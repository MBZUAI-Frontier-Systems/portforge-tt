# SPDX-FileCopyrightText: © 2026 PortForge-TT (baseline wrapper around Tenstorrent's ResNet-50 expert code)
# SPDX-License-Identifier: Apache-2.0
"""Expert baseline: Tenstorrent's hand-written ResNet-50 behind the ports/resnet50 contract.

Operator-only: never seeded into an agent workspace. The model code and the tt_cnn pipeline are
vendored unchanged in behaviour from tt-metal (see VENDORED.md); weights are the pinned torchvision
IMAGENET1K_V1 checkpoint, BatchNorm-folded on the host at load time by the expert's preprocessor.

Execution follows upstream's fastest end-to-end recipe (test_perf_e2e_resnet50 test_perf_trace_2cqs):
one traced model per batch size behind MultiCQTracedModelOverlappedInputExecutor, which writes inputs
on command queue 1 while queue 0 replays the trace and reads outputs. classify() enqueues one batch,
classify_stream() enqueues all its batches back to back, then both synchronize once.

Modes:
  bfp8_b  the upstream configuration: bfloat8_b weights and activations, LoFi, fp32_dest_acc_en=False,
          math_approx_mode=True.
  bf16    bfloat16 weights and activations, HiFi4, fp32_dest_acc_en=True, math_approx_mode=False: an
          accuracy-modified configuration, not an upstream one. Batch 32 does not fit L1 with the
          expert's knobs in this mode (VENDORED.md); its calls fail with the error recorded at load.
"""
import os

# The sandbox clears the environment; torch's inductor cache (imported via torchvision) needs a user name.
os.environ.setdefault("USER", "portforge")

import numpy as np
import torch
import torchvision
from safetensors.torch import load_file
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from expert_model_utils import is_blackhole_p100
from expert_pipeline import PipelineConfig, create_pipeline_from_config
from expert_preprocessing import create_custom_mesh_preprocessor
from expert_resnet50 import resnet50 as ExpertResNet50
from tt_py_test_utils_common.utility_functions import divup

IMAGE_SIZE, NUM_CLASSES, BATCH_SIZES = 224, 1000, (16, 32)
# Upstream's trace_2cqs device parameters; the trace region holds both batch sizes' traces.
DEVICE_OPTIONS = {"l1_small_size": 32768, "trace_region_size": 16 * 1024 * 1024, "num_command_queues": 2}

POLICY_BFP8 = dict(
    mode="bfp8_b", weights="bfp8_b", activations="bfp8_b",
    accumulation="expert bfp8_b config, as upstream: LoFi math, fp32_dest_acc_en=False, math_approx_mode=True, "
                 "packer_l1_acc per upstream call site",
    exceptions=["input pixels are uploaded as bfloat16 (expert test-infra convention) and folded on device "
                "before the bfloat8_b stem convolution"])
POLICY_BF16 = dict(
    mode="bf16", weights="bf16", activations="bf16",
    accumulation="HiFi4 math, fp32_dest_acc_en=True, packer_l1_acc=True, math_approx_mode=False "
                 "(expert code, accuracy-modified config)",
    exceptions=["expert code, accuracy-modified config, not an upstream configuration: upstream runs these layers with "
                "fp32_dest_acc_en=False and math_approx_mode=True; this mode sets fp32 destination accumulation on and "
                "approximate math off (HiFi4 throughout, including the fc and the average pool)"])


def _model_config(precision):
    if precision == "bfp8_b":
        return {"MATH_FIDELITY": ttnn.MathFidelity.LoFi, "WEIGHTS_DTYPE": ttnn.bfloat8_b,
                "ACTIVATIONS_DTYPE": ttnn.bfloat8_b, "MATH_APPROX_MODE": True, "FP32_DEST_ACC_EN": False}, POLICY_BFP8
    if precision == "bf16":
        return {"MATH_FIDELITY": ttnn.MathFidelity.HiFi4, "WEIGHTS_DTYPE": ttnn.bfloat16,
                "ACTIVATIONS_DTYPE": ttnn.bfloat16, "MATH_APPROX_MODE": False, "FP32_DEST_ACC_EN": True}, POLICY_BF16
    raise ValueError("expert baseline supports bf16 and bfp8_b only; got " + str(precision))


def _input_memory_configs(device, batch):
    """resnet50_test_infra.setup_dram_sharded_input: L1 height-sharded fold input and its DRAM staging."""
    grid = ttnn.CoreGrid(y=8, x=6) if batch == 16 else ttnn.CoreGrid(y=10, x=13)
    if batch == 32 and is_blackhole_p100(device):
        grid = ttnn.CoreGrid(y=8, x=8)
    rows = batch * 3 * IMAGE_SIZE
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    l1 = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1,
                           ttnn.ShardSpec(shard_grid, (divup(rows, grid.x * grid.y), IMAGE_SIZE),
                                          ttnn.ShardOrientation.ROW_MAJOR))
    dram_grid = device.dram_grid_size()
    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid.x - 1, dram_grid.y - 1))}),
        [divup(rows, dram_grid.x), IMAGE_SIZE], ttnn.ShardOrientation.ROW_MAJOR))
    return l1, dram


def _host_input(pixel_values):
    """Host bfloat16 row-major NCHW tensor, as the expert test infra builds it (torch cast, then from_torch;
    the same bits as casting in from_torch at about a tenth of the host time)."""
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(pixel_values, dtype=np.float32)).to(torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def _failure(exc):
    return RuntimeError(str(exc).split("backtrace:")[0].strip()[-600:])


class Backend:
    def __init__(self, pipelines, policy):
        self._pipelines = pipelines
        self.precision_policy = policy

    def _pipeline(self, batch):
        pipeline = self._pipelines.get(batch)
        if pipeline is None:
            raise ValueError("batch size must be one of %s" % (BATCH_SIZES,))
        if isinstance(pipeline, Exception):
            raise RuntimeError("expert model unavailable at batch %d: %s" % (batch, pipeline))
        return pipeline

    def _run(self, batches):
        batch = int(batches[0].shape[0])
        outputs = self._pipeline(batch).enqueue([_host_input(values) for values in batches]).pop_all()
        logits = np.concatenate([ttnn.to_torch(output).to(torch.float32).numpy().reshape(batch, NUM_CLASSES)
                                 for output in outputs])
        return {"logits": np.ascontiguousarray(logits, dtype=np.float32)}

    def classify(self, pixel_values):
        return self._run([pixel_values])

    def classify_stream(self, batches):
        if len({int(values.shape[0]) for values in batches}) != 1:
            raise ValueError("every batch of a stream has the same size")
        return self._run(batches)


def create_backend(weights_path, config, device, *, precision="bf16"):
    model_config, policy = _model_config(precision)
    path = weights_path if os.path.isfile(weights_path) else os.path.join(weights_path, "model.safetensors")
    reference = torchvision.models.resnet50(weights=None)
    reference.load_state_dict(load_file(path, device="cpu"), strict=True)
    reference.eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference, custom_preprocessor=create_custom_mesh_preprocessor(None), device=None)
    pipelines = {}
    # Largest batch first: capturing batch 16 before batch 32 clashes the batch-32 compile pass with a
    # persistent L1 buffer of the batch-16 trace (measured on the card in the pilot).
    for batch in sorted(BATCH_SIZES, reverse=True):
        try:
            model = ExpertResNet50(device=device, parameters=parameters, batch_size=batch, model_config=model_config,
                                   input_shape=(batch, 3, IMAGE_SIZE, IMAGE_SIZE), kernel_size=3, stride=2,
                                   dealloc_input=True, final_output_mem_config=ttnn.L1_MEMORY_CONFIG)
            l1, dram = _input_memory_configs(device, batch)
            pipeline = create_pipeline_from_config(
                PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
                model=lambda tensor, model=model: model(tensor, device, {}), device=device,
                l1_input_memory_config=l1, dram_input_memory_config=dram)
            pipeline.compile(_host_input(np.zeros((batch, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)))
            pipelines[batch] = pipeline
        except RuntimeError as exc:
            pipelines[batch] = _failure(exc)
    return Backend(pipelines, dict(policy, exceptions=list(policy["exceptions"])))
