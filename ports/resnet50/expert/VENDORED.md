# Expert baseline: Tenstorrent's hand-written ResNet-50

Operator-only. The campaign never copies this directory into an agent workspace; it is the
yardstick the agents' port is compared with afterwards, measured through the same evaluator:

    python -m portforge_tt.control.baseline --port ports/resnet50 --cluster CLUSTER.yaml \
        --corpus CORPUS.json --source ports/resnet50/expert --precision bfp8_b --stages smoke,bringup,perf,full

Source: tt-metal `0df0dde3316322cf44451cb86f1f09555c6119cc` (the TT host's checkout). The three model
files are byte-identical upstream at `573f4b35e8bc`, where the 2026-11-04 pilot first vendored them.
License: Apache-2.0; SPDX headers kept.

| File | Upstream path | Upstream SHA-256 (first 16 hex) |
|---|---|---|
| `expert_resnet50.py` | `models/demos/vision/classification/resnet50/ttnn_resnet/tt/ttnn_functional_resnet50.py` | `d5feef0f8d64e7e2` |
| `expert_preprocessing.py` | `.../ttnn_resnet/tt/custom_preprocessing.py` | `a9ecb4c439b678d4` |
| `expert_model_utils.py` | `.../ttnn_resnet/tt/ttnn_functional_resnet50_model_utils.py` | `4fbdb6d9f1de5d3a` |
| `expert_executor.py` | `models/tt_cnn/tt/executor.py` | unchanged |
| `expert_pipeline.py` | `models/tt_cnn/tt/pipeline.py` | import line only |

## Changes from upstream (complete)

1. Imports: `models.demos...model_utils` becomes `expert_model_utils`, `models.common.utility_functions`
   becomes `tt_py_test_utils_common.utility_functions` (upstream's own shim target), and the pipeline
   imports `expert_executor`. The sandbox hides `models/`.
2. `expert_resnet50.py`, global average pool: upstream hard-codes `bfloat8_b` and LoFi; it now follows
   `model_config`, so the bf16 mode does not silently run a BFP8 op. Identical in the bfp8_b mode.
3. `expert_resnet50.py`: every compute-kernel config goes through `_compute_config`, which takes
   `MATH_APPROX_MODE` (default True) and `FP32_DEST_ACC_EN` (default False) from `model_config`. The
   defaults are upstream's, so the bfp8_b mode is unchanged.

Both patches are marked `PORTFORGE PATCH`.

## `backend.py` (PortForge-written)

It reproduces upstream's fastest end-to-end recipe, `test_perf_e2e_resnet50.py::test_perf_trace_2cqs`:
- the `resnet50_test_infra` input placement (L1 height-sharded fold input, DRAM staging);
- `PipelineConfig(use_trace=True, num_command_queues=2)`, i.e. `MultiCQTracedModelOverlappedInputExecutor`.

There is one pipeline per batch size, captured largest first: capturing batch 16 first clashes the
batch-32 compile with a persistent L1 buffer.

Modes:
- `bfp8_b` is upstream's configuration: bfloat8_b weights and activations, LoFi.
- `bf16` is accuracy-modified, not upstream: HiFi4, fp32 accumulation, approximate math off. Batch 32
  does not fit L1 in it (static circular buffers clash), so its calls fail.

Unlike upstream's harness, our timing protocol converts each float32 batch to a bfloat16 host tensor
and reads the logits back to NumPy inside the timed call. The same holds for the agents' backends.

## Measured on the campaign's TT host, one Blackhole p150 card (2026-11-05)

Random-pixel inputs, outside the harness, per `classify` call or 256-image `classify_stream`:

| mode | b16 call | b32 call | stream 16x16 | stream 8x32 | max row NRMSE (vs FP32) | top-1 |
|---|---|---|---|---|---|---|
| bfp8_b | 2.46 ms, 6,500 img/s | 3.88 ms, 8,240 img/s | 7,670 img/s | 9,710 img/s | 0.094 | 63/64 |
| bf16 | 3.30 ms, 4,850 img/s | fails | 5,350 img/s | fails | 0.018 | 64/64 |

Upstream's own targets for the same recipe on Blackhole are 1.8 ms (b16) and 2.7 ms (b32), with
pre-converted host inputs.
