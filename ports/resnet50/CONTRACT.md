# ResNet-50 backend contract

Write `backend.py` from scratch for the pinned `timm/resnet50.tv_in1k` (model
key `resnet50`, license bsd-3-clause): torchvision's ImageNet-1k
`IMAGENET1K_V1` weights in `model.safetensors`, with torchvision `resnet50`
state-dict names. PORTING_RULES.md holds the rules every port follows; this
file adds the model's interface, envelope and gates.

```python
DEVICE_OPTIONS = {}  # optional: trace_region_size (<=256 MiB), l1_small_size (<=128 KiB), num_command_queues (1-2)

def create_backend(weights_path, config, device, *, precision="bf16"):
    # Return an object with the methods below. The harness owns the TTNN device.
    ...

class Backend:
    # Required instance declaration; it must match the requested precision.
    precision_policy = {"mode": "bf16", "weights": "bf16", "activations": "bf16",
                        "accumulation": "fp32", "exceptions": []}

    def classify(self, pixel_values):
        # pixel_values: read-only CPU numpy float32, C-contiguous [B,3,224,224], B in {16, 32},
        # already normalized with the ImageNet mean and std. Return {"logits": CPU numpy
        # float32 [B,1000]}.
        ...

    def classify_stream(self, batches):
        # batches: a list of N read-only float32 arrays, each [B,3,224,224], same B in {16, 32}.
        # Return {"logits": float32 [N*B,1000]}, rows in input order. You may overlap transfers,
        # compute and readback across batches (e.g. two command queues); return only when every
        # logit is on the host. A loop over classify() is a valid first version.
        ...
```

Load the pinned checkpoint; CPU weight conversion at load time is allowed,
including folding BatchNorm into the convolutions. All network compute in
`classify` and `classify_stream` runs on the TT device: convolutions, pooling,
residual additions, activations and the classifier. Host work in the timed call
is limited to moving and casting data. Rows are independent: a row never depends
on the other rows of its batch or stream. Repeated identical calls return
bit-identical logits. Never write into the inputs, and never cache inputs or
outputs across calls: every call computes its logits.

## Envelope

- Batches of 16 and 32 images, 224x224 RGB, single centre crop (the corpus
  builder resizes and crops; the evaluator normalizes, outside the timed call).
- 1000 logits per image. No test-time augmentation, no change to the pinned
  weight values beyond exact BatchNorm folding.
- One card. Out of scope: other batch sizes, resolutions or heads, multi-card
  execution.

The oracle is FP32 `torchvision.models.resnet50` on the pinned weights, on the
reference host (CUDA with TF32 off, or CPU, recorded). Its BF16 run is a
diagnostic and never replaces the oracle.

## Gates

This track judges task behaviour; per-row numerics are a sanity check. Every
gate is decided by `evaluate.py`.

| Gate | Where | Bound |
| --- | --- | --- |
| Logits NRMSE per row against the oracle | every case | at most **0.30** |
| Behaviour: rows against the same images in another case | reversed, mixed, repeat and stream cases | at most **0.05** per row |
| Top-1 agreement with the oracle | every smoke, bringup and perf case whose images all carry labels (not the synthetic batch) | at least **14/16** (28/32) |
| Held-out top-1 drop | full stage, 1,024 held-out images | oracle top-1 accuracy minus yours at most **1.0 pp** |
| Held-out top-1 agreement | full stage | at least **0.95** |
| Determinism | every timed repeat | bit-identical |

Top-5 agreement and the held-out NRMSE are reported, not gated. Corpus
provenance and the reasons for each bound: BENCHMARK.md.

## Stages

- `smoke`: one batch of 16 dev images.
- `bringup`: batch 16, its reversed order, two more batches of 16, batch 32, a
  batch of 16 built from two slices of the batch of 32, a repeat, five
  synthetic images with eleven dev images, and `stream-b16x4` (four batches of
  16 in one `classify_stream` call).
- `perf`: bringup plus `sweep-b16`, `sweep-b32` (one `classify` call each) and
  `stream-b16x16`, `stream-b32x8` (256 images in one `classify_stream` call).
  Each case repeats at least 100 times and 20 seconds.
- `full` (controller only, held-out): 64 batches of 16, a batch of 32, a
  reversed batch, the synthetic batch and a stream of 16 batches of 16.

Timing protocol `normalized_pixels_to_logits_with_sync_excluding_load_normalization_progress_v1`:
the timed span starts when `classify` or `classify_stream` is called and ends
after device synchronization. It includes host-to-device transfer, device
compute and the readback of the logits; it excludes weight loading and the
uint8-to-float32 normalization. A case's images per second is its image count
over the p50 of its timed repeats.

## Objective

Passing sources are ranked on a Pareto front, separately for `bf16` and
`bfp8_b` (both delivery precisions), over:

- p50 and p99 latency of the single `classify` calls;
- throughput: the best images per second over the sweep and stream cases;
- board energy per image;
- resident device DRAM;
- max NRMSE against the FP32 oracle.

Every gate above is a hard floor. Within it, fidelity is a tradeoff axis like
the others: a faster or leaner configuration with higher NRMSE and a
higher-fidelity one can both be on the front. Weight and activation formats,
math fidelity and accumulation are free per op group while every gate holds;
declare them in `precision_policy`.

## Deliverable

A tt-metal style package: `tt/` (the TTNN implementation), `demo/` (classify
an image batch, print top-5 classes, explicit checkpoint, device and precision
arguments), `reference/`, `tests/` (pytest with the `device` fixture),
`benchmarks/`, and a README with measured numbers and limitations. SPDX
headers; Apache-2.0-compatible code; the weights are bsd-3-clause.
