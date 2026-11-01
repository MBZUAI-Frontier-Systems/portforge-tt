# Porting rules (every pack)

These rules hold for every model. The pack's CONTRACT.md adds the model's
interface, envelope and gates; where it is more specific, it wins.

## Backend

Write `backend.py` from scratch. `create_backend(weights_path, config, device, *,
precision="bf16")` returns an object with the methods CONTRACT.md names. Read
configuration values rather than hardcoding architecture dimensions; the
registry in `models.json` is authoritative.

- Load the pinned official checkpoint. CPU weight conversion and deterministic
  preprocessing are allowed; learned inference runs on TT.
- The harness owns the TTNN device: never open or close another one. On one
  card `device` is a 1×1 `MeshDevice`; a mesh stage passes a 1×N one, so the
  same backend can serve both (`device.get_num_devices()`).
- Optional `DEVICE_OPTIONS`: `trace_region_size` 0–128 MiB,
  `l1_small_size` 0–128 KiB, `num_command_queues` 1–2.
- Never mutate input arrays. Batched rows stay independent. Return
  CPU-visible numpy results. Repeated identical calls must be bit-identical.
- During timed calls, PyTorch learned ops on the host CPU are rejected; host
  matmuls pass only when no operand carries model weights (e.g. RoPE angles
  from positions). A call that reaches no TT op fails.

## Precision

Evaluation selects `bf16` (default), `fp32`, `fp16` or `bfp8_b` explicitly and
passes it to the factory. Declare the actual dominant weight and activation
formats, accumulation and every higher-precision exception in the instance's
`precision_policy` (`mode`, `weights`, `activations`, `accumulation`,
`exceptions`); the weight format and mode must match the request, and mixed
activations need an entry in `exceptions`. Unsupported modes fail clearly; never
substitute another format silently. TT `bfloat8_b` is block floating point with
shared exponents, not IEEE FP8.

Per-op-group choices go in an optional `groups` field, e.g.
`"groups": {"ff_matmul": {"weights": "bfp8_b", "math_fidelity": "LoFi"}}`
(keys: weights, activations, math_fidelity LoFi/HiFi2/HiFi3/HiFi4,
fp32_dest_acc, packer_l1_acc, math_approx). Any group that departs from the mode
needs an `exceptions` entry.

The correctness oracle is the official implementation in FP32 on the reference
host (CUDA preferred, CPU recorded). CUDA BF16 comparisons are context only;
they never replace the oracle or relax gates.

## Baseline before optimization

Before any optimization claim, record measured baselines on the fixed suites
from the same TT host: PyTorch on the host CPU and, if the installed stack
supports it, an ONNX/forge path. Every speedup cites the matched baseline
artifact, the same suite, precision and card. TT-vs-TT ratios never mix with
CUDA context numbers.

## Stages

- `smoke`: one real case. `bringup`: about eight cases covering batch,
  padding, masks, length boundaries and edge inputs. One warmup and one
  measured call each.
- `perf`: the bringup cases plus a batch sweep within the envelope, repeated
  about 100 times or for at least 20 s per case (at most 120 s). It keeps every
  accuracy gate and reports latency percentiles (cases of batch ≤4), the
  throughput curve and board energy. Optimizations are ranked on it.
- `mesh` (packs that declare it): the perf workload data parallel over a 1×N
  mesh of one host's cards, ranked on its own frontier.
- `port_profile` runs one Tracy-profiled pass and returns the per-op table; for
  large batches profile one perf case, e.g. `port_profile(stage="perf", case="sweep-b1024")`.
- `full`: the frozen held-out corpus plus consistency cases, run once on frozen
  source through `port_submit`. Early passes are debugging milestones:
  `verdict.accepted` is true only after a successful full evaluation.

## Objective after the gates

Passing candidates are ranked by the pack's `optimization` block (overridable
at launch): a Pareto front over p50/p99 latency, throughput, energy per item and
device memory by default, or a primary metric under constraints. Record what
each source applies in `TRANSFORMS.json` (catalogue ids from
TT_OPTIMIZATIONS.md, its parent source, anything novel).

## Evidence and sandbox

Collect results with `port_status(compact=True)`; `port_log` returns
stdout/stderr only. `port_run` jobs see public inputs, never verdicts or oracle
outputs, and mount `/models.json`, `/portforge_models.py`, prepared `/input`
files and routed `/weights` read-only; inspect them with a bounded `port_run`
before assuming a schema. Candidate-authored pass flags are not evidence: the
controller's job receipts are. Keep job IDs outside source code so recording a
result does not change the source hash.

Timing covers array transfer, host control, learned compute and device
synchronisation, and excludes weight loading. Do not exploit known test inputs,
precompute outputs, run hidden CPU inference or alter the trusted evaluator. Do
not call a handful of samples a tail-latency SLA.

## tt-metal deliverable

A tt-metal model package for an upstream PR under `models/experimental/<model>`:
`tt/` (TTNN implementation), `demo/` (CLI), `reference/` (FP32 reference
utilities), `tests/` (pytest, importable without sandbox mounts), `benchmarks/`,
`docs/`, and a README with model-card links, measured evidence, limitations and
a performance table. SPDX headers on source files; only Apache-2.0-compatible
code. The full checklist is UPSTREAM_PR.md. The worker prepares branch, commit
message and PR body; the operator reviews, signs and submits.
