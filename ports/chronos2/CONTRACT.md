# Chronos-2 contract: make upstream `chronos_forecast` faster, keep its whole API

This round does not write a new port. The workspace starts with Tenstorrent's own Chronos-2
package, `models/experimental/chronos_forecast` at tt-metal `f8f728b7627`, and a fixed user-level
`backend.py` that drives it. The deliverable is a pull request to that package. Every speedup must
live inside `models/experimental/chronos_forecast/`, so that `TtChronos` itself is faster for every
caller.

Upstream at this commit equals the seed except for one line: `ops/common.py` resolves kernel
paths from its own directory, so the vendored kernels are the ones compiled.

## Interface

`backend.py` stays user code: load with `TtChronos.from_torch_model`, choose TT's preset
(`TtChronosPrecision()` for `bf16`, `.performance()` for `bfp8_b`), call `prepare_inputs`,
and replay through `TtChronosTraceRunner`. You may change it only:

- to call a new public API that the package exposes and documents;
- to tune `DEVICE_OPTIONS` (`trace_region_size` 0–256 MiB, `l1_small_size` 0–128 KiB,
  `num_command_queues` 1–2);
- to tune the documented L1 choice.

Never move compute, caching or batching tricks into it.

```python
backend.forecast(past_values, past_observed_mask, prediction_length, group_ids=None, future_covariates=None)
# past_values, past_observed_mask: float32 [B, T], 1 = observed (masked cells hold 0; the backend makes them NaN)
# group_ids: int64 [B] or None (None: every row its own group). Rows of a group attend to each other.
# future_covariates: float32 [B, prediction_length] or None; known future values, NaN for rows to forecast
# returns {"quantiles": float32 [B, prediction_length, 21]}
```

## The public API that must keep working

`TtChronos.from_torch_model`, `TtChronos.forward` and `prepare_inputs`/`upload_inputs`/
`forward_device`/`postprocess_output`, `TtChronosTraceRunner` (`capture`, `execute`,
`execute_pipelined`, `stream`, `release`), `TtChronosPrecision` and its presets, mesh
(1×N data parallel), `group_ids`, `future_covariates`, contexts up to 2048 and horizons up to
four 16-step patches.

- Keep signatures backward compatible. Add options with defaults rather than changing behaviour.
- **No device state may outlive a public call** unless the caller opted into it and can release
  it, as `TtChronosTraceRunner` does. A live trace overwrites buffers allocated after it.
- Upstream's tests (`/opt/tt-metal/models/experimental/chronos_forecast/tests`, readable from
  `port_run`) must still pass. The operator runs them on your final source.

## Envelope and gates

- **Accuracy:** max-row quantile NRMSE ≤ 0.04 against the official FP32 `Chronos2Model.forward` on
  every numerical case, at `bf16` and `bfp8_b`.
- **Consistency:** batch, single, reorder and group-isolation cases agree within ≤ 0.01.
- **Determinism:** repeated identical calls are bit-identical.
- **Cases** (BENCHMARK.md):
  - contexts 17, 64, 65, 100, 160, 256, 512 and 2048;
  - horizons 16, 32, 48 and 64;
  - a constant series;
  - groups of four aligned variates;
  - one target with two known-future covariates.
- **Perf (one card):** the bringup cases plus:
  - context 512 at b1, 2, 4, 16, 64, 256 and 1,024;
  - the paper shape (context 2048, b1,024), unique and in groups of four.
- **Mesh (1×4):** b4, 6, 64, 256 and 1,024 at context 512, plus the paper-shape cases at
  b1,024. It stops at 1,024 because TT's own encoder misses the gate at 4,096 (bfp8_b
  0.040–0.042, grouped bf16 0.050).
- **Full (held out, at submit):**
  - windows of all ten series at contexts 64, 128, 512 and 2048;
  - a seven-variate group;
  - a covariate case;
  - weighted quantile-loss inflation ≤ 2%.

Not gated, because upstream cannot do them either; list them under limitations:
- interior missing values: `prepare_inputs` accepts all-valid masks only;
- context 1: TT's encoder gives NRMSE 0.09–0.13;
- short contexts at large batch: at context 160 with 1,024–4,096 rows, TT's encoder's worst row
  reaches 0.050–0.084, while 40 rows give 0.013.

Context 17 at `bfp8_b` sits at 0.039 in TT's own encoder, so changes there have almost no
accuracy margin.

The learned compute runs on TT. Host work allowed is non-learned arithmetic, as upstream already
does it: instance norm, arcsinh and sinh, patching, masks, RoPE tables.

## What counts as faster

The first passing evaluation of each stage and precision is the **baseline**. Evaluate the
unmodified seed first, at perf and mesh, at `bf16` and `bfp8_b`. That baseline is TT's best
configuration:

- its trace runner;
- its `performance()` preset at `bfp8_b`;
- L1 chunks from 16 series per chip.

A candidate counts only if, against that baseline at the same precision and stage:

- **no cell is slower:** every sweep case's p50 and the b≤4 latencies are compared, not just
  the best throughput;
- **device memory does not grow:** at most +10%;
- **all gates pass.**

Report every cell, wins and ties. Never claim a number that one same-source job does not show.

## Deliverable

At the end, the source tree under `models/experimental/chronos_forecast/` must be ready to
replace upstream's files one to one:

- SPDX headers;
- no sandbox paths;
- tests for every new code path, including:
  - short contexts at large batch;
  - an explicit `context_mask`;
  - two models on one device;
  - a user tensor allocated between calls;
- `README.md` and `optimizations.md` updated with measured before/after numbers and job ids, in
  the style of their existing ranked table.

Prepare a PR title and body. The operator reviews, re-measures, opens the PR and pushes.
Agents never push.
