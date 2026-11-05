# Benchmark: what the port is measured on, and why

The harness benchmark is a **porting benchmark**, not a forecasting-leaderboard
claim. It answers three questions quickly, on real data, with held-out ground
truth: (1) is the TT implementation numerically faithful to the FP32 oracle,
(2) does it preserve batch/mask/revisit semantics, and (3) is it getting
faster relative to its own recorded baseline. Task-level forecast quality is
measured only at the full gate, and only as *agreement with the FP32 reference*
— the model's absolute accuracy is Chronos-2's problem, not the port's.

## What is in the frozen corpus (`corpus.json`)

| Series | Origin | Why it is here |
|---|---|---|
| `etth1_hufl…etth1_ot` (7) | ETTh1, last 1536 hourly rows (Jun–Aug 2018), [ETDataset](https://github.com/zhouhaoyi/ETDataset), MIT | Real-world electricity-transformer telemetry; the classic long-horizon forecasting fixture used by PatchTST/iTransformer papers and by Tenstorrent's own time-series bounty issues (#32139/#32140). Different scales and regimes per column stress normalization and arcsinh scaling honestly. |
| `syn_sine_mixed` | deterministic: 168/24-hour sines + seeded noise (`Random(1729)`) | Smooth doubly-periodic signal; easy to eyeball, checks that periodic structure survives patching. |
| `syn_arcsinh_stress` | deterministic: `sinh(4·sin)` + tiny noise (`Random(4104)`) | Alternating tiny/huge magnitudes (|x| up to ~27). Directly stresses `use_arcsinh` scaling, FP range behavior and BFP8_B shared-exponent loss. |
| `syn_constant` | `7.0` × 1536 | Degenerate input: zero variance after normalization; catches NaN/div-by-zero paths in scaling and quantile heads. |
| `etth1_*_long` (7) | ETTh1, last 4352 hourly rows of the same columns, values rounded to 6 decimals (the 1536-point tails are their last rows) | Context 2048, the shape of TT's own benchmark, and aligned variates for group and covariate cases. |

All values are frozen in the file (canonical-JSON sha256
`e22aaed68e558fdffe239e765a1fa28baaccfb4adeaba75ec4fb69c6dd47699a`); synthetic
series are frozen outputs, not regenerated, so every campaign and every
reference run sees byte-identical data.

## Workload shape and stage sizes

The envelope is TtChronos's public API (CONTRACT.md):
- batch 1–1,024 per call, on one card or on the 1×4 mesh;
- context 1–2048;
- horizon 1–64 (four 16-step patches);
- unique or grouped rows, with optional known future covariates;
- the model's 21 native quantiles.

Windows carve `[context | future]` slices. Futures are ground truth for the full-stage task metric
and stay in the private oracle directory. Grouped cases use aligned ETTh1 columns, so the variates
of a group share timestamps. The oracle is the official `Chronos2Model.forward` in FP32, given
the same context (NaN where masked), `group_ids` and `future_covariates` as `TtChronos.forward`.
It is chunked on group boundaries.

| Stage | Cases | Contents |
|---|---|---|
| `smoke` | 1 | ETTh1 OT, context 512 → horizon 64 |
| `bringup` | 15 | the original cases without the masked tail (batch/single/reorder, contexts 64/65, constant), contexts 17/100/160/256 at 40 rows with horizons 16–64, context 2048, two groups of four plus a group-isolation check, a target with two known-future covariates |
| `perf` | 24 | bringup + context 512 at b1/2/4/16/64/256/1,024 + paper shape (context 2048) at b1,024, unique and groups of four |
| `mesh` | 22 | bringup + b4/6/64/256/1,024 + the paper-shape cases at b1,024 (capped at 1,024: TT's encoder misses the gate at 4,096) |
| `full` | 42 | 9 series × 3 windows (ctx 512/128/64) + 7 long series at context 2048 + batch4, constant, behavior/revisit cases + a 7-variate group + a covariate case |

Timing: 1 warmup + 1 measured repeat per case in short stages, 3 measured
repeats in full; protocol `cpu_arrays_to_cpu_arrays_with_sync_excluding_load_
preprocessing_progress_v1`.

## Gates

Interior missing values and context 1 are not gated. `TtChronos` accepts all-valid contexts
only, and TT's encoder misses the NRMSE gate at context 1 (0.09–0.13, measured 2026-11-02), so
neither side can pass them. Context 160 at 1,024 or more rows is not gated either: TT's encoder's
worst row reaches 0.050–0.084 there. Left padding of short contexts (17, 64, 65, 100) is covered.

- `quantile_max_row_nrmse ≤ 0.04` vs the FP32 oracle (per batch row, over the
  full [horizon × 21-quantile] block) — every numerical case.
- Behavior: same series forecast through different batch shapes/orders must
  agree within NRMSE ≤ 0.01; repeated identical calls must be bit-identical.
- Full stage only: weighted quantile-loss (mean pinball over steps and all 21
  levels, scaled by mean |y|) on held-out futures may inflate by ≤2% versus
  the FP32 reference's own forecasts. This is agreement with the reference,
  not a leaderboard score.
- Diagnostics (non-gating): quantile-crossing counts, first-call latency,
  forecasts/second, memory snapshots.

## Why not the big public benchmarks

- **GIFT-Eval / fev-bench / Chronos Benchmark II**: these are where Chronos-2's
  zero-shot SOTA claims live, but they span tens of datasets (many tens of GB),
  need their own loaders/scorers, and measure model accuracy rather than port
  fidelity. Using them as the porting loop's inner gate would repeat the NLLB
  mistake of spending turns waiting on evaluation instead of engineering. They
  are the right *post-port* validation if we later want an accuracy claim, and
  SOURCES.md links them for context.
- **Full ETTh1 (17k rows) / ETTm / Electricity / Traffic**: bigger windows add
  wall time without changing what they can detect about the TT port at this
  envelope; the frozen 1536-point tails keep every suite short while remaining
  real data with published provenance.
- **Random-noise-only corpora**: cheap, but hide exactly the failure modes we
  care about (periodicity, magnitude spread, degenerate variance) — hence the
  three purpose-built synthetic series instead.

If a later phase wants broader coverage, extend the corpus file and pins
deliberately (new sha256, new references, documented provenance) — never by
quietly editing series the agent has already been evaluated against.
