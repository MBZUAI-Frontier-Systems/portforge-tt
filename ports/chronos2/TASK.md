# Chronos-2: make Tenstorrent's `chronos_forecast` faster everywhere

The workspace holds upstream `models/experimental/chronos_forecast` and a user-level `backend.py`
(CONTRACT.md). Make `TtChronos` faster at every measured cell without losing any feature: groups,
covariates, context 2048, mesh, trace runner. The budget is short, about three hours, so work
from the evidence below rather than rediscovering it.

`prior_art/` holds our earlier standalone fast path, read-only reference code: `tt/fast_path*.py`
and `ops/split_rope`. It beat this package at bf16 from b64 up and at bfp8_b at b64, but lost at
bfp8_b b4, single-card b1,024 and on four cards. It also had bugs, so learn from its mechanisms
and do not copy it whole:

- It segfaulted on short contexts. `ttnn.slice` over the full range returns its input, and
  deallocating the input freed the result.
- Its program configs were tuned only for S=37 and hit L1 clashes elsewhere.
- Its live trace overwrote later allocations.

## Order

0. **No CPU or ONNX baselines, no packaging first.** The seed is the baseline. CPU-heavy jobs on
   the TT host corrupt every timing. Perf, mesh and profile jobs hold every card while they run,
   so queue them deliberately. Do not build `tests/`, `demo/` or docs before a perf receipt shows
   a win.
1. **Baseline first.** Evaluate the unmodified seed:
   - `port_evaluate` smoke, then bringup;
   - perf and mesh at `bf16` and `bfp8_b`.

   Record each cell in BASELINE.md with job ids. If an upstream case fails a gate, stop and
   report it; do not "fix" the evaluator.
2. **Profile.** Run `port_profile` on `sweep-b1024` and `paper-b1024` at both precisions. Find
   what dominates per op.
3. **Optimize single card at context 512** (most cells), one lever at a time, starting with
   lever 1 (token packing): it attacks the largest measured waste. Keep a lever only
   when no cell regresses (CONTRACT.md); record each attempt in TRANSFORMS.json.
4. **Paper shape and groups:** confirm the gains hold at context 2048 and in groups of four.
5. **Mesh:** improve 4-card scaling.
6. **Deliverable:**
   - tests for new paths, plus the regression tests named in CONTRACT.md;
   - README and `optimizations.md` rows with job ids;
   - a PR title and body.

   With about 40 minutes left: freeze, re-run perf and mesh at both precisions, and
   `port_submit`.

## What the seed measures (operator precheck, 2026-11-02)

These come from the same backend on this host, outside the harness. Your harness baseline is
the authority.

| Precision | Cards | b1 | b4 | b64 | b1,024 | paper b1,024 | paper groups-4 b1,024 |
|---|---|---|---|---|---|---|---|
| bfp8_b | 1 | 2.17 ms | 1,575/s | 5,697/s | 12,270/s | 5,305/s | 3,412/s |
| bf16 | 1 | 2.32 ms | 1,408/s | 5,896/s | 8,230/s | 3,613/s | 2,171/s |

On the 4-card mesh, b1,024 (the mesh stage's largest batch) runs at 28,206/s (bfp8_b) and
23,322/s (bf16), about 2.3× and 2.8× one card. At 4,096 rows TT's encoder misses the accuracy gate,
so the mesh stage stops at 1,024.

Every gated case passes at both precisions. The tightest is context 17 at bfp8_b, at 0.0392.

There is an upstream accuracy bug worth fixing if you have time, because a fix is a real PR item.
At context 160 / horizon 32 with 1,024 rows, the worst row reaches 0.050 at bf16 and 0.084 at
bfp8_b, against 0.013 with 40 rows. It is not gated; check it with a `port_run` probe.

Small grouped calls are slow: 8 rows in groups of four take 8.6–11 ms, against 2.5 ms for 2
unique rows.

## Levers, ranked by evidence

1. **Token packing on the time axis.**
   - The gap: at context 512 and horizon 64 a series has 37 tokens and is padded to 64, so 42%
     of every matmul and SDPA row is padding. At context 160 / horizon 32, 13 tokens are padded
     to 32.
   - The idea: put several series in one tile-padded sequence, with a block-diagonal time
     mask and per-series RoPE positions. Our earlier port gained +21% at bf16 from this (packing
     up to six 37-token series into 224). See `prior_art/tt/fast_path_ops.py` (segment map, RoPE
     offsets, packing plan).
   - Use it only when packing saves at least 25% of tokens. The paper shape (133 → 160) gains
     little.
   - It must compose with their L1 series chunks: a chunk holds whole packed sequences. It must
     also compose with group blocks: pack only unique-group rows, or whole groups.
2. **Group attention with unique groups is an identity mix.** Each row sees only itself, so the
   sublayer is `x + W_o W_v RMSNorm(x)`. Upstream already skips the mask (diagonal path). Fold
   `W_o·W_v` into one d×d linear on the host when every group is unique; this saves about 7% of
   FLOPs.
3. **Batch-gated fidelity.** HiFi2 on wo, ff_up and ff_down only above batch 4 kept accuracy in
   our port. Upstream's presets do not depend on batch.
4. **Small batch (b1–b4) uses the non-L1 path.** Our 1D in0-mcast grid widening (24 → 72–96
   cores at Mt ≤ 2) helped there. Upstream b1 is already about 2.1 ms, so measure before
   keeping anything.
5. **L1 chunk budgets for this grid.** `TtChronos` warns that its `_L1_CHUNK_TOKENS_*` were
   measured on an 11×10 worker grid, and these cards have 13×10. Re-measure them with
   `/opt/tt-metal/models/experimental/chronos_forecast/sweeps/sweep_l1_chunk.py` (through
   `port_run`). Make the budget follow the grid, not a constant.
6. **Mesh:**
   - b1,024 reaches 2.3× one card at bfp8_b. The timing includes host arrays in and out, so the
     host prep, the readback and the unscale dominate.
   - Host prep and readback dominate. Overlap them with replay (`execute_pipelined`, `stream`),
     and avoid per-chip host round trips.
   - Row-major split before tilize is already done.

## Known losers (do not retry)

- **Accuracy:**
  - LoFi on QKV, FF-up or SDPA beyond what `performance()` already uses pushed max-row NRMSE
    over 0.04;
  - bfp4 weights;
  - SDPA chunk 64;
  - our custom RMSNorm kernels.
- **Slower:**
  - sharded `rms_norm` (layout conversions cost more than the norm);
  - DRAM-sharded matmul at b≤4;
  - K-split FF-down (2.6–3.7× slower);
  - grid gy4 at large M;
  - SDPA pass sizes 168 and 224;
  - threaded host overlap on the mesh.
- **Already in upstream:**
  - batch folded into M;
  - L1 chunks;
  - fused QKV head split + RoPE;
  - bank-local add;
  - bf8 RMSNorm output;
  - LoFi attention in `performance()`;
  - two-CQ input refresh.

## Working rules

- Read CONTRACT.md, LESSONS.md and UPSTREAM_PR.md first.
- Only `port_write` persists edits. Files over 24 KB (`tt/model.py`) need your native Read with
  an offset; `port_read` truncates.
- Keep STATE.md short: done, current hypothesis, pending job ids, next action.
- Read logs with `port_log` paging; do not rerun jobs to recover them.
- Compute stays on TT with real weights. Never exploit the corpus, cache answers or change the
  evaluator.
