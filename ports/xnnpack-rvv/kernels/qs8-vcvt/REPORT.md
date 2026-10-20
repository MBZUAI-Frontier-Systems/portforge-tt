# REPORT: qs8-vcvt → RVV (`xnn_qs8_vcvt_ukernel__rvv_pf1`)

Campaign: fix-and-port the family SALTyRN got wrong, then package it for
upstream. Final artifact: `kernel.c` (sha256 `83b4aab051b53a3a…`, identical in
the local checkout and on the box), registered from the `neon_u8` template
line as `xnn_qs8_vcvt_ukernel__rvv_pf1`.

## 1. Arithmetic derivation (scalar kernel = spec)

`xnn_qs8_vcvt_ukernel__scalar_u1` (and u2/u4) compute, per element:

```
vbias = (int32)(((uint32)(int32) output_zero_point) << 8)
        - multiplier * (int32) input_zero_point + 0x80
y     = sat8( (vbias + x * multiplier) >> 8 )       // arithmetic (floor) shift
```

- The shift is fixed at 8; `xnn_init_qs8_cvt_scalar_params` pre-scales the
  multiplier: `multiplier = lrintf(256 * input_scale / output_scale)`, asserted
  to `[1, 32768]`. Zero points are int8-range in every producer path (the
  params struct says int16_t; the verifier's own domain table says
  `(-128, 127)`).
- `+0x80` is the round-half-up constant for `>> 8`, so the shift must truncate
  (floor): `vsmul` with RNU (ties up) is the exact RVV equivalent; a rounding
  shift *on top of* the +0x80 would round twice.
- **Width bound:** `|x - izp| ≤ 255`, `mult ≤ 32768` ⟹ `|(x-izp)*mult| ≤ 2^23`,
  everything before the final clamp fits int32 with margin — the scalar is
  exact, and in i16 the only stages that could overflow are the multiplier
  itself (see §2) — hence two multiplier regimes in the kernel:
  - `multiplier ∈ [1, 32767]`: i16 chain `vwsub(izp) → <<7 → vsmul(RNU) →
    vsadd(ozp) → vnclip(sat8)`. Each stage exact or redundant in range
    (`|(x-izp)<<7| ≤ 32640` fits i16; the vsmul result ≤ 32640 fits i16;
    `+ozp` fits i16), so it is bit-exact against the scalar and the Neon
    kernel; intermediate saturations cannot trigger for producer params.
  - `multiplier == 32768`: not representable as an i16 vsmul operand.
    `(x-izp) * 32768/256 = (x-izp)*128` is an exact integer, so the multiply
    and its rounding drop out; the chain is `vwsub → <<7 → vsadd → vnclip`.
- The final `sat8` via narrowing `vnclip` equals the scalar's
  `max(-128)/min(127)` before the `(int8_t)` cast (clamping twice to nested
  ranges equals clamping once to the inner range).

Final loop: one `vsetvl`-driven loop per regime, tail = smaller `vl` of the
same code path, no scalar epilogue, no OOB reads, e8m4 in / i16m8 math
(widest legal layout — the i32 alternative caps at e8m2, §4).

## 2. Why SALTyRN's kernel failed

`src/optimized-rvv-kernels/qs8-vcvt.c` computes
`shifted_multiplier = (int16_t)((int32_t)multiplier << 7)` and feeds it to
`vsmul`:

1. **int16 overflow on the pre-shift.** XNNPACK produces `multiplier ≥ 256`
   for any scale ratio ≥ 1 (the default 1.0/1.0 config has multiplier 256!).
   `256 << 7 = 32768` already wraps negative in int16; `512/1024 << 7` wrap to
   0/negative — outputs flip sign or zero out exactly where XNNPACK's asserts
   say such multipliers are normal. This is the RVV_GUIDE trap: never narrow a
   parameter before the arithmetic.
2. **Semantic confusion of `vsmul`:** it already shifts by SEW-1; the extra
   `<<7` double-shifts even when the value happened to fit (multiplier < 256).

Side finding for upstream: the *existing upstream* RVV kernels
(`rvv_u1v/u2v/u4v`) load `const int16_t multiplier = params->scalar.multiplier`
— an int16 truncation of the int32 field. At `multiplier == 32768` (the init's
inclusive bound, e.g. input scale = 128 × output scale) the multiplier becomes
-32768 and their `vsmul` chain flips the sign of every output; the scalar and
Neon kernels are exact there. The gtest never hits it (its ratios give
multiplier 4..16384), which is exactly the "the tests never do that" trap the
CONTRACT warns about. My kernel handles it as a first-class regime, and gate 2
formally verifies the equivalence against the Neon source *at that config*.

## 3. Gate receipts (appended in run order, nothing rewritten)

Wrapper: `run_gate.py` = 20-line script calling the documented
`evaluate.test / verify / bench`; the evaluator was never modified.

**Gate 1 — XNNPACK gtests (`evaluate.test`)**, kernel = XNNPACK's own test
matrix under QEMU at VLEN 128/256/512/1024 and on the X280 (VLEN 512
hardware), 8 gtests per kernel (batch eq/div/lt/gt, output/input scale,
output/input zero point):

| # | kernel variant | result |
|---|---|---|
| 1 | v1: exact-i32 chain, e8m2→e32m8 | **PASS** all wheres, first try |
| 2 | v2: two-regime i16 chain, e8m4 (= final design) | **PASS** all wheres |
| 3 | load-ahead first cut (lookahead folded into the `do-while`) | **FAIL** at QEMU 256/512/1024 + X280: 4 of 8 tests, e.g. `batch_eq` at i=128/256: `y=33, y_ref=-41, tol=1` — the epilogue-less lookahead left the last chunk unprocessed. Root-caused (loop exits when `batch` hits 0 after the *lookahead* load), fixed by moving the final chunk's compute after the loop. Failure kept verbatim in the receipt logs and the artifact `kernel-loadahead-firstcut-gtestfail.c`. |
| 4 | load-ahead fixed (while + epilogue) | **PASS** all wheres |
| 5 | **final `kernel.c`** (plain two-regime; sha `83b4aab0…`) | **PASS** all wheres (receipt: gate-1 rerun after the archive mix-up in §5, on the exact submitted hash) |

**Gate 2 — bounded formal equivalence (`evaluate.verify`)**: cvc5, configs
mode, vs the SALTyRN-prepped Neon u8 source, VLEN 512, `max_batch=8`,
budget 1500 s.

- Params: `params.json`, 17 structs produced by the real
  `xnn_init_qs8_cvt_scalar_params`, compiled from the pinned tree by
  `dump_params.c` (no hand-invented values). Coverage: multiplier extremes
  1 / 32768 / 32765, the gtest ratios (4..16384), non-power ratios
  (384, 396, 69), zero-point extremes ±128/±127 and opposite-sign pairs.
- v1: **ALL_PASSED**, 17/17 configs verified to batch 8, 17.5 s.
- v2 (two-regime): **ALL_PASSED**, 17/17 to batch 8 (both regimes exercised:
  mult ≤ 32767 configs take the vsmul path, mult = 32768 takes the exact path).
- load-ahead fixed: **ALL_PASSED** (superseded).
- **Final kernel.c: ALL_PASSED, 17/17 configs to batch 8, 19.1 s.**
  Notably `multiplier=32768` is *in* the verified config set — the config
  where all three upstream RVV variants cannot match the Neon oracle.

**Gate 3 — benchmark (`evaluate.bench`), X280 hardware** (taskset-pinned,
5 repetitions, median). Final kernel, two consecutive runs:

```
shape        pf1(final)  scalar_u1  scalar_u4   rvv_u1v  rvv_u2v  rvv_u4v
N:16320        2815.5      84281      77288       5845     3296     2811
N:130560      22618.8     711708     647604      46829    26408    22621
(ns; run 133121Z: 2816 / 22619 — same within 0.15%)

vs scalar_u4:  27.5x (N:16320)   28.6x (N:130560)
vs scalar_u1:  29.9x             31.5x
vs rvv_u1v:    2.08x geomean
vs rvv_u2v:    1.17x geomean
vs rvv_u4v:    0.999 geomean  (N:16320 0.998, N:130560 1.000)
```

Verdict: **clearly faster than scalar (27-32x)**; **statistically tied with
the best upstream RVV variant** (u4v): pf1 22618.8 vs u4v 22621 ns and 2815.5
vs 2811 ns — differences ≤ 0.15% across runs, i.e. measurement noise
(per-run cv ≈ 0.5%). The evaluator's mechanical `passed` flag reads false only
because one shape landed at 0.998 (< 1.0 by 0.2%). This is expected: the final
loop is instruction-for-instruction isomorphic to u4v's generated loop
(verified by disassembly: `vsetvli, vle8, vwsub, vsll, vsmul, vsadd, vnclip,
vse8` + identical scalar bookkeeping) — same op count, same width, no spills.
What the kernel adds is the multiplier-32768 regime (u4v sign-flips there) at
identical steady-state speed; that is the upstream argument per CONTRACT
("not slower than the best RVV variant unless REPORT.md argues why upstream
still wants the kernel").

Rejected on measurement (kept in history per the honesty rules):

| variant | bench result | why rejected |
|---|---|---|
| v1 exact-i32 (e8m2→e32m8) | 15.6x / 15.0x vs scalar, **0.501x vs u4v** | the vf4 widening caps input at e8m2 = half of u4v's lanes; correct but half speed |
| load-ahead first cut | gate-1 FAIL (last chunk unprocessed) | gtest caught it; fixed below |
| load-ahead fixed (while+epilogue) | **0.961x vs u4v** — regressed | GCC protects the live chunk with a `vmv4r` 4-register group copy plus an extra `vsetvli` per iteration; that costs more than the hidden load latency (the plain loop is throughput-bound, not latency-bound). Reverted. |

## 4. Per-shape timing (final, X280)

See the table in §3. Both benchmark shapes (L1/L2 characteristic sizes
16320 and 130560 elements) with the benchmark's default quantization
(multiplier 256 — a multiplier the SALTyRN kernel gets wrong, note).

## 5. Process note (archive mix-up, for the record)

While iterating I mislabeled the kernel archive files: `kernel-2path-e8m4-v2.c`
came to hold the *broken* lookahead variant, so two bench runs (receipts
132005Z/132344Z, 0.964-0.972) measured the broken lookahead kernel, not the
plain design, and one gate-1 "PASS" I first attributed to the final design was
actually the fixed load-ahead variant. I reconstructed the plain two-regime
kernel, confirmed the submitted sha256 `83b4aab0…` on both machines, and
re-ran all three gates on that exact file (receipts in §3, rows marked
"final"). The archived variants are renamed honestly:
`kernel-i32-e8m2-v1.c`, `kernel-loadahead-firstcut-gtestfail.c`,
`kernel-loadahead-pipelined-rejected.c`.

## 6. Tooling notes (evaluator honesty, nothing modified)

- `evaluate.bench` crashes in its *reporting* line
  (`evaluate.py:159: TypeError: 'float' object is not subscriptable` —
  `peers` maps name → per-shape float, then indexes it again). All
  measurements complete and are written to disk before the crash; I computed
  the per-shape table and the `compare_bench` geomeans from the raw
  `vunary-bench.x280.json` receipts with the same formulas
  (`kernels/qs8-vcvt/bench_table.py`). Recommend
  `round(t[shape]/ns, …)` → `round(t/ns, …)` upstream of this pack.
- `dump_params.c` needs the `pthreadpool.h` stub from
  `research/qs8_vadd_smoke` (xnnpack.h includes it for one opaque type) and
  `-Wl,--gc-sections` so unreferenced init functions in
  `microparams-init.c` don't drag in f16 helpers. Documented in its header.
- The formal gate's per-kernel fixture derived from the stem `qs8-vcvt`
  worked as documented; cvc5 proved all 17 configs × batches 1-8 in ~19 s.

## 7. What an upstream PR would need

- **Generator template:** this family is generated from `src/qs8-vcvt/rvv.c.in`
  via `tools/xngen` (one body expanded into u1v/u2v/u4v). The i16 chain here is
  the u4v body plus the multiplier-32768 regime; the natural packaging is a
  second body in `rvv.c.in` (or a sibling `rvv-fixed.c.in`) emitting the
  two-regime kernel at e8m4, replacing or joining u1v/u2v/u4v. The `.inc` row:
  `XNN_UKERNEL(xnn_arch_riscv_vector, xnn_qs8_vcvt_ukernel__rvv_<suffix>, <tile>, true, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)`.
  (The portforge registration here clones the Neon u8 line — tile 8,
  `vector_tile=false` — because the tooling keys off the Neon template; a PR
  would register VLA rows `(4, true)`-style under the RVV guard.)
- **Test additions:** a gtest case with multiplier 32768 (input/output scales
  128.0/1.0) would have caught the upstream int16 truncation; add it in the
  same PR. Also worth a gtest sweep at multiplier 255/256/257 (the SALTyRN
  breaking point).
- **Selection/argument:** upstream should either adopt this two-regime body
  for the RVV slot (perf-neutral vs u4v, correct across the full init
  envelope) or minimally fix u1v/u2v/u4v's int16 multiplier truncation; the
  first is strictly better since it also removes the sign-flip at
  multiplier = 32768 that all three current variants share.
- **Benchmarks:** no new benchmark needed; the family's `vunary` bench
  registered my kernel automatically (shapes N:16320/N:130560 above).

## Round 2 (`kernel-v2.c`, `xnn_qs8_vcvt_ukernel__rvv_pf2` — beats rvv_u4v 1.178x geomean)

Goal: beat the hand-written upstream rvv_u4v by a real margin at the standard
shapes. Result: **1.178x / 1.179x per shape, 1.178x geomean, all three gates
green** (gate 3's mechanical pass flag true for every shape). Artifact:
`kernel-v2.c`, sha256 `0c807425d968c97c058356328a01dcaa97827b864ec1da230fefc334ef113d4b`
(identical locally and on the box).

### 1. The benchmark multiplier is 183, not 256 (corrects §4 above)

§4 said the benchmark's default quantization gives multiplier 256. That is
wrong — it assumed the gtest's 1.0/1.0 scales. The vunary benchmark
(`bench/vunary.cc`) specializes int8 quantization to input `{zp 1, 0.5}` /
output `{zp -1, 0.7}`, so the pinned init (`src/microparams-init.c:1249`,
`lrintf(256 * 0.5f/0.7f)`) yields **multiplier 183** with `izp=1, ozp=-1`
(reproduced with the real init semantics: ratio f32 0.714285731 → 183).
Multipliers 183 and 256 both ran round-1's full 5-op chain, so the error cost
round 1 nothing — but it hid the actual optimization target: an odd multiplier
in the general regime, exactly where pf1 was instruction-for-instruction
isomorphic to u4v.

### 2. Derivation of the fast regimes

Spec (§1): `y = sat8( rhu((x - izp) * m / 256) + ozp )`, `rhu(z) = floor(z + 1/2)`,
`m ∈ [1, 32768]`, zero points and x int8-range. The kernel is compute-bound at
the bench shapes (copy ceiling 20.8-21.4 GB/s vs 10.3-10.4 GB/s for the 5-op
chain; X280 executes vector ops in whole LMUL-sized groups, so one op removed
is ~20% there), so every regime below dispatches on the per-call scalar params
to the fewest ops that stays bit-exact on the regime's whole domain:

- **A (m = 256, s = izp - ozp ∈ [-128, 127]):** the division by 256 is exact
  (m·d/256 = d for every d), the rounding drops out, and y = sat8(x - s) is
  ONE op: e8 saturating subtract (`vssub_vx`). s fits the int8 scalar operand
  by the regime condition. Disassembled loop: `vsetvli, vle8, vssub.vx,
  vse8`. Measured 2.02-2.12x vs u4v (§4 table) — essentially the vcopy
  ceiling.
- **B (m = 256, s outside int8):** y = sat8((x - izp) + ozp) in i16:
  `vwsub, vsadd, vnclip(0)` — 3 ops. |(x-izp)+ozp| ≤ 383, no intermediate
  saturation.
- **C (direct i16, m ≠ 256, v(x) = m·x + b fits i16 for all int8 x, b =
  256·ozp - m·izp):** move both zero points into a scalar bias:
  y = sat8( floor((m·x + b + 128) / 256) ) — the +0x80 rounding constant is
  supplied by the narrowing op itself (`vnclip` shift 8 in RNU mode adds
  exactly 2^7 before the shift). Chain `vwadd(0), vmul(m), vadd(b),
  vnclip(8, RNU)` — 4 ops, no fractional multiply. The i16 range predicate is
  `b - 128·m ≥ -32768 ∧ b + 127·m ≤ 32767`; it implies b fits int16 (v(0) =
  b). vmul/vadd are exact mod 2^16 and the predicate guarantees the true sum
  fits i16, so wrapped intermediates are harmless. **This is the benchmark
  regime: m=183, b=-439, v ∈ [-23863, 22802].** Disassembled loop:
  `vsetvli(e16), vle8, vsext.vf2, vmul.vx, vadd.vx, vsetvli(e8),
  vnclip.wi 8, vse8` — 4 vector ALU ops vs u4v's 5, no spills.
- **D1 (m = 2^p, p ∈ [9, 15]):** m/256 = 2^e (e ∈ [1,7]) is exact, multiply
  and rounding drop out: `vwsub, vsll(e), vsadd, vnclip(0)` — 4 ops.
  |(x-izp) << e| ≤ 255·2^7 = 32640, +ozp reaches 32767 exactly: no i16 stage
  can overflow for producer params. This generalizes round-1's m=32768 regime
  (which upstream's rvv variants still get wrong via int16 truncation; kept
  first-class here).
- **D2 (everything else, m ≤ 32767):** the §1 five-op vsmul chain, unchanged
  and proven there. The dispatch is complete and disjoint over the init's
  whole output range: A/B take m=256; C takes what fits i16; D1 the powers of
  two ≥ 512 (C cannot take them: 128·m > 32767); D2 the rest. A power of two
  below 512 with extreme zero points falls to D2 (its exact quotient would
  need rounding: e.g. m=128 ⇒ d/2), which the vsmul chain handles.

Dead ends (derived, then rejected without spending a gate run): folding the
bias into the widen's scalar (i8 operand, |b| ≫ 127); folding it into
`vsmul` via a pre-biased input (needs m | 128·(127 + 256·ozp) — false for
183); a 3-op chain via `vmacc` with a hoisted bias vector (vmacc destroys its
accumulator register, so the broadcast costs one op per iteration — no win);
byte-packed dual-element lanes (vnclip narrows whole i16 values, the high
byte pollutes); float rounds (RNE ties differ from half-up). A b == 0 3-op
sub-regime of C (`vwadd, vmul, vnclip`) is possible but was skipped: it is
never hit at the bench params and the kernel was already gates-green.

### 3. Gate receipts (round 2, run order preserved)

Wrapper `run_gate.py` (round-1 §3); evaluator unmodified. Kernel sha
`0c807425…` on both machines (receipt dirs under `results/qs8-vcvt/` on the
box).

**Gate 1 — `evaluate.test`: PASS, 40/40, first try** (receipt
`test-20261019T164746Z`): 8 gtests × {qemu:128, qemu:256, qemu:512,
qemu:1024, x280}, zero failures. Regime hit map: batch eq/div/lt/gt +
input/output zero-point tests run m=256 with zps {0}/{2,5,8} → regime A;
output_scale (m ∈ {64, 16, 4}, b=0) → regime C; input_scale (m ∈ {1024,
4096, 16384}) → regime D1. B and D2 are not gtest-reachable (the gtest never
combines m=256 with |s| > 127, nor a non-pow2 m ≥ 512): both are formally
verified in gate 2 and argued here per CONTRACT's regime-dispatch rule.

**Gate 2 — `evaluate.verify` (cvc5, configs mode, Neon oracle, VLEN 512):
ALL_PASSED, 17/17 configs to batch 8, 17.2 s** (receipt
`verify-20261019T164813Z`). Same `params.json` as round 1 (17 producer-true
structs; untouched). Regime → config coverage: A ← {0,256,0}, {-128,256,-128},
{127,256,127}; B ← {-128,256,127} (s=-255); C ← {0,64,0}, {0,4,0}, {0,1,0},
{9,1,-9}; D1 ← {0,1024,0}, {5,4096,3}, {0,16384,0}, {0,32768,0},
{100,512,-100}; D2 ← {-5,32765,7}, {0,384,0}, {-3,396,2}, {-128,69,127}.
Every regime is formally proven on its config set; no MISSING_INTRINSIC
(all five new intrinsics — `vssub_vx_i8m4`, `vwadd_vx_i16m8`, `vmul_vx_i16m8`,
`vadd_vx_i16m8`, and the shift-8 RNU `vnclip_wx_i8m4` — are shimmed in the
verifier).

**Gate 3 — `evaluate.bench` (X280, taskset-pinned, 5 reps, median): PASS**
(receipt `bench-20261019T164852Z`):

```
shape        pf2       rvv_u4v  vs u4v   scalar_u1  vs scalar   u1v      u2v
N:16320      2386.8     2810.9   1.178x   81938.3     34.3x     2.45x    1.38x
N:130560    19191.4    22619.0   1.179x  686416.4     35.8x     2.44x    1.38x
geomean vs best upstream RVV (rvv_u4v): 1.178x   (evaluator "passed": true)
```

The benchmark parameters (m=183, s=+2) take regime C — the 4-op direct-i16
chain — which is where the 1.18x comes from (predicted ~1.25x from the pure
op-count model; the last ~5% is the two-`vsetvli` overhead and memory
overlap the model ignores).

### 4. Pre-gate de-risk measurement (sizebench-v2, X280, best-of-5 cycles)

`sizebench-v2.c` (kept next to round-1's `sizebench.c`) timed all variants at
the REAL bench params {izp=1, m=183, ozp=-1} and at {izp=5, m=256, ozp=-3},
and diffed pf2 against the scalar oracle over all 17 gate-2 configs plus the
gtest-equivalent param sets (32 configs, int8 edges pinned): **0 mismatches**;
m=32768 pf2 MATCH / u4v DIFFER (upstream truncation, as in round 1). Cycle
ratios pf2/u4v: 1.114 (N:16320) and 1.159 (N:130560) at m=183; 2.12/2.02 at
m=256 (regime A ≈ copy ceiling). The official gate 3 shows a cleaner 1.178x
because its harness carries less per-iteration overhead than perf-cycle
normalization at these sizes — direction and magnitude agree.

### 5. Honest limits

- The win exists only where the family is compute-bound. At N ≥ 1M (DRAM)
  everything converges to the copy ceiling (~1.0x, sizebench-v2 rows
  1048576/16777216 for both param sets); the benchmark shapes are exactly
  where it pays.
- The 2x regime A speed at m=256 is real but the official benchmark never
  measures m=256; the headline 1.178x is the C regime. A producer running
  scale ratio 1.0 (m=256, equal zero points) gets the 2x.
- sizebench-v2's m=256 row shows pf1 ≈ u4v ≈ 25.1k cycles vs pf2 12.4k: the
  round-1 tie was a property of the 5-op chain, not of the problem.

