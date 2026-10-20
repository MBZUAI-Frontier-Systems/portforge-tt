# REPORT: qs8-vlrelu → RVV (`xnn_qs8_vlrelu_ukernel__rvv_pf1`)

Campaign: clean-room port of the quantized leaky-ReLU unary family from the
scalar spec + Neon kernel to RVV, then optimize as far as the physics allows.
Final artifact: `kernel.c`, sha256
`d6d0fdb726437a5f140901c6564d6689c4c5b78af8831c41f4977ddcd6430b57`
(identical in the local checkout and on the box), registered from the
`neon_u8` template line (tile 8, `vector_tile=false`) as
`xnn_qs8_vlrelu_ukernel__rvv_pf1`.

All three gates green on the final kernel:

- gate 1 `evaluate.test`: **PASS**, 10 gtests × {qemu:128, qemu:256,
  qemu:512, qemu:1024, x280}, zero failures (receipt `test-20261019T182039Z`)
- gate 2 `evaluate.verify`: **ALL_PASSED**, 33/33 producer-true configs to
  batch 8, cvc5, vs the SALTyRN-prepped Neon source, VLEN 512, 108.9 s
  (receipt `verify-20261019T182110Z`)
- gate 3 `evaluate.bench`: **PASS**, X280, geomean **2.597x** vs the best
  upstream RVV variant (`rvv_u2v`); 2.599x / 2.595x per shape, 28.6x / 29.6x
  vs scalar (receipt `bench-20261019T182320Z`)

The clean-room rule was respected: `src/optimized-rvv-kernels/qs8-vlrelu.c`
was first opened only after the round-1 kernel had passed all three gates
(§7). Everything before that was derived from `scalar-select.c.in` (the
spec), `neon.c.in` (the oracle), and the family's upstream `rvv.c.in`.

## 1. Arithmetic derivation (scalar kernel = spec)

`xnn_qs8_vlrelu_ukernel__scalar_select_u1` computes, per element:

```
d    = (int32) x - input_zero_point
m    = (d >= 0) ? positive_multiplier : negative_multiplier   // sign test on the DE-BIASED value
vacc = (int32) (((uint32) (int32) output_zero_point) << 8) + 0x80 + d * m
y    = sat8( asr(vacc, 8) )                                  // arithmetic (floor) shift
```

i.e.

```
y = sat8( ozp + rhu( d * m(d) / 256 ) ),   rhu(z) = floor(z + 1/2).
```

`xnn_init_qs8_lrelu_scalar_params` produces, with int8-range zero points
(the verifier's own domain table says (-128, 127)):

- `positive_multiplier = lrintf(256 * in_scale / out_scale)` in **[1, 32768]**
  (asserts, inclusive both ends);
- `negative_multiplier = lrintf(256 * in_scale * slope / out_scale)` in
  **[-32767, 32768]**, `!= 0` (a 0 slope trips the init's assert, so a zero
  multiplier is not producible; slope can be negative, giving negative
  multipliers, and > 1, giving negative_multiplier > positive_multiplier).

The family's classic off-by-one is the sign test: it compares the DE-BIASED
`d = x - izp`, not the raw input; `d == 0` takes the positive multiplier,
where the product is 0 so the choice is immaterial.

**Neon (gate-2 oracle) equivalence.** `neon.c.in` computes the same map with
a negated-operand trick: `vacc = vsubw_s8(izp, x)` = `-d`;
`mask = vacc < 0` = `d > 0`; multiplier' = `-pos` on mask, `-neg` otherwise
(note `vdupq_n_s16(-multiplier)`; the negation maps the init's range
[-32768, 32768] into int16 exactly); `vqrdmulh(vacc << 7, m')` =
`rhu(d*m/256)` bit-exactly (the doubling rounding-multiply-high with RNU
equals `floor((d*2^7*m + 2^14)/2^15)`); `vqadd(+ozp)`; `vqmovn` = sat8.
Its intermediate saturations provably never trigger on the producer domain
(`|rhu(d*m/256)| <= 255`, `|result + ozp| <= 383`), so Neon == scalar ==
this kernel bit-exactly everywhere the formal gate looks.

**Width bounds** (used by every regime below): `|d| <= 255`;
`|d * m| <= 255 * 32768 = 2^23` (fits i32, hence the scalar is exact);
`|d << 7| <= 32640` (fits i16); `|rhu(d*m/256)| <= 255` for `|m| <= 32767`;
`|+ozp| <= 383` (fits i16); the only stage that ever saturates is the final
sat8, which is exactly the scalar's clamp.

## 2. Regime analysis and dispatch

The kernel dispatches on the per-call scalar params (fixed for one call) to
the fewest vector ops that is bit-exact on that regime's whole domain.
Three top-level regimes, complete and disjoint over the init's output range:

1. **`pos == neg`** (slope rounds to exactly 1.0; `lrintf` maps slope 1 to
   identical multipliers): the op is plain requantization with one
   multiplier m, and the qs8-vcvt round-2 regime ladder applies unchanged
   (same identity `rhu(a/256) = floor((a + 128)/256)`):
   - m == 256, `s = izp - ozp` in [-128, 127]: **A1**, 1 op, e8 saturating
     subtract (`vssub`): `y = sat8(x - s)`.
   - m == 256 otherwise: **A2**, 3 ops (`vwsub, vsadd, vnclip(0)`).
   - direct-i16 (`b = 256*ozp - m*izp`, predicate `b - 128*m >= -32768` and
     `b + 127*m <= 32767`): **C**, 4 ops (`vwadd(0), vmul(m), vadd(b),
     vnclip(8, RNU)`). The predicate keeps `m*x + b` in i16 for every int8
     x; the RNU narrowing contributes exactly the +0x80 rounding constant.
   - m = 2^p, p in [9, 15]: **D1**, 4 ops (`vwsub, vsll(p-8), vsadd,
     vnclip(0)`); the quotient m/256 = 2^e is exact, no rounding exists.
     e = 7 (m = 32768) reaches `|d << 7| = 32640`, `+ozp` exactly 32767.
   - everything else (1 <= m <= 32767): **D2**, 5 ops (`vwsub, vsll 7,
     vsmul(RNU), vsadd, vnclip(0)`).
2. **general two-sided**, both multipliers in [-32767, 32767]: the 7-op i16
   chain of the final kernel (§4), the benchmark's regime.
3. **a multiplier == 32768**: not representable in int16 (it reads as
   -32768). An i32 chain shaped like the upstream rvv template
   (`vsra(31)/vand/vxor` select, `vmacc` against a hoisted
   `0x80 + 256*ozp` bias vector, `vnclip(8, RDN)`, `vnclip(0)`); exact:
   `|d*m| <= 2^23`, no i32 overflow, no intermediate saturation. Runs at
   e8m2 in / i32m8 middle (m4 input would make the i32 stage an illegal
   m16 type - which is also why this family's upstream template tops out at
   u2v; the i16 chains here have no such ceiling).

Dispatch order: `pos == neg` first (so `pos == neg == 32768` takes the exact
D1 shift form, not the i32 fallback), then the int16-range general chain,
then i32. Every (pos, neg) the init can emit takes exactly one branch; the
33 configs of `params.json` map onto them (see `dump_params.c` comments).

Paper bounds per chain stage are in the kernel header; gate 2 proves each
config formally (the configs cover A1, A2, C incl. the m = 128 boundary and
C-ineligible zero points, D1 incl. m = 32768, D2, general incl. negative and
slope > 1 multipliers, and the i32 regime on both sides).

## 3. Why upstream has no u4v, and what this kernel adds

The family's upstream RVV variants are only `rvv_u1v` / `rvv_u2v`. The
generated template (`rvv.c.in`) hoists a **bias broadcast vector at i32**;
at LMUL 4 that would be an `i32m16` type - illegal (max LMUL 8). The i16
chains in this kernel never need an i32 stage (the fractional multiply is
`vsmul` on <<7-scaled i16), so the m4-width layout (e8m4 in, i16m8 middle -
a would-be u4v's footprint) is legal, and the kernel is the family's first
variant at that width. Only the correctness regime 3 needs i32, and it stays
at e8m2 in / i32m8, upstream's own u2v shape.

## 4. Design iterations (all measured; receipts in §5)

The benchmark (`bench/vunary.cc`) runs slope **0.5** with int8 quantization
in {zp 1, scale 0.5} / out {zp -1, scale 0.7}, so the measured params are
**pos 183, neg 91** - the *general two-sided* regime (unlike qs8-vcvt, whose
benchmark hit a fast single-multiplier regime). Every optimization below
targets that chain: `widen, sign-select, <<7, rounding multiply, +ozp,
narrow` = 6 semantic stages + load/store. The select is mandatory (two
different multipliers); the minimum proven chain is 7 vector ops.

| round | general chain | width | X280 @16K (cyc/el) | official bench vs u2v |
|---|---|---|---|---|
| v1 | `vmslt` mask + `vmerge_vxm`, hoisted i16m4 broadcast (7 ops) | e8m2/i16m4 | 0.62 | 2.031x |
| v2 | `vsra/vand/vxor` arithmetic select (8 ops) | e8m4/i16m8 | 0.62-0.63 | 1.994x |
| v3 = final | `vmsgt` on the raw i8 input + `vmerge_vxm`, hoisted i16m8 broadcast (7 ops) | e8m4/i16m8 | **0.51** | **2.597x** |

Findings along the way (each is a measurement, not a guess):

- **m2 -> m4 with the same op count buys ~nothing** (v1 7 ops @ m2 = 0.62;
  v2 8 ops @ m4 = 0.62): per-element ALU work dominates. **Fewer ops at the
  wider width is the win**: v3's 7 ops @ m4 = 0.51 (~1.25x from v1/v2).
- The select should ride on the **narrow** input: `vmsgt` on i8 + one merge
  (1 m4-width op + 1 m8-width op) beats sign arithmetic on the widened
  value (`vsra+vand+vxor`, 3 m8-width ops). The mask and the widen both read
  x; the select result then feeds only the multiply.
- The mask+merge select needs one broadcast vector live across the loop.
  At i16m8 that is an 8-register group; GCC keeps it loop-invariant when it
  is built with `vsetvlmax` (full-length, prefix-read by later iterations -
  vl is monotonically non-increasing). With the and/xor select (v2) I first
  assumed that group would force spills at m4/m8; it does not: the
  disassembly of both v2 and v3 has **zero group-spill moves**
  (`objdump -d`, no `vsd`/`vld` between the math).
- **`vmerge_vxm` operand order is mask=1 -> the SCALAR, mask=0 -> the vector
  base.** Round 3's first cut passed `vmerge(vpos_v, neg_scalar, mask)` -
  inverting every select. The pre-gate differential driver caught it (329
  failing (config, batch) pairs, all general-regime, e.g. in = -128, izp 1,
  pos 183, neg 91: got -93 = the pos-multiplier result, want -47) before any
  gate spent a run. Three configs escaped by luck: 256/128, 69/35 with
  zero points at both extremes saturate both branches to the same clamp
  (127) on most inputs - a reminder that saturation can mask select bugs.
- One difftest run initially reported all-pass on the *broken* variant: the
  static binary had been linked against the previous kernel.o (stale
  artifact, same trap as the qs8-vcvt REPORT §5). Re-linked and re-run
  before anything else; kept here for the record.
- Rejected without spending gates: folding ozp into the multiply (breaks
  the rounding identity unless m divides 2^15*ozp - not true for the bench
  params 183/91); folding the select into the shift (no hardware hook);
  2x software pipelining (the X280 general loop is throughput-bound, and
  the third-party kernel's 3-stage pipeline buys nothing measurable at the
  bench shapes, see §7); a |d|-based branch for slope = -1 (equal op count
  to the general chain).

Loop structure: one vsetvl-driven do-while per regime, tail = smaller vl of
the same code path, no scalar epilogue. In-place safe (load chunk, compute,
store same chunk). No OOB reads, no OOB writes (`vle8`/`vse8` touch exactly
vl elements; the family's Neon template is XNN_OOB_READS, this kernel does
not need it).

## 5. Gate receipts (run order preserved, nothing rewritten)

Kernel shas per round: v1/v2 were iterated in place (archive
reconstructions: `kernel-v1-m2-vmerge.c`, `kernel-v2-m4-andxor.c`, both
compile-clean; the final `kernel.c` is byte-identical on both machines,
sha `d6d0fdb7...`). Receipt dirs under `results/qs8-vlrelu/` on the box.

| # | receipt | kernel | result |
|---|---|---|---|
| 1 | test-20261019T180034Z | v1 | **PASS** 10 gtests × 5 wheres, first try |
| 2 | verify-20261019T180204Z | v1 | **ALL_PASSED** 33/33 configs to batch 8 (107.5 s) |
| 3 | bench-20261019T180414Z | v1 | **PASS** geomean 2.031x vs rvv_u2v |
| 4 | test-20261019T181322Z | v2 | **PASS** all wheres |
| 5 | verify-20261019T181348Z | v2 | **ALL_PASSED** 33/33 (98.7 s) |
| 6 | bench-20261019T181548Z | v2 | **PASS** geomean 1.994x (regressed vs v1: the 8th op ate the width gain) |
| 7 | test-20261019T182039Z | v3 = final | **PASS** all wheres |
| 8 | verify-20261019T182110Z | v3 = final | **ALL_PASSED** 33/33 to batch 8 (108.9 s) |
| 9 | bench-20261019T182320Z | v3 = final | **PASS** geomean **2.597x**, per-shape 2.599x / 2.595x |

Pre-gate differential driver (`difftest.c` + `difftest_build.sh`): the real
scalar kernel vs mine under QEMU at VLEN 128/256/512/1024, 33 configs × 31
batch sizes (1..12345, every power-of-two boundary and vl±1) = **1023
checks, 0 failures** on the final kernel (and on v1/v2 after the v3
inversion was fixed). It also checks upstream u1v/u2v against the scalar at
the bench config: MATCH (this family's upstream variants are i32-exact - no
qs8-vcvt-style truncation bug to report upstream).

**Gate 2 params**: `params.json`, 33 structs produced by the real
`xnn_init_qs8_lrelu_scalar_params` (compiled from the pinned tree by
`dump_params.c`, same pattern as the vcvt pack: pthreadpool stub include,
`-Wl,--gc-sections`). Coverage map (config -> regime): A1 <- {0/256/256/0};
A2 <- {-128/256/256/127}, {127/256/256/-128}; C <- {1/183/183/-1},
{0/64}, {0/1}, {-128/69/127}, {0/128} (pow2-but-C boundary);
D1 <- {0/1024}, {5/4096/3}, {0/32768} (e=7), {100/512/-100};
D2 <- {-128/128/127} (pow2 under 512, C-ineligible), {-5/32765/7},
{0/384}, {-3/396/2}; general <- {1/183/91/-1} (**the bench params**),
{256/128} × 2 zero-point sets, {1024/512}, {16384/8192}, {512/256},
{69/35}, {4096/2048}, {256/3}, {256/77}, {256/333} (gtest slopes),
{183/-183}, {183/-2} (negative multipliers), {183/238} (slope > 1);
i32 <- {16384/32768}, {32768/-32735}, {32768/9830} (nonzero zero points).
Every regime of §2 is formally proven on its configs; no MISSING_INTRINSIC
(all intrinsics used are shimmed, including `vsmul_vv_i16m8`,
`vmerge_vxm_i16m8`, `vmsgt_vx_i8m4_b2`).

## 6. Per-shape timings and the working-set curve (X280)

Official gate 3 (google-benchmark, taskset-pinned, 5 reps, median,
`bench-20261019T182320Z`):

```
shape        pf1(final)  scalar_u1   rvv_u1v   rvv_u2v   vs u2v   vs scalar
N:16320        3517.0    100641.9     2.865x    2.599x   2.599x    28.62x
N:130560      28182.8    834465.3     2.863x    2.595x   2.595x    29.61x
(ns)                                          geomean vs best upstream: 2.597x
```

Evaluator `passed` flag: true (faster than the best RVV variant on every
shape).

Working-set curve (`sizebench-vlrelu.c`, best-of-5 perf cycles, bench
params 183/91; GB/s = 2N bytes / ns at 1.753 GHz; the raw vcopy column is
the family's memory ceiling - one vle8 + one vse8, no arithmetic):

```
   N      scalar_u1     rvv_u1v     rvv_u2v     rvv_pf1    vcopy(ceiling)
 16320    13.35 cy/el   1.22 (2.9)  1.12 (3.1)  0.51 (6.8)  0.21-0.22 (16.2)
 130560   13.47         1.12 (3.1)  1.02 (3.4)  0.41 (8.5)  0.17-0.18 (19.5)
 1048576  13.57         1.32 (2.7)  1.22 (2.9)  0.97 (3.6)  0.83-0.95 (3.7-4.0)
16777216  13.63         3.09 (1.1)  3.04 (1.2)  3.04 (1.2)  2.90-3.03 (1.2)
```

Reading of the curve (the honest statement of where the ceiling is):

- **Compute-bound at and below the benchmark shapes**: pf1 runs 2.3-2.4x
  above the copy ceiling (0.51 vs 0.21 cy/el at 16K; 0.41 vs 0.17-0.18 at
  130K), so every removed vector op shows up in the timing - that is why
  the op-count lever paid 1.25x per op saved and why the kernel beats
  upstream by 2.2-2.6x exactly there.
- **Memory-bound from ~1M**: pf1 0.97 vs copy 0.83-0.95 at 1M and 3.04 vs
  2.90-3.03 at 16M - all variants converge to the ceiling; the remaining
  gap at 16M is run-to-run noise (the 16M rows of every variant and the
  copy agree within ~3%). Optimizing compute at DRAM-resident sizes is
  pointless, as the vcvt round-2 REPORT already concluded; the numbers
  above are that statement, measured.
- Upstream u1v vs u2v spacing (1.22 vs 1.12) reproduces the known
  per-instruction fixed cost; the pf1 -> ceiling gap is the 7-op chain's
  intrinsic compute, not fixable by width (m4 was the widest legal layout,
  §3).

Bit-exactness on hardware (`sizebench` spot checks, best-of-5 binary
compare vs the scalar kernel): pf1, u1v, u2v all MATCH at the bench config
and at the 32768/-32735 corner.

## 7. Post-hoc comparison: the third-party kernel (read only after round 1 passed all gates)

`src/optimized-rvv-kernels/qs8-vlrelu.c` (`test_rvv`) was opened after the
round-1 gates were green, per the clean-room rule. Findings:

- **Same core idea, independently derived**: a 7-op i16 chain
  (`vmsgt` on raw x, `vwsub`, `vmerge_vxm`, `vsll 7`, `vsmul_vv RNU`,
  `vsadd`, `vnclip`) at e8m4/i16m8 - the same minimal op set and width the
  final kernel converged to, which is reassuring for the derivation.
- **Correctness bug at the init's inclusive bound** (the qs8-vcvt disease,
  one family later): it truncates both multipliers to `int16_t`
  (`(int16_t) params->scalar.positive_multiplier` and the hoisted
  `vneg_mult` broadcast). At multiplier = 32768 - inside the init's
  asserted range - the value reads as -32768 and the affected side sign-
  flips. My kernel routes any 32768 multiplier to the exact i32 regime;
  three of the 33 gate-2 configs pin that corner (16384/32768,
  32768/-32735, 32768/9830) and pass formally, and the sizebench corner
  check confirms on hardware. For every config with both multipliers in
  [-32767, 32767] their kernel is bit-exact (verified by reading; its
  mask polarity `vmsgt(x, izp)` -> scalar pos is correct, unlike my first
  cut, and its in-place pipeline ordering is sound).
- **Speed**: head-to-head on the X280 at the bench params (added as a row
  to `sizebench-vlrelu-cmp.c`, best-of-5 cycles):
  16K: theirs 0.51 vs mine 0.52 cy/el; 130K: 0.41 vs 0.41; 1M: 0.84 vs
  0.85; 16M: 2.91 vs 2.91 - **statistically tied at every size** (the
  per-run cv is ~1-2%). Their 3-stage software pipeline (load i+1 /
  compute i / store i-1) does not measurably beat the plain loop: both are
  throughput-bound at these widths. What the final kernel adds is the
  regime dispatch (1-op requantize path at slope 1, exact i32 path at
  multiplier 32768) and the formal proofs per regime; what theirs has that
  mine does not: nothing functional, and its assert set skips the
  `batch % sizeof(int8_t)` check.

Net for an upstream PR: this kernel is a drop-in replacement for the
third-party design with the 32768 bug fixed and each regime formally
verified; at the benchmark shapes they are equally fast, both ~2.6x faster
than the best existing upstream variant.

## 8. Tooling notes (evaluator honesty; nothing modified)

- `run_gate.py` must be invoked from the repo root (`cd
  ~/portforge-cpu/repo`), not the workdir root - the kernel path is
  resolved against cwd.
- The mask-producing compare intrinsics need GCC's mask-suffix form
  (`__riscv_vmslt_vx_i16m4_b4`, `vmsgt_vx_i8m4_b2`); the verifier's shims
  use the same suffixed names - the earlier grep for unsuffixed names was
  my error, the coverage is complete.
- `vmerge_vxm(a, b, mask)` = `mask ? b : a` (scalar into vector base);
  the verifier shim documents this (`mask=1 -> b`) - worth a line in the
  RVV guide; it cost one differential round here.
- The static difftest/sizebench binaries must be relinked after every
  kernel edit (stale-artifact trap, §4).
- All nine gate receipts from this campaign are preserved under
  `results/qs8-vlrelu/` on the box, in run order, including the round-2
  bench regression (1.994x) - kept, not rewritten.

## 9. What an upstream PR would need

- Generator template: this family is generated; the natural packaging is a
  second body in `src/qs8-vlrelu/rvv.c.in` (or a sibling `rvv-wide.c.in`)
  emitting the i16 chains at e8m4/i16m8 plus the i32 body for the 32768
  corner, registered as VLA rows under the RVV guard (the portforge
  registration here clones the Neon u8 line - tile 8, vector_tile=false -
  because the tooling keys off the Neon template; same note as the vcvt
  pack).
- Test additions: a gtest config with a 32768 multiplier on either side
  (e.g. scales 128.0/1.0 with slope 2.0) would have caught the
  third-party kernel's truncation; a slope-1 config would exercise the
  1-op regime. Both are in `params.json` as formal configs already.
- Benchmarks: none needed; the family's `vunary` bench picked the kernel
  up automatically (the two shapes above).
