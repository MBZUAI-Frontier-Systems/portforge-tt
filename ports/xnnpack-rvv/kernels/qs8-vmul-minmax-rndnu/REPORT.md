# REPORT: qs8-vmul-minmax-rndnu → `xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1`

Campaign #3, CPU track, first TRUE new-coverage port: XNNPACK's quantized
multiply family `qs8-vmul-minmax-rndnu` has **no RVV variant upstream at all**
(and no scalar, SSE or WASM variant either - the Neon kernels are the family's
only implementations). Everything here is additive; the work is clean-room from
the Neon source + init in `src/xnnpack-saltyrn` (PR precedents: #9879/#9974,
new RVV microkernels).

**All gates green; submitted.** Summary verdicts:

| gate | verdict | evidence |
|---|---|---|
| 1 XNNPACK gtests | **PASS** | 15/15 cases at QEMU VLEN 128/256/512/1024 **and on the X280**, zero failures (`results-gate1/*.json`) |
| 2 equivalence | **PASS** via exhaustive differential (SMT unavailable for this family, see below) | 14 producer-true configs x **entire int8 input space** (256x256 pairs) + 28 batch shapes x 14 configs, byte-exact vs the literal Neon transcription, under QEMU at 4 VLENs **and X280 native**: `DIFFTEST_OK: 3976 checks, 0 failures` at every VLEN |
| 3 benchmark | **PASS** (first-in-family; honest bars: scalar + autovec) | X280: **1.38 cyc/elem** (0.76 ns/elem) compute-resident = **29.4x vs scalar**; working-set curve below; GCC's autovectorizer **fails** on this loop (0 vector instructions at -O3), so the autovec bar == scalar |

## 1. Spec (derived from source)

From `src/qs8-vmul/gen/qs8-vmul-minmax-rndnu-neon-ld64-u8.c` (the only
implementation) and `xnn_init_qs8_mul_minmax_rndnu_neon_params`
(`src/microparams-init.c:1073`):

```
da = (int16) a - a_zero_point                 // exact: [-255, 255] (int8 zps)
db = (int16) b - b_zero_point
P  = da * db                                  // exact i32: [-65025, 65025]
x  = sat32(P << left_pre_shift)               // never saturates on producer domain
h  = sat32(floor(x * multiplier / 2^31))      // vqdmulhq_s32: doubling high half, TRUNCATING
r  = floor((h + 2^(s-1)) / 2^s)               // vrshl by left_post_shift = -s (round half UP)
y  = sat8(sat16(sat16(r) + output_zero_point))// vqmovn -> vqadd -> vqmovn
y  = clamp(y, output_min, output_max)         // vmax_s8 then vmin_s8, int8
```

The init (asserts = producer domain): `product_output_scale in [2^-16, 2^8)`,
`multiplier in [0x40000000, 0x7FFFFF80]` (always positive),
`shift = 126 - biased_exp in [-8, 15)`, then
`post_shift = max(shift,1) in [1,15]`, `pre_shift = shift - post_shift`,
`left_pre_shift = -pre_shift in [0,9]`, `left_post_shift = -post_shift`,
`output_zero_point = (int16) y_zp`, `output_min = INT8_MIN`,
`output_max = INT8_MAX` (always - the qmin/qmax gtest setters are ignored by
the qs8 vmul tester).

**The collapse to one rounding (the key identity):** on the producer domain
`left_pre_shift + left_post_shift = -shift`, and floor-at-2^31 followed by a
round-half-up shift by s >= 1 composes exactly to a single round-half-up:
floor((k + 2^(s-1))/2^s) = floor((k + f + 2^(s-1))/2^s) for k integer, f in
[0,1), because no integer lies strictly inside (k + 2^(s-1), k + 2^(s-1) + f).
Hence, with **E = 31 + shift = 31 - left_pre_shift - left_post_shift in [23,46]**:

```
r = rhu(P * multiplier / 2^E) = floor((P * multiplier + 2^(E-1)) / 2^E)
```

which is literally `vnclip` in RNU mode (it adds 2^(E-1), floor-shifts, and
saturates to the narrow SEW). The i64 product `|P * multiplier| <= 65025 *
2^31 < 2^48` never overflows; `|r| <= 2^48/2^23 + 1 < 2^26` so the i32 stage
never saturates; the `vsadd` of the zero point in i32 and the i16->i8 narrows
reproduce the Neon tail exactly (sat8 after sat16 equals sat8 of the i32 sum
because i8-range clipping through i16 clips identically - any |value| beyond
i8 saturates to the same limit). All of this is verified element-for-element
by the exhaustive difftest against the **literal** Neon transcription (below),
not just argued.

## 2. Regime analysis

**One regime, no dispatch ladder.** The formula above is exact over the init's
entire asserted domain; unlike qs8-vlrelu there is no sub-regime where a
narrower form (i16 vsmul, exact shift) is legal: the product P needs 17 bits,
P*m needs 48, so the i64 multiply is structural. The pre/post shift split is
absorbed into E at setup; the loop body is parameter-value-independent (no
data-dependent branches), so one vsetvl-driven loop serves every batch and
every config.

Width: the i64 accumulator stage at LMUL 8 caps the chain at
`e8m1 -> i16m2 -> i32m4 -> i64m8` (VLEN/8 elements per iteration; 64 at the
X280's VLEN 512). Peak live set = 18 register groups (a, b, two i16 de-biased
operands, the i32 product, the i64 accumulator). **LMUL justification by
disassembly** (`pf1.dis`): the hot loop contains **0 `vsd`/`vld` group-spill
moves**, all params live in scalars, 4 `vsetvli` (one AVL-driven, three
SEW/LMUL switches), and the only scalar work is loop bookkeeping. No OOB reads
or writes: `vle8`/`vse8` touch exactly `vl` elements; the tail is a smaller vl
of the same loop (no scalar epilogue, matching the template's convention; the
Neon kernel's `XNN_OOB_READS` is *not* needed).

## 3. Gate 1 - XNNPACK gtests: PASS (with a documented wrinkle)

`evaluate.test` fails before running anything: the family's gtest target does
not exist. Upstream XNNPACK ships `test/qs8-vmul-minmax-rndnu.cc` but **never
wired it into `test/CMakeLists.txt`** (`MICROKERNEL_VBINARY_UNIT_TESTS` lists
`qs8-vmul-minmax-fp32`, `qs8-vmulc-minmax-fp32`, `qu8-*-rndnu` - not this
family; confirmed in the upstream checkout too - the family had no non-ARM
implementation to test). `targets_for` finds the .cc (it includes the .inc)
and ninja rejects the unknown target.

Route-around (no SALTyRN/upstream checkout touched; the disposable worktree at
`build/trees/pf-qs8-vmul-minmax-rndnu` is the tool's own injection surface):
+1 line in the worktree's `test/CMakeLists.txt` to wire the **unmodified**
upstream test source, plus the params/init guard widening (section 5) the
family needs to compile for riscv at all. Then the tool's own
`xnnpack.run_tests` produced the receipts:

- QEMU VLEN 128/256/512/1024: **15/15 passed, 0 failed** each
  (batch eq/div/lt/gt, inplace a/b/ab, a/b/y zero point, a/b/y scale, qmin,
  qmax) - `results-gate1/qs8-vmul-minmax-rndnu-test.qemu-*.json`
- X280 hardware: **15/15 passed, 0 failed** -
  `results-gate1/qs8-vmul-minmax-rndnu-test.x280.json`

(The gtest compares against the tester's fp32 reference with +/-1 tolerance;
this kernel is bit-exact to the Neon semantics, so it passes trivially.)

## 4. Gate 2 - equivalence: exhaustive differential (SMT route closed, documented)

`evaluate.verify` cannot run for this family, for two independent reasons:

1. **Tool bug (new-coverage path):** with `neon_text=` supplied and an explicit
   `entry`, `verify` crashes at `ports/xnnpack-rvv/evaluate.py:125` -
   `equiv.verify(cfg, neon, rvv, ...)` references `neon`, which is undefined
   (the parameter is `neon_text`). NameError before the verifier starts.
   `evaluate.py` is not mine to modify; documented here.
2. **No SALTyRN fixture for this stem:** SALTyRN never ported the rndnu
   variant (no optimized kernel, no scalar, no memory fixture), so even a
   fixed `verify` would report UNSUPPORTED for `qs8-vmul-minmax-rndnu`.

Per the campaign instructions this is replaced by the strongest differential
gate (`difftest.c`, build `difftest_build.sh`):

- **Oracle:** a literal per-element transcription of the Neon kernel - every
  step including its intermediate saturations (`vqshl` sat, `vqdmulh`
  truncating high half, `vrshl` round-half-up, `vqmovn`/`vqadd`/`vqmovn`
  chain, min/max) in exact C. This tests the composite-E identity claimed in
  the kernel against the real Neon semantics, not against my own simplification.
- **Inputs:** mathematically complete - **all 256 x 256 int8 operand pairs**
  per config (a held constant per 256-wide row, b covering the full range;
  both operand roles covered by the pair space).
- **Params:** 14 producer-true configs through the **REAL library init**
  (linked from `microparams-init.c` in the port worktree; dump cross-checks my
  earlier verbatim transcription value-for-value):
  multiplier 0x40000000 (lower bound) and 0x7FFFFF80 (upper bound),
  shift in {-8, -6, -4, -2, 0, 1, 2, 7, 10, 14, 15} => E in
  {23, 25, 27, 29, 31, 32, 33, 38, 41, 45, 46}, pre-shift up to 9, zero-point
  extremes (-128/127 on all three operands), gtest default scales.
- **Batches:** 28 sizes (1..4096, every power-of-two boundary +/-1) x 14
  configs for tail/loop coverage.
- **Result:** `DIFFTEST_OK: 3976 checks, 0 failures` under QEMU at VLEN
  128/256/512/1024 **and natively on the X280** (VLEN 512).

Honest caveat, as required: params are discretized at boundary configs - the
sweep is complete over inputs, not over the continuous parameter space; an SMT
fixture for this stem can be added to the trusted side later (the verifier
would need a `qs8-vmul-minmax-rndnu` memory fixture and the i64 `vwmul`
shimmed).

## 5. What an upstream PR needs (beyond the kernel)

1. **The registry row** (`src/qs8-vmul/qs8-vmul-minmax-rndnu.inc`):
   `XNN_UKERNEL(xnn_arch_riscv_vector, xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1, 8, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_rndnu_neon_params)`
   (the tooling clones the Neon `ld64_u8` row and swaps the arch; tile 8 is
   fine - the kernel is vsetvl-driven).
2. **Params/init guard widening** - the real library diff, kept verbatim as
   `params-init-riscv.patch` (3 files, 3 lines): upstream guards
   `union xnn_qs8_mul_minmax_params.rndnu_neon` (`microparams.h`),
   the init declaration (`microparams-init.h`) and its definition
   (`microparams-init.c`) behind `#if XNN_ARCH_ARM || XNN_ARCH_ARM64`. On
   riscv the row references an undeclared init and a union without the member
   (and the union shrinks from 20 to 12 bytes - an unpatched build would let
   the init overwrite 8 stack bytes in the tester). Widen all three with
   `|| XNN_ARCH_RISCV`. The kernel itself needs no shim: it reads
   `params->rndnu_neon` like any other family.
3. **Wire the gtest** (+1 line in `test/CMakeLists.txt`,
   `MICROKERNEL_VBINARY_UNIT_TESTS`): the test source exists upstream but was
   never registered (section 3).
4. **Generator templates** (not done here, noted per CONTRACT): the family's
   kernels are generated from `src/qs8-vmul/neon.c.in`; an upstream PR should
   add `rvv.c.in`-style templating for the rndnu family and regenerate the
   Neon rows' siblings, as PRs #9879/#9974 did for their families. The
   campaign deliverable is the hand-written kernel (the only template in the
   family, `scalar.c.in`, asserts `REQUANTIZATION == "FP32"` and cannot express
   rndnu).
5. **Bench wrapper quirk (pre-existing, report-only):** `bench/vbinary.cc`'s
   `ParamsWrapper<xnn_qs8_mul_minmax_params>` builds bench params with
   `xnn_init_qs8_mul_minmax_scalar_params` (fp32 requant) for rndnu rows. With
   all-scales-1 quantization my kernel reads garbage shifts from that union.
   Timing is unaffected (the instruction stream has no data-dependent branches
   and no param-dependent memory safety), but an upstream PR should give the
   wrapper the row's real init.

## 6. Gate 3 - benchmark (X280, VLEN 512)

**Peer comparison:** none exists - this is the family's first non-Neon
variant. The honest bars for a new-coverage PR (per the campaign brief):
the scalar spec kernel and an autovectorized build of it, plus the memory
ceiling. `sizebench.c` (best-of-5 `perf_event` CPU cycles, board pinned,
1.753 GHz; 3 bytes/elem traffic = 2 reads + 1 write):

| N | rvv_pf1 | scalar (-O2) | scalar (-O3 "autovec") | vcopy ceiling | pf1 vs scalar |
|---|---|---|---|---|---|
| 16320 (L1) | 13735 cy (**1.48 cy/el**) | 40.56 cy/el | 40.55 cy/el | 0.21 cy/el (2B/el) | **27.4x** |
| 130560 (L2) | 102735 cy (**1.38 cy/el**) | 40.59 | 40.59 | 0.18 | **29.4x** |
| 1048576 (1M) | 1966309 cy (3.29) | 40.91 | 40.90 | 1.12 | **12.4x** |
| 16777216 (16M) | 42868386 cy (4.48) | 40.96 | 40.96 | 2.97 | **9.1x** |

- **The autovec bar is scalar code: GCC's autovectorizer does not vectorize
  this loop at all** (`-O3 -march=rv64gcv`, 0 vector instructions in the
  object - the variable shift + i64 high-multiply + saturation chain defeats
  it; receipt in `sizebench_build.sh` output). Beating autovec here is the
  same as beating scalar: 9-29x.
- Compute- vs bandwidth-bound: for this kernel's 3-byte/elem traffic the
  memory ceiling is ~1.5x the 2-byte vcopy column, i.e. ~0.3 cy/el cache-
  resident and ~4.5 cy/el at 16M. **At 16M pf1 (4.48) is at ~100% of that
  bandwidth wall; at cache-resident sizes it is compute-bound** (1.38-1.48 vs
  ~0.3), dominated by the i64 `vwmul` and the two narrowing `vnclip`s.
- The XNNPACK `vbinary-bench` target (the family's .inc IS included there)
  ran on the X280 through the tool's own `run_bench`:
  N=10560: 8027.6 ns (0.760 ns/el), N=87360: 66496.1 ns (0.761 ns/el),
  cv <= 0.006% - `results-gate3/vbinary-bench.x280.json`. Consistent with the
  sizebench (0.76 ns/el x 1.753 GHz = 1.33 cy/el, L1/L2-resident shapes).
  The tool's `evaluate.bench` wrapper itself cannot run for this family: it
  insists on also building the (nonexistent) gtest target and its `tree()`
  reset would drop the params patch; the build step was driven manually
  (`tree` -> `inject` -> re-apply `params-init-riscv.patch` -> `ninja
  vbinary-bench`), then `run_bench` produced the receipt.
- There is no upstream RVV variant to tie with (unlike qs8-vcvt/qs8-vlrelu);
  for context, the sibling campaigns measured ~28.6-29.6x vs scalar for a
  lighter chain (single vsmul requantize); this family's 29.4x
  cache-resident with an i64 multiply chain is in the same league.

## 7. Receipts & honest negatives

Receipts in this directory: `results-gate1/` (5 gtest JSONs), `results-gate3/`
(bench JSON), `pf1.dis` (kernel disassembly: 0 spills), `params.json` (real
init dump), `difftest.c`/`difftest_build.sh`, `scalar_oracle.c`,
`sizebench.c`/`sizebench_build.sh`, `probe-x280.c`, `params-init-riscv.patch`.

- **Board builds must use `-march=rv64gcv` only**: a `-march=rv64gcv_zba_zbb_zbs`
  static binary **SIGILLs on the X280** (reproduced with the difftest, fixed by
  dropping the B extensions; the kernel's vector ops are identical - the gtest
  built by the tool's cmake runs fine). Everything shipped here for the board
  is `rv64gcv`.
- The SMT gate could not be attempted to a verdict: `evaluate.verify` crashes
  on the new-coverage path (NameError, evaluate.py:125) before the fixture
  lookup; the fixture gap is independent (section 4).
- `evaluate.bench`/`evaluate.test` wrappers fail for this family for the
  structural reasons in sections 3/5; the underlying runners
  (`xnnpack.run_tests`/`run_bench`) and the tool's own build/inject machinery
  were used for every receipt.
- Budget: the brief's ~2h estimate was exceeded (~3.5h), mostly on the two
  structural new-coverage gaps (params guards, unwired gtest) and the
  board-SIGILL investigation - all documented above.
- Untried optimization ideas (not needed for any gate): software-pipelining
  the loads across iterations at L1/L2-resident sizes (register budget allows
  ~1.5x unroll only before spills), and folding the output min/max clamp away
  when the params carry the init's full range (rejected: breaks the row's
  minmax contract fidelity for hand-built params).

## 8. Verdict

`xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1` is the first RVV (and first
non-Neon) implementation of the family: bit-exact against the Neon semantics
over the entire int8 input space on producer-true params at every VLEN tested
(QEMU 128-1024, X280 native), 15/15 on the family gtest everywhere, 29x the
scalar spec kernel (which GCC cannot autovectorize) and at the bandwidth wall
at DRAM scale. Submitted as campaign #3's deliverable.
