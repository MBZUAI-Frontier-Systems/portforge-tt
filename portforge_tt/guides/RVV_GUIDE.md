# RVV_GUIDE: porting and optimizing XNNPACK microkernels for RISC-V Vector

Things that worked, things that broke, and places where a new idea could win.
Learn the rules from existing kernels first; deviate only when you can measure
that you won. (X = XNNPACK upstream at the pinned revisions; S = SALTyRN's
published kernels; PR = the three merged XNNPACK RVV pull requests.)

## Read before writing

1. The scalar kernel is the specification. Write down its formula, including
   the rounding mode, clamp order, and where saturation happens, before you
   write a single intrinsic. Ambiguities you leave unresolved become bugs the
   solver will find later, at a much worse price.
2. Read the family's existing RVV kernels (most families have u1v/u2v/u4v
   variants) and the Neon kernel. You are not translating Neon mechanically —
   that is what Neon2RVV does, and it leaves 4/52 of the vector width used.
3. In XNNPACK, `batch` is a *byte* count for many families (check!). The tail
   is "batch != 0 after the main loop", not "batch < vl".

## Requantized int8 kernels (qs8-*/qu8-*) — where SALTyRN's kernels died

- The requantization chain is: widen to i32, multiply by the per-tensor
  multiplier, rounding arithmetic shift right, add output zero point, saturate
  to output type, clamp to [qmin, qmax]. RVV gives you exactly these:
  `vmul`/`vmacc` (i32), `vssra` with `__RISCV_VXRM_RNU` (rounding shift),
  `vnclip` (shift + saturate to narrow width).
- **Never narrow a parameter before the arithmetic.** S died twice here:
  `output_zero_point << shift` in int32 (127 << 30 wraps) in qs8-vadd, and
  `(int16_t)(multiplier << 7)` in qs8-vcvt (multiplier >= 256 flips signs).
  XNNPACK multipliers reach 2^21 and shifts 30; do the wide-domain math in
  i32/i64 exactly as wide as the scalar kernel does, and derive that width
  from the scalar code, not from the test values.
- **Do not add terms the scalar kernel already folded.** S's qu8-vadd added a
  rounding term on top of a bias that XNNPACK's init had already folded it
  into; every test failed. The params struct fields are *inputs*, their
  meanings are fixed by the init function.
- SEW/LMUL progression for the widening chain: e8m2 -> e16m4 -> e32m8 keeps
  the widest stage within LMUL=8 and is what the working kernels do. If
  registers spill (check the disassembly: `vsd`/`vld` storms between math),
  drop one LMUL step before unrolling differently.
- The rounding mode is a parameter of `vssra`/`vnclip`, not a global: match
  the scalar kernel's shift semantics (round-to-nearest-even is `RNU` for
  `vssra`; `vnclip` on the narrow step is usually `RDN` with a zero round bit —
  check the scalar's rounding, not a sibling kernel's).

## Float kernels

- Memory-bound families (vadd/vmul/vmax/...) at VLEN 512 will sit near 1.0x vs
  upstream no matter what you do; the win there is correctness or a better
  unroll, not fireworks. Say so in the REPORT instead of chasing noise.
- NaN is where float kernels go wrong: `vfmax`/`vfmin` differ from Neon's
  compare+select on ±0 and NaN payload ordering (argmaxpool returned one
  element's value with another's index; f16-vcvt turned NaNs into ±inf).
  If the scalar uses explicit compares, you use explicit compares; only use
  vfmin/vfmax if you have proven the family is NaN-free by construction.
- `--input-range` fencing in the verifier exists because FP-multiply families
  explode the solver on NaN/inf paths. Fence, don't ignore: an UNSAT inside a
  fenced range plus a special-values driver outside it is the honest pair.
- FMA: RVV `vfmacc` is fused; Neon `vfmaq` under AArch64 is fused but `vmlaq`
  is not. A fused-vs-unfused mismatch shows up as single-ulp differences on
  subnormals — the qd8-gemm counterexample nobody has explained yet.

## Structure and speed (ranked by measured payoff)

1. One `vsetvl` per outer loop, driven by the remaining count; the tail is a
   smaller `vl`, never a scalar copy of the loop. (Everything merged does this.)
2. Process width vs register pressure: e32m8 fills the register file; e32m4
   with 2x unroll often beats it once accumulators appear. Measure both; S's
   wins over upstream (1.5-2.6x on quantized kernels) were mostly wider
   stages + fewer stores, and their loses were spills.
3. `vmacc` over `vmul`+`vadd`; immediate/scalar forms (`vx`) over vector forms
   when one operand is a broadcast param.
4. Segment loads (`vlseg`) for interleaved layouts, `vslidedown` for
   broadcast-within-vector; S's x32-transposec won with these and lost to the
   hand-written 32x8 elsewhere — study PR #9996 for the shape of an honest
   optimization PR.
5. Tail policy: `_ta` (agnostic) destinations when the tail is fully masked
   out at the store; `_tu` needs the previous value and costs a dependency.
   Note the verifier has no shims for `_tu`/`_mu` — using them costs you gate 2.
6. Compile with `-march=rv64gcv_zba_zbb_zbs` at `-O2`; check the disassembly
   (`riscv64-linux-gnu-objdump -d`) for scalar fallbacks and spills. The
   anti-cheat check greps for exactly that: the hot loop must be vector.

## What would be novel (unsolved, measured, honest)

- VLEN-adaptive tile selection at runtime: choose u-variant from `vlenb` once
  per process — upstream picks at compile time; a measured runtime pick could
  win across the 128..1024 range. (Must be a pure speed choice, never semantic.)
- Tail-free kernels via guaranteed-multiple batch configs (register tiles
  matching XNNPACK's padding) — check whether the harness lets you promise it.
- Better segmented access for packw/transposec; a correct f32-dwconv with
  vector tiles (S's is broken at partial channel groups — the packing contract
  is documented in the dwconv header).
- Beating upstream where S claimed speed but was wrong (qs8-vcvt, qu8-vadd):
  a correct kernel at their measured speed would be an immediate upstream PR.

## Process law (from this campaign's history)

- Parameters for the formal gate must come from the library's init code. Every
  published kernel that skipped this is wrong.
- XNNPACK's gtests are necessary, not sufficient: they passed the broken
  qs8-vadd. The trio (gtests + SMT with real params + special-value driver) is
  the gate set; passing two of three means "not done".
- Keep every gate receipt. A speed claim without a receipt is a rumor.
