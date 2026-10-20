# TASK: port and fix the qs8-vcvt RVV kernel

You are porting XNNPACK's `qs8-vcvt` (quantized convert with requantization)
microkernel family to RVV. SALTyRN's published optimized kernel for this family
is **wrong**: `(int16_t)(multiplier << 7)` overflows int16 as soon as the
scale ratio is >= 1 (multiplier >= 256), flipping every output's sign. XNNPACK
produces such multipliers routinely. Your job: produce a kernel that is *correct
for the whole envelope* and *fast*, then package it.

Work on the x86 host (`ssh tt_box`), repo at `~/portforge-cpu/repo`, workdir
`~/portforge-cpu`. All tools are Python: `from portforge_tt.cpu import ...` with
the venv or system python3 from the repo root. Read
`portforge_tt/cpu/README.md` first, then the modules you use. Edit files in the
repo checkout on the box (or edit locally and rsync; the box copy is what runs).

## Steps

1. **Study the family** (~30 min). Read, under `src/xnnpack-saltyrn`:
   `src/qs8-vcvt/qs8-vcvt.inc`, the scalar kernel
   (`gen/qs8-vcvt-scalar-*.c`), the Neon kernel, and the three existing RVV
   kernels (`rvv_u1v`, `rvv_u2v`, `rvv_u4v`). Understand the exact arithmetic:
   requantization is `y = sat<qmin,qmax>( round<>((x*mult) >> shift) + bias-ish )`
   — read the scalar kernel and `xnn_init_qs8_cvt_scalar_params` and write the
   formula in your notes before coding. Read SALTyRN's broken kernel
   (`src/optimized-rvv-kernels/qs8-vcvt.c`) and identify exactly why it wraps.
2. **Write the first correct kernel** in `kernels/qs8-vcvt/kernel.c`. Simplest
   correct skeleton: e8m2/e16m4/e32m8-style widening like the upstream vbinary
   RVV kernels do for qs8 — do the multiply-accumulate in i32 (or i64 if
   needed), use `vssra`/`vnclip` for the rounding shift with the VXRM the
   scalar kernel implies, apply bias in the wide domain, clamp last. Derive the
   rounding mode from the scalar code; do not guess.
3. **Gate 1** — `evaluate.test`: register as
   `xnn_qs8_vcvt_ukernel__rvv_<suffix>` from the Neon template and run the
   family gtest at VLEN 128/256/512/1024 + X280. Fix until green.
4. **Gate 2** — `evaluate.verify`: cvc5 vs the Neon source, configs mode with
   XNNPACK-produced params (write the tiny `--dump-params`-style helper for the
   cvt params struct; copy the pattern from `research/qs8_vadd_smoke/`),
   `max_batch` 8, budget 600+. Must be `ALL_PASSED`.
5. **Optimize.** Now that it is correct, make it fast. Compare per shape
   against `rvv_u1v/u2v/u4v` and scalar. Ideas to try, cheapest first: process
   width per iteration (LMUL) vs register pressure; one `vsetvl` per outer
   loop; i32 vs i64 accumulator width (the scalar formula tells you the
   minimum); hoisting parameter broadcasts; tail policy (`_tu` vs dist /
   agnostic) to drop a merge op. Measure after each change; keep receipts in
   `REPORT.md`. If an idea regresses, revert it and write down the number.
6. **Gate 3** — `evaluate.bench`: final numbers, per shape, vs scalar and each
   RVV variant.
7. **Package** — REPORT.md with: the arithmetic derivation, why the SALTyRN
   kernel failed, gate receipts (commands + verdict IDs), per-shape timing
   table, what you would change for the upstream PR (generator template).
8. **Submit** — run `evaluate.submit(cfg, "qs8-vcvt")` to freeze the receipt.

## Budget

Aim for a correct kernel within ~2 hours of work and spend the rest on speed.
Never trade correctness for speed: gates 1-2 are pass/fail, gate 3 is a
trade-off you must document.
