# TASK: campaign 1 - investigate the mct hardware failure, then port the loop filter

Two linked objectives. The first is a gate-integrity issue for this whole
repo lane; the second is the first from-scratch dav1d port.

## Part A - mc_8bpc.mct fails on real hardware but not under QEMU

On the X280, a full checkasm run reports `mc_8bpc.mct [FAILED]`; the same
binary under QEMU (VLEN 512) passes all 916 tests. `--list-functions` shows
the 8bpc `mct_8tap_regular_sharp_*` functions are overridden by RVV asm
(`(c, rvv)`) while the 16bpc ones are C-only.

1. Reproduce deterministically: run `checkasm --function=<pattern>` on the
   X280 with several fixed seeds until you have a failing input recorded
   (checkasm prints the failing case with `-v`; seeds must be logged).
2. Narrow: reduce the failing case to the smallest (width, alignment, value)
   that still diverges between the RVV asm and the C reference. Compare QEMU
   and hardware on that case - if they disagree with each other, you have a
   QEMU model gap OR a hardware/erratum issue; if they agree with each other
   but differ from C, it is a plain asm bug that QEMU's seed missed.
3. Write the finding to `kernels/mct-hardware/REPORT.md` with the exact
   failing case, the suspected instruction (single-step with QEMU's `-d`
   trace and/or gdb under QEMU), and the fix. If it is an upstream asm bug:
   prepare the one-function fix as the campaign deliverable. If it is a QEMU
   gap: document precisely (instruction + sequence) - that changes which
   gates this repo may trust QEMU for.

## Part B - port the loop filter to RVV (new coverage)

`src/loopfilter.h` (`Dav1dLoopFilterDSPContext`) has no riscv implementation
at all - pure C on RISC-V today, while x86/ARM/loongarch all have assembly.

1. Study `src/loopfilter_tmpl.c` (the C reference: `loop_filter` +
   `loop_filter_128`, horizontal/vertical, bitdepth-templated) and the
   existing asm ports for the register-level conventions (start with
   `src/loongarch/` if the riscv asm files are too terse; `src/arm/64/` for
   the algorithm decomposition).
2. Write `src/riscv/64/loopfilter.S` (+ the loopfilter.h init wiring in
   `src/riscv/`) for the 8bpc functions first; 16bpc after 8bpc is green.
3. Gates: checkasm (`loopfilter` tests) under QEMU at all VLENs + X280, a
   targeted driver for the filter's boundary strengths/flatness masks, then
   `checkasm --bench` on the X280 vs the C reference.
4. REPORT.md with derivation, receipts, and the upstream MR shape.

Budget: Part A ~90 minutes, Part B the rest. Correctness gates before any
benchmarking, seeds logged, failures verbatim, honest negatives.
