# CONTRACT: dav1d RVV kernel ports

Read fully before writing assembly. dav1d is a different machine from XNNPACK:
hand-written assembly (not intrinsics), meson (not CMake), checkasm (not
gtest), and a stricter correctness contract - **assembly is bit-exact to the C
reference, no tolerances**.

## Deliverable

Per campaign, in `kernels/<name>/`:

1. The assembly: one or more files in the worktree's `src/riscv/64/`, functions
   named `<function>_<8bpc|16bpc>_rvv`, using the tree's `asm.S` macros
   (`function name, export=1, ext="v,zbb"` style; note existing riscv asm
   requires Zbb - confirm what the X280 executes before relying on an ext).
2. Init wiring: one assignment line per function in `src/riscv/<dsp>.h`'s
   inline init (e.g. `c->lf.loop_v = dav1d_loop_filter_v_8bpc_rvv;`) inside the
   right bitdepth/feature guard.
3. `REPORT.md`: the algorithm derivation from the C reference, the register
   allocation and lane-layout reasoning, gate receipts in run order (with the
   checkasm seeds used), benchmark table, and honest negatives.

## Gates (run them yourself, in this order)

1. **checkasm correctness** - `dav1d.build` then `dav1d.run_checkasm`:
   the family's tests green under QEMU at VLEN 128, 256, 512, 1024 and natively
   on the X280 (when up). checkasm compares bit-exactly against the C
   reference over randomized cases (alignments, widths, edge sizes). Run the
   WHOLE suite too - a new function must not break another (shared temporaries
   happen).
2. **Targeted differential** - checkasm's random cases do not cover every
   boundary. Write a small driver (pattern: ports/xnnpack-rvv/kernels/*/
   difftest.c) for the ported function's corners: minimum/maximum dimensions,
   alignment edges, the sign/rounding boundaries of any shifts, and - if the
   function is bitdepth-templated - both bitdepths.
3. **Benchmark** - `checkasm --bench -f <pattern>` on the X280 (its timing
   source is linux-perf, which works under tt-bh-linux) vs the C reference and
   vs any existing RVV implementation. Report cycles per call at the seeds you
   used, plus - for memory-heavy functions - the working-set effect. QEMU
   timings are NOT evidence of speed.
4. **Review pass** - as a dav1d maintainer would: naming, macro usage, no
   dead code, comments explaining the lane layout and the tail handling, and
   the upstream MR shape (asm file + init line + meson wiring if needed).

## Honesty rules

- The C reference and checkasm are the trusted side. Never patch them to make
  a failure disappear (the disposable worktree may need *build* wiring like
  params-guard widening - that is allowed and must be recorded as the
  upstream-PR library diff).
- Record every checkasm seed. Failures are kept verbatim in REPORT.md.
- QEMU passing does not prove hardware correctness and vice versa (campaign 1
  exists precisely because of this). A port is done when BOTH agree.
- If hardware is wedged (connection refused), reboot it with
  `x280.boot(cfg)` - that is a supported recovery, not a workaround.

## Assembly-specific rules

- Follow the neighboring .S files' conventions exactly (macro usage, register
  preservation, stack usage, `.balign`).
- Say which vector extensions you rely on (`ext=` in the function macro) and
  confirm they exist on the X280 (`/proc/cpuinfo` isa string) AND under QEMU.
- The tail is handled by `vl` (setvl with the remaining count), never by a
  scalar epilogue, unless the C reference's semantics force it - and then it
  must be proven for the boundary cases.
