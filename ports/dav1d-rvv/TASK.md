# TASK: one dav1d family on RVV

The family, its functions, formula and domain are in SPEC.md; context/ holds the C reference,
checkasm's test, the other ISAs' implementations (arm64 Neon, x86, LoongArch where they exist) and
the riscv asm macros with an example file.

1. **Read before writing.** The C reference and the test until you know every edge case (widths,
   heights, edge flags, the extreme values the clipping and rounding reach); then the other ISAs'
   structure; then TECHNIQUES.md, RVV_GUIDE.md and LESSONS.md.
2. **Port** (when the source is a placeholder). Choose a structure: follow the arm64 decomposition,
   follow the x86 one, or design directly from the C reference for RVV (strip-mined rows, masks,
   widening ops). Get a correct version first: `port_run(command="check")` green at VLEN 128 and
   1024, then `port_evaluate(stage="check")`.
3. **Measure.** `port_evaluate(stage="perf")` for the per-cell speedup over C, and `port_profile` for
   the hot loops' predicted cycles and bound class.
4. **Optimize.** One technique (or a small combination) per step from TECHNIQUES.md and
   RVV_TECHNIQUES.md, chosen against the measured bottleneck; record each in TRANSFORMS.json and
   keep NOTES.md current. Re-check after every change; a perf receipt decides.
5. **Submit** the best measured source with `port_submit` when the supervisor asks or the frontier
   stops moving; the controller certifies it.
