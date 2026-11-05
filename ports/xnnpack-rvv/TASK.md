# TASK: one XNNPACK family on RVV

The family's formula, parameter domain and width bounds are in SPEC.md with the regime x shape
matrix; context/ holds the scalar reference and every upstream RVV kernel of the family. Speed is
measured against the fastest correct upstream RVV kernel in each cell.

1. **Read before writing.** SPEC.md's domain, the reference, the upstream kernels (what each does
   well in which cells), then TECHNIQUES.md, RVV_GUIDE.md and LESSONS.md.
2. **Start correct.** `port_run(command="check")` on the starting source, then
   `port_evaluate(stage="check")` and `port_evaluate(stage="perf")` for its per-cell numbers.
3. **Optimize.** One technique (or a small combination) per step from TECHNIQUES.md and
   RVV_TECHNIQUES.md, chosen against the measured bottleneck (`port_profile` for the hot loops'
   predicted cycles and bound class; the per-cell speedups for which regimes lose). Record each in
   TRANSFORMS.json, prove every new dispatch predicate and narrowing in NOTES.md, and re-check after
   every change; a perf receipt decides.
4. **Submit** the best measured source with `port_submit` when the supervisor asks or the frontier
   stops moving; the controller certifies it (held-out regimes, second session, cvc5).
