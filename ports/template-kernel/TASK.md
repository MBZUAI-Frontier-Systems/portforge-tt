# TASK: one <library> family on RVV

1. **Read before writing.** SPEC.md (formula, domain, matrix), the reference and other
   implementations in context/, then TECHNIQUES.md, RVV_GUIDE.md and LESSONS.md.
2. **Port or start correct.** `port_run(command="check")`, then `port_evaluate(stage="check")`.
3. **Measure.** `port_evaluate(stage="perf")` and `port_profile`.
4. **Optimize** with techniques from TECHNIQUES.md and RVV_TECHNIQUES.md, recorded in TRANSFORMS.json;
   a perf receipt decides.
5. **Submit** the best measured source with `port_submit`.
