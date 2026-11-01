# <Model> → Tenstorrent port

<!-- The worker's ordered plan. Keep the order; replace TODOs with model detail. -->

Build one correct, fast, configurable TTNN implementation of `TODO/hf-repo-id`,
packaged in tt-metal style for an upstream PR. CONTRACT.md defines the
interface and acceptance envelope; PORTING_RULES.md the rules every port
follows. Read LESSONS.md before the first device job.

1. Read PORTING_RULES.md, TT_GUIDE.md, SOURCES.md, BENCHMARK.md and LESSONS.md. Inspect the
   pinned checkpoint and the reference implementation. Save concise findings
   and citations in RESEARCH.md. TODO: list the architecture details that must
   be understood before implementing (e.g. masking, positions, normalization,
   decoding rules).
2. Baseline first: measure PyTorch on the TT host's CPU (and ONNX/forge if
   installed) on the smoke and bringup suites. Record commands, samples and job
   IDs in BASELINE_*.md.
3. Write `backend.py` and small component checks. Use `port_run` for bounded
   experiments. Edit files only with `port_write`; files written inside a job
   are discarded.
4. Run `port_evaluate(stage="smoke")`, collect results with
   `port_status(compact=True)`, and fix one hypothesis at a time.
5. Pass `bringup` before optimizing, then run `port_evaluate(stage="perf")` on
   that source: it is the measured starting point of the search. Keep
   `TRANSFORMS.json` (parent, applied catalogue ids, novel) current from here on.
6. Standard catalogue (TT_OPTIMIZATIONS.md): `port_profile`, then fix dispatch
   (trace, persistent buffers), grid use (matmul program configs, batch folded
   into M), L1 residency, and per-op-group precision/fidelity. Each step is one
   child of a frontier source, ranked on `perf`. Retry earlier rejections in new
   combinations rather than ruling them out.
7. TT-native work, guided by the profile: custom `ttnn.generic_op` kernels for
   dominant or fused ops, NoC-local and multicast data movement, sub-devices.
   Record anything outside the catalogue in `novel` with the op-level evidence.
8. Write REPORT.md: correctness evidence, commands, the frontier on `perf`
   (latency percentiles, throughput, energy, memory) versus the starting point,
   rejected combinations with reasons, limitations and sources. Keep STATE.md short: completed
   work, current hypothesis, pending job IDs, next action.
9. Build the tt-metal package (PORTING_RULES.md, UPSTREAM_PR.md) (tests with the `device`
   fixture, an end-to-end test, demo, README perf table) and prepare the branch
   name, commit message and PR body early.
10. Rerun smoke, bringup and perf on the final source, finish all documents,
   then call `port_submit` exactly once. Every file becomes read-only after submitting.
