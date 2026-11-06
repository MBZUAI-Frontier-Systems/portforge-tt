# ResNet-50 → Tenstorrent port

Build one correct, fast, configurable TTNN implementation of
`timm/resnet50.tv_in1k` (ImageNet-1k classification), packaged in tt-metal
style for an upstream PR. CONTRACT.md defines the interface and acceptance
envelope; PORTING_RULES.md the rules every port follows. Read LESSONS.md before
the first device job.

1. Read PORTING_RULES.md, TT_GUIDE.md, SOURCES.md, BENCHMARK.md and LESSONS.md. Inspect the
   pinned checkpoint's tensor names and shapes from the safetensors header and
   the pinned config. Save concise findings and citations in RESEARCH.md.
   Settle before implementing:
   - the layer graph: the 7x7 stride-2 stem convolution with BatchNorm and ReLU,
     the 3x3 stride-2 max pool (padding 1), four stages of 3, 4, 6 and 3
     bottleneck blocks (stride on the 3x3 convolution, projection shortcut in
     each stage's first block), the global average pool and the 2048-to-1000
     classifier;
   - BatchNorm folding into convolution weights and biases on the CPU at load
     time, checked against the unfolded PyTorch math on the CPU first;
   - which TTNN operations implement convolution, pooling, the residual
     addition and the classifier in the installed TTNN, and the tensor layouts
     and shapes each expects (read their docstrings and op unit tests).
2. The baseline is your own first passing source on `perf` (step 5). Do not
   run PyTorch CPU or ONNX baselines.
3. Write `backend.py` and small component checks (the stem, one bottleneck,
   the head) against a CPU reference of the same folded weights. Use `port_run`
   for bounded experiments. Edit files only with `port_write`; files written
   inside a job are discarded.
4. Run `port_evaluate(stage="smoke")`, collect results with
   `port_status(compact=True)`, and fix one hypothesis at a time.
5. Pass `bringup` before optimizing, then run `port_evaluate(stage="perf")` on
   that source at both delivery precisions: it is the measured starting point of
   the search. Keep `TRANSFORMS.json` (parent, applied catalogue ids, novel)
   current from here on.
6. Standard catalogue (TT_OPTIMIZATIONS.md): `port_profile`, then fix dispatch
   (trace, persistent buffers), transfers (input upload, readback, two command
   queues for `classify_stream`), grid use, L1 residency, and per-op-group
   precision and fidelity. Each step is one child of a frontier source, ranked
   on `perf`. Retry earlier rejections in new combinations rather than ruling
   them out.
7. TT-native work, guided by the profile: custom `ttnn.generic_op` kernels for
   dominant or fused ops, NoC-local and multicast data movement, sub-devices.
   Record anything outside the catalogue in `novel` with the op-level evidence.
8. Write REPORT.md: correctness evidence, commands, the frontier on `perf`
   (images per second per case, latency percentiles, energy, memory) versus the
   starting point, rejected combinations with reasons, limitations and sources.
   Keep STATE.md short: completed work, current hypothesis, pending job IDs,
   next action.
9. Build the tt-metal package (PORTING_RULES.md, UPSTREAM_PR.md) (tests with the `device`
   fixture, an end-to-end test, demo, README perf table) and prepare the branch
   name, commit message and PR body early.
10. Rerun smoke, bringup and perf on the final source, finish all documents,
   then call `port_submit` exactly once. Every file becomes read-only after submitting.
