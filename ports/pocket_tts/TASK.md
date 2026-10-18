# Pocket TTS → Tenstorrent port

Build one correct, fast, configurable TTNN implementation of
`kyutai/pocket-tts` (english_2026-09_2), packaged in tt-metal style for an
upstream PR. CONTRACT.md defines the interface and acceptance envelope. Read
LESSONS.md before the first device job.

1. Read TT_GUIDE.md, SOURCES.md, BENCHMARK.md and LESSONS.md. Inspect the
   pinned checkpoint (`safetensors` keys: `flow_lm.*` 127 tensors, `mimi.*`
   87) and the official `pocket-tts==3.3.0` sources. Save concise findings and
   citations in RESEARCH.md. The details that must be understood before
   implementing: the streaming transformer's KV-cache layout and the voice
   ModelState format (`cache` [2,1,T,16,64], `offset`, `pad` per layer), RoPE
   application, the AdaLN flow head's conditioning (s,t for LSD), latent
   normalization (`emb_mean`/`emb_std`), the EOS threshold rule (frames ≥ 6,
   logit > -4.0, stop at eos + frames_after_eos), and the Mimi decoder's
   streaming conv state (offsets advance 2 encoder steps per latent).
2. Baseline first: measure PyTorch on the TT host's CPU on the smoke and
   bringup suites (pocket-tts runs CPU real-time with 1 thread by default;
   record both 1 and 12 threads). Record commands, samples and job IDs in
   BASELINE_*.md.
3. Write `backend.py` and small component checks. Start with the teacher-forced
   `predict_latent` path (text embedding + transformer + flow head on TT,
   Mimi decode later); use `port_run` for bounded experiments. Edit files only
   with `port_write`; files written inside a job are discarded.
4. Run `port_evaluate(stage="smoke")`, collect results with
   `port_status(compact=True)`, and fix one hypothesis at a time. The anchor
   NRMSE gate isolates single-step numerics; free-run gates add drift.
5. Pass `bringup` before optimizing. Profile, then compare changes on identical
   workloads against the recorded baselines. The AR loop is dispatch-bound at
   ~300 frames per case: plan trace capture over the per-frame step early, and
   batch the Mimi decode across frames rather than decoding per frame. Start
   with BF16, compare FP32 when diagnosing numerics, and treat BFP8_B as a
   separate experiment (shared exponents are risky for the flow head and the
   SEANet decoder).
6. Write REPORT.md: correctness evidence, commands, baseline versus optimized
   latency and memory, limitations and sources. Keep STATE.md short: completed
   work, current hypothesis, pending job IDs, next action.
7. Build the tt-metal package from CONTRACT.md and prepare the branch name,
   commit message and PR body early.
8. Rerun smoke and bringup on the final source, finish all documents, then call
   `port_submit` exactly once. Every file becomes read-only after submitting.
