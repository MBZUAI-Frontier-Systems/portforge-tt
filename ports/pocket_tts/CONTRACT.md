# Pocket TTS backend contract

Write `backend.py` from scratch for the pinned `kyutai/pocket-tts` checkpoint
`languages/english_2026-09_2/model.safetensors` at revision `7f364c05`
(model key `pocket_tts`, weights CC-BY-4.0, reference code MIT). Read
configuration values from `config.json` rather than hardcoding architecture
dimensions; the registry in `models.json` is authoritative.

The model (CALM, arXiv:2509.06926): a 6-layer streaming transformer
(d=1024, 16 heads, RoPE, KV-cache) consumes text token embeddings plus a
voice-prompt KV prefix and, one frame at a time, the previous 32-dim Mimi
latent; from the last hidden state an EOS head (`out_eos`, threshold -4.0)
decides stopping and a depth-6 AdaLN flow MLP (`flow_net`) maps one Gaussian
noise row to the next latent via one-step LSD decoding. Latents are normalized
space (denormalize with `emb_std`/`emb_mean` before the Mimi decoder), 12.5 Hz,
1920 audio samples per frame at 24 kHz.

```python
def create_backend(weights_path, config, device, *, precision="bf16"):
    # Return an object with the methods below. The harness owns the TTNN device.
    ...

class Backend:
    # Required instance declaration; update it for the requested precision.
    precision_policy = {"mode": "bf16", "weights": "bf16", "activations": "bf16",
                        "accumulation": "fp32", "exceptions": []}

    def generate(self, text_tokens, voice, noise, frames_after_eos, max_frames):
        # text_tokens: int64 [L] prepared token ids (see inputs.npz; ids < 4001).
        # voice: dict of float32/int64 arrays keyed transformer_layers_<i>_self_attn__{cache,offset,pad};
        #        cache is [2, 1, T_v, 16, 64] (K,V) — the official ModelState KV
        #        prefix for the voice prompt. Seed your KV cache from it.
        # noise: float32 [F, 32], std sqrt(0.3); consume the rows strictly in
        #        order: row 0 is drawn (and discarded) by the text prefill — the
        #        official forward always samples, even when its output is unused —
        #        and latent f is produced from noise[f + 1].
        # frames_after_eos, max_frames: int scalars from the public inputs.
        # Returns {"latents": float32 [T, 32] normalized space (T computed frames,
        #          including the frame whose EOS trigger stops the run),
        #          "eos_frame": int (first frame with EOS logit > -4.0, counting
        #                      only frames >= 6; -1 if the frame cap was reached),
        #          "audio": float32 [S] at 24 kHz — the Mimi decode of
        #                   latents * emb_std + emb_mean for every computed frame
        #                   EXCEPT a stop-triggering final frame (EOS stop renders
        #                   (T-1) frames; a capped run renders all T), with the
        #                   first sample_rate//200 samples multiplied by
        #                   linspace(0, 1, n)}
        ...

    def predict_latent(self, text_tokens, voice, latents_prefix, noise_row):
        # Teacher-forced probe (development stages only): replay the prefix
        # (float32 [t, 32]) from the voice+text prefix (consuming the prefill
        # draw and one row per prefix frame) and return the flow head's next
        # latent float32 [32] produced from noise_row. The evaluator hands
        # predict_latent exactly the row that produced oracle latent t, so the
        # prediction must equal generate()'s frame t on a matching prefix.
        ...
```

Never mutate caller arrays. Batch is always 1 (the reference implementation
has none). Load the pinned official checkpoint (CPU weight conversion and
deterministic preprocessing are allowed); learned inference — text embedding
lookup, transformer, flow head, EOS head, Mimi decoder — must run on TT. The
oracle is the official `pocket-tts==3.3.0` `TTSModel` in FP32 on the reference
host. Optional `DEVICE_OPTIONS`: `trace_region_size` 0–128 MiB,
`l1_small_size` 0–128 KiB, `num_hw_cqs` 1–2. Return CPU-visible results.

## Precision

Evaluation selects `bf16` (default), `fp32`, `fp16`, or `bfp8_b` explicitly and
passes it to the factory. Declare the actual dominant weight and activation
formats, accumulation, and every higher-precision exception in
`precision_policy`; the weight format and mode must match the request. Mixed
activations require an explanation in `exceptions`. Unsupported modes must fail
clearly and never silently substitute another format. TT `bfloat8_b` is block
floating point with shared exponents, not IEEE FP8. Autoregressive drift makes
this model sensitive: measure the free-run effect of any precision choice; the
reference implementation itself keeps the transformer output and flow head in
FP32 even with BF16 weights.

The correctness oracle stays FP32 on the reference host (CUDA preferred, CPU
recorded). CUDA BF16 comparisons are context only; they never replace the
oracle or relax gates.

## Envelope and gates

Envelope: batch 1; single text chunk (≤50 prepared tokens); voices alba, vera,
michael (pinned precomputed voice states); temperature 0.3 with 1 LSD decode
step; `max_frames = ceil((L/3 + 2) * 12.5)`; generation stops at
`eos_frame + frames_after_eos` or the frame cap. Out of scope: other languages,
the 24-layer variants, raw-audio voice cloning (the Mimi encoder), streaming
APIs, batching, chunked long-form synthesis.

Gates, in words:

- Teacher-forced anchor NRMSE ≤ 0.04 against the FP32 oracle: given the
  oracle's latent prefix `latents[:t]` (t = 4, 12, 32; development stages
  only), the predicted next latent matches per-row NRMSE ≤ 0.04.
- Free-run latent NRMSE ≤ 0.10 over the common prefix with the oracle run on
  the same noise pool; EOS frame within ±2 frames of the oracle's (both -1 on
  cap exhaustion is a match).
- Audio log-mel NRMSE ≤ 0.12 against the oracle waveform; audio length ratio
  ≥ 0.85.
- Revisit consistency: rerunning a case in the same suite matches its first
  run within NRMSE 0.01 and the same EOS frame; repeated identical calls are
  bit-identical.
- Full stage only: WER of an independent ASR (parakeet TDT) on the generated
  audio inflates ≤ 0.02 relative to the same scoring of the oracle audio.
- Timing follows `TIMING_PROTOCOL` in evaluate.py: text tokens + voice state +
  noise to latents/eos/audio, device-synchronized, excluding weight loading.

Public inputs never contain oracle outputs except the bounded anchor prefixes;
do not copy them into free-run outputs — that is fabrication, not porting.

## Baseline before optimization

Before any optimization claim, record measured baselines on the fixed suites
from the same TT host: PyTorch on the host CPU (pocket-tts runs CPU
real-time; pin threads) and, if the installed stack supports it, an
ONNX/forge path. Every speedup cites the matched baseline artifact, the same
suite, precision and card. TT-vs-TT ratios never mix with CUDA context
numbers. Real-time factor (audio seconds / wall seconds) is the headline
metric; a 10 s text should beat 1.0 RTF after optimization.

## Stages

`smoke` (one real case) and `bringup` (eight cases: short and long texts,
one-word, digits, punctuation, uppercase, a near-envelope-maximum text, and a
revisit) have one warmup and one measured call each. `full` is the frozen
held-out corpus (16 Harvard sentences + edges + revisit) run once on frozen
source through `port_submit`. Early passes are debugging milestones, never
acceptance: `verdict.accepted` is true only after a successful full
evaluation. Corpus provenance: BENCHMARK.md.

## Evidence and sandbox

Collect results with `port_status(compact=True)`; `port_log` returns
stdout/stderr only. `port_run` jobs see public inputs, never verdicts or
oracle outputs, and mount `/models.json`, `/models.py`, prepared `/input`
files and routed `/weights` read-only. Do not exploit known test inputs,
precompute outputs, run hidden CPU inference or alter the trusted evaluator.
Timing includes transfers, host control, learned compute and synchronization,
and excludes weight loading.

## tt-metal deliverable

A tt-metal model package for an upstream PR under
`models/experimental/pocket_tts`: `tt/` (TTNN implementation), `demo/` (CLI),
`reference/` (FP32 reference utilities), `tests/` (pytest, importable without
sandbox mounts), `benchmarks/`, `docs/`, and a README with model-card links,
measured evidence and limitations. SPDX headers on source files; only
Apache-2.0-compatible code (the MIT-licensed reference may be studied but
copied code must carry its notices; CC-BY-4.0 weights are never redistributed
inside the package). The worker prepares branch, commit message and PR body;
the operator reviews, signs and submits.
