# Start from public evidence

Read CONTRACT.md first. Target: `kyutai/pocket-tts` revision `7f364c05`,
recipe `languages/english_2026-09_2` — a CALM speech model: 6-layer streaming
transformer (d=1024, 16 heads, RoPE, hidden_scale 4) with a depth-6/512 AdaLN
flow head predicting continuous 32-dim Mimi latents at 12.5 Hz (one-step LSD,
temp 0.3), a linear EOS head (threshold -4.0), and a Mimi SEANet+2-layer-
transformer decoder rendering 24 kHz audio. ~100M generative parameters; the
checkpoint bundles flow_lm (127 tensors), mimi (87) and the speaker projection.
Envelope: batch 1, ≤50 text tokens, three pinned voice states, single chunk.

- [Model card](https://huggingface.co/kyutai/pocket-tts) and the pinned
  [english_2026-09_2 recipe files](https://huggingface.co/kyutai/pocket-tts/tree/7f364c05d1be3faf2b234cdbe4cbb68a026d4218/languages/english_2026-09_2)
  (gated: accept terms; CC-BY-4.0 weights).
- [CALM paper: Continuous Audio Language Models](https://arxiv.org/abs/2509.06926)
  (Rouard et al.): why continuous latents replace discrete tokens, consistency/
  LSD decoding, and the Pocket TTS release results (WER 1.84, CPU real-time).
- [Kyutai technical report](https://kyutai.org/pocket-tts-technical-report/) and
  [pocket-tts GitHub](https://github.com/kyutai-labs/pocket-tts) (MIT): the
  exact reference semantics the oracle runs — `tts_model.py` (AR loop, EOS rule,
  chunking, `frames_after_eos` heuristics), `flow_lm.py` (LSD decode, noise
  draw, latent normalization), `modules/transformer.py` (streaming attention,
  KV cache, RoPE), `modules/mlp.py` (AdaLN flow head), `models/mimi.py`
  (decode_from_latent streaming state). Study, do not paste without notices.
- Closest TT precedents in the tt-metal tree: `models/experimental/voxtral_tts`
  (Kyutai flow-TTS: codec decoder to 24 kHz, flow head, AR pipeline — the
  nearest architectural neighbour), `models/experimental/speecht5_tts`, and the
  Whisper demos in `models/demos/audio` (conv frontends, AR decode loops).
  Similarity is a starting point, not proof of equivalent semantics.
- [TTNN docs](https://docs.tenstorrent.com/tt-metal/latest/ttnn/): conv1d/
  transposed convs for SEANet, SDPA vs manual attention, traces. The per-frame
  AR loop is the dispatch-overhead risk; the NLLB and Chronos-2 campaigns'
  trace/batching wins transfer directly, and Parakeet's decode-loop lessons
  apply.

This is a shared machine (many users' home directories). Never read, write or
probe beyond this campaign's workspace and mounts; never reset cards; keep all
artifacts inside the run workspace. Precision guidance (BF16 primary, BFP8_B
as a separate block-float experiment, sensitive-op FP32 exceptions with
measured justification) matches the previous contracts. Use web search for
existing public ports and optimization write-ups; confirm claims in primary
sources; cite licenses.
