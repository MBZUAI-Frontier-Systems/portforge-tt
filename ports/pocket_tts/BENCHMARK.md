# Pocket TTS benchmark: what the port is measured on, and why

A **porting benchmark**, not a TTS-leaderboard claim. It answers: (1) is the TT
implementation numerically faithful to the FP32 oracle (bounded teacher-forced
next-latent predictions and free-run sequences), (2) does it preserve
determinism and revisit semantics, (3) does full-corpus audio stay intelligible
(WER vs the same scoring of the oracle audio), and (4) is it faster relative to
its own recorded CPU baseline. Absolute speech quality is Pocket TTS's problem;
we gate only *agreement with the FP32 reference*.

## Frozen corpus (`corpus.json`, canonical sha256
`2160008909385c5384b2e3fc7fa72ada9b463aa5c21e189e5189b5a15a187964`)

| Cases | Origin | Why |
|---|---|---|
| 25 × Harvard sentences (IEEE 1965, public domain; text from the [stringr data-raw mirror](https://raw.githubusercontent.com/tidyverse/stringr/master/data-raw/harvard-sentences.txt), accessed 2026-10-15) | deterministic seeded selection `random.Random(20261015)` | The classic TTS intelligibility fixture: phonetically balanced, license-clean, human-readable for WER scoring. 5–12 words per sentence exercises the short end; multi-sentence composites reach the 50-token chunk limit. |
| edge-one-word, edge-digits, edge-punct, edge-upper, full-edge-question, full-edge-apostrophe | authored, deterministic | Degenerate/adversarial text: single word (minimum token count), numerals, heavy punctuation and case, questions, contractions — catch tokenizer and EOS-path failures. |
| voices alba / vera / michael | `languages/english_2026-09_2/embeddings/*.safetensors`, CC-BY-4.0 | Precomputed ModelState KV prefixes shipped with the weights; three distinct speakers, rotated across cases. |

29 cases total; texts and voice assignments are frozen in the JSON (7.2 KB).
The reference prepares text with the official chunker (asserting the
single-chunk envelope), tokenizes, and derives `max_frames` and
`frames_after_eos` per case; candidates receive token ids, never raw text
parsing.

| Stage | Cases | Contents |
|---|---|---|
| `smoke` | 1 | shortest Harvard sentence, voice alba |
| `bringup` | 8 | short, long, one-word, digits, punctuation, uppercase, near-max-length text, revisit (same text+voice as short) |
| `full` | 20 | 16 held-out Harvard + question + apostrophe + long, plus revisit — all WER-gated |

## Gates and why

- **Anchor NRMSE ≤ 0.04** (teacher-forced, t = 4/12/32, development stages):
  the only drift-free numeric probe of the transformer+flow-head stack; the
  paper's standard per-row threshold.
- **Free-run latent NRMSE ≤ 0.10** over the common prefix: a continuous AR
  model amplifies bf16 rounding over hundreds of frames; 0.10 catches gross
  divergence while admitting accumulation the anchor gate proves bounded.
- **EOS frame ± 2** (160 ms): EOS is a threshold on a continuous logit, so
  near-ties can flip exactly like the Parakeet transducer argmax; two frames
  is the smallest tolerance that does not make the gate a coin flip. Both
  sides hitting the frame cap (-1) is a match; one-sided cap is a runaway.
- **Audio mel-NRMSE ≤ 0.12 + length ratio ≥ 0.85**: the decoder is a pure
  function of the latents, so this mostly re-checks the free-run gate
  end-to-end through Mimi.
- **Revisit ≤ 0.01 + identical EOS; repeats bit-identical**: state must not
  leak between cases or calls.
- **Full-stage WER inflation ≤ 0.02**: parakeet TDT (pinned revision) scores
  candidate audio and oracle audio against the input text on the reference
  host; identical scoring isolates porting damage from TTS quality.
- Diagnostics (non-gating): per-case real-time factor, first-call latency,
  A100/RTX comparison context, memory snapshots.

## Declared reference policies

- **Noise as a public input.** The official sampler draws per-frame Gaussian
  noise from the global torch RNG, which differs between a CUDA oracle and a
  TT device. The evaluator feeds both sides the same frozen pool
  (`numpy PCG64`, seed derived from the corpus hash + case name) through a
  narrow `torch.nn.init.normal_` patch that matches only the sampler's draw
  (numel 32, std √0.3). Nothing else is intercepted; the pool exhausting means
  generation overshot its frame budget and fails loudly. Row 0 is consumed by
  the text prefill (the official forward samples even when its output is
  discarded); latent f comes from row f + 1.
- **Two oracle passes per case.** Official `TTSModel.generate_audio` produces
  the audio; a faithful replay of the official AR loop produces latents/EOS.
  Both must consume the same number of noise rows and agree on length. The
  official decoder thread batches frames by wall-clock timing, so its audio is
  only repeatable to ~1e-6 (recorded, tolerated); latents and EOS are
  bit-deterministic.
- **Checkpoint layout.** The gated HF repository stores the recipe in
  subdirectories the registry cannot pin; `evaluate.py`/`prepare_checkpoint.py`
  download the pinned subpaths, flatten them (hash-verified), and write the
  authored `config.json` + a materialized recipe YAML with local paths. Use
  `prepare_checkpoint.py` instead of the generic `--download` shortcut.

## Held-out discipline

`full` cases are never shown to agents during development; anchor prefixes
(oracle latents[:t], t ≤ 32) exist only on smoke/bringup. Copying any oracle
array into free-run outputs is fabrication, not porting. If an operator shares
a failing full case, later full results are no longer blind for it.
