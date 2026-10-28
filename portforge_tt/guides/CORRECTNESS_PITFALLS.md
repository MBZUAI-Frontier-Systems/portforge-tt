# Correctness pitfalls when porting a model

Most of these were found in public model releases by teams that ship fast ports
(Unsloth's bug reports on Gemma, Llama, Phi and Qwen; llama.cpp's backend tests)
or are documented TT hardware behaviour. Check every item that applies before
blaming the hardware or loosening a hypothesis. The FP32 oracle catches the
symptom; this list shortens the search for the cause.

## Constants and numerics

- **Constants in FP32.** Scale factors such as `sqrt(hidden_size)` lose digits
  in BF16 (Gemma's sqrt(3072) = 55.4256 rounds to 55.5). Compute them in FP32
  on the host and fold them into weights exactly.
- **Norms in FP32.** RMSNorm/LayerNorm statistics (sum of squares, variance,
  `eps`) need FP32 accumulation; check the model's `eps` and whether its gamma
  is applied as `w` or `(1 + w)` (Gemma).
- **RoPE in FP32.** Compute cos/sin tables in FP32, check the scaling variant
  (linear, NTK, YaRN, Llama-3 frequency bands) and `rope_theta`, and whether
  the model rotates interleaved pairs or halves.
- **Division form.** `y * (1/x)` and `y / x` round differently; match the
  reference when a gate is tight.
- **Overflow.** Activations can exceed the FP16 range (Gemma 3 > 65504) and BF16
  loses small residual updates. Keep the residual stream FP32 when a gate fails
  only on deep layers; never clip activations to hide it.

## Activation and attention variants

- **GELU variant.** Exact (erf) vs tanh approximation is a model choice; the
  wrong one passes short tests and drifts on long ones. Same for SiLU vs Swish-β.
- **Softcapping.** Gemma 2 caps attention logits (50) and final logits (30);
  standard fused attention kernels do not.
- **Sliding windows.** Check off-by-one in window size (Phi-3: 2047 vs 2048) and
  which layers are local vs global.
- **Masks and padding.** Padded rows must not change other rows (the bringup
  single/reorder/padding cases test this); check left vs right padding, the
  mask value (-inf vs large negative in BF16) and fully masked rows (NaN from
  softmax over all -inf).

## Tokens and embeddings

- **BOS/EOS.** Missing or duplicated BOS changes every output (Gemma, Llama 3).
- **pad = eos.** Generation that never stops or stops at once often means the
  pad token equals EOS (Qwen 2.5, Phi-4).
- **Untrained rows.** Reserved or special-token embedding rows can be zero or
  garbage (Llama 3: 287 rows); outputs on those ids are not meaningful.
- **Tied embeddings.** Check whether the LM head shares the input embedding.

## TT hardware behaviour

- **Not IEEE.** Denormals flush to zero; NaN/Inf only set sticky flags
  (`tech_reports/Handling_Special_Value`). Check finiteness explicitly on
  outputs.
- **Block floating point.** BFP8_B shares one exponent per 16 values; a
  row with one large outlier loses its small values. Mantissa rounding
  saturates instead of renormalizing (`tech_reports/data_formats`).
- **Math fidelity.** LoFi/HiFi2 drop mantissa bits of the operands inside the
  matrix engine; accumulation precision is a separate switch
  (`fp32_dest_acc_en`).
- **Approximate SFPU.** `math_approx_mode` and approximate exp change softmax
  and GELU tails.
- **Blackhole NoC flushes.** Custom kernels need explicit NoC flushes that
  Wormhole did not; missing ones give mismatches or hangs
  (`tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md`).
- **Trace semantics.** A replayed trace reads whatever is at the captured
  addresses: copy new inputs into the persistent buffers, never reallocate,
  and execute the trace before reading outputs captured with it.

## How to localize a mismatch

Compare per op against the reference (llama.cpp's `test-backend-ops` idea): run
one layer at a time on the same inputs, record NRMSE per op output, and fix the
first op that exceeds its tolerance. Keep the per-op tolerance with a comment
on why it is what it is.
