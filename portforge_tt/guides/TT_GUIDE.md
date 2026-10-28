# Porting to Tenstorrent with TTNN

Read CONTRACT.md first, then SOURCES.md (if the pack has one) for this model's
primary sources. This guide collects what earlier campaigns measured on
Blackhole; treat it as starting hypotheses to confirm, not as facts about your
runtime. TT_OPTIMIZATIONS.md is the catalogue of TT techniques (with exemplars
in /opt/tt-metal) and CORRECTNESS_PITFALLS.md the checklist for mismatches.

## Getting correct

- **Compose sensitive reductions.** At masked, non-tile-aligned sequence
  lengths a softmax built from `max/subtract/exp/sum/reciprocal` in FP32 was
  exact (about 1e-7), while `ttnn.softmax` lost about 1e-2. RMSNorm and
  LayerNorm composed from elementwise ops also beat the fused kernels. Those
  fused kernels were measured with default compute configs and alone; with
  HiFi4 + `fp32_dest_acc_en`, an FP32 residual, or sharded program configs they
  are untested combinations, not ruled out (`native-norm-softmax`).
- **TT FP32 matmul is not IEEE FP32.** At default fidelity it measured about
  1e-3 relative error per linear, only about 2x better than BF16. Full FP32 is
  therefore not a cure-all; HiFi4 with `fp32_dest_acc_en` and targeted FP32
  weights (subsampling convolutions, decoder heads) helped more.
- **BF16 error grows with depth and with dynamic range.** Keeping the residual
  stream in FP32 while matmul inputs stay BF16, plus an FP32 final norm and
  head, halved error in a 12-layer encoder at about 11% latency.
- **Check your reference's numerics too.** An A100 "FP32" oracle silently ran
  cuDNN convolutions in TF32 until `torch.backends.cudnn.allow_tf32 = False`
  was set; the harness evaluator now does this.
- **Folds are free, accuracy-neutral wins.** RMSNorm weights fold into the next
  linear, RoPE `rotate_half` folds into q/k weight columns, and attention over
  independent rows can collapse to a single linear. Verify each fold exactly
  in FP32 on the host first.

## Getting fast

- **Measure on `perf`, explain with `port_profile`.** Bringup has one timed
  sample per case; the perf stage has ~100 samples or 20 s per case, a batch
  sweep and board energy, and is what the frontier ranks. The profile splits
  time into device kernels vs op-to-op gaps and shows each op's efficiency
  against TT's performance model: fix dispatch first, then the dominant ops.
- **Use the whole grid.** Default matmul configs on batched inputs ran B small
  matmuls on part of the grid; folding the batch into M with a 2D-multicast
  program config was the largest single win in TT's own Chronos-2 port
  (`matmul-2d-mcast-fuse-batch`). Query `compute_with_storage_grid_size()`.
- **Keep activations in L1.** Chunk the batch so the encoder runs on L1-resident
  activations, DRAM only between chunks (`l1-interleaved-activations`).
- **Small models are dispatch-bound.** Eager execution of a few hundred small
  ops was flat in sequence length. Capturing the graph with metal trace and
  replaying it gave 5-9x end to end.
- **Keep one trace alive.** Allocating a new shape's buffers while another
  trace existed corrupted memory and hung the device on a later replay.
  Release the live trace before capturing a new shape, or allocate every
  persistent buffer before any capture.
- **Reduce op count before tuning kernels.** Fusing q/k/v, folding norms and
  using `ttnn.transformer` head split/concat cut a graph from about 900 to 500
  ops and 15-20% of replay time.
- **BFP8_B saved memory, not time, while the graph was dispatch-bound**: weight
  memory halved with no latency change and about 1.3x the BF16 error. Once the
  graph is traced and compute-bound, BFP8 operands with HiFi2/LoFi
  matmuls in L1 are a real speed lever (TT's Chronos-2 `performance()` preset:
  548 -> 399 ms). Choose precision and fidelity per op group and sweep them
  (`per-group-precision-sweep`) instead of one global switch.
- **Autoregressive loops pay a host round trip per step.** Profile per-step
  time and move loop control onto traced device work where possible.

## Engineering habits

- Keep a short RESEARCH.md: source URL, revision or date, the finding, and how
  it was confirmed.
- Prefer primary sources: the model card, the reference implementation you
  are gated against, TTNN documentation, and existing tt-metal model ports.
  Respect licenses; do not copy incompatible code.
- This is a shared machine. Stay inside the campaign workspace and mounts and
  never reset cards.
