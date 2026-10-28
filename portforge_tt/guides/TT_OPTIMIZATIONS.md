# TT optimization catalogue

Generated from `optimizations.json` (the controller reads the JSON; keep both in sync).
Every entry is a Blackhole/TT-Metal technique with a working exemplar in the TT-Metal
checkout (`/opt/tt-metal` in the sandbox). Read the exemplar before implementing.

## How to use it

- **Measure first.** `port_profile` gives the per-op table (device time, op-to-op gap share,
  TT performance-model efficiency, FPU/DRAM utilization). A large gap share means dispatch
  bound (trace, fewer ops); a dominant op with low efficiency is where program configs,
  layout, precision/fidelity or a custom kernel pay off.
- **Declare what a source is.** Keep `TRANSFORMS.json` in the workspace:
  `{"parent": "<code_sha256 you started from>", "applied": ["<ids>"], "novel": "...", "note": "..."}`.
  The controller builds the search tree from it and from trusted receipts.
- **Rejections are contextual.** A technique that failed on one parent (too slow, or
  accuracy) is still eligible in another combination: fused norms that failed alone may pass
  with `fp32-dest-acc` or `fp32-residual`; BFP8 that did not speed up alone may with
  `math-fidelity-ladder` and L1 residency. Record parent, combination and reason.
- **Rank on `perf`.** Only the perf stage (many repeats, batch sweep, board energy) feeds the
  frontier. Bringup proves correctness; it is not a benchmark.
- **Go beyond the list.** The north star is a TT-native technique nobody has written down
  yet (NoC dataflow, L1 placement, multicast, sub-devices). Put it in `novel` with profile
  evidence (the op-level delta) so it can be checked.

## dispatch: Remove host/dispatch overhead between device ops

### `trace-capture` — Metal trace replay of the whole forward

- **Mechanism:** Capture the op sequence once (program cache warm, no allocations during capture) and replay it with execute_trace; op-to-op gaps drop to a few microseconds.
- **Helps when:** Any fixed-shape forward; largest win when host dispatch dominates (small batch, many small ops).
- **Combine/conflict with:** `persistent-io-buffers`, `two-command-queues`
- **Risk:** hang/wrong output if buffers move or shapes change; keep one live trace per shape
- **Exemplar:** `models/experimental/chronos_forecast/tt/trace_runner.py` (source: tech_reports/AdvancedPerformanceOptimizationsForModels)

### `persistent-io-buffers` — Address-stable device input/output buffers

- **Mechanism:** Allocate inputs/outputs once, copy new data into them (copy_host_to_device_tensor) before each replay.
- **Helps when:** Required by trace-capture; also removes per-call allocation.
- **Combine/conflict with:** `trace-capture`
- **Risk:** stale data if a copy is skipped
- **Exemplar:** `models/experimental/chronos_forecast/tt/trace_runner.py` (source: tech_reports/AdvancedPerformanceOptimizationsForModels)

### `two-command-queues` — 2 CQs with events: I/O on CQ1, compute on CQ0

- **Mechanism:** DEVICE_OPTIONS num_command_queues=2; write the next input and read back the previous output on CQ1 while CQ0 replays; synchronize with record/wait events.
- **Helps when:** Host<->device transfer is a visible share of the step (large inputs/outputs, streaming).
- **Combine/conflict with:** `trace-capture`, `host-overlap`
- **Risk:** event ordering bugs; measured ~0 gain on chronos_forecast
- **Exemplar:** `models/demos/sentence_bert/runner/performant_runner.py` (source: tech_reports/AdvancedPerformanceOptimizationsForModels)

### `host-overlap` — Overlap host preprocessing with replay

- **Mechanism:** Prepare/tokenize/pad the next batch on the host while the device runs the current one (non-blocking replay).
- **Helps when:** Host preprocessing is a measurable share of end-to-end time.
- **Combine/conflict with:** `two-command-queues`
- **Risk:** low
- **Exemplar:** `models/experimental/chronos_forecast/tt/trace_runner.py` (source: chronos_forecast optimizations.md)

### `on-device-loop-control` — Keep autoregressive control on device

- **Mechanism:** Argmax/sampling, EOS masks and cache updates as device ops so a decode step needs no host readback.
- **Helps when:** Token-by-token decoders (ITL dominated by readback/dispatch).
- **Combine/conflict with:** `trace-capture`
- **Risk:** exact-token gates
- **Exemplar:** `models/tt_transformers/tt/generator.py` (source: tech_reports/LLMs/llms.md)

### `op-count-reduction` — Fewer dispatched ops

- **Mechanism:** Algebraic folds, reshapes removed, merged elementwise chains; count ops in the Tracy CSV.
- **Helps when:** Untraced paths, or when trace is impossible.
- **Combine/conflict with:** `norm-gamma-fold`, `fused-qkv`
- **Risk:** low
- **Exemplar:** `models/experimental/chronos_forecast/optimizations.md` (source: PortForge chronos2 (-44% ops))

## grid: Use the whole Tensix grid in matmuls

### `matmul-2d-mcast-fuse-batch` — 2D multicast matmul with the batch folded into M

- **Mechanism:** MatmulMultiCoreReuseMultiCastProgramConfig(fuse_batch=True, per_core_M/N over the full grid) so B small matmuls become one large one.
- **Helps when:** Batched encoders/prefill where default configs run B tiny matmuls on part of the grid (chronos_forecast: 2659 -> 922 ms, the largest single win).
- **Combine/conflict with:** `l1-interleaved-activations`, `matmul-blocking`
- **Risk:** L1 overflow at large per-core blocks
- **Exemplar:** `models/experimental/chronos_forecast/tt/program_configs.py` (source: tech_reports/LLMs/llms.md 4.3)

### `matmul-1d-mcast` — 1D multicast matmul (in0 or in1 mcast)

- **Mechanism:** MatmulMultiCoreReuseMultiCast1DProgramConfig when N (or M) is too small to fill 2D grid columns.
- **Helps when:** Narrow outputs (heads, projections with small N) or tall-skinny activations.
- **Combine/conflict with:** `matmul-blocking`
- **Risk:** low
- **Exemplar:** `models/experimental/chronos_forecast/tt/program_configs.py` (source: chronos_forecast sweeps/sweep_matmul_l1.py)

### `matmul-blocking` — Tune in0_block_w and output subblocks

- **Mechanism:** in0_block_w >= 2, out_subblock_h*w as large as dst allows (8 tiles, 4 with fp32 dest acc); sweep per shape rather than guess.
- **Helps when:** After a program config family is chosen; compute-bound matmuls.
- **Combine/conflict with:** `fp32-dest-acc`, `math-fidelity-ladder`
- **Risk:** low
- **Exemplar:** `models/experimental/chronos_forecast/sweeps/sweep_matmul_l1.py` (source: tech_reports/LLMs/llms.md 4.6)

### `dram-sharded-matmul` — DRAM-sharded weights read by the nearest cores

- **Mechanism:** Shard weights across the DRAM banks (create_dram_sharded_mem_config) and use the DRAM-sharded matmul program config.
- **Helps when:** Small M (decode, batch 1) where matmuls are DRAM-bandwidth bound.
- **Combine/conflict with:** `bfp8-weights`
- **Risk:** low
- **Exemplar:** `models/tt_transformers/tt/model_config.py` (source: tech_reports/Saturating_DRAM_bandwidth)

### `core-grid-sizing` — Size the core grid per op

- **Mechanism:** Query device.compute_with_storage_grid_size() (p150 exposes fewer columns than the die) and pick grids that divide the work evenly.
- **Helps when:** Ops whose tile counts leave cores idle or unbalanced.
- **Combine/conflict with:** `matmul-2d-mcast-fuse-batch`
- **Risk:** low
- **Exemplar:** `models/experimental/chronos_forecast/tt/program_configs.py` (source: tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md)

## memory: Keep data in L1 / near the core that uses it

### `l1-interleaved-activations` — Activations resident in L1, processed in chunks

- **Mechanism:** Run the encoder on batch chunks that fit L1 (ttnn.L1_MEMORY_CONFIG), DRAM only between chunks.
- **Helps when:** Activation traffic to DRAM dominates; chronos_forecast kept L1-interleaved over block-sharded after benchmarking both.
- **Combine/conflict with:** `matmul-2d-mcast-fuse-batch`, `deallocate-early`
- **Risk:** L1 OOM at large chunks
- **Exemplar:** `models/experimental/chronos_forecast/sweeps/sweep_l1_chunk.py` (source: chronos_forecast optimizations.md)

### `sharded-activations` — Width/height/block-sharded activations

- **Mechanism:** Shard chains of ops in L1 with matching sharded program configs (matmul, layer_norm, softmax) and avoid interleaved<->sharded conversions.
- **Helps when:** Long chains of ops on the same layout; encoders like BERT.
- **Combine/conflict with:** `native-norm-softmax`
- **Risk:** conversion overhead if the chain is short
- **Exemplar:** `models/demos/sentence_bert/ttnn` (source: tech_reports/tensor_sharding)

### `deallocate-early` — Deallocate intermediates as soon as possible

- **Mechanism:** ttnn.deallocate dead tensors so larger L1 chunks/blocks fit.
- **Helps when:** Pushing chunk sizes or L1 residency.
- **Combine/conflict with:** `l1-interleaved-activations`
- **Risk:** use-after-free inside traces
- **Exemplar:** `models/tt_transformers/tt/attention.py` (source: tech_reports/LLMs/llms.md)

## numerics: Trade precision for speed per op group, measured against the FP32 oracle

### `bfp8-weights` — BFP8_B weights (and outputs) per op group

- **Mechanism:** bfloat8_b halves weight bytes; with HiFi2/LoFi matmuls in L1 it is also faster, not only smaller.
- **Helps when:** DRAM-bound matmuls; together with math-fidelity-ladder.
- **Combine/conflict with:** `math-fidelity-ladder`, `fp32-residual`
- **Risk:** accuracy; shared exponent per 16 values
- **Exemplar:** `models/tt_transformers/tt/model_config.py` (source: tech_reports/data_formats)

### `math-fidelity-ladder` — LoFi / HiFi2 / HiFi4 per op group

- **Mechanism:** LoFi is ~4x and HiFi2 ~2x HiFi4 matmul throughput per core; choose per op group, not globally.
- **Helps when:** Compute-bound matmuls; pairs with BFP8 operands.
- **Combine/conflict with:** `bfp8-weights`, `fp32-dest-acc`
- **Risk:** accuracy
- **Exemplar:** `models/tt_transformers/tt/model_config.py` (source: tech_reports/matrix_engine, tech_reports/GEMM_FLOPS)

### `fp32-dest-acc` — fp32_dest_acc_en where accumulation needs it

- **Mechanism:** FP32 destination accumulation improves long reductions but halves dst tiles (8 -> 4), shrinking matmul subblocks.
- **Helps when:** Long-K matmuls, norms and softmax that fail accuracy in bf16 accumulation.
- **Combine/conflict with:** `matmul-blocking`, `native-norm-softmax`
- **Risk:** speed
- **Exemplar:** `models/tt_transformers/tt/model_config.py` (source: tech_reports/matrix_engine)

### `packer-l1-acc` — packer_l1_acc for K-loop accumulation

- **Mechanism:** Accumulate partial sums in L1 in the packer instead of re-reading.
- **Helps when:** Matmuls with many K blocks.
- **Combine/conflict with:** `fp32-dest-acc`
- **Risk:** low (disable for SDPA where it hurt)
- **Exemplar:** `models/experimental/chronos_forecast/tt/program_configs.py` (source: chronos_forecast)

### `approx-sfpu` — Approximate SFPU math (exp, gelu, recip)

- **Mechanism:** math_approx_mode / SDPAProgramConfig(exp_approx_mode=True).
- **Helps when:** Softmax/activation-heavy ops.
- **Combine/conflict with:** `sdpa-op`
- **Risk:** accuracy
- **Exemplar:** `models/experimental/chronos_forecast/tt/mha_core.py` (source: chronos_forecast)

### `per-group-precision-sweep` — Scripted datatype/fidelity sweep per op group

- **Mechanism:** Expose per-group dtype/fidelity knobs (precision_policy groups) and sweep them against the oracle instead of one global choice.
- **Helps when:** Always after bringup; replaces one-shot global BFP8 trials.
- **Combine/conflict with:** `bfp8-weights`, `math-fidelity-ladder`
- **Risk:** search cost
- **Exemplar:** `models/tt_transformers/tt/model_config.py` (source: autoports doc/datatype_sweep, ModelOptimizations)

### `fp32-residual` — FP32 residual stream

- **Mechanism:** Keep the residual add path in fp32 while sublayers run lower precision.
- **Helps when:** Unlocks lower precision elsewhere that failed alone.
- **Combine/conflict with:** `bfp8-weights`, `native-norm-softmax`
- **Risk:** speed/memory
- **Exemplar:** `models/experimental/esm2` (source: PortForge esm2/chronos2)

## fusion: Fewer, larger device ops

### `fused-activation-matmul` — Activation fused into the matmul

- **Mechanism:** fused_activation=UnaryWithParam(...) in the matmul program config.
- **Helps when:** Linear followed by GELU/ReLU/SiLU.
- **Combine/conflict with:** `matmul-2d-mcast-fuse-batch`
- **Risk:** GELU variant must match the model (tanh vs erf)
- **Exemplar:** `models/experimental/chronos_forecast/tt/residual_block.py` (source: tech_reports/LLMs/llms.md)

### `sdpa-op` — ttnn.transformer.scaled_dot_product_attention

- **Mechanism:** Flash-style fused attention with SDPAProgramConfig (q/k chunk sizes); one chunk per sequence for short sequences.
- **Helps when:** Attention over moderate sequence lengths; check mask support against the oracle.
- **Combine/conflict with:** `approx-sfpu`, `fused-qkv`
- **Risk:** masking/padding semantics
- **Exemplar:** `models/experimental/chronos_forecast/sweeps/sweep_sdpa.py` (source: tech_reports/FlashAttention)

### `fused-qkv` — Single QKV matmul plus head split

- **Mechanism:** Concatenate Wq/Wk/Wv, one matmul, then ttnn.experimental.nlp_create_qkv_heads (or a custom QKV+RoPE kernel).
- **Helps when:** Every attention layer.
- **Combine/conflict with:** `rope-fused`, `sdpa-op`
- **Risk:** low
- **Exemplar:** `models/tt_transformers/tt/attention.py` (source: tech_reports/LLMs/llms.md)

### `rope-fused` — Fused rotary embedding with a shared cos/sin cache

- **Mechanism:** Device RoPE op (or custom kernel) with cos/sin computed once and shared across the batch.
- **Helps when:** Models with RoPE.
- **Combine/conflict with:** `fused-qkv`
- **Risk:** RoPE must be computed in fp32 (scaling variants)
- **Exemplar:** `models/experimental/chronos_forecast/sweeps/bench_rope.py` (source: chronos_forecast)

### `norm-gamma-fold` — Fold norm gamma (and other constants) into the next linear

- **Mechanism:** W' = diag(gamma) W exactly; removes an elementwise multiply per norm.
- **Helps when:** Norm followed by a linear.
- **Combine/conflict with:** `op-count-reduction`
- **Risk:** none if exact
- **Exemplar:** `models/experimental/chronos_forecast/tt/model_preprocessing.py` (source: chronos_forecast, PortForge chronos2)

### `native-norm-softmax` — Native ttnn norms/softmax with explicit compute configs

- **Mechanism:** ttnn.rms_norm / layer_norm / softmax (sharded program configs where applicable) with HiFi4 + fp32 dest acc instead of composed elementwise ops.
- **Helps when:** Composed norms/softmax are a visible share of op time. Rejected alone in the hackathon on accuracy: retry with fp32-dest-acc, fp32-residual or a different fidelity.
- **Combine/conflict with:** `fp32-dest-acc`, `fp32-residual`, `sharded-activations`
- **Risk:** accuracy
- **Exemplar:** `models/demos/sentence_bert/ttnn` (source: PortForge hackathon rejections)

## kernel: Custom TT kernels (reader/compute/writer) for what stock ops do poorly

### `generic-op-kernel` — Custom kernel via ttnn.generic_op

- **Mechanism:** KernelDescriptor/CBDescriptor reader-compute-writer kernels for fused elementwise chains, custom norms with low-precision output, or layouts stock ops handle poorly; batch several tiles per NoC barrier.
- **Helps when:** The profile shows a stock op far from its roofline or a chain of small ops.
- **Combine/conflict with:** `bank-local-elementwise`, `noc-multicast`
- **Risk:** correctness and hangs; Blackhole needs explicit NoC flushes
- **Exemplar:** `models/experimental/chronos_forecast/ops` (source: chronos_forecast; tech_reports/Blackhole)

### `bank-local-elementwise` — Bank-local elementwise ops (no NoC traffic)

- **Mechanism:** Each core reads only the tiles in its own L1 bank for residual adds and similar ops.
- **Helps when:** L1-interleaved elementwise ops.
- **Combine/conflict with:** `l1-interleaved-activations`, `generic-op-kernel`
- **Risk:** layout assumptions
- **Exemplar:** `models/experimental/chronos_forecast/ops` (source: chronos_forecast optimizations.md)

### `noc-multicast` — Multicast shared operands to a core rectangle

- **Mechanism:** One NoC multicast instead of N unicast reads of the same weights/tables (Blackhole supports strided and L-shaped multicast).
- **Helps when:** Many cores read identical data.
- **Combine/conflict with:** `generic-op-kernel`
- **Risk:** hangs if semaphores are wrong
- **Exemplar:** `tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md` (source: Blackhole bring-up guide; arXiv 2606.09879)

## scaleout: More than one card or concurrent sub-devices

### `data-parallel-mesh` — Data parallel across a card mesh

- **Mechanism:** Replicate weights on a MeshDevice and shard the batch (chronos_forecast: 4x p150 ~4.36x).
- **Helps when:** Throughput objectives with more than one card.
- **Combine/conflict with:** `trace-capture`
- **Risk:** needs mesh access in the sandbox
- **Exemplar:** `models/experimental/chronos_forecast/tests` (source: chronos_forecast)

### `subdevice-prefetch` — Sub-devices with a DRAM prefetcher

- **Mechanism:** Split the grid into sub-devices; a prefetcher streams the next weights into global CBs while compute runs.
- **Helps when:** DRAM-bound decode on large models.
- **Combine/conflict with:** `dram-sharded-matmul`
- **Risk:** complexity
- **Exemplar:** `models/tt_transformers/tt/prefetcher.py` (source: tech_reports/SubDevices)
