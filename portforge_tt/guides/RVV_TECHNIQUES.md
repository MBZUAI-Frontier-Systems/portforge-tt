# RVV techniques

The technique catalogue of kernel campaigns: the controller parses the `## <category>` and
``### `<id>`: <title>`` headings (ranking/search.py), so keep that shape when adding entries.
Whether a technique worked is recorded per parent and combination in each run's search tree,
not here: a technique rejected once is still worth combining with others.

## config

vector configuration: SEW, LMUL, vsetvl placement

### `max-lmul`: Raise LMUL until the widest stage is m8

- **Mechanism.** Larger register groups mean fewer instructions per element and longer chimes that hide latency.
- **Helps when.** The hot loop runs at LMUL <= 2 and has no spills; latency-bound chains (mul, fixed-point) at small LMUL.
- **Bound.** compute; knobs: LMUL
- **Correctness risk.** low: register-group alignment errors at compile time
- **Preconditions.** widest stage of a widening chain must stay within m8
- **Verifier.** SMT yes
- **X280.** ALU cost is 2 cycles per LMUL (DLEN 256) so throughput per element is flat, but latency (add 3, mul 7 cycles at m1/m2) is hidden only from m4 up
- **Interactions.** synergy: vla-stripmine; conflicts: multi-accumulator-unroll
- **Exemplar.** xnnpack:`src/qs8-vcvt/gen/qs8-vcvt-rvv-u4v.c`

### `vsetvl-hoist`: Fewer vsetvli per iteration; group same-vtype ops

- **Mechanism.** Order instructions so ops sharing SEW/LMUL are adjacent; set vtype once per stage.
- **Helps when.** The static profile shows more vsetvli per iteration than distinct stages.
- **Bound.** compute
- **Correctness risk.** low: using a stale vl after a vtype change
- **Preconditions.** none
- **Verifier.** SMT yes
- **X280.** a vtype-changing vsetvli costs ~1 cycle, small next to 8-16 cycle m4/m8 ops

### `no-whole-register-moves`: Avoid vmv<n>r copies in the hot loop

- **Mechanism.** Structure data flow so values are produced in the register group that consumes them.
- **Helps when.** The profile shows vmv1r/2r/4r/8r inside the loop (load-ahead pipelining often introduces them).
- **Bound.** compute
- **Correctness risk.** low
- **Preconditions.** none
- **Verifier.** SMT yes
- **X280.** a vmv4r costs ~8 cycles: load-ahead's copy made it 0.97x
- **Interactions.** conflicts: load-ahead

## isel

instruction selection: cheaper or fused instructions for the same math

### `vnclip-requant`: Use vnclip for shift + round + saturate + narrow

- **Mechanism.** One vnclip with the right vxrm replaces separate rounding shift, clamp and narrow.
- **Helps when.** A requantization tail shifts, saturates and narrows.
- **Bound.** compute; knobs: vxrm
- **Correctness risk.** medium: wrong rounding mode (ties)
- **Preconditions.** rounding mode must match the scalar kernel's (round-half-up is RNU)
- **Verifier.** SMT yes
- **X280.** vnclip costs 4 cycles per LMUL of the destination (reads a 2x source)
- **Interactions.** synergy: fold-bias-scalar
- **Exemplar.** xnnpack:`src/qs8-vcvt/gen/qs8-vcvt-rvv-u4v.c`

### `vx-scalar-operands`: Scalar-operand forms (.vx/.vi/.vf) for broadcast parameters

- **Mechanism.** Use the scalar register operand instead of splatting a parameter into a vector register.
- **Helps when.** The loop holds splatted constants that cost registers (forcing lower LMUL) or a vmv.v.x per iteration.
- **Bound.** compute
- **Correctness risk.** low: immediate out of range
- **Preconditions.** scalar operand fits the element width (vx sign-extends from XLEN; .vi is 5-bit)
- **Verifier.** SMT yes
- **X280.** same cost as .vv; the win is register pressure
- **Interactions.** synergy: max-lmul

### `fused-macc`: vmacc/vmadd (and vwmacc) instead of vmul + vadd

- **Mechanism.** One multiply-accumulate replaces two ALU ops.
- **Helps when.** A multiply is immediately followed by an add of a vector or broadcast value.
- **Bound.** compute
- **Correctness risk.** medium: float fused vs unfused ulp differences
- **Preconditions.** integer: exact; float: vfmacc is fused - only if the reference is fused too (Neon vmlaq is not)
- **Verifier.** SMT yes
- **X280.** vmacc throughput equals vadd (2 cycles per LMUL), so one op saved is a full op
- **Interactions.** synergy: fold-bias-scalar

### `widen-fused`: Widening arithmetic instead of extend + op

- **Mechanism.** vwadd/vwsub/vwmul widen as part of the arithmetic, saving a separate vsext/vzext.
- **Helps when.** The loop extends an input and then immediately adds/subtracts/multiplies it.
- **Bound.** compute
- **Correctness risk.** low
- **Preconditions.** none
- **Verifier.** SMT yes
- **X280.** vwadd costs the same as vsext at the source LMUL
- **Interactions.** synergy: narrow-sew-chain
- **Exemplar.** xnnpack:`src/qs8-vcvt/gen/qs8-vcvt-rvv-u4v.c`

### `branchless-merge`: vmerge or arithmetic masks instead of data-dependent branches

- **Mechanism.** Compute both sides and select with vmerge, or fold the selection into arithmetic (and/xor for lrelu).
- **Helps when.** Per-element conditions such as leaky-relu slopes, clamps, argmax updates.
- **Bound.** compute
- **Correctness risk.** medium: NaN and signed-zero ordering (SALTyRN argmaxpool)
- **Preconditions.** NaN/±0 behaviour must match the scalar compare (vfmax differs from compare+select)
- **Verifier.** SMT yes
- **X280.** vmerge costs like vadd
- **Exemplar.** portforge:`ports/xnnpack-rvv/kernels/qs8-vlrelu/kernel.c`

### `explicit-nan-compare`: Explicit compare+select where the reference does, not vfmin/vfmax

- **Mechanism.** Mirror the scalar kernel's comparisons so NaN and ±0 results match bit for bit.
- **Helps when.** Float min/max/argmax/clamp kernels.
- **Bound.** compute
- **Correctness risk.** low
- **Preconditions.** none
- **Verifier.** SMT yes
- **X280.** same cost class as vfmax

## algorithm

change the arithmetic itself while staying bit-exact

### `narrow-sew-chain`: Do the arithmetic at the narrowest exact width

- **Mechanism.** Prove a bound on every intermediate and run the chain at e16 instead of e32 (or e8 instead of e16): half the register groups, half the cycles per element.
- **Helps when.** A widening chain goes to i32 but the values provably fit i16 for the whole producer domain.
- **Bound.** compute
- **Correctness risk.** high: overflow at parameter extremes (SALTyRN qs8-vcvt: (int16)(m<<7) at m>=256)
- **Preconditions.** range proof over the init's whole output domain (not the test values); intermediate products and sums, not only inputs
- **Verifier.** SMT yes
- **X280.** cost per instruction is per LMUL, so halving SEW at fixed element count halves LMUL and the cycles
- **Interactions.** synergy: regime-dispatch
- **Exemplar.** portforge:`ports/xnnpack-rvv/kernels/qs8-vcvt/kernel.c`

### `single-op-regime`: Collapse a regime to one saturating op

- **Mechanism.** When the requantization is an exact shift by zero (m == 256 in qs8-vcvt), the whole chain is one e8 saturating add/subtract of a scalar.
- **Helps when.** A regime where multiplier and shift cancel; usually the identity/default quantization.
- **Bound.** memory
- **Correctness risk.** medium: scalar operand outside int8 range
- **Preconditions.** the scalar operand fits the narrow element type, else a widened path
- **Verifier.** SMT yes
- **X280.** a 3-instruction loop is memory-bound at 32 B/cycle in L1
- **Interactions.** synergy: regime-dispatch; requires: regime-dispatch
- **Exemplar.** portforge:`ports/xnnpack-rvv/kernels/qs8-vcvt/kernel-v2.c`

### `fold-bias-scalar`: Fold zero points and rounding into one scalar bias

- **Mechanism.** Move every per-call constant (zero points times multipliers, output zero point shifted, rounding constant) into a scalar computed once, so the loop does mul + add + narrow.
- **Helps when.** Requantization with zero points applied per element.
- **Bound.** compute
- **Correctness risk.** high: bias overflow at shift extremes; double-counted rounding
- **Preconditions.** the folded bias must not overflow its type for any producer parameters (SALTyRN qs8-vadd: ozp<<shift wraps int32 at shift 30); do not re-add a term the init already folded (SALTyRN qu8-vadd)
- **Verifier.** SMT yes
- **X280.** removes one ALU op per element: ~20% of a 5-op chain
- **Interactions.** synergy: regime-dispatch, vx-scalar-operands
- **Exemplar.** portforge:`ports/xnnpack-rvv/kernels/qs8-vcvt/kernel-v2.c`

## dispatch

specialize on runtime parameters

### `regime-dispatch`: Dispatch on runtime parameters to the cheapest exact path

- **Mechanism.** Check scalar parameter predicates once per call and run a loop specialized for that regime (fewer ops where the math simplifies), keeping a general path.
- **Helps when.** The formula collapses for common parameter values (multiplier a power of two, unit scale, zero offsets).
- **Bound.** compute
- **Correctness risk.** medium: off-by-one predicate boundary (an SMT-config gate missed one for vcvt pf2)
- **Preconditions.** each predicate must make its path exact on the regime's whole domain; dispatch is by proven predicate, never by special-casing test values; regimes complete and disjoint over the init's range
- **Verifier.** SMT yes
- **X280.** branches before the loop cost a few cycles; libgcc calls in the predicate (ctz without Zbb) cost ~18% at batch 256
- **Interactions.** synergy: fold-bias-scalar, single-op-regime
- **Exemplar.** portforge:`ports/xnnpack-rvv/kernels/qs8-vcvt/kernel-v2.c`

### `runtime-vlen-tiles`: Pick unroll/LMUL from vlenb at runtime

- **Mechanism.** Choose among compiled variants once per process from the actual VLEN, instead of a compile-time choice.
- **Helps when.** A family's best variant differs between VLEN 128 and 512/1024 (check QEMU instruction counts and the board).
- **Bound.** any
- **Correctness risk.** low
- **Preconditions.** purely a speed choice: every variant must be exact
- **Verifier.** SMT yes
- **X280.** only VLEN 512 can be timed here
- **Interactions.** synergy: max-lmul

## memory

how data moves between memory and registers

### `load-ahead`: Software-pipeline the next load

- **Mechanism.** Issue the next iteration's load before this iteration's compute so memory latency overlaps.
- **Helps when.** Memory latency is exposed (L2/DRAM working sets) and the loop is short.
- **Bound.** memory
- **Correctness risk.** medium: out-of-bounds read at the tail
- **Preconditions.** must not read past XNN_OOB_READS slack
- **Verifier.** SMT partial
- **X280.** rejected on qs8-vcvt in L1 (0.97x): the vector unit is decoupled already
- **Interactions.** conflicts: no-whole-register-moves
- **Exemplar.** portforge:`ports/xnnpack-rvv/kernels/qs8-vcvt/kernel-loadahead-pipelined-rejected.c`

### `avoid-gather`: Slides or segment loads instead of vrgather

- **Mechanism.** Replace arbitrary register permutations with vslide or vlseg/vsseg for regular patterns.
- **Helps when.** The profile shows vrgather/vcompress in the hot loop.
- **Bound.** compute
- **Correctness risk.** medium: lane-order mistakes
- **Preconditions.** none
- **Verifier.** SMT partial (segment-tuple types (vuint32m2x4_t) have no SALTyRN shims)
- **X280.** vrgather is element-serial: 68 cycles at e8m1 (64 elements), 516 at e8m8
- **Interactions.** synergy: segment-loads

### `gather-low-lmul`: If a gather is unavoidable, run it at LMUL 1

- **Mechanism.** Split a high-LMUL vrgather into LMUL=1 gathers; cost grows with the table size per instruction.
- **Helps when.** Lookup-table kernels (vrgather as LUT) at LMUL >= 2.
- **Bound.** compute
- **Correctness risk.** medium: index out of the table
- **Preconditions.** the table fits one register (VLEN/8 bytes at e8)
- **Verifier.** SMT partial
- **X280.** measured gather cost doubles per LMUL step (68/132/260/516 cycles at e8)

### `segment-loads`: Unit-stride segment loads/stores for interleaved data

- **Mechanism.** vlseg<nf>/vsseg<nf> de-interleave AoS to SoA in one instruction instead of strided loads.
- **Helps when.** Interleaved layouts: complex numbers (vcmul), channels (packw), transposes.
- **Bound.** memory; knobs: NF
- **Correctness risk.** medium: field order
- **Preconditions.** none
- **Verifier.** SMT no (tuple types are not shimmed: gate with the exact check)
- **X280.** e32, VLEN 512, L1, 128-byte stride: strided segment loads vlsseg2/4/8 at m1 take 17/19/32 cycles (about one segment per cycle up to nf=4: 4 elements/cycle at nf=8); unit-stride vlseg2 m1 5 cycles, vlseg4 19, vlseg8 32
- **Interactions.** synergy: avoid-gather
- **Exemplar.** xnnpack:`src/x32-transposec/rvv-u.c.in`

### `strided-avoid`: Prefer unit-stride over strided or indexed memory ops

- **Mechanism.** Re-layout or block the loop so the vector unit reads contiguous memory.
- **Helps when.** vlse/vsse or vluxei in the hot loop.
- **Bound.** memory
- **Correctness risk.** low
- **Preconditions.** none
- **Verifier.** SMT yes
- **X280.** measured: vlse32 is 1 element/cycle (16 cycles at m1, 64 at m4), vsse32 0.5 element/cycle (32 at m1), indexed vluxei32/vsuxei32 about 2 cycles per element; unit vle32/vse32 cost 5 cycles at m1-m2 and 8 at m4 (32 B/cycle)
- **Interactions.** synergy: segment-loads

### `fewer-stores`: Fewer, wider stores; no stores of intermediates

- **Mechanism.** Keep intermediates in registers; store once per output element at the narrowest type.
- **Helps when.** The loop writes intermediates or stores at a wider type than needed.
- **Bound.** memory
- **Correctness risk.** low
- **Preconditions.** none
- **Verifier.** SMT yes
- **X280.** L1 stores 32 B/cycle; L2 copy only ~2.6 B/cycle

### `segment-stores`: Strided segment stores instead of strided segment loads

- **Mechanism.** vssseg<nf> writes nf consecutive fields per segment at a byte stride. A 2-D reorder (transpose, packing) can load contiguous rows with unit-stride vle and scatter them with one strided segment store, instead of gathering with vlsseg and storing unit-stride.
- **Helps when.** Data movement kernels whose gather side uses strided or strided-segment loads (x32-transposec, packw).
- **Bound.** memory; knobs: nf, lmul
- **Correctness risk.** medium: field order swapped (rows vs columns); writes into row padding past block_height
- **Preconditions.** nf * LMUL <= 8; output segments (nf consecutive elements) are contiguous in the destination
- **Verifier.** SMT no shim for tuple intrinsics (exact check only)
- **X280.** measured e32, VLEN 512, L1: vssseg2/4/8 at m1 = 16 cycles whatever nf (one segment per cycle: 8 elements/cycle at nf=8), twice the rate of vlsseg8 (32 cycles); vssseg2 m4 64 cycles; unit vle32 m4 loads 64 elements in 8 cycles
- **Interactions.** synergy: max-lmul, cache-blocking

## loop

loop structure: strip-mining, unrolling, pipelining, hoisting

### `vla-stripmine`: One vsetvl-driven loop; the tail is a smaller vl

- **Mechanism.** Replace a fixed-width main loop plus scalar tail with one loop whose vl comes from the remaining count.
- **Helps when.** Ported from Neon/SSE with a scalar tail; small batches dominated by tail code.
- **Bound.** any
- **Correctness risk.** low: off-by-one vl at the last iteration
- **Preconditions.** batch is in bytes for many XNNPACK families: divide by element size once
- **Verifier.** SMT yes
- **X280.** baseline structure of every merged RVV kernel
- **Interactions.** synergy: max-lmul
- **Exemplar.** xnnpack:`src/qs8-vcvt/gen/qs8-vcvt-rvv-u4v.c`

### `multi-accumulator-unroll`: Unroll with several independent accumulators

- **Mechanism.** Split a loop-carried dependence (reductions, running max) across accumulators to hide latency.
- **Helps when.** Reductions or running extrema at LMUL <= 2 where latency (7 cycles for mul) is exposed.
- **Bound.** compute; knobs: ACC
- **Correctness risk.** medium: float reduction order changes results
- **Preconditions.** integer only, or float when the reference order is not part of the contract
- **Verifier.** SMT yes
- **X280.** at m4/m8 latency is already hidden, so extra accumulators mostly add register pressure
- **Interactions.** synergy: deferred-reduction; conflicts: max-lmul

### `deferred-reduction`: Element-wise accumulate in the loop, reduce once at the end

- **Mechanism.** Keep a vector accumulator with vadd in the loop and do one vredsum after it.
- **Helps when.** A reduction calls vredsum (or similar) every iteration.
- **Bound.** compute
- **Correctness risk.** medium: accumulator overflow on long rows
- **Preconditions.** accumulator width must not overflow over the longest batch
- **Verifier.** SMT partial
- **X280.** vredsum costs 29-67 cycles vs 2-16 for vadd
- **Interactions.** synergy: multi-accumulator-unroll

### `hoist-invariants`: Hoist loop-invariant work and splats out of the loop

- **Mechanism.** Compute splats, rounding constants and derived scalars once before the loop.
- **Helps when.** vmv.v.x or scalar recomputation inside the hot loop.
- **Bound.** compute
- **Correctness risk.** low
- **Preconditions.** none
- **Verifier.** SMT yes
- **X280.** scalar work overlaps with the decoupled vector unit; vector splats cost a full op
- **Interactions.** synergy: vx-scalar-operands

### `cache-blocking`: Walk large 2-D blocks in cache-sized sub-blocks

- **Mechanism.** When one call covers a block larger than L1, iterate over sub-blocks sized so every input and output cache line touched is reused before eviction, instead of streaming whole column strips.
- **Helps when.** Large single calls (transposes, packing) where measured cycles per element jump at sizes beyond L1/L2, especially with power-of-two strides that alias cache sets.
- **Bound.** memory; knobs: sub_block
- **Correctness risk.** low: edge sub-blocks skipped or done twice
- **Preconditions.** results do not depend on traversal order (pure data movement or independent outputs)
- **Verifier.** SMT yes (loop structure only)
- **X280.** L1 D 32 KiB-class working set (32 B/cycle), L2 ~13 B/cycle, DRAM ~0.7 B/cycle; upstream x32 transposes run 13-14 cycles/element on 1024x1024 (4096-byte strides) vs ~1 cycle/element at 256x256
- **Interactions.** synergy: segment-stores

## policy

tail/mask policies and predication

### `tail-agnostic`: Tail/mask agnostic policies (ta, ma)

- **Mechanism.** Use _ta/_ma intrinsics so destinations need not be read; undisturbed policies add a dependency.
- **Helps when.** The kernel uses _tu/_mu intrinsics without needing the old values.
- **Bound.** compute
- **Correctness risk.** medium: garbage in lanes that are later stored
- **Preconditions.** only when the tail/masked-off lanes are never stored
- **Verifier.** SMT yes (_tu/_mu variants have NO shims: using them forfeits the SMT gate)
- **X280.** not yet calibrated

### `vl-predication`: Restrict work with vl instead of masks

- **Mechanism.** When only leading elements are active, shorten vl rather than computing and masking all lanes.
- **Helps when.** A mask selects a prefix of the vector.
- **Bound.** compute
- **Correctness risk.** low
- **Preconditions.** none
- **Verifier.** SMT yes
- **X280.** not yet calibrated

## scalar

the scalar side: prologue, bookkeeping, calls

### `avoid-libcalls`: No libgcc calls in the prologue or loop

- **Mechanism.** Replace __ctzdi2/__clzdi2/__divdi3-style calls with arithmetic the target has (XNNPACK compiles RVV with -march=rv64gcv: no Zbb), or with a vector equivalent, or precompute in the init.
- **Helps when.** The static profile lists calls; small batches dominate.
- **Bound.** any
- **Correctness risk.** low
- **Preconditions.** none
- **Verifier.** SMT yes
- **X280.** pf2's __ctzdi2 call cost ~18% at batch 256
- **Interactions.** synergy: regime-dispatch

## offload

north star: work moved to other engines on the same card

### `tensix-offload`: North star: offload a bulk stage to the Tensix cores on the same Blackhole

- **Mechanism.** The X280 drives control flow; a dense, regular stage (GEMM-like, stencil, LUT over large batches) runs on Tensix cores reached over the NoC, if the round trip amortizes.
- **Helps when.** Large batches of BF16-friendly dense work; never for tiny calls.
- **Bound.** any
- **Correctness risk.** high: BF16/SFPU numerics differ from the CPU reference
- **Preconditions.** TT-Metal and tt-bh-linux must coexist on the card (unverified); end-to-end time including the round trip must beat RVV
- **Verifier.** SMT no
- **X280.** no published X280<->Tensix work; spike in research/tt-offload first
