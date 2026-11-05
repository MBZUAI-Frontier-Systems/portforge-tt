# Lessons from earlier kernel campaigns (binding rules)

Distilled from the XNNPACK (qs8-vcvt, qs8-vadd-minmax, qs8-vlrelu, x32-transposec) and dav1d
(splat_mv, wiener, sgr, loopfilter, load_tmvs) campaigns on the SiFive X280, October-November 2026.
Each rule cost hours when it was broken.

## Correctness

1. **Every VLEN, not the board's.** The X280 has VLEN 512, but the gates run 128, 256, 512 and 1024.
   Upstream XNNPACK transposes that hard-code 8 or 16 lanes are wrong at VLEN 128 (and 32x8 at 512).
   Strip-mine with `vsetvli` and derive every tile size from `vl` or `vlenb`.
2. **QEMU passing is not hardware passing.** Strided accesses whose offset is not a multiple of the
   element width are implementation-defined; one dav1d kernel passed QEMU and failed on the X280.
   The perf stage re-checks on the board before timing.
3. **The domain is the init's, not the tests'.** Width bounds come from what the library's init can
   produce (every quantization parameter, every tail), never from test values. Every narrowing needs a
   range argument in NOTES.md; every dispatch predicate needs a proof that each path is exact on its
   whole domain. Held-out regimes are drawn from that domain at certification.
4. **Tails and edges are where ports break.** Run `port_run(command="check")` after every change; a
   kernel that is right on full vectors and wrong on the last 3 elements is wrong.

## Performance on the X280

5. **ALU ops cost about 2 cycles per LMUL, narrowing about 4; latency hides only from m4 up.** Raise
   LMUL until the widest stage of a widening chain is m8 unless the profile shows spills.
6. **`vrgather` is element-serial** (cost grows with VL): prefer slides, segment loads or a small
   in-register table at LMUL 1. A table gather won +45% in sgr3x3 only because the table fit one register.
7. **Bandwidth bounds large cells.** L1 is about 32 bytes/cycle, DRAM about 0.7: 4M-element cells sat at
   1.00x for every candidate. Do not spend turns on ALU tricks for DRAM-bound cells; the profile's bound
   class says which cells can still move.
8. **No libcalls in the hot loop.** XNNPACK builds RVV with `-march=rv64gcv` (no Zbb), so
   `__builtin_ctz` becomes a `__ctzdi2` call: one such call cost a 0.79x cell.
9. **Linux clobbers `vtype` on every syscall**; never rely on vector state across a call.
10. **Code alignment moves timings by ~2%.** Identical kernels differed 2.4% unaligned and under 1%
    aligned; treat sub-2% differences between structurally different kernels with suspicion.

## Measurement

11. **Only perf receipts rank.** Static profiles predict X280 cycles within a few percent for simple
    loops; use them to choose what to try, never as the result.
12. **One flagged cell is a suspicion.** Cross-session drift at DRAM-sized shapes exceeds the
    in-session noise floor; a flag counts only when a second session at twice the rounds repeats it.
13. **The board is shared.** The harness waits for it to be idle and leases it; never time anything
    yourself and never interrupt another job.

## Search

14. **A rejected technique is rejected for one parent.** Width dispatch (+57% in splat_mv) and the
    edge pre-pass (+13.7% in wiener) both lost earlier under other parents. Record every attempt in
    TRANSFORMS.json so the search can tell combinations apart.
15. **Read the other ISAs' implementations, then design for RVV.** The arm64 and x86 decompositions
    are a good start, but RVV's length-agnostic loops, masks and widening ops often allow a simpler
    structure (fewer passes, no transposes).
