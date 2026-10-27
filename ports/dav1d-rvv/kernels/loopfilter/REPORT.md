# REPORT: loop filter RVV port (8bpc) - campaign 1, Part B

Status: **RED - honest stop at the budget line.** The port is written,
wired, assembles clean, and runs end-to-end (no crashes) on QEMU and the
X280; the hev path, the flat6/flat8/flat16 arithmetic and the wrapper
control flow are verified correct in isolated exact-value tests. checkasm
still reports data mismatches on all 10 8bpc `lpf_*` tests
(`loopfilter.c:184`, bit-exactness), so the campaign gate is **not** met.
This file documents what exists, what is proven, what is broken, and the
fastest path to green for the next campaign.

## Deliverables in this directory

- `loopfilter.S` (= `lf-final.S`): `src/riscv/64/loopfilter.S` -
  4 exported wrappers (`dav1d_lpf_{h,v}_sb_{y,uv}_8bpc_rvv`,
  `export=1, ext="v"`) + internal `lpf_core_rvv`.
- `lf-init.h` (= `loopfilter-init.h`): `src/riscv/loopfilter.h` -
  decls + inline `loop_filter_dsp_init_riscv()` guarded by
  `DAV1D_RISCV_CPU_FLAG_V` and `BITDEPTH == 8`.
- `lf-wiring.diff`: the two-line library diff in
  `src/loopfilter_tmpl.c` (ARCH_RISCV include + `loop_filter_dsp_init_riscv(c)`
  call - the documented "no riscv hook existed" diff) and `src/meson.build`
  (registering `riscv/64/loopfilter.S` in the 8bpc asm list).
- `lf_diff.c`: exact differential driver (C `loop_filter` verbatim as
  reference) - the tool that pinned two real asm bugs (below).
- `lf_run.c`, `lf_run2.c`: earlier crash-hunt drivers.
- `mc-fix.diff`: stale copy - see `kernels/mct-hardware/mc.S-fix.diff`.

## Design (as implemented)

C reference: `src/loopfilter_tmpl.c` `loop_filter()` + the four sb128
wrappers. Per set vmask bit, the C filters 4 independent pixel rows
(horizontal, strideb=1) or columns (vertical, strideb=stride) with the
13-tap AV1 deblocking filter of width wd in {4,6,8,16}.

- Wrappers: vmask iteration and per-bit `L -> (E,I,H,wd)` lookups stay
  scalar (control flow, not pixel data), mirroring the C exactly; loop
  state lives in s0-s7 (the core clobbers all temporaries).
- `lpf_core_rvv(dst, E, I, H, stridea, strideb, wd)`: vectorizes the 4
  independent lines of one bit as 4 vector lanes. VL is always exactly 4
  (the C loop bound is the constant 4) - no setvl tail, no scalar
  epilogue. Taps p6..q6 are strided loads (`vlse8.v`, stride = stridea,
  base = dst + k*strideb), widened via `vzext.vf2`; stores are the
  mirrored strided `vsse8.v` masked with `v0.t`. All arithmetic is
  lane-wise e16 replicating the C int semantics, including iclip_diff's
  [-128,127] clamp and iclip_pixel's [0,255].
- Per-lane flat/hev decisions are mask registers: the plain (hev-branch)
  stores go first with fm / fm&!hev masks, then the flat paths re-store
  the lanes they own (same addresses - last masked store per lane wins,
  exactly like the C if/else-if chain). A lane the C does not write
  (dst[-2] of a hev lane) is never stored.
- Flat outputs are computed as `round(Sum(w_i * tap_i) + r) >> s` via
  `vmacc.vx` with scalar weights - 6 outputs for flat8, 4 for flat6,
  12 for flat16; wd16 lanes that are flat8in-but-not-flat16 run the flat8
  sub-path (the C else-if chain).
- `ext="v"` only: no Zbb/Zba instructions are used (board-safe plain
  rv64gcv; the board's isa string has zba/zbb but the code avoids them).

## What is proven correct

- Deterministic exact-value tests (`lf_diff.c` known test + H=0
  sub-cases, fixed taps, expected outputs computed by hand): the hev
  path with H>0 and H=0, including the `hev ? iclip_diff(p1-q1) : 0`
  merge, f1/f2 derivation and the [-2]/[+1] non-hev writes, matches C
  bit-exactly for all 4 lanes.
- Full single-edge sweeps: 3200 randomized edges (all wd, both
  directions, 400 layouts each) bit-exact.
- Multi-edge blocks (32 bits set, per-bit random L): wd8/wd16/wd6
  classes bit-exact in the differential (after the fixes below).

## Bugs found and fixed during Part B (all real, all in the new asm)

1. `vzext.vf2` with vd == vs2 - illegal-instruction on QEMU; taps are now
   staged through v19.
2. load_tap/store_lane offset macros: the -6 offset computed 8*strideb and
   the -7 branch had been lost in an earlier edit (wd16 read p5/p6 from the
   wrong addresses). Rewritten with a programmatically balanced if-chain.
3. Wrapper loop state (vm, y, vmask words, l pointer) lived in t-registers
   that the core clobbers; the l pointer was also destroyed by the per-bit
   H argument. Moved to s0-s7 with save/restore, H passed via t4.
4. hev-path values used v12/v13 - the wd16 q4/q5 tap registers - corrupting
   the flat16 sums. Moved to dead-after-use v15/v19.
5. `3*(q0-p0)` was computed as `2*(q0-p0) + q0` (wrong: adds q0, not the
   difference). Fixed to `2*(q0-p0) - (p0-q0)`. This was found by the
   exact-value sub-tests after a fork()-in-driver red herring (child writes
   are COW-invisible to the parent - the differential driver must not fork;
   that artifact cost the most time).

## What is still red

checkasm, seed 1, QEMU 512, `--function=lpf_*`:
**10 of 10 tests failed** (all `lpf_*_8bpc_rvv`, all at
`tests/checkasm/loopfilter.c:184` = the pixel comparison; no crashes).
The X280 agrees with QEMU (same mismatches), so this is a plain
functional gap, not a hw/qemu disagreement.

Remaining discrepancy (not root-caused within budget): under checkasm's
exact input geometry (135-pitch h buffers / 128-pitch v buffers, 16-pixel
packing, randomized per-bit L), some lines diverge where my standalone
differential (32-byte pitch, same wrapper/core code, same C reference)
matches. The residual suspects, in order of likelihood:

1. The differential's buffer geometry still differs from checkasm's
   (16/128-pitch packing with edges at offset 8; ±7 taps reach the
   neighboring 16-byte rows in the h case - checkasm's lines are packed
   16 bytes apart, so a stray write one row off is invisible in my
   32-byte-pitch layout but visible in checkasm's).
2. Lane-to-line mapping at the wrapper boundary for the v direction
   (consecutive-byte lines): a one-byte line-base error would be invisible
   in my h-focused sweep but shifts every v line.
3. The flat16 sub-path selection for wd16 (flat8in-but-not-flat16 lanes
   running flat8) is only exercised by checkasm's w16 test; my
   differential's wd16 class forces all bits into vmask[0..2] but may not
   produce flat8in-without-flat16 lines as often.

Next campaign should start from: reproduce with checkasm itself under
`-d in_asm` + the ptrace tracer (crash-free, so instead use a
differential that copies checkasm's `init_lpf_border` verbatim - including
its 16-byte pitch and `b4_stride=2/32` - into `lf_diff.c`), then bisect
the wd4 hev path first (smallest code).

## Honest negatives

- No checkasm gate is green for the loop filter: not QEMU, not the board.
- No benchmark numbers were taken (correctness gate unmet; per CONTRACT,
  benchmarking only after correctness).
- 16bpc: not started (blocked behind 8bpc green, as planned).
- The debug markers/instrumentation in the drivers (`lf_diff.c`) are
  development leftovers; the asm itself is clean of them.

## Seeds log

QEMU 512 `--function=lpf_*` seed 1 (xN during bisection), QEMU
128/256/512/1024 seed 1 (identical 10/10 failures each - confirming the
gap is VLEN-independent), board `--function=lpf_*` seed 1 (10/10, same).
All runs through the supported `dav1d.run_checkasm`/`x280` APIs;
every failure kept verbatim above and in
`~/portforge-cpu/results/mct-hardware/` on the box.
