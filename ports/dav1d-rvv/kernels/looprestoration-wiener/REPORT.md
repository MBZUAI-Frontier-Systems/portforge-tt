# REPORT: loop-restoration Wiener filter RVV port (8bpc) — campaign 2 (`c2-wiener`)

Status: **GREEN.** From-scratch RVV port of dav1d's 8bpc loop-restoration
Wiener filter (`wiener[0]` = 7-tap, `wiener[1]` = 5-tap). The looprestoration
family had zero RISC-V support upstream (pure C); everything here is additive.
All three gates pass: checkasm bit-exact under QEMU at VLEN 128/256/512/1024
and natively on the X280 (full suite 918/918 on both), a 5-level targeted
differential driver is clean, and `checkasm --bench` on the X280 measures
**~5.0–5.2× over the C reference**. SGR (`sgr_3x3` stretch goal): **not
started** — an honest not-started beats a half-working state.

## Derived spec (from `src/looprestoration_tmpl.c`, 8bpc constants)

The block function `wiener_c(p, stride, left, lpf, w, h, params, edges)` is a
separable 7-tap filter with a uint16 intermediate buffer (6 rows ×
`REST_UNIT_STRIDE = 390` = 256·1.5 + 3 + 3) and edge flags
LR_HAVE_{LEFT,RIGHT,TOP,BOTTOM}. Per-row `left` (4 px) supplies out-of-buffer
columns on the left; `lpf` (2 top rows + 2 bottom rows at `lpf + 6·stride`)
supplies rows above/below the filtered area.

**Horizontal pass** (`wiener_filter_h`, 8bpc):

```
round_bits_h   = 3                      # 3 + (bitdepth == 12) * 2
rounding_off_h = 1 << 2                 # = 4
clip_limit     = 1 << (8 + 1 + 7 - 3)   # = 8192
sum  = 1 << 14                          # 1 << (bitdepth + 6)
sum += src[x] * 128                     # 8bpc-only center contribution
sum += Σ_{i=0..6} fh[i] · t(x + i - 3)  # fh[3] is applied AND +128·src[x]
out[x] = iclip((sum + 4) >> 3, 0, 8191) # arithmetic shift; u16 intermediate
```

with, per index `j = x + i - 3`:
`j < 0`: `left[0][4+j]` if a left buffer is present, else `src[j]` (reads
3 px before the row) when LR_HAVE_LEFT, else `src[0]`;
`j ≥ w`: `src[j]` (reads up to 3 px past) when LR_HAVE_RIGHT, else `src[w-1]`.
The effective center tap is `fh[3] + 128` (checkasm fills
`filter[0][3] = -(f0+f1+f2)·2` for 8bpc so DC gain = 128); both contributions
are real and both are applied.

**Vertical pass** (`wiener_filter_v` / the v part of `wiener_filter_hv`, 8bpc):

```
round_bits_v   = 11                     # 11 - (bitdepth == 12) * 2
rounding_off_v = 1 << 10                # = 1024
round_offset   = 1 << (8 + 10)          # = 262144
sum  = -262144 + Σ_{k=0..5} rows[k][i] · fv[k] + row6[i] · fv[6]
out[i] = iclip_pixel((sum + 1024) >> 11)  # arithmetic shift; clamp [0,255]
```

`row6` = `ptrs[5]` (7th row duplicates the 6th) in the pure vertical pass, or
the freshly filtered row in the combined hv pass. The two additive constants
fold exactly into one `Σ - 261120` (no intermediate rounding), then `>> 11`
(arithmetic), then clamp.

**Tap ranges exercised** (from checkasm's generator = the legal envelope):
f0 ∈ [-5,10], f1 ∈ [-23,8], f2 ∈ [-17,46], f3 = 128 − 2·(f0+f1+f2) (vertical)
or −2·(f0+f1+f2) (horizontal, 8bpc). 5-tap = 7-tap with f0 = f6 = 0, so both
entry points share one driver and one set of kernels.

## Design as implemented

- `src/riscv/64/looprestoration.S` (only `ext=v`, plain RVV 1.0 — no Zbb/Zba;
  matches both the X280 and the QEMU cpu model, which has neither):
  - `dav1d_wiener_filter_h_8bpc_rvv(uint16_t *dst, const pixel (*left)[4],
    const pixel *src, const int16_t fh[8], int w, enum LrEdgeFlags edges)`:
    materializes the padded row `pad[0..w+5]` (w+6 ≤ 390 bytes, stack) with the
    C's three head cases and two tail cases, then convolves in vector blocks:
    7 sliding `vle8` loads (base+offset via a scalar temp — RVV loads have no
    immediate offset), two-step widen u8→e16→e32, `vmacc.vx` with the 7 scalar
    taps (fh[3] pre-incremented by 128), one `vadd` of 16388
    (= (1<<14) + rounding_off_h), `vsra.vi 3`, clamp [0,8191], narrow to e16,
    `vse16`. Block VL = VLEN/8 (16 @ VLEN 128 … 128 @ 1024); tail purely via
    `vsetvli` on the remaining count, no scalar epilogue.
  - `dav1d_wiener_filter_v_8bpc_rvv(pixel *dst, const uint16_t *const rows[7],
    const int16_t fv[8], int w)`: loads the 7 u16 rows as e16 m1 (block VL =
    VLEN/16), widens, `vmacc` with fv[0..6] (row 6 loaded explicitly, so the
    pure-v duplication and the hv fresh-row case share the kernel), one `vadd`
    of −261120 (= −round_offset + rounding_off_v), `vsra.vi 11`, clamp
    [0,255], double narrow e32→e16→e8 (`vncvt` into an `mf2` destination),
    `vse8`. Register map: rows in t0–t6, fv[0..6] in a4–a7+s0–s2 (s-regs saved),
    a1/a2 scratch, loop state a0 (dst), a3 (remaining), a2 (block vl).
- `src/riscv/looprestoration.h`: decls + the block driver `wiener_filter_rvv`
  (a verbatim transcription of `wiener_c`'s control flow: hor[6·390] rows,
  ptrs rotation, startup shapes, bottom padding) calling the two kernels;
  small static helpers `wiener_filter_hv_rvv` (h into tmp, v over rows[6] with
  `rows[6] = tmp`, 390-entry memcpy, rotate) and `wiener_filter_v_rvv`
  (`rows[6] = ptrs[5]`); `loop_restoration_dsp_init_riscv(c)` assigns
  `c->wiener[0] = c->wiener[1] = wiener_filter_rvv` under
  `DAV1D_RISCV_CPU_FLAG_V`, 8bpc only (16bpc build gets a no-op).

## Bugs found and fixed (all real, all in the new code)

1. **RVV loads have no immediate offset** — `vle8.v v9, 1(a1)` is illegal
   operands; address arithmetic goes through a scalar temp.
2. **`function` macro prefixing** — names in `.S` files must omit `dav1d_`
   (the macro's `EXTERN` adds `PRIVATE_PREFIX`); my first build exported
   `dav1d_dav1d_wiener_*` and the link failed.
3. **Byte-interleaved v-store** (found by the `vprobe.c` micro-test): after
   `vncvt` the vector is e16; storing it with `vse8` under an `e16,m1` vtype
   writes the *low byte of each 16-bit element*, producing value,0,value,0.
   The e32→e8 pack needs a second `vncvt` into an `e8,mf2` destination, then
   `vse8` under that vtype.
4. **Tap-register clobber in the v loop** (found by the 3-level differential,
   which localized the failure to block 2+): `li a4, -261120` destroyed
   `a4 = fv[0]`; every block after the first accumulated with a garbage tap.
   The scratch moved to a1 (dead after the row-pointer loads). The h loop was
   audited for the same class (it only writes a6/a7 scratch) and passes.
5. **Missing `goto v1;` in the bottom-label chain — my transcription bug,
   present identically in the header driver AND the differential's reference
   copy**, which is why levels 1–4 (self-consistent) were green while checkasm
   stayed red. Discovered by linking the library's own
   `dav1d_loop_restoration_dsp_init_8bpc`: under QEMU user-mode the V bit is
   not in AT_HWCAP, so the init resolved to pure `wiener_c`, and level 5
   (real C vs my driver) failed on exactly the last output row of shapes with
   h ≥ 4 — the real C's `v3 → v2 → goto v1` chain issues **three** standalone
   v calls; my two omitted the third, leaving the final row unfiltered.
   Fixed in both copies; checkasm went green immediately after.

## Gate receipts (run order, seeds)

Gate 1 — checkasm (bit-exact vs C):
- RED (bug 1–2 era): `qemu:512` seed 2384887318 — 2/2 failed (81x58, 225x36,
  edges 0000, `looprestoration.c:112`).
- RED (bug 3–4 era): `qemu:128` seeds 2678839097, 1427837379 — 2/2 failed
  (41x26; 249x43 + 3x7).
- GREEN `wiener_*`: `qemu:128` seed 383332238, `qemu:256` seed 939386273,
  `qemu:512` seed 1476593007, `qemu:1024` seed 2002715360; X280 seeds
  1413659980 and 2538311048 — all pass (`results/c2-wiener/wiener-*.log`).
- FULL suite: `qemu:512` **918/918**, X280 **918/918**
  (`fullsuite-*.log`). `--list-functions`:
  `wiener_7tap_8bpc (c, rvv)`, `wiener_5tap_8bpc (c, rvv)`; 10/12bpc stay C.
- Note: the board full-suite initially bus-errored in the **pre-existing**
  upstream `mct_8tap_smooth_sharp_w128_*` misaligned-load bug (the mct-hardware
  campaign's known issue, absent from its worktree only because of its local
  `mc.S-fix.diff`). That documented fix is applied in this worktree solely to
  make the gate runnable; it is not part of the LR deliverable.

Gate 2 — targeted differential (`lr_diff.c`, 5 levels, all bit-exact):
1. h kernel alone vs transcribed reference: 7 tap sets × {5-tap,7-tap} × 4
   edge combos × left/NULL × 27 widths (1..384 incl. all alignment edges) — ok
2. v kernel alone: intermediates forced to the clip boundaries
   {0, 8191, 2047, 2048} ∪ random — ok
3. whole block: widths × heights (1..64 incl. the h=1..4 startup shapes) ×
   all 16 edge combos × tap extremes × content modes {random, all-0, all-255,
   checker, clip-heavy} — ok
4. exact checkasm mirror (stride 448, 64-byte-aligned pointers, cleared
   padding, `init_tmp` content model, 3000 configs × 16 edges) — ok
5. **library-built reference** (`dav1d_loop_restoration_dsp_init_8bpc`
   resolving to pure `wiener_c` under QEMU) vs the driver: 3000 configs ×
   16 edges — ok (this is the level that caught bug 5)
All levels green at VLEN 128/256/512/1024 (`lr_diff-all-vlens.log`).

Gate 3 — bench, X280, timing = linux-perf, seed 125395212 + final rerun
(`bench-x280.log`), checkasm bench geometry 256×64, edges 0xf:

| function            | C (ticks/call) | RVV (ticks/call) | speedup |
|---------------------|---------------:|-----------------:|--------:|
| wiener_7tap_8bpc_c  |      1,093,507 | 217,798 (5.02x)  |  5.02×  |
| wiener_5tap_8bpc_c  |      1,099,647 | 220,003 (5.00x)  |  5.00×  |
| rerun: 7tap         |      1,084,483 | 209,972          |  5.16×  |
| rerun: 5tap         |      1,088,263 | 210,332          |  5.17×  |

## Working-set note (two-pass structure)

The intermediate buffer is stream-sized: `hor` is 6 × 390 × 2 B = 4.68 KB
(plus a 780 B hv temp), i.e. fully L1-resident on the X280. The cost is
bandwidth, not capacity: per output pixel the H pass writes 2 B and the V pass
re-reads up to 7 × 2 B ≈ 14 B of intermediates — ~16 B of L1 traffic per
pixel, against 1 B in / 1 B out of DRAM. The H pass also re-reads its
padded row 7× (sliding-window loads) from a ≤390 B stack buffer. The kernel
is therefore L1-bandwidth-bound, which is the price of the C-exact
u16-intermediate two-pass contract; a fused single-pass (recomputing h per
output row into registers, 7-row sliding window of raw pixels) would remove
most of it but changes rounding order and would no longer be a drop-in
`wiener_c` replacement.

## Upstream-MR shape (the artifact)

1. **new** `src/riscv/64/looprestoration.S` — 2 exported functions
   (`function ..., export=1, ext=v`).
2. **new** `src/riscv/looprestoration.h` — per-family init header modeled on
   `src/riscv/cdef.h` (decls + static driver + inline init).
3. **library diff** `src/looprestoration_tmpl.c` (+3 lines ×2): the
   `#elif ARCH_RISCV` include of the header and the
   `loop_restoration_dsp_init_riscv(c)` call — the hook genuinely did not
   exist (no riscv LR ever existed upstream).
4. **meson diff** `src/meson.build` (+1): `'riscv/64/looprestoration.S'` in
   the 8bpc asm list (alphabetical, between itx and mc).
Full diff: `wiring.diff` (excludes the unrelated `mc.S` board fix above).

## Honest negatives

- SGR (`sgr_5x5/3x3/mix`): **not started** — stretch goal, wiener consumed
  the budget (2 asm bugs + 1 transcription bug + the checkasm-vs-reference
  tooling detour).
- No QEMU timing claims: all speed numbers are X280 linux-perf.
- The differential's reference is a transcription of `wiener_c`; it was
  validated *against* the library's own `wiener_c` (level 5) only after the
  level-5 mechanism existed — earlier green levels were self-consistent, not
  absolute. Any future LR campaign should build level 5 first.
- 16bpc Wiener: out of scope (BITDEPTH==8 guard); the 16bpc build compiles
  clean with a no-op init.

## Files

- `looprestoration.S` — the asm (`src/riscv/64/looprestoration.S`)
- `looprestoration-init.h` — the init header (`src/riscv/looprestoration.h`)
- `wiring.diff` — template hook + meson registration
- `lr_diff.c`, `vprobe.c`, `whichfn.c` — differential driver + debug probes
- `receipts/` — checkasm/bench/diff logs from the box
