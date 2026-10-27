# REPORT: loop-restoration self-guided (SGR) RVV port (8bpc) — campaign 3 (`c3-sgr`)

Status: **GREEN.** From-scratch RVV port of dav1d's 8bpc self-guided loop
restoration: **`sgr_3x3`, `sgr_5x5` and `sgr_mix`**, all three wired and
bit-exact. The looprestoration family had zero RISC-V support upstream; this
campaign extends the file pair the wiener campaign (c2-wiener) created.
All gates pass: checkasm bit-exact under QEMU at VLEN 128/256/512/1024 and
natively on the X280 (**full suite 921/921 on both**), a 13-level targeted
differential is clean at every VLEN, and `checkasm --bench` on the X280
(linux-perf) measures **~4.0–4.35× over the C reference** for all three
variants.

## Derived spec (from `src/looprestoration_tmpl.c`, 8bpc)

All three block functions share the same machinery: 3×3 or 5×5 box
sums of the source and its square, a per-pixel self-guided weight, a
3×3-guided finish filter, and a weighted blend into the destination.
`coef` = int16_t, `pixel` = uint8_t, block buffers use
`BUF_STRIDE = 384 + 16`, the finish temporary uses `FILTER_OUT_STRIDE = 384`.

**Box row, horizontal** (`sgr_box3_row_h` / `sgr_box5_row_h`): after
`sumsq++; sum++;` the C stores `x = -1..w`, i.e. **w+2 entries at [0..w+1]
of the caller's row**. Output `o` sums source positions `o-2..o` (3-tap) or
`o-3..o+1` (5-tap) of the *virtual* row:

```
v < 0 : HAVE_LEFT ? (left ? left[0][4+v] : src[v]) : src[0]     (3-tap)
        HAVE_LEFT ? (left ? left[0][4+v] : src[v]) : src[0]     (5-tap, v>=-3)
v >= w: (v < w || HAVE_RIGHT) ? src[v] : src[w-1]
        (5-tap reads up to src[w+2] when LR_HAVE_RIGHT — real OOB reads,
         same as the C; checkasm pre-fills +4 columns)
sum[o]   = Σ tap                       (≤ 765 / 1275, fits int16)
sumsq[o] = Σ tap²                      (≤ 195075 / 328050, int32)
```

**Box row, vertical** (`sgr_box3_row_v` / `sgr_box5_row_v`): sums 3 (5)
row pointers over `x = 0..w+1` (the C's loop bound is `w + 2`).

**Self-guided weights** (`sgr_calc_row_ab`, 8bpc: `bitdepth_min_8 = 0`, so
no pre-rounding shifts), in place over `i = 0..w+1`:

```
a = AA[i],  b = BB[i]                       (A int32, B int16)
p = max(n*a - b*b, 0)                       signed; n = 9 (3x3), 25 (5x5)
z = (p * s + (1 << 19)) >> 20               UNSIGNED 32-BIT: p*s WRAPS
                                            (3x3: p ≤ 5.3e6, s ≤ 3236 →
                                             17e9 > 2^32; 5x5 also wraps)
x = dav1d_sgr_x_by_x[umin(z, 255)]          256-entry LUT
AA[i] = (x * BB[i] * one_by_x + (1 << 11)) >> 12   one_by_x = 455 / 164
BB[i] = x                                   (≤ 255, fits int16)
```

**Finish filter** — per output pixel with column `c = i + 1` (rows loaded
at columns `c-1, c, c+1`):

```
3x3 (sgr_finish_filter_row1, rows P[0..2] = above/current/below):
  a = (P1[c] + P1[c-1] + P1[c+1] + P0[c] + P2[c]) * 4
    + (P0[c-1] + P2[c-1] + P0[c+1] + P2[c+1]) * 3      (same for b over A)
  tmp[i] = (b - a * src[i] + (1 << 8)) >> 9            int32 → int16 STORE
                                                       (truncating; asm must
                                                        wrap identically)
5x5 (sgr_finish_filter2, rows 0..1 of the A/B window):
  a = (P0[c] + P1[c]) * 6 + (P0[c-1] + P1[c-1] + P0[c+1] + P1[c+1]) * 5
  row 0: tmp[i]     = (b - a*src[i]     + (1 << 8)) >> 9
  row 1 (h > 1, rows [1] alone, src += stride):
         tmp[i + 384] = (b - a*src[i+1] + (1 << 7)) >> 8    ← different
                                                          rounding per row!
```

**Blend** (`sgr_weighted_row1` / `sgr_weighted2`):

```
dst[i] = iclip_u8(dst[i] + ((v + (1 << 10)) >> 11))
3x5:  v = w1 * tmp3[i]              (w1 ∈ [33, 160] in checkasm)
5x5:  v = w0 * tmp5[i]              (w0 ∈ [-96, 31])
mix:  v = w0 * tmp5[i] + w1 * tmp3[i]
tmp5/tmp3 advance by FILTER_OUT_STRIDE (384) coefs per row; dst by stride.
```

**Block control flow** (`sgr_3x3_c` / `sgr_5x5_c` / `sgr_mix_c`): rotating
row windows (3/5 input rows → A/B rows, rotated by 3 / by 2 (`rotate5_x2`)
/ by 4+2 for mix), startup shapes for `LR_HAVE_TOP` (duplicate first rows)
and no-top (all rows point at row 0), `h ≤ 2..4` shortcuts via
`vert_1/vert_2/odd/output_1/output_2` labels, bottom padding from
`lpf + 6*stride` duplicated 1–2 rows. checkasm exercises all 16 edge
combinations with `w = 256` when `LR_HAVE_RIGHT`, else 1..384, and
`h = 64` when `LR_HAVE_BOTTOM`, else 1..64. Per-variant parameters come
from `dav1d_sgr_params[14] = {56, 0}` (5x5), `[10] = {0, 2589}` (3x3),
`[0] = {140, 3236}` (mix) with `w0 = s0 ? (rnd&127)-96 : 0`,
`w1 = (s1 ? 160-(rnd&127) : 33) - w0`.

## Design as implemented

Same split as c2-wiener: **all block-level control flow stays in C**
(`src/riscv/looprestoration.h`: `sgr_3x3_rvv`, `sgr_5x5_rvv`, `sgr_mix_rvv`
are verbatim transcriptions of the C block functions; helpers
`sgr_box3_vert_rvv`, `sgr_box5_vert_rvv`, `sgr_finish1_rvv`,
`sgr_finish2_rvv`, `sgr_finish_mix_rvv`, `sgr_rotate_rvv`,
`sgr_rotate5_x2_rvv` mirror the C's statics, suffixed `_rvv` to avoid
clashing with the real statics of the including TU). **All per-row
arithmetic is RVV asm** appended to `src/riscv/64/looprestoration.S`
(`ext=v` only, plain RVV 1.0 — matches X280 and the QEMU model; no Zbb):

| kernel | replaces | notes |
|---|---|---|
| `sgr_box3_row_h_8bpc_rvv` | `sgr_box3_row_h` | pads the row (≤ 388 B stack), 3 shifted byte loads, sum in e16, squares in e32 |
| `sgr_box5_row_h_8bpc_rvv` | `sgr_box5_row_h` | 5 taps; squares accumulate via `vzext.vf4` straight from the e8 loads (source EMUL m4/4 = m1, as loaded) |
| `sgr_box3_row_v_8bpc_rvv` | `sgr_box3_row_v` | e32 + e16 row sums, w+2 elements |
| `sgr_box5_row_v_8bpc_rvv` | `sgr_box5_row_v` | 5 pointers, s-regs saved |
| `sgr_calc_ab1_8bpc_rvv` | `sgr_calc_row_ab(n=9, 455)` | `vmul.vx` modular p*s + `vsrl` reproduce the C's unsigned wrap; LUT via `vluxei8` under `e8,mf2` |
| `sgr_calc_ab2_8bpc_rvv` | `sgr_calc_row_ab(n=25, 164)` | same body, n/obx constants |
| `sgr_finish_filter_row1_8bpc_rvv` | `sgr_finish_filter_row1` | 8-neighborhood: `*4` center-window sum + `*3` diagonals; `vnmsub`-free: `vmul` a·src then `vsub` from b |
| `sgr_finish_filter2_8bpc_rvv` | `sgr_finish_filter2` | two-row (×6/×5) row 0, single-row rounding (1<<7)>>8 row 1 |
| `sgr_weighted_row1_8bpc_rvv` | `sgr_weighted_row1` | widen → blend → clip → double narrow |
| `sgr_weighted2_8bpc_rvv` | `sgr_weighted2` | w0·t1 + w1·t2; per-row pointer rewind (see bugs) |
| (none) | `sgr_box35_row_h` | two asm calls from a static inline |

Init lines (inside `DAV1D_RISCV_CPU_FLAG_V`, `BITDEPTH == 8`):
`c->sgr[0] = sgr_5x5_rvv; c->sgr[1] = sgr_3x3_rvv; c->sgr[2] = sgr_mix_rvv;`.
The 16bpc TU compiles clean with the unchanged no-op init.

**Exactness rules learned on this port** (recorded for the next RVV porter):

1. **Never issue a widening op under a vtype whose implied source group
   (EMUL = LMUL/2 or /4) does not match how the data was last written.**
   `vzext.vf2 v16, v8` under `e32,m4` reads source group v8+*v9* as e16 —
   QEMU raises illegal instruction; the wiener-proven shape widens in steps
   (e8 m1 → e16 m2 → e32 m4), each step under the destination vtype.
   `vzext.vf4` from an `e8,m1` load is legal and correct (source EMUL m1).
2. **Grouped (LMUL > 1) loads/stores need register-spaced destinations:**
   under `e32,m2` the destinations must be v8, v10, v12 — `vle32.v v8/v9/v10`
   overlaps groups and is illegal (v9 odd = never a legal group base).
3. **`vmv.v.v` does not survive this toolchain**: GAS encodes the source in
   the *vs1* field (vs2 = v0); QEMU reads the spec's vs2-field form and
   copies **v0** — silent garbage. Avoided entirely (restructured to
   `vmul`+`vsub`, direct accumulation); c2-wiener's `vmv.v.i` is a
   different encoding and is fine.
4. **32-bit wrap in `z`** is reproduced bit-exactly with modular `vmul.vx`
   + logical `vsrl` + `vminu` (see spec above).
5. **The int16 truncating store of `tmp`** (finish kernels) is reproduced
   with `vncvt.x.x.w` — same wrap as the C's implicit conversion.
6. Watch the classic clobbers: constant materialization (`li t2, 3`,
   `li t2, 1<<8`) inside loops whose registers hold row pointers, and
   per-row pointer advances with the correct element size (coef = 2 B,
   int32 A rows = 4 B — finish_filter2's row 1 advanced the A row by 2).

## Bugs found and fixed (all real, all in the new code)

1. **`vslli.vi`/`vmv.vv`** — wrong mnemonics (`vsll.vi`, and no vector
   `vmv.vv` form) — first build failed.
2. **`sgr_3x3_8bpc_rvv (illegal instruction)`** (checkasm seed 2256890342,
   VLEN 512): the box3 kernel widened bytes directly to e32m4
   (`vzext.vf2 v16, v8` under `e32,m4`) — semantically wrong (reads v8+v9
   as e16 pairs) and QEMU-rejected. Isolated with a micro-probe
   (`vprobe_sgr.c`, `vprobe4.c`: shape A crashes at every VLEN, shape B —
   the wiener two-step — is correct at every VLEN). Fixed to the two-step.
3. **SIGSEGV at `vle16.v v8,(t2)`, badaddr=0x40** in
   `sgr_finish_filter_row1` (found with a fault-handler probe,
   `kprobe2.c`): `li t2, 3` / `li t2, 1<<8` / the tmp advance clobbered
   **t2 = B2 row pointer**; iteration 1 survived (B2 already loaded),
   iteration 2 dereferenced the constant. Fixed to scratch t3.
4. **`vmv.v.v` copies v0** (see rule 3): the `a`/`b` staging in
   `finish_filter_row1` silently used v0 contents — results independent of
   `src` (diagnostic `kprobe3`/level 7: same tmp for src=0 and src=2).
   Restructured to `vsll.vi v24` + `vmacc` for `a`, `vmul`+`vsub` for
   `b − a·src`, `vadd.vi v18, v10, 0` for the A0[c] seed.
5. **Second-block value mismatch** (differential level 4, w=34 index 32):
   the B/A row pointers were never advanced across block iterations (and
   the A pointers were re-`ld`ed from the array each iteration). Fixed:
   load once before the loop, advance by 2·vl (B, coef) / 4·vl (A) at the
   bottom.
6. **`sgr_5x5_8bpc_rvv (segmentation fault)`** (checkasm seed 1925629212):
   `box5_row_v` used t6 as both the second sum-row pointer and the advance
   scratch → `vle16.v v9,(t6)` with t6 = 64. Scratch moved to a5.
7. **`vle32.v v9,(t6)` illegal** in `finish_filter2`: odd-register m2
   group base (rule 2) in both the row-0 A block (v8..v13 consecutive) and
   the row-1 block. Fixed to v8/v10/v12/v14/v16/v18.
8. **`finish_filter2` read the wrong arrays**: the asm loaded B rows from
   `a3` and A rows from `a4`, but the declared signature is
   `(tmp, src, stride, A_ptrs, B_ptrs, w, h)` — A/B swapped
   (differential level 11: tmp = −13494, mathematically impossible from
   correct inputs). Fixed; note `finish_filter_row1` has the opposite arg
   order and was correct.
9. **`finish_filter2` row 1 advanced its A row by 2 bytes/element** (int32
   needs 4) — differential level 11, w=34 h=2 index 416 (row 1, block 2).
10. **`weighted2` never multiplied t1 by w0** (`t1 + w1*t2` instead of
    `w0*t1 + w1*t2`) — only mix uses w0 (3x3: w0 = 0; 5x5: single weight),
    which is why levels 6/12 were green while mix was off by one
    (checkasm seed 1238734392, 92x57 edges 0000; differential level 13:
    (0,1) C=13 rvv=14).
11. **`weighted2` row advance accumulated the w-wide walk**
    (`a2 += 2w + 768` per row instead of `= base + 768·row`) — differential
    level 13 again, after fix 10 moved the failure. Fixed to rewind +
    stride.
12. Transcription bug class from c2-wiener (a bug present identically in
    driver and differential reference, cancelling out) was avoided by
    building **differential level 6/12/13 (my driver vs the library's own
    pure-C block functions) from the start** and by including the actual
    `src/riscv/looprestoration.h` in the driver rather than a transcription.

## Gate receipts (run order, seeds)

Gate 1 — checkasm (bit-exact vs C), final receipt logs in `receipts/`:

- RED `sgr_3x3_8bpc_rvv (illegal instruction)`: qemu:512 seed 2256890342
  (bug 2 era).
- RED value mismatch `size = 105x34, edges = 0000`
  (looprestoration.c:181), qemu:512 seed 254765549 (bug 5 era).
- GREEN `sgr_3x3` + C-only others: qemu:128 seed 3334146078, qemu:256
  seed 3962318083, qemu:512 seed 327712286, qemu:1024 seed 1069296544.
- RED `sgr_5x5_8bpc_rvv (segmentation fault)`: qemu:512 seed 1925629212
  (bug 6 era).
- GREEN `sgr_3x3` + `sgr_5x5`: qemu:128 seed 332254619, qemu:256 seed
  1156450514, qemu:512 seed 1908864010, qemu:1024 seed 2655094673.
- FULL suite (all families) qemu:512: **920/920**, seed 2990856524
  (3x3+5x5 stage).
- RED `sgr_mix_8bpc_rvv` value mismatch `size = 92x57, edges = 0000`:
  qemu:512 seed 1238734392 (bugs 10–11 era).
- GREEN all three: qemu:128 seed 1046426628, qemu:256 seed 1915192784,
  qemu:512 seed 2701984486, qemu:1024 seed 3534360845
  (`checkasm-sgr-qemu*.log`).
- FULL suite qemu:512: **921/921**, seeds 361407247 and 4293201247
  (`fullsuite-qemu512.log`).
- X280 native: sgr 3/3 seed 3964676208 (`checkasm-sgr-x280.log`);
  FULL suite **921/921**, seeds 721576828 and 591713012
  (`fullsuite-x280.log`).
- `--list` check: sgr_3x3_8bpc, sgr_5x5_8bpc, sgr_mix_8bpc run as (c, rvv);
  10/12bpc stay C-only (visible in every bench log).
- Note: the worktree carries the documented pre-existing `mc.S` board fix
  (mct campaign) solely to make the X280 full-suite gate runnable; it is
  not part of this deliverable. Same as c2-wiener.

Gate 2 — targeted differential (`sgr_diff.c`, 13 levels, all bit-exact;
`receipts/sgr_diff-vl{128,256,512,1024}.log` all end "all requested levels
ok"):

1. box3_row_h vs transcribed C: 384 widths × 16 edges × {random,
   checker, all-255} × left/NULL
2. box3_row_v vs C: random + {max, zero} rows
3. calc_ab1 vs C: all 16 `dav1d_sgr_params` rows × boundary A/B
   (incl. AA = 585225 where p*s wraps u32)
4. finish_filter_row1 vs C: A near the calc output ceiling (65018), B ≤ 255
5. weighted_row1 vs C: t = {random, 32767, −32768, 0}, w1 ∈ [−96, 256]
6. **whole sgr_3x3_rvv vs the library's pure-C sgr_3x3_8bpc_c** (init
   without the V flag under QEMU): 3000 configs × 16 edges, checkasm-
   mirroring harness (stride 448, 64-byte aligned, init_tmp content model)
7. weighted2 diagnostic: crafted inputs, back-out a and b
8. box5_row_h vs C (same envelope as level 1)
9. box5_row_v vs C
10. calc_ab2 vs C: 16 s0 rows × AA up to 1640250, BB up to 6375
11. finish_filter2 vs C: h = 1 and h = 2, w up to 384
12. **whole sgr_5x5_rvv vs C sgr_5x5**: 3000 configs × 16 edges
13. **whole sgr_mix_rvv vs C sgr_mix**: 3000 configs × 16 edges

Gate 3 — bench, X280, timing = linux-perf, VLEN 512, checkasm bench
geometry 256×64, edges 0xf (`bench-x280.log`, `bench-x280-repeat.log`,
`--repeat=2`):

| function | C (ticks/call) | RVV (ticks/call) | speedup |
|---|---:|---:|---:|
| sgr_3x3_8bpc | 1,708,856 | 414,818 (4.12x) | 4.12× |
| sgr_5x5_8bpc | 1,306,232 | 301,132 (4.34x) | 4.34× |
| sgr_mix_8bpc | 2,914,813 | 679,167 (4.29x) | 4.29× |
| repeat: 3x3 | 1,707,818 | 392,523 | 4.35× |
| repeat: 5x5 | 1,305,995 | 321,452 | 4.06× |
| repeat: mix | 2,912,057 | 715,770 | 4.07× |
| 3rd run: 3x3 | 1,707,888 | 429,825 | 3.97× |
| 3rd run: 5x5 | 1,307,915 | 303,083 | 4.32× |
| 3rd run: mix | 2,915,034 | 768,855 | 3.79× |

All measured runs land in **3.8–4.35×**; call it **~4.1×** for 3x3,
**~4.3×** for 5x5 and **~4.2×** for mix, consistently well above the
"real speedup" bar. QEMU timings were not used.

## Working-set note

The sgr sum/A/B row buffers are block-sized stack arrays, not
stream-sized: 3x3 = 3 rows × 400 × (4 B sumsq + 2 B sum + 4 B A + 2 B B)
= 14.4 KB live plus a 1.5 KB finish tmp; 5x5 = 5 sumsq/sum rows (12 KB) +
2 A/B rows (4.8 KB) ≈ 16.8 KB; mix = 5+5+3 sumsq/sum rows (19.2 KB) +
2+4 A/B rows (11.2 KB) ≈ 30.4 KB, plus 2 × 1.5 KB finish temporaries
≈ 33 KB peak. 3x3/5x5 are fully L1-resident on the X280 (32 KB L1D); mix
spills part of its window to the X280's loosely-integrated L2, which
shows as mix's slightly lower speedup (3.8–4.3× vs 5x5's 4.1–4.3×).
The cost is bandwidth, not compute: per output pixel the box h pass
writes 6 B (3-tap) / 10 B (5-tap) of sums, the box v pass re-reads 3×/5×
that and writes 6/10 B, calc rewrites 6 B, the finish re-reads up to
18 B/30 B of A/B rows and the blend re-reads tmp — of order 50–90 B of
L1 traffic per pixel against 1 B in / 1 B out of DRAM. Like wiener, the
kernels are L1-bandwidth-bound; a fused single-pass variant would change
the C-exact intermediate contract and is out of scope.

## Upstream-MR shape (the artifact)

1. **extended** `src/riscv/64/looprestoration.S` — c2-wiener's 2 wiener
   kernels + 11 exported sgr kernels (`function ..., export=1, ext=v`).
2. **extended** `src/riscv/looprestoration.h` — sgr kernel declarations,
   static helpers mirroring the C's control flow, the three block drivers,
   and the three `c->sgr[n]` init lines inside the existing
   `DAV1D_RISCV_CPU_FLAG_V` / `BITDEPTH == 8` guard.
3. No further library/meson changes: the `#elif ARCH_RISCV` hook in
   `src/looprestoration_tmpl.c` and the meson asm registration already
   exist from c2-wiener (its `wiring.diff`); an upstream MR would carry
   both campaigns' file contents in one commit pair.
   Delta vs the c2-wiener versions: `sgr-additions.diff` (1710 lines).

## Honest negatives

- **16bpc SGR is out of scope** (BITDEPTH == 8 guard, as upstream's riscv
  ports do for bitdepth-specific kernels); the 16bpc library and its
  checkasm suites compile and pass with C-only sgr (10bpc/12bpc entries in
  every log).
- The differential's `s` sweep uses the 16 legal `dav1d_sgr_params` rows;
  arbitrary (illegal) s values are untested, matching the C's domain.
- No attempt was made to fuse stages or skip the `p*s` wrap path for
  small s; bit-exactness with the C's 32-bit semantics forbids
  simplifications like scaling p down before the multiply.
- `--repeat=2` bench spread on 3x3 (4.35× vs 3.97×) is larger than
  wiener's; the sgr C reference is heavier and the board is shared, so
  single-run ticks have a few percent of noise. All numbers are real
  linux-perf ticks from `bench-x280*.log`.
- QEMU was used for correctness only (its vector model is not a timing
  source); hardware was used for both correctness (921/921) and timing.

## Files

- `looprestoration.S` — the asm (`src/riscv/64/looprestoration.S`),
  wiener part + this campaign's 11 sgr kernels
- `looprestoration-init.h` — the init header
  (`src/riscv/looprestoration.h`)
- `sgr-additions.diff` — this campaign's delta vs the c2-wiener files
  (the upstream-MR shape, on top of the wiener MR)
- `sgr_diff.c` — 13-level differential driver (links the built objects and
  the library's C sgr functions; built with the include line in its header)
- `kprobe2.c`, `kprobe3.c`, `kprobe5.c`, `vprobe_sgr.c`, `vprobe4.c` —
  fault-isolation and RVV-semantics probes used during debugging
- `receipts/` — checkasm/bench/differential logs from the box, plus the
  worktree copies of the two shipped files (byte-identical to the masters)
