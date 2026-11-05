/* dav1d chroma loop filter (sb128 uv, wd 4 and 6), 8bpc: lifted RVV vs
 * src/loopfilter_tmpl.c's loop_filter_h_sb128uv_c / loop_filter_v_sb128uv_c.
 *
 * Domain (decoder and checkasm): vmask[0] and vmask[1] bits 0..NBLKS-1 (a bit
 * in vmask[1] selects wd 6 for that block, otherwise wd 4; a bit set only in
 * vmask[1] also filters); vmask[2..3] are never read by the uv wrappers. The
 * per-block level L is in [0, 63]; L == 0 skips the block. The filter limits
 * come from the Av1FilterLUT, filled from sharpness in [0, 7] by checkasm's
 * formula (mirroring dav1d_calc_eih). Pixels 0..255.
 *
 * The vmask words are CONCRETE per configuration (-DVM0, -DVM1): a symbolic
 * vmask makes the reference wrapper's bit loop (guard vm & ~(y-1), y <<= 1)
 * cost CBMC minutes per unwind check, while the bits' data effects (which
 * blocks filter, at which wd) are covered by enumerating patterns across
 * configurations. Levels, the lut's sharpness and all pixel data stay
 * symbolic, so the fm/flat/hev decisions inside the filter core are all
 * covered data-dependently.
 *
 * Memory contract (checkasm geometry): for the h wrapper (vertical edges)
 * stride is 16, dst points 8 pixels into each row, blocks advance 4 rows
 * each; for the v wrapper (horizontal edges) stride is 64, dst points 8 rows
 * into the buffer, blocks advance 4 columns each. With wd <= 6 the filter
 * reads up to 3 pixels and writes up to 2 on each side of an edge; every
 * other byte must be untouched (checkasm compares the destination exactly,
 * no padding allowance). Reads are allowed anywhere in dst, mask, lvl and
 * lut.
 *
 * Bounded configuration: -DDIR (0 = h, 1 = v), -DNBLKS (blocks exercised),
 * -DVM0/-DVM1 (the two vmask words). */
#include <stdlib.h>
#define BITDEPTH 8
#include "fv_runtime.h"
#include "lifted.c"
#include "src/loopfilter_tmpl.c"

#ifndef DIR
#define DIR 0
#endif
#ifndef NBLKS
#define NBLKS 4
#endif
#ifndef VM0
#define VM0 0xf
#endif
#ifndef VM1
#define VM1 0
#endif
#ifndef ENTRY
#define ENTRY fvf_dav1d_lpf_h_sb_uv_8bpc_rvv
#endif

#define STRIDE (DIR ? 64 : 16)
#define B4_STRIDE (DIR ? 32 : 2)
/* only what the geometry needs: h touches rows 0..4*NBLKS-1 of a 16-pixel
 * buffer starting at column 8; v touches rows 5..10 (edge at row 8) of a
 * stride-64 buffer. Keeps the symbolic fill and the compare loop small. */
#define BUF_N (DIR ? 12 * 64 : 16 + 4 * NBLKS * 16)
#define DST_OFF (DIR ? 8 * 64 : 8)
#define LVL_N (64 * 4)
#define LVL_OFF (DIR ? 32 * 4 + 2 : 4 + 2)
#define DST_ADDR 0x1000000ULL
#define MASK_ADDR 0x2000000ULL
#define LVL_ADDR 0x3000000ULL
#define LUT_ADDR 0x3100000ULL

static uint8_t dst_r[BUF_N];
static uint8_t *dst_c, *mask_c, *lvl_c, *lut_c;

int main(int argc, char **argv)
{
#ifndef FV_CBMC
    if (argc > 1) fv_rng = strtoull(argv[1], 0, 10) * 0x9E3779B97F4A7C15ULL + 1;
    if (argc > 2) fv_agnostic_ones = atoi(argv[2]);
#else
    (void)argc; (void)argv;
#endif
    dst_c = fv_alloc(BUF_N); mask_c = fv_alloc(16); lvl_c = fv_alloc(LVL_N);
    lut_c = fv_alloc(sizeof(Av1FilterLUT));
    fv_fill(dst_c, BUF_N);
    fv_fill(mask_c, 16);
    fv_fill(lvl_c, LVL_N);
    fv_fill(lut_c, sizeof(Av1FilterLUT));

    /* Domain: vmask[0..1] are the configuration's concrete words (see header),
     * vmask[2..3] = 0 (unread); level bytes (column 2 of the lvl array) in
     * [0, 63]. */
    const uint32_t m0 = VM0, m1 = VM1;
    memcpy(mask_c + 0, &m0, 4);
    memcpy(mask_c + 4, &m1, 4);
    memset(mask_c + 8, 0, 8);
    for (int j = 0; j < 64; j++)
        lvl_c[4 * j + 2] &= 63;

    /* The lut, from a symbolic sharpness in [0, 7], exactly as checkasm's
     * loopfilter.c computes it. */
    Av1FilterLUT lut;
    const int sharp = lut_c[144 - 1] & 7;
    for (int level = 0; level < 64; level++) {
        int limit = level;
        if (sharp > 0) {
            limit >>= (sharp + 3) >> 2;
            limit = imin(limit, 9 - sharp);
        }
        limit = imax(limit, 1);
        lut.i[level] = limit;
        lut.e[level] = 2 * (level + 2) + limit;
    }
    lut.sharp[0] = (sharp + 3) >> 2;
    lut.sharp[1] = sharp ? 9 - sharp : 0xff;
    memcpy(lut_c, &lut, sizeof lut);

    memcpy(dst_r, dst_c, BUF_N);
#ifndef FV_CBMC
    static uint8_t chk[BUF_N];   /* native self-check: pre-filter snapshot */
    memcpy(chk, dst_r, BUF_N);
#endif

    fv_region_add("dst", DST_ADDR, BUF_N, dst_c, 1);
    fv_region_add("mask", MASK_ADDR, 16, mask_c, 0);
    fv_region_add("lvl", LVL_ADDR, LVL_N, lvl_c, 0);
    fv_region_add("lut", LUT_ADDR, sizeof lut, lut_c, 0);

    /* Reference computation. The C wrapper's own driver loop
     * (for (y = 1; vm & ~(y-1); y <<= 1, ...)) cannot be unwound by CBMC
     * when vm comes from memory (the unwind check never closes; verified
     * 2026-11-03: >11000 unwinds of a loop that exits after <= 33). The
     * semantics are replicated here with a concrete-bounded driver over the
     * same vmask words, calling the C's own loop_filter() core from
     * src/loopfilter_tmpl.c. A native self-check (below, cosim only) proves
     * the driver equal to the real wrapper on every cosim input. */
    pixel *dref = dst_r + DST_OFF;
    const uint8_t (*lref)[4] = (const uint8_t (*)[4])(lvl_c + LVL_OFF);
    const uint32_t vm = m0 | m1;
    for (int j = 0; j < NBLKS; j++) {
        if (vm & (1U << j)) {
            const int L = DIR ? (lref[0][0] ? lref[0][0] : lref[-B4_STRIDE][0])
                              : (lref[0][0] ? lref[0][0] : lref[-1][0]);
            if (L) {
                const int H = L >> 4;
                const int E = lut.e[L], I = lut.i[L];
                const int idx = !!(m1 & (1U << j));
                if (DIR)
                    loop_filter(dref, E, I, H, 1, STRIDE, 4 + 2 * idx);
                else
                    loop_filter(dref, E, I, H, STRIDE, 1, 4 + 2 * idx);
            }
        }
        if (DIR) {
            dref += 4;
            lref++;
        } else {
            dref += 4 * STRIDE;
            lref += B4_STRIDE;
        }
    }

#ifndef FV_CBMC
    /* native self-check: the replicated driver must match the C wrapper */
    {
        pixel *dc = chk + DST_OFF;
        if (DIR)
            loop_filter_v_sb128uv_c(dc, STRIDE, (const uint32_t *)mask_c,
                                    (const uint8_t (*)[4])(lvl_c + LVL_OFF), B4_STRIDE, &lut, NBLKS);
        else
            loop_filter_h_sb128uv_c(dc, STRIDE, (const uint32_t *)mask_c,
                                    (const uint8_t (*)[4])(lvl_c + LVL_OFF), B4_STRIDE, &lut, NBLKS);
        for (int i = 0; i < BUF_N; i++)
            if (chk[i] != dst_r[i]) {
                fprintf(stderr, "FV-FAIL: harness driver replicates the C wrapper incorrectly at %d\n", i);
                exit(5);
            }
    }
#endif

    const uint64_t args[8] = {DST_ADDR + DST_OFF, STRIDE, MASK_ADDR,
                              LVL_ADDR + LVL_OFF, B4_STRIDE, LUT_ADDR, NBLKS, 0};
#ifndef FV_NO_CAND
    fv_call(ENTRY, args, 7);
#endif

    for (int i = 0; i < BUF_N; i++) {
#ifdef FV_CBMC
        __CPROVER_assert(dst_c[i] == dst_r[i], "equivalence: output differs from the C reference");
#else
        if (dst_c[i] != dst_r[i]) {
            fprintf(stderr, "FV-FAIL: dst differs at row %d col %d: candidate %u reference %u\n",
                    i / STRIDE, i % STRIDE, dst_c[i], dst_r[i]);
            exit(4);
        }
#endif
    }
    return 0;
}
