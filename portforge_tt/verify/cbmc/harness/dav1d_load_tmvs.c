/* dav1d load_tmvs: lifted RVV vs src/refmvs.c's load_tmvs_c (via reference.c:
 * mv_projection + load_tmvs_c verbatim).
 *
 * Domain (decoder and checkasm): n_tile_threads == 1 (tile_row_idx 0);
 * 0 <= row_start8 < row_end8 <= 16 and row_end8 <= ih8; columns
 * 0 <= col_start8 < col_end8 <= iw8; the C reads rp_ref rows over the widened
 * column range [col_start8 - 8, col_end8 + 8) (clamped) and writes rp_proj
 * only inside [row_start8, row_end8) x [col_start8, col_end8). n_mfmvs in
 * [1, 3]; mfmv_ref[n] in [0, 7); the per-candidate MV and ref (0..7) are
 * arbitrary, including invalid entries (ref == 0, ref2cur ==
 * INVALID_REF2CUR, ref2ref == 0): the C's skips cover them.
 *
 * Memory contract: rp_proj is the only writable buffer; rf, the rp_ref
 * pointer array and all rp_ref blocks are read-only. Every byte of rp_proj
 * is compared (the C writes exactly the window above).
 *
 * Bounded configuration: -DW8 (iw8), -DCS8 -DCE8 (column window),
 * -DRS8 -DRE8 (row window), -DNMV (references in play). */
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "fv_runtime.h"
#include "src/refmvs.h"
#include "lifted.c"
#include "reference.c"

#ifndef W8
#define W8 16
#endif
#ifndef CS8
#define CS8 0
#endif
#ifndef CE8
#define CE8 14
#endif
#ifndef RS8
#define RS8 0
#endif
#ifndef RE8
#define RE8 2
#endif
#ifndef NMV
#define NMV 3
#endif
#ifndef ENTRY
#define ENTRY fvf_dav1d_load_tmvs_rvv
#endif

#define STRIDE (W8 + 16)                    /* rp_stride */
#define IH8 17
#define TB 5                                /* sizeof(refmvs_temporal_block), packed */
#define RPPROJ_N ((16 * STRIDE + W8) * TB)
#define RPREF_N ((17 * STRIDE + W8) * TB)

#define RF_ADDR 0x1000000ULL
#define RPPROJ_ADDR 0x2000000ULL
#define RPPTRS_ADDR 0x3000000ULL
#define RPREF_ADDR(i) (0x4000000ULL + (uint64_t)(i) * 0x100000ULL)

static uint8_t proj_r[RPPROJ_N], refs_r[7][RPREF_N];
static uint8_t *rf_c, *proj_c, *rpptrs_c, *refs_c[7];

int main(int argc, char **argv)
{
#ifndef FV_CBMC
    if (argc > 1) fv_rng = strtoull(argv[1], 0, 10) * 0x9E3779B97F4A7C15ULL + 1;
    if (argc > 2) fv_agnostic_ones = atoi(argv[2]);
#else
    (void)argc; (void)argv;
#endif
    rf_c = fv_alloc(sizeof(refmvs_frame));
    proj_c = fv_alloc(RPPROJ_N);
    rpptrs_c = fv_alloc(7 * 8);
    for (int i = 0; i < 7; i++) refs_c[i] = fv_alloc(RPREF_N);

    fv_fill(proj_c, RPPROJ_N);
    for (int i = 0; i < 7; i++) {
        fv_fill(refs_c[i], RPREF_N);
        for (uint64_t b = 4; b < RPREF_N; b += TB)   /* ref byte of each block: 0..7 */
            refs_c[i][b] &= 7;
    }

    /* the native frame for the C reference, then an arena copy whose pointer
     * fields hold the synthetic addresses the lifted code will read */
    refmvs_frame rf;
    memset(&rf, 0, sizeof rf);
    refmvs_temporal_block *refs_native[7];
    rf.n_tile_threads = 1; rf.n_frame_threads = 1;
    rf.iw8 = W8; rf.ih8 = IH8; rf.iw4 = 2 * W8; rf.ih4 = 2 * IH8;
    rf.rp_stride = STRIDE;
    rf.n_mfmvs = NMV;
    fv_fill(rf_c, sizeof rf);
    for (int n = 0; n < NMV; n++) {
        /* mv_projection asserts den in [1, 31] and num in (-32, 31): legal
         * distances are 5-bit; 0 (ref2ref) and -32 (ref2cur) are the C's
         * skip values. */
        rf.mfmv_ref[n] = rf_c[offsetof(refmvs_frame, mfmv_ref) + n] % 7;
        rf.mfmv_ref2cur[n] = (int8_t)((rf_c[offsetof(refmvs_frame, mfmv_ref2cur) + n] & 63) - 32);
        for (int r = 0; r < 7; r++)
            rf.mfmv_ref2ref[n][r] = rf_c[offsetof(refmvs_frame, mfmv_ref2ref) + n * 7 + r] & 31;
    }
    rf.rp_proj = (refmvs_temporal_block *)proj_r;
    rf.rp_ref = (refmvs_temporal_block *const *)refs_native;
    for (int i = 0; i < 7; i++) refs_native[i] = (refmvs_temporal_block *)refs_r[i];

    memcpy(rf_c, &rf, sizeof rf);
    uint64_t v = RPPROJ_ADDR;
    memcpy(rf_c + offsetof(refmvs_frame, rp_proj), &v, 8);
    v = RPPTRS_ADDR;
    memcpy(rf_c + offsetof(refmvs_frame, rp_ref), &v, 8);
    for (int i = 0; i < 7; i++) {
        v = RPREF_ADDR(i);
        memcpy(rpptrs_c + 8 * i, &v, 8);
    }
    memcpy(proj_r, proj_c, RPPROJ_N);
    for (int i = 0; i < 7; i++)
        memcpy(refs_r[i], refs_c[i], RPREF_N);

    fv_region_add("rf", RF_ADDR, sizeof rf, rf_c, 0);
    fv_region_add("rp_proj", RPPROJ_ADDR, RPPROJ_N, proj_c, 1);
    fv_region_add("rpptrs", RPPTRS_ADDR, 7 * 8, rpptrs_c, 0);
    for (int i = 0; i < 7; i++)
        fv_region_add("rp_ref", RPREF_ADDR(i), RPREF_N, refs_c[i], 0);

#ifndef FV_NO_REF   /* diagnostics: time one side alone */
    load_tmvs_c(&rf, 0, CS8, CE8, RS8, RE8);
#endif

    const uint64_t args[6] = {RF_ADDR, 0, CS8, CE8, RS8, RE8};
#ifndef FV_NO_CAND
    fv_call(ENTRY, args, 6);
#endif

    for (uint64_t i = 0; i < RPPROJ_N; i++) {
#ifdef FV_CBMC
        __CPROVER_assert(proj_c[i] == proj_r[i], "equivalence: rp_proj differs from the C reference");
#else
        if (proj_c[i] != proj_r[i]) {
            fprintf(stderr, "FV-FAIL: rp_proj differs at byte %llu: candidate %u reference %u\n",
                    (unsigned long long)i, proj_c[i], proj_r[i]);
            exit(4);
        }
#endif
    }
    return 0;
}
