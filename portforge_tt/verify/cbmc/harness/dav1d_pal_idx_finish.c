/* dav1d pal_idx_finish: lifted RVV vs src/pal.c's pal_idx_finish_c.
 * Domain (the C reference's asserts, the decoder's palette): bw, bh in {4..64}
 * powers of two, 4 <= w <= bw and 4 <= h <= bh multiples of 4, indices < 8.
 * Bounded configuration: -DBW -DBH -DW -DH. */
#include <stdlib.h>
#include "fv_runtime.h"
#include "lifted.c"
#include "src/pal.c"

#ifndef BW
#define BW 8
#endif
#ifndef BH
#define BH 4
#endif
#ifndef W
#define W BW
#endif
#ifndef H
#define H BH
#endif
#ifndef ENTRY
#define ENTRY fvf_dav1d_pal_idx_finish_rvv
#endif

#define SRC_N (BW * H)
#define DST_N ((BW / 2) * BH)
static uint8_t src_r[SRC_N], dst_r[DST_N];

int main(int argc, char **argv)
{
#ifndef FV_CBMC
    if (argc > 1) fv_rng = strtoull(argv[1], 0, 10) * 0x9E3779B97F4A7C15ULL + 1;
    if (argc > 2) fv_agnostic_ones = atoi(argv[2]);
#else
    (void)argc; (void)argv;
#endif
    uint8_t *src_c = fv_alloc(SRC_N), *dst_c = fv_alloc(DST_N);
    fv_fill(src_c, SRC_N);
    fv_fill(dst_c, DST_N);
    for (int i = 0; i < SRC_N; i++) {
#ifdef FV_CBMC
        __CPROVER_assume(src_c[i] < 8);
#else
        src_c[i] &= 7;
#endif
    }
    memcpy(src_r, src_c, SRC_N);
    memcpy(dst_r, dst_c, DST_N);
    fv_region_add("src", 0x200000, SRC_N, src_c, 0);
    fv_region_add("dst", 0x300000, DST_N, dst_c, 1);

    pal_idx_finish_c(dst_r, src_r, BW, BH, W, H);
    const uint64_t args[6] = {0x300000, 0x200000, BW, BH, W, H};
    fv_call(ENTRY, args, 6);
    fv_compare("dst", dst_c, dst_r, DST_N);
    return 0;
}
