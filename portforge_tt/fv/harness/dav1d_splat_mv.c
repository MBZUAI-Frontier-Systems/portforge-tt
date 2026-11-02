/* dav1d splat_mv: lifted RVV vs src/refmvs.c's splat_mv_c.
 * Domain (decoder and checkasm): bw4 in {1,2,4,8,16,32}, bh4 >= 1, bx4 a
 * multiple of bw4 inside a 32-block row; rows reached only through rr[]
 * (independent addresses, 64-byte aligned, gaps between them); rmv 4-byte
 * aligned. Every byte of every row is checked, and rr/rmv are read-only.
 * Bounded configuration: -DBW4 -DBH4 -DBX4. */
#include <stdlib.h>
#include "fv_runtime.h"
#include "lifted.c"
#include "reference.c"   /* splat_mv_c, verbatim from the pinned src/refmvs.c */

#ifndef BW4
#define BW4 4
#endif
#ifndef BH4
#define BH4 2
#endif
#ifndef BX4
#define BX4 0
#endif
#ifndef ENTRY
#define ENTRY fvf_dav1d_splat_mv_rvv
#endif

#define ROW_BYTES (32 * 12)
#define ROW_ADDR(k) (0x400000ULL + (uint64_t)(k) * 0x1000ULL)
#define RR_ADDR 0x200000ULL
#define RMV_ADDR 0x300004ULL      /* 4-byte aligned, not 8 or 16 */

static uint8_t rows_r[BH4][ROW_BYTES];

int main(int argc, char **argv)
{
#ifndef FV_CBMC
    if (argc > 1) fv_rng = strtoull(argv[1], 0, 10) * 0x9E3779B97F4A7C15ULL + 1;
    if (argc > 2) fv_agnostic_ones = atoi(argv[2]);
#else
    (void)argc; (void)argv;
#endif
    _Static_assert(BX4 % BW4 == 0 && BX4 + BW4 <= 32, "bx4 outside the domain");
    uint8_t *rows_c[BH4], *rmv_c = fv_alloc(12), *rr_c = fv_alloc(8 * BH4);
    for (int k = 0; k < BH4; k++) rows_c[k] = fv_alloc(ROW_BYTES);
    for (int k = 0; k < BH4; k++) {
        fv_fill(rows_c[k], ROW_BYTES);
        memcpy(rows_r[k], rows_c[k], ROW_BYTES);
        fv_region_add("row", ROW_ADDR(k), ROW_BYTES, rows_c[k], 1);
        for (int b = 0; b < 8; b++) rr_c[8 * k + b] = (uint8_t)(ROW_ADDR(k) >> (8 * b));
    }
    fv_fill(rmv_c, 12);
    fv_region_add("rr", RR_ADDR, 8 * BH4, rr_c, 0);
    fv_region_add("rmv", RMV_ADDR, 12, rmv_c, 0);

    refmvs_block *rr_ref[BH4];
    for (int k = 0; k < BH4; k++) rr_ref[k] = (refmvs_block *)rows_r[k];
    refmvs_block rmv_ref;
    memcpy(&rmv_ref, rmv_c, 12);
    splat_mv_c(rr_ref, &rmv_ref, BX4, BW4, BH4);

    const uint64_t args[5] = {RR_ADDR, RMV_ADDR, BX4, BW4, BH4};
    fv_call(ENTRY, args, 5);
    for (int k = 0; k < BH4; k++) fv_compare("row", rows_c[k], rows_r[k], ROW_BYTES);
    return 0;
}
