/* Call each sgr kernel from the built object; last line before SIGILL = culprit. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

void dav1d_sgr_box3_row_h_8bpc_rvv(int32_t *sumsq, int16_t *sum,
                                   const uint8_t (*left)[4], const uint8_t *src,
                                   int w, int edges);
void dav1d_sgr_box3_row_v_8bpc_rvv(int32_t **sumsq, int16_t **sum,
                                   int32_t *so, int16_t *sumo, int w);
void dav1d_sgr_calc_ab1_8bpc_rvv(int32_t *AA, int16_t *BB, int w,
                                 unsigned s, const uint8_t *x);
void dav1d_sgr_finish_filter_row1_8bpc_rvv(int16_t *tmp, const uint8_t *src,
                                           int32_t **A, int16_t **B, int w);
void dav1d_sgr_weighted_row1_8bpc_rvv(uint8_t *dst, const int16_t *t1,
                                      int w, int w1);

static uint8_t lut[256];
static uint8_t srcbuf[64], dst[64], left[4][4];
static int32_t aa[64], so[64], r32[3][64];
static int16_t bb[64], sumo[64], tmp[64], r16[3][64];

int main(void) {
    for (int i = 0; i < 256; i++) lut[i] = (uint8_t)i;
    for (int i = 0; i < 64; i++) srcbuf[i] = 128;
    memset(left, 7, sizeof(left));

    int w = 37;
    printf("box3_row_h w=%d\n", w); fflush(stdout);
    dav1d_sgr_box3_row_h_8bpc_rvv(aa, bb, left, srcbuf, w, 0xf);
    printf("  -> aa[0..2]=%d %d %d sum[0..2]=%d %d %d\n",
           aa[0], aa[1], aa[2], bb[0], bb[1], bb[2]); fflush(stdout);

    printf("box3_row_h w=1\n"); fflush(stdout);
    dav1d_sgr_box3_row_h_8bpc_rvv(aa, bb, NULL, srcbuf, 1, 0);
    printf("  ok\n"); fflush(stdout);

    { int32_t *p32[3] = { r32[0], r32[1], r32[2] };
      int16_t *p16[3] = { r16[0], r16[1], r16[2] };
      printf("box3_row_v w=%d\n", w); fflush(stdout);
      dav1d_sgr_box3_row_v_8bpc_rvv(p32, p16, so, sumo, w);
      printf("  ok\n"); fflush(stdout); }

    printf("calc_ab1 w=%d s=2589\n", w); fflush(stdout);
    dav1d_sgr_calc_ab1_8bpc_rvv(aa, bb, w, 2589, lut);
    printf("  -> aa[0]=%d bb[0]=%d\n", aa[0], bb[0]); fflush(stdout);

    { int32_t *A[3] = { r32[0], r32[1], r32[2] };
      int16_t *B[3] = { r16[0], r16[1], r16[2] };
      printf("finish_filter_row1 w=%d\n", w); fflush(stdout);
      dav1d_sgr_finish_filter_row1_8bpc_rvv(tmp, srcbuf, A, B, w);
      printf("  -> tmp[0]=%d\n", tmp[0]); fflush(stdout); }

    printf("weighted_row1 w=%d w1=100\n", w); fflush(stdout);
    dav1d_sgr_weighted_row1_8bpc_rvv(dst, tmp, w, 100);
    printf("  -> dst[0]=%d\n", dst[0]); fflush(stdout);

    printf("all kernels ok\n");
    return 0;
}
