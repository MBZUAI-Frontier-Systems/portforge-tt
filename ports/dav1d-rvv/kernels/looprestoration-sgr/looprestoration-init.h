/*****************************************************************************
 * looprestoration.h: riscv 8bpc loop restoration (Wiener filter)
 *****************************************************************************/

#ifndef DAV1D_SRC_RISCV_LOOPRESTORATION_H
#define DAV1D_SRC_RISCV_LOOPRESTORATION_H

#include "src/cpu.h"
#include "src/looprestoration.h"

#if BITDEPTH == 8

// The two kernels live in src/riscv/64/looprestoration.S; they are the
// arithmetic of wiener_filter_h() and wiener_filter_v() from
// src/looprestoration_tmpl.c, bit-exact.  The block-level control flow
// (row buffers, ptrs rotation, top/bottom padding) is kept in C below, so
// 7-tap and 5-tap share one driver: the C reference treats the 5-tap filter
// as the 7-tap one with fh/fv[0] = fh/fv[6] = 0, and both kernels apply the
// taps unconditionally.
void BF(dav1d_wiener_filter_h, rvv)(uint16_t *dst, const pixel (*left)[4],
                                    const pixel *src, const int16_t fh[8],
                                    const int w, const enum LrEdgeFlags edges);
void BF(dav1d_wiener_filter_v, rvv)(pixel *dst, const uint16_t *const rows[7],
                                    const int16_t fv[8], const int w);

// 256 * 1.5 + 3 + 3 = 390
#define REST_UNIT_STRIDE_RVV (390)

// wiener_filter_hv() from the C reference: horizontal pass into a temporary
// row, vertical pass consuming the six stored rows plus the temporary, then
// store the temporary as the newest row and rotate the window.  rows[6] is
// passed to the kernel explicitly (the C reads ptrs[5] twice there), so the
// kernel never aliases the row the temporary is copied into.
static void wiener_filter_hv_rvv(pixel *p, uint16_t **ptrs,
                                 const pixel (*left)[4],
                                 const pixel *src, const int16_t filter[2][8],
                                 const int w, const enum LrEdgeFlags edges)
{
    ALIGN_STK_16(uint16_t, tmp, REST_UNIT_STRIDE_RVV,);
    const uint16_t *rows[7];

    BF(dav1d_wiener_filter_h, rvv)(tmp, left, src, filter[0], w, edges);

    for (int k = 0; k < 6; k++)
        rows[k] = ptrs[k];
    rows[6] = tmp;
    BF(dav1d_wiener_filter_v, rvv)(p, rows, filter[1], w);

    memcpy(ptrs[6], tmp, sizeof(uint16_t) * REST_UNIT_STRIDE_RVV);

    // Rotate the window of pointers. Shift the 6 pointers downwards one step.
    for (int i = 0; i < 6; i++)
        ptrs[i] = ptrs[i + 1];
    ptrs[6] = ptrs[0];
}

// wiener_filter_v() from the C reference, at the bottom of the filtered area
// (the 7th row is assumed identical to the 6th).
static void wiener_filter_v_rvv(pixel *p, uint16_t **ptrs, const int16_t fv[8],
                                const int w)
{
    const uint16_t *rows[7];

    for (int k = 0; k < 6; k++)
        rows[k] = ptrs[k];
    rows[6] = ptrs[5];
    BF(dav1d_wiener_filter_v, rvv)(p, rows, fv, w);

    for (int i = 0; i < 5; i++)
        ptrs[i] = ptrs[i + 1];
}

static void wiener_filter_rvv(pixel *p, const ptrdiff_t stride,
                              const pixel (*left)[4],
                              const pixel *lpf, const int w, int h,
                              const LooprestorationParams *const params,
                              const enum LrEdgeFlags edges HIGHBD_DECL_SUFFIX)
{
    // Values stored between horizontal and vertical filtering don't
    // fit in a uint8_t.
    uint16_t hor[6 * REST_UNIT_STRIDE_RVV];
    uint16_t *ptrs[7], *rows[6];
    for (int i = 0; i < 6; i++)
        rows[i] = &hor[i * REST_UNIT_STRIDE_RVV];
    const int16_t (*const filter)[8] = params->filter;
    const int16_t *fh = params->filter[0];
    const int16_t *fv = params->filter[1];
    const pixel *lpf_bottom = lpf + 6*PXSTRIDE(stride);

    const pixel *src = p;
    if (edges & LR_HAVE_TOP) {
        ptrs[0] = rows[0];
        ptrs[1] = rows[0];
        ptrs[2] = rows[1];
        ptrs[3] = rows[2];
        ptrs[4] = rows[2];
        ptrs[5] = rows[2];

        BF(dav1d_wiener_filter_h, rvv)(rows[0], NULL, lpf, fh, w, edges);
        lpf += PXSTRIDE(stride);
        BF(dav1d_wiener_filter_h, rvv)(rows[1], NULL, lpf, fh, w, edges);

        BF(dav1d_wiener_filter_h, rvv)(rows[2], left, src, fh, w, edges);
        left++;
        src += PXSTRIDE(stride);

        if (--h <= 0)
            goto v1;

        ptrs[4] = ptrs[5] = rows[3];
        BF(dav1d_wiener_filter_h, rvv)(rows[3], left, src, fh, w, edges);
        left++;
        src += PXSTRIDE(stride);

        if (--h <= 0)
            goto v2;

        ptrs[5] = rows[4];
        BF(dav1d_wiener_filter_h, rvv)(rows[4], left, src, fh, w, edges);
        left++;
        src += PXSTRIDE(stride);

        if (--h <= 0)
            goto v3;
    } else {
        ptrs[0] = rows[0];
        ptrs[1] = rows[0];
        ptrs[2] = rows[0];
        ptrs[3] = rows[0];
        ptrs[4] = rows[0];
        ptrs[5] = rows[0];

        BF(dav1d_wiener_filter_h, rvv)(rows[0], left, src, fh, w, edges);
        left++;
        src += PXSTRIDE(stride);

        if (--h <= 0)
            goto v1;

        ptrs[4] = ptrs[5] = rows[1];
        BF(dav1d_wiener_filter_h, rvv)(rows[1], left, src, fh, w, edges);
        left++;
        src += PXSTRIDE(stride);

        if (--h <= 0)
            goto v2;

        ptrs[5] = rows[2];
        BF(dav1d_wiener_filter_h, rvv)(rows[2], left, src, fh, w, edges);
        left++;
        src += PXSTRIDE(stride);

        if (--h <= 0)
            goto v3;

        ptrs[6] = rows[3];
        wiener_filter_hv_rvv(p, ptrs, left, src, filter, w, edges);
        left++;
        src += PXSTRIDE(stride);
        p += PXSTRIDE(stride);

        if (--h <= 0)
            goto v3;

        ptrs[6] = rows[4];
        wiener_filter_hv_rvv(p, ptrs, left, src, filter, w, edges);
        left++;
        src += PXSTRIDE(stride);
        p += PXSTRIDE(stride);

        if (--h <= 0)
            goto v3;
    }

    ptrs[6] = ptrs[5] + REST_UNIT_STRIDE_RVV;
    do {
        wiener_filter_hv_rvv(p, ptrs, left, src, filter, w, edges);
        left++;
        src += PXSTRIDE(stride);
        p += PXSTRIDE(stride);
    } while (--h > 0);

    if (!(edges & LR_HAVE_BOTTOM))
        goto v3;

    wiener_filter_hv_rvv(p, ptrs, NULL, lpf_bottom, filter, w, edges);
    lpf_bottom += PXSTRIDE(stride);
    p += PXSTRIDE(stride);

    wiener_filter_hv_rvv(p, ptrs, NULL, lpf_bottom, filter, w, edges);
    p += PXSTRIDE(stride);
v1:
    wiener_filter_v_rvv(p, ptrs, fv, w);

    return;

v3:
    wiener_filter_v_rvv(p, ptrs, fv, w);
    p += PXSTRIDE(stride);
v2:
    wiener_filter_v_rvv(p, ptrs, fv, w);
    p += PXSTRIDE(stride);
    goto v1;
}

// ---------------------------------------------------------------------------
// SGR (self-guided restoration).  The stage kernels live in
// src/riscv/64/looprestoration.S and are bit-exact transcriptions of the
// 8bpc arithmetic in src/looprestoration_tmpl.c; the block-level control
// flow below mirrors sgr_3x3_c() exactly (buffers, pointer rotation,
// startup shapes, bottom padding).  Named with an _rvv suffix throughout
// so the static helpers of looprestoration_tmpl.c (rotate, sgr_box3_vert,
// ...) stay unambiguous in the including translation unit.
// ---------------------------------------------------------------------------

void BF(dav1d_sgr_box3_row_h, rvv)(int32_t *sumsq, coef *sum,
                                   const pixel (*left)[4],
                                   const pixel *src, const int w,
                                   const enum LrEdgeFlags edges);
void BF(dav1d_sgr_box3_row_v, rvv)(int32_t **sumsq, coef **sum,
                                   int32_t *sumsq_out, coef *sum_out,
                                   const int w);
void BF(dav1d_sgr_calc_ab1, rvv)(int32_t *AA, coef *BB, const int w,
                                 const unsigned s, const uint8_t *x_by_x);
void BF(dav1d_sgr_finish_filter_row1, rvv)(coef *tmp, const pixel *src,
                                           int32_t **A_ptrs, coef **B_ptrs,
                                           const int w);
void BF(dav1d_sgr_weighted_row1, rvv)(pixel *dst, const coef *t1,
                                      const int w, const int w1);
void BF(dav1d_sgr_box5_row_h, rvv)(int32_t *sumsq, coef *sum,
                                   const pixel (*left)[4],
                                   const pixel *src, const int w,
                                   const enum LrEdgeFlags edges);
void BF(dav1d_sgr_box5_row_v, rvv)(int32_t **sumsq, coef **sum,
                                   int32_t *sumsq_out, coef *sum_out,
                                   const int w);
void BF(dav1d_sgr_calc_ab2, rvv)(int32_t *AA, coef *BB, const int w,
                                 const unsigned s, const uint8_t *x_by_x);
void BF(dav1d_sgr_finish_filter2, rvv)(coef *tmp, const pixel *src,
                                       const ptrdiff_t src_stride,
                                       int32_t **A_ptrs, coef **B_ptrs,
                                       const int w, const int h);
void BF(dav1d_sgr_weighted2, rvv)(pixel *dst, const ptrdiff_t dst_stride,
                                  const coef *t1, const coef *t2,
                                  const int w, const int h,
                                  const int w0, const int w1);

extern const uint8_t dav1d_sgr_x_by_x[256];

// BUF_STRIDE in the C reference
#define SGR_BUF_STRIDE_RVV (384 + 16)
// FILTER_OUT_STRIDE in the C reference
#define SGR_FILTER_OUT_STRIDE_RVV (384)

static NOINLINE void sgr_rotate_rvv(int32_t **sumsq_ptrs, coef **sum_ptrs,
                                    const int n)
{
    int32_t *tmp32 = sumsq_ptrs[0];
    coef *tmpc = sum_ptrs[0];
    for (int i = 0; i < n - 1; i++) {
        sumsq_ptrs[i] = sumsq_ptrs[i + 1];
        sum_ptrs[i] = sum_ptrs[i + 1];
    }
    sumsq_ptrs[n - 1] = tmp32;
    sum_ptrs[n - 1] = tmpc;
}

// sgr_box3_vert(): vertical box sum + the per-pixel a/b weights + rotation.
static void sgr_box3_vert_rvv(int32_t **sumsq, coef **sum,
                              int32_t *AA, coef *BB,
                              const int w, const unsigned s)
{
    BF(dav1d_sgr_box3_row_v, rvv)(sumsq, sum, AA, BB, w);
    BF(dav1d_sgr_calc_ab1, rvv)(AA, BB, w, s, dav1d_sgr_x_by_x);
    sgr_rotate_rvv(sumsq, sum, 3);
}

// sgr_box3_hv(): one fresh input row through the horizontal box, then the
// vertical box/weights over the 3-row window.
static void sgr_box3_hv_rvv(int32_t **sumsq, coef **sum,
                            int32_t *AA, coef *BB,
                            const pixel (*left)[4],
                            const pixel *src, const int w,
                            const unsigned s, const enum LrEdgeFlags edges)
{
    BF(dav1d_sgr_box3_row_h, rvv)(sumsq[2], sum[2], left, src, w, edges);
    sgr_box3_vert_rvv(sumsq, sum, AA, BB, w, s);
}

// sgr_finish1(): one output row from the 8-neighborhood filter + blend.
static void sgr_finish1_rvv(pixel **dst, const ptrdiff_t stride,
                            int32_t **A_ptrs, coef **B_ptrs,
                            const int w, const int w1)
{
    ALIGN_STK_16(coef, tmp, SGR_FILTER_OUT_STRIDE_RVV,);

    BF(dav1d_sgr_finish_filter_row1, rvv)(tmp, *dst, A_ptrs, B_ptrs, w);
    BF(dav1d_sgr_weighted_row1, rvv)(*dst, tmp, w, w1);
    *dst += PXSTRIDE(stride);
    sgr_rotate_rvv(A_ptrs, B_ptrs, 3);
}

static NOINLINE void sgr_rotate5_x2_rvv(int32_t **sumsq_ptrs, coef **sum_ptrs)
{
    int32_t *tmp32[2];
    coef *tmpc[2];
    memcpy(tmp32, sumsq_ptrs, 2 * sizeof(int32_t*));
    memcpy(tmpc, sum_ptrs, 2 * sizeof(coef*));
    memmove(&sumsq_ptrs[0], &sumsq_ptrs[2], 3 * sizeof(int32_t*));
    memmove(&sum_ptrs[0], &sum_ptrs[2], 3 * sizeof(coef*));
    memcpy(&sumsq_ptrs[3], tmp32, 2 * sizeof(int32_t*));
    memcpy(&sum_ptrs[3], tmpc, 2 * sizeof(coef*));
}

// sgr_box5_vert(): vertical box sum + the per-pixel a/b weights + rotation.
static void sgr_box5_vert_rvv(int32_t **sumsq, coef **sum,
                              int32_t *AA, coef *BB,
                              const int w, const unsigned s)
{
    BF(dav1d_sgr_box5_row_v, rvv)(sumsq, sum, AA, BB, w);
    BF(dav1d_sgr_calc_ab2, rvv)(AA, BB, w, s, dav1d_sgr_x_by_x);
    sgr_rotate5_x2_rvv(sumsq, sum);
}

// sgr_finish2(): up to two output rows from the 6-neighborhood filter.
static void sgr_finish2_rvv(pixel **dst, const ptrdiff_t stride,
                            int32_t **A_ptrs, coef **B_ptrs,
                            const int w, const int h, const int w0)
{
    ALIGN_STK_16(coef, tmp, 2 * SGR_FILTER_OUT_STRIDE_RVV,);

    BF(dav1d_sgr_finish_filter2, rvv)(tmp, *dst, stride, A_ptrs, B_ptrs, w, h);
    BF(dav1d_sgr_weighted_row1, rvv)(*dst, tmp, w, w0);
    *dst += PXSTRIDE(stride);
    if (h > 1) {
        BF(dav1d_sgr_weighted_row1, rvv)(*dst, tmp + SGR_FILTER_OUT_STRIDE_RVV,
                                         w, w0);
        *dst += PXSTRIDE(stride);
    }
    sgr_rotate_rvv(A_ptrs, B_ptrs, 2);
}

static void sgr_3x3_rvv(pixel *p, const ptrdiff_t stride,
                        const pixel (*left)[4],
                        const pixel *lpf, const int w, int h,
                        const LooprestorationParams *const params,
                        const enum LrEdgeFlags edges)
{
    ALIGN_STK_16(int32_t, sumsq_buf, SGR_BUF_STRIDE_RVV * 3 + 16,);
    ALIGN_STK_16(coef, sum_buf, SGR_BUF_STRIDE_RVV * 3 + 16,);
    int32_t *sumsq_ptrs[3], *sumsq_rows[3];
    coef *sum_ptrs[3], *sum_rows[3];
    for (int i = 0; i < 3; i++) {
        sumsq_rows[i] = &sumsq_buf[i * SGR_BUF_STRIDE_RVV];
        sum_rows[i] = &sum_buf[i * SGR_BUF_STRIDE_RVV];
    }

    ALIGN_STK_16(int32_t, A_buf, SGR_BUF_STRIDE_RVV * 3 + 16,);
    ALIGN_STK_16(coef, B_buf, SGR_BUF_STRIDE_RVV * 3 + 16,);
    int32_t *A_ptrs[3];
    coef *B_ptrs[3];
    for (int i = 0; i < 3; i++) {
        A_ptrs[i] = &A_buf[i * SGR_BUF_STRIDE_RVV];
        B_ptrs[i] = &B_buf[i * SGR_BUF_STRIDE_RVV];
    }
    const pixel *src = p;
    const pixel *lpf_bottom = lpf + 6*PXSTRIDE(stride);

    if (edges & LR_HAVE_TOP) {
        sumsq_ptrs[0] = sumsq_rows[0];
        sumsq_ptrs[1] = sumsq_rows[1];
        sumsq_ptrs[2] = sumsq_rows[2];
        sum_ptrs[0] = sum_rows[0];
        sum_ptrs[1] = sum_rows[1];
        sum_ptrs[2] = sum_rows[2];

        BF(dav1d_sgr_box3_row_h, rvv)(sumsq_rows[0], sum_rows[0],
                                      NULL, lpf, w, edges);
        lpf += PXSTRIDE(stride);
        BF(dav1d_sgr_box3_row_h, rvv)(sumsq_rows[1], sum_rows[1],
                                      NULL, lpf, w, edges);

        sgr_box3_hv_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                        left, src, w, params->sgr.s1, edges);
        left++;
        src += PXSTRIDE(stride);
        sgr_rotate_rvv(A_ptrs, B_ptrs, 3);

        if (--h <= 0)
            goto vert_1;

        sgr_box3_hv_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                        left, src, w, params->sgr.s1, edges);
        left++;
        src += PXSTRIDE(stride);
        sgr_rotate_rvv(A_ptrs, B_ptrs, 3);

        if (--h <= 0)
            goto vert_2;
    } else {
        sumsq_ptrs[0] = sumsq_rows[0];
        sumsq_ptrs[1] = sumsq_rows[0];
        sumsq_ptrs[2] = sumsq_rows[0];
        sum_ptrs[0] = sum_rows[0];
        sum_ptrs[1] = sum_rows[0];
        sum_ptrs[2] = sum_rows[0];

        BF(dav1d_sgr_box3_row_h, rvv)(sumsq_rows[0], sum_rows[0],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        sgr_box3_vert_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                          w, params->sgr.s1);
        sgr_rotate_rvv(A_ptrs, B_ptrs, 3);

        if (--h <= 0)
            goto vert_1;

        sumsq_ptrs[2] = sumsq_rows[1];
        sum_ptrs[2] = sum_rows[1];

        sgr_box3_hv_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                        left, src, w, params->sgr.s1, edges);
        left++;
        src += PXSTRIDE(stride);
        sgr_rotate_rvv(A_ptrs, B_ptrs, 3);

        if (--h <= 0)
            goto vert_2;

        sumsq_ptrs[2] = sumsq_rows[2];
        sum_ptrs[2] = sum_rows[2];
    }

    do {
        sgr_box3_hv_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                        left, src, w, params->sgr.s1, edges);
        left++;
        src += PXSTRIDE(stride);

        sgr_finish1_rvv(&p, stride, A_ptrs, B_ptrs, w, params->sgr.w1);
    } while (--h > 0);

    if (!(edges & LR_HAVE_BOTTOM))
        goto vert_2;

    sgr_box3_hv_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                    NULL, lpf_bottom, w, params->sgr.s1, edges);
    lpf_bottom += PXSTRIDE(stride);

    sgr_finish1_rvv(&p, stride, A_ptrs, B_ptrs, w, params->sgr.w1);

    sgr_box3_hv_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                    NULL, lpf_bottom, w, params->sgr.s1, edges);

    sgr_finish1_rvv(&p, stride, A_ptrs, B_ptrs, w, params->sgr.w1);
    return;

vert_2:
    sumsq_ptrs[2] = sumsq_ptrs[1];
    sum_ptrs[2] = sum_ptrs[1];
    sgr_box3_vert_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                      w, params->sgr.s1);

    sgr_finish1_rvv(&p, stride, A_ptrs, B_ptrs, w, params->sgr.w1);

output_1:
    sumsq_ptrs[2] = sumsq_ptrs[1];
    sum_ptrs[2] = sum_ptrs[1];
    sgr_box3_vert_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                      w, params->sgr.s1);

    sgr_finish1_rvv(&p, stride, A_ptrs, B_ptrs, w, params->sgr.w1);
    return;

vert_1:
    sumsq_ptrs[2] = sumsq_ptrs[1];
    sum_ptrs[2] = sum_ptrs[1];
    sgr_box3_vert_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                      w, params->sgr.s1);
    sgr_rotate_rvv(A_ptrs, B_ptrs, 3);
    goto output_1;
}

static void sgr_5x5_rvv(pixel *p, const ptrdiff_t stride,
                        const pixel (*left)[4],
                        const pixel *lpf, const int w, int h,
                        const LooprestorationParams *const params,
                        const enum LrEdgeFlags edges)
{
    ALIGN_STK_16(int32_t, sumsq_buf, SGR_BUF_STRIDE_RVV * 5 + 16,);
    ALIGN_STK_16(coef, sum_buf, SGR_BUF_STRIDE_RVV * 5 + 16,);
    int32_t *sumsq_ptrs[5], *sumsq_rows[5];
    coef *sum_ptrs[5], *sum_rows[5];
    for (int i = 0; i < 5; i++) {
        sumsq_rows[i] = &sumsq_buf[i * SGR_BUF_STRIDE_RVV];
        sum_rows[i] = &sum_buf[i * SGR_BUF_STRIDE_RVV];
    }

    ALIGN_STK_16(int32_t, A_buf, SGR_BUF_STRIDE_RVV * 2 + 16,);
    ALIGN_STK_16(coef, B_buf, SGR_BUF_STRIDE_RVV * 2 + 16,);
    int32_t *A_ptrs[2];
    coef *B_ptrs[2];
    for (int i = 0; i < 2; i++) {
        A_ptrs[i] = &A_buf[i * SGR_BUF_STRIDE_RVV];
        B_ptrs[i] = &B_buf[i * SGR_BUF_STRIDE_RVV];
    }
    const pixel *src = p;
    const pixel *lpf_bottom = lpf + 6*PXSTRIDE(stride);

    if (edges & LR_HAVE_TOP) {
        sumsq_ptrs[0] = sumsq_rows[0];
        sumsq_ptrs[1] = sumsq_rows[0];
        sumsq_ptrs[2] = sumsq_rows[1];
        sumsq_ptrs[3] = sumsq_rows[2];
        sumsq_ptrs[4] = sumsq_rows[3];
        sum_ptrs[0] = sum_rows[0];
        sum_ptrs[1] = sum_rows[0];
        sum_ptrs[2] = sum_rows[1];
        sum_ptrs[3] = sum_rows[2];
        sum_ptrs[4] = sum_rows[3];

        BF(dav1d_sgr_box5_row_h, rvv)(sumsq_rows[0], sum_rows[0],
                                      NULL, lpf, w, edges);
        lpf += PXSTRIDE(stride);
        BF(dav1d_sgr_box5_row_h, rvv)(sumsq_rows[1], sum_rows[1],
                                      NULL, lpf, w, edges);

        BF(dav1d_sgr_box5_row_h, rvv)(sumsq_rows[2], sum_rows[2],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        if (--h <= 0)
            goto vert_1;

        BF(dav1d_sgr_box5_row_h, rvv)(sumsq_rows[3], sum_rows[3],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);
        sgr_box5_vert_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                          w, params->sgr.s0);
        sgr_rotate_rvv(A_ptrs, B_ptrs, 2);

        if (--h <= 0)
            goto vert_2;

        // ptrs are rotated by 2; both [3] and [4] now point at rows[0]; set
        // one of them to point at the previously unused rows[4].
        sumsq_ptrs[3] = sumsq_rows[4];
        sum_ptrs[3] = sum_rows[4];
    } else {
        sumsq_ptrs[0] = sumsq_rows[0];
        sumsq_ptrs[1] = sumsq_rows[0];
        sumsq_ptrs[2] = sumsq_rows[0];
        sumsq_ptrs[3] = sumsq_rows[0];
        sumsq_ptrs[4] = sumsq_rows[0];
        sum_ptrs[0] = sum_rows[0];
        sum_ptrs[1] = sum_rows[0];
        sum_ptrs[2] = sum_rows[0];
        sum_ptrs[3] = sum_rows[0];
        sum_ptrs[4] = sum_rows[0];

        BF(dav1d_sgr_box5_row_h, rvv)(sumsq_rows[0], sum_rows[0],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        if (--h <= 0)
            goto vert_1;

        sumsq_ptrs[4] = sumsq_rows[1];
        sum_ptrs[4] = sum_rows[1];

        BF(dav1d_sgr_box5_row_h, rvv)(sumsq_rows[1], sum_rows[1],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        sgr_box5_vert_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                          w, params->sgr.s0);
        sgr_rotate_rvv(A_ptrs, B_ptrs, 2);

        if (--h <= 0)
            goto vert_2;

        sumsq_ptrs[3] = sumsq_rows[2];
        sumsq_ptrs[4] = sumsq_rows[3];
        sum_ptrs[3] = sum_rows[2];
        sum_ptrs[4] = sum_rows[3];

        BF(dav1d_sgr_box5_row_h, rvv)(sumsq_rows[2], sum_rows[2],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        if (--h <= 0)
            goto odd;

        BF(dav1d_sgr_box5_row_h, rvv)(sumsq_rows[3], sum_rows[3],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        sgr_box5_vert_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                          w, params->sgr.s0);
        sgr_finish2_rvv(&p, stride, A_ptrs, B_ptrs, w, 2, params->sgr.w0);

        if (--h <= 0)
            goto vert_2;

        // ptrs are rotated by 2; both [3] and [4] now point at rows[0]; set
        // one of them to point at the previously unused rows[4].
        sumsq_ptrs[3] = sumsq_rows[4];
        sum_ptrs[3] = sum_rows[4];
    }

    do {
        BF(dav1d_sgr_box5_row_h, rvv)(sumsq_ptrs[3], sum_ptrs[3],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        if (--h <= 0)
            goto odd;

        BF(dav1d_sgr_box5_row_h, rvv)(sumsq_ptrs[4], sum_ptrs[4],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        sgr_box5_vert_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                          w, params->sgr.s0);
        sgr_finish2_rvv(&p, stride, A_ptrs, B_ptrs, w, 2, params->sgr.w0);
    } while (--h > 0);

    if (!(edges & LR_HAVE_BOTTOM))
        goto vert_2;

    BF(dav1d_sgr_box5_row_h, rvv)(sumsq_ptrs[3], sum_ptrs[3],
                                  NULL, lpf_bottom, w, edges);
    lpf_bottom += PXSTRIDE(stride);
    BF(dav1d_sgr_box5_row_h, rvv)(sumsq_ptrs[4], sum_ptrs[4],
                                  NULL, lpf_bottom, w, edges);

output_2:
    sgr_box5_vert_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                      w, params->sgr.s0);
    sgr_finish2_rvv(&p, stride, A_ptrs, B_ptrs, w, 2, params->sgr.w0);
    return;

vert_2:
    // Duplicate the last row twice more
    sumsq_ptrs[3] = sumsq_ptrs[2];
    sumsq_ptrs[4] = sumsq_ptrs[2];
    sum_ptrs[3] = sum_ptrs[2];
    sum_ptrs[4] = sum_ptrs[2];
    goto output_2;

odd:
    // Copy the last row as padding once
    sumsq_ptrs[4] = sumsq_ptrs[3];
    sum_ptrs[4] = sum_ptrs[3];

    sgr_box5_vert_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                      w, params->sgr.s0);
    sgr_finish2_rvv(&p, stride, A_ptrs, B_ptrs, w, 2, params->sgr.w0);

output_1:
    // Duplicate the last row twice more
    sumsq_ptrs[3] = sumsq_ptrs[2];
    sumsq_ptrs[4] = sumsq_ptrs[2];
    sum_ptrs[3] = sum_ptrs[2];
    sum_ptrs[4] = sum_ptrs[2];

    sgr_box5_vert_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                      w, params->sgr.s0);
    // Output only one row
    sgr_finish2_rvv(&p, stride, A_ptrs, B_ptrs, w, 1, params->sgr.w0);
    return;

vert_1:
    // Copy the last row as padding once
    sumsq_ptrs[4] = sumsq_ptrs[3];
    sum_ptrs[4] = sum_ptrs[3];

    sgr_box5_vert_rvv(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                      w, params->sgr.s0);
    sgr_rotate_rvv(A_ptrs, B_ptrs, 2);

    goto output_1;
}

// sgr_finish_mix(): the 5x5 rows through the 6-neighborhood filter, the 3x3
// rows through the 8-neighborhood one, then the two-row weighted blend.
static void sgr_finish_mix_rvv(pixel **dst, const ptrdiff_t stride,
                               int32_t **A5_ptrs, coef **B5_ptrs,
                               int32_t **A3_ptrs, coef **B3_ptrs,
                               const int w, const int h,
                               const int w0, const int w1)
{
    ALIGN_STK_16(coef, tmp5, 2 * SGR_FILTER_OUT_STRIDE_RVV,);
    ALIGN_STK_16(coef, tmp3, 2 * SGR_FILTER_OUT_STRIDE_RVV,);

    BF(dav1d_sgr_finish_filter2, rvv)(tmp5, *dst, stride, A5_ptrs, B5_ptrs, w, h);
    BF(dav1d_sgr_finish_filter_row1, rvv)(tmp3, *dst, A3_ptrs, B3_ptrs, w);
    if (h > 1)
        BF(dav1d_sgr_finish_filter_row1, rvv)(tmp3 + SGR_FILTER_OUT_STRIDE_RVV,
                                              *dst + PXSTRIDE(stride),
                                              &A3_ptrs[1], &B3_ptrs[1], w);
    BF(dav1d_sgr_weighted2, rvv)(*dst, stride, tmp5, tmp3, w, h, w0, w1);
    *dst += h*PXSTRIDE(stride);
    sgr_rotate_rvv(A5_ptrs, B5_ptrs, 2);
    sgr_rotate_rvv(A3_ptrs, B3_ptrs, 4);
}

static void sgr_mix_rvv(pixel *p, const ptrdiff_t stride,
                        const pixel (*left)[4],
                        const pixel *lpf, const int w, int h,
                        const LooprestorationParams *const params,
                        const enum LrEdgeFlags edges)
{
    ALIGN_STK_16(int32_t, sumsq5_buf, SGR_BUF_STRIDE_RVV * 5 + 16,);
    ALIGN_STK_16(coef, sum5_buf, SGR_BUF_STRIDE_RVV * 5 + 16,);
    int32_t *sumsq5_ptrs[5], *sumsq5_rows[5];
    coef *sum5_ptrs[5], *sum5_rows[5];
    for (int i = 0; i < 5; i++) {
        sumsq5_rows[i] = &sumsq5_buf[i * SGR_BUF_STRIDE_RVV];
        sum5_rows[i] = &sum5_buf[i * SGR_BUF_STRIDE_RVV];
    }
    ALIGN_STK_16(int32_t, sumsq3_buf, SGR_BUF_STRIDE_RVV * 3 + 16,);
    ALIGN_STK_16(coef, sum3_buf, SGR_BUF_STRIDE_RVV * 3 + 16,);
    int32_t *sumsq3_ptrs[3], *sumsq3_rows[3];
    coef *sum3_ptrs[3], *sum3_rows[3];
    for (int i = 0; i < 3; i++) {
        sumsq3_rows[i] = &sumsq3_buf[i * SGR_BUF_STRIDE_RVV];
        sum3_rows[i] = &sum3_buf[i * SGR_BUF_STRIDE_RVV];
    }

    ALIGN_STK_16(int32_t, A5_buf, SGR_BUF_STRIDE_RVV * 2 + 16,);
    ALIGN_STK_16(coef, B5_buf, SGR_BUF_STRIDE_RVV * 2 + 16,);
    int32_t *A5_ptrs[2];
    coef *B5_ptrs[2];
    for (int i = 0; i < 2; i++) {
        A5_ptrs[i] = &A5_buf[i * SGR_BUF_STRIDE_RVV];
        B5_ptrs[i] = &B5_buf[i * SGR_BUF_STRIDE_RVV];
    }
    ALIGN_STK_16(int32_t, A3_buf, SGR_BUF_STRIDE_RVV * 4 + 16,);
    ALIGN_STK_16(coef, B3_buf, SGR_BUF_STRIDE_RVV * 4 + 16,);
    int32_t *A3_ptrs[4];
    coef *B3_ptrs[4];
    for (int i = 0; i < 4; i++) {
        A3_ptrs[i] = &A3_buf[i * SGR_BUF_STRIDE_RVV];
        B3_ptrs[i] = &B3_buf[i * SGR_BUF_STRIDE_RVV];
    }
    const pixel *src = p;
    const pixel *lpf_bottom = lpf + 6*PXSTRIDE(stride);

    if (edges & LR_HAVE_TOP) {
        sumsq5_ptrs[0] = sumsq5_rows[0];
        sumsq5_ptrs[1] = sumsq5_rows[0];
        sumsq5_ptrs[2] = sumsq5_rows[1];
        sumsq5_ptrs[3] = sumsq5_rows[2];
        sumsq5_ptrs[4] = sumsq5_rows[3];
        sum5_ptrs[0] = sum5_rows[0];
        sum5_ptrs[1] = sum5_rows[0];
        sum5_ptrs[2] = sum5_rows[1];
        sum5_ptrs[3] = sum5_rows[2];
        sum5_ptrs[4] = sum5_rows[3];

        sumsq3_ptrs[0] = sumsq3_rows[0];
        sumsq3_ptrs[1] = sumsq3_rows[1];
        sumsq3_ptrs[2] = sumsq3_rows[2];
        sum3_ptrs[0] = sum3_rows[0];
        sum3_ptrs[1] = sum3_rows[1];
        sum3_ptrs[2] = sum3_rows[2];

        BF(dav1d_sgr_box3_row_h, rvv)(sumsq3_rows[0], sum3_rows[0],
                                      NULL, lpf, w, edges);
        BF(dav1d_sgr_box5_row_h, rvv)(sumsq5_rows[0], sum5_rows[0],
                                      NULL, lpf, w, edges);
        lpf += PXSTRIDE(stride);
        BF(dav1d_sgr_box3_row_h, rvv)(sumsq3_rows[1], sum3_rows[1],
                                      NULL, lpf, w, edges);
        BF(dav1d_sgr_box5_row_h, rvv)(sumsq5_rows[1], sum5_rows[1],
                                      NULL, lpf, w, edges);

        BF(dav1d_sgr_box3_row_h, rvv)(sumsq3_rows[2], sum3_rows[2],
                                      left, src, w, edges);
        BF(dav1d_sgr_box5_row_h, rvv)(sumsq5_rows[2], sum5_rows[2],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, params->sgr.s1);
        sgr_rotate_rvv(A3_ptrs, B3_ptrs, 4);

        if (--h <= 0)
            goto vert_1;

        BF(dav1d_sgr_box3_row_h, rvv)(sumsq3_ptrs[2], sum3_ptrs[2],
                                      left, src, w, edges);
        BF(dav1d_sgr_box5_row_h, rvv)(sumsq5_rows[3], sum5_rows[3],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);
        sgr_box5_vert_rvv(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                          w, params->sgr.s0);
        sgr_rotate_rvv(A5_ptrs, B5_ptrs, 2);
        sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, params->sgr.s1);
        sgr_rotate_rvv(A3_ptrs, B3_ptrs, 4);

        if (--h <= 0)
            goto vert_2;

        // ptrs are rotated by 2; both [3] and [4] now point at rows[0]; set
        // one of them to point at the previously unused rows[4].
        sumsq5_ptrs[3] = sumsq5_rows[4];
        sum5_ptrs[3] = sum5_rows[4];
    } else {
        sumsq5_ptrs[0] = sumsq5_rows[0];
        sumsq5_ptrs[1] = sumsq5_rows[0];
        sumsq5_ptrs[2] = sumsq5_rows[0];
        sumsq5_ptrs[3] = sumsq5_rows[0];
        sumsq5_ptrs[4] = sumsq5_rows[0];
        sum5_ptrs[0] = sum5_rows[0];
        sum5_ptrs[1] = sum5_rows[0];
        sum5_ptrs[2] = sum5_rows[0];
        sum5_ptrs[3] = sum5_rows[0];
        sum5_ptrs[4] = sum5_rows[0];

        sumsq3_ptrs[0] = sumsq3_rows[0];
        sumsq3_ptrs[1] = sumsq3_rows[0];
        sumsq3_ptrs[2] = sumsq3_rows[0];
        sum3_ptrs[0] = sum3_rows[0];
        sum3_ptrs[1] = sum3_rows[0];
        sum3_ptrs[2] = sum3_rows[0];

        BF(dav1d_sgr_box3_row_h, rvv)(sumsq3_rows[0], sum3_rows[0],
                                      left, src, w, edges);
        BF(dav1d_sgr_box5_row_h, rvv)(sumsq5_rows[0], sum5_rows[0],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, params->sgr.s1);
        sgr_rotate_rvv(A3_ptrs, B3_ptrs, 4);

        if (--h <= 0)
            goto vert_1;

        sumsq5_ptrs[4] = sumsq5_rows[1];
        sum5_ptrs[4] = sum5_rows[1];

        sumsq3_ptrs[2] = sumsq3_rows[1];
        sum3_ptrs[2] = sum3_rows[1];

        BF(dav1d_sgr_box3_row_h, rvv)(sumsq3_rows[1], sum3_rows[1],
                                      left, src, w, edges);
        BF(dav1d_sgr_box5_row_h, rvv)(sumsq5_rows[1], sum5_rows[1],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        sgr_box5_vert_rvv(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                          w, params->sgr.s0);
        sgr_rotate_rvv(A5_ptrs, B5_ptrs, 2);
        sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, params->sgr.s1);
        sgr_rotate_rvv(A3_ptrs, B3_ptrs, 4);

        if (--h <= 0)
            goto vert_2;

        sumsq5_ptrs[3] = sumsq5_rows[2];
        sumsq5_ptrs[4] = sumsq5_rows[3];
        sum5_ptrs[3] = sum5_rows[2];
        sum5_ptrs[4] = sum5_rows[3];

        sumsq3_ptrs[2] = sumsq3_rows[2];
        sum3_ptrs[2] = sum3_rows[2];

        BF(dav1d_sgr_box3_row_h, rvv)(sumsq3_rows[2], sum3_rows[2],
                                      left, src, w, edges);
        BF(dav1d_sgr_box5_row_h, rvv)(sumsq5_rows[2], sum5_rows[2],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, params->sgr.s1);
        sgr_rotate_rvv(A3_ptrs, B3_ptrs, 4);

        if (--h <= 0)
            goto odd;

        BF(dav1d_sgr_box3_row_h, rvv)(sumsq3_ptrs[2], sum3_ptrs[2],
                                      left, src, w, edges);
        BF(dav1d_sgr_box5_row_h, rvv)(sumsq5_rows[3], sum5_rows[3],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        sgr_box5_vert_rvv(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                          w, params->sgr.s0);
        sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, params->sgr.s1);
        sgr_finish_mix_rvv(&p, stride, A5_ptrs, B5_ptrs, A3_ptrs, B3_ptrs,
                           w, 2, params->sgr.w0, params->sgr.w1);

        if (--h <= 0)
            goto vert_2;

        // ptrs are rotated by 2; both [3] and [4] now point at rows[0]; set
        // one of them to point at the previously unused rows[4].
        sumsq5_ptrs[3] = sumsq5_rows[4];
        sum5_ptrs[3] = sum5_rows[4];
    }

    do {
        BF(dav1d_sgr_box3_row_h, rvv)(sumsq3_ptrs[2], sum3_ptrs[2],
                                      left, src, w, edges);
        BF(dav1d_sgr_box5_row_h, rvv)(sumsq5_ptrs[3], sum5_ptrs[3],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, params->sgr.s1);
        sgr_rotate_rvv(A3_ptrs, B3_ptrs, 4);

        if (--h <= 0)
            goto odd;

        BF(dav1d_sgr_box3_row_h, rvv)(sumsq3_ptrs[2], sum3_ptrs[2],
                                      left, src, w, edges);
        BF(dav1d_sgr_box5_row_h, rvv)(sumsq5_ptrs[4], sum5_ptrs[4],
                                      left, src, w, edges);
        left++;
        src += PXSTRIDE(stride);

        sgr_box5_vert_rvv(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                          w, params->sgr.s0);
        sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, params->sgr.s1);
        sgr_finish_mix_rvv(&p, stride, A5_ptrs, B5_ptrs, A3_ptrs, B3_ptrs,
                           w, 2, params->sgr.w0, params->sgr.w1);
    } while (--h > 0);

    if (!(edges & LR_HAVE_BOTTOM))
        goto vert_2;

    BF(dav1d_sgr_box3_row_h, rvv)(sumsq3_ptrs[2], sum3_ptrs[2],
                                  NULL, lpf_bottom, w, edges);
    BF(dav1d_sgr_box5_row_h, rvv)(sumsq5_ptrs[3], sum5_ptrs[3],
                                  NULL, lpf_bottom, w, edges);
    lpf_bottom += PXSTRIDE(stride);
    sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                      w, params->sgr.s1);
    sgr_rotate_rvv(A3_ptrs, B3_ptrs, 4);

    BF(dav1d_sgr_box3_row_h, rvv)(sumsq3_ptrs[2], sum3_ptrs[2],
                                  NULL, lpf_bottom, w, edges);
    BF(dav1d_sgr_box5_row_h, rvv)(sumsq5_ptrs[4], sum5_ptrs[4],
                                  NULL, lpf_bottom, w, edges);

output_2:
    sgr_box5_vert_rvv(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                      w, params->sgr.s0);
    sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                      w, params->sgr.s1);
    sgr_finish_mix_rvv(&p, stride, A5_ptrs, B5_ptrs, A3_ptrs, B3_ptrs,
                       w, 2, params->sgr.w0, params->sgr.w1);
    return;

vert_2:
    // Duplicate the last row twice more
    sumsq5_ptrs[3] = sumsq5_ptrs[2];
    sumsq5_ptrs[4] = sumsq5_ptrs[2];
    sum5_ptrs[3] = sum5_ptrs[2];
    sum5_ptrs[4] = sum5_ptrs[2];

    sumsq3_ptrs[2] = sumsq3_ptrs[1];
    sum3_ptrs[2] = sum3_ptrs[1];
    sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                      w, params->sgr.s1);
    sgr_rotate_rvv(A3_ptrs, B3_ptrs, 4);

    sumsq3_ptrs[2] = sumsq3_ptrs[1];
    sum3_ptrs[2] = sum3_ptrs[1];

    goto output_2;

odd:
    // Copy the last row as padding once
    sumsq5_ptrs[4] = sumsq5_ptrs[3];
    sum5_ptrs[4] = sum5_ptrs[3];

    sumsq3_ptrs[2] = sumsq3_ptrs[1];
    sum3_ptrs[2] = sum3_ptrs[1];

    sgr_box5_vert_rvv(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                      w, params->sgr.s0);
    sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                      w, params->sgr.s1);
    sgr_finish_mix_rvv(&p, stride, A5_ptrs, B5_ptrs, A3_ptrs, B3_ptrs,
                       w, 2, params->sgr.w0, params->sgr.w1);

output_1:
    // Duplicate the last row twice more
    sumsq5_ptrs[3] = sumsq5_ptrs[2];
    sumsq5_ptrs[4] = sumsq5_ptrs[2];
    sum5_ptrs[3] = sum5_ptrs[2];
    sum5_ptrs[4] = sum5_ptrs[2];

    sumsq3_ptrs[2] = sumsq3_ptrs[1];
    sum3_ptrs[2] = sum3_ptrs[1];

    sgr_box5_vert_rvv(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                      w, params->sgr.s0);
    sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                      w, params->sgr.s1);
    sgr_rotate_rvv(A3_ptrs, B3_ptrs, 4);
    // Output only one row
    sgr_finish_mix_rvv(&p, stride, A5_ptrs, B5_ptrs, A3_ptrs, B3_ptrs,
                       w, 1, params->sgr.w0, params->sgr.w1);
    return;

vert_1:
    // Copy the last row as padding once
    sumsq5_ptrs[4] = sumsq5_ptrs[3];
    sum5_ptrs[4] = sum5_ptrs[3];

    sumsq3_ptrs[2] = sumsq3_ptrs[1];
    sum3_ptrs[2] = sum3_ptrs[1];

    sgr_box5_vert_rvv(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                      w, params->sgr.s0);
    sgr_rotate_rvv(A5_ptrs, B5_ptrs, 2);
    sgr_box3_vert_rvv(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                      w, params->sgr.s1);
    sgr_rotate_rvv(A3_ptrs, B3_ptrs, 4);

    goto output_1;
}

static ALWAYS_INLINE void loop_restoration_dsp_init_riscv(Dav1dLoopRestorationDSPContext *const c)
{
    const unsigned flags = dav1d_get_cpu_flags();
    if (!(flags & DAV1D_RISCV_CPU_FLAG_V)) return;

    c->wiener[0] = c->wiener[1] = wiener_filter_rvv;
    c->sgr[0] = sgr_5x5_rvv;
    c->sgr[1] = sgr_3x3_rvv;
    c->sgr[2] = sgr_mix_rvv;
}

#else

static ALWAYS_INLINE void loop_restoration_dsp_init_riscv(Dav1dLoopRestorationDSPContext *const c)
{
    (void)c;
}

#endif /* BITDEPTH == 8 */

#endif /* DAV1D_SRC_RISCV_LOOPRESTORATION_H */
