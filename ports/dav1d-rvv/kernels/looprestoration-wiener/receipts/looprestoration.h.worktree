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

static ALWAYS_INLINE void loop_restoration_dsp_init_riscv(Dav1dLoopRestorationDSPContext *const c)
{
    const unsigned flags = dav1d_get_cpu_flags();
    if (!(flags & DAV1D_RISCV_CPU_FLAG_V)) return;

    c->wiener[0] = c->wiener[1] = wiener_filter_rvv;
}

#else

static ALWAYS_INLINE void loop_restoration_dsp_init_riscv(Dav1dLoopRestorationDSPContext *const c)
{
    (void)c;
}

#endif /* BITDEPTH == 8 */

#endif /* DAV1D_SRC_RISCV_LOOPRESTORATION_H */
