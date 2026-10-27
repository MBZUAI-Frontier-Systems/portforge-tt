/* Differential driver for the 8bpc Wiener RVV port (campaign c2-wiener).
 *
 * Three levels, all bit-exact against the C reference math of
 * src/looprestoration_tmpl.c (copied verbatim below, 8bpc constants):
 *   1. horizontal kernel vs wiener_filter_h()
 *   2. vertical kernel   vs wiener_filter_v()
 *   3. whole block       vs wiener_c() via the same driver flow the
 *      src/riscv/looprestoration.h header uses
 *
 * Build (in the toolchain image):
 *   riscv64-linux-gnu-gcc-14 -march=rv64gcv -O1 -static lr_diff.c \
 *       -o lr_diff -L/pf/build/dav1d-c2-wiener-rv64/src -ldav1d
 * Run: qemu-riscv64 -L /usr/riscv64-linux-gnu -cpu rv64,v=true,... ./lr_diff
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

typedef uint8_t pixel;

#define LR_HAVE_LEFT   (1 << 0)
#define LR_HAVE_RIGHT  (1 << 1)
#define LR_HAVE_TOP    (1 << 2)
#define LR_HAVE_BOTTOM (1 << 3)

#define REST_UNIT_STRIDE (390)

#define CHECKASM_ALIGN_PIXEL __attribute__((aligned(64)))

static unsigned long long rng_state = 0x12345678ULL;
static unsigned rnd(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned)(rng_state >> 33);
}

static int iclip(int x, int min, int max) {
    return x < min ? min : x > max ? max : x;
}

typedef struct {
    int16_t filter[2][8];
} LRParams;

/* ---------------- reference: verbatim from looprestoration_tmpl.c -------- */

static void ref_h(uint16_t *dst, const pixel (*left)[4],
                  const pixel *src, const int16_t fh[8],
                  const int w, const int edges)
{
    const int round_bits_h = 3;
    const int rounding_off_h = 1 << (round_bits_h - 1);
    const int clip_limit = 1 << (8 + 1 + 7 - round_bits_h);

    if (w < 6) {
        for (int x = 0; x < w; x++) {
            int sum = (1 << 14);
            sum += src[x] * 128;
            for (int i = 0; i < 7; i++) {
                int idx = x + i - 3;
                if (idx < 0) {
                    if (!(edges & LR_HAVE_LEFT))
                        sum += src[0] * fh[i];
                    else if (left)
                        sum += left[0][4 + idx] * fh[i];
                    else
                        sum += src[idx] * fh[i];
                } else if (idx >= w && !(edges & LR_HAVE_RIGHT)) {
                    sum += src[w - 1] * fh[i];
                } else
                    sum += src[idx] * fh[i];
            }
            sum = iclip((sum + rounding_off_h) >> round_bits_h, 0, clip_limit - 1);
            dst[x] = sum;
        }
        return;
    }

    int start = 3;
    if (!(edges & LR_HAVE_LEFT)) {
        for (int x = 0; x < 3; x++) {
            int sum = (1 << 14);
            sum += src[x] * 128;
            for (int i = 0; i < 7; i++) {
                int idx = x + i - 3;
                if (idx < 0)
                    sum += src[0] * fh[i];
                else
                    sum += src[idx] * fh[i];
            }
            sum = iclip((sum + rounding_off_h) >> round_bits_h, 0, clip_limit - 1);
            dst[x] = sum;
        }
    } else if (left) {
        for (int x = 0; x < 3; x++) {
            int sum = (1 << 14);
            sum += src[x] * 128;
            for (int i = 0; i < 7; i++) {
                int idx = x + i - 3;
                if (idx < 0)
                    sum += left[0][4 + idx] * fh[i];
                else
                    sum += src[idx] * fh[i];
            }
            sum = iclip((sum + rounding_off_h) >> round_bits_h, 0, clip_limit - 1);
            dst[x] = sum;
        }
    } else {
        start = 0;
    }
    int end = w - 3;
    if (edges & LR_HAVE_RIGHT)
        end = w;

    for (int x = start; x < end; x++) {
        int sum = (1 << 14);
        sum += src[x] * 128;
        for (int i = 0; i < 7; i++) {
            int idx = x + i - 3;
            sum += src[idx] * fh[i];
        }
        sum = iclip((sum + rounding_off_h) >> round_bits_h, 0, clip_limit - 1);
        dst[x] = sum;
    }

    for (int x = end; x < w; x++) {
        int sum = (1 << 14);
        sum += src[x] * 128;
        for (int i = 0; i < 7; i++) {
            int idx = x + i - 3;
            if (idx >= w)
                sum += src[w - 1] * fh[i];
            else
                sum += src[idx] * fh[i];
        }
        sum = iclip((sum + rounding_off_h) >> round_bits_h, 0, clip_limit - 1);
        dst[x] = sum;
    }
}

static void ref_v(pixel *p, uint16_t **ptrs, const int16_t fv[8], const int w)
{
    const int round_bits_v = 11;
    const int rounding_off_v = 1 << (round_bits_v - 1);
    const int round_offset = 1 << (8 + (round_bits_v - 1));

    for (int i = 0; i < w; i++) {
        int sum = -round_offset;
        for (int k = 0; k < 6; k++)
            sum += ptrs[k][i] * fv[k];
        sum += ptrs[5][i] * fv[6];
        p[i] = iclip((sum + rounding_off_v) >> round_bits_v, 0, 255);
    }
    for (int i = 0; i < 5; i++)
        ptrs[i] = ptrs[i + 1];
}

static void ref_hv(pixel *p, uint16_t **ptrs, const pixel (*left)[4],
                   const pixel *src, const int16_t filter[2][8],
                   const int w, const int edges)
{
    const int round_bits_v = 11;
    const int rounding_off_v = 1 << (round_bits_v - 1);
    const int round_offset = 1 << (8 + (round_bits_v - 1));

    const int16_t *fh = filter[0];
    const int16_t *fv = filter[1];

    uint16_t tmp[REST_UNIT_STRIDE];
    ref_h(tmp, left, src, fh, w, edges);

    for (int i = 0; i < w; i++) {
        int sum = -round_offset;
        for (int k = 0; k < 6; k++)
            sum += ptrs[k][i] * fv[k];
        sum += tmp[i] * fv[6];
        p[i] = iclip((sum + rounding_off_v) >> round_bits_v, 0, 255);
    }

    memcpy(ptrs[6], tmp, sizeof(uint16_t) * REST_UNIT_STRIDE);
    for (int i = 0; i < 6; i++)
        ptrs[i] = ptrs[i + 1];
    ptrs[6] = ptrs[0];
}

static void ref_wiener(pixel *p, const ptrdiff_t stride,
                       const pixel (*left)[4],
                       const pixel *lpf, const int w, int h,
                       const LRParams *const params, const int edges)
{
    uint16_t hor[6 * REST_UNIT_STRIDE];
    uint16_t *ptrs[7], *rows[6];
    for (int i = 0; i < 6; i++)
        rows[i] = &hor[i * REST_UNIT_STRIDE];
    const int16_t (*const filter)[8] = params->filter;
    const int16_t *fh = params->filter[0];
    const int16_t *fv = params->filter[1];
    const pixel *lpf_bottom = lpf + 6 * stride;

    const pixel *src = p;
    if (edges & LR_HAVE_TOP) {
        ptrs[0] = rows[0];
        ptrs[1] = rows[0];
        ptrs[2] = rows[1];
        ptrs[3] = rows[2];
        ptrs[4] = rows[2];
        ptrs[5] = rows[2];

        ref_h(rows[0], NULL, lpf, fh, w, edges);
        lpf += stride;
        ref_h(rows[1], NULL, lpf, fh, w, edges);

        ref_h(rows[2], left, src, fh, w, edges);
        left++;
        src += stride;

        if (--h <= 0)
            goto v1;

        ptrs[4] = ptrs[5] = rows[3];
        ref_h(rows[3], left, src, fh, w, edges);
        left++;
        src += stride;

        if (--h <= 0)
            goto v2;

        ptrs[5] = rows[4];
        ref_h(rows[4], left, src, fh, w, edges);
        left++;
        src += stride;

        if (--h <= 0)
            goto v3;
    } else {
        ptrs[0] = rows[0];
        ptrs[1] = rows[0];
        ptrs[2] = rows[0];
        ptrs[3] = rows[0];
        ptrs[4] = rows[0];
        ptrs[5] = rows[0];

        ref_h(rows[0], left, src, fh, w, edges);
        left++;
        src += stride;

        if (--h <= 0)
            goto v1;

        ptrs[4] = ptrs[5] = rows[1];
        ref_h(rows[1], left, src, fh, w, edges);
        left++;
        src += stride;

        if (--h <= 0)
            goto v2;

        ptrs[5] = rows[2];
        ref_h(rows[2], left, src, fh, w, edges);
        left++;
        src += stride;

        if (--h <= 0)
            goto v3;

        ptrs[6] = rows[3];
        ref_hv(p, ptrs, left, src, filter, w, edges);
        left++;
        src += stride;
        p += stride;

        if (--h <= 0)
            goto v3;

        ptrs[6] = rows[4];
        ref_hv(p, ptrs, left, src, filter, w, edges);
        left++;
        src += stride;
        p += stride;

        if (--h <= 0)
            goto v3;
    }

    ptrs[6] = ptrs[5] + REST_UNIT_STRIDE;
    do {
        ref_hv(p, ptrs, left, src, filter, w, edges);
        left++;
        src += stride;
        p += stride;
    } while (--h > 0);

    if (!(edges & LR_HAVE_BOTTOM))
        goto v3;

    ref_hv(p, ptrs, NULL, lpf_bottom, filter, w, edges);
    lpf_bottom += stride;
    p += stride;

    ref_hv(p, ptrs, NULL, lpf_bottom, filter, w, edges);
    p += stride;
v1:
    ref_v(p, ptrs, fv, w);
    return;

v3:
    ref_v(p, ptrs, fv, w);
    p += stride;
v2:
    ref_v(p, ptrs, fv, w);
    p += stride;
    goto v1;
}

/* ---------------- kernels under test ------------------------------------- */

void dav1d_wiener_filter_h_8bpc_rvv(uint16_t *dst, const pixel (*left)[4],
                                    const pixel *src, const int16_t fh[8],
                                    const int w, const int edges);
void dav1d_wiener_filter_v_8bpc_rvv(pixel *dst, const uint16_t *const rows[7],
                                    const int16_t fv[8], const int w);

/* clone of the header driver */
static void rvv_hv(pixel *p, uint16_t **ptrs, const pixel (*left)[4],
                   const pixel *src, const int16_t filter[2][8],
                   const int w, const int edges)
{
    uint16_t tmp[REST_UNIT_STRIDE];
    const uint16_t *rows[7];

    dav1d_wiener_filter_h_8bpc_rvv(tmp, left, src, filter[0], w, edges);

    for (int k = 0; k < 6; k++)
        rows[k] = ptrs[k];
    rows[6] = tmp;
    dav1d_wiener_filter_v_8bpc_rvv(p, rows, filter[1], w);

    memcpy(ptrs[6], tmp, sizeof(uint16_t) * REST_UNIT_STRIDE);
    for (int i = 0; i < 6; i++)
        ptrs[i] = ptrs[i + 1];
    ptrs[6] = ptrs[0];
}

static void rvv_v(pixel *p, uint16_t **ptrs, const int16_t fv[8], const int w)
{
    const uint16_t *rows[7];
    for (int k = 0; k < 6; k++)
        rows[k] = ptrs[k];
    rows[6] = ptrs[5];
    dav1d_wiener_filter_v_8bpc_rvv(p, rows, fv, w);
    for (int i = 0; i < 5; i++)
        ptrs[i] = ptrs[i + 1];
}

static void rvv_wiener(pixel *p, const ptrdiff_t stride,
                       const pixel (*left)[4],
                       const pixel *lpf, const int w, int h,
                       const LRParams *const params, const int edges)
{
    uint16_t hor[6 * REST_UNIT_STRIDE];
    uint16_t *ptrs[7], *rows[6];
    for (int i = 0; i < 6; i++)
        rows[i] = &hor[i * REST_UNIT_STRIDE];
    const int16_t (*const filter)[8] = params->filter;
    const int16_t *fh = params->filter[0];
    const int16_t *fv = params->filter[1];
    const pixel *lpf_bottom = lpf + 6 * stride;

    const pixel *src = p;
    if (edges & LR_HAVE_TOP) {
        ptrs[0] = rows[0];
        ptrs[1] = rows[0];
        ptrs[2] = rows[1];
        ptrs[3] = rows[2];
        ptrs[4] = rows[2];
        ptrs[5] = rows[2];

        dav1d_wiener_filter_h_8bpc_rvv(rows[0], NULL, lpf, fh, w, edges);
        lpf += stride;
        dav1d_wiener_filter_h_8bpc_rvv(rows[1], NULL, lpf, fh, w, edges);

        dav1d_wiener_filter_h_8bpc_rvv(rows[2], left, src, fh, w, edges);
        left++;
        src += stride;

        if (--h <= 0)
            goto v1;

        ptrs[4] = ptrs[5] = rows[3];
        dav1d_wiener_filter_h_8bpc_rvv(rows[3], left, src, fh, w, edges);
        left++;
        src += stride;

        if (--h <= 0)
            goto v2;

        ptrs[5] = rows[4];
        dav1d_wiener_filter_h_8bpc_rvv(rows[4], left, src, fh, w, edges);
        left++;
        src += stride;

        if (--h <= 0)
            goto v3;
    } else {
        ptrs[0] = rows[0];
        ptrs[1] = rows[0];
        ptrs[2] = rows[0];
        ptrs[3] = rows[0];
        ptrs[4] = rows[0];
        ptrs[5] = rows[0];

        dav1d_wiener_filter_h_8bpc_rvv(rows[0], left, src, fh, w, edges);
        left++;
        src += stride;

        if (--h <= 0)
            goto v1;

        ptrs[4] = ptrs[5] = rows[1];
        dav1d_wiener_filter_h_8bpc_rvv(rows[1], left, src, fh, w, edges);
        left++;
        src += stride;

        if (--h <= 0)
            goto v2;

        ptrs[5] = rows[2];
        dav1d_wiener_filter_h_8bpc_rvv(rows[2], left, src, fh, w, edges);
        left++;
        src += stride;

        if (--h <= 0)
            goto v3;

        ptrs[6] = rows[3];
        rvv_hv(p, ptrs, left, src, filter, w, edges);
        left++;
        src += stride;
        p += stride;

        if (--h <= 0)
            goto v3;

        ptrs[6] = rows[4];
        rvv_hv(p, ptrs, left, src, filter, w, edges);
        left++;
        src += stride;
        p += stride;

        if (--h <= 0)
            goto v3;
    }

    ptrs[6] = ptrs[5] + REST_UNIT_STRIDE;
    do {
        rvv_hv(p, ptrs, left, src, filter, w, edges);
        left++;
        src += stride;
        p += stride;
    } while (--h > 0);

    if (!(edges & LR_HAVE_BOTTOM))
        goto v3;

    rvv_hv(p, ptrs, NULL, lpf_bottom, filter, w, edges);
    lpf_bottom += stride;
    p += stride;

    rvv_hv(p, ptrs, NULL, lpf_bottom, filter, w, edges);
    p += stride;
v1:
    rvv_v(p, ptrs, fv, w);
    return;

v3:
    rvv_v(p, ptrs, fv, w);
    p += stride;
v2:
    rvv_v(p, ptrs, fv, w);
    p += stride;
    goto v1;
}

/* ---------------- test scaffolding --------------------------------------- */

static int failures = 0;

/* widths: minimum, small (C's w<6 path), power-of-2 and alignment edges,
 * checkasm's max widths; heights include the h=1..4 startup paths */
static const int widths[] = { 1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16, 17, 31, 32,
                              63, 64, 65, 127, 128, 129, 191, 192, 255, 256,
                              257, 383, 384 };
static const int heights[] = { 1, 2, 3, 4, 5, 6, 7, 8, 16, 63, 64 };

/* tap extremes allowed by the bitstream (see the C test harness ranges):
 * f0 in [-5,10], f1 in [-23,8], f2 in [-17,46], f3 = 128 - 2*(f0+f1+f2) */
static const int tap_sets[][3] = {
    {  10,   8,  46 },   /* all max positive: f3 = -128 */
    {  -5, -23, -17 },   /* all min negative: f3 = +90  */
    {  -5,  -5,  -5 },   /* mild negative */
    {  10, -23,  46 },   /* mixed, max |f1| */
    {  -5,   8, -17 },   /* mixed */
    {   0,   0,   0 },   /* identity-ish: f3 = 128 */
    {   7,  15,  63 },   /* not bitstream-legal but exercises clip: f3 = -60 */
};
static const int n_tap_sets = (int)(sizeof(tap_sets) / sizeof(tap_sets[0]));

static void set_taps(int16_t fh[8], int16_t fv[8], const int t3[3], int five_tap)
{
    for (int pass = 0; pass < 2; pass++) {
        int16_t *f = pass ? fv : fh;
        f[0] = f[6] = five_tap ? 0 : t3[0];
        f[1] = f[5] = t3[1];
        f[2] = f[4] = t3[2];
        if (pass)
            f[3] = 128 - (f[0] + f[1] + f[2]) * 2;
        else
            f[3] = -(f[0] + f[1] + f[2]) * 2;   /* 8bpc horizontal */
    }
}

static void fill_pixels(pixel *buf, int n, int mode)
{
    for (int i = 0; i < n; i++) {
        switch (mode) {
        case 0: buf[i] = (pixel)rnd(); break;
        case 1: buf[i] = 0; break;
        case 2: buf[i] = 255; break;
        case 3: buf[i] = (pixel)((i & 8) ? 255 : 0); break;
        case 4: buf[i] = (pixel)((rnd() & 15) ? rnd() : (rnd() & 1) * 255); break;
        }
    }
}

/* ---- level 1: horizontal kernel ---- */
static void test_h(void)
{
    static pixel src[64 + 384 + 64];
    static pixel left[64][4];
    uint16_t ref[384], got[384];

    for (int ti = 0; ti < n_tap_sets; ti++) {
        for (int five = 0; five < 2; five++) {
            int16_t fh[8], fv[8];
            set_taps(fh, fv, tap_sets[ti], five);
            for (unsigned e = 0; e < 4; e++) {
                for (int use_left = 0; use_left < 2; use_left++) {
                    for (int wi = 0; wi < (int)(sizeof(widths)/sizeof(widths[0])); wi++) {
                        int w = widths[wi];
                        if ((e & LR_HAVE_RIGHT) && w > 256) continue;
                        int mode = rnd() & 7;
                        if (mode > 4) mode = 0;
                        fill_pixels(src + 32, 384 + 16, mode);
                        fill_pixels(&left[0][0], 4, rnd() & 1 ? 0 : 2);
                        const pixel (*lp)[4] = use_left ? left : NULL;
                        memset(ref, 0xAA, sizeof(ref));
                        memset(got, 0xAA, sizeof(got));
                        ref_h(ref, lp, src + 32, fh, w, e);
                        dav1d_wiener_filter_h_8bpc_rvv(got, lp, src + 32, fh, w, e);
                        if (memcmp(ref, got, sizeof(uint16_t) * w)) {
                            printf("H-FAIL taps=%d five=%d edges=%u left=%d w=%d\n",
                                   ti, five, e, use_left, w);
                            for (int x = 0; x < w; x++)
                                if (ref[x] != got[x]) {
                                    printf("  x=%d ref=%u got=%u\n", x, ref[x], got[x]);
                                    break;
                                }
                            failures++;
                            if (failures > 20) exit(1);
                            goto next;
                        }
                    }
                }
            }
next: ;
        }
    }
    printf("level 1 (h kernel): %s\n", failures ? "FAILED" : "ok");
}

/* ---- level 2: vertical kernel ---- */
static void test_v(void)
{
    static uint16_t rowsbuf[8][384];
    uint16_t *ptrs[6];
    static pixel refout[8][384], gotout[8][384];

    for (int ti = 0; ti < n_tap_sets; ti++) {
        for (int five = 0; five < 2; five++) {
            int16_t fh[8], fv[8];
            set_taps(fh, fv, tap_sets[ti], five);
            for (int wi = 0; wi < (int)(sizeof(widths)/sizeof(widths[0])); wi++) {
                int w = widths[wi];
                if (w > 384) continue;
                /* intermediates: random in [0,8191], plus the clip boundaries */
                for (int r = 0; r < 8; r++)
                    for (int x = 0; x < 384; x++) {
                        unsigned m = rnd() & 15;
                        rowsbuf[r][x] = (uint16_t)(m == 0 ? 0 : m == 1 ? 8191 :
                                                   m == 2 ? 2047 : m == 3 ? 2048 :
                                                   rnd() & 8191);
                    }
                uint16_t *rp[6] = { rowsbuf[0], rowsbuf[1], rowsbuf[2],
                                    rowsbuf[3], rowsbuf[4], rowsbuf[5] };
                for (int k = 0; k < 6; k++) ptrs[k] = rp[k];
                memset(refout, 0xAA, sizeof(refout));
                memset(gotout, 0xAA, sizeof(gotout));

                /* reference: ref_v twice (rows advance by 5, but we only
                 * produce one output row per call here) */
                for (int k = 0; k < 6; k++) ptrs[k] = rp[k];
                ref_v(refout[0], ptrs, fv, w);
                /* driver under test */
                for (int k = 0; k < 6; k++) ptrs[k] = rp[k];
                rvv_v(gotout[0], ptrs, fv, w);

                if (memcmp(refout[0], gotout[0], w)) {
                    printf("V-FAIL taps=%d five=%d w=%d\n", ti, five, w);
                    for (int x = 0; x < w; x++)
                        if (refout[0][x] != gotout[0][x]) {
                            printf("  x=%d ref=%d got=%d\n", x, refout[0][x], gotout[0][x]);
                            break;
                        }
                    failures++;
                    if (failures > 20) exit(1);
                }
            }
        }
    }
    printf("level 2 (v kernel): %s\n", failures ? "FAILED" : "ok");
}

/* ---- level 3: whole block ---- */
static void test_block(void)
{
    /* padded canvas: 3 px left margin for LR_HAVE_LEFT-without-left reads,
     * 3 px right overread with LR_HAVE_RIGHT, as checkasm initializes */
    enum { MAXW = 384, MAXH = 64 };
    static pixel canvas[(MAXH + 8) * (MAXW + 8)];
    static pixel lpfc[8 * (MAXW + 8)];
    static pixel left[MAXH][4];
    static pixel refb[(MAXH + 8) * (MAXW + 8)];
    static pixel gotb[(MAXH + 8) * (MAXW + 8)];

    for (int ti = 0; ti < n_tap_sets; ti++) {
        for (int five = 0; five < 2; five++) {
            for (int hc = 0; hc < 2; hc++) {   /* fh from set, fv random-ish */
                LRParams p;
                int16_t fv2[8];
                set_taps(p.filter[0], fv2, tap_sets[ti], five);
                if (hc) {
                    int16_t fv3[8];
                    set_taps(fv3, p.filter[1], tap_sets[(ti + 3) % n_tap_sets], five ^ 1);
                } else {
                    for (int k = 0; k < 8; k++) p.filter[1][k] = fv2[k];
                }
                for (unsigned e = 0; e < 16; e++) {
                    for (int wi = 0; wi < (int)(sizeof(widths)/sizeof(widths[0])); wi++) {
                        int w = widths[wi];
                        if ((e & LR_HAVE_RIGHT) && w > 256) continue;
                        for (int hi = 0; hi < (int)(sizeof(heights)/sizeof(heights[0])); hi++) {
                            int h = heights[hi];
                            if ((e & LR_HAVE_BOTTOM) && h < 64 && (rnd() & 3)) continue;
                            /* corners exhaustive, sampled middle: keeps the
                             * QEMU runtime bounded */
                            if ((rnd() & 3) && w > 8 && w != 255 && w != 256 &&
                                w != 384 && h != 1 && h != 2 && h != 64) continue;
                            int mode = rnd() & 7;
                            if (mode > 4) mode = 0;
                            fill_pixels(canvas, sizeof(canvas), mode);
                            fill_pixels(lpfc, sizeof(lpfc), mode);
                            fill_pixels(&left[0][0], sizeof(left), mode);
                            /* the driver may read src[-3..-1] relative to p
                             * when LR_HAVE_LEFT is set without a left buffer:
                             * ensure canvas has meaningful data there (it
                             * does: 3px margin per row) */
                            ptrdiff_t stride = MAXW + 8;
                            pixel *p0 = canvas + 3;      /* p = canvas + 3 */
                            memcpy(refb, canvas, sizeof(canvas));
                            memcpy(gotb, canvas, sizeof(canvas));
                            pixel *rp = refb + 3, *gp = gotb + 3;
                            ref_wiener(rp, stride, left, lpfc, w, h, &p, e);
                            rvv_wiener(gp, stride, left, lpfc, w, h, &p, e);
                            if (memcmp(refb, gotb, sizeof(canvas))) {
                                printf("BLOCK-FAIL taps=%d five=%d fvmix=%d "
                                       "edges=%u w=%d h=%d\n",
                                       ti, five, hc, e, w, h);
                                for (int y = 0; y < h; y++)
                                    for (int x = 0; x < w; x++)
                                        if (refb[3 + y * stride + x] !=
                                            gotb[3 + y * stride + x]) {
                                            printf("  first diff y=%d x=%d ref=%d got=%d\n",
                                                   y, x, refb[3 + y*stride + x],
                                                   gotb[3 + y*stride + x]);
                                            goto done;
                                        }
done:
                                failures++;
                                if (failures > 20) exit(1);
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
    printf("level 3 (whole block): %s\n", failures ? "FAILED" : "ok");
}

/* ---- level 4: checkasm mirror (exact geometry + content model) ---------- */

static void init_tmp_mirror(pixel *buf, ptrdiff_t stride, int w, int h)
{
    int x_off = rnd() & 7, y_off = rnd() & 7;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++)
            buf[x] = (((x + x_off) ^ (y + y_off)) & 8 ? 255 : 0) ^ (rnd() & 15);
        buf += stride;
    }
}

static void test_checkasm(void)
{
    /* PIXEL_RECT(name, 384, 64): buf_w = 448, buf_h = 96,
     * p = buf + stride*16 + 64 (64-byte aligned) */
    static CHECKASM_ALIGN_PIXEL pixel cbuf[96 * 448 + 64];
    static CHECKASM_ALIGN_PIXEL pixel abuf[96 * 448 + 64];
    static pixel hbuf[40 * 448 + 64] __attribute__((aligned(64)));
    pixel *c_dst = cbuf + 448 * 16 + 64;
    pixel *a_dst = abuf + 448 * 16 + 64;
    pixel *h_edge = hbuf + 448 * 16 + 64;
    pixel left[64][4];

    for (int iter = 0; iter < 3000; iter++) {
        LRParams p;
        int five = iter & 1;
        /* checkasm's legal ranges */
        int f0 = (int)(rnd() & 15) - 5, f1 = (int)(rnd() & 31) - 23,
            f2 = (int)(rnd() & 63) - 17;
        p.filter[0][0] = p.filter[0][6] = five ? 0 : f0;
        p.filter[0][1] = p.filter[0][5] = f1;
        p.filter[0][2] = p.filter[0][4] = f2;
        p.filter[0][3] = -(p.filter[0][0] + p.filter[0][1] + p.filter[0][2]) * 2;
        p.filter[1][0] = p.filter[1][6] = five ? 0 : (int)(rnd() & 15) - 5;
        p.filter[1][1] = p.filter[1][5] = (int)(rnd() & 31) - 23;
        p.filter[1][2] = p.filter[1][4] = (int)(rnd() & 63) - 17;
        p.filter[1][3] = 128 - (p.filter[1][0] + p.filter[1][1] +
                                p.filter[1][2]) * 2;

        int base_w = 1 + (int)(rnd() % 384);
        int base_h = 1 + (int)(rnd() & 63);

        memset(cbuf, 0, sizeof(cbuf));
        init_tmp_mirror(c_dst - 4, 448, 384 + 4, 64);
        init_tmp_mirror(h_edge - 4, 448, 384 + 4, 8);
        init_tmp_mirror((pixel *) left, 4 * (int)sizeof(pixel), 4, 64);

        for (unsigned e = 0; e < 16; e++) {
            int w = (e & LR_HAVE_RIGHT) ? 256 : base_w;
            int h = (e & LR_HAVE_BOTTOM) ? 64 : base_h;
            memcpy(abuf, cbuf, sizeof(cbuf));
            ref_wiener(c_dst, 448, left, h_edge, w, h, &p, e);
            rvv_wiener(a_dst, 448, left, h_edge, w, h, &p, e);
            /* compare region: round up w to the 64-element overwrite
             * allowance, +8 padding; h + 8 rows */
            int cw = ((w + 63) & ~63) + 8;
            int ch = h + 8;
            if (memcmp(c_dst, a_dst, 0)) {}
            for (int y = 0; y < ch; y++) {
                if (memcmp(c_dst + y * 448, a_dst + y * 448, cw)) {
                    printf("MIRROR-FAIL iter=%d five=%d edges=%u w=%d h=%d "
                           "row=%d\n", iter, five, e, w, h, y);
                    for (int x = 0; x < cw; x++)
                        if (c_dst[y * 448 + x] != a_dst[y * 448 + x]) {
                            printf("  x=%d ref=%d got=%d\n", x,
                                   c_dst[y * 448 + x], a_dst[y * 448 + x]);
                            break;
                        }
                    failures++;
                    if (failures > 10) exit(1);
                    goto next_iter;
                }
            }
        }
next_iter: ;
    }
    printf("level 4 (checkasm mirror): %s\n", failures ? "FAILED" : "ok");
}

/* ---- level 5: the library's own init (what checkasm actually calls) ----- */

typedef void (*lrfn)(pixel *, ptrdiff_t, const pixel (*)[4], const pixel *,
                     int, int, const void *, int);
extern void dav1d_loop_restoration_dsp_init_8bpc(void *c, int bpc);

static lrfn get_lib_wiener(void)
{
    lrfn fns[5];   /* Dav1dLoopRestorationDSPContext: wiener[2] + sgr[3] */
    dav1d_loop_restoration_dsp_init_8bpc(fns, 8);
    if (fns[0] != fns[1])
        printf("note: lib wiener[0] != wiener[1]\n");
    return fns[0];
}

static void test_lib(void)
{
    static CHECKASM_ALIGN_PIXEL pixel cbuf[96 * 448 + 64];
    static CHECKASM_ALIGN_PIXEL pixel abuf[96 * 448 + 64];
    static pixel hbuf[40 * 448 + 64] __attribute__((aligned(64)));
    pixel *c_dst = cbuf + 448 * 16 + 64;
    pixel *a_dst = abuf + 448 * 16 + 64;
    pixel *h_edge = hbuf + 448 * 16 + 64;
    pixel left[64][4];
    lrfn fn = get_lib_wiener();
    printf("lib fn = %p, local driver = %p\n", (void *)fn, (void *)rvv_wiener);
    if (fn == (lrfn)rvv_wiener) {
        printf("level 5: lib init resolved to the LOCAL driver - test void\n");
        return;
    }

    for (int iter = 0; iter < 3000; iter++) {
        LRParams p;
        int five = iter & 1;
        int f0 = (int)(rnd() & 15) - 5, f1 = (int)(rnd() & 31) - 23,
            f2 = (int)(rnd() & 63) - 17;
        p.filter[0][0] = p.filter[0][6] = five ? 0 : f0;
        p.filter[0][1] = p.filter[0][5] = f1;
        p.filter[0][2] = p.filter[0][4] = f2;
        p.filter[0][3] = -(p.filter[0][0] + p.filter[0][1] + p.filter[0][2]) * 2;
        p.filter[1][0] = p.filter[1][6] = five ? 0 : (int)(rnd() & 15) - 5;
        p.filter[1][1] = p.filter[1][5] = (int)(rnd() & 31) - 23;
        p.filter[1][2] = p.filter[1][4] = (int)(rnd() & 63) - 17;
        p.filter[1][3] = 128 - (p.filter[1][0] + p.filter[1][1] +
                                p.filter[1][2]) * 2;

        int base_w = 1 + (int)(rnd() % 384);
        int base_h = 1 + (int)(rnd() & 63);

        memset(cbuf, 0, sizeof(cbuf));
        init_tmp_mirror(c_dst - 4, 448, 384 + 4, 64);
        init_tmp_mirror(h_edge - 4, 448, 384 + 4, 8);
        init_tmp_mirror((pixel *) left, 4 * (int)sizeof(pixel), 4, 64);

        for (unsigned e = 0; e < 16; e++) {
            int w = (e & LR_HAVE_RIGHT) ? 256 : base_w;
            int h = (e & LR_HAVE_BOTTOM) ? 64 : base_h;
            memcpy(abuf, cbuf, sizeof(cbuf));
            ref_wiener(c_dst, 448, left, h_edge, w, h, &p, e);
            fn(a_dst, 448, left, h_edge, w, h, &p, e);
            int cw = ((w + 63) & ~63) + 8;
            int ch = h + 8;
            for (int y = 0; y < ch; y++) {
                if (memcmp(c_dst + y * 448, a_dst + y * 448, cw)) {
                    printf("LIB-FAIL iter=%d five=%d edges=%u w=%d h=%d row=%d\n",
                           iter, five, e, w, h, y);
                    for (int x = 0; x < cw; x++)
                        if (c_dst[y * 448 + x] != a_dst[y * 448 + x]) {
                            printf("  x=%d ref=%d got=%d\n", x,
                                   c_dst[y * 448 + x], a_dst[y * 448 + x]);
                            break;
                        }
                    failures++;
                    if (failures > 10) exit(1);
                    goto next_iter;
                }
            }
        }
next_iter: ;
    }
    printf("level 5 (lib driver): %s\n", failures ? "FAILED" : "ok");
}

int main(void)
{
    test_h();
    test_v();
    test_block();
    test_checkasm();
    test_lib();
    printf("%s (%d failures)\n", failures ? "DIFF FAILED" : "ALL OK", failures);
    return failures != 0;
}
