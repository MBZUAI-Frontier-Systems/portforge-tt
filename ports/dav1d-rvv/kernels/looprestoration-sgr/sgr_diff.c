/* Differential driver for the 8bpc SGR RVV port (campaign c3-sgr).
 *
 * Levels, all bit-exact:
 *   1. box3_row_h kernel  vs the transcribed C (sgr_box3_row_h, 8bpc)
 *   2. box3_row_v kernel  vs the transcribed C (sgr_box3_row_v)
 *   3. calc_ab1 kernel    vs the transcribed C (sgr_calc_row_ab, n=9, obx=455)
 *   4. finish_row1 kernel vs the transcribed C (sgr_finish_filter_row1)
 *   5. weighted_row1      vs the transcribed C (sgr_weighted_row1)
 *   6. whole block: the header's sgr_3x3_rvv driver vs the LIBRARY's pure-C
 *      sgr_3x3_8bpc_c (dav1d_loop_restoration_dsp_init_8bpc resolves to C
 *      under QEMU user-mode: no V bit in AT_HWCAP).  This is the level that
 *      catches driver transcription bugs against the real C (the c2-wiener
 *      lesson: build it first).
 *
 * Build (toolchain image):
 *   riscv64-linux-gnu-gcc-14 -march=rv64gcv -O1 -static \
 *       -I<t>/src -I<t>/include -I<b> -DBITDEPTH=8 sgr_diff.c \
 *       <t>/src/riscv/64/looprestoration.S.o -o sgr_diff \
 *       -L<pf>/build/dav1d-c3-sgr-rv64/src -ldav1d
 * Run: qemu-riscv64 -L /usr/riscv64-linux-gnu -cpu rv64,v=true,... ./sgr_diff
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define BITDEPTH 8
#include "src/riscv/looprestoration.h"

/* ---------------- the asm kernels under test ---------------- */

/* declared by the included header; short local aliases */
#define K_BOX3H BF(dav1d_sgr_box3_row_h, rvv)
#define K_BOX3V BF(dav1d_sgr_box3_row_v, rvv)
#define K_CALC1 BF(dav1d_sgr_calc_ab1, rvv)
#define K_FIN1  BF(dav1d_sgr_finish_filter_row1, rvv)
#define K_WROW1 BF(dav1d_sgr_weighted_row1, rvv)
#define K_BOX5H BF(dav1d_sgr_box5_row_h, rvv)
#define K_BOX5V BF(dav1d_sgr_box5_row_v, rvv)
#define K_CALC2 BF(dav1d_sgr_calc_ab2, rvv)
#define K_FIN2  BF(dav1d_sgr_finish_filter2, rvv)

/* ---------------- reference transcriptions (8bpc constants) ---------------- */

static int iclip(int x, int min, int max) {
    return x < min ? min : x > max ? max : x;
}
static int imax__(const int a, const int b) { return a > b ? a : b; }
static unsigned umin__(const unsigned a, const unsigned b) { return a <= b ? a : b; }

#define LR_HAVE_LEFT   (1 << 0)
#define LR_HAVE_RIGHT  (1 << 1)
#define LR_HAVE_TOP    (1 << 2)
#define LR_HAVE_BOTTOM (1 << 3)

static void ref_box3_row_h(int32_t *sumsq, int16_t *sum,
                           const uint8_t (*left)[4], const uint8_t *src,
                           const int w, const int edges)
{
    sumsq++;
    sum++;
    int a = edges & LR_HAVE_LEFT ? (left ? left[0][2] : src[-2]) : src[0];
    int b = edges & LR_HAVE_LEFT ? (left ? left[0][3] : src[-1]) : src[0];
    for (int x = -1; x < w + 1; x++) {
        int c = (x + 1 < w || (edges & LR_HAVE_RIGHT)) ? src[x + 1] : src[w - 1];
        sum[x] = a + b + c;
        sumsq[x] = a * a + b * b + c * c;
        a = b;
        b = c;
    }
}

static void ref_box3_row_v(int32_t **sumsq, int16_t **sum,
                           int32_t *sumsq_out, int16_t *sum_out, const int w)
{
    for (int x = 0; x < w + 2; x++) {
        sumsq_out[x] = sumsq[0][x] + sumsq[1][x] + sumsq[2][x];
        sum_out[x] = sum[0][x] + sum[1][x] + sum[2][x];
    }
}

static void ref_calc_ab1(int32_t *AA, int16_t *BB, const int w, const unsigned s)
{
    const int n = 9, one_by_x = 455;
    for (int i = 0; i < w + 2; i++) {
        const int a = AA[i];            /* 8bpc: no pre-shift */
        const int b = BB[i];
        const unsigned p = imax__(a * n - b * b, 0);
        const unsigned z = (p * s + (1 << 19)) >> 20;
        const unsigned x = dav1d_sgr_x_by_x[umin__(z, 255)];
        AA[i] = (x * BB[i] * one_by_x + (1 << 11)) >> 12;
        BB[i] = x;
    }
}
static void ref_finish_row1(int16_t *tmp, const uint8_t *src,
                            int32_t **A_ptrs, int16_t **B_ptrs, const int w)
{
#define EIGHT_NEIGHBORS(P, i)\
    ((P[1][i] + P[1][i - 1] + P[1][i + 1] + P[0][i] + P[2][i]) * 4 + \
     (P[0][i - 1] + P[2][i - 1] +                           \
      P[0][i + 1] + P[2][i + 1]) * 3)
    for (int i = 0; i < w; i++) {
        const int a = EIGHT_NEIGHBORS(B_ptrs, i + 1);
        const int b = EIGHT_NEIGHBORS(A_ptrs, i + 1);
        tmp[i] = (b - a * src[i] + (1 << 8)) >> 9;
    }
#undef EIGHT_NEIGHBORS
}

static void ref_weighted_row1(uint8_t *dst, const int16_t *t1,
                              const int w, const int w1)
{
    for (int i = 0; i < w; i++) {
        const int v = w1 * t1[i];
        dst[i] = iclip(dst[i] + ((v + (1 << 10)) >> 11), 0, 255);
    }
}


static void ref_box5_row_h(int32_t *sumsq, int16_t *sum,
                           const uint8_t (*left)[4], const uint8_t *src,
                           const int w, const int edges)
{
    sumsq++;
    sum++;
    int a = edges & LR_HAVE_LEFT ? (left ? left[0][1] : src[-3]) : src[0];
    int b = edges & LR_HAVE_LEFT ? (left ? left[0][2] : src[-2]) : src[0];
    int c = edges & LR_HAVE_LEFT ? (left ? left[0][3] : src[-1]) : src[0];
    int d = src[0];
    for (int x = -1; x < w + 1; x++) {
        int e = (x + 2 < w || (edges & LR_HAVE_RIGHT)) ? src[x + 2] : src[w - 1];
        sum[x] = a + b + c + d + e;
        sumsq[x] = a * a + b * b + c * c + d * d + e * e;
        a = b;
        b = c;
        c = d;
        d = e;
    }
}

static void ref_box5_row_v(int32_t **sumsq, int16_t **sum,
                           int32_t *sumsq_out, int16_t *sum_out, const int w)
{
    for (int x = 0; x < w + 2; x++) {
        sumsq_out[x] = sumsq[0][x] + sumsq[1][x] + sumsq[2][x] +
                       sumsq[3][x] + sumsq[4][x];
        sum_out[x] = sum[0][x] + sum[1][x] + sum[2][x] + sum[3][x] + sum[4][x];
    }
}

static void ref_calc_ab(int32_t *AA, int16_t *BB, const int w, const unsigned s,
                        const int n, const int one_by_x)
{
    for (int i = 0; i < w + 2; i++) {
        const int a = AA[i];
        const int b = BB[i];
        const unsigned p = imax__(a * n - b * b, 0);
        const unsigned z = (p * s + (1 << 19)) >> 20;
        const unsigned x = dav1d_sgr_x_by_x[umin__(z, 255)];
        AA[i] = (x * BB[i] * one_by_x + (1 << 11)) >> 12;
        BB[i] = x;
    }
}

static void ref_finish_filter2(int16_t *tmp, const uint8_t *src,
                               const long src_stride,
                               int32_t **A_ptrs, int16_t **B_ptrs,
                               const int w, const int h)
{
#define SIX_NEIGHBORS(P, i)    ((P[0][i]     + P[1][i]) * 6 +   \
     (P[0][i - 1] + P[1][i - 1] +    \
      P[0][i + 1] + P[1][i + 1]) * 5)
    for (int i = 0; i < w; i++) {
        const int a = SIX_NEIGHBORS(B_ptrs, i + 1);
        const int b = SIX_NEIGHBORS(A_ptrs, i + 1);
        tmp[i] = (b - a * src[i] + (1 << 8)) >> 9;
    }
    if (h <= 1)
        return;
    tmp += 384;
    src += src_stride;
    const int32_t *A = &A_ptrs[1][1];
    const int16_t *B = &B_ptrs[1][1];
    for (int i = 0; i < w; i++) {
        const int a = B[i] * 6 + (B[i - 1] + B[i + 1]) * 5;
        const int b = A[i] * 6 + (A[i - 1] + A[i + 1]) * 5;
        tmp[i] = (b - a * src[i] + (1 << 7)) >> 8;
    }
#undef SIX_NEIGHBORS
}

/* ---------------- harness ---------------- */

static unsigned long long rng_state = 0x53475233ULL;
static unsigned rnd(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned)(rng_state >> 33);
}

#define CHECKASM_ALIGN __attribute__((aligned(64)))

static int failures = 0;

static void dump_row(const char *what, const int16_t *a, const int16_t *b, int n) {
    fprintf(stderr, "%s mismatch:\n", what);
    for (int i = 0; i < n; i++)
        if (a[i] != b[i]) { fprintf(stderr, "  [%d]: ref=%d asm=%d\n", i, a[i], b[i]); if (i > 12) break; }
}

int main(int argc, char **argv) {
    int level = argc > 1 ? atoi(argv[1]) : 0;
    const uint8_t *x_by_x = dav1d_sgr_x_by_x;

    uint8_t lut[256];
    memcpy(lut, x_by_x, 256);

    if (!level || level == 1) {
        /* level 1: box3_row_h — widths 1..384 incl. alignment edges, all
         * edge combos, left buffer or NULL, random and extreme content */
        static uint8_t CHECKASM_ALIGN srcbuf[448 + 16];
        static uint8_t leftbuf[64][4];
        static int32_t CHECKASM_ALIGN sq_a[512], sq_b[512];
        static int16_t CHECKASM_ALIGN sm_a[512], sm_b[512];
        int bad = 0;
        for (int w = 1; w <= 384; w++) {
            for (int edges = 0; edges < 16; edges++) {
                for (int mode = 0; mode < 3; mode++) {
                    for (int i = 0; i < w + 8; i++) {
                        switch (mode) {
                        case 0: srcbuf[i] = rnd(); break;
                        case 1: srcbuf[i] = (i & 1) * 255; break;
                        default: srcbuf[i] = 255; break;
                        }
                    }
                    for (int i = 0; i < 64 * 4; i++) ((uint8_t *) leftbuf)[i] = rnd();
                    const uint8_t (*left)[4] = (rnd() & 1) ? leftbuf : NULL;
                    memset(sq_a, 0xAA, sizeof sq_a); memset(sq_b, 0xAA, sizeof sq_b);
                    memset(sm_a, 0x5A, sizeof sm_a); memset(sm_b, 0x5A, sizeof sm_b);
                    ref_box3_row_h(sq_a, sm_a, left, srcbuf, w, edges);
                    K_BOX3H(sq_b, sm_b, left, srcbuf, w, edges);
                    for (int i = 0; i < w + 2; i++) {
                        if (sq_a[i] != sq_b[i] || sm_a[i] != sm_b[i]) {
                            fprintf(stderr, "level 1 FAIL w=%d edges=%d mode=%d left=%d [%d]: "
                                    "sq %d vs %d, sm %d vs %d\n", w, edges, mode, !!left,
                                    i, sq_a[i], sq_b[i], sm_a[i], sm_b[i]);
                            bad = failures = 1;
                            goto L1;
                        }
                    }
                }
            }
        }
L1:
        if (!bad) printf("level 1: box3_row_h ok (384 widths x 16 edges x 3 modes)\n");
    }

    if (!level || level == 2) {
        static int32_t CHECKASM_ALIGN r32[3][512], out32a[512], out32b[512];
        static int16_t CHECKASM_ALIGN r16[3][512], out16a[512], out16b[512];
        int bad = 0;
        for (int w = 1; w <= 384; w++) {
            for (int t = 0; t < 4; t++) {
                for (int k = 0; k < 3; k++)
                    for (int i = 0; i < w + 2; i++) {
                        r32[k][i] = t < 2 ? (int32_t)rnd() % 195075 : (t == 2 ? 195075 : 0);
                        r16[k][i] = t < 2 ? (int16_t)(rnd() % 765) : (t == 2 ? 765 : 0);
                    }
                int32_t *p32[3] = { r32[0], r32[1], r32[2] };
                int16_t *p16[3] = { r16[0], r16[1], r16[2] };
                ref_box3_row_v(p32, p16, out32a, out16a, w);
                K_BOX3V(p32, p16, out32b, out16b, w);
                for (int i = 0; i < w + 2; i++)
                    if (out32a[i] != out32b[i] || out16a[i] != out16b[i]) {
                        fprintf(stderr, "level 2 FAIL w=%d t=%d [%d]: %d/%d vs %d/%d\n",
                                w, t, i, out32a[i], out16a[i], out32b[i], out16b[i]);
                        bad = failures = 1;
                        goto L2;
                    }
            }
        }
L2:
        if (!bad) printf("level 2: box3_row_v ok\n");
    }

    if (!level || level == 3) {
        /* calc_ab1: random A/B plus forced boundaries, all 16 s rows */
        static const uint16_t sgr_params[16][2] = {
            { 140, 3236 }, { 112, 2158 }, {  93, 1618 }, {  80, 1438 },
            {  70, 1295 }, {  58, 1177 }, {  47, 1079 }, {  37,  996 },
            {  30,  925 }, {  25,  863 }, {   0, 2589 }, {   0, 1618 },
            {   0, 1177 }, {   0,  925 }, {  56,    0 }, {  22,    0 },
        };
        static int32_t CHECKASM_ALIGN Aa[512], Ab[512];
        static int16_t CHECKASM_ALIGN Ba[512], Bb[512];
        int bad = 0;
        for (int w = 1; w <= 384; w += 7) {
            for (int row = 0; row < 16; row++) {
                for (int t = 0; t < 5; t++) {
                    for (int i = 0; i < w + 2; i++) {
                        switch (t) {
                        case 0: Aa[i] = rnd() % 585226; Ba[i] = rnd() % 2296; break;
                        case 1: Aa[i] = 585225; Ba[i] = 2295; break;   /* max */
                        case 2: Aa[i] = 0; Ba[i] = 0; break;           /* min */
                        case 3: Aa[i] = 585225; Ba[i] = 0; break;      /* p max */
                        default: Aa[i] = 1; Ba[i] = 2295; break;       /* p = 0 edge */
                        }
                    }
                    memcpy(Ab, Aa, sizeof(int32_t) * (w + 2));
                    memcpy(Bb, Ba, sizeof(int16_t) * (w + 2));
                    ref_calc_ab1(Aa, Ba, w, sgr_params[row][1]);
                    K_CALC1(Ab, Bb, w, sgr_params[row][1], lut);
                    for (int i = 0; i < w + 2; i++)
                        if (Aa[i] != Ab[i] || Ba[i] != Bb[i]) {
                            fprintf(stderr, "level 3 FAIL w=%d row=%d t=%d [%d]: "
                                    "A %d vs %d, B %d vs %d\n", w, row, t, i,
                                    Aa[i], Ab[i], Ba[i], Bb[i]);
                            bad = failures = 1;
                            goto L3;
                        }
                }
            }
        }
L3:
        if (!bad) printf("level 3: calc_ab1 ok (incl. p*s 32-bit wrap)\n");
    }

    if (!level || level == 4) {
        static int32_t CHECKASM_ALIGN Arow[3][512];
        static int16_t CHECKASM_ALIGN Brow[3][512], tmpa[512], tmpb[512];
        static uint8_t CHECKASM_ALIGN srcbuf[512];
        int bad = 0;
        for (int w = 1; w <= 384; w += 3) {
            for (int t = 0; t < 4; t++) {
                for (int k = 0; k < 3; k++)
                    for (int i = 0; i < w + 2; i++) {
                        /* A values near the calc_ab1 output ceiling, B <= 255 */
                        Arow[k][i] = t == 0 ? rnd() % 65019 : (t == 1 ? 65018 : (t == 2 ? 0 : 255));
                        Brow[k][i] = t == 0 ? rnd() % 256 : (t == 1 ? 255 : (t == 2 ? 0 : 3));
                    }
                for (int i = 0; i < w; i++) srcbuf[i] = t == 3 ? 255 : rnd();
                int32_t *A[3] = { Arow[0], Arow[1], Arow[2] };
                int16_t *B[3] = { Brow[0], Brow[1], Brow[2] };
                ref_finish_row1(tmpa, srcbuf, A, B, w);
                K_FIN1(tmpb, srcbuf, A, B, w);
                for (int i = 0; i < w; i++)
                    if (tmpa[i] != tmpb[i]) {
                        fprintf(stderr, "level 4 FAIL w=%d t=%d [%d]: %d vs %d\n",
                                w, t, i, tmpa[i], tmpb[i]);
                        for (int k = 0; k < 3; k++) {
                            fprintf(stderr, "  B%d:", k);
                            for (int j = 0; j < w + 2; j++) fprintf(stderr, " %d", Brow[k][j]);
                            fprintf(stderr, "\n  A%d:", k);
                            for (int j = 0; j < w + 2; j++) fprintf(stderr, " %d", Arow[k][j]);
                            fprintf(stderr, "\n");
                        }
                        fprintf(stderr, "  src:");
                        for (int j = 0; j < w; j++) fprintf(stderr, " %d", srcbuf[j]);
                        fprintf(stderr, "\n");
                        bad = failures = 1;
                        goto L4;
                    }
            }
        }
L4:
        if (!bad) printf("level 4: finish_filter_row1 ok\n");
    }

    if (!level || level == 5) {
        static int16_t CHECKASM_ALIGN t1a[512], t1b[512];
        static uint8_t CHECKASM_ALIGN dsta[512], dstb[512];
        int bad = 0;
        for (int w = 1; w <= 384; w += 5) {
            for (int t = 0; t < 4; t++) {
                for (int i = 0; i < w; i++) {
                    t1a[i] = t1b[i] = t == 0 ? (int16_t)rnd() : (t == 1 ? 32767 :
                             (t == 2 ? -32768 : 0));
                    dsta[i] = dstb[i] = t == 0 ? rnd() : (t == 1 ? 255 : (t == 2 ? 0 : 128));
                }
                int w1 = t == 0 ? (int)(rnd() % 257) - 96 : (t == 1 ? 160 : (t == 2 ? -96 : 0));
                ref_weighted_row1(dsta, t1a, w, w1);
                K_WROW1(dstb, t1b, w, w1);
                for (int i = 0; i < w; i++)
                    if (dsta[i] != dstb[i]) {
                        fprintf(stderr, "level 5 FAIL w=%d t=%d w1=%d [%d]: %d vs %d\n",
                                w, t, w1, i, dsta[i], dstb[i]);
                        bad = failures = 1;
                        goto L5;
                    }
            }
        }
L5:
        if (!bad) printf("level 5: weighted_row1 ok\n");
    }

    if (!level || level == 6) {
        /* whole block vs the library's pure-C sgr_3x3_8bpc_c */
        Dav1dLoopRestorationDSPContext c;
        dav1d_loop_restoration_dsp_init_8bpc(&c, 8);
        if (c.sgr[1] == (looprestorationfilter_fn) sgr_3x3_rvv) {
            fprintf(stderr, "level 6: library init resolved to the RVV asm - "
                            "the differential would compare asm against itself\n");
            failures++;
        } else {
            /* checkasm-mirroring harness: 64-byte-aligned dst with padding,
             * h_edge 384x8, left 64x4, init_tmp content model */
#define BUF_STRIDE 448
            static uint8_t CHECKASM_ALIGN c_buf[BUF_STRIDE * 72], a_buf[BUF_STRIDE * 72];
            static uint8_t CHECKASM_ALIGN h_edge[BUF_STRIDE * 8];
            static uint8_t left[64][4];
            int bad = 0;
            for (int cfg_i = 0; cfg_i < 3000 && !bad; cfg_i++) {
                const unsigned s = (const unsigned[3]){ 2589, 3236, 863 }[cfg_i % 3];
                const int w1 = 160 - (int)(rnd() & 127);
                LooprestorationParams params;
                memset(&params, 0, sizeof params);
                params.sgr.s0 = 0;
                params.sgr.s1 = s;
                params.sgr.w0 = 0;
                params.sgr.w1 = w1;
                const int base_w = 1 + (int)(rnd() % 384);
                const int base_h = 1 + (int)(rnd() & 63);
                const int x_off = rnd() & 7, y_off = rnd() & 7;

                for (int y = 0; y < 64; y++)
                    for (int x = 0; x < 384 + 4; x++)
                        c_buf[y * BUF_STRIDE + x] =
                            (((x + x_off) ^ (y + y_off)) & 8 ? 255 : 0) ^ (rnd() & 15);
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 384 + 4; x++)
                        h_edge[y * BUF_STRIDE + x] =
                            (((x + x_off) ^ (y + y_off)) & 8 ? 255 : 0) ^ (rnd() & 15);
                for (int i = 0; i < 64 * 4; i++) ((uint8_t *) left)[i] = rnd();

                for (int edges = 0; edges < 16 && !bad; edges++) {
                    const int w = edges & LR_HAVE_RIGHT ? 256 : base_w;
                    const int h = edges & LR_HAVE_BOTTOM ? 64 : base_h;
                    memset(c_buf, 0, sizeof c_buf);
                    for (int y = 0; y < 64; y++)
                        for (int x = 0; x < 384 + 4; x++)
                            c_buf[y * BUF_STRIDE + x] =
                                (((x + x_off) ^ (y + y_off)) & 8 ? 255 : 0) ^ (rnd() & 15);
                    memcpy(a_buf, c_buf, sizeof a_buf);

                    c.sgr[1](c_buf, BUF_STRIDE, left, h_edge, w, h, &params, edges);
                    sgr_3x3_rvv(a_buf, BUF_STRIDE, left, h_edge, w, h, &params, edges);

                    for (int y = 0; y < h && !bad; y++)
                        for (int x = 0; x < w && !bad; x++)
                            if (c_buf[y * BUF_STRIDE + x] != a_buf[y * BUF_STRIDE + x]) {
                                fprintf(stderr, "level 6 FAIL cfg=%d s=%u w1=%d %dx%d edges=%d "
                                        "at (%d,%d): C=%d rvv=%d\n", cfg_i, s, w1, w, h,
                                        edges, x, y, c_buf[y * BUF_STRIDE + x],
                                        a_buf[y * BUF_STRIDE + x]);
                                bad = failures = 1;
                            }
                }
            }
            if (!bad) printf("level 6: sgr_3x3_rvv == sgr_3x3_8bpc_c (3000 configs x 16 edges)\n");
        }
    }

    if (!level || level == 7) {
        /* diagnostic: crafted inputs, back out the kernel's effective a and b
         * B0={10,20,30} B1={40,50,60} B2={70,80,90}
         * A0={100,200,300} A1={400,500,600} A2={700,800,900}
         * => a = 250*4 + 200*3 = 1600, b = 2500*4 + 2000*3 = 16000
         * src=0: tmp = (16000 + 256) >> 9 = 31    (b+256 = 16256 -> 31)
         * src=2: tmp = (16000 - 3200 + 256) >> 9 = 25
         */
        static int32_t CHECKASM_ALIGN A[3] = { 0 };  /* unused; rows below */
        static int32_t CHECKASM_ALIGN Arow2[3][8];
        static int16_t CHECKASM_ALIGN Brow2[3][8];
        static int16_t ta[8], tb[8];
        static uint8_t sb[8];
        int32_t *Ap[3] = { Arow2[0], Arow2[1], Arow2[2] };
        int16_t *Bp[3] = { Brow2[0], Brow2[1], Brow2[2] };
        int Bv[3][3] = { {10,20,30}, {40,50,60}, {70,80,90} };
        int Av[3][3] = { {100,200,300}, {400,500,600}, {700,800,900} };
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 3; j++) { Brow2[k][j] = Bv[k][j]; Arow2[k][j] = Av[k][j]; }
        for (int j = 0; j < 3; j++) sb[j] = 0;
        ref_finish_row1(ta, sb, Ap, Bp, 1);
        K_FIN1(tb, sb, Ap, Bp, 1);
        printf("level 7: src=0 -> ref tmp=%d asm tmp=%d (expect 31)\n", ta[0], tb[0]);
        for (int j = 0; j < 3; j++) sb[j] = 2;
        ref_finish_row1(ta, sb, Ap, Bp, 1);
        K_FIN1(tb, sb, Ap, Bp, 1);
        printf("level 7: src=2 -> ref tmp=%d asm tmp=%d (expect 25)\n", ta[0], tb[0]);
        (void)A;
    }

    if (!level || level == 8) {
        static uint8_t CHECKASM_ALIGN srcbuf[448 + 16];
        static uint8_t leftbuf[64][4];
        static int32_t CHECKASM_ALIGN sq_a[512], sq_b[512];
        static int16_t CHECKASM_ALIGN sm_a[512], sm_b[512];
        int bad = 0;
        for (int w = 1; w <= 384; w++) {
            for (int edges = 0; edges < 16; edges++) {
                for (int mode = 0; mode < 2; mode++) {
                    for (int i = 0; i < w + 8; i++)
                        srcbuf[i] = mode ? (i & 1) * 255 : rnd();
                    for (int i = 0; i < 64 * 4; i++) ((uint8_t *) leftbuf)[i] = rnd();
                    const uint8_t (*left)[4] = (rnd() & 1) ? leftbuf : NULL;
                    memset(sq_a, 0xAA, sizeof sq_a); memset(sq_b, 0xAA, sizeof sq_b);
                    memset(sm_a, 0x5A, sizeof sm_a); memset(sm_b, 0x5A, sizeof sm_b);
                    ref_box5_row_h(sq_a, sm_a, left, srcbuf, w, edges);
                    K_BOX5H(sq_b, sm_b, left, srcbuf, w, edges);
                    for (int i = 0; i < w + 2; i++)
                        if (sq_a[i] != sq_b[i] || sm_a[i] != sm_b[i]) {
                            fprintf(stderr, "level 8 FAIL w=%d edges=%d mode=%d left=%d [%d]: "
                                    "sq %d vs %d, sm %d vs %d\n", w, edges, mode, !!left,
                                    i, sq_a[i], sq_b[i], sm_a[i], sm_b[i]);
                            bad = failures = 1;
                            goto L8;
                        }
                }
            }
        }
L8:
        if (!bad) printf("level 8: box5_row_h ok\n");
    }

    if (!level || level == 9) {
        static int32_t CHECKASM_ALIGN r32[5][512], out32a[512], out32b[512];
        static int16_t CHECKASM_ALIGN r16[5][512], out16a[512], out16b[512];
        int bad = 0;
        for (int w = 1; w <= 384; w++) {
            for (int t = 0; t < 3; t++) {
                for (int k = 0; k < 5; k++)
                    for (int i = 0; i < w + 2; i++) {
                        r32[k][i] = t == 0 ? (int32_t)rnd() % 328050 : (t == 1 ? 328050 : 0);
                        r16[k][i] = t == 0 ? (int16_t)(rnd() % 1276) : (t == 1 ? 1275 : 0);
                    }
                int32_t *p32[5] = { r32[0], r32[1], r32[2], r32[3], r32[4] };
                int16_t *p16[5] = { r16[0], r16[1], r16[2], r16[3], r16[4] };
                ref_box5_row_v(p32, p16, out32a, out16a, w);
                K_BOX5V(p32, p16, out32b, out16b, w);
                for (int i = 0; i < w + 2; i++)
                    if (out32a[i] != out32b[i] || out16a[i] != out16b[i]) {
                        fprintf(stderr, "level 9 FAIL w=%d t=%d [%d]: %d/%d vs %d/%d\n",
                                w, t, i, out32a[i], out16a[i], out32b[i], out16b[i]);
                        bad = failures = 1;
                        goto L9;
                    }
            }
        }
L9:
        if (!bad) printf("level 9: box5_row_v ok\n");
    }

    if (!level || level == 10) {
        static const uint16_t sgr_params[16][2] = {
            { 140, 3236 }, { 112, 2158 }, {  93, 1618 }, {  80, 1438 },
            {  70, 1295 }, {  58, 1177 }, {  47, 1079 }, {  37,  996 },
            {  30,  925 }, {  25,  863 }, {   0, 2589 }, {   0, 1618 },
            {   0, 1177 }, {   0,  925 }, {  56,    0 }, {  22,    0 },
        };
        static int32_t CHECKASM_ALIGN Aa[512], Ab[512];
        static int16_t CHECKASM_ALIGN Ba[512], Bb[512];
        int bad = 0;
        for (int w = 1; w <= 384; w += 7) {
            for (int row = 0; row < 16; row++) {
                for (int t = 0; t < 5; t++) {
                    for (int i = 0; i < w + 2; i++) {
                        switch (t) {
                        case 0: Aa[i] = rnd() % 1640251; Ba[i] = rnd() % 6376; break;
                        case 1: Aa[i] = 1640250; Ba[i] = 6375; break;
                        case 2: Aa[i] = 0; Ba[i] = 0; break;
                        case 3: Aa[i] = 1640250; Ba[i] = 0; break;
                        default: Aa[i] = 1; Ba[i] = 6375; break;
                        }
                    }
                    memcpy(Ab, Aa, sizeof(int32_t) * (w + 2));
                    memcpy(Bb, Ba, sizeof(int16_t) * (w + 2));
                    ref_calc_ab(Aa, Ba, w, sgr_params[row][0], 25, 164);
                    K_CALC2(Ab, Bb, w, sgr_params[row][0], lut);
                    for (int i = 0; i < w + 2; i++)
                        if (Aa[i] != Ab[i] || Ba[i] != Bb[i]) {
                            fprintf(stderr, "level 10 FAIL w=%d row=%d t=%d [%d]: "
                                    "A %d vs %d, B %d vs %d\n", w, row, t, i,
                                    Aa[i], Ab[i], Ba[i], Bb[i]);
                            bad = failures = 1;
                            goto L10;
                        }
                }
            }
        }
L10:
        if (!bad) printf("level 10: calc_ab2 ok\n");
    }

    if (!level || level == 11) {
        static int32_t CHECKASM_ALIGN Arow[2][512];
        static int16_t CHECKASM_ALIGN Brow[2][512], tmpa[2 * 384], tmpb[2 * 384];
        static uint8_t CHECKASM_ALIGN srcbuf[512];
        int bad = 0;
        for (int w = 1; w <= 384; w += 3) {
            for (int t = 0; t < 3; t++) {
                for (int k = 0; k < 2; k++)
                    for (int i = 0; i < w + 2; i++) {
                        Arow[k][i] = t == 0 ? rnd() % 65019 : (t == 1 ? 65018 : 0);
                        Brow[k][i] = t == 0 ? rnd() % 256 : (t == 1 ? 255 : 0);
                    }
                for (int i = 0; i < w; i++) srcbuf[i] = rnd();
                int32_t *A[2] = { Arow[0], Arow[1] };
                int16_t *B[2] = { Brow[0], Brow[1] };
                for (int h = 1; h <= 2; h++) {
                    memset(tmpa, 0x5A, sizeof tmpa); memset(tmpb, 0x5A, sizeof tmpb);
                    ref_finish_filter2(tmpa, srcbuf, 448, A, B, w, h);
                    K_FIN2(tmpb, srcbuf, 448, A, B, w, h);
                    const int n = h == 1 ? w : 384 + w;
                    for (int i = 0; i < n; i++)
                        if (tmpa[i] != tmpb[i]) {
                            fprintf(stderr, "level 11 FAIL w=%d t=%d h=%d [%d]: %d vs %d\n",
                                    w, t, h, i, tmpa[i], tmpb[i]);
                            bad = failures = 1;
                            goto L11;
                        }
                }
            }
        }
L11:
        if (!bad) printf("level 11: finish_filter2 ok (h=1 and h=2)\n");
    }

    if (!level || level == 12) {
        /* whole sgr_5x5_rvv vs the library's pure-C sgr_5x5 (c.sgr[0]) */
        Dav1dLoopRestorationDSPContext c;
        dav1d_loop_restoration_dsp_init_8bpc(&c, 8);
        if (c.sgr[0] == (looprestorationfilter_fn) sgr_5x5_rvv) {
            fprintf(stderr, "level 12: library init resolved to the RVV asm\n");
            failures++;
        } else {
#define BUF_STRIDE 448
            static uint8_t CHECKASM_ALIGN c_buf[BUF_STRIDE * 72], a_buf[BUF_STRIDE * 72];
            static uint8_t CHECKASM_ALIGN h_edge[BUF_STRIDE * 8];
            static uint8_t left[64][4];
            int bad = 0;
            for (int cfg_i = 0; cfg_i < 3000 && !bad; cfg_i++) {
                const unsigned s = (const unsigned[3]){ 56, 140, 22 }[cfg_i % 3];
                const int w0 = s ? (int)(rnd() & 127) - 96 : 0;
                LooprestorationParams params;
                memset(&params, 0, sizeof params);
                params.sgr.s0 = s;
                params.sgr.s1 = 0;
                params.sgr.w0 = w0;
                params.sgr.w1 = 33 - w0;
                const int base_w = 1 + (int)(rnd() % 384);
                const int base_h = 1 + (int)(rnd() & 63);
                const int x_off = rnd() & 7, y_off = rnd() & 7;

                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 384 + 4; x++)
                        h_edge[y * BUF_STRIDE + x] =
                            (((x + x_off) ^ (y + y_off)) & 8 ? 255 : 0) ^ (rnd() & 15);
                for (int i = 0; i < 64 * 4; i++) ((uint8_t *) left)[i] = rnd();

                for (int edges = 0; edges < 16 && !bad; edges++) {
                    const int w = edges & LR_HAVE_RIGHT ? 256 : base_w;
                    const int h = edges & LR_HAVE_BOTTOM ? 64 : base_h;
                    for (int y = 0; y < 64; y++)
                        for (int x = 0; x < 384 + 4; x++)
                            c_buf[y * BUF_STRIDE + x] =
                                (((x + x_off) ^ (y + y_off)) & 8 ? 255 : 0) ^ (rnd() & 15);
                    memcpy(a_buf, c_buf, sizeof a_buf);

                    c.sgr[0](c_buf, BUF_STRIDE, left, h_edge, w, h, &params, edges);
                    sgr_5x5_rvv(a_buf, BUF_STRIDE, left, h_edge, w, h, &params, edges);

                    for (int y = 0; y < h && !bad; y++)
                        for (int x = 0; x < w && !bad; x++)
                            if (c_buf[y * BUF_STRIDE + x] != a_buf[y * BUF_STRIDE + x]) {
                                fprintf(stderr, "level 12 FAIL cfg=%d s=%u w0=%d %dx%d edges=%d "
                                        "at (%d,%d): C=%d rvv=%d\n", cfg_i, s, w0, w, h,
                                        edges, x, y, c_buf[y * BUF_STRIDE + x],
                                        a_buf[y * BUF_STRIDE + x]);
                                bad = failures = 1;
                            }
                }
            }
            if (!bad) printf("level 12: sgr_5x5_rvv == C sgr_5x5 (3000 configs x 16 edges)\n");
        }
    }

    if (!level || level == 13) {
        /* whole sgr_mix_rvv vs the library's pure-C sgr_mix (c.sgr[2]) */
        Dav1dLoopRestorationDSPContext c;
        dav1d_loop_restoration_dsp_init_8bpc(&c, 8);
        if (c.sgr[2] == (looprestorationfilter_fn) sgr_mix_rvv) {
            fprintf(stderr, "level 13: library init resolved to the RVV asm\n");
            failures++;
        } else {
#define BUF_STRIDE 448
            static uint8_t CHECKASM_ALIGN c_buf[BUF_STRIDE * 72], a_buf[BUF_STRIDE * 72];
            static uint8_t CHECKASM_ALIGN h_edge[BUF_STRIDE * 8];
            static uint8_t left[64][4];
            int bad = 0;
            for (int cfg_i = 0; cfg_i < 3000 && !bad; cfg_i++) {
                LooprestorationParams params;
                memset(&params, 0, sizeof params);
                /* checkasm's mix row: dav1d_sgr_params[0] = { 140, 3236 } */
                params.sgr.s0 = 140;
                params.sgr.s1 = 3236;
                const int w0 = (int)(rnd() & 127) - 96;
                params.sgr.w0 = w0;
                params.sgr.w1 = 160 - (int)(rnd() & 127) - w0;
                const int base_w = 1 + (int)(rnd() % 384);
                const int base_h = 1 + (int)(rnd() & 63);
                const int x_off = rnd() & 7, y_off = rnd() & 7;

                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 384 + 4; x++)
                        h_edge[y * BUF_STRIDE + x] =
                            (((x + x_off) ^ (y + y_off)) & 8 ? 255 : 0) ^ (rnd() & 15);
                for (int i = 0; i < 64 * 4; i++) ((uint8_t *) left)[i] = rnd();

                for (int edges = 0; edges < 16 && !bad; edges++) {
                    const int w = edges & LR_HAVE_RIGHT ? 256 : base_w;
                    const int h = edges & LR_HAVE_BOTTOM ? 64 : base_h;
                    for (int y = 0; y < 64; y++)
                        for (int x = 0; x < 384 + 4; x++)
                            c_buf[y * BUF_STRIDE + x] =
                                (((x + x_off) ^ (y + y_off)) & 8 ? 255 : 0) ^ (rnd() & 15);
                    memcpy(a_buf, c_buf, sizeof a_buf);

                    c.sgr[2](c_buf, BUF_STRIDE, left, h_edge, w, h, &params, edges);
                    sgr_mix_rvv(a_buf, BUF_STRIDE, left, h_edge, w, h, &params, edges);

                    for (int y = 0; y < h && !bad; y++)
                        for (int x = 0; x < w && !bad; x++)
                            if (c_buf[y * BUF_STRIDE + x] != a_buf[y * BUF_STRIDE + x]) {
                                fprintf(stderr, "level 13 FAIL cfg=%d w0=%d w1=%d %dx%d "
                                        "edges=%d at (%d,%d): C=%d rvv=%d\n", cfg_i,
                                        params.sgr.w0, params.sgr.w1, w, h, edges, x, y,
                                        c_buf[y * BUF_STRIDE + x],
                                        a_buf[y * BUF_STRIDE + x]);
                                bad = failures = 1;
                            }
                }
            }
            if (!bad) printf("level 13: sgr_mix_rvv == C sgr_mix (3000 configs x 16 edges)\n");
        }
    }

    if (failures) { printf("DIFF FAILURES\n"); return 1; }
    printf("all requested levels ok\n");
    return 0;
}
