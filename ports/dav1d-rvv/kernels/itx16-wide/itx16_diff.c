/* Campaign-4 differential driver for the 16x16 itx path (gate 2).
 *
 * Compares the RVV asm (linked straight out of the worktree build) against
 * the C reference (via dav1d_itx_dsp_init_8bpc_c) over: all twelve 16x16
 * transform types, an eob sweep including the asm's sparse thresholds
 * (8 and 36) and the extremes, several dst strides/alignments, and value
 * patterns at the int16 clipping edges.  Coefficient generation replicates
 * checkasm's tests/checkasm/itx.c guarantees (scan-order zeroing beyond
 * eob, random fill of the 32x32 padding).
 *
 * Build (in the container, from ~/portforge-cpu):
 *   riscv64-linux-gnu-gcc -O2 -DBITDEPTH=8 -std=c99 -D_GNU_SOURCE \
 *     -I build/dav1d-c4-opt-rv64 -I build/trees/c4-opt \
 *     -I build/dav1d-c4-opt-rv64/include/dav1d -I build/trees/c4-opt/include/dav1d \
 *     -I build/trees/c4-opt/include \
 *     c4/itx16_diff.c -o c4/itx16_diff \
 *     build/dav1d-c4-opt-rv64/src/libdav1d.a build/dav1d-c4-opt-rv64/src/libdav1d_bitdepth_8.a
 * Run: qemu-riscv64 -cpu rv64,v=true,vlen=<vlen>,... c4/itx16_diff [seed]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "src/itx.h"
#include "src/tables.h"
#include "src/scan.h"

/* the twelve 16x16 asm entry points */
#define D16(name) extern void dav1d_inv_txfm_add_##name##_16x16_8bpc_rvv( \
    uint8_t *dst, ptrdiff_t stride, int16_t *coeff, int eob);
D16(dct_dct) D16(identity_identity) D16(dct_adst) D16(dct_flipadst)
D16(dct_identity) D16(adst_dct) D16(adst_adst) D16(adst_flipadst)
D16(flipadst_dct) D16(flipadst_adst) D16(flipadst_flipadst) D16(identity_dct)
#undef D16

static const itxfm_fn asm_fns[12] = {
    dav1d_inv_txfm_add_dct_dct_16x16_8bpc_rvv,
    dav1d_inv_txfm_add_identity_identity_16x16_8bpc_rvv,
    dav1d_inv_txfm_add_dct_adst_16x16_8bpc_rvv,
    dav1d_inv_txfm_add_dct_flipadst_16x16_8bpc_rvv,
    dav1d_inv_txfm_add_dct_identity_16x16_8bpc_rvv,
    dav1d_inv_txfm_add_adst_dct_16x16_8bpc_rvv,
    dav1d_inv_txfm_add_adst_adst_16x16_8bpc_rvv,
    dav1d_inv_txfm_add_adst_flipadst_16x16_8bpc_rvv,
    dav1d_inv_txfm_add_flipadst_dct_16x16_8bpc_rvv,
    dav1d_inv_txfm_add_flipadst_adst_16x16_8bpc_rvv,
    dav1d_inv_txfm_add_flipadst_flipadst_16x16_8bpc_rvv,
    dav1d_inv_txfm_add_identity_dct_16x16_8bpc_rvv,
};

/* order must match itx_tmpl.c's assign_itx_all_fn16 wiring:
 * enum DCT_ADST  <-> fn adst_dct   (enum lists types[0]_types[1]; the fn
 *  name lists txfm1_txfm2 = the reverse order) */
static const int txtps[12] = {
    DCT_DCT, IDTX, ADST_DCT, FLIPADST_DCT, H_DCT,
    DCT_ADST, ADST_ADST, FLIPADST_ADST, DCT_FLIPADST, ADST_FLIPADST,
    FLIPADST_FLIPADST, V_DCT,
};

static const char *names[12] = {
    "dct_dct", "identity_identity", "dct_adst", "dct_flipadst", "dct_identity",
    "adst_dct", "adst_adst", "adst_flipadst", "flipadst_dct", "flipadst_adst",
    "flipadst_flipadst", "identity_dct",
};

#define TX TX_16X16
#define W 16
#define H 16

static unsigned long long rng_state;
static unsigned rnd(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned)(rng_state >> 33);
}

/* ---- checkasm's forward-transform coefficient generator (realistic range) */
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524401
#endif
enum Tx1D { X_DCT, X_ADST, X_FLIPADST, X_IDENTITY, X_WHT };

static const uint8_t tx1d_types[N_TX_TYPES_PLUS_LL][2] = {
    [DCT_DCT]           = { X_DCT,      X_DCT      },
    [ADST_DCT]          = { X_DCT,      X_ADST     },
    [DCT_ADST]          = { X_ADST,     X_DCT      },
    [ADST_ADST]         = { X_ADST,     X_ADST     },
    [FLIPADST_DCT]      = { X_DCT,      X_FLIPADST },
    [DCT_FLIPADST]      = { X_FLIPADST, X_DCT      },
    [FLIPADST_FLIPADST] = { X_FLIPADST, X_FLIPADST },
    [ADST_FLIPADST]     = { X_FLIPADST, X_ADST     },
    [FLIPADST_ADST]     = { X_ADST,     X_FLIPADST },
    [IDTX]              = { X_IDENTITY, X_IDENTITY },
    [V_DCT]             = { X_IDENTITY, X_DCT      },
    [H_DCT]             = { X_DCT,      X_IDENTITY },
};

static void fdct_1d(double *const out, const double *const in, const int sz)
{
    for (int i = 0; i < sz; i++) {
        out[i] = 0.0;
        for (int j = 0; j < sz; j++)
            out[i] += in[j] * cos(M_PI * (2 * j + 1) * i / (sz * 2.0));
    }
    out[0] *= M_SQRT1_2;
}

static void fadst_1d(double *const out, const double *const in, const int sz)
{
    for (int i = 0; i < sz; i++) {
        out[i] = 0.0;
        for (int j = 0; j < sz; j++)
            out[i] += in[j] * sin(M_PI *
            (sz == 4 ? (    j + 1) * (2 * i + 1) / (8.0 + 1.0) :
                       (2 * j + 1) * (2 * i + 1) / (sz * 4.0)));
    }
}

/* verbatim structure of checkasm's ftx(): forward transform of random
 * pixels, rounded to int16 - the coefficient range checkasm validated
 * the upstream asm against */
static void gen_coeff(int16_t *coeff, const int txtp, const unsigned pattern)
{
    double out[64 * 64], temp[64 * 64];
    const double scale = 1.0; /* scaling_factors[ctz(16*16) - 4] */
    const int w = W, h = H;

    for (int i = 0; i < h; i++) {
        double in[64], t_out[64];
        for (int j = 0; j < w; j++)
            in[j] = (double)((int)(rnd() & 511) - 255);
        switch (tx1d_types[txtp][0]) {
        case X_DCT: fdct_1d(t_out, in, w); break;
        case X_ADST:
        case X_FLIPADST: fadst_1d(t_out, in, w); break;
        default: memcpy(t_out, in, w * sizeof(*t_out)); break;
        }
        for (int j = 0; j < w; j++)
            temp[j * h + i] = t_out[j] * scale;
    }
    for (int i = 0; i < w; i++) {
        switch (tx1d_types[txtp][0]) {
        case X_DCT: fdct_1d(&out[i * h], &temp[i * h], h); break;
        case X_ADST:
        case X_FLIPADST: fadst_1d(&out[i * h], &temp[i * h], h); break;
        default: memcpy(&out[i * h], &temp[i * h], h * sizeof(*out)); break;
        }
    }
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            double v = out[y * w + x];
            if (pattern == 3 && (rnd() & 1)) v = 0.0;
            coeff[y * W + x] = (int16_t)(v + 0.5);
        }
}

static void zero_beyond_eob(int16_t *coeff, const int txtp, const int eob)
{
    const enum TxClass tc = dav1d_tx_type_class[txtp];
    const uint16_t *const scan = dav1d_scans[TX];
    if (tc == TX_CLASS_2D) {
        for (int n = eob + 1; n < W * H; n++)
            coeff[scan[n]] = 0;
    } else if (tc == TX_CLASS_H) {
        for (int n = eob + 1; n < W * H; n++)
            coeff[n] = 0;
    } else { /* TX_CLASS_V */
        int rcx = eob / W, rcy = eob % W;
        for (; rcx < H; rcx++, rcy = -1)
            while (++rcy < W)
                coeff[rcy * H + rcx] = 0;
    }
    for (int n = W * H; n < 32 * 32; n++)
        coeff[n] = (int16_t)(rnd() % 65536 - 32768); /* padding garbage */
}

int main(int argc, char **argv)
{
    static int16_t coeff[2][32 * 32] __attribute__((aligned(64)));
    static uint8_t dst[2][64 * 96] __attribute__((aligned(64)));
    rng_state = argc > 1 ? (unsigned long long)atoll(argv[1]) : 1;
    Dav1dInvTxfmDSPContext c;
    extern void dav1d_itx_dsp_init_8bpc(Dav1dInvTxfmDSPContext *c, int bpc);
    dav1d_itx_dsp_init_8bpc(&c, 8);

    if (argc > 2 && argv[2][0] == 'd') {
        /* DC-only probe: coeff[0]=dc, eob=255, stride 16; dump both grids */
        const int dc = argc > 3 ? atoi(argv[3]) : 100;
        memset(coeff, 0, sizeof(coeff));
        coeff[0][0] = dc; memcpy(coeff[1], coeff[0], sizeof(coeff[1]));
        for (int n = 0; n < 64 * 96; n++)
            dst[0][n] = dst[1][n] = 0;
        c.itxfm_add[TX][DCT_DCT](dst[0] + 16, 16, coeff[0], 255);
        asm_fns[0](dst[1] + 16, 16, coeff[1], 255);
        printf("dc=%d  (C grid then asm grid, row 0)\n", dc);
        for (int k = 0; k < 2; k++) {
            printf("%s rows 0-1:", k ? "asm" : "C  ");
            for (int j = 0; j < 32; j++)
                printf(" %3d", dst[k][16 + j]);
            printf("\n");
        }
        int bad = 0;
        for (int r = 0; r < 16; r++) {
            int rowbad = 0;
            for (int x = 0; x < 16; x++)
                if (dst[0][16 + r * 16 + x] != dst[1][16 + r * 16 + x]) rowbad++;
            if (rowbad) {
                printf("row %d differs (c vs asm):", r);
                for (int x = 0; x < 16; x++)
                    printf(" %d/%d", dst[0][16 + r * 16 + x], dst[1][16 + r * 16 + x]);
                printf("\n");
            }
            bad += rowbad;
        }
        printf("diffs=%d\n", bad);
        return 0;
    }

    if (argc > 2 && argv[2][0] == 'p') {
        /* single-coefficient sweep: coef=K at raster pos p, eob=255 */
        const int K = argc > 3 ? atoi(argv[3]) : 100;
        int total = 0;
        int one = argc > 4 ? atoi(argv[4]) : -1;
        int fn = argc > 5 ? atoi(argv[5]) : 0;
        for (int p = 0; p < 256; p++) {
            memset(coeff, 0, sizeof(coeff));
            coeff[0][p] = K; memcpy(coeff[1], coeff[0], sizeof(coeff[1]));
            for (int n = 0; n < 64 * 96; n++)
                dst[0][n] = dst[1][n] = 0;
            c.itxfm_add[TX][txtps[fn]](dst[0] + 16, 16, coeff[0], 255);
            asm_fns[fn](dst[1] + 16, 16, coeff[1], 255);
            int rowbad = 0;
            for (int r = 0; r < 16; r++)
                for (int x = 0; x < 16; x++)
                    if (dst[0][16 + r * 16 + x] != dst[1][16 + r * 16 + x]) rowbad++;
            if (rowbad) {
                total += rowbad;
                if (one < 0 || p == one) {
                    printf("coef@(r=%d,c=%d): %d elem diffs\n", p / 16, p % 16, rowbad);
                    for (int r = 0; r < 16; r++)
                        for (int x = 0; x < 16; x++)
                            if (dst[0][16 + r * 16 + x] != dst[1][16 + r * 16 + x])
                                printf("  (%d,%d) c=%d asm=%d\n", r, x,
                                       dst[0][16 + r * 16 + x], dst[1][16 + r * 16 + x]);
                }
            }
        }
        printf("single-coef sweep total diffs=%d\n", total);
        return 0;
    }

    long cases = 0;
    int fails = 0;
    static int fail_eob[12][32];
    memset(fail_eob, 0, sizeof(fail_eob));

    /* strides: tight, odd (misaligns every row), padded */
    const ptrdiff_t strides[] = { argc > 3 ? (ptrdiff_t)atoi(argv[3]) : W, W + 1, 32, 64 };
    const int nstrides = (int)(sizeof(strides) / sizeof(strides[0]));

    /* eob sweep: full first (main datapath), then sparse thresholds */
    static const int eobs[] = {
        255, 254, 200, 128, 100, 64, 55, 48, 40, 37, 36, 35, 32, 31, 24,
        20, 17, 16, 12, 9, 8, 7, 5, 3, 2, 1,
    };
    const int neob = (int)(sizeof(eobs) / sizeof(eobs[0]));

    for (int fn = 0; fn < 12; fn++)
      for (int ei = 0; ei < neob; ei++)
        for (int pat = 0; pat < 4; pat++)
          for (int si = 0; si < nstrides; si++)
            for (int rep = 0; rep < 3; rep++) {
                int eob = eobs[ei];
                const ptrdiff_t stride = strides[si];
                if (eob == 0 && txtps[fn] != DCT_DCT)
                    continue; /* checkasm never generates eob=0 for non-DC types */
                gen_coeff(coeff[0], txtps[fn], pat);
                zero_beyond_eob(coeff[0], txtps[fn], eob);
                memcpy(coeff[1], coeff[0], sizeof(coeff[1]));
                for (int n = 0; n < 64 * 96; n++)
                    dst[0][n] = dst[1][n] = (uint8_t)(rnd() & 0xff);

                c.itxfm_add[TX][txtps[fn]](dst[0] + 16, stride, coeff[0], eob);
                if (argc > 2 && argv[2][0] == 'c')   /* harness sanity mode */
                    c.itxfm_add[TX][txtps[fn]](dst[1] + 16, stride, coeff[1], eob);
                else
                    asm_fns[fn](dst[1] + 16, stride, coeff[1], eob);

                cases++;
                int bad = -1;
                for (int n = 0; n < 64 * 96 && bad < 0; n++)
                    if (dst[0][n] != dst[1][n]) bad = n;
                if (bad >= 0 || memcmp(coeff[0], coeff[1], sizeof(coeff[0]))) {
                    fails++;
                    fail_eob[fn][ei]++;
                    if (fails <= 10)
                        printf("FAIL %s eob=%d stride=%d pat=%d rep=%d: "
                               "dst[%d] c=%d asm=%d%s\n", names[fn], eob,
                               (int)stride, pat, rep, bad,
                               bad >= 0 ? dst[0][bad] : -1,
                               bad >= 0 ? dst[1][bad] : -1,
                               memcmp(coeff[0], coeff[1], sizeof(coeff[0]))
                                   ? " (+coeff not zeroed)" : "");
                }
            }
    printf("%s: %ld cases, %d failures (seed %llu)\n",
           fails ? "DIFFS" : "OK", cases, fails,
           argc > 1 ? (unsigned long long)atoll(argv[1]) : 1ULL);
    for (int fn = 0; fn < 12; fn++) {
        printf("%-18s", names[fn]);
        for (int ei = 0; ei < neob; ei++)
            printf(" %d:%d", eobs[ei], !!fail_eob[fn][ei]);
        printf("\n");
    }
    return !!fails;
}
