/* dav1d Wiener loop-restoration filter, 8bpc: lifted RVV vs src/looprestoration_tmpl.c's wiener_c.
 *
 * Domain (decoder and checkasm): 1 <= w, 1 <= h <= 64; any of the 16 edge
 * combinations; symmetric taps with f0 in [-5,10] (0 for the 5-tap filter),
 * f1 in [-23,8], f2 in [-17,46], filter[0][3] = -2(f0+f1+f2) and
 * filter[1][3] = 128 - 2(f0+f1+f2); pixels 0..255.
 *
 * Memory contract: the filter runs in place on p (rows 0..h-1). It may write
 * row y only in [0, align64(w)) (checkasm's padded-align rule; bytes in
 * [w, align64(w)) are not compared) and must leave every other byte of p, lpf,
 * left and params unchanged. Reads are allowed anywhere in the p rows (plus 16
 * bytes before), in the 8 lpf rows (plus 16 bytes before), left and params.
 *
 * Bounded configuration: -DW -DH -DEDGES -DTAPS (7 or 5) and -DTAPSET (0 =
 * symbolic taps over the whole domain, k > 0 = the k-th fixed tap set). */
#include <stdlib.h>
#define BITDEPTH 8
#include "fv_runtime.h"
#include "lifted.c"
#include "src/looprestoration_tmpl.c"

#ifndef W
#define W 7
#endif
#ifndef H
#define H 2
#endif
#ifndef EDGES
#define EDGES 15
#endif
#ifndef TAPS
#define TAPS 7
#endif
#ifndef TAPSET
#define TAPSET 0
#endif
#ifndef ENTRY
#define ENTRY fvf_dav1d_wiener_filter7_8bpc_rvv
#endif

#define ALIGN64(x) (((x) + 63) & ~63)
#define STRIDE (ALIGN64(W + 3) + 64)
#define PRE 16
#define P_ADDR 0x1000000ULL
#define LPF_ADDR 0x2000000ULL
#define LEFT_ADDR 0x3000000ULL
#define PARAMS_ADDR 0x3100000ULL

#define P_N (PRE + H * STRIDE)
#define LPF_N (PRE + 8 * STRIDE)
static uint8_t p_r[P_N];
static uint8_t *p_c, *lpf_c, *left_c, *params_c;

static const int8_t fixed_taps[][6] = {
    /* f0 f1 f2 (horizontal), f0 f1 f2 (vertical): the envelope's corners and a typical set */
    {0, 0, 0, 0, 0, 0},
    {10, 8, 46, 10, 8, 46},
    {-5, -23, -17, -5, -23, -17},
    {3, -7, 15, -1, -6, 27},
    {10, -23, 46, -5, 8, -17},
};

static int16_t tap(int k, int i)
{
    const int16_t *f = (const int16_t *)params_c;
    return f[8 * k + i];
}

static void set_tap(int k, int i, int16_t v)
{
    params_c[2 * (8 * k + i)] = (uint8_t)v;
    params_c[2 * (8 * k + i) + 1] = (uint8_t)((uint16_t)v >> 8);
}

int main(int argc, char **argv)
{
#ifndef FV_CBMC
    if (argc > 1) fv_rng = strtoull(argv[1], 0, 10) * 0x9E3779B97F4A7C15ULL + 1;
    if (argc > 2) fv_agnostic_ones = atoi(argv[2]);
#else
    (void)argc; (void)argv;
#endif
    p_c = fv_alloc(P_N); lpf_c = fv_alloc(LPF_N); left_c = fv_alloc(4 * H); params_c = fv_alloc(32);
    fv_fill(p_c, P_N);
    fv_fill(lpf_c, LPF_N);
    fv_fill(left_c, 4 * H);
    fv_fill(params_c, 32);
    for (int k = 0; k < 2; k++) {
        int16_t f0, f1, f2;
#if TAPSET == 0
        f0 = (int16_t)(int8_t)params_c[2 * (8 * k + 0)];
        f1 = (int16_t)(int8_t)params_c[2 * (8 * k + 1)];
        f2 = (int16_t)(int8_t)params_c[2 * (8 * k + 2)];
#ifdef FV_CBMC
        __CPROVER_assume(f0 >= -5 && f0 <= 10 && f1 >= -23 && f1 <= 8 && f2 >= -17 && f2 <= 46);
#else
        f0 = (int16_t)(f0 % 16 - 5 < -5 ? -5 : f0 % 16 - 5);
        f1 = (int16_t)((uint8_t)f1 % 32 - 23);
        f2 = (int16_t)((uint8_t)f2 % 64 - 17);
        if (f0 > 10) f0 = 10;
#endif
#else
        f0 = fixed_taps[TAPSET % 5][3 * k + 0];
        f1 = fixed_taps[TAPSET % 5][3 * k + 1];
        f2 = fixed_taps[TAPSET % 5][3 * k + 2];
#endif
        if (TAPS == 5) f0 = 0;
        set_tap(k, 0, f0); set_tap(k, 6, f0);
        set_tap(k, 1, f1); set_tap(k, 5, f1);
        set_tap(k, 2, f2); set_tap(k, 4, f2);
        set_tap(k, 3, (int16_t)((k ? 128 : 0) - 2 * (f0 + f1 + f2)));
        /* filter[k][7] is never read by the reference: it stays an arbitrary input */
    }
    memcpy(p_r, p_c, P_N);

    /* candidate regions: each row's writable window first (first match wins), then
     * the read-only remainder of p */
    for (int y = 0; y < H; y++)
        fv_region_add("p-row", P_ADDR + (uint64_t)y * STRIDE, ALIGN64(W), p_c + PRE + y * STRIDE, 1);
    fv_region_add("p", P_ADDR - PRE, P_N, p_c, 0);
    fv_region_add("lpf", LPF_ADDR - PRE, LPF_N, lpf_c, 0);
    fv_region_add("left", LEFT_ADDR, 4 * H, left_c, 0);
    fv_region_add("params", PARAMS_ADDR, 32, params_c, 0);

    LooprestorationParams params;
    memcpy(&params, params_c, 32);
#ifndef FV_NO_REF   /* diagnostics: time one side alone */
    wiener_c(p_r + PRE, STRIDE, (const pixel (*)[4])left_c, lpf_c + PRE, W, H, &params, (enum LrEdgeFlags)EDGES);
#endif

    const uint64_t args[8] = {P_ADDR, STRIDE, LEFT_ADDR, LPF_ADDR + 0, W, H, PARAMS_ADDR, EDGES};
#ifndef FV_NO_CAND
    fv_call(ENTRY, args, 8);
#endif

    for (int i = 0; i < P_N; i++) {
        int off = i - PRE, y = off >= 0 ? off / STRIDE : -1, x = off >= 0 ? off % STRIDE : -1;
        if (y >= 0 && y < H && x >= W && x < ALIGN64(W))
            continue;   /* padding the filter may overwrite */
#ifdef ONLY_Y   /* one output pixel per query (sliced formula); ONLY_Y=-1: every non-output byte */
        if (ONLY_Y >= 0 ? !(y == ONLY_Y && x == ONLY_X) : (y >= 0 && y < H && x >= 0 && x < W))
            continue;
#endif
#ifdef FV_CBMC
        __CPROVER_assert(p_c[i] == p_r[i], "equivalence: output differs from the C reference");
#else
        if (p_c[i] != p_r[i]) {
            fprintf(stderr, "FV-FAIL: p differs at row %d col %d: candidate %u reference %u\n", y, x, p_c[i], p_r[i]);
            exit(4);
        }
#endif
    }
    (void)tap;
    return 0;
}
