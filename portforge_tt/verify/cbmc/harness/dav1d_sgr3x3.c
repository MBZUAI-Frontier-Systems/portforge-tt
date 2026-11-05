/* dav1d SGR 3x3 loop-restoration filter, 8bpc: lifted RVV vs src/looprestoration_tmpl.c's
 * sgr_3x3_c.
 *
 * Domain (decoder side, lr_apply_tmpl.c): this entry point (dsp->lr.sgr[1])
 * is selected exactly when the sgr_params row has s0 == 0 and s1 != 0, i.e.
 * rows 10..13 of dav1d_sgr_params (s1 in {2589, 1618, 1177, 925}); then
 * w0 = 0 and w1 = 128 - sw1 with sw1 in [-32, 95], so w1 in [33, 160].
 * 1 <= w, 1 <= h <= 64; any of the 16 edge combinations; pixels 0..255.
 * (checkasm exercises only row 10; the proof sweeps the other rows at the
 * smallest size.)
 *
 * Memory contract: the filter runs in place on p (rows 0..h-1). It may write
 * row y only in [0, align64(w)) (checkasm's padded-align rule; bytes in
 * [w, align64(w)) are not compared) and must leave every other byte of p,
 * lpf, left and params unchanged. Reads are allowed anywhere in the p rows
 * (plus 16 bytes before), in the 8 lpf rows (plus 16 bytes before), left and
 * params.
 *
 * Bounded configuration: -DW -DH -DEDGES and -DSROW (a dav1d_sgr_params row
 * with s0 == 0: 10, 11, 12 or 13). Configurations never set LR_HAVE_RIGHT:
 * in the decoder envelope that only occurs at w >= 32 (between restoration
 * units), out of CBMC reach here; it is covered by checkasm and the held-out
 * seeds, not by this proof. */
#include <stdlib.h>
#define BITDEPTH 8
#include "fv_runtime.h"
#include "lifted.c"
#include "src/looprestoration_tmpl.c"

#ifndef W
#define W 2
#endif
#ifndef H
#define H 2
#endif
#ifndef EDGES
#define EDGES 13
#endif
#ifndef SROW
#define SROW 10
#endif
#ifndef ENTRY
#define ENTRY fvf_dav1d_sgr_filter_3x3_8bpc_rvv
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

    /* sgr[1] domain: s0 = 0, s1 from the config's table row, w0 = 0,
     * w1 in [33, 160] (see header). The rest of the union stays an
     * arbitrary input: the reference never reads it. */
    int16_t w1;
#ifdef FV_CBMC
    w1 = (int16_t)((uint16_t)(params_c[10] | (params_c[11] << 8)));
    __CPROVER_assume(w1 >= 33 && w1 <= 160);
#else
    w1 = (int16_t)(33 + params_c[10] % 128);
#endif
    /* the sgr fields are written as bytes and the reference's union is read
     * back from them (as in dav1d_wiener.c): field stores into a local union
     * followed by a byte memcpy make CBMC 6.6's flattening fail an internal
     * invariant (boolbv_map literal width) and its SMT output unparsable */
    fv_put(params_c + 0, 0, 4);
    fv_put(params_c + 4, dav1d_sgr_params[SROW][1], 4);
    fv_put(params_c + 8, 0, 2);
    fv_put(params_c + 10, (uint16_t)w1, 2);
    LooprestorationParams params;
    memcpy(&params, params_c, 32);
    memcpy(p_r, p_c, P_N);

    /* candidate regions: each row's writable window first (first match wins),
     * then the read-only remainder of p */
    for (int y = 0; y < H; y++)
        fv_region_add("p-row", P_ADDR + (uint64_t)y * STRIDE, ALIGN64(W), p_c + PRE + y * STRIDE, 1);
    fv_region_add("p", P_ADDR - PRE, P_N, p_c, 0);
    fv_region_add("lpf", LPF_ADDR - PRE, LPF_N, lpf_c, 0);
    fv_region_add("left", LEFT_ADDR, 4 * H, left_c, 0);
    fv_region_add("params", PARAMS_ADDR, 32, params_c, 0);

#ifndef FV_NO_REF   /* diagnostics: time one side alone */
    sgr_3x3_c(p_r + PRE, STRIDE, (const pixel (*)[4])left_c, lpf_c + PRE,
              W, H, &params, (enum LrEdgeFlags)EDGES);
#endif

    const uint64_t args[8] = {P_ADDR, STRIDE, LEFT_ADDR, LPF_ADDR + 0, W, H, PARAMS_ADDR, EDGES};
#ifndef FV_NO_CAND
    fv_call(ENTRY, args, 8);
#endif

    for (int i = 0; i < P_N; i++) {
        int off = i - PRE, y = off >= 0 ? off / STRIDE : -1, x = off >= 0 ? off % STRIDE : -1;
        if (y >= 0 && y < H && x >= W && x < ALIGN64(W))
            continue;   /* padding the filter may overwrite */
#ifdef FV_CBMC
        __CPROVER_assert(p_c[i] == p_r[i], "equivalence: output differs from the C reference");
#else
        if (p_c[i] != p_r[i]) {
            fprintf(stderr, "FV-FAIL: p differs at row %d col %d: candidate %u reference %u\n",
                    y, x, p_c[i], p_r[i]);
            exit(4);
        }
#endif
    }
    return 0;
}
