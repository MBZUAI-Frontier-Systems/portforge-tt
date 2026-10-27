#include <stdio.h>
#include <stdint.h>
#include <string.h>

void dav1d_sgr_finish_filter_row1_8bpc_rvv(int16_t *tmp, const uint8_t *src,
                                           int32_t **A, int16_t **B, int w);

static int16_t ta[8], tb[8];
static uint8_t sb[8];
static int32_t Arow2[3][8];
static int16_t Brow2[3][8];

int main(void) {
    int Bv[3][3] = { {10,20,30}, {40,50,60}, {70,80,90} };
    int Av[3][3] = { {100,200,300}, {400,500,600}, {700,800,900} };
    int32_t *Ap[3] = { Arow2[0], Arow2[1], Arow2[2] };
    int16_t *Bp[3] = { Brow2[0], Brow2[1], Brow2[2] };
    for (int k = 0; k < 3; k++)
        for (int j = 0; j < 3; j++) { Brow2[k][j] = Bv[k][j]; Arow2[k][j] = Av[k][j]; }
    for (int j = 0; j < 8; j++) { sb[j] = 0; ta[j] = tb[j] = -9999; }

    dav1d_sgr_finish_filter_row1_8bpc_rvv(tb, sb, Ap, Bp, 1);
    printf("src=0: tmp0=%d (expect 31)\n", tb[0]);

    sb[0] = 2;
    dav1d_sgr_finish_filter_row1_8bpc_rvv(tb, sb, Ap, Bp, 1);
    printf("src=2: tmp0=%d (expect 25)\n", tb[0]);

    /* w=3 across the row */
    for (int j = 0; j < 8; j++) sb[j] = j;
    dav1d_sgr_finish_filter_row1_8bpc_rvv(tb, sb, Ap, Bp, 3);
    printf("w=3: %d %d %d (expect -40 -40 -40)\n", tb[0], tb[1], tb[2]);
    return 0;
}
