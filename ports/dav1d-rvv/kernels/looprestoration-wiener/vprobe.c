/* minimal v-kernel probe */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

void dav1d_wiener_filter_v_8bpc_rvv(uint8_t *dst, const uint16_t *const rows[7],
                                    const int16_t fv[8], const int w);

static uint16_t rows[7][512];
int main(void)
{
    int16_t fv[8];
    for (int i = 0; i < 8; i++) fv[i] = 0;
    fv[3] = 128;                      /* center only */
    for (int x = 0; x < 512; x++)
        for (int r = 0; r < 7; r++)
            rows[r][x] = (uint16_t)(100 + r);
    uint8_t out[512];
    memset(out, 0xAA, sizeof(out));
    const uint16_t *rp[7] = { rows[0], rows[1], rows[2], rows[3],
                              rows[4], rows[5], rows[6] };
    dav1d_wiener_filter_v_8bpc_rvv(out, rp, fv, 8);
    /* expect: (128*100 - 261120 + 1024) >> 11 = (12800 - 260096) >> 11
     * = -247296 >> 11 = -120.7 -> arithmetic -121 -> clamp 0 */
    for (int x = 0; x < 8; x++) printf("%d ", out[x]);
    printf("(center-only, expect 0s)\n");

    fv[3] = 0; for (int i = 0; i < 8; i++) fv[i] = 0; fv[1] = 128;  /* rows[1] only */
    memset(out, 0xAA, sizeof(out));
    dav1d_wiener_filter_v_8bpc_rvv(out, rp, fv, 8);
    /* expect: (128*101 + 1024 - 261120) >> 11 = (12928 - 260096) >> 11 <0 -> 0 */
    for (int x = 0; x < 8; x++) printf("%d ", out[x]);
    printf("(row1-only, expect 0s)\n");

    /* make sums positive: rows = 3000 */
    for (int x = 0; x < 512; x++)
        for (int r = 0; r < 7; r++)
            rows[r][x] = 3000;
    fv[3] = 128;
    memset(out, 0xAA, sizeof(out));
    dav1d_wiener_filter_v_8bpc_rvv(out, rp, fv, 8);
    /* expect: (128*3000 + 1024 - 261120) >> 11 = (384000+1024-261120)>>11
     * = 123904 >> 11 = 60 (60.5 -> 60) */
    for (int x = 0; x < 8; x++) printf("%d ", out[x]);
    printf("(3000 center, expect 60 x8)\n");
    return 0;
}
