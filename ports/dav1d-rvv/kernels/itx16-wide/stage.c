/* Stage-wise check vs exact C formulas:
 *  H: stack[r][c] must equal (2x + ((x*1697+1024)>>11) + 2) >> 2  (clipped s16)
 *     where x = coefs[r][c]  (identity horz + driver shift)
 *  V: dst = clip_u8(dst + (2y + ((y*1697+1024)>>11) + 8) >> 4)
 *     where y = stack[r][c]  (identity vert + final shift)                 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

extern void wrap_horz(long coefs, long stack, int eob, int thr, void *helper);
extern void wrap_vert(long stack, long dst, long stride, void *kern, void *helper);
extern void dav1d_inv_txfm_horz_identity_16_rvv(void);
extern void dav1d_inv_txfm_add_vert_16_rvv(void);
extern void dav1d_inv_identity_e16_x16_rvv(void);

static int16_t coefs[16][16] __attribute__((aligned(64)));
static int16_t stack_[16][16] __attribute__((aligned(64)));
static uint8_t dst[16][16] __attribute__((aligned(64)));

static unsigned st = 20250101;
static unsigned rnd(void) { st = st * 1103515245u + 12345u; return st >> 8; }

static int iclip(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

int main(void)
{
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            coefs[i][j] = (int16_t)((int)(rnd() % 4096) - 2048);
            dst[i][j] = (uint8_t)(rnd() & 0xff);
        }

    wrap_horz((long)coefs, (long)stack_, 255, 8, dav1d_inv_txfm_horz_identity_16_rvv);

    int hbad = 0;
    for (int r = 0; r < 16 && hbad < 4; r++)
        for (int c = 0; c < 16 && hbad < 4; c++) {
            int x = coefs[r][c];
            int y = 2 * x + ((x * 1697 + 1024) >> 11);
            y = (y + 2) >> 2;
            y = iclip(y, INT16_MIN, INT16_MAX);
            if (stack_[r][c] != y) {
                printf("H (%d,%d) x=%d asm=%d exp=%d\n", r, c, x, stack_[r][c], y);
                hbad++;
            }
        }
    printf("stage H diffs: %s\n", hbad ? "FAIL" : "ok");

    wrap_vert((long)stack_, (long)dst, 16, dav1d_inv_identity_e16_x16_rvv,
              dav1d_inv_txfm_add_vert_16_rvv);

    int vbad = 0;
    for (int r = 0; r < 16 && vbad < 4; r++)
        for (int c = 0; c < 16 && vbad < 4; c++) {
            int y = stack_[r][c];
            int v = 2 * y + ((y * 1697 + 1024) >> 11);
            v = (v + 8) >> 4;
            int e = iclip(dst[r][c] + v, 0, 255);
            if (dst[r][c] != e) {
                printf("V (%d,%d) y=%d v=%d asm=%d exp=%d\n", r, c, y, v, dst[r][c], e);
                vbad++;
            }
        }
    printf("stage V diffs: %s\n", vbad ? "FAIL" : "ok");
    return 0;
}
