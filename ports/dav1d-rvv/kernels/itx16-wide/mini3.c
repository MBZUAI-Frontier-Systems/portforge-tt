/* Isolated vnclipu.wi at (e8, mf2 vs m1, vl=16) with known source. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static int16_t src[16] __attribute__((aligned(64)));
static uint8_t out[16] __attribute__((aligned(64)));

int main(void)
{
    for (int i = 0; i < 16; i++) src[i] = (int16_t)(3000 - 700 * i); /* mixed sign */
    long ps = (long)src, po = (long)out;
    for (int mode = 0; mode < 2; mode++) {
        if (mode == 0) {
            __asm__ volatile(
                "vsetivli zero, 16, e16, m1, ta, ma\n"
                "mv t4, %[s]\n"
                "vle16.v v24, 0(t4)\n"
                "vsetivli zero, 16, e8, m1, ta, ma\n"
                "vnclipu.wi v16, v24, 0\n"
                "mv t5, %[o]\n"
                "vse8.v v16, 0(t5)\n"
                :: [s]"r"(ps), [o]"r"(po)
                : "t4","t5","v16","v24","v25","memory");
        } else {
            __asm__ volatile(
                "vsetivli zero, 16, e16, m1, ta, ma\n"
                "mv t4, %[s]\n"
                "vle16.v v24, 0(t4)\n"
                "vsetivli zero, 16, e8, mf2, ta, ma\n"
                "vnclipu.wi v16, v24, 0\n"
                "mv t5, %[o]\n"
                "vse8.v v16, 0(t5)\n"
                :: [s]"r"(ps), [o]"r"(po)
                : "t4","t5","v16","v24","v25","memory");
        }
        printf("(e8,%s,vl=16) vnclipu:", mode ? "mf2" : "m1");
        for (int i = 0; i < 8; i++) printf(" %d", out[i]);
        printf("\n   exp:");
        for (int i = 0; i < 8; i++) {
            int v = 3000 - 700 * i;
            printf(" %d", v < 0 ? 0 : v > 255 ? 255 : v);
        }
        printf("\n");
    }
    return 0;
}
