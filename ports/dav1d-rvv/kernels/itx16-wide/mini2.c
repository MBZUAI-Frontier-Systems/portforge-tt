/* Minimal: vwaddu.wv at (e8, mf2 vs m1, vl=16) with known inputs. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static int16_t vert[16] __attribute__((aligned(64)));
static uint8_t dstv[16] __attribute__((aligned(64)));
static int16_t out[16] __attribute__((aligned(64)));

int main(void)
{
    for (int i = 0; i < 16; i++) { vert[i] = 100 + i; dstv[i] = 200; }
    long pv = (long)vert, pd = (long)dstv, po = (long)out;
    for (int mode = 0; mode < 2; mode++) {
        if (mode == 0) {
            __asm__ volatile(
                "vsetivli zero, 16, e16, m1, ta, ma\n"
                "mv t4, %[v]\n"
                "vle16.v v0, 0(t4)\n"
                "vsetivli zero, 16, e8, m1, ta, ma\n"
                "mv t5, %[d]\n"
                "vle8.v v16, 0(t5)\n"
                "vwaddu.wv v24, v0, v16\n"
                "vsetivli zero, 16, e16, m1, ta, ma\n"
                "mv t5, %[o]\n"
                "vse16.v v24, 0(t5)\n"
                :: [v]"r"(pv), [d]"r"(pd), [o]"r"(po)
                : "t4","t5","v0","v16","v24","v25","memory");
        } else {
            __asm__ volatile(
                "vsetivli zero, 16, e16, m1, ta, ma\n"
                "mv t4, %[v]\n"
                "vle16.v v0, 0(t4)\n"
                "vsetivli zero, 16, e8, mf2, ta, ma\n"
                "mv t5, %[d]\n"
                "vle8.v v16, 0(t5)\n"
                "vwaddu.wv v24, v0, v16\n"
                "vsetivli zero, 16, e16, m1, ta, ma\n"
                "mv t5, %[o]\n"
                "vse16.v v24, 0(t5)\n"
                :: [v]"r"(pv), [d]"r"(pd), [o]"r"(po)
                : "t4","t5","v0","v16","v24","v25","memory");
        }
        printf("(e8,%s,vl=16):", mode ? "mf2" : "m1");
        for (int i = 0; i < 4; i++) printf(" %d", out[i]);
        printf("   (exp 300 301 302 303)\n");
    }
    return 0;
}
