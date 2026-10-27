/* Reproduce box3_row_h's exact inner-block sequence in isolation. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static uint8_t pad[512];
static int16_t sum[512];
static int32_t sq[512];
int main(int argc, char **argv) {
    int w = argc > 1 ? atoi(argv[1]) : 37;
    for (int i = 0; i < 512; i++) pad[i] = (uint8_t)i;
    long n = w + 2;
    const uint8_t *p = pad;
    int16_t *s = sum; int32_t *q = sq;
    asm volatile(
        "1:\n"
        " vsetvli a7, %4, e32, m4, ta, ma\n"
        " vsetvli zero, a7, e8, m1, ta, ma\n"
        " vle8.v v8, (%0)\n"
        " add t0, %0, 1\n"
        " vle8.v v9, (t0)\n"
        " add t0, %0, 2\n"
        " vle8.v v10, (t0)\n"
        " vsetvli zero, a7, e32, m4, ta, ma\n"
        " vzext.vf2 v16, v8\n"
        " vzext.vf2 v20, v9\n"
        " vzext.vf2 v24, v10\n"
        " vadd.vv v28, v16, v20\n"
        " vadd.vv v28, v28, v24\n"
        " vmul.vv v16, v16, v16\n"
        " vmacc.vv v16, v20, v20\n"
        " vmacc.vv v16, v24, v24\n"
        " vsetvli zero, a7, e16, m2, ta, ma\n"
        " vncvt.x.x.w v12, v28\n"
        " vse16.v v12, (%1)\n"
        " vsetvli zero, a7, e32, m4, ta, ma\n"
        " vse32.v v16, (%2)\n"
        " slli t0, a7, 2\n"
        " add %2, %2, t0\n"
        " slli t0, a7, 1\n"
        " add %1, %1, t0\n"
        " add %0, %0, a7\n"
        " sub %4, %4, a7\n"
        " bnez %4, 1b\n"
        : "+r"(p), "+r"(s), "+r"(q), "+r"(n)
        :: "t0", "a7", "memory");
    printf("ok, sum[0]=%d sq[0]=%d\n", sum[0], sq[0]);
    return 0;
}
