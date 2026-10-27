/* Shape A (direct e8 -> e32m4 widen) vs shape B (two-step via e16m2).
 * Operands: %0 = pad, %1 = sum, %2 = sq, %3 = n.  */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static uint8_t pad[4096];
static int16_t sum[4096];
static int32_t sq[4096];

int main(int argc, char **argv) {
    int shape = atoi(argv[1]);
    int w = atoi(argv[2]);
    for (int i = 0; i < 4096; i++) pad[i] = (uint8_t)i;
    long n = w + 2;
    const uint8_t *p = pad;
    int16_t *s = sum; int32_t *q = sq;
    if (shape == 0) {
        asm volatile(
            "1:\n"
            " vsetvli a7, %3, e32, m4, ta, ma\n"
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
            " sub %3, %3, a7\n"
            " bnez %3, 1b\n"
            : "+r"(p), "+r"(s), "+r"(q), "+r"(n)
            :: "t0", "a7", "memory");
    } else {
        asm volatile(
            "1:\n"
            " vsetvli a7, %3, e32, m4, ta, ma\n"
            " vsetvli zero, a7, e8, m1, ta, ma\n"
            " vle8.v v8, (%0)\n"
            " add t0, %0, 1\n"
            " vle8.v v9, (t0)\n"
            " add t0, %0, 2\n"
            " vle8.v v10, (t0)\n"
            " vsetvli zero, a7, e16, m2, ta, ma\n"
            " vzext.vf2 v12, v8\n"
            " vzext.vf2 v14, v9\n"
            " vzext.vf2 v16, v10\n"
            " vadd.vv v18, v12, v14\n"
            " vadd.vv v18, v18, v16\n"
            " vse16.v v18, (%1)\n"
            " vsetvli zero, a7, e32, m4, ta, ma\n"
            " vzext.vf2 v20, v12\n"
            " vzext.vf2 v24, v14\n"
            " vzext.vf2 v28, v16\n"
            " vmul.vv v20, v20, v20\n"
            " vmacc.vv v20, v24, v24\n"
            " vmacc.vv v20, v28, v28\n"
            " vse32.v v20, (%2)\n"
            " slli t0, a7, 2\n"
            " add %2, %2, t0\n"
            " slli t0, a7, 1\n"
            " add %1, %1, t0\n"
            " add %0, %0, a7\n"
            " sub %3, %3, a7\n"
            " bnez %3, 1b\n"
            : "+r"(p), "+r"(s), "+r"(q), "+r"(n)
            :: "t0", "a7", "memory");
    }
    printf("shape %d ok: sum0=%d (exp %d) sq0=%d (exp %d)\n", shape,
           sum[0], pad[0] + pad[1] + pad[2],
           sq[0], pad[0]*pad[0] + pad[1]*pad[1] + pad[2]*pad[2]);
    return 0;
}
