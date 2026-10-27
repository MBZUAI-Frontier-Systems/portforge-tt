/* Which RVV instruction does QEMU reject? Run each candidate in a child-ish
 * sequence and print progress; the last line before SIGILL is the culprit. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static uint8_t lut[256], src[64];
static int32_t aa[64];
static int16_t bb[64], tmp[64];
static uint8_t dst[64];
static int32_t rows32[3][64];
static int16_t rows16[3][64];

#define VSET(e, m) " vsetvli t0, t6, e" #e ", " #m ", ta, ma\n"

int main(void) {
    for (int i = 0; i < 256; i++) lut[i] = (uint8_t)i;
    for (int i = 0; i < 64; i++) { src[i] = 200; aa[i] = 500000; bb[i] = 2000; }
    long n = 16;

    printf("A: vsetvli e32m2 + vle32/vsext/vmul/vnmsac/vmax/vmul/vadd/vsrl/vminu\n"); fflush(stdout);
    asm volatile(
        " mv t6, %0\n" VSET(32, m2)
        " vle32.v v8, (%1)\n"
        VSET(16, m1)
        " vle16.v v12, (%2)\n"
        VSET(32, m2)
        " vsext.vf2 v10, v12\n"
        " vmul.vx v14, v8, t3\n"
        " vnmsac.vv v14, v10, v10\n"
        " vmax.vx v14, v14, zero\n"
        " vmul.vx v14, v14, %3\n"
        " li t2, 524288\n"
        " vadd.vx v14, v14, t2\n"
        " vsrl.vi v14, v14, 20\n"
        " li t3, 255\n"
        " vminu.vx v14, v14, t3\n"
        :: "r"(n), "r"(aa), "r"(bb), "r"(5u) : "t0", "t2", "t3", "t6");

    printf("B: ncvt chain + vluxei8 under e8 mf2\n"); fflush(stdout);
    asm volatile(
        " mv t6, %0\n"
        VSET(32, m2)
        " vse32.v v14, (%1)\n"   /* z values in aa[] */
        VSET(16, m1)
        " vncvt.x.x.w v12, v14\n"
        VSET(8, mf2)
        " vncvt.x.x.w v15, v12\n"
        " vluxei8.v v13, (%2), v15\n"
        :: "r"(n), "r"(aa), "r"(lut) : "t0", "t6");

    printf("C: vsll.vi + vmacc.vx under e32m2\n"); fflush(stdout);
    asm volatile(
        " mv t6, %0\n" VSET(32, m2)
        " vle32.v v18, (%1)\n"
        " vsll.vi v18, v18, 2\n"
        " li t2, 3\n"
        " vmacc.vx v18, t2, v18\n"
        :: "r"(n), "r"(aa) : "t0", "t2", "t6");

    printf("D: vnmsub.vv + ncvt + e8mf2 store (finish/weighted shape)\n"); fflush(stdout);
    asm volatile(
        " mv t6, %0\n" VSET(32, m2)
        " vle32.v v18, (%1)\n"
        " vle32.v v24, (%1)\n"
        " vle32.v v28, (%1)\n"
        " vmv.v.v v30, v18\n"
        " vnmsub.vv v30, v24, v28\n"
        " li t2, 256\n"
        " vadd.vx v30, v30, t2\n"
        " vsra.vi v30, v30, 9\n"
        VSET(16, m1)
        " vncvt.x.x.w v12, v30\n"
        " vse16.v v12, (%2)\n"
        VSET(8, mf2)
        " vncvt.x.x.w v9, v8\n"
        :: "r"(n), "r"(aa), "r"(tmp) : "t0", "t2", "t6");

    printf("E: weighted shape incl. e8m1 load + double narrow + vse8 mf2\n"); fflush(stdout);
    asm volatile(
        " mv t6, %0\n" VSET(32, m2)
        " vle32.v v12, (%1)\n"
        " vmax.vx v12, v12, zero\n"
        " li t0, 255\n"
        " vmin.vx v12, v12, t0\n"
        VSET(16, m1)
        " vncvt.x.x.w v8, v12\n"
        VSET(8, mf2)
        " vncvt.x.x.w v9, v8\n"
        " vse8.v v9, (%2)\n"
        :: "r"(n), "r"(aa), "r"(dst) : "t0", "t6");

    printf("all ok\n");
    return 0;
}
