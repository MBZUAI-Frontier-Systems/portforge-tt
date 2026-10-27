/* Authoritative dct-kernel check: asm e16_x16 kernel at vl=16 vs the C
 * inv_dct_1d (dav1d_tx1d_fns[2][DCT]) applied per lane to the same input. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

extern void dav1d_inv_dct_e16_x16_rvv(void);
typedef void (*itx_1d_fn)(int32_t *c, int stride, int clip_min, int clip_max);
extern const itx_1d_fn dav1d_tx1d_fns[6][4];

static int16_t in[16][16] __attribute__((aligned(64)));
static int16_t out[16][16] __attribute__((aligned(64)));

#define VLD "vle16.v v0,0(t4)\n addi t4,t4,32\n vle16.v v1,0(t4)\n addi t4,t4,32\n vle16.v v2,0(t4)\n addi t4,t4,32\n vle16.v v3,0(t4)\n addi t4,t4,32\n vle16.v v4,0(t4)\n addi t4,t4,32\n vle16.v v5,0(t4)\n addi t4,t4,32\n vle16.v v6,0(t4)\n addi t4,t4,32\n vle16.v v7,0(t4)\n addi t4,t4,32\n vle16.v v8,0(t4)\n addi t4,t4,32\n vle16.v v9,0(t4)\n addi t4,t4,32\n vle16.v v10,0(t4)\n addi t4,t4,32\n vle16.v v11,0(t4)\n addi t4,t4,32\n vle16.v v12,0(t4)\n addi t4,t4,32\n vle16.v v13,0(t4)\n addi t4,t4,32\n vle16.v v14,0(t4)\n addi t4,t4,32\n vle16.v v15,0(t4)\n"
#define VST "vse16.v v0,0(t5)\n addi t5,t5,32\n vse16.v v1,0(t5)\n addi t5,t5,32\n vse16.v v2,0(t5)\n addi t5,t5,32\n vse16.v v3,0(t5)\n addi t5,t5,32\n vse16.v v4,0(t5)\n addi t5,t5,32\n vse16.v v5,0(t5)\n addi t5,t5,32\n vse16.v v6,0(t5)\n addi t5,t5,32\n vse16.v v7,0(t5)\n addi t5,t5,32\n vse16.v v8,0(t5)\n addi t5,t5,32\n vse16.v v9,0(t5)\n addi t5,t5,32\n vse16.v v10,0(t5)\n addi t5,t5,32\n vse16.v v11,0(t5)\n addi t5,t5,32\n vse16.v v12,0(t5)\n addi t5,t5,32\n vse16.v v13,0(t5)\n addi t5,t5,32\n vse16.v v14,0(t5)\n addi t5,t5,32\n vse16.v v15,0(t5)\n"

static void run(long vl)
{
    unsigned st = 12345;
    for (int k = 0; k < 16; k++)
        for (int j = 0; j < 16; j++) {
            st = st * 1103515245u + 12345u;
            in[k][j] = (int16_t)((st >> 16) % 4096 - 2048);  /* dense mixed-sign */
        }
    register long a_in __asm__("a0") = (long)in;
    register long a_out __asm__("a1") = (long)out;
    register long a_vl __asm__("a2") = vl;
    register long a_kfn __asm__("a3") = (long)dav1d_inv_dct_e16_x16_rvv;
    __asm__ volatile(
        "vsetivli zero, 16, e16, m1, ta, ma\n"
        "mv t4, a0\n"
        VLD
        "vsetvli zero, a2, e16, m1, ta, ma\n"
        "jalr t0, a3\n"
        "vsetivli zero, 16, e16, m1, ta, ma\n"
        "mv t5, a1\n"
        VST
        : "+r"(a_in), "+r"(a_out)
        : "r"(a_vl), "r"(a_kfn)
        : "t0","t1","t2","t3","t4","t5",
          "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",
          "v12","v13","v14","v15","v16","v17","v18","v19","v20","v21",
          "v22","v23","v24","v25","v26","v27","v28","v29","v30","v31",
          "memory");
}

int main(void)
{
    run(16);
    int16_t expect[16][16];
    for (int lane = 0; lane < 16; lane++) {
        int32_t c[16];
        for (int i = 0; i < 16; i++) c[i] = in[i][lane]; /* column lane */
        dav1d_tx1d_fns[2][0](c, 1, INT16_MIN, INT16_MAX);
        for (int o = 0; o < 16; o++) {
            int32_t v = c[o];
            expect[o][lane] = (int16_t)(v < INT16_MIN ? INT16_MIN :
                                        v > INT16_MAX ? INT16_MAX : v);
        }
    }
    for (long vl = 16; vl >= 8; vl -= 8) {
        if (vl == 8) run(vl);   /* rerun: overwrites in/out identically */
        int bad = 0, first = -1;
        for (int o = 0; o < 16; o++)
            for (int lane = 0; lane < 16; lane++)
                if (out[o][lane] != expect[o][lane]) {
                    bad++;
                    if (first < 0) first = o * 16 + lane;
                }
        printf("vl=%ld diffs=%d\n", vl, bad);
        if (bad && vl == 16) {
            printf("out reg (row) = transform output for one basis column;\n");
            for (int o = 0; o < 16; o++) {
                int rb = 0;
                for (int lane = 0; lane < 16; lane++) rb += out[o][lane] != expect[o][lane];
                if (rb) {
                    printf("v%2d asm :", o);
                    for (int lane = 0; lane < 16; lane++) printf(" %4d", out[o][lane]);
                    printf("\n    exp :");
                    for (int lane = 0; lane < 16; lane++) printf(" %4d", expect[o][lane]);
                    printf("\n");
                }
            }
        }
    }
    return 0;
}
