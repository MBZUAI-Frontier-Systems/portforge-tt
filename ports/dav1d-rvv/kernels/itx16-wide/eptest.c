/* The exact vert-epilog sequence at (e8, mf2, vl=16): vle8 dst rows,
 * vwaddu.wv via scratch, vnclipu.wi, vse8. Compared against C. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static int16_t vertout[16][16] __attribute__((aligned(64)));
static uint8_t dst[16][16] __attribute__((aligned(64)));
static uint8_t got[16][16] __attribute__((aligned(64)));
static int16_t wide[16] __attribute__((aligned(64)));
static int16_t reg0[16] __attribute__((aligned(64)));
static int16_t reg8[16] __attribute__((aligned(64)));
static int state[2];

int main(void)
{
    unsigned st = 777;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            st = st * 1103515245u + 12345u;
            vertout[r][c] = (int16_t)((st >> 16) % 65536 - 32768) / 8; /* ±4096 */
            dst[r][c] = (uint8_t)(st >> 24);
        }
    register long pv __asm__("a0") = (long)vertout;
    register long pd __asm__("a1") = (long)dst;
    register long pg __asm__("a2") = (long)got;
    register long pw __asm__("a4") = (long)wide;
    register long pq __asm__("a5") = (long)state;
    long stride = 16;
    register long ps __asm__("a3") = stride;
    __asm__ volatile(
        "mv t5, %[d]\n"
        "vsetivli zero, 16, e16, m1, ta, ma\n"
        "mv t4, %[v]\n"
        "vle16.v v0, 0(t4)\n"
        "addi t4, t4, 32\n vle16.v v1, 0(t4)\n addi t4, t4, 32\n vle16.v v2, 0(t4)\n"
        "addi t4, t4, 32\n vle16.v v3, 0(t4)\n addi t4, t4, 32\n vle16.v v4, 0(t4)\n"
        "addi t4, t4, 32\n vle16.v v5, 0(t4)\n addi t4, t4, 32\n vle16.v v6, 0(t4)\n"
        "addi t4, t4, 32\n vle16.v v7, 0(t4)\n addi t4, t4, 32\n vle16.v v8, 0(t4)\n"
        "addi t4, t4, 32\n vle16.v v9, 0(t4)\n addi t4, t4, 32\n vle16.v v10, 0(t4)\n"
        "addi t4, t4, 32\n vle16.v v11, 0(t4)\n addi t4, t4, 32\n vle16.v v12, 0(t4)\n"
        "addi t4, t4, 32\n vle16.v v13, 0(t4)\n addi t4, t4, 32\n vle16.v v14, 0(t4)\n"
        "addi t4, t4, 32\n vle16.v v15, 0(t4)\n"
        "vsetivli zero, 16, e8, mf2, ta, ma\n"
        "vle8.v v16, (t5)\n"
        "add t0, t5, %[s]\n"
        "vle8.v v17, (t0)\n"
        "add t0, t0, %[s]\n"
        "vle8.v v18, (t0)\n"
        "add t0, t0, %[s]\n"
        "vle8.v v19, (t0)\n"
        "add t0, t0, %[s]\n"
        "vle8.v v20, (t0)\n"
        "add t0, t0, %[s]\n"
        "vle8.v v21, (t0)\n"
        "add t0, t0, %[s]\n"
        "vle8.v v22, (t0)\n"
        "add t0, t0, %[s]\n"
        "vle8.v v23, (t0)\n"
        "vwaddu.wv v0, v0, v16\n"
        "vwaddu.wv v1, v1, v17\n"
        "vwaddu.wv v2, v2, v18\n"
        "vwaddu.wv v3, v3, v19\n"
        "vwaddu.wv v4, v4, v20\n"
        "vwaddu.wv v5, v5, v21\n"
        "vwaddu.wv v6, v6, v22\n"
        "vwaddu.wv v7, v7, v23\n"
        "vsetivli zero, 16, e16, m1, ta, ma\n"
        "vsetivli zero, 16, e8, mf2, ta, ma\n"
        "vnclipu.wi v16, v0, 0\n"
        "vnclipu.wi v17, v1, 0\n"
        "vnclipu.wi v18, v2, 0\n"
        "vnclipu.wi v19, v3, 0\n"
        "vnclipu.wi v20, v4, 0\n"
        "vnclipu.wi v21, v5, 0\n"
        "vnclipu.wi v22, v6, 0\n"
        "vnclipu.wi v23, v7, 0\n"
        "vse8.v v16, (t5)\n"
        "add t5, t5, %[s]\n"
        "vse8.v v17, (t5)\n"
        "add t5, t5, %[s]\n"
        "vse8.v v18, (t5)\n"
        "add t5, t5, %[s]\n"
        "vse8.v v19, (t5)\n"
        "add t5, t5, %[s]\n"
        "vse8.v v20, (t5)\n"
        "add t5, t5, %[s]\n"
        "vse8.v v21, (t5)\n"
        "add t5, t5, %[s]\n"
        "vse8.v v22, (t5)\n"
        "add t5, t5, %[s]\n"
        "vse8.v v23, (t5)\n"
        "csrr t0, vl\n"
        "sw t0, 0(%[q])\n"
        "csrr t0, vtype\n"
        "sw t0, 4(%[q])\n"
        "vsetivli zero, 16, e16, m1, ta, ma\n"
        "mv t6, %[w]\n"
        "vse16.v v0, 0(t6)\n"
        "addi t6, t6, 64\n"
        "vse16.v v8, 0(t6)\n"
        :
        : [d]"r"(pd), [v]"r"(pv), [s]"r"(ps), [q]"r"(pq), [w]"r"(pw)
        : "t0","t4","t5","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9",
          "v10","v11","v12","v13","v14","v15","v16","v17","v18","v19","v20",
          "v21","v22","v23","v24","v25","memory");

    int bad = 0;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            int v = vertout[r][c] + dst[r][c];
            v = v < 0 ? 0 : v > 255 ? 255 : v;
            if (got[r][c] != v) {
                bad++;
                if (bad <= 6)
                    printf("(%d,%d) vert=%d dst=%d got=%d exp=%d\n",
                           r, c, vertout[r][c], dst[r][c], got[r][c], v);
            }
        }
    printf("vl=%d vtype=0x%x\n", state[0], state[1]);
    printf("v0 (vert+dst sums, exp 1371 875 1427 ...):");
    for (int i = 0; i < 8; i++) printf(" %d", reg0[i]);
    printf("\nv8:");
    for (int i = 0; i < 8; i++) printf(" %d", reg8[i]);
    printf("\n");
    printf("epilog diffs=%d\n", bad);
    for (int r = 0; r < 4; r++) {
        printf("row%02d got:", r);
        for (int c = 0; c < 16; c++) printf(" %3d", got[r][c]);
        printf("\n     exp:");
        for (int c = 0; c < 16; c++) {
            int v = vertout[r][c] + dst[r][c];
            v = v < 0 ? 0 : v > 255 ? 255 : v;
            printf(" %3d", v);
        }
        printf("\n     dst:");
        for (int c = 0; c < 16; c++) printf(" %3d", dst[r][c]);
        printf("\n");
    }
    return 0;
}
