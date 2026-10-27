/* Exact-replica test of the vert-16 helper epilog: load 16 e16 rows,
 * 16 e8 dst rows at (e8,mf2,vl=16), scratch-form vwaddu/vnclipu. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static int16_t vert[16][16] __attribute__((aligned(64)));
static uint8_t dst[16][16] __attribute__((aligned(64)));
static uint8_t got[16][16] __attribute__((aligned(64)));

int main(void)
{
    unsigned st = 31337;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            st = st * 1103515245u + 12345u;
            vert[r][c] = (int16_t)((st >> 16) % 8192 - 4096);
            dst[r][c] = (uint8_t)(st >> 24);
        }
    long pv = (long)vert, pd = (long)dst, pg = (long)got, stride = 16;
    static int16_t wide[16] __attribute__((aligned(64)));
    long pw = (long)wide;
    __asm__ volatile(
        "vsetivli zero, 16, e16, m1, ta, ma\n"
        "mv t4, %[v]\n"
        "vle16.v v0, 0(t4)\n addi t4,t4,32\n vle16.v v1, 0(t4)\n addi t4,t4,32\n"
        "vle16.v v2, 0(t4)\n addi t4,t4,32\n vle16.v v3, 0(t4)\n addi t4,t4,32\n"
        "vle16.v v4, 0(t4)\n addi t4,t4,32\n vle16.v v5, 0(t4)\n addi t4,t4,32\n"
        "vle16.v v6, 0(t4)\n addi t4,t4,32\n vle16.v v7, 0(t4)\n addi t4,t4,32\n"
        "vle16.v v8, 0(t4)\n addi t4,t4,32\n vle16.v v9, 0(t4)\n addi t4,t4,32\n"
        "vle16.v v10, 0(t4)\n addi t4,t4,32\n vle16.v v11, 0(t4)\n addi t4,t4,32\n"
        "vle16.v v12, 0(t4)\n addi t4,t4,32\n vle16.v v13, 0(t4)\n addi t4,t4,32\n"
        "vle16.v v14, 0(t4)\n addi t4,t4,32\n vle16.v v15, 0(t4)\n"
        "vsetivli zero, 16, e8, mf2, ta, ma\n"
        "mv t5, %[d]\n"
        "vle8.v v16, 0(t5)\n add t0,t5,%[s]\n vle8.v v17, (t0)\n add t0,t0,%[s]\n"
        "vle8.v v18, (t0)\n add t0,t0,%[s]\n vle8.v v19, (t0)\n add t0,t0,%[s]\n"
        "vle8.v v20, (t0)\n add t0,t0,%[s]\n vle8.v v21, (t0)\n add t0,t0,%[s]\n"
        "vle8.v v22, (t0)\n add t0,t0,%[s]\n vle8.v v23, (t0)\n"
        "vwaddu.wv v24, v0, v16\n"
        "vsetivli zero, 16, e16, m1, ta, ma\n mv t6, %[w]\n vse16.v v24, 0(t6)\n"
        "vsetivli zero, 16, e8, mf2, ta, ma\n vnclipu.wi v16, v24, 0\n"
        "vwaddu.wv v24, v1, v17\n vnclipu.wi v17, v24, 0\n"
        "vwaddu.wv v24, v2, v18\n vnclipu.wi v18, v24, 0\n"
        "vwaddu.wv v24, v3, v19\n vnclipu.wi v19, v24, 0\n"
        "vwaddu.wv v24, v4, v20\n vnclipu.wi v20, v24, 0\n"
        "vwaddu.wv v24, v5, v21\n vnclipu.wi v21, v24, 0\n"
        "vwaddu.wv v24, v6, v22\n vnclipu.wi v22, v24, 0\n"
        "vwaddu.wv v24, v7, v23\n vnclipu.wi v23, v24, 0\n"
        "mv t6, %[g]\n"
        "vse8.v v16, 0(t6)\n addi t6,t6,16\n vse8.v v17, 0(t6)\n addi t6,t6,16\n"
        "vse8.v v18, 0(t6)\n addi t6,t6,16\n vse8.v v19, 0(t6)\n addi t6,t6,16\n"
        "vse8.v v20, 0(t6)\n addi t6,t6,16\n vse8.v v21, 0(t6)\n addi t6,t6,16\n"
        "vse8.v v22, 0(t6)\n addi t6,t6,16\n vse8.v v23, 0(t6)\n"
        "add t5, t5, %[s]\n"
        "mv t0, t5\n"
        "vle8.v v16, (t0)\n add t0,t0,%[s]\n vle8.v v17, (t0)\n add t0,t0,%[s]\n"
        "vle8.v v18, (t0)\n add t0,t0,%[s]\n vle8.v v19, (t0)\n add t0,t0,%[s]\n"
        "vle8.v v20, (t0)\n add t0,t0,%[s]\n vle8.v v21, (t0)\n add t0,t0,%[s]\n"
        "vle8.v v22, (t0)\n add t0,t0,%[s]\n vle8.v v23, (t0)\n"
        "vwaddu.wv v24, v8, v16\n vnclipu.wi v16, v24, 0\n"
        "vwaddu.wv v24, v9, v17\n vnclipu.wi v17, v24, 0\n"
        "vwaddu.wv v24, v10, v18\n vnclipu.wi v18, v24, 0\n"
        "vwaddu.wv v24, v11, v19\n vnclipu.wi v19, v24, 0\n"
        "vwaddu.wv v24, v12, v20\n vnclipu.wi v20, v24, 0\n"
        "vwaddu.wv v24, v13, v21\n vnclipu.wi v21, v24, 0\n"
        "vwaddu.wv v24, v14, v22\n vnclipu.wi v22, v24, 0\n"
        "vwaddu.wv v24, v15, v23\n vnclipu.wi v23, v24, 0\n"
        "addi t6, t6, -128\n"
        "vse8.v v16, 0(t6)\n addi t6,t6,16\n vse8.v v17, 0(t6)\n addi t6,t6,16\n"
        "vse8.v v18, 0(t6)\n addi t6,t6,16\n vse8.v v19, 0(t6)\n addi t6,t6,16\n"
        "vse8.v v20, 0(t6)\n addi t6,t6,16\n vse8.v v21, 0(t6)\n addi t6,t6,16\n"
        "vse8.v v22, 0(t6)\n addi t6,t6,16\n vse8.v v23, 0(t6)\n"
"mv t6, %[g]\n"
        "vse8.v v16, 0(t6)\n addi t6,t6,16\n vse8.v v17, 0(t6)\n addi t6,t6,16\n"
        "vse8.v v18, 0(t6)\n addi t6,t6,16\n vse8.v v19, 0(t6)\n addi t6,t6,16\n"
        "vse8.v v20, 0(t6)\n addi t6,t6,16\n vse8.v v21, 0(t6)\n addi t6,t6,16\n"
        "vse8.v v22, 0(t6)\n addi t6,t6,16\n vse8.v v23, 0(t6)\n"
        :: [v]"r"(pv), [d]"r"(pd), [g]"r"(pg), [s]"r"(stride), [w]"r"(pw)
        : "t0","t4","t5","t6","v0","v1","v2","v3","v4","v5","v6","v7","v8",
          "v9","v10","v11","v12","v13","v14","v15","v16","v17","v18","v19",
          "v20","v21","v22","v23","v24","v25","memory");

    printf("v24[0..3] = %d %d %d %d (exp %d %d %d %d)\n",
           wide[0], wide[1], wide[2], wide[3],
           vert[0][0] + dst[0][0], vert[0][1] + dst[0][1],
           vert[0][2] + dst[0][2], vert[0][3] + dst[0][3]);
    int bad = 0;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            int v = vert[r][c] + dst[r][c];
            v = v < 0 ? 0 : v > 255 ? 255 : v;
            if (got[r][c] != v) {
                bad++;
                if (bad <= 5)
                    printf("(%d,%d) vert=%d dst=%d got=%d exp=%d\n",
                           r, c, vert[r][c], dst[r][c], got[r][c], v);
            }
        }
    printf("replica diffs=%d\n", bad);
    return 0;
}
