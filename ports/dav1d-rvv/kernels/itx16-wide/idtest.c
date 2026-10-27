/* vsmul/vsra/vaadd identity-scale chain vs the C reference semantics. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static int16_t asmout[16] __attribute__((aligned(64)));

static int c_model(int x)
{
    /* inv_identity16_1d_c */
    int y = 2 * x + ((x * 1697 + 1024) >> 11);
    /* driver: (tmp + rnd) >> shift, shift=2, rnd=2 */
    return (y + 2) >> 2;
}

int main(void)
{
    int16_t in[16] __attribute__((aligned(64)));
    long pa = (long)in, po = (long)asmout;
    int bad = 0, shown = 0;
    for (int base = -32768; base < 32768; base += 16) {
        for (int i = 0; i < 16; i++) in[i] = (int16_t)(base + i);
        __asm__ volatile(
            "vsetivli zero, 16, e16, m1, ta, ma\n"
            "csrw vxrm, zero\n"
            "mv t4, %[a]\n"
            "vle16.v v0, 0(t4)\n"
            "li t1, 2*(5793-4096)*8\n"
            "vsmul.vx v16, v0, t1\n"
            "vsra.vi v16, v16, 1\n"
            "vaadd.vv v0, v0, v16\n"
            "mv t5, %[o]\n"
            "vse16.v v0, 0(t5)\n"
            :: [a]"r"(pa), [o]"r"(po)
            : "t1","t4","t5","v0","v16","memory");
        for (int i = 0; i < 16; i++) {
            int x = base + i;
            int exp = c_model(x);
            if (asmout[i] != exp) {
                bad++;
                if (shown++ < 8)
                    printf("x=%6d asm=%6d c=%6d\n", x, asmout[i], exp);
            }
        }
    }
    printf("identity-scale chain diffs=%d\n", bad);
    return 0;
}
