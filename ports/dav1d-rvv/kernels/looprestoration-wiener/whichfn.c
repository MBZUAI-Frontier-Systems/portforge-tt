#include <stdio.h>
typedef void (*lrfn)();
extern void dav1d_loop_restoration_dsp_init_8bpc(void *c, int bpc);
int main(void) {
    lrfn fns[5];
    dav1d_loop_restoration_dsp_init_8bpc(fns, 8);
    for (int i = 0; i < 5; i++) printf("fns[%d] = %p\n", i, fns[i]);
    return 0;
}
