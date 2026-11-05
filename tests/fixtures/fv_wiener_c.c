/* Test fixture for portforge_tt/verify/cbmc: dav1d's own C wiener filter compiled for
 * RV64GCV (GCC auto-vectorizes it), lifted and proved equal to its source. */
#define BITDEPTH 8
#include "src/looprestoration_tmpl.c"

void fv_test_wiener(pixel *p, const ptrdiff_t stride, const pixel (*left)[4], const pixel *lpf, const int w,
                    int h, const LooprestorationParams *const params, const enum LrEdgeFlags edges)
{
    wiener_c(p, stride, left, lpf, w, h, params, edges);
}

int main(void) { return 0; }
