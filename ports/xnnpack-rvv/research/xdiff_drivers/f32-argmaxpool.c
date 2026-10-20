// xdiff driver: f32-argmaxpool (9p8x signature). One output pixel per case;
// pooling windows of 1..10 elements over 1..20 channels, values drawn from the
// float specials (NaN, +-0, +-inf, subnormals) and random finite numbers.
#include <stdlib.h>

void KERNEL(size_t output_pixels, size_t pooling_elements, size_t channels, const float** input,
            size_t input_offset, size_t input_pixel_stride, float* output, uint32_t* index,
            size_t input_increment, size_t output_increment, size_t index_increment);

#define MAXK 10
#define MAXC 20
#define PAD 64

int main(void) {
  static float rows[MAXK][MAXC + PAD];
  static float out[MAXC + PAD];
  static uint32_t idx[MAXC + PAD];
  const float* ptrs[MAXK + 8];
  uint64_t seed = 1;
  int id = 0;
  for (size_t k = 1; k <= MAXK; k++) {
    for (size_t c = 1; c <= MAXC; c++) {
      for (int trial = 0; trial < 40; trial++, id++) {
        for (size_t p = 0; p < k; p++) {
          for (size_t j = 0; j < c + PAD; j++) {
            const uint64_t r = xdiff_rand(&seed);
            rows[p][j] = (r & 3) == 0 ? xdiff_u2f(xdiff_f32_specials[(r >> 8) % XDIFF_NSPECIALS])
                                      : (float) ((int) (r >> 40) % 9 - 4);  // small ints: many ties
          }
          ptrs[p] = rows[p];
        }
        memset(out, 0xA5, sizeof out);
        memset(idx, 0xA5, sizeof idx);
        KERNEL(1, k, c, ptrs, 0, 0, out, idx, 0, 0, 0);
        printf("case %d k=%zu c=%zu in=", id, k, c);
        for (size_t p = 0; p < k; p++) { xdiff_hex(rows[p], c * 4); printf("|"); }
        printf(" -> ");
        xdiff_hex(out, c * 4);
        printf(" ");
        xdiff_hex(idx, c * 4);
        printf("\n");
      }
    }
  }
  return 0;
}
