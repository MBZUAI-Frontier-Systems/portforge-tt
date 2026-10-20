// xdiff driver: f32-f16-vcvt. Every float special plus a dense sweep of bit
// patterns around fp16 rounding boundaries, in batches of 1..67 elements.
void KERNEL(size_t batch, const float* input, uint16_t* output, const void* params);

int main(void) {
  static float in[4096 + 64];
  static uint16_t out[4096 + 64];
  uint64_t seed = 7;
  int id = 0;
  for (int round = 0; round < 400; round++) {
    const size_t n = 1 + round % 67;
    for (size_t i = 0; i < n + 64; i++) {
      const uint64_t r = xdiff_rand(&seed);
      uint32_t bits;
      switch (r & 3) {
        case 0: bits = xdiff_f32_specials[(r >> 8) % XDIFF_NSPECIALS]; break;
        case 1: bits = (uint32_t) (r >> 16); break;                                  // any pattern
        case 2: bits = (0x33000000u + (uint32_t) ((r >> 16) % 0x14800000u)) | ((r >> 63) << 31); break;  // fp16 range
        default: bits = ((uint32_t) (r >> 20) & 0xffffe000u) | 0x1000u;             // exact ties
      }
      in[i] = xdiff_u2f(bits);
    }
    memset(out, 0xA5, sizeof out);
    KERNEL(n * sizeof(float), in, out, NULL);
    printf("case %d n=%zu in=", id++, n);
    xdiff_hex(in, n * 4);
    printf(" -> ");
    xdiff_hex(out, (n + 4) * 2);  // 4 extra elements catch writes past the end
    printf("\n");
  }
  return 0;
}
