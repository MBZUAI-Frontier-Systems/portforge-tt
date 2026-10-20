// Differential driver: bit-exact check of xnn_qs8_vlrelu_ukernel__rvv_pf1
// against the family's scalar spec kernel (scalar_select_u1) under QEMU, over
// a params sweep produced by the REAL init function (same configs as
// params.json) and batch sizes around every power of two and the vl boundary.
//
// Build (toolchain container):
//   riscv64-linux-gnu-gcc -O2 -march=rv64gcv_zba_zbb_zbs -static \
//     -I/pf/src/xnnpack-saltyrn -I/pf/src/xnnpack-saltyrn/include \
//     -I/pf/repo/ports/xnnpack-rvv/research/qs8_vadd_smoke \
//     difftest.c /pf/src/xnnpack-saltyrn/src/microparams-init.c \
//     /pf/src/xnnpack-saltyrn/src/qs8-vlrelu/gen/qs8-vlrelu-scalar-select-u1.c \
//     /pf/repo/ports/xnnpack-rvv/kernels/qs8-vlrelu/kernel.c \
//     -o /pf/build/difftest-vlrelu
// Run: qemu-riscv64 -cpu rv64,vlen=512,v=true,vext_spec=v1.0 /pf/build/difftest-vlrelu
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"

void xnn_qs8_vlrelu_ukernel__scalar_select_u1(
    size_t, const int8_t*, int8_t*, const struct xnn_qs8_lrelu_params*);
void xnn_qs8_vlrelu_ukernel__rvv_pf1(
    size_t, const int8_t*, int8_t*, const struct xnn_qs8_lrelu_params*);
void xnn_qs8_vlrelu_ukernel__rvv_u1v(
    size_t, const int8_t*, int8_t*, const struct xnn_qs8_lrelu_params*);
void xnn_qs8_vlrelu_ukernel__rvv_u2v(
    size_t, const int8_t*, int8_t*, const struct xnn_qs8_lrelu_params*);

// (input_scale, input_zp, output_scale, output_zp, negative_slope):
// covers every dispatch regime of the kernel, all through the real init.
static const struct {
  float is, os, slope;
  int32_t izp, ozp;
  const char* regime;
} configs[] = {
  {1.0f, 1.0f, 1.0f, 0, 0, "A1 vssub (m=256, s=0)"},
  {1.0f, 1.0f, 1.0f, -128, 127, "A2 i16 (m=256, s=-255)"},
  {1.0f, 1.0f, 1.0f, 127, -128, "A2 i16 (m=256, s=+255)"},
  {0.5f, 0.7f, 1.0f, 1, -1, "C direct-i16 (m=183)"},
  {1.0f, 4.0f, 1.0f, 0, 0, "C direct-i16 (m=64)"},
  {1.0f, 256.0f, 1.0f, 0, 0, "C direct-i16 (m=1)"},
  {0.0123f, 0.0456f, 1.0f, -128, 127, "C direct-i16 (m=69, zps extreme)"},
  {1.0f, 2.0f, 1.0f, 0, 0, "C direct-i16 (m=128 boundary)"},
  {4.0f, 1.0f, 1.0f, 0, 0, "D1 shift (m=1024)"},
  {16.0f, 1.0f, 1.0f, 5, 3, "D1 shift (m=4096)"},
  {128.0f, 1.0f, 1.0f, 0, 0, "D1 shift (m=32768, e=7)"},
  {2.0f, 1.0f, 1.0f, 100, -100, "D1 shift (m=512)"},
  {1.0f, 2.0f, 1.0f, -128, 127, "D2 vsmul (m=128, C-ineligible zps)"},
  {127.99f, 1.0f, 1.0f, -5, 7, "D2 vsmul (m=32765)"},
  {1.5f, 1.0f, 1.0f, 0, 0, "D2 vsmul (m=384)"},
  {1.7f, 1.1f, 1.0f, -3, 2, "D2 vsmul (m=396)"},
  {0.5f, 0.7f, 0.5f, 1, -1, "GENERAL (bench: pos 183, neg 91)"},
  {1.0f, 1.0f, 0.5f, -128, 127, "GENERAL (256/128)"},
  {1.0f, 1.0f, 0.5f, 127, -128, "GENERAL (256/128 swapped)"},
  {4.0f, 1.0f, 0.5f, 0, 0, "GENERAL (1024/512)"},
  {64.0f, 1.0f, 0.5f, 0, 0, "GENERAL (16384/8192)"},
  {2.0f, 1.0f, 0.5f, 100, -100, "GENERAL (512/256)"},
  {0.0123f, 0.0456f, 0.5f, -128, 127, "GENERAL (69/35)"},
  {16.0f, 1.0f, 0.5f, 5, 3, "GENERAL (4096/2048)"},
  {1.0f, 1.0f, 0.01f, 0, 0, "GENERAL (256/3, gtest slope)"},
  {1.0f, 1.0f, 0.3f, 0, 0, "GENERAL (256/77, gtest slope)"},
  {1.0f, 1.0f, 1.3f, 0, 0, "GENERAL (256/333, gtest slope)"},
  {0.5f, 0.7f, -1.0f, 1, -1, "GENERAL (183/-183, neg mult)"},
  {0.5f, 0.7f, -0.01f, 1, -1, "GENERAL (183/-2)"},
  {0.5f, 0.7f, 1.3f, 1, -1, "GENERAL (183/238, slope>1)"},
  {64.0f, 1.0f, 2.0f, 0, 0, "I32 (16384/32768)"},
  {128.0f, 1.0f, -0.999f, 0, 0, "I32 (32768/-32737)"},
  {128.0f, 1.0f, 0.3f, 5, -7, "I32 (32768/9830, nonzero zps)"},
};
#define NCONFIGS (sizeof(configs) / sizeof(configs[0]))

static const size_t batches[] = {
  1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
  130, 131, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4096, 12345,
};

int main(void) {
  const size_t maxn = 65536;
  int8_t* in = malloc(maxn);
  int8_t* out = malloc(maxn);
  int8_t* ref = malloc(maxn);
  uint64_t seed = 0x9e3779b97f4a7c15ull;
  for (size_t i = 0; i < maxn; i++) {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    in[i] = (int8_t)(seed >> 56);
  }
  // pin the int8 edges into the pattern
  in[0] = -128; in[1] = 127; in[2] = 0; in[3] = -1; in[4] = 1;

  size_t fails = 0, checks = 0;
  for (size_t c = 0; c < NCONFIGS; c++) {
    union xnn_unary_params op = {0};
    op.leaky_relu.negative_slope = configs[c].slope;
    union xnn_unary_uparams up;
    const struct xnn_quantization_params iq = {configs[c].izp, configs[c].is};
    const struct xnn_quantization_params oq = {configs[c].ozp, configs[c].os};
    xnn_init_qs8_lrelu_scalar_params(&up, &op, &iq, &oq);
    const struct xnn_qs8_lrelu_params* p = &up.qs8_lrelu;
    for (size_t b = 0; b < sizeof(batches) / sizeof(batches[0]); b++) {
      const size_t n = batches[b];
      memset(out, 0x5A, n); memset(ref, 0xA5, n);
      xnn_qs8_vlrelu_ukernel__scalar_select_u1(n, in, ref, p);
      xnn_qs8_vlrelu_ukernel__rvv_pf1(n, in, out, p);
      checks++;
      if (memcmp(out, ref, n) != 0) {
        fails++;
        size_t first = 0;
        while (first < n && out[first] == ref[first]) first++;
        printf("FAIL cfg %zu (%s: izp %" PRId32 " pos %" PRId32 " neg %" PRId32
               " ozp %" PRId32 ") batch %zu at [%zu]: got %d want %d\n",
               c, configs[c].regime, configs[c].izp,
               p->scalar.positive_multiplier, p->scalar.negative_multiplier,
               configs[c].ozp, n, first, out[first], ref[first]);
      }
    }
  }
  // spot-check upstream variants at the bench config for the REPORT
  union xnn_unary_params op = {0};
  op.leaky_relu.negative_slope = 0.5f;
  union xnn_unary_uparams up;
  const struct xnn_quantization_params iq = {1, 0.5f};
  const struct xnn_quantization_params oq = {-1, 0.7f};
  xnn_init_qs8_lrelu_scalar_params(&up, &op, &iq, &oq);
  xnn_qs8_vlrelu_ukernel__scalar_select_u1(4096, in, ref, &up.qs8_lrelu);
  xnn_qs8_vlrelu_ukernel__rvv_u1v(4096, in, out, &up.qs8_lrelu);
  printf("bench cfg: u1v vs scalar %s; ",
         memcmp(out, ref, 4096) == 0 ? "MATCH" : "DIFFER");
  xnn_qs8_vlrelu_ukernel__rvv_u2v(4096, in, out, &up.qs8_lrelu);
  printf("u2v vs scalar %s\n",
         memcmp(out, ref, 4096) == 0 ? "MATCH" : "DIFFER");

  printf("%s: %zu checks, %zu failures\n", fails ? "DIFFTEST_FAIL" : "DIFFTEST_OK",
         checks, fails);
  return fails != 0;
}
