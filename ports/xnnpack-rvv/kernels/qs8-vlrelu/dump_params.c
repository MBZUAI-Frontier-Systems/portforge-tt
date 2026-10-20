// Prints the exact param structs xnn_init_qs8_lrelu_scalar_params produces
// for a sweep of (input_scale, input_zp, output_scale, output_zp,
// negative_slope) quantization/op configs, as the JSON list gate 2
// (evaluate.verify configs mode) consumes.
//
// The init function is the real one, compiled from the pinned tree: nothing
// here invents parameter values. Build (inside the toolchain container; the
// pthreadpool.h stub is the one the qs8_vadd_smoke build uses):
//
//   gcc-14 -O2 -ffunction-sections -fdata-sections \
//     -I/pf/repo/ports/xnnpack-rvv/research/qs8_vadd_smoke \
//     -I/pf/src/xnnpack-saltyrn -I/pf/src/xnnpack-saltyrn/include \
//     /pf/repo/ports/xnnpack-rvv/kernels/qs8-vlrelu/dump_params.c \
//     /pf/src/xnnpack-saltyrn/src/microparams-init.c -lm -Wl,--gc-sections \
//     -o /pf/build/dump_params_vlrelu
//   /pf/build/dump_params_vlrelu > .../kernels/qs8-vlrelu/params.json
//
// The configs cover every dispatch regime of the kernel on the init's whole
// asserted domain (positive_multiplier in [1, 32768], negative_multiplier in
// [-32767, 32768] \ {0}, int8-range zero points): slope 1.0 (the pure
// requantize ladder), the benchmark's slope 0.5 with the benchmark's own
// quantization, the gtest slopes 0.01/0.3/1.3, negative slopes (negative
// multipliers), and the multiplier-32768 corner on both sides.
#include <inttypes.h>
#include <stdio.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"

static const struct {
  float input_scale;
  int32_t input_zero_point;
  float output_scale;
  int32_t output_zero_point;
  float negative_slope;
  const char* regime;
} configs[] = {
  {1.0f, 0, 1.0f, 0, 1.0f, "A1 vssub (m=256, s=0)"},
  {1.0f, -128, 1.0f, 127, 1.0f, "A2 i16 (m=256, s=-255)"},
  {1.0f, 127, 1.0f, -128, 1.0f, "A2 i16 (m=256, s=+255)"},
  {0.5f, 1, 0.7f, -1, 1.0f, "C direct-i16 (m=183)"},
  {1.0f, 0, 4.0f, 0, 1.0f, "C direct-i16 (m=64)"},
  {1.0f, 0, 256.0f, 0, 1.0f, "C direct-i16 (m=1)"},
  {0.0123f, -128, 0.0456f, 127, 1.0f, "C direct-i16 (m=69, zps extreme)"},
  {1.0f, 0, 2.0f, 0, 1.0f, "C direct-i16 (m=128 boundary)"},
  {4.0f, 0, 1.0f, 0, 1.0f, "D1 shift (m=1024)"},
  {16.0f, 5, 1.0f, 3, 1.0f, "D1 shift (m=4096)"},
  {128.0f, 0, 1.0f, 0, 1.0f, "D1 shift (m=32768, e=7)"},
  {2.0f, 100, 1.0f, -100, 1.0f, "D1 shift (m=512)"},
  {1.0f, -128, 2.0f, 127, 1.0f, "D2 vsmul (m=128, C-ineligible zps)"},
  {127.99f, -5, 1.0f, 7, 1.0f, "D2 vsmul (m=32765)"},
  {1.5f, 0, 1.0f, 0, 1.0f, "D2 vsmul (m=384)"},
  {1.7f, -3, 1.1f, 2, 1.0f, "D2 vsmul (m=396)"},
  {0.5f, 1, 0.7f, -1, 0.5f, "GENERAL (bench: pos 183, neg 91)"},
  {1.0f, -128, 1.0f, 127, 0.5f, "GENERAL (256/128)"},
  {1.0f, 127, 1.0f, -128, 0.5f, "GENERAL (256/128 swapped)"},
  {4.0f, 0, 1.0f, 0, 0.5f, "GENERAL (1024/512)"},
  {64.0f, 0, 1.0f, 0, 0.5f, "GENERAL (16384/8192)"},
  {2.0f, 100, 1.0f, -100, 0.5f, "GENERAL (512/256)"},
  {0.0123f, -128, 0.0456f, 127, 0.5f, "GENERAL (69/35)"},
  {16.0f, 5, 1.0f, 3, 0.5f, "GENERAL (4096/2048)"},
  {1.0f, 0, 1.0f, 0, 0.01f, "GENERAL (256/3, gtest slope)"},
  {1.0f, 0, 1.0f, 0, 0.3f, "GENERAL (256/77, gtest slope)"},
  {1.0f, 0, 1.0f, 0, 1.3f, "GENERAL (256/333, gtest slope)"},
  {0.5f, 1, 0.7f, -1, -1.0f, "GENERAL (183/-183, neg mult)"},
  {0.5f, 1, 0.7f, -1, -0.01f, "GENERAL (183/-2)"},
  {0.5f, 1, 0.7f, -1, 1.3f, "GENERAL (183/238, slope>1)"},
  {64.0f, 0, 1.0f, 0, 2.0f, "I32 (16384/32768)"},
  {128.0f, 0, 1.0f, 0, -0.999f, "I32 (32768/-32737)"},
  {128.0f, 5, 1.0f, -7, 0.3f, "I32 (32768/9830, nonzero zps)"},
};
#define NCONFIGS (sizeof(configs) / sizeof(configs[0]))

int main(void) {
  printf("[\n");
  for (size_t i = 0; i < NCONFIGS; i++) {
    union xnn_unary_params op = {0};
    op.leaky_relu.negative_slope = configs[i].negative_slope;
    union xnn_unary_uparams uparams;
    const struct xnn_quantization_params input_quantization = {
      configs[i].input_zero_point, configs[i].input_scale,
    };
    const struct xnn_quantization_params output_quantization = {
      configs[i].output_zero_point, configs[i].output_scale,
    };
    xnn_init_qs8_lrelu_scalar_params(&uparams, &op, &input_quantization,
                                     &output_quantization);
    printf("  {\"input_zero_point\": %" PRId32 ", \"positive_multiplier\": %" PRId32
           ", \"negative_multiplier\": %" PRId32 ", \"output_zero_point\": %" PRId32
           "}%s\n",
           (int32_t) uparams.qs8_lrelu.scalar.input_zero_point,
           uparams.qs8_lrelu.scalar.positive_multiplier,
           uparams.qs8_lrelu.scalar.negative_multiplier,
           (int32_t) uparams.qs8_lrelu.scalar.output_zero_point,
           i + 1 < NCONFIGS ? "," : "");
  }
  printf("]\n");
  return 0;
}
