// Prints the exact param structs xnn_init_qs8_cvt_scalar_params produces for a
// sweep of (input_scale, input_zp, output_scale, output_zp) quantization
// configs, as the JSON list gate 2 (evaluate.verify configs mode) consumes.
//
// The init function is the real one, compiled from the pinned tree: nothing
// here invents parameter values. Build (inside the toolchain container, with
// the workdir mounted at /pf; the pthreadpool.h stub is the one the
// qs8_vadd_smoke build uses):
//
//   gcc-14 -O2 -ffunction-sections -fdata-sections \
//     -I/pf/repo/ports/xnnpack-rvv/research/qs8_vadd_smoke \
//     -I/pf/src/xnnpack-saltyrn \
//     /pf/repo/ports/xnnpack-rvv/kernels/qs8-vcvt/dump_params.c \
//     /pf/src/xnnpack-saltyrn/src/microparams-init.c \
//     -lm -Wl,--gc-sections -o /pf/build/dump_params
//   /pf/build/dump_params > .../kernels/qs8-vcvt/params.json
//
// The configs cover the init's asserted input domain, |ratio| in [2^-8, 2^7],
// i.e. multiplier in [1, 32768]: both extremes, the gtest's ratios, non-power
// ratios that exercise lrintf, and zero-point extremes and combinations.
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
} configs[] = {
  {1.0f, 0, 1.0f, 0},          // identity scales: multiplier 256 (the gtest default)
  {4.0f, 0, 1.0f, 0},          // gtest input_scale: multiplier 1024
  {16.0f, 5, 1.0f, 3},         // gtest input_scale with non-zero points: 4096
  {64.0f, 0, 1.0f, 0},         // gtest input_scale max: 16384
  {128.0f, 0, 1.0f, 0},        // init assert max, multiplier 32768: the int16-breaking edge
  {127.99f, -5, 1.0f, 7},      // just under the max: lrintf -> 32765
  {1.0f, 0, 4.0f, 0},          // gtest output_scale: multiplier 64
  {1.0f, 0, 64.0f, 0},         // gtest output_scale max: multiplier 4
  {1.0f, 0, 256.0f, 0},        // init assert min, multiplier 1
  {0.004f, 9, 1.0f, -9},       // near the min: lrintf -> 1
  {1.5f, 0, 1.0f, 0},          // non-power ratio: 384
  {1.7f, -3, 1.1f, 2},         // non-power ratio with points: lrintf(395.6) = 396
  {0.0123f, -128, 0.0456f, 127},  // small odd ratio, zero points at both extremes
  {1.0f, -128, 1.0f, -128},    // zero points at int8 min
  {1.0f, 127, 1.0f, 127},      // zero points at int8 max
  {1.0f, -128, 1.0f, 127},     // zero points of opposite signs at extremes
  {2.0f, 100, 1.0f, -100},     // large same-magnitude opposite points
};
#define NCONFIGS (sizeof(configs) / sizeof(configs[0]))

int main(void) {
  printf("[\n");
  for (size_t i = 0; i < NCONFIGS; i++) {
    const union xnn_unary_params op = {0};
    union xnn_unary_uparams uparams;
    const struct xnn_quantization_params input_quantization = {
      configs[i].input_zero_point, configs[i].input_scale,
    };
    const struct xnn_quantization_params output_quantization = {
      configs[i].output_zero_point, configs[i].output_scale,
    };
    xnn_init_qs8_cvt_scalar_params(&uparams, &op, &input_quantization, &output_quantization);
    printf("  {\"input_zero_point\": %" PRId32 ", \"multiplier\": %" PRId32
           ", \"output_zero_point\": %" PRId32 "}%s\n",
           (int32_t) uparams.qs8_cvt.scalar.input_zero_point,
           uparams.qs8_cvt.scalar.multiplier,
           (int32_t) uparams.qs8_cvt.scalar.output_zero_point,
           i + 1 < NCONFIGS ? "," : "");
  }
  printf("]\n");
  return 0;
}
