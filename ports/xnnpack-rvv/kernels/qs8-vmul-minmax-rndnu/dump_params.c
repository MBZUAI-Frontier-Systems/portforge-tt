// Prints the exact param structs xnn_init_qs8_mul_minmax_rndnu_neon_params
// (the REAL library init, linked from microparams-init.c; it compiles for
// riscv in the port worktree where the ARM-only guards are widened - see
// kernel.c / REPORT.md) produces for a sweep of (a_scale, b_scale, y_scale,
// a_zp, b_zp, y_zp) quantization configs, as the JSON list gate 2
// (evaluate.verify configs mode) consumes. Nothing here invents parameter
// values.
//
// Build (toolchain container): bash difftest_build.sh runs this; the JSON
// lands in params.json next to this file.
#include <inttypes.h>
#include <stdio.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"

static const struct {
  float a_scale, b_scale, y_scale;
  int32_t a_zp, b_zp, y_zp;
  const char* why;
} configs[] = {
  // multiplier / shift boundaries of the init's asserted domain:
  {1.0f, 1.0f, 1.0f, 0, 0, 0, "mult 0x40000000 (lower bound), shift 0"},
  {1.0f, 1.0f, 65536.0f, 0, 0, 0, "pos 2^-16 (assert edge), shift 15, E=46 (max)"},
  {1.0f, 1.0f, 32768.0f, 0, 0, 0, "pos 2^-15, shift 14, E=45"},
  {1.0f, 1.0f, 0.0078125f, 0, 0, 0, "pos 2^7, shift -8 (assert edge), pre-shift 9, E=23 (min)"},
  {1.0f, 1.0f, 1.0f / 255.99998474121094f, 0, 0, 0, "mult 0x7FFFFF80 (upper bound), shift -8"},
  {1.0f, 1.0f, 2.0f, 0, 0, 0, "pos 0.5, shift 0 (pre/post split edge)"},
  {1.0f, 1.0f, 0.5f, 0, 0, 0, "pos 2, shift -2 (pre-shift 3, post 1)"},
  {1.0f, 1.0f, 1.5f, 0, 0, 0, "pos 2/3, shift -1 (pre-shift 2, post 1)"},
  {1.0f, 1.0f, 0.25f, 0, 0, 0, "pos 4, shift 2 (pure post-shift)"},
  // gtest defaults and sweep extremes, with zero point extremes:
  {0.75f, 1.25f, 0.96875f, -128, -128, 0, "gtest default scales, a/b zp -128"},
  {9.72f, 1.25f, 0.96875f, 127, -128, 127, "gtest a_scale top, pos 12.54, shift -4"},
  {3.096f, 3.096f, 0.314f, 100, -100, 3, "pos 30.5, shift -6, E=25, extreme zps"},
  {0.1f, 0.1f, 9.72f, 5, -5, -7, "pos 2^-9.9, shift 7, E=38, mixed zps"},
  {0.0123f, 0.0456f, 1.0f, -128, 127, -128, "tiny pos, extreme zps"},
};
#define NCONFIGS ((int) (sizeof(configs) / sizeof(configs[0])))

int main(void) {
  printf("[\n");
  for (int i = 0; i < NCONFIGS; i++) {
    const struct xnn_quantization_params aq = {configs[i].a_zp, configs[i].a_scale};
    const struct xnn_quantization_params bq = {configs[i].b_zp, configs[i].b_scale};
    const struct xnn_quantization_params yq = {configs[i].y_zp, configs[i].y_scale};
    union xnn_qs8_mul_minmax_params up;
    xnn_init_qs8_mul_minmax_rndnu_neon_params(&up, &aq, &bq, &yq);
    printf("  {\"a_zero_point\": %d, \"b_zero_point\": %d, \"left_pre_shift\": %" PRId32
           ", \"multiplier\": %" PRId32 ", \"left_post_shift\": %" PRId32
           ", \"output_zero_point\": %d, \"output_min\": %d, \"output_max\": %d, \"why\": \"%s\"}%s\n",
           (int) up.rndnu_neon.a_zero_point, (int) up.rndnu_neon.b_zero_point,
           up.rndnu_neon.left_pre_shift, up.rndnu_neon.multiplier,
           up.rndnu_neon.left_post_shift, (int) up.rndnu_neon.output_zero_point,
           (int) up.rndnu_neon.output_min, (int) up.rndnu_neon.output_max,
           configs[i].why, i + 1 < NCONFIGS ? "," : "");
  }
  printf("]\n");
  return 0;
}
