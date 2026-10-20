// Differential driver for xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1: bit-exact
// check against a LITERAL transcription of the Neon kernel's per-element
// sequence (the family's only upstream implementation - there is no scalar
// variant to link against), over the ENTIRE int8 input space (all 256x256
// operand pairs) for every producer-true param config, plus batch-size/tail
// coverage at every VLEN. Params come from the REAL library init
// (xnn_init_qs8_mul_minmax_rndnu_neon_params, linked from microparams-init.c;
// it compiles for riscv in the port worktree, where the ARM-only guards are
// widened - see kernel.c / REPORT.md).
//
// Gate-2 role: SALTyRN's verifier has no memory fixture for this stem (the
// family has no SALTyRN port and no scalar kernel), so this exhaustive
// differential sweep is the equivalence gate: inputs are mathematically
// complete (every int8 pair); params are the init's own outputs at its
// asserted-domain boundaries (see dump_params.c / params.json; params are
// therefore discretized at boundary configs - an SMT fixture on the trusted
// side can close that later).
//
// Build (toolchain container):
//   bash difftest_build.sh          (from the repo root on the box)
// Run: under QEMU at VLEN 128/256/512/1024 (inside the script).
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"

void xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1(
    size_t, const int8_t*, const int8_t*, int8_t*,
    const union xnn_qs8_mul_minmax_params*);

// LITERAL per-element transcription of
// xnn_qs8_vmul_minmax_rndnu_ukernel__neon_ld64_u8: every Neon step, including
// its intermediate saturations, in exact C. This is the oracle; the composite
// rhu(P*m/2^(31+shift)) identity claimed in kernel.c must reproduce it
// bit-exactly or the difftest fails.
static int8_t ref_elem(int8_t a, int8_t b, const union xnn_qs8_mul_minmax_params* q) {
  // vsubl_s8: vxa = (int16)(a - azp), vxb = (int16)(b - bzp)
  const int32_t vxa = (int32_t) a - (int32_t) q->rndnu_neon.a_zero_point;
  const int32_t vxb = (int32_t) b - (int32_t) q->rndnu_neon.b_zero_point;
  // vmull_s16: i32 product
  int64_t vacc = (int64_t) (vxa * vxb);
  // vqshlq_s32(vacc, vleft_pre_shift): saturating left shift by >= 0
  vacc = vacc << q->rndnu_neon.left_pre_shift;
  if (vacc > INT32_MAX) vacc = INT32_MAX;
  if (vacc < INT32_MIN) vacc = INT32_MIN;
  // vqdmulhq_s32(vacc, vmultiplier): sat32((vacc * mult * 2) >> 32)
  int64_t high = (vacc * (int64_t) q->rndnu_neon.multiplier) << 1;
  high >>= 32;  // arithmetic shift == floor for the truncating high half
  if (high > INT32_MAX) high = INT32_MAX;
  if (high < INT32_MIN) high = INT32_MIN;
  // vrshlq_s32(vacc, vleft_post_shift): left_post_shift <= -1, so this is the
  // ARM rounding right shift: (v + 2^(s-1)) >> s, s = -left_post_shift in [1,15]
  {
    const int64_t s = -q->rndnu_neon.left_post_shift;
    high = (high + (INT64_C(1) << (s - 1))) >> s;
  }
  // vqmovn_s32
  int32_t v16 = (int32_t) high;
  if (v16 > INT16_MAX) v16 = INT16_MAX;
  if (v16 < INT16_MIN) v16 = INT16_MIN;
  // vqaddq_s16(vacc, voutput_zero_point)
  v16 = v16 + q->rndnu_neon.output_zero_point;
  if (v16 > INT16_MAX) v16 = INT16_MAX;
  if (v16 < INT16_MIN) v16 = INT16_MIN;
  // vqmovn_s16
  int32_t v8 = v16;
  if (v8 > INT8_MAX) v8 = INT8_MAX;
  if (v8 < INT8_MIN) v8 = INT8_MIN;
  // vmax_s8(output_min), vmin_s8(output_max)
  if (v8 < q->rndnu_neon.output_min) v8 = q->rndnu_neon.output_min;
  if (v8 > q->rndnu_neon.output_max) v8 = q->rndnu_neon.output_max;
  return (int8_t) v8;
}

// Producer-true configs (the init's asserted domain): the same quantization
// configs dump_params.c feeds through the real init into params.json.
static const struct {
  float a_scale, b_scale, y_scale;
  int32_t a_zp, b_zp, y_zp;
  const char* why;
} configs[] = {
  {1.0f, 1.0f, 1.0f, 0, 0, 0, "mult 0x40000000 (lower bound), shift 0"},
  {1.0f, 1.0f, 65536.0f, 0, 0, 0, "pos 2^-16 (assert edge), shift 15, E=46 (max)"},
  {1.0f, 1.0f, 32768.0f, 0, 0, 0, "pos 2^-15, shift 14, E=45"},
  {1.0f, 1.0f, 0.0078125f, 0, 0, 0, "pos 2^7, shift -8 (assert edge), pre-shift 9, E=23 (min)"},
  {1.0f, 1.0f, 1.0f / 255.99998474121094f, 0, 0, 0, "mult 0x7FFFFF80 (upper bound), shift -8"},
  {1.0f, 1.0f, 2.0f, 0, 0, 0, "pos 0.5, shift 0 (pre/post split edge)"},
  {1.0f, 1.0f, 0.5f, 0, 0, 0, "pos 2, shift -2 (pre-shift 3, post 1)"},
  {1.0f, 1.0f, 1.5f, 0, 0, 0, "pos 2/3, shift -1 (pre-shift 2, post 1)"},
  {1.0f, 1.0f, 0.25f, 0, 0, 0, "pos 4, shift 2 (pure post-shift)"},
  {0.75f, 1.25f, 0.96875f, -128, -128, 0, "gtest default scales, a/b zp -128"},
  {9.72f, 1.25f, 0.96875f, 127, -128, 127, "gtest a_scale top, pos 12.54, shift -4"},
  {3.096f, 3.096f, 0.314f, 100, -100, 3, "pos 30.5, shift -6, E=25, extreme zps"},
  {0.1f, 0.1f, 9.72f, 5, -5, -7, "pos 2^-9.9, shift 7, E=38, mixed zps"},
  {0.0123f, 0.0456f, 1.0f, -128, 127, -128, "tiny pos, extreme zps"},
};
#define NCONFIGS ((int) (sizeof(configs) / sizeof(configs[0])))

static const size_t batches[] = {
  1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
  255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4096,
};
#define NBATCHES ((int) (sizeof(batches) / sizeof(batches[0])))

int main(void) {
  static int8_t in_a[4096], in_b[4096], out[4096], ref[4096];
  uint64_t seed = 0x9e3779b97f4a7c15ull;
  for (size_t i = 0; i < 4096; i++) {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    in_a[i] = (int8_t) (seed >> 56);
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    in_b[i] = (int8_t) (seed >> 56);
  }
  in_a[0] = -128; in_a[1] = 127; in_a[2] = 0; in_a[3] = -1;
  in_b[0] = 127; in_b[1] = -128; in_b[2] = 0; in_b[3] = 1;

  size_t checks = 0, fails = 0;
  for (int c = 0; c < NCONFIGS; c++) {
    const struct xnn_quantization_params aq = {configs[c].a_zp, configs[c].a_scale};
    const struct xnn_quantization_params bq = {configs[c].b_zp, configs[c].b_scale};
    const struct xnn_quantization_params yq = {configs[c].y_zp, configs[c].y_scale};
    union xnn_qs8_mul_minmax_params up;
    xnn_init_qs8_mul_minmax_rndnu_neon_params(&up, &aq, &bq, &yq);
    const union xnn_qs8_mul_minmax_params* q = &up;

    // (1) Exhaustive sweep: every int8 operand pair, batch 256 rows
    // (a constant per row, b covering all 256 values).
    for (int32_t a = -128; a <= 127; a++) {
      memset(in_a, (int8_t) a, 256);
      for (int32_t b = -128; b <= 127; b++) in_b[b + 128] = (int8_t) b;
      memset(out, 0x5A, 256); memset(ref, 0xA5, 256);
      xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1(256, in_a, in_b, out, q);
      for (int32_t b = -128; b <= 127; b++) ref[b + 128] = ref_elem((int8_t) a, (int8_t) b, q);
      checks++;
      if (memcmp(out, ref, 256) != 0) {
        fails++;
        size_t first = 0;
        while (first < 256 && out[first] == ref[first]) first++;
        printf("FAIL exhaustive cfg %d (%s: mult %" PRId32 " lps %" PRId32 " lpp %" PRId32
               " ozp %d) a=%d b=%d: got %d want %d\n",
               c, configs[c].why, q->rndnu_neon.multiplier, q->rndnu_neon.left_pre_shift,
               q->rndnu_neon.left_post_shift, (int) q->rndnu_neon.output_zero_point,
               a, (int) first - 128, out[first], ref[first]);
        if (fails > 8) { printf("... stopping after 8 failures\n"); return 1; }
      }
    }

    // (2) Batch/tail coverage on the LCG pattern.
    for (int bi = 0; bi < NBATCHES; bi++) {
      const size_t n = batches[bi];
      memset(out, 0x5A, n); memset(ref, 0xA5, n);
      for (size_t i = 0; i < n; i++) ref[i] = ref_elem(in_a[i], in_b[i], q);
      xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1(n, in_a, in_b, out, q);
      checks++;
      if (memcmp(out, ref, n) != 0) {
        fails++;
        size_t first = 0;
        while (first < n && out[first] == ref[first]) first++;
        printf("FAIL batch cfg %d (%s) n=%zu at [%zu]: got %d want %d\n",
               c, configs[c].why, n, first, out[first], ref[first]);
        if (fails > 8) { printf("... stopping after 8 failures\n"); return 1; }
      }
    }
  }
  printf("%s: %zu checks (%d exhaustive-config x 256x256 + %d batch x %d configs), %zu failures\n",
         fails ? "DIFFTEST_FAIL" : "DIFFTEST_OK", checks, NCONFIGS, NBATCHES, NCONFIGS, fails);
  return fails != 0;
}
