// Scalar oracle for qs8-vmul-minmax-rndnu: a LITERAL per-element transcription
// of xnn_qs8_vmul_minmax_rndnu_ukernel__neon_ld64_u8 (the family's only
// upstream implementation) in exact C - the spec this campaign's kernel is
// gated against (there is no scalar variant upstream to link).
//
// Compiled twice by sizebench_build.sh: -O2 as the scalar baseline, and
// -O3 -march=rv64gcv as the autovectorization bar (symbol renamed by macro).
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#ifndef SCALAR_ORACLE_NAME
#define SCALAR_ORACLE_NAME xnn_qs8_vmul_minmax_rndnu_scalar_oracle
#endif

static inline int8_t oracle_elem(
    int8_t a, int8_t b, const union xnn_qs8_mul_minmax_params* restrict q) {
  // vsubl_s8
  const int32_t vxa = (int32_t) a - (int32_t) q->rndnu_neon.a_zero_point;
  const int32_t vxb = (int32_t) b - (int32_t) q->rndnu_neon.b_zero_point;
  // vmull_s16
  int64_t vacc = (int64_t) (vxa * vxb);
  // vqshlq_s32 (left_pre_shift >= 0 on the producer domain)
  vacc = vacc << q->rndnu_neon.left_pre_shift;
  if (vacc > INT32_MAX) vacc = INT32_MAX;
  if (vacc < INT32_MIN) vacc = INT32_MIN;
  // vqdmulhq_s32: sat32((vacc * mult * 2) >> 32)
  int64_t high = (vacc * (int64_t) q->rndnu_neon.multiplier) << 1;
  high >>= 32;
  if (high > INT32_MAX) high = INT32_MAX;
  if (high < INT32_MIN) high = INT32_MIN;
  // vrshlq_s32 by left_post_shift <= -1: (v + 2^(s-1)) >> s
  {
    const int64_t s = -q->rndnu_neon.left_post_shift;
    high = (high + (INT64_C(1) << (s - 1))) >> s;
  }
  // vqmovn_s32 -> vqaddq_s16 -> vqmovn_s16 -> vmax/vmin
  int32_t v16 = (int32_t) high;
  if (v16 > INT16_MAX) v16 = INT16_MAX;
  if (v16 < INT16_MIN) v16 = INT16_MIN;
  v16 = v16 + q->rndnu_neon.output_zero_point;
  if (v16 > INT16_MAX) v16 = INT16_MAX;
  if (v16 < INT16_MIN) v16 = INT16_MIN;
  int32_t v8 = v16;
  if (v8 > INT8_MAX) v8 = INT8_MAX;
  if (v8 < INT8_MIN) v8 = INT8_MIN;
  if (v8 < q->rndnu_neon.output_min) v8 = q->rndnu_neon.output_min;
  if (v8 > q->rndnu_neon.output_max) v8 = q->rndnu_neon.output_max;
  return (int8_t) v8;
}

void SCALAR_ORACLE_NAME(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params* restrict params) {
  for (size_t i = 0; i < batch; i++) {
    output[i] = oracle_elem(input_a[i], input_b[i], params);
  }
}
