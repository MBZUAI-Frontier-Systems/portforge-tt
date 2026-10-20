// clang-format off
// qs8-vcvt (requantized int8 convert) RVV microkernel, round 2.
//
// Arithmetic (derived from xnn_qs8_vcvt_ukernel__scalar_u1, the spec):
//
//   vbias = (int32)(((uint32)(int32) output_zero_point) << 8)
//           - multiplier * (int32) input_zero_point + 0x80
//   y     = sat8( (vbias + x * multiplier) >> 8 )       // arithmetic shift
//
// xnn_init_qs8_cvt_scalar_params produces multiplier = lrintf(256 * in_scale /
// out_scale) in [1, 32768] with int8-range zero points, so per element
//
//   y = sat8( round_half_up((x - input_zero_point) * multiplier / 256)
//             + output_zero_point ),   round_half_up(z) = floor(z + 1/2).
//
// The ukernel is compute-bound at the benchmark shapes (a raw vector copy
// reaches 20.8 GB/s where this family runs at 10.3 GB/s on the X280), so the
// only lever is vector operations per element. The X280 executes each op on
// whole LMUL-sized register groups, so the design below dispatches on the
// per-call scalar params (they are fixed for one call) to the *fewest ops*
// that is still bit-exact for that whole regime. Four regimes, all derived
// from the same identity round_half_up(a/256) = floor((a + 128)/256):
//
// - multiplier == 256: the division is exact and the rounding drops out,
//   y = sat8(x - (input_zero_point - output_zero_point)). When
//   s = izp - ozp fits int8 this is ONE op: e8 saturating subtract
//   (vssub). Otherwise y = sat8((x - izp) + ozp) in i16: widen-subtract,
//   saturating add, narrow (3 ops).
// - "direct i16" (multiplier small enough that m*x + b fits i16 with
//   b = 256*ozp - m*izp, i.e. b - 128*m >= -32768 and b + 127*m <= 32767):
//   moving the zero points into a scalar bias and the +0x80 rounding into
//   the narrowing op's RNU (round-half-up adds exactly 2^(8-1) before the
//   shift),
//     y = sat8( floor((m*x + b + 128) / 256) )
//   needs no fractional multiply at all: widen x, vmul by m, add b, and
//   vnclip by shift 8 in RNU mode saturates to int8 (4 ops, one fewer than
//   the general chain). Wrapped intermediates are harmless: vmul/vadd are
//   exact modulo 2^16 and the regime predicate guarantees the true sum
//   fits i16, so the stored i16 value equals m*x + b. This regime covers
//   XNNPACK's own vunary benchmark parameters (scales 0.5/0.7, zero points
//   1/-1: multiplier 183, b -439), where the general chain would run the
//   same 5 ops as upstream.
// - multiplier a power of two >= 512: the quotient m/256 = 2^e (e in
//   [1,7]) is an exact integer, the multiply and its rounding drop out:
//   widen-subtract, shift by e, saturating add, narrow (4 ops). |(x-izp)
//   << e| <= 255 * 2^7 = 32640 and +ozp reaches 32767 exactly, so no i16
//   stage can overflow for producer params. This generalizes the
//   multiplier == 32768 regime of the round-1 kernel (which upstream's RVV
//   variants get wrong: they truncate the int32 field to int16).
// - everything else (1 <= multiplier <= 32767): the i16 fractional-multiply
//   chain, proven in REPORT.md section 1: |(x - izp) << 7| <= 32640 fits
//   i16; vsmul (RNU, ties up) reproduces the +0x80 rounding exactly and its
//   result fits i16; vsadd(+ozp) fits i16; vnclip applies sat8. 5 ops.
//
// The dispatch is complete and disjoint over the init's whole output range:
// every (multiplier, zero point) triple the producer can emit takes exactly
// one regime, and each regime is exact on its whole domain (formally
// verified per config by the cvc5 gate; regimes map to the configs in
// params.json as A: m=256/|s|<=127, B: m=256/|s|>127, C: the small-m
// configs, D1: 512..32768 powers of two, D2: the rest).
//
// Loop structure: one vsetvl-driven loop per regime; the tail is a smaller
// vl of the same code path (no scalar epilogue, no OOB reads: vle8/vse8
// touch exactly vl elements). Regimes B/D1/D2 run e8m4 in / i16m8 math, the
// widest legal layout; regime A stays e8m4 end to end; regime C widens once
// and runs e16m8. All lanes-per-iteration maxima match upstream rvv_u4v.
#include <assert.h>
#include <stdint.h>

#include <riscv_vector.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"

void xnn_qs8_vcvt_ukernel__rvv_pf2(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vinput_zero_point = params->scalar.input_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;
  const int32_t vmultiplier = params->scalar.multiplier;
  assert(vmultiplier >= 1 && vmultiplier <= 32768);

  if (vmultiplier == 256) {
    // Exact quotient: y = sat8(x - (izp - ozp)).
    const int32_t vshift_zero_point = vinput_zero_point - voutput_zero_point;
    if (vshift_zero_point >= -128 && vshift_zero_point <= 127) {
      // The whole map is one e8 saturating subtract.
      const int8_t vs = (int8_t) vshift_zero_point;
      do {
        size_t vl = __riscv_vsetvl_e8m4(batch); batch -= vl;
        const vint8m4_t vx = __riscv_vle8_v_i8m4(input, vl); input += vl;
        const vint8m4_t vy = __riscv_vssub_vx_i8m4(vx, vs, vl);
        __riscv_vse8_v_i8m4(output, vy, vl); output += vl;
      } while (batch != 0);
      return;
    }
    // s outside int8: keep the widening subtract, y = sat8((x - izp) + ozp).
    do {
      size_t vl = __riscv_vsetvl_e8m4(batch); batch -= vl;
      const vint8m4_t vx = __riscv_vle8_v_i8m4(input, vl); input += vl;
      vint16m8_t vacc = __riscv_vwsub_vx_i16m8(vx, (int8_t) vinput_zero_point, vl);
      // |vacc + ozp| <= 255 + 128 = 383: cannot saturate.
      vacc = __riscv_vsadd_vx_i16m8(vacc, (int16_t) voutput_zero_point, vl);
      const vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_i8m4(output, vy, vl); output += vl;
    } while (batch != 0);
    return;
  }

  // Direct i16 form: y = sat8(floor((m*x + b + 128) / 256)) with the zero
  // points folded into the scalar bias b = 256*ozp - m*izp. The predicate
  // keeps m*x + b inside i16 for every int8 x; b itself then fits i16 too
  // (it is the value at x = 0).
  const int32_t vbias = 256 * voutput_zero_point - vmultiplier * vinput_zero_point;
  if (vbias >= 128 * vmultiplier - 32768 && vbias <= 32767 - 127 * vmultiplier) {
    const int16_t vm = (int16_t) vmultiplier;
    const int16_t vb = (int16_t) vbias;
    do {
      size_t vl = __riscv_vsetvl_e8m4(batch); batch -= vl;
      const vint8m4_t vx = __riscv_vle8_v_i8m4(input, vl); input += vl;
      vint16m8_t vacc = __riscv_vwadd_vx_i16m8(vx, 0, vl);
      vacc = __riscv_vmul_vx_i16m8(vacc, vm, vl);
      vacc = __riscv_vadd_vx_i16m8(vacc, vb, vl);
      // RNU adds exactly the +0x80 rounding constant before the >>8.
      const vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 8, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_i8m4(output, vy, vl); output += vl;
    } while (batch != 0);
    return;
  }

  if (vmultiplier >= 512 && (vmultiplier & (vmultiplier - 1)) == 0) {
    // Exact power-of-two quotient m/256 = 2^e, e in [1, 7]: no rounding.
    const uint32_t vshift = (uint32_t) __builtin_ctz((unsigned) vmultiplier) - 8;
    do {
      size_t vl = __riscv_vsetvl_e8m4(batch); batch -= vl;
      const vint8m4_t vx = __riscv_vle8_v_i8m4(input, vl); input += vl;
      vint16m8_t vacc = __riscv_vwsub_vx_i16m8(vx, (int8_t) vinput_zero_point, vl);
      // |vacc << e| <= 255 * 128 = 32640; +ozp reaches 32767 exactly.
      vacc = __riscv_vsll_vx_i16m8(vacc, vshift, vl);
      vacc = __riscv_vsadd_vx_i16m8(vacc, (int16_t) voutput_zero_point, vl);
      const vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_i8m4(output, vy, vl); output += vl;
    } while (batch != 0);
    return;
  }

  // General regime (1 <= multiplier <= 32767): i16 fractional-multiply
  // chain, exact per REPORT.md section 1.
  assert(vmultiplier <= INT16_MAX);
  do {
    size_t vl = __riscv_vsetvl_e8m4(batch); batch -= vl;
    const vint8m4_t vx = __riscv_vle8_v_i8m4(input, vl); input += vl;
    vint16m8_t vacc = __riscv_vwsub_vx_i16m8(vx, (int8_t) vinput_zero_point, vl);
    vacc = __riscv_vsll_vx_i16m8(vacc, 7, vl);
    // RNU ties up: identical to the scalar's "+0x80 then floor-shift".
    vacc = __riscv_vsmul_vx_i16m8(vacc, (int16_t) vmultiplier, __RISCV_VXRM_RNU, vl);
    vacc = __riscv_vsadd_vx_i16m8(vacc, (int16_t) voutput_zero_point, vl);
    const vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_i8m4(output, vy, vl); output += vl;
  } while (batch != 0);
}
