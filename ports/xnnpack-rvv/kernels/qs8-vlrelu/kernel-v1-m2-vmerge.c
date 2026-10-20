// clang-format off
// ARCHIVE: qs8-vlrelu round-1 variant (reconstructed from the session edit
// history; not byte-identical to what the round-1 gates measured, but the
// same chains: all regimes e8m2 in / i16m4 middle, general regime =
// vmslt-mask + vmerge_vxm with a hoisted i16m4 broadcast). Measured on the
// X280: 0.62 cyc/elem at N:16320, official bench 2.031x geomean vs upstream
// rvv_u2v. Superseded by kernel.c (7 ops at e8m4/i16m8). See REPORT.md.
// qs8-vlrelu (quantized leaky ReLU) RVV microkernel.
//
// Arithmetic (derived from xnn_qs8_vlrelu_ukernel__scalar_select_u1, the spec):
//
//   d    = (int32) x - input_zero_point
//   m    = (d >= 0) ? positive_multiplier : negative_multiplier
//   vacc = (int32) (((uint32) (int32) output_zero_point) << 8) + 0x80 + d * m
//   y    = sat8( vacc >> 8 )                     // arithmetic (floor) shift
//
// i.e. per element
//
//   y = sat8( output_zero_point
//             + round_half_up( d * m(d) / 256 ) ),   rhu(z) = floor(z + 1/2),
//
// with the sign test taken on the DE-BIASED value d (this is the classic
// off-by-one trap of the family: the raw input sign is wrong at
// x == input_zero_point and below it).
//
// xnn_init_qs8_lrelu_scalar_params produces, for int8-range zero points:
//   positive_multiplier = lrintf(256 * in_scale / out_scale)  in [1, 32768]
//   negative_multiplier = positive_multiplier * slope, rounded, in
//                         [-32767, 32768], never 0 (a 0 slope rounds to an
//   assert failure in the init, so every producer value is covered by the
//   regimes below and nothing else can arrive).
//
// The Neon kernel (the formal-gate oracle) computes the same map with a
// negated-operand trick: d' = izp - x, multiplier' = -(selected multiplier),
// vqrdmulh(d' << 7, m') = rhu(d * m / 256) bit-exactly, then vqadd(+ozp) and
// a saturating narrow. Its intermediate saturations (vqrdmulh/vqadd) provably
// never trigger on the producer domain (|rhu(d*m/256)| <= 255, |+ozp| <= 383),
// so Neon == scalar == this kernel bit-exact everywhere.
//
// The kernel is compute-bound at the benchmark shapes (see REPORT.md), and
// the X280 executes each vector op on whole LMUL-sized register groups, so
// the design dispatches on the per-call scalar params (fixed for one call)
// to the fewest vector ops that stays bit-exact on that regime's whole
// domain. Three regimes, complete and disjoint over the init's output range:
//
// - positive == negative (slope rounds to exactly 1): the op is plain
//   requantization, and the qs8-vcvt regime ladder applies unchanged
//   (y = sat8(ozp + rhu(d * m / 256)) with a single m): the exact-quotient
//   vssub one-op form at m = 256; the 3-op i16 form at m = 256 with far-apart
//   zero points; the 4-op direct-i16 form when m*x + b fits i16
//   (b = 256*ozp - m*izp); the 4-op exact-shift form at powers of two >= 512
//   (m/256 = 2^e, e in [1, 7], no rounding); the 5-op vsmul chain otherwise.
//   Every stage of every form is exact or redundant in range for producer
//   params, exactly as proven for qs8-vcvt (same identity
//   rhu(a/256) = floor((a + 128)/256), same i16 bounds).
// - otherwise, if both multipliers fit int16 (|m| <= 32767): the general
//   two-sided chain, all i16, 7 ops at the e8m2 in / i16m4 middle width:
//     mask = x > izp (i8 compare)               // exactly d > 0
//     d    = vwsub(x, izp)                      // [-255, 255]
//     m    = merge(pos, neg, mask)              // d < 0 ? neg : pos
//     d  <<= 7                                  // |d << 7| <= 32640
//     r    = vsmul_vv(d, m, RNU)                // rhu(d*m/256), |r| <= 255
//     r   += ozp  (vsadd)                       // |r + ozp| <= 383
//     y    = vnclip(r, 0)                       // sat8 = the scalar clamp
//   The select is bit-identical to the scalar's: the mask is false at
//   d == 0 and selects the hoisted (positive) multiplier, where the product
//   is 0 either way. The compare rides on the raw i8 input, so the select
//   costs one narrow op plus one merge - the fewest-ops form of this chain;
//   the arithmetic and/xor select needs three m8-wide ops instead (measured
//   1.2x slower per element at this width). The hoisted broadcast is
//   full-length (vsetvlmax), loop-invariant, and later iterations read only
//   its prefix. vsmul in RNU mode adds exactly 2^14 before the >>15, which
//   reproduces the scalar's "+0x80 then floor >>8" on the <<7-scaled
//   operand; its internal product |d * m| <= 255*32767 < 2^30 cannot
//   overflow, and its result is bounded by 255, so no intermediate
//   saturates before the final sat8 narrow.
// - otherwise (a multiplier is 32768, not representable in int16): an i32
//   chain shaped like the upstream rvv template (their u1v/u2v): sign
//   select via vsra(31)/vand/vxor in i32, vmacc against a hoisted bias
//   vector 0x80 + 256*ozp (the round-half-up constant and the output zero
//   point folded, so one RDN shift by 8 is the exact scalar asr), then a
//   saturating narrow to i8. |d * m| <= 255 * 32768 = 2^23 and the bias
//   fits i32 with room to spare; the i16 stage after the shift-8 cannot
//   saturate for producer params.
//
// Loop structure: one vsetvl-driven loop per regime; the tail is a smaller
// vl of the same code path (no scalar epilogue). No OOB reads, no OOB
// writes: vle8/vse8 touch exactly vl elements. The i16 chains run at the
// widest legal layout (e8m2 in/out, i16m4 middle - the same
// lanes-per-iteration as a would-be upstream u4v); the i32 regime stays at
// e8m2 in / i32m8 middle because its wide stage is i32 (m4 input would make
// it an illegal m16), the same layout as the upstream rvv template. Live
// sets stay within the 32 X280 registers in both (checked in the
// disassembly: no group spill moves).
#include <assert.h>

#include <riscv_vector.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"

void xnn_qs8_vlrelu_ukernel__rvv_pf1(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_lrelu_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vinput_zero_point = params->scalar.input_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;
  const int32_t vpositive_multiplier = params->scalar.positive_multiplier;
  const int32_t vnegative_multiplier = params->scalar.negative_multiplier;

  if (vpositive_multiplier == vnegative_multiplier) {
    // Slope == 1 exactly: plain requantization, y = sat8(ozp + rhu(d*m/256)).
    const int32_t vmultiplier = vpositive_multiplier;
    if (vmultiplier == 256) {
      // Exact quotient: the rounding and multiply drop out,
      // y = sat8(x - (izp - ozp)).
      const int32_t vshift_zero_point = vinput_zero_point - voutput_zero_point;
      if (vshift_zero_point >= -128 && vshift_zero_point <= 127) {
        // The whole map is one e8 saturating subtract.
        const int8_t vs = (int8_t) vshift_zero_point;
        do {
          size_t vl = __riscv_vsetvl_e8m2(batch); batch -= vl;
          const vint8m2_t vx = __riscv_vle8_v_i8m2(input, vl); input += vl;
          const vint8m2_t vy = __riscv_vssub_vx_i8m2(vx, vs, vl);
          __riscv_vse8_v_i8m2(output, vy, vl); output += vl;
        } while (batch != 0);
        return;
      }
      // s outside int8: keep the widening subtract, y = sat8((x - izp) + ozp).
      do {
        size_t vl = __riscv_vsetvl_e8m2(batch); batch -= vl;
        const vint8m2_t vx = __riscv_vle8_v_i8m2(input, vl); input += vl;
        vint16m4_t vacc = __riscv_vwsub_vx_i16m4(vx, (int8_t) vinput_zero_point, vl);
        // |vacc + ozp| <= 255 + 128 = 383: cannot saturate.
        vacc = __riscv_vsadd_vx_i16m4(vacc, (int16_t) voutput_zero_point, vl);
        const vint8m2_t vy = __riscv_vnclip_wx_i8m2(vacc, 0, __RISCV_VXRM_RNU, vl);
        __riscv_vse8_v_i8m2(output, vy, vl); output += vl;
      } while (batch != 0);
      return;
    }
    // Direct i16 form: y = sat8(floor((m*x + b + 128) / 256)) with the zero
    // points folded into the scalar bias b = 256*ozp - m*izp. The predicate
    // keeps m*x + b inside i16 for every int8 x; b itself then fits i16 too
    // (it is the value at x = 0). Wrapped intermediates are harmless:
    // vmul/vadd are exact modulo 2^16 and the predicate guarantees the true
    // sum fits i16.
    const int32_t vbias = 256 * voutput_zero_point - vmultiplier * vinput_zero_point;
    if (vbias >= 128 * vmultiplier - 32768 && vbias <= 32767 - 127 * vmultiplier) {
      const int16_t vm = (int16_t) vmultiplier;
      const int16_t vb = (int16_t) vbias;
      do {
        size_t vl = __riscv_vsetvl_e8m2(batch); batch -= vl;
        const vint8m2_t vx = __riscv_vle8_v_i8m2(input, vl); input += vl;
        vint16m4_t vacc = __riscv_vwadd_vx_i16m4(vx, 0, vl);
        vacc = __riscv_vmul_vx_i16m4(vacc, vm, vl);
        vacc = __riscv_vadd_vx_i16m4(vacc, vb, vl);
        // RNU adds exactly the +0x80 rounding constant before the >>8.
        const vint8m2_t vy = __riscv_vnclip_wx_i8m2(vacc, 8, __RISCV_VXRM_RNU, vl);
        __riscv_vse8_v_i8m2(output, vy, vl); output += vl;
      } while (batch != 0);
      return;
    }
    if (vmultiplier >= 512 && (vmultiplier & (vmultiplier - 1)) == 0) {
      // Exact power-of-two quotient m/256 = 2^e, e in [1, 7]: no rounding.
      const uint32_t vshift = (uint32_t) __builtin_ctz((unsigned) vmultiplier) - 8;
      do {
        size_t vl = __riscv_vsetvl_e8m2(batch); batch -= vl;
        const vint8m2_t vx = __riscv_vle8_v_i8m2(input, vl); input += vl;
        vint16m4_t vacc = __riscv_vwsub_vx_i16m4(vx, (int8_t) vinput_zero_point, vl);
        // |vacc << e| <= 255 * 128 = 32640; +ozp reaches 32767 exactly.
        vacc = __riscv_vsll_vx_i16m4(vacc, vshift, vl);
        vacc = __riscv_vsadd_vx_i16m4(vacc, (int16_t) voutput_zero_point, vl);
        const vint8m2_t vy = __riscv_vnclip_wx_i8m2(vacc, 0, __RISCV_VXRM_RNU, vl);
        __riscv_vse8_v_i8m2(output, vy, vl); output += vl;
      } while (batch != 0);
      return;
    }
    // Fractional multiply, 1 <= m <= 32767 (32768 is a power of two and took
    // the shift form above). Proven bound chain in the header comment.
    do {
      size_t vl = __riscv_vsetvl_e8m2(batch); batch -= vl;
      const vint8m2_t vx = __riscv_vle8_v_i8m2(input, vl); input += vl;
      vint16m4_t vacc = __riscv_vwsub_vx_i16m4(vx, (int8_t) vinput_zero_point, vl);
      vacc = __riscv_vsll_vx_i16m4(vacc, 7, vl);
      // RNU ties up: identical to the scalar's "+0x80 then floor >> 8".
      vacc = __riscv_vsmul_vx_i16m4(vacc, (int16_t) vmultiplier, __RISCV_VXRM_RNU, vl);
      vacc = __riscv_vsadd_vx_i16m4(vacc, (int16_t) voutput_zero_point, vl);
      const vint8m2_t vy = __riscv_vnclip_wx_i8m2(vacc, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_i8m2(output, vy, vl); output += vl;
    } while (batch != 0);
    return;
  }

    // ARCHIVE of the round-1 variant: same chains as the final kernel but at
    // the e8m2/i16m4 width, with the general regime's mask+merge select and a
    // hoisted i16m4 broadcast. Measured 0.62-0.63 cyc/elem vs the final
    // 0.51 at N:16320 on the X280 (see REPORT.md round table).

    if (vpositive_multiplier <= INT16_MAX && vnegative_multiplier <= INT16_MAX) {
    // General two-sided regime: both multipliers fit int16 (negative can be
    // as low as -32767 by the init's asserts). Sign-select with a mask and a
    // scalar-merge: one hoisted full-length broadcast of the positive
    // multiplier (vsetvlmax, so it is loop-invariant and never rematerialized
    // inside the loop; later iterations use a smaller vl and read a prefix),
    // the negative multiplier as the merge scalar. d == 0 takes the positive
    // side, where the product is 0 either way - same as the scalar.
    const vint16m4_t vpos =
        __riscv_vmv_v_x_i16m4((int16_t) vpositive_multiplier, __riscv_vsetvlmax_e16m4());
    const int16_t vneg = (int16_t) vnegative_multiplier;
    do {
      size_t vl = __riscv_vsetvl_e8m2(batch); batch -= vl;
      const vint8m2_t vx = __riscv_vle8_v_i8m2(input, vl); input += vl;
      vint16m4_t vacc = __riscv_vwsub_vx_i16m4(vx, (int8_t) vinput_zero_point, vl);
      const vbool4_t vmask = __riscv_vmslt_vx_i16m4_b4(vacc, 0, vl);
      vint16m4_t vmult = __riscv_vmerge_vxm_i16m4(vpos, vneg, vmask, vl);
      // |vacc << 7| <= 255 * 128 = 32640 fits i16.
      vacc = __riscv_vsll_vx_i16m4(vacc, 7, vl);
      // RNU ties up: rhu(d * m / 256), |result| <= 255, never saturates.
      vacc = __riscv_vsmul_vv_i16m4(vacc, vmult, __RISCV_VXRM_RNU, vl);
      // |+ozp| <= 383: cannot saturate.
      vacc = __riscv_vsadd_vx_i16m4(vacc, (int16_t) voutput_zero_point, vl);
      const vint8m2_t vy = __riscv_vnclip_wx_i8m2(vacc, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_i8m2(output, vy, vl); output += vl;
    } while (batch != 0);
    return;
  }

  // A multiplier is 32768: not representable in int16. i32 chain, the
  // upstream rvv template's shape, exact across this regime's whole domain.
  assert(vpositive_multiplier == 32768 || vnegative_multiplier == 32768);
  const int32_t vbias = 0x80 + 256 * voutput_zero_point;
  const int32_t n = (int32_t) __riscv_vsetvl_e8m2(batch);
  const vint32m8_t vbiasv = __riscv_vmv_v_x_i32m8(vbias, n);
  do {
    size_t vl = __riscv_vsetvl_e8m2(batch); batch -= vl;
    const vint8m2_t vx = __riscv_vle8_v_i8m2(input, vl); input += vl;
    const vint16m4_t vd16 = __riscv_vwsub_vx_i16m4(vx, (int8_t) vinput_zero_point, vl);
    vint32m8_t vacc = __riscv_vwcvt_x_x_v_i32m8(vd16, vl);
    const vint32m8_t vsign = __riscv_vsra_vx_i32m8(vacc, 31, vl);
    vint32m8_t vmult = __riscv_vand_vx_i32m8(vsign, vnegative_multiplier ^ vpositive_multiplier, vl);
    vmult = __riscv_vxor_vx_i32m8(vmult, vpositive_multiplier, vl);
    // vacc = 0x80 + 256*ozp + d*m: |d*m| <= 2^23, no i32 overflow.
    vacc = __riscv_vmacc_vv_i32m8(vbiasv, vacc, vmult, vl);
    // RDN shift 8 is the exact scalar asr; the i16 stage cannot saturate
    // (|vacc >> 8| <= 32712 for producer params).
    vint16m4_t vout16 = __riscv_vnclip_wx_i16m4(vacc, 8, __RISCV_VXRM_RDN, vl);
    const vint8m2_t vy = __riscv_vnclip_wx_i8m2(vout16, 0, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_i8m2(output, vy, vl); output += vl;
  } while (batch != 0);
}
