// clang-format off
// qs8-vmul-minmax-rndnu: quantized elementwise multiply with round-to-nearest-up
// requantization and a min/max clamp. First RVV variant of the family (upstream
// has Neon only; there is no scalar, SSE or WASM rndnu VMUL either).
//
// Spec, derived from src/qs8-vmul/gen/qs8-vmul-minmax-rndnu-neon-ld64-u8.c (the
// family's only implementation) and xnn_init_qs8_mul_minmax_rndnu_neon_params
// (src/microparams-init.c):
//
//   da   = (int16) a - a_zero_point          // exact: [-255, 255] for int8 zps
//   db   = (int16) b - b_zero_point
//   P    = da * db                           // exact in i32: [-65025, 65025]
//   x    = P << left_pre_shift               // saturating; never saturates on
//                                             //   the producer domain (below)
//   h    = floor(x * multiplier / 2^31)      // vqdmulhq_s32 (doubling high
//                                             //   half, truncating, sat32)
//   r    = floor((h + 2^(s-1)) / 2^s)        // vrshl by left_post_shift = -s,
//                                             //   s = post_shift in [1, 15]
//   y    = sat8(sat16(sat16(r) + output_zero_point))
//   y    = clamp(y, output_min, output_max)  // both int8
//
// The init produces multiplier in [0x40000000, 0x7FFFFF80] and
// shift = 127 + 31 - 32 - (scale_bits >> 23) in [-8, 15], then splits it
//   post_shift  = max(shift, 1),          left_pre_shift  = post_shift - shift,
//   pre_shift   = shift - post_shift,     left_post_shift = -post_shift.
// On that whole domain left_pre_shift in [0, 9] (|x| <= 65025 * 2^9 < 2^31, so
// its saturation never fires) and left_pre_shift + left_post_shift = -shift,
// which collapses the pipeline to ONE rounding of the exact i64 product:
//
//   r = rhu(P * multiplier / 2^(31 + shift))
//     = floor((P * multiplier + 2^(30 + shift - left_pre_shift)) / 2^(31 + shift - left_pre_shift))
//
// (floor at 2^31 followed by a round-half-up shift right by s >= 1 composes
// exactly to a single round-half-up at 2^(31+s): adding f in [0,1) to the
// integer k + 2^(s-1) cannot cross a multiple of 2^s. Verified symbolically
// and by the exhaustive difftest against the literal Neon transcription.)
// With E := 31 + shift - left_pre_shift = 31 - left_pre_shift - left_post_shift
// in [23, 46], the kernel is:
//
//   y = clamp8(sat8(sat16(floor((P * multiplier + 2^(E-1)) / 2^E) + ozp)))
//
// and `vnclip` in RNU mode performs exactly the "+2^(E-1), floor >>E" step.
// The composite identity was checked element-for-element against the literal
// Neon sequence over the entire int8 input space x all difftest configs.
//
// Envelope (everything the init can produce): zero points int8, output zero
// point int16, multiplier positive in [0x40000000, 0x7FFFFF80], E in
// [23, 46], output_min <= output_max within int8. |P * multiplier| < 2^48 and
// |r| <= 2^48/2^23 + 1 < 2^26, so every intermediate below fits: the i64
// product cannot overflow, the i32 stage cannot saturate, and the i16/i8
// narrows reproduce the Neon kernel's saturations exactly.
//
// Upstream PR note: upstream guards the family's params member
// (union xnn_qs8_mul_minmax_params.rndnu_neon) and the init
// xnn_init_qs8_mul_minmax_rndnu_neon_params behind
// #if XNN_ARCH_ARM || XNN_ARCH_ARM64 (src/xnnpack/microparams.h,
// src/xnnpack/microparams-init.h, src/microparams-init.c), so an RVV build
// references an undeclared init and a union without the member. The RVV PR
// therefore widens those three guards with || XNN_ARCH_RISCV (that library
// patch is applied in the port's worktree; the kernel itself needs no shim and
// reads params->rndnu_neon like any other family). See REPORT.md.
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vbinary.h"

void xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t va_zero_point = params->rndnu_neon.a_zero_point;
  const int32_t vb_zero_point = params->rndnu_neon.b_zero_point;
  // E = 31 + shift = 31 - (left_pre_shift + left_post_shift) in [23, 46].
  const uint32_t vright_shift =
      (uint32_t) (31 - (int32_t) params->rndnu_neon.left_pre_shift
                    - (int32_t) params->rndnu_neon.left_post_shift);
  assert(vright_shift >= 23 && vright_shift <= 46);
  const int32_t vmultiplier = params->rndnu_neon.multiplier;
  assert(vmultiplier >= INT32_C(0x40000000));
  const int32_t voutput_zero_point = params->rndnu_neon.output_zero_point;
  const int8_t voutput_min = params->rndnu_neon.output_min;
  const int8_t voutput_max = params->rndnu_neon.output_max;

  // One vsetvl-driven loop at the widest legal layout: the i64 product stage
  // (LMUL 8) caps the chain at e8m1 -> i16m2 -> i32m4 -> i64m8, i.e. VLEN/8
  // elements per iteration. Peak live set is 18 register groups (a, b, two
  // i16 de-biased operands, the i32 product, the i64 accumulator) - checked
  // in the disassembly: no group spills. The tail is a smaller vl of the same
  // code path; vle8/vse8 touch exactly vl elements (no OOB reads or writes).
  do {
    size_t vl = __riscv_vsetvl_e8m1(batch); batch -= vl;
    const vint8m1_t va = __riscv_vle8_v_i8m1(input_a, vl); input_a += vl;
    const vint8m1_t vb = __riscv_vle8_v_i8m1(input_b, vl); input_b += vl;
    // De-biased operands, exact in i16: [-255, 255] for int8 zero points.
    const vint16m2_t vda = __riscv_vwsub_vx_i16m2(va, (int8_t) va_zero_point, vl);
    const vint16m2_t vdb = __riscv_vwsub_vx_i16m2(vb, (int8_t) vb_zero_point, vl);
    // Product of the de-biased operands, exact in i32: [-65025, 65025].
    const vint32m4_t vprod = __riscv_vwmul_vv_i32m4(vda, vdb, vl);
    // P * multiplier in i64: |P * multiplier| <= 65025 * 2^31 < 2^48.
    const vint64m8_t vacc = __riscv_vwmul_vx_i64m8(vprod, vmultiplier, vl);
    // Round-half-up shift right by E: RNU adds exactly 2^(E-1) before the
    // floor shift. The i32 stage cannot saturate (|r| <= 2^26 + 1).
    vint32m4_t vacc32 = __riscv_vnclip_wx_i32m4(vacc, vright_shift, __RISCV_VXRM_RNU, vl);
    // + output zero point, exact in i32 (|vacc32 + ozp| <= 2^26 + 2^15).
    // Equal to the Neon tail sat16(vacc) +qadd ozp -> sat8 because any value
    // beyond i8 saturates to the same limit through the i16 stage.
    vacc32 = __riscv_vsadd_vx_i32m4(vacc32, voutput_zero_point, vl);
    // Saturating narrows to i16 and i8: the Neon vqmovn/vqadd/vqmovn tail.
    const vint16m2_t vacc16 = __riscv_vnclip_wx_i16m2(vacc32, 0, __RISCV_VXRM_RDN, vl);
    vint8m1_t vout = __riscv_vnclip_wx_i8m1(vacc16, 0, __RISCV_VXRM_RDN, vl);
    // The family's min/max clamp, where the Neon kernel applies it (the init
    // pins output_min/output_max to INT8_MIN/INT8_MAX, so this is redundant
    // for producer params but keeps the row's minmax contract).
    vout = __riscv_vmax_vx_i8m1(vout, voutput_min, vl);
    vout = __riscv_vmin_vx_i8m1(vout, voutput_max, vl);
    __riscv_vse8_v_i8m1(output, vout, vl); output += vl;
  } while (batch != 0);
}
