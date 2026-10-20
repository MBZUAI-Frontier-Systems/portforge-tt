// clang-format off
// qs8-vcvt (requantized int8 convert) RVV microkernel.
//
// Arithmetic (derived from xnn_qs8_vcvt_ukernel__scalar_u1, the spec):
//
//   vbias = (int32)(((uint32)(int32) output_zero_point) << 8)
//           - multiplier * (int32) input_zero_point + 0x80
//   y     = sat8( (vbias + x * multiplier) >> 8 )       // arithmetic shift
//
// The +0x80 is the round-half-up constant for >> 8, and
// xnn_init_qs8_cvt_scalar_params produces multiplier = lrintf(256 * in_scale /
// out_scale) in [1, 32768] with int8-range zero points, so per element
//
//   y = sat8( round_half_up((x - input_zero_point) * multiplier / 256)
//             + output_zero_point ).
//
// Two multiplier regimes cover the init's whole output range:
//
// - multiplier in [1, 32767]: the i16 fractional-multiply chain. Every stage
//   is exact or redundant in range: |(x - izp) << 7| <= 32640 fits i16;
//   vsmul (RNU, ties up) reproduces the +0x80 rounding exactly and its result
//   |round((x-izp)*mult/256)| <= 32640 fits i16; vsadd(+ozp) fits i16; the
//   narrowing vnclip applies sat8, which is the scalar kernel's max/min clamp.
//   None of the intermediate saturations can trigger for producer params, so
//   the chain is bit-exact against the scalar (and the Neon) kernel.
// - multiplier == 32768: not representable as an i16 vsmul operand (upstream's
//   rvv variants truncate the int32 field to int16 and flip every sign there).
//   (x - izp) * 32768 / 256 = (x - izp) * 128 is an exact integer, so the
//   multiply and its rounding drop out of the chain entirely.
//
// Loop structure (fast path): the next chunk's vle8 is issued before the
// current chunk's arithmetic, so the load fills while the vector pipe works;
// chunks are independent. The chunk loop loads ahead, and the final chunk is
// finished by the epilogue after the loop (an early version that folded the
// epilogue into the loop left the last chunk unprocessed - the gtest caught
// it). The slow path is a plain loop. The tail is always a smaller vl of the
// same code path (no scalar epilogue, no OOB reads: vle8/vse8 touch exactly
// vl elements, and the lookahead runs with the next chunk's vl, 0 when the
// current chunk is the last). Both paths run e8m4 in / i16m8 math, the widest
// legal layout for this chain; LMUL 8 + the lookahead stays within the
// register file (no spills in the disassembly).
#include <assert.h>

#include <riscv_vector.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"

void xnn_qs8_vcvt_ukernel__rvv_pf1(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int16_t vinput_zero_point = params->scalar.input_zero_point;
  const int16_t voutput_zero_point = params->scalar.output_zero_point;
  const int32_t vmultiplier = params->scalar.multiplier;

  if (XNN_LIKELY(vmultiplier <= INT16_MAX)) {
    size_t vl = __riscv_vsetvl_e8m4(batch); batch -= vl;
    vint8m4_t vx = __riscv_vle8_v_i8m4(input, vl); input += vl;
    while (batch != 0) {
      // Load-ahead: the arithmetic below does not depend on the next chunk.
      const size_t vl_next = __riscv_vsetvl_e8m4(batch); batch -= vl_next;
      const vint8m4_t vx_next = __riscv_vle8_v_i8m4(input, vl_next); input += vl_next;

      vint16m8_t vacc = __riscv_vwsub_vx_i16m8(vx, vinput_zero_point, vl);
      vacc = __riscv_vsll_vx_i16m8(vacc, 7, vl);
      // RNU ties up: identical to the scalar's "+0x80 then floor-shift".
      vacc = __riscv_vsmul_vx_i16m8(vacc, (int16_t) vmultiplier, __RISCV_VXRM_RNU, vl);
      vacc = __riscv_vsadd_vx_i16m8(vacc, voutput_zero_point, vl);
      const vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_i8m4(output, vy, vl); output += vl;

      vx = vx_next; vl = vl_next;
    }
    // The chunk loaded last (or the only chunk) - not covered by the loop.
    vint16m8_t vacc = __riscv_vwsub_vx_i16m8(vx, vinput_zero_point, vl);
    vacc = __riscv_vsll_vx_i16m8(vacc, 7, vl);
    vacc = __riscv_vsmul_vx_i16m8(vacc, (int16_t) vmultiplier, __RISCV_VXRM_RNU, vl);
    vacc = __riscv_vsadd_vx_i16m8(vacc, voutput_zero_point, vl);
    const vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_i8m4(output, vy, vl); output += vl;
  } else {
    // multiplier == 32768: (x - izp) << 7 already is (x - izp) * mult / 256,
    // exactly; no fractional multiply exists at this scale in i16.
    assert(vmultiplier == 32768);
    do {
      size_t vl = __riscv_vsetvl_e8m4(batch); batch -= vl;
      const vint8m4_t vx = __riscv_vle8_v_i8m4(input, vl); input += vl;
      vint16m8_t vacc = __riscv_vwsub_vx_i16m8(vx, vinput_zero_point, vl);
      vacc = __riscv_vsll_vx_i16m8(vacc, 7, vl);
      vacc = __riscv_vsadd_vx_i16m8(vacc, voutput_zero_point, vl);
      const vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_i8m4(output, vy, vl); output += vl;
    } while (batch != 0);
  }
}
