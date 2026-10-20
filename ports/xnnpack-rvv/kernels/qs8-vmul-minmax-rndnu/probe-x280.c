// Probe: which vector op does the X280 reject? Executes one op at a time and
// prints a marker after each, so the last printed line names the culprit.
#include <stdio.h>
#include <riscv_vector.h>

int main(void) {
  size_t vl = __riscv_vsetvlmax_e8m1();
  printf("vlenb=%zu vl_e8m1=%zu\n", __riscv_vlenb(), vl); fflush(stdout);

  vint8m1_t a8 = __riscv_vmv_v_x_i8m1(3, vl);
  vint16m2_t a16 = __riscv_vmv_v_x_i16m2(5, __riscv_vsetvlmax_e16m2());
  vint32m4_t a32 = __riscv_vmv_v_x_i32m4(7, __riscv_vsetvlmax_e32m4());

  printf("vwsub\n"); fflush(stdout);
  vint16m2_t d16 = __riscv_vwsub_vx_i16m2(a8, 1, vl);
  printf("vwmul i32\n"); fflush(stdout);
  vint32m4_t p32 = __riscv_vwmul_vv_i32m4(d16, d16, __riscv_vsetvlmax_e32m4());
  printf("vwmul i64\n"); fflush(stdout);
  vint64m8_t p64 = __riscv_vwmul_vx_i64m8(p32, 1073741824, __riscv_vsetvlmax_e64m8());
  printf("vnclip i64->i32 RNU 23\n"); fflush(stdout);
  vint32m4_t r32 = __riscv_vnclip_wx_i32m4(p64, 23, __RISCV_VXRM_RNU, __riscv_vsetvlmax_e32m4());
  printf("vnclip i64->i32 RNU 46\n"); fflush(stdout);
  r32 = __riscv_vnclip_wx_i32m4(p64, 46, __RISCV_VXRM_RNU, __riscv_vsetvlmax_e32m4());
  printf("vsadd i32\n"); fflush(stdout);
  r32 = __riscv_vsadd_vx_i32m4(r32, 5, __riscv_vsetvlmax_e32m4());
  printf("vnclip i32->i16\n"); fflush(stdout);
  vint16m2_t r16 = __riscv_vnclip_wx_i16m2(r32, 0, __RISCV_VXRM_RDN, __riscv_vsetvlmax_e16m2());
  printf("vnclip i16->i8\n"); fflush(stdout);
  vint8m1_t r8 = __riscv_vnclip_wx_i8m1(r16, 0, __RISCV_VXRM_RDN, vl);
  printf("vmax/vmin i8\n"); fflush(stdout);
  r8 = __riscv_vmax_vx_i8m1(r8, -128, vl);
  r8 = __riscv_vmin_vx_i8m1(r8, 127, vl);
  printf("vse8 first=%d\n", (int) __riscv_vmv_x_s_i8m1_i8(r8)); fflush(stdout);
  return 0;
}
