/*****************************************************************************
 * looprestoration.S: riscv 8bpc loop restoration (Wiener filter)
 *****************************************************************************
 * Clean-room port of wiener_filter_h() and wiener_filter_v() from
 * src/looprestoration_tmpl.c.  The block-level buffer management stays in C
 * (see src/riscv/looprestoration.h); these two functions are the arithmetic.
 *
 * Horizontal pass, per output x (8bpc: round_bits_h = 3):
 *   sum  = 1 << 14                       (1 << (bitdepth + 6))
 *   sum += src[x] * 128                  (8bpc-only center contribution)
 *   sum += sum_{i=0..6} fh[i] * t(x + i - 3)
 *   dst[x] = iclip((sum + 4) >> 3, 0, (1 << 13) - 1)
 * with rounding_off_h = 4 and clip_limit = 1 << (8 + 1 + 7 - 3).  t(j) is
 * the source row with the C's edge rules:
 *   j <  0: left[0][4+j] if left, else src[j] if LR_HAVE_LEFT, else src[0]
 *   j >= w: src[j] if LR_HAVE_RIGHT, else src[w-1]
 * We materialize pad[0..w+5] = t(-3..w+2) and convolve it directly.
 *
 * Vertical pass, per output i (8bpc: round_bits_v = 11):
 *   sum  = -(1 << 18)                    (round_offset)
 *   sum += sum_{k=0..5} rows[k][i] * fv[k] + rows[6][i] * fv[6]
 *   dst[i] = iclip_pixel((sum + 1024) >> 11)
 * with rounding_off_v = 1 << 10; rows[6] is the 7th input row (identical to
 * rows[5] in the pure vertical pass, the fresh horizontal output in the hv
 * pass).  The -round_offset and +rounding_off_v constants are folded into a
 * single -261120 add; the >> is arithmetic and the clamp is [0, 255].
 *****************************************************************************/

#include "src/riscv/asm.S"

// void dav1d_wiener_filter_h_8bpc_rvv(uint16_t *dst, const pixel (*left)[4],
//                                     const pixel *src, const int16_t fh[8],
//                                     const int w, const enum LrEdgeFlags edges)
function wiener_filter_h_8bpc_rvv, export=1, ext=v
    addi    sp, sp, -416        # pad[0..w+5], w + 6 <= 390 bytes
    mv      t5, sp              # t5 = pad

    # ---- pad[0..2]: head edge handling
    andi    t0, a5, 1           # LR_HAVE_LEFT
    beqz    t0, L(no_left)
    bnez    a1, L(left_buf)
    # LR_HAVE_LEFT without a separate left buffer: copy src[-3..-1]
    addi    t1, a2, -3
    lbu     t2, 0(t1)
    sb      t2, 0(t5)
    lbu     t2, 1(t1)
    sb      t2, 1(t5)
    lbu     t2, 2(t1)
    sb      t2, 2(t5)
    j       L(head_done)
L(left_buf):
    addi    t1, a1, 1           # left[0][1..3]
    lbu     t2, 0(t1)
    sb      t2, 0(t5)
    lbu     t2, 1(t1)
    sb      t2, 1(t5)
    lbu     t2, 2(t1)
    sb      t2, 2(t5)
    j       L(head_done)
L(no_left):
    lbu     t2, 0(a2)
    sb      t2, 0(t5)
    sb      t2, 1(t5)
    sb      t2, 2(t5)
L(head_done):
    # ---- pad[3..w+2] = src[0..w-1]
    mv      t1, a2
    addi    t2, t5, 3
    mv      t3, a4
1:
    vsetvli t4, t3, e8, m1, ta, ma
    vle8.v  v16, (t1)
    vse8.v  v16, (t2)
    add     t1, t1, t4
    add     t2, t2, t4
    sub     t3, t3, t4
    bnez    t3, 1b
    # ---- pad[w+3..w+5]: tail edge handling
    add     t2, t5, 3
    add     t2, t2, a4
    andi    t0, a5, 2           # LR_HAVE_RIGHT
    bnez    t0, L(have_right)
    add     t1, a2, a4
    addi    t1, t1, -1          # src[w-1], replicated
    lbu     t3, 0(t1)
    sb      t3, 0(t2)
    sb      t3, 1(t2)
    sb      t3, 2(t2)
    j       L(tail_done)
L(have_right):
    add     t1, a2, a4          # src[w..w+2], read 3 past w like the C
    lbu     t3, 0(t1)
    sb      t3, 0(t2)
    lbu     t3, 1(t1)
    sb      t3, 1(t2)
    lbu     t3, 2(t1)
    sb      t3, 2(t2)
L(tail_done):
    # ---- taps: fh'[0..6], fh'[3] = fh[3] + 128 (8bpc center fold)
    lh      t0, 0(a3)
    lh      t1, 2(a3)
    lh      t2, 4(a3)
    lh      t3, 6(a3)
    addi    t3, t3, 128
    lh      t4, 8(a3)
    lh      t5, 10(a3)
    lh      t6, 12(a3)

    mv      a3, a0              # dst cursor
    mv      a1, sp              # pad cursor (left is dead now)
    mv      a5, a4              # remaining outputs (edges is dead now)
2:
    vsetvli a7, a5, e32, m4, ta, ma
    vsetvli zero, a7, e8, m1, ta, ma
    vle8.v  v8,  (a1)
    add     a0, a1, 1
    vle8.v  v9,  (a0)
    add     a0, a1, 2
    vle8.v  v10, (a0)
    add     a0, a1, 3
    vle8.v  v11, (a0)
    add     a0, a1, 4
    vle8.v  v12, (a0)
    add     a0, a1, 5
    vle8.v  v13, (a0)
    add     a0, a1, 6
    vle8.v  v14, (a0)
    vsetvli zero, a7, e16, m2, ta, ma
    vzext.vf2 v16, v8
    vsetvli zero, a7, e32, m4, ta, ma
    vzext.vf2 v20, v16
    vmv.v.i v28, 0
    vmacc.vx v28, t0, v20
    vsetvli zero, a7, e16, m2, ta, ma
    vzext.vf2 v16, v9
    vsetvli zero, a7, e32, m4, ta, ma
    vzext.vf2 v20, v16
    vmacc.vx v28, t1, v20
    vsetvli zero, a7, e16, m2, ta, ma
    vzext.vf2 v16, v10
    vsetvli zero, a7, e32, m4, ta, ma
    vzext.vf2 v20, v16
    vmacc.vx v28, t2, v20
    vsetvli zero, a7, e16, m2, ta, ma
    vzext.vf2 v16, v11
    vsetvli zero, a7, e32, m4, ta, ma
    vzext.vf2 v20, v16
    vmacc.vx v28, t3, v20
    vsetvli zero, a7, e16, m2, ta, ma
    vzext.vf2 v16, v12
    vsetvli zero, a7, e32, m4, ta, ma
    vzext.vf2 v20, v16
    vmacc.vx v28, t4, v20
    vsetvli zero, a7, e16, m2, ta, ma
    vzext.vf2 v16, v13
    vsetvli zero, a7, e32, m4, ta, ma
    vzext.vf2 v20, v16
    vmacc.vx v28, t5, v20
    vsetvli zero, a7, e16, m2, ta, ma
    vzext.vf2 v16, v14
    vsetvli zero, a7, e32, m4, ta, ma
    vzext.vf2 v20, v16
    vmacc.vx v28, t6, v20
    # (sum + rounding_off_h) >> round_bits_h, clip to [0, clip_limit - 1]
    li      a6, 16388           # (1 << 14) + rounding_off_h
    vadd.vx v28, v28, a6
    vsra.vi v28, v28, 3
    vmax.vx v28, v28, zero
    li      a6, 8191            # clip_limit - 1
    vmin.vx v28, v28, a6
    vsetvli zero, a7, e16, m2, ta, ma
    vncvt.x.x.w v16, v28
    vse16.v v16, (a3)
    slli    a6, a7, 1
    add     a3, a3, a6
    add     a1, a1, a7
    sub     a5, a5, a7
    bnez    a5, 2b

    addi    sp, sp, 416
    ret
endfunc

// void dav1d_wiener_filter_v_8bpc_rvv(pixel *dst,
//                                     const uint16_t *const rows[7],
//                                     const int16_t fv[8], const int w)
function wiener_filter_v_8bpc_rvv, export=1, ext=v
    addi    sp, sp, -32
    sd      s0, 0(sp)
    sd      s1, 8(sp)
    sd      s2, 16(sp)
    ld      t0, 0(a1)
    ld      t1, 8(a1)
    ld      t2, 16(a1)
    ld      t3, 24(a1)
    ld      t4, 32(a1)
    ld      t5, 40(a1)
    ld      t6, 48(a1)
    lh      a4, 0(a2)           # fv[0]
    lh      a5, 2(a2)           # fv[1]
    lh      a6, 4(a2)           # fv[2]
    lh      a7, 6(a2)           # fv[3]
    lh      s0, 8(a2)           # fv[4]
    lh      s1, 10(a2)          # fv[5]
    lh      s2, 12(a2)          # fv[6]
1:
    vsetvli a2, a3, e16, m1, ta, ma
    vle16.v v8,  (t0)
    vle16.v v9,  (t1)
    vle16.v v10, (t2)
    vle16.v v11, (t3)
    vle16.v v12, (t4)
    vle16.v v13, (t5)
    vle16.v v14, (t6)
    vsetvli zero, a2, e32, m2, ta, ma
    vzext.vf2 v16, v8
    vmv.v.i v24, 0
    vmacc.vx v24, a4, v16
    vzext.vf2 v16, v9
    vmacc.vx v24, a5, v16
    vzext.vf2 v16, v10
    vmacc.vx v24, a6, v16
    vzext.vf2 v16, v11
    vmacc.vx v24, a7, v16
    vzext.vf2 v16, v12
    vmacc.vx v24, s0, v16
    vzext.vf2 v16, v13
    vmacc.vx v24, s1, v16
    vzext.vf2 v16, v14
    vmacc.vx v24, s2, v16
    # (sum + rounding_off_v) >> round_bits_v, clamp to pixel range:
    # -round_offset + rounding_off_v = -261120 for 8bpc (a1 is free: the
    # rows[] array pointer was consumed by the ld's above)
    li      a1, -261120
    vadd.vx v24, v24, a1
    vsra.vi v24, v24, 11
    vmax.vx v24, v24, zero
    li      a1, 255
    vmin.vx v24, v24, a1
    vsetvli zero, a2, e16, m1, ta, ma
    vncvt.x.x.w v15, v24
    vsetvli zero, a2, e8, mf2, ta, ma
    vncvt.x.x.w v14, v15
    vse8.v  v14, (a0)
    slli    a1, a2, 1
    add     t0, t0, a1
    add     t1, t1, a1
    add     t2, t2, a1
    add     t3, t3, a1
    add     t4, t4, a1
    add     t5, t5, a1
    add     t6, t6, a1
    add     a0, a0, a2
    sub     a3, a3, a2
    bnez    a3, 1b

    ld      s0, 0(sp)
    ld      s1, 8(sp)
    ld      s2, 16(sp)
    addi    sp, sp, 32
    ret
endfunc
