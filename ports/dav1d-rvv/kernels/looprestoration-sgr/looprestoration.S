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

/*****************************************************************************
 * SGR (self-guided restoration), 8bpc.
 *
 * The stage kernels below are bit-exact transcriptions of the per-row
 * arithmetic of sgr_3x3_c()/sgr_5x5_c()/sgr_mix_c() from
 * src/looprestoration_tmpl.c; the block-level control flow (row buffers,
 * pointer rotation, startup shapes, bottom padding) stays in C in
 * src/riscv/looprestoration.h, mirroring the C reference.
 *
 * Exactness notes:
 *  - sgr_calc_row_ab computes z = (p * s + (1 << 19)) >> 20 in 32-bit
 *    unsigned arithmetic, and p * s does overflow for legal s values
 *    (e.g. 3x3: p <= 9*3*255^2, s = 3236).  The vector path keeps the
 *    modular product (vmul.vx) and the logical shift (vsrl) so the wrap
 *    matches the C bit for bit.
 *  - the A/B rows are indexed x = 0..w+1: the box row-h kernels write w+2
 *    entries (the C increments its pointers and stores x = -1..w, which
 *    covers the same memory), and the row-v / calc / finish kernels use
 *    the same range.
 *  - the finish kernels narrow their e32 result to coef (int16_t) with
 *    vncvt, reproducing the C's truncating int16 store exactly.
 *****************************************************************************/

// void dav1d_sgr_box3_row_h_8bpc_rvv(int32_t *sumsq, coef *sum,
//                                     const pixel (*left)[4],
//                                     const pixel *src, const int w,
//                                     const enum LrEdgeFlags edges)
// Writes sumsq[0..w+1] and sum[0..w+1]: output o is the 3-wide box over
// virtual positions o-2..o, materialized as pad[0..w+3] (v = j - 2):
//   pad[0..1] = HAVE_LEFT ? (left ? left[0][2..3] : src[-2..-1]) : src[0]
//   pad[j]    = src[j-2]                       (2 <= j <= w+1)
//   pad[w+2]  = HAVE_RIGHT ? src[w]   : src[w-1]
//   pad[w+3]  = HAVE_RIGHT ? src[w+1] : src[w-1]
function sgr_box3_row_h_8bpc_rvv, export=1, ext=v
    addi    sp, sp, -400        # pad[0..w+3], w + 4 <= 388 bytes
    mv      t5, sp
    andi    t0, a5, 1           # LR_HAVE_LEFT
    beqz    t0, L(sgr3_no_left)
    bnez    a2, L(sgr3_left_buf)
    addi    t1, a3, -2          # src[-2], src[-1]
    lbu     t2, 0(t1)
    sb      t2, 0(t5)
    lbu     t2, 1(t1)
    sb      t2, 1(t5)
    j       L(sgr3_head_done)
L(sgr3_left_buf):
    addi    t1, a2, 2           # left[0][2], left[0][3]
    lbu     t2, 0(t1)
    sb      t2, 0(t5)
    lbu     t2, 1(t1)
    sb      t2, 1(t5)
    j       L(sgr3_head_done)
L(sgr3_no_left):
    lbu     t2, 0(a3)
    sb      t2, 0(t5)
    sb      t2, 1(t5)
L(sgr3_head_done):
    mv      t1, a3              # pad[2..w+1] = src[0..w-1]
    addi    t2, t5, 2
    mv      t3, a4
1:
    vsetvli t4, t3, e8, m1, ta, ma
    vle8.v  v16, (t1)
    vse8.v  v16, (t2)
    add     t1, t1, t4
    add     t2, t2, t4
    sub     t3, t3, t4
    bnez    t3, 1b
    add     t2, t5, 2
    add     t2, t2, a4          # &pad[w+2]
    andi    t0, a5, 2           # LR_HAVE_RIGHT
    bnez    t0, L(sgr3_have_right)
    add     t1, a3, a4
    addi    t1, t1, -1
    lbu     t3, 0(t1)           # src[w-1], replicated
    sb      t3, 0(t2)
    sb      t3, 1(t2)
    j       L(sgr3_tail_done)
L(sgr3_have_right):
    add     t1, a3, a4          # src[w], src[w+1], read past w like the C
    lbu     t3, 0(t1)
    sb      t3, 0(t2)
    lbu     t3, 1(t1)
    sb      t3, 1(t2)
L(sgr3_tail_done):
    addi    a5, a4, 2           # n = w + 2 outputs (edges is dead)
    mv      a2, sp              # pad cursor (left is dead)
2:
    vsetvli a7, a5, e32, m4, ta, ma
    vsetvli zero, a7, e8, m1, ta, ma
    vle8.v  v8,  (a2)
    add     t0, a2, 1
    vle8.v  v9,  (t0)
    add     t0, a2, 2
    vle8.v  v10, (t0)
    # widen in steps: e8m1 -> e16m2 -> e32m4, each step under the vtype
    # whose implied source group matches how the data was last written
    vsetvli zero, a7, e16, m2, ta, ma
    vzext.vf2 v12, v8
    vzext.vf2 v14, v9
    vzext.vf2 v16, v10
    vadd.vv  v18, v12, v14
    vadd.vv  v18, v18, v16      # box3 sum (max 3*255), already coef-sized
    vse16.v  v18, (a1)
    vsetvli zero, a7, e32, m4, ta, ma
    vzext.vf2 v20, v12
    vzext.vf2 v24, v14
    vzext.vf2 v28, v16
    vmul.vv  v20, v20, v20
    vmacc.vv v20, v24, v24
    vmacc.vv v20, v28, v28      # box3 sum of squares (max 3*255^2)
    vse32.v  v20, (a0)
    slli    t0, a7, 2
    add     a0, a0, t0
    slli    t0, a7, 1
    add     a1, a1, t0
    add     a2, a2, a7
    sub     a5, a5, a7
    bnez    a5, 2b

    addi    sp, sp, 400
    ret
endfunc

// void dav1d_sgr_box3_row_v_8bpc_rvv(int32_t **sumsq, coef **sum,
//                                     int32_t *sumsq_out, coef *sum_out,
//                                     const int w)
// Sums rows [0..2] over x = 0..w+1 (the C's loop bound is w + 2).
function sgr_box3_row_v_8bpc_rvv, export=1, ext=v
    ld      t0, 0(a0)
    ld      t1, 8(a0)
    ld      t2, 16(a0)
    ld      t3, 0(a1)
    ld      t4, 8(a1)
    ld      t5, 16(a1)
    addi    a4, a4, 2
1:
    vsetvli a6, a4, e32, m2, ta, ma
    vle32.v v8,  (t0)
    vle32.v v10, (t1)
    vle32.v v12, (t2)
    vadd.vv v8, v8, v10
    vadd.vv v8, v8, v12
    vse32.v v8, (a2)
    vsetvli zero, a6, e16, m1, ta, ma
    vle16.v v8,  (t3)
    vle16.v v9,  (t4)
    vle16.v v10, (t5)
    vadd.vv v8, v8, v9
    vadd.vv v8, v8, v10
    vse16.v v8, (a3)
    slli    t6, a6, 2
    add     t0, t0, t6
    add     t1, t1, t6
    add     t2, t2, t6
    add     a2, a2, t6
    slli    t6, a6, 1
    add     t3, t3, t6
    add     t4, t4, t6
    add     t5, t5, t6
    add     a3, a3, t6
    sub     a4, a4, a6
    bnez    a4, 1b
    ret
endfunc

// void dav1d_sgr_calc_ab1_8bpc_rvv(int32_t *AA, coef *BB, const int w,
//                                   const unsigned s, const uint8_t *x_by_x)
// sgr_calc_row_ab() with the sgr_box3_vert constants (n = 9,
// one_by_x = 455), in place over i = 0..w+1:
//   a = AA[i], b = BB[i]                    (8bpc: no pre-rounding shift)
//   p = max(9*a - b*b, 0)                   signed
//   z = (p * s + (1 << 19)) >> 20           unsigned 32-bit (p*s wraps!)
//   x = x_by_x[umin(z, 255)]
//   AA[i] = (x * BB[i] * 455 + (1 << 11)) >> 12
//   BB[i] = x
function sgr_calc_ab1_8bpc_rvv, export=1, ext=v
    li      t0, 9               # n
    li      t1, 455             # one_by_x
    addi    a5, a2, 2           # w + 2 entries (a2 dead)
1:
    vsetvli a6, a5, e32, m2, ta, ma
    vle32.v v8, (a0)            # a = AA[i]
    vsetvli zero, a6, e16, m1, ta, ma
    vle16.v v12, (a1)           # b = BB[i]
    vsetvli zero, a6, e32, m2, ta, ma
    vsext.vf2 v10, v12
    vmul.vx  v14, v8, t0        # 9*a
    vnmsac.vv v14, v10, v10     # - b*b
    vmax.vx  v14, v14, zero     # p (signed >= 0)
    vmul.vx  v14, v14, a3       # p*s, modular like the C's unsigned wrap
    li      t2, 1 << 19
    vadd.vx  v14, v14, t2
    vsrl.vi  v14, v14, 20       # unsigned >> 20
    li      t3, 255
    vminu.vx v14, v14, t3       # umin(z, 255)
    vsetvli zero, a6, e16, m1, ta, ma
    vncvt.x.x.w v12, v14
    vsetvli zero, a6, e8, mf2, ta, ma
    vncvt.x.x.w v15, v12        # z as byte indices
    vluxei8.v v13, (a4), v15    # x = x_by_x[z]
    vsetvli zero, a6, e16, m1, ta, ma
    vzext.vf2 v12, v13
    vsetvli zero, a6, e32, m2, ta, ma
    vzext.vf2 v14, v12          # x as e32
    vmul.vv  v16, v14, v10      # x * BB
    vmul.vx  v16, v16, t1       # * one_by_x
    li      t2, 1 << 11
    vadd.vx  v16, v16, t2
    vsra.vi  v16, v16, 12
    vse32.v  v16, (a0)          # AA[i]
    vsetvli zero, a6, e16, m1, ta, ma
    vncvt.x.x.w v12, v14
    vse16.v  v12, (a1)          # BB[i] = x
    slli    t6, a6, 2
    add     a0, a0, t6
    slli    t6, a6, 1
    add     a1, a1, t6
    sub     a5, a5, a6
    bnez    a5, 1b
    ret
endfunc

// void dav1d_sgr_finish_filter_row1_8bpc_rvv(coef *tmp, const pixel *src,
//                                            int32_t **A_ptrs,
//                                            coef **B_ptrs, const int w)
// sgr_finish_filter_row1(): with c = i + 1,
//   a = (P1[c] + P1[c-1] + P1[c+1] + P0[c] + P2[c]) * 4
//     + (P0[c-1] + P2[c-1] + P0[c+1] + P2[c+1]) * 3   (B rows)
//   b = same over the A rows
//   tmp[i] = (b - a * src[i] + (1 << 8)) >> 9       (truncating i16 store)
// Rows are loaded at columns c-1, c, c+1 = base, base+1, base+2.
function sgr_finish_filter_row1_8bpc_rvv, export=1, ext=v
    ld      t0, 0(a3)           # B0
    ld      t1, 8(a3)           # B1
    ld      t2, 16(a3)          # B2
    ld      t4, 0(a2)           # A0
    ld      t5, 8(a2)           # A1
    ld      t6, 16(a2)          # A2
1:
    vsetvli a6, a4, e32, m2, ta, ma
    # ---- a: from the B rows (coef = int16_t)
    vsetvli zero, a6, e16, m1, ta, ma
    vle16.v v8,  (t0)           # B0[c-1..]
    add     t3, t0, 2
    vle16.v v9,  (t3)           # B0[c..]
    add     t3, t0, 4
    vle16.v v10, (t3)           # B0[c+1..]
    vsetvli zero, a6, e32, m2, ta, ma
    vsext.vf2 v16, v8
    vsext.vf2 v18, v9
    vsext.vf2 v20, v10
    vadd.vv  v22, v16, v20      # B0[c-1] + B0[c+1]
    vsetvli zero, a6, e16, m1, ta, ma
    vle16.v v8,  (t2)           # B2 rows, same three loads
    add     t3, t2, 2
    vle16.v v9,  (t3)
    add     t3, t2, 4
    vle16.v v10, (t3)
    vsetvli zero, a6, e32, m2, ta, ma
    vsext.vf2 v16, v8
    vsext.vf2 v24, v9
    vsext.vf2 v20, v10
    vadd.vv  v22, v22, v16      # + B2[c-1]
    vadd.vv  v22, v22, v20      # + B2[c+1]      (the *3 sum)
    vadd.vv  v18, v18, v24      # B0[c] + B2[c]
    vsetvli zero, a6, e16, m1, ta, ma
    vle16.v v8,  (t1)           # B1 rows
    add     t3, t1, 2
    vle16.v v9,  (t3)
    add     t3, t1, 4
    vle16.v v10, (t3)
    vsetvli zero, a6, e32, m2, ta, ma
    vsext.vf2 v16, v8
    vsext.vf2 v24, v9
    vsext.vf2 v20, v10
    vadd.vv  v18, v18, v16      # + B1[c-1]
    vadd.vv  v18, v18, v20      # + B1[c+1]
    vadd.vv  v18, v18, v24      # the *4 sum
    vsll.vi v24, v18, 2        # * 4
    li      t3, 3               # t2 still holds the B2 row pointer
    vmacc.vx v24, t3, v22       # a = *4 sum + 3 * (the *3 sum), in v24
    # ---- b: from the A rows (int32_t), same neighborhood
    vle32.v v8,  (t4)           # m2 groups are 2 registers apart: v8, v10, v12
    add     t3, t4, 4
    vle32.v v10, (t3)
    add     t3, t4, 8
    vle32.v v12, (t3)
    vadd.vv  v22, v8, v12       # A0[c-1] + A0[c+1]
    vadd.vi  v18, v10, 0     # A0[c]
    vle32.v v8,  (t6)
    add     t3, t6, 4
    vle32.v v10, (t3)
    add     t3, t6, 8
    vle32.v v12, (t3)
    vadd.vv  v22, v22, v8       # + A2[c-1]
    vadd.vv  v22, v22, v12      # + A2[c+1]   (the *3 sum)
    vadd.vv  v18, v18, v10      # + A2[c]
    vle32.v v8,  (t5)
    add     t3, t5, 4
    vle32.v v10, (t3)
    add     t3, t5, 8
    vle32.v v12, (t3)
    vadd.vv  v18, v18, v8       # + A1[c-1]
    vadd.vv  v18, v18, v10      # + A1[c+1]
    vadd.vv  v18, v18, v12      # the *4 sum
    vsll.vi v18, v18, 2
    li      t3, 3
    vmacc.vx v18, t3, v22       # b
    # ---- tmp[i] = (b - a * src[i] + (1 << 8)) >> 9
    vsetvli zero, a6, e8, mf2, ta, ma
    vle8.v  v8, (a1)
    vsetvli zero, a6, e16, m1, ta, ma
    vzext.vf2 v9, v8
    vsetvli zero, a6, e32, m2, ta, ma
    vzext.vf2 v28, v9
    vmul.vv  v30, v24, v28      # a * src[i]
    vsub.vv  v30, v18, v30      # b - a * src[i]
    li      t3, 1 << 8          # t2 still holds the B2 row pointer
    vadd.vx  v30, v30, t3
    vsra.vi  v30, v30, 9
    vsetvli zero, a6, e16, m1, ta, ma
    vncvt.x.x.w v12, v30        # truncating store to coef, like the C
    vse16.v  v12, (a0)
    slli    t3, a6, 1           # 2*vl bytes: tmp and the B rows
    add     a0, a0, t3
    add     t0, t0, t3
    add     t1, t1, t3
    add     t2, t2, t3
    add     a1, a1, a6          # src: 1 byte per pixel
    slli    t3, a6, 2           # 4*vl bytes: the A rows
    add     t4, t4, t3
    add     t5, t5, t3
    add     t6, t6, t3
    sub     a4, a4, a6
    bnez    a4, 1b
    ret
endfunc

// void dav1d_sgr_weighted_row1_8bpc_rvv(pixel *dst, const coef *t1,
//                                       const int w, const int w1)
// dst[i] = iclip_pixel(dst[i] + ((w1 * t1[i] + (1 << 10)) >> 11))
function sgr_weighted_row1_8bpc_rvv, export=1, ext=v
1:
    vsetvli a5, a2, e32, m2, ta, ma
    vsetvli zero, a5, e16, m1, ta, ma
    vle16.v v8, (a1)
    vsetvli zero, a5, e32, m2, ta, ma
    vsext.vf2 v10, v8
    vmul.vx  v10, v10, a3       # w1 * t1[i]
    li      t0, 1 << 10
    vadd.vx  v10, v10, t0
    vsra.vi  v10, v10, 11
    vsetvli zero, a5, e8, mf2, ta, ma
    vle8.v  v8, (a0)
    vsetvli zero, a5, e16, m1, ta, ma
    vzext.vf2 v9, v8
    vsetvli zero, a5, e32, m2, ta, ma
    vzext.vf2 v12, v9
    vadd.vv  v12, v12, v10
    vmax.vx  v12, v12, zero
    li      t0, 255
    vmin.vx  v12, v12, t0
    vsetvli zero, a5, e16, m1, ta, ma
    vncvt.x.x.w v8, v12
    vsetvli zero, a5, e8, mf2, ta, ma
    vncvt.x.x.w v9, v8
    vse8.v   v9, (a0)
    add     a0, a0, a5
    slli    t0, a5, 1
    add     a1, a1, t0
    sub     a2, a2, a5
    bnez    a2, 1b
    ret
endfunc

// void dav1d_sgr_box5_row_h_8bpc_rvv(int32_t *sumsq, coef *sum,
//                                     const pixel (*left)[4],
//                                     const pixel *src, const int w,
//                                     const enum LrEdgeFlags edges)
// The 5-wide box: output o covers virtual positions o-3..o+1, materialized
// as pad[0..w+5] (v = j - 3):
//   pad[0..2] = HAVE_LEFT ? (left ? left[0][1..3] : src[-3..-1]) : src[0]
//   pad[j]    = src[j-3]                       (3 <= j <= w+2)
//   pad[w+3..w+5] = HAVE_RIGHT ? src[w..w+2] : src[w-1]
function sgr_box5_row_h_8bpc_rvv, export=1, ext=v
    addi    sp, sp, -400        # pad[0..w+5], w + 6 <= 390 bytes
    mv      t5, sp
    andi    t0, a5, 1           # LR_HAVE_LEFT
    beqz    t0, L(sgr5_no_left)
    bnez    a2, L(sgr5_left_buf)
    addi    t1, a3, -3          # src[-3..-1]
    lbu     t2, 0(t1)
    sb      t2, 0(t5)
    lbu     t2, 1(t1)
    sb      t2, 1(t5)
    lbu     t2, 2(t1)
    sb      t2, 2(t5)
    j       L(sgr5_head_done)
L(sgr5_left_buf):
    addi    t1, a2, 1           # left[0][1..3]
    lbu     t2, 0(t1)
    sb      t2, 0(t5)
    lbu     t2, 1(t1)
    sb      t2, 1(t5)
    lbu     t2, 2(t1)
    sb      t2, 2(t5)
    j       L(sgr5_head_done)
L(sgr5_no_left):
    lbu     t2, 0(a3)
    sb      t2, 0(t5)
    sb      t2, 1(t5)
    sb      t2, 2(t5)
L(sgr5_head_done):
    mv      t1, a3              # pad[3..w+2] = src[0..w-1]
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
    add     t2, t5, 3
    add     t2, t2, a4          # &pad[w+3]
    andi    t0, a5, 2           # LR_HAVE_RIGHT
    bnez    t0, L(sgr5_have_right)
    add     t1, a3, a4
    addi    t1, t1, -1
    lbu     t3, 0(t1)           # src[w-1], replicated
    sb      t3, 0(t2)
    sb      t3, 1(t2)
    sb      t3, 2(t2)
    j       L(sgr5_tail_done)
L(sgr5_have_right):
    add     t1, a3, a4          # src[w..w+2], read past w like the C
    lbu     t3, 0(t1)
    sb      t3, 0(t2)
    lbu     t3, 1(t1)
    sb      t3, 1(t2)
    lbu     t3, 2(t1)
    sb      t3, 2(t2)
L(sgr5_tail_done):
    addi    a5, a4, 2           # n = w + 2 outputs (edges is dead)
    mv      a2, sp              # pad cursor (left is dead)
2:
    vsetvli a7, a5, e32, m4, ta, ma
    vsetvli zero, a7, e8, m1, ta, ma
    vle8.v  v8,  (a2)
    add     t0, a2, 1
    vle8.v  v9,  (t0)
    add     t0, a2, 2
    vle8.v  v10, (t0)
    add     t0, a2, 3
    vle8.v  v11, (t0)
    add     t0, a2, 4
    vle8.v  v12, (t0)
    # box5 sum in e16 (max 5*255)
    vsetvli zero, a7, e16, m2, ta, ma
    vzext.vf2 v16, v8
    vzext.vf2 v18, v9
    vadd.vv  v16, v16, v18
    vzext.vf2 v18, v10
    vadd.vv  v16, v16, v18
    vzext.vf2 v18, v11
    vadd.vv  v16, v16, v18
    vzext.vf2 v18, v12
    vadd.vv  v16, v16, v18
    vse16.v  v16, (a1)
    # box5 sum of squares in e32 (max 5*255^2); widen straight from the
    # byte loads with vf4 (source EMUL = m4/4 = m1, as loaded)
    vsetvli zero, a7, e32, m4, ta, ma
    vmv.v.i v20, 0
    vzext.vf4 v24, v8
    vmacc.vv v20, v24, v24
    vzext.vf4 v24, v9
    vmacc.vv v20, v24, v24
    vzext.vf4 v24, v10
    vmacc.vv v20, v24, v24
    vzext.vf4 v24, v11
    vmacc.vv v20, v24, v24
    vzext.vf4 v24, v12
    vmacc.vv v20, v24, v24
    vse32.v  v20, (a0)
    slli    t0, a7, 2
    add     a0, a0, t0
    slli    t0, a7, 1
    add     a1, a1, t0
    add     a2, a2, a7
    sub     a5, a5, a7
    bnez    a5, 2b

    addi    sp, sp, 400
    ret
endfunc

// void dav1d_sgr_box5_row_v_8bpc_rvv(int32_t **sumsq, coef **sum,
//                                     int32_t *sumsq_out, coef *sum_out,
//                                     const int w)
// Sums rows [0..4] over x = 0..w+1.
function sgr_box5_row_v_8bpc_rvv, export=1, ext=v
    ld      t0, 0(a0)
    ld      t1, 8(a0)
    ld      t2, 16(a0)
    ld      t3, 24(a0)
    ld      t4, 32(a0)
    ld      t5, 0(a1)
    ld      t6, 8(a1)
    addi    sp, sp, -32
    sd      s0, 0(sp)
    sd      s1, 8(sp)
    sd      s2, 16(sp)
    ld      s0, 16(a1)
    ld      s1, 24(a1)
    ld      s2, 32(a1)
    addi    a4, a4, 2
1:
    vsetvli a6, a4, e32, m2, ta, ma
    vle32.v v8,  (t0)
    vle32.v v10, (t1)
    vle32.v v12, (t2)
    vle32.v v14, (t3)
    vle32.v v16, (t4)
    vadd.vv v8, v8, v10
    vadd.vv v8, v8, v12
    vadd.vv v8, v8, v14
    vadd.vv v8, v8, v16
    vse32.v v8, (a2)
    vsetvli zero, a6, e16, m1, ta, ma
    vle16.v v8,  (t5)
    vle16.v v9,  (t6)
    vle16.v v10, (s0)
    vle16.v v11, (s1)
    vle16.v v12, (s2)
    vadd.vv v8, v8, v9
    vadd.vv v8, v8, v10
    vadd.vv v8, v8, v11
    vadd.vv v8, v8, v12
    vse16.v v8, (a3)
    slli    a5, a6, 2           # a5 is scratch here: t6 is a row pointer
    add     t0, t0, a5
    add     t1, t1, a5
    add     t2, t2, a5
    add     t3, t3, a5
    add     t4, t4, a5
    add     a2, a2, a5
    slli    a5, a6, 1
    add     t5, t5, a5
    add     t6, t6, a5
    add     s0, s0, a5
    add     s1, s1, a5
    add     s2, s2, a5
    add     a3, a3, a5
    sub     a4, a4, a6
    bnez    a4, 1b
    ld      s0, 0(sp)
    ld      s1, 8(sp)
    ld      s2, 16(sp)
    addi    sp, sp, 32
    ret
endfunc

// void dav1d_sgr_calc_ab2_8bpc_rvv(int32_t *AA, coef *BB, const int w,
//                                   const unsigned s, const uint8_t *x_by_x)
// sgr_calc_row_ab() with the sgr_box5_vert constants (n = 25,
// one_by_x = 164); identical arithmetic to calc_ab1, see there.
function sgr_calc_ab2_8bpc_rvv, export=1, ext=v
    li      t0, 25              # n
    li      t1, 164             # one_by_x
    addi    a5, a2, 2           # w + 2 entries (a2 dead)
1:
    vsetvli a6, a5, e32, m2, ta, ma
    vle32.v v8, (a0)            # a = AA[i]
    vsetvli zero, a6, e16, m1, ta, ma
    vle16.v v12, (a1)           # b = BB[i]
    vsetvli zero, a6, e32, m2, ta, ma
    vsext.vf2 v10, v12
    vmul.vx  v14, v8, t0        # 25*a
    vnmsac.vv v14, v10, v10     # - b*b
    vmax.vx  v14, v14, zero     # p (signed >= 0)
    vmul.vx  v14, v14, a3       # p*s, modular like the C's unsigned wrap
    li      t2, 1 << 19
    vadd.vx  v14, v14, t2
    vsrl.vi  v14, v14, 20       # unsigned >> 20
    li      t3, 255
    vminu.vx v14, v14, t3       # umin(z, 255)
    vsetvli zero, a6, e16, m1, ta, ma
    vncvt.x.x.w v12, v14
    vsetvli zero, a6, e8, mf2, ta, ma
    vncvt.x.x.w v15, v12        # z as byte indices
    vluxei8.v v13, (a4), v15    # x = x_by_x[z]
    vsetvli zero, a6, e16, m1, ta, ma
    vzext.vf2 v12, v13
    vsetvli zero, a6, e32, m2, ta, ma
    vzext.vf2 v14, v12          # x as e32
    vmul.vv  v16, v14, v10      # x * BB
    vmul.vx  v16, v16, t1       # * one_by_x
    li      t2, 1 << 11
    vadd.vx  v16, v16, t2
    vsra.vi  v16, v16, 12
    vse32.v  v16, (a0)          # AA[i]
    vsetvli zero, a6, e16, m1, ta, ma
    vncvt.x.x.w v12, v14
    vse16.v  v12, (a1)          # BB[i] = x
    slli    t6, a6, 2
    add     a0, a0, t6
    slli    t6, a6, 1
    add     a1, a1, t6
    sub     a5, a5, a6
    bnez    a5, 1b
    ret
endfunc

// void dav1d_sgr_finish_filter2_8bpc_rvv(coef *tmp, const pixel *src,
//                                        const ptrdiff_t src_stride,
//                                        int32_t **A_ptrs, coef **B_ptrs,
//                                        const int w, const int h)
// sgr_finish_filter2(): row 0 blends the two A/B rows (weight 6 center,
// 5 sides) and stores tmp[i] = (b - a*src[i] + (1 << 8)) >> 9; if h > 1 a
// second row from A/B_ptrs[1] alone with rounding (1 << 7) >> 8, stored at
// tmp + FILTER_OUT_STRIDE.  Column c = i + 1, rows loaded at columns
// c-1, c, c+1 = base, base+1, base+2.
function sgr_finish_filter2_8bpc_rvv, export=1, ext=v
    addi    sp, sp, -16
    sd      s1, 0(sp)
    mv      s1, a5              # s1 = w (both loops need it)
    ld      t0, 0(a4)           # B0 (a3 = A_ptrs, a4 = B_ptrs here)
    ld      t1, 8(a4)           # B1
    ld      t4, 0(a3)           # A0
    ld      t5, 8(a3)           # A1
    li      t2, 6
    li      t3, 5
    # ---- row 0: two-row neighborhood
1:
    vsetvli a7, a5, e32, m2, ta, ma
    vsetvli zero, a7, e16, m1, ta, ma
    vle16.v v8,  (t0)           # B0[c-1..]
    add     t6, t0, 2
    vle16.v v9,  (t6)           # B0[c..]
    add     t6, t0, 4
    vle16.v v10, (t6)           # B0[c+1..]
    vle16.v v11, (t1)           # B1[c-1..]
    add     t6, t1, 2
    vle16.v v12, (t6)           # B1[c..]
    add     t6, t1, 4
    vle16.v v13, (t6)           # B1[c+1..]
    vsetvli zero, a7, e32, m2, ta, ma
    vsext.vf2 v16, v9
    vsext.vf2 v18, v12
    vadd.vv  v22, v16, v18      # (B0c + B1c)
    vsext.vf2 v16, v8
    vsext.vf2 v18, v10
    vsext.vf2 v24, v11
    vsext.vf2 v26, v13
    vadd.vv  v24, v24, v26      # B1l + B1r
    vadd.vv  v24, v24, v16      # + B0l
    vadd.vv  v24, v24, v18      # + B0r      (the *5 sum)
    vmul.vx  v30, v22, t2       # * 6
    vmacc.vx v30, t3, v24       # a = *6 + 5 * (the *5 sum), in v30
    # ---- the same over the A rows (m2 groups: v8, v10, v12, v14, v16, v18)
    vle32.v v8,  (t4)
    add     t6, t4, 4
    vle32.v v10, (t6)
    add     t6, t4, 8
    vle32.v v12, (t6)
    vle32.v v14, (t5)
    add     t6, t5, 4
    vle32.v v16, (t6)
    add     t6, t5, 8
    vle32.v v18, (t6)
    vadd.vv  v22, v10, v16      # (A0c + A1c)
    vadd.vv  v24, v14, v18      # A1l + A1r
    vadd.vv  v24, v24, v8       # + A0l
    vadd.vv  v24, v24, v12      # + A0r
    vmul.vx  v18, v22, t2
    vmacc.vx v18, t3, v24       # b
    # ---- tmp[i] = (b - a * src[i] + (1 << 8)) >> 9
    vsetvli zero, a7, e8, mf2, ta, ma
    vle8.v  v8, (a1)
    vsetvli zero, a7, e16, m1, ta, ma
    vzext.vf2 v9, v8
    vsetvli zero, a7, e32, m2, ta, ma
    vzext.vf2 v28, v9
    vmul.vv  v16, v30, v28      # a * src[i]
    vsub.vv  v30, v18, v16      # b - a * src[i]
    li      t6, 1 << 8
    vadd.vx  v30, v30, t6
    vsra.vi  v30, v30, 9
    vsetvli zero, a7, e16, m1, ta, ma
    vncvt.x.x.w v12, v30
    vse16.v  v12, (a0)
    # advance row-0 cursors
    slli    t6, a7, 1
    add     a0, a0, t6
    add     t0, t0, t6
    add     t1, t1, t6
    add     a1, a1, a7
    slli    t6, a7, 2
    add     t4, t4, t6
    add     t5, t5, t6
    sub     a5, a5, a7
    bnez    a5, 1b

    beqz    a6, L(sgrf2_done)
    # ---- row 1 (h > 1): from A/B_ptrs[1] alone, rounding (1 << 7) >> 8.
    # tmp = tmp_base + FILTER_OUT_STRIDE (384 coefs); src = src_base + stride.
    # a0/a1 currently sit at base + w: undo w, add the full offsets.
    slli    t6, s1, 1
    sub     a0, a0, t6
    addi    a0, a0, 2 * 384     # 384 coef * 2 bytes
    sub     a1, a1, s1
    add     a1, a1, a2          # src += src_stride
    ld      t0, 8(a4)           # B1 (base of the row-1 neighborhood)
    ld      t4, 8(a3)           # A1
    mv      a5, s1
1:
    vsetvli a7, a5, e32, m2, ta, ma
    vsetvli zero, a7, e16, m1, ta, ma
    vle16.v v8,  (t0)           # B[c-1..]
    add     t6, t0, 2
    vle16.v v9,  (t6)           # B[c..]
    add     t6, t0, 4
    vle16.v v10, (t6)           # B[c+1..]
    vsetvli zero, a7, e32, m2, ta, ma
    vsext.vf2 v22, v9
    vsext.vf2 v24, v8
    vsext.vf2 v26, v10
    vmul.vx  v16, v22, t2       # center * 6
    vmacc.vx v16, t3, v24       # + left * 5
    vmacc.vx v16, t3, v26       # + right * 5    (a)
    vle32.v v8,  (t4)           # m2 groups: v8, v10, v12
    add     t6, t4, 4
    vle32.v v10, (t6)
    add     t6, t4, 8
    vle32.v v12, (t6)
    vmul.vx  v18, v10, t2       # center * 6
    vmacc.vx v18, t3, v8        # + left * 5
    vmacc.vx v18, t3, v12       # + right * 5   (b)
    vsetvli zero, a7, e8, mf2, ta, ma
    vle8.v  v8, (a1)
    vsetvli zero, a7, e16, m1, ta, ma
    vzext.vf2 v9, v8
    vsetvli zero, a7, e32, m2, ta, ma
    vzext.vf2 v28, v9
    vmul.vv  v30, v16, v28
    vsub.vv  v30, v18, v30
    li      t6, 1 << 7
    vadd.vx  v30, v30, t6
    vsra.vi  v30, v30, 8
    vsetvli zero, a7, e16, m1, ta, ma
    vncvt.x.x.w v12, v30
    vse16.v  v12, (a0)
    slli    t6, a7, 1
    add     a0, a0, t6
    add     t0, t0, t6
    add     a1, a1, a7
    slli    t6, a7, 2           # the A row is int32: 4 bytes per element
    add     t4, t4, t6
    sub     a5, a5, a7
    bnez    a5, 1b
L(sgrf2_done):
    ld      s1, 0(sp)
    addi    sp, sp, 16
    ret
endfunc

// void dav1d_sgr_weighted2_8bpc_rvv(pixel *dst, const ptrdiff_t dst_stride,
//                                    const coef *t1, const coef *t2,
//                                    const int w, const int h,
//                                    const int w0, const int w1)
// dst[i] = iclip_pixel(dst[i] + ((w0*t1[i] + w1*t2[i] + (1 << 10)) >> 11));
// t1/t2 advance by FILTER_OUT_STRIDE (384) coefs per row, dst by dst_stride.
function sgr_weighted2_8bpc_rvv, export=1, ext=v
    addi    sp, sp, -32
    sd      s0, 0(sp)
    sd      s1, 8(sp)
    sd      s2, 16(sp)
    mv      s0, a4               # w
    mv      s1, a6               # w0
    mv      s2, a7               # w1
1:
    mv      a4, s0
2:
    vsetvli a6, a4, e32, m2, ta, ma
    vsetvli zero, a6, e16, m1, ta, ma
    vle16.v v8,  (a2)
    vle16.v v9,  (a3)
    vsetvli zero, a6, e32, m2, ta, ma
    vsext.vf2 v10, v8
    vsext.vf2 v12, v9
    vmul.vx  v10, v10, s1       # w0 * t1[i]
    vmacc.vx v10, s2, v12       # v = w0*t1[i] + w1*t2[i]
    li      t0, 1 << 10
    vadd.vx  v10, v10, t0
    vsra.vi  v10, v10, 11
    vsetvli zero, a6, e8, mf2, ta, ma
    vle8.v  v8, (a0)
    vsetvli zero, a6, e16, m1, ta, ma
    vzext.vf2 v9, v8
    vsetvli zero, a6, e32, m2, ta, ma
    vzext.vf2 v12, v9
    vadd.vv  v12, v12, v10
    vmax.vx  v12, v12, zero
    li      t0, 255
    vmin.vx  v12, v12, t0
    vsetvli zero, a6, e16, m1, ta, ma
    vncvt.x.x.w v8, v12
    vsetvli zero, a6, e8, mf2, ta, ma
    vncvt.x.x.w v9, v8
    vse8.v   v9, (a0)
    add     a0, a0, a6
    slli    t0, a6, 1
    add     a2, a2, t0
    add     a3, a3, t0
    sub     a4, a4, a6
    bnez    a4, 2b
    # next output row: rewind the w-wide walk, then apply the C's strides
    # (dst += dst_stride, t1/t2 += FILTER_OUT_STRIDE coefs)
    slli    t0, s0, 1
    sub     a2, a2, t0
    sub     a3, a3, t0
    sub     a0, a0, s0
    add     a0, a0, a1
    li      t0, 384 * 2
    add     a2, a2, t0
    add     a3, a3, t0
    addi    a5, a5, -1
    bnez    a5, 1b
    ld      s0, 0(sp)
    ld      s1, 8(sp)
    ld      s2, 16(sp)
    addi    sp, sp, 32
    ret
endfunc
