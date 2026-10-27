/* Loop filter (deblocking) RVV port - 8bpc.
 *
 * C reference: src/loopfilter_tmpl.c loop_filter() (+ the four sb128
 * wrappers). Per set bit in the vmask, the C filters the 4 pixel rows
 * (horizontal variant, strideb = 1) or columns (vertical variant,
 * strideb = stride) crossing the block edge with the AV1 deblocking filter
 * of width wd in {4, 6, 8, 16}.
 *
 * Structure of this port:
 *  - the wrappers keep the vmask iteration and per-bit level lookups in
 *    scalar code (control flow, not pixel data) and mirror the C exactly;
 *  - lpf_core_rvv vectorizes the 4 independent lines of one vmask bit as
 *    4 vector lanes. VL is always exactly 4 (the C loop bound is the
 *    constant 4) - there is no tail and no scalar epilogue. The taps
 *    p6..q6 are strided loads (stride = stridea, base = dst + k*strideb),
 *    the stores are the mirrored strided stores; all arithmetic is
 *    lane-wise e16 replicating the C's int arithmetic, including
 *    iclip_diff's [-128, 127] clamp and iclip_pixel's [0, 255] clamp;
 *  - the per-lane flat/hev decisions become mask registers, so a mixed
 *    edge behaves exactly like the C writes it: the plain (hev-branch)
 *    stores go first with fm / fm&!hev masks, then the flat paths re-store
 *    the lanes they own with their own masks (same addresses, so the last
 *    masked store per lane wins, exactly like the C's if/else-if chain).
 *    A lane the C does not write (dst[-2] of a hev lane) is never stored.
 */

#include "src/riscv/asm.S"

/* clip signed e16 vector \vec to [-128, 127] (iclip_diff, 8bpc) */
.macro clip_diff vec
        li t6, 127
        vmin.vx \vec, \vec, t6
        li t6, -128
        vmax.vx \vec, \vec, t6
.endm

/* clamp signed e16 vector to [0, 255] (iclip_pixel, 8bpc) */
.macro clip_u8 vec
        vmax.vx \vec, \vec, zero
        li t6, 255
        vmin.vx \vec, \vec, t6
.endm

/* store clamped e16 \vec to dst + \off*strideb, lanes enabled by v0.
 * \off in [-6, 5]; kills t1; expects t2 = 2*strideb, t3 = 3*strideb,
 * t4 = 4*strideb. */
.macro store_lane vec, off
        clip_u8 \vec
.if \off == 0
        mv t1, a0
.else
.if \off == -1
        sub t1, a0, a5
.else
.if \off == -2
        sub t1, a0, t2
.else
.if \off == -3
        sub t1, a0, t3
.else
.if \off == -4
        sub t1, a0, t4
.else
.if \off == -5
        sub t1, a0, t4
        sub t1, t1, a5
.else
.if \off == -6
        sub t1, a0, t4
        sub t1, t1, t2
.else
.if \off == 1
        add t1, a0, a5
.else
.if \off == 2
        add t1, a0, t2
.else
.if \off == 3
        add t1, a0, t3
.else
.if \off == 4
        add t1, a0, t4
.else
        add t1, a0, t4
        add t1, t1, a5
.endif
.endif
.endif
.endif
.endif
.endif
.endif
.endif
.endif
.endif
.endif
        vsse8.v \vec, (t1), a4, v0.t
.endm

/* load tap dst + \off*strideb widened to e16 into \vec (kills t1, t2) */
.macro load_tap vec, off
        .if \off == 0
        vlse8.v v19, (a0), a4
    .else
        .if \off == -1
li t2, 0
add t2, t2, a5
sub t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
        .if \off == -2
li t2, 0
slli t1, a5, 1
add t2, t2, t1
sub t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
        .if \off == -3
li t2, 0
add t2, t2, a5
slli t1, a5, 1
add t2, t2, t1
sub t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
        .if \off == -4
li t2, 0
slli t1, a5, 2
add t2, t2, t1
sub t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
        .if \off == -5
li t2, 0
add t2, t2, a5
slli t1, a5, 2
add t2, t2, t1
sub t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
        .if \off == -6
li t2, 0
slli t1, a5, 1
add t2, t2, t1
slli t1, a5, 2
add t2, t2, t1
sub t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
        .if \off == -7
li t2, 0
add t2, t2, a5
slli t1, a5, 1
add t2, t2, t1
slli t1, a5, 2
add t2, t2, t1
sub t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
        .if \off == 1
li t2, 0
add t2, t2, a5
add t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
        .if \off == 2
li t2, 0
slli t1, a5, 1
add t2, t2, t1
add t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
        .if \off == 3
li t2, 0
add t2, t2, a5
slli t1, a5, 1
add t2, t2, t1
add t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
        .if \off == 4
li t2, 0
slli t1, a5, 2
add t2, t2, t1
add t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
        .if \off == 5
li t2, 0
add t2, t2, a5
slli t1, a5, 2
add t2, t2, t1
add t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
        .if \off == 6
li t2, 0
slli t1, a5, 1
add t2, t2, t1
slli t1, a5, 2
add t2, t2, t1
add t1, a0, t2
        vlse8.v v19, (t1), a4
    .else
li t2, 0
add t2, t2, a5
slli t1, a5, 1
add t2, t2, t1
slli t1, a5, 2
add t2, t2, t1
add t1, a0, t2
        vlse8.v v19, (t1), a4
.endif
.endif
.endif
.endif
.endif
.endif
.endif
.endif
.endif
.endif
.endif
.endif
.endif
.endif
        vzext.vf2 \vec, v19   /* widening ops need vd != vs2 */
.endm

/* \out = |\a - \b|, kills \tmp */
.macro abs_diff out, a, b, tmp
        vsub.vv \out, \a, \b
        vneg.v \tmp, \out
        vmax.vv \out, \out, \tmp
.endm

/* \m &= (|\a - \b| <= F), F = 1 for 8bpc; kills \out, \tmp */
.macro flat1 m, a, b, out, tmp
        abs_diff \out, \a, \b, \tmp
        vmsle.vi \out, \out, 1
        vmand.mm \m, \m, \out
.endm

/*
 * flat8 6 outputs (wd8; also the flat8in sub-path of wd16); taps
 * p3 v7, p2 v5, p1 v1, p0 v2, q0 v3, q1 v4, q2 v6, q3 v8; every store is
 * masked with \sel (moved into v0).
 */
.macro flat8_body sel
        /* dst[-3] = (3*p3 + 2*p2 + p1 + p0 + q0 + 4) >> 3 */
        vmv.v.i v15, 4
        li t6, 3
        vmacc.vx v15, t6, v7
        li t6, 2
        vmacc.vx v15, t6, v5
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        vsrl.vi v15, v15, 3
        vmmv.m v0, \sel
        store_lane v15, -3
        /* dst[-2] = (2*p3 + p2 + 2*p1 + p0 + q0 + q1 + 4) >> 3 */
        vmv.v.i v15, 4
        li t6, 2
        vmacc.vx v15, t6, v7
        li t6, 1
        vmacc.vx v15, t6, v5
        li t6, 2
        vmacc.vx v15, t6, v1
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        vsrl.vi v15, v15, 3
        vmmv.m v0, \sel
        store_lane v15, -2
        /* dst[-1] = (p3 + p2 + p1 + 2*p0 + q0 + q1 + q2 + 4) >> 3 */
        vmv.v.i v15, 4
        li t6, 1
        vmacc.vx v15, t6, v7
        li t6, 1
        vmacc.vx v15, t6, v5
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 2
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        li t6, 1
        vmacc.vx v15, t6, v6
        vsrl.vi v15, v15, 3
        vmmv.m v0, \sel
        store_lane v15, -1
        /* dst[+0] = (p2 + p1 + p0 + 2*q0 + q1 + q2 + q3 + 4) >> 3 */
        vmv.v.i v15, 4
        li t6, 1
        vmacc.vx v15, t6, v5
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 2
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        li t6, 1
        vmacc.vx v15, t6, v6
        li t6, 1
        vmacc.vx v15, t6, v8
        vsrl.vi v15, v15, 3
        vmmv.m v0, \sel
        store_lane v15, 0
        /* dst[+1] = (p1 + p0 + q0 + 2*q1 + q2 + 2*q3 + 4) >> 3 */
        vmv.v.i v15, 4
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        li t6, 2
        vmacc.vx v15, t6, v4
        li t6, 1
        vmacc.vx v15, t6, v6
        li t6, 2
        vmacc.vx v15, t6, v8
        vsrl.vi v15, v15, 3
        vmmv.m v0, \sel
        store_lane v15, 1
        /* dst[+2] = (p0 + q0 + q1 + 2*q2 + 3*q3 + 4) >> 3 */
        vmv.v.i v15, 4
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        li t6, 2
        vmacc.vx v15, t6, v6
        li t6, 3
        vmacc.vx v15, t6, v8
        vsrl.vi v15, v15, 3
        vmmv.m v0, \sel
        store_lane v15, 2
.endm

/*
 * lpf_core_rvv(pixel *dst, int E, int I, int H,
 *              ptrdiff_t stridea, ptrdiff_t strideb, int wd)
 * Internal (not exported): called by the four sb wrappers with a0..a6.
 * Clobbers t0..t6, a1..a6 and the vector registers.
 *
 * Vector registers (e16, VL = 4):
 *   v1..v4   p1 p0 q0 q1              v5..v8   p2 q2 p3 q3
 *   v9..v14  p4 p5 p6 q4 q5 q6        v15..v19 temporaries
 *   v16 |p1-p0|, v17 |q1-q0| (live until hev is computed)
 *   v20 fm, v21 hev, v22 flat8in, v23 flat8out/sel, v24..v28 constants
 *   v0 store mask
 */
function lpf_core_rvv, ext="v"
        csrwi vxrm, 0            /* round-to-nearest for the narrowing ops */
        vsetivli zero, 4, e16, m1, ta, ma

        load_tap v1, -2
        load_tap v2, -1
        load_tap v3, 0
        load_tap v4, 1

        abs_diff v16, v1, v2, v19 /* |p1-p0| (fm + hev) */
        abs_diff v17, v4, v3, v19 /* |q1-q0| (fm + hev) */

        /* fm = |p1-p0| <= I && |q1-q0| <= I && 2*|p0-q0| + (|p1-q1|>>1) <= E */
        vmv.v.x v24, a2          /* I */
        vmv.v.x v25, a1          /* E */
        vmsle.vv v20, v16, v24
        vmsle.vv v19, v17, v24
        vmand.mm v20, v20, v19
        abs_diff v18, v2, v3, v19 /* |p0-q0| */
        vsll.vi v18, v18, 1
        abs_diff v19, v1, v4, v15 /* |p1-q1| */
        vsrl.vi v19, v19, 1
        vadd.vv v18, v18, v19
        vmsle.vv v19, v18, v25
        vmand.mm v20, v20, v19    /* v20 = fm */

        /* per-width loads and extended fm terms */
        li t0, 16
        beq a6, t0, L(core_wd16_pre)
        li t0, 8
        beq a6, t0, L(core_wd8_pre)
        li t0, 6
        beq a6, t0, L(core_wd6_pre)
        j L(core_hev)            /* wd == 4 */

L(core_wd6_pre):
        load_tap v5, -3          /* p2 = dst[-3*strideb] */
        load_tap v6, 2           /* q2 = dst[+2*strideb] */
        /* fm &= |p2-p1| <= I && |q2-q1| <= I */
        abs_diff v18, v5, v1, v19
        vmsle.vv v19, v18, v24
        vmand.mm v20, v20, v19
        abs_diff v18, v6, v4, v19
        vmsle.vv v19, v18, v24
        vmand.mm v20, v20, v19
        j L(core_hev)

L(core_wd8_pre):
        load_tap v5, -3
        load_tap v6, 2
        load_tap v7, -4          /* p3 = dst[-4*strideb] */
        load_tap v8, 3
        /* fm &= |p2-p1|, |q2-q1|, |p3-p2|, |q3-q2| <= I */
        abs_diff v18, v5, v1, v19
        vmsle.vv v19, v18, v24
        vmand.mm v20, v20, v19
        abs_diff v18, v6, v4, v19
        vmsle.vv v19, v18, v24
        vmand.mm v20, v20, v19
        abs_diff v18, v7, v5, v19
        vmsle.vv v19, v18, v24
        vmand.mm v20, v20, v19
        abs_diff v18, v8, v6, v19
        vmsle.vv v19, v18, v24
        vmand.mm v20, v20, v19
        j L(core_hev)

L(core_wd16_pre):
        load_tap v5, -3
        load_tap v6, 2
        load_tap v7, -4
        load_tap v8, 3
        load_tap v9, -5          /* p4 = dst[-5*strideb] */
        load_tap v10, -6         /* p5 = dst[-6*strideb] */
        load_tap v11, -7         /* p6 = dst[-7*strideb] */
        load_tap v12, 4          /* q4 = dst[+4*strideb] */
        load_tap v13, 5          /* q5 */
        load_tap v14, 6          /* q6 */
        /* the same four extended fm terms as wd8 */
        abs_diff v18, v5, v1, v19
        vmsle.vv v19, v18, v24
        vmand.mm v20, v20, v19
        abs_diff v18, v6, v4, v19
        vmsle.vv v19, v18, v24
        vmand.mm v20, v20, v19
        abs_diff v18, v7, v5, v19
        vmsle.vv v19, v18, v24
        vmand.mm v20, v20, v19
        abs_diff v18, v8, v6, v19
        vmsle.vv v19, v18, v24
        vmand.mm v20, v20, v19

L(core_hev):
        /* strideb multiples for store_lane (no load_tap past this point) */
        slli t2, a5, 1
        slli t3, a5, 1
        add t3, t3, a5
        slli t4, a5, 2

        /* skip everything when no lane passes fm */
        vfirst.m t0, v20
        bltz t0, L(core_done)

        /* hev = |p1-p0| > H || |q1-q0| > H */
        vmv.v.x v26, a3          /* H */
        vmsgt.vv v21, v16, v26
        vmsgt.vv v19, v17, v26
        vmor.mm v21, v21, v19     /* v21 = hev */

        /* plain path (the C's else branch), correct for every lane:
         * f  = iclip_diff(3*(q0-p0) + (hev ? iclip_diff(p1-q1) : 0))
         * f1 = imin(f + 4, 127) >> 3 ; f2 = imin(f + 3, 127) >> 3
         * dst[-1] = clip_u8(p0 + f2) ; dst[+0] = clip_u8(q0 - f1)
         * !hev:   dst[-2] = clip_u8(p1 + (f1+1)>>1), dst[+1] = clip_u8(q1-..)
         */
        vsub.vv v18, v3, v2      /* q0-p0 */
        vsll.vi v18, v18, 1
        vneg.v v19, v18
        vsub.vv v15, v3, v2      /* q0-p0 */
        vadd.vv v18, v18, v15    /* 3*(q0-p0) */
        vsub.vv v19, v1, v4      /* p1-q1 */
        clip_diff v19
        vmv.v.i v15, 0
        vmmv.m v0, v21
        vmerge.vvm v19, v15, v19, v0 /* hev ? clip(p1-q1) : 0 */
        vadd.vv v18, v18, v19
        clip_diff v18            /* v18 = f */
        li t0, 127
        vmv.v.x v28, t0
        vadd.vi v19, v18, 4
        vmin.vv v19, v19, v28
        vsrl.vi v19, v19, 3      /* v19 = f1 */
        vadd.vi v15, v18, 3
        vmin.vv v15, v15, v28
        vsrl.vi v15, v15, 3      /* v15 = f2 */
        vadd.vv v16, v2, v15
        clip_u8 v16              /* dst[-1]; v15 (f2) dead below */
        vsub.vv v17, v3, v19
        clip_u8 v17              /* dst[+0]; v19 (f1) dead below */
        vadd.vi v18, v19, 1
        vsrl.vi v18, v18, 1      /* (f1 + 1) >> 1 */
        vadd.vv v15, v1, v18
        clip_u8 v15              /* dst[-2] (non-hev lanes); v12/v13 (q4/q5
                                  * for wd16) must survive the hev section */
        vsub.vv v19, v4, v18
        clip_u8 v19              /* dst[+1] (non-hev lanes) */

        vmmv.m v0, v20
        store_lane v16, -1
        store_lane v17, 0
        vmnand.mm v0, v20, v21    /* !(fm & hev) */
        vmand.mm v0, v0, v20      /* fm & !hev */
        store_lane v15, -2
        store_lane v19, 1

        li t0, 16
        beq a6, t0, L(core_wd16_flat)
        li t0, 8
        beq a6, t0, L(core_wd8_flat)
        li t0, 6
        beq a6, t0, L(core_wd6_flat)
        j L(core_done)           /* wd == 4: no flat path */

/* ---- wd == 6: flat8in only (chroma) ------------------------------------ */
L(core_wd6_flat):
        vmv.v.i v22, -1          /* flat8in, and-merged below */
        flat1 v22, v5, v2, v18, v19 /* |p2-p0| <= F */
        flat1 v22, v1, v2, v18, v19 /* |p1-p0| <= F */
        flat1 v22, v4, v3, v18, v19 /* |q1-q0| <= F */
        flat1 v22, v6, v2, v18, v19 /* |q2-q0| <= F */
        vmand.mm v23, v20, v22    /* sel6 = fm & flat8in */
        vfirst.m t0, v23
        bltz t0, L(core_done)
        /* dst[-2] = (p2 + 2*p2 + 2*p1 + 2*p0 + q0 + 4) >> 3 */
        vmv.v.i v15, 4
        li t6, 3
        vmacc.vx v15, t6, v5
        li t6, 2
        vmacc.vx v15, t6, v1
        li t6, 2
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        vsrl.vi v15, v15, 3
        vmmv.m v0, v23
        store_lane v15, -2
        /* dst[-1] = (p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3 */
        vmv.v.i v15, 4
        li t6, 1
        vmacc.vx v15, t6, v5
        li t6, 2
        vmacc.vx v15, t6, v1
        li t6, 2
        vmacc.vx v15, t6, v2
        li t6, 2
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        vsrl.vi v15, v15, 3
        vmmv.m v0, v23
        store_lane v15, -1
        /* dst[+0] = (p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4) >> 3 */
        vmv.v.i v15, 4
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 2
        vmacc.vx v15, t6, v2
        li t6, 2
        vmacc.vx v15, t6, v3
        li t6, 2
        vmacc.vx v15, t6, v4
        li t6, 1
        vmacc.vx v15, t6, v6
        vsrl.vi v15, v15, 3
        vmmv.m v0, v23
        store_lane v15, 0
        /* dst[+1] = (p0 + 2*q0 + 2*q1 + 3*q2 + 4) >> 3 */
        vmv.v.i v15, 4
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 2
        vmacc.vx v15, t6, v3
        li t6, 2
        vmacc.vx v15, t6, v4
        li t6, 3
        vmacc.vx v15, t6, v6
        vsrl.vi v15, v15, 3
        vmmv.m v0, v23
        store_lane v15, 1
        j L(core_done)

/* ---- wd == 8: flat8in (luma) -------------------------------------------- */
L(core_wd8_flat):
        vmv.v.i v22, -1
        flat1 v22, v5, v2, v18, v19 /* |p2-p0| <= F */
        flat1 v22, v1, v2, v18, v19 /* |p1-p0| <= F */
        flat1 v22, v4, v3, v18, v19 /* |q1-q0| <= F */
        flat1 v22, v6, v2, v18, v19 /* |q2-q0| <= F */
        flat1 v22, v7, v2, v18, v19 /* |p3-p0| <= F */
        flat1 v22, v8, v2, v18, v19 /* |q3-q0| <= F */
        vmand.mm v23, v20, v22    /* sel8 = fm & flat8in */
        vfirst.m t0, v23
        bltz t0, L(core_done)
        flat8_body v23
        j L(core_done)

/* ---- wd == 16: flat16 / flat8in (luma) ---------------------------------- */
L(core_wd16_flat):
        vmv.v.i v22, -1
        flat1 v22, v5, v2, v18, v19 /* |p2-p0| <= F */
        flat1 v22, v1, v2, v18, v19 /* |p1-p0| <= F */
        flat1 v22, v4, v3, v18, v19 /* |q1-q0| <= F */
        flat1 v22, v6, v2, v18, v19 /* |q2-q0| <= F */
        flat1 v22, v7, v2, v18, v19 /* |p3-p0| <= F */
        flat1 v22, v8, v2, v18, v19 /* |q3-q0| <= F */
        /* flat8out = |p6..p4 - p0| <= F && |q4..q6 - q0| <= F */
        vmv.v.i v23, -1
        flat1 v23, v11, v2, v18, v19 /* |p6-p0| <= F */
        flat1 v23, v10, v2, v18, v19 /* |p5-p0| <= F */
        flat1 v23, v9, v2, v18, v19  /* |p4-p0| <= F */
        flat1 v23, v12, v3, v18, v19 /* |q4-q0| <= F */
        flat1 v23, v13, v3, v18, v19 /* |q5-q0| <= F */
        flat1 v23, v14, v3, v18, v19 /* |q6-q0| <= F */
        vmand.mm v23, v23, v22
        vmand.mm v23, v23, v20    /* sel16 = fm & flat8in & flat8out */
        /* sel8' = fm & flat8in & !sel16 (disjoint lanes, order is free) */
        vmnand.mm v19, v23, v23
        vmand.mm v19, v19, v22
        vmand.mm v19, v19, v20
        vfirst.m t0, v23
        bltz t0, 92f             /* no flat16 lanes; flat8 sub-path may run */
        /* dst[-6] = (7*p6 + 2*p5 + 2*p4 + p3 + p2 + p1 + p0 + q0 + 8) >> 4 */
        vmv.v.i v15, 8
        li t6, 7
        vmacc.vx v15, t6, v11
        li t6, 2
        vmacc.vx v15, t6, v10
        li t6, 2
        vmacc.vx v15, t6, v9
        li t6, 1
        vmacc.vx v15, t6, v7
        li t6, 1
        vmacc.vx v15, t6, v5
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        vsrl.vi v15, v15, 4
        vmmv.m v0, v23
        store_lane v15, -6
        /* dst[-5] = (5*p6 + 2*p5 + 2*p4 + 2*p3 + p2 + p1 + p0 + q0 + q1) >> 4 */
        vmv.v.i v15, 8
        li t6, 5
        vmacc.vx v15, t6, v11
        li t6, 2
        vmacc.vx v15, t6, v10
        li t6, 2
        vmacc.vx v15, t6, v9
        li t6, 2
        vmacc.vx v15, t6, v7
        li t6, 1
        vmacc.vx v15, t6, v5
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        vsrl.vi v15, v15, 4
        vmmv.m v0, v23
        store_lane v15, -5
        /* dst[-4] = (4*p6 + p5 + 2*p4 + 2*p3 + 2*p2 + p1 + p0 + q0 + q1
         *            + q2 + 8) >> 4 */
        vmv.v.i v15, 8
        li t6, 4
        vmacc.vx v15, t6, v11
        li t6, 1
        vmacc.vx v15, t6, v10
        li t6, 2
        vmacc.vx v15, t6, v9
        li t6, 2
        vmacc.vx v15, t6, v7
        li t6, 2
        vmacc.vx v15, t6, v5
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        li t6, 1
        vmacc.vx v15, t6, v6
        vsrl.vi v15, v15, 4
        vmmv.m v0, v23
        store_lane v15, -4
        /* dst[-3] = (3*p6 + p5 + p4 + 2*p3 + 2*p2 + 2*p1 + p0 + q0 + q1
         *            + q2 + q3 + 8) >> 4 */
        vmv.v.i v15, 8
        li t6, 3
        vmacc.vx v15, t6, v11
        li t6, 1
        vmacc.vx v15, t6, v10
        li t6, 1
        vmacc.vx v15, t6, v9
        li t6, 2
        vmacc.vx v15, t6, v7
        li t6, 2
        vmacc.vx v15, t6, v5
        li t6, 2
        vmacc.vx v15, t6, v1
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        li t6, 1
        vmacc.vx v15, t6, v6
        li t6, 1
        vmacc.vx v15, t6, v8
        vsrl.vi v15, v15, 4
        vmmv.m v0, v23
        store_lane v15, -3
        /* dst[-2] = (2*p6 + p5 + p4 + p3 + 2*p2 + 2*p1 + 2*p0 + q0 + q1
         *            + q2 + q3 + q4 + 8) >> 4 */
        vmv.v.i v15, 8
        li t6, 2
        vmacc.vx v15, t6, v11
        li t6, 1
        vmacc.vx v15, t6, v10
        li t6, 1
        vmacc.vx v15, t6, v9
        li t6, 1
        vmacc.vx v15, t6, v7
        li t6, 2
        vmacc.vx v15, t6, v5
        li t6, 2
        vmacc.vx v15, t6, v1
        li t6, 2
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        li t6, 1
        vmacc.vx v15, t6, v6
        li t6, 1
        vmacc.vx v15, t6, v8
        li t6, 1
        vmacc.vx v15, t6, v12
        vsrl.vi v15, v15, 4
        vmmv.m v0, v23
        store_lane v15, -2
        /* dst[-1] = (p6 + p5 + p4 + p3 + p2 + 2*p1 + 2*p0 + 2*q0 + q1 + q2
         *            + q3 + q4 + q5 + 8) >> 4 */
        vmv.v.i v15, 8
        li t6, 1
        vmacc.vx v15, t6, v11
        li t6, 1
        vmacc.vx v15, t6, v10
        li t6, 1
        vmacc.vx v15, t6, v9
        li t6, 1
        vmacc.vx v15, t6, v7
        li t6, 1
        vmacc.vx v15, t6, v5
        li t6, 2
        vmacc.vx v15, t6, v1
        li t6, 2
        vmacc.vx v15, t6, v2
        li t6, 2
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        li t6, 1
        vmacc.vx v15, t6, v6
        li t6, 1
        vmacc.vx v15, t6, v8
        li t6, 1
        vmacc.vx v15, t6, v12
        li t6, 1
        vmacc.vx v15, t6, v13
        vsrl.vi v15, v15, 4
        vmmv.m v0, v23
        store_lane v15, -1
        /* dst[+0] = (p5 + p4 + p3 + p2 + p1 + 2*p0 + 2*q0 + 2*q1 + q2 + q3
         *            + q4 + q5 + q6 + 8) >> 4 */
        vmv.v.i v15, 8
        li t6, 1
        vmacc.vx v15, t6, v10
        li t6, 1
        vmacc.vx v15, t6, v9
        li t6, 1
        vmacc.vx v15, t6, v7
        li t6, 1
        vmacc.vx v15, t6, v5
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 2
        vmacc.vx v15, t6, v2
        li t6, 2
        vmacc.vx v15, t6, v3
        li t6, 2
        vmacc.vx v15, t6, v4
        li t6, 1
        vmacc.vx v15, t6, v6
        li t6, 1
        vmacc.vx v15, t6, v8
        li t6, 1
        vmacc.vx v15, t6, v12
        li t6, 1
        vmacc.vx v15, t6, v13
        li t6, 1
        vmacc.vx v15, t6, v14
        vsrl.vi v15, v15, 4
        vmmv.m v0, v23
        store_lane v15, 0
        /* dst[+1] = (p4 + p3 + p2 + p1 + p0 + 2*q0 + 2*q1 + 2*q2 + q3 + q4
         *            + q5 + 2*q6 + 8) >> 4 */
        vmv.v.i v15, 8
        li t6, 1
        vmacc.vx v15, t6, v9
        li t6, 1
        vmacc.vx v15, t6, v7
        li t6, 1
        vmacc.vx v15, t6, v5
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 2
        vmacc.vx v15, t6, v3
        li t6, 2
        vmacc.vx v15, t6, v4
        li t6, 2
        vmacc.vx v15, t6, v6
        li t6, 1
        vmacc.vx v15, t6, v8
        li t6, 1
        vmacc.vx v15, t6, v12
        li t6, 1
        vmacc.vx v15, t6, v13
        li t6, 2
        vmacc.vx v15, t6, v14
        vsrl.vi v15, v15, 4
        vmmv.m v0, v23
        store_lane v15, 1
        /* dst[+2] = (p3 + p2 + p1 + p0 + q0 + 2*q1 + 2*q2 + 2*q3 + q4 + q5
         *            + 3*q6 + 8) >> 4 */
        vmv.v.i v15, 8
        li t6, 1
        vmacc.vx v15, t6, v7
        li t6, 1
        vmacc.vx v15, t6, v5
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        li t6, 2
        vmacc.vx v15, t6, v4
        li t6, 2
        vmacc.vx v15, t6, v6
        li t6, 2
        vmacc.vx v15, t6, v8
        li t6, 1
        vmacc.vx v15, t6, v12
        li t6, 1
        vmacc.vx v15, t6, v13
        li t6, 3
        vmacc.vx v15, t6, v14
        vsrl.vi v15, v15, 4
        vmmv.m v0, v23
        store_lane v15, 2
        /* dst[+3] = (p2 + p1 + p0 + q0 + q1 + 2*q2 + 2*q3 + 2*q4 + q5
         *            + 4*q6 + 8) >> 4 */
        vmv.v.i v15, 8
        li t6, 1
        vmacc.vx v15, t6, v5
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        li t6, 2
        vmacc.vx v15, t6, v6
        li t6, 2
        vmacc.vx v15, t6, v8
        li t6, 2
        vmacc.vx v15, t6, v12
        li t6, 1
        vmacc.vx v15, t6, v13
        li t6, 4
        vmacc.vx v15, t6, v14
        vsrl.vi v15, v15, 4
        vmmv.m v0, v23
        store_lane v15, 3
        /* dst[+4] = (p1 + p0 + q0 + q1 + q2 + 2*q3 + 2*q4 + 2*q5 + 5*q6
         *            + 8) >> 4 */
        vmv.v.i v15, 8
        li t6, 1
        vmacc.vx v15, t6, v1
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        li t6, 1
        vmacc.vx v15, t6, v6
        li t6, 2
        vmacc.vx v15, t6, v8
        li t6, 2
        vmacc.vx v15, t6, v12
        li t6, 2
        vmacc.vx v15, t6, v13
        li t6, 5
        vmacc.vx v15, t6, v14
        vsrl.vi v15, v15, 4
        vmmv.m v0, v23
        store_lane v15, 4
        /* dst[+5] = (p0 + q0 + q1 + q2 + q3 + 2*q4 + 2*q5 + 7*q6 + 8) >> 4 */
        vmv.v.i v15, 8
        li t6, 1
        vmacc.vx v15, t6, v2
        li t6, 1
        vmacc.vx v15, t6, v3
        li t6, 1
        vmacc.vx v15, t6, v4
        li t6, 1
        vmacc.vx v15, t6, v6
        li t6, 1
        vmacc.vx v15, t6, v8
        li t6, 2
        vmacc.vx v15, t6, v12
        li t6, 2
        vmacc.vx v15, t6, v13
        li t6, 7
        vmacc.vx v15, t6, v14
        vsrl.vi v15, v15, 4
        vmmv.m v0, v23
        store_lane v15, 5
92:
        /* wd16 else-if chain: lanes with flat8in but not flat16 run flat8 */
        vfirst.m t0, v19
        bltz t0, L(core_done)
        flat8_body v19

L(core_done):
        ret
endfunc

/* ------------------------------------------------------------------------
 * Superblock wrappers: scalar vmask iteration + per-bit level lookups,
 * then lpf_core_rvv for the 4 lines of the bit.
 *
 * Loop state lives in callee-saved registers (the core is free to clobber
 * all temporaries): s0 stride, s1 b4_stride, s2 lut, s3 l (across the call),
 * s4 vm, s5 vmask[1] word, s6 vmask[2] word, s7 y.
 * ------------------------------------------------------------------------ */

.macro lpf_wrapper_body dir, chroma
        addi sp, sp, -80
        sd ra, 0(sp)
        sd s0, 8(sp)
        sd s1, 16(sp)
        sd s2, 24(sp)
        sd s3, 32(sp)
        sd s4, 40(sp)
        sd s5, 48(sp)
        sd s6, 56(sp)
        sd s7, 64(sp)
        mv s0, a1                /* stride */
        mv s1, a4                /* b4_stride */
        mv s2, a5                /* lut */
        mv s3, a3                /* l */
        lw s4, 0(a2)             /* vm */
        lw s5, 4(a2)
.if !\chroma
        lw s6, 8(a2)
        or s4, s4, s5
        or s4, s4, s6
.else
        or s4, s4, s5
.endif
        li s7, 1                 /* y */
        /* DEBUG markers */
1:
        addi t4, s7, -1
        not t4, t4
        and t4, t4, s4
        beqz t4, 9f              /* no more bits */
        and t4, s4, s7
        beqz t4, 8f              /* bit not set */
        lbu t4, 0(s3)            /* L = l[0][0] ? l[0][0] : l[-1][0] (h) */
        bnez t4, 2f
.if \dir
        slli t5, s1, 2
        sub t5, s3, t5
        lbu t4, 0(t5)            /* l[-b4_stride][0] */
.else
        lbu t4, -4(s3)           /* l[-1][0] */
.endif
2:
        beqz t4, 8f              /* L == 0: skip */
        add t5, s2, t4
        lbu a1, 0(t5)            /* E = lut->e[L] */
        addi t5, t5, 64
        lbu a2, 0(t5)            /* I = lut->i[L] */
        srli t4, t4, 4           /* H = L >> 4 (passed to the core below) */
.if \chroma
        /* wd = 4 + 2*idx, idx = !!(vmask[1] & y) */
        and t5, s5, s7
        beqz t5, 3f
        li a6, 6
        j 4f
3:
        li a6, 4
4:
.else
        /* wd = 4 << idx, idx = (vmask[2] & y) ? 2 : !!(vmask[1] & y) */
        and t5, s6, s7
        bnez t5, 5f              /* idx == 2: wd 16 */
        and t5, s5, s7
        beqz t5, 6f              /* idx == 0: wd 4 */
        li a6, 8
        j 7f
5:
        li a6, 16
        j 7f
6:
        li a6, 4
7:
.endif
.if \dir
        li a4, 1                 /* stridea = 1 */
        mv a5, s0                /* strideb = stride */
.else
        mv a4, s0                /* stridea = stride */
        li a5, 1                 /* strideb = 1 */
.endif
        mv a3, t4                /* H */
        jal lpf_core_rvv
8:
        slli t5, s7, 1
        mv s7, t5                /* y <<= 1 */
.if \dir
        addi a0, a0, 4           /* dst += 4 */
        addi s3, s3, 4           /* l++ */
.else
        slli t5, s0, 2
        add a0, a0, t5           /* dst += 4 * stride */
        slli t5, s1, 2
        add s3, s3, t5           /* l += 4 * b4_stride */
.endif
        j 1b
9:
        ld ra, 0(sp)
        ld s0, 8(sp)
        ld s1, 16(sp)
        ld s2, 24(sp)
        ld s3, 32(sp)
        ld s4, 40(sp)
        ld s5, 48(sp)
        ld s6, 56(sp)
        ld s7, 64(sp)
        addi sp, sp, 80
        ret
.endm

function lpf_h_sb_y_8bpc_rvv, export=1, ext="v"
        lpf_wrapper_body 0, 0
endfunc

function lpf_v_sb_y_8bpc_rvv, export=1, ext="v"
        lpf_wrapper_body 1, 0
endfunc

function lpf_h_sb_uv_8bpc_rvv, export=1, ext="v"
        lpf_wrapper_body 0, 1
endfunc

function lpf_v_sb_uv_8bpc_rvv, export=1, ext="v"
        lpf_wrapper_body 1, 1
endfunc
