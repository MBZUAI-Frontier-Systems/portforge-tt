/* C-callable wrappers for the wide 16x16 pipeline helpers (campaign 4). */

    .text

/* wrap_horz(a0=coefs, a1=stack, a2=eob, a3=threshold, a4=helper)
 * helper contract: t4=coefs, t5=stack, t6=32, t3=threshold, a3=eob,
 * a4 unused here (identity variant), a7 = return */
    .globl wrap_horz
wrap_horz:
    vsetivli zero, 16, e16, m1, ta, ma
    csrw vxrm, zero
    mv t4, a0
    mv t5, a1
    li t6, 32
    mv t3, a3
    mv a3, a2
    jalr a7, a4
    ret

/* wrap_vert(a0=stack, a1=dst, a2=stride, a3=kern, a4=helper)
 * helper contract: t4=stack, t5=dst, t6=32, a5=kern, a1=stride, a7=ret */
    .globl wrap_vert
wrap_vert:
    vsetivli zero, 16, e16, m1, ta, ma
    csrw vxrm, zero
    mv t4, a0
    mv t5, a1
    li t6, 32
    mv a1, a2
    mv a5, a3
    jalr a7, a4
    ret
