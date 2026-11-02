/* Infrastructure fixture for tests (not a campaign seed): a deliberately simple
 * RVV splat_mv to exercise the dav1d adapter and the formal check. */
#include "src/riscv/asm.S"

function splat_mv_rvv, export=1, ext=v
    li          t0, 12
    mul         a2, a2, t0          // bx4 * 12 bytes
    slli        t1, a3, 1
    add         t1, t1, a3          // 3 * bw4 words per row
    vsetivli    zero, 3, e32, m1, ta, ma
    vle32.v     v16, (a1)
1:
    ld          t2, 0(a0)
    addi        a0, a0, 8
    add         t2, t2, a2
    mv          t3, t1
    li          t4, 0
2:
    vsetvli     t5, t3, e32, m4, ta, ma
    vid.v       v4
    vadd.vx     v4, v4, t4
    li          t6, 3
    vremu.vx    v4, v4, t6
    vrgather.vv v8, v16, v4
    vse32.v     v8, (t2)
    sub         t3, t3, t5
    add         t4, t4, t5
    slli        t5, t5, 2
    add         t2, t2, t5
    bnez        t3, 2b
    addi        a4, a4, -1
    bnez        a4, 1b
    ret
endfunc
