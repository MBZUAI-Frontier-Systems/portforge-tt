/* RVV 1.0 + RV64 instruction model for lifted machine code (portforge_tt/fv).
 *
 * The lifter (lift.py) turns a function's disassembly into C that calls the
 * helpers below; the same C runs natively (co-simulation against QEMU and the
 * X280) and under CBMC (bounded equivalence against the project's C reference).
 * Trust comes from validation, not from provenance: validate.py runs every
 * opcode/SEW/LMUL/policy combination the census needs through QEMU, the board
 * and this model and requires identical results.
 *
 * Choices that matter for soundness:
 *   - VLEN is a compile-time constant (FV_VLEN), one query per legal VLEN.
 *   - tail- and mask-agnostic elements are nondeterministic per instruction
 *     (undisturbed or all ones), so a kernel that relies on either fails.
 *   - vill is set at entry and vxrm is nondeterministic: code must set both.
 *   - memory goes through fv_load/fv_store, defined by the harness, which
 *     asserts every access lies in a buffer the contract lets the kernel touch.
 */
#ifndef PORTFORGE_FV_RVV_MODEL_H
#define PORTFORGE_FV_RVV_MODEL_H

#include <stdint.h>

#ifndef FV_VLEN
#define FV_VLEN 512
#endif
#define VLENB (FV_VLEN / 8)

#ifdef FV_CBMC
_Bool nondet_bool(void);
uint64_t nondet_u64(void);
uint8_t nondet_u8(void);
#define FV_ASSERT(c, msg) __CPROVER_assert((c), msg)
#define FV_ASSUME(c) __CPROVER_assume(c)
#define FV_AGNOSTIC() nondet_bool()
#else
#include <stdio.h>
#include <stdlib.h>
extern int fv_agnostic_ones;  /* native runs: 0 = undisturbed, 1 = all ones */
extern void fv_fail(const char *msg);
#define FV_ASSERT(c, msg) do { if (!(c)) fv_fail(msg); } while (0)
#define FV_ASSUME(c) ((void)0)
#define FV_AGNOSTIC() fv_agnostic_ones
#endif

typedef struct {
    uint64_t x[32];
    uint64_t f[32];                 /* raw bits; only moved, never computed on */
    uint8_t v[32 * VLENB];
    uint64_t vl;
    unsigned sew, lmul8;            /* lmul8 = LMUL * 8: 1 (mf8) .. 64 (m8) */
    unsigned vta, vma, vill;
    unsigned vxrm, vxsat;
} fv_state;

extern fv_state S;

/* harness-provided memory: little-endian, n in {1, 2, 4, 8} */
uint64_t fv_load(uint64_t addr, unsigned n);
void fv_store(uint64_t addr, unsigned n, uint64_t value);

static inline uint64_t fv_mask(unsigned bits) { return bits >= 64 ? ~0ULL : ((1ULL << bits) - 1); }
static inline uint64_t fv_sx(uint64_t v, unsigned bits)
{
    if (bits >= 64) return v;
    uint64_t m = 1ULL << (bits - 1);
    v &= fv_mask(bits);
    return (v ^ m) - m;
}
static inline int64_t fv_s(uint64_t v, unsigned bits) { return (int64_t)fv_sx(v, bits); }

/* ------------------------------------------------------------------ vtype */

static inline uint64_t fv_vlmax_of(unsigned sew, unsigned lmul8) { return (uint64_t)FV_VLEN * lmul8 / 8 / sew; }

static inline uint64_t fv_set_vtype(uint64_t vtype, uint64_t avl, int keep_vl, int vlmax_avl)
{
    unsigned vsew = (vtype >> 3) & 7, vlmul = vtype & 7;
    static const unsigned lmul8_of[8] = {8, 16, 32, 64, 0, 1, 2, 4};
    unsigned sew = 8u << vsew, lmul8 = lmul8_of[vlmul];
    /* reserved encodings, and SEW > LMUL * ELEN (e.g. e64,mf2) */
    int ill = vsew > 3 || lmul8 == 0 || (vtype >> 8) != 0 || (uint64_t)sew * 8 > 64ULL * lmul8;
    if (ill) {
        S.vill = 1; S.vl = 0;
        return 0;
    }
    S.vill = 0; S.sew = sew; S.lmul8 = lmul8;
    S.vta = (vtype >> 6) & 1; S.vma = (vtype >> 7) & 1;
    uint64_t vlmax = fv_vlmax_of(sew, lmul8);
    if (keep_vl) {
        /* rd = x0, rs1 = x0: keep vl; only legal when VLMAX is unchanged */
        FV_ASSERT(S.vl <= vlmax, "vsetvl x0,x0 changed VLMAX below vl");
    } else if (vlmax_avl) {
        S.vl = vlmax;
    } else {
        S.vl = avl < vlmax ? avl : vlmax;   /* deterministic choice allowed by the spec */
    }
    return S.vl;
}

static inline uint64_t fv_vtype_imm(unsigned sew, unsigned lmul8, unsigned ta, unsigned ma)
{
    unsigned vsew = sew == 8 ? 0 : sew == 16 ? 1 : sew == 32 ? 2 : 3;
    unsigned vlmul = lmul8 == 8 ? 0 : lmul8 == 16 ? 1 : lmul8 == 32 ? 2 : lmul8 == 64 ? 3 :
                     lmul8 == 1 ? 5 : lmul8 == 2 ? 6 : 7;
    return vlmul | (vsew << 3) | (ta << 6) | (ma << 7);
}

static inline uint64_t fv_vtype_csr(void)
{
    if (S.vill) return 1ULL << 63;
    return fv_vtype_imm(S.sew, S.lmul8, S.vta, S.vma);
}

/* ------------------------------------------------------------------ register file */

/* register group of EEW elements with EMUL = eew / sew * lmul; checks alignment */
static inline unsigned fv_emul8(unsigned eew) { return (unsigned)((uint64_t)S.lmul8 * eew / S.sew); }

static inline void fv_group_check(unsigned reg, unsigned emul8, const char *what)
{
    unsigned regs = emul8 <= 8 ? 1 : emul8 / 8;
    (void)what;  /* CBMC wants literal descriptions */
    FV_ASSERT(emul8 >= 1 && emul8 <= 64, "register group: EMUL outside 1/8..8");
    FV_ASSERT(reg % regs == 0 && reg + regs <= 32, "register group: misaligned or past v31");
}

static inline uint64_t fv_vget(unsigned reg, uint64_t i, unsigned eew)
{
    uint64_t off = (uint64_t)reg * VLENB + i * (eew / 8);
    FV_ASSERT(off + eew / 8 <= 32 * VLENB, "vector element outside the register file");
    uint64_t v = 0;
    for (unsigned b = 0; b < eew / 8; b++) v |= (uint64_t)S.v[off + b] << (8 * b);
    return v;
}

static inline void fv_vset(unsigned reg, uint64_t i, unsigned eew, uint64_t value)
{
    uint64_t off = (uint64_t)reg * VLENB + i * (eew / 8);
    FV_ASSERT(off + eew / 8 <= 32 * VLENB, "vector element outside the register file");
    for (unsigned b = 0; b < eew / 8; b++) S.v[off + b] = (uint8_t)(value >> (8 * b));
}

static inline int fv_mbit(unsigned reg, uint64_t i) { return (S.v[reg * VLENB + i / 8] >> (i % 8)) & 1; }
static inline void fv_mset(unsigned reg, uint64_t i, int bit)
{
    uint8_t *p = &S.v[reg * VLENB + i / 8];
    *p = (uint8_t)((*p & ~(1u << (i % 8))) | ((bit & 1u) << (i % 8)));
}

/* elements of a group of EMUL registers (tail runs to the end of the group, or of
 * the one register for fractional EMUL) */
static inline uint64_t fv_tail_end(unsigned eew, unsigned emul8)
{
    unsigned regs = emul8 <= 8 ? 1 : emul8 / 8;
    return (uint64_t)regs * VLENB / (eew / 8);
}

static inline void fv_vcheck(void) { FV_ASSERT(!S.vill, "vector instruction with vill set (no vsetvl before use)"); }

/* Data-dependent choices are made with value selects, never branches: under CBMC a
 * branch on a symbolic value splits the path and every join merges the whole machine
 * state (wiener W=2: 176 s of symbolic execution for 176k steps). */

/* tail elements [from, end) of vd: undisturbed or all ones when vta */
static inline void fv_tail(unsigned vd, uint64_t from, unsigned eew, unsigned emul8, unsigned agnostic)
{
    if (!agnostic) return;
    _Bool ones = FV_AGNOSTIC();
    uint64_t end = fv_tail_end(eew, emul8);
    for (uint64_t i = from; i < end; i++) fv_vset(vd, i, eew, ones ? fv_mask(eew) : fv_vget(vd, i, eew));
}

/* the value of a masked-off element: undisturbed, or all ones when vma */
static inline uint64_t fv_off(unsigned vd, uint64_t i, unsigned eew, int agnostic_choice)
{
    return (S.vma && agnostic_choice) ? fv_mask(eew) : fv_vget(vd, i, eew);
}

static inline void fv_inactive(unsigned vd, uint64_t i, unsigned eew, int agnostic_choice)
{
    if (S.vma) fv_vset(vd, i, eew, fv_off(vd, i, eew, agnostic_choice));
}

static inline int fv_active(int masked, uint64_t i) { return !masked || fv_mbit(0, i); }

/* ------------------------------------------------------------------ fixed-point helpers */

static inline uint64_t fv_rnd(uint64_t v, unsigned d, unsigned xrm)
{
    if (d == 0) return 0;
    uint64_t d1 = (v >> (d - 1)) & 1;
    uint64_t low = d >= 2 ? (v & fv_mask(d - 1)) : 0;
    uint64_t bd = d < 64 ? (v >> d) & 1 : 0;
    switch (xrm & 3) {
    case 0: return d1;                                   /* rnu */
    case 1: return d1 & ((low != 0) | bd);               /* rne */
    case 2: return 0;                                    /* rdn */
    default: return (!bd) & ((d1 | (low != 0)) != 0);    /* rod */
    }
}

static inline int64_t fv_sat_s(int64_t v, unsigned bits)
{
    int64_t hi = (int64_t)(fv_mask(bits - 1)), lo = -hi - 1;
    S.vxsat |= (v > hi) | (v < lo);
    return v > hi ? hi : v < lo ? lo : v;
}

static inline uint64_t fv_sat_u(uint64_t v, unsigned bits)
{
    if (bits >= 64) return v;
    S.vxsat |= v > fv_mask(bits);
    return v > fv_mask(bits) ? fv_mask(bits) : v;
}

/* ------------------------------------------------------------------ single-width integer ops */

enum {
    OP_ADD, OP_SUB, OP_RSUB, OP_AND, OP_OR, OP_XOR, OP_SLL, OP_SRL, OP_SRA,
    OP_MIN, OP_MINU, OP_MAX, OP_MAXU, OP_MUL, OP_MULH, OP_MULHU, OP_MULHSU,
    OP_DIV, OP_DIVU, OP_REM, OP_REMU, OP_SADD, OP_SADDU, OP_SSUB, OP_SSUBU,
    OP_AADD, OP_AADDU, OP_ASUB, OP_ASUBU, OP_SMUL, OP_SSRL, OP_SSRA, OP_MV,
    /* compares (mask results) */
    OP_MSEQ, OP_MSNE, OP_MSLT, OP_MSLTU, OP_MSLE, OP_MSLEU, OP_MSGT, OP_MSGTU,
    /* multiply-add (three operands) */
    OP_MACC, OP_NMSAC, OP_MADD, OP_NMSUB,
};

static inline uint64_t fv_alu(int op, uint64_t a, uint64_t b, unsigned sew)
{
    uint64_t m = fv_mask(sew);
    int64_t sa = fv_s(a, sew), sb = fv_s(b, sew);
    uint64_t ua = a & m, ub = b & m;
    unsigned sh = (unsigned)(b & (sew - 1));
    switch (op) {
    case OP_ADD: return (ua + ub) & m;
    case OP_SUB: return (ua - ub) & m;
    case OP_RSUB: return (ub - ua) & m;
    case OP_AND: return ua & ub;
    case OP_OR: return ua | ub;
    case OP_XOR: return ua ^ ub;
    case OP_SLL: return (ua << sh) & m;
    case OP_SRL: return ua >> sh;
    case OP_SRA: return (uint64_t)(sa >> sh) & m;
    case OP_MIN: return sa < sb ? ua : ub;
    case OP_MINU: return ua < ub ? ua : ub;
    case OP_MAX: return sa > sb ? ua : ub;
    case OP_MAXU: return ua > ub ? ua : ub;
    case OP_MUL: return (ua * ub) & m;
    case OP_MULH:
        if (sew == 64) return (uint64_t)(((__int128)sa * (__int128)sb) >> 64);
        return (uint64_t)((sa * sb) >> sew) & m;
    case OP_MULHU:
        if (sew == 64) return (uint64_t)(((unsigned __int128)ua * ub) >> 64);
        return ((ua * ub) >> sew) & m;
    case OP_MULHSU:
        if (sew == 64) return (uint64_t)(((__int128)sa * (__int128)(unsigned __int128)ub) >> 64);
        return (uint64_t)((sa * (int64_t)ub) >> sew) & m;
    case OP_DIV:
        if (sb == 0) return m;
        if (sew < 64 ? 0 : (sa == INT64_MIN && sb == -1)) return ua;
        if (sa == -(int64_t)(1ULL << (sew - 1)) && sb == -1 && sew < 64) return ua;
        return (uint64_t)(sa / sb) & m;
    case OP_DIVU: return ub == 0 ? m : ua / ub;
    case OP_REM:
        if (sb == 0) return ua;
        if (sb == -1) return 0;
        return (uint64_t)(sa % sb) & m;
    case OP_REMU: return ub == 0 ? ua : ua % ub;
    case OP_SADD: return (uint64_t)fv_sat_s(sa + sb, sew) & m;   /* sew <= 32 asserted by caller */
    case OP_SADDU:
        if (sew == 64) {   /* the 64-bit sum itself can wrap: saturate on carry */
            uint64_t s = ua + ub;
            S.vxsat |= s < ua;
            return s < ua ? ~0ULL : s;
        }
        return fv_sat_u(ua + ub, sew);
    case OP_SSUB: return (uint64_t)fv_sat_s(sa - sb, sew) & m;
    case OP_SSUBU: S.vxsat |= ua < ub; return ua < ub ? 0 : ua - ub;
    case OP_AADD: { int64_t s = sa + sb; return (uint64_t)((s >> 1) + (int64_t)fv_rnd((uint64_t)s, 1, S.vxrm)) & m; }
    case OP_AADDU: { uint64_t s = ua + ub; return ((s >> 1) + fv_rnd(s, 1, S.vxrm)) & m; }
    case OP_ASUB: { int64_t s = sa - sb; return (uint64_t)((s >> 1) + (int64_t)fv_rnd((uint64_t)s, 1, S.vxrm)) & m; }
    case OP_ASUBU: { uint64_t s = ua - ub; /* sew+1-bit difference */
        int64_t d = (int64_t)ua - (int64_t)ub;
        (void)s; return (uint64_t)((d >> 1) + (int64_t)fv_rnd((uint64_t)d, 1, S.vxrm)) & m; }
    case OP_SMUL: {
        int64_t p = sa * sb;
        int64_t r = (p >> (sew - 1)) + (int64_t)fv_rnd((uint64_t)p, sew - 1, S.vxrm);
        return (uint64_t)fv_sat_s(r, sew) & m; }
    case OP_SSRL: return (ua >> sh) + fv_rnd(ua, sh, S.vxrm);
    case OP_SSRA: return (uint64_t)((sa >> sh) + (int64_t)fv_rnd((uint64_t)sa, sh, S.vxrm)) & m;
    case OP_MV: return ub;
    case OP_MSEQ: return ua == ub;
    case OP_MSNE: return ua != ub;
    case OP_MSLT: return sa < sb;
    case OP_MSLTU: return ua < ub;
    case OP_MSLE: return sa <= sb;
    case OP_MSLEU: return ua <= ub;
    case OP_MSGT: return sa > sb;
    case OP_MSGTU: return ua > ub;
    }
    FV_ASSERT(0, "unknown vector alu op");
    return 0;
}

#define FV_SRC_V 0   /* operand b is vs1 */
#define FV_SRC_X 1   /* operand b is a scalar (x register or immediate) */

static inline int fv_ops_need_narrow_sew(int op)
{
    return op == OP_SADD || op == OP_SSUB || op == OP_AADD || op == OP_ASUB || op == OP_AADDU ||
           op == OP_ASUBU || op == OP_SMUL;
}

/* vd = op(vs2, vs1 | scalar), masked by v0 when masked */
static inline void fv_vop(int op, unsigned vd, unsigned vs2, int src, unsigned vs1, uint64_t scalar, int masked)
{
    fv_vcheck();
    unsigned sew = S.sew;
    if (fv_ops_need_narrow_sew(op)) FV_ASSERT(sew <= 32, "model: saturating/averaging op at SEW 64");
    fv_group_check(vd, S.lmul8, "vd group");
    fv_group_check(vs2, S.lmul8, "vs2 group");
    if (src == FV_SRC_V) fv_group_check(vs1, S.lmul8, "vs1 group");
    if (masked) FV_ASSERT(vd != 0, "masked op writes v0");
    int agn = FV_AGNOSTIC();
    for (uint64_t i = 0; i < S.vl; i++) {
        uint64_t a = fv_vget(vs2, i, sew);
        uint64_t b = src == FV_SRC_V ? fv_vget(vs1, i, sew) : scalar;
        unsigned sat = S.vxsat;
        uint64_t r = fv_alu(op, a, b, sew);
        int on = fv_active(masked, i);
        S.vxsat = on ? S.vxsat : sat;   /* a masked-off element never saturates */
        fv_vset(vd, i, sew, on ? r : fv_off(vd, i, sew, agn));
    }
    fv_tail(vd, S.vl, sew, S.lmul8, S.vta);
}

/* vmv.v.v / vmv.v.x / vmv.v.i and vmerge (masked select) */
static inline void fv_vmerge(unsigned vd, unsigned vs2, int src, unsigned vs1, uint64_t scalar, int merge)
{
    fv_vcheck();
    unsigned sew = S.sew;
    fv_group_check(vd, S.lmul8, "vd group");
    for (uint64_t i = 0; i < S.vl; i++) {
        int take = !merge || fv_mbit(0, i);
        uint64_t b = src == FV_SRC_V ? fv_vget(vs1, i, sew) : scalar & fv_mask(sew);
        fv_vset(vd, i, sew, take ? b : fv_vget(vs2, i, sew));
    }
    fv_tail(vd, S.vl, sew, S.lmul8, S.vta);
}

/* integer compares: mask register vd (tail always agnostic) */
static inline void fv_vcmp(int op, unsigned vd, unsigned vs2, int src, unsigned vs1, uint64_t scalar, int masked)
{
    fv_vcheck();
    unsigned sew = S.sew;
    fv_group_check(vs2, S.lmul8, "vs2 group");
    int agn = FV_AGNOSTIC();
    uint8_t res[VLENB];
    for (unsigned b = 0; b < VLENB; b++) res[b] = S.v[vd * VLENB + b];
    for (uint64_t i = 0; i < S.vl; i++) {
        uint64_t a = fv_vget(vs2, i, sew);
        uint64_t b = src == FV_SRC_V ? fv_vget(vs1, i, sew) : scalar;
        int computed = (int)fv_alu(op, a, b, sew);
        int off = (S.vma && agn) ? 1 : ((res[i / 8] >> (i % 8)) & 1);
        int bit = fv_active(masked, i) ? computed : off;
        res[i / 8] = (uint8_t)((res[i / 8] & ~(1u << (i % 8))) | ((unsigned)bit << (i % 8)));
    }
    _Bool tail_ones = FV_AGNOSTIC();
    for (uint64_t i = S.vl; i < (uint64_t)FV_VLEN; i++)
        res[i / 8] = (uint8_t)(tail_ones ? (res[i / 8] | (1u << (i % 8))) : res[i / 8]);
    for (unsigned b = 0; b < VLENB; b++) S.v[vd * VLENB + b] = res[b];
}

/* vmacc/vnmsac/vmadd/vnmsub: vd, vs1|rs1, vs2 */
static inline void fv_vmuladd(int op, unsigned vd, int src, unsigned vs1, uint64_t scalar, unsigned vs2, int masked)
{
    fv_vcheck();
    unsigned sew = S.sew;
    uint64_t m = fv_mask(sew);
    fv_group_check(vd, S.lmul8, "vd group");
    int agn = FV_AGNOSTIC();
    for (uint64_t i = 0; i < S.vl; i++) {
        if (masked && !fv_mbit(0, i)) { fv_inactive(vd, i, sew, agn); continue; }   /* rare: masked multiply-add */
        uint64_t a = src == FV_SRC_V ? fv_vget(vs1, i, sew) : scalar;
        uint64_t b = fv_vget(vs2, i, sew), d = fv_vget(vd, i, sew), r;
        switch (op) {
        case OP_MACC: r = a * b + d; break;
        case OP_NMSAC: r = d - a * b; break;
        case OP_MADD: r = a * d + b; break;
        default: r = b - a * d; break;   /* OP_NMSUB */
        }
        fv_vset(vd, i, sew, r & m);
    }
    fv_tail(vd, S.vl, sew, S.lmul8, S.vta);
}

/* ------------------------------------------------------------------ widening / narrowing */

enum {
    W_ADD, W_ADDU, W_SUB, W_SUBU, W_ADD_W, W_ADDU_W, W_SUB_W, W_SUBU_W,
    W_MUL, W_MULU, W_MULSU, W_MACC, W_MACCU, W_MACCSU, W_MACCUS,
};

/* vd (2*SEW) = op(vs2, vs1 | scalar); for the W_*_W forms vs2 is already 2*SEW;
 * for the W_MACC* forms the operands are (vs1|rs1, vs2) and vd accumulates */
static inline void fv_vwop(int op, unsigned vd, unsigned vs2, int src, unsigned vs1, uint64_t scalar, int masked)
{
    fv_vcheck();
    unsigned sew = S.sew, wsew = 2 * sew;
    FV_ASSERT(sew <= 32, "widening op at SEW 64");
    unsigned wl8 = 2 * S.lmul8;
    fv_group_check(vd, wl8, "widening vd group");
    int wide2 = op == W_ADD_W || op == W_ADDU_W || op == W_SUB_W || op == W_SUBU_W;
    fv_group_check(vs2, wide2 ? wl8 : S.lmul8, "widening vs2 group");
    uint64_t wm = fv_mask(wsew);
    int agn = FV_AGNOSTIC();
    for (uint64_t i = 0; i < S.vl; i++) {
        if (masked && !fv_mbit(0, i)) { fv_inactive(vd, i, wsew, agn); continue; }
        uint64_t b = src == FV_SRC_V ? fv_vget(vs1, i, sew) : scalar & fv_mask(sew);
        uint64_t a = fv_vget(vs2, i, wide2 ? wsew : sew);
        int64_t sa = fv_s(a, wide2 ? wsew : sew), sb = fv_s(b, sew);
        uint64_t ua = a, ub = b, r;
        switch (op) {
        case W_ADD: case W_ADD_W: r = (uint64_t)(sa + sb); break;
        case W_ADDU: case W_ADDU_W: r = ua + ub; break;
        case W_SUB: case W_SUB_W: r = (uint64_t)(sa - sb); break;
        case W_SUBU: case W_SUBU_W: r = ua - ub; break;
        case W_MUL: r = (uint64_t)(sa * sb); break;
        case W_MULU: r = ua * ub; break;
        case W_MULSU: r = (uint64_t)(sa * (int64_t)ub); break;   /* vs2 signed, vs1 unsigned */
        default: {
            /* vwmacc*: vd += vs1|rs1 * vs2; b is vs1|rs1, a is vs2 */
            int64_t sb2 = sb;
            uint64_t d = fv_vget(vd, i, wsew);
            uint64_t p;
            if (op == W_MACC) p = (uint64_t)(sb2 * fv_s(a, sew));
            else if (op == W_MACCU) p = ub * (a & fv_mask(sew));
            else if (op == W_MACCSU) p = (uint64_t)(sb2 * (int64_t)(a & fv_mask(sew)));  /* signed rs1/vs1, unsigned vs2 */
            else p = (uint64_t)((int64_t)ub * fv_s(a, sew));                          /* W_MACCUS: unsigned rs1, signed vs2 */
            r = d + p;
        }
        }
        fv_vset(vd, i, wsew, r & wm);
    }   /* (masked widening ops keep the branch: rare in kernels) */
    fv_tail(vd, S.vl, wsew, wl8, S.vta);
}

enum { N_SRL, N_SRA, N_CLIP, N_CLIPU };

/* vd (SEW) = narrow(vs2 (2*SEW), shift = vs1|scalar) */
static inline void fv_vnop(int op, unsigned vd, unsigned vs2, int src, unsigned vs1, uint64_t scalar, int masked)
{
    fv_vcheck();
    unsigned sew = S.sew, wsew = 2 * sew;
    FV_ASSERT(sew <= 32, "narrowing op at SEW 64");
    fv_group_check(vd, S.lmul8, "narrowing vd group");
    fv_group_check(vs2, 2 * S.lmul8, "narrowing vs2 group");
    int agn = FV_AGNOSTIC();
    /* in-order is safe when vd is the low part of vs2: narrow element i overlaps wide
     * element i / 2, which is read in the same or an earlier iteration */
    for (uint64_t i = 0; i < S.vl; i++) {
        if (masked && !fv_mbit(0, i)) { fv_inactive(vd, i, sew, agn); continue; }
        uint64_t a = fv_vget(vs2, i, wsew);
        uint64_t b = src == FV_SRC_V ? fv_vget(vs1, i, sew) : scalar;
        unsigned sh = (unsigned)(b & (wsew - 1));
        uint64_t r;
        if (op == N_SRL) r = a >> sh;
        else if (op == N_SRA) r = (uint64_t)(fv_s(a, wsew) >> sh);
        else if (op == N_CLIPU) r = fv_sat_u((a >> sh) + fv_rnd(a, sh, S.vxrm), sew);
        else r = (uint64_t)fv_sat_s((fv_s(a, wsew) >> sh) + (int64_t)fv_rnd(fv_sx(a, wsew), sh, S.vxrm), sew);
        fv_vset(vd, i, sew, r & fv_mask(sew));
    }
    fv_tail(vd, S.vl, sew, S.lmul8, S.vta);
}

/* vzext/vsext.vf{2,4,8} */
static inline void fv_vext(unsigned vd, unsigned vs2, unsigned frac, int sign, int masked)
{
    fv_vcheck();
    unsigned sew = S.sew, ssew = sew / frac;
    FV_ASSERT(ssew >= 8, "extension source EEW below 8");
    fv_group_check(vd, S.lmul8, "vd group");
    int agn = FV_AGNOSTIC();
    for (uint64_t i = 0; i < S.vl; i++) {
        if (masked && !fv_mbit(0, i)) { fv_inactive(vd, i, sew, agn); continue; }
        uint64_t a = fv_vget(vs2, i, ssew);
        fv_vset(vd, i, sew, (sign ? fv_sx(a, ssew) : a) & fv_mask(sew));
    }
    fv_tail(vd, S.vl, sew, S.lmul8, S.vta);
}

/* ------------------------------------------------------------------ permutes, moves, reductions */

static inline void fv_vmv_x_s(unsigned rd, unsigned vs2)
{
    fv_vcheck();
    if (rd) S.x[rd] = fv_sx(fv_vget(vs2, 0, S.sew), S.sew);
}

static inline void fv_vmv_s_x(unsigned vd, uint64_t value)
{
    fv_vcheck();
    if (S.vl > 0) {
        fv_vset(vd, 0, S.sew, value & fv_mask(S.sew));
        if (S.vta) {
            _Bool ones = FV_AGNOSTIC();
            for (uint64_t i = 1; i < (uint64_t)VLENB * 8 / S.sew; i++)
                fv_vset(vd, i, S.sew, ones ? fv_mask(S.sew) : fv_vget(vd, i, S.sew));
        }
    }
}

static inline void fv_vmvnr(unsigned vd, unsigned vs2, unsigned n)
{
    FV_ASSERT(vd % n == 0 && vs2 % n == 0 && vd + n <= 32 && vs2 + n <= 32, "vmv<n>r register alignment");
    for (unsigned b = 0; b < n * VLENB; b++) S.v[vd * VLENB + b] = S.v[vs2 * VLENB + b];
}

static inline void fv_vid(unsigned vd, int masked)
{
    fv_vcheck();
    int agn = FV_AGNOSTIC();
    for (uint64_t i = 0; i < S.vl; i++) {
        if (masked && !fv_mbit(0, i)) { fv_inactive(vd, i, S.sew, agn); continue; }
        fv_vset(vd, i, S.sew, i & fv_mask(S.sew));
    }
    fv_tail(vd, S.vl, S.sew, S.lmul8, S.vta);
}

/* vrgather.vv (idx from vs1, EEW = SEW), vrgatherei16.vv (EEW 16), vrgather.vx/vi */
static inline void fv_vrgather(unsigned vd, unsigned vs2, int src, unsigned vs1, uint64_t scalar, unsigned idx_eew,
                               int masked)
{
    fv_vcheck();
    unsigned sew = S.sew;
    uint64_t vlmax = fv_vlmax_of(sew, S.lmul8);
    FV_ASSERT(vd != vs2 && (src != FV_SRC_V || vd != vs1), "vrgather destination overlaps a source");
    int agn = FV_AGNOSTIC();
    for (uint64_t i = 0; i < S.vl; i++) {
        if (masked && !fv_mbit(0, i)) { fv_inactive(vd, i, sew, agn); continue; }
        uint64_t idx = src == FV_SRC_V ? fv_vget(vs1, i, idx_eew) : scalar;
        fv_vset(vd, i, sew, idx < vlmax ? fv_vget(vs2, idx, sew) : 0);
    }
    fv_tail(vd, S.vl, sew, S.lmul8, S.vta);
}

static inline void fv_vslideup(unsigned vd, unsigned vs2, uint64_t off, int masked)
{
    fv_vcheck();
    unsigned sew = S.sew;
    FV_ASSERT(vd != vs2, "vslideup destination overlaps the source");
    int agn = FV_AGNOSTIC();
    for (uint64_t i = off; i < S.vl; i++) {
        if (masked && !fv_mbit(0, i)) { fv_inactive(vd, i, sew, agn); continue; }
        fv_vset(vd, i, sew, fv_vget(vs2, i - off, sew));
    }
    fv_tail(vd, S.vl, sew, S.lmul8, S.vta);
}

static inline void fv_vslidedown(unsigned vd, unsigned vs2, uint64_t off, int masked)
{
    fv_vcheck();
    unsigned sew = S.sew;
    uint64_t vlmax = fv_vlmax_of(sew, S.lmul8);
    int agn = FV_AGNOSTIC();
    /* reads ahead of writes: copy first (vd may equal vs2 for slidedown) */
    uint64_t tmp[FV_VLEN];
    for (uint64_t i = 0; i < S.vl; i++)
        tmp[i] = (off < vlmax && i + off < vlmax) ? fv_vget(vs2, i + off, sew) : 0;
    for (uint64_t i = 0; i < S.vl; i++) {
        if (masked && !fv_mbit(0, i)) { fv_inactive(vd, i, sew, agn); continue; }
        fv_vset(vd, i, sew, tmp[i]);
    }
    fv_tail(vd, S.vl, sew, S.lmul8, S.vta);
}

static inline void fv_vslide1up(unsigned vd, unsigned vs2, uint64_t x, int masked)
{
    fv_vcheck();
    unsigned sew = S.sew;
    FV_ASSERT(vd != vs2, "vslide1up destination overlaps the source");
    int agn = FV_AGNOSTIC();
    for (uint64_t i = 0; i < S.vl; i++) {
        if (masked && !fv_mbit(0, i)) { fv_inactive(vd, i, sew, agn); continue; }
        fv_vset(vd, i, sew, i == 0 ? x & fv_mask(sew) : fv_vget(vs2, i - 1, sew));
    }
    fv_tail(vd, S.vl, sew, S.lmul8, S.vta);
}

static inline void fv_vslide1down(unsigned vd, unsigned vs2, uint64_t x, int masked)
{
    fv_vcheck();
    unsigned sew = S.sew;
    int agn = FV_AGNOSTIC();
    for (uint64_t i = 0; i < S.vl; i++) {
        if (masked && !fv_mbit(0, i)) { fv_inactive(vd, i, sew, agn); continue; }
        fv_vset(vd, i, sew, i + 1 == S.vl ? x & fv_mask(sew) : fv_vget(vs2, i + 1, sew));
    }
    fv_tail(vd, S.vl, sew, S.lmul8, S.vta);
}

static inline void fv_vcompress(unsigned vd, unsigned vs2, unsigned vs1)
{
    fv_vcheck();
    unsigned sew = S.sew;
    FV_ASSERT(vd != vs2 && vd != vs1, "vcompress overlap");
    uint64_t k = 0;
    for (uint64_t i = 0; i < S.vl; i++)
        if (fv_mbit(vs1, i)) fv_vset(vd, k++, sew, fv_vget(vs2, i, sew));
    fv_tail(vd, k, sew, S.lmul8, 1);
}

enum { R_SUM, R_MAX, R_MAXU, R_MIN, R_MINU, R_AND, R_OR, R_XOR, R_WSUM, R_WSUMU };

/* vd[0] = reduce(vs1[0], vs2[*]) */
static inline void fv_vred(int op, unsigned vd, unsigned vs2, unsigned vs1, int masked)
{
    fv_vcheck();
    unsigned sew = S.sew, dsew = (op == R_WSUM || op == R_WSUMU) ? 2 * sew : sew;
    if (S.vl == 0) return;
    uint64_t acc = fv_vget(vs1, 0, dsew), m = fv_mask(dsew);
    for (uint64_t i = 0; i < S.vl; i++) {
        uint64_t before = acc;
        uint64_t a = fv_vget(vs2, i, sew);
        switch (op) {
        case R_SUM: acc = (acc + a) & m; break;
        case R_WSUM: acc = (acc + (uint64_t)fv_s(a, sew)) & m; break;
        case R_WSUMU: acc = (acc + a) & m; break;
        case R_MAX: acc = fv_s(a, sew) > fv_s(acc, sew) ? a : acc; break;
        case R_MAXU: acc = a > acc ? a : acc; break;
        case R_MIN: acc = fv_s(a, sew) < fv_s(acc, sew) ? a : acc; break;
        case R_MINU: acc = a < acc ? a : acc; break;
        case R_AND: acc &= a; break;
        case R_OR: acc |= a; break;
        default: acc ^= a; break;
        }
        acc = fv_active(masked, i) ? acc : before;
    }
    fv_vset(vd, 0, dsew, acc);
    if (S.vta) {
        _Bool ones = FV_AGNOSTIC();
        for (uint64_t i = 1; i < (uint64_t)VLENB * 8 / dsew; i++)
            fv_vset(vd, i, dsew, ones ? fv_mask(dsew) : fv_vget(vd, i, dsew));
    }
}

/* mask-register logical ops over vl bits (vmand.mm vd, vs2, vs1, ...) */
enum { M_AND, M_NAND, M_ANDN, M_XOR, M_OR, M_NOR, M_ORN, M_XNOR };

static inline void fv_vmlogic(int op, unsigned vd, unsigned vs2, unsigned vs1)
{
    fv_vcheck();
    uint8_t res[VLENB];
    for (unsigned b = 0; b < VLENB; b++) res[b] = S.v[vd * VLENB + b];
    for (uint64_t i = 0; i < S.vl; i++) {
        int a = fv_mbit(vs2, i), b = fv_mbit(vs1, i), r;
        switch (op) {
        case M_AND: r = a & b; break;
        case M_NAND: r = !(a & b); break;
        case M_ANDN: r = a & !b; break;
        case M_XOR: r = a ^ b; break;
        case M_OR: r = a | b; break;
        case M_NOR: r = !(a | b); break;
        case M_ORN: r = a | !b; break;
        default: r = !(a ^ b); break;
        }
        res[i / 8] = (uint8_t)((res[i / 8] & ~(1u << (i % 8))) | ((unsigned)r << (i % 8)));
    }
    _Bool tail_ones = FV_AGNOSTIC();
    for (uint64_t i = S.vl; i < (uint64_t)FV_VLEN; i++)
        res[i / 8] = (uint8_t)(tail_ones ? (res[i / 8] | (1u << (i % 8))) : res[i / 8]);
    for (unsigned b = 0; b < VLENB; b++) S.v[vd * VLENB + b] = res[b];
}

static inline uint64_t fv_vcpop(unsigned vs2, int masked)
{
    fv_vcheck();
    uint64_t n = 0;
    for (uint64_t i = 0; i < S.vl; i++)
        if ((!masked || fv_mbit(0, i)) && fv_mbit(vs2, i)) n++;
    return n;
}

static inline uint64_t fv_vfirst(unsigned vs2, int masked)
{
    fv_vcheck();
    for (uint64_t i = 0; i < S.vl; i++)
        if ((!masked || fv_mbit(0, i)) && fv_mbit(vs2, i)) return i;
    return ~0ULL;
}

/* ------------------------------------------------------------------ memory */

enum { A_UNIT, A_STRIDED, A_INDEXED };

/* loads and stores, segment forms included: nf fields, data EEW deew (= SEW for
 * indexed), index EEW ieew for indexed; field f of element i lives in register
 * vd + f * max(1, EMUL) */
static inline void fv_vmem(int store, int mode, unsigned vd, uint64_t base, uint64_t stride, unsigned vidx,
                           unsigned deew, unsigned ieew, unsigned nf, int masked)
{
    fv_vcheck();
    unsigned emul8 = mode == A_INDEXED ? S.lmul8 : fv_emul8(deew);
    unsigned regs = emul8 <= 8 ? 1 : emul8 / 8;
    FV_ASSERT(nf * regs <= 8, "segment group larger than 8 registers");
    for (unsigned f = 0; f < nf; f++) fv_group_check(vd + f * regs, emul8, "memory vd group");
    if (mode == A_INDEXED) fv_group_check(vidx, (unsigned)((uint64_t)S.lmul8 * ieew / S.sew), "index group");
    unsigned bytes = deew / 8;
    int agn = FV_AGNOSTIC();
    for (uint64_t i = 0; i < S.vl; i++) {
        if (masked && !fv_mbit(0, i)) {
            if (!store) for (unsigned f = 0; f < nf; f++) fv_inactive(vd + f * regs, i, deew, agn);
            continue;
        }
        uint64_t elem = mode == A_UNIT ? base + i * nf * bytes
                      : mode == A_STRIDED ? base + i * stride
                      : base + fv_vget(vidx, i, ieew);
        for (unsigned f = 0; f < nf; f++) {
            uint64_t addr = elem + f * bytes;
            FV_ASSERT(addr % bytes == 0, "misaligned vector element access");
            if (store) fv_store(addr, bytes, fv_vget(vd + f * regs, i, deew));
            else fv_vset(vd + f * regs, i, deew, fv_load(addr, bytes));
        }
    }
    if (!store)
        for (unsigned f = 0; f < nf; f++) fv_tail(vd + f * regs, S.vl, deew, emul8, S.vta);
}

/* whole-register loads/stores: n registers, ignore vl and vtype */
static inline void fv_vmem_whole(int store, unsigned vd, uint64_t base, unsigned n)
{
    FV_ASSERT(vd % n == 0 && vd + n <= 32, "whole-register group alignment");
    for (unsigned b = 0; b < n * VLENB; b++) {
        if (store) fv_store(base + b, 1, S.v[vd * VLENB + b]);
        else S.v[vd * VLENB + b] = (uint8_t)fv_load(base + b, 1);
    }
}

/* vlm.v / vsm.v: ceil(vl / 8) bytes */
static inline void fv_vmem_mask(int store, unsigned vd, uint64_t base)
{
    fv_vcheck();
    uint64_t n = (S.vl + 7) / 8;
    for (uint64_t b = 0; b < n; b++) {
        if (store) fv_store(base + b, 1, S.v[vd * VLENB + b]);
        else S.v[vd * VLENB + b] = (uint8_t)fv_load(base + b, 1);
    }
}

/* ------------------------------------------------------------------ scalar helpers */

static inline uint64_t fv_clz(uint64_t v, unsigned bits)
{
    v &= fv_mask(bits);
    uint64_t n = 0;
    for (int i = (int)bits - 1; i >= 0 && !((v >> i) & 1); i--) n++;
    return n;
}

static inline uint64_t fv_ctz(uint64_t v, unsigned bits)
{
    v &= fv_mask(bits);
    uint64_t n = 0;
    for (unsigned i = 0; i < bits && !((v >> i) & 1); i++) n++;
    return n;
}

static inline uint64_t fv_cpop(uint64_t v, unsigned bits)
{
    v &= fv_mask(bits);
    uint64_t n = 0;
    for (unsigned i = 0; i < bits; i++) n += (v >> i) & 1;
    return n;
}

static inline uint64_t fv_div(uint64_t a, uint64_t b, unsigned bits, int is_signed, int rem)
{
    uint64_t m = fv_mask(bits);
    if (is_signed) {
        int64_t sa = fv_s(a, bits), sb = fv_s(b, bits);
        if (sb == 0) return fv_sx(rem ? a : m, bits);
        int64_t mn = bits == 64 ? INT64_MIN : -(int64_t)(1ULL << (bits - 1));
        if (sa == mn && sb == -1) return fv_sx(rem ? 0 : a, bits);
        return fv_sx((uint64_t)(rem ? sa % sb : sa / sb), bits);
    }
    uint64_t ua = a & m, ub = b & m;
    if (ub == 0) return fv_sx(rem ? ua : m, bits);
    return fv_sx(rem ? ua % ub : ua / ub, bits);
}

static inline uint64_t fv_csr_read(unsigned csr)
{
    switch (csr) {
    case 0xc22: return VLENB;          /* vlenb */
    case 0xc20: return S.vl;           /* vl */
    case 0xc21: return fv_vtype_csr(); /* vtype */
    case 0x00a: return S.vxrm;         /* vxrm */
    case 0x009: return S.vxsat;        /* vxsat */
    case 0x00f: return (S.vxrm << 1) | S.vxsat; /* vcsr */
    case 0x008: return 0;              /* vstart */
    }
    FV_ASSERT(0, "unmodelled CSR read");
    return 0;
}

static inline void fv_csr_write(unsigned csr, uint64_t v)
{
    switch (csr) {
    case 0x00a: S.vxrm = v & 3; return;
    case 0x009: S.vxsat = v & 1; return;
    case 0x00f: S.vxrm = (v >> 1) & 3; S.vxsat = v & 1; return;
    case 0x008: FV_ASSERT(v == 0, "nonzero vstart"); return;
    }
    FV_ASSERT(0, "unmodelled CSR write");
}

#endif
