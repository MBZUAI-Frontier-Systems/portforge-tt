/* Harness runtime for lifted kernels (portforge_tt/fv): memory regions, the
 * call wrapper with psABI checks, and input generation that is symbolic under
 * CBMC and seeded-random natively.
 *
 * A harness declares its buffers as regions in the candidate's address space
 * (concrete addresses, so the solver sees constant indices), fills them with
 * fv_fill(), copies them for the reference, calls the reference on the copy and
 * the lifted kernel through fv_call(), and compares with fv_compare().
 *
 * Every candidate access is checked: loads must fall inside a readable region
 * or a data symbol the code addresses, stores inside a writable region or the
 * stack. Misaligned scalar accesses are a separate property ("alignment") that
 * the runner reports as a performance finding, not a correctness failure: Linux
 * emulates them on the X280 at ~300x the cost.
 */
#ifndef PORTFORGE_FV_RUNTIME_H
#define PORTFORGE_FV_RUNTIME_H

#include <stdint.h>
#include <string.h>
#include "rvv_model.h"

fv_state S;

typedef struct {
    const char *name;
    uint64_t base, size;
    uint8_t *mem;           /* the candidate's copy */
    int writable;
} fv_region;

typedef struct {
    uint64_t base, size;
    const uint8_t *bytes;
} fv_data;

#ifndef FV_MAX_REGIONS
#define FV_MAX_REGIONS 80
#endif
static fv_region fv_regions[FV_MAX_REGIONS];
static int fv_nregions;

/* Every candidate-side buffer (and the stack) lives in one arena. With separate
 * arrays CBMC cannot tell which object a region's pointer names, so each byte
 * store became an update of every buffer (wiener W=2: >800 s in SSA conversion);
 * with one object and constant offsets each byte is its own variable. */
#ifndef FV_ARENA_SIZE
#define FV_ARENA_SIZE 32768
#endif
static uint8_t fv_arena[FV_ARENA_SIZE];
static uint64_t fv_arena_used;

static inline uint8_t *fv_alloc(uint64_t n)
{
    uint8_t *p = fv_arena + fv_arena_used;
    fv_arena_used = (fv_arena_used + n + 15) & ~15ULL;
    FV_ASSERT(fv_arena_used <= FV_ARENA_SIZE, "harness: arena too small (raise FV_ARENA_SIZE)");
    return p;
}

/* lifted code defines these (data symbols its auipc pairs address) */
extern const fv_data FV_DATA[];
extern const int FV_NDATA;

#ifndef FV_CBMC
int fv_agnostic_ones;
static unsigned long long fv_rng = 0x9e3779b97f4a7c15ULL;
void fv_fail(const char *msg)
{
    fprintf(stderr, "FV-FAIL: %s\n", msg);
    exit(3);
}
static uint8_t fv_rand8(void)
{
    fv_rng ^= fv_rng << 13; fv_rng ^= fv_rng >> 7; fv_rng ^= fv_rng << 17;
    return (uint8_t)(fv_rng >> 24);
}
#define FV_INPUT8() fv_rand8()
#else
#define FV_INPUT8() nondet_u8()
#endif

static inline void fv_region_add(const char *name, uint64_t base, uint64_t size, uint8_t *mem, int writable)
{
    FV_ASSERT(fv_nregions < FV_MAX_REGIONS, "too many regions");
    fv_regions[fv_nregions++] = (fv_region){name, base, size, mem, writable};
}

/* fill a buffer with inputs: symbolic under CBMC, random natively */
static inline void fv_fill(uint8_t *mem, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++) mem[i] = FV_INPUT8();
}

uint64_t fv_load(uint64_t a, unsigned n)
{
#ifdef FV_CHECK_ALIGN
    __CPROVER_assert(a % n == 0, "alignment: misaligned scalar load");
#endif
    for (int r = 0; r < fv_nregions; r++) {
        if (a >= fv_regions[r].base && a + n <= fv_regions[r].base + fv_regions[r].size) {
            uint64_t v = 0;
            for (unsigned b = 0; b < n; b++) v |= (uint64_t)fv_regions[r].mem[a - fv_regions[r].base + b] << (8 * b);
            return v;
        }
    }
    for (int d = 0; d < FV_NDATA; d++) {
        if (a >= FV_DATA[d].base && a + n <= FV_DATA[d].base + FV_DATA[d].size) {
            uint64_t v = 0;
            for (unsigned b = 0; b < n; b++) v |= (uint64_t)FV_DATA[d].bytes[a - FV_DATA[d].base + b] << (8 * b);
            return v;
        }
    }
    FV_ASSERT(0, "memory: load outside every buffer the kernel may read");
    return 0;
}

void fv_store(uint64_t a, unsigned n, uint64_t value)
{
#ifdef FV_CHECK_ALIGN
    __CPROVER_assert(a % n == 0, "alignment: misaligned scalar store");
#endif
    for (int r = 0; r < fv_nregions; r++) {
        if (a >= fv_regions[r].base && a + n <= fv_regions[r].base + fv_regions[r].size) {
            FV_ASSERT(fv_regions[r].writable, "memory: store into a read-only buffer");
            for (unsigned b = 0; b < n; b++) fv_regions[r].mem[a - fv_regions[r].base + b] = (uint8_t)(value >> (8 * b));
            return;
        }
    }
    FV_ASSERT(0, "memory: store outside every buffer the kernel may write");
}

/* ------------------------------------------------------------------ calls */

#define FV_RETURN_ADDR 0x7ff0000000ULL   /* not an instruction address: returning there ends the call */
#define FV_STACK_BASE 0x10000000ULL
#ifndef FV_STACK_SIZE
#define FV_STACK_SIZE 8192
#endif

static const int fv_callee_saved[] = {2, 3, 4, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

/* run a lifted function with up to 8 integer arguments under the psABI */
static inline void fv_call(void (*fn)(void), const uint64_t *args, int nargs)
{
    uint64_t saved[15], fsaved[12];
    static int stack_added;
    if (!stack_added) {
        fv_region_add("stack", FV_STACK_BASE, FV_STACK_SIZE, fv_alloc(FV_STACK_SIZE), 1);
        stack_added = 1;
    }
    for (int r = 0; r < 32; r++) {
        uint64_t v = 0;
        for (int b = 0; b < 8; b++) v |= (uint64_t)FV_INPUT8() << (8 * b);
        S.x[r] = v;
    }
    S.x[0] = 0;
    for (int r = 0; r < 32; r++) S.f[r] = S.x[r] ^ 0x5555555555555555ULL;
    for (uint64_t b = 0; b < sizeof S.v; b++) S.v[b] = FV_INPUT8();
    for (int i = 0; i < nargs; i++) S.x[10 + i] = args[i];
    S.x[1] = FV_RETURN_ADDR;
    S.x[2] = FV_STACK_BASE + FV_STACK_SIZE;          /* 16-byte aligned */
    S.vill = 1; S.vl = 0;
    S.vxrm = FV_INPUT8() & 3;
    S.vxsat = FV_INPUT8() & 1;
    for (int k = 0; k < 15; k++) saved[k] = S.x[fv_callee_saved[k]];
    for (int k = 0; k < 12; k++) fsaved[k] = S.f[k < 2 ? 8 + k : 16 + k];
    fn();
    for (int k = 0; k < 15; k++)
        FV_ASSERT(S.x[fv_callee_saved[k]] == saved[k], "abi: callee-saved register (sp, gp, tp, s0-s11) not restored");
    for (int k = 0; k < 12; k++)
        FV_ASSERT(S.f[k < 2 ? 8 + k : 16 + k] == fsaved[k], "abi: callee-saved fs register not restored");
}

/* equality of the candidate's and the reference's copies of one buffer */
static inline void fv_compare(const char *what, const uint8_t *cand, const uint8_t *ref, uint64_t n)
{
#ifdef FV_CBMC
    for (uint64_t i = 0; i < n; i++) __CPROVER_assert(cand[i] == ref[i], "equivalence: output differs from the C reference");
#else
    for (uint64_t i = 0; i < n; i++)
        if (cand[i] != ref[i]) {
            fprintf(stderr, "FV-FAIL: %s differs at byte %llu: candidate %u reference %u\n", what,
                    (unsigned long long)i, cand[i], ref[i]);
            exit(4);
        }
#endif
}

#endif
