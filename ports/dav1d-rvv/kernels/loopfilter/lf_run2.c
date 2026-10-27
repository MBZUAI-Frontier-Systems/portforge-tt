/* checkasm-geometry replica for the lpf RVV port: same buffers, pointer
 * arithmetic, vmask randomization and per-bit L lookups as
 * tests/checkasm/loopfilter.c check_lpf_sb(), calling the RVV functions
 * directly. Any SIGILL/SIGSEGV prints the faulting PC. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <ucontext.h>

typedef void (*lpf_fn)(uint8_t *dst, ptrdiff_t stride, const uint32_t *mask,
                       const uint8_t (*l)[4], ptrdiff_t b4_stride,
                       const void *lut, int w);

extern void dav1d_lpf_h_sb_y_8bpc_rvv(uint8_t *, ptrdiff_t, const uint32_t *,
                                      const uint8_t (*)[4], ptrdiff_t, const void *, int);
extern void dav1d_lpf_v_sb_y_8bpc_rvv(uint8_t *, ptrdiff_t, const uint32_t *,
                                      const uint8_t (*)[4], ptrdiff_t, const void *, int);
extern void dav1d_lpf_h_sb_uv_8bpc_rvv(uint8_t *, ptrdiff_t, const uint32_t *,
                                       const uint8_t (*)[4], ptrdiff_t, const void *, int);
extern void dav1d_lpf_v_sb_uv_8bpc_rvv(uint8_t *, ptrdiff_t, const uint32_t *,
                                       const uint8_t (*)[4], ptrdiff_t, const void *, int);

static uint64_t rs = 0x123456789abcdefULL;
static uint32_t rnd(void)
{
    rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
    return (uint32_t)(rs * 0x2545F4914F6CDD1DULL >> 32);
}

static void handler(int sig, siginfo_t *si, void *uc)
{
    ucontext_t *ctx = uc;
    fprintf(stderr, "TRAP sig=%d addr=%p pc=%p\n", sig, si->si_addr,
            (void *)ctx->uc_mcontext.__gregs[0]);
    _exit(100);
}

static void run_case(lpf_fn fn, const char *name, int n_blks, int lf_idx,
                     int is_chroma, int dir)
{
    static uint8_t c_dst_mem[128 * 16] __attribute__((aligned(64)));
    uint32_t vmask[4] = { 0 };
    uint8_t l[64][4];
    memset(c_dst_mem, 0x80, sizeof(c_dst_mem));
    memset(l, 0, sizeof(l));

    uint8_t *dst;
    ptrdiff_t stride, b4_stride;
    if (dir) {
        dst = c_dst_mem + n_blks * 4 * 8;
        stride = n_blks * 4;
        b4_stride = 32;
    } else {
        dst = c_dst_mem + 8;
        stride = 16;
        b4_stride = 2;
    }
    struct { uint8_t e[64]; uint8_t i[64]; uint64_t sharp[2]; } lut;
    const int sharp = rnd() & 7;
    for (int level = 0; level < 64; level++) {
        int limit = level;
        if (sharp > 0) { limit >>= (sharp + 3) >> 2; limit = limit < 9 - sharp ? limit : 9 - sharp; }
        limit = limit < 1 ? 1 : limit;
        lut.i[level] = limit;
        lut.e[level] = 2 * (level + 2) + limit;
    }
    const int n_strengths = is_chroma ? 2 : 3;
    for (int j = 0; j < n_blks; j++) {
        const int idx = rnd() % (n_strengths + 1);
        if (idx) vmask[idx - 1] |= 1U << j;
        if (dir) {
            l[j][lf_idx] = rnd() & 63;
            l[j + 32][lf_idx] = rnd() & 63;
        } else {
            l[j * 2][lf_idx] = rnd() & 63;
            l[j * 2 + 1][lf_idx] = rnd() & 63;
        }
    }
    const int n_bits = 4 * n_blks;
    /* worst-case: every possible bit set, several seeds */
    for (int pattern = 0; pattern < 3; pattern++) {
        if (pattern == 0) { vmask[0] = vmask[1] = vmask[2] = -1u; }
        else if (pattern == 1) { vmask[0] = -1u; vmask[1] = vmask[2] = 0; }
        else { vmask[0] = 0; vmask[1] = -1u; vmask[2] = 0; }
        memcpy(c_dst_mem + (dir ? 1024 : 0), c_dst_mem, 128 * 16 - (dir ? 1024 : 0));
        pid_t pid = fork();
        if (pid == 0) {
            fn(dst, stride, vmask, (const uint8_t (*)[4])((uint8_t *)l + (dir ? 128 : 4) + lf_idx),
               b4_stride, &lut, n_blks);
            _exit(0);
        }
        int st;
        waitpid(pid, &st, 0);
        if (WIFSIGNALED(st) || WEXITSTATUS(st)) {
            printf("%s pattern %d: FAILED (%d/%d)\n", name, pattern,
                   WIFSIGNALED(st) ? WTERMSIG(st) : 0, WEXITSTATUS(st));
            fflush(stdout);
            return;
        }
    }
    printf("%s: ok\n", name);
    fflush(stdout);
}

int main(void)
{
    struct sigaction sa = { .sa_sigaction = handler, .sa_flags = SA_SIGINFO };
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);

    run_case(dav1d_lpf_h_sb_y_8bpc_rvv, "h_sb_y", 32, 0, 0, 0);
    run_case(dav1d_lpf_v_sb_y_8bpc_rvv, "v_sb_y", 32, 1, 0, 1);
    run_case(dav1d_lpf_h_sb_uv_8bpc_rvv, "h_sb_uv", 16, 2, 1, 0);
    run_case(dav1d_lpf_v_sb_uv_8bpc_rvv, "v_sb_uv", 16, 2, 1, 1);
    return 0;
}
