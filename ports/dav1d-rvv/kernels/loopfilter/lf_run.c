/* minimal lpf runner under qemu: calls the 4 sb wrappers once each with a
 * small vmask, catches SIGILL/SIGBUS/SIGSEGV and prints the faulting PC. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
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

static void handler(int sig, siginfo_t *si, void *uc)
{
    ucontext_t *ctx = uc;
    fprintf(stderr, "TRAP sig=%d pc=%p\n", sig,
            (void *)ctx->uc_mcontext.__gregs[0]);
    _exit(100);
}

int main(void)
{
    struct sigaction sa = { .sa_sigaction = handler, .sa_flags = SA_SIGINFO };
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);

    static uint8_t buf[128 * 128];
    static uint8_t l[64][4];
    uint32_t vmask[4] = { 0 };
    struct { uint8_t e[64]; uint8_t i[64]; uint64_t sharp[2]; } lut;
    for (int i = 0; i < 64; i++) { lut.e[i] = 10; lut.i[i] = 5; }
    memset(buf, 0x80, sizeof(buf));
    memset(l, 32, sizeof(l));
    vmask[0] = 1; /* one edge */

    lpf_fn fns[4] = {
        (lpf_fn)dav1d_lpf_h_sb_y_8bpc_rvv, (lpf_fn)dav1d_lpf_v_sb_y_8bpc_rvv,
        (lpf_fn)dav1d_lpf_h_sb_uv_8bpc_rvv, (lpf_fn)dav1d_lpf_v_sb_uv_8bpc_rvv };
    const char *names[4] = { "h_sb_y", "v_sb_y", "h_sb_uv", "v_sb_uv" };
    int wd[4] = { 4, 4, 4, 4 };
    for (int i = 0; i < 4; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            fns[i](buf, 128, vmask, (const uint8_t (*)[4])l + 8, 8, &lut, wd[i]);
            printf("%s: ok\n", names[i]);
            fflush(stdout);
            _exit(0);
        }
        int st;
        waitpid(pid, &st, 0);
        if (WIFSIGNALED(st))
            printf("%s: SIGNAL %d\n", names[i], WTERMSIG(st));
        else if (WEXITSTATUS(st))
            printf("%s: exit %d\n", names[i], WEXITSTATUS(st));
        fflush(stdout);
    }
    return 0;
}
