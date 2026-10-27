/* checkasm-replica driver for the mct/mc SIGBUS: calls the verbatim upstream
 * mc.S slice (prep+put filter_row helpers, chooser, dispatch) exactly like
 * check_mct/check_mc do, catching SIGBUS per case. Board build:
 *   gcc -O2 -march=rv64gcv -I/tmp/mctrepro -I<src tree> -o mct_repro
 *       mct_repro.c mc_slice.S   */
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <alloca.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <ucontext.h>
#include <sys/wait.h>

typedef void (*mct_fn)(int16_t *dst, const uint8_t *src, ptrdiff_t stride,
                       int w, int h, int mx, int my);
typedef void (*mc_fn)(uint8_t *dst, ptrdiff_t dst_stride, const uint8_t *src,
                      ptrdiff_t src_stride, int w, int h, int mx, int my);

extern void dav1d_prep_8tap_regular_sharp_8bpc_rvv(int16_t *, const uint8_t *,
                                                   ptrdiff_t, int, int, int, int);
extern void dav1d_put_8tap_regular_sharp_8bpc_rvv(uint8_t *, ptrdiff_t,
                                                  const uint8_t *, ptrdiff_t,
                                                  int, int, int, int);
/* upstream name the chooser references via X(mc_subpel_filters) */
const uint8_t dav1d_mc_subpel_filters[6][15][8] = {
    [0] = { {128, 0, 0, 0, 0, 0, 0, 0} }, /* values irrelevant to the trap */
};

static int case_w, case_h, case_mx, case_my, case_is_mct;
static volatile int trapped;
static unsigned long trap_pc, trap_addr;

static void handler(int sig, siginfo_t *si, void *uc)
{
    ucontext_t *ctx = uc;
    fprintf(stderr, "  TRAP sig=%d code=%d addr=%p pc=%p\n",
            sig, si->si_code, si->si_addr, (void *)ctx->uc_mcontext.__gregs[0]);
    _exit(100);
}

int main(void)
{
    /* checkasm keeps src_buf + two 128x128 int16 tmps in ~84KB of stack frame;
     * the trapping ld lands ~35KB above sp. Replicate that placement. */
    static uint8_t *src_buf;
    static int16_t *tmp;
    static uint8_t *dst;
    uint8_t *stack_slab = alloca(64 + 135 * 135 + 64 + 2 * (128 * 128 * 2 + 64) + 64);
    src_buf = (uint8_t *)(((unsigned long)stack_slab + 63) & ~63UL);
    tmp = (int16_t *)(((unsigned long)(src_buf + 135 * 135) + 63) & ~63UL);
    dst = (uint8_t *)(((unsigned long)(tmp + 128 * 128) + 63) & ~63UL);
    const int sign = -1;
    for (int y = 0; y < 135; y++)
        for (int x = 0; x < 135; x++)
            src_buf[135 * y + x] = ((x | y) < 8 ? ((x ^ y ^ sign) & 1 ? 0xff : 0)
                                                : (uint8_t)(x * 7 + y));
    const uint8_t *src = src_buf + 135 * 3 + 3;
    struct sigaction sa = { .sa_sigaction = handler, .sa_flags = SA_SIGINFO };
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    {
        char cmd[64];
        snprintf(cmd, sizeof(cmd), "grep mct_repro /proc/%d/maps 1>&2", getpid());
        system(cmd);
    }

    int id = 0;
    for (int is_mct = 0; is_mct <= 1; is_mct++)
        for (int w = 4; w <= 32; w <<= 1)
            for (int mxy = 0; mxy < 4; mxy++) {
                int mx = (mxy & 1) ? 8 : 0, my = (mxy & 2) ? 8 : 0;
                int h = w * 2 > 128 ? 128 : w * 2;
                id++;
                pid_t pid = fork();
                if (pid == 0) {
                    /* checkasm runs the C reference immediately before the asm
                     * call, writing ~64KB of tmp buffers -> cold L1 for the
                     * src line the asm then loads. Replicate that. */
                    memset(tmp, 0x55, 128 * 128 * 2);
                    memset(dst, 0x55, 128 * 128);
                    if (is_mct)
                        dav1d_prep_8tap_regular_sharp_8bpc_rvv(
                            tmp, src, 135, w, h, mx, my);
                    else
                        dav1d_put_8tap_regular_sharp_8bpc_rvv(
                            dst, 136, src, 135, w, h, mx, my);
                    printf("case %2d %s w=%-3d h=%-3d mxy=%d : OK\n",
                           id, is_mct ? "mct" : "mc ", w, h, mxy);
                    fflush(stdout);
                    _exit(0);
                }
                int st;
                waitpid(pid, &st, 0);
                if (WIFSIGNALED(st))
                    printf("case %2d %s w=%-3d h=%-3d mxy=%d : SIGNAL %d\n",
                           id, is_mct ? "mct" : "mc ", w, h, mxy, WTERMSIG(st));
                else if (WEXITSTATUS(st) != 0)
                    printf("case %2d %s w=%-3d h=%-3d mxy=%d : exit %d\n",
                           id, is_mct ? "mct" : "mc ", w, h, mxy, WEXITSTATUS(st));
                fflush(stdout);
            }
    return 0;
}
