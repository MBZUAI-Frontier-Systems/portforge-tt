/* Minimal reproducer for the mc_8bpc.mct SIGBUS on the X280:
 * does the board trap on a misaligned scalar ld (8-byte load)?
 *
 * handler prints sig, si_code, si_addr and the faulting PC from ucontext.
 * gcc -O2 -march=rv64gcv -o misalign_test misalign_test.c
 */
#include <stdio.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <ucontext.h>

static void handler(int sig, siginfo_t *si, void *uc)
{
    ucontext_t *ctx = uc;
    fprintf(stderr, "CAUGHT sig=%d si_code=%d si_addr=%p pc=%p\n",
            sig, si->si_code, si->si_addr, (void *)ctx->uc_mcontext.__gregs[0]);
    _exit(42);
}

int main(int argc, char **argv)
{
    struct sigaction sa = { .sa_sigaction = handler, .sa_flags = SA_SIGINFO };
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);

    char buf[64] __attribute__((aligned(64)));
    memset(buf, 0xAB, sizeof(buf));
    /* byte-granularity offsets, like dav1d's src = base + 3 + row*135 */
    for (int off = 0; off < 8; off++) {
        uint64_t v;
        __asm__ volatile ("ld %0, 0(%1)" : "=r"(v) : "r"(buf + off) : "memory");
        printf("ld  @+%d ok: %016lx\n", off, v);
        fflush(stdout);
    }
    /* and a misaligned vector element access for contrast: vle16 at odd addr */
    for (int off = 0; off < 2; off++) {
        uint64_t out[8] __attribute__((aligned(64)));
        const uint16_t *p = (const uint16_t *)(buf + 1 + off);
        __asm__ volatile (
            "vsetvli t0, %1, e16, m1, ta, ma\n\t"
            "vle16.v v1, (%2)\n\t"
            "vse16.v v1, (%3)"
            : : "r"(0), "r"(4UL), "r"(p), "r"(out) : "t0", "memory");
        printf("vle16 @+1+%d ok\n", off);
        fflush(stdout);
    }
    printf("no traps\n");
    return 0;
}
