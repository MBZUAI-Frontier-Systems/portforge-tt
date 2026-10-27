/* Does a misaligned scalar ld trap when VLSU state is live?
 * Cases: ld alone / after vsetvli / after vle8.v (vl=8 e16) / after vle8 + vsll,
 * at alignment 0..7, fork per case so a SIGBUS only kills that case.
 * gcc -O2 -march=rv64gcv -o misalign3 misalign3.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <ucontext.h>

static int case_id;
static char buf[256] __attribute__((aligned(64)));

static void handler(int sig, siginfo_t *si, void *uc)
{
    ucontext_t *ctx = uc;
    fprintf(stderr, "CASE %d: sig=%d code=%d addr=%p pc=%p\n", case_id, sig,
            si->si_code, si->si_addr, (void *)ctx->uc_mcontext.__gregs[0]);
    _exit(100);
}

static void run_case(int id, int preload, int off)
{
    case_id = id;
    pid_t pid = fork();
    if (pid == 0) {
        unsigned long v = 0, vl = 8;
        const char *src = buf + 16 + off;
        switch (preload) {
        case 0: /* bare ld */
            __asm__ volatile ("ld %0, 0(%1)" : "=r"(v) : "r"(src) : "memory");
            break;
        case 1: /* vsetvli then ld */
            __asm__ volatile ("vsetvli t0, %1, e16, m1, ta, ma\n\t"
                              "ld %0, 0(%2)"
                              : "=r"(v) : "r"(vl), "r"(src) : "t0", "memory");
            break;
        case 2: /* vsetvli, vle8, ld — the mc.S sequence */
            __asm__ volatile ("vsetvli t0, %1, e16, m1, ta, ma\n\t"
                              "vle8.v v30, (%2)\n\t"
                              "ld %0, 0(%3)"
                              : "=r"(v) : "r"(vl), "r"(buf + 16), "r"(src)
                              : "t0", "memory");
            break;
        case 3: /* vle8, ld, rd==rs1 as t3 like upstream */
            __asm__ volatile ("vsetvli t0, %1, e16, m1, ta, ma\n\t"
                              "vle8.v v30, (%2)\n\t"
                              "add t3, %3, %4\n\t"
                              "ld t3, 0(t3)"
                              : "=r"(v) : "r"(vl), "r"(buf + 16), "r"(src), "r"(0UL)
                              : "t0", "t3", "memory");
            break;
        }
        printf("CASE %d ok v=%016lx\n", id, v);
        _exit(0);
    }
    int st;
    waitpid(pid, &st, 0);
    if (WIFEXITED(st) && WEXITSTATUS(st) == 0)
        ;
    else if (WIFEXITED(st))
        fprintf(stderr, "CASE %d: trapped (exit %d)\n", id, WEXITSTATUS(st));
    else if (WIFSIGNALED(st))
        fprintf(stderr, "CASE %d: died by signal %d\n", id, WTERMSIG(st));
}

int main(void)
{
    struct sigaction sa = { .sa_sigaction = handler, .sa_flags = SA_SIGINFO };
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    memset(buf, 0x5A, sizeof(buf));

    int id = 1;
    for (int preload = 0; preload <= 3; preload++)
        for (int off = 0; off < 8; off++, id++)
            run_case(id, preload, off);
    return 0;
}
