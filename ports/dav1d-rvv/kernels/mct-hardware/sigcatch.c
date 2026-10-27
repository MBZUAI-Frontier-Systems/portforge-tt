/* LD_PRELOAD SIGBUS/SIGSEGV/SIGILL catcher: prints sig, si_code, si_addr, PC
 * and a /proc/self/maps excerpt around the PC, then re-raises. Build on the
 * board: gcc -shared -fPIC -O2 -o sigcatch.so sigcatch.c */
#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ucontext.h>

static void handler(int sig, siginfo_t *si, void *uc)
{
    ucontext_t *ctx = uc;
    unsigned long pc = ctx->uc_mcontext.__gregs[0];
    fprintf(stderr, "\nSIGCATCH sig=%d si_code=%d si_addr=%p pc=0x%lx\n",
            sig, si->si_code, si->si_addr, pc);
    fflush(stderr);
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "grep -F '' /proc/%d/maps | awk '$3 ~ /checkasm/' >&2", getpid());
    system(cmd);
    /* restore default and re-raise so the exit status reflects the signal */
    signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor)) static void install(void)
{
    struct sigaction sa = { .sa_sigaction = handler, .sa_flags = SA_SIGINFO };
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
}
