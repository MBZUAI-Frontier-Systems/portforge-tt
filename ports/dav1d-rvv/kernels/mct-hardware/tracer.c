/* ptrace tracer: run a child, report every SIGBUS/SIGSEGV/SIGILL delivery-stop
 * (sig, si_code, si_addr, PC) then re-inject the signal so the child's own
 * handler sees it. gcc -O2 -o tracer tracer.c ; ./tracer ./checkasm args... */
#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>
#include <elf.h>
#include <errno.h>

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: tracer cmd args...\n"); return 2; }
    pid_t pid = fork();
    if (pid == 0) {
        ptrace(PTRACE_TRACEME, 0, 0, 0);
        execv(argv[1], argv + 1);
        _exit(127);
    }
    int status;
    long pc = 0;
    while (1) {
        if (waitpid(pid, &status, 0) < 0) { perror("waitpid"); return 3; }
        if (WIFEXITED(status)) { printf("child exit %d\n", WEXITSTATUS(status)); break; }
        if (WIFSIGNALED(status)) { printf("child killed by %d\n", WTERMSIG(status)); break; }
        int sig = WSTOPSIG(status);
        int deliver = sig;
        if (sig == SIGTRAP) { deliver = 0; } /* exec + any stray trap */
        else if (sig == SIGBUS || sig == SIGSEGV || sig == SIGILL) {
            siginfo_t si; memset(&si, 0, sizeof(si));
            if (ptrace(PTRACE_GETSIGINFO, pid, 0, &si) == 0) {
                struct iovec iov; unsigned long regs[33];
                iov.iov_base = regs; iov.iov_len = sizeof(regs);
                long ok = ptrace(PTRACE_GETREGSET, pid, (void *)NT_PRSTATUS, &iov);
                unsigned long upc = ok == 0 ? regs[0] : 0;
                printf("SIG %d code=%d addr=%p pc=0x%lx\n", sig, si.si_code,
                       si.si_addr, upc);
                if (ok == 0) {
                    static const char *abi[32] = {
                        "zero","ra","sp","gp","tp","t0","t1","t2","s0","s1",
                        "a0","a1","a2","a3","a4","a5","a6","a7","s2","s3",
                        "s4","s5","s6","s7","s8","s9","s10","s11","t3","t4",
                        "t5","t6"};
                    for (int r = 0; r < 32; r++)
                        printf("  %-4s = 0x%016lx\n", abi[r], regs[r]);
                }
                fflush(stdout);
                char cmd[128];
                snprintf(cmd, sizeof(cmd),
                         "head -6 /proc/%d/maps >&2; echo .... >&2; "
                         "grep -E 'checkasm|stack' /proc/%d/maps >&2", pid, pid);
                system(cmd);
            }
        }
        if (ptrace(PTRACE_CONT, pid, 0, (void *)(long)deliver) < 0) {
            if (errno == ESRCH) break;
            perror("PTRACE_CONT"); return 4;
        }
    }
    return 0;
}
