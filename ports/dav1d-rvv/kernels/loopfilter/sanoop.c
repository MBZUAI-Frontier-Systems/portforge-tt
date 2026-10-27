/* LD_PRELOAD shim for checkasm under qemu-user: refuses to install handlers
 * for SIGILL/SIGBUS/SIGSEGV so the first crash is fatal and QEMU reports the
 * guest PC (uncaught target signal). Everything else passes through. */
#define _GNU_SOURCE
#include <signal.h>
#include <string.h>

extern int __sigaction(int, const struct sigaction *, struct sigaction *);

int sigaction(int signum, const struct sigaction *act, struct sigaction *oldact)
{
    if ((signum == SIGILL || signum == SIGBUS || signum == SIGSEGV) && act) {
        struct sigaction dfl;
        memset(&dfl, 0, sizeof(dfl));
        dfl.sa_handler = SIG_DFL;
        return __sigaction(signum, &dfl, oldact);
    }
    return __sigaction(signum, act, oldact);
}
