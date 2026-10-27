/* Which misaligned-scalar-ld cases trap on this board? Matrix over:
 *   region: main stack / heap / mmap with untouched second page
 *   pinning: default vs hart 1 (checkasm uses taskset -c 1)
 *   crossing: into mapped memory vs into a never-faulted page
 * gcc -O2 -march=rv64gcv -o misalign2 misalign2.c  */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static void try_ld(const char *what, const volatile char *p)
{
    unsigned long v;
    __asm__ volatile ("ld %0, 0(%1)" : "=r"(v) : "r"(p) : "memory");
    printf("  %-46s ok (%016lx)\n", what, v);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    long pagesize = sysconf(_SC_PAGESIZE);

    char stackbuf[64] __attribute__((aligned(64)));
    memset(stackbuf, 0x11, sizeof(stackbuf));
    char *heapbuf = aligned_alloc(64, 4096);
    memset(heapbuf, 0x22, 4096);
    char *mm = mmap(NULL, 2 * pagesize, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(mm, 0x33, pagesize); /* touch only page 0 */

    printf("pagesize=%ld argv0=%s\n", pagesize, argv[0]);
    for (int off = 1; off < 8; off++) {
        char what[64];
        snprintf(what, sizeof(what), "ld stack+%d", off);
        try_ld(what, stackbuf + off);
    }
    for (int off = 1; off < 8; off++) {
        char what[64];
        snprintf(what, sizeof(what), "ld heap+%d", off);
        try_ld(what, heapbuf + off);
    }
    /* crossing into the never-touched second page */
    for (int off = 1; off < 8; off++) {
        char what[64];
        snprintf(what, sizeof(what), "ld mmap page0 end -8 +%d (crossing)", off);
        try_ld(what, mm + pagesize - 8 + off);
    }
    printf("no traps\n");
    return 0;
}
