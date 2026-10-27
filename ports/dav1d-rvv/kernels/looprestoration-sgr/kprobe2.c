/* kprobe + SIGILL/SIGSEGV handler that prints the faulting guest PC. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>

void dav1d_sgr_box3_row_h_8bpc_rvv(int32_t *, int16_t *, const uint8_t (*)[4],
                                   const uint8_t *, int, int);
void dav1d_sgr_box3_row_v_8bpc_rvv(int32_t **, int16_t **, int32_t *, int16_t *, int);
void dav1d_sgr_calc_ab1_8bpc_rvv(int32_t *, int16_t *, int, unsigned, const uint8_t *);
void dav1d_sgr_finish_filter_row1_8bpc_rvv(int16_t *, const uint8_t *,
                                           int32_t **, int16_t **, int);
void dav1d_sgr_weighted_row1_8bpc_rvv(uint8_t *, const int16_t *, int, int);

static uint8_t lut[256];
static uint8_t srcbuf[64], dst[64], left[4][4];
static int32_t aa[64], so[64], r32[3][64];
static int16_t bb[64], sumo[64], tmp[64], r16[3][64];

static void handler(int sig, siginfo_t *si, void *uc) {
    ucontext_t *u = uc;
    char buf[128];
    int n = snprintf(buf, sizeof buf, "signal %d pc=0x%llx badaddr=0x%llx\n",
                     sig, (unsigned long long)u->uc_mcontext.__gregs[0],
                     (unsigned long long)(uintptr_t)si->si_addr);
    write(2, buf, n);
    _exit(3);
}

int main(void) {
    struct sigaction sa = { .sa_sigaction = handler, .sa_flags = SA_SIGINFO };
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);

    for (int i = 0; i < 256; i++) lut[i] = (uint8_t)i;
    for (int i = 0; i < 64; i++) srcbuf[i] = 128;
    memset(left, 7, sizeof(left));

    int w = 37;
    printf("box3_row_h\n"); fflush(stdout);
    dav1d_sgr_box3_row_h_8bpc_rvv(aa, bb, left, srcbuf, w, 0xf);
    printf("box3_row_v\n"); fflush(stdout);
    { int32_t *p32[3] = { r32[0], r32[1], r32[2] };
      int16_t *p16[3] = { r16[0], r16[1], r16[2] };
      dav1d_sgr_box3_row_v_8bpc_rvv(p32, p16, so, sumo, w); }
    printf("calc_ab1\n"); fflush(stdout);
    dav1d_sgr_calc_ab1_8bpc_rvv(aa, bb, w, 2589, lut);
    printf("finish_filter_row1\n"); fflush(stdout);
    { int32_t *A[3] = { r32[0], r32[1], r32[2] };
      int16_t *B[3] = { r16[0], r16[1], r16[2] };
      dav1d_sgr_finish_filter_row1_8bpc_rvv(tmp, srcbuf, A, B, w); }
    printf("weighted_row1\n"); fflush(stdout);
    dav1d_sgr_weighted_row1_8bpc_rvv(dst, tmp, w, 100);
    printf("all kernels ok\n");
    return 0;
}
