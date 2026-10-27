#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>

void dav1d_sgr_box5_row_h_8bpc_rvv(int32_t *, int16_t *, const uint8_t (*)[4],
                                   const uint8_t *, int, int);
void dav1d_sgr_box5_row_v_8bpc_rvv(int32_t **, int16_t **, int32_t *, int16_t *, int);
void dav1d_sgr_calc_ab2_8bpc_rvv(int32_t *, int16_t *, int, unsigned, const uint8_t *);
void dav1d_sgr_finish_filter2_8bpc_rvv(int16_t *, const uint8_t *, long,
                                       int32_t **, int16_t **, int, int);

static uint8_t lut[256];
static uint8_t srcbuf[64], left[4][4];
static int32_t aa[64], so[64], r32[5][64];
static int16_t bb[64], sumo[64], tmp[2*384], r16[5][64];

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
    printf("box5_row_h w=%d edges=0xf\n", w); fflush(stdout);
    dav1d_sgr_box5_row_h_8bpc_rvv(aa, bb, left, srcbuf, w, 0xf);
    printf("  -> sum=%d sq=%d\n", bb[0], aa[0]); fflush(stdout);

    printf("box5_row_v w=%d\n", w); fflush(stdout);
    { int32_t *p32[5] = { r32[0], r32[1], r32[2], r32[3], r32[4] };
      int16_t *p16[5] = { r16[0], r16[1], r16[2], r16[3], r16[4] };
      dav1d_sgr_box5_row_v_8bpc_rvv(p32, p16, so, sumo, w); }
    printf("  ok\n"); fflush(stdout);

    printf("calc_ab2 w=%d s=56\n", w); fflush(stdout);
    dav1d_sgr_calc_ab2_8bpc_rvv(aa, bb, w, 56, lut);
    printf("  -> A=%d B=%d\n", aa[0], bb[0]); fflush(stdout);

    printf("finish_filter2 w=%d h=2\n", w); fflush(stdout);
    { int32_t *A[2] = { r32[0], r32[1] };
      int16_t *B[2] = { r16[0], r16[1] };
      dav1d_sgr_finish_filter2_8bpc_rvv(tmp, srcbuf, 448, A, B, w, 2); }
    printf("  -> tmp0=%d tmp384=%d\n", tmp[0], tmp[384]); fflush(stdout);

    printf("finish_filter2 w=%d h=1\n", w); fflush(stdout);
    { int32_t *A[2] = { r32[0], r32[1] };
      int16_t *B[2] = { r16[0], r16[1] };
      dav1d_sgr_finish_filter2_8bpc_rvv(tmp, srcbuf, 448, A, B, w, 1); }
    printf("  -> tmp0=%d\n", tmp[0]); fflush(stdout);

    printf("all 5x5 kernels ok\n");
    return 0;
}
