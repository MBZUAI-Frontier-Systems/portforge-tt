// sizebench: qs8-vmul-minmax-rndnu on the X280. Times
//   - rvv_pf1            (this campaign's kernel)
//   - scalar_oracle      (literal Neon transcription, -O2)
//   - scalar_oracle_o3   (SAME source, -O3 -march=rv64gcv: the autovec bar)
//   - vcopy              (raw 1-load+1-store memory ceiling)
// at working sets from L1-resident to DRAM-resident, with the benchmark's own
// quantization (all scales 1.0, zero points 0) through the REAL rndnu init,
// plus bit-exactness spot checks of pf1 vs the scalar oracle on 4 producer
// configs (mult bounds, shift edges).
//
// Built with -march=rv64gcv ONLY (no zba_zbb_zbs: that build SIGILLs on this
// board - see REPORT.md), static, run on the board: taskset -c <cpu> ./sizebench
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <riscv_vector.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"

void xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1(size_t, const int8_t*, const int8_t*,
                                                int8_t*, const union xnn_qs8_mul_minmax_params*);
void xnn_qs8_vmul_minmax_rndnu_scalar_oracle(size_t, const int8_t*, const int8_t*,
                                             int8_t*, const union xnn_qs8_mul_minmax_params*);
void xnn_qs8_vmul_minmax_rndnu_scalar_oracle_o3(size_t, const int8_t*, const int8_t*,
                                                int8_t*, const union xnn_qs8_mul_minmax_params*);

// Raw memory ceiling: one vle8 + one vse8 per iteration, no arithmetic.
static void vcopy(size_t batch, const int8_t* input_a, const int8_t* input_b,
                  int8_t* output, const union xnn_qs8_mul_minmax_params* params) {
  (void) input_b; (void) params;
  do {
    size_t vl = __riscv_vsetvl_e8m2(batch); batch -= vl;
    vint8m2_t v = __riscv_vle8_v_i8m2(input_a, vl); input_a += vl;
    __riscv_vse8_v_i8m2(output, v, vl); output += vl;
  } while (batch != 0);
}

typedef void (*ukernel_fn)(size_t, const int8_t*, const int8_t*, int8_t*,
                           const union xnn_qs8_mul_minmax_params*);

static long long timed_min_cycles(ukernel_fn fn, size_t n, int8_t* a, int8_t* b, int8_t* out,
                                  const union xnn_qs8_mul_minmax_params* p, int reps) {
  struct perf_event_attr attr;
  memset(&attr, 0, sizeof attr);
  attr.type = PERF_TYPE_HARDWARE;
  attr.size = sizeof attr;
  attr.config = PERF_COUNT_HW_CPU_CYCLES;
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;
  int fd = (int) syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
  if (fd < 0) return -1;
  fn(n, a, b, out, p);  // warm icache/branch predictors once
  long long best = -1;
  for (int i = 0; i < reps; i++) {
    long long before = 0, after = 0;
    if (read(fd, &before, sizeof before) != sizeof before) break;
    fn(n, a, b, out, p);
    if (read(fd, &after, sizeof after) != sizeof after) break;
    long long delta = after - before;
    if (best < 0 || delta < best) best = delta;
  }
  close(fd);
  return best;
}

int main(void) {
  const size_t sizes[] = {16320, 130560, 1u << 20, 1u << 24};
  const size_t max_n = 1u << 24;
  int8_t* a = malloc(max_n);
  int8_t* b = malloc(max_n);
  int8_t* out = malloc(max_n);
  int8_t* ref = malloc(max_n);
  if (!a || !b || !out || !ref) { perror("malloc"); return 1; }
  uint64_t seed = 0x9e3779b97f4a7c15ull;
  for (size_t i = 0; i < max_n; i++) {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    a[i] = (int8_t) (seed >> 56);
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    b[i] = (int8_t) (seed >> 56);
  }

  // The XNNPACK benchmark's own quantization (all scales 1.0, zero points 0)
  // through the REAL rndnu init: multiplier 0x40000000, shift 0, E=31.
  const struct xnn_quantization_params q0 = {0, 1.0f};
  union xnn_qs8_mul_minmax_params p;
  xnn_init_qs8_mul_minmax_rndnu_neon_params(&p, &q0, &q0, &q0);

  printf("   N        rvv_pf1      scalar_O2   scalar_O3(autovec)       vcopy(ceiling)   (best-of-5 cycles; cyc/elem; GB/s at 1.753 GHz, 3 bytes/elem)\n");
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
    size_t n = sizes[s];
    long long c[4];
    c[0] = timed_min_cycles(xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1, n, a, b, out, &p, 5);
    c[1] = timed_min_cycles(xnn_qs8_vmul_minmax_rndnu_scalar_oracle, n, a, b, out, &p, 5);
    c[2] = timed_min_cycles(xnn_qs8_vmul_minmax_rndnu_scalar_oracle_o3, n, a, b, out, &p, 5);
    c[3] = timed_min_cycles(vcopy, n, a, b, out, &p, 5);
    printf("%8zu", n);
    for (int k = 0; k < 4; k++) {
      if (c[k] < 0) { printf("          n/a          "); continue; }
      double ns = c[k] / 1.753;               // cycles -> ns at 1.753 GHz
      double gbs = 3.0 * n / ns;              // 2 reads + 1 write per element
      printf("  %8.0f (%5.2f, %4.1f)", ns, (double) c[k] / n, gbs);
    }
    printf("\n");
  }

  // Bit-exactness spot checks vs the scalar oracle: 4 producer configs.
  struct {
    float as, bs, ys; int32_t azp, bzp, yzp;
    const char* what;
  } checks[] = {
    {1.0f, 1.0f, 1.0f, 0, 0, 0, "mult 0x40000000, shift 0, E=31"},
    {1.0f, 1.0f, 0.0078125f, 0, 0, 0, "shift -8, pre-shift 9, E=23"},
    {1.0f, 1.0f, 65536.0f, 0, 0, 0, "shift 15, E=46"},
    {0.75f, 1.25f, 0.96875f, -128, -128, 0, "gtest defaults, zps -128"},
  };
  const size_t n = 4096;
  printf("\nbit-exactness vs scalar oracle (4096 elems, X280):\n");
  int all_ok = 1;
  for (size_t k = 0; k < sizeof(checks) / sizeof(checks[0]); k++) {
    const struct xnn_quantization_params aq = {checks[k].azp, checks[k].as};
    const struct xnn_quantization_params bq = {checks[k].bzp, checks[k].bs};
    const struct xnn_quantization_params yq = {checks[k].yzp, checks[k].ys};
    union xnn_qs8_mul_minmax_params pc;
    xnn_init_qs8_mul_minmax_rndnu_neon_params(&pc, &aq, &bq, &yq);
    xnn_qs8_vmul_minmax_rndnu_scalar_oracle(n, a, b, ref, &pc);
    xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1(n, a, b, out, &pc);
    int ok = memcmp(out, ref, n) == 0;
    all_ok &= ok;
    printf("  %-34s %s\n", checks[k].what, ok ? "MATCH" : "DIFFER");
  }
  printf(all_ok ? "SPOTCHECKS_OK\n" : "SPOTCHECKS_FAIL\n");
  return all_ok ? 0 : 1;
}
