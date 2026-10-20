// sizebench: cache-residency sweep for qs8-vlrelu on the X280. Times
// scalar_u1 / rvv_u1v / rvv_u2v / rvv_pf1 and a raw vcopy ceiling at working
// sets from L1/L2-resident to DRAM-resident, at the benchmark's own params
// (slope 0.5, in {zp 1, scale 0.5}, out {zp -1, scale 0.7}: multipliers
// 183/91 - the general two-sided regime), plus a bit-exactness spot-check of
// every variant against the scalar kernel on that config and on the
// multiplier-32768 corner.
//
// Built in the toolchain container (cross riscv64, -static), run on the X280:
//   taskset -c 1 ./sizebench-vlrelu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <riscv_vector.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <stdint.h>

struct xnn_qs8_lrelu_params {
  struct {
    int32_t input_zero_point;
    int32_t positive_multiplier;
    int32_t negative_multiplier;
    int32_t output_zero_point;
  } scalar;
};

void xnn_qs8_vlrelu_ukernel__scalar_select_u1(size_t, const int8_t*, int8_t*, const struct xnn_qs8_lrelu_params*);
void xnn_qs8_vlrelu_ukernel__rvv_u1v(size_t, const int8_t*, int8_t*, const struct xnn_qs8_lrelu_params*);
void xnn_qs8_vlrelu_ukernel__rvv_u2v(size_t, const int8_t*, int8_t*, const struct xnn_qs8_lrelu_params*);
void xnn_qs8_vlrelu_ukernel__rvv_pf1(size_t, const int8_t*, int8_t*, const struct xnn_qs8_lrelu_params*);

// Raw memory ceiling: one vle8 + one vse8 per iteration, no arithmetic.
static void vcopy(size_t batch, const int8_t* input, int8_t* output,
                  const struct xnn_qs8_lrelu_params* params) {
  (void) params;
  do {
    size_t vl = __riscv_vsetvl_e8m2(batch); batch -= vl;
    vint8m2_t v = __riscv_vle8_v_i8m2(input, vl); input += vl;
    __riscv_vse8_v_i8m2(output, v, vl); output += vl;
  } while (batch != 0);
}

typedef void (*ukernel_fn)(size_t, const int8_t*, int8_t*, const struct xnn_qs8_lrelu_params*);

static long long timed_min_cycles(ukernel_fn fn, size_t n, int8_t* in, int8_t* out,
                                  struct xnn_qs8_lrelu_params* p, int reps) {
  struct perf_event_attr attr;
  memset(&attr, 0, sizeof attr);
  attr.type = PERF_TYPE_HARDWARE;
  attr.size = sizeof attr;
  attr.config = PERF_COUNT_HW_CPU_CYCLES;
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;
  int fd = (int) syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
  if (fd < 0) return -1;
  fn(n, in, out, p);  // warm icache/branch predictors once
  long long best = -1;
  for (int i = 0; i < reps; i++) {
    long long before = 0, after = 0;
    if (read(fd, &before, sizeof before) != sizeof before) break;
    fn(n, in, out, p);
    if (read(fd, &after, sizeof after) != sizeof after) break;
    long long delta = after - before;
    if (best < 0 || delta < best) best = delta;
  }
  close(fd);
  return best;
}

int main(void) {
  const size_t sizes[] = {16320, 130560, 1u << 20, 1u << 24};
  const size_t max_n = sizes[sizeof(sizes) / sizeof(sizes[0]) - 1];
  int8_t* in = malloc(max_n);
  int8_t* out = malloc(max_n);
  if (!in || !out) { perror("malloc"); return 1; }
  uint64_t seed = 0x9e3779b97f4a7c15ull;
  for (size_t i = 0; i < max_n; i++) {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    in[i] = (int8_t)(seed >> 56);
  }

  // The benchmark's own quantization and slope: multipliers 183/91.
  struct xnn_qs8_lrelu_params p = {{1, 183, 91, -1}};
  printf("   N      scalar_u1      rvv_u1v      rvv_u2v     rvv_pf1    vcopy(ceiling)   (best-of-5 cycles; cyc/elem; GB/s at 1.753 GHz)\n");
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
    size_t n = sizes[s];
    long long c[5];
    c[0] = timed_min_cycles(xnn_qs8_vlrelu_ukernel__scalar_select_u1, n, in, out, &p, 5);
    c[1] = timed_min_cycles(xnn_qs8_vlrelu_ukernel__rvv_u1v, n, in, out, &p, 5);
    c[2] = timed_min_cycles(xnn_qs8_vlrelu_ukernel__rvv_u2v, n, in, out, &p, 5);
    c[3] = timed_min_cycles(xnn_qs8_vlrelu_ukernel__rvv_pf1, n, in, out, &p, 5);
    c[4] = timed_min_cycles(vcopy, n, in, out, &p, 5);
    printf("%8zu", n);
    for (int k = 0; k < 5; k++) {
      if (c[k] < 0) { printf("      n/a          "); continue; }
      double ns = c[k] / 1.753;               // cycles -> ns at 1.753 GHz
      double gbs = 2.0 * n / ns;              // in+out bytes per ns == GB/s
      printf("  %8.0f (%5.2f, %4.1f)", ns, (double) c[k] / n, gbs);
    }
    printf("\n");
  }

  // Bit-exactness spot-checks vs the scalar kernel: the benchmark config and
  // the m=32768 corner (init's inclusive bound) on non-trivial input.
  const size_t n = 4096;
  int8_t* ref = malloc(n);
  struct xnn_qs8_lrelu_params big = {{7, 32768, -32735, -11}};
  xnn_qs8_vlrelu_ukernel__scalar_select_u1(n, in, ref, &big);
  const char* names[4] = {"pf1", "u1v", "u2v", "vcopy"};
  ukernel_fn fns[4] = {xnn_qs8_vlrelu_ukernel__rvv_pf1,
                       xnn_qs8_vlrelu_ukernel__rvv_u1v,
                       xnn_qs8_vlrelu_ukernel__rvv_u2v, vcopy};
  printf("\nmult corner 32768/-32735 vs scalar: ");
  for (int k = 0; k < 3; k++) {
    fns[k](n, in, out, &big);
    printf("%s %s  ", names[k], memcmp(out, ref, n) == 0 ? "MATCH" : "DIFFER");
  }
  xnn_qs8_vlrelu_ukernel__scalar_select_u1(n, in, ref, &p);
  printf("\nbench config 183/91 vs scalar:      ");
  for (int k = 0; k < 3; k++) {
    fns[k](n, in, out, &p);
    printf("%s %s  ", names[k], memcmp(out, ref, n) == 0 ? "MATCH" : "DIFFER");
  }
  printf("\n");
  return 0;
}
