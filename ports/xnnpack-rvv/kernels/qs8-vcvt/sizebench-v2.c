// sizebench-v2: round-2 de-risk bench for qs8-vcvt on the X280.
//
// Differences from sizebench.c (round 1):
// - runs at the REAL XNNPACK vunary benchmark quantization. bench/vunary.cc
//   specializes int8 to input {zp 1, scale 0.5} / output {zp -1, scale 0.7},
//   so xnn_init_qs8_cvt_scalar_params yields multiplier lrintf(256*0.5/0.7)
//   = 183, NOT 256 as the round-1 REPORT section 4 said. multiplier 183 is
//   odd: rvv_u4v and rvv_pf1 both run the full 5-op chain there.
// - adds rvv_pf2 (round-2 kernel) and prints cycles/element + GB/s.
// - sweeps the benchmark shape and the DRAM shape, plus a correctness pass:
//   every params.json config (the gate-2 set) and the gtest-equivalent param
//   sets, full int8 edge values + pseudo-random fill, pf2 vs scalar_u4.
//
// Built in the toolchain container (cross riscv64, -static), run on the X280:
//   taskset -c 1 ./sizebench-v2
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <riscv_vector.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <stdint.h>

struct xnn_qs8_cvt_params {
  struct {
    int16_t input_zero_point;
    int32_t multiplier;
    int16_t output_zero_point;
  } scalar;
};

void xnn_qs8_vcvt_ukernel__scalar_u4(size_t, const int8_t*, int8_t*, const struct xnn_qs8_cvt_params*);
void xnn_qs8_vcvt_ukernel__rvv_u4v(size_t, const int8_t*, int8_t*, const struct xnn_qs8_cvt_params*);
void xnn_qs8_vcvt_ukernel__rvv_pf1(size_t, const int8_t*, int8_t*, const struct xnn_qs8_cvt_params*);
void xnn_qs8_vcvt_ukernel__rvv_pf2(size_t, const int8_t*, int8_t*, const struct xnn_qs8_cvt_params*);

// Raw memory ceiling: one vle8 + one vse8 per iteration, no arithmetic.
static void vcopy(size_t batch, const int8_t* input, int8_t* output,
                  const struct xnn_qs8_cvt_params* params) {
  (void) params;
  do {
    size_t vl = __riscv_vsetvl_e8m4(batch); batch -= vl;
    vint8m4_t v = __riscv_vle8_v_i8m4(input, vl); input += vl;
    __riscv_vse8_v_i8m4(output, v, vl); output += vl;
  } while (batch != 0);
}

typedef void (*ukernel_fn)(size_t, const int8_t*, int8_t*, const struct xnn_qs8_cvt_params*);

static long long timed_min_cycles(ukernel_fn fn, size_t n, int8_t* in, int8_t* out,
                                  struct xnn_qs8_cvt_params* p, int reps) {
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

static int check_all(size_t n, int8_t* in, int8_t* out, int8_t* ref,
                     struct xnn_qs8_cvt_params* p) {
  memset(out, 0x5a, n);
  xnn_qs8_vcvt_ukernel__scalar_u4(n, in, ref, p);
  xnn_qs8_vcvt_ukernel__rvv_pf2(n, in, out, p);
  return memcmp(out, ref, n) == 0;
}

int main(void) {
  const size_t sizes[] = {16320, 130560, 1u << 20, 1u << 24};
  const size_t max_n = sizes[sizeof(sizes) / sizeof(sizes[0]) - 1];
  int8_t* in = malloc(max_n);
  int8_t* out = malloc(max_n);
  int8_t* ref = malloc(max_n);
  if (!in || !out || !ref) { perror("malloc"); return 1; }
  uint64_t seed = 0x9e3779b97f4a7c15ull;
  for (size_t i = 0; i < max_n; i++) {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    in[i] = (int8_t)(seed >> 56);
  }
  // Pin the int8 edges (the generator above hits them with ~zero chance).
  const size_t edges[] = {0, 1, 2, 3, 126, 127, 128, 254, max_n - 255,
                          max_n - 129, max_n - 128, max_n - 127, max_n - 2, max_n - 1};
  for (size_t k = 0; k < sizeof(edges) / sizeof(edges[0]); k++) in[edges[k]] = (int8_t) (int) (k * 17 - 103);

  // The REAL vunary benchmark params (bench/vunary.cc int8 specialization).
  struct xnn_qs8_cvt_params bench = {{1, 183, -1}};
  // multiplier 256 (round-1 REPORT section 4's assumed bench params).
  struct xnn_qs8_cvt_params m256 = {{5, 256, -3}};

  printf("   N     params    scalar_u4     rvv_u4v      rvv_pf1      rvv_pf2    vcopy  (best-of-5 cyc/elem; GB/s @1.753GHz)\n");
  for (int variant = 0; variant < 2; variant++) {
    struct xnn_qs8_cvt_params* p = variant ? &m256 : &bench;
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
      size_t n = sizes[s];
      long long c[5];
      c[0] = timed_min_cycles(xnn_qs8_vcvt_ukernel__scalar_u4, n, in, out, p, 5);
      c[1] = timed_min_cycles(xnn_qs8_vcvt_ukernel__rvv_u4v, n, in, out, p, 5);
      c[2] = timed_min_cycles(xnn_qs8_vcvt_ukernel__rvv_pf1, n, in, out, p, 5);
      c[3] = timed_min_cycles(xnn_qs8_vcvt_ukernel__rvv_pf2, n, in, out, p, 5);
      c[4] = timed_min_cycles(vcopy, n, in, out, p, 5);
      printf("%8zu %s%3d/%+4d", n, variant ? "m=256 " : "m=183 ", p->scalar.multiplier % 256,
             p->scalar.input_zero_point - p->scalar.output_zero_point);
      for (int k = 0; k < 5; k++) {
        if (c[k] < 0) { printf("      n/a       "); continue; }
        double ns = c[k] / 1.753;
        double gbs = 2.0 * n / ns;
        printf("  %8.0f(%5.2f,%4.1f)", ns, (double) c[k] / n, gbs);
      }
      if (c[1] > 0 && c[3] > 0)
        printf("  pf2/u4v=%.3f", (double) c[1] / (double) c[3]);
      printf("\n");
    }
  }

  // Correctness: every gate-2 config (params.json) plus the gtest-equivalent
  // param sets, at a size that crosses the e8m4 tile boundary many times and
  // the +1 tail, against the scalar oracle.
  struct xnn_qs8_cvt_params cfgs[] = {
    {{0, 256, 0}}, {{0, 1024, 0}}, {{5, 4096, 3}}, {{0, 16384, 0}},
    {{0, 32768, 0}}, {{-5, 32765, 7}}, {{0, 64, 0}}, {{0, 4, 0}},
    {{0, 1, 0}}, {{9, 1, -9}}, {{0, 384, 0}}, {{-3, 396, 2}},
    {{-128, 69, 127}}, {{-128, 256, -128}}, {{127, 256, 127}},
    {{-128, 256, 127}}, {{100, 512, -100}},
    // gtest-equivalent: scales {1,1}, zero-point walks; output/input scale walks.
    {{2, 256, 0}}, {{5, 256, 0}}, {{8, 256, 0}},
    {{0, 256, 2}}, {{0, 256, 5}}, {{0, 256, 8}},
    {{0, 1024, 0}}, {{0, 4096, 0}}, {{0, 16384, 0}},
    {{0, 64, 0}}, {{0, 16, 0}}, {{0, 4, 0}},
    {{127, 256, -128}}, {{128 - 256, 183, 127}}, {{1, 183, -1}},
  };
  const size_t n = 130561;
  int fails = 0;
  for (size_t k = 0; k < sizeof(cfgs) / sizeof(cfgs[0]); k++) {
    if (!check_all(n, in, out, ref, &cfgs[k])) {
      printf("MISMATCH: izp=%d m=%d ozp=%d\n", cfgs[k].scalar.input_zero_point,
             cfgs[k].scalar.multiplier, cfgs[k].scalar.output_zero_point);
      fails++;
    }
  }
  printf("correctness: %zu configs, %d mismatches (pf2 vs scalar_u4)\n",
         sizeof(cfgs) / sizeof(cfgs[0]), fails);

  // multiplier 32768: upstream RVV truncates int32 -> int16 (-32768) and
  // flips every sign; scalar and pf2 must agree, u4v must differ.
  struct xnn_qs8_cvt_params big = {{7, 32768, -11}};
  xnn_qs8_vcvt_ukernel__scalar_u4(4096, in, ref, &big);
  xnn_qs8_vcvt_ukernel__rvv_pf2(4096, in, out, &big);
  printf("mult=32768: pf2 vs scalar %s; ", memcmp(out, ref, 4096) == 0 ? "MATCH" : "DIFFER (!!)");
  xnn_qs8_vcvt_ukernel__rvv_u4v(4096, in, out, &big);
  printf("u4v vs scalar %s (upstream truncation)\n",
         memcmp(out, ref, 4096) == 0 ? "MATCH" : "DIFFER");
  return fails != 0;
}
