// Differential test and timing for the XNNPACK qs8-vadd-minmax microkernel.
//
// The same deterministic case list runs on every ISA. Each kernel built for the
// current ISA is compared byte-for-byte with XNNPACK's scalar kernel in the same
// process, and the digest over all cases is printed so outputs can also be
// compared across machines (x86 scalar/AVX2 vs X280 RVV vs QEMU at other VLENs).
//
//   difftest            correctness: one JSON line per kernel, exit 1 on mismatch
//   difftest --bench    also time each kernel at a few batch sizes
//   difftest --dump-params   print the XNNPACK-initialized params of every
//                       qconfig as JSON (the verifier's concrete configs)
//
// Kernels compiled in, by target:
//   all      xnn ... scalar_u1        (the oracle)
//   x86-64   xnn ... sse41_mul32_ld32_u16, avx2_mul32_ld64_u16
//   aarch64  xnn ... neon_ld64_u16    (SALTyRN's Neon source kernel)
//   riscv64  xnn ... rvv_u2v          (upstream hand-written RVV)
//            saltyrn_qs8_vadd_rvv     (SALTyRN optimized RVV)
#define _GNU_SOURCE
#include <inttypes.h>
#include <math.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#endif

#include "src/xnnpack/microparams.h"

typedef void (*vadd_fn)(size_t, const int8_t*, const int8_t*, int8_t*,
                        const struct xnn_qs8_add_minmax_params*);

#define DECL(name) void name(size_t, const int8_t*, const int8_t*, int8_t*, \
                             const struct xnn_qs8_add_minmax_params*)
DECL(xnn_qs8_vadd_minmax_ukernel__scalar_u1);
#if defined(__x86_64__)
DECL(xnn_qs8_vadd_minmax_ukernel__sse41_mul32_ld32_u16);
DECL(xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_u16);
#elif defined(__aarch64__)
DECL(xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16);
#elif defined(__riscv)
DECL(xnn_qs8_vadd_minmax_ukernel__rvv_u2v);
DECL(saltyrn_qs8_vadd_rvv);
#endif

static bool always(void) { return true; }
#if defined(__x86_64__)
static bool has_sse41(void) { return __builtin_cpu_supports("sse4.1"); }
static bool has_avx2(void) { return __builtin_cpu_supports("avx2"); }
#endif

static const struct {
  const char* name;
  vadd_fn fn;
  bool (*supported)(void);
} kernels[] = {
  {"xnn_scalar_u1", xnn_qs8_vadd_minmax_ukernel__scalar_u1, always},
#if defined(__x86_64__)
  {"xnn_sse41_mul32_ld32_u16", xnn_qs8_vadd_minmax_ukernel__sse41_mul32_ld32_u16, has_sse41},
  {"xnn_avx2_mul32_ld64_u16", xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_u16, has_avx2},
#elif defined(__aarch64__)
  {"xnn_neon_ld64_u16", xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16, always},
#elif defined(__riscv)
  {"xnn_rvv_u2v", xnn_qs8_vadd_minmax_ukernel__rvv_u2v, always},
  {"saltyrn_rvv_opt", saltyrn_qs8_vadd_rvv, always},
#endif
};
#define NKERNELS (sizeof(kernels) / sizeof(kernels[0]))

// Copied from xnn_init_qs8_add_minmax_scalar_params (src/microparams-init.c at
// the pinned XNNPACK revision) so params are the ones XNNPACK would really pass.
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static void init_params(struct xnn_qs8_add_minmax_params* p,
                        float a_scale, int8_t a_zp, float b_scale, int8_t b_zp,
                        float out_scale, int8_t out_zp, int8_t qmin, int8_t qmax) {
  const float a_output_scale = a_scale / out_scale;
  const float b_output_scale = b_scale / out_scale;
  const float abs_a = fabsf(a_output_scale), abs_b = fabsf(b_output_scale);
  const float max_abs = abs_a > abs_b ? abs_a : abs_b;
  const int32_t max_scale_exponent = (int32_t) (f2u(max_abs) >> 23) - 127;
  const uint32_t shift = (uint32_t) (20 - max_scale_exponent);
  const int32_t abs_am = (int32_t) lrintf(u2f(f2u(abs_a) + (shift << 23)));
  const int32_t abs_bm = (int32_t) lrintf(u2f(f2u(abs_b) + (shift << 23)));
  const int32_t am = signbit(a_output_scale) ? -abs_am : abs_am;
  const int32_t bm = signbit(b_output_scale) ? -abs_bm : abs_bm;
  const int32_t rounding = INT32_C(1) << (shift - 1);
  memset(p, 0, sizeof(*p));
  p->scalar.bias = rounding - am * (int32_t) a_zp - bm * (int32_t) b_zp;
  p->scalar.a_zero_point = a_zp;
  p->scalar.b_zero_point = b_zp;
  p->scalar.a_multiplier = am;
  p->scalar.b_multiplier = bm;
  p->scalar.shift = (int32_t) shift;
  p->scalar.output_zero_point = out_zp;
  p->scalar.output_min = qmin;  // XNNPACK sets INT8_MIN/MAX; the operator narrows
  p->scalar.output_max = qmax;  // them for fused clamps, so both are swept here
}

// Quantization configurations: XNNPACK's valid range is |scale ratio| in
// [2^-10, 2^8), which gives shift in [12, 30]. Edges first, then typical.
static const struct {
  float a_scale; int8_t a_zp; float b_scale; int8_t b_zp;
  float out_scale; int8_t out_zp; int8_t qmin, qmax;
} qconfigs[] = {
  {1.0f, 0, 1.0f, 0, 1.0f, 0, INT8_MIN, INT8_MAX},              // identity-ish
  {0.5f, 7, 0.75f, -5, 1.0f, 4, INT8_MIN, INT8_MAX},
  {0.0123f, -128, 0.0456f, 127, 0.0378f, -1, INT8_MIN, INT8_MAX},
  {1.0f, 127, 1.0f, -128, 1.0f, 127, INT8_MIN, INT8_MAX},       // zp extremes
  {1.0f, -128, 1.0f, -128, 1.0f, -128, INT8_MIN, INT8_MAX},
  {0x1.0p-10f, 3, 0x1.0p-10f, -3, 1.0f, 0, INT8_MIN, INT8_MAX}, // shift = 30
  {0x1.0p-10f, 0, 0x1.0p-10f, 0, 1.0f, 127, INT8_MIN, INT8_MAX}, // shift 30, ozp 127
  {0x1.fp+7f, 1, 0x1.0p-2f, 2, 1.0f, -7, INT8_MIN, INT8_MAX},   // shift = 13
  {200.0f, 0, 200.0f, 0, 1.0f, 0, INT8_MIN, INT8_MAX},          // saturating
  {-0.25f, 10, 0.3f, -10, 0.2f, 5, INT8_MIN, INT8_MAX},         // negative scale
  {0.1f, -20, 0.2f, 30, 0.15f, 3, -100, 90},                    // fused clamp
  {0.1f, -20, 0.2f, 30, 0.15f, 3, 0, 0},                        // degenerate clamp
};
#define NQCONFIGS (sizeof(qconfigs) / sizeof(qconfigs[0]))

// Batch sizes straddle every vector width in play: Neon 16, AVX2 16/32,
// RVV e8m2 VLMAX = 2*VLEN/8 (32 at VLEN=128 ... 256 at VLEN=1024).
static const size_t batches[] = {
  1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
  255, 256, 257, 511, 512, 513, 1000, 4099,
};
#define NBATCHES (sizeof(batches) / sizeof(batches[0]))

enum { PAT_RANDOM, PAT_MIN, PAT_MAX, PAT_ALTERNATE, PAT_RAMP, NPATTERNS };
static const char* pattern_names[] = {"random", "min", "max", "alternate", "ramp"};

static uint64_t splitmix64(uint64_t* s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

static void fill(int8_t* x, size_t n, int pattern, uint64_t seed, bool second) {
  uint64_t s = seed;
  for (size_t i = 0; i < n; i++) {
    switch (pattern) {
      case PAT_RANDOM: x[i] = (int8_t) splitmix64(&s); break;
      case PAT_MIN: x[i] = second ? INT8_MAX : INT8_MIN; break;
      case PAT_MAX: x[i] = INT8_MAX; break;
      case PAT_ALTERNATE: x[i] = (i ^ second) & 1 ? INT8_MAX : INT8_MIN; break;
      default: x[i] = (int8_t) (second ? 255 - i : i); break;
    }
  }
}

// FNV-1a over output bytes; the digest of a kernel chains every case in order.
static uint64_t fnv1a(uint64_t h, const void* data, size_t n) {
  const uint8_t* p = data;
  for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001B3ull; }
  return h;
}

#define PAD 256      // > XNN_EXTRA_BYTES: kernels may read past the end
#define GUARD 64     // bytes after the output that must stay untouched
#define CANARY 0xA5

static int check_correctness(void) {
  const size_t maxn = 4099 + 3;
  int8_t* a = aligned_alloc(64, maxn + PAD);
  int8_t* b = aligned_alloc(64, maxn + PAD);
  int8_t* ref = aligned_alloc(64, maxn + PAD);
  int8_t* out = aligned_alloc(64, maxn + PAD);
  uint64_t digest[NKERNELS];
  size_t mismatches[NKERNELS], guard_hits[NKERNELS], cases = 0;
  for (size_t k = 0; k < NKERNELS; k++) { digest[k] = 0xCBF29CE484222325ull; mismatches[k] = guard_hits[k] = 0; }
  char first_fail[NKERNELS][160];
  memset(first_fail, 0, sizeof first_fail);

  for (size_t q = 0; q < NQCONFIGS; q++) {
    struct xnn_qs8_add_minmax_params params;
    init_params(&params, qconfigs[q].a_scale, qconfigs[q].a_zp, qconfigs[q].b_scale,
                qconfigs[q].b_zp, qconfigs[q].out_scale, qconfigs[q].out_zp,
                qconfigs[q].qmin, qconfigs[q].qmax);
    for (size_t bi = 0; bi < NBATCHES; bi++) {
      for (int pat = 0; pat < NPATTERNS; pat++) {
        const size_t n = batches[bi];
        const size_t misalign = (q + bi + pat) % 4;  // unaligned pointers too
        const uint64_t seed = 0x5EEDull * (q + 1) + 7919ull * bi + pat;
        memset(a, 0x11, maxn + PAD);
        memset(b, 0x22, maxn + PAD);
        fill(a + misalign, n, pat, seed, false);
        fill(b + misalign, n, pat, seed ^ 0xABCDEFull, true);
        memset(ref, CANARY, maxn + PAD);
        kernels[0].fn(n, a + misalign, b + misalign, ref + misalign, &params);
        cases++;
        for (size_t k = 0; k < NKERNELS; k++) {
          if (!kernels[k].supported()) continue;
          memset(out, CANARY, maxn + PAD);
          kernels[k].fn(n, a + misalign, b + misalign, out + misalign, &params);
          digest[k] = fnv1a(digest[k], out + misalign, n);
          size_t bad = n;
          for (size_t i = 0; i < n; i++) if (out[misalign + i] != ref[misalign + i]) { bad = i; break; }
          bool guard_ok = true;
          for (size_t i = 0; i < misalign; i++) guard_ok &= (uint8_t) out[i] == CANARY;
          for (size_t i = 0; i < GUARD; i++) guard_ok &= (uint8_t) out[misalign + n + i] == CANARY;
          if (bad != n) {
            if (mismatches[k]++ == 0)
              snprintf(first_fail[k], sizeof first_fail[k],
                       "qconfig=%zu batch=%zu pattern=%s i=%zu a=%d b=%d got=%d want=%d",
                       q, n, pattern_names[pat], bad, a[misalign + bad], b[misalign + bad],
                       out[misalign + bad], ref[misalign + bad]);
          }
          if (!guard_ok && guard_hits[k]++ == 0 && !first_fail[k][0])
            snprintf(first_fail[k], sizeof first_fail[k],
                     "qconfig=%zu batch=%zu pattern=%s wrote outside output", q, n, pattern_names[pat]);
        }
      }
    }
  }
  int failed = 0;
  for (size_t k = 0; k < NKERNELS; k++) {
    if (!kernels[k].supported()) {
      printf("{\"kernel\":\"%s\",\"status\":\"unsupported\"}\n", kernels[k].name);
      continue;
    }
    const bool ok = mismatches[k] == 0 && guard_hits[k] == 0;
    failed |= !ok;
    printf("{\"kernel\":\"%s\",\"status\":\"%s\",\"cases\":%zu,\"mismatched_cases\":%zu,"
           "\"guard_violations\":%zu,\"digest\":\"%016" PRIx64 "\",\"first_failure\":\"%s\"}\n",
           kernels[k].name, ok ? "pass" : "fail", cases, mismatches[k], guard_hits[k],
           digest[k], first_fail[k]);
  }
  free(a); free(b); free(ref); free(out);
  return failed;
}

#ifdef __linux__
static int open_cycles(void) {
  struct perf_event_attr attr;
  memset(&attr, 0, sizeof attr);
  attr.type = PERF_TYPE_HARDWARE;
  attr.size = sizeof attr;
  attr.config = PERF_COUNT_HW_CPU_CYCLES;
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;
  return (int) syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
}
#else
static int open_cycles(void) { return -1; }
#endif

static uint64_t read_cycles(int fd) {
  uint64_t c = 0;
  if (fd >= 0 && read(fd, &c, sizeof c) != sizeof c) c = 0;
  return c;
}

static double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static int cmp_double(const void* x, const void* y) {
  double a = *(const double*) x, b = *(const double*) y;
  return (a > b) - (a < b);
}

// Each sample times a block of calls lasting about 2 ms; report the median and
// minimum per call over 21 samples. Cycles come from perf_event when available.
static void bench(void) {
  static const size_t sizes[] = {64, 1024, 16384, 262144};
  int fd = open_cycles();
  struct xnn_qs8_add_minmax_params params;
  init_params(&params, 0.0123f, -128, 0.0456f, 127, 0.0378f, -1, INT8_MIN, INT8_MAX);
  for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
    const size_t n = sizes[s];
    int8_t* a = aligned_alloc(64, n + PAD);
    int8_t* b = aligned_alloc(64, n + PAD);
    int8_t* out = aligned_alloc(64, n + PAD);
    fill(a, n, PAT_RANDOM, 1, false);
    fill(b, n, PAT_RANDOM, 2, true);
    for (size_t k = 0; k < NKERNELS; k++) {
      if (!kernels[k].supported()) continue;
      size_t reps = 1;
      for (;;) {
        const double t0 = now_ns();
        for (size_t r = 0; r < reps; r++) kernels[k].fn(n, a, b, out, &params);
        if (now_ns() - t0 > 2e6 || reps > (1u << 24)) break;
        reps *= 2;
      }
      enum { SAMPLES = 21 };
      double ns[SAMPLES], cyc[SAMPLES];
      for (int i = 0; i < SAMPLES; i++) {
        const uint64_t c0 = read_cycles(fd);
        const double t0 = now_ns();
        for (size_t r = 0; r < reps; r++) kernels[k].fn(n, a, b, out, &params);
        const double t1 = now_ns();
        const uint64_t c1 = read_cycles(fd);
        ns[i] = (t1 - t0) / reps;
        cyc[i] = (double) (c1 - c0) / reps;
      }
      qsort(ns, SAMPLES, sizeof ns[0], cmp_double);
      qsort(cyc, SAMPLES, sizeof cyc[0], cmp_double);
      printf("{\"bench\":\"%s\",\"batch\":%zu,\"reps\":%zu,\"ns_median\":%.1f,\"ns_min\":%.1f,"
             "\"cycles_median\":%.0f,\"cycles_per_elem\":%.3f,\"cycles_source\":\"%s\"}\n",
             kernels[k].name, n, reps, ns[SAMPLES / 2], ns[0], cyc[SAMPLES / 2],
             cyc[SAMPLES / 2] / n, fd >= 0 ? "perf_event" : "none");
    }
    free(a); free(b); free(out);
  }
}

static void dump_params(void) {
  printf("[\n");
  for (size_t q = 0; q < NQCONFIGS; q++) {
    struct xnn_qs8_add_minmax_params p;
    init_params(&p, qconfigs[q].a_scale, qconfigs[q].a_zp, qconfigs[q].b_scale,
                qconfigs[q].b_zp, qconfigs[q].out_scale, qconfigs[q].out_zp,
                qconfigs[q].qmin, qconfigs[q].qmax);
    printf("  {\"a_zero_point\": %d, \"b_zero_point\": %d, \"bias\": %" PRId32
           ", \"a_multiplier\": %" PRId32 ", \"b_multiplier\": %" PRId32 ", \"shift\": %" PRId32
           ", \"output_zero_point\": %d, \"output_min\": %d, \"output_max\": %d}%s\n",
           p.scalar.a_zero_point, p.scalar.b_zero_point, p.scalar.bias, p.scalar.a_multiplier,
           p.scalar.b_multiplier, p.scalar.shift, p.scalar.output_zero_point,
           p.scalar.output_min, p.scalar.output_max, q + 1 < NQCONFIGS ? "," : "");
  }
  printf("]\n");
}

int main(int argc, char** argv) {
  bool do_bench = false;
  int cpu = -1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--bench")) do_bench = true;
    else if (!strcmp(argv[i], "--dump-params")) { dump_params(); return 0; }
    else if (!strcmp(argv[i], "--cpu") && i + 1 < argc) cpu = atoi(argv[++i]);
  }
  if (cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof set, &set);
  }
  const int failed = check_correctness();
  if (do_bench) bench();
  return failed;
}
