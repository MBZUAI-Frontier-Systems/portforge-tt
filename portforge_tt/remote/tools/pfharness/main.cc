// pfharness: the harness-owned driver for one XNNPACK registry (.inc).
//
// It includes the family's registry with XNN_UKERNEL redefined into a table of
// {name, function, init}, the way XNNPACK's own tests expand registries, so every
// registered kernel (upstream variants and injected candidates alike) is
// callable by name, and every kernel's params come from its *own* registered
// init function: nothing here invents parameter values.
//
// Built per family by portforge_tt/remote/harness.py with
//   -DPF_INC="src/<dir>/<family>.inc"  -DPF_DECL="src/xnnpack/<header>.h"
//   -DPF_BINARY=0|1                    (vunary/vcvt vs vbinary signature)
//   -DPF_TRANSPOSE=1                   (transposec registries: XNN_TRANSPOSE_UKERNEL)
//
// Commands (all output is JSON lines on stdout):
//   pfharness list
//   pfharness dump   REGIMES KERNEL...
//   pfharness check  REGIMES BATCHES SEED REF CAND...
//   pfharness bench  REGIMES SHAPES ROUNDS TARGET_NS SEED REF KERNEL...
// REGIMES is a file of lines "regime <id> <a_scale> <a_zp> <b_scale> <b_zp>
//   <out_scale> <out_zp> <op0> <op1> <dist> <lo> <hi>" written by regimes.py;
//   dist is full (whole integer range), uniform (floats in [lo, hi]) or
//   specials (uniform plus NaN/inf/subnormal/signed-zero values).
// BATCHES and SHAPES are comma-separated element counts.
//
// Transpose mode (PF_TRANSPOSE=1) keeps the commands but changes two inputs:
//   REGIMES lines are "tregime <id> <tile> <in_pad> <out_pad> <offset>": the
//     memory layout of a call. tile 0 is one ukernel call over the whole block
//     (XNNPACK's bench); tile T sweeps the block in TxT tiles with the full
//     matrix strides, the way XNNPACK's transpose operator calls the ukernel.
//     Row strides are width+in_pad (input) and height+out_pad (output)
//     elements; offset shifts both base pointers by that many elements.
//   SHAPES and BATCHES are comma-separated "WxH" blocks (W input columns, H rows).
// The check compares against the definition of a transpose (not a kernel) and
// verifies that row padding, the bytes before the output and a guard after it
// are untouched.
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"
#include PF_DECL

#ifndef PF_BINARY
#define PF_BINARY 0
#endif
#ifndef PF_TRANSPOSE
#define PF_TRANSPOSE 0
#endif

enum Kind { K_I8, K_U8, K_I16, K_U16, K_I32, K_U32, K_F16, K_BF16, K_F32 };
template <class T> constexpr Kind kind_of();
template <> constexpr Kind kind_of<int8_t>() { return K_I8; }
template <> constexpr Kind kind_of<uint8_t>() { return K_U8; }
template <> constexpr Kind kind_of<int16_t>() { return K_I16; }
template <> constexpr Kind kind_of<uint16_t>() { return K_U16; }
template <> constexpr Kind kind_of<int32_t>() { return K_I32; }
template <> constexpr Kind kind_of<uint32_t>() { return K_U32; }
template <> constexpr Kind kind_of<xnn_float16>() { return K_F16; }
template <> constexpr Kind kind_of<xnn_bfloat16>() { return K_BF16; }
template <> constexpr Kind kind_of<float>() { return K_F32; }

struct Kernel {
  const char* name;
  const char* arch;
  void* fn;
  void* init;
  size_t in_size, out_size;
  Kind in_kind, out_kind;
};

// Registry lines come in two shapes: (arch, ukernel, tile, vector_tile, T, params,
// init) and (arch, ukernel, tile, vector_tile, Tin, Tout, params, init). Tile
// arguments are dropped: some reference xnn_init_hardware_config().
#define XNN_QUANTIZED(T) T
#define PF_PICK(_1, _2, _3, _4, _5, _6, _7, _8, NAME, ...) NAME
#define XNN_UKERNEL(...) PF_PICK(__VA_ARGS__, PF_UK8, PF_UK7, _6, _5, _4, _3, _2, _1)(__VA_ARGS__)
#define PF_UK7(arch, uk, tile, vtile, T, P, init) PF_UK8(arch, uk, tile, vtile, T, T, P, init)
#define PF_UK8(arch, uk, tile, vtile, Tin, Tout, P, init)                                  \
  {#uk, #arch, reinterpret_cast<void*>(&uk), reinterpret_cast<void*>(init), sizeof(Tin), \
   sizeof(Tout), kind_of<Tin>(), kind_of<Tout>()},

#define XNN_TRANSPOSE_UKERNEL(arch, uk, bits, T, bw, bh) \
  {#uk, #arch, reinterpret_cast<void*>(&uk), nullptr, (bits) / 8, (bits) / 8, kind_of<T>(), kind_of<T>()},

static const Kernel kKernels[] = {
#include PF_INC
};
#undef XNN_UKERNEL
#undef XNN_TRANSPOSE_UKERNEL
#undef XNN_QUANTIZED

static const size_t kNumKernels = sizeof(kKernels) / sizeof(kKernels[0]);

typedef void (*unary_fn)(size_t, const void*, void*, const void*);
typedef void (*binary_fn)(size_t, const void*, const void*, void*, const void*);

// ---------------------------------------------------------------------------- helpers

static const Kernel* find_kernel(const char* name) {
  for (size_t i = 0; i < kNumKernels; i++)
    if (!strcmp(kKernels[i].name, name)) return &kKernels[i];
  fprintf(stderr, "pfharness: no kernel %s in %s\n", name, PF_INC);
  exit(2);
}

#if !PF_TRANSPOSE
struct Regime {
  std::string id, dist;
  xnn_quantization_params a, b, out;
  float op[2];
  double lo, hi;
};

static std::vector<Regime> load_regimes(const char* path) {
  std::vector<Regime> out;
  FILE* f = fopen(path, "r");
  if (!f) { perror(path); exit(2); }
  char id[128], dist[32];
  Regime r;
  while (fscanf(f, " regime %127s %f %d %f %d %f %d %f %f %31s %lf %lf", id, &r.a.scale, &r.a.zero_point,
                &r.b.scale, &r.b.zero_point, &r.out.scale, &r.out.zero_point, &r.op[0], &r.op[1], dist,
                &r.lo, &r.hi) == 12) {
    r.id = id;
    r.dist = dist;
    out.push_back(r);
  }
  fclose(f);
  return out;
}

#endif  // !PF_TRANSPOSE

static std::vector<size_t> parse_list(const char* text) {
  std::vector<size_t> out;
  for (const char* p = text; *p;) {
    out.push_back(strtoull(p, (char**) &p, 10));
    if (*p == ',') p++;
  }
  return out;
}

#if !PF_TRANSPOSE
// Params for one kernel under one regime, from the kernel's own init. Returns
// the byte count the init reports (0 when the kernel takes no params).
struct Params {
  alignas(64) unsigned char bytes[1024];
  size_t size;
};

static Params make_params(const Kernel& k, const Regime& r) {
  Params p;
  memset(&p, 0, sizeof p);
  if (!k.init) return p;
#if PF_BINARY
  p.size = reinterpret_cast<xnn_init_binary_params_fn>(k.init)(
      reinterpret_cast<xnn_binary_uparams*>(p.bytes), &r.a, &r.b, &r.out);
#else
  xnn_unary_params op;
  memset(&op, 0, sizeof op);
  memcpy(&op, r.op, sizeof(float) * 2);  // every member of the union is float-only
  p.size = reinterpret_cast<xnn_init_unary_uparams_fn>(k.init)(
      reinterpret_cast<xnn_unary_uparams*>(p.bytes), &op, &r.a, &r.out);
#endif
  return p;
}

static void call(const Kernel& k, size_t n, const void* a, const void* b, void* y, const Params& p) {
#if PF_BINARY
  reinterpret_cast<binary_fn>(k.fn)(n * k.in_size, a, b, y, p.bytes);
#else
  (void) b;
  reinterpret_cast<unary_fn>(k.fn)(n * k.in_size, a, y, p.bytes);
#endif
}

#endif  // !PF_TRANSPOSE

static uint64_t splitmix(uint64_t* s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

#if !PF_TRANSPOSE
static const uint32_t kFloatSpecials[] = {
    0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00001u, 0x7f800001u,
    0x00000001u, 0x807fffffu, 0x7f7fffffu, 0xff7fffffu, 0x3f800000u, 0xbf800000u};

// Input element i under a regime: patterns are 0 random, 1 type minimum, 2 type
// maximum, 3 alternating extremes, 4 ramp (every value of an 8-bit type in turn).
static void fill(void* buf, Kind kind, size_t n, const Regime& r, int pattern, uint64_t seed) {
  uint64_t s = seed;
  for (size_t i = 0; i < n; i++) {
    uint64_t x = splitmix(&s);
    switch (kind) {
      case K_I8: case K_U8: {
        uint8_t v = pattern == 0 ? (uint8_t) x : pattern == 1 ? (kind == K_I8 ? 0x80 : 0) :
                    pattern == 2 ? (kind == K_I8 ? 0x7f : 0xff) :
                    pattern == 3 ? ((i & 1) ? (kind == K_I8 ? 0x7f : 0xff) : (kind == K_I8 ? 0x80 : 0)) :
                    (uint8_t) i;
        ((uint8_t*) buf)[i] = v;
        break;
      }
      case K_I16: case K_U16: case K_F16: case K_BF16:
        ((uint16_t*) buf)[i] = pattern == 4 ? (uint16_t) i : (uint16_t) x;
        break;
      case K_I32: case K_U32:
        ((uint32_t*) buf)[i] = pattern == 4 ? (uint32_t) i : (uint32_t) x;
        break;
      case K_F32: {
        float v = (float) (r.lo + (r.hi - r.lo) * ((x >> 11) * 0x1.0p-53));
        uint32_t bits;
        memcpy(&bits, &v, 4);
        if (r.dist == "specials" && (x & 7) == 0) bits = kFloatSpecials[(x >> 3) % 13];
        if (pattern == 1) v = (float) r.lo, memcpy(&bits, &v, 4);
        if (pattern == 2) v = (float) r.hi, memcpy(&bits, &v, 4);
        ((uint32_t*) buf)[i] = bits;
        break;
      }
    }
  }
}

// Difference between two output elements: |a - b| in LSB for integers, ULP for
// floats (NaN equals NaN, any NaN vs non-NaN is "infinite").
static double element_err(Kind kind, const void* a, const void* b, size_t i) {
  switch (kind) {
    case K_I8: return fabs((double) ((const int8_t*) a)[i] - ((const int8_t*) b)[i]);
    case K_U8: return fabs((double) ((const uint8_t*) a)[i] - ((const uint8_t*) b)[i]);
    case K_I16: return fabs((double) ((const int16_t*) a)[i] - ((const int16_t*) b)[i]);
    case K_U16: case K_F16: case K_BF16:
      return fabs((double) ((const uint16_t*) a)[i] - ((const uint16_t*) b)[i]);
    case K_I32: return fabs((double) ((const int32_t*) a)[i] - ((const int32_t*) b)[i]);
    case K_U32: return fabs((double) ((const uint32_t*) a)[i] - ((const uint32_t*) b)[i]);
    case K_F32: {
      float x = ((const float*) a)[i], y = ((const float*) b)[i];
      if (std::isnan(x) || std::isnan(y)) return (std::isnan(x) && std::isnan(y)) ? 0 : INFINITY;
      int32_t ix, iy;
      memcpy(&ix, &x, 4);
      memcpy(&iy, &y, 4);
      if (ix < 0) ix = INT32_MIN - ix;  // map to a monotonic integer line
      if (iy < 0) iy = INT32_MIN - iy;
      return fabs((double) ix - (double) iy);
    }
  }
  return INFINITY;
}

#endif  // !PF_TRANSPOSE

static uint64_t fnv1a(const void* p, size_t n) {
  uint64_t h = 0xCBF29CE484222325ull;
  for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t*) p)[i]) * 0x100000001B3ull;
  return h;
}

static double now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

struct Counters {
  int fd = -1;
  bool ok() const { return fd >= 0; }
  Counters() {
    perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.type = PERF_TYPE_HARDWARE;
    a.size = sizeof a;
    a.config = PERF_COUNT_HW_CPU_CYCLES;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.read_format = PERF_FORMAT_GROUP;
    fd = (int) syscall(SYS_perf_event_open, &a, 0, -1, -1, 0);
    if (fd < 0) return;
    a.config = PERF_COUNT_HW_INSTRUCTIONS;
    if ((int) syscall(SYS_perf_event_open, &a, 0, -1, fd, 0) < 0) { close(fd); fd = -1; }
  }
  // {cycles, instructions}
  void read2(uint64_t out[2]) const {
    uint64_t buf[3] = {0, 0, 0};
    out[0] = out[1] = 0;
    if (fd >= 0 && read(fd, buf, sizeof buf) == sizeof buf) out[0] = buf[1], out[1] = buf[2];
  }
};

#define PAD 4096       // > XNN_EXTRA_BYTES: kernels may read past the end
#define GUARD 64       // output bytes past the end that must stay untouched
#define CANARY 0xA5

// ---------------------------------------------------------------------------- commands

static int cmd_list() {
  for (size_t i = 0; i < kNumKernels; i++) {
    const Kernel& k = kKernels[i];
    printf("{\"kernel\":\"%s\",\"arch\":\"%s\",\"in_bytes\":%zu,\"out_bytes\":%zu,\"init\":%s}\n",
           k.name, k.arch, k.in_size, k.out_size, k.init ? "true" : "false");
  }
  return 0;
}

#if !PF_TRANSPOSE
static int cmd_dump(int argc, char** argv) {
  std::vector<Regime> regimes = load_regimes(argv[0]);
  for (int i = 1; i < argc; i++) {
    const Kernel& k = *find_kernel(argv[i]);
    for (const Regime& r : regimes) {
      Params p = make_params(k, r);
      printf("{\"kernel\":\"%s\",\"regime\":\"%s\",\"size\":%zu,\"hex\":\"", k.name, r.id.c_str(), p.size);
      for (size_t j = 0; j < p.size; j++) printf("%02x", p.bytes[j]);
      printf("\"}\n");
    }
  }
  return 0;
}

// Bit-exact comparison of each candidate with the reference kernel, both fed
// params from their own inits, over every regime x batch x input pattern, with
// misaligned pointers and guard bytes after the output.
static int cmd_check(int argc, char** argv) {
  std::vector<Regime> regimes = load_regimes(argv[0]);
  std::vector<size_t> batches = parse_list(argv[1]);
  const uint64_t seed = strtoull(argv[2], nullptr, 10);
  const Kernel& ref = *find_kernel(argv[3]);
  size_t maxn = 0;
  for (size_t n : batches) maxn = std::max(maxn, n);
  const size_t in_bytes = maxn * ref.in_size + PAD, out_bytes = maxn * ref.out_size + PAD;
  std::vector<uint8_t> a(in_bytes), b(in_bytes), y_ref(out_bytes), y(out_bytes);
  int failed = 0;
  for (int c = 4; c < argc; c++) {
    const Kernel& k = *find_kernel(argv[c]);
    size_t cases = 0, bad_cases = 0, guard_hits = 0;
    double max_err = 0;
    char first[512] = "";
    for (const Regime& r : regimes) {
      Params pr = make_params(ref, r), pk = make_params(k, r);
      for (size_t bi = 0; bi < batches.size(); bi++) {
        const size_t n = batches[bi];
        for (int pattern = 0; pattern < 5; pattern++) {
          const size_t off = (bi + pattern) % 4;  // misaligned pointers too
          const uint64_t s = seed * 1000003u + bi * 131 + pattern;
          memset(a.data(), 0x11, in_bytes);
          memset(b.data(), 0x22, in_bytes);
          fill(a.data() + off * ref.in_size, ref.in_kind, n, r, pattern, s);
          fill(b.data() + off * ref.in_size, ref.in_kind, n, r, pattern, s ^ 0xABCDEF);
          memset(y_ref.data(), CANARY, out_bytes);
          memset(y.data(), CANARY, out_bytes);
          call(ref, n, a.data() + off * ref.in_size, b.data() + off * ref.in_size,
               y_ref.data() + off * ref.out_size, pr);
          call(k, n, a.data() + off * k.in_size, b.data() + off * k.in_size, y.data() + off * k.out_size, pk);
          cases++;
          double case_err = 0;
          size_t worst = 0;
          for (size_t i = 0; i < n; i++) {
            double e = element_err(ref.out_kind, y_ref.data() + off * ref.out_size, y.data() + off * k.out_size, i);
            if (e > case_err) case_err = e, worst = i;
          }
          bool guard_ok = true;
          for (size_t i = 0; i < off * k.out_size; i++) guard_ok &= y[i] == CANARY;
          for (size_t i = 0; i < GUARD; i++) guard_ok &= y[off * k.out_size + n * k.out_size + i] == CANARY;
          if (!guard_ok) guard_hits++;
          if (case_err > 0) {
            bad_cases++;
            if (!first[0])
              snprintf(first, sizeof first, "regime=%s batch=%zu pattern=%d index=%zu err=%g",
                       r.id.c_str(), n, pattern, worst, case_err);
          } else if (!guard_ok && !first[0]) {
            snprintf(first, sizeof first, "regime=%s batch=%zu pattern=%d wrote outside its output",
                     r.id.c_str(), n, pattern);
          }
          max_err = std::max(max_err, case_err);
        }
      }
    }
    failed |= bad_cases || guard_hits;
    printf("{\"kernel\":\"%s\",\"reference\":\"%s\",\"cases\":%zu,\"mismatched_cases\":%zu,"
           "\"guard_violations\":%zu,\"max_err\":%s,\"first_failure\":\"%s\"}\n",
           k.name, ref.name, cases, bad_cases, guard_hits,
           std::isinf(max_err) ? "\"inf\"" : std::to_string(max_err).c_str(), first);
  }
  return failed;
}

// Raw timing samples. For every cell (regime x shape) and round, the kernels run
// in a seeded random order, at a rotating buffer offset, each as one timed block
// of calls lasting about TARGET_NS. After timing, every kernel's output on the
// cell's canonical input is hashed together with the reference kernel's, so a
// fast-but-wrong kernel cannot hide in the bench.
static int cmd_bench(int argc, char** argv) {
  std::vector<Regime> regimes = load_regimes(argv[0]);
  std::vector<size_t> shapes = parse_list(argv[1]);
  const int rounds = atoi(argv[2]);
  const double target_ns = atof(argv[3]);
  uint64_t seed = strtoull(argv[4], nullptr, 10);
  const Kernel& ref = *find_kernel(argv[5]);
  std::vector<const Kernel*> ks;
  for (int i = 6; i < argc; i++) ks.push_back(find_kernel(argv[i]));
  static const size_t kOffsets[] = {0, 16, 48, 4096 + 32};
  Counters ctr;
  for (const Regime& r : regimes) {
    std::vector<Params> ps;
    for (const Kernel* k : ks) ps.push_back(make_params(*k, r));
    for (size_t n : shapes) {
      const size_t in_bytes = n * ref.in_size + PAD + 8192, out_bytes = n * ref.out_size + PAD + 8192;
      std::vector<uint8_t> a(in_bytes), b(in_bytes), y(out_bytes);
      fill(a.data(), ref.in_kind, n + PAD / ref.in_size, r, 0, seed + n);
      fill(b.data(), ref.in_kind, n + PAD / ref.in_size, r, 0, seed + n + 1);
      // Block size per kernel: enough calls for one block to last target_ns.
      std::vector<size_t> reps(ks.size(), 1);
      for (size_t j = 0; j < ks.size(); j++) {
        for (;;) {
          double t0 = now_ns();
          for (size_t q = 0; q < reps[j]; q++) call(*ks[j], n, a.data(), b.data(), y.data(), ps[j]);
          if (now_ns() - t0 >= target_ns || reps[j] >= (1u << 24)) break;
          reps[j] *= 2;
        }
      }
      std::vector<size_t> order(ks.size());
      for (int round = 0; round < rounds; round++) {
        for (size_t j = 0; j < order.size(); j++) order[j] = j;
        for (size_t j = order.size(); j > 1; j--) std::swap(order[j - 1], order[splitmix(&seed) % j]);
        const size_t off = kOffsets[round % 4];
        for (size_t j : order) {
          uint64_t c0[2], c1[2];
          ctr.read2(c0);
          double t0 = now_ns();
          for (size_t q = 0; q < reps[j]; q++)
            call(*ks[j], n, a.data() + off, b.data() + off, y.data() + off, ps[j]);
          double t1 = now_ns();
          ctr.read2(c1);
          printf("{\"regime\":\"%s\",\"shape\":%zu,\"round\":%d,\"offset\":%zu,\"slot\":%zu,\"kernel\":\"%s\","
                 "\"reps\":%zu,\"ns\":%.3f,\"cycles\":%.3f,\"instret\":%.3f}\n",
                 r.id.c_str(), n, round, off, j, ks[j]->name, reps[j], (t1 - t0) / reps[j],
                 (double) (c1[0] - c0[0]) / reps[j], (double) (c1[1] - c0[1]) / reps[j]);
        }
      }
      // correctness after timing
      Params pr = make_params(ref, r);
      std::vector<uint8_t> yr(out_bytes);
      call(ref, n, a.data(), b.data(), yr.data(), pr);
      const uint64_t href = fnv1a(yr.data(), n * ref.out_size);
      for (size_t j = 0; j < ks.size(); j++) {
        memset(y.data(), 0, out_bytes);
        call(*ks[j], n, a.data(), b.data(), y.data(), ps[j]);
        uint64_t h = fnv1a(y.data(), n * ks[j]->out_size);
        printf("{\"regime\":\"%s\",\"shape\":%zu,\"slot\":%zu,\"kernel\":\"%s\",\"hash\":\"%016" PRIx64
               "\",\"reference_hash\":\"%016" PRIx64 "\"}\n",
               r.id.c_str(), n, j, ks[j]->name, h, href);
      }
      fflush(stdout);
    }
  }
  if (!ctr.ok()) fprintf(stderr, "pfharness: no perf_event counters; cycles are 0\n");
  return 0;
}

#else  // PF_TRANSPOSE

typedef void (*transpose_fn)(const void*, void*, size_t, size_t, size_t, size_t);

struct TRegime {
  std::string id;
  size_t tile, in_pad, out_pad, offset;
};

struct Block {
  size_t w, h;
};

static std::vector<TRegime> load_regimes(const char* path) {
  std::vector<TRegime> out;
  FILE* f = fopen(path, "r");
  if (!f) { perror(path); exit(2); }
  char id[128];
  TRegime r;
  while (fscanf(f, " tregime %127s %zu %zu %zu %zu", id, &r.tile, &r.in_pad, &r.out_pad, &r.offset) == 5) {
    r.id = id;
    out.push_back(r);
  }
  fclose(f);
  return out;
}

static std::vector<Block> parse_blocks(const char* text) {
  std::vector<Block> out;
  for (const char* p = text; *p;) {
    Block b;
    b.w = strtoull(p, (char**) &p, 10);
    if (*p != 'x') { fprintf(stderr, "pfharness: shapes are WxH, got %s\n", text); exit(2); }
    b.h = strtoull(p + 1, (char**) &p, 10);
    out.push_back(b);
    if (*p == ',') p++;
  }
  return out;
}

// One layout of a W x H block: element strides and buffer sizes in bytes.
struct Layout {
  size_t es, in_ld, out_ld, in_bytes, out_bytes, off;
  Layout(const Kernel& k, const TRegime& r, Block b) {
    es = k.in_size;
    in_ld = b.w + r.in_pad;
    out_ld = b.h + r.out_pad;
    off = r.offset * es;
    in_bytes = off + in_ld * b.h * es + PAD;
    out_bytes = off + out_ld * b.w * es + GUARD;
  }
};

// The ukernel applied to a block: one call (tile 0) or XNNPACK's operator sweep
// of TxT tiles with the full-matrix strides. x and y are buffer starts; the
// block begins L.off bytes in.
static void apply(const Kernel& k, const TRegime& r, Block b, const Layout& L, const uint8_t* x, uint8_t* y) {
  transpose_fn fn = reinterpret_cast<transpose_fn>(k.fn);
  x += L.off;
  y += L.off;
  if (r.tile == 0) {
    fn(x, y, L.in_ld * L.es, L.out_ld * L.es, b.w, b.h);
    return;
  }
  for (size_t i = 0; i < b.h; i += r.tile) {
    const size_t th = std::min(r.tile, b.h - i);
    for (size_t j = 0; j < b.w; j += r.tile) {
      const size_t tw = std::min(r.tile, b.w - j);
      fn(x + (i * L.in_ld + j) * L.es, y + (j * L.out_ld + i) * L.es, L.in_ld * L.es, L.out_ld * L.es, tw, th);
    }
  }
}

// Every input element distinct and seed-dependent, so a misplaced element is
// always visible; padding gets its own pattern.
static void fill_block(uint8_t* x, const Layout& L, Block b, uint64_t seed) {
  uint64_t s = seed;
  for (size_t i = 0; i < L.in_bytes; i++) x[i] = (uint8_t) splitmix(&s);
  for (size_t r = 0; r < b.h; r++)
    for (size_t c = 0; c < b.w; c++) {
      uint64_t v = ((uint64_t) r << 32 | c) ^ (seed * 0x9E3779B97F4A7C15ull);
      memcpy(x + L.off + (r * L.in_ld + c) * L.es, &v, L.es);
    }
}

static uint64_t hash_output(const uint8_t* y, const Layout& L, Block b) {
  uint64_t h = 0xCBF29CE484222325ull;
  for (size_t c = 0; c < b.w; c++) h = h * 31 + fnv1a(y + L.off + c * L.out_ld * L.es, b.h * L.es);
  return h;
}

static int cmd_dump(int, char**) { return 0; }  // transposes take no params

static int cmd_check(int argc, char** argv) {
  std::vector<TRegime> regimes = load_regimes(argv[0]);
  std::vector<Block> blocks = parse_blocks(argv[1]);
  const uint64_t seed = strtoull(argv[2], nullptr, 10);
  const Kernel& ref = *find_kernel(argv[3]);
  int failed = 0;
  for (int c = 4; c < argc; c++) {
    const Kernel& k = *find_kernel(argv[c]);
    size_t cases = 0, bad_cases = 0, guard_hits = 0;
    char first[512] = "";
    for (const TRegime& r : regimes) {
      for (size_t bi = 0; bi < blocks.size(); bi++) {
        const Block b = blocks[bi];
        const Layout L(k, r, b);
        std::vector<uint8_t> x(L.in_bytes), y(L.out_bytes);
        fill_block(x.data(), L, b, seed * 1000003u + bi);
        memset(y.data(), CANARY, L.out_bytes);
        apply(k, r, b, L, x.data(), y.data());
        cases++;
        size_t wrong = 0, wr = 0, wc = 0;
        for (size_t col = 0; col < b.w; col++)
          for (size_t row = 0; row < b.h; row++)
            if (memcmp(x.data() + L.off + (row * L.in_ld + col) * L.es,
                       y.data() + L.off + (col * L.out_ld + row) * L.es, L.es)) {
              if (!wrong) wr = row, wc = col;
              wrong++;
            }
        bool guard_ok = true;
        for (size_t i = 0; i < L.off; i++) guard_ok &= y[i] == CANARY;
        for (size_t col = 0; col < b.w; col++)
          for (size_t i = b.h * L.es; i < L.out_ld * L.es; i++) guard_ok &= y[L.off + col * L.out_ld * L.es + i] == CANARY;
        for (size_t i = L.off + L.out_ld * b.w * L.es; i < L.out_bytes; i++) guard_ok &= y[i] == CANARY;
        if (!guard_ok) guard_hits++;
        if (wrong) {
          bad_cases++;
          if (!first[0])
            snprintf(first, sizeof first, "regime=%s block=%zux%zu wrong=%zu first at input row %zu col %zu",
                     r.id.c_str(), b.w, b.h, wrong, wr, wc);
        } else if (!guard_ok && !first[0]) {
          snprintf(first, sizeof first, "regime=%s block=%zux%zu wrote outside its output (padding or guard)",
                   r.id.c_str(), b.w, b.h);
        }
      }
    }
    failed |= bad_cases || guard_hits;
    printf("{\"kernel\":\"%s\",\"reference\":\"definition (%s for bench hashes)\",\"cases\":%zu,"
           "\"mismatched_cases\":%zu,\"guard_violations\":%zu,\"max_err\":%s,\"first_failure\":\"%s\"}\n",
           k.name, ref.name, cases, bad_cases, guard_hits, bad_cases ? "\"inf\"" : "0.000000", first);
  }
  return failed;
}

static int cmd_bench(int argc, char** argv) {
  std::vector<TRegime> regimes = load_regimes(argv[0]);
  std::vector<Block> blocks = parse_blocks(argv[1]);
  const int rounds = atoi(argv[2]);
  const double target_ns = atof(argv[3]);
  uint64_t seed = strtoull(argv[4], nullptr, 10);
  const Kernel& ref = *find_kernel(argv[5]);
  std::vector<const Kernel*> ks;
  for (int i = 6; i < argc; i++) ks.push_back(find_kernel(argv[i]));
  static const size_t kOffsets[] = {0, 16, 48, 4096 + 32};
  Counters ctr;
  for (const TRegime& r : regimes) {
    for (Block b : blocks) {
      const Layout L(ref, r, b);
      const size_t slack = 4096 + 64;
      std::vector<uint8_t> x(L.in_bytes + slack), y(L.out_bytes + slack);
      fill_block(x.data(), L, b, seed + b.w * 7919 + b.h);
      std::vector<size_t> reps(ks.size(), 1);
      for (size_t j = 0; j < ks.size(); j++) {
        for (;;) {
          double t0 = now_ns();
          for (size_t q = 0; q < reps[j]; q++) apply(*ks[j], r, b, L, x.data(), y.data());
          if (now_ns() - t0 >= target_ns || reps[j] >= (1u << 24)) break;
          reps[j] *= 2;
        }
      }
      std::vector<size_t> order(ks.size());
      for (int round = 0; round < rounds; round++) {
        for (size_t j = 0; j < order.size(); j++) order[j] = j;
        for (size_t j = order.size(); j > 1; j--) std::swap(order[j - 1], order[splitmix(&seed) % j]);
        const size_t off = kOffsets[round % 4];
        for (size_t j : order) {
          uint64_t c0[2], c1[2];
          ctr.read2(c0);
          double t0 = now_ns();
          for (size_t q = 0; q < reps[j]; q++) apply(*ks[j], r, b, L, x.data() + off, y.data() + off);
          double t1 = now_ns();
          ctr.read2(c1);
          printf("{\"regime\":\"%s\",\"shape\":\"%zux%zu\",\"elems\":%zu,\"round\":%d,\"offset\":%zu,\"slot\":%zu,"
                 "\"kernel\":\"%s\",\"reps\":%zu,\"ns\":%.3f,\"cycles\":%.3f,\"instret\":%.3f}\n",
                 r.id.c_str(), b.w, b.h, b.w * b.h, round, off, j, ks[j]->name, reps[j], (t1 - t0) / reps[j],
                 (double) (c1[0] - c0[0]) / reps[j], (double) (c1[1] - c0[1]) / reps[j]);
        }
      }
      std::vector<uint8_t> yr(L.out_bytes);
      apply(ref, r, b, L, x.data(), yr.data());
      const uint64_t href = hash_output(yr.data(), L, b);
      for (size_t j = 0; j < ks.size(); j++) {
        memset(y.data(), 0, y.size());
        apply(*ks[j], r, b, L, x.data(), y.data());
        printf("{\"regime\":\"%s\",\"shape\":\"%zux%zu\",\"slot\":%zu,\"kernel\":\"%s\",\"hash\":\"%016" PRIx64
               "\",\"reference_hash\":\"%016" PRIx64 "\"}\n",
               r.id.c_str(), b.w, b.h, j, ks[j]->name, hash_output(y.data(), L, b), href);
      }
      fflush(stdout);
    }
  }
  if (!ctr.ok()) fprintf(stderr, "pfharness: no perf_event counters; cycles are 0\n");
  return 0;
}
#endif  // PF_TRANSPOSE

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "list")) return cmd_list();
  if (argc >= 4 && !strcmp(argv[1], "dump")) return cmd_dump(argc - 2, argv + 2);
  if (argc >= 7 && !strcmp(argv[1], "check")) return cmd_check(argc - 2, argv + 2);
  if (argc >= 9 && !strcmp(argv[1], "bench")) return cmd_bench(argc - 2, argv + 2);
  fprintf(stderr, "usage: pfharness list | dump R K... | check R BATCHES SEED REF CAND... | "
                  "bench R SHAPES ROUNDS TARGET_NS SEED REF K...\n");
  return 2;
}
