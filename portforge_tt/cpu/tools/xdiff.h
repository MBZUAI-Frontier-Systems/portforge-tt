// Helpers for portforge-cpu xdiff drivers (force-included into kernel and driver).
#pragma once
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Special float values every float driver should sweep.
static const uint32_t xdiff_f32_specials[] = {
  0x00000000u, 0x80000000u,              // +0, -0
  0x3f800000u, 0xbf800000u,              // 1, -1
  0x00000001u, 0x807fffffu,              // smallest / largest-magnitude subnormal
  0x7f7fffffu, 0xff7fffffu,              // +-FLT_MAX
  0x7f800000u, 0xff800000u,              // +-inf
  0x7fc00000u, 0xffc00001u, 0x7f800001u, // quiet NaN, negative NaN with payload, signalling NaN
  0x38800000u, 0x477fe000u, 0x477ff000u, // fp16 edges: min normal, max finite, overflow tie
  0x33000000u, 0x33000001u,              // fp16 smallest subnormal tie / just above
};
#define XDIFF_NSPECIALS (sizeof(xdiff_f32_specials) / sizeof(xdiff_f32_specials[0]))

static inline float xdiff_u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t xdiff_f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static inline void xdiff_hex(const void* p, size_t n) {
  const uint8_t* b = (const uint8_t*) p;
  for (size_t i = 0; i < n; i++) printf("%02x", b[i]);
}

static inline uint64_t xdiff_rand(uint64_t* s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
