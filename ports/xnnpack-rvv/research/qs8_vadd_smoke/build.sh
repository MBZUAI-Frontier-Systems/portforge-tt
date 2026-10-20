#!/usr/bin/env bash
# Build the qs8-vadd difftest for x86-64, aarch64 and riscv64 (static binaries).
# Runs inside portforge-rvv-host. Usage: build.sh <src-dir> <out-dir>
#   <src-dir> holds the pinned checkouts: xnnpack/ and optimized-rvv-kernels/
set -euo pipefail
src=$1 out=$2
here=$(cd "$(dirname "$0")" && pwd)
x=$src/xnnpack/src/qs8-vadd/gen/qs8-vadd-minmax
inc=(-I"$here/../../tools" -I"$src/xnnpack" -I"$src/xnnpack/include" -I"$src/xnnpack/src")
cflags=(-std=c11 -O2 -DNDEBUG -Wall -Wno-unused-function "${inc[@]}")

build() {  # build <arch> <cc> <march flags...> -- <extra objects as "file:flags">
  local arch=$1 cc=$2; shift 2
  local dir=$out/$arch; mkdir -p "$dir"
  local objs=() march=()
  while [[ $# -gt 0 && $1 != -- ]]; do march+=("$1"); shift; done
  shift
  "$cc" "${cflags[@]}" "${march[@]}" -c "$x-scalar-u1.c" -o "$dir/scalar.o"; objs+=("$dir/scalar.o")
  for spec in "$@"; do
    local file=${spec%%|*} flags=${spec#*|}
    local obj=$dir/$(basename "${file%.c}").o
    # shellcheck disable=SC2086
    "$cc" "${cflags[@]}" "${march[@]}" $flags -c "$file" -o "$obj"; objs+=("$obj")
  done
  "$cc" "${cflags[@]}" "${march[@]}" -c "$here/difftest.c" -o "$dir/difftest.o"
  "$cc" -static -o "$dir/difftest" "$dir/difftest.o" "${objs[@]}" -lm
  echo "built $dir/difftest ($("$cc" --version | head -1))"
}

build x86_64 gcc-14 -march=x86-64 -- \
  "$x-sse41-mul32-ld32-u16.c|-msse4.1" \
  "$x-avx2-mul32-ld64-u16.c|-mavx2"

build aarch64 aarch64-linux-gnu-gcc-14 -march=armv8-a -- \
  "$x-neon-ld64-u16.c|"

build riscv64 riscv64-linux-gnu-gcc-14 -march=rv64gcv -mabi=lp64d -- \
  "$x-rvv-u2v.c|" \
  "$src/optimized-rvv-kernels/qs8-vadd-minmax.c|-include src/xnnpack/microparams.h -Dtest_rvv=saltyrn_qs8_vadd_rvv"
