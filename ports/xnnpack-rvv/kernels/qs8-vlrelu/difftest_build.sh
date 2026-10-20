#!/bin/bash
# Build + run the qs8-vlrelu differential driver in the toolchain container.
# Usage (on tt_box): bash ports/xnnpack-rvv/kernels/qs8-vlrelu/difftest_build.sh
set -e
SM=/pf/repo/ports/xnnpack-rvv/research/qs8_vadd_smoke
KF=/pf/repo/ports/xnnpack-rvv/kernels/qs8-vlrelu
SRC=/pf/src/xnnpack-saltyrn
cd /pf/build
FLAGS="-O2 -march=rv64gcv_zba_zbb_zbs -I$SRC -I$SRC/include -I$SM"
riscv64-linux-gnu-gcc $FLAGS -c $KF/difftest.c -o vlrelu-difftest.o
riscv64-linux-gnu-gcc $FLAGS -c $KF/kernel.c -o vlrelu-kernel.o
riscv64-linux-gnu-gcc $FLAGS -c $SRC/src/microparams-init.c -o vlrelu-init.o
for g in scalar-select-u1 rvv-u1v rvv-u2v; do
  riscv64-linux-gnu-gcc $FLAGS -c $SRC/src/qs8-vlrelu/gen/qs8-vlrelu-$g.c -o vlrelu-$g.o
done
riscv64-linux-gnu-gcc -static vlrelu-difftest.o vlrelu-kernel.o vlrelu-init.o \
  vlrelu-scalar-select-u1.o vlrelu-rvv-u1v.o vlrelu-rvv-u2v.o -lm -o difftest-vlrelu
echo LINK_OK
for vlen in 128 256 512 1024; do
  echo "== VLEN $vlen =="
  qemu-riscv64 -cpu rv64,vlen=$vlen,v=true,vext_spec=v1.0 /pf/build/difftest-vlrelu
done
