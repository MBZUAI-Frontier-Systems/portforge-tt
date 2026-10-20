#!/bin/bash
# Build + run the qs8-vmul-minmax-rndnu differential driver in the toolchain
# container: exhaustive int8 input space x 14 producer-true configs (params
# from the REAL library init), bit-exact vs the literal Neon transcription,
# under QEMU at 4 VLENs.
#
# Headers and microparams-init.c come from the port's injected worktree
# (build/trees/pf-qs8-vmul-minmax-rndnu), where the ARM-only guards around the
# family's params member and init are widened for riscv - the patch is kept as
# params-init-riscv.patch next to this script (it is also the upstream-PR
# library diff). Re-apply it to a fresh worktree before rebuilding here.
# Usage (on tt_box): bash ports/xnnpack-rvv/kernels/qs8-vmul-minmax-rndnu/difftest_build.sh
set -e
KF=/pf/repo/ports/xnnpack-rvv/kernels/qs8-vmul-minmax-rndnu
WT=/pf/build/trees/pf-qs8-vmul-minmax-rndnu
SM=/pf/repo/ports/xnnpack-rvv/research/qs8_vadd_smoke
cd /pf/build
FLAGS="-O2 -march=rv64gcv_zba_zbb_zbs -I$WT -I$WT/include -I$SM"
riscv64-linux-gnu-gcc $FLAGS -c $KF/difftest.c -o vmul-rndnu-difftest.o
riscv64-linux-gnu-gcc $FLAGS -c $KF/kernel.c -o vmul-rndnu-kernel.o
riscv64-linux-gnu-gcc $FLAGS -c $WT/src/microparams-init.c -o vmul-rndnu-init.o
riscv64-linux-gnu-gcc -static vmul-rndnu-difftest.o vmul-rndnu-kernel.o vmul-rndnu-init.o \
  -lm -o difftest-vmul-rndnu
echo LINK_OK
riscv64-linux-gnu-gcc -O2 -I$WT -I$WT/include -I$SM $KF/dump_params.c \
  $WT/src/microparams-init.c -lm -o dump-params-vmul-rndnu
qemu-riscv64 -L /usr/riscv64-linux-gnu /pf/build/dump-params-vmul-rndnu > $KF/params.json
echo PARAMS_OK
for vlen in 128 256 512 1024; do
  echo "== VLEN $vlen =="
  qemu-riscv64 -cpu rv64,vlen=$vlen,v=true,vext_spec=v1.0 /pf/build/difftest-vmul-rndnu
done
