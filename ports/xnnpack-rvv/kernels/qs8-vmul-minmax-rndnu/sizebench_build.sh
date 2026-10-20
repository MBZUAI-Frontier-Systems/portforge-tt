#!/bin/bash
# Build the qs8-vmul-minmax-rndnu sizebench for the X280.
# -march=rv64gcv ONLY: zba_zbb_zbs builds SIGILL on this board (REPORT.md).
# The scalar oracle is built twice: -O2 (baseline) and -O3 -march=rv64gcv
# (the autovectorization bar; same source, symbol renamed).
set -e
KF=/pf/repo/ports/xnnpack-rvv/kernels/qs8-vmul-minmax-rndnu
WT=/pf/build/trees/pf-qs8-vmul-minmax-rndnu
SM=/pf/repo/ports/xnnpack-rvv/research/qs8_vadd_smoke
cd /pf/build
FLAGS="-O2 -march=rv64gcv -I$WT -I$WT/include -I$SM"
riscv64-linux-gnu-gcc $FLAGS -c $KF/sizebench.c -o vmul-sizebench.o
riscv64-linux-gnu-gcc $FLAGS -c $KF/kernel.c -o vmul-sizebench-kernel.o
riscv64-linux-gnu-gcc $FLAGS -c $KF/scalar_oracle.c -o vmul-scalar-o2.o
riscv64-linux-gnu-gcc -O3 -march=rv64gcv -DSCALAR_ORACLE_NAME=xnn_qs8_vmul_minmax_rndnu_scalar_oracle_o3 \
  -I$WT -I$WT/include -I$SM -c $KF/scalar_oracle.c -o vmul-scalar-o3.o
riscv64-linux-gnu-gcc $FLAGS -c $WT/src/microparams-init.c -o vmul-sizebench-init.o
riscv64-linux-gnu-gcc -static vmul-sizebench.o vmul-sizebench-kernel.o vmul-scalar-o2.o \
  vmul-scalar-o3.o vmul-sizebench-init.o -lm -o sizebench-vmul-rndnu
echo SIZEBENCH_OK
# Did GCC autovectorize the O3 oracle? (vector ops in its disassembly)
echo "vector instructions in the O3 oracle object:"
riscv64-linux-gnu-objdump -d vmul-scalar-o3.o | grep -cE '^\s+[0-9a-f]+:\s+[0-9a-f]+\s+v' || true
echo "vector instructions in the O2 oracle object:"
riscv64-linux-gnu-objdump -d vmul-scalar-o2.o | grep -cE '^\s+[0-9a-f]+:\s+[0-9a-f]+\s+v' || true
echo "spill/group moves in pf1 (should be no vsd/vld between vector math):"
riscv64-linux-gnu-objdump -d vmul-sizebench-kernel.o | awk '/^[0-9a-f]+ <xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1>:/{f=1} f' | grep -cE 'vsd|vld' || true
