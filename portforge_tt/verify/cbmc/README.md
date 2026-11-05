# verify/cbmc: formal checks of RVV machine code

Verifies the code that ships, whatever it was written in (intrinsics, `.S`, inline asm):

1. `lift.py` turns `objdump -d` of the built binary into C, one function per symbol, over
   the machine state in `rvv_model.h` (RV64GC + Zba/Zbb/Zbs subset + RVV 1.0 integer).
   Anything not modelled raises `Unsupported`: the check fails closed.
2. A harness (`harness/<project>_<function>.c`) includes `fv_runtime.h`, the lifted C and
   the project's C reference, declares the buffers as memory regions (what may be read,
   what may be written), fills inputs (symbolic under CBMC, random natively), calls both
   and compares every byte the contract fixes.
3. `run.cosim` compiles the harness natively and runs seeds (cheap check of lifter and
   model); `run.prove` runs CBMC at one VLEN and one bounded configuration. `dav1d.py`
   sweeps a family's configurations at VLEN 128, 256, 512 and 1024.

Soundness choices: VLEN is fixed per query; tail- and mask-agnostic elements are
nondeterministic (undisturbed or all ones); vill is set and vxrm is arbitrary at entry;
callee-saved registers must be restored; every load must hit a readable region and every
store a writable one; misaligned vector elements fail, misaligned scalar accesses are
reported separately ("alignment", a performance finding on the X280).

Trust base: `rvv_model.h` (validated by `validate.py` against QEMU and the board), the
lifter (validated by the same runs and by co-simulation), CBMC, and each harness.
