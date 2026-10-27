# REPORT: mc_8bpc.mct hardware failure on the X280

Campaign 1, Part A. Status: **SOLVED** - deterministic reproduction, root cause
proven at instruction level, one-function fix prepared and validated
(checkasm 916/916 on the X280 and under QEMU at VLEN 128/256/512/1024).

## TL;DR

The 8bpc 8tap filter row helpers in `src/riscv/64/mc.S`
(`prep_filter_row_{4,6,8}tap_rvv` and `put_filter_row_{4,6,8}tap_rvv`) perform
a **scalar 8-byte `ld t3, 0(t3)` at a general (unaligned) address** - the
"right end incl. padding" load that feeds the `vslide1down` padding path.
The X280 hardware raises a **load-address-misaligned exception** (cause 4)
for these loads; the tt-bh-linux guest kernel has no misaligned-access
emulation and delivers SIGBUS (si_code = BUS_ADRALN). QEMU-user executes the
same instruction sequence without any trap (it models no misalignment
behavior for scalar loads), which is exactly why the full checkasm suite
passes 916/916 under QEMU VLEN 512 and dies on the board.

Every checkasm test whose RVV override executes one of these helpers with a
non-zero subpel offset fails with "bus error": all 8tap `mct` and `mc`
functions with mx != 0 (the `_h_` and `_hv_` variants of every filter
composition, ~90 tests). All mx = 0 variants pass because they take the
`copy` htap path, which has no such `ld`.

Note this is **broader than the campaign brief**: the brief recorded a single
`mc_8bpc.mct [FAILED]` observation; in fact the `mc_8bpc.mc` (put) tests with
mx != 0 fail the same way when run standalone (verified with
`--function=mc_8tap_regular_sharp_w8_h_8bpc`: SIGBUS at
`put_filter_row_8tap_rvv+0x8`, the same `ld t3, 0(t3)` shape). A full-suite
run aborts at the first failing family, so earlier failures were hidden in
the original observation.

## Evidence chain (in run order)

1. **Deterministic reproduction on the X280** (build `dav1d-c1-mct-rv64`,
   checkasm at the pinned dav1d rev `d7a83f6e0f9360188a38ceae0883f3d132d87b47`):
   - `checkasm --function=mct_8tap_regular_sharp* -v <seed>` for seeds 1..8:
     **12 of 24 tests failed on every seed** (all 8bpc `w{4,8,16,32,64,128}`
     x `{h,hv}` variants; all `_0_`/`_v_` variants pass), each reported as
     `FAILURE: mct_8tap_regular_sharp_*_8bpc_rvv (bus error)`.
   - The failure is data-independent (same 12 tests for every seed) and
     100% reproducible across processes and board reboots.
2. **Crash-site capture** (ptrace tracer, `kernels/mct-hardware/tracer.c` -
   checkasm installs its own handlers so LD_PRELOAD loses the race):
   `SIG 7 code=1 addr=0x... pc=0x...` with the guest mapping dumped around it.
3. **Guest-kernel corroboration** (board dmesg, twice):
   ```
   checkasm[4186]: unhandled signal 7 code 0x1 at 0x000055558a8ced88
       in checkasm[79d88,55558a855000+c5000]
   epc : 000055558a8ced88    badaddr: 00007fffd57d8af7
   status: 0000000200004420  cause: 0000000000000004   <- load address misaligned
   Code: ... 0e33 00b8 8f07 0205 (3e03) 000e              <- ld t3,0(t3)
   ```
   `cause 4` = load-address-misaligned; `badaddr` = the data address = `t3`
   = `src + chunk_w`; offset `0x79d88` =
   `prep_filter_row_8tap_rvv+0x8`, i.e. exactly
   ```
   add   t3, a6, a1        ; chunk_w + src
   vle8.v v30, (a1)
   ld    t3, 0(t3)         ; <-- trap; 8-byte load, t3 & 7 in {5,7,...}
   ```
4. **Control experiments** (board, gcc -O2 -march=rv64gcv, pinned and
   unpinned, seeds fixed): a bare misaligned scalar `ld` at stack/heap/mmap
   addresses - including page-crossing ones and ones immediately following a
   `vle8.v` - does **not** trap in a standalone process
   (`misalign_test.c`, `misalign2.c`, `misalign3.c`: 32/32 + 21/21 cases
   pass). A verbatim recompile of the upstream `mc.S` slice (chooser +
   filter_row helpers) in a standalone driver also passes 32/32. The trap
   therefore needs the checkasm execution context (process/binary layout,
   L1/cache state, physical mapping - not determinable from user space).
   What is *not* in doubt: in checkasm the trap is precise, at this `ld`,
   with cause 4, on every seed and every run, and identical misaligned
   `ld`s in the same process context would be required for the C reference
   to behave differently. The classification that matters for the port:
   **RISC-V does not guarantee misaligned scalar loads; code that issues
   them is not portable, and this board rejects them in this context.**
5. **Fix** (`mc.S-fix.diff`): replace the single misaligned `ld` in each of
   the three `filter_row_{4,6,8}tap` helpers with two aligned 8-byte loads
   plus a funnel shift (11 scalar instructions, no new registers beyond the
   temporaries the helpers already own, no branches):
   ```
   andi t0, t3, -8
   ld   t1, 0(t0)
   ld   t2, 8(t0)
   sub  t4, t3, t0        ; misalignment in bytes
   sltu t3, zero, t4      ; 1 iff misaligned
   slli t4, t4, 3
   srl  t1, t1, t4
   neg  t4, t4
   addi t4, t4, 64
   sll  t2, t2, t4        ; (shift mod 64 makes t4==0 keep W1, masked below)
   neg  t3, t3
   and  t2, t2, t3
   or   t3, t1, t2
   ```
6. **Post-fix gates** (build `dav1d-c1-mct-rv64`, worktree
   `~/portforge-cpu/build/trees/c1-mct`):
   - X280, full suite, seed 1: **checkasm: all 916 tests passed** (rc 0)
   - X280, `--function=mct_8tap_*`, seeds 2, 3, 42: rc 0 each
   - QEMU VLEN 128/256/512/1024, full suite, seed 1:
     **all 916 tests passed** at every VLEN

## Root-cause classification

Outcome (b/a) hybrid, per the campaign's taxonomy:

- It is an **upstream asm bug** in the portability sense: the RISC-V ISA
  allows implementations to trap on misaligned scalar loads; the asm issues
  one at an address that is misaligned for essentially every checkasm input
  (`src_stride = 135`, plus the -3/-2/-1 htap offsets).
- It reached upstream because **QEMU cannot catch it**: qemu-user implements
  misaligned scalar accesses unconditionally. This recalibrates this repo's
  gates: **QEMU passing proves functional semantics only; fault-behavior
  classes (misalignment, non-natural access width, PMP/access faults) need
  real hardware.** Concretely: any future RVV kernel that issues scalar
  loads/stores wider than the element alignment must be audited for
  natural alignment before it can trust a green QEMU run.
- Not a hardware erratum: the exception is architecturally correct behavior
  (cause 4, precise), deterministic, and reproducible across processes,
  seeds and (after reboots) boots. The residual puzzle - the same misaligned
  `ld` succeeds in freshly-started standalone processes on the same board -
  is documented above as an open observation; it does not affect the fix.

## Why the mx=0 variants pass

`filter_fn_chooser` with mx==0 sets function id 0 -> the `copy` htap
(`prep_copy_*`), whose row helper (`filter_row_copy_rvv`) has no padding
`ld` - it only widens bytes. The failing tests are exactly the mx!=0 ones,
which dispatch to `filter_row_{4,6,8}tap`.

## Deliverable

- `mc.S-fix.diff` (this directory): the upstream-MR-shaped fix, 3 sites,
  +17/-1 lines, applied to the disposable worktree, validated per above.
  Upstream MR shape: one patch, subject
  `riscv: mc: avoid misaligned 8-byte load in filter_row_{4,6,8}tap`
  (portable per RISC-V spec; benefits any core without native misaligned
  support; no behavior change on cores that tolerate misalignment).

## Seeds log

Board: checkasm seeds 1..8 (pre-fix, failing 12/24), seed 1 (full suite,
pre-fix: mc_8bpc.mct FAILED at ~90 tests), seeds 1 (full suite), 2, 3, 42
(family) post-fix - all green. QEMU: seeds 1 (full suite x4 VLENs). Every
checkasm invocation used `taskset -c 1` on the board via the supported
`dav1d.run_checkasm`/`x280` APIs.
