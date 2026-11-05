# CONTRACT: dav1d RVV kernels

dav1d's contract is strict: the assembly is **bit-exact with the C reference, no tolerance**. A
candidate is one assembly file (`kernel.S`) that implements every function named in SPEC.md. The
harness owns everything else: the init wiring that points dav1d's function table at your symbols,
the C side, meson, checkasm and the board.

## Rules (the gates enforce all of them)

1. **Bit-exact everywhere.** checkasm (dav1d's own test: random inputs, sizes and edge flags against
   the C reference) passes under QEMU at VLEN 128, 256, 512 and 1024 AND on the X280 itself. Held-out
   checkasm seeds and the formal check run at certification.
2. **Complete and self-contained.** Implement every function in SPEC.md completely in your file. No
   calls or jumps to anything outside it (no C fallback, no libc, no other dav1d asm): the gate rejects
   any call, tail call or jump to a symbol you do not define.
3. **dav1d's conventions.** Use the project's asm macros (`function`/`endfunc` from `src/riscv/asm.S`,
   see context/riscv_asm.S and the example in context/) and the RISC-V psABI (callee-saved s0-s11 and
   fs0-fs11; 16-byte stack alignment). The X280 executes rv64gcv with Zba and Zbb, not Zbs.
4. **Vector work.** The hot loops must be vector code; a scalar port is rejected. Never read or write
   outside the buffers the C reference touches (checkasm guards the destination).
5. **Tails by `vl`.** Handle widths with `vsetvli` on the remaining count, never a scalar epilogue,
   unless the C reference's semantics force it; then prove the boundary cases in NOTES.md.

## Gates

| gate | what | where |
|---|---|---|
| G0 | builds; defines every symbol, references nothing outside the file; static profile | x86 host |
| G1 | checkasm for the family at VLEN 128/256/512/1024 | QEMU |
| G2 | 8 more checkasm seeds at VLEN 128 and 1024 | QEMU |
| G3/G4 | checkasm on the X280, then `checkasm --bench` rounds pairing your kernel with the C reference in the same process | X280 |
| G5 | held-out seeds at every VLEN and on the X280; the family's verify methods (CBMC over your lifted machine code against the C reference) | certification |

`port_evaluate(stage="check")` runs G0-G2, `stage="perf"` runs G0-G4; G5 runs after `port_submit`.

## Deliverable

`kernel.S`, `TRANSFORMS.json` (parent, applied technique ids, novel ideas) and `NOTES.md`: the
derivation from the C reference, the lane layout and register allocation, how each edge case and
tail is handled, and every narrowing's range argument. A reviewer reads NOTES.md first.

## Honesty rules

- The C reference, checkasm and the gates are the trusted side; you cannot change them.
- A kernel that is correct but slow is a result; a fast kernel with a loophole is not.
