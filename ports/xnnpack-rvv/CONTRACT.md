# CONTRACT: XNNPACK RVV microkernels

A candidate is one C file (`kernel.c`) with `<riscv_vector.h>` intrinsics implementing the family's
ukernel. The harness registers it next to its template kernel, builds XNNPACK, and owns the tests,
the regime matrix, the board and the formal check.

## Rules (the gates enforce all of them)

1. **Exact on the init's whole domain.** Bit-exact with the scalar reference for EVERY parameter value
   XNNPACK's init can produce (qs8 multipliers reach 2^21 and shifts 30; float families see NaN, ±0,
   ±inf and subnormals), every batch size including tails, every input, and every VLEN from 128 to 1024.
   Width bounds come from the init's domain, never from test values. Held-out parameter regimes are
   checked at certification.
2. **Proven dispatch only.** Parameter dispatch is allowed only on predicates you prove make each path
   exact on its whole domain; never special-case particular values. Write each predicate's proof
   sketch, and every narrowing's range argument, in NOTES.md.
3. **Interface.** Keep the template's signature, params struct and `XNN_OOB_READS` semantics: you may
   over-read within the padded tile, never over-write. No `_tu`/`_mu` intrinsics (the formal check has
   no shims for them). No libgcc calls in the hot path (XNNPACK builds RVV with `-march=rv64gcv`: no
   Zbb, so `__builtin_ctz` becomes a call).
4. **Vector work.** The hot loop must be vector code; no scalar fallback for the bulk of the work.

Transpose families (`x32-transposec`) instead: exact for every block shape (tails included), every
stride at least the block's row size (any multiple of 4 bytes), 4-byte-aligned pointers and every
VLEN; bytes between `block_height*4` and `output_stride` in each output row stay untouched; dispatch
only on the block shape, the strides and `vlenb`, never on data.

## Gates

| gate | what | where |
|---|---|---|
| G0 | compiles in XNNPACK's build; static profile | x86 host |
| G1 | the family's gtests at VLEN 128/512/1024 | QEMU |
| G2 | bit-exact check over every (non-held-out) regime of the matrix | QEMU |
| G3/G4 | the same check on the X280, then bench rounds over the matrix against the best upstream RVV kernel per cell | X280 |
| G5 | held-out regimes under QEMU and on the X280; cvc5 equivalence against the Neon kernel over params from the init | certification |

`port_evaluate(stage="check")` runs G0-G2, `stage="perf"` runs G0-G4; G5 runs after `port_submit`.

## Deliverable

`kernel.c`, `TRANSFORMS.json` (parent, applied technique ids, novel ideas) and `NOTES.md` (design,
proofs of dispatch predicates and narrowings, measured numbers). For an upstream PR the kernel is
later expressed through XNNPACK's generator templates (see SOURCES.md).

## Honesty rules

- The tests, the oracle and the gates are the trusted side; you cannot change them.
- A kernel that is correct but slower than upstream is a result; a fast kernel with a loophole is not.
