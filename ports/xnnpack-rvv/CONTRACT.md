# CONTRACT: XNNPACK RVV kernel ports

Read this fully before writing code. It defines what "done" means. The gates are
mechanical: either the tool reports a pass or it does not, and you can run every
one of them yourself before submitting.

## Deliverable

One RVV microkernel per campaign, delivered as:

1. `kernels/<family>/kernel.c` — the ukernel, function name
   `xnn_<family>_ukernel__rvv_<yoursuffix>`, pure C with `<riscv_vector.h>`
   intrinsics (no inline asm for the v1 packs), self-contained apart from
   XNNPACK headers it genuinely needs.
2. A registration line for the family's `.inc` registry (the tools clone the
   template's line; you check tiles/params still make sense).
3. `kernels/<family>/REPORT.md` — what you tried, what the gates said, and the
   final numbers. This is the document a reviewer (human or agent) reads first.

For an actual upstream PR the kernel must additionally be expressed through
XNNPACK's Python generator templates where the family uses them (see the merged
PRs in SOURCES.md); the campaign REPORT says exactly what is left to do for that.

## Interface rules

- Keep the template ukernel's signature, tile parameters, and params struct.
  The registry line is cloned from the template (usually the Neon variant);
  if your kernel needs different tiles (e.g. vector-length tiles), say so in
  REPORT.md and register from the closest RVV template instead.
- `XNN_OOB_READS` families may over-read within the padded tile; they may never
  over-write. If you rely on OOB reads, say where.
- Params come to you through XNNPACK's init functions. Assume every value those
  functions can produce will occur. qs8 multipliers reach 2^21 and shifts 30;
  float families produce NaN/±0/inf inputs. "The tests never do that" is not an
  argument; several published kernels broke exactly this way.

## Gates (all must pass; run them yourself)

Order matters: cheap and specific first, expensive and general last.

1. **Build + XNNPACK tests** — `evaluate.test(cfg, kernel, ...)`: your kernel
   registered next to its template, the family's gtest binary under QEMU at
   VLEN 128, 256, 512, 1024, and on the X280 when available. Zero failures.
   The gtest compares against XNNPACK's scalar kernel in-process.
2. **Bounded formal equivalence** — `evaluate.verify(cfg, kernel, ...)`: cvc5,
   Neon source vs your kernel, `configs` mode with parameters from XNNPACK's
   init code (`--dump-params` pattern), `max_batch` capped so every config is
   reached, budget >= 600 s. Verdict must be `ALL_PASSED`. `PARTIAL` means the
   sweep never finished: raise the budget or lower `max_batch`, never submit it.
   `COUNTEREXAMPLE` is gold: replay it via `evaluate.xdiff`/a driver, fix, retry.
   `MISSING_INTRINSIC` means the verifier lacks a shim for an intrinsic you used;
   either pick a shimmed intrinsic or report it and justify the gap.
3. **Benchmark** — `evaluate.bench(cfg, kernel, ...)`: XNNPACK's benchmark on
   the X280 (or note FireSim/QEMU-only). Compare per shape against (a) the
   scalar ukernel, (b) every existing RVV variant of the family. A pass needs:
   clearly faster than scalar; not slower than the best RVV variant unless
   REPORT.md argues why upstream still wants the kernel (e.g. it removes the
   only non-RVV fallback at some VLEN).
4. **Review pass** — reread your kernel as if you were the XNNPACK maintainer:
   naming, comments explaining the loop structure and the tail policy, no dead
   variants, no magic numbers without derivation, no shape/value special-casing.

Without an X280, gate 3 runs under QEMU-timing only for smoke purposes and the
campaign REPORT must say "no hardware numbers"; QEMU timing is never evidence
of speed.

## Honesty rules

- The evaluator and the oracle are not yours to modify. Never edit files under
  `portforge_tt/` to make a gate pass; if a tool looks wrong, say so in
  REPORT.md and work around it.
- Report failures verbatim. A fixed kernel keeps its history: gate receipts are
  appended to REPORT.md, never rewritten.
- If you cannot meet a gate, deliver the best state with REPORT.md saying
  exactly which gate fails and why. A correct kernel at 0.9x is a result; a
  fast kernel with a loophole is not.
