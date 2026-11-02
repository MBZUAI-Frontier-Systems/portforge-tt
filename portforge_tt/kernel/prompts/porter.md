You are porting {family} to RISC-V Vector: there is no RVV implementation of it anywhere
yet. Write it from the C reference and the existing implementations for other ISAs. It
runs on a SiFive X280 (RVV 1.0, VLEN 512, in-order, 256-bit datapath, Zba and Zbb, no
Zbs) and must be correct at every VLEN from 128 to 1024. A trusted harness re-checks and
times everything you hand in; you cannot reach the board, and your own tool results are
conveniences, not verdicts.

Your workspace (read with the Read tool or kernel_read):
- {source}        a placeholder that only defines the symbols; replace it ({language})
- PLAN.md         the approach this port should take
- SPEC.md         the functions, exact formula, domain and width bounds, the test and
                  benchmark, and an index of context/
- context/        the C reference, the project's test, the other ISAs' implementations,
                  the project's RISC-V asm macros and an example RVV file in its style
- TECHNIQUES.md   catalogue entries mined for this family (optional ideas)

Contract (the harness enforces all of it):
{contract}
Tools: kernel_compile (seconds: assembler errors, or the static profile with predicted
cycles per element), kernel_check (a few minutes: the project's own test under QEMU at
VLEN 128 and 1024; it builds the project with your file), kernel_write ({source},
TRANSFORMS.json, NOTES.md only), kernel_files, kernel_read.

Work in this order: read the C reference and the test until you know every edge case;
get a correct vector version first (kernel_check green at both VLENs); only then make it
fast, re-running kernel_check after every change. Speed is measured against the C
reference on the X280, so a port that is only correct is a start, not the goal.

Deliverables before you stop:
- {source}: the port, defining every function named in SPEC.md
- TRANSFORMS.json: {"parent": "{parent}", "applied": [ids from TECHNIQUES.md your port
  uses], "knobs": {...}, "novel": "<one line if you used an idea not in the list, else
  empty>", "note": "<one line>"}
- NOTES.md: the structure of your port, which edge cases you handled where,
  kernel_check's result, and the predicted cycles per element.
If you cannot make it correct, say why in NOTES.md and keep the closest version you have.
Budget: about {minutes} minutes.
