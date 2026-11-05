# xnnpack-rvv: porting XNNPACK microkernels to RISC-V Vector

Campaign pack for kernel ports from ARM Neon to RVV, gated by XNNPACK's own
tests, bounded formal equivalence (cvc5), and X280 hardware benchmarks.
Deliverables target upstream XNNPACK PRs. See `CONTRACT.md` for what "done"
means, `TASK.md` for the plan a campaign's worker follows, and `portforge_tt/guides/RVV_GUIDE.md`
for engineering tactics and known traps.

## Layout

```
port.json        machine-readable manifest: repos, hardware, gates, anti-cheat rules
CONTRACT.md      deliverable + gates + honesty rules (binding)
TASK.md          the ordered plan of one family campaign (seeded into the workspace)
SOURCES.md       upstream revisions, merged PR precedents, in-repo evidence
evaluate.py      the gate adapter the host's runner calls (prepare/check/bench_matrix, verify); assess/submit
                 remain from the manual campaigns
kernels/<family>/   per-kernel campaign workspaces (kernel.c, params.json, REPORT.md)
research/        SALTyRN survey provenance: registry, bug evidence, the first smoke
```

## Current campaigns

- `kernels/qs8-vcvt/` — **DONE 2026-10-19** (`xnn_qs8_vcvt_ukernel__rvv_pf1`,
  sha `83b4aab0…`): gates 1-2 PASS/ALL_PASSED (17 producer-true configs,
  incl. multiplier 32768), X280 28x vs scalar, tied with upstream `rvv_u4v`
  (0.998-1.000). Side finding, solver-confirmed: all three upstream RVV
  variants truncate the int32 multiplier to int16 and sign-flip every output
  at multiplier = 32768 (the init's inclusive bound); gtests never reach it.
  See `kernels/qs8-vcvt/REPORT.md` and the submission receipt.
- `kernels/qs8-vlrelu/` — **DONE 2026-10-19**
  (`xnn_qs8_vlrelu_ukernel__rvv_pf1`, sha `d6d0fdb7…`): clean-room port.
  Gates 1-2 PASS/ALL_PASSED (33 producer-true configs from the real lrelu
  init, incl. negative multipliers and the 32768 corner on both sides),
  X280 2.597x geomean vs upstream `rvv_u2v` (28.6-29.6x vs scalar), tied
  with the third-party optimized-rvv kernel at every measured size while
  fixing its int16 multiplier truncation. First variant of the family at
  the e8m4/i16m8 width (the upstream template's i32 bias vector caps it at
  u2v). See `kernels/qs8-vlrelu/REPORT.md` and the submission receipt.
- `kernels/qs8-vmul-minmax-rndnu/` — **DONE 2026-10-20** (campaign #3,
  first true new-coverage port: the family had no RVV variant upstream at
  all, Neon only; clean-room from spec). `xnn_qs8_vmul_minmax_rndnu_ukernel__rvv_pf1`,
  sha `a3f7a651…`: gates 1-2 PASS (gate 1 = 15/15 family gtests at QEMU
  VLEN 128-1024 + X280; gate 2 = exhaustive differential, all 256x256
  int8 pairs x 14 producer-true configs from the real init, byte-exact vs
  a literal-Neon oracle at 4 VLENs + X280 native — SMT verify unavailable
  for this stem, documented). X280 29.4x vs scalar (1.38 cyc/el
  cache-resident), GCC's autovectorizer fails on the loop (the honest bar
  is scalar), at the bandwidth wall at 16M. Upstream-PR delta beyond the
  kernel: the .inc row, a 3-line ARM-guard widening for the family's
  params/init (`params-init-riscv.patch`), and wiring the gtest target
  upstream never registered. See `kernels/qs8-vmul-minmax-rndnu/REPORT.md`.

## Running the gates

Everything runs on the x86 host (`ssh tt_box`, repo `~/portforge-cpu/repo`).
Python API only:

```python
from portforge_tt.remote import config
import sys; sys.path.insert(0, "ports/xnnpack-rvv"); import evaluate
cfg = config.load()
evaluate.assess(cfg, "qs8-vcvt", "kernels/qs8-vcvt/kernel.c", name="xnn_qs8_vcvt_ukernel__rvv_pf1")
```

`research/` scripts are provenance (the 2026-10-18 survey that found five wrong
published kernels); they are kept runnable but are not part of the gates.
