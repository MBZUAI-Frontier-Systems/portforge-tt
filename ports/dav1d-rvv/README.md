# dav1d-rvv: porting dav1d's DSP kernels to RISC-V Vector

Campaign pack for dav1d's hand-written assembly kernels, validated by checkasm
(bit-exact vs the C reference) on QEMU and real RVV hardware, benchmarked on
the X280. Different repo, same trusted-gate discipline as `ports/xnnpack-rvv/`
- read `CONTRACT.md` for what "done" means and `portforge_tt/remote/dav1d.py` for
the tooling.

## Layout

```
port.json        manifest: pinned upstream, gates (checkasm + differential + bench), anti-cheat
CONTRACT.md      deliverable + gates + honesty rules (binding)
TASK.md          the ordered plan of one family campaign (seeded into the workspace)
SOURCES.md       pinned revisions, coverage survey, known hardware findings
kernels/<name>/  per-campaign workspaces (asm, drivers, REPORT.md, submissions)
```

## Repo structure notes (dav1d vs XNNPACK)

- Hand-written `.S` assembly per ISA (`src/<arch>/`), bitdepth-templated
  (`_8bpc`/`_16bpc`) - no intrinsics, so the SMT gate does not apply;
  correctness is checkasm (bit-exact) + targeted differential drivers.
- Per-family inline init headers (`src/riscv/<dsp>.h`) are the injection
  surface: one assignment line per ported function.
- Tests/benchmarks: checkasm (meson subproject, videolan/checkasm v1.2.0) -
  randomized differential tests vs the C reference plus cycle benchmarks.
  Full suite at the pinned revision: 916 tests.
- Build: meson + `tools/riscv64.cross`; `tests/checkasm` is
  `build_by_default=false` - build it explicitly; `-Dtrim_dsp=false` is
  required or checkasm has no C reference to compare against.

## Campaigns (all verified independently)

| campaign | target | mode | result |
|---|---|---|---|
| 1 | `mc_8bpc.mct` + `mc.S` | investigation + fix | upstream RVV asm uses misaligned scalar loads; kernel emulates (~290x/access), checkasm SIGBUSes; page-safe fix `mc.S-fix-v2.diff` 916/916 on X280 + QEMU. Upstream MR candidate; full report `reports/dav1d-rvv-mc-misaligned-load/` |
| 2 | `looprestoration_wiener_8bpc` | new coverage | bit-exact, **5.0-5.2x vs C**; first RVV implementation of the family |
| 3 | `looprestoration_sgr_3x3` (+5x5, mix) | new coverage | bit-exact, **3.98x / 4.35x / 3.76x vs C** (independently re-measured); depends on the wiener files as base |
| 4 | 16x16 ITX functions | optimize existing RVV | bit-exact; 16x16 geomean **1.17-1.27x** vs previous RVV asm (board noise between runs), total itx_8bpc cycles -22% |
| — | `loopfilter` port | new coverage | **honest red** (2026-10-21): bit-exactness not reached; asm + differential + next steps preserved in `kernels/loopfilter/` |

## State of RVV coverage (pinned revision)

104 riscv asm functions (6 families: ipred 37, itx 32, mc 28, filmgrain 2,
cdef 2, pal 1) against an arm64 surface of 347. `loopfilter` and
`looprestoration` have no riscv support at all. Details in SOURCES.md.
