# SOURCES: XNNPACK RVV porting

## Upstream

- XNNPACK current: `96eaa532df4826aa02c0915bc2320dc1e8e263ff` (checked out as `src/xnnpack`)
- The revision SALTyRN pinned (kernel params match this tree): `867d5a344790802ee067be62f572c2e2722bf6fb` (`src/xnnpack-saltyrn`)
- Family lives in `src/qs8-vcvt/` (registry `qs8-vcvt.inc`, generated kernels under `gen/`, params in `src/xnnpack/qs8-vcvt.h` or `microparams.h`)

## Merged RVV PRs to imitate (acceptance precedent)

- google/XNNPACK #9879 — new RVV `f32-vmulcaddc-minmax` microkernel
- google/XNNPACK #9974 — new RVV `f16-vmulcaddc` microkernel
- google/XNNPACK #9996 — RVV `x32-transposec` optimization, 1.25x geomean
  These show the expected shape of an acceptable contribution: registry line,
  generator template where the family uses one, tests via the existing harness.

## In-repo evidence (this pack)

- `research/saltyrn.py` + `research/saltyrn_kernels.json` — the SALTyRN kernel
  survey registry: which upstream Neon kernel each optimized kernel came from,
  and the verdicts (5 wrong, 3 do not build, formal verdicts incl. counterexamples).
- `research/qs8_vadd_smoke/` — the first end-to-end smoke: differential test +
  timing harness (difftest.c with `--dump-params`), SMT verification scripts.
  The `--dump-params` pattern is what gate 2 needs for the cvt params struct.
- `research/xdiff_drivers/` — special-value differential drivers (argmaxpool,
  f32-f16-vcvt) that caught NaN bugs the other gates missed.

## SALTyRN

- Repo: https://github.com/keaganchern/SALTy-rn (verifier baked into the host
  image at a pinned commit); optimized kernels:
  https://github.com/keaganchern/optimized-rvv-kernels
- Their method: LLM Neon->RVV translation, bounded SMT equivalence (cvc5),
  optimization via Autocomp beam search, timing on FireSim/Saturn (VLEN 512).
- Known verifier limits (do not repeat): stock param edge values stop at
  shift 15 / multiplier 127; the total time budget can expire before later
  configs run; no shims for `_tu`/`_mu` policy intrinsics and segment tuples.

## Hardware

- SiFive X280 on the Blackhole card via tt-bh-linux: RVV 1.0, VLEN 512,
  4 harts, ~1.75 GHz, perf_event cycle counters (user rdcycle is trapped).
  Optional: without it, all gates except hardware timing still run (QEMU).
- SiFive's RVV XNNPACK work (17 fp32 kernels, Dec 2024) and the RISE SpeexDSP
  post are good external references for kernel style.
