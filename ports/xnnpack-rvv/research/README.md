# research/ — SALTyRN survey provenance (2026-10-18)

Artifacts from the survey that ran all 35 of SALTyRN's published optimized RVV
kernels through the gates. Findings (details in `../SOURCES.md` and the survey
report on the x86 host at `~/portforge-cpu/results/saltyrn/survey-1/`):

- 5 kernels wrong on reachable inputs (qs8-vadd-minmax, qs8-vcvt,
  qu8-vadd-minmax, f32-argmaxpool, f32-f16-vcvt); f32-dwconv-minmax does not
  match XNNPACK's packing contract.
- 2 do not build in XNNPACK (f32-f16-vcvt signature, qu8-rsum inline asm).
- SALTyRN's verifier, as published, accepted every wrong quantized kernel:
  its stock param edges stop at shift 15 / multiplier 127, and its total time
  budget can expire before later configs run.

Files: `saltyrn.py` + `saltyrn_kernels.json` (registry: which upstream Neon
kernel each optimized kernel came from + verdicts), `qs8_vadd_smoke/` (the
first end-to-end differential + SMT smoke; `--dump-params` is the params
pattern gate 2 needs), `xdiff_drivers/` (special-value drivers that caught the
NaN bugs). Scripts assume the pre-restructure layout and are kept for
provenance; the live tools are `portforge_tt.remote` + `../evaluate.py`.
