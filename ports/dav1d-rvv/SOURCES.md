# SOURCES: dav1d RVV porting

## Upstream

- dav1d pinned revision `d7a83f6e0f9360188a38ceae0883f3d132d87b47` (master, 2026-10-17), checked out as `src/dav1d`.
- checkasm is a meson subproject: `subprojects/checkasm.wrap` -> videolan/checkasm `v1.2.0` (auto-fetched on first configure; needs network once).
- Key layout: `src/<dsp>_tmpl.c` (bitdepth-templated C reference, the spec) + `src/<dsp>.h` (DSP struct) + `src/<arch>/` (per-ISA: `.h` inline init + `.S` asm). RISC-V init entry points are called from the tmpl files (`mc_dsp_init_riscv` etc.).
- RISC-V asm conventions: `src/riscv/asm.S` macros; symbols are unprefixed (`function blend_8bpc_rvv, export=1, ext="v,zbb"`); existing code requires Zbb; bitdepth suffix `_8bpc`/`_16bpc`.

## Coverage at the pinned revision (2026-10-20 survey)

- RISC-V: **104 exported asm functions** across 6 families (ipred 37, itx 32, mc 28, filmgrain 2, cdef 2, pal 1 - plus 2 in cpu.S). ARM64 reference surface: **347 functions**. RVV covers roughly 30% of the arm64 asm surface.
- Families with **no riscv support at all**: `loopfilter` (3 struct pointers x 2 bitdepths), `looprestoration` (wiener + sgr). Both run pure C on RISC-V; both are hot per-block-edge/per-pixel paths.
- Intra-family gaps: mc (arm64 ~150 functions vs riscv 28), itx and ipred similar ratios - every covered family still has unported functions.
- checkasm at HEAD has 916 tests; a full run under QEMU VLEN 512 passes 916/916.

## Known hardware findings (this pack, 2026-10-19/21)

- `mc_8bpc.mct` **fails checkasm on the X280** (real RVV hardware) while
  passing under QEMU - root-caused 2026-10-21: the 8bpc mc row helpers do a
  **misaligned scalar `ld`**, which the kernel emulates in software (~290x
  slower per access on this board); checkasm v1.2.0 sets
  `PR_SET_UNALIGN(SIGBUS)` so its runs fail instead of silently running
  ~290x slower. On emulation cores upstream's small-block RVV mc is up to
  ~4x slower than plain C. Fix validated (`mc.S-fix-v2.diff`, page-safe
  bracket loads): 916/916 on X280 + QEMU with the prctl active. Full report:
  `reports/dav1d-rvv-mc-misaligned-load/`. Gate calibration: QEMU green
  proves semantics only - fault/perf-behavior classes need hardware.
- The X280 guest Linux panics every few hours (register dump in x280-boot.log); recovery is `x280.boot(cfg)`. Also: binaries built with `zba_zbb_zbs` SIGILL on the board - use plain `rv64gcv` for board builds (QEMU builds can keep the extras).
- checkasm's timing source on this kernel is linux-perf ("Timing source: linux (perf)"), so `--bench` works natively on the X280.

## Prior art

- RISC-V Summit Europe 2024 (Nathan Egge): RISE case study "Adding RVV 1.0 to dav1d" - the initial riscv asm and checkasm integration came from that effort.
- The loongarch port (`src/loongarch/`) is the most recent fresh-ISA addition; its incremental structure (which functions first, how the init file grows) is the closest model for a from-scratch riscv family port.
