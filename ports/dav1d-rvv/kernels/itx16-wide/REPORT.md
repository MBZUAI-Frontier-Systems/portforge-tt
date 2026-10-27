# REPORT: itx_8bpc 16x16 VL=16 rework (campaign 4, branch `cpu`)

**TL;DR** — The weakest hot spot in dav1d's existing RVV assembly is the
entire 16x16 inverse-transform block: every 16x16 variant runs at
1.5-2.8x vs C while the *same transforms* at 8x16/16x8 run 10-12x. Root
cause: the driver runs all four 1-D kernel invocations at `vsetivli
zero, 8` (8 active lanes of a 512-bit machine) with two row-halves and a
per-half stack round trip. This campaign rewrites the 16x16 path at
vl=16 for VLEN>=256 (dispatcher falls back to the verbatim upstream
driver below that), with a restructured sparse path. On the X280 the
16x16 family improves by **1.268x geomean** (17 of 25 bench functions
>= 1.25x, max 1.62x), total 16x16 RVV cycles drop 22%, with zero
correctness regressions: checkasm 916/916 under QEMU at VLEN
128/256/512/1024 (seeds 1/2/42) and natively on the X280, plus a
~90k-case differential driver green at all VLENs.

## 1. Measured survey (choose with data)

X280, `checkasm --bench --duration=100000 --json --test=<fam> 1`
(pinned dav1d rev `d7a83f6e0f9360188a38ceae0883f3d132d87b47` + the
campaign-1 `mc.S` bus-error fix so board runs survive; full per-function
JSON in `receipts/bench-*.full`, parsed table in
`receipts/survey-rows.json`):

| family      | rvv fns | median vs-C | min | max |
|-------------|---------|-------------|-----|-----|
| mc_8bpc     | 164     | 7.91x       | 0.79x | 43.9x |
| itx_8bpc    | 226     | 7.03x       | 0.94x | 12.8x |
| ipred_8bpc  | 71      | 4.35x       | 0.21x | 35.6x |

The mc big-watt functions (8tap w64/w128) already sit at 9-14x; mc's weak
spots (w_mask_4xx_w4 at 0.8x, w2 paths at ~1x) are low-cycle tails.
The 16x16 itx block is the thin spot *with* real decode weight: every
variant (including `dct_dct`, the most common real-world type) sits at
1.5-2.8x vs C, ~6000 cycles per call at the bench point, while the same
transform families at 8x16/16x8 run 10-12x. ipred's pal_pred_w4 (0.21x)
is a rare-mode tail.

**Target: the twelve `inv_txfm_add_*_16x16_8bpc` functions.**

Root cause (from `src/riscv/64/itx.S`): `inv_txfm_add_16x16_rvv` splits
the block into two row halves and runs the 1-D `e16_x16` kernels four
times per block, all at `vsetivli zero, 8` - 8 active lanes of a
512-bit machine, 4 kernel invocations, plus per-half stack round trips.

## 2. The change (upstream-MR shape)

One file: `src/riscv/64/itx.S` (`c4-itx16-wide.diff`, 266 insertions /
2 deletions against the pinned rev; no meson, no `riscv/itx.h` changes -
the entry-point assignments are unchanged).

1. Upstream `inv_txfm_add_16x16_rvv` renamed to
   `inv_txfm_add_16x16_narrow_rvv`, body verbatim: it remains the driver
   for VLEN < 256 (16 e16 lanes do not fit an m1 group there).
2. New wide driver `inv_txfm_add_16x16_wide_rvv` (VLEN >= 256, selected
   by `vlenb` in each entry function): the whole block lives in v0-v15
   at vl=16/e16/m1 - column j in v_j for the horizontal pass, row j
   after the transpose through the same 512-byte stack buffer - so the
   1-D kernels run twice per block instead of four times. The kernels
   are untouched: they are vtype-agnostic (elementwise, no `vsetvli`
   inside) and inherit vl=16.
3. New helpers `inv_txfm_horz{_identity,}_16_rvv` and
   `inv_txfm_add_vert_16_rvv`, mirroring the upstream per-half helpers
   at twice the lanes.
4. Sparse path restructured: for `eob < ` the upstream threshold (36, or
   8 for identity-first types) the horizontal pass runs at **vl=8 on the
   top half only** - exactly like upstream, since C excludes coef rows
   8..15 there (`last_nonzero_col < 8`) and the kernels mix across
   registers, so a vl=16 pass would spend half of every widened op on
   lanes that would be discarded again. The stack bottom half is zeroed
   (one m8 store) and the vertical pass stays at vl=16.

## 3. Gates (run order, all on the final binary)

1. **checkasm correctness** (seeds logged):
   `qemu:{128,256,512,1024}` seed 1: **916/916** each;
   `qemu:512` seeds 2 and 42: **916/916**;
   **X280 native seed 1: 916/916**. (`receipts/gate1-final2.txt`;
   earlier iterations: `receipts/vlen-results4.txt` for the 16x16
   subset.) The narrow path itself is bit-identical upstream code, also
   green in every run.
2. **Differential driver** (`itx16_diff.c`): 12 types x 26-point eob
   sweep (0..255 incl. both sparse thresholds) x 4 coefficient patterns
   (checkasm's ftx generator + small/sparse mixes) x 4 strides
   (16/17/32/64) x 3 reps, asm vs the C reference via
   `dav1d_itx_dsp_init_8bpc`, dst and coefficient buffers compared.
   **14976 cases x {VLEN 128/256/512/1024 x seeds 1/7/12345} = 0
   failures** (`receipts/gate2-final.txt`, `gate2-relink.txt`).
3. **X280 benchmark** (seed 1, 100 ms/function, `--json`):
   `receipts/bench-itx_8bpc-before-seed1.full` (pinned tree + mc.S fix)
   vs `receipts/bench-itx_8bpc-after-seed1.full`; table in
   `receipts/cmp-bench-final2.txt`:

   | metric | value |
   |---|---|
   | 16x16 family geomean (rvv after / rvv before) | **1.268x** |
   | 16x16 functions >= 1.25x faster | 17 of 25 (max 1.62x) |
   | 16x16 total rvv cycles | 146020 -> 113598 (0.778) |
   | slowest 16x16 function | dct_dct_0 0.99x (DC-shortcut path, untouched: noise) |
   | whole itx_8bpc bench geomean | 1.034x (226 functions) |

   Selected rows (cycles, nop-subtracted medians):

   | function | before | after | speedup |
   |---|---|---|---|
   | inv_txfm_add_16x16_adst_adst_2     | 6663.6 | 4111.7 | **1.62x** |
   | inv_txfm_add_16x16_adst_flipadst_2 | 6952.9 | 4582.3 | **1.52x** |
   | inv_txfm_add_16x16_dct_adst_2      | 6904.9 | 4787.1 | **1.44x** |
   | inv_txfm_add_16x16_flipadst_adst_2 | 6936.6 | 4911.5 | **1.41x** |
   | inv_txfm_add_16x16_identity_identity_2 | 5917.2 | 4272.0 | **1.39x** |
   | inv_txfm_add_16x16_identity_dct_2  | 6470.7 | 4791.6 | **1.35x** |
   | inv_txfm_add_16x16_adst_adst_1     | 6177.2 | 4617.8 | **1.34x** |
   | inv_txfm_add_16x16_dct_dct_2       | 6174.7 | 4980.7 | 1.24x |
   | inv_txfm_add_16x16_dct_dct_1       | 4997.9 | 4900.2 | 1.02x |
   | inv_txfm_add_16x16_dct_dct_0 (DC)  | 179.6  | 180.7  | 0.99x |

   Run-to-run variance on the shared board is ~±10% per function; the
   after2 intermediate build measured geomean 1.17x in a busier window
   with the same code path for the full-eob cases.

## 4. Round-by-round ledger (every measurement kept)

* **R1 - SIGILL, 12/25.** `vwaddu.wv` at (e8,m1,vl=16) reads its wide
  operand as an (e16,m2) group; an odd register base (v1 -> group
  v1,v2) is an illegal misaligned group. Found by qemu `-d in_asm`
  (last translated insn) + isolated op tests. Fix: run the dst epilog
  at (e8,mf2) where every operand is a single register.
* **R2 - value mismatches, 24/25.** The sparse path zeroed the *input*
  lanes 8..15; the kernels mix across registers, so transform(zeros) is
  not zero in the bottom *output* rows - C zero-fills tmp rows beyond
  `last_nonzero_col` after the horz pass. Fix: zero the *output* lanes
  (post-kernel) in the sparse branch; garbage inputs then only reach
  discarded lanes.
* **R3 - dst row 7 written twice, row 15 never.** The second 8-row
  chunk's stores started at the row-7 pointer. Found by a DC-only probe
  (exactly 16 diffs, one zero row). Fix: advance the pointer once.
* **R4 - vnclipu does not clamp negatives.** `vnclipu.wi` treats the
  source as unsigned bits: a negative sum (0xFC..) saturates to **255**,
  not 0 (isolated test: src=-500 -> 255). The upstream `vmax.vx v, v,
  zero` before the clip is mandatory; my first epilog had dropped it as
  "redundant". Restored; this was the last correctness bug (the
  checkasm dump's scattered `00` vs `ff` pairs).
* **R5 - the narrow driver ran my helpers at every VLEN.** The rewired
  `la` lines in `def_fn_16x16` fed the *wide* helper pointers to the
  upstream narrow driver, whose stack layout differs - 12/25 failures
  even at VLEN=128 where the wide path never executes. Fix: vlenb
  branch inside `def_fn_16x16` with per-driver helper wiring.
* **R6 - attempted and rejected:** routing sparse cases through
  upstream's `vert_8x16` pair (my sparse stack is row-major 16-wide,
  not upstream's interleaved half-block layout; measured worse and
  reverted); scratch-register `vwaddu` forms; processing sparse blocks
  at vl=16 with input-lane zeroing.

Debug probes kept in this directory: `kcheck.c` (1-D kernel vs the C
`inv_dct_1d` per lane, vl sweep), `idtest.c` (identity vsmul/vsra/vaadd
chain vs C over all int16), `mini2.c`/`mini3.c` (isolated vwaddu /
vnclipu at both LMULs), `replica.c` (the vert epilog replica that
exposed the vnclipu semantics), `itx16_diff.c` (gate 2 driver),
`stage.c`+`wrap.S` (per-stage harness).

## 5. Honest negatives / physics

* The survey premise "vl=8 wastes 75% of the datapath" does **not**
  convert into speed on the X280: vector cost is lane-proportional
  (a 16-lane e16 op costs ~2x an 8-lane op), so halving the kernel
  invocation count is roughly compute-neutral. The measured wins come
  from halved instruction/issue count, halved driver and stack
  overhead, and the restructured sparse path - hence 1.24x on
  full-eob-heavy variants and ~1.0x where the sparse path dominates.
* `dct_dct_1` (mixed eob) measured 1.02x; `dct_dct_0` (DC-only
  shortcut, untouched) 0.99x = noise.
* Pre-existing upstream domain notes found while building the driver
  (out of scope, not touched): for `eob=0` with non-DCT_DCT types the
  upstream asm differs from C by +-1 (checkasm never generates that:
  `subsh = !!txtp`); with raw +-32768 coefficients (outside checkasm's
  ftx range) C vs asm diverge on intermediate clipping order. Both are
  invisible to checkasm's generator and unaffected by this MR.
* QEMU vs hardware: every gate ran on both; the group-alignment SIGILL
  and the vnclipu semantics behave identically on the X280 (Gate 1
  X280 green on the final binary).

## 6. Reproduction

```sh
# box: tt_box, repo ~/portforge-cpu/repo, workdir ~/portforge-cpu
cd ~/portforge-cpu/repo
python3 - <<'EOF'
from portforge_tt.cpu import config, dav1d
from pathlib import Path
import subprocess
cfg = config.load()
tree = dav1d.tree(cfg, "c4-opt")
fix = Path.home()/"portforge-cpu/repo/ports/dav1d-rvv/kernels/mct-hardware/mc.S-fix.diff"
subprocess.run(["patch","-p1","--quiet"], cwd=tree, stdin=open(fix), check=True)
subprocess.run(["python3","/home/abror/portforge-cpu/c4/apply_itx16_wide.py",str(tree)], check=True)
dav1d.build(cfg, tree, "c4-opt")
EOF
python3 - <<'EOF'
from portforge_tt.cpu import config, dav1d
cfg = config.load()
b = cfg.build / "dav1d-c4-opt-rv64"
for vlen in (128, 256, 512, 1024):
    dav1d.run_checkasm(cfg, b, where=f"qemu:{vlen}", args=["1"])
dav1d.run_checkasm(cfg, b, where="x280", args=["1"])
dav1d.run_checkasm(cfg, b, where="x280",
                   args=["--bench","--duration=100000","--json","--test=itx_8bpc","1"])
EOF
```

Artifacts on the box: worktree `~/portforge-cpu/build/trees/c4-opt`,
build `~/portforge-cpu/build/dav1d-c4-opt-rv64`, benches/logs/driver in
`~/portforge-cpu/c4/`.
