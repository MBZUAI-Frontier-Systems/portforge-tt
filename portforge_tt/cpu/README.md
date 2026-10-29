# portforge_tt.cpu: CPU kernel-porting machinery

Target machinery for porting CPU kernels (RISC-V Vector first). Library only —
no CLI; agents and evaluators import these modules. One project adapter per
upstream repo lives here (XNNPACK today, dav1d/dav2d later); everything
specific to a *port* (which kernel, what the agent wrote, campaign receipts)
lives under `ports/<pack>/`.

## Modules

| module | role |
|---|---|
| `config` | one YAML (`<workdir>/cpu.yaml`) describing machines; every key has a default, every gate degrades to QEMU-only when the board is absent |
| `host` | the x86 host: pinned sources (`setup`d under `<workdir>/src`), the toolchain container (gcc-14 native + riscv64 + aarch64 cross, QEMU, cvc5, SALTyRN verifier), path translation host<->container |
| `x280` | the optional SiFive X280 board (RVV 1.0, VLEN 512) via tt-bh-linux: probe, boot, exclusive lease, run |
| `equiv` | bounded formal equivalence of a Neon and an RVV kernel via SALTyRN's verifier + cvc5; verdicts ALL_PASSED / PARTIAL / COUNTEREXAMPLE / MISSING_INTRINSIC |
| `xdiff` | cross-ISA differential runs from a small driver: same cases into both kernels, exact per-case comparison, counterexample inputs printed |
| `xnnpack` | XNNPACK adapter: inject a kernel into a clean worktree the way an upstream PR would, build riscv64, run family gtests/benchmarks, parse results, per-shape bench comparison |
| `kernelsrc` | strips local copies of XNNPACK types from outside kernels, recording every removal; extracts one ukernel from multi-variant files |
| `regimes` | the shape x parameter-regime matrix (`kernels/<family>/regimes.json`): init *arguments*, check-only extremes, held-out sampler |
| `harness` | builds and runs `tools/pfharness`, a harness-owned driver that includes the family registry and calls every kernel with params from its own init: `list`, `dump`, `check` (bit-exact vs the reference, all regimes/batches/patterns, misaligned, guard bytes), `bench` (raw paired samples + post-timing output hashes) |
| `stats` | paired log ratios, Hodges-Lehmann, bootstrap CIs, A/A noise floor, flag/improve rules |
| `bench` | the trusted protocol: lease, idle checks before/after (the board is shared), one session per matrix, fastest *correct* upstream baseline per cell, verdicts |
| `profile` | static profile from objdump: hot loops, instruction classes, vsetvl configs, spills, named libcalls; calibrated cost model (`predict`) and `roofline` |
| `calibrate` | X280 microbenchmarks: per-op throughput/latency by SEW x LMUL, bandwidth per memory level (`data/x280-calib.json` is the 2026-10-28 table) |

`data/sources.json` pins every input repo (base image digest and cvc5 zip are
pinned in `docker/Dockerfile`).

## Machines

- x86 host (any Linux with docker/podman): builds, QEMU at VLEN 128..1024, the
  verifier. This is all you need for correctness work.
- SiFive X280 on a Blackhole card (`ssh -p 2222 debian@localhost` from the x86
  host): real-hardware timing through perf_event. **Optional** — boot with
  `x280.boot(cfg)`, lease it for every timing run. Sites without the card run
  the whole pipeline except benchmarks.

## Typical session (in the pack, e.g. ports/xnnpack-rvv)

```python
import sys; sys.path.insert(0, "ports/xnnpack-rvv")
import evaluate
from portforge_tt.cpu import config
cfg = config.load()
result = evaluate.assess(cfg, "qs8-vcvt", "kernels/qs8-vcvt/kernel.c",
                         name="xnn_qs8_vcvt_ukernel__rvv_pf1")
print(result["hints"])   # exactly what to fix next, gate by gate
```

For a whole family under the regime matrix (what the kernel orchestrator uses):

```python
tree, exe, dropped = evaluate.prepare_matrix(cfg, "qs8-vcvt", {"xnn_qs8_vcvt_ukernel__rvv_pf1": "kernels/qs8-vcvt/kernel.c"})
evaluate.check_matrix(cfg, "qs8-vcvt", exe, ["xnn_qs8_vcvt_ukernel__rvv_pf1"])     # G2
evaluate.bench_matrix(cfg, "qs8-vcvt", exe, ["xnn_qs8_vcvt_ukernel__rvv_pf1"])     # G4, on the X280
```

## Measured facts about the X280 (2026-10-28)

- Vector ALU, mul, fixed-point, widen and extend ops cost 2 cycles per LMUL (512-bit
  registers through a 256-bit datapath) at every SEW; narrowing costs 4 per LMUL.
  Latency (add 3, mul 7 cycles) is exposed at LMUL 1-2 and hidden from LMUL 4.
- `vrgather` is element-serial (68 cycles at e8m1, 516 at e8m8); reductions 29-67.
- `vsetvli` changing vtype: about 1 cycle. L1 32 B/cycle; L2 ~13 B/cycle load;
  DRAM ~0.7 B/cycle.
- Linux discards vector state on every syscall: set vtype after any syscall.
- A/A noise of the paired protocol: worst cell 0.21%, so the floor δ is 0.5%.
  Across sessions DRAM-sized cells drift a little more (one single-session flag at
  0.994 on qs8-vcvt 4M elements did not reproduce at 2R), so a flag counts only
  when a confirmation session in a new lease flags the same cell.
- The static cost model reproduces measured loops within 3-6% (qs8-vcvt pf1/pf2).
- Memory forms at e32 m1 (cycles): vle32 5, vse32 5, vlse32 16, vsse32 32, vlsseg2/4/8
  17/19/32, vssseg2/4/8 16 each, indexed ~2 per element. Strided segment *stores* move
  8 elements/cycle, twice the strided segment loads upstream transposes use.
- Identical code at different addresses timed up to 2.4% apart (x32-transposec);
  `-falign-functions=64 -falign-loops=64` brought it under 1%.
- DRAM streaming is ~0.6 B/cycle for copies: 1024x1024 x32 transposes sit at ~13
  cycles/element for every kernel (bandwidth bound).

`evaluate.assess` returns gate results with structured detail (per-test failure
messages, solver counterexamples, per-shape timings vs scalar and each upstream
RVV variant) plus markdown. Submit with `evaluate.submit` when all gates pass.
`research/` in the pack holds the SALTyRN survey that chose the first targets.
