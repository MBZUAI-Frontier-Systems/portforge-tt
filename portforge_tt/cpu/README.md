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
| `kernelsrc` | strips local copies of XNNPACK types from outside kernels, recording every removal |

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

`evaluate.assess` returns gate results with structured detail (per-test failure
messages, solver counterexamples, per-shape timings vs scalar and each upstream
RVV variant) plus markdown. Submit with `evaluate.submit` when all gates pass.
`research/` in the pack holds the SALTyRN survey that chose the first targets.
