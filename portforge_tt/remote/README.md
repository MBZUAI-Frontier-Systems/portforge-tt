# portforge_tt/remote: what the x86 host runs

The controller uploads a run's frozen harness to `<work_dir>/runs/<run>/harness` and starts
`python3 -m portforge_tt.remote.runner CMD ARGS.json` there, detached (control/cluster.remote).
The runner answers `health`, `context`, `gates`, `bench`, `dev_compile`, `dev_check`, `certify`
and `log`; a pack's adapter (`ports/<pack>/evaluate.py`) may implement any command itself as
`remote_<command>` (dav1d does). Host settings (workdir, container image, board) live in
`<workdir>/cpu.yaml` on the host (`config.py`).

| module | role |
|---|---|
| `runner` | the command dispatcher above |
| `config` | the host's `cpu.yaml`; every key has a default |
| `host` | pinned sources (`data/sources.json`), the toolchain container (`docker/`), host/container paths |
| `x280` | the SiFive X280 (RVV 1.0, VLEN 512) via tt-bh-linux: probe, boot, exclusive lease, idle check, run |
| `xnnpack`, `dav1d`, `kernelsrc` | project adapters: inject a kernel into a pinned tree, build, run its tests and benchmarks |
| `regimes`, `harness` | the regime x shape matrix and `tools/pfharness`, the driver that checks and times every kernel with params from its own init |
| `bench`, `stats` | the timing protocol (lease, idle checks, paired rounds, best correct baseline per cell) and its statistics |
| `profile`, `calibrate` | static objdump profile with the calibrated cost model; the X280 microbenchmarks behind it |

The formal methods are in `portforge_tt/verify/`.

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
