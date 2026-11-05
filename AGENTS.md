# Optimizing a CPU kernel family (guide for coding agents)

You are helping a user port or optimize a kernel family for RISC-V Vector with this harness.
Work with them step by step; the decisions marked **Ask** belong to the user. Never start a
campaign, boot the X280, spend board time or touch remote machines without their go-ahead,
and check the board is idle before timing (other agents share it). The finished result is a
family in a kernel pack (`ports/<pack>/`) and, after a campaign, a certified kernel.

Read `README.md` first, then the closest existing pack:

| Kernel shape | Closest pack |
|---|---|
| C intrinsics in a library with its own tests and init functions | `ports/xnnpack-rvv` (gtests, regime matrix from the init, cvc5 vs the Neon kernel) |
| Hand-written assembly checked by the project's own test | `ports/dav1d-rvv` (checkasm, C baseline, CBMC over the lifted machine code) |

This branch runs kernel packs only; the Tenstorrent model packs under `ports/` run from the
`tt` branch.

## 1. Choose the family and its baseline

**Ask**: which kernel family, and what counts as the baseline. The default is the fastest
correct upstream RVV variant per cell. Families with no RVV anywhere use the C reference,
labelled as such.

- Add the family to `ports/<pack>/families.json`. XNNPACK entries list the registry, the Neon
  ukernel, signature, init function and upstream RVV variants (see `evaluate.build_families`).
  Assembly entries (dav1d) list the functions, symbols, wiring and `context` (which source
  excerpts the worker sees).
- List its formal methods as `"verify": ["cbmc"]` or `["cvc5"]` (empty when none applies;
  see `portforge_tt/verify/contract.py`).
- For a function with no RVV anywhere, add `kernels/<family>/placeholder.S` (or `.c`). The
  campaign starts from it and the worker ports first.
- **Check `context`.** Every other ISA's implementation of the function must be in the
  excerpts; `tests/test_verify.py` checks the dav1d ones against a checkout.

Data-movement families (transposes) take no params: their regimes are call layouts over "WxH"
shapes (see `kernels/x32-transposec/regimes.json`). A family may build from current upstream
(`base`), add compile flags (`cflags`) and register candidates with a different gtest tile
(`overrides`).

## 2. Write the regime matrix with the user

**Ask**: the parameter regimes that matter and the shapes.

`kernels/<family>/regimes.json` lists init *arguments* (quantization, op params), not param
structs; the harness calls each kernel's own init. Include:

- the bench and test params;
- the init envelope's extremes;
- one regime per code path you know of;
- `check_extra` for correctness-only extremes;
- a `heldout` sampler that respects the init's asserts (only certification sees it);
- the `noise_floor` from an A/A session.

Schema: `portforge_tt/remote/regimes.py`.

## 3. Write the technique menu

`kernels/<family>/TECHNIQUES.json` holds:

- `spec`: the exact formula, domain and width bounds;
- `candidates`: ids from `portforge_tt/guides/RVV_TECHNIQUES.md`, or local `k:<name>` ideas, each with how it applies here and why it might win.

Mine it from every other ISA's implementation, upstream commits, earlier reports and the catalogue. Never record outcomes in it: the search tree does that, per parent and combination, so a technique that lost once stays available. The worker gets it as `TECHNIQUES.md` from the first turn, including when porting from a placeholder.

New catalogue entries go in `RVV_TECHNIQUES.md` under a `## <category>` heading as ``### `<id>`: <title>``; the controller parses those headings.

## 4. Calibrate and dry-run (once per board, then per family)

**Ask** before using the board.

- **Calibration.** `remote.calibrate.run(cfg)` on the host measures per-op costs and bandwidth (`remote/data/x280-calib.json` is the reference copy). A/A sessions (`evaluate.noise_session`) give the noise floor.
- **Dry run.** Run the trusted gates on known kernels without agents:
  ```sh
  portforge --port ports/<pack> --family F --seed-kernel upstream=path --seed-kernel mine=path --dry-run
  ```
  It runs G0 compile and the static profile, G1 tests, G2 bit-exact, and G3/G4 on the board unless `--board-minutes 0`. The ranking must match what is known about these kernels before a campaign is worth running.

## 5. Launch and supervise

**Ask**: the agent backend, hours, board minutes and whether to watch the run together.

```sh
portforge --port ports/<pack> --family F --cluster my-cluster.yaml --hours 6 --board-minutes 120 --detach
# optional: --seed-kernel name=path (baselines; the first starts the workspace), --placeholder path,
#           --optimize worst_cell --constraint "text_bytes<=1.5*baseline", --beam 4, --agent claude
```

- **Watch:** `runs/<run>/status.json`, the turn files and `runs/<run>/search/frontier.json`. `python -m portforge_tt.ranking.report runs/<run>` prints the ranked candidates and the certification.
- **Steer:** stop the controller and `--resume runs/<run>` with a `--resume-note`. Editing the workspace's STATE.md does not reach the supervisor as a trusted instruction.
- **Objective:** the default is the geomean speedup over the matrix with no cell slower than the baseline beyond the noise floor. A flagged cell is re-measured at twice the rounds in a second session, and only a flag seen in both counts. The Pareto front over (speedup, worst cell, code size) is always reported.
- **Certification:** when the worker submits (or, without a submission, when the budget ends) the controller certifies the source. That means held-out regimes under QEMU and on the X280, a second bench session that must agree, and the family's verify methods. Nothing is a result before that.

## 6. Formal verification

Each method lives in `portforge_tt/verify/<method>/` behind `verify/contract.py`: one verdict vocabulary (`ALL_PASSED`, `PARTIAL`, `COUNTEREXAMPLE`, `UNSUPPORTED`, `ERROR`), and only `ALL_PASSED` passes.

- **`cbmc`:** the built machine code is lifted (`lift.py`) into C over an RVV model (`rvv_model.h`) and proved equal to the project's C reference by CBMC on bounded sizes at every VLEN, after a native co-simulation. The model is trusted only through its validation receipts (`python -m portforge_tt.verify.cbmc.validate`: every modelled instruction against QEMU, and the X280 when allowed).
- **`cvc5`:** SALTyRN's shims compare the RVV kernel with its Neon source over params from the library's own init (`kernels/<family>/params.json`).

A harness (`verify/cbmc/harness/*.c`) states the domain and which bytes the kernel may read and write; it is reviewed once per family by the user and never edited by agents.
