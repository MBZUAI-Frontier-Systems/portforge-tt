# Adding a model to portforge-tt (guide for coding agents)

You are helping a user port a new model to Tenstorrent with this harness. Work
with them step by step; the decisions marked **Ask** belong to the user. Never
run a campaign, touch remote machines, or spend GPU/TT time without their
go-ahead. The finished result is a new port pack in `ports/<name>/`.

Read `README.md` first, then the closest existing pack:

| Model shape | Closest pack |
|---|---|
| Encoder, float outputs | `ports/esm2` (logits), `ports/chronos2` (quantile forecasts) |
| Encoder-decoder or autoregressive tokens | `ports/nllb` (translation), `ports/parakeet` (speech) |

## 1. Choose the model and pin it

**Ask**: the Hugging Face repo and license, which checkpoint sizes to deliver, and
the envelope (batch sizes, input lengths, output lengths, languages or modes).

- Copy `ports/template/` to `ports/<name>/`.
- In `models.json`, set `repo_id`, an immutable `revision` (40-character commit),
  the weight files and their SHA-256 and byte sizes (from the Hugging Face LFS
  API, not from memory), and the `config.json` fields the port relies on under
  `architecture`.
- Check it: `python -m portforge_tt.models <key> /tmp/ckpt --download --registry ports/<name>/models.json`.

## 2. Choose the benchmark with the user

**Ask**: which public data represents real use, its license, and which edge
cases matter (empty or constant inputs, extreme values, shortest and longest
lengths, masking).

- Build `corpus.json` (schema in `ports/template/BENCHMARK.md`): real items from
  the chosen data plus seeded synthetic edge cases, and a `selection` for
  `smoke` (1 case), `bringup` (about 8) and `full` (held out).
- Record provenance, licenses, generators and seeds in `BENCHMARK.md`.
- Compute its identity and pin it in both `port.json` and `evaluate.py`:
  `python -c "import json; from portforge_tt.cluster import identity; print(identity(json.load(open('ports/<name>/corpus.json'))))"`.
- Keep `full` cases out of development; agents never see them.

**Ask**: the gates. Earlier packs use max-row NRMSE ≤ 0.04 against the FP32 oracle,
≤ 0.01 between batch and single-row runs, bit-identical repeated calls, exact
tokens for decoders, and a task metric on `full` (for example ≤ 2% worse WER).

## 3. Write the evaluator

Implement the four functions in `ports/<name>/evaluate.py`:

- `make_suite(corpus, stage, model)` builds the three suites and returns
  `(spec, labels)`. Every `compare_to` must name a case of the same stage;
  `portforge_eval.check_suite` enforces this and the launcher rejects bad suites.
- `reference(args)` runs the official implementation in FP32 on the reference
  host and writes public inputs and private oracle outputs. Disable TF32 for
  matmuls and cuDNN convolutions.
- `candidate(args)` is generic; fill in `forward_case()` to call the backend
  method named in `CONTRACT.md`.
- `assess(args)` compares actual and oracle outputs and writes the verdict.

Add `ports/<name>/tests/test_evaluate.py` (see an existing pack) and run it with
`python -m unittest discover -s ports/<name>/tests`.

## 4. Write the agent documents

- `CONTRACT.md`: the backend interface, envelope, gates, and deliverable.
- `TASK.md`: the ordered plan; keep the template's order.
- `SOURCES.md`: model card, papers, reference code, closest tt-metal ports.
- `port.json`: title, task, delivery models, precisions, corpus hash, and the
  reference host's packages and import checks.

Run `python -m pytest tests/test_ports.py -q`: it loads
every pack and checks hashes, suites, tool names and site-specific leaks.

## 5. Set up the machines with the user

**Ask**: which TT host and which reference host (GPU preferred) to use, SSH users and
keys, and which cards the campaign may lease.

- TT host: a built TT-Metal checkout, its commit, a Python environment with
  `ttnn`, `torch`, `numpy`, `pytest`, `bwrap`, and the checkpoint downloaded with
  `python -m portforge_tt.models --download`. List the cards with `ls /dev/tenstorrent` or `tt-smi`.
- Reference host: copy `port.json` and run `bash setup_reference.sh port.json`.
- Fill a copy of `cluster.yaml`; keep it out of version control.
- Reference smoke on the reference host, once, before any campaign:
  `PORTFORGE_REFERENCE_EXECUTION=direct python evaluate.py reference --stage smoke --corpus corpus.json --out /tmp/ref-smoke`
  (with `evaluate.py`, `portforge_eval.py`, `models.py` and `models.json` copied beside it).

## 6. Launch and supervise

**Ask**: the agent backend (`--agent opencode`, `claude` or `codex`), budget hours, and
whether to watch the run together.

```sh
portforge --port ports/<name> --cluster my-cluster.yaml --hours 6 --detach
```

While it runs, read `runs/<run>/status.json` and the turn files. To steer the
agents, stop the controller and `--resume` with a `--resume-note`; editing the
workspace's STATE.md does not reach the supervisor as a trusted instruction. If
you share a failing held-out case with the agents, record that its later result
is no longer blind.

# Optimizing a CPU kernel family (RVV; `ports/xnnpack-rvv` is the reference pack)

Kernel campaigns use their own orchestrator (`python -m portforge_tt.kernel`), not the
model loop above. The controller runs the agents; every trusted gate runs on the x86
host (`tt_box`: toolchain image, QEMU, XNNPACK builds) and the SiFive X280 on the
Blackhole card. Read `portforge_tt/cpu/README.md` and `portforge_tt/guides/RVV_TECHNIQUES.md`
first. As above, decisions marked **Ask** belong to the user; never start a campaign,
boot the X280 or spend board time without their go-ahead, and check the board is idle
before timing (other agents share it).

## 1. Choose the family and its baseline

**Ask**: which kernel family, and what counts as the baseline (default: the fastest
correct upstream RVV variant per cell; families without one use autovectorized scalar,
labelled as such).

- Add the family to `ports/<pack>/families.json` (`evaluate.build_families`): its
  registry, the Neon ukernel it ports from, signature, init function, upstream RVV
  variants and whether the formal gate has a prepped Neon source.

Assembly projects (`ports/dav1d-rvv`) use the same orchestrator through their adapter's
`remote_<command>` functions: a candidate is one `.S` file, the adapter owns the wiring
(init header, C init call, meson entry) in a pinned tree, G1/G2 are the project's checkasm
under QEMU at VLEN 128-1024, the bench pairs each candidate with the C reference inside
`checkasm --bench` on the X280, and certification adds held-out seeds and the formal check
(`portforge_tt/fv`). For a function with no RVV anywhere, start from the family's
`placeholder.S` (`--placeholder path`): a porting stage of independent ports, each told a
different approach, runs until one passes G0-G2, then the search optimizes from the ports.

Data-movement families (transposes) take no params: their regimes are call layouts
(one call, XNNPACK's 32x32 operator sweep, padded views) over "WxH" shapes; see
`kernels/x32-transposec/regimes.json`. A family may build from current upstream
(`base`), add compile flags (`cflags`, e.g. code alignment when identical kernels time
apart) and register candidates with a different gtest tile (`overrides`).

## 2. Write the regime matrix with the user

**Ask**: the parameter regimes that matter and the shapes. A kernel's speed and
correctness depend on its parameters, so `kernels/<family>/regimes.json` lists init
*arguments* (quantization, op params), not param structs: the harness calls each
kernel's own init. Include the XNNPACK bench params, the gtest params, the init
envelope's extremes and one regime per code path you know of; add `check_extra` for
correctness-only extremes and a `heldout` sampler (it must respect the init's asserts).
Schema: `portforge_tt/cpu/regimes.py`.

## 3. Write the technique file

`kernels/<family>/TECHNIQUES.json` (schema: `catalogue.validate_kernel`): the exact
formula, domain and width bounds, and candidate techniques (catalogue ids or local
`k:<name>` ideas) with how each applies here and why it might win. Mine it from every
arch variant of the family (`xnnpack.variants`), upstream commits, existing REPORTs
and the catalogue. Never record outcomes in it: the search tree does that, per parent
and combination, so a technique that lost once stays available.

## 4. Calibrate and dry-run (once per board, then per family)

**Ask** before using the board.

- `cpu.calibrate.run(cfg)` measures per-op costs and bandwidth (`data/x280-calib.json`
  is the reference copy). A/A sessions (`evaluate.noise_session`) give the noise floor δ.
- `python -m portforge_tt.kernel evaluate --family F --seed-kernel name=path [--import ...]`
  runs the trusted gates (G0 compile + static profile, G1 gtests, G2 bit-exact check,
  G3/G4 board bench) on known kernels without agents. The ordering must match what is
  known about them before a campaign is worth running.

## 5. Launch and supervise

**Ask**: the agent backend, workers, rounds, hours, board minutes and planner
(`thompson`, mechanical; or `llm`).

```sh
.venv/bin/python -m portforge_tt.kernel run --family F --seed-kernel best=path \
    --rounds 10 --workers 4 --beam 4 --plans 6 --board-minutes 240 --agent claude
```

To stop without losing work, `touch runs/kernel/<run>/STOP`: the controller exits after
the current round; rerun the same command to resume (board seconds and active hours
already spent still count; nodes lost to host or network errors are re-evaluated).

Read `runs/kernel/<run>/status.json`, `plans/round-*.json` and `REPORT.md`
(`python -m portforge_tt.kernel report runs/kernel/<run>`). The default objective is
the geomean speedup over the matrix with no cell slower than the baseline beyond δ.
A flagged cell is re-measured with twice the rounds in a new lease, and only a flag
seen in both sessions makes a candidate infeasible. The Pareto front over (speedup, worst cell, code size) is always reported. The report also shows, from
the same timing rows, the speedup over upstream kernels correct at every VLEN ("portable")
and over the kernel XNNPACK dispatches on the X280 (`dispatched_x280` in families.json). Certify the
frontier (held-out regimes, SMT where the family has a prepped Neon source, X280
gtests, a second bench session) before calling anything a result.

The formal check (`portforge_tt/fv`, any project): the built machine code is lifted
(`lift.py`) into C over an RVV instruction model (`rvv_model.h`) and proved equal to the
project's own C reference by CBMC on bounded sizes at every VLEN (`run.prove`), after a
native co-simulation (`run.cosim`). The model is trusted only through its validation
receipts (`python -m portforge_tt.fv.validate`: every modelled instruction against QEMU,
and the X280 when allowed). A harness (`fv/harness/*.c`) states the domain and which bytes
the kernel may read and write; it is reviewed once per family and never edited by agents.
