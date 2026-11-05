# portforge-tt (cpu)

Agents port and optimize CPU kernels for RISC-V Vector. A supervisor plans one step per turn; a worker writes the kernel and calls a few MCP tools. Trusted gates check every candidate bit-exactly against the project's own reference under QEMU at VLEN 128-1024 and time it on a SiFive X280 against the best existing kernel. The final source is certified on held-out regimes and by formal verification of the shipped machine code.

This branch is the CPU side. The Tenstorrent model packs (`ports/chronos2`, `esm2`, `nllb`, `parakeet`, `pocket_tts`, `template`) run from the `tt` branch.

## Run

```sh
pip install -e .
python -m pytest tests -q                         # no hardware needed
portforge --port ports/dav1d-rvv --family refmvs-load_tmvs --cluster my-cluster.yaml --hours 6 --board-minutes 120 --detach
```

You need an x86 host, reachable over SSH, holding the toolchain image (`portforge_tt/remote/docker`), QEMU and the pinned sources; it also reaches the X280. Copy `cluster.yaml` and fill it in.

- **Progress:** `runs/<run>/status.json`.
- **Ranking:** `python -m portforge_tt.ranking.report runs/<run>` ranks the candidates.
- **Dry run:** `--dry-run --seed-kernel name=path` evaluates known kernels without agents.
- **Resume:** `--resume runs/<run>` continues a campaign.
- **Agent:** `--agent claude|codex|opencode` picks the agent.

`AGENTS.md` covers adding a kernel family.

## Layout

| Path | What |
|---|---|
| `ports/<pack>/` | a pack: families, regime matrices, technique menus, the gate adapter, agent contract (start from `ports/template-kernel/`) |
| `portforge_tt/control/` | the campaign: run creation, the loop, the agents' tools, jobs |
| `portforge_tt/ranking/` | objective, search tree, report |
| `portforge_tt/remote/` | what runs on the x86 host: gate runner, toolchain container, QEMU, X280 bench, static profiler |
| `portforge_tt/verify/` | formal methods behind one contract: `cbmc/` (lifted machine code vs the C reference), `cvc5/` (SMT vs the Neon kernel) |
| `portforge_tt/agents/` | opencode, Claude Code and Codex wrappers |
| `portforge_tt/guides/`, `prompts/` | what the agents read |

## Results

All measured on the X280 against the baseline named for each family. Every listed kernel passed the gates (tests at four VLENs plus the X280, bit-exact checks and held-out seeds).

| family | baseline | speedup (geomean) | formal verification |
|---|---|---|---|
| dav1d splat_mv | C | 5.87x | CBMC: all 180 configs proved |
| dav1d wiener (7- and 5-tap) | C | 50.7x | CBMC: 37/192 (1-2 rows); 9,320-config co-simulation sweep across the dav1d families |
| dav1d sgr 3x3 / 5x5 / mix | C | 18.8x / 14.4x / 11.9x | CBMC: degenerate sizes only |
| dav1d loopfilter uv | C | 2.5-3.0x | CBMC: every 4-tap pattern, no 6-tap |
| dav1d load_tmvs | C | 1.78x | not attempted |
| XNNPACK qs8-vadd-minmax | best upstream RVV per cell | 2.61x | cvc5 ALL_PASSED (12 init configs) |
| XNNPACK qs8-vcvt | best upstream RVV per cell | 1.55x | cvc5 ALL_PASSED |
| XNNPACK x32-transposec | best of 7 upstream RVV | 1.04x | none listed for this family |

None of the seven dav1d functions has hand-written RVV upstream. Receipts are in `runs/kernel/` (not versioned).

## License

BSD-3-Clause ([LICENSE](LICENSE)). The agent wrappers and MCP server base are adapted from [CHIA](https://github.com/ucb-bar/chia) and keep their upstream headers.
