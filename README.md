# portforge-tt

Agents port pretrained models to Tenstorrent hardware with TTNN. A supervisor plans one step per turn, a worker writes the backend and runs sandboxed jobs on TT cards, and a trusted evaluator checks every candidate against an FP32 reference. The held-out corpus runs once, on frozen source. Among passing candidates, a perf stage (latency, throughput, board energy) ranks them on a Pareto front.

## Run

```sh
pip install -e .
python -m pytest tests -q                         # no hardware needed
portforge --port ports/chronos2 --cluster my-cluster.yaml --hours 6 --detach
```

You need a TT host (TT-Metal, `bwrap`, the checkpoint) and a GPU host for the FP32 oracle, both over SSH; copy `cluster.yaml` and fill it in. Progress is in `runs/<run>/status.json`; `python -m portforge_tt.ranking.report runs/<run>` ranks the candidates. `--resume runs/<run>` continues a campaign, `--agent claude|codex|opencode` picks the agent. `AGENTS.md` covers adding a model.

## Layout

| Path | What |
|---|---|
| `ports/<model>/` | a pack: pinned model, corpus, evaluator, agent contract (start from `ports/template/`) |
| `portforge_tt/control/` | the campaign: run creation, the loop, the agents' tools, jobs |
| `portforge_tt/ranking/` | objective, search tree, report |
| `portforge_tt/remote/` | what runs on the TT and GPU hosts: sandbox runner, telemetry, the evaluator kit |
| `portforge_tt/agents/` | opencode, Claude Code and Codex wrappers |
| `portforge_tt/guides/`, `prompts/` | what the agents read |

## Results

Blackhole p150, FP32 references on an A100:

| pack | smoke | bringup | full | tt-metal PR |
|---|---|---|---|---|
| nllb (bf16) | PASS | PASS 8/8 | 17/20 | [#57398](https://github.com/tenstorrent/tt-metal/pull/57398) |
| chronos2 (bf16) | PASS (NRMSE 0.009) | PASS (NRMSE 0.043) | not run | [#57764](https://github.com/tenstorrent/tt-metal/pull/57764) |
| esm2 (bf16) | PASS (NRMSE 0.030) | FAIL 4/8 | not run | [#57833](https://github.com/tenstorrent/tt-metal/pull/57833) |
| parakeet (bf16) | PASS (NRMSE 0.018) | PASS 8/8 (NRMSE 0.033) | 18/20 | [#57845](https://github.com/tenstorrent/tt-metal/pull/57845) |

## Paper

**PortForge-TT: Enabling Agentic Cross-Platform Model Adaptation using CHIA.** Abror Shopulatov, Dongning Ma, Abdulrahman Mahmoud (MBZUAI). A³ Workshop @ MICRO 2026. Preprint coming soon.

## License

BSD-3-Clause ([LICENSE](LICENSE)). The agent wrappers and MCP server base are adapted from [CHIA](https://github.com/ucb-bar/chia) and keep their upstream headers.
