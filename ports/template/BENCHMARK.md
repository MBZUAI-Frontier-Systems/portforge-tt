# <Model> benchmark

<!-- Provenance and rationale for corpus.json. Agents read this; reviewers audit it. -->

## Frozen corpus (`corpus.json`)

TODO: what the corpus contains, where each item comes from (URL, revision or
access date, license), and how items were selected. Include deterministic
synthetic edge cases (silence, constants, extreme dynamic range, minimum and
maximum lengths) with their generator and seed.

The corpus is a JSON object with `schema_version: 1`, a `sources` list
(provenance and licenses), `cases` (one entry per clip, keyed by name, holding
the input data and an optional `label`), and `selection` naming the clips used
by each stage (`smoke`, `bringup`, `full`). Its identity is the SHA-256 of its
canonical JSON; pin that in `port.json` and `evaluate.py`.

The `perf` stage reuses the bringup selection plus a batch sweep of the smoke
item, so it needs no extra corpus entries.

Add a `robustness` selection for inputs that real use will produce but the
bringup set does not stress: out-of-distribution data, added noise, the
longest envelope inputs, extreme and constant values, empty or all-padding
rows. Mix input distributions and generate synthetic ones from recorded seeds
(KernelBench-Verified found single-distribution suites overstate kernels).
These cases are gated like the others (finite outputs, oracle NRMSE, task
metric within its bound) and belong to the held-out side unless a case is
deliberately public.

## Gates and why

TODO: each gate in `GATES`, what it protects against, and why the threshold is
appropriate for this task.

## Held-out discipline

`full` cases are never shown to agents during development. If an operator
shares a failing full case, later full results are no longer blind for it.
