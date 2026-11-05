# Upstream PR checklist (tt-metal)

What reviewers of our earlier tt-metal PRs (`models/experimental/{nllb,esm2,parakeet}`) asked
for. Check every item before you call the deliverable ready; the operator reviews it again at
PR time. These are requirements of tt-metal, not of this harness: passing the harness gates does
not make a package mergeable.

The deliverable is one directory (e.g. `models/experimental/<name>/`) containing:

- **`tt/`**: the TTNN implementation, importable without the sandbox (no `import backend`).
- **`tests/`**: real pytest tests.
  - Each test is a `def test_...` function with `assert`s.
  - Each test takes the repository `device` (or `mesh_device`) fixture; never call `open_device` yourself.
  - At least one end-to-end test runs the real checkpoint (`tests/*e2e*` or `tests/*demo*`).
- **`demo/`**: a runnable script with explicit checkpoint, device and precision arguments.
- **`README.md`**:
  - model card links;
  - a performance table in the style of `models/model_targets.yaml`: latency or throughput (ms, series/s, t/s, fps, rtf) with batch and shape stated;
  - limitations.

Every Python file:

- starts with an SPDX header (`SPDX-FileCopyrightText`, `SPDX-License-Identifier: Apache-2.0`);
- has no hard-coded sandbox paths (`/weights`, `/input`, `/output`, `/work`); take paths as arguments or fixtures;
- contains no vendored third-party code unless it has license attribution (Apache-2.0 compatible).

When the PR improves an existing upstream port, its claims must survive a hostile review:

- **Baseline:** compare against upstream's best documented configuration (presets, opt-in
  flags such as L1 chunking, their trace runner), not its constructor defaults. First reproduce a
  number upstream publishes in its own README within 10%; if you cannot, the baseline is wrong.
- **Same harness, one run:** every number in the PR body and README comes from one measurement run
  of both implementations; state the job or log it came from.
- **Tested envelope:** any routing or dispatch condition (context length, horizon, batch, groups)
  covers only shapes that tests exercise; sweep the edges (shortest and longest inputs, every
  horizon) at both small and large batch.
- **Library semantics:** no device state survives a public call unless the user opts in and can
  release it (a live trace overwrites buffers allocated after it). Memory figures count everything
  the public object holds.

The worker prepares the branch name, commit message and PR body. The human operator reviews, signs (DCO) and opens the PR; agents never push.
