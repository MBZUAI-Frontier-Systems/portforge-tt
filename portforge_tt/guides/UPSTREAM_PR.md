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

The worker prepares the branch name, commit message and PR body. The human operator reviews, signs (DCO) and opens the PR; agents never push.
