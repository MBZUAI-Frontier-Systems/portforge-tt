# Task: one kernel family

1. Pick the family with the user; add it to `families.json`.
2. Write `kernels/<family>/regimes.json` with the user (see `portforge_tt/cpu/regimes.py`).
3. Research: write `kernels/<family>/TECHNIQUES.json` from every arch variant of the
   family, upstream history, the catalogue and earlier reports.
4. Dry run without agents: `python -m portforge_tt.kernel evaluate` on the upstream
   kernels and any archived variants; the measured ordering must make sense.
5. Campaign: `python -m portforge_tt.kernel run` (rounds of plan, implement, gate,
   measure).
6. Certify the frontier, write `kernels/<family>/REPORT.md` from the run's `REPORT.md`,
   and prepare the upstream change.
