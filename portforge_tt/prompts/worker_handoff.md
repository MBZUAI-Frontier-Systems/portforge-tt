<!-- Appended to every worker turn. Placeholders: {title}, {repo_id}, {key}. -->
FINAL WORKING PROTOCOL: A routine work unit covers a few edits with port_run checks and at most one or
two trusted evaluations (roughly ten minutes). Aim to save your handoff within about eight minutes;
that is a strong target, not a hard rule: when a high-promise step is mid-flight and clearly
progressing, finish it, but first append a one-line progress checkpoint to STATE.md so an interrupted
turn stays recoverable. Never run past thirty minutes in one turn; the runtime cuts off beyond that.
Stop early on unexpected failure, drift or blockage; reconcile pending jobs without duplicating work.
The controller already saves complete requests, results and logs. Keep STATE.md concise: current
source and its code_sha256, check/perf status, pending job IDs, what was tried with which result,
and the next step. Do not embed raw receipts or logs.
