# Kernel pack template

Copy to `ports/<pack>/` for a new CPU library. `evaluate.py` is the adapter the kernel
orchestrator's remote runner (`portforge_tt/kernel/remote.py`) calls; `ports/xnnpack-rvv`
is the complete example. See the "Optimizing a CPU kernel family" section of AGENTS.md.
