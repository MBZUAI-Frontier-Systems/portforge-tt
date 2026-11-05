# Kernel pack template

Copy to `ports/<pack>/` for a new CPU library. `evaluate.py` is the adapter the host's gate
runner (`portforge_tt/remote/runner.py`) calls; `ports/xnnpack-rvv` (C intrinsics) and
`ports/dav1d-rvv` (assembly) are complete examples. See AGENTS.md.
