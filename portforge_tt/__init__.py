"""PortForge: autonomous kernel porting to RISC-V Vector.

A supervisor agent plans one bounded unit of work at a time; a worker agent
ports and optimizes one kernel family through scoped MCP tools. Every
candidate is gated by the project's own tests and a bit-exact check under
QEMU at VLEN 128-1024, timed on a SiFive X280 against the best existing
kernel, and the final source is certified on held-out regimes and by formal
verification of the shipped code.

``control/`` runs a campaign (``portforge`` is control/controller.py),
``ranking/`` ranks passing candidates, ``remote/`` is what the x86 host runs
(toolchain image, QEMU, board), ``verify/`` holds the formal methods behind
one contract, and ``agents/`` wraps the opencode, claude and codex CLIs. A
campaign freezes this package and its pack into ``runs/<run>/harness/`` and
runs from that copy on both machines.
"""

__version__ = "2.0.0"
