"""PortForge-TT: autonomous model porting to Tenstorrent hardware.

A supervisor agent plans one bounded unit of work at a time; a worker agent
researches the model, writes the TTNN backend, and runs isolated jobs on a TT
card through scoped MCP tools. A trusted evaluator compares every candidate
with an independent FP32 reference, first on small suites and finally, once,
on a frozen held-out corpus.

``control/`` runs a campaign (``portforge`` is control/controller.py),
``ranking/`` ranks passing candidates, ``remote/`` is what the TT and
reference hosts run, and ``agents/`` wraps the opencode, claude and codex CLIs.
Every module also runs *flat*: a campaign copies the code directories into one
``runs/<run>/harness/`` and re-executes from there, and the remote hosts receive
plain files, which is why imports use the ``if __package__:`` shim.
"""

__version__ = "2.0.0"
