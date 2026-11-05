"""CBMC over lifted RVV machine code: the built binary's instructions are lifted to C over
an RVV model (rvv_model.h) and proved equal to the project's C reference per config and
VLEN (see README.md). Project tables (harnesses, configs) live per project: dav1d.py."""
from .. import contract


def certify(cfg, family, entry, candidate, out, budget):
    """Contract entry point; ``candidate['build_dir']`` is the project build holding the candidate."""
    if entry.get("project") != "dav1d":
        return contract.result("cbmc", "UNSUPPORTED", error="cbmc has harnesses for dav1d families only")
    from . import dav1d
    summary = dav1d.certify(cfg, family, entry, candidate["build_dir"], out, budget=budget)
    counter = summary["failed"][:3] if summary["verdict"] == "COUNTEREXAMPLE" else None
    detail = {k: v for k, v in summary.items()
              if k not in ("passed", "verdict", "proved", "configs", "attempted", "seconds", "error")}
    return contract.result("cbmc", summary["verdict"], proved=summary.get("proved", 0),
                           configs=summary.get("configs", 0), attempted=summary.get("attempted", 0),
                           counterexample=counter, seconds=summary.get("seconds", 0), error=summary.get("error"),
                           **detail)
