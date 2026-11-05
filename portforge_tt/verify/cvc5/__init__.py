"""Bounded SMT equivalence (cvc5, SALTyRN's intrinsic shims) of an RVV C kernel and its Neon
oracle, over parameter structs produced by the library's own init (see equiv.py)."""
from pathlib import Path

from .. import contract

# equiv.py verdicts that say the method could not judge this candidate, and that it broke
UNSUPPORTED = ("NO_PARAMS", "MISSING_INTRINSIC")


def certify(cfg, family, entry, candidate, out, budget):
    """Contract entry point; ``candidate`` holds ``source`` (RVV C path), ``oracle`` (the Neon
    kernel text) and ``configs`` (param structs from the init), plus optional ``max_batch``."""
    from . import equiv
    if not candidate.get("configs"):
        return contract.result("cvc5", "UNSUPPORTED", error="no params: write kernels/<family>/params.json")
    v = equiv.verify(cfg, candidate["oracle"], Path(candidate["source"]).read_text(), family, out, mode="configs",
                     vlen=candidate.get("vlen", 512), budget=budget, configs=candidate["configs"],
                     max_batch=candidate.get("max_batch", 8))
    if "missing intrinsics: " in (v.error or ""):
        verdict = "UNSUPPORTED"
    elif v.verdict in contract.VERDICTS:
        verdict = v.verdict
    else:
        verdict = "UNSUPPORTED" if v.verdict in UNSUPPORTED else "ERROR"
    return contract.result("cvc5", verdict, proved=sum(1 for c in v.configs if c.get("verified_up_to")),
                           configs=len(candidate["configs"]), attempted=len(v.configs),
                           counterexample=v.counterexample or None, seconds=v.seconds, error=v.error or None,
                           equiv_verdict=v.verdict, max_verified_batch=v.max_verified_batch,
                           failed_config=v.failed_config, vlen=v.vlen)
