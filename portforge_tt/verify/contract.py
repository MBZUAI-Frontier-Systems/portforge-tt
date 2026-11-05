"""The shared contract of formal verification methods (verify/<method>/).

A method proves a candidate kernel equal to a trusted reference on a bounded
domain. Each method is a package exposing

    certify(cfg, family, entry, candidate, out, budget) -> dict

where ``candidate`` is a dict of method-specific inputs the pack adapter
provides (cbmc: ``build_dir``, the project build holding the candidate's
machine code; cvc5: ``source``, ``oracle`` and ``configs``), ``out`` is a
host directory for logs and ``budget`` is wall seconds. It returns
``result(...)``: one verdict from VERDICTS plus counts and receipts.

  ALL_PASSED      every bounded configuration proved
  PARTIAL         no counterexample, but some configurations unproved (timeout or budget)
  COUNTEREXAMPLE  a configuration where the candidate and the reference differ
  UNSUPPORTED     the method cannot express this candidate (unmodelled instruction, no params)
  ERROR           the method or its harness failed; says nothing about the candidate

Only ALL_PASSED passes. A family lists its methods in families.json ("verify":
["cbmc"]); certification runs each one through ``run``.
"""
import importlib

VERDICTS = ("ALL_PASSED", "PARTIAL", "COUNTEREXAMPLE", "UNSUPPORTED", "ERROR")
METHODS = {"cbmc": "portforge_tt.verify.cbmc", "cvc5": "portforge_tt.verify.cvc5"}


def result(method, verdict, *, proved=0, configs=0, attempted=0, counterexample=None, seconds=0,
           error=None, **detail):
    """One method's normalized result; ``detail`` keeps the method's own receipt fields."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
    return {"method": method, "verdict": verdict, "passed": verdict == "ALL_PASSED", "proved": proved,
            "configs": configs, "attempted": attempted, "counterexample": counterexample,
            "seconds": seconds, "error": error, "detail": detail}


def run(method, cfg, family, entry, candidate, out, budget):
    """Run one method; an exception is an ERROR result, never a pass."""
    if method not in METHODS:
        return result(method, "UNSUPPORTED", error=f"unknown verification method {method!r}")
    try:
        return importlib.import_module(METHODS[method]).certify(cfg, family, entry, candidate, out, budget)
    except Exception as error:  # a broken method must not block or pass certification
        return result(method, "ERROR", error=f"{type(error).__name__}: {error}"[-2000:])


def passed(results):
    """Certification needs every listed method to pass (an empty list passes: nothing was claimed)."""
    return all(r.get("passed") for r in results)
