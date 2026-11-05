"""Run SALTyRN's verifier with explicit, producer-consistent concrete params.

SALTyRN's concrete mode builds configs by varying one struct field at a time
around fixed edge values (engine/emit.py: _gen_configs). For params with derived
fields (qs8-vadd's bias = rounding - a_mult*a_zp - b_mult*b_zp) that yields
structs XNNPACK never produces, and its edge table stops at shift=15 while
XNNPACK uses shift in [12, 30]. This wrapper replaces the config generator with
whole structs read from a JSON list (field name -> value), e.g. the output of
`difftest --dump-params`, then runs the stock orchestrator CLI.

Usage (inside portforge-rvv-host, cwd $SALTYRN):
  python saltyrn_configs.py <configs.json> <orchestrator args...>
"""
import json
import sys

from workflow.verification.engine import emit
from workflow.verification import orchestrator

configs = json.load(open(sys.argv[1]))
del sys.argv[1]


def explicit_configs(leaves, cap=0):
    paths = [path for path, _ctype, _ in leaves if path != "_"]
    names = {path: path.split(".")[-1] for path in paths}
    missing = {names[p] for p in paths} - set(configs[0])
    if missing:
        raise SystemExit(f"configs lack fields {sorted(missing)}")
    combos = [tuple(str(int(c[names[p]])) for p in paths) for c in configs]
    return paths, combos


emit._gen_configs = explicit_configs
orchestrator.main()
