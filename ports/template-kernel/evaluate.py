"""Adapter skeleton for a kernel pack. ports/xnnpack-rvv/evaluate.py is the reference.

The host's gate runner (portforge_tt/remote/runner.py) calls:

  families() -> {family: entry}                     entry from families.json
  regimes_spec(family) -> dict                       kernels/<family>/regimes.json
  prepare_matrix(cfg, family, {name: path}, tag)     inject candidates, build, build the harness
                                                     -> (tree_name, harness_exe, dropped)
  check_matrix(cfg, family, exe, names, wheres, out, heldout_seed=None)
                                                     G2 bit-exact check -> {name: {where: row, passed}}
  bench_matrix(cfg, family, exe, names, out, rounds, delta, seed)
                                                     G3/G4 one board session -> ({name: verdict}, receipt)
  verify(cfg, family, kernel_path, ...)              bounded formal equivalence (optional)
"""
import json
from pathlib import Path

PACK = Path(__file__).resolve().parent


def families():
    path = PACK / "families.json"
    return json.loads(path.read_text())["families"] if path.exists() else {}


def regimes_spec(family):
    raise NotImplementedError("load kernels/<family>/regimes.json with portforge_tt.remote.regimes.load")


def prepare_matrix(cfg, family, variants, tag="matrix", entry=None):
    raise NotImplementedError


def check_matrix(cfg, family, exe, candidates, wheres=("qemu:128", "qemu:1024"), out=None, heldout_seed=None):
    raise NotImplementedError


def bench_matrix(cfg, family, exe, candidates, out=None, rounds=15, delta=None, upstream=None, entry=None, seed=1):
    raise NotImplementedError
