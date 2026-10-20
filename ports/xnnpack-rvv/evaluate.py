"""Gates for ports/xnnpack-rvv campaigns, as importable functions (no CLI).

An agent edits kernels/<family>/kernel.c, then asks for feedback with one call:

    from ports.xnnpack_rvv import evaluate
    result = evaluate.assess(cfg, "qs8-vcvt", "kernel.c", name="xnn_qs8_vcvt_ukernel__rvv_pf1")

`assess` runs the three gates (tests, formal equivalence, benchmark) that are
runnable right now, and returns a dict whose `report` is a human/agent-readable
markdown summary and whose `hints` say what to do next. Every gate result is
also written under results/ so receipts survive across turns.

The evaluator is not the agent's to modify: fixes belong in the kernel, or in a
REPORT.md note arguing the tool is wrong.
"""
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from portforge_tt.cpu import config, equiv, host, x280, xnnpack  # noqa: E402
sys.path.insert(0, str(_REPO / "ports" / "xnnpack-rvv" / "research"))
import saltyrn  # noqa: E402

PACK = _REPO / "ports" / "xnnpack-rvv"
VLENS = [128, 256, 512, 1024]


def _registry_entry(family, entry=None):
    """Pack metadata for a family: an explicit entry (new-coverage families that
    predate the survey registry) or the SALTyRN registry lookup."""
    if entry is not None:
        return entry
    entry = saltyrn.load_registry().get(family)
    if entry is None:
        raise SystemExit(f"{family!r} is not in the research registry; pass entry={{'inc': ..., "
                         f"'neon_ukernel': ...}} explicitly (new-coverage family)")
    return entry


def _x280_ok(cfg):
    return x280.port_open(cfg)


def _register(cfg, family, kernel_path, name, entry):
    """A clean worktree with only this kernel injected; returns (tree, targets)."""
    tree = xnnpack.tree(cfg, f"pf-{family}")
    inj = xnnpack.Injection(source=Path(kernel_path).resolve(), inc=entry["inc"],
                            template=entry["neon_ukernel"], name=name)
    targets = xnnpack.targets_for(tree, inj.inc)
    if not targets["test"]:
        raise SystemExit(f"no gtest target includes {inj.inc}")
    injections, dropped = _build_with_injections(cfg, f"pf-{family}", [inj],
                                                 sorted({*targets["test"], *targets["bench"]}))
    if dropped:
        raise SystemExit(f"kernel does not compile in XNNPACK: {dropped}")
    return tree, targets


def _build_with_injections(cfg, name, injections, targets, log=print):
    """Build with the injections; drop kernels that do not compile and retry once."""
    dropped = {}
    while True:
        tree = xnnpack.tree(cfg, name)
        added = xnnpack.inject(tree, injections)
        try:
            xnnpack.build(cfg, tree, name, targets, log=log)
            return injections, dropped
        except xnnpack.BuildError as error:
            bad = {}
            for inj, rel in zip(injections, added):
                hits = [e for e in error.errors if Path(rel).name in e or inj.name in e]
                if hits:
                    bad[inj.name] = hits
            if not bad:
                raise
            for bad_name, hits in bad.items():
                dropped[bad_name] = [h.split("error:", 1)[-1].strip()[:240] for h in hits[:3]]
                log(f"dropping {bad_name}: does not compile")
            injections = [i for i in injections if i.name not in dropped]
            if not injections:
                return injections, dropped


def test(cfg, family, kernel_path, name, out=None, entry=None):
    """Gate 1: XNNPACK gtests, QEMU at every VLEN plus X280 when reachable."""
    entry = _registry_entry(family, entry)
    tree, targets = _register(cfg, family, kernel_path, name, entry)
    out = Path(out) if out else _results(cfg, family, "test")
    wheres = [f"qemu:{v}" for v in VLENS] + (["x280"] if _x280_ok(cfg) else [])
    matrix = {}
    for where in wheres:
        for target in targets["test"]:
            matrix[f"{target}@{where}"] = xnnpack.run_tests(cfg, f"pf-{family}", target, f"*{name}*", where, out)
    passed = all(s["failed"] == 0 and not s.get("errors") for s in matrix.values())
    return {"gate": "xnnpack_tests", "passed": passed, "wheres": list(matrix), "detail": matrix,
            "failures": {k: v.get("messages", {}) for k, v in matrix.items() if v.get("failed")}}


def verify(cfg, family, kernel_path, name=None, budget=600, max_batch=8, out=None, entry=None,
           neon_text=None):
    """Gate 2: cvc5 equivalence vs the Neon oracle, XNNPACK params.

    `neon_text` overrides the SALTyRN-prepped Neon source (new-coverage families
    that have no prepped copy pass the upstream Neon kernel text here)."""
    params_file = PACK / "kernels" / family / "params.json"
    if not params_file.exists():
        return {"gate": "formal_equivalence", "passed": False, "verdict": "NO_PARAMS",
                "hint": f"write {params_file}: a JSON list of param structs exactly as "
                        f"XNNPACK's init produces them (12+ edge configs; see "
                        f"research/qs8_vadd_smoke/difftest.c --dump-params for the pattern). "
                        f"Formal verification without producer-true params is what let the "
                        f"buggy SALTyRN kernels pass."}
    configs = json.loads(params_file.read_text())
    if neon_text is None:
        entry = _registry_entry(family, entry)
        neon_text = (cfg.src / "saltyrn" / entry["neon_source"]).read_text()
    rvv = Path(kernel_path).read_text()
    out = Path(out) if out else _results(cfg, family, "verify")
    verdict = equiv.verify(cfg, neon_text, rvv, family, out, mode="configs", vlen=512,
                           budget=budget, configs=configs, max_batch=max_batch)
    passed = verdict.verdict == "ALL_PASSED"
    return {"gate": "formal_equivalence", "passed": passed, "verdict": verdict.verdict,
            "max_verified_batch": verdict.max_verified_batch, "configs": len(verdict.configs),
            "counterexample": verdict.counterexample, "failed_config": verdict.failed_config,
            "missing_intrinsics": "missing intrinsics: " in (verdict.error or ""),
            "error": verdict.error, "seconds": verdict.seconds}


def bench(cfg, family, kernel_path, name, out=None, repetitions=5, entry=None):
    """Gate 3: XNNPACK benchmark on the X280; candidate vs scalar and every RVV variant."""
    entry = _registry_entry(family, entry)
    tree, targets = _register(cfg, family, kernel_path, name, entry)
    if not targets["bench"]:
        return {"gate": "benchmark", "passed": False, "hint": "family has no benchmark target"}
    out = Path(out) if out else _results(cfg, family, "bench")
    if not _x280_ok(cfg):
        return {"gate": "benchmark", "passed": None, "qemu_only": True,
                "hint": "X280 unreachable; QEMU timing is not evidence of speed. "
                        "Finish correctness gates; get board access for numbers."}
    token = xnnpack.family_token(entry["neon_ukernel"])
    rows = {}
    for target in targets["bench"]:
        rows.update(xnnpack.run_bench(cfg, f"pf-{family}", target, f"xnn_{token}_ukernel", out))
    cmp = xnnpack.compare_bench(rows, name, token)
    scalar_ns, by_uk = {}, {}
    for bench_name, row in rows.items():
        if "ns" not in row:
            continue
        uk, shape = xnnpack.split_bench_name(bench_name)
        by_uk.setdefault(uk, {})[shape] = row["ns"]
        if uk and "scalar" in uk:
            scalar_ns[shape] = row["ns"]
    table = []
    for shape, ns in sorted(cmp["ours_ns"].items()):
        peers = {uk: t[shape] for uk, t in cmp["upstream_ns"].items() if shape in t}
        best = min(peers.values()) if peers else None
        scalar = scalar_ns.get(shape)
        table.append({"shape": shape.strip("/"), "ours_ns": round(ns, 1),
                      "scalar_ns": round(scalar, 1) if scalar else None,
                      "vs_scalar": round(scalar / ns, 2) if scalar and ns else None,
                      "best_rvv_ns": round(best, 1) if best else None,
                      "vs_best_rvv": round(best / ns, 3) if best and ns else None,
                      "vs_rvv": {uk.split("__")[-1]: round(t / ns, 3) for uk, t in peers.items() if ns}})
    faster_than_best = all(t["vs_best_rvv"] is not None and t["vs_best_rvv"] >= 1.0 for t in table) if table else False
    return {"gate": "benchmark", "passed": bool(table) and faster_than_best,
            "geomean_vs_best_rvv": cmp["speedup_vs_best"], "per_shape": table,
            "hint": None if faster_than_best else
            "not faster than the best hand-written RVV variant on every shape; "
            "iterate on width/unrolling (see RVV_GUIDE.md) or argue the case in REPORT.md"}


def assess(cfg, family, kernel_path, name, budget=600, out=None):
    """All runnable gates + a markdown report + next-step hints."""
    results = {"test": test(cfg, family, kernel_path, name, out=out),
               "verify": verify(cfg, family, kernel_path, budget=budget, out=out),
               "bench": bench(cfg, family, kernel_path, name, out=out)}
    results["hints"] = _hints(results)
    results["report"] = _report(family, name, results)
    out = Path(out) if out else _results(cfg, family, "assess")
    (out / "assess.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    print(results["report"])
    print("results:", out)
    return results


def submit(cfg, family, kernel_path, name, note=""):
    """Freeze the current kernel + latest assess receipt as a submission."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = PACK / "kernels" / family / f"submission-{stamp}"
    dst.mkdir(parents=True, exist_ok=True)
    src = Path(kernel_path).resolve()
    (dst / "kernel.c").write_text(src.read_text())
    latest = sorted((cfg.results / family).glob("*")) if (cfg.results / family).exists() else []
    assess_files = [p for d in latest for p in [d / "assess.json"] if p.exists()]
    receipt = {"family": family, "kernel": str(src), "name": name,
               "kernel_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
               "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "note": note}
    if assess_files:
        receipt["assess"] = json.loads(assess_files[-1].read_text())
    (dst / "RECEIPT.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")
    print("submitted:", dst)
    return receipt


# ----------------------------------------------------------------------------- reporting

def _hints(results):
    hints = []
    t, v, b = results["test"], results["verify"], results["bench"]
    if not t["passed"]:
        failing = [k for k, s in t["detail"].items() if s.get("failed") or s.get("errors")]
        hints.append(f"gate 1 FAILS on {failing}: fix correctness before optimizing. "
                     f"First messages: {json.dumps(t['failures'], default=str)[:600]}")
    if v.get("verdict") == "COUNTEREXAMPLE":
        hints.append(f"gate 2 found inputs where you differ from Neon "
                     f"(batch {v['counterexample'].get('batch', '?')}, config {v['failed_config']}); "
                     f"make the failing config fail gate 1's tester too if you can, fix, rerun")
    elif v.get("verdict") == "PARTIAL":
        hints.append(f"gate 2 only proved batches up to {v['max_verified_batch']} across "
                     f"{v['configs']} configs: raise budget or lower max_batch; PARTIAL is not a pass")
    elif v.get("verdict") == "MISSING_INTRINSIC":
        hints.append("gate 2: the verifier has no shim for an intrinsic you used; "
                     "switch to a shimmed equivalent or document the gap")
    elif v.get("verdict") == "NO_PARAMS":
        hints.append(v["hint"])
    elif not v.get("passed"):
        hints.append(f"gate 2 verdict {v.get('verdict')}: {str(v.get('error'))[:300]}")
    if b.get("passed") is False:
        hints.append(b["hint"])
    if t["passed"] and v.get("passed") and b.get("passed"):
        hints.append("all gates pass: write REPORT.md and run evaluate.submit(...)")
    return hints


def _report(family, name, results):
    t, v, b = results["test"], results["verify"], results["bench"]
    lines = [f"# assess {family} — {name}", ""]
    lines.append(f"- gate 1 XNNPACK tests: **{'PASS' if t['passed'] else 'FAIL'}** "
                 f"({', '.join(t['wheres'])})")
    vtxt = v.get("verdict", "-")
    lines.append(f"- gate 2 formal equivalence (cvc5, configs mode, VLEN 512): **{vtxt}**"
                 + (f" up to batch {v['max_verified_batch']}, {v['configs']} configs" if v.get("configs") else ""))
    if b.get("passed") is None:
        lines.append(f"- gate 3 benchmark: **SKIPPED** ({b.get('hint', 'no X280')})")
    else:
        lines.append(f"- gate 3 benchmark (X280): **{'PASS' if b['passed'] else 'FAIL'}**"
                     + (f", geomean vs best upstream RVV {b['geomean_vs_best_rvv']:.2f}x" if b.get("geomean_vs_best_rvv") else ""))
        for row in b.get("per_shape", []):
            lines.append(f"    - {row['shape']}: ours {row['ours_ns']} ns"
                         + (f", scalar {row['scalar_ns']} ns ({row['vs_scalar']}x)" if row.get("vs_scalar") else "")
                         + (f", best RVV {row['best_rvv_ns']} ns ({row['vs_best_rvv']}x)" if row["best_rvv_ns"] else ""))
    if results["hints"]:
        lines += ["", "## What to do next"] + [f"- {h}" for h in results["hints"]]
    return "\n".join(lines) + "\n"


def _results(cfg, family, gate):
    out = cfg.results / family / f"{gate}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True, exist_ok=True)
    return out
