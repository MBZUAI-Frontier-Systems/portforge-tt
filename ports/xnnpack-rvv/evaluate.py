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

from portforge_tt.cpu import bench as pbench  # noqa: E402
from portforge_tt.cpu import config, equiv, harness, host, kernelsrc, regimes, x280, xnnpack  # noqa: E402
sys.path.insert(0, str(_REPO / "ports" / "xnnpack-rvv" / "research"))
import saltyrn  # noqa: E402

PACK = _REPO / "ports" / "xnnpack-rvv"
FAMILIES = PACK / "families.json"
VLENS = [128, 256, 512, 1024]
# Families with no SALTyRN-prepped Neon source: the Neon ukernel they port from.
NEW_COVERAGE = {"qs8-vmul-minmax-rndnu": "xnn_qs8_vmul_minmax_rndnu_ukernel__neon_ld64_u8"}
# Hand-set fields build_families keeps. x32-transposec builds from current upstream,
# whose RVV variants include three newer (Autocomp-optimized) 8xv kernels; its
# injected candidates register with a 16x16 tile so the gtests sweep blocks up to
# 32x32; it takes no params, so its formal gate is the exhaustive layout check.
FAMILY_OVERRIDES = {
    # aligned code for every timed family since x32-transposec showed identical kernels
    # timing 2.4% apart at different addresses
    "qs8-vadd-minmax": {"cflags": "-falign-functions=64 -falign-loops=64"},
    "x32-transposec": {"base": "xnnpack", "signature": "transposec", "init": None,
                       # identical kernels at different addresses differed by up to 2.4% per
                       # cell; aligned functions and loops bring that under 1%
                       "cflags": "-falign-functions=64 -falign-loops=64",
                       # transpose-config.c: vlenb == 64 (the X280's VLEN 512) selects 16x8
                       "dispatched_x280": "xnn_x32_transposec_ukernel__16x8_rvv",
                       "overrides": {"4": "16", "5": "16"}, "smt": "none",
                       "upstream_rvv": ["xnn_x32_transposec_ukernel__4x4_rvv", "xnn_x32_transposec_ukernel__8x8_rvv",
                                        "xnn_x32_transposec_ukernel__16x8_rvv", "xnn_x32_transposec_ukernel__32x8_rvv",
                                        "xnn_x32_transposec_ukernel__8xv1_rvv", "xnn_x32_transposec_ukernel__8xv2_rvv",
                                        "xnn_x32_transposec_ukernel__8xv4_rvv"]},
}


def families():
    return json.loads(FAMILIES.read_text())["families"] if FAMILIES.exists() else {}


def _registry_entry(family, entry=None):
    """Pack metadata for a family: an explicit entry, else families.json, else
    the SALTyRN survey registry."""
    if entry is not None:
        return entry
    entry = families().get(family) or saltyrn.load_registry().get(family)
    if entry is None:
        raise SystemExit(f"{family!r} is in neither families.json nor the research registry; "
                         f"add it with build_families() or pass entry={{'inc': ..., 'neon_ukernel': ...}}")
    return entry


def build_families(cfg, names=None, base="xnnpack-saltyrn"):
    """Write families.json from the pinned tree: for each family its registry, the
    Neon ukernel it ports from, signature, init function, upstream RVV variants,
    and whether the SMT gate has a prepped Neon source (SALTyRN) to compare with."""
    tree = cfg.src / base
    survey = saltyrn.load_registry()
    wanted = names or sorted({*survey, *NEW_COVERAGE})
    incs = {p: p.read_text(errors="ignore") for p in (tree / "src").rglob("*.inc")}
    out = {}
    for family in wanted:
        known = survey.get(family, {})
        neon = NEW_COVERAGE.get(family) or known.get("neon_ukernel")
        if not neon:
            continue
        inc, line = None, None
        for path, text in incs.items():
            for _macro, args, _raw in xnnpack.registry_lines(text):
                if len(args) > 1 and args[1] == neon:
                    inc, line = path, args
        if inc is None:
            continue
        rel_inc = str(inc.relative_to(tree))
        source = ({"repo": "saltyrn", "path": known["neon_source"]} if known.get("neon_source")
                  else {"repo": base, "path": str(xnnpack.find_ukernel_file(tree, neon).relative_to(tree))})
        out[family] = {
            "inc": rel_inc, "neon_ukernel": neon, "neon_source": source,
            "signature": xnnpack.signature_of(tree, rel_inc),
            "init": line[-1] if line[-1].startswith("xnn_init") else None,
            # by name too: upstream registers some RVV kernels as xnn_arch_none
            # inside an #if XNN_ARCH_RISCV guard (e.g. qs8-f32-vcvt)
            "upstream_rvv": [u for arch, uks in xnnpack.variants(tree, rel_inc).items() for u in uks
                             if "riscv" in arch or "__rvv" in u or u.endswith("_rvv")],
            "smt": "saltyrn" if source["repo"] == "saltyrn" else "none",
            **FAMILY_OVERRIDES.get(family, {}),
        }
    FAMILIES.write_text(json.dumps({"comment": f"generated by evaluate.build_families against {base}",
                                    "families": out}, indent=1) + "\n")
    return out


def neon_text_for(cfg, entry):
    """The Neon oracle's source: just the ported ukernel, even when the upstream
    file defines several unroll variants."""
    source = entry.get("neon_source")
    if isinstance(source, str):  # SALTyRN registry form
        source = {"repo": "saltyrn", "path": source}
    text = (cfg.src / source["repo"] / source["path"]).read_text()
    if source["repo"] == "saltyrn":
        return text
    return kernelsrc.extract_function(text, entry["neon_ukernel"])


def _x280_ok(cfg):
    return x280.port_open(cfg)


def _register(cfg, family, kernel_path, name, entry, tag="dev"):
    """A clean worktree with only this kernel injected; returns (tree name, targets).

    `tag` names the worktree and build, so concurrent evaluations (orchestrator
    workers) never reset each other's trees."""
    tree_name = f"pf-{family}-{tag}"
    tree = xnnpack.tree(cfg, tree_name, base=base_of(entry))
    inj = _injection(entry, kernel_path, name)
    targets = xnnpack.targets_for(tree, inj.inc)
    if not targets["test"]:
        raise SystemExit(f"no gtest target includes {inj.inc}")
    injections, dropped = _build_with_injections(cfg, tree_name, [inj],
                                                 sorted({*targets["test"], *targets["bench"]}), base=base_of(entry),
                                                 cflags=entry.get("cflags", ""))
    if dropped:
        raise SystemExit(f"kernel does not compile in XNNPACK: {dropped}")
    return tree_name, targets


def base_of(entry):
    """The XNNPACK checkout a family builds from (sources.json key): SALTyRN's pin
    by default; families whose newest upstream RVV kernels postdate it name the
    current upstream so the baseline is what XNNPACK ships today."""
    return entry.get("base", "xnnpack-saltyrn")


def _injection(entry, src, name):
    """Register a candidate by cloning the template's registry line; `overrides`
    replaces registry arguments (transposes: the tile the gtests sweep)."""
    return xnnpack.Injection(source=Path(src).resolve(), inc=entry["inc"],
                             template=entry.get("template", entry["neon_ukernel"]), name=name,
                             overrides=entry.get("overrides"))


def _build_with_injections(cfg, name, injections, targets, log=print, base="xnnpack-saltyrn", cflags=""):
    """Build with the injections; drop kernels that do not compile and retry once."""
    dropped = {}
    while True:
        tree = xnnpack.tree(cfg, name, base=base)
        added = xnnpack.inject(tree, injections)
        try:
            xnnpack.build(cfg, tree, name, targets, log=log, cflags=cflags)
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


def test(cfg, family, kernel_path, name, out=None, entry=None, tag="dev"):
    """Gate 1: XNNPACK gtests, QEMU at every VLEN plus X280 when reachable.

    Passes only if every run ran exactly the tests the filter selects (listed up
    front) and failed none: a crash, timeout or empty filter is a failure."""
    entry = _registry_entry(family, entry)
    tree_name, targets = _register(cfg, family, kernel_path, name, entry, tag)
    out = Path(out) if out else _results(cfg, family, "test")
    wheres = [f"qemu:{v}" for v in VLENS] + (["x280"] if _x280_ok(cfg) else [])
    matrix, problems = {}, []
    for target in targets["test"]:
        expected = len(xnnpack.list_tests(cfg, tree_name, target, f"*{name}*"))
        runs = []
        for where in wheres:
            run = xnnpack.run_tests(cfg, tree_name, target, f"*{name}*", where, out)
            run["expected"] = expected
            matrix[f"{target}@{where}"] = run
            runs.append(run)
        problems += xnnpack.test_verdict(runs, expected)[1]
    return {"gate": "xnnpack_tests", "passed": not problems, "wheres": list(matrix), "detail": matrix,
            "problems": problems,
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
    neon = neon_text if neon_text is not None else neon_text_for(cfg, _registry_entry(family, entry))
    rvv = Path(kernel_path).read_text()
    out = Path(out) if out else _results(cfg, family, "verify")
    verdict = equiv.verify(cfg, neon, rvv, family, out, mode="configs", vlen=512,
                           budget=budget, configs=configs, max_batch=max_batch)
    passed = verdict.verdict == "ALL_PASSED"
    return {"gate": "formal_equivalence", "passed": passed, "verdict": verdict.verdict,
            "max_verified_batch": verdict.max_verified_batch, "configs": len(verdict.configs),
            "counterexample": verdict.counterexample, "failed_config": verdict.failed_config,
            "missing_intrinsics": "missing intrinsics: " in (verdict.error or ""),
            "error": verdict.error, "seconds": verdict.seconds}


def bench(cfg, family, kernel_path, name, out=None, repetitions=5, entry=None, tag="dev"):
    """Gate 3: XNNPACK benchmark on the X280; candidate vs scalar and every RVV variant."""
    entry = _registry_entry(family, entry)
    tree_name, targets = _register(cfg, family, kernel_path, name, entry, tag)
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
        rows.update(xnnpack.run_bench(cfg, tree_name, target, f"xnn_{token}_ukernel", out))
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


def assess(cfg, family, kernel_path, name, budget=600, out=None, entry=None, neon_text=None, tag="dev"):
    """All runnable gates + a markdown report + next-step hints."""
    results = {"test": test(cfg, family, kernel_path, name, out=out, entry=entry, tag=tag),
               "verify": verify(cfg, family, kernel_path, budget=budget, out=out, entry=entry,
                                neon_text=neon_text),
               "bench": bench(cfg, family, kernel_path, name, out=out, entry=entry, tag=tag)}
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


# ----------------------------------------------------------------------------- regime matrix

def regimes_spec(family):
    return regimes.load(PACK / "kernels" / family / "regimes.json")


def prepare_matrix(cfg, family, variants, tag="matrix", entry=None):
    """Inject every variant ({ukernel name: source path}) into one tree, build the
    family's targets and pfharness. Returns (tree_name, exe, dropped)."""
    entry = _registry_entry(family, entry)
    tree_name = f"pf-{family}-{tag}"
    tree = xnnpack.tree(cfg, tree_name, base=base_of(entry))
    targets = xnnpack.targets_for(tree, entry["inc"])
    injections = [_injection(entry, src, name) for name, src in variants.items()]
    injections, dropped = _build_with_injections(cfg, tree_name, injections,
                                                 sorted({*targets["test"], *targets["bench"]}), base=base_of(entry),
                                                 cflags=entry.get("cflags", ""))
    exe = harness.build(cfg, tree_name, entry)
    return tree_name, exe, dropped


def check_matrix(cfg, family, exe, candidates, wheres=("qemu:128", "qemu:1024"), out=None, heldout_seed=None):
    """G2: bit-exact check of each candidate vs the family's reference kernel over
    every regime (and held-out regimes when a seed is given), per where."""
    spec = regimes_spec(family)
    out = Path(out) if out else _results(cfg, family, "check")
    out.mkdir(parents=True, exist_ok=True)
    regime_list = regimes.check_regimes(spec)
    if heldout_seed is not None:
        regime_list += regimes.draw_heldout(spec, heldout_seed)
    rfile = harness.write_regimes(spec, regime_list, out / "regimes.txt")
    result = {}
    for where in wheres:
        rows = harness.check(cfg, exe, rfile, regimes.check_sizes(spec), spec["reference"], candidates, where)
        for name, row in rows.items():
            result.setdefault(name, {})[where] = row
    for name, by_where in result.items():
        by_where["passed"] = all(isinstance(r, dict) and not r.get("error") and r.get("mismatched_cases") == 0
                                 and r.get("guard_violations") == 0 for w, r in by_where.items() if w != "passed")
    (out / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def bench_matrix(cfg, family, exe, candidates, out=None, rounds=15, delta=None, upstream=None, entry=None,
                 seed=1):
    """G4: one X280 session timing every candidate and every upstream RVV variant
    on the regime matrix; returns {candidate: bench.analyze verdict}."""
    entry = _registry_entry(family, entry)
    spec = regimes_spec(family)
    out = Path(out) if out else _results(cfg, family, "bench-matrix")
    out.mkdir(parents=True, exist_ok=True)
    upstream = list(upstream if upstream is not None else entry["upstream_rvv"])
    kernels = upstream + list(candidates)
    rfile = harness.write_regimes(spec, regimes.bench_regimes(spec), out / "regimes.txt")
    receipt, rows = pbench.session(cfg, exe, rfile, spec["bench_shapes"], spec["reference"], kernels, out,
                                   rounds=rounds, seed=seed)
    delta = delta if delta is not None else spec.get("noise_floor", 0.01)
    verdicts = {name: pbench.analyze(rows, kernels, len(upstream) + i, list(range(len(upstream))), delta=delta,
                                     reference_cell=spec.get("reference_cell"))
                for i, name in enumerate(candidates)}
    # Secondary views of the same rows (reported, never the objective): against upstream
    # kernels correct at every VLEN, and against the kernel XNNPACK dispatches on the X280.
    views = {}
    specific = vlen_specific(cfg, family, exe, upstream)
    if specific and len(specific) < len(upstream):
        views["portable"] = ([i for i, k in enumerate(upstream) if k not in specific],
                             "upstream kernels correct at every VLEN (excludes " + ", ".join(
                                 k.split("__")[-1] for k in specific) + ")")
    if entry.get("dispatched_x280") in upstream:
        views["dispatched"] = ([upstream.index(entry["dispatched_x280"])],
                               "the kernel XNNPACK dispatches on the X280: " + entry["dispatched_x280"].split("__")[-1])
    for i, name in enumerate(candidates):
        verdicts[name]["views"] = {}
        for view, (slots, why) in views.items():
            v = pbench.analyze(rows, kernels, len(upstream) + i, slots, delta=delta)
            verdicts[name]["views"][view] = {"why": why, "speedup_geomean": v["metrics"]["speedup_geomean"],
                                             "worst_cell": v["metrics"]["worst_cell"], "flagged": v["flagged"]}
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1, default=str) + "\n")
    return verdicts, receipt


def vlen_specific(cfg, family, exe, upstream):
    """Upstream kernels that are exact at VLEN 1024 but not at 128 (written for one
    VLEN, dispatched only there). Cached next to the harness binary."""
    cache = Path(exe).parent / "upstream-vlen.json"
    key = hashlib.sha256(json.dumps([harness.sha(), sorted(upstream),
                                     (PACK / "kernels" / family / "regimes.json").read_text()]).encode()).hexdigest()
    if cache.exists():
        data = json.loads(cache.read_text())
        if data.get("key") == key:
            return data["specific"]
    check = check_matrix(cfg, family, exe, list(upstream), wheres=("qemu:128", "qemu:1024"),
                         out=Path(exe).parent / "upstream-vlen-check")
    ok = lambda r: isinstance(r, dict) and r.get("mismatched_cases") == 0 and r.get("guard_violations") == 0
    specific = [k for k in upstream if ok(check[k].get("qemu:1024")) and not ok(check[k].get("qemu:128"))]
    cache.write_text(json.dumps({"key": key, "specific": specific}) + "\n")
    return specific


def noise_session(cfg, family, exe, kernel, out=None, rounds=15, seed=1):
    """A/A: one upstream kernel against itself; returns per-cell speedup estimates."""
    spec = regimes_spec(family)
    out = Path(out) if out else _results(cfg, family, "aa")
    out.mkdir(parents=True, exist_ok=True)
    rfile = harness.write_regimes(spec, regimes.bench_regimes(spec), out / "regimes.txt")
    _, rows = pbench.session(cfg, exe, rfile, spec["bench_shapes"], spec["reference"], [kernel, kernel], out,
                             rounds=rounds, seed=seed)
    verdict = pbench.analyze(rows, [kernel, kernel], 1, [0], delta=0.0)
    return {k: c["speedup"]["s"] for k, c in verdict["cells"].items() if "speedup" in c}


# ----------------------------------------------------------------------------- reporting

def _hints(results):
    hints = []
    t, v, b = results["test"], results["verify"], results["bench"]
    if not t["passed"]:
        hints.append(f"gate 1 FAILS: {'; '.join(t['problems'][:6])}. Fix correctness before "
                     f"optimizing. First messages: {json.dumps(t['failures'], default=str)[:600]}")
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
    """A fresh result directory; microseconds and pid keep concurrent gates apart."""
    out = cfg.results / family / f"{gate}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S.%fZ}-{os.getpid()}"
    out.mkdir(parents=True, exist_ok=True)
    return out
