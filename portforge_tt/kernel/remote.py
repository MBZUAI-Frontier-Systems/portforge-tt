"""Gate runner on the x86 host (tt_box): python3 -m portforge_tt.kernel.remote CMD ARGS.json

The orchestrator runs on the controller next to the agent CLIs; every trusted
measurement runs here, from the repo copy synced at campaign start, through the
pack adapter (ports/<pack>/evaluate.py). Each command reads a JSON argument
file and prints one JSON object on the last line of stdout.

  gates        G0 compile + static profile, G1 gtests (QEMU), G2 bit-exact check
               for a batch of candidates injected into one tree
  bench        G3/G4: one X280 session over the regime matrix
  dev_compile  untrusted worker convenience: compile one source, static profile
  dev_check    untrusted worker convenience: G2 check of one source under QEMU
  certify      G5: held-out regimes, SMT (when the family has params), X280 gtests,
               second bench session

A pack may implement any command itself as remote_<command>(cfg, args) in its
evaluate.py (ports/dav1d-rvv: assembly kernels gated by checkasm); the result has
the same shape.
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

from portforge_tt.cpu import calibrate, config, harness, host, profile, regimes, x280, xnnpack

REPO = Path(__file__).resolve().parents[2]


def adapter(pack):
    path = REPO / pack / "evaluate.py"
    spec = importlib.util.spec_from_file_location("pack_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def _predict(cfg, tree_name, names, calib, per="vl"):
    """Static profile plus per-loop predicted cycles/element for each kernel."""
    out_dir = xnnpack.build_dir(cfg, tree_name)
    libs = " ".join(host.in_container(cfg, out_dir / f"libxnnpack-microkernels-{k}.a") for k in ("all", "prod"))
    result = {}
    for name in names:
        text = host.run(cfg, f"riscv64-linux-gnu-objdump -dr --no-show-raw-insn --disassemble={name} {libs} "
                             f"2>/dev/null", timeout=300).stdout
        funcs = profile.parse(text)
        if name not in funcs:
            continue
        insns = funcs[name]
        p = profile.function_profile(insns)
        preds = []
        for start, end in profile.loops(insns):
            pr = profile.predict(insns, start, end, calib, per=per) if calib else None
            if pr:
                preds.append({"start": hex(start), **pr})
        p["predicted"] = preds
        hot = [x for x in preds if x]
        p["bound"] = max(hot, key=lambda x: x["compute_cycles"])["bound"] if hot else None
        result[name] = p
    sizes = profile.disassemble(cfg, tree_name, list(result))
    for name, p in result.items():
        p["text_bytes"] = sizes.get(name, {}).get("text_bytes")
    return result


def cmd_gates(cfg, args):
    ev = adapter(args["pack"])
    family, tag = args["family"], args["tag"]
    variants = {name: str(cfg.workdir / rel) for name, rel in args["kernels"].items()}
    tree_name, exe, dropped = ev.prepare_matrix(cfg, family, variants, tag=tag)
    entry = ev.families()[family]
    targets = xnnpack.targets_for(cfg.build / "trees" / tree_name, entry["inc"])
    alive = [n for n in variants if n not in dropped]
    out = {n: {"G0": {"passed": False, "errors": dropped[n]}} for n in dropped}
    calib = calibrate.load(cfg) or json.loads((REPO / "portforge_tt/cpu/data/x280-calib.json").read_text())
    per = "loads" if entry.get("signature") == "transposec" else "vl"
    profiles = _predict(cfg, tree_name, alive, calib, per) if alive else {}
    for name in alive:
        out[name] = {"G0": {"passed": True}, "profile": profiles.get(name)}
    # G1: the family's gtests filtered to each kernel, under QEMU
    wheres = args.get("g1_wheres", ["qemu:128", "qemu:512", "qemu:1024"])
    for name in alive:
        problems, runs = [], {}
        for target in targets["test"]:
            expected = len(xnnpack.list_tests(cfg, tree_name, target, f"*{name}*"))
            batch = []
            for where in wheres:
                r = xnnpack.run_tests(cfg, tree_name, target, f"*{name}*", where,
                                      cfg.workdir / args["out"] / "gtests" / name)
                runs[f"{target}@{where}"] = {k: r.get(k) for k in ("passed", "failed", "error")}
                batch.append(r)
            problems += xnnpack.test_verdict(batch, expected)[1]
        out[name]["G1"] = {"passed": not problems, "problems": problems[:8], "runs": runs}
    # G2: bit-exact check over every (non-held-out) regime
    g1_ok = [n for n in alive if out[n]["G1"]["passed"]]
    if g1_ok:
        check = ev.check_matrix(cfg, family, exe, g1_ok, wheres=tuple(args.get("g2_wheres", ["qemu:128", "qemu:1024"])),
                                out=cfg.workdir / args["out"] / "check")
        for name in g1_ok:
            out[name]["G2"] = {"passed": check[name]["passed"],
                               "detail": {w: r for w, r in check[name].items() if w != "passed"}}
    return {"tree": tree_name, "results": out}


def cmd_bench(cfg, args):
    ev = adapter(args["pack"])
    exe = cfg.build / f"xnnpack-pf-{args['family']}-{args['tag']}-rv64" / "pfharness"
    verdicts, receipt = ev.bench_matrix(cfg, args["family"], exe, args["kernels"],
                                        out=cfg.workdir / args["out"], rounds=args.get("rounds", 15),
                                        delta=args.get("delta"), seed=args.get("seed", 1))
    return {"verdicts": verdicts, "receipt": receipt}


def cmd_dev_compile(cfg, args):
    """Compile one source exactly as the injection would, without building XNNPACK."""
    ev = adapter(args["pack"])
    entry = ev.families()[args["family"]]
    tree = xnnpack.tree(cfg, f"dev-{args['family']}-{args['tag']}", base=ev.base_of(entry))
    src = cfg.workdir / args["source"]
    template = xnnpack.find_ukernel_file(tree, entry.get("template", entry["neon_ukernel"]))
    unit = Path(str(src) + ".unit.c")
    unit.write_text(xnnpack.kernel_source(src.read_text(), "xnn_dev_kernel", template.read_text()))
    obj = Path(str(src) + ".o")
    t = host.in_container(cfg, tree)
    proc = host.run(cfg, f"riscv64-linux-gnu-gcc -O2 -march=rv64gcv -mabi=lp64d -std=c99 -DXNN_ENABLE_RISCV_VECTOR=1 "
                         f"-I{t} -I{t}/include -I{t}/src -I{host.in_container(cfg, harness.pthreadpool_include(cfg))} "
                         f"-c {host.in_container(cfg, unit)} -o {host.in_container(cfg, obj)} 2>&1 && "
                         f"riscv64-linux-gnu-objdump -dr --no-show-raw-insn {host.in_container(cfg, obj)}",
                    check=False, timeout=300)
    if proc.returncode != 0:
        return {"compiled": False, "errors": proc.stdout[-4000:]}
    funcs = profile.parse(proc.stdout)
    insns = funcs.get("xnn_dev_kernel", [])
    p = profile.function_profile(insns)
    calib = calibrate.load(cfg) or json.loads((REPO / "portforge_tt/cpu/data/x280-calib.json").read_text())
    per = "loads" if entry.get("signature") == "transposec" else "vl"
    p["predicted"] = [{"start": hex(s), **(profile.predict(insns, s, e, calib, per=per) or {})}
                      for s, e in profile.loops(insns)]
    return {"compiled": True, "profile": p}


def cmd_dev_check(cfg, args):
    ev = adapter(args["pack"])
    name = f"xnn_{xnnpack.family_token(ev.families()[args['family']]['neon_ukernel'])}_ukernel__rvv_dev"
    tree_name, exe, dropped = ev.prepare_matrix(cfg, args["family"], {name: str(cfg.workdir / args["source"])},
                                                tag=f"dev-{args['tag']}")
    if dropped:
        return {"compiled": False, "errors": dropped}
    check = ev.check_matrix(cfg, args["family"], exe, [name], wheres=("qemu:128", "qemu:1024"),
                            out=cfg.workdir / args["out"])
    return {"compiled": True, "check": check[name]}


def cmd_certify(cfg, args):
    ev = adapter(args["pack"])
    family, tag, name = args["family"], args["tag"], args["kernel"]
    exe = cfg.build / f"xnnpack-pf-{family}-{tag}-rv64" / "pfharness"
    if not exe.exists() or name not in {k["kernel"] for k in harness.kernels(cfg, exe)}:
        # the campaign tree holds only the last evaluated batch: build one with this kernel
        _, exe, dropped = ev.prepare_matrix(cfg, family, {name: str(cfg.workdir / args["source"])},
                                            tag=f"{tag}-certify"[:48])
        if dropped:
            return {"heldout": {"passed": False}, "error": f"does not compile: {dropped}"}
    out = {"heldout": ev.check_matrix(cfg, family, exe, [name], wheres=("qemu:128", "qemu:1024", "x280"),
                                      out=cfg.workdir / args["out"] / "heldout", heldout_seed=args["heldout_seed"])[name]}
    params = ev.PACK / "kernels" / family / "params.json"
    if params.exists() and ev.families()[family].get("smt") != "none":
        out["smt"] = ev.verify(cfg, family, cfg.workdir / args["source"], budget=args.get("smt_budget", 600),
                               out=cfg.workdir / args["out"] / "smt")
    return out


def cmd_context(cfg, args):
    """Single-function sources of the reference and upstream kernels (for SPEC.md)."""
    from portforge_tt.cpu import kernelsrc
    ev = adapter(args["pack"])
    entry = ev.families()[args["family"]]
    spec = ev.regimes_spec(args["family"])
    tree = cfg.build / "trees" / f"pf-{args['family']}-context"
    tree = xnnpack.tree(cfg, tree.name, base=ev.base_of(entry))
    out = {}
    for name in [spec["reference"], *entry["upstream_rvv"]][:args.get("limit", 10)]:
        try:
            out[name] = kernelsrc.extract_function(xnnpack.find_ukernel_file(tree, name).read_text(), name)
        except (FileNotFoundError, ValueError) as error:
            out[name] = f"// unavailable: {error}"
    return {"sources": out}


COMMANDS = {"context": cmd_context, "gates": cmd_gates, "bench": cmd_bench, "dev_compile": cmd_dev_compile, "dev_check": cmd_dev_check,
            "certify": cmd_certify}


def main(argv=None):
    argv = argv or sys.argv[1:]
    command, args_file = argv[0], argv[1]
    args = json.loads(Path(args_file).read_text())
    cfg = config.load()
    try:
        # a pack whose project is not XNNPACK (dav1d: assembly + checkasm) implements
        # the command itself as remote_<command>(cfg, args) in its evaluate.py
        own = getattr(adapter(args["pack"]), f"remote_{command}", None) if "pack" in args else None
        result = {"ok": True, **(own or COMMANDS[command])(cfg, args)}
    except Exception as error:  # report, don't hang the controller
        result = {"ok": False, "error": f"{type(error).__name__}: {error}"[-3000:],
                  "trace": traceback.format_exc()[-3000:]}
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
