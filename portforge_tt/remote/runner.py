"""Gate runner on the x86 host: python3 -m portforge_tt.remote.runner CMD ARGS.json

The controller runs next to the agent CLIs; every trusted measurement runs here,
from the run's frozen harness (control/cluster.remote uploads it and starts this
module detached), through the pack adapter (ports/<pack>/evaluate.py). Each
command reads a JSON argument file and prints one JSON object on the last line
of stdout. Paths in the arguments are relative to the host's work_dir.

  health       container image tools, pinned sources, and whether the X280 answers
  log          a byte page of an earlier command's stdout/stderr (agents' port_log)

  gates        G0 compile + static profile, G1 gtests (QEMU), G2 bit-exact check
               for a batch of candidates injected into one tree
  bench        G3/G4: one X280 session over the regime matrix
  dev_compile  untrusted worker convenience: compile one source, static profile
  dev_check    untrusted worker convenience: G2 check of one source under QEMU
  certify      G5: held-out regimes under QEMU and on the X280, then the family's
               verify methods (verify/contract.py)

A pack may implement any command itself as remote_<command>(cfg, args) in its
evaluate.py (ports/dav1d-rvv: assembly kernels gated by checkasm); the result has
the same shape.
"""
import importlib.util
import json
from pathlib import Path
import re
import sys
import traceback

from . import calibrate, config, harness, host, profile, regimes, x280, xnnpack

REPO = Path(__file__).resolve().parents[2]  # the harness (or repository) root holding ports/


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def safe_file(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not path.parts:
        raise ValueError('unsafe relative path')
    target = Path(root) / path
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('path escapes workspace')
    return target


def validate_log_request(rpc, stream, offset_bytes, limit_bytes):
    if not isinstance(rpc, str) or not re.fullmatch(r'runs/[A-Za-z0-9_.-]{1,120}/rpc/[a-z_]{1,20}-[0-9a-f]{12}', rpc):
        raise ValueError('invalid job log reference')
    if stream not in ('stdout', 'stderr'):
        raise ValueError('stream must be stdout or stderr')
    if type(offset_bytes) is not int or not 0 <= offset_bytes < 2**63:
        raise ValueError('offset_bytes must be a nonnegative signed 64-bit integer')
    if type(limit_bytes) is not int or not 1 <= limit_bytes <= 65536:
        raise ValueError('limit_bytes must be an integer in 1..65536')


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
    calib = calibrate.load(cfg) or json.loads((REPO / "portforge_tt/remote/data/x280-calib.json").read_text())
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
    calib = calibrate.load(cfg) or json.loads((REPO / "portforge_tt/remote/data/x280-calib.json").read_text())
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
    out["verify"] = [ev.verify(cfg, family, cfg.workdir / args["source"], budget=args.get("verify_budget", 600),
                               out=cfg.workdir / args["out"] / method)
                     for method in ev.families()[family].get("verify", []) if method == "cvc5"]
    return out


def cmd_context(cfg, args):
    """Single-function sources of the reference and upstream kernels (for SPEC.md)."""
    from . import kernelsrc
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


def cmd_health(cfg, args):
    """Is the host ready for gates, and is the board reachable (timing needs it, correctness does not)?"""
    tools = host.probe(cfg)
    board = {"reachable": x280.port_open(cfg)}
    if board["reachable"] and args.get("board_probe"):
        board.update(x280.probe(cfg))
    return {"host": tools, "board": board, "workdir": str(cfg.workdir)}


def cmd_log(cfg, args):
    """One byte page of an earlier command's stdout or stderr; offsets count bytes."""
    rpc, stream = args["rpc"], args.get("stream", "stdout")
    offset, limit = args.get("offset_bytes", 0), args.get("limit_bytes", 65536)
    validate_log_request(rpc, stream, offset, limit)
    path = cfg.workdir / (rpc + (".out" if stream == "stdout" else ".err"))
    if not path.is_file():
        return {"size_bytes": 0, "text": "", "eof": True, "next_offset_bytes": offset}
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(limit)
    return {"size_bytes": size, "text": data.decode(errors="replace"), "eof": offset + len(data) >= size,
            "next_offset_bytes": offset + len(data)}


COMMANDS = {"health": cmd_health, "log": cmd_log, "context": cmd_context, "gates": cmd_gates, "bench": cmd_bench, "dev_compile": cmd_dev_compile, "dev_check": cmd_dev_check,
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
