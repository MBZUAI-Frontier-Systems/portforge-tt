"""Build and run pfharness (tools/pfharness/main.cc) for one XNNPACK family.

The harness is compiled against an injected XNNPACK tree (xnnpack.tree/inject/
build), so it sees every registered kernel of the family, candidates included,
and links the same microkernel library the family's own bench target does.
It is harness-owned: candidates never see or change it, and its sha goes into
every receipt.
"""
import hashlib
import json
from pathlib import Path
import shlex

from . import host, regimes, x280, xnnpack
from .config import TOOLS

SOURCE = TOOLS / "pfharness" / "main.cc"
BENCH_LINK_OBJECTS = ("xnnpack-microparams-init", "xnnpack-hardware-config", "xnnpack-logging")


def sha():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def pthreadpool_include(cfg):
    """pthreadpool's include dir from any XNNPACK build (xnnpack.h needs the header)."""
    for path in sorted(cfg.build.glob("xnnpack-*-rv64/pthreadpool-source/include")):
        return path
    raise FileNotFoundError("no XNNPACK build has fetched pthreadpool yet; build any family first")


def decl_header(tree, inc):
    """The src/xnnpack/*.h header that declares the family's ukernels (it
    includes the registry), e.g. src/xnnpack/vcvt.h for qs8-vcvt."""
    for header in sorted((Path(tree) / "src" / "xnnpack").glob("*.h")):
        if f'"{inc}"' in header.read_text(errors="ignore"):
            return f"src/xnnpack/{header.name}"
    raise FileNotFoundError(f"no src/xnnpack header includes {inc}")


def build(cfg, tree_name, entry, binary=None):
    """Compile pfharness for the family into the tree's build dir; returns its path.

    Needs the tree built (xnnpack.build) with the family's bench target, which
    produces the microkernel library and the init/hardware-config/logging
    objects linked here.
    """
    tree = cfg.build / "trees" / tree_name
    out_dir = xnnpack.build_dir(cfg, tree_name)
    exe = out_dir / "pfharness"
    binary = entry.get("signature") == "vbinary" if binary is None else binary
    tree_c, out_c = host.in_container(cfg, tree), host.in_container(cfg, out_dir)
    objects = " ".join(f"$(find {out_c}/CMakeFiles/{t}.dir -name '*.o')" for t in BENCH_LINK_OBJECTS)
    script = (
        f"riscv64-linux-gnu-g++ -O2 -std=c++17 -static -w "
        f"-DPF_INC='\"{entry['inc']}\"' -DPF_DECL='\"{decl_header(tree, entry['inc'])}\"' "
        f"-DPF_BINARY={int(bool(binary))} -DPF_TRANSPOSE={int(entry.get('signature') == 'transposec')} "
        f"-DXNN_ENABLE_RISCV_VECTOR=1 -DXNN_ENABLE_RISCV_FP16_VECTOR=0 "
        f"-I{out_c}/pthreadpool-source/include -I{tree_c} -I{tree_c}/include -I{tree_c}/src "
        f"/tools/pfharness/main.cc -o {out_c}/pfharness "
        f"-Wl,--start-group {objects} {out_c}/libxnnpack-microkernels-all.a {out_c}/libxnnpack-microkernels-prod.a "
        f"{out_c}/cpuinfo/libcpuinfo.a {out_c}/pthreadpool/libpthreadpool.a -Wl,--end-group -lm -lpthread "
        f"> {out_c}/pfharness.build.log 2>&1 || {{ tail -40 {out_c}/pfharness.build.log; exit 1; }}")
    proc = host.run(cfg, script, check=False, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"pfharness build failed:\n{proc.stdout[-4000:]}{proc.stderr[-2000:]}")
    return exe


def _rows(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def run(cfg, exe, args, where, files=(), timeout=3600):
    """Run `pfharness args...` under QEMU ('qemu:<vlen>') or on the X280 ('x280').

    `files` are local paths referenced by args; they are copied next to the
    binary on the board. Returns (returncode, stdout, stderr). On the X280 the
    caller is responsible for the lease and idle checks (bench.py does both).
    """
    if where.startswith("qemu:"):
        vlen = int(where.split(":")[1])
        cmd = f"{host.qemu_prefix(vlen)} {host.in_container(cfg, exe)} " + " ".join(
            shlex.quote(host.in_container(cfg, a) if Path(str(a)).is_absolute() and Path(str(a)).exists()
                        else str(a)) for a in args)
        proc = host.run(cfg, cmd, check=False, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    if where == "x280":
        remote = x280.push(cfg, [exe, *files])
        mapping = {str(Path(f)): r for f, r in zip(files, remote[1:])}
        cmd = shlex.quote(remote[0]) + " " + " ".join(shlex.quote(mapping.get(str(a), str(a))) for a in args)
        proc = x280.run(cfg, f"env -i taskset -c {cfg.x280.cpu} {cmd}", timeout=timeout, check=False)
        return proc.returncode, proc.stdout, proc.stderr
    raise ValueError(f"unknown target {where!r}")


def write_regimes(spec, regime_list, path):
    Path(path).write_text(regimes.flat(spec, regime_list))
    return Path(path)


def kernels(cfg, exe, where="qemu:128"):
    code, out, err = run(cfg, exe, ["list"], where)
    if code != 0:
        raise RuntimeError(f"pfharness list failed: {err[-800:]}")
    return _rows(out)


def dump(cfg, exe, regimes_file, names, where="qemu:128"):
    code, out, err = run(cfg, exe, ["dump", regimes_file, *names], where, files=[regimes_file])
    if code != 0:
        raise RuntimeError(f"pfharness dump failed: {err[-800:]}")
    return _rows(out)


def check(cfg, exe, regimes_file, batches, reference, candidates, where, seed=1):
    """{candidate: row} from the bit-exact check against `reference`."""
    code, out, err = run(cfg, exe, ["check", regimes_file, ",".join(map(str, batches)), str(seed),
                                    reference, *candidates], where, files=[regimes_file])
    rows = {r["kernel"]: r for r in _rows(out)}
    for name in candidates:
        if name not in rows:
            rows[name] = {"kernel": name, "error": f"no result (exit {code}): {err[-400:]}"}
    return rows


def bench_raw(cfg, exe, regimes_file, shapes, rounds, target_ns, seed, reference, names, where="x280"):
    """Raw timing rows and post-timing output hashes (pfharness bench)."""
    code, out, err = run(cfg, exe, ["bench", regimes_file, ",".join(map(str, shapes)), str(rounds),
                                    str(target_ns), str(seed), reference, *names],
                         where, files=[regimes_file], timeout=7200)
    rows = _rows(out)
    if code != 0 or not rows:
        raise RuntimeError(f"pfharness bench failed (exit {code}): {err[-800:]}")
    return rows, err
