"""XNNPACK as a test and benchmark harness for RVV microkernels.

XNNPACK lists every microkernel of a family in a registry (``src/<dir>/<family>.inc``,
one ``XNN_UKERNEL(arch, name, tile..., params, init)`` line each). Its unit
tests and benchmarks expand those registries, so registering a new kernel is
all it takes to get the family's full upstream test matrix and benchmark for it,
which is also exactly what an upstream PR does.

Flow:  tree() -> inject() -> build() -> run_tests() / run_bench()
Trees are git worktrees under <workdir>/build/trees/<name>, reset on reuse, so
the pinned checkouts under src/ stay untouched.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex

from . import host, x280
from .kernelsrc import strip_local_xnn_types

RVV_ARCH = "xnn_arch_riscv_vector"
RVV_GUARD = "#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR"
CMAKE_LIST = "cmake/gen/rvv_microkernels.cmake"
CMAKE_ANCHOR = "SET(NON_PROD_RVV_MICROKERNEL_SRCS"
REGISTRY_LINE = re.compile(r"^\s*(XNN_\w+)\((.*)\)\s*$")
KERNEL_DEF = re.compile(r"\bvoid\s+((?:xnn_|test_)\w*)\s*\(")


@dataclass
class Injection:
    """Register `source` in `inc` as ukernel `name`, cloning `template`'s registry line.

    The template is usually the kernel the new one was ported from (the Neon
    variant), so tile sizes and params match what that kernel was tested with.
    """
    source: Path
    inc: str              # registry path relative to the tree, e.g. src/qs8-vadd/qs8-vadd-minmax.inc
    template: str         # existing ukernel whose registry line is cloned
    name: str             # new ukernel name, e.g. xnn_qs8_vadd_minmax_ukernel__rvv_saltyrn
    arch: str = RVV_ARCH
    overrides: dict = None  # registry argument index -> replacement (tile sizes for VLA kernels)


def registry_lines(inc_text):
    """[(macro, [args...], raw_line)] for every registry entry in a .inc file."""
    entries = []
    for line in inc_text.splitlines():
        m = REGISTRY_LINE.match(line)
        if m:
            entries.append((m.group(1), [a.strip() for a in _split_args(m.group(2))], line))
    return entries


def _split_args(text):
    args, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            args.append(cur)
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    return args + [cur]


def clone_registry_line(inc_text, template, name, arch=RVV_ARCH, overrides=None):
    for macro, args, _ in registry_lines(inc_text):
        if len(args) > 1 and args[1] == template:
            args = [arch, name] + args[2:]
            for index, value in (overrides or {}).items():
                args[int(index)] = value
            return f"{macro}({', '.join(args)})"
    raise ValueError(f"{template} is not registered in this .inc")


def kernel_source(kernel_text, name, template_text):
    """The kernel renamed to `name`, with the template kernel's XNNPACK includes."""
    kernel_text, _ = strip_local_xnn_types(kernel_text)
    m = KERNEL_DEF.search(kernel_text)
    if not m:
        raise ValueError("no xnn_* or test_* function definition in kernel source")
    body = kernel_text[:m.start(1)] + name + kernel_text[m.end(1):]
    includes = ["#include <riscv_vector.h>"]
    for line in template_text.splitlines():
        s = line.strip()
        if s.startswith("#include") and not re.search(r"arm|neon|immintrin|x86", s):
            includes.append(s)
    return "// Registered by portforge-cpu; not part of upstream XNNPACK.\n" + \
        "\n".join(dict.fromkeys(includes)) + "\n\n" + body


def insert_registry_lines(inc_text, template, lines):
    """Add a guarded block of registry lines just before the #if section holding
    `template`. Registries can wrap entries in file-local macros (XNN_QUANTIZED)
    that are #undef'd at the end of the file, so appending at the end breaks them."""
    rows = inc_text.splitlines()
    at = next((i for i, row in enumerate(rows)
               if REGISTRY_LINE.match(row) and re.search(rf"[(,]\s*{re.escape(template)}\s*,", row)), None)
    if at is None:
        raise ValueError(f"{template} is not registered in this .inc")
    while at > 0 and not rows[at].lstrip().startswith("#if"):
        at -= 1
    if not rows[at].lstrip().startswith("#if"):  # template is not in a guarded section
        at = next(i for i, row in enumerate(rows) if REGISTRY_LINE.match(row))
    block = ["// portforge-cpu injections", RVV_GUARD, *lines, "#endif", ""]
    return "\n".join(rows[:at] + block + rows[at:]) + "\n"


def find_ukernel_file(tree, ukernel):
    pattern = re.compile(rf"\bvoid\s+{re.escape(ukernel)}\s*\(")
    for path in sorted((Path(tree) / "src").rglob("*.c")):
        try:
            if pattern.search(path.read_text(errors="ignore")):
                return path
        except OSError:
            continue
    raise FileNotFoundError(f"no source defines {ukernel}")


def inject(tree, injections):
    """Apply injections to a tree; returns the list of new source paths (relative)."""
    tree = Path(tree)
    cmake = tree / CMAKE_LIST
    cmake_text = cmake.read_text()
    added = []
    by_inc = {}
    for inj in injections:
        inc_path = tree / inj.inc
        line = clone_registry_line(inc_path.read_text(), inj.template, inj.name, inj.arch, inj.overrides)
        by_inc.setdefault(inj.inc, (inj.template, []))[1].append(line)
        template_file = find_ukernel_file(tree, inj.template)
        rel = Path(inj.inc).parent / "portforge" / f"{inj.name}.c"
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(kernel_source(Path(inj.source).read_text(), inj.name,
                                              template_file.read_text()))
        added.append(str(rel))
    for inc, (template, lines) in by_inc.items():
        path = tree / inc
        path.write_text(insert_registry_lines(path.read_text(), template, lines))
    if CMAKE_ANCHOR not in cmake_text:
        raise ValueError(f"{CMAKE_LIST} has no {CMAKE_ANCHOR}")
    cmake.write_text(cmake_text.replace(CMAKE_ANCHOR, CMAKE_ANCHOR + "".join(f"\n  {a}" for a in added), 1))
    return added


def tree(cfg, name, base="xnnpack-saltyrn"):
    """A clean worktree of src/<base> at build/trees/<name>."""
    spec = host.sources()["repos"][base]
    repo = cfg.src / spec.get("worktree_of", base)
    path = cfg.build / "trees" / name
    if path.exists():
        host.git("reset", "-q", "--hard", spec["revision"], cwd=path)
        host.git("clean", "-fdq", cwd=path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        host.git("worktree", "add", "-q", "--force", "--detach", str(path), spec["revision"], cwd=repo)
    return path


def targets_for(tree, inc):
    """Test and bench targets whose sources include the registry `inc`."""
    tree = Path(tree)
    found = {"test": [], "bench": []}
    for kind in found:
        for cc in sorted((tree / kind).glob("*.cc")):
            if f'"{inc}"' in cc.read_text(errors="ignore"):
                found[kind].append(f"{cc.stem}-{kind}")
    return found


class BuildError(RuntimeError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("XNNPACK build failed:\n" + "\n".join(errors[:40]))


def build_dir(cfg, name):
    return cfg.build / f"xnnpack-{name}-rv64"


def build(cfg, tree_path, name, targets=None, log=print):
    """Configure (once) and build the riscv64 tree; targets None builds everything."""
    out = build_dir(cfg, name)
    src_c, out_c = host.in_container(cfg, tree_path), host.in_container(cfg, out)
    configure = ""
    if not (out / "build.ninja").exists():
        configure = (f"cmake -S {src_c} -B {out_c} -G Ninja -DCMAKE_BUILD_TYPE=Release "
                     f"-DCMAKE_TOOLCHAIN_FILE={src_c}/cmake/riscv64.toolchain "
                     "-DXNNPACK_ENABLE_RISCV_FP16_VECTOR=OFF -DXNNPACK_BUILD_TESTS=ON "
                     "-DXNNPACK_BUILD_BENCHMARKS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache "
                     "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache > " + f"{out_c}.configure.log 2>&1 && ")
        out.mkdir(parents=True, exist_ok=True)
    tlist = " ".join(shlex.quote(t) for t in (targets or []))
    script = (f"export CCACHE_DIR=/pf/build/.ccache; {configure}"
              f"ninja -k 0 -C {out_c} -j{cfg.jobs} {tlist} > {out_c}.build.log 2>&1")
    proc = host.run(cfg, script, check=False, timeout=7200)
    if proc.returncode != 0:
        log_path = Path(f"{out}.build.log")
        text = log_path.read_text(errors="ignore") if log_path.exists() else proc.stderr
        errors = [line for line in text.splitlines() if "error:" in line]
        raise BuildError(errors or text.splitlines()[-40:])
    log(f"built {name}: {len(targets or [])} targets" if targets else f"built {name}: all")
    return out


def binary(cfg, name, target):
    kind = "test" if target.endswith("-test") else "bench"
    return build_dir(cfg, name) / kind / target


# ----------------------------------------------------------------------------- tests

def parse_gtest(report):
    """{test_name: "pass"|"fail"|"skip"} plus failure messages from a gtest JSON report."""
    status, messages = {}, {}
    for suite in report.get("testsuites", []):
        for case in suite.get("testsuite", []):
            name = f"{suite['name']}.{case['name']}"
            if case.get("result") == "SKIPPED" or case.get("status") == "NOTRUN":
                status[name] = "skip"
            elif case.get("failures"):
                status[name] = "fail"
                messages[name] = case["failures"][0].get("failure", "")[:600]
            else:
                status[name] = "pass"
    return status, messages


def run_tests(cfg, name, target, gtest_filter, where, out_dir):
    """Run one test binary under QEMU (where='qemu:<vlen>') or on the X280 ('x280')."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = binary(cfg, name, target)
    report = out_dir / f"{target}.{where.replace(':', '-')}.json"
    if where.startswith("qemu"):
        vlen = int(where.split(":")[1])
        script = (f"cd {host.in_container(cfg, exe.parent)} && {host.qemu_prefix(vlen)} ./{target} "
                  f"--gtest_filter={shlex.quote(gtest_filter)} "
                  f"--gtest_output=json:{host.in_container(cfg, report)} > /dev/null 2>&1; echo rc=$?")
        host.run(cfg, script, check=False, timeout=3600)
    elif where == "x280":
        with x280.lease(cfg):
            remote = x280.push(cfg, [exe])[0]
            proc = x280.run(cfg, f"{remote} --gtest_filter={shlex.quote(gtest_filter)} "
                                 f"--gtest_output=json:{remote}.json > /dev/null 2>&1; cat {remote}.json",
                            timeout=3600, check=False)
            report.write_text(proc.stdout)
    else:
        raise ValueError(f"unknown test target {where!r}")
    try:
        data = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError):
        return {"where": where, "target": target, "error": "no gtest report (crash or timeout)",
                "passed": 0, "failed": 0, "status": {}, "messages": {}}
    status, messages = parse_gtest(data)
    return {"where": where, "target": target, "passed": sum(s == "pass" for s in status.values()),
            "failed": sum(s == "fail" for s in status.values()), "status": status, "messages": messages}


# ----------------------------------------------------------------------------- benchmarks

def parse_bench(report):
    """{benchmark_name: {"ns": median real time, "cv": coefficient of variation, ...}}."""
    rows = {}
    for b in report.get("benchmarks", []):
        if b.get("run_type") != "aggregate":
            continue
        base = b.get("run_name") or b["name"].rsplit("_", 1)[0]
        scale = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}[b.get("time_unit", "ns")]
        entry = rows.setdefault(base, {})
        agg = b.get("aggregate_name")
        if agg == "median":
            entry["ns"] = b["real_time"] * scale
            entry.update({k: v for k, v in b.items() if k in ("elements", "bytes", "cpufreq")})
        elif agg == "cv":
            entry["cv"] = b["real_time"]
    return rows


def run_bench(cfg, name, target, regex, out_dir, repetitions=5, min_time="0.2s"):
    """Run a benchmark binary on the X280, pinned to cfg.x280.cpu; returns parse_bench rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = binary(cfg, name, target)
    report = out_dir / f"{target}.x280.json"
    with x280.lease(cfg):
        remote = x280.push(cfg, [exe])[0]
        proc = x280.run(cfg, f"taskset -c {cfg.x280.cpu} {remote} --benchmark_filter={shlex.quote(regex)} "
                             f"--benchmark_repetitions={repetitions} --benchmark_min_time={min_time} "
                             f"--benchmark_report_aggregates_only=true --benchmark_format=json 2>/dev/null",
                        timeout=7200, check=False)
    report.write_text(proc.stdout)
    try:
        return parse_bench(json.loads(proc.stdout))
    except json.JSONDecodeError:
        return {}


# ----------------------------------------------------------------------------- bench analytics

BENCH_UKERNEL = re.compile(r"(xnn_\w+?)(?=/|$)")


def family_token(ukernel):
    """xnn_qs8_vadd_minmax_ukernel__... -> qs8_vadd_minmax (matches bench and test names)."""
    return re.sub(r"^xnn_", "", ukernel.split("_ukernel")[0])


def ukernel_name(neon_ukernel, suffix="rvv_saltyrn"):
    """xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16 -> ...__rvv_<suffix>"""
    return neon_ukernel.rsplit("__", 1)[0] + "__" + suffix


def split_bench_name(name):
    """'vbinary/xnn_..._ukernel__rvv_u2v/N:8192/real_time' -> (ukernel, '/N:8192/real_time')."""
    m = BENCH_UKERNEL.search(name)
    if not m:
        return None, name
    return m.group(1), name[m.end():]


def geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    product = 1.0
    for v in values:
        product *= v
    return product ** (1 / len(values))


def compare_bench(rows, ukernel, token, exclude="saltyrn"):
    """Speedup of `ukernel` over the hand-written RVV variants of its family, per shape.

    rows: parse_bench output. Names containing `exclude` (e.g. other candidate
    suffixes of the same family) are ignored unless they are `ukernel` itself.
    Returns the geomean over shapes of (fastest upstream RVV time / our time),
    and the same against each variant.
    """
    ours, theirs = {}, {}
    for name, row in rows.items():
        if "ns" not in row:
            continue
        uk, shape = split_bench_name(name)
        if uk == ukernel:
            ours[shape] = row["ns"]
        elif (uk and uk.startswith(f"xnn_{token}_ukernel") and "rvv" in uk
              and (not exclude or exclude not in uk)):
            theirs.setdefault(uk, {})[shape] = row["ns"]
    best, per_variant = [], {uk: [] for uk in theirs}
    for shape, ns in ours.items():
        peers = {uk: t[shape] for uk, t in theirs.items() if shape in t}
        if not peers:
            continue
        best.append(min(peers.values()) / ns)
        for uk, t in peers.items():
            per_variant[uk].append(t / ns)
    return {"speedup_vs_best": geomean(best), "shapes": len(best),
            "vs": {uk: geomean(v) for uk, v in per_variant.items() if v},
            "ours_ns": ours, "upstream_ns": theirs}
