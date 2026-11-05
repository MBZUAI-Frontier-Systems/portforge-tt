"""dav1d-rvv adapter: the trusted gates of `portforge --port ports/dav1d-rvv --family <f>`.

A candidate is one assembly file (kernel.S) that implements a family's
functions in src/riscv/64/. The harness owns everything else: the per-family
wiring (init header, the C init call, the meson entry) is applied to a pinned
dav1d tree once per slot, and only the .S changes between candidates.

Gates (each command below is remote_<command> for portforge_tt.remote.runner):
  G0  the tree builds; the object defines every required symbol and references
      no undefined symbol (no C fallback, no other dav1d asm); static profile
  G1  checkasm (the project's own test: C reference vs RVV, bit-exact, random
      sizes, edges and data) under QEMU at VLEN 128, 256, 512, 1024
  G2  checkasm on 8 more seeds at VLEN 128 and 1024
  G3/G4  X280: checkasm correctness first, then R rounds of `checkasm --bench`
      over every candidate in a shuffled order; each round pairs the candidate
      with the C reference timed in the same process (baseline = C: these
      families have no upstream RVV)
  G5  held-out seeds (never shown to workers) under QEMU at every VLEN and on
      the X280, and the family's verify methods (verify/contract.py; cbmc: the built machine code
      lifted and proved equal to the C reference by CBMC on bounded sizes)
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import shlex
import shutil
import threading
import time

from portforge_tt.remote import bench as pbench, calibrate, dav1d, host, profile, x280
from portforge_tt.verify import contract

PACK = Path(__file__).resolve().parent
REPO = PACK.parents[1]
VLENS = (128, 256, 512, 1024)
QEMU = "qemu-riscv64 -L /usr/riscv64-linux-gnu -cpu rv64,v=true,vlen={vlen},elen=64,vext_spec=v1.0,zba=true,zbb=true,zbs=false"


def families():
    return json.loads((PACK / "families.json").read_text())["families"]


def regimes_spec(family):
    path = PACK / "kernels" / family / "regimes.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ----------------------------------------------------------------------------- wiring

def _patch(path, old, new):
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"wiring: anchor not found in {path}: {old[:60]!r}")
    path.write_text(text.replace(old, new, 1))


def _meson_add(tree, after, line):
    _patch(tree / "src" / "meson.build", f"'{after}',\n", f"'{after}',\n                '{line}',\n")


LICENSE = "/* portforge wiring (harness-owned): points dav1d's function table at the candidate */\n"


def _wire_refmvs(tree):
    (tree / "src" / "riscv" / "refmvs.h").write_text(LICENSE + """
#include "src/cpu.h"
#include "src/refmvs.h"

decl_splat_mv_fn(dav1d_splat_mv_rvv);

static ALWAYS_INLINE void refmvs_dsp_init_riscv(Dav1dRefmvsDSPContext *const c) {
    const unsigned flags = dav1d_get_cpu_flags();
    if (!(flags & DAV1D_RISCV_CPU_FLAG_V)) return;

    c->splat_mv = dav1d_splat_mv_rvv;
}
""")
    src = tree / "src" / "refmvs.c"
    _patch(src, '#elif ARCH_X86\n#include "src/x86/refmvs.h"\n#endif',
           '#elif ARCH_X86\n#include "src/x86/refmvs.h"\n#elif ARCH_RISCV\n#include "src/riscv/refmvs.h"\n#endif')
    _patch(src, "#elif ARCH_X86\n    refmvs_dsp_init_x86(c);\n#endif",
           "#elif ARCH_X86\n    refmvs_dsp_init_x86(c);\n#elif ARCH_RISCV\n    refmvs_dsp_init_riscv(c);\n#endif")
    _meson_add(tree, "riscv/64/pal.S", "riscv/64/refmvs.S")


def _wire_looprestoration(tree):
    (tree / "src" / "riscv" / "looprestoration.h").write_text(LICENSE + """
#include "src/cpu.h"
#include "src/looprestoration.h"

void BF(dav1d_wiener_filter7, rvv)(pixel *p, const ptrdiff_t stride,
                                   const pixel (*left)[4], const pixel *lpf,
                                   const int w, int h,
                                   const LooprestorationParams *const params,
                                   const enum LrEdgeFlags edges HIGHBD_DECL_SUFFIX);
void BF(dav1d_wiener_filter5, rvv)(pixel *p, const ptrdiff_t stride,
                                   const pixel (*left)[4], const pixel *lpf,
                                   const int w, int h,
                                   const LooprestorationParams *const params,
                                   const enum LrEdgeFlags edges HIGHBD_DECL_SUFFIX);

static ALWAYS_INLINE void loop_restoration_dsp_init_riscv(Dav1dLoopRestorationDSPContext *const c,
                                                          const int bpc)
{
    const unsigned flags = dav1d_get_cpu_flags();
    if (!(flags & DAV1D_RISCV_CPU_FLAG_V)) return;
    (void)bpc;
#if BITDEPTH == 8
    c->wiener[0] = BF(dav1d_wiener_filter7, rvv);
    c->wiener[1] = BF(dav1d_wiener_filter5, rvv);
#endif
}
""")
    src = tree / "src" / "looprestoration_tmpl.c"
    _patch(src, '#elif ARCH_X86\n#include "src/x86/looprestoration.h"\n#endif',
           '#elif ARCH_X86\n#include "src/x86/looprestoration.h"\n#elif ARCH_RISCV\n#include "src/riscv/looprestoration.h"\n#endif')
    _patch(src, "#elif ARCH_X86\n    loop_restoration_dsp_init_x86(c, bpc);\n#endif",
           "#elif ARCH_X86\n    loop_restoration_dsp_init_x86(c, bpc);\n#elif ARCH_RISCV\n"
           "    loop_restoration_dsp_init_riscv(c, bpc);\n#endif")
    _meson_add(tree, "riscv/64/mc.S", "riscv/64/looprestoration.S")


def _wire_sgr3x3(tree):
    (tree / "src" / "riscv" / "looprestoration.h").write_text(LICENSE + """
#include "src/cpu.h"
#include "src/looprestoration.h"

void BF(dav1d_sgr_filter_3x3, rvv)(pixel *p, const ptrdiff_t stride,
                                   const pixel (*left)[4], const pixel *lpf,
                                   const int w, int h,
                                   const LooprestorationParams *const params,
                                   const enum LrEdgeFlags edges HIGHBD_DECL_SUFFIX);

static ALWAYS_INLINE void loop_restoration_dsp_init_riscv(Dav1dLoopRestorationDSPContext *const c,
                                                          const int bpc)
{
    const unsigned flags = dav1d_get_cpu_flags();
    if (!(flags & DAV1D_RISCV_CPU_FLAG_V)) return;
    (void)bpc;
#if BITDEPTH == 8
    c->sgr[1] = BF(dav1d_sgr_filter_3x3, rvv);
#endif
}
""")
    src = tree / "src" / "looprestoration_tmpl.c"
    _patch(src, '#elif ARCH_X86\n#include "src/x86/looprestoration.h"\n#endif',
           '#elif ARCH_X86\n#include "src/x86/looprestoration.h"\n#elif ARCH_RISCV\n#include "src/riscv/looprestoration.h"\n#endif')
    _patch(src, "#elif ARCH_X86\n    loop_restoration_dsp_init_x86(c, bpc);\n#endif",
           "#elif ARCH_X86\n    loop_restoration_dsp_init_x86(c, bpc);\n#elif ARCH_RISCV\n"
           "    loop_restoration_dsp_init_riscv(c, bpc);\n#endif")
    _meson_add(tree, "riscv/64/mc.S", "riscv/64/looprestoration.S")


def _wire_loopfilter_uv(tree):
    (tree / "src" / "riscv" / "loopfilter.h").write_text(LICENSE + """
#include "src/cpu.h"
#include "src/loopfilter.h"

decl_loopfilter_sb_fn(BF(dav1d_lpf_h_sb_uv, rvv));
decl_loopfilter_sb_fn(BF(dav1d_lpf_v_sb_uv, rvv));

static ALWAYS_INLINE void loop_filter_dsp_init_riscv(Dav1dLoopFilterDSPContext *const c) {
    const unsigned flags = dav1d_get_cpu_flags();
    if (!(flags & DAV1D_RISCV_CPU_FLAG_V)) return;
#if BITDEPTH == 8
    c->loop_filter_sb[1][0] = BF(dav1d_lpf_h_sb_uv, rvv);
    c->loop_filter_sb[1][1] = BF(dav1d_lpf_v_sb_uv, rvv);
#endif
}
""")
    src = tree / "src" / "loopfilter_tmpl.c"
    _patch(src, '#elif ARCH_X86\n#include "src/x86/loopfilter.h"\n#endif',
           '#elif ARCH_X86\n#include "src/x86/loopfilter.h"\n#elif ARCH_RISCV\n#include "src/riscv/loopfilter.h"\n#endif')
    _patch(src, "#elif ARCH_X86\n    loop_filter_dsp_init_x86(c);\n#endif",
           "#elif ARCH_X86\n    loop_filter_dsp_init_x86(c);\n#elif ARCH_RISCV\n    loop_filter_dsp_init_riscv(c);\n#endif")
    _meson_add(tree, "riscv/64/mc.S", "riscv/64/loopfilter.S")


def _wire_sgr5x5(tree):
    (tree / "src" / "riscv" / "looprestoration.h").write_text(LICENSE + """
#include "src/cpu.h"
#include "src/looprestoration.h"

void BF(dav1d_sgr_filter_5x5, rvv)(pixel *p, const ptrdiff_t stride,
                                   const pixel (*left)[4], const pixel *lpf,
                                   const int w, int h,
                                   const LooprestorationParams *const params,
                                   const enum LrEdgeFlags edges HIGHBD_DECL_SUFFIX);

static ALWAYS_INLINE void loop_restoration_dsp_init_riscv(Dav1dLoopRestorationDSPContext *const c,
                                                          const int bpc)
{
    const unsigned flags = dav1d_get_cpu_flags();
    if (!(flags & DAV1D_RISCV_CPU_FLAG_V)) return;
    (void)bpc;
#if BITDEPTH == 8
    c->sgr[0] = BF(dav1d_sgr_filter_5x5, rvv);
#endif
}
""")
    src = tree / "src" / "looprestoration_tmpl.c"
    _patch(src, '#elif ARCH_X86\n#include "src/x86/looprestoration.h"\n#endif',
           '#elif ARCH_X86\n#include "src/x86/looprestoration.h"\n#elif ARCH_RISCV\n#include "src/riscv/looprestoration.h"\n#endif')
    _patch(src, "#elif ARCH_X86\n    loop_restoration_dsp_init_x86(c, bpc);\n#endif",
           "#elif ARCH_X86\n    loop_restoration_dsp_init_x86(c, bpc);\n#elif ARCH_RISCV\n"
           "    loop_restoration_dsp_init_riscv(c, bpc);\n#endif")
    _meson_add(tree, "riscv/64/mc.S", "riscv/64/looprestoration.S")


def _wire_sgrmix(tree):
    (tree / "src" / "riscv" / "looprestoration.h").write_text(LICENSE + """
#include "src/cpu.h"
#include "src/looprestoration.h"

void BF(dav1d_sgr_filter_mix, rvv)(pixel *p, const ptrdiff_t stride,
                                   const pixel (*left)[4], const pixel *lpf,
                                   const int w, int h,
                                   const LooprestorationParams *const params,
                                   const enum LrEdgeFlags edges HIGHBD_DECL_SUFFIX);

static ALWAYS_INLINE void loop_restoration_dsp_init_riscv(Dav1dLoopRestorationDSPContext *const c,
                                                          const int bpc)
{
    const unsigned flags = dav1d_get_cpu_flags();
    if (!(flags & DAV1D_RISCV_CPU_FLAG_V)) return;
    (void)bpc;
#if BITDEPTH == 8
    c->sgr[2] = BF(dav1d_sgr_filter_mix, rvv);
#endif
}
""")
    src = tree / "src" / "looprestoration_tmpl.c"
    _patch(src, '#elif ARCH_X86\n#include "src/x86/looprestoration.h"\n#endif',
           '#elif ARCH_X86\n#include "src/x86/looprestoration.h"\n#elif ARCH_RISCV\n#include "src/riscv/looprestoration.h"\n#endif')
    _patch(src, "#elif ARCH_X86\n    loop_restoration_dsp_init_x86(c, bpc);\n#endif",
           "#elif ARCH_X86\n    loop_restoration_dsp_init_x86(c, bpc);\n#elif ARCH_RISCV\n"
           "    loop_restoration_dsp_init_riscv(c, bpc);\n#endif")
    _meson_add(tree, "riscv/64/mc.S", "riscv/64/looprestoration.S")


def _wire_refmvs_tmvs(tree):
    (tree / "src" / "riscv" / "refmvs.h").write_text(LICENSE + """
#include "src/cpu.h"
#include "src/refmvs.h"

decl_load_tmvs_fn(dav1d_load_tmvs_rvv);

static ALWAYS_INLINE void refmvs_dsp_init_riscv(Dav1dRefmvsDSPContext *const c) {
    const unsigned flags = dav1d_get_cpu_flags();
    if (!(flags & DAV1D_RISCV_CPU_FLAG_V)) return;

    c->load_tmvs = dav1d_load_tmvs_rvv;
}
""")
    src = tree / "src" / "refmvs.c"
    _patch(src, '#elif ARCH_X86\n#include "src/x86/refmvs.h"\n#endif',
           '#elif ARCH_X86\n#include "src/x86/refmvs.h"\n#elif ARCH_RISCV\n#include "src/riscv/refmvs.h"\n#endif')
    _patch(src, "#elif ARCH_X86\n    refmvs_dsp_init_x86(c);\n#endif",
           "#elif ARCH_X86\n    refmvs_dsp_init_x86(c);\n#elif ARCH_RISCV\n    refmvs_dsp_init_riscv(c);\n#endif")
    _meson_add(tree, "riscv/64/pal.S", "riscv/64/refmvs.S")


WIRING = {"refmvs": _wire_refmvs, "looprestoration": _wire_looprestoration, "sgr3x3": _wire_sgr3x3,
          "loopfilter_uv": _wire_loopfilter_uv, "sgr5x5": _wire_sgr5x5, "sgrmix": _wire_sgrmix,
          "refmvs_tmvs": _wire_refmvs_tmvs}


# ----------------------------------------------------------------------------- slots and builds

def _slot(cfg, family, slot):
    """A wired tree for one build slot (created once; only the .S changes later)."""
    entry = families()[family]
    name = f"pf-{family}-{slot}"
    path = cfg.build / "trees" / name
    marker = path / ".pf-wired"
    if not marker.exists():
        dav1d.tree(cfg, name)
        # checkasm is a meson subproject fetched at configure time; the build
        # container has no network, so copy the main checkout's copy
        sub = cfg.src / "dav1d" / "subprojects" / "checkasm"
        if sub.exists() and not (path / "subprojects" / "checkasm").exists():
            shutil.copytree(sub, path / "subprojects" / "checkasm", symlinks=True)
        WIRING[entry["wiring"]](path)
        stub = PACK / "kernels" / family / "placeholder.S"
        (path / entry["asm_path"]).write_text(stub.read_text())
        marker.write_text(entry["wiring"] + "\n")
    return path, name


def build(cfg, family, slot, source_text):
    """Install the candidate into a slot and build checkasm. Returns (build_dir, errors)."""
    entry = families()[family]
    path, name = _slot(cfg, family, slot)
    asm = path / entry["asm_path"]
    if not asm.exists() or asm.read_text() != source_text:
        asm.write_text(source_text)
    out = cfg.build / f"dav1d-{name}-rv64"
    src_c, out_c = host.in_container(cfg, path), host.in_container(cfg, out)
    jobs = max(4, cfg.jobs // 4)
    script = (f"if [ ! -f {out_c}/build.ninja ]; then meson setup {out_c} {src_c} --cross-file /tools/riscv64.cross "
              f"{dav1d.MESON_FLAGS} > {out_c}.configure.log 2>&1 || {{ tail -30 {out_c}.configure.log; exit 1; }}; fi; "
              f"ninja -C {out_c} -j{jobs} tests/checkasm > {out_c}.build.log 2>&1 || "
              f"{{ grep -E 'error|Error|undefined' {out_c}.build.log | head -40; tail -15 {out_c}.build.log; exit 1; }}")
    proc = host.run(cfg, script, timeout=3600, check=False)
    if proc.returncode != 0:
        return out, (proc.stdout + proc.stderr)[-4000:] or "build failed"
    return out, None


def _object(out, entry):
    stem = Path(entry["asm_path"]).name
    hits = sorted(out.glob(f"src/**/*{stem}.o"))
    return hits[0] if hits else None


def static_gate(cfg, family, out):
    """G0 checks on the candidate's object: required symbols, no outside references,
    and the static profile of its code."""
    entry = families()[family]
    obj = _object(out, entry)
    if obj is None:
        return {"passed": False, "errors": "object file not found"}, None
    o = host.in_container(cfg, obj)
    undefined = host.run(cfg, f"riscv64-linux-gnu-nm -u {o}", check=False).stdout.split()
    # a family may name read-only dav1d tables the asm can use (sgr: dav1d_sgr_x_by_x)
    undefined = [u for u in undefined if u not in ("U",) and u not in entry.get("allowed_externs", [])]
    defined = host.run(cfg, f"riscv64-linux-gnu-nm --defined-only {o}", check=False).stdout
    missing = [s for s in entry["symbols"] if not re.search(rf"\bT {re.escape(s)}$", defined, re.M)]
    problems = []
    if undefined:
        problems.append("references symbols outside the file (no calls into C or other asm): " + ", ".join(undefined))
    if missing:
        problems.append("required global functions missing: " + ", ".join(missing))
    text = host.run(cfg, f"riscv64-linux-gnu-objdump -d --no-show-raw-insn {o}", check=False).stdout
    funcs = profile.parse(text)
    insns = sorted((i for f in funcs.values() for i in f), key=lambda i: i[0])
    prof = None
    try:
        prof = profile.function_profile(insns)
        calib = calibrate.load(cfg) or json.loads((REPO / "portforge_tt/remote/data/x280-calib.json").read_text())
        prof["predicted"] = [{"start": hex(s), **(profile.predict(insns, s, e, calib) or {})}
                             for s, e in profile.loops(insns)]
        hot = [x for x in prof["predicted"] if x.get("compute_cycles")]
        prof["bound"] = max(hot, key=lambda x: x["compute_cycles"])["bound"] if hot else None
        size = host.run(cfg, f"riscv64-linux-gnu-size -A {o}", check=False).stdout
        m = re.search(r"^\.text\s+(\d+)", size, re.M)
        prof["text_bytes"] = int(m.group(1)) if m else None
        prof["functions"] = sorted(funcs)
    except Exception as error:  # the profile is advisory; never fail G0 on it
        prof = {"error": f"{type(error).__name__}: {error}"}
    if not any(v for f in funcs.values() for v in f if _is_vector(v)):
        problems.append("no vector instructions: a scalar port is not accepted")
    return {"passed": not problems, "errors": "; ".join(problems)}, prof


def _is_vector(insn):
    return insn[1].startswith("v")


# ----------------------------------------------------------------------------- checkasm runs

def checkasm(cfg, out, family, where, seed, repeat=1, bench=False, duration=None, timeout=1800):
    """Run checkasm for the family's test and functions. Returns {passed, tests, failed, rvv_tested, output}."""
    entry = families()[family]
    exe = dav1d.checkasm_path(cfg, out)
    args = f"--test={entry['test']} --function={shlex.quote(entry['functions'])}"
    if repeat > 1:
        args += f" --repeat={repeat}"
    if bench:
        args += f" --bench --json --duration={duration or 2000}"
    if where.startswith("qemu"):
        vlen = int(where.split(":")[1])
        proc = host.run(cfg, f"{QEMU.format(vlen=vlen)} {host.in_container(cfg, exe)} {args} {seed}",
                        timeout=timeout, check=False)
    else:
        raise ValueError(where)
    return parse_checkasm(proc.stdout + "\n" + proc.stderr, proc.returncode)


def parse_checkasm(text, code):
    rvv = re.search(r"^RVV:\n((?: - .*\n?)+)", text, re.M)
    rvv_lines = rvv.group(1).splitlines() if rvv else []
    failed = len(re.findall(r"\[FAILED\]", text))
    m = re.search(r"checkasm: all (\d+) tests passed", text)
    tests = int(m.group(1)) if m else None
    passed = code == 0 and failed == 0 and bool(m) and any("[OK]" in l for l in rvv_lines)
    return {"passed": passed, "tests": tests, "failed": failed, "rvv_tested": len(rvv_lines), "exit": code,
            "output": text[-3000:] if not passed else ""}


def _row(r):
    """The devtools/report row shape for one checkasm run."""
    return {"cases": r["tests"] or 0, "mismatched_cases": r["failed"] if r["passed"] or r["failed"] else 1,
            "guard_violations": 0, "max_err": None,
            "error": None if r["passed"] else (f"exit {r['exit']}" + ("" if r["rvv_tested"] else
                                                                       "; no RVV function was tested")),
            "output": r["output"]}


_REGISTRY_LOCK = threading.Lock()  # remote_gates builds candidates in parallel threads


def _registry(cfg, family):
    return cfg.build / f"dav1d-pf-{family}-binaries.json"


def _record_binary(cfg, family, name, out):
    """Keep a copy of a candidate's checkasm for the bench and certify steps."""
    dest = cfg.build / "pf-bins" / family / name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dav1d.checkasm_path(cfg, out), dest / "checkasm")
    reg = _registry(cfg, family)
    with _REGISTRY_LOCK:
        data = json.loads(reg.read_text()) if reg.exists() else {}
        data[name] = str(dest / "checkasm")
        reg.write_text(json.dumps(data, indent=1) + "\n")
    return dest / "checkasm"


# ----------------------------------------------------------------------------- runner commands (remote_<command>)

def remote_gates(cfg, args):
    family = args["family"]
    names = list(args["kernels"])
    sources = {n: (cfg.workdir / rel).read_text() for n, rel in args["kernels"].items()}
    out = {}
    seed = int(args.get("seed", 1))

    def one(k_name):
        k, name = k_name
        build_dir, errors = build(cfg, family, f"g{k}", sources[name])
        if errors:
            return name, {"G0": {"passed": False, "errors": errors}}
        g0, prof = static_gate(cfg, family, build_dir)
        res = {"G0": g0, "profile": prof}
        if not g0["passed"]:
            return name, res
        runs = {f"qemu:{v}": checkasm(cfg, build_dir, family, f"qemu:{v}", seed) for v in VLENS}
        res["G1"] = {"passed": all(r["passed"] for r in runs.values()),
                     "detail": {w: _row(r) for w, r in runs.items()}}
        if not res["G1"]["passed"]:
            return name, res
        extra = {f"qemu:{v}": checkasm(cfg, build_dir, family, f"qemu:{v}", seed + 1000, repeat=8)
                 for v in (128, 1024)}
        res["G2"] = {"passed": all(r["passed"] for r in extra.values()),
                     "detail": {w: _row(r) for w, r in extra.items()}}
        if res["G2"]["passed"]:
            res["binary"] = str(_record_binary(cfg, family, name, build_dir))
        return name, res

    with ThreadPoolExecutor(max_workers=min(4, len(names))) as pool:
        for name, res in pool.map(one, enumerate(names)):
            out[name] = res
    return {"tree": f"pf-{family}", "results": out}


def _x280_run(cfg, remote, args, timeout=900):
    proc = x280.run(cfg, f"taskset -c {cfg.x280.cpu} {remote} {args}", timeout=timeout, check=False)
    return proc.stdout, proc.stderr, proc.returncode


def _wait_idle(cfg, wait_seconds=1800, log=print):
    deadline, delay = time.time() + wait_seconds, 15
    while True:
        state = x280.idle(cfg)
        if state.get("ok"):
            return state
        if time.time() > deadline:
            raise pbench.Contended(f"board busy for {wait_seconds}s")
        log(f"board busy ({state}); waiting {delay}s")
        time.sleep(delay)
        delay = min(delay * 2, 240)


def bench_session(cfg, family, binaries, out_dir, rounds=10, seed=1, duration=None):
    """X280: correctness then R shuffled rounds of checkasm --bench. Returns (rows per name, receipt)."""
    entry = families()[family]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = duration or entry.get("bench_duration_us", 2000)
    rng = random.Random(seed)
    rows = {n: [] for n in binaries}
    correct, log = {}, []
    with x280.lease(cfg):
        before = _wait_idle(cfg)
        t0 = time.time()
        remote = {}
        for k, (name, path) in enumerate(binaries.items()):
            rdir = f"{cfg.x280.remote_dir}/pf-{family}-{k}"
            remote[name] = x280.push(cfg, [path], rdir)[0]
        test_args = f"--test={entry['test']} --function={shlex.quote(entry['functions'])}"
        for name, rpath in remote.items():
            ok = True
            for s in (seed % 2_000_000_000, (seed + 1) % 2_000_000_000):
                o, e, code = _x280_run(cfg, rpath, f"{test_args} {s}")
                r = parse_checkasm(o + "\n" + e, code)
                ok = ok and r["passed"]
                if not r["passed"]:
                    log.append({"name": name, "seed": s, "output": r["output"][-1500:]})
            correct[name] = ok
        for rnd in range(rounds):
            order = list(remote)
            rng.shuffle(order)
            for name in order:
                # checkasm takes a 32-bit seed: an overflowing one only prints its usage
                o, e, code = _x280_run(cfg, remote[name], f"{test_args} --bench --json --duration={duration} "
                                                          f"{(seed * 7919 + rnd * 104729) % 2_000_000_000}")
                try:
                    data = json.loads(o[o.index("{"):])
                except ValueError:
                    log.append({"name": name, "round": rnd, "error": (o + e)[-800:]})
                    continue
                for fn, info in data.get("functions", {}).items():
                    vers = info.get("versions", {})
                    if "c" not in vers or "rvv" not in vers:
                        continue
                    for slot, ver in ((0, "c"), (1, "rvv")):
                        cyc = vers[ver]["adjustedCycles"]["median"]
                        rows[name].append({"regime": fn, "shape": "call", "slot": slot, "round": rnd,
                                           "ns": cyc, "cycles": cyc, "instret": None, "elems": 1})
        after = x280.idle(cfg)
    for name in rows:
        cells = {r["regime"] for r in rows[name]}
        for fn in cells:
            for slot in (0, 1):
                rows[name].append({"regime": fn, "shape": "call", "slot": slot, "hash": 1,
                                   "reference_hash": 1 if (slot == 0 or correct.get(name)) else 0})
    receipt = {"board": x280.board_id(cfg), "rounds": rounds, "seed": seed, "duration_us": duration,
               "idle_before": before, "idle_after": after, "seconds": round(time.time() - t0, 1),
               "correct_on_x280": correct, "problems": log[:20]}
    (out_dir / "rows.json").write_text(json.dumps(rows) + "\n")
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    return rows, receipt


def remote_bench(cfg, args):
    family = args["family"]
    reg = json.loads(_registry(cfg, family).read_text())
    binaries = {n: reg[n] for n in args["kernels"]}
    delta = args.get("delta") or regimes_spec(family).get("noise_floor", 0.02)
    rows, receipt = bench_session(cfg, family, binaries, cfg.workdir / args["out"], rounds=args.get("rounds", 10),
                                  seed=args.get("seed", 1) % 2_000_000_000)
    empty = [n for n, r in rows.items() if not any("ns" in x for x in r)]
    if empty:   # no timing at all is a failed session, never a measurement
        raise RuntimeError(f"no timing rows for {empty}: {json.dumps(receipt['problems'])[:1500]}")
    verdicts = {}
    for name, r in rows.items():
        v = pbench.analyze(r, ["c", name], 1, [0], delta=delta, reference_cell=regimes_spec(family).get("reference_cell"))
        v["views"] = {}
        verdicts[name] = v
    (cfg.workdir / args["out"] / "verdicts.json").write_text(json.dumps(verdicts, indent=1, default=str) + "\n")
    return {"verdicts": verdicts, "receipt": receipt}


def remote_dev_compile(cfg, args):
    src = (cfg.workdir / args["source"]).read_text()
    out, errors = build(cfg, args["family"], f"dev-{args['tag']}"[:40], src)
    if errors:
        return {"compiled": False, "errors": errors}
    g0, prof = static_gate(cfg, args["family"], out)
    if not g0["passed"]:
        return {"compiled": False, "errors": g0["errors"]}
    return {"compiled": True, "profile": prof}


def remote_dev_check(cfg, args):
    src = (cfg.workdir / args["source"]).read_text()
    out, errors = build(cfg, args["family"], f"dev-{args['tag']}"[:40], src)
    if errors:
        return {"compiled": False, "errors": errors}
    g0, _ = static_gate(cfg, args["family"], out)
    if not g0["passed"]:
        return {"compiled": False, "errors": g0["errors"]}
    runs = {f"qemu:{v}": checkasm(cfg, out, args["family"], f"qemu:{v}", 7) for v in (128, 1024)}
    check = {"passed": all(r["passed"] for r in runs.values())}
    check.update({w: _row(r) for w, r in runs.items()})
    return {"compiled": True, "check": check}


def remote_context(cfg, args):
    """Sources a worker gets (context/): the C reference, other ISAs' asm, the test, the macros."""
    entry = families()[args["family"]]
    tree = cfg.src / "dav1d"
    out = {}
    for item in entry["context"]:
        text = (tree / item["path"]).read_text(errors="replace")
        if item.get("lines"):
            a, b = item["lines"]
            text = f"// {item['path']} lines {a}-{b}\n" + "\n".join(text.splitlines()[a - 1:b])
        out[item["name"]] = text
    wiring = PACK / "kernels" / args["family"] / "WIRING.h"
    if wiring.exists():
        out["wiring.h"] = wiring.read_text()
    return {"sources": out}


def remote_certify(cfg, args):
    """G5: held-out seeds at every VLEN and on the X280, then the formal check."""
    family = args["family"]
    src = (cfg.workdir / args["source"]).read_text()
    build_dir, errors = build(cfg, family, "cert", src)
    if errors:
        return {"heldout": {"passed": False}, "error": f"does not build: {errors[-800:]}"}
    seed = int(args["heldout_seed"]) % 2_000_000_000
    held = {f"qemu:{v}": _row(checkasm(cfg, build_dir, family, f"qemu:{v}", seed, repeat=16)) for v in VLENS}
    passed = all(r["mismatched_cases"] == 0 and not r["error"] for r in held.values())
    entry = families()[family]
    with x280.lease(cfg):
        _wait_idle(cfg)
        rpath = x280.push(cfg, [dav1d.checkasm_path(cfg, build_dir)], f"{cfg.x280.remote_dir}/pf-{family}-cert")[0]
        o, e, code = _x280_run(cfg, rpath, f"--test={entry['test']} --function={shlex.quote(entry['functions'])} "
                                           f"--repeat=16 {seed}", timeout=1800)
    x = parse_checkasm(o + "\n" + e, code)
    held["x280"] = _row(x)
    passed = passed and x["passed"]
    result = {"heldout": {"passed": passed, **held}}
    result["verify"] = [contract.run(method, cfg, family, entry, {"build_dir": build_dir},
                                     cfg.workdir / args["out"] / method, args.get("verify_budget", 7200))
                        for method in entry.get("verify", [])]
    return result
