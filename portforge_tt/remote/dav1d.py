"""dav1d adapter: meson-built AV1 decoder, per-ISA hand-written assembly kernels.

Differences from the XNNPACK adapter shape:
- kernels are hand-written .S assembly (bitdepth-templated 8bpc/16bpc), so the
  intrinsics-shim SMT gate does not apply: correctness = checkasm (bit-exact
  against the C reference) plus targeted differential drivers; optimization =
  checkasm --bench (its timing source on tt-bh-linux is linux perf).
- a port = a .S file in src/riscv/64/ + one assignment line in
  src/riscv/<dsp>.h (the per-family inline init) - wired per-DSP by hand in the
  disposable worktree; dav1d has no generator/registry machinery to script.
- the family's C reference must exist in the build: build with
  -Dtrim_dsp=false or checkasm reports "reference functions unavailable".
- checkasm itself is a meson subproject (videolan/checkasm, pinned by wrap):
  fetched automatically on reconfigure; needs network on first configure.

Build conventions proven on tt_box (see ports/dav1d-rvv/):
  cross file  tools/riscv64.cross (gcc-14 cross, needs_exe_wrapper)
  setup       meson setup <build> <tree> --cross-file ... -Denable_tests=true
              -Denable_asm=true -Denable_examples=false -Denable_tools=false
              -Ddefault_library=static -Dtrim_dsp=false
  target      ninja -C <build> tests/checkasm   (build_by_default is false)
"""
from pathlib import Path
import shlex

from .config import TOOLS
from . import host, x280

REPO = "dav1d"
CROSS = TOOLS / "riscv64.cross"
MESON_FLAGS = ("-Denable_tests=true -Denable_asm=true -Denable_examples=false "
               "-Denable_tools=false -Ddefault_library=static -Dtrim_dsp=false")
QEMU_CPU = "rv64,v=true,vlen={vlen},elen=64,vext_spec=v1.0"

def tree(cfg, name, base=REPO):
    """A clean git worktree of src/dav1d at the pinned revision under build/trees."""
    spec = host.sources()["repos"][base]
    repo = cfg.src / base
    path = cfg.build / "trees" / name
    if path.exists():
        host.git("reset", "-q", "--hard", spec["revision"], cwd=path)
        host.git("clean", "-fdq", cwd=path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        host.git("worktree", "add", "-q", "--force", "--detach", str(path), spec["revision"], cwd=repo)
    return path


def build(cfg, tree_path, name, targets=("tests/checkasm",), log=print):
    """meson configure (once; reconfigure after edits) + compile. Returns build dir.

    Always configures -Dtrim_dsp=false: checkasm needs the C reference functions."""
    src_c = host.in_container(cfg, tree_path)
    out = cfg.build / f"dav1d-{name}-rv64"
    out_c = host.in_container(cfg, out)
    setup = f"meson setup {out_c} {src_c} --cross-file /tools/riscv64.cross {MESON_FLAGS} " \
            f"> {out_c}.configure.log 2>&1"
    script = (
        f"if [ -f {out_c}/build.ninja ]; then "
        f"meson configure {out_c} >> {out_c}.configure.log 2>&1; "
        f"else {setup}; fi; "
        f"ninja -C {out_c} -j{cfg.jobs} {' '.join(shlex.quote(t) for t in targets)} >> {out_c}.build.log 2>&1")
    host.run(cfg, script, timeout=3600)
    log(f"built dav1d/{name}: {', '.join(targets)}")
    return out


def checkasm_path(cfg, build_dir):
    return Path(build_dir) / "tests" / "checkasm"


def run_checkasm(cfg, build_dir, where="qemu:512", args=(), timeout=3600):
    """Run the checkasm binary; where='qemu:<vlen>' or 'x280'. Returns combined output.

    checkasm writes its progress and summary to stderr, so both streams are
    concatenated."""
    exe = host.in_container(cfg, checkasm_path(cfg, build_dir))
    if where.startswith("qemu"):
        vlen = int(where.split(":")[1])
        script = f"qemu-riscv64 -L /usr/riscv64-linux-gnu -cpu {QEMU_CPU.format(vlen=vlen)} {exe} {' '.join(args)}"
        proc = host.run(cfg, script, timeout=timeout)
        return proc.stdout + "\n" + proc.stderr
    if where == "x280":
        with x280.lease(cfg):
            remote = x280.push(cfg, [checkasm_path(cfg, build_dir)])[0]
            proc = x280.run(cfg, f"taskset -c {cfg.x280.cpu} {remote} {' '.join(args)}",
                            timeout=timeout)
            return proc.stdout + "\n" + proc.stderr
    raise ValueError(f"unknown checkasm target {where!r}")


def coverage(tree_path):
    """RVV coverage per DSP family, parsed from the tree.

    Returns {family: {"riscv_init_lines": n, "has_asm_file": bool}} for the
    families with a src/riscv/<dsp>.h, plus {"missing": [families without any
    riscv support]}. The C-side surface lives in src/<dsp>.h structs (counted
    separately - init lines under-count multi-variant assignments are fine for
    gap analysis).
    """
    tree_path = Path(tree_path)
    riscv = tree_path / "src" / "riscv"
    out = {}
    for header in sorted(riscv.glob("*.h")):
        if header.name in ("asm-offsets.h", "cpu.h"):
            continue
        family = header.stem  # cdef, filmgrain, ipred, itx, mc, pal
        text = header.read_text(errors="ignore")
        out[family] = {"riscv_init_lines": text.count("c->"), "has_asm_file": True}
    known = set(out) | {"loopfilter", "looprestoration", "msac", "refmvs"}
    out["missing"] = sorted(f for f in known if f not in out
                            and not (tree_path / "src" / f"{f}.h").exists())
    return out
