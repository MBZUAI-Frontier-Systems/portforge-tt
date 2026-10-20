"""Cross-ISA differential runs: the same driver against a Neon and an RVV kernel.

A driver is a small C file with a `main` that enumerates cases, calls
`KERNEL(...)` and prints one line per case: the inputs, then the output bytes
in hex. The driver is compiled once per side with KERNEL bound to that side's
function, Neon under qemu-aarch64 and RVV under qemu-riscv64 (any VLEN) or on
the X280; the per-case lines are compared. This answers "on which inputs do they
differ" concretely, where the solver only says "they differ".

Driver conventions:
  - include "xdiff.h" (tools/xdiff.h) for XNNPACK types and print helpers
  - print   case <id> <inputs...> -> <output hex>   per case
"""
from dataclasses import dataclass, field
from pathlib import Path
import shlex

from . import host, x280
from .equiv import KERNEL_DEF
from .kernelsrc import strip_local_xnn_types

SIDES = {"neon": ("aarch64-linux-gnu-gcc", "-march=armv8.2-a+fp16", "<arm_neon.h>"),
         "rvv": ("riscv64-linux-gnu-gcc", "-march=rv64gcv_zfh_zvfhmin", "<riscv_vector.h>")}


@dataclass
class Diff:
    cases: int
    differing: int
    first: list = field(default_factory=list)    # [(case line neon, case line rvv)]
    where: str = ""


def _kernel_unit(text, side, xnn_include):
    text, _ = strip_local_xnn_types(text)
    m = KERNEL_DEF.search(text)
    if not m:
        raise ValueError(f"no kernel definition in the {side} source")
    text = text[:m.start(1)] + f"xdiff_kernel_{side}" + text[m.end(1):]
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("#include")]
    header = [f"#include {SIDES[side][2]}", "#include <assert.h>", "#include <stddef.h>",
              "#include <stdint.h>", "#include <math.h>", f'#include "{xnn_include}"',
              "#ifndef XNN_OOB_READS\n#define XNN_OOB_READS\n#endif"]
    return "\n".join(header + lines) + "\n"


def build(cfg, driver, neon_text, rvv_text, out_dir, xnnpack="xnnpack-saltyrn",
          xnn_include="src/xnnpack/microparams.h"):
    """Compile driver+kernel for both sides as static binaries; returns {side: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "driver.c").write_text(Path(driver).read_text())
    binaries, cmds = {}, []
    xnn = host.in_container(cfg, cfg.src / xnnpack)
    inc = f"-I/tools -I{xnn} -I{xnn}/include -I{xnn}/src"
    for side, text in (("neon", neon_text), ("rvv", rvv_text)):
        if text is None:
            continue
        cc, march, _ = SIDES[side]
        (out_dir / f"{side}.c").write_text(_kernel_unit(text, side, xnn_include))
        d = host.in_container(cfg, out_dir)
        cmds.append(f"{cc} -O2 {march} -std=gnu11 -w {inc} -include /tools/xdiff.h "
                    f"-c {d}/{side}.c -o {d}/{side}.o "
                    f"&& {cc} -O2 {march} -std=gnu11 -w {inc} -include /tools/xdiff.h -DKERNEL=xdiff_kernel_{side} "
                    f"-DXDIFF_SIDE_{side.upper()} -c {d}/driver.c -o {d}/driver_{side}.o "
                    f"&& {cc} -static -o {d}/{side} {d}/driver_{side}.o {d}/{side}.o -lm")
        binaries[side] = out_dir / side
    host.run(cfg, " && ".join(cmds), timeout=600)
    return binaries


def _cases(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("case "):
            parts = line.split(" ", 2)
            out[parts[1]] = line
    return out


def compare(neon_out, rvv_out, where, limit=10):
    a, b = _cases(neon_out), _cases(rvv_out)
    keys = [k for k in a if k in b]
    bad = [k for k in keys if a[k] != b[k]]
    missing = len(set(a) ^ set(b))
    return Diff(cases=len(keys), differing=len(bad) + missing,
                first=[(a[k], b[k]) for k in bad[:limit]], where=where)


def run(cfg, binaries, vlens=(128, 512), board=True):
    """Run both sides; Neon once under QEMU, RVV under QEMU per VLEN and on the X280."""
    neon = host.run(cfg, f"qemu-aarch64 -cpu max {shlex.quote(host.in_container(cfg, binaries['neon']))}",
                    timeout=1800).stdout
    diffs = []
    for vlen in vlens:
        rvv = host.run(cfg, f"{host.qemu_prefix(vlen)},zfh=true {shlex.quote(host.in_container(cfg, binaries['rvv']))}",
                       timeout=1800).stdout
        diffs.append(compare(neon, rvv, f"qemu:{vlen}"))
    if board:
        with x280.lease(cfg):
            remote = x280.push(cfg, [binaries["rvv"]])[0]
            rvv = x280.run(cfg, remote, timeout=1800).stdout
        diffs.append(compare(neon, rvv, "x280"))
    return diffs
