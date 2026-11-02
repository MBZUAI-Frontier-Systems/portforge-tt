"""Validate the instruction model (rvv_model.h) and the lifter against real executions.

Every test is one small RISC-V function that loads the whole vector register
file, a few scalar registers, vxrm and a memory buffer from random inputs, sets
vtype, executes one instruction under test, and stores everything back. The
same function runs
  - for real: under QEMU at a given VLEN (and on the X280 when allowed), and
  - lifted: through lift.py + rvv_model.h compiled natively at the same VLEN,
on the same seeded inputs; any difference in any output byte is a model or
lifter bug. Undisturbed policies (tu, mu) keep every output deterministic; the
model treats agnostic elements as nondeterministic in proofs anyway.

    python -m portforge_tt.fv.validate --vlen 128 --vlen 512 --trials 64
"""
import argparse
import itertools
import json
from pathlib import Path
import re

from portforge_tt.cpu import config, host

from . import lift, run

FV_DIR = Path(__file__).resolve().parent
SEWS = (8, 16, 32, 64)
LMULS = ("mf8", "mf4", "mf2", "m1", "m2", "m4", "m8")
L8 = {"mf8": 1, "mf4": 2, "mf2": 4, "m1": 8, "m2": 16, "m4": 32, "m8": 64}


def _legal(sew, lmul):
    return sew * 8 <= 64 * L8[lmul]


def _regs(lmul8, widen=False):
    """vd, vs2, vs1 aligned for the group size (widening doubles vd's group)."""
    return ("v8", "v16", "v24") if lmul8 <= 32 else ("v8", "v16", "v24") if not widen else None


# (template, kind): kind decides which SEW/LMUL combinations are legal
#   s: single width   w: widening (vd 2*SEW)   wv: widening with a wide vs2   n: narrowing (vs2 2*SEW)
#   m: mask result    r: reduction   x: scalar result   p: permutation (vd != sources)
SINGLE = ["vadd", "vsub", "vand", "vor", "vxor", "vsll", "vsrl", "vsra", "vmin", "vminu", "vmax", "vmaxu", "vmul",
          "vmulh", "vmulhu", "vmulhsu", "vdiv", "vdivu", "vrem", "vremu", "vsadd", "vsaddu", "vssub", "vssubu",
          "vaadd", "vaaddu", "vasub", "vasubu", "vsmul", "vssrl", "vssra"]
VI_OK = {"vadd", "vand", "vor", "vxor", "vsll", "vsrl", "vsra", "vsadd", "vsaddu", "vssrl", "vssra", "vrsub"}
NOT_AT_64 = {"vsadd", "vssub", "vaadd", "vasub", "vaaddu", "vasubu", "vsmul"}


def templates():
    out = []
    for op in SINGLE:
        out.append((f"{op}.vv v8, v16, v24", "s", op))
        out.append((f"{op}.vx v8, v16, t2", "s", op))
        if op in VI_OK:
            out.append((f"{op}.vi v8, v16, {7 if op.startswith(('vs', 'vss')) else -3}", "s", op))
        out.append((f"{op}.vv v8, v16, v24, v0.t", "s", op))
    out += [("vrsub.vx v8, v16, t2", "s", "vrsub"), ("vrsub.vi v8, v16, 5", "s", "vrsub")]
    for op in ("vmacc", "vnmsac", "vmadd", "vnmsub"):
        out += [(f"{op}.vv v8, v24, v16", "s", op), (f"{op}.vx v8, t2, v16", "s", op)]
    for op in ("vmseq", "vmsne", "vmslt", "vmsltu", "vmsle", "vmsleu"):
        out += [(f"{op}.vv v1, v16, v24", "m", op), (f"{op}.vx v1, v16, t2", "m", op)]
    out += [("vmsgt.vx v1, v16, t2", "m", "vmsgt"), ("vmsgtu.vx v1, v16, t2", "m", "vmsgtu"),
            ("vmseq.vi v1, v16, 3", "m", "vmseq"), ("vmsle.vi v1, v16, -2", "m", "vmsle")]
    for op in ("vwadd", "vwaddu", "vwsub", "vwsubu", "vwmul", "vwmulu", "vwmulsu"):
        out += [(f"{op}.vv v8, v16, v24", "w", op), (f"{op}.vx v8, v16, t2", "w", op)]
    for op in ("vwadd", "vwaddu", "vwsub", "vwsubu"):
        out += [(f"{op}.wv v8, v16, v24", "wv", op), (f"{op}.wx v8, v16, t2", "wv", op)]
    for op in ("vwmacc", "vwmaccu", "vwmaccsu"):
        out += [(f"{op}.vv v8, v24, v16", "w", op), (f"{op}.vx v8, t2, v16", "w", op)]
    out.append(("vwmaccus.vx v8, t2, v16", "w", "vwmaccus"))
    for op in ("vnsrl", "vnsra", "vnclip", "vnclipu"):
        out += [(f"{op}.wv v8, v16, v24", "n", op), (f"{op}.wx v8, v16, t2", "n", op),
                (f"{op}.wi v8, v16, 3", "n", op)]
    for f in ("vf2", "vf4", "vf8"):
        out += [(f"vzext.{f} v8, v16", f"e{f}", "vzext"), (f"vsext.{f} v8, v16", f"e{f}", "vsext")]
    out += [("vmv.v.v v8, v16", "s", "vmv"), ("vmv.v.x v8, t2", "s", "vmv"), ("vmv.v.i v8, -7", "s", "vmv"),
            ("vmerge.vvm v8, v16, v24, v0", "s", "vmerge"), ("vmerge.vxm v8, v16, t2, v0", "s", "vmerge"),
            ("vmerge.vim v8, v16, 9, v0", "s", "vmerge"), ("vmv.x.s t3, v16", "x", "vmv.x.s"),
            ("vmv.s.x v8, t2", "s", "vmv.s.x"), ("vid.v v8", "s", "vid"),
            ("vrgather.vv v8, v16, v24", "p", "vrgather"), ("vrgather.vx v8, v16, t4", "p", "vrgather"),
            ("vrgather.vi v8, v16, 3", "p", "vrgather"), ("vrgatherei16.vv v8, v16, v24", "p16", "vrgatherei16"),
            ("vslideup.vx v8, v16, t4", "p", "vslideup"), ("vslideup.vi v8, v16, 3", "p", "vslideup"),
            ("vslidedown.vx v8, v16, t4", "p", "vslidedown"), ("vslidedown.vi v8, v16, 5", "p", "vslidedown"),
            ("vslide1up.vx v8, v16, t2", "p", "vslide1up"), ("vslide1down.vx v8, v16, t2", "p", "vslide1down"),
            ("vcompress.vm v8, v16, v1", "p", "vcompress")]
    for op in ("vredsum", "vredmax", "vredmaxu", "vredmin", "vredminu", "vredand", "vredor", "vredxor"):
        out.append((f"{op}.vs v8, v16, v24", "r", op))
    out += [("vwredsum.vs v8, v16, v24", "wr", "vwredsum"), ("vwredsumu.vs v8, v16, v24", "wr", "vwredsumu")]
    for op in ("vmand", "vmnand", "vmandn", "vmxor", "vmor", "vmnor", "vmorn", "vmxnor"):
        out.append((f"{op}.mm v1, v2, v3", "mm", op))
    out += [("vcpop.m t3, v2", "xm", "vcpop"), ("vfirst.m t3, v2", "xm", "vfirst")]
    for eew in (8, 16, 32, 64):
        out += [(f"vle{eew}.v v8, (a4)", f"ld{eew}", "vle"), (f"vse{eew}.v v16, (a4)", f"ld{eew}", "vse"),
                (f"vle{eew}.v v8, (a4), v0.t", f"ld{eew}", "vle"),
                (f"vlse{eew}.v v8, (a4), t5", f"ld{eew}", "vlse"), (f"vsse{eew}.v v16, (a4), t5", f"ld{eew}", "vsse"),
                (f"vluxei{eew}.v v8, (a4), v24", f"ix{eew}", "vluxei"),
                (f"vsuxei{eew}.v v16, (a4), v24", f"ix{eew}", "vsuxei")]
        for nf in (2, 3, 4, 8):
            out += [(f"vlseg{nf}e{eew}.v v8, (a4)", f"sg{eew}x{nf}", "vlseg"),
                    (f"vsseg{nf}e{eew}.v v8, (a4)", f"sg{eew}x{nf}", "vsseg"),
                    (f"vlsseg{nf}e{eew}.v v8, (a4), t5", f"sg{eew}x{nf}", "vlsseg"),
                    (f"vssseg{nf}e{eew}.v v8, (a4), t5", f"sg{eew}x{nf}", "vssseg")]
    out += [("vl1re8.v v8, (a4)", "whole", "vl1r"), ("vl2re16.v v8, (a4)", "whole", "vl2r"),
            ("vs1r.v v16, (a4)", "whole", "vs1r"), ("vs2r.v v16, (a4)", "whole", "vs2r"),
            ("vmv1r.v v8, v16", "whole", "vmv1r"), ("vmv2r.v v8, v16", "whole", "vmv2r"),
            ("vmv4r.v v8, v16", "whole", "vmv4r"), ("vmv8r.v v8, v16", "whole", "vmv8r")]
    return out


def configs(kind, op):
    """(sew, lmul) pairs to test a template at: every legal combination for its kind."""
    out = []
    for sew, lmul in itertools.product(SEWS, LMULS):
        if not _legal(sew, lmul):
            continue
        l8 = L8[lmul]
        if kind in ("s", "m", "r", "x", "p") and op in NOT_AT_64 and sew == 64:
            continue
        if kind in ("w", "wv", "n", "wr") and (sew > 32 or l8 > 32):
            continue
        if kind.startswith("evf"):
            f = int(kind[3])
            if sew // f < 8 or l8 < f:
                continue
        if kind == "p16" and not 1 <= l8 * 16 // sew <= 64:
            continue
        if kind[:2] in ("ld", "ix", "sg"):
            eew = int(re.match(r"\D+(\d+)", kind).group(1))
            emul8 = l8 * eew // sew
            if l8 * eew % sew or not 1 <= emul8 <= 64:
                continue
            if kind.startswith("sg") and int(kind.split("x")[1]) * max(1, emul8 // 8) > 8:
                continue
            if kind.startswith("ix") and eew != sew:
                continue   # the index mask below runs at SEW
        if kind in ("mm", "xm", "whole") and (sew, lmul) not in ((8, "m1"), (32, "m4"), (16, "mf2")):
            continue
        out.append((sew, lmul))
    return out


PROLOGUE = """    .text
    .option arch, +v, +zba, +zbb
"""


def test_function(k, insn, sew, lmul):
    """a0 = in, a1 = out, a2 = avl, a3 = vxrm, a4 = memory buffer (64 KB), a5 = index mask."""
    m = re.search(r"e(?:i)?(\d+)\.v", insn)
    eew = int(m.group(1)) if m else 8
    stride_mask = 63 & ~(eew // 8 - 1)
    return f"""
    .globl t_{k}
    .type t_{k}, @function
t_{k}:
    csrr    t1, vlenb
    slli    t1, t1, 3
    mv      t6, a0
    vl8re8.v v0, (t6)
    add     t6, t6, t1
    vl8re8.v v8, (t6)
    add     t6, t6, t1
    vl8re8.v v16, (t6)
    add     t6, t6, t1
    vl8re8.v v24, (t6)
    add     t6, t6, t1
    ld      t2, 0(t6)
    ld      t4, 8(t6)
    andi    t4, t4, 127
    ld      t5, 16(t6)
    andi    t5, t5, {stride_mask}
    ld      t3, 24(t6)
    csrw    vxrm, a3
    csrwi   vxsat, 0
    vsetvli t0, a2, e{sew}, {lmul}, tu, mu
    {"vand.vx v24, v24, a5" if "ei" in insn and "v24" in insn else "nop"}
    {insn}
    csrr    t0, vl
    csrr    a6, vxsat
    mv      t6, a1
    vs8r.v  v0, (t6)
    add     t6, t6, t1
    vs8r.v  v8, (t6)
    add     t6, t6, t1
    vs8r.v  v16, (t6)
    add     t6, t6, t1
    vs8r.v  v24, (t6)
    add     t6, t6, t1
    sd      t0, 0(t6)
    sd      t3, 8(t6)
    sd      a6, 16(t6)
    ret
    .size t_{k}, .-t_{k}
"""


DRIVER = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NT %(n)d
%(protos)s
typedef void (*tfn)(uint8_t *, uint8_t *, uint64_t, uint64_t, uint8_t *, uint64_t);
static const tfn T[NT] = {%(table)s};
static const unsigned char HM[NT] = {%(hashmem)s};   /* store tests: hash the memory buffer too */
static uint64_t rng;
static uint8_t r8(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint8_t)(rng >> 24); }
#define VLENB_MAX 128
static uint8_t in[32 * VLENB_MAX + 64], out[32 * VLENB_MAX + 64], mem[65536 + 64];
static uint64_t fnv(const uint8_t *p, size_t n, uint64_t h) { for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; } return h; }
int main(int argc, char **argv)
{
    int trials = atoi(argv[1]);
    uint64_t vlenb = %(vlenb)s;
    for (int k = 0; k < NT; k++) {
        for (int t = 0; t < trials; t++) {
            rng = 0x9E3779B97F4A7C15ULL * (uint64_t)(k * 1000 + t + 1);
            for (size_t i = 0; i < sizeof in; i++) in[i] = r8();
            for (size_t i = 0; i < sizeof mem; i++) mem[i] = r8();
            memset(out, 0, sizeof out);
            uint64_t avl = t %% 4 == 0 ? 1000000 : r8() %% (8 * vlenb + 3);
            uint64_t xrm = r8() & 3;
            T[k](in, out, avl, xrm, ADDR(mem), 0xff8 & ~((uint64_t)7));
            uint64_t h = fnv(out, 32 * vlenb + 24, 0xcbf29ce484222325ULL);
            if (HM[k]) h = fnv(mem, 65536, h);
            printf("%%d %%d %%016llx\n", k, t, (unsigned long long)h);
        }
    }
    return 0;
}
"""


def build_suite(vlens_unused=None):
    tests = []
    for insn, kind, op in templates():
        for sew, lmul in configs(kind, op):
            tests.append({"insn": insn, "kind": kind, "sew": sew, "lmul": lmul})
    return tests


def run_validation(cfg, out_dir, vlens=(128, 512), trials=32, limit=None, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tests = build_suite()[:limit] if limit else build_suite()
    asm = PROLOGUE + "".join(test_function(k, t["insn"], t["sew"], t["lmul"]) for k, t in enumerate(tests))
    (out_dir / "vtests.S").write_text(asm)
    (out_dir / "tests.json").write_text(json.dumps(tests, indent=0) + "\n")
    protos = "".join(f"void t_{k}(uint8_t *, uint8_t *, uint64_t, uint64_t, uint8_t *, uint64_t);\n"
                     for k in range(len(tests)))
    table = ", ".join(f"t_{k}" for k in range(len(tests)))
    ic = lambda p: host.in_container(cfg, p)
    hashmem = ", ".join("1" if re.match(r"vs(s?seg|e|se|uxei|\dr)", t["insn"]) else "0" for t in tests)
    rv_driver = DRIVER % {"n": len(tests), "protos": protos, "table": table, "hashmem": hashmem, "vlenb": "({ uint64_t v; __asm__ volatile(\"csrr %0, vlenb\" : \"=r\"(v)); v; })"}
    (out_dir / "driver_rv.c").write_text(rv_driver.replace("ADDR(mem)", "mem"))
    proc = host.run(cfg, f"cd {ic(out_dir)} && riscv64-linux-gnu-gcc -O1 -march=rv64gcv_zba_zbb -static driver_rv.c "
                         f"vtests.S -o vtests.elf 2>&1 | tail -30", timeout=1800, check=False)
    if not (out_dir / "vtests.elf").exists():
        raise RuntimeError("assembling the tests failed:\n" + proc.stdout)
    text = run.disassemble(cfg, out_dir / "vtests.elf")
    funcs = lift.parse(text)
    unsupported, bodies = [], []
    for k in range(len(tests)):
        try:
            bodies.append(lift.Lifter(funcs, f"t_{k}").lift())
        except lift.Unsupported as error:
            unsupported.append({"test": k, **tests[k], "error": str(error)})
            bodies.append(f"static void fvf_t_{k}(void) {{ FV_ASSERT(0, \"unsupported\"); }}\n")
    results = {"tests": len(tests), "unsupported": unsupported, "vlens": {}}
    for vlen in vlens:
        native = (f"#define FV_VLEN {vlen}\n#include \"fv_runtime.h\"\nconst fv_data FV_DATA[] = {{{{0, 0, 0}}}};\n"
                  f"const int FV_NDATA = 0;\n" + "\n".join(bodies) + "\n"
                  + NATIVE_DRIVER % {"n": len(tests), "vlenb": vlen // 8, "hashmem": hashmem,
                                     "table": ", ".join(f"fvf_t_{k}" for k in range(len(tests)))})
        (out_dir / f"native_{vlen}.c").write_text(native)
        proc = host.run(cfg, f"cd {ic(out_dir)} && gcc -O1 -w -I{ic(FV_DIR)} native_{vlen}.c -o native_{vlen} 2>&1 "
                             f"| tail -20", timeout=3600, check=False)
        if not (out_dir / f"native_{vlen}").exists():
            raise RuntimeError("native build failed:\n" + proc.stdout)
        qemu = (f"qemu-riscv64 -cpu rv64,v=true,vlen={vlen},elen=64,vext_spec=v1.0,zba=true,zbb=true "
                f"{ic(out_dir / 'vtests.elf')} {trials}")
        a = host.run(cfg, f"cd {ic(out_dir)} && {qemu} > qemu_{vlen}.txt; ./native_{vlen} {trials} > native_{vlen}.txt; "
                          f"echo done", timeout=7200, check=False)
        qe = (out_dir / f"qemu_{vlen}.txt").read_text().splitlines()
        na = (out_dir / f"native_{vlen}.txt").read_text().splitlines()
        bad = {}
        for lq, ln in zip(qe, na):
            if lq != ln:
                k = int(lq.split()[0])
                bad.setdefault(k, 0)
                bad[k] += 1
        if len(qe) != len(na):
            bad[-1] = f"line counts differ: qemu {len(qe)} native {len(na)}"
        results["vlens"][vlen] = {"checked": len(qe), "mismatched_tests": len(bad),
                                  "mismatches": [{"test": k, **(tests[k] if k >= 0 else {}), "trials": n}
                                                 for k, n in sorted(bad.items())][:200]}
        log(f"VLEN {vlen}: {len(qe)} runs, {len(bad)} tests differ")
    (out_dir / "validation.json").write_text(json.dumps(results, indent=1) + "\n")
    return results


NATIVE_DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
#define NT %(n)d
typedef void (*tfn)(void);
static const tfn T[NT] = {%(table)s};
static const unsigned char HM[NT] = {%(hashmem)s};   /* store tests: hash the memory buffer too */
static uint64_t rng;
static uint8_t r8(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint8_t)(rng >> 24); }
#define VLENB_MAX 128
static uint8_t in[32 * VLENB_MAX + 64], out[32 * VLENB_MAX + 64], mem[65536 + 64];
static uint64_t fnv(const uint8_t *p, size_t n, uint64_t h) { for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; } return h; }
int main(int argc, char **argv)
{
    int trials = atoi(argv[1]);
    uint64_t vlenb = %(vlenb)d;
    fv_region_add("in", 0x100000, sizeof in, in, 0);
    fv_region_add("out", 0x200000, sizeof out, out, 1);
    fv_region_add("mem", 0x300000, sizeof mem, mem, 1);
    for (int k = 0; k < NT; k++) {
        for (int t = 0; t < trials; t++) {
            rng = 0x9E3779B97F4A7C15ULL * (uint64_t)(k * 1000 + t + 1);
            for (size_t i = 0; i < sizeof in; i++) in[i] = r8();
            for (size_t i = 0; i < sizeof mem; i++) mem[i] = r8();
            memset(out, 0, sizeof out);
            uint64_t avl = t %% 4 == 0 ? 1000000 : r8() %% (8 * vlenb + 3);
            uint64_t xrm = r8() & 3;
            memset(&S, 0, sizeof S);
            S.x[10] = 0x100000; S.x[11] = 0x200000; S.x[12] = avl; S.x[13] = xrm; S.x[14] = 0x300000;
            S.x[15] = 0xff8 & ~((uint64_t)7);
            S.x[1] = FV_RETURN_ADDR; S.x[2] = 0x10000000ULL + 4096; S.vill = 1;
            T[k]();
            uint64_t h = fnv(out, 32 * vlenb + 24, 0xcbf29ce484222325ULL);
            if (HM[k]) h = fnv(mem, 65536, h);
            printf("%%d %%d %%016llx\n", k, t, (unsigned long long)h);
        }
    }
    return 0;
}
"""


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m portforge_tt.fv.validate")
    p.add_argument("--vlen", type=int, action="append")
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--limit", type=int)
    p.add_argument("--out", default="results/fv/validate")
    a = p.parse_args(argv)
    cfg = config.load()
    r = run_validation(cfg, cfg.workdir / a.out, vlens=tuple(a.vlen or (128, 512)), trials=a.trials, limit=a.limit)
    print(json.dumps({"tests": r["tests"], "unsupported": len(r["unsupported"]),
                      "vlens": {v: {k: x[k] for k in ("checked", "mismatched_tests")} for v, x in r["vlens"].items()}}))


if __name__ == "__main__":
    main()
