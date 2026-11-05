"""X280 calibration: per-op costs and memory bandwidth, measured once per board.

Generates a C microbenchmark (inline-asm loops of `.rept` blocks between
perf_event cycle reads), runs it on the board under the lease, and stores
<workdir>/results/calib/<board>.json with:

  ops[op][e<sew>,m<lmul>] = {"throughput_cpi": ..., "latency_cpi": ...}
      cycles per instruction for independent chains (throughput) and for a
      dependent chain (latency), whole-LMUL-group instructions;
  memory[kind][working-set] = bytes per cycle for vle8/vse8/copy at m8;
  memops[form][e32,m<lmul>] = cycles per instruction of one e32 memory form
      (unit, strided, segment, strided segment, indexed) with L1-resident data
      and a 128-byte stride: what a transpose's loads and stores really cost;
  vsetvl_cpi: cost of a vsetvli that changes vtype.

predict() turns a static loop profile into cycles per element (sum of
throughput costs: the X280 vector unit issues in order) and roofline() places a
measured cell against the compute and memory ceilings.
"""
import json
from pathlib import Path

from . import host, x280

SEWS = (8, 16, 32)
LMULS = (1, 2, 4, 8)
# op name -> (asm template, kind). {d} {a} {b} are vector registers; kinds:
#   same   all operands at LMUL
#   widen  destination at 2*LMUL (sources at LMUL)
#   narrow source at 2*LMUL (destination at LMUL)
#   extend vtype gives the *destination* width (vsext/vzext): SEW >= 16, source at LMUL/2
OPS = {
    "vadd.vv": ("vadd.vv {d}, {a}, {b}", "same"),
    "vmul.vv": ("vmul.vv {d}, {a}, {b}", "same"),
    "vmacc.vv": ("vmacc.vv {d}, {a}, {b}", "same"),
    "vsmul.vv": ("vsmul.vv {d}, {a}, {b}", "same"),
    "vsadd.vx": ("vsadd.vx {d}, {a}, t0", "same"),
    "vsll.vi": ("vsll.vi {d}, {a}, 3", "same"),
    "vmseq.vv": ("vmseq.vv v0, {a}, {b}", "same"),
    "vmerge.vvm": ("vmerge.vvm {d}, {a}, {b}, v0", "same"),
    "vrgather.vv": ("vrgather.vv {d}, {a}, {b}", "same"),
    "vslidedown.vi": ("vslidedown.vi {d}, {a}, 1", "same"),
    "vredsum.vs": ("vredsum.vs {d}, {a}, {b}", "same"),
    "vwadd.vv": ("vwadd.vv {d}, {a}, {b}", "widen"),
    "vsext.vf2": ("vsext.vf2 {d}, {a}", "extend"),
    "vnclip.wi": ("vnclip.wi {d}, {a}, 4", "narrow"),
    "vfadd.vv": ("vfadd.vv {d}, {a}, {b}", "same"),
    "vfmacc.vv": ("vfmacc.vv {d}, {a}, {b}", "same"),
}
FLOAT_OPS = {"vfadd.vv", "vfmacc.vv"}
# Ops whose result can legally feed their own next instance (vd may overlap a
# source): only these get a dependent-chain (latency) measurement.
CHAINABLE = {"vadd.vv", "vmul.vv", "vmacc.vv", "vsmul.vv", "vsadd.vx", "vsll.vi", "vmerge.vvm",
             "vslidedown.vi", "vfadd.vv", "vfmacc.vv"}
MEMORY_SETS = {"L1": 16 << 10, "L2": 1 << 20, "DRAM": 64 << 20}
# e32 memory forms: name -> asm with {v} (register group), %0 base, %1 stride (128 B),
# v24 index vector (byte offsets 128*i). Segment forms need nf * LMUL <= 8.
MEMOPS = {"vle32.v": "vle32.v {v}, (%0)", "vse32.v": "vse32.v {v}, (%0)",
          "vlse32.v": "vlse32.v {v}, (%0), %1", "vsse32.v": "vsse32.v {v}, (%0), %1",
          "vluxei32.v": "vluxei32.v {v}, (%0), v24", "vsuxei32.v": "vsuxei32.v {v}, (%0), v24"}
for _nf in (2, 4, 8):
    MEMOPS.update({f"vlseg{_nf}e32.v": f"vlseg{_nf}e32.v {{v}}, (%0)", f"vsseg{_nf}e32.v": f"vsseg{_nf}e32.v {{v}}, (%0)",
                   f"vlsseg{_nf}e32.v": f"vlsseg{_nf}e32.v {{v}}, (%0), %1",
                   f"vssseg{_nf}e32.v": f"vssseg{_nf}e32.v {{v}}, (%0), %1"})
MEMOP_LMULS = (1, 2, 4)
REPT = 32
ITERS = 2000


def _regs(lmul, kind):
    """Register groups (d, a, b) and 4 independent destinations, aligned to the group sizes."""
    if kind in ("same", "extend"):
        width = lmul
    else:
        width = 2 * lmul
    step = max(width, lmul)
    groups = [f"v{i}" for i in range(step, 32, step)][:6]
    return groups


def cases():
    out = []
    for op, (template, kind) in OPS.items():
        for sew in SEWS:
            if op in FLOAT_OPS and sew != 32:
                continue
            for lmul in LMULS:
                if kind in ("widen", "narrow") and (lmul == 8 or sew == 32):
                    continue
                if kind == "extend" and sew == 8:
                    continue
                out.append((op, sew, lmul))
    return out


def _body(op, sew, lmul, dependent):
    template, kind = OPS[op]
    groups = _regs(lmul, kind)
    if len(groups) < 3:
        return None
    if dependent:  # every instruction reads the previous one's destination
        if op not in CHAINABLE:
            return None
        lines = [template.format(d=groups[0], a=groups[0], b=groups[2])] * 4
    else:          # four destinations, sources never written
        srcs = groups[-2:]
        dests = groups[:max(1, min(4, len(groups) - 2))]
        lines = [template.format(d=dests[i % len(dests)], a=srcs[0], b=srcs[1]) for i in range(4)]
    return "\\n".join(lines)


def memop_cases():
    out = []
    for op in MEMOPS:
        nf = int(op[op.index("seg") + 3]) if "seg" in op else 1
        for lmul in MEMOP_LMULS:
            if nf * lmul <= 8 and not (op.startswith(("vlux", "vsux")) and lmul > 2):
                out.append((op, lmul))
    return out


def source():
    """The microbenchmark C source; prints one JSON line per measurement."""
    funcs, calls = [], []
    for n, (op, lmul) in enumerate(memop_cases()):
        body = "\\n".join([MEMOPS[op].format(v="v8")] * 4)
        funcs.append(f"""
static double mem_{n}(void) {{
  uint64_t c0 = cycles();
  size_t vl; asm volatile("vsetvli %0, zero, e32, m{lmul}, ta, ma\\n vid.v v24\\n vsll.vi v24, v24, 7" : "=r"(vl));
  for (int i = 0; i < {ITERS}; i++)
    asm volatile(".rept {REPT // 4}\\n{body}\\n.endr" :: "r"(buf_a), "r"((size_t) 128) : "memory");
  return (double) (cycles() - c0) / ({ITERS} * {REPT});
}}""")
        calls.append(f'  emit("memop", "{op}", 32, {lmul}, 0, mem_{n}());')
    for n, (op, sew, lmul) in enumerate(cases()):
        for dep in (0, 1):
            body = _body(op, sew, lmul, dep)
            if body is None:
                continue
            funcs.append(f"""
static double op_{n}_{dep}(void) {{
  // Linux discards vector state on every syscall (vtype becomes vill), so the
  // counter is read *before* vtype is set; one vsetvli in 64k ops is noise.
  uint64_t c0 = cycles();
  size_t vl; asm volatile("vsetvli %0, zero, e{sew}, m{lmul}, ta, ma\\n li t0, 3\\n csrwi vxrm, 0" : "=r"(vl) :: "t0");
  for (int i = 0; i < {ITERS}; i++) asm volatile(".rept {REPT // 4}\\n{body}\\n.endr" ::: "memory", "t0");
  return (double) (cycles() - c0) / ({ITERS} * {REPT});
}}""")
            calls.append(f'  emit("op", "{op}", {sew}, {lmul}, {dep}, op_{n}_{dep}());')
    mem = []
    for name, size in MEMORY_SETS.items():
        mem.append(f'  emit_mem("{name}", {size});')
    return f"""// Generated by portforge_tt/remote/calibrate.py; do not edit.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>

static int fd = -1;
static uint64_t cycles(void) {{
  uint64_t c = 0;
  if (fd >= 0 && read(fd, &c, sizeof c) != sizeof c) c = 0;
  return c;
}}
static void emit(const char* kind, const char* op, int sew, int lmul, int dep, double cpi) {{
  printf("{{\\"kind\\":\\"%s\\",\\"op\\":\\"%s\\",\\"sew\\":%d,\\"lmul\\":%d,\\"dependent\\":%d,\\"cpi\\":%.4f}}\\n",
         kind, op, sew, lmul, dep, cpi);
}}
static uint8_t* buf_a; static uint8_t* buf_b;
static double stream(int kind, size_t bytes) {{  // 0 load, 1 store, 2 copy; m8 e8
  double best = 1e30;
  for (int rep = 0; rep < 5; rep++) {{
    uint64_t c0 = cycles();
    size_t passes = (64u << 20) / bytes + 1;
    for (size_t p = 0; p < passes; p++) {{
      uint8_t* a = buf_a; uint8_t* b = buf_b; size_t n = bytes;
      while (n) {{
        size_t vl;
        asm volatile("vsetvli %0, %1, e8, m8, ta, ma" : "=r"(vl) : "r"(n));
        if (kind != 1) asm volatile("vle8.v v8, (%0)" :: "r"(a) : "memory");
        if (kind != 0) asm volatile("vse8.v v8, (%0)" :: "r"(b) : "memory");
        a += vl; b += vl; n -= vl;
      }}
    }}
    double bpc = (double) (bytes * passes * (kind == 2 ? 2 : 1)) / (double) (cycles() - c0);
    if (1 / bpc < best) best = 1 / bpc;
  }}
  return 1 / best;
}}
static void emit_mem(const char* set, size_t bytes) {{
  static const char* names[] = {{"load", "store", "copy"}};
  for (int k = 0; k < 3; k++)
    printf("{{\\"kind\\":\\"mem\\",\\"op\\":\\"%s\\",\\"set\\":\\"%s\\",\\"bytes\\":%zu,\\"bytes_per_cycle\\":%.4f}}\\n",
           names[k], set, bytes, stream(k, bytes));
}}
static double vsetvl_switch(void) {{
  uint64_t c0 = cycles();  // vsetvli itself re-arms vtype after the syscall
  for (int i = 0; i < {ITERS}; i++)
    asm volatile(".rept 16\\nvsetvli t1, zero, e8, m4, ta, ma\\nvsetvli t1, zero, e16, m8, ta, ma\\n.endr" ::: "t1");
  return (double) (cycles() - c0) / ({ITERS} * 32);
}}
{''.join(funcs)}

int main(void) {{
  struct perf_event_attr attr;
  memset(&attr, 0, sizeof attr);
  attr.type = PERF_TYPE_HARDWARE; attr.size = sizeof attr; attr.config = PERF_COUNT_HW_CPU_CYCLES;
  attr.exclude_kernel = 1; attr.exclude_hv = 1;
  fd = (int) syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
  if (fd < 0) {{ fprintf(stderr, "no perf_event cycles\\n"); return 1; }}
  buf_a = aligned_alloc(4096, 64u << 20); buf_b = aligned_alloc(4096, 64u << 20);
  memset(buf_a, 1, 64u << 20); memset(buf_b, 2, 64u << 20);
{chr(10).join(mem)}
  printf("{{\\"kind\\":\\"vsetvl\\",\\"cpi\\":%.4f}}\\n", vsetvl_switch());
{chr(10).join(calls)}
  return 0;
}}
"""


def run(cfg, out_dir=None, log=print):
    """Build and run the calibration on the X280 (idle + lease); returns the table."""
    out_dir = Path(out_dir or cfg.results / "calib")
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "x280_calib.c"
    src.write_text(source())
    exe = out_dir / "x280_calib"
    host.run(cfg, f"riscv64-linux-gnu-gcc -O2 -march=rv64gcv -static {host.in_container(cfg, src)} "
                  f"-o {host.in_container(cfg, exe)}", timeout=600)
    with x280.lease(cfg):
        state = x280.idle(cfg)
        if not state.get("ok"):
            raise RuntimeError(f"board busy, not calibrating: {state}")
        remote = x280.push(cfg, [exe])[0]
        proc = x280.run(cfg, f"env -i taskset -c {cfg.x280.cpu} {remote}", timeout=3600)
    rows = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith("{")]
    table = tabulate(rows)
    table["board"] = x280.board_id(cfg)
    path = out_dir / f"{table['board'].replace(' ', '_').replace('/', '_')}.json"
    path.write_text(json.dumps(table, indent=1) + "\n")
    log(f"calibration: {path}")
    return table


def tabulate(rows):
    table = {"ops": {}, "memory": {}, "vsetvl_cpi": None}
    for r in rows:
        if r["kind"] == "op":
            cell = table["ops"].setdefault(r["op"], {}).setdefault(f"e{r['sew']},m{r['lmul']}", {})
            cell["latency_cpi" if r["dependent"] else "throughput_cpi"] = r["cpi"]
        elif r["kind"] == "memop":
            table.setdefault("memops", {}).setdefault(r["op"], {})[f"e32,m{r['lmul']}"] = r["cpi"]
        elif r["kind"] == "mem":
            table["memory"].setdefault(r["op"], {})[r["set"]] = r["bytes_per_cycle"]
        elif r["kind"] == "vsetvl":
            table["vsetvl_cpi"] = r["cpi"]
    return table


def load(cfg):
    paths = sorted((cfg.results / "calib").glob("*.json")) if (cfg.results / "calib").exists() else []
    return json.loads(paths[-1].read_text()) if paths else None
