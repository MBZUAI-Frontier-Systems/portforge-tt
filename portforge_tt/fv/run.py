"""Drive one formal check: disassemble -> lift -> co-simulate -> CBMC.

Runs on the x86 host, in the toolchain image (cbmc, gcc, riscv64 binutils).

  lifted = lift_binary(cfg, binary, entry, work)          # lifted.c (+ data symbols)
  cosim(cfg, harness, work, includes, defines, runs=200)   # native: lifted vs C reference
  prove(cfg, harness, work, includes, defines, vlen=512)   # CBMC: every input, bounded sizes

A harness is a C file that #includes fv_runtime.h, "lifted.c" and the project's
reference source, and builds main() from regions, fv_fill, fv_call, fv_compare
(see harness/). Defines select the bounded configuration (sizes, parameters).

CBMC verdicts: "proved" (every property holds, unwinding complete),
"counterexample" (with the failing property and the input bytes), "alignment"
(only misaligned scalar accesses fail: a performance finding), "timeout",
"error". The verdict records the binary's sha256, the harness sha256 and the
CBMC version.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import time

from portforge_tt.cpu import host

from . import lift as lift_mod

FV_DIR = Path(__file__).resolve().parent
OBJDUMP = "riscv64-linux-gnu-objdump"


@dataclass
class Verdict:
    verdict: str
    config: dict
    vlen: int
    seconds: float = 0.0
    failed: list = field(default_factory=list)       # [{"property", "description"}]
    alignment: list = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    error: str = ""
    log: str = ""

    @property
    def proved(self):
        return self.verdict == "proved"


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


INSN = re.compile(r"^(\s*[0-9a-f]+:\s+)\.insn\s+(\d),\s*0x([0-9a-f]+)\s*$")


def disassemble(cfg, binary):
    """objdump -d, with raw `.insn` words decoded for the full RV64GCV + Zba/Zbb/Zbs ISA.

    objdump picks the ISA from the binary's mapping symbols and attributes; in a
    linked binary a function assembled under `.option arch, +zba` can still print
    as `.insn 4, 0x...`. Those words are reassembled into a helper object whose
    attributes name every extension we model, and disassembled from there."""
    text = host.run(cfg, f"{OBJDUMP} -d --no-show-raw-insn {host.in_container(cfg, binary)}",
                    timeout=600).stdout
    words = sorted({(int(m.group(2)), m.group(3)) for m in map(INSN.match, text.splitlines()) if m})
    if not words:
        return text
    work = Path(binary).parent
    src = work / "fv_insn_words.S"
    src.write_text("    .option norvc\n    .text\n" + "".join(
        f"w_{n}_{w}:\n    .insn {n}, 0x{w}\n" for n, w in words))
    w_c = host.in_container(cfg, work)
    out = host.run(cfg, f"cd {w_c} && riscv64-linux-gnu-gcc -march=rv64gcv_zba_zbb_zbs -mabi=lp64d -c "
                        f"fv_insn_words.S -o fv_insn_words.o && {OBJDUMP} -d --no-show-raw-insn fv_insn_words.o",
                   timeout=300, check=False).stdout
    decoded, label = {}, None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <w_(\d)_([0-9a-f]+)>:", line)
        if m:
            label = (int(m.group(1)), m.group(2))
            continue
        m = re.match(r"^\s*[0-9a-f]+:\s+(.*)$", line)
        if m and label and label not in decoded:
            decoded[label] = m.group(1).strip()
    lines = []
    for line in text.splitlines():
        m = INSN.match(line)
        if m and (int(m.group(2)), m.group(3)) in decoded and not decoded[(int(m.group(2)), m.group(3))].startswith(".insn"):
            line = m.group(1) + decoded[(int(m.group(2)), m.group(3))]
        lines.append(line)
    return "\n".join(lines) + "\n"


def _symbols(cfg, binary):
    """[(addr, size, section, name)] of every symbol with a size."""
    out = host.run(cfg, f"{OBJDUMP} -t {host.in_container(cfg, binary)}", timeout=300).stdout
    syms = []
    for line in out.splitlines():
        m = re.match(r"^([0-9a-f]{16})\s.{7}\s(\S+)\s+([0-9a-f]{16})\s+(?:\.hidden\s+)?(\S+)$", line)
        if m and int(m.group(3), 16) > 0:
            syms.append((int(m.group(1), 16), int(m.group(3), 16), m.group(2), m.group(4)))
    return syms


def _bytes(cfg, binary, start, stop):
    out = host.run(cfg, f"{OBJDUMP} -s --start-address={start:#x} --stop-address={stop:#x} "
                        f"{host.in_container(cfg, binary)}", timeout=300).stdout
    data = bytearray()
    for line in out.splitlines():
        m = re.match(r"^\s+([0-9a-f]+)\s((?:[0-9a-f]{2,8}\s){1,4})", line)
        if m:
            addr = int(m.group(1), 16)
            chunk = bytes.fromhex("".join(m.group(2).split()))
            if addr - start != len(data):
                data.extend(b"\0" * (addr - start - len(data)))
            data.extend(chunk)
    return bytes(data[:stop - start])


def referenced_addresses(funcs, names):
    """Absolute addresses the code computes (objdump annotates auipc pairs)."""
    out = set()
    for n in names:
        insns = funcs[n].insns
        for k, i in enumerate(insns):
            m = re.match(r"([0-9a-f]+)", i.comment or "")
            if not m or i.mnem in ("jal", "j", "call", "tail") or i.mnem.startswith("b"):
                continue
            # only pc-relative pairs: an auipc into one of this instruction's registers shortly before
            regs = set(re.findall(r"\b[a-z]+\d*\b", " ".join(i.ops)))
            if any(p.mnem == "auipc" and p.ops and p.ops[0] in regs for p in insns[max(0, k - 4):k]):
                out.add(int(m.group(1), 16))
    return out


def lift_binary(cfg, binary, entry, work):
    """lifted.c for entry (and its callees) plus the data symbols the code addresses."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    text = disassemble(cfg, binary)
    funcs = lift_mod.parse(text)
    if entry not in funcs:  # dav1d's function macro puts a local and an EXTERN label on one address
        alias = entry[len("dav1d_"):] if entry.startswith("dav1d_") else "dav1d_" + entry
        if alias not in funcs:
            raise lift_mod.Unsupported(f"{entry} not in {binary}")
        entry = alias
    lifter = lift_mod.Lifter(funcs, entry)
    body = lifter.lift()
    addrs = referenced_addresses(funcs, lifter.needed)
    code = {(f.start, f.end) for f in funcs.values() if f.insns}
    data = []
    syms = _symbols(cfg, binary)
    for a in sorted(addrs):
        if any(s <= a < e for s, e in code):
            continue   # an address inside code (a jump target), not data
        hit = [s for s in syms if s[0] <= a < s[0] + s[1] and s[2] != ".text"]
        if not hit:
            # no sized symbol: take the rest of the section up to the next symbol, capped
            nxt = min((s[0] for s in syms if s[0] > a), default=a + 256)
            hit = [(a, min(nxt - a, 4096), "?", f"anon_{a:x}")]
        base, size, _, name = hit[0]
        if any(d[1] == base for d in data):
            continue
        data.append((name, base, _bytes(cfg, binary, base, base + size)))
    lines = ["/* lifted by portforge_tt.fv.lift from " + Path(binary).name + " entry " + entry + " */"]
    for k, (name, base, blob) in enumerate(data):
        lines.append(f"/* {name} */ static const uint8_t fv_data_{k}[{max(len(blob), 1)}] = {{"
                     + ", ".join(str(b) for b in blob) + "};")
    lines.append("const fv_data FV_DATA[] = {" + "".join(
        f"{{{base:#x}ULL, {len(blob)}, fv_data_{k}}}, " for k, (_, base, blob) in enumerate(data))
                 + ("{0, 0, 0}" if not data else "") + "};")
    lines.append(f"const int FV_NDATA = {len(data)};")
    (work / "lifted.c").write_text("\n".join(lines) + "\n\n" + body)
    meta = {"entry": entry, "functions": lifter.needed, "binary_sha256": sha_file(binary),
            "data": [{"name": n, "base": hex(b), "size": len(blob)} for n, b, blob in data],
            "instructions": sum(len(funcs[n].insns) for n in lifter.needed)}
    (work / "lifted.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def _flags(includes, defines):
    """includes are container paths"""
    inc = " ".join(f"-I{i}" for i in includes)
    dfn = " ".join(f"-D{k}={v}" for k, v in defines.items())
    return inc, dfn


def extract_reference(source, functions, headers, work):
    """reference.c: the named functions copied verbatim from the pinned source
    (for references whose file goto-cc cannot take whole), behind the headers
    they need. Returns its sha256."""
    text = Path(source).read_text()
    parts = []
    for name in functions:
        m = re.search(rf"^(?:static\s+)?(?:\w+\s+)*{re.escape(name)}\s*\(", text, re.M)
        if not m:
            raise ValueError(f"{name} not found in {source}")
        depth, i = 0, text.index("{", m.end())
        for j in range(i, len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                parts.append(text[m.start():j + 1])
                break
    body = (f"/* verbatim from {source}: {', '.join(functions)} */\n#include \"config.h\"\n"
            + "".join(f"#include \"{h}\"\n" for h in headers) + "\n" + "\n\n".join(parts) + "\n")
    (Path(work) / "reference.c").write_text(body)
    return hashlib.sha256(body.encode()).hexdigest()


def dav1d_config(cfg, build_dir, work):
    """config.h for compiling dav1d's C reference on the x86 host: the build's own
    config with assembly and every architecture switched off."""
    text = (Path(build_dir) / "config.h").read_text()
    text = re.sub(r"#define (HAVE_ASM|ARCH_\w+) 1", r"#define \1 0", text)
    (Path(work) / "config.h").write_text(text)


def cosim(cfg, harness, work, includes, defines, runs=100, seed=1, sources=()):
    """Native build of the harness (random inputs), run `runs` seeds. Returns
    {"passed", "runs", "failure"}."""
    work = Path(work)
    inc, dfn = _flags([host.in_container(cfg, p) for p in includes], defines)
    exe = work / "cosim"
    src = " ".join([host.in_container(cfg, harness)] + [host.in_container(cfg, s) for s in sources])
    proc = host.run(cfg, f"cd {host.in_container(cfg, work)} && gcc -O1 -w -std=gnu11 -I{host.in_container(cfg, FV_DIR)} "
                         f"-I. {inc} {dfn} {src} -o cosim 2>&1 | tail -30", timeout=900, check=False)
    if not exe.exists():
        return {"passed": False, "runs": 0, "error": True, "failure": "compile: " + proc.stdout[-3000:]}
    script = (f"cd {host.in_container(cfg, work)} && for s in $(seq {seed} {seed + runs - 1}); do "
              f"for a in 0 1; do ./cosim $s $a || {{ echo FAILED seed=$s agnostic=$a; exit 1; }}; done; done; echo ALL_OK")
    out = host.run(cfg, script, timeout=3600, check=False)
    text = out.stdout + out.stderr
    return {"passed": "ALL_OK" in text, "runs": runs, "failure": "" if "ALL_OK" in text else text[-2000:]}


def prove(cfg, harness, work, includes, defines, vlen=512, timeout=1800, solver=None, extra=""):
    """goto-cc + cbmc on the harness at one VLEN and one bounded configuration."""
    work = Path(work)
    tag = "-".join(f"{k}{v}" for k, v in sorted(defines.items())) + f"-vlen{vlen}"
    tag = re.sub(r"[^\w.-]", "_", tag)[:120]
    inc, dfn = _flags([host.in_container(cfg, p) for p in includes], {**defines, "FV_VLEN": vlen})
    w = host.in_container(cfg, work)
    src = host.in_container(cfg, harness)
    gb = f"{tag}.goto"
    log = work / f"{tag}.cbmc.log"
    solver_flag = f"--{solver}" if solver else ""
    script = (f"cd {w} && goto-cc -std=gnu11 -I{host.in_container(cfg, FV_DIR)} -I. {inc} {dfn} -DFV_CBMC=1 -DFV_CHECK_ALIGN "
              f"{src} -o {gb} > {tag}.gotocc.log 2>&1 || {{ echo GOTOCC_FAILED; tail -40 {tag}.gotocc.log; exit 0; }}; "
              f"timeout {timeout} cbmc {gb} --unwinding-assertions --object-bits 16 "
              # constant-index accesses to the register file, stack and buffers become plain
              # variables instead of array terms: W=1 wiener 5+ min -> 23 s
              f"--max-field-sensitivity-array-size 70000 {solver_flag} {extra} "
              f"--trace --json-ui > {tag}.cbmc.json 2> {tag}.cbmc.err; echo EXIT=$?")
    t0 = time.time()
    proc = host.run(cfg, script, timeout=timeout + 600, check=False)
    seconds = round(time.time() - t0, 1)
    out = proc.stdout
    if "GOTOCC_FAILED" in out:
        return Verdict("error", defines, vlen, seconds, error="goto-cc: " + out[-3000:], log=str(log))
    exit_code = int(re.search(r"EXIT=(\d+)", out).group(1)) if re.search(r"EXIT=(\d+)", out) else -1
    if exit_code == 124:
        return Verdict("timeout", defines, vlen, seconds, log=str(work / f"{tag}.cbmc.json"))
    err = work / f"{tag}.cbmc.err"
    if not solver and err.exists() and "invariant violation" in err.read_text(errors="replace"):
        # CBMC 6.6's SAT flattening hits an internal invariant (boolbv_map literal width)
        # on some lifted programs; the SMT backend does not go through it
        v = prove(cfg, harness, work, includes, defines, vlen=vlen, timeout=timeout, solver="z3", extra=extra)
        v.error = (v.error + " " if v.error else "") + "[z3 backend: CBMC SAT backend hit an internal invariant]"
        return v
    try:
        result = json.loads((work / f"{tag}.cbmc.json").read_text())
    except (OSError, ValueError) as error:
        err = (work / f"{tag}.cbmc.err").read_text()[-2000:] if (work / f"{tag}.cbmc.err").exists() else ""
        return Verdict("error", defines, vlen, seconds, error=f"cbmc output unreadable ({error}): {err}")
    return _verdict(result, defines, vlen, seconds, str(work / f"{tag}.cbmc.json"))


def _verdict(result, defines, vlen, seconds, log):
    props = []
    for item in result:
        if "result" in item:
            props = item["result"]
    failed = [p for p in props if p.get("status") == "FAILURE"]
    hard = [p for p in failed if not p.get("description", "").startswith("alignment:")]
    align = [p for p in failed if p.get("description", "").startswith("alignment:")]
    trace = {}
    if hard and hard[0].get("trace"):
        trace = _inputs(hard[0]["trace"])
    errors = [m.get("messageText", "") for item in result for m in ([item] if "messageType" in item else [])
              if item.get("messageType") == "ERROR"]
    if errors and not props:
        return Verdict("error", defines, vlen, seconds, error="; ".join(errors)[-2000:], log=log)
    verdict = "counterexample" if hard else "alignment" if align else "proved" if props else "error"
    return Verdict(verdict, defines, vlen, seconds,
                   failed=[{"property": p["property"], "description": p.get("description")} for p in hard][:20],
                   alignment=[{"property": p["property"], "description": p.get("description")} for p in align][:20],
                   trace=trace, log=log)


def _inputs(trace):
    """Assignments to harness input buffers in a CBMC counterexample (first 4 KB)."""
    out = {}
    for step in trace:
        if step.get("stepType") == "assignment" and step.get("assignmentType") == "variable":
            lhs = step.get("lhs", "")
            if lhs.startswith(("in_", "buf_", "fv_stack")) or "nondet" in lhs:
                out[lhs] = step.get("value", {}).get("data")
        if len(out) > 400:
            break
    return out


def report(verdicts):
    return [asdict(v) for v in verdicts]
