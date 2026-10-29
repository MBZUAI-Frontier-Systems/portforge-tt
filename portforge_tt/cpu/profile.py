"""Static profile of a compiled RVV kernel: what the hot loops actually execute.

From the kernel's object code (objdump of the library the harness links), per
innermost loop (found from backward branches):
  - vsetvli configs (SEW, LMUL) and how many there are per iteration,
  - vector instructions by class (load, store, widen, narrow, fixed-point, mul,
    permute, mask, reduction, whole-register move, other ALU) and scalar ones,
  - spills (sp-relative loads/stores) and calls inside the loop,
and for the whole function: prologue calls (e.g. libgcc's __ctzdi2 when the
target lacks Zbb), conditional branches before the first loop (dispatch cost),
and .text bytes. Several innermost loops usually mean parameter dispatch, one
loop per regime.

It is the cheapest signal an agent gets: it explains most speed differences
before any board time (pf2's 0.82x at batch 256 is a __ctzdi2 call in its
dispatch; load-ahead's loss is extra vsetvli and a vmv4r per iteration).
"""
import re

from . import host, xnnpack

INSN = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*?)\s*(?:#.*)?$")
FUNC = re.compile(r"^[0-9a-f]+ <([\w.]+)>:$")
VSETVL = re.compile(r"e(8|16|32|64),(mf8|mf4|mf2|m1|m2|m4|m8)")
BRANCH = re.compile(r"^(b\w*|j)$")
RELOC_CALL = re.compile(r"^\s*([0-9a-f]+):\s+R_RISCV_CALL\w*\s+(\S+)")
CLASSES = [
    ("vsetvl", re.compile(r"^vset(i)?vli?$")),
    ("load", re.compile(r"^vl(e\d+|se\d+|uxei\d+|oxei\d+|\dre\d+|e\d+ff|m)\.v$|^vl\dr|^vl(s|ux|ox)?seg\d+e\d+(ff)?\.v$")),
    ("store", re.compile(r"^vs(e\d+|se\d+|uxei\d+|oxei\d+|\dr|m)\.v$|^vs\dr|^vs(s|ux|ox)?seg\d+e\d+\.v$")),
    ("move", re.compile(r"^vmv\dr\.v$|^vmv\.v\.v$")),
    ("fixed_point", re.compile(r"^v(s(add|sub|mul|sra|srl)|aadd|asub|nclip)u?\.")),
    ("widen", re.compile(r"^v(f)?w|^v(s|z)ext\.")),
    ("narrow", re.compile(r"^v(f)?n")),
    ("reduction", re.compile(r"^v(f|w)?red")),
    ("permute", re.compile(r"^v(rgather|slide|compress)")),
    ("mask", re.compile(r"^vm(seq|sne|sltu?|sleu?|sgtu?|sgeu?|flt|fle|feq|fne|fgt|fge|n?and|andn|n?or|orn|x?nor|xor|"
                        r"sbf|sif|sof|adc|sbc|v\.)|^vmerge|^vfmerge|^vfirst|^vcpop|^vid\.|^viota")),
    ("mul", re.compile(r"^v(f)?(mul|macc|madd|nmsac|nmsub|div|rem|sqrt)")),
]
LMUL = {"mf8": 0.125, "mf4": 0.25, "mf2": 0.5, "m1": 1, "m2": 2, "m4": 4, "m8": 8}


def classify(mnemonic):
    if not mnemonic.startswith("v"):
        return "scalar"
    for name, pattern in CLASSES:
        if pattern.search(mnemonic):
            return name
    return "alu"


def parse(text):
    """{function: [(addr, mnemonic, operands)]} from objdump -dr --no-show-raw-insn.
    Call relocations become ("call", <symbol>) entries so callees are named."""
    funcs, current = {}, None
    for line in text.splitlines():
        r = RELOC_CALL.match(line)
        if r and current is not None:
            current.append((int(r.group(1), 16), "reloc-call", r.group(2)))
            continue
        m = FUNC.match(line.strip())
        if m:
            name = m.group(1)
            if not name.startswith(".L"):  # local labels belong to the enclosing function
                current = funcs.setdefault(name, [])
            continue
        m = INSN.match(line)
        if m and current is not None:
            current.append((int(m.group(1), 16), m.group(2), m.group(3)))
    return funcs


def _target(operands):
    m = re.search(r"\b([0-9a-f]+) <", operands)
    return int(m.group(1), 16) if m else None


def loops(insns):
    """[(start, end)] address ranges closed by a backward branch, innermost only."""
    found = []
    for addr, mnem, ops in insns:
        if BRANCH.match(mnem):
            target = _target(ops)
            if target is not None and target <= addr:
                found.append((target, addr))
    inner = [l for l in found if not any(o != l and l[0] <= o[0] and o[1] <= l[1] for o in found)]
    return sorted(set(inner))


def loop_profile(insns, start, end):
    body = [(a, m, o) for a, m, o in insns if start <= a <= end]
    counts = {}
    for _, mnem, _ in body:
        if mnem == "reloc-call":
            continue
        cls = classify(mnem)
        counts[cls] = counts.get(cls, 0) + 1
    configs = [VSETVL.search(o) for _, m, o in body if classify(m) == "vsetvl"]
    configs = [(int(c.group(1)), c.group(2)) for c in configs if c]
    hoisted = None
    if not configs:  # vtype set once above the loop: the last vsetvli before it applies
        before = [VSETVL.search(o) for a, m, o in insns if a < start and classify(m) == "vsetvl"]
        before = [b for b in before if b]
        if before:
            hoisted = (int(before[-1].group(1)), before[-1].group(2))
    # register-group size of the memory ops: GCC sets vl with e8,mf4 and issues
    # vlsseg8e32 at EMUL 1, so the vtype LMUL alone understates the groups in use
    sew, lmul = (configs or [hoisted] if hoisted else configs or [(8, "m1")])[0]
    emuls, eews = [], []
    for _, m, _ in body:
        if classify(m) in ("load", "store"):
            eew = EEW.search(m)
            if eew and "xei" not in m:
                emuls.append(LMUL[lmul] * int(eew.group(1)) / sew)
                eews.append(int(eew.group(1)))
    spills = sum(1 for _, m, o in body if "(sp)" in o and m in ("sd", "ld", "sw", "lw", "fsd", "fld")
                 or "(sp)" in o and m.startswith(("vs", "vl")))
    calls = sum(1 for _, m, _ in body if m == "reloc-call")
    vector = sum(n for k, n in counts.items() if k not in ("scalar", "vsetvl"))
    return {"start": hex(start), "end": hex(end), "instructions": sum(1 for _, m, _ in body if m != "reloc-call"), "vector_ops": vector,
            "counts": counts, "vsetvl": [f"e{sew},{lmul}" for sew, lmul in configs],
            "vtype_hoisted": f"e{hoisted[0]},{hoisted[1]}" if hoisted else None,
            "max_lmul": max([LMUL[l] for _, l in configs or ([hoisted] if hoisted else [])] + emuls, default=None),
            "widest_sew": max([s for s, _ in configs or ([hoisted] if hoisted else [])] + eews, default=None),
            "spills": spills, "calls": calls}


def function_profile(insns, size=None):
    hot = [loop_profile(insns, s, e) for s, e in loops(insns)]
    hot = [h for h in hot if h["vector_ops"]] or hot
    first_loop = min((int(h["start"], 16) for h in hot), default=None)
    prologue = [(a, m, o) for a, m, o in insns if first_loop is None or a < first_loop]
    calls = [o for _, m, o in insns if m == "reloc-call"]
    profile = {
        "instructions": sum(1 for _, m, _ in insns if m != "reloc-call"), "text_bytes": size,
        "loops": hot,
        "prologue_instructions": sum(1 for _, m, _ in prologue if m != "reloc-call"),
        "prologue_branches": sum(1 for _, m, _ in prologue if BRANCH.match(m) and m != "j"),
        "calls": calls,
        "mnemonics": sorted({m for a, m, _ in insns if m != "reloc-call"
                             and any(int(h["start"], 16) <= a <= int(h["end"], 16) for h in hot)}),
    }
    profile["descriptors"] = descriptors(profile)
    return profile


def descriptors(profile):
    """MAP-Elites cell of a kernel: hot-loop LMUL, widest SEW, unroll, dispatch."""
    hot = max(profile["loops"], key=lambda h: h["vector_ops"], default=None)
    if hot is None:
        return {"lmul": None, "sew": None, "unroll": None, "dispatch": False}
    loads = hot["counts"].get("load", 0)
    return {"lmul": hot["max_lmul"], "sew": hot["widest_sew"],
            "unroll": "1" if loads <= 2 else "2" if loads <= 4 else "4+",
            "dispatch": len(profile["loops"]) > 1}


def disassemble(cfg, tree_name, ukernels):
    """{ukernel: profile} from the tree's built microkernel libraries."""
    out_dir = xnnpack.build_dir(cfg, tree_name)
    libs = " ".join(host.in_container(cfg, out_dir / f"libxnnpack-microkernels-{k}.a") for k in ("all", "prod"))
    script = "; ".join(
        [f"riscv64-linux-gnu-nm -S --defined-only {libs} 2>/dev/null | grep -E ' T ({'|'.join(ukernels)})$'"]
        + [f"echo '@@@ {u}'; riscv64-linux-gnu-objdump -dr --no-show-raw-insn --disassemble={u} {libs} "
           f"2>/dev/null | sed -n '/^[0-9a-f]* <{u}>:/,$p'" for u in ukernels])
    text = host.run(cfg, script, timeout=600).stdout
    head, _, rest = text.partition("@@@ ")
    sizes = {}
    for line in head.splitlines():
        parts = line.split()
        if len(parts) == 4:
            sizes[parts[3]] = int(parts[1], 16)
    result = {}
    for chunk in ("@@@ " + rest).split("@@@ ")[1:]:
        name, _, body = chunk.partition("\n")
        funcs = parse(body)
        if name.strip() in funcs:
            result[name.strip()] = function_profile(funcs[name.strip()], sizes.get(name.strip()))
    return result


def compare(candidate, baseline):
    """Lines an agent reads: hot loops side by side with the upstream baseline's."""
    def loop_line(h):
        c = h["counts"]
        return (f"{h['vector_ops']} vector ops ({', '.join(f'{k} {v}' for k, v in sorted(c.items()) if k not in ('scalar',))}), "
                f"{c.get('scalar', 0)} scalar, vsetvl {h['vsetvl']}, spills {h['spills']}, calls {h['calls']}")
    lines = []
    for label, p in (("candidate", candidate), ("baseline", baseline)):
        if not p:
            continue
        lines.append(f"{label}: {len(p['loops'])} hot loop(s), {p['text_bytes']} bytes, prologue "
                     f"{p['prologue_instructions']} insns / {p['prologue_branches']} branches, calls {p['calls'] or 'none'}")
        lines += [f"  loop {i}: {loop_line(h)}" for i, h in enumerate(p["loops"])]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- cost model

REPRESENTATIVE = {"alu": "vadd.vv", "mul": "vmul.vv", "widen": "vwadd.vv", "narrow": "vnclip.wi",
                  "reduction": "vredsum.vs", "mask": "vmseq.vv"}


def _op_for(mnemonic, cls):
    if cls == "fixed_point":
        return "vnclip.wi" if mnemonic.startswith("vnclip") else "vsmul.vv"
    if cls == "widen" and mnemonic.startswith(("vsext", "vzext")):
        return "vsext.vf2"
    if cls == "permute":
        return "vrgather.vv" if mnemonic.startswith(("vrgather", "vcompress")) else "vslidedown.vi"
    if mnemonic.startswith(("vmerge", "vfmerge")):
        return "vmerge.vvm"
    if mnemonic.startswith(("vfmacc", "vfmadd", "vfnmsac", "vfmsac")):
        return "vfmacc.vv"
    if mnemonic.startswith("vf") and cls == "alu":
        return "vfadd.vv"
    return REPRESENTATIVE.get(cls)


SEGMENT = re.compile(r"seg(\d)e")
EEW = re.compile(r"e(8|16|32|64)(?:ff)?\.v$")


def predict(insns, start, end, calib, vlen=512, per="vl"):
    """Predicted cycles per element of one loop from the calibrated per-op costs.

    Walks the loop in order tracking vtype (each vsetvli), sums the throughput
    cost of every vector instruction (the X280 vector unit issues in order), and
    divides by the elements one iteration handles: VLMAX of the loop's first
    vsetvli (per="vl", elementwise kernels) or every element the loop loads
    (per="loads", data movement such as transposes). Loads and stores cost the
    calibrated cycles of their exact form (strided and segment forms are far
    slower than unit stride on the X280) when the table has it, else bytes over
    L1 bandwidth; memory overlaps compute, so the loop costs max(compute, memory).
    """
    body = [(a, m, o) for a, m, o in insns if start <= a <= end and m != "reloc-call"]
    sew, lmul, first, compute, memory, unknown, loaded = 8, 1.0, None, 0.0, 0.0, [], 0.0
    if not any(classify(m) == "vsetvl" for _, m, _ in body):
        # vtype set once above the loop (hoisted vsetvli): take the last one before it
        before = [VSETVL.search(o) for a, m, o in insns if a < start and classify(m) == "vsetvl"]
        before = [b for b in before if b]
        if before:
            sew, lmul = int(before[-1].group(1)), LMUL[before[-1].group(2)]
            first = (sew, lmul)
    l1 = (calib.get("memory", {}).get("load", {}) or {}).get("L1") or 32.0
    memops = calib.get("memops", {})
    for _, mnem, ops in body:
        cls = classify(mnem)
        if cls == "vsetvl":
            m = VSETVL.search(ops)
            if m:
                sew, lmul = int(m.group(1)), LMUL[m.group(2)]
                if first is None:
                    first = (sew, lmul)
            compute += calib.get("vsetvl_cpi") or 1.0
            continue
        if cls == "scalar":
            continue
        if cls in ("load", "store"):
            seg = SEGMENT.search(mnem)
            nf = int(seg.group(1)) if seg else 1
            # unit/strided/segment forms carry their own element width: EMUL = EEW/SEW * LMUL
            # (GCC often sets vl with e8,mf4 and then issues vlsseg8e32 at EMUL 1)
            eew = EEW.search(mnem)
            eew = int(eew.group(1)) if eew and "xei" not in mnem else sew
            emul = lmul * eew / sew
            if cls == "load":
                loaded += nf * vlen * emul / eew
            exact = memops.get(mnem, {}).get(f"e{eew},m{int(emul)}") if emul >= 1 else None
            if exact is None:
                memory += nf * (vlen * emul / 8) / l1
                if memops:
                    unknown.append(mnem)
            else:
                memory += exact
            continue
        if cls == "move":
            compute += 2.0 * int(re.search(r"vmv(\d)r", mnem).group(1)) if re.search(r"vmv(\d)r", mnem) else 2.0 * lmul
            continue
        op = _op_for(mnem, cls)
        key = f"e{sew},m{int(lmul)}" if lmul >= 1 else None
        cost = calib.get("ops", {}).get(op, {}).get(key, {}).get("throughput_cpi") if op and key else None
        if cost is None:
            cost = 2.0 * max(lmul, 1)  # the common case on the X280: 2 cycles per register
            unknown.append(mnem)
        compute += cost
    if first is None:
        return None
    elements = loaded if per == "loads" and loaded else vlen * first[1] / first[0]
    return {"cycles_per_elem": max(compute, memory) / elements, "compute_cycles": compute,
            "memory_cycles": memory, "elements_per_iteration": elements,
            "bound": "compute" if compute >= memory else "memory", "uncalibrated": sorted(set(unknown))}


WORKING_SET_LEVELS = (("L1", 32 << 10), ("L2", 2 << 20), ("DRAM", float("inf")))


def roofline(measured_cpe, predicted, bytes_per_elem, shape, calib):
    """Where a measured cell sits: compute floor (the static prediction) vs the
    memory floor at the cell's working-set level; efficiency = floor / measured."""
    level = next(name for name, size in WORKING_SET_LEVELS if shape * bytes_per_elem <= size)
    bw = (calib.get("memory", {}).get("copy", {}) or {}).get(level)
    memory_floor = bytes_per_elem / bw if bw else None
    compute_floor = predicted["cycles_per_elem"] if predicted else None
    floors = [f for f in (compute_floor, memory_floor) if f]
    floor = max(floors) if floors else None
    return {"level": level, "compute_floor": compute_floor, "memory_floor": memory_floor,
            "bound": None if not floors else ("memory" if memory_floor and floor == memory_floor else "compute"),
            "efficiency": floor / measured_cpe if floor and measured_cpe else None}
