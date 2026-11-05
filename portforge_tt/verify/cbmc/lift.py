"""Lift RISC-V (RV64GC + Zba/Zbb/Zbs + V 1.0) machine code to C over rvv_model.h.

Input is `objdump -d --no-show-raw-insn` of a linked binary (addresses and
relocations resolved) plus the bytes of the data symbols the code addresses.
Every function reachable from the entry by direct calls becomes one C function
over the global machine state S; a label per instruction address; branches are
gotos; an indirect jump becomes a switch over the function's instruction
addresses, and a jump that leaves the function is a return, checked against the
return address the function was entered with. Anything not modelled fails
closed: the lifter raises Unsupported, it never guesses.

The same C compiles natively (co-simulation) and under CBMC (bounded proofs).
"""
from dataclasses import dataclass, field
import re


class Unsupported(Exception):
    pass


ABI = {"zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4, "t0": 5, "t1": 6, "t2": 7, "s0": 8, "fp": 8, "s1": 9,
       **{f"a{i}": 10 + i for i in range(8)}, **{f"s{i}": 16 + i for i in range(2, 12)},
       **{f"t{i}": 25 + i for i in range(3, 7)}}
ABI.update({f"x{i}": i for i in range(32)})
FABI = {**{f"ft{i}": i for i in range(8)}, "fs0": 8, "fs1": 9, **{f"fa{i}": 10 + i for i in range(8)},
        **{f"fs{i}": 16 + i for i in range(2, 12)}, **{f"ft{i}": 20 + i for i in range(8, 12)}}
FABI.update({f"f{i}": i for i in range(32)})
CSRS = {"vlenb": 0xc22, "vl": 0xc20, "vtype": 0xc21, "vxrm": 0x00a, "vxsat": 0x009, "vcsr": 0x00f, "vstart": 0x008}
LMUL8 = {"m1": 8, "m2": 16, "m4": 32, "m8": 64, "mf2": 4, "mf4": 2, "mf8": 1}

LINE = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)(?:\s+(.*?))?\s*$")
HEAD = re.compile(r"^([0-9a-f]+) <(.+)>:\s*$")


@dataclass
class Insn:
    addr: int
    mnem: str
    ops: list
    comment: str = ""


@dataclass
class Function:
    name: str
    start: int
    insns: list = field(default_factory=list)

    @property
    def end(self):
        return self.insns[-1].addr + 4 if self.insns else self.start

    def contains(self, addr):
        return any(i.addr == addr for i in self.insns)


def parse(text):
    """{name: Function} from objdump -d --no-show-raw-insn output."""
    funcs, cur = {}, None
    for line in text.splitlines():
        m = HEAD.match(line)
        if m:
            cur = Function(m.group(2), int(m.group(1), 16))
            funcs[cur.name] = cur
            continue
        m = LINE.match(line)
        if m and cur is not None:
            rest = m.group(3) or ""
            body, _, comment = rest.partition("#")
            body = re.sub(r"\s*<[^>]*>", "", body).strip()
            ops = [o.strip() for o in _split_ops(body)] if body else []
            cur.insns.append(Insn(int(m.group(1), 16), m.group(2), ops, comment.strip()))
    return funcs


def _split_ops(body):
    out, depth, cur = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def xr(name):
    if name not in ABI:
        raise Unsupported(f"not an x register: {name}")
    return ABI[name]


def vr(name):
    m = re.fullmatch(r"v(\d+)", name)
    if not m:
        raise Unsupported(f"not a vector register: {name}")
    return int(m.group(1))


def imm(text):
    text = text.strip()
    return int(text, 16) if text.lower().startswith(("0x", "-0x")) else int(text, 0)


def mem(text):
    """'imm(reg)' or '(reg)' -> (imm, reg)."""
    m = re.fullmatch(r"(-?(?:0x)?[0-9a-fA-F]*)\((\w+)\)", text.strip())
    if not m:
        raise Unsupported(f"memory operand {text!r}")
    return (imm(m.group(1)) if m.group(1) else 0), xr(m.group(2))


def X(r):
    return "0ULL" if r == 0 else f"S.x[{r}]"


def SX(expr, bits):
    return f"fv_sx({expr}, {bits})"


class Lifter:
    def __init__(self, funcs, entry, data=None):
        self.funcs = funcs
        self.entry = entry
        self.data = data or []          # [(name, addr, bytes)] the code may read
        self.by_start = {f.start: f for f in funcs.values() if f.insns}
        self.needed = []

    # ---------------------------------------------------------------- driver

    def lift(self, entries=None):
        """C for the entry (or several entries) and everything they call."""
        order, todo = [], list(entries or [self.entry])
        while todo:
            name = todo.pop()
            if name in order:
                continue
            if name not in self.funcs:
                raise Unsupported(f"no disassembly for {name}")
            order.append(name)
            for target in self.callees(self.funcs[name]):
                todo.append(target)
        self.needed = order
        protos = "".join(f"static void {cname(n)}(void);\n" for n in order)
        bodies = "\n".join(self.function(self.funcs[n]) for n in order)
        return protos + "\n" + bodies

    def callees(self, f):
        out = []
        for i in f.insns:
            if i.mnem in ("jal", "j", "call", "tail") or (i.mnem.startswith("b") and i.ops):
                target = self.target(i)
                if target is not None and not f.contains(target):
                    if target not in self.by_start:
                        raise Unsupported(f"{f.name}: {i.mnem} at {i.addr:#x} jumps into the middle of another function")
                    out.append(self.by_start[target].name)
        return out

    def target(self, i):
        for op in reversed(i.ops):
            if re.fullmatch(r"[0-9a-f]+", op):
                return int(op, 16)
        return None

    # ---------------------------------------------------------------- one function

    def function(self, f):
        addrs = [i.addr for i in f.insns]
        lines = [f"static void {cname(f.name)}(void)", "{", "    const uint64_t entry_ra = S.x[1];",
                 "    (void)entry_ra;"]
        for k, i in enumerate(f.insns):
            nxt = f.insns[k + 1].addr if k + 1 < len(f.insns) else None
            try:
                body = self.insn(f, i, nxt, addrs)
            except Unsupported as error:
                raise Unsupported(f"{f.name}+{i.addr - f.start:#x}: {i.mnem} {', '.join(i.ops)}: {error}") from None
            lines.append(f"L_{i.addr:x}: /* {i.mnem} {', '.join(i.ops)} */")
            lines.append("    " + body)
        lines.append('    FV_ASSERT(0, "fell off the end of ' + f.name + '");')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def jump(self, f, target_expr, addrs, link_rd=0, link=None):
        """Indirect jump: a switch over this function's addresses, else a return."""
        pre = f"{{ uint64_t t_ = {target_expr}; "
        if link_rd:
            pre += f"S.x[{link_rd}] = {link:#x}ULL; "
        cases = " ".join(f"case {a:#x}ULL: goto L_{a:x};" for a in addrs)
        return (pre + f"switch (t_) {{ {cases} default: "
                f'FV_ASSERT(t_ == entry_ra, "indirect jump outside the function that is not a return"); return; }} }}')

    def direct(self, f, target, rd, nxt):
        """jal / j / branches to a known address."""
        if f.contains(target):
            link = f"S.x[{rd}] = {nxt:#x}ULL; " if rd else ""
            return f"{{ {link}goto L_{target:x}; }}"
        callee = self.by_start[target]
        if rd == 0:      # tail call: the callee returns to our caller
            return f"{{ {cname(callee.name)}(); return; }}"
        if rd != 1:
            raise Unsupported("call that links a register other than ra")
        return f"{{ S.x[1] = {nxt:#x}ULL; {cname(callee.name)}(); goto L_{nxt:x}; }}"

    # ---------------------------------------------------------------- instructions

    def insn(self, f, i, nxt, addrs):
        m, o = i.mnem, i.ops
        # ------------------------------------------------ control flow
        if m == "j":
            return self.direct(f, self.target(i), 0, nxt)
        if m == "jal":
            rd = xr(o[0]) if len(o) == 2 else 1
            return self.direct(f, self.target(i), rd, nxt)
        if m in ("call", "tail"):
            return self.direct(f, self.target(i), 1 if m == "call" else 0, nxt)
        if m == "ret":
            return self.jump(f, "S.x[1]", addrs)
        if m == "jr":
            off, rs = (0, xr(o[0])) if "(" not in o[0] else mem(o[0])
            return self.jump(f, f"{X(rs)} + {off}ULL", addrs)
        if m == "jalr":
            if len(o) == 1:
                off, rs = (0, xr(o[0])) if "(" not in o[0] else mem(o[0])
                rd = 1
            elif len(o) == 2:
                rd = xr(o[0])
                off, rs = mem(o[1]) if "(" in o[1] else (0, xr(o[1]))
            else:
                rd, rs, off = xr(o[0]), xr(o[1]), imm(o[2])
            if rd == 0:
                return self.jump(f, f"{X(rs)} + {off}ULL", addrs)
            if rd == 1 and nxt is not None:
                # indirect call to a lifted function entry
                cases = " ".join(f"case {fn.start:#x}ULL: {cname(fn.name)}(); break;"
                                 for fn in (self.funcs[n] for n in self.needed) if fn.insns)
                return (f"{{ uint64_t t_ = {X(rs)} + {off}ULL; S.x[1] = {nxt:#x}ULL; "
                        f"if (t_ >= {f.start:#x}ULL && t_ < {f.end:#x}ULL) {{ "
                        + self.jump(f, "t_", addrs)[1:-1] + f" }} else {{ switch (t_) {{ {cases} "
                        f'default: FV_ASSERT(0, "indirect call to an unknown function"); }} goto L_{nxt:x}; }} }}')
            raise Unsupported("jalr linking a register other than ra")
        branch = self.branch(i)
        if branch:
            cond, target = branch
            if f.contains(target):
                return f"if ({cond}) goto L_{target:x};"
            return f"if ({cond}) {self.direct(f, target, 0, nxt)}"
        # ------------------------------------------------ scalar
        s = self.scalar(i)
        if s is not None:
            return s
        if m.startswith("v"):
            return self.vector(i)
        raise Unsupported("unknown mnemonic")

    def branch(self, i):
        m, o = i.mnem, i.ops
        two = {"beq": "==", "bne": "!=", "blt": "<s", "bge": ">=s", "bltu": "<", "bgeu": ">=",
               "bgt": ">s", "ble": "<=s", "bgtu": ">", "bleu": "<="}
        zero = {"beqz": "==", "bnez": "!=", "bltz": "<s", "bgez": ">=s", "bgtz": ">s", "blez": "<=s"}
        if m in two:
            a, b, op = X(xr(o[0])), X(xr(o[1])), two[m]
        elif m in zero:
            a, b, op = X(xr(o[0])), "0ULL", zero[m]
        else:
            return None
        if op.endswith("s"):
            return f"(int64_t){a} {op[:-1]} (int64_t){b}", self.target(i)
        return f"{a} {op} {b}", self.target(i)

    def scalar(self, i):
        m, o = i.mnem, i.ops

        def w(rd, expr):
            return f"S.x[{rd}] = {expr};" if rd else f"(void)({expr});"

        if m == "nop" or m.startswith("fence") or m == "pause":
            return ";"
        if m == "mv":
            return w(xr(o[0]), X(xr(o[1])))
        if m == "li":
            return w(xr(o[0]), f"(uint64_t){imm(o[1])}LL")
        if m == "lui":
            return w(xr(o[0]), f"(uint64_t)(int64_t)(int32_t)({imm(o[1]) & 0xfffff}u << 12)")
        if m == "auipc":
            return w(xr(o[0]), f"{i.addr:#x}ULL + (uint64_t)(int64_t)(int32_t)({imm(o[1]) & 0xfffff}u << 12)")
        if m in ("neg", "negw", "not", "sext.w", "seqz", "snez", "sltz", "sgtz", "zext.b", "zext.h", "zext.w",
                 "sext.b", "sext.h"):
            rd, a = xr(o[0]), X(xr(o[1]))
            expr = {"neg": f"0ULL - {a}", "negw": SX(f"0ULL - {a}", 32), "not": f"~{a}", "sext.w": SX(a, 32),
                    "seqz": f"(uint64_t)({a} == 0)", "snez": f"(uint64_t)({a} != 0)",
                    "sltz": f"(uint64_t)((int64_t){a} < 0)", "sgtz": f"(uint64_t)((int64_t){a} > 0)",
                    "zext.b": f"{a} & 0xffULL", "zext.h": f"{a} & 0xffffULL", "zext.w": f"{a} & 0xffffffffULL",
                    "sext.b": SX(a, 8), "sext.h": SX(a, 16)}[m]
            return w(rd, expr)
        r3 = {"add": "{a} + {b}", "sub": "{a} - {b}", "and": "{a} & {b}", "or": "{a} | {b}", "xor": "{a} ^ {b}",
              "sll": "{a} << ({b} & 63)", "srl": "{a} >> ({b} & 63)",
              "sra": "(uint64_t)((int64_t){a} >> ({b} & 63))",
              "slt": "(uint64_t)((int64_t){a} < (int64_t){b})", "sltu": "(uint64_t)({a} < {b})",
              "addw": SX("{a} + {b}", 32), "subw": SX("{a} - {b}", 32),
              "sllw": SX("{a} << ({b} & 31)", 32), "srlw": SX("({a} & 0xffffffffULL) >> ({b} & 31)", 32),
              "sraw": "(uint64_t)(int64_t)((int32_t){a} >> ({b} & 31))",
              "mul": "{a} * {b}", "mulw": SX("{a} * {b}", 32),
              "mulh": "(uint64_t)(((__int128)(int64_t){a} * (__int128)(int64_t){b}) >> 64)",
              "mulhu": "(uint64_t)(((unsigned __int128){a} * (unsigned __int128){b}) >> 64)",
              "mulhsu": "(uint64_t)(((__int128)(int64_t){a} * (__int128)(unsigned __int128){b}) >> 64)",
              "div": "fv_div({a}, {b}, 64, 1, 0)", "divu": "fv_div({a}, {b}, 64, 0, 0)",
              "rem": "fv_div({a}, {b}, 64, 1, 1)", "remu": "fv_div({a}, {b}, 64, 0, 1)",
              "divw": "fv_div({a}, {b}, 32, 1, 0)", "divuw": "fv_div({a}, {b}, 32, 0, 0)",
              "remw": "fv_div({a}, {b}, 32, 1, 1)", "remuw": "fv_div({a}, {b}, 32, 0, 1)",
              "sh1add": "({a} << 1) + {b}", "sh2add": "({a} << 2) + {b}", "sh3add": "({a} << 3) + {b}",
              "add.uw": "({a} & 0xffffffffULL) + {b}", "sh1add.uw": "(({a} & 0xffffffffULL) << 1) + {b}",
              "sh2add.uw": "(({a} & 0xffffffffULL) << 2) + {b}", "sh3add.uw": "(({a} & 0xffffffffULL) << 3) + {b}",
              "andn": "{a} & ~{b}", "orn": "{a} | ~{b}", "xnor": "~({a} ^ {b})",
              "max": "((int64_t){a} > (int64_t){b} ? {a} : {b})", "maxu": "({a} > {b} ? {a} : {b})",
              "min": "((int64_t){a} < (int64_t){b} ? {a} : {b})", "minu": "({a} < {b} ? {a} : {b})",
              "rol": "(({a} << ({b} & 63)) | ({a} >> ((64 - ({b} & 63)) & 63)))",
              "ror": "(({a} >> ({b} & 63)) | ({a} << ((64 - ({b} & 63)) & 63)))",
              "bset": "{a} | (1ULL << ({b} & 63))", "bclr": "{a} & ~(1ULL << ({b} & 63))",
              "binv": "{a} ^ (1ULL << ({b} & 63))", "bext": "(({a} >> ({b} & 63)) & 1)"}
        if m in r3 and len(o) == 3 and not _is_imm(o[2]):
            return w(xr(o[0]), r3[m].format(a=X(xr(o[1])), b=X(xr(o[2]))))
        ri = {"addi": "{a} + {b}", "andi": "{a} & {b}", "ori": "{a} | {b}", "xori": "{a} ^ {b}",
              "slti": "(uint64_t)((int64_t){a} < (int64_t){b})", "sltiu": "(uint64_t)({a} < {b})",
              "slli": "{a} << {b}", "srli": "{a} >> {b}", "srai": "(uint64_t)((int64_t){a} >> {b})",
              "addiw": SX("{a} + {b}", 32), "slliw": SX("{a} << {b}", 32),
              "srliw": SX("({a} & 0xffffffffULL) >> {b}", 32), "sraiw": "(uint64_t)(int64_t)((int32_t){a} >> {b})",
              "slli.uw": "(({a} & 0xffffffffULL) << {b})",
              "rori": "(({a} >> {b}) | ({a} << ((64 - {b}) & 63)))",
              "roriw": SX("((({a} & 0xffffffffULL) >> {b}) | ({a} << ((32 - {b}) & 31)))", 32),
              "bseti": "{a} | (1ULL << {b})", "bclri": "{a} & ~(1ULL << {b})", "binvi": "{a} ^ (1ULL << {b})",
              "bexti": "(({a} >> {b}) & 1)"}
        alias_imm = {"add": "addi", "and": "andi", "or": "ori", "xor": "xori", "sll": "slli", "srl": "srli",
                     "sra": "srai", "addw": "addiw", "sllw": "slliw", "srlw": "srliw", "sraw": "sraiw"}
        if m in alias_imm and len(o) == 3 and _is_imm(o[2]):
            m = alias_imm[m]
        if m in ri:
            return w(xr(o[0]), ri[m].format(a=X(xr(o[1])), b=f"(uint64_t){imm(o[2])}LL"))
        un = {"clz": "fv_clz({a}, 64)", "ctz": "fv_ctz({a}, 64)", "cpop": "fv_cpop({a}, 64)",
              "clzw": "fv_clz({a}, 32)", "ctzw": "fv_ctz({a}, 32)", "cpopw": "fv_cpop({a}, 32)",
              "rev8": "__builtin_bswap64({a})",
              "orc.b": "fv_orcb({a})"}
        if m in un:
            if m == "orc.b":
                raise Unsupported("orc.b")
            return w(xr(o[0]), un[m].format(a=X(xr(o[1]))))
        loads = {"lb": (1, 8), "lbu": (1, 0), "lh": (2, 16), "lhu": (2, 0), "lw": (4, 32), "lwu": (4, 0),
                 "ld": (8, 0)}
        if m in loads:
            n, sx = loads[m]
            off, rs = mem(o[1])
            val = f"fv_load({X(rs)} + (uint64_t){off}LL, {n})"
            return w(xr(o[0]), SX(val, sx) if sx else val)
        stores = {"sb": 1, "sh": 2, "sw": 4, "sd": 8}
        if m in stores:
            off, rs = mem(o[1])
            return f"fv_store({X(rs)} + (uint64_t){off}LL, {stores[m]}, {X(xr(o[0]))});"
        if m in ("fld", "flw"):
            off, rs = mem(o[1])
            return f"S.f[{FABI[o[0]]}] = fv_load({X(rs)} + (uint64_t){off}LL, {8 if m == 'fld' else 4});"
        if m in ("fsd", "fsw"):
            off, rs = mem(o[1])
            return f"fv_store({X(rs)} + (uint64_t){off}LL, {8 if m == 'fsd' else 4}, S.f[{FABI[o[0]]}]);"
        if m in ("csrr", "csrw", "csrwi", "csrrw", "csrrwi", "csrs", "csrc", "csrsi", "csrci"):
            return self.csr(m, o)
        return None

    def csr(self, m, o):
        def num(name):
            if name in CSRS:
                return CSRS[name]
            raise Unsupported(f"csr {name}")
        if m == "csrr":
            rd = xr(o[0])
            return f"S.x[{rd}] = fv_csr_read({num(o[1])});" if rd else ";"
        if m == "csrw":
            return f"fv_csr_write({num(o[0])}, {X(xr(o[1]))});"
        if m == "csrwi":
            return f"fv_csr_write({num(o[0])}, {imm(o[1])});"
        if m in ("csrrw", "csrrwi"):
            rd, c = xr(o[0]), num(o[1])
            src = X(xr(o[2])) if m == "csrrw" else str(imm(o[2]))
            save = f"S.x[{rd}] = old_;" if rd else ""
            return f"{{ uint64_t old_ = fv_csr_read({c}); fv_csr_write({c}, {src}); {save} }}"
        if m in ("csrs", "csrc", "csrsi", "csrci"):
            c = num(o[0])
            src = X(xr(o[1])) if m in ("csrs", "csrc") else str(imm(o[1]))
            op = "|" if m.startswith("csrs") else "& ~"
            return f"fv_csr_write({c}, fv_csr_read({c}) {op}({src}));"
        raise Unsupported(m)

    # ---------------------------------------------------------------- vector

    def vector(self, i):
        m, o = i.mnem, list(i.ops)
        masked = 0
        if o and o[-1] == "v0.t":
            masked = 1
            o = o[:-1]
        if m in ("vsetvli", "vsetivli"):
            rd = xr(o[0])
            sew = int(o[2][1:])
            lmul8 = LMUL8[o[3]]
            ta = 1 if "ta" in o[4:] else 0
            ma = 1 if "ma" in o[4:] else 0
            vt = f"fv_vtype_imm({sew}, {lmul8}, {ta}, {ma})"
            if m == "vsetivli":
                call = f"fv_set_vtype({vt}, {imm(o[1])}ULL, 0, 0)"
            else:
                rs1 = xr(o[1])
                keep = int(rs1 == 0 and rd == 0)
                vmax = int(rs1 == 0 and rd != 0)
                call = f"fv_set_vtype({vt}, {X(rs1)}, {keep}, {vmax})"
            return f"S.x[{rd}] = {call};" if rd else f"(void){call};"
        if m == "vsetvl":
            rd, rs1, rs2 = xr(o[0]), xr(o[1]), xr(o[2])
            call = f"fv_set_vtype({X(rs2)}, {X(rs1)}, {int(rs1 == 0 and rd == 0)}, {int(rs1 == 0 and rd != 0)})"
            return f"S.x[{rd}] = {call};" if rd else f"(void){call};"
        # aliases
        alias = {"vneg.v": ("vrsub.vx", lambda o: [o[0], o[1], "zero"]),
                 "vnot.v": ("vxor.vi", lambda o: [o[0], o[1], "-1"]),
                 "vncvt.x.x.w": ("vnsrl.wx", lambda o: [o[0], o[1], "zero"]),
                 "vwcvt.x.x.v": ("vwadd.vx", lambda o: [o[0], o[1], "zero"]),
                 "vwcvtu.x.x.v": ("vwaddu.vx", lambda o: [o[0], o[1], "zero"]),
                 "vmmv.m": ("vmand.mm", lambda o: [o[0], o[1], o[1]]),
                 "vmnot.m": ("vmnand.mm", lambda o: [o[0], o[1], o[1]]),
                 "vmclr.m": ("vmxor.mm", lambda o: [o[0], o[0], o[0]]),
                 "vmset.m": ("vmxnor.mm", lambda o: [o[0], o[0], o[0]])}
        if m in alias:
            m, o = alias[m][0], alias[m][1](o)
        # memory
        r = self.vmem(m, o, masked)
        if r:
            return r
        if m in ("vmv.v.v", "vmv.v.x", "vmv.v.i"):
            vd = vr(o[0])
            if m == "vmv.v.v":
                return f"fv_vmerge({vd}, 0, FV_SRC_V, {vr(o[1])}, 0, 0);"
            val = X(xr(o[1])) if m == "vmv.v.x" else f"(uint64_t){imm(o[1])}LL"
            return f"fv_vmerge({vd}, 0, FV_SRC_X, 0, {val}, 0);"
        if m in ("vmerge.vvm", "vmerge.vxm", "vmerge.vim"):
            vd, vs2 = vr(o[0]), vr(o[1])
            if m == "vmerge.vvm":
                return f"fv_vmerge({vd}, {vs2}, FV_SRC_V, {vr(o[2])}, 0, 1);"
            val = X(xr(o[2])) if m == "vmerge.vxm" else f"(uint64_t){imm(o[2])}LL"
            return f"fv_vmerge({vd}, {vs2}, FV_SRC_X, 0, {val}, 1);"
        if m == "vmv.x.s":
            return f"fv_vmv_x_s({xr(o[0])}, {vr(o[1])});"
        if m == "vmv.s.x":
            return f"fv_vmv_s_x({vr(o[0])}, {X(xr(o[1]))});"
        mm = re.fullmatch(r"vmv(\d)r\.v", m)
        if mm:
            return f"fv_vmvnr({vr(o[0])}, {vr(o[1])}, {mm.group(1)});"
        if m == "vid.v":
            return f"fv_vid({vr(o[0])}, {masked});"
        mm = re.fullmatch(r"v([zs])ext\.vf([248])", m)
        if mm:
            return f"fv_vext({vr(o[0])}, {vr(o[1])}, {mm.group(2)}, {int(mm.group(1) == 's')}, {masked});"
        base, _, form = m.partition(".")
        # integer single-width
        single = {"vadd": "OP_ADD", "vsub": "OP_SUB", "vrsub": "OP_RSUB", "vand": "OP_AND", "vor": "OP_OR",
                  "vxor": "OP_XOR", "vsll": "OP_SLL", "vsrl": "OP_SRL", "vsra": "OP_SRA", "vmin": "OP_MIN",
                  "vminu": "OP_MINU", "vmax": "OP_MAX", "vmaxu": "OP_MAXU", "vmul": "OP_MUL", "vmulh": "OP_MULH",
                  "vmulhu": "OP_MULHU", "vmulhsu": "OP_MULHSU", "vdiv": "OP_DIV", "vdivu": "OP_DIVU",
                  "vrem": "OP_REM", "vremu": "OP_REMU", "vsadd": "OP_SADD", "vsaddu": "OP_SADDU",
                  "vssub": "OP_SSUB", "vssubu": "OP_SSUBU", "vaadd": "OP_AADD", "vaaddu": "OP_AADDU",
                  "vasub": "OP_ASUB", "vasubu": "OP_ASUBU", "vsmul": "OP_SMUL", "vssrl": "OP_SSRL",
                  "vssra": "OP_SSRA"}
        compare = {"vmseq": "OP_MSEQ", "vmsne": "OP_MSNE", "vmslt": "OP_MSLT", "vmsltu": "OP_MSLTU",
                   "vmsle": "OP_MSLE", "vmsleu": "OP_MSLEU", "vmsgt": "OP_MSGT", "vmsgtu": "OP_MSGTU"}
        unsigned_imm = {"vsll", "vsrl", "vsra", "vssrl", "vssra", "vnsrl", "vnsra", "vnclip", "vnclipu",
                        "vslideup", "vslidedown", "vrgather"}
        if base in single or base in compare:
            fn = "fv_vop" if base in single else "fv_vcmp"
            op = single.get(base) or compare[base]
            vd, vs2 = vr(o[0]), vr(o[1])
            if form == "vv":
                return f"{fn}({op}, {vd}, {vs2}, FV_SRC_V, {vr(o[2])}, 0, {masked});"
            if form == "vx":
                return f"{fn}({op}, {vd}, {vs2}, FV_SRC_X, 0, {X(xr(o[2]))}, {masked});"
            if form == "vi":
                v = imm(o[2])
                if base in unsigned_imm:
                    v &= 31
                return f"{fn}({op}, {vd}, {vs2}, FV_SRC_X, 0, (uint64_t){v}LL, {masked});"
            raise Unsupported(f"form {form}")
        muladd = {"vmacc": "OP_MACC", "vnmsac": "OP_NMSAC", "vmadd": "OP_MADD", "vnmsub": "OP_NMSUB"}
        if base in muladd:
            vd = vr(o[0])
            vs2 = vr(o[2])
            if form == "vv":
                return f"fv_vmuladd({muladd[base]}, {vd}, FV_SRC_V, {vr(o[1])}, 0, {vs2}, {masked});"
            return f"fv_vmuladd({muladd[base]}, {vd}, FV_SRC_X, 0, {X(xr(o[1]))}, {vs2}, {masked});"
        widen = {("vwadd", "vv"): "W_ADD", ("vwadd", "vx"): "W_ADD", ("vwaddu", "vv"): "W_ADDU",
                 ("vwaddu", "vx"): "W_ADDU", ("vwsub", "vv"): "W_SUB", ("vwsub", "vx"): "W_SUB",
                 ("vwsubu", "vv"): "W_SUBU", ("vwsubu", "vx"): "W_SUBU",
                 ("vwadd", "wv"): "W_ADD_W", ("vwadd", "wx"): "W_ADD_W", ("vwaddu", "wv"): "W_ADDU_W",
                 ("vwaddu", "wx"): "W_ADDU_W", ("vwsub", "wv"): "W_SUB_W", ("vwsub", "wx"): "W_SUB_W",
                 ("vwsubu", "wv"): "W_SUBU_W", ("vwsubu", "wx"): "W_SUBU_W",
                 ("vwmul", "vv"): "W_MUL", ("vwmul", "vx"): "W_MUL", ("vwmulu", "vv"): "W_MULU",
                 ("vwmulu", "vx"): "W_MULU", ("vwmulsu", "vv"): "W_MULSU", ("vwmulsu", "vx"): "W_MULSU"}
        if (base, form) in widen:
            vd, vs2 = vr(o[0]), vr(o[1])
            if form in ("vv", "wv"):
                return f"fv_vwop({widen[base, form]}, {vd}, {vs2}, FV_SRC_V, {vr(o[2])}, 0, {masked});"
            return f"fv_vwop({widen[base, form]}, {vd}, {vs2}, FV_SRC_X, 0, {X(xr(o[2]))}, {masked});"
        wmacc = {"vwmacc": "W_MACC", "vwmaccu": "W_MACCU", "vwmaccsu": "W_MACCSU", "vwmaccus": "W_MACCUS"}
        if base in wmacc:
            vd, vs2 = vr(o[0]), vr(o[2])
            if form == "vv":
                return f"fv_vwop({wmacc[base]}, {vd}, {vs2}, FV_SRC_V, {vr(o[1])}, 0, {masked});"
            return f"fv_vwop({wmacc[base]}, {vd}, {vs2}, FV_SRC_X, 0, {X(xr(o[1]))}, {masked});"
        narrow = {"vnsrl": "N_SRL", "vnsra": "N_SRA", "vnclip": "N_CLIP", "vnclipu": "N_CLIPU"}
        if base in narrow:
            vd, vs2 = vr(o[0]), vr(o[1])
            if form == "wv":
                return f"fv_vnop({narrow[base]}, {vd}, {vs2}, FV_SRC_V, {vr(o[2])}, 0, {masked});"
            if form == "wx":
                return f"fv_vnop({narrow[base]}, {vd}, {vs2}, FV_SRC_X, 0, {X(xr(o[2]))}, {masked});"
            return f"fv_vnop({narrow[base]}, {vd}, {vs2}, FV_SRC_X, 0, (uint64_t){imm(o[2]) & 31}LL, {masked});"
        if base == "vrgather" or base == "vrgatherei16":
            vd, vs2 = vr(o[0]), vr(o[1])
            if form == "vv":
                eew = "16" if base == "vrgatherei16" else "S.sew"
                return f"fv_vrgather({vd}, {vs2}, FV_SRC_V, {vr(o[2])}, 0, {eew}, {masked});"
            val = X(xr(o[2])) if form == "vx" else f"(uint64_t){imm(o[2]) & 31}LL"
            return f"fv_vrgather({vd}, {vs2}, FV_SRC_X, 0, {val}, 0, {masked});"
        if base in ("vslideup", "vslidedown"):
            vd, vs2 = vr(o[0]), vr(o[1])
            val = X(xr(o[2])) if form == "vx" else f"(uint64_t){imm(o[2]) & 31}LL"
            return f"fv_{base}({vd}, {vs2}, {val}, {masked});"
        if base in ("vslide1up", "vslide1down"):
            return f"fv_{base}({vr(o[0])}, {vr(o[1])}, {X(xr(o[2]))}, {masked});"
        if m == "vcompress.vm":
            return f"fv_vcompress({vr(o[0])}, {vr(o[1])}, {vr(o[2])});"
        reds = {"vredsum": "R_SUM", "vredmax": "R_MAX", "vredmaxu": "R_MAXU", "vredmin": "R_MIN",
                "vredminu": "R_MINU", "vredand": "R_AND", "vredor": "R_OR", "vredxor": "R_XOR",
                "vwredsum": "R_WSUM", "vwredsumu": "R_WSUMU"}
        if base in reds and form == "vs":
            return f"fv_vred({reds[base]}, {vr(o[0])}, {vr(o[1])}, {vr(o[2])}, {masked});"
        mlog = {"vmand": "M_AND", "vmnand": "M_NAND", "vmandn": "M_ANDN", "vmxor": "M_XOR", "vmor": "M_OR",
                "vmnor": "M_NOR", "vmorn": "M_ORN", "vmxnor": "M_XNOR"}
        if base in mlog and form == "mm":
            return f"fv_vmlogic({mlog[base]}, {vr(o[0])}, {vr(o[1])}, {vr(o[2])});"
        if m == "vcpop.m":
            rd = xr(o[0])
            return f"S.x[{rd}] = fv_vcpop({vr(o[1])}, {masked});" if rd else ";"
        if m == "vfirst.m":
            rd = xr(o[0])
            return f"S.x[{rd}] = fv_vfirst({vr(o[1])}, {masked});" if rd else ";"
        raise Unsupported("vector instruction not modelled")

    def vmem(self, m, o, masked):
        mm = re.fullmatch(r"v([ls])(?:seg(\d))?e(\d+)(ff)?\.v", m)
        if mm:
            store, nf, eew, ff = mm.group(1) == "s", int(mm.group(2) or 1), int(mm.group(3)), mm.group(4)
            if ff:
                raise Unsupported("fault-only-first loads are not modelled yet")
            _, rs = mem(o[1])
            return f"fv_vmem({int(store)}, A_UNIT, {vr(o[0])}, {X(rs)}, 0, 0, {eew}, 0, {nf}, {masked});"
        mm = re.fullmatch(r"v([ls])s(?:seg(\d))?e(\d+)\.v", m)
        if mm:
            store, nf, eew = mm.group(1) == "s", int(mm.group(2) or 1), int(mm.group(3))
            _, rs = mem(o[1])
            return (f"fv_vmem({int(store)}, A_STRIDED, {vr(o[0])}, {X(rs)}, {X(xr(o[2]))}, 0, {eew}, 0, {nf}, "
                    f"{masked});")
        mm = re.fullmatch(r"v([ls])[uo]x(?:seg(\d))?ei(\d+)\.v", m)
        if mm:
            store, nf, ieew = mm.group(1) == "s", int(mm.group(2) or 1), int(mm.group(3))
            _, rs = mem(o[1])
            return (f"fv_vmem({int(store)}, A_INDEXED, {vr(o[0])}, {X(rs)}, 0, {vr(o[2])}, S.sew, {ieew}, {nf}, "
                    f"{masked});")
        mm = re.fullmatch(r"vl(\d)re(\d+)\.v|vl(\d)r\.v", m)
        if mm:
            _, rs = mem(o[1])
            return f"fv_vmem_whole(0, {vr(o[0])}, {X(rs)}, {mm.group(1) or mm.group(3)});"
        mm = re.fullmatch(r"vs(\d)r\.v", m)
        if mm:
            _, rs = mem(o[1])
            return f"fv_vmem_whole(1, {vr(o[0])}, {X(rs)}, {mm.group(1)});"
        if m in ("vlm.v", "vsm.v"):
            _, rs = mem(o[1])
            return f"fv_vmem_mask({int(m == 'vsm.v')}, {vr(o[0])}, {X(rs)});"
        return None


def _is_imm(text):
    return bool(re.fullmatch(r"-?(0x[0-9a-fA-F]+|\d+)", text.strip()))


def cname(name):
    return "fvf_" + re.sub(r"\W", "_", name)


def lift(objdump_text, entry, data=None):
    """C source for the entry function and everything it calls."""
    funcs = parse(objdump_text)
    return Lifter(funcs, entry, data).lift()
