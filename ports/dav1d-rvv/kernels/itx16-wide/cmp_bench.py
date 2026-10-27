"""Compare itx_8bpc X280 bench: survey (before) vs c4-opt (after)."""
import json
import math
from pathlib import Path

C4 = Path("/home/abror/portforge-cpu/c4")


def parse(full):
    text = Path(full).read_text()
    start = text.find("{")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def collect(fam, tag):
    obj = parse(C4 / f"bench-{fam}-{tag}.full")
    nop = obj["nopCycles"]["median"]
    rows = {}
    for name, fn in obj["functions"].items():
        vs = fn["versions"]
        if "rvv" not in vs:
            continue
        c = vs["c"]["rawCycles"]["median"] - nop
        r = vs["rvv"]["rawCycles"]["median"] - nop
        rows[name] = (c, r)
    return rows


before = collect("itx_8bpc", "seed1")
after = collect("itx_8bpc", "final2-seed1")

rows = []
for name in sorted(after):
    if name not in before:
        continue
    c0, r0 = before[name]
    c1, r1 = after[name]
    rows.append((name, c0, r0, c1, r1, r0 / r1))

print("%-52s %9s %9s %9s %9s %7s" % ("function", "c_bef", "rvv_bef", "c_aft", "rvv_aft", "speedup"))
for name, c0, r0, c1, r1, sp in sorted(rows, key=lambda x: -x[5]):
    print("%-52s %9.1f %9.1f %9.1f %9.1f %7.2f" % (name, c0, r0, c1, r1, sp))

# geomean over the 16x16 family (all *_16x16_* rvv functions)
fam = [r for r in rows if "_16x16_" in r[0]]
g = math.exp(sum(math.log(r[5]) for r in fam) / len(fam))
print("\n16x16 family: %d functions, geomean rvv-after/rvv-before = %.3f" % (len(fam), g))
old = [r for r in rows if "_16x16_" in r[0]]
g_cyc = math.exp(sum(math.log(r[3] / r[1]) for r in fam) / len(fam))
print("16x16 family: geomean cycles-after/cycles-before = %.3f" % g_cyc)
tot_b = sum(r[2] for r in fam)
tot_a = sum(r[4] for r in fam)
print("16x16 family: total rvv cycles before=%.0f after=%.0f (after/before=%.3f)" % (tot_b, tot_a, tot_a / tot_b))
slow = [r for r in rows if r[5] < 1.0 and "_16x16_" in r[0]]
print("16x16 functions slower than before: %d" % len(slow), [r[0] for r in slow])
faster = [r for r in rows if r[5] >= 1.25 and "_16x16_" in r[0]]
print("16x16 functions >=1.25x faster: %d" % len(faster), [r[0] for r in faster])
# whole-itx geomean
g_all = math.exp(sum(math.log(r[5]) for r in rows) / len(rows))
print("whole itx_8bpc bench: %d functions, geomean rvv ratio = %.3f" % (len(rows), g_all))
