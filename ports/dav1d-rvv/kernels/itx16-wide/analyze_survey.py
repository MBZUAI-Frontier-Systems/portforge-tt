"""Parse checkasm --json bench outputs for the campaign-4 survey."""
import json
import sys
from pathlib import Path

C4 = Path("/home/abror/portforge-cpu/c4")


def parse(full):
    text = Path(full).read_text()
    start = text.find("{")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def collect(fams=("mc_8bpc", "itx_8bpc", "ipred_8bpc"), tag="seed1"):
    rows = []
    for fam in fams:
        obj = parse(C4 / f"bench-{fam}-{tag}.full")
        nop = obj["nopCycles"]["median"]
        for name, fn in obj["functions"].items():
            vs = fn["versions"]
            if "rvv" not in vs:
                continue
            c = vs["c"]["rawCycles"]["median"] - nop
            r = vs["rvv"]["rawCycles"]["median"] - nop
            rows.append({"family": fam, "name": name, "report": fn["reportName"],
                         "c": round(c, 1), "rvv": round(r, 1), "speedup": round(c / r, 3)})
    rows.sort(key=lambda x: x["speedup"])
    return rows


if __name__ == "__main__":
    rows = collect()
    (C4 / "survey-rows.json").write_text(json.dumps(rows, indent=1))
    print(len(rows), "functions with rvv+c")
    for fam in ("mc_8bpc", "itx_8bpc", "ipred_8bpc"):
        sp = sorted(r["speedup"] for r in rows if r["family"] == fam)
        print("%-11s n=%3d  median %.2f  min %.2f  max %.2f" %
              (fam, len(sp), sp[len(sp) // 2], sp[0], sp[-1]))
    print()
    print("== 30 worst (lowest speedup vs C, nop-subtracted) ==")
    print("%-10s %-32s %9s %9s %7s" % ("family", "function", "c_cyc", "rvv_cyc", "speedup"))
    for r in rows[:30]:
        print("%-10s %-32s %9.1f %9.1f %7.2f" %
              (r["family"], r["name"], r["c"], r["rvv"], r["speedup"]))
