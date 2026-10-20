import json, sys
sys.path.insert(0, "/home/abror/portforge-cpu/repo")
from portforge_tt.cpu import xnnpack

rows = xnnpack.parse_bench(json.load(open("/home/abror/portforge-cpu/results/qs8-vcvt/bench-20261019T125931Z/vunary-bench.x280.json")))
ours = "xnn_qs8_vcvt_ukernel__rvv_pf1"
cmp = xnnpack.compare_bench(rows, ours, "qs8_vcvt")
scalar_ns, by_uk = {}, {}
for bench_name, row in rows.items():
    if "ns" not in row:
        continue
    uk, shape = xnnpack.split_bench_name(bench_name)
    by_uk.setdefault(uk, {})[shape] = row["ns"]
    if uk and "scalar" in uk:
        scalar_ns[shape] = row["ns"]
hdr = ("shape", "ours_ns", "scalar_ns", "vs_scalar", "vs_u1v", "vs_u2v", "vs_u4v", "vs_best")
print("%-14s %9s %9s %9s %8s %8s %8s %8s" % hdr)
faster = True
for shape, ns in sorted(cmp["ours_ns"].items()):
    peers = {uk.split("__")[-1]: t[shape] for uk, t in cmp["upstream_ns"].items() if shape in t}
    s = scalar_ns.get(shape)
    best = min(peers.values())
    faster &= (ns <= best)
    print("%-14s %9.1f %9s %9s %8s %8s %8s %8s" % (
        shape.strip("/"), ns, "%.1f" % s if s else "-",
        "%.2f" % (s / ns) if s and ns else "-",
        *(("%.3f" % (peers[k] / ns)) if k in peers else "-" for k in ("u1v", "u2v", "u4v")),
        "%.3f" % (best / ns)))
print("geomean vs best upstream RVV: %.3f over %d shapes" % (cmp["speedup_vs_best"], cmp["shapes"]))
print("gate-3 verdict (ours <= best upstream RVV on every shape):", faster)
print("vs geomeans:", {k.split("__")[-1]: round(v, 3) for k, v in cmp["vs"].items()})
