"""Formal check of a dav1d candidate (certify step of ports/dav1d-rvv).

The candidate's functions are lifted from the checkasm binary the gates built,
co-simulated natively against the C reference on random inputs (a cheap check
of the lifter and the model), then proved equal to the C reference by CBMC on
every bounded configuration below, at VLEN 128, 256, 512 and 1024. Inputs are
fully symbolic inside each configuration (pixels, taps, block contents, the
vector register file, vxrm); the configurations fix sizes and edge flags.
"""
from concurrent.futures import ThreadPoolExecutor
import itertools
from pathlib import Path
import time

from . import lift, run

HARNESS = Path(__file__).resolve().parent / "harness"
VLENS = (128, 256, 512, 1024)


def splat_configs():
    out = []
    for bw4 in (1, 2, 4, 8, 16, 32):
        for bh4 in (1, 2, 5):
            for bx4 in sorted({0, bw4, 32 - bw4}):
                if bx4 % bw4 == 0 and bx4 + bw4 <= 32:
                    out.append({"BW4": bw4, "BH4": bh4, "BX4": bx4})
    return out


def wiener_configs(taps):
    """Every edge combination without LR_HAVE_RIGHT at up to two output pixels.

    Proof cost grows steeply with the number of outputs (one pixel ~25 s, two
    ~3 min, four > 20 min: CBMC on the 7x7 window arithmetic), so the bounded
    proof covers the edge and tail logic at the smallest sizes. LR_HAVE_RIGHT
    only occurs with w >= 32 (between restoration units), out of reach here: it
    is covered by checkasm and the held-out seeds, not by this proof."""
    out = []
    for edges in range(16):
        if edges & 2:
            continue
        for w, h in ((1, 1), (2, 1), (1, 2)):
            out.append({"TAPS": taps, "W": w, "H": h, "EDGES": edges, "TAPSET": 3})
    return out


FAMILIES = {
    "refmvs-splat_mv": {"harness": "dav1d_splat_mv.c", "entries": {"": "dav1d_splat_mv_rvv"},
                        "reference": {"file": "src/refmvs.c", "functions": ["splat_mv_c"],
                                      "headers": ["src/refmvs.h"]},
                        "configs": lambda key: splat_configs()},
    "looprestoration-wiener-8bpc": {"harness": "dav1d_wiener.c",
                                    "entries": {"7": "dav1d_wiener_filter7_8bpc_rvv",
                                                "5": "dav1d_wiener_filter5_8bpc_rvv"},
                                    "configs": lambda key: wiener_configs(int(key)),
                                    "defines": {"FV_STACK_SIZE": 16384, "FV_ARENA_SIZE": 65536},
                                    "sources": ["src/tables.c"]},
}


def certify(cfg, family, entry, build_dir, out, budget=3600, vlens=VLENS, configs=None, parallel=12):
    spec = FAMILIES[family]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    binary = Path(build_dir) / "tests" / "checkasm"
    includes = [cfg.src / "dav1d", cfg.src / "dav1d" / "include", Path(build_dir), Path(build_dir) / "include"]
    t0 = time.time()
    summary = {"passed": False, "verdict": "PARTIAL", "proved": 0, "configs": 0, "failed": [], "alignment": [],
               "timeouts": [], "cosim": {}, "lifted": {}}
    total, proved = 0, 0
    for key, symbol in spec["entries"].items():
        work = out / (key or "main")
        work.mkdir(parents=True, exist_ok=True)
        try:
            meta = run.lift_binary(cfg, binary, symbol, work)
        except Exception as error:  # an unmodelled instruction fails closed
            summary.update(verdict="UNSUPPORTED", error=f"{symbol}: {error}")
            return summary
        summary["lifted"][symbol] = {k: meta[k] for k in ("functions", "instructions", "binary_sha256")}
        fn = lift.cname(meta["entry"])
        run.dav1d_config(cfg, build_dir, work)
        ref = spec.get("reference")
        if ref:
            summary["reference_sha256"] = run.extract_reference(cfg.src / "dav1d" / ref["file"], ref["functions"],
                                                                ref["headers"], work)
        harness = HARNESS / spec["harness"]
        cfgs = configs if configs is not None else spec["configs"](key)
        extra = [cfg.src / "dav1d" / s for s in spec.get("sources", [])]
        cos = run.cosim(cfg, harness, work, [work] + includes, {**cfgs[-1], **spec.get("defines", {}), "ENTRY": fn},
                        runs=50, sources=extra)
        summary["cosim"][symbol] = cos
        if not cos["passed"]:
            harness_fault = cos.get("error") or "FV-FAIL: harness:" in cos.get("failure", "")
            summary.update(verdict="ERROR" if harness_fault else "COUNTEREXAMPLE",
                           failed=[{"symbol": symbol, "cosim": cos["failure"][-1500:]}])
            return summary
        jobs = [(c, vlen) for c in cfgs for vlen in vlens]
        total += len(jobs)

        def one(job):
            c, vlen = job
            if time.time() - t0 > budget:
                return c, vlen, None
            return c, vlen, run.prove(cfg, harness, work, [work] + includes, {**c, **spec.get("defines", {}),
                                                                             "ENTRY": fn}, vlen=vlen,
                                      timeout=min(1800, max(60, int(budget - (time.time() - t0)))))

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for c, vlen, v in pool.map(one, jobs):
                if v is None:
                    summary["timeouts"].append({"symbol": symbol, **c, "vlen": vlen, "why": "budget"})
                elif v.verdict in ("proved", "alignment"):
                    proved += 1
                    if "z3 backend" in (v.error or ""):
                        summary["z3_fallback"] = summary.get("z3_fallback", 0) + 1
                    if v.verdict == "alignment":
                        summary["alignment"].append({"symbol": symbol, **c, "vlen": vlen, "where": v.alignment[:3]})
                elif v.verdict == "timeout":
                    summary["timeouts"].append({"symbol": symbol, **c, "vlen": vlen})
                else:
                    summary["failed"].append({"symbol": symbol, **c, "vlen": vlen, "verdict": v.verdict,
                                              "properties": v.failed[:5], "error": v.error[-800:],
                                              "trace": dict(list(v.trace.items())[:40])})
        if any(f["verdict"] == "counterexample" for f in summary["failed"]):
            summary.update(verdict="COUNTEREXAMPLE", proved=proved, configs=total, seconds=round(time.time() - t0))
            return summary
    summary.update(proved=proved, configs=total, seconds=round(time.time() - t0),
                   verdict="ALL_PASSED" if proved == total and not summary["failed"] else "PARTIAL")
    summary["passed"] = summary["verdict"] == "ALL_PASSED"
    return summary
