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


def sgr5x5_configs():
    """Same shape as sgr3x3: edge sweep (no LR_HAVE_RIGHT) at up to two output
    pixels on the canonical 5x5 row (idx 14), the other legal row (15) at the
    smallest size, and a full-border cosim config last."""
    out = []
    for edges in range(16):
        if edges & 2:
            continue
        for w, h in ((1, 1), (2, 1), (1, 2)):
            out.append({"W": w, "H": h, "EDGES": edges, "SROW": 14})
    out.append({"W": 1, "H": 1, "EDGES": 0, "SROW": 15})
    out.append({"W": 2, "H": 2, "EDGES": 13, "SROW": 14})   # cosim: every border but RIGHT
    return out


def sgrmix_configs():
    """Same shape as sgr3x3 on the canonical mix row (idx 0); two more legal
    rows (5, 9) at the smallest size; full-border cosim config last."""
    out = []
    for edges in range(16):
        if edges & 2:
            continue
        for w, h in ((1, 1), (2, 1), (1, 2)):
            out.append({"W": w, "H": h, "EDGES": edges, "SROW": 0})
    for srow in (5, 9):
        out.append({"W": 1, "H": 1, "EDGES": 0, "SROW": srow})
    out.append({"W": 2, "H": 2, "EDGES": 13, "SROW": 0})   # cosim: every border but RIGHT
    return out


def lpf_uv_configs(dir_key):
    """One direction (h or v): vmask patterns over 4 blocks. The words are
    concrete per configuration (a symbolic vmask makes the reference
    wrapper's bit loop cost CBMC minutes per unwind check; the bits' effects
    are covered by enumeration): all-wd4, all-wd6, a wd4/wd6 mix, a single
    block, and a single wd6 block. Levels, sharpness and pixels stay
    symbolic. The last config is the cosim one (richest mix)."""
    d = 0 if dir_key == "h" else 1
    pats = [(0xf, 0x0), (0xf, 0xf), (0x1, 0x0), (0x0, 0x8), (0x5, 0xa)]
    return [{"DIR": d, "NBLKS": 4, "VM0": vm0, "VM1": vm1} for vm0, vm1 in pats]


def sgr3x3_configs():
    """Every edge combination without LR_HAVE_RIGHT at up to two output pixels,
    at the decoder's canonical 3x3 param row (idx 10), plus the other three
    legal s1 rows at the smallest size. Same proof-cost shape as wiener: the
    per-pixel work (box sums, the u32-wrap p*s, the LUT) grows steeply with
    the number of outputs, so the bounded proof covers the edge and startup
    logic at the smallest sizes. LR_HAVE_RIGHT only occurs with w >= 32
    (between restoration units), out of reach here: covered by checkasm and
    the held-out seeds, not by this proof."""
    out = []
    for edges in range(16):
        if edges & 2:
            continue
        for w, h in ((1, 1), (2, 1), (1, 2)):
            out.append({"W": w, "H": h, "EDGES": edges, "SROW": 10})
    for srow in (11, 12, 13):
        out.append({"W": 1, "H": 1, "EDGES": 0, "SROW": srow})
    out.append({"W": 2, "H": 2, "EDGES": 13, "SROW": 10})   # cosim: every border but RIGHT
    return out


def tmvs_configs():
    """Column/row windows over the small bounded frame: both single-reference
    and three-reference runs, windows touching the frame edges and inside.
    The block-walk, the projection windowing and the invalid-entry skips are
    all data-driven (symbolic MVs, refs and distances), so the configs bound
    only the geometry. The last config is the cosim one."""
    return [
        {"W8": 16, "CS8": 0, "CE8": 4, "RS8": 0, "RE8": 1, "NMV": 1},
        {"W8": 16, "CS8": 0, "CE8": 4, "RS8": 0, "RE8": 2, "NMV": 3},
        {"W8": 16, "CS8": 2, "CE8": 14, "RS8": 0, "RE8": 2, "NMV": 3},
        {"W8": 16, "CS8": 0, "CE8": 16, "RS8": 0, "RE8": 2, "NMV": 3},   # cosim: full width
    ]


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
    "looprestoration-sgr3x3-8bpc": {"harness": "dav1d_sgr3x3.c",
                                    "entries": {"": "dav1d_sgr_filter_3x3_8bpc_rvv"},
                                    "configs": lambda key: sgr3x3_configs(),
                                    "defines": {"FV_STACK_SIZE": 16384, "FV_ARENA_SIZE": 65536},
                                    "sources": ["src/tables.c"]},
    "looprestoration-sgr5x5-8bpc": {"harness": "dav1d_sgr5x5.c",
                                    "entries": {"": "dav1d_sgr_filter_5x5_8bpc_rvv"},
                                    "configs": lambda key: sgr5x5_configs(),
                                    "defines": {"FV_STACK_SIZE": 32768, "FV_ARENA_SIZE": 131072},
                                    "sources": ["src/tables.c"]},
    "looprestoration-sgrmix-8bpc": {"harness": "dav1d_sgrmix.c",
                                    "entries": {"": "dav1d_sgr_filter_mix_8bpc_rvv"},
                                    "configs": lambda key: sgrmix_configs(),
                                    "defines": {"FV_STACK_SIZE": 49152, "FV_ARENA_SIZE": 196608},
                                    "sources": ["src/tables.c"]},
    "loopfilter-uv-8bpc": {"harness": "dav1d_lpf_uv.c",
                           "entries": {"h": "dav1d_lpf_h_sb_uv_8bpc_rvv",
                                       "v": "dav1d_lpf_v_sb_uv_8bpc_rvv"},
                           "configs": lpf_uv_configs,
                           "defines": {"FV_STACK_SIZE": 16384, "FV_ARENA_SIZE": 65536}},
    "refmvs-load_tmvs": {"harness": "dav1d_load_tmvs.c",
                         "entries": {"": "dav1d_load_tmvs_rvv"},
                         "reference": {"file": "src/refmvs.c",
                                       "functions": ["mv_projection", "load_tmvs_c"],
                                       "headers": ["src/refmvs.h"]},
                         "configs": lambda key: tmvs_configs(),
                         "defines": {"FV_STACK_SIZE": 16384, "FV_ARENA_SIZE": 131072}},
}


def certify(cfg, family, entry, build_dir, out, budget=3600, vlens=VLENS, configs=None, parallel=12,
            proof_timeout=1800):
    spec = FAMILIES[family]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    binary = Path(build_dir) / "tests" / "checkasm"
    includes = [cfg.src / "dav1d", cfg.src / "dav1d" / "include", Path(build_dir), Path(build_dir) / "include"]
    t0 = time.time()
    summary = {"passed": False, "verdict": "PARTIAL", "proved": 0, "configs": 0, "failed": [], "alignment": [],
               "timeouts": [], "cosim": {}, "lifted": {}}
    total, proved, attempted = 0, 0, 0
    for key, symbol in spec["entries"].items():
        if not (configs if configs is not None else spec["configs"](key)):
            continue   # a subset run with nothing to prove for this entry
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
                                      timeout=min(proof_timeout, max(60, int(budget - (time.time() - t0)))),
                                      field_sensitivity=spec.get("field_sensitivity", True), sources=extra)

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for c, vlen, v in pool.map(one, jobs):
                if v is None:
                    summary["timeouts"].append({"symbol": symbol, **c, "vlen": vlen, "why": "budget"})
                    continue
                attempted += 1
                if v.verdict in ("proved", "alignment"):
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
            summary.update(verdict="COUNTEREXAMPLE", proved=proved, configs=total, attempted=attempted,
                           seconds=round(time.time() - t0))
            return summary
    # attempted < configs: the rest never ran (budget), which is not a solver timeout
    summary.update(proved=proved, configs=total, attempted=attempted, seconds=round(time.time() - t0),
                   verdict="ALL_PASSED" if proved == total and not summary["failed"] else "PARTIAL")
    summary["passed"] = summary["verdict"] == "ALL_PASSED"
    return summary
