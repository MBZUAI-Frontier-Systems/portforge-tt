"""SALTyRN's optimized RVV kernels: registry and injection specs (research provenance).

The registry (saltyrn_kernels.json, next to this file) records, per kernel, the
upstream Neon ukernel SALTyRN translated (found by content match against
SALTyRN's pinned XNNPACK), its registry .inc, and the optimized RVV file, plus
what the 2026-10-18 survey found (see ../README.md). The generic helpers it
once carried (bench comparison, name mangling) live in portforge_tt.cpu.xnnpack.

Run from the repo root:  python ports/xnnpack-rvv/research/saltyrn.py ...
"""
import difflib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from portforge_tt.cpu import xnnpack  # noqa: E402

REGISTRY = Path(__file__).resolve().parent / "saltyrn_kernels.json"
SUFFIX = "rvv_saltyrn"
# optimized-rvv-kernels names two files differently from kernels/source
OPTIMIZED_ALIASES = {"f32-dwconv2d-chw": "dwconv2d-chw", "x32-transposec": "x32transpose"}
# Upstream writes these with the xnn_simd_f32_t wrappers (src/xnnpack/simd/f32-neon.h);
# SALTyRN expanded the wrappers into raw intrinsics, so text matching scores low or
# picks a lookalike. Checked by hand against the expanded wrapper definitions.
MATCH_OVERRIDES = {
    "f32-vsqrt": "xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4",
    "f32-maxpool": "xnn_f32_maxpool_minmax_ukernel_9p__neon_u4",
}


def _normalized_body(text):
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    text = re.sub(r"#include[^\n]*", "", text)
    m = re.search(r"\bvoid\s+\w+\s*\(", text)
    text = text[m.start():] if m else text
    text = re.sub(r"void\s+\w+\s*\(", "void F(", text, count=1)
    return re.sub(r"\s+", " ", text).strip()


FUNCTION_START = re.compile(r"^(?:static\s+)?void\s+(xnn_\w+)\s*\(", re.M)


def _functions(text):
    """{ukernel: normalized body} for every ukernel defined in a source file."""
    starts = [(m.start(), m.group(1)) for m in FUNCTION_START.finditer(text)]
    out = {}
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        out[name] = _normalized_body(text[pos:end])
    return out


def build_registry(saltyrn_dir, xnnpack_dir, optimized_dir):
    """Match every SALTyRN Neon source to the upstream ukernel it was prepped from."""
    xnn = Path(xnnpack_dir)
    candidates = {}                       # ukernel -> (file, normalized body)
    for path in (xnn / "src").rglob("*.c"):
        if "neon" in path.name or "aarch64" in path.name:
            for name, body in _functions(path.read_text(errors="ignore")).items():
                candidates[name] = (path, body)
    incs = {p: p.read_text(errors="ignore") for p in (xnn / "src").rglob("*.inc")}
    registry = {}
    for source in sorted((Path(saltyrn_dir) / "kernels" / "source").glob("*.c")):
        name = source.stem
        body = _normalized_body(source.read_text())
        near = [(uk, b) for uk, (_, b) in candidates.items() if 0.6 < len(b) / max(len(body), 1) < 1.6]
        quick = sorted(((difflib.SequenceMatcher(None, body, b, autojunk=False).quick_ratio(), uk)
                        for uk, b in near), reverse=True)[:8]
        ratio, ukernel = max((difflib.SequenceMatcher(None, body, candidates[uk][1], autojunk=False).ratio(), uk)
                             for _, uk in quick)
        if name in MATCH_OVERRIDES:
            ukernel = MATCH_OVERRIDES[name]
            ratio = difflib.SequenceMatcher(None, body, candidates[ukernel][1], autojunk=False).ratio()
        best = candidates[ukernel][0]
        inc = next((p for p, t in incs.items()
                    if any(len(a) > 1 and a[1] == ukernel for _, a, _ in xnnpack.registry_lines(t))), None)
        opt = Path(optimized_dir) / f"{OPTIMIZED_ALIASES.get(name, name)}.c"
        upstream_rvv = []
        if inc is not None:
            upstream_rvv = [a[1] for _, a, _ in xnnpack.registry_lines(incs[inc])
                            if len(a) > 1 and "rvv" in a[1]]
        registry[name] = {
            "neon_source": f"kernels/source/{source.name}",
            "neon_ukernel": ukernel,
            "neon_file": str(best.relative_to(xnn)),
            "match_ratio": round(ratio, 3),
            "match": "manual" if name in MATCH_OVERRIDES else "content",
            "inc": str(inc.relative_to(xnn)) if inc else None,
            "optimized": opt.name if opt.exists() else None,
            "upstream_rvv": upstream_rvv,
        }
    return registry


# Kernels whose RVV version does not keep the Neon kernel's data layout, so they are
# registered from an upstream RVV template with vector-length tiles instead.
VLEN_TILES = "4 * (xnn_init_hardware_config()->vlenb / sizeof(float))"
REGISTRATION_OVERRIDES = {
    # loads weights in groups of VLMAX(e32m4) channels: XNNPACK must pack them 3p(4v)c, not 3p4c
    "f32-dwconv-minmax": {"template": "xnn_f32_dwconv_minmax_ukernel_3p8vc__rvv",
                          "name": "xnn_f32_dwconv_minmax_ukernel_3p4vc__rvv_saltyrn",
                          "overrides": {2: VLEN_TILES, 4: VLEN_TILES}},
}


def load_registry():
    return json.loads(REGISTRY.read_text())["kernels"]


# Re-exported from the package so old call sites keep working.
ukernel_name = xnnpack.ukernel_name
family_token = xnnpack.family_token
geomean = xnnpack.geomean
split_bench_name = xnnpack.split_bench_name


def registered_name(name, entry):
    """The ukernel name a SALTyRN kernel is registered under in XNNPACK."""
    return REGISTRATION_OVERRIDES.get(name, {}).get("name", ukernel_name(entry["neon_ukernel"]))


def injectable(entry):
    """Kernels XNNPACK's registry-driven tests can take; GEMM-style families are
    generated from YAML specs instead and need their own wiring."""
    return bool(entry["inc"] and entry["optimized"]
                and (entry["match_ratio"] >= 0.9 or entry.get("match") == "manual"))


def injections(cfg, registry, names=None):
    out = []
    for name, entry in registry.items():
        if names and name not in names:
            continue
        if not injectable(entry):
            continue
        reg = REGISTRATION_OVERRIDES.get(name, {})
        out.append(xnnpack.Injection(source=cfg.src / "optimized-rvv-kernels" / entry["optimized"],
                                     inc=entry["inc"], template=reg.get("template", entry["neon_ukernel"]),
                                     name=registered_name(name, entry),
                                     overrides=reg.get("overrides")))
    return out


def compare_bench(rows, ukernel, token):
    """compare_bench bound to this survey's candidate suffix."""
    return xnnpack.compare_bench(rows, ukernel, token, exclude=SUFFIX)
