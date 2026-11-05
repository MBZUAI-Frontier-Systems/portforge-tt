"""Parameter regimes: the shape x parameter matrix every kernel is judged on.

A kernel's speed and correctness depend on its parameters (qs8-vcvt's pf2 was
2x faster at multiplier 256, 1.18x at the bench's 183 and 1.0x elsewhere), so
a family declares its regimes in kernels/<family>/regimes.json:

    {"family": "qs8-vcvt",
     "reference": "xnn_qs8_vcvt_ukernel__scalar_u4",       # bit-exact oracle for the check
     "bench_shapes": [256, 16320, 130560, 4194304],        # elements per call
     "check_batches": [1, 2, 3, ..., 4099],
     "inputs": {"dist": "full"},                           # default input distribution
     "regimes": [
       {"id": "bench", "a": {"scale": 0.5, "zero_point": 1}, "out": {"scale": 0.7, "zero_point": -1},
        "why": "XNNPACK's bench quantization (multiplier 183)"},
       {"id": "d1-m32768", ..., "baseline_correct": false},
       ...],
     "check_extra": [...],                                 # correctness-only regimes
     "heldout": {"count": 3,
                 "sample": {"a.scale": ["loguniform", 0.004, 128], "a.zero_point": ["int", -128, 127], ...},
                 "accept": "1/256 <= a_scale / out_scale <= 128"}}

A regime is the *arguments* of the family's init function: quantization of
input a (and b for binary kernels) and of the output, plus up to two op floats
(clamp min/max, leaky-relu slope). The harness calls each kernel's own init
with them, so params are what XNNPACK itself would produce. Held-out regimes
are drawn at certification with a seed agents never see.

Transposes ("kind": "transpose") take no params: a regime is the memory layout
of a call instead ({"id", "tile", "in_pad", "out_pad", "offset"}: one call or
XNNPACK's TxT operator sweep, row padding in elements, base-pointer offset),
shapes are "WxH" blocks, and "check_shapes" replaces "check_batches" (a list,
or {"grid": [lo, hi], "extra": [...]} for every W, H in [lo, hi] plus extras).
Held-out samplers then name top-level fields ("tile", "in_pad", ...).
"""
import json
import math
import random
import re
from pathlib import Path

QUANT_FIELDS = ("a", "b", "out")
LAYOUT_FIELDS = ("tile", "in_pad", "out_pad", "offset")
SHAPE = re.compile(r"^[1-9][0-9]*x[1-9][0-9]*$")
DISTS = ("full", "uniform", "specials")
SAMPLERS = ("loguniform", "uniform", "int", "choice")


def load(path):
    spec = json.loads(Path(path).read_text())
    validate(spec)
    return spec


def is_transpose(spec):
    return spec.get("kind") == "transpose"


def check_sizes(spec):
    """Batches (element counts) or, for transposes, "WxH" blocks the check runs."""
    if not is_transpose(spec):
        return list(spec["check_batches"])
    shapes = spec["check_shapes"]
    if isinstance(shapes, dict):
        lo, hi = shapes["grid"]
        return [f"{w}x{h}" for w in range(lo, hi + 1) for h in range(lo, hi + 1)] + list(shapes.get("extra", []))
    return list(shapes)


def validate(spec):
    """Raise ValueError naming the first problem in a regimes spec."""
    if is_transpose(spec):
        return _validate_transpose(spec)
    for key in ("family", "reference", "bench_shapes", "check_batches", "regimes"):
        if key not in spec:
            raise ValueError(f"regimes spec lacks {key!r}")
    ids = [r.get("id") for r in spec["regimes"] + spec.get("check_extra", [])]
    if len(ids) != len(set(ids)) or not all(ids):
        raise ValueError(f"regime ids must be unique and non-empty: {ids}")
    for regime in spec["regimes"] + spec.get("check_extra", []):
        for field in QUANT_FIELDS:
            q = regime.get(field, {})
            if set(q) - {"scale", "zero_point"}:
                raise ValueError(f"{regime['id']}.{field}: only scale and zero_point")
        if len(regime.get("op", [])) > 2:
            raise ValueError(f"{regime['id']}: at most two op floats")
        if regime.get("dist", spec.get("inputs", {}).get("dist", "full")) not in DISTS:
            raise ValueError(f"{regime['id']}: dist must be one of {DISTS}")
    if not all(isinstance(n, int) and n > 0 for n in spec["bench_shapes"] + spec["check_batches"]):
        raise ValueError("shapes and batches are positive element counts")
    for name, sampler in spec.get("heldout", {}).get("sample", {}).items():
        if sampler[0] not in SAMPLERS:
            raise ValueError(f"heldout sampler for {name}: one of {SAMPLERS}")


def _validate_transpose(spec):
    for key in ("family", "reference", "bench_shapes", "check_shapes", "regimes"):
        if key not in spec:
            raise ValueError(f"regimes spec lacks {key!r}")
    ids = [r.get("id") for r in spec["regimes"] + spec.get("check_extra", [])]
    if len(ids) != len(set(ids)) or not all(ids):
        raise ValueError(f"regime ids must be unique and non-empty: {ids}")
    for regime in spec["regimes"] + spec.get("check_extra", []):
        extra = set(regime) - {"id", "why", "baseline_correct", *LAYOUT_FIELDS}
        if extra:
            raise ValueError(f"{regime['id']}: transpose regimes take only {LAYOUT_FIELDS}, not {sorted(extra)}")
        if not all(isinstance(regime.get(f, 0), int) and regime.get(f, 0) >= 0 for f in LAYOUT_FIELDS):
            raise ValueError(f"{regime['id']}: layout fields are non-negative integers")
    shapes = spec["bench_shapes"] + check_sizes(spec)
    bad = [s for s in shapes if not isinstance(s, str) or not SHAPE.match(s)]
    if bad:
        raise ValueError(f"transpose shapes are 'WxH' strings: {bad[:5]}")
    for name, sampler in spec.get("heldout", {}).get("sample", {}).items():
        if name not in LAYOUT_FIELDS or sampler[0] not in SAMPLERS:
            raise ValueError(f"heldout sampler {name}: a layout field with one of {SAMPLERS}")


def _line(spec, r):
    if is_transpose(spec):
        return "tregime {} {} {} {} {}".format(r["id"], *(int(r.get(f, 0)) for f in LAYOUT_FIELDS))
    inputs = {**spec.get("inputs", {}), **{k: r[k] for k in ("dist", "lo", "hi") if k in r}}
    q = {f: {"scale": 1.0, "zero_point": 0, **r.get(f, {})} for f in QUANT_FIELDS}
    op = (list(r.get("op", [])) + [0.0, 0.0])[:2]
    return ("regime {id} {a[scale]!r} {a[zero_point]} {b[scale]!r} {b[zero_point]} {out[scale]!r} "
            "{out[zero_point]} {op0!r} {op1!r} {dist} {lo!r} {hi!r}").format(
        id=r["id"], a=q["a"], b=q["b"], out=q["out"], op0=float(op[0]), op1=float(op[1]),
        dist=inputs.get("dist", "full"), lo=float(inputs.get("lo", -1.0)), hi=float(inputs.get("hi", 1.0)))


def flat(spec, regimes):
    """The harness's line format (see tools/pfharness/main.cc)."""
    return "".join(_line(spec, r) + "\n" for r in regimes)


def bench_regimes(spec):
    return list(spec["regimes"])


def check_regimes(spec):
    return list(spec["regimes"]) + list(spec.get("check_extra", []))


def cells(spec, regimes=None, shapes=None):
    """(regime id, shape) pairs of the bench matrix."""
    regimes = regimes if regimes is not None else bench_regimes(spec)
    return [(r["id"], n) for r in regimes for n in (shapes or spec["bench_shapes"])]


def _sample(sampler, rng):
    kind, *args = sampler
    if kind == "loguniform":
        lo, hi = args
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    if kind == "uniform":
        return rng.uniform(*args)
    if kind == "int":
        return rng.randint(*args)
    return rng.choice(args[0])


def _accepts(expr, regime):
    names = {f"{f}_{k}": regime.get(f, {}).get(k, 1.0 if k == "scale" else 0)
             for f in QUANT_FIELDS for k in ("scale", "zero_point")}
    names.update({f"op{i}": v for i, v in enumerate(regime.get("op", []))})
    names.update({f: regime.get(f, 0) for f in LAYOUT_FIELDS})
    return bool(eval(expr, {"__builtins__": {}, "abs": abs, "min": min, "max": max}, names))


def draw_heldout(spec, seed, tries=10000):
    """Held-out regimes drawn from the family's sampler with a secret seed."""
    held = spec.get("heldout")
    if not held:
        return []
    rng = random.Random(seed)
    out = []
    for _ in range(tries):
        if len(out) >= held.get("count", 3):
            break
        regime = {"id": f"heldout-{len(out)}", "heldout": True}
        for name, sampler in held["sample"].items():
            value = _sample(sampler, rng)
            if "." not in name:  # transpose layout field
                regime[name] = int(value)
                continue
            field, key = name.split(".")
            if field == "op":
                regime.setdefault("op", [0.0, 0.0])[int(key)] = float(value)
            else:
                regime.setdefault(field, {})[key] = int(value) if key == "zero_point" else float(value)
        if not held.get("accept") or _accepts(held["accept"], regime):
            out.append(regime)
    if len(out) < held.get("count", 3):
        raise ValueError("held-out sampler rejected too many draws; loosen 'accept'")
    return out
