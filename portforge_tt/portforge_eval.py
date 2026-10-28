"""Model-independent parts of a PortForge trusted evaluator.

Every port's ``evaluate.py`` builds its own suites, reference outputs and gates,
but checkpoint identity, receipts, precision declarations, progress reporting
and input-integrity checks are the same for any model. They live here so a new
port only writes the model-specific parts (see ports/template/evaluate.py).

This module runs on the reference host, inside the TT sandbox, and in the
controller, so it depends only on the standard library and ``portforge_models.py``.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

if __package__:  # installed as part of the portforge_tt package
    from .portforge_models import model_spec, verify_checkpoint
else:  # flat: a frozen run harness or a direct script execution
    from portforge_models import model_spec, verify_checkpoint

# TT storage formats an evaluator may accept. BFP8_B is block floating point (one shared
# exponent per 16 values), not IEEE FP8. BFP4_B is deferred until the pipeline is proven on BF16/BFP8.
TT_PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TT_DTYPES = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16", "bfp8_b": "bfloat8_b"}
POLICY_KEYS = {"mode", "weights", "activations", "accumulation", "exceptions"}
# Optional per-op-group overrides (tt_transformers ModelOptimizations style): group -> fields.
GROUP_KEYS = {"weights", "activations", "math_fidelity", "fp32_dest_acc", "packer_l1_acc", "math_approx"}
MATH_FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
SUITE_VERSION = 2
# Gated stages. "perf" repeats the bringup workload long enough for latency
# percentiles, a batch scaling curve and energy; it keeps the accuracy gates.
STAGES = ("smoke", "bringup", "perf", "full")
MAX_TIMED_REPEATS = 10000
# Perf: aim for 100 samples (a real p99) and >= 20 s per case (a sustained power window),
# but stop a slow case after 120 s once it has at least 10 samples.
PERF_REPEATS, PERF_MIN_SECONDS, PERF_MAX_SECONDS, PERF_MIN_REPEATS = 100, 20.0, 120.0, 10
PERF_BATCHES = (1, 2, 4)
REFERENCE_MAX_REPEATS = 3
# DEVICE_OPTIONS a backend may declare: ttnn.open_device keywords and their bounds.
# (The printed open_device signature omits num_command_queues; the binding accepts it.)
DEVICE_OPTION_BOUNDS = {"trace_region_size": (0, 128 * 1024 ** 2), "l1_small_size": (0, 128 * 1024),
                        "num_command_queues": (1, 2)}
# Matmul-class ops as the dispatcher sees them (torch.matmul and linear decompose into these).
HOST_MATMUL_OPS = ("aten.mm.", "aten.bmm.", "aten.addmm.", "aten.baddbmm.", "aten.addbmm.", "aten.mv.",
                   "aten.addmv.", "aten.dot.", "aten.vdot.", "aten.matmul.", "aten.linear.", "aten._addmm_activation.")


def stage_repeats(stage):
    """Timed repeats per numerical case: perf needs enough samples for a p99."""
    return {"full": 3, "perf": PERF_REPEATS}.get(stage, 1)


def perf_fields(stage):
    """Extra suite fields of the perf stage only, so smoke/bringup/full suite hashes are unchanged."""
    return (dict(min_timed_seconds=PERF_MIN_SECONDS, max_timed_seconds=PERF_MAX_SECONDS,
                 min_timed_repeats=PERF_MIN_REPEATS) if stage == "perf" else {})


def reference_repeats(spec, case):
    """The FP32 reference's timing is context only; never repeat it 100 times."""
    return min(spec["timed_repeats"] if case["numerical"] else 1, REFERENCE_MAX_REPEATS)


def repeat_count_ok(spec, case, count):
    """Exact repeat count, except perf suites, which run at least the count (minimum wall time)."""
    expected = spec["timed_repeats"] if case["numerical"] else 1
    if "min_timed_repeats" in spec and case["numerical"]:
        return count >= min(expected, spec["min_timed_repeats"])
    return count == expected


def suite_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def model_identity(name):
    model = model_spec(name)
    digest = hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(model_key=name, model=model["repo_id"], revision=model["revision"], model_manifest_sha256=digest)


def suite_model(spec, default_key, gates):
    """Check a prepared suite against the pinned registry and frozen gates."""
    name = spec.get("model_key", default_key)
    identity = model_identity(name)
    if spec.get("version") != SUITE_VERSION or any(spec.get(k) != v for k, v in identity.items()):
        raise ValueError("suite model identity does not match the pinned registry")
    if spec.get("gates") != gates:
        raise ValueError("suite changed the frozen gates")
    return model_spec(name), identity


def check_suite(spec):
    """Every consistency case must compare against a case the same suite contains.

    Single-row, reversed-batch and revisit cases compare their outputs against rows
    of another case. Naming a case the stage does not build makes the gate
    impossible to pass, so reject such suites when they are built.
    """
    names = {case["name"] for case in spec["cases"]}
    for case in spec["cases"]:
        if "compare_to" not in case:
            continue
        if case["compare_to"] not in names:
            raise ValueError(f"{spec['stage']} case {case['name']} compares to missing case {case['compare_to']}")
        rows = case.get("compare_rows")
        if (not isinstance(rows, list) or not rows or len(set(rows)) != len(rows)
                or any(type(row) is not int or row < 0 for row in rows)):
            raise ValueError(f"{spec['stage']} case {case['name']} needs distinct nonnegative compare_rows")
    return spec


def validate_config(config, model):
    for key, value in model["architecture"].items():
        if config.get(key) != value:
            raise ValueError("configuration does not match pinned model architecture: " + key)


def validate_receipt(receipt, identity, model, spec=None):
    required = ["config.json", *model["weight_files"], *([model["weight_index"]] if model["weight_index"] else [])]
    if not isinstance(receipt, dict) or not isinstance(receipt.get("verified_files"), dict):
        raise ValueError("malformed checkpoint verification receipt")
    if (receipt.get("model") != identity["model_key"] or receipt.get("repo_id") != identity["model"]
            or receipt.get("revision") != identity["revision"]
            or receipt.get("checkpoint_format") != model["checkpoint_format"]
            or receipt.get("model_manifest_sha256") != identity["model_manifest_sha256"]
            or any(receipt.get("verified_files", {}).get(k) != model["files"][k] for k in required)):
        raise ValueError("checkpoint verification receipt does not match pinned model")
    if spec is not None and receipt.get("suite_sha256") != suite_hash(spec):
        raise ValueError("checkpoint receipt belongs to a different prepared suite")


def validate_artifact(artifact, spec, identity, timing_protocol):
    if (not isinstance(artifact, dict) or any(artifact.get(k) != v for k, v in identity.items())
            or artifact.get("suite_sha256") != suite_hash(spec)
            or artifact.get("timing_protocol") != timing_protocol):
        raise ValueError("artifact model/suite/timing identity mismatch")


def verify_weights(path, name):
    """Single-file checkpoint or an official checkpoint directory; hash bytes, never execute."""
    path, model = Path(path), model_spec(name)
    if path.is_dir():
        return verify_checkpoint(path, name)["verified_files"]
    if model["weight_index"]:
        raise ValueError("sharded model requires a checkpoint directory")
    filename = model["weight_files"][0]
    expected, checksum = model["files"][filename], hashlib.sha256()
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise ValueError("candidate weight size mismatch")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != expected["sha256"]:
        raise ValueError("candidate weight SHA-256 mismatch")
    return {filename: dict(expected)}


def device_options(backend_module):
    """The backend's DEVICE_OPTIONS, checked against DEVICE_OPTION_BOUNDS (ttnn.open_device keywords)."""
    options = getattr(backend_module, "DEVICE_OPTIONS", {})
    if (not isinstance(options, dict) or set(options) - set(DEVICE_OPTION_BOUNDS)
            or any(type(v) is not int or not DEVICE_OPTION_BOUNDS[k][0] <= v <= DEVICE_OPTION_BOUNDS[k][1]
                   for k, v in options.items())):
        raise ValueError("invalid DEVICE_OPTIONS; allowed keys: " + ", ".join(DEVICE_OPTION_BOUNDS))
    return dict(options)


def smallest_weight_matrix(path, name):
    """Elements in the checkpoint's smallest weight matrix (ndim >= 2), from tensor headers only."""
    import math
    path, model = Path(path), model_spec(name)
    sizes = []
    for file in ([path / f for f in model["weight_files"]] if path.is_dir() else [path]):
        if file.suffix == ".safetensors":
            with file.open("rb") as stream:
                header = json.loads(stream.read(int.from_bytes(stream.read(8), "little")))
            shapes = [entry["shape"] for key, entry in header.items() if key != "__metadata__"]
        else:
            import torch
            state = torch.load(file, map_location="cpu", mmap=True, weights_only=True)
            shapes = [list(value.shape) for value in state.values() if hasattr(value, "shape")]
        sizes += [math.prod(shape) for shape in shapes if len(shape) >= 2]
    if not sizes:
        raise ValueError("checkpoint has no weight matrices")
    return min(sizes)


class HostComputeGuard:
    """CPU tripwire around candidate calls; build it after create_backend.

    Learned ops named in ``forbidden`` (norms, softmax, activations, convolutions)
    always fail on the host. Matmuls (HOST_MATMUL_OPS) fail only when an operand
    carries model weights: a host tensor or array the backend holds (walked from
    the backend object and its module) at least as large as the checkpoint's
    smallest weight matrix, a view of one, or a tensor computed from one during a
    guarded call. Weight-free host math, such as RoPE angles from positions, passes.
    NumPy math and host TTNN tensors are invisible to it: it catches PyTorch
    fallbacks, and the TT call count and timing cover the rest.
    """

    MARK = "_portforge_weights"

    def __init__(self, forbidden, floor, *roots):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves
        self.forbidden, self.floor, self.roots = tuple(forbidden), int(floor), roots
        guard = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs, name = kwargs or {}, str(func)
                if name.startswith(guard.forbidden):
                    raise RuntimeError("CPU learned compute forbidden: " + name)
                weighted = any(guard.carries(x) for x in tree_leaves((args, kwargs)) if isinstance(x, torch.Tensor))
                if weighted and name.startswith(HOST_MATMUL_OPS):
                    raise RuntimeError("CPU learned compute forbidden (host matmul on model weights): " + name)
                out = func(*args, **kwargs)
                if weighted:  # taint results and in-place/out= targets
                    written = [args[i] if i < len(args) else kwargs.get(a.name)
                               for i, a in enumerate(func._schema.arguments)
                               if a.alias_info is not None and a.alias_info.is_write]
                    for x in tree_leaves((out, written)):
                        if isinstance(x, torch.Tensor):
                            setattr(x, guard.MARK, True)
                return out

        self.mode = Mode
        self.refresh()

    def refresh(self):
        """Re-walk what the backend holds; call outside timed spans (e.g. before each timed window)."""
        import types
        import numpy as np
        import torch
        skip = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType,
                str, bytes, int, float, complex, bool, type(None))
        regions, seen, stack = [], {}, [vars(root) if isinstance(root, types.ModuleType) else root
                                           for root in self.roots]
        while stack and len(seen) < 1_000_000:
            item = stack.pop()
            if id(item) in seen:
                continue
            seen[id(item)] = item  # held, so ids stay unique during the walk
            if isinstance(item, torch.Tensor):
                if item.device.type == "cpu" and item.numel() >= self.floor:
                    storage = item.untyped_storage()
                    regions.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes(), item))
            elif isinstance(item, np.ndarray):
                while isinstance(item.base, np.ndarray):
                    item = item.base
                if item.size >= self.floor:
                    regions.append((item.ctypes.data, item.ctypes.data + item.nbytes, item))
            elif isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, (list, tuple, set, frozenset)):
                stack.extend(item)
            elif not isinstance(item, skip):
                stack.extend(getattr(item, "__dict__", {}).values())
                stack.extend(getattr(item, s) for s in getattr(type(item), "__slots__", ()) if hasattr(item, s))
        merged = []  # sorted, non-overlapping [start, end); the tensors are kept so addresses stay valid
        for start, end, owner in sorted(regions, key=lambda r: r[0]):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]), merged[-1][2] + [owner])
            else:
                merged.append((start, end, [owner]))
        self.regions, self.starts = merged, [start for start, _, _ in merged]

    def carries(self, tensor):
        import bisect
        if getattr(tensor, self.MARK, False):
            return True
        try:
            if tensor.device.type != "cpu" or not tensor.numel():
                return False
            address = tensor.data_ptr()
        except RuntimeError:
            return False
        index = bisect.bisect_right(self.starts, address) - 1
        return index >= 0 and address < self.regions[index][1]

    def __call__(self):
        return self.mode()


def reference_execution():
    """The reference must run on an explicitly configured compute host, never a login node."""
    mode = os.getenv("PORTFORGE_REFERENCE_EXECUTION", "slurm")
    if mode not in ("slurm", "direct") or socket.gethostname().lower().startswith(("login", "lo-")):
        raise RuntimeError("reference requires an explicitly configured compute host")
    if mode == "slurm" and not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Slurm reference requires a compute allocation")
    return mode


def require_tt_precision(mode, runtime, precisions=TT_PRECISIONS):
    if mode not in precisions:
        raise ValueError("unsupported precision; TT BFP8 is not IEEE FP8")
    dtype = TT_DTYPES[mode]
    if runtime is not None and getattr(runtime, dtype, None) is None:
        raise ValueError(f"installed TTNN does not expose native {dtype}; no silent precision substitution")


def validate_precision(mode, policy, runtime=None, precisions=TT_PRECISIONS):
    """Validate the backend's declaration; this is not a proof of every tensor's dtype."""
    require_tt_precision(mode, runtime, precisions)
    if not isinstance(policy, dict) or not POLICY_KEYS <= set(policy) <= POLICY_KEYS | {"groups"}:
        raise ValueError("backend must declare precision_policy fields mode/weights/activations/accumulation/exceptions")
    if policy["mode"] != mode or policy["weights"] != mode or policy["activations"] not in precisions:
        raise ValueError("precision declaration disagrees with requested mode")
    require_tt_precision(policy["activations"], runtime, precisions)
    if not isinstance(policy["accumulation"], str) or not policy["accumulation"].strip():
        raise ValueError("declare accumulation policy explicitly")
    if not isinstance(policy["exceptions"], list) or any(not isinstance(x, str) or not x.strip() for x in policy["exceptions"]):
        raise ValueError("precision exceptions must be explicit nonempty descriptions")
    groups = validate_groups(policy.get("groups", {}), runtime, precisions)
    mixed = policy["activations"] != mode or any(g.get(k, mode) != mode for g in groups.values()
                                                 for k in ("weights", "activations"))
    if mixed and not policy["exceptions"]:
        raise ValueError("mixed activation precision requires an explanation")
    effective = dict(policy, exceptions=list(policy["exceptions"]))
    if groups:
        effective["groups"] = groups
    return dict(requested=mode, effective=effective, declaration_only=True,
                format_note="block floating point; not IEEE FP8/FP4" if mode.startswith("bfp") else mode)


def validate_groups(groups, runtime=None, precisions=TT_PRECISIONS):
    """Per-op-group overrides, e.g. {"attention_matmul": {"weights": "bfp8_b", "math_fidelity": "HiFi2"}}."""
    if not isinstance(groups, dict) or len(groups) > 64:
        raise ValueError("precision groups must map at most 64 group names to settings")
    checked = {}
    for name, settings in groups.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError("precision group names must be short nonempty strings")
        if not isinstance(settings, dict) or not settings or set(settings) - GROUP_KEYS:
            raise ValueError("precision group " + name + " may set only " + "/".join(sorted(GROUP_KEYS)))
        for key in ("weights", "activations"):
            if key in settings:
                require_tt_precision(settings[key], runtime, precisions)
        if "math_fidelity" in settings and settings["math_fidelity"] not in MATH_FIDELITIES:
            raise ValueError("math_fidelity must be one of " + ", ".join(MATH_FIDELITIES))
        if any(type(settings[k]) is not bool for k in ("fp32_dest_acc", "packer_l1_acc", "math_approx") if k in settings):
            raise ValueError("precision group flags must be booleans")
        checked[name] = dict(settings)
    return checked


def repeat_schedule(spec, case):
    """Untimed first call (repeat 0), then the suite's timed repeats.

    A perf suite also sets min_timed_seconds: numerical cases keep repeating until
    both the count and the wall time are reached (MLPerf-style sustained windows,
    so sampled board power covers the workload), bounded by MAX_TIMED_REPEATS.
    A slow case stops at max_timed_seconds once it has min_timed_repeats samples.
    """
    count = spec["timed_repeats"] if case["numerical"] else 1
    minimum = float(spec.get("min_timed_seconds", 0)) if case["numerical"] else 0.0
    maximum = float(spec.get("max_timed_seconds", 0)) if case["numerical"] else 0.0
    floor = min(count, int(spec.get("min_timed_repeats", count)))
    repeat, start = 0, None
    while True:
        elapsed = time.perf_counter() - start if start is not None else 0.0
        if repeat > count and (start is None or elapsed >= minimum or repeat > MAX_TIMED_REPEATS):
            return
        if maximum and repeat > floor and elapsed >= maximum:
            return
        yield repeat
        if repeat == 0:
            start = time.perf_counter()
        repeat += 1


def signpost(name):
    """Tracy signpost when the runner profiles this job (profile jobs set PORTFORGE_PROFILE)."""
    if os.getenv("PORTFORGE_PROFILE") == "1":
        try:
            from tracy import signpost as mark
        except ImportError:
            return
        mark(name)


_PROFILED = []


def profile_device(device):
    """Register the opened device so profile jobs drain its profiler at every case boundary.

    TT-Metal's device profiler holds about 1000 programs and drops markers past that
    (Tracy's report then fails); it drains only on an explicit ReadDeviceProfiler.
    """
    if os.getenv("PORTFORGE_PROFILE") == "1":
        _PROFILED.append(device)
    return device


def flush_profiler():
    if _PROFILED:
        import ttnn
        for device in _PROFILED:
            ttnn.ReadDeviceProfiler(device)


class TimedWindow:
    """Wall-clock window of one case's timed repeats.

    The trusted runner integrates sampled board power over [start, end]; the
    profile job keeps device ops between the two signposts. Call open() before
    the first timed repeat (after the untimed first call) and close() after the
    last; record() returns the fields measurements.json carries per case.
    """

    def __init__(self, items):
        self.items, self.start, self.end = int(items), None, None
        flush_profiler()  # profile jobs: the previous case's ops

    def open(self):
        if self.start is None:
            flush_profiler()  # the untimed first call
            signpost("portforge_timed_start")
            self.start = time.time()

    def close(self):
        if self.start is not None and self.end is None:
            self.end = time.time()
            signpost("portforge_timed_stop")
            flush_profiler()  # the timed repeats

    def record(self):
        return dict(window=[self.start, self.end] if self.start is not None else None, items=self.items)


def latency_summary(samples):
    """Nearest-rank percentiles of positive timing samples (seconds)."""
    import math
    values = sorted(float(x) for x in samples)
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("invalid timing samples")
    def rank(q):
        return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]
    mean = sum(values) / len(values)
    deviation = (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** .5 if len(values) > 1 else 0.0
    return dict(samples=len(values), p50_seconds=rank(.5), p90_seconds=rank(.9), p95_seconds=rank(.95),
                p99_seconds=rank(.99), max_seconds=values[-1], mean_seconds=mean, stdev_seconds=deviation,
                cv=deviation / mean, percentile_method="nearest_rank",
                tail_note=None if len(values) >= 100 else "fewer than 100 samples: p99 is the maximum-order statistic")


def verdict_latency(checks):
    """Run-level latency fields for a verdict from checks that carry raw "seconds" samples.

    Adds the per-case percentile summary to each check in place and returns the
    aggregate the objective reads (objective.metrics): median of case p50s, worst
    case p99 and worst first-call time.
    """
    import statistics
    p50, p99, first = [], [], []
    for check in checks:
        samples = check.get("seconds")
        if not samples:
            continue
        summary = latency_summary(samples)
        check.update({k: v for k, v in summary.items() if k not in ("samples",)})
        p50.append(summary["p50_seconds"])
        p99.append(summary["p99_seconds"])
        if type(check.get("first_seconds")) in (int, float):
            first.append(check["first_seconds"])
    return dict(median_case_p50_seconds=statistics.median(p50) if p50 else None,
                max_case_p99_seconds=max(p99, default=None), max_first_seconds=max(first, default=None),
                measured_cases=len(p50), min_samples_per_case=min((c.get("timing_samples") or len(c.get("seconds") or [])
                                                                  for c in checks if c.get("seconds")), default=0))


def scaling_curve(checks, cases):
    """Throughput per batch size from the perf stage's batch-sweep cases (case["batch"] rows)."""
    batches = {case["name"]: case["batch"] for case in cases if type(case.get("batch")) is int}
    curve = []
    for check in checks:
        batch = batches.get(check.get("name"))
        if batch and check.get("p50_seconds"):
            curve.append(dict(case=check["name"], batch=batch, items=batch, p50_seconds=check["p50_seconds"],
                              p99_seconds=check.get("p99_seconds"), items_per_second=batch / check["p50_seconds"]))
    return sorted(curve, key=lambda row: row["batch"])


def validate_input_preservation(expected, observed):
    """Reject caller-array mutation, including changes to shape or dtype."""
    import numpy as np
    if expected.keys() != observed.keys():
        raise ValueError("backend input set changed")
    for name, original in expected.items():
        value = observed[name]
        if not isinstance(value, np.ndarray) or value.dtype != original.dtype:
            raise ValueError("backend mutated caller input: " + name)
        # equal_nan is only valid for float dtypes; string/integer inputs compare exactly.
        equal = (np.array_equal(value, original, equal_nan=True) if original.dtype.kind == "f"
                 else np.array_equal(value, original))
        if not equal:
            raise ValueError("backend mutated caller input: " + name)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    os.replace(temporary, path)


class Progress:
    """Heartbeat file the runner uses to tell a slow phase from a hang."""

    def __init__(self, directory, stage):
        self.path, self.sequence, self.stage = Path(directory) / "progress.json", 0, stage

    def mark(self, phase, case=None, repeat=None, initializing=False):
        self.sequence += 1
        event = dict(version=1, sequence=self.sequence, updated_unix_seconds=time.time(),
                     phase=phase, case=case, repeat=repeat,
                     timeout_seconds=600 if initializing else (300 if self.stage == "full" else 180))
        write_json(self.path, event)
        print(json.dumps({"progress": event}), flush=True)
        return event
