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
# Latency percentiles describe single-stream use; larger sweep batches feed only the throughput curve.
LATENCY_MAX_BATCH = 4
# Optional stage a pack may add: the perf workload data parallel over a 1xN mesh of one host's
# cards (the suite's "mesh_cards"). Ranked on its own frontier, never against single-card perf.
MESH_STAGE = "mesh"
MESH_CARDS_BOUNDS = (2, 8)
ALL_STAGES = STAGES + (MESH_STAGE,)            # what a TT host accepts
GATED_STAGES = ("smoke", "bringup", "perf", MESH_STAGE)  # what an agent may evaluate; full is controller-only
# Files every evaluating host receives (reference host, TT host, sandbox), and the TT host's extra runner files.
EVALUATOR_FILES = ("evaluate.py", "portforge_eval.py", "portforge_models.py", "models.json")
TT_HOST_FILES = ("runner.py", "telemetry.py", "opprofile.py") + EVALUATOR_FILES
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
    return {"full": 3, "perf": PERF_REPEATS, MESH_STAGE: PERF_REPEATS}.get(stage, 1)


def perf_fields(stage):
    """Extra suite fields of the perf stage only, so smoke/bringup/full suite hashes are unchanged."""
    return (dict(min_timed_seconds=PERF_MIN_SECONDS, max_timed_seconds=PERF_MAX_SECONDS,
                 min_timed_repeats=PERF_MIN_REPEATS) if stage in ("perf", MESH_STAGE) else {})


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
    cards = spec.get("mesh_cards")
    if (spec["stage"] == MESH_STAGE) != (cards is not None) or (
            cards is not None and (type(cards) is not int or not MESH_CARDS_BOUNDS[0] <= cards <= MESH_CARDS_BOUNDS[1])):
        raise ValueError(f"the mesh stage, and only it, needs integer mesh_cards in {MESH_CARDS_BOUNDS}")
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


def open_tt_device(ttnn, spec, options):
    """The suite's device: one card (open_device already returns a 1x1 MeshDevice), or for the
    mesh stage a 1xN MeshDevice over every card the runner leased, for data parallelism."""
    if spec.get("mesh_cards"):
        return ttnn.open_mesh_device(ttnn.MeshShape(1, spec["mesh_cards"]), **options)
    return ttnn.open_device(device_id=0, **options)


def close_tt_device(ttnn, spec, device):
    (ttnn.close_mesh_device if spec.get("mesh_cards") else ttnn.close_device)(device)


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

    TT-Metal's device profiler holds a fixed number of programs (the runner raises it to
    PROFILE_PROGRAMS for profile jobs) and drops markers past that (Tracy's report then
    fails); it drains only on an explicit ReadDeviceProfiler.
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


def verdict_latency(checks, cases=()):
    """Run-level latency fields for a verdict from checks that carry raw "seconds" samples.

    Adds the per-case percentile summary to each check in place and returns the
    aggregate the objective reads (objective.metrics): median of case p50s, worst
    case p99 and worst first-call time. Sweep cases above LATENCY_MAX_BATCH (named
    in ``cases``) get their summary but stay out of the aggregate: they are throughput points.
    """
    import statistics
    throughput_only = {case["name"] for case in cases
                       if type(case.get("batch")) is int and case["batch"] > LATENCY_MAX_BATCH}
    p50, p99, first = [], [], []
    for check in checks:
        samples = check.get("seconds")
        if not samples:
            continue
        summary = latency_summary(samples)
        check.update({k: v for k, v in summary.items() if k not in ("samples",)})
        if check.get("name") in throughput_only:
            continue
        p50.append(summary["p50_seconds"])
        p99.append(summary["p99_seconds"])
        if type(check.get("first_seconds")) in (int, float):
            first.append(check["first_seconds"])
    return dict(median_case_p50_seconds=statistics.median(p50) if p50 else None,
                max_case_p99_seconds=max(p99, default=None), max_first_seconds=max(first, default=None),
                measured_cases=len(p50), min_samples_per_case=min((c.get("timing_samples") or len(c.get("seconds") or [])
                                                                  for c in checks if c.get("seconds")), default=0),
                latency_max_batch=LATENCY_MAX_BATCH if throughput_only else None)


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


# ---------------------------------------------------------------------------
# The evaluator kit: the scaffold every pack's evaluate.py shares. A pack keeps its
# constants, make_suite and a few model-specific hooks; ports/template/evaluate.py
# is the worked example. The four entry points a pack exposes stay thin:
#
#   def reference(args):  ...uses ReferenceRun...
#   def candidate(args):  return portforge_eval.run_candidate(args, globals())
#   def assess(args):     return portforge_eval.run_assess(args, globals())
#   if __name__ == "__main__": portforge_eval.main(globals())
#
# Packs pass globals(), not sys.modules[__name__]: under `python -m tracy` (port_profile) the script
# runs inside tracy's own __main__ module.

class Pack:
    """A pack's evaluate.py namespace (its globals()), read live as attributes."""

    def __init__(self, namespace):
        self.__dict__ = namespace


def as_pack(pack):
    return Pack(pack) if isinstance(pack, dict) else pack


# ttnn entry points counted as TT compute; a timed call that reaches none of them fails.
TT_COMPUTE_OPS = ("matmul", "linear", "embedding", "layer_norm", "conv1d", "conv2d", "execute_trace")


def case_arrays(inputs, case):
    """This case's arrays from inputs.npz, keyed without the "<case>__" prefix."""
    prefix = case["name"] + "__"
    return {key[len(prefix):]: inputs[key] for key in inputs.files if key.startswith(prefix)}


PROFILE_REPEATS = 3


def profile_subset(spec):
    """A profile job of one perf case (PORTFORGE_PROFILE_CASE): only that case, a few timed repeats.
    Profiles are never assessed, so the suite may shrink; everything else keeps the full suite."""
    case = os.getenv("PORTFORGE_PROFILE_CASE")
    if os.getenv("PORTFORGE_PROFILE") != "1" or not case:
        return spec
    cases = [c for c in spec["cases"] if c["name"] == case]
    if not cases:
        raise ValueError("profile case not in the suite: " + case)
    fields = ("min_timed_seconds", "max_timed_seconds", "min_timed_repeats")
    return dict({k: v for k, v in spec.items() if k not in fields}, cases=cases,
                timed_repeats=min(spec["timed_repeats"], PROFILE_REPEATS))


def run_candidate(args, pack):
    """TT sandbox: import the agent's backend.py, check its precision declaration, time every case
    with device synchronisation, and write actual.npz and measurements.json.

    The pack supplies:
      FORBIDDEN_HOST_OPS   learned aten ops that must never run on the host CPU (HostComputeGuard)
      ROWS                 the case field listing its rows (throughput items), or None for one item
      forward_case(backend, case, inputs, spec, config) -> (outputs, supplied)
          One timed call. ``inputs`` holds the case's arrays; pass the backend copies and return them
          as ``supplied`` (mutation is detected) with every output array to compare to the oracle.
    Optional:
      probe_case(...)      same signature; one untimed call on numerical cases (e.g. a decoder's
                           encoder states and logits), synchronised and checked the same way
      case_inputs(inputs, case, spec) -> dict    default: case_arrays(inputs, case)
    """
    pack = as_pack(pack)
    import importlib
    import resource
    import sys
    public, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((public / "suite.json").read_text())
    selected, identity = suite_model(spec, pack.KEY, pack.GATES)
    config = json.loads((public / "config.json").read_text())
    validate_config(config, selected)
    progress = Progress(out, spec["stage"])
    progress.mark("verify_checkpoint", initializing=True)
    validate_receipt(json.loads((public / "checkpoint_receipt.json").read_text()), identity, selected, spec)
    verified = verify_weights(args.weights, identity["model_key"])
    spec = profile_subset(spec)
    progress.mark("import_runtime", initializing=True)
    import numpy as np
    import torch  # noqa: F401  (weight conversion helpers on the TT host)
    import ttnn
    require_tt_precision(args.precision, ttnn, pack.PRECISIONS)
    calls = [0]
    for name in TT_COMPUTE_OPS:
        original = getattr(ttnn, name, None)
        if original:
            def counted(*a, _op=original, **kw):
                calls[0] += 1
                return _op(*a, **kw)
            setattr(ttnn, name, counted)
    sys.path.insert(0, os.getcwd())
    backend_module = importlib.import_module("backend")
    options = device_options(backend_module)
    inputs = np.load(public / "inputs.npz", allow_pickle=False)
    select = getattr(pack, "case_inputs", None) or (lambda inputs, case, spec: case_arrays(inputs, case))
    probe = getattr(pack, "probe_case", None)
    arrays, measured = {}, dict(cases=[], device_options=options, **identity, suite_sha256=suite_hash(spec),
                                timing_protocol=pack.TIMING_PROTOCOL, verified_weight_files=verified)

    def call(forward, backend, case, given):
        """One guarded call; returns (outputs, elapsed seconds including the device synchronisation)."""
        start = time.perf_counter()
        with guard():
            outputs, supplied = forward(backend, case, given, spec, config)
        elapsed = time.perf_counter() - start
        progress.mark("synchronize", case["name"])
        start = time.perf_counter()
        ttnn.synchronize_device(device)
        elapsed += time.perf_counter() - start
        validate_input_preservation({key: given[key] for key in supplied if key in given}, supplied)
        return {key: np.asarray(value) for key, value in outputs.items()}, elapsed

    progress.mark("open_device", initializing=True)
    device = profile_device(open_tt_device(ttnn, spec, options))
    try:
        progress.mark("create_backend", initializing=True)
        start = time.perf_counter()
        backend = backend_module.create_backend(args.weights, config, device, precision=args.precision)
        measured["precision"] = validate_precision(args.precision, getattr(backend, "precision_policy", None),
                                                   ttnn, pack.PRECISIONS)
        progress.mark("load_synchronize", initializing=True)
        ttnn.synchronize_device(device)
        measured["load_seconds"] = time.perf_counter() - start
        guard = HostComputeGuard(pack.FORBIDDEN_HOST_OPS, smallest_weight_matrix(args.weights, identity["model_key"]),
                                 backend, backend_module)
        view = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
        measured["resident_dram_bytes"] = int(view.num_banks) * int(view.total_bytes_allocated_per_bank)
        measured["trace_reserved_bytes"] = int(view.num_banks) * options.get("trace_region_size", 0)
        for case in spec["cases"]:
            name = case["name"]
            progress.mark("case_begin", name)
            given = select(inputs, case, spec)
            if probe and case["numerical"]:
                progress.mark("probe", name)
                outputs, _ = call(probe, backend, case, given)
                arrays.update({name + "__" + key: value.copy() for key, value in outputs.items()})
            first, samples, cold = None, [], None
            window = TimedWindow(len(case[pack.ROWS]) if pack.ROWS else 1)
            for repeat in repeat_schedule(spec, case):
                if repeat == 1:
                    guard.refresh()  # state the untimed first call built
                    window.open()
                progress.mark("pre_synchronize", name, repeat)
                ttnn.synchronize_device(device)
                progress.mark("forward", name, repeat)
                before = calls[0]
                outputs, elapsed = call(pack.forward_case, backend, case, given)
                if calls[0] == before:
                    raise ValueError("missing TT compute")
                if first is None:
                    first, cold = {key: value.copy() for key, value in outputs.items()}, elapsed
                elif outputs.keys() != first.keys() or any(not np.array_equal(first[k], outputs[k]) for k in first):
                    raise ValueError("TT nondeterminism")
                else:
                    samples.append(elapsed)
            arrays.update({name + "__" + key: value for key, value in first.items()})
            window.close()
            measured["cases"].append(dict(name=name, seconds=samples, first_seconds=cold, **window.record()))
            progress.mark("case_complete", name)
        progress.mark("write_outputs")
        np.savez_compressed(out / "actual.npz", **arrays)
        measured.update(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        observed_tt_calls=calls[0], memory_scope="resident allocator snapshot, not peak")
        write_json(out / "measurements.json", measured)
    finally:
        progress.mark("close_device")
        close_tt_device(ttnn, spec, device)
    progress.mark("complete")


class Assessment:
    """What assess() compares, loaded once. Identity and timing-coverage problems are already in
    ``failures``; ``checks`` collects one dict per case; ``notes`` holds a pack's cross-case state."""

    def __init__(self, args, pack):
        import numpy as np
        self.pack, self.failures, self.checks, self.notes = pack, [], [], {}
        self.public, self.private, self.out = Path(args.input), Path(args.oracle), Path(args.output)
        self.spec = spec = json.loads((self.public / "suite.json").read_text())
        self.selected, self.identity = suite_model(spec, pack.KEY, pack.GATES)
        self.actual = np.load(self.out / "actual.npz", allow_pickle=False)
        self.expected = np.load(self.private / "oracle.npz", allow_pickle=False)
        self.measured = json.loads((self.out / "measurements.json").read_text())
        self.timings = {entry["name"]: entry for entry in self.measured["cases"]}
        self._inputs = None
        try:
            selected, identity = self.selected, self.identity
            validate_config(json.loads((self.public / "config.json").read_text()), selected)
            validate_receipt(json.loads((self.public / "checkpoint_receipt.json").read_text()), identity, selected, spec)
            validate_artifact(self.measured, spec, identity, pack.TIMING_PROTOCOL)
            needed = [*selected["weight_files"], *([selected["weight_index"]] if selected["weight_index"] else [])]
            verified = self.measured.get("verified_weight_files")
            if not isinstance(verified, dict) or any(verified.get(k) != selected["files"][k] for k in needed):
                raise ValueError("candidate weight verification identity mismatch")
            self.reference = json.loads((self.private / "reference.json").read_text())
            validate_artifact(self.reference, spec, identity, pack.TIMING_PROTOCOL)
            validate_receipt(self.reference.get("checkpoint_receipt", {}), identity, selected, spec)
            if self.reference.get("precision", {}).get("requested") != "fp32":
                raise ValueError("correctness oracle must be FP32")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.failures.append("identity: " + str(exc))

    @property
    def inputs(self):
        """The public inputs.npz the candidate received."""
        if self._inputs is None:
            import numpy as np
            self._inputs = np.load(self.public / "inputs.npz", allow_pickle=False)
        return self._inputs

    def require_arrays(self, keys, candidate_only=()):
        """The oracle holds exactly ``keys``; the candidate those plus ``candidate_only``."""
        if set(self.actual.files) != set(keys) | set(candidate_only) or set(self.expected.files) != set(keys):
            self.failures.append("missing/unexpected arrays")

    def timing(self, check, case):
        """Validate a case's raw timing samples against the suite and record them in its check."""
        import math
        import statistics
        entry = self.timings[case["name"]]
        samples = entry["seconds"]
        if not samples or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError("invalid timing samples")
        if not repeat_count_ok(self.spec, case, len(samples)):
            raise ValueError("timing repeat count disagrees with prepared suite")
        cold = entry.get("first_seconds")
        if type(cold) not in (int, float) or not math.isfinite(cold) or cold <= 0:
            raise ValueError("invalid first-call timing")
        check.update(p50_seconds=statistics.median(samples), timing_samples=len(samples), seconds=samples,
                     first_seconds=cold)

    def labels(self):
        """The private ground truth for quality cases ({} with a recorded failure when unreadable)."""
        try:
            return json.loads((self.private / "labels.json").read_text())
        except (OSError, ValueError) as exc:
            self.failures.append("labels: " + str(exc))
            return {}

    def verdict(self, quality=None, max_nrmse=..., **extra):
        """Assemble, write and return verdict.json. max_nrmse defaults to the largest "*_nrmse" check field."""
        import statistics
        checks, spec, failures = self.checks, self.spec, self.failures
        if max_nrmse is ...:
            max_nrmse = max((v for c in checks for k, v in c.items() if k.endswith("_nrmse") and v is not None),
                            default=None)
        timings = [c["p50_seconds"] for c in checks if "p50_seconds" in c]
        verdict = dict(stage=spec["stage"], **self.identity, suite_sha256=suite_hash(spec),
                       timing_protocol=self.pack.TIMING_PROTOCOL,
                       passed=not failures, accepted=not failures and spec["stage"] == "full",
                       failures=failures, case_count=len(checks), max_nrmse=max_nrmse, **extra,
                       timing={"median_case_p50_seconds": statistics.median(timings) if timings else None,
                               "measured_cases": len(timings)},
                       latency=verdict_latency(checks, spec["cases"]), scaling=scaling_curve(checks, spec["cases"]),
                       precision=self.measured.get("precision"), checks=checks, quality=quality or {},
                       production_certified=False)
        write_json(self.out / "verdict.json", verdict)
        return verdict


def run_assess(args, pack):
    """Reference host: compare actual.npz with oracle.npz case by case and write verdict.json.

    The pack supplies:
      output_keys(case, spec) -> names of the arrays a case produces ("<case>__<name>" in both files)
      check_case(check, case, a) -> {unit: count} or None
          Numerics for one case against ``a`` (an Assessment): record fields in ``check`` and raise
          ValueError when a gate fails. Returned counts become "<unit>_per_second" rates.
    Optional:
      candidate_keys(case, a) -> extra "<case>__<name>" arrays only the candidate writes (e.g. probes
                           scored against oracle rows)
      quality(a) -> dict   task metrics on the full stage (only when every case passed); append to
                           a.failures when a quality gate fails
      verdict_fields(a) -> dict   extra verdict fields (e.g. its own max_nrmse)
    """
    pack = as_pack(pack)
    a = Assessment(args, pack)
    spec = a.spec
    extra = getattr(pack, "candidate_keys", None)
    a.require_arrays({case["name"] + "__" + key for case in spec["cases"] for key in pack.output_keys(case, spec)},
                     {key for case in spec["cases"] for key in (extra(case, a) if extra else ())})
    if len(a.measured["cases"]) != len(spec["cases"]) or set(a.timings) != {c["name"] for c in spec["cases"]}:
        a.failures.append("missing/duplicate timings")
    for case in spec["cases"]:
        check = {"name": case["name"]}
        try:
            counts = pack.check_case(check, case, a) or {}
            if "p50_seconds" not in check:  # a check_case that needs p50 may have validated timing itself
                a.timing(check, case)
            check.update({unit + "_per_second": count / check["p50_seconds"] if count else None
                          for unit, count in counts.items()})
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            a.failures.append(case["name"] + ": " + str(exc))
        a.checks.append(check)
    quality = pack.quality(a) if hasattr(pack, "quality") and spec["stage"] == "full" and not a.failures else {}
    return a.verdict(quality, **(pack.verdict_fields(a) if hasattr(pack, "verdict_fields") else {}))


class ReferenceRun:
    """The reference host's half of an evaluator: the FP32 oracle (``reference``) or a diagnostic
    precision benchmark of the official implementation (``reference-benchmark``).

        run = portforge_eval.ReferenceRun(args, pack)
        torch, device = run.torch()                # seeded, TF32 off, CUDA when available
        directory = run.checkpoint()               # download + verify the pinned files
        model = <load the official implementation from directory>
        run.loaded(parameter_dtypes=[...])
        for case in run.cases():
            inputs = run.prepared(case) or <build the case's arrays from run.corpus>
            outputs = run.time(case, lambda: <call>, lambda result: {<name>: array})
            run.record(case, inputs, outputs)
        return run.finish(labels)

    Public files: input/{suite.json, config.json, checkpoint_receipt.json, inputs.npz};
    private: oracle/{oracle.npz, reference.json, labels.json}. A benchmark run reuses those inputs,
    writes actual.npz and measurements.json beside them and returns the pack's assessment.
    """

    def __init__(self, args, pack, precisions=("fp32", "bf16")):
        pack = as_pack(pack)
        self.execution = reference_execution()
        self.pack, self.args = pack, args
        self.benchmark = getattr(args, "command", "reference") == "reference-benchmark"
        self.precision = args.precision if self.benchmark else "fp32"
        if self.precision not in precisions:
            raise ValueError("the reference exposes " + "/".join({"fp32": "float32", "bf16": "bfloat16",
                             "fp16": "float16"}.get(p, p) for p in precisions) + " only")
        self.root = Path(args.output if self.benchmark else args.out)
        self.root.mkdir(parents=True, exist_ok=True)
        self.public, self.private = self.root / "input", self.root / "oracle"
        self.corpus, self.labels, self.inputs, self.expected = None, {}, {}, {}
        if self.benchmark:
            import numpy as np
            self.public, self.private = Path(args.input), Path(args.oracle)
            oracle = json.loads((self.private / "reference.json").read_text())
            if oracle.get("precision", {}).get("requested", "fp32") != "fp32":
                raise ValueError("precision comparisons require the fixed FP32 oracle")
            self.spec = json.loads((self.public / "suite.json").read_text())
            if self.spec["stage"] not in ("smoke", "bringup"):
                raise ValueError("precision benchmark is diagnostic: use smoke or bringup inputs")
            self._prepared = np.load(self.public / "inputs.npz", allow_pickle=False)
        else:
            self.public.mkdir(exist_ok=True)
            self.private.mkdir(exist_ok=True)
            self.corpus = json.loads(Path(args.corpus).read_text())
            self.spec, self.labels = pack.make_suite(self.corpus, args.stage, getattr(args, "model", None) or pack.KEY)
        self.selected, self.identity = suite_model(self.spec, pack.KEY, pack.GATES)
        if self.benchmark:
            if getattr(args, "model", None) not in (None, self.identity["model_key"]):
                raise ValueError("benchmark model differs from prepared inputs")
            validate_artifact(oracle, self.spec, self.identity, pack.TIMING_PROTOCOL)
            validate_config(json.loads((self.public / "config.json").read_text()), self.selected)
            validate_receipt(oracle.get("checkpoint_receipt", {}), self.identity, self.selected, self.spec)
        self.progress = Progress(self.root, self.spec["stage"])
        self.progress.mark("load_model", initializing=True)
        self.started, self.device, self.device_note, self.receipt, self.meta = time.perf_counter(), None, None, None, None

    def torch(self, min_free_bytes=None):
        """Seeded torch on CUDA when available (FP32 matmuls and convolutions: TF32 and reduced-precision
        reductions off), else CPU. With min_free_bytes, a nearly full shared GPU moves the oracle to CPU
        (recorded); a precision benchmark always needs the GPU."""
        import torch
        torch.set_num_threads(12)
        torch.manual_seed(1729)
        available = torch.cuda.is_available()
        free = torch.cuda.mem_get_info()[0] if available and min_free_bytes else None
        if available and (self.benchmark or free is None or free >= min_free_bytes):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            self.device = "cuda"
            if self.precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError("this CUDA device lacks native BF16 support")
        else:
            self.device = "cpu"
            if free is not None:
                self.device_note = f"CUDA had {free} free bytes (< {min_free_bytes}); FP32 oracle on CPU"
        self._torch = torch
        return torch, self.device

    def checkpoint(self, token=False):
        """Download the pinned files, verify them, and publish config.json; returns the directory."""
        from huggingface_hub import hf_hub_download
        self.started = time.perf_counter()
        selected = self.selected
        for filename in selected["files"]:
            self.progress.mark("download_checkpoint", filename, initializing=True)
            path = Path(hf_hub_download(selected["repo_id"], filename, revision=selected["revision"], token=token))
        directory = path.parent
        self.progress.mark("verify_checkpoint", initializing=True)
        self.receipt = verify_checkpoint(directory, self.identity["model_key"], all_files=True)
        self.receipt["model_manifest_sha256"] = self.identity["model_manifest_sha256"]
        self.publish_config((directory / "config.json").read_text())
        return directory

    def publish_config(self, text):
        validate_config(json.loads(text), self.selected)
        if not self.benchmark:
            (self.public / "config.json").write_text(text)

    def loaded(self, parameter_dtypes, **extra):
        """Call once the model is loaded: starts reference.json (load time, host, versions, precision)."""
        torch, device = self._torch, self.device
        self.meta = dict(load_seconds=time.perf_counter() - self.started, device=device, device_note=self.device_note,
                         gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                         hostname=socket.gethostname(), execution=self.execution, job_id=os.getenv("SLURM_JOB_ID"),
                         torch_version=torch.__version__, **self.identity, checkpoint_receipt=self.receipt,
                         verified_weight_files=self.receipt["verified_files"], timing_protocol=self.pack.TIMING_PROTOCOL,
                         parameter_dtypes=list(parameter_dtypes),
                         precision=validate_precision(self.precision, dict(
                             mode=self.precision, weights=self.precision, activations=self.precision,
                             accumulation=f"PyTorch {device} operator-dependent; no autocast", exceptions=[]),
                             None, self.pack.PRECISIONS), cases=[], **extra)
        return self.meta

    def cases(self):
        for case in self.spec["cases"]:
            self.progress.mark("case_begin", case["name"])
            yield case
            self.progress.mark("case_complete", case["name"])

    def prepared(self, case):
        """A benchmark's prepared arrays for this case (copies), or None when building the oracle."""
        return {key: value.copy() for key, value in case_arrays(self._prepared, case).items()} if self.benchmark else None

    def time(self, case, call, read):
        """Untimed first call, then the suite's reference repeats: ``call()`` is timed through a CUDA
        synchronisation, ``read(result)`` (untimed) returns its arrays, which must repeat exactly.
        Records timing and peak device memory; returns the first call's arrays."""
        import numpy as np
        torch, cuda = self._torch, self.device == "cuda"
        name, first, samples, cold = case["name"], None, [], None
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        for repeat in range(reference_repeats(self.spec, case) + 1):
            self.progress.mark("reference_call", name, repeat)
            start = time.perf_counter()
            result = call()
            elapsed = time.perf_counter() - start
            self.progress.mark("reference_synchronize", name, repeat)
            start = time.perf_counter()
            if cuda:
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            arrays = {key: np.asarray(value) for key, value in read(result).items()}
            if first is None:
                first, cold = arrays, elapsed
            elif arrays.keys() != first.keys() or any(not np.array_equal(first[k], arrays[k]) for k in first):
                raise ValueError("reference nondeterminism")
            else:
                samples.append(elapsed)
        self.meta["cases"].append(dict(name=name, seconds=samples, first_seconds=cold,
                                       peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                                       peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None))
        return first

    def record(self, case, inputs, outputs):
        """Keep a case's public inputs and oracle outputs (``{name: array}``) for finish()."""
        self.inputs.update({case["name"] + "__" + key: value for key, value in (inputs or {}).items()})
        self.expected.update({case["name"] + "__" + key: value for key, value in outputs.items()})

    def finish(self, labels=None):
        """Write the public inputs and private oracle, or for a benchmark assess it; returns that verdict."""
        import numpy as np
        import resource
        self.progress.mark("write_outputs")
        self.meta["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.meta["suite_sha256"] = self.receipt["suite_sha256"] = suite_hash(self.spec)
        if self.benchmark:
            np.savez_compressed(self.root / "actual.npz", **self.expected)
            write_json(self.root / "measurements.json", self.meta)
            result = self.pack.assess(self.args)
            write_json(self.root / "verdict.json", result)
            self.progress.mark("complete")
            return result
        np.savez_compressed(self.public / "inputs.npz", **self.inputs)
        np.savez_compressed(self.private / "oracle.npz", **self.expected)
        write_json(self.public / "suite.json", self.spec)
        write_json(self.public / "checkpoint_receipt.json", self.receipt)
        write_json(self.private / "labels.json", self.labels if labels is None else labels)
        write_json(self.private / "reference.json", self.meta)
        self.progress.mark("complete")


def main(pack):
    """The evaluate.py command line every pack shares; the TT runner and the reference jobs call it."""
    pack = as_pack(pack)
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=pack.__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--stage", choices=ALL_STAGES, required=True)
    ref.add_argument("--model", default=pack.KEY)
    for flag in ("corpus", "out"):
        ref.add_argument("--" + flag, required=True)
    for command, flags in (("candidate", ("input", "output", "weights")), ("assess", ("input", "oracle", "output")),
                           ("reference-benchmark", ("input", "oracle", "output"))):
        child = sub.add_parser(command)
        for flag in flags:
            child.add_argument("--" + flag, required=True)
        if command == "candidate":
            child.add_argument("--precision", choices=pack.PRECISIONS, default="bf16")
        elif command == "reference-benchmark":
            child.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
            child.add_argument("--model", default=pack.KEY)
    args = parser.parse_args()
    result = {"reference": pack.reference, "reference-benchmark": pack.reference, "candidate": pack.candidate,
              "assess": pack.assess}[args.command](args)
    if args.command in ("assess", "reference-benchmark"):
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
        sys.exit(0 if result["passed"] else 1)
