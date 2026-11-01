"""Chronos-2 trusted evaluator: quantile-forecast suites, FP32 oracle, TT measurement and gates.

The shared scaffold is portforge_eval's evaluator kit; this file holds what is specific to
Chronos-2: the suites, the official pipeline as the oracle, the backend's forecast() call and
the quantile gates. Only input/ is exposed to the candidate. Run reference/assess on the
configured reference compute host (CUDA preferred, CPU allowed and recorded).
"""
import hashlib
import json
import math
import sys

import portforge_eval
from portforge_eval import (Progress, reference_execution, suite_hash, validate_input_preservation,  # noqa: F401
                            validate_receipt, write_json)
from portforge_models import model_spec

KEY = "chronos2"
MODEL = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
WEIGHT_SHA256 = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
CORPUS_SHA256 = "f0b08df4fe80385da1d3e9a05e3779708c4e808b2da581aa974fea9505d64ae3"
GATES = {"nrmse": .04, "behavior_max_nrmse": .01, "max_wql_inflation": .02}
PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TIMING_PROTOCOL = "cpu_arrays_to_cpu_arrays_with_sync_excluding_load_preprocessing_progress_v1"
MAX_CONTEXT = 512
MAX_HORIZON = 64
# Offline throughput: one card is swept to 1024 series per call; the mesh stage runs the same
# workload data parallel over MESH_CARDS cards of one host, up to MESH_CARDS * 1024 series.
MAX_BATCH = 1024
MESH_CARDS = 4
PERF_SWEEP = (1, 2, 4, 16, 64, 256, 1024)
MESH_SWEEP = (4, 6, 64, 1024, 4096)  # 6 does not split evenly over the cards
REFERENCE_BATCH = 256  # series per reference predict call
REFERENCE_MIN_FREE_BYTES = 8 * 1024 ** 3  # less free GPU memory than this: run the FP32 oracle on CPU
SWEEP_SERIES = ("etth1_hufl", "etth1_ot", "etth1_mull", "etth1_lufl", "etth1_lull")  # bringup series only
QUANTILES = tuple(model_spec("chronos2")["architecture"]["chronos_config"]["quantiles"])


def model_identity(name=KEY):
    return portforge_eval.model_identity(name)


def suite_model(spec):
    return portforge_eval.suite_model(spec, KEY, GATES)


def validate_precision(mode, policy, runtime=None):
    return portforge_eval.validate_precision(mode, policy, runtime, PRECISIONS)


def series_window(series, context, horizon, window, start=None):
    """Deterministic disjoint [context|future] slices; the future tail is ground truth only.
    An explicit start (sweep rows) overrides the disjoint window position."""
    start = window * (context + horizon) if start is None else start
    if start + context + horizon > len(series):
        raise ValueError("window exceeds the frozen series")
    return series[start:start + context], series[start + context:start + context + horizon]


def masked_context(values, mask_spec):
    """Apply the frozen missing-value pattern; masked cells keep zero placeholders."""
    mask = [1.0] * len(values)
    context = list(values)
    if mask_spec:
        if "tail" in mask_spec:
            for i in range(len(context) - mask_spec["tail"], len(context)):
                context[i], mask[i] = 0.0, 0.0
        for hole in mask_spec.get("holes", ()):
            context[hole], mask[hole] = 0.0, 0.0
    return context, mask


def sweep_rows(batch):
    """batch distinct windows: the bringup series in turn, each revisit starting one step later,
    so no two rows are equal (a backend cannot answer a sweep by deduplicating rows)."""
    keys = [SWEEP_SERIES[i % len(SWEEP_SERIES)] for i in range(batch)]
    return keys, [i // len(SWEEP_SERIES) for i in range(batch)]


def make_suite(corpus, stage, model="chronos2"):
    selected, model_fields = model_spec(model), model_identity(model)
    identity = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if identity != CORPUS_SHA256 or stage not in portforge_eval.STAGES + (portforge_eval.MESH_STAGE,):
        raise ValueError("unrecognized corpus or stage")
    series = corpus["series"]
    cases, labels = [], {}
    def add(name, keys, context, horizon, window=0, mask=None, starts=None, **extra):
        rows, masks, futures = [], [], []
        for i, key in enumerate(keys):
            past, future = series_window(series[key], context, horizon, window, starts[i] if starts else None)
            values, observed = masked_context(past, mask)
            rows.append(values)
            masks.append(observed)
            futures.append(future)
        cases.append(dict(name=name, series=list(keys), context=context, horizon=horizon,
                          window=window, mask=mask, numerical=True, **({"starts": starts} if starts else {}), **extra))
        labels[name] = futures
    if stage == "smoke":
        add("short", ["etth1_ot"], 512, 64)
    elif stage in ("bringup", "perf", portforge_eval.MESH_STAGE):
        add("batch", ["etth1_hufl", "etth1_ot"], 512, 64)
        for name, rows in (("single0", [0]), ("single1", [1]), ("reverse", [1, 0])):
            add(name, [["etth1_hufl", "etth1_ot"][i] for i in rows], 512, 64,
                compare_to="batch", compare_rows=rows)
        add("masked-tail", ["etth1_mull"], 512, 64, mask={"tail": 16})
        add("length-64", ["etth1_lufl"], 64, 64)
        add("length-65", ["etth1_lull"], 65, 64)
        add("constant", ["syn_constant"], 512, 64)
        if stage != "bringup":  # batch scaling sweep within the contract envelope
            for n in PERF_SWEEP if stage == "perf" else MESH_SWEEP:
                keys, starts = sweep_rows(n)
                add(f"sweep-b{n}", keys, 512, 64, starts=starts, batch=n)
    else:
        for key in [k for k in series if k != "syn_constant"]:
            add(f"{key}-w0", [key], 512, 64, window=0, quality=True)
            add(f"{key}-w1", [key], 128, 16, window=1, quality=True)
            add(f"{key}-w2", [key], 64, 64, window=2, quality=True)
        add("batch4", ["etth1_hufl", "etth1_ot", "syn_sine_mixed", "syn_arcsinh_stress"], 512, 64, quality=True)
        add("masked-holes", ["etth1_mull"], 512, 64, mask={"holes": [3, 17, 251, 500]}, quality=True)
        add("constant", ["syn_constant"], 512, 64, quality=True)
        for name, rows in (("single0", [0]), ("single1", [1]), ("reverse", [3, 2, 1, 0])):
            add("batch4-" + name, [["etth1_hufl", "etth1_ot", "syn_sine_mixed", "syn_arcsinh_stress"][i] for i in rows],
                512, 64, compare_to="batch4", compare_rows=rows)
        add("revisit", ["etth1_hufl", "etth1_ot", "syn_sine_mixed", "syn_arcsinh_stress"], 512, 64,
            compare_to="batch4", compare_rows=[0, 1, 2, 3])
        add("masked-tail", ["etth1_mull"], 512, 64, mask={"tail": 16})
    return dict(version=2, stage=stage, **model_fields,
                weight_sha256=selected["files"][selected["weight_files"][0]]["sha256"] if not selected["weight_index"] else None,
                corpus_sha256=CORPUS_SHA256, max_context=MAX_CONTEXT, gates=GATES.copy(),
                quantiles=list(QUANTILES), timed_repeats=portforge_eval.stage_repeats(stage), **portforge_eval.perf_fields(stage),
                **({"mesh_cards": MESH_CARDS} if stage == portforge_eval.MESH_STAGE else {}), cases=cases), labels


def quantile_nrmse(actual, expected):
    """Max per-row NRMSE over the (horizon, quantiles) forecast block."""
    import numpy as np
    if actual.shape != expected.shape or actual.dtype.kind != "f" or expected.dtype.kind != "f":
        raise ValueError("quantile shape/dtype check")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite quantile values")
    axes = tuple(range(1, actual.ndim))
    error = np.sqrt(np.mean((actual - expected) ** 2, axis=axes)) / np.maximum(
        1e-12, np.sqrt(np.mean(expected ** 2, axis=axes)))
    if not np.isfinite(error).all():
        raise ValueError("nonfinite NRMSE")
    return float(np.max(error))


def pinball_loss(quantiles, future, levels):
    """Mean pinball loss over steps and quantiles, scaled by the mean absolute target."""
    import numpy as np
    tau = np.asarray(levels, dtype=np.float64).reshape((1, -1))
    diff = future[:, :, None] - quantiles
    loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)
    scale = float(np.mean(np.abs(future)))
    return float(np.mean(loss) / max(scale, 1e-12))


def load_pipeline(directory, dtype, device):
    """Load through the public chronos-forecasting package; verify the pinned revision."""
    from chronos import Chronos2Pipeline
    try:
        return Chronos2Pipeline.from_pretrained(str(directory), dtype=dtype, device_map=device)
    except TypeError:
        return Chronos2Pipeline.from_pretrained(str(directory), torch_dtype=dtype, device_map=device)


def extract_quantiles(predictions, rows, horizon):
    """predict() -> list of (n_variates, n_quantiles, horizon) tensors; univariate only here."""
    import torch
    if len(predictions) != rows:
        raise ValueError("pipeline returned one tensor per univariate series expected")
    stacked = []
    for item in predictions:
        tensor = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != horizon:
            raise ValueError(f"unexpected forecast tensor shape {tuple(tensor.shape)}")
        stacked.append(tensor[0])  # [Q, H]
    return torch.stack(stacked).cpu().float().numpy()  # [B, Q, H]

def reference(args):
    """Reference host: the chronos-forecasting pipeline's quantile forecasts (FP32 oracle or benchmark)."""
    run = portforge_eval.ReferenceRun(args, sys.modules[__name__])
    if not run.benchmark and run.precision != "fp32":
        raise ValueError("the immutable correctness oracle is FP32")
    # A shared GPU can be nearly full of other users' processes; the FP32 oracle then runs on
    # CPU (allowed by the contract and recorded). Diagnostic benchmarks still need the GPU.
    torch, device = run.torch(min_free_bytes=REFERENCE_MIN_FREE_BYTES)
    dtype = {"fp32": "float32", "bf16": "bfloat16"}[run.precision]
    directory = run.checkpoint()
    run.progress.mark("load_model", initializing=True)
    pipeline = load_pipeline(directory, dtype, device)
    if device == "cuda" and device not in str(getattr(pipeline, "model", pipeline).device):
        raise RuntimeError("reference pipeline did not land on CUDA")
    run.loaded(parameter_dtypes=[dtype])
    with torch.inference_mode():
        for case in run.cases():
            arrays = run.prepared(case)
            if arrays is None:
                rows = [masked_context(series_window(run.corpus["series"][key], case["context"], case["horizon"],
                                                     case["window"], case["starts"][i] if "starts" in case else None)[0],
                                       case.get("mask")) for i, key in enumerate(case["series"])]
                arrays = dict(past_values=torch.tensor([r[0] for r in rows], dtype=torch.float32).numpy(),
                              past_observed_mask=torch.tensor([r[1] for r in rows], dtype=torch.float32).numpy())
            context = torch.from_numpy(arrays["past_values"].copy())
            context[torch.from_numpy(arrays["past_observed_mask"]) == 0] = float("nan")  # the package's missing marker
            batch = context.unsqueeze(1)  # [B, 1 variate, T]
            # Chunked: the package's group attention is quadratic in the batch it is given
            # (27.75 GiB at 4096 rows); univariate rows are independent groups either way.
            outputs = run.time(case, lambda: pipeline.predict(batch, prediction_length=case["horizon"],
                                                              batch_size=min(batch.shape[0], REFERENCE_BATCH),
                                                              limit_prediction_length=True),
                               lambda predictions: {"quantiles": extract_quantiles(
                                   predictions, batch.shape[0], case["horizon"]).transpose(0, 2, 1)})  # [B, H, Q]
            run.record(case, arrays, outputs)
    return run.finish()


# Candidate (TT sandbox): portforge_eval.run_candidate times forward_case.
FORBIDDEN_HOST_OPS = ("aten.embedding.", "aten.native_layer_norm.", "aten.layer_norm.", "aten._softmax.",
                      "aten.softmax.", "aten.gelu.", "aten.relu.", "aten.scaled_dot_product", "aten.sinh.")
ROWS = "series"


def forward_case(backend, case, inputs, spec, config):
    import numpy as np
    supplied = {key: inputs[key].copy() for key in ("past_values", "past_observed_mask")}
    result = backend.forecast(supplied["past_values"], supplied["past_observed_mask"], case["horizon"])
    quantiles = np.asarray(result["quantiles"], dtype=np.float32)
    if quantiles.shape != (len(case["series"]), case["horizon"], len(spec["quantiles"])):
        raise ValueError("forecast quantiles must be [B,H,Q]")
    return {"quantiles": quantiles}, supplied


def candidate(args):
    return portforge_eval.run_candidate(args, sys.modules[__name__])


# Assessment (reference host): portforge_eval.run_assess calls check_case per case, then quality.
def output_keys(case, spec):
    return ("quantiles",)


def check_case(check, case, a):
    import numpy as np
    name = case["name"]
    quantiles = a.actual[name + "__quantiles"]
    if quantiles.shape != (len(case["series"]), case["horizon"], len(a.spec["quantiles"])) or quantiles.dtype.kind != "f":
        raise ValueError("invalid quantiles shape/dtype")
    error = quantile_nrmse(quantiles.astype(np.float64), a.expected[name + "__quantiles"].astype(np.float64))
    check.update(rows=len(case["series"]), quantile_max_row_nrmse=error)
    if error > GATES["nrmse"]:
        raise ValueError("quantile NRMSE exceeds 0.04")
    if "compare_to" in case:
        base = a.actual[case["compare_to"] + "__quantiles"].astype(np.float64)
        for row, source in enumerate(case["compare_rows"]):
            if quantile_nrmse(quantiles[row][None], base[source][None]) > GATES["behavior_max_nrmse"]:
                raise ValueError("batch/order/revisit semantics changed for row " + str(row))
    check.update(forecast_points=int(quantiles.shape[0] * quantiles.shape[1]),
                 quantile_crossings=int(np.sum(np.diff(np.sort(quantiles, axis=-1), axis=-1) < 0)))
    return {"forecasts": quantiles.shape[0]}


def quality(a):
    """Full stage: weighted quantile loss against the held-out futures, relative to the FP32 oracle."""
    import numpy as np
    labels, quality = a.labels(), {}
    for case in a.spec["cases"]:
        if not case.get("quality"):
            continue
        name = case["name"]
        try:
            future = np.asarray(labels[name], dtype=np.float64)
            candidate_loss, reference_loss = (pinball_loss(data[name + "__quantiles"].astype(np.float64), future,
                                                           a.spec["quantiles"]) for data in (a.actual, a.expected))
            if not math.isfinite(candidate_loss) or not math.isfinite(reference_loss):
                raise ValueError("nonfinite task loss")
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            a.failures.append(name + ": " + str(exc))
            continue
        inflation = candidate_loss / max(reference_loss, 1e-12) - 1
        quality[name] = dict(wql=candidate_loss, reference_wql=reference_loss, inflation=inflation)
        if inflation > GATES["max_wql_inflation"]:
            a.failures.append(name + f": weighted quantile-loss inflation exceeds {GATES['max_wql_inflation']:.0%}")
    return quality


def assess(args):
    return portforge_eval.run_assess(args, sys.modules[__name__])


if __name__ == "__main__":
    portforge_eval.main(sys.modules[__name__])
