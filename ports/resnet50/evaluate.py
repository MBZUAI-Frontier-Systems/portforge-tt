"""ResNet-50 trusted evaluator: ImageNet-1k suites, FP32 oracle, TT measurement and task-level gates.

The shared scaffold is portforge_eval's evaluator kit; this file holds what is specific to ResNet-50.
Only input/ is exposed to the candidate: uint8 224x224 RGB crops per case, never labels or oracle
logits. The oracle is torchvision resnet50 in FP32 (CUDA preferred, CPU recorded) on the pinned
safetensors. The corpus is private (ImageNet terms of access): build_corpus.py builds it and the
launcher takes it with --corpus.
"""
import base64
from fractions import Fraction
import hashlib
import io
import json

import portforge_eval
from portforge_eval import (check_suite, reference_execution, suite_hash, validate_input_preservation,  # noqa: F401
                            validate_receipt, write_json)
from portforge_models import model_spec

KEY = "resnet50"
MODEL = "timm/resnet50.tv_in1k"
REVISION = "78f3ecfdb38e06d9b8397f662e7ab8fee96026fa"
# Pinned after build_corpus.py has produced the private corpus (it prints the identity).
CORPUS_SHA256 = "67c2b735c39017c2203f747d32156548dabd1fa1ef728366753f01e265c6d030"
# Task-level track.
#   nrmse                max per-row logits NRMSE against the FP32 oracle; sanity only (wrong weights or layout)
#   behavior_max_nrmse   per-row NRMSE between a consistency case and the same images in its compare_to case
#   top1_agreement       min top-1 agreement with the oracle per all-labelled smoke/bringup/perf case (14/16)
#   full_top1_drop_pp    max (oracle top-1 - candidate top-1) over the held-out images, percentage points
#   full_top1_agreement  min top-1 agreement with the oracle over the held-out images
GATES = {"nrmse": .30, "behavior_max_nrmse": .05, "top1_agreement": .875,
         "full_top1_drop_pp": 1.0, "full_top1_agreement": .95}
PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TIMING_PROTOCOL = "normalized_pixels_to_logits_with_sync_excluding_load_normalization_progress_v1"
IMAGE_SIZE, BATCH_SIZES, TOP_K = 224, (16, 32), 5
DEV_COUNT, HELDOUT_COUNT, SYNTHETIC_COUNT = 48, 1024, 5
STREAM_IMAGES = 256  # perf stream cases: 16 batches of 16, 8 batches of 32
REFERENCE_CHUNK = 64


def model_identity(name=KEY):
    return portforge_eval.model_identity(name)


def suite_model(spec):
    return portforge_eval.suite_model(spec, KEY, GATES)


def validate_precision(mode, policy, runtime=None):
    return portforge_eval.validate_precision(mode, policy, runtime, PRECISIONS)


def corpus_identity(corpus):
    return hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def make_suite(corpus, stage, model=KEY):
    """Frozen suites from the corpus pools (selection: dev, heldout, synthetic).

    smoke: one batch of 16 dev images.
    bringup: batches of 16 and 32, a reversed batch, a mixed batch from two slices of the batch of 32,
    a repeat, five synthetic images with eleven dev images, and a stream of 4 batches of 16.
    perf: bringup plus the sweep (one call of 16 or 32) and streams of 256 images (16x16, 8x32).
    full: 64 held-out batches of 16 (quality, 1,024 images), a batch of 32, a reversed batch, the
    synthetic batch and a held-out stream of 16 batches of 16.
    A case with "stream": n is n back-to-back batches of "batch" rows in one classify_stream call.
    Returns (spec, labels): per-row ImageNet labels for each case (None for synthetic rows).
    """
    if corpus.get("schema_version") != 1 or stage not in portforge_eval.STAGES:
        raise ValueError("unsupported corpus schema or stage")
    if corpus_identity(corpus) != CORPUS_SHA256:
        raise ValueError("unrecognized corpus: identity does not match the pinned CORPUS_SHA256")
    selected, identity = model_spec(model), model_identity(model)
    pools = corpus["selection"]
    dev, heldout, synthetic = list(pools["dev"]), list(pools["heldout"]), list(pools["synthetic"])
    if (len(dev), len(heldout), len(synthetic)) != (DEV_COUNT, HELDOUT_COUNT, SYNTHETIC_COUNT):
        raise ValueError("corpus pools must hold 48 dev, 1024 held-out and 5 synthetic images")
    entries = {entry["name"]: entry for entry in corpus["cases"]}
    if len(entries) != len(corpus["cases"]):
        raise ValueError("duplicate corpus case")
    cases, labels = [], {}

    def add(name, clips, stream=None, **extra):
        size = len(clips) // (stream or 1)
        if size not in BATCH_SIZES or size * (stream or 1) != len(clips) or any(c not in entries for c in clips):
            raise ValueError(name + ": batches hold 16 or 32 corpus images")
        if stream:
            extra.update(batch=size, stream=stream)
        cases.append(dict(name=name, clips=list(clips), numerical=True, **extra))
        labels[name] = [entries[clip].get("label") for clip in clips]

    def ring(pool, count):
        return [pool[i % len(pool)] for i in range(count)]

    if stage == "smoke":
        add("batch16", dev[0:16])
    elif stage in ("bringup", "perf"):
        add("batch16", dev[0:16])
        add("batch16-reverse", dev[0:16][::-1], compare_to="batch16", compare_rows=list(range(15, -1, -1)))
        add("batch16-b", dev[16:32])
        add("batch32", dev[0:32])
        add("batch16-mixed", dev[0:8] + dev[16:24], compare_to="batch32",
            compare_rows=list(range(8)) + list(range(16, 24)))
        add("batch16-c", dev[32:48])
        add("synthetic", synthetic + dev[32:43])
        add("repeat", dev[0:16], compare_to="batch16", compare_rows=list(range(16)))
        add("stream-b16x4", ring(dev, 64), stream=4, compare_to="batch16", compare_rows=list(range(16)))
        if stage == "perf":  # throughput within the envelope: one call per batch, and streams
            for n in BATCH_SIZES:
                add(f"sweep-b{n}", dev[0:n], batch=n)
            for n in BATCH_SIZES:
                add(f"stream-b{n}x{STREAM_IMAGES // n}", ring(dev, STREAM_IMAGES), stream=STREAM_IMAGES // n,
                    compare_to=f"sweep-b{n}", compare_rows=list(range(n)))
    else:
        for index in range(HELDOUT_COUNT // 16):
            add(f"ho-batch-{index:02d}", heldout[16 * index:16 * index + 16], quality=True)
        add("ho-batch32", heldout[0:32])
        add("ho-batch-00-reverse", heldout[0:16][::-1], compare_to="ho-batch-00", compare_rows=list(range(15, -1, -1)))
        add("synthetic", synthetic + dev[32:43])
        add("ho-stream-b16x16", heldout[0:256], stream=16, compare_to="ho-batch-00", compare_rows=list(range(16)))
    config = selected["architecture"]["pretrained_cfg"]
    spec = dict(version=2, stage=stage, **identity, corpus_sha256=CORPUS_SHA256, gates=GATES.copy(),
                image_size=IMAGE_SIZE, num_classes=selected["architecture"]["num_classes"],
                mean=list(config["mean"]), std=list(config["std"]),
                timed_repeats=portforge_eval.stage_repeats(stage), **portforge_eval.perf_fields(stage), cases=cases)
    return check_suite(spec), labels


def decode_case(entry):
    """Corpus PNG -> uint8 [224,224,3]; the stored pixel hash must match what decodes."""
    import numpy as np
    from PIL import Image
    pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(entry["png_b64"], validate=True))).convert("RGB"),
                        dtype=np.uint8)
    if pixels.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(entry["name"] + ": corpus image must be 224x224 RGB")
    if hashlib.sha256(pixels.tobytes()).hexdigest() != entry["pixels_sha256"]:
        raise ValueError(entry["name"] + ": decoded pixels do not match the corpus pixels_sha256")
    return pixels


def normalize(pixels, spec):
    """uint8 [B,224,224,3] -> float32 [B,3,224,224] with the pinned mean and std.

    Plain float32 NumPy on both hosts, so the oracle and the candidate get bit-identical inputs."""
    import numpy as np
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("pixels must be uint8 [B,224,224,3]")
    mean, std = np.asarray(spec["mean"], dtype=np.float32), np.asarray(spec["std"], dtype=np.float32)
    values = (pixels.astype(np.float32) / np.float32(255.0) - mean) / std
    return np.ascontiguousarray(values.transpose(0, 3, 1, 2))


def max_row_nrmse(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or actual.dtype.kind != "f" or expected.dtype.kind != "f":
        raise ValueError("shape/dtype check")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite values")
    error = np.sqrt(np.mean((actual - expected) ** 2, axis=1)) / np.maximum(1e-12, np.sqrt(np.mean(expected ** 2, axis=1)))
    if not np.isfinite(error).all():
        raise ValueError("nonfinite NRMSE")
    return float(np.max(error))


def reference(args):
    """Reference host: torchvision resnet50 logits (FP32 oracle or BF16 benchmark)."""
    import numpy as np
    run = portforge_eval.ReferenceRun(args, globals())
    torch, device = run.torch()
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[run.precision]
    directory = run.checkpoint()
    run.progress.mark("load_model", initializing=True)
    import torchvision
    from safetensors.torch import load_file
    model = torchvision.models.resnet50(weights=None)
    model.load_state_dict(load_file(str(directory / run.selected["weight_files"][0]), device="cpu"), strict=True)
    model = model.eval().to(device=device, dtype=dtype)
    run.loaded(parameter_dtypes=[str(dtype).split(".")[-1]], torchvision_version=torchvision.__version__)
    entries = {entry["name"]: entry for entry in run.corpus["cases"]} if run.corpus else None
    with torch.inference_mode():
        for case in run.cases():
            arrays = run.prepared(case)
            if arrays is None:
                arrays = {"pixels": np.stack([decode_case(entries[clip]) for clip in case["clips"]])}
            values = torch.from_numpy(normalize(arrays["pixels"], run.spec)).to(device=device, dtype=dtype)
            outputs = run.time(case, lambda: [model(chunk) for chunk in values.split(REFERENCE_CHUNK)],
                               lambda result: {"logits": torch.cat(result).float().cpu().numpy()})
            run.record(case, arrays, outputs)
    return run.finish()


# Candidate (TT sandbox): portforge_eval.run_candidate times forward_case.
FORBIDDEN_HOST_OPS = ("aten.convolution.", "aten._convolution.", "aten.conv2d.", "aten.cudnn_convolution",
                      "aten.mkldnn_convolution.", "aten.max_pool2d", "aten.avg_pool2d.", "aten.adaptive_avg_pool2d.",
                      "aten._adaptive_avg_pool2d.", "aten.mean.", "aten.native_batch_norm.",
                      "aten._native_batch_norm_legit", "aten.batch_norm.", "aten.relu.", "aten.relu_.",
                      "aten.add.", "aten.add_.", "aten._softmax.", "aten.softmax.")
ROWS = "clips"


def case_inputs(inputs, case, spec):
    """Normalized pixels, built once per case outside the timed calls and passed read-only rather
    than copied per call (a batch of 16 is 9.6 MB). Writing into them raises in NumPy; a backend that
    mutates them through another view changes its later repeats, which must be bit-identical."""
    values = normalize(inputs[case["name"] + "__pixels"], spec)
    values.setflags(write=False)
    return {"pixel_values": values}


def forward_case(backend, case, inputs, spec, config):
    import numpy as np
    values = inputs["pixel_values"]
    if case.get("stream"):
        size = case["batch"]
        result = backend.classify_stream([values[i * size:(i + 1) * size] for i in range(case["stream"])])
    else:
        result = backend.classify(values)
    logits = np.asarray(result["logits"], dtype=np.float32)
    if logits.shape != (len(case["clips"]), spec["num_classes"]):
        raise ValueError("logits must be [images, num_classes] in input order")
    return {"logits": logits}, {"pixel_values": values}


def candidate(args):
    return portforge_eval.run_candidate(args, globals())


# Assessment (reference host): portforge_eval.run_assess calls check_case per case, then quality on full.
def output_keys(case, spec):
    return ("logits",)


def case_labels(a, name):
    if "labels" not in a.notes:
        a.notes["labels"] = a.labels()
    return a.notes["labels"].get(name)


def check_case(check, case, a):
    import numpy as np
    name = case["name"]
    a.timing(check, case)  # first, so a case that fails a gate still reports its speed
    actual = a.actual[name + "__logits"].astype(np.float64)
    oracle = a.expected[name + "__logits"].astype(np.float64)
    check["logits_max_row_nrmse"] = error = max_row_nrmse(actual, oracle)
    check["rows"] = rows = actual.shape[0]
    check["top1_agreement_count"] = agree = int((actual.argmax(axis=1) == oracle.argmax(axis=1)).sum())
    if error > GATES["nrmse"]:
        raise ValueError(f"logits NRMSE {error:.4f} exceeds {GATES['nrmse']}")
    if "compare_to" in case:
        source = case["compare_rows"]
        base = a.actual[case["compare_to"] + "__logits"].astype(np.float64)
        check["behavior_max_row_error"] = worst = max_row_nrmse(actual[:len(source)], base[source])
        if worst > GATES["behavior_max_nrmse"]:
            raise ValueError(f"batch/order/repeat/stream semantics changed: row NRMSE {worst:.4f} against "
                             f"{case['compare_to']}")
    labels = case_labels(a, name)
    if a.spec["stage"] != "full" and labels and None not in labels:
        if Fraction(agree, rows) < Fraction(repr(GATES["top1_agreement"])):
            raise ValueError(f"top-1 agreement with the oracle {agree}/{rows} below {GATES['top1_agreement']}")
    return {"images": rows}


def quality_summary(candidate, oracle, labels):
    """Held-out metrics from exact counts. top5_agreement: the oracle's top-1 is in the candidate's top-5."""
    import numpy as np
    candidate, oracle = np.asarray(candidate, dtype=np.float64), np.asarray(oracle, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if candidate.shape != oracle.shape or candidate.ndim != 2 or labels.shape != candidate.shape[:1] or not len(labels):
        raise ValueError("quality needs [rows, classes] logits on both sides and one label per row")
    if not (np.isfinite(candidate).all() and np.isfinite(oracle).all()):
        raise ValueError("nonfinite quality logits")
    if ((labels < 0) | (labels >= candidate.shape[1])).any():
        raise ValueError("label outside the class range")
    images = int(labels.shape[0])
    mine, theirs = candidate.argmax(axis=1), oracle.argmax(axis=1)
    top5 = np.argsort(-candidate, axis=1, kind="stable")[:, :TOP_K]
    counts = dict(images=images, oracle_top1_correct=int((theirs == labels).sum()),
                  candidate_top1_correct=int((mine == labels).sum()),
                  top1_agreement_count=int((mine == theirs).sum()),
                  top5_agreement_count=int((top5 == theirs[:, None]).any(axis=1).sum()))
    return dict(counts, oracle_top1_accuracy=counts["oracle_top1_correct"] / images,
                candidate_top1_accuracy=counts["candidate_top1_correct"] / images,
                top1_drop_pp=100.0 * (counts["oracle_top1_correct"] - counts["candidate_top1_correct"]) / images,
                top1_agreement=counts["top1_agreement_count"] / images,
                top5_agreement=counts["top5_agreement_count"] / images)


def quality_failures(summary):
    """Exact comparison of the integer counts against the decimal gates."""
    failures, images = [], summary["images"]
    drop = Fraction(summary["oracle_top1_correct"] - summary["candidate_top1_correct"], images)
    if drop * 100 > Fraction(repr(GATES["full_top1_drop_pp"])):
        failures.append(f"held-out top-1 drop {float(drop) * 100:.3f} pp exceeds {GATES['full_top1_drop_pp']} pp")
    if Fraction(summary["top1_agreement_count"], images) < Fraction(repr(GATES["full_top1_agreement"])):
        failures.append(f"held-out top-1 agreement {summary['top1_agreement_count']}/{images} "
                        f"below {GATES['full_top1_agreement']}")
    return failures


def quality(a):
    import numpy as np
    cases = [case for case in a.spec["cases"] if case.get("quality")]
    try:
        rows = [label for case in cases for label in case_labels(a, case["name"])]
        summary = quality_summary(np.concatenate([a.actual[c["name"] + "__logits"] for c in cases]),
                                  np.concatenate([a.expected[c["name"] + "__logits"] for c in cases]), rows)
    except (KeyError, TypeError, ValueError) as exc:
        a.failures.append("quality: " + str(exc))
        return {}
    a.failures.extend(quality_failures(summary))
    return summary


def assess(args):
    return portforge_eval.run_assess(args, globals())


if __name__ == "__main__":
    portforge_eval.main(globals())
