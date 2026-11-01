"""Parakeet trusted evaluator: real-audio suites, FP32 oracle, TT measurement and gates.

The shared scaffold is portforge_eval's evaluator kit; this file holds what is specific to
Parakeet. Only input/ is exposed to the candidate (mel features + lengths; never raw audio,
never transcripts). Run reference/assess on the configured reference host (CUDA preferred,
CPU allowed and recorded); the candidate imports the agent's backend.py in its current working
directory on the TT host. Deterministic mel preprocessing (STFT/log-mel) is declared policy and
runs on the host CPU, like tokenization did for translation; learned compute must run on TT/CUDA.
"""
import base64
import hashlib
import io
import json
import math
import sys

import portforge_eval
from portforge_eval import (Progress, reference_execution, suite_hash, validate_input_preservation,  # noqa: F401
                            validate_receipt, write_json)
from portforge_models import model_spec

KEY = "parakeet"
MODEL = "nvidia/parakeet-tdt-0.6b-v3"
REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
WEIGHT_SHA256 = "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11"
CORPUS_SHA256 = '62fb9c517719931999ccd72b9c097e889e843aaf8345567f62f16e95e8248d02'
GATES = {"nrmse": .04, "max_wer_inflation": .02, "behavior_max_nrmse": .01}
PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TIMING_PROTOCOL = "mel_features_to_tokens_with_sync_excluding_load_preprocessing_progress_v1"
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 32


def model_identity(name=KEY):
    return portforge_eval.model_identity(name)


def suite_model(spec):
    return portforge_eval.suite_model(spec, KEY, GATES)


def validate_precision(mode, policy, runtime=None):
    return portforge_eval.validate_precision(mode, policy, runtime, PRECISIONS)


def load_audio(entry):
    """Decode one frozen corpus clip to mono float32 samples at 16 kHz."""
    import numpy as np
    import soundfile as sf
    data, rate = sf.read(io.BytesIO(base64.b64decode(entry["audio_b64"])), dtype="float32")
    if data.ndim != 1:
        data = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise ValueError("frozen corpus clips are pinned at 16 kHz; got " + str(rate))
    if len(data) > MAX_AUDIO_SECONDS * SAMPLE_RATE:
        raise ValueError("clip exceeds the frozen envelope")
    return np.asarray(data, dtype=np.float32)


def make_suite(corpus, stage, model="parakeet"):
    selected, model_fields = model_spec(model), model_identity(model)
    identity = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if identity != CORPUS_SHA256 or stage not in portforge_eval.STAGES:
        raise ValueError("unrecognized corpus or stage")
    entries = {entry["name"]: entry for entry in corpus["cases"]}
    cases, labels = [], {}
    def add(name, keys, numerical=True, **extra):
        rows = [entries[key] for key in keys]
        cases.append(dict(name=name, clips=list(keys), numerical=numerical, **extra))
        labels[name] = [row["transcript"] for row in rows]
    if stage == "smoke":
        add("short", [corpus["selection"]["smoke"]])
    elif stage in ("bringup", "perf"):
        pick = corpus["selection"]["bringup"]
        add("batch", [pick["batch0"], pick["batch1"]])
        for name, rows in (("single0", [0]), ("single1", [1]), ("reverse", [1, 0])):
            add(name, [pick["batch" + str(i)] for i in rows], compare_to="batch", compare_rows=rows)
        add("long", [pick["long"]])
        add("short", [pick["short"]])
        add("silence", [pick["silence"]])
        add("tone", ["syn-tone"])
        if stage == "perf":  # batch scaling sweep within the contract envelope
            for n in portforge_eval.PERF_BATCHES:
                add(f"sweep-b{n}", [corpus["selection"]["smoke"]] * n, batch=n)
    else:
        for key in corpus["selection"]["full"]:
            add(key, [key], quality=True)
        pick = corpus["selection"]["bringup"]
        batch4 = [pick["batch0"], pick["batch1"], corpus["selection"]["full"][0], corpus["selection"]["full"][-1]]
        add("batch4", batch4, quality=True)
        add("batch4-reverse", batch4[::-1], compare_to="batch4", compare_rows=[3, 2, 1, 0])
        # batch4 rows 0-1 hold the same two clips; the full stage has no "batch" case.
        add("revisit", batch4[:2], compare_to="batch4", compare_rows=[0, 1])
        add("silence", [pick["silence"]])
        add("long", [pick["long"]])
    return dict(version=2, stage=stage, **model_fields,
                weight_sha256=selected["files"][selected["weight_files"][0]]["sha256"] if not selected["weight_index"] else None,
                corpus_sha256=CORPUS_SHA256, max_audio_seconds=MAX_AUDIO_SECONDS, gates=GATES.copy(),
                timed_repeats=portforge_eval.stage_repeats(stage), **portforge_eval.perf_fields(stage), cases=cases), labels


def canonical_tokens(row, blank, pad, vocab):
    """Strip padding after the first blank-free greedy sequence; validate ids."""
    row = [int(x) for x in row]
    if any(x < 0 or x >= vocab for x in row):
        raise ValueError("token outside pinned vocabulary")
    end = len(row)
    while end > 0 and row[end - 1] == pad:
        end -= 1
    return row[:end]


def word_error_rate(hypothesis, reference_text):
    """Word-level Levenshtein distance over whitespace tokens, normalized by reference length."""
    hyp, ref = hypothesis.split(), reference_text.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(ref) + 1))
    for i, word in enumerate(hyp, 1):
        current = [i]
        for j, ref_word in enumerate(ref, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (word != ref_word)))
        previous = current
    return previous[-1] / len(ref)


def quantile_free_nrmse(actual, expected):
    """Max per-row NRMSE over the encoder time dimension."""
    import numpy as np
    if actual.shape != expected.shape or actual.dtype.kind != "f" or expected.dtype.kind != "f":
        raise ValueError("encoder shape/dtype check")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite encoder values")
    axes = tuple(range(1, actual.ndim))
    error = np.sqrt(np.mean((actual - expected) ** 2, axis=axes)) / np.maximum(
        1e-12, np.sqrt(np.mean(expected ** 2, axis=axes)))
    if not np.isfinite(error).all():
        raise ValueError("nonfinite NRMSE")
    return float(np.max(error))


def load_parakeet(directory, dtype, device):
    """Load through the public transformers implementation of ParakeetForTDT."""
    import torch
    import transformers
    cls = getattr(transformers, "AutoModelForTDT", None) or getattr(transformers, "ParakeetForTDT", None)
    if cls is None:
        raise RuntimeError("installed transformers lacks Parakeet TDT support (need >= 5.6)")
    return cls.from_pretrained(str(directory), dtype=dtype, local_files_only=True,
                               trust_remote_code=False, weights_only=True, use_safetensors=True).eval().to(device)


def encode_reference(model, mel, frame_mask, device):
    """Encoder last hidden states through the public implementation, with shape fallbacks."""
    import torch
    inner = model.model if hasattr(model, "model") else model
    with torch.inference_mode():
        dtype = next(model.parameters()).dtype  # bf16 study: features must match the weights
        enc = inner.encoder(input_features=mel.to(device=device, dtype=dtype),
                            attention_mask=frame_mask.to(device))
        states = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc
        if states is None:
            raise RuntimeError("reference encoder produced no hidden states; verify the installed transformers API")
        return states



def reference(args):
    """Reference host: transformers ParakeetForTDT encoder states and greedy tokens (FP32 oracle or benchmark)."""
    import numpy as np
    run = portforge_eval.ReferenceRun(args, sys.modules[__name__])
    torch, device = run.torch()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[run.precision]
    directory = run.checkpoint()
    run.progress.mark("load_model", initializing=True)
    from transformers import AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    model = load_parakeet(directory, dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    run.loaded(parameter_dtypes=[str(dtype).split(".")[-1]])
    architecture = run.selected["architecture"]
    blank, pad, vocab = architecture["blank_token_id"], architecture["pad_token_id"], architecture["vocab_size"]
    entries = {entry["name"]: entry for entry in run.corpus["cases"]} if run.corpus else None
    with torch.inference_mode():
        for case in run.cases():
            name = case["name"]
            arrays = run.prepared(case)
            if arrays is None:
                waveforms = [load_audio(entries[key]) for key in case["clips"]]
                if any(len(w) == 0 for w in waveforms):
                    raise ValueError("empty frozen clip")
                features = processor(waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
                # Parakeet processor: input_features [B, T, 128 mel] (time-major),
                # attention_mask [B, T] with row sums = true frame counts.
                mel = features.input_features
                if not torch.is_tensor(mel):
                    mel = torch.stack([torch.as_tensor(m) for m in mel])
                frame_mask = features.attention_mask.long()
                arrays = dict(mel=mel.to(torch.float32).numpy().astype(np.float32),
                              mel_lengths=frame_mask.sum(dim=-1).numpy().astype(np.int64),
                              attention_mask=frame_mask.numpy().astype(np.int64),
                              audio_seconds=np.array([len(w) / SAMPLE_RATE for w in waveforms], dtype=np.float32))
            mel = torch.from_numpy(arrays["mel"])
            frame_mask = torch.from_numpy(arrays["attention_mask"]).long()
            oracle = {}
            if case["numerical"]:
                run.progress.mark("encode", name)
                encoded = encode_reference(model, mel, frame_mask, device)
                run.progress.mark("encode_readback", name)
                oracle["encoder"] = encoded.float().cpu().numpy()
            oracle["tokens"] = run.time(
                case, lambda: model.generate(input_features=mel.to(device=device, dtype=next(model.parameters()).dtype),
                                             attention_mask=frame_mask.to(device), return_dict_in_generate=True),
                lambda generated: {"tokens": generated.sequences.cpu().numpy()})["tokens"]
            canonical = [canonical_tokens(row, blank, pad, vocab) for row in oracle["tokens"]]
            run.meta["cases"][-1]["transcripts"] = tokenizer.batch_decode([torch.tensor(row) for row in canonical],
                                                                          skip_special_tokens=True)
            if not run.benchmark:
                arrays["tokens_reference_shape"] = np.array(oracle["tokens"].shape, dtype=np.int64)
            run.record(case, arrays, oracle)
    return run.finish()


# Candidate (TT sandbox): portforge_eval.run_candidate probes encode and times transcribe.
# Learned ops that never run on the host CPU. Host matmuls are allowed only without
# model weights (portforge_eval.HostComputeGuard), e.g. RoPE angles from positions.
FORBIDDEN_HOST_OPS = ("aten.embedding.", "aten.native_layer_norm.", "aten.layer_norm.", "aten._softmax.",
                      "aten.softmax.", "aten.gelu.", "aten.silu.", "aten.scaled_dot_product",
                      "aten.conv2d.", "aten.conv1d.", "aten._convolution.")
ROWS = "clips"


def probe_case(backend, case, inputs, spec, config):
    """Untimed, numerical cases only: the encoder states compared with the oracle."""
    import numpy as np
    supplied = {key: inputs[key].copy() for key in ("mel", "mel_lengths")}
    result = backend.encode(supplied["mel"], supplied["mel_lengths"])
    encoder = np.asarray(result["encoder"], dtype=np.float32)
    mel, encoder_config = inputs["mel"], config["encoder_config"]
    expected_frames = -(-mel.shape[1] // encoder_config["subsampling_factor"])
    if encoder.shape != (mel.shape[0], expected_frames, encoder_config["hidden_size"]):
        raise ValueError("encoder output must be [B, ceil(T/8), hidden]")
    return {"encoder": encoder}, supplied


def forward_case(backend, case, inputs, spec, config):
    import numpy as np
    supplied = {key: inputs[key].copy() for key in ("mel", "mel_lengths")}
    tokens = np.asarray(backend.transcribe(supplied["mel"], supplied["mel_lengths"])["tokens"])
    if tokens.dtype.kind not in "iu" or tokens.ndim != 2:
        raise ValueError("transcribe must return integer token ids [B,L]")
    return {"tokens": tokens}, supplied


def candidate(args):
    return portforge_eval.run_candidate(args, sys.modules[__name__])


# Assessment (reference host): portforge_eval.run_assess calls check_case per case, then quality.
def output_keys(case, spec):
    return ("tokens", "encoder") if case["numerical"] else ("tokens",)


def check_case(check, case, a):
    import numpy as np
    name = case["name"]
    blank, pad, vocab = (a.selected["architecture"][k] for k in ("blank_token_id", "pad_token_id", "vocab_size"))
    tokens = a.actual[name + "__tokens"]
    if tokens.ndim != 2 or tokens.shape[0] != len(case["clips"]) or tokens.dtype.kind not in "iu":
        raise ValueError("invalid tokens shape/dtype")
    canonical = [canonical_tokens(row, blank, pad, vocab) for row in tokens]
    oracle = [canonical_tokens(row, blank, pad, vocab) for row in a.expected[name + "__tokens"]]
    check["rows"] = len(case["clips"])
    check["exact_reference_rows"] = sum(x == y for x, y in zip(canonical, oracle))
    if case["numerical"]:
        error = quantile_free_nrmse(a.actual[name + "__encoder"].astype(np.float64),
                                    a.expected[name + "__encoder"].astype(np.float64))
        check["encoder_max_row_nrmse"] = error
        if error > GATES["nrmse"]:
            raise ValueError("encoder NRMSE exceeds 0.04")
    if check["exact_reference_rows"] != check["rows"]:
        raise ValueError("greedy tokens differ from FP32 reference")
    if "compare_to" in case:
        base = [canonical_tokens(row, blank, pad, vocab) for row in a.actual[case["compare_to"] + "__tokens"]]
        for row, source in enumerate(case["compare_rows"]):
            if canonical[row] != base[source]:
                raise ValueError("batch/order/revisit semantics changed for row " + str(row))
    # real_time_factor needs the validated p50, so the timing is checked here (run_assess repeats it).
    a.timing(check, case)
    audio_seconds = float(a.inputs[name + "__audio_seconds"].sum())
    check.update(audio_seconds=audio_seconds,
                 real_time_factor=audio_seconds / check["p50_seconds"] if audio_seconds else None)


def quality(a):
    """Full stage: WER against the human transcripts, relative to the FP32 oracle's transcripts."""
    blank, pad, vocab = (a.selected["architecture"][k] for k in ("blank_token_id", "pad_token_id", "vocab_size"))
    labels, quality = a.labels(), {}
    reference_transcripts = {c["name"]: c["transcripts"] for c in a.reference["cases"]}
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.selected["repo_id"], revision=a.selected["revision"],
                                                  trust_remote_code=False, local_files_only=True)
    except (OSError, ValueError, ImportError) as exc:
        a.failures.append("labels: tokenizer unavailable on assessor (" + str(exc)[:120] + ")")
        return quality
    for case in a.spec["cases"]:
        if not case.get("quality"):
            continue
        name = case["name"]
        try:
            rows = [canonical_tokens(r, blank, pad, vocab) for r in a.actual[name + "__tokens"]]
            hyp = tokenizer.batch_decode(rows, skip_special_tokens=True)
        except (ValueError, TypeError) as exc:
            a.failures.append(name + ": " + str(exc))
            continue
        human = labels[name]
        oracle_wer = max(word_error_rate(o, h) for o, h in zip(reference_transcripts[name], human))
        candidate_wer = max(word_error_rate(y, h) for y, h in zip(hyp, human))
        if not math.isfinite(candidate_wer) or not math.isfinite(oracle_wer):
            a.failures.append(name + ": nonfinite WER")
            continue
        inflation = candidate_wer - oracle_wer
        quality[name] = dict(wer=candidate_wer, reference_wer=oracle_wer, wer_inflation=inflation)
        if inflation > GATES["max_wer_inflation"]:
            a.failures.append(name + f": WER inflation exceeds {GATES['max_wer_inflation']:.0%}")
    return quality


def assess(args):
    return portforge_eval.run_assess(args, sys.modules[__name__])


if __name__ == "__main__":
    portforge_eval.main(sys.modules[__name__])
