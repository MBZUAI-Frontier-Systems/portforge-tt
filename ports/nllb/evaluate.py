"""NLLB-200 trusted evaluator: translation suites, CUDA FP32 oracle, TT measurement and gates.

The shared scaffold is portforge_eval's evaluator kit; this file holds what is specific to NLLB:
the suites, the transformers model as the oracle (encoder states, last-position logits and greedy
tokens), the backend's forward() and generate() calls, the token gates and chrF++ on full.
Only input/ is exposed to the candidate. Run reference/assess on GPU compute nodes;
candidate imports the agent's backend.py in its current working directory.
"""
import hashlib
import json
import math

import portforge_eval
from portforge_eval import (Progress, reference_execution, suite_hash, validate_config,  # noqa: F401
                            validate_input_preservation, validate_receipt, verify_weights, write_json)
from portforge_models import model_spec

MODEL = "facebook/nllb-200-distilled-600M"
REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
WEIGHT_SHA256 = "c266c2cfd19758b6d09c1fc31ecdf1e485509035f6b51dfe84f1ada83eefcc42"
CORPUS_SHA256 = "c9b03ca2525db184b6feaf341a52383ec1d9d74311682462c29ba036e74327d1"
LANGUAGES = ("fra_Latn", "uzn_Latn", "arb_Arab", "zho_Hans")
GATES = {"nrmse": .04, "max_chrf_loss": .5, "behavior_exact": True}
PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TIMING_PROTOCOL = "cpu_tokens_to_cpu_tokens_with_sync_excluding_load_tokenization_progress_v1"

KEY = "600m"


def model_identity(name=KEY):
    return portforge_eval.model_identity(name)


def suite_model(spec):
    return portforge_eval.suite_model(spec, KEY, GATES)


def validate_precision(mode, policy, runtime=None):
    return portforge_eval.validate_precision(mode, policy, runtime, PRECISIONS)


def make_suite(corpus, stage, model="600m"):
    selected, model_fields = model_spec(model), model_identity(model)
    identity = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if identity != CORPUS_SHA256 or stage not in portforge_eval.STAGES:
        raise ValueError("unrecognized corpus or stage")
    cases, labels = [], {}
    def add(name, texts, src="eng_Latn", tgt="fra_Latn", cap=8, **extra):
        cases.append(dict(name=name, texts=texts, src=src, tgt=tgt,
                          max_new_tokens=cap, numerical=True, **extra))
    if stage == "smoke":
        add("short", ["Hello world."], cap=4)
    elif stage in ("bringup", "perf"):
        add("batch", ["Hello world.", "Good morning, everyone."])
        for name, rows in (("single0", [0]), ("single1", [1]), ("reverse", [1, 0]), ("padding", [0, 1])):
            add(name, [cases[0]["texts"][i] for i in rows], compare_to="batch", compare_rows=rows,
                extra_padding=32 if name == "padding" else 0)
        for length in (32, 33):
            add(f"length-{length}", ["The students are testing translation. " * 100], token_length=length)
        add("unicode-empty", ["", "Hello 👋! Café — €50."])
        if stage == "perf":  # batch scaling sweep within the contract envelope
            for n in portforge_eval.PERF_BATCHES:
                add(f"sweep-b{n}", ["Good morning, everyone."] * n, batch=n)
    else:
        for src, tgt in [("eng_Latn", l) for l in LANGUAGES] + [(l, "eng_Latn") for l in LANGUAGES]:
            for offset in range(0, 1012, 4):
                name = f"{src}-{tgt}-{offset}"
                add(name, corpus[src][offset:offset + 4], src, tgt, cap=64, quality=True)
                cases[-1]["numerical"] = offset == 0
                labels[name] = corpus[tgt][offset:offset + 4]
            first = next(c for c in cases if c["name"] == f"{src}-{tgt}-0")
            for tag, rows in (("single0", [0]), ("single1", [1]), ("single2", [2]),
                              ("single3", [3]), ("pair", [0, 1]), ("triple", [0, 1, 2]),
                              ("reverse", [3, 2, 1, 0]), ("padding", [0, 1, 2, 3])):
                add(first["name"] + "-" + tag, [first["texts"][i] for i in rows], src, tgt,
                    cap=64, compare_to=first["name"], compare_rows=rows,
                    extra_padding=32 if tag == "padding" else 0)
        for length in (2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256):
            add(f"length-{length}", ["The students are testing translation. " * 100], token_length=length)
        add("unicode-empty", ["", " \n\t", "Hello 👋! Café — €50.", "<script>hello</script>"], cap=32)
        for cap in (2, 3, 31, 32, 33, 64):
            add(f"cap-{cap}", ["Hello world."], cap=cap)
        for first in list(cases):
            if first.get("quality") and first["numerical"]:
                add(first["name"] + "-revisit", first["texts"], first["src"], first["tgt"], cap=64,
                    compare_to=first["name"], compare_rows=[0, 1, 2, 3])
    return dict(version=2, stage=stage, **model_fields,
                weight_sha256=selected["files"][selected["weight_files"][0]]["sha256"] if not selected["weight_index"] else None,
                corpus_sha256=CORPUS_SHA256, max_source_tokens=256, gates=GATES.copy(),
                timed_repeats=portforge_eval.stage_repeats(stage), **portforge_eval.perf_fields(stage), cases=cases), labels


def canonical_tokens(row, target, cap, config=None):
    config = config or model_spec("600m")["architecture"]
    start, eos, pad, vocab = (config[k] for k in ("decoder_start_token_id", "eos_token_id", "pad_token_id", "vocab_size"))
    row = [int(x) for x in row]
    if len(row) < 2 or len(row) > cap + 1 or row[:2] != [start, target] or any(x < 0 or x >= vocab for x in row):
        raise ValueError("invalid token range/start/target/length")
    end = row.index(eos, 2) if eos in row[2:] else len(row)
    if pad in row[2:end] or any(x != pad for x in row[end + 1:]) or (end == len(row) and len(row) != cap + 1):
        raise ValueError("invalid EOS/padding/termination")
    return row[:end + 1]


def reference(args):
    """Reference host: transformers NLLB encoder states, last-position logits and greedy tokens
    (FP32 oracle or benchmark). CUDA only: CPU is not a reference compute substitute here."""
    reference_execution()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU is not a reference compute substitute")
    run = portforge_eval.ReferenceRun(args, globals(), precisions=("fp32", "fp16", "bf16"))
    torch, device = run.torch()
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[run.precision]
    directory = run.checkpoint()
    run.progress.mark("load_model", initializing=True)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(str(directory), dtype=dtype, local_files_only=True,
        attn_implementation="eager", trust_remote_code=False, weights_only=True, use_safetensors=False).eval().cuda()
    run.progress.mark("load_tokenizer", initializing=True)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False, local_files_only=True)
    # The staged config.json is the loaded configuration's serialization, not the official file (CONTRACT.md).
    validate_config(model.config.to_dict(), run.selected)
    if not run.benchmark:
        model.config.to_json_file(run.public / "config.json")
    run.loaded(parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}))
    architecture, spec = run.selected["architecture"], run.spec
    with torch.inference_mode():
        for case in run.cases():
            name, arrays = case["name"], run.prepared(case)
            if arrays is None:
                tokenizer.src_lang = case["src"]
                values = tokenizer(case["texts"], padding=True, truncation=True, max_length=spec["max_source_tokens"], return_tensors="pt")
                if "token_length" in case:
                    values = {k: v[:, :case["token_length"]].clone() for k, v in values.items()}
                    values["input_ids"][:, -1] = architecture["eos_token_id"]
                pad = min(case.get("extra_padding", 0), spec["max_source_tokens"] - values["input_ids"].shape[1])
                values = {k: torch.nn.functional.pad(v, (0, pad), value=architecture["pad_token_id"] if k == "input_ids" else 0) for k, v in values.items()}
                case["target_id"] = tokenizer.convert_tokens_to_ids(case["tgt"])
                prefix = torch.tensor([[architecture["decoder_start_token_id"], case["target_id"]]] * len(case["texts"]))
                arrays = {k: v.numpy() for k, v in {**values, "decoder_input_ids": prefix}.items()}
            values = {k: torch.from_numpy(arrays[k]) for k in ("input_ids", "attention_mask")}
            outputs = {}
            if case["numerical"]:
                run.progress.mark("forward", name)
                output = model(**{k: v.cuda() for k, v in values.items()},
                               decoder_input_ids=torch.from_numpy(arrays["decoder_input_ids"]).cuda(), use_cache=False)
                run.progress.mark("forward_readback", name)
                outputs["encoder"] = output.encoder_last_hidden_state.float().cpu().numpy()
                outputs["logits"] = output.logits[:, -1].float().cpu().numpy()
            # Timed: host-to-device copies, greedy generation and the token readback.
            outputs.update(run.time(case, lambda: model.generate(
                **{k: v.cuda() for k, v in values.items()}, forced_bos_token_id=case["target_id"],
                max_new_tokens=case["max_new_tokens"], do_sample=False, num_beams=1, use_cache=True).cpu(),
                lambda tokens: {"tokens": tokens.numpy()}))
            run.meta["cases"][-1]["translations"] = tokenizer.batch_decode(outputs["tokens"], skip_special_tokens=True)
            run.record(case, arrays, outputs)
    result = run.finish()
    if run.benchmark:
        from sacrebleu.metrics import CHRF
        hypotheses = [text for c in run.meta["cases"] for text in c["translations"]]
        original = json.loads((run.private / "reference.json").read_text())
        translations = [text for c in original["cases"] for text in c["translations"]]
        result["reference_agreement"] = dict(chrf_vs_fp32=CHRF(word_order=2).corpus_score(hypotheses, [translations]).score,
                                            scope="short-output agreement with FP32; not human-reference quality")
        write_json(run.root / "verdict.json", result)
    return result


# Candidate (TT sandbox): portforge_eval.run_candidate probes numerical cases, then times forward_case.
# Learned ops that never run on the host CPU. Host matmuls are allowed only without
# model weights (portforge_eval.HostComputeGuard), e.g. RoPE angles from positions.
FORBIDDEN_HOST_OPS = ("aten.embedding.", "aten.native_layer_norm.", "aten.layer_norm.", "aten._softmax.",
                      "aten.softmax.", "aten.gelu.", "aten.scaled_dot_product")
ROWS = "texts"


def probe_case(backend, case, inputs, spec, config):
    """Untimed teacher-forced forward: encoder states and the last decoder position's logits."""
    import numpy as np
    supplied = {key: inputs[key].copy() for key in ("input_ids", "attention_mask", "decoder_input_ids")}
    result = backend.forward(supplied["input_ids"], supplied["attention_mask"], supplied["decoder_input_ids"])
    logits = np.asarray(result["logits"])
    if logits.shape != (*supplied["decoder_input_ids"].shape, config["vocab_size"]):
        raise ValueError("forward logits must be [B,T,V]")
    return {"encoder": np.asarray(result["encoder"]), "logits": logits[:, -1]}, supplied


def forward_case(backend, case, inputs, spec, config):
    """Timed greedy generation: CPU token ids in, CPU token ids out."""
    import numpy as np
    supplied = {key: inputs[key].copy() for key in ("input_ids", "attention_mask")}
    tokens = np.asarray(backend.generate(supplied["input_ids"], supplied["attention_mask"],
                                         case["target_id"], case["max_new_tokens"]))
    if tokens.dtype.kind not in "iu":
        raise ValueError("invalid tokens")
    return {"tokens": tokens}, supplied


def candidate(args):
    return portforge_eval.run_candidate(args, globals())


# Assessment (reference host): portforge_eval.run_assess calls check_case per case, then quality.
def output_keys(case, spec):
    return ("tokens", "encoder", "logits") if case["numerical"] else ("tokens",)


def check_case(check, case, a):
    import numpy as np
    name, config = case["name"], a.selected["architecture"]
    canonical = a.notes.setdefault("canonical", {})  # cross-case: comparisons and full-stage decoding
    tokens = a.actual[name + "__tokens"]
    if tokens.ndim != 2 or tokens.shape[0] != len(case["texts"]) or tokens.dtype.kind not in "iu":
        raise ValueError("invalid tokens shape/dtype")
    canonical[name] = [canonical_tokens(r, case["target_id"], case["max_new_tokens"], config) for r in tokens]
    oracle = [canonical_tokens(r, case["target_id"], case["max_new_tokens"], config) for r in a.expected[name + "__tokens"]]
    if len(oracle) != len(tokens):
        raise ValueError("oracle token batch shape mismatch")
    check["exact_reference_rows"] = sum(x == y for x, y in zip(canonical[name], oracle))
    check["rows"] = len(tokens)
    if case["numerical"]:
        for key in ("encoder", "logits"):
            if a.actual[name + "__" + key].dtype.kind != "f":
                raise ValueError(key + " must contain floating-point values")
            mine, ref = (data[name + "__" + key].astype(np.float64) for data in (a.actual, a.expected))
            shape = ((len(tokens), a.inputs[name + "__input_ids"].shape[1], config["d_model"])
                     if key == "encoder" else (len(tokens), config["vocab_size"]))
            if mine.shape != shape or ref.shape != shape:
                raise ValueError(key + " dimensions disagree with pinned architecture/input")
            if not np.isfinite(mine).all() or not np.isfinite(ref).all():
                raise ValueError(key + " shape/finite check")
            axes = tuple(range(1, mine.ndim))
            error = float(np.max(np.sqrt(np.mean((mine - ref) ** 2, axis=axes)) / np.maximum(1e-12, np.sqrt(np.mean(ref ** 2, axis=axes)))))
            if not math.isfinite(error):
                raise ValueError(key + " nonfinite NRMSE")
            check[key + "_max_row_nrmse"] = error
            if error > GATES["nrmse"]:
                raise ValueError(key + " NRMSE exceeds 0.04")
    if "compare_to" in case and canonical[name] != [canonical[case["compare_to"]][i] for i in case["compare_rows"]]:
        raise ValueError("batch/order/padding/revisit semantics changed")
    check["generated_tokens"] = sum(len(row) - 1 for row in canonical[name])
    return {"tokens": check["generated_tokens"]}


def quality(a):
    """Full stage: chrF++ per direction against the human references, relative to the FP32 oracle's."""
    from transformers import AutoTokenizer
    from sacrebleu.metrics import CHRF
    selected, canonical, quality = a.selected, a.notes["canonical"], {}
    tokenizer = AutoTokenizer.from_pretrained(selected["repo_id"], revision=selected["revision"], trust_remote_code=False, local_files_only=True)
    labels = a.labels()
    reference = {c["name"]: c for c in a.reference["cases"]}
    for src, tgt in [("eng_Latn", l) for l in LANGUAGES] + [(l, "eng_Latn") for l in LANGUAGES]:
        subset = [c["name"] for c in a.spec["cases"] if c.get("quality") and (c["src"], c["tgt"]) == (src, tgt)]
        hyp, ref, human = [], [], []
        for name in subset:
            hyp.extend(tokenizer.batch_decode(canonical[name], skip_special_tokens=True))
            ref.extend(reference[name]["translations"])
            human.extend(labels.get(name, ()))
        metric = CHRF(word_order=2)
        if len(human) != 1012 or len(hyp) != len(human) or len(ref) != len(human):
            a.failures.append(src + "->" + tgt + ": incomplete full quality coverage")
            continue
        score, baseline = (metric.corpus_score(x, [human]).score for x in (hyp, ref))
        quality[src + "->" + tgt] = dict(chrf=score, cuda_chrf=baseline, rows=len(human))
        if not math.isfinite(score) or not math.isfinite(baseline) or baseline - score > GATES["max_chrf_loss"]:
            a.failures.append(src + "->" + tgt + ": chrF++ loss exceeds 0.5")
    return quality


def assess(args):
    return portforge_eval.run_assess(args, globals())


if __name__ == "__main__":
    portforge_eval.main(globals())
