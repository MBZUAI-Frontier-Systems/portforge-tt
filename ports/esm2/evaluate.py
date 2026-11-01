"""ESM-2 trusted evaluator: protein masked-LM suites, FP32 oracle, TT measurement and gates.

The shared scaffold is portforge_eval's evaluator kit; this file holds what is specific to ESM-2.
Only input/ is exposed to the candidate: already-masked token ids, the attention mask and the
frozen masked-position list, never the raw sequences' recovery answers. The oracle runs the public
transformers EsmForMaskedLM in FP32 on the configured reference host (CUDA preferred, CPU recorded).
"""
import hashlib
import json
import sys

import portforge_eval
from portforge_eval import (Progress, reference_execution, suite_hash, validate_input_preservation,  # noqa: F401
                            validate_receipt, write_json)
from portforge_models import model_spec

KEY = "esm2"
MODEL = "facebook/esm2_t33_650M_UR50D"
REVISION = "08e4846e537177426273712802403f7ba8261b6c"
WEIGHT_SHA256 = "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"
CORPUS_SHA256 = "369c475fa78f1aad39963eee3fc43344c62be766793ed529e12fc1fdb4326a9e"
GATES = {"nrmse": .04, "behavior_max_nrmse": .01}
PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TIMING_PROTOCOL = "token_ids_to_logits_with_sync_excluding_load_tokenization_progress_v1"
MAX_RESIDUES = 1024
MASK_STRIDE, MASK_OFFSET = 16, 5  # deterministic masked positions: 5, 21, 37, ...


def model_identity(name=KEY):
    return portforge_eval.model_identity(name)


def suite_model(spec):
    return portforge_eval.suite_model(spec, KEY, GATES)


def validate_precision(mode, policy, runtime=None):
    return portforge_eval.validate_precision(mode, policy, runtime, PRECISIONS)


def masked_positions(length):
    return [p for p in range(MASK_OFFSET, length, MASK_STRIDE)]


def make_suite(corpus, stage, model="esm2"):
    selected, model_fields = model_spec(model), model_identity(model)
    identity = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if identity != CORPUS_SHA256 or stage not in portforge_eval.STAGES:
        raise ValueError("unrecognized corpus or stage")
    entries = {entry["name"]: entry for entry in corpus["cases"]}
    cases = []
    def add(name, keys, **extra):
        cases.append(dict(name=name, sequences=list(keys), numerical=True, **extra))
    if stage == "smoke":
        add("short", [corpus["selection"]["smoke"]])
    elif stage in ("bringup", "perf"):
        pick = corpus["selection"]["bringup"]
        add("batch", [pick["batch0"], pick["batch1"]])
        for name, rows in (("single0", [0]), ("single1", [1]), ("reverse", [1, 0])):
            add(name, [pick["batch" + str(i)] for i in rows], compare_to="batch", compare_rows=rows)
        add("long", [pick["long"]])
        add("short", [pick["short"]])
        add("unknown-residues", [pick["unknown"]])
        add("single-residue", ["syn-single"])
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
        # batch4 rows 0-1 hold the same two sequences; the full stage has no "batch" case.
        add("revisit", batch4[:2], compare_to="batch4", compare_rows=[0, 1])
        add("single-residue", ["syn-single"])
    return dict(version=2, stage=stage, **model_fields,
                weight_sha256=selected["files"][selected["weight_files"][0]]["sha256"] if not selected["weight_index"] else None,
                corpus_sha256=CORPUS_SHA256, max_residues=MAX_RESIDUES, gates=GATES.copy(),
                mask_stride=MASK_STRIDE, mask_offset=MASK_OFFSET, vocab_size=selected["architecture"]["vocab_size"],
                timed_repeats=portforge_eval.stage_repeats(stage), **portforge_eval.perf_fields(stage), cases=cases), {}


def max_row_nrmse(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or actual.dtype.kind != "f" or expected.dtype.kind != "f":
        raise ValueError("shape/dtype check")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite values")
    axes = tuple(range(1, actual.ndim))
    error = np.sqrt(np.mean((actual - expected) ** 2, axis=axes)) / np.maximum(
        1e-12, np.sqrt(np.mean(expected ** 2, axis=axes)))
    if not np.isfinite(error).all():
        raise ValueError("nonfinite NRMSE")
    return float(np.max(error))


def reference(args):
    """Reference host: transformers EsmForMaskedLM logits and last hidden states (FP32 oracle or benchmark)."""
    import numpy as np
    run = portforge_eval.ReferenceRun(args, sys.modules[__name__])
    torch, device = run.torch()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[run.precision]
    directory = run.checkpoint()
    run.progress.mark("load_model", initializing=True)
    from transformers import EsmForMaskedLM, EsmTokenizerFast
    model = EsmForMaskedLM.from_pretrained(str(directory), dtype=dtype, local_files_only=True,
                                           trust_remote_code=False, weights_only=True,
                                           use_safetensors=True).eval().to(device)
    tokenizer = EsmTokenizerFast.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    run.loaded(parameter_dtypes=[str(dtype).split(".")[-1]])
    mask_id = run.selected["architecture"]["mask_token_id"]
    entries = {entry["name"]: entry for entry in run.corpus["cases"]} if run.corpus else None
    with torch.inference_mode():
        for case in run.cases():
            arrays = run.prepared(case)
            if arrays is None:
                seqs = [entries[key]["sequence"] for key in case["sequences"]]
                if any(len(s) > MAX_RESIDUES for s in seqs):
                    raise ValueError("sequence exceeds the frozen envelope")
                enc = tokenizer(seqs, padding=True, return_tensors="pt")
                ids = enc.input_ids.clone()
                # Published in the suite: the candidate sees which positions are masked, never their answers.
                case["masked_positions"] = [masked_positions(len(seq)) for seq in seqs]
                for row, positions in enumerate(case["masked_positions"]):
                    for p in positions:  # residue p sits at token index p+1 (cls at 0)
                        ids[row, p + 1] = mask_id
                arrays = dict(input_ids=ids.numpy().astype(np.int64), attention_mask=enc.attention_mask.numpy().astype(np.int64),
                              residues=np.array([len(s) for s in seqs], dtype=np.int64))
            ids, mask = torch.from_numpy(arrays["input_ids"]).long(), torch.from_numpy(arrays["attention_mask"]).long()
            run.progress.mark("forward", case["name"])
            output = model(input_ids=ids.to(device), attention_mask=mask.to(device), output_hidden_states=True)
            oracle = dict(logits=output.logits.float().cpu().numpy(), hidden=output.hidden_states[-1].float().cpu().numpy())
            run.time(case, lambda: model(input_ids=ids.to(device), attention_mask=mask.to(device)),
                     lambda output: {"logits": output.logits.float().cpu().numpy()})
            run.record(case, arrays, oracle)
    return run.finish()


# Candidate (TT sandbox): portforge_eval.run_candidate times forward_case.
FORBIDDEN_HOST_OPS = ("aten.embedding.", "aten.native_layer_norm.", "aten.layer_norm.", "aten._softmax.",
                      "aten.softmax.", "aten.gelu.", "aten.silu.", "aten.relu.", "aten.scaled_dot_product")
ROWS = "sequences"


def forward_case(backend, case, inputs, spec, config):
    import numpy as np
    supplied = {key: inputs[key].copy() for key in ("input_ids", "attention_mask")}
    result = backend.embed(supplied["input_ids"], supplied["attention_mask"])
    logits = np.asarray(result["logits"], dtype=np.float32)
    hidden = np.asarray(result["hidden"], dtype=np.float32)
    shape = supplied["input_ids"].shape
    if logits.shape != shape + (config["vocab_size"],):
        raise ValueError("logits must be [B,L,vocab]")
    if hidden.shape != shape + (config["hidden_size"],):
        raise ValueError("hidden must be [B,L,hidden]")
    return {"logits": logits, "hidden": hidden}, supplied


def candidate(args):
    return portforge_eval.run_candidate(args, sys.modules[__name__])


# Assessment (reference host): portforge_eval.run_assess calls check_case per case.
def output_keys(case, spec):
    return ("logits", "hidden")


def check_case(check, case, a):
    name = case["name"]
    logits, oracle = a.actual[name + "__logits"], a.expected[name + "__logits"]
    for key in ("logits", "hidden"):
        error = max_row_nrmse(a.actual[name + "__" + key].astype("float64"), a.expected[name + "__" + key].astype("float64"))
        check[key + "_max_row_nrmse"] = error
        if error > GATES["nrmse"]:
            raise ValueError(key + " NRMSE exceeds 0.04")
    check["rows"] = logits.shape[0]
    check["exact_masked_rows"] = sum(
        [int(logits[row, p + 1].argmax()) for p in positions] == [int(oracle[row, p + 1].argmax()) for p in positions]
        for row, positions in enumerate(case.get("masked_positions", [])))
    if check["exact_masked_rows"] != check["rows"]:
        raise ValueError("masked-position argmax differs from FP32 reference")
    if "compare_to" in case:
        base = a.actual[case["compare_to"] + "__hidden"].astype("float64")
        mine = a.actual[name + "__hidden"].astype("float64")
        for row, source in enumerate(case["compare_rows"]):
            if max_row_nrmse(mine[row][None], base[source][None]) > GATES["behavior_max_nrmse"]:
                raise ValueError("batch/order/revisit semantics changed for row " + str(row))
    check["residues"] = int(a.inputs[name + "__residues"].sum())
    return {"residues": check["residues"]}


def assess(args):
    return portforge_eval.run_assess(args, sys.modules[__name__])


if __name__ == "__main__":
    portforge_eval.main(sys.modules[__name__])
