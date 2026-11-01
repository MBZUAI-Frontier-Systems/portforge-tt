"""Trusted evaluator template: suites, FP32 oracle, TT measurement and independent gates.

Copy ports/template/ to ports/<your-model>/ and fill in every TODO. The shared scaffold
(receipts, timing, determinism, host-compute guard, mesh devices, verdicts, CLI) is
portforge_eval's evaluator kit; a pack writes only what is specific to its model:

  make_suite(corpus, stage, model) -> (spec, labels)   the frozen smoke/bringup/perf/full suites
  reference(args)      reference host: the official implementation's FP32 outputs (ReferenceRun)
  forward_case(...)    TT sandbox: one timed call of the agent's backend (run_candidate)
  check_case(...)      reference host: the gates for one case (run_assess), plus quality() on full

Closest working examples:
  float outputs, one forward pass     ports/esm2 (logits), ports/chronos2 (quantiles)
  token outputs, autoregressive loop  ports/nllb (translation), ports/parakeet (ASR), ports/pocket_tts (TTS)

Keep the candidate's view minimal: input/ holds only what the backend needs (never oracle
outputs or ground-truth labels). Learned compute must run on TT; the kit rejects CPU
matmul/attention on model weights while the backend runs.
"""

import portforge_eval
from portforge_eval import check_suite

KEY = "my-model"                     # registry key in models.json and port.json
MODEL = "TODO/hf-repo-id"            # must match models.json
REVISION = "TODO-40-char-commit-sha"
# After building corpus.json, from the harness directory:
#   python -c "import json, cluster; print(cluster.identity(json.load(open('ports/<name>/corpus.json'))))"
CORPUS_SHA256 = "0" * 64
GATES = {"nrmse": .04, "behavior_max_nrmse": .01}   # add task metrics (WER, chrF, ...) as needed
PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TIMING_PROTOCOL = "inputs_to_outputs_with_sync_excluding_load_preprocessing_progress_v1"


def model_identity(name=KEY):
    return portforge_eval.model_identity(name)


def suite_model(spec):
    return portforge_eval.suite_model(spec, KEY, GATES)


def validate_precision(mode, policy, runtime=None):
    return portforge_eval.validate_precision(mode, policy, runtime, PRECISIONS)


def make_suite(corpus, stage, model=KEY):
    """Frozen suites. smoke: 1 case; bringup: ~8 cases covering batching, masking and edge
    inputs; perf: the bringup cases plus a batch sweep; full: the whole held-out corpus plus
    consistency cases.

    Each case is a dict with a unique "name", its rows (e.g. "clips": keys into the corpus),
    "numerical": True when it is timed, optional "quality": True for task-metric cases,
    and optional "compare_to"/"compare_rows" for single-row / reversed / revisit
    consistency against another case of the same stage.
    """
    if corpus.get("schema_version") != 1:
        raise ValueError("unsupported corpus schema")
    identity = model_identity(model)
    cases, labels = [], {}

    def add(name, clips, numerical=True, **extra):
        cases.append(dict(name=name, clips=list(clips), numerical=numerical, **extra))
        labels[name] = [corpus["cases"][clip].get("label") for clip in clips]

    # TODO: build the stages from corpus["selection"], e.g.
    #   if stage == "smoke": add("short", [corpus["selection"]["smoke"]])
    #   elif stage in ("bringup", "perf"): add("batch", [...]); add("single0", [...], compare_to="batch", compare_rows=[0]) ...
    #       and for perf a sweep within the envelope:
    #       for n in portforge_eval.PERF_BATCHES: add(f"sweep-b{n}", [smoke_clip] * n, batch=n)
    #   else: one case per held-out clip with quality=True, then batch4 / batch4-reverse / revisit.
    raise NotImplementedError("make_suite: define the smoke, bringup, perf and full suites")
    spec = dict(version=2, stage=stage, **identity, corpus_sha256=CORPUS_SHA256, gates=GATES.copy(),  # noqa
                timed_repeats=portforge_eval.stage_repeats(stage), **portforge_eval.perf_fields(stage), cases=cases)
    return check_suite(spec), labels


def reference(args):
    """Reference host: run the official implementation in FP32 (see portforge_eval.ReferenceRun).

    ReferenceRun handles the oracle/benchmark modes, the checkpoint download and receipt, FP32
    torch (TF32 off for matmuls and cuDNN), progress, the timed repeats with a determinism
    check, and writes input/ (public) and oracle/ (private). ports/esm2/evaluate.py is the
    shortest complete example.
    """
    run = portforge_eval.ReferenceRun(args, globals())
    torch, device = run.torch()
    directory = run.checkpoint()
    model = None  # TODO: load the official implementation from directory onto device, in FP32
    run.loaded(parameter_dtypes=["float32"])
    with torch.inference_mode():
        for case in run.cases():
            inputs = run.prepared(case)
            if inputs is None:
                inputs = {}  # TODO: this case's public arrays, built from run.corpus
            outputs = run.time(case, lambda: model(inputs), lambda result: {})  # TODO: {name: numpy array}
            run.record(case, inputs, outputs)
    raise NotImplementedError("reference: load the model, build inputs and return its outputs")
    return run.finish()  # noqa


# Candidate (TT sandbox): portforge_eval.run_candidate times forward_case on every case.
# Learned ops that never run on the host CPU; host matmuls are allowed only without model weights
# (portforge_eval.HostComputeGuard), e.g. RoPE angles from positions.
FORBIDDEN_HOST_OPS = ("aten.embedding.", "aten.native_layer_norm.", "aten.layer_norm.", "aten._softmax.",
                      "aten.softmax.", "aten.gelu.", "aten.silu.", "aten.relu.", "aten.scaled_dot_product",
                      "aten.convolution.")
ROWS = "clips"   # the case field listing its rows (throughput items); None for one item per case


def forward_case(backend, case, inputs, spec, config):
    """TODO: call the agent's backend once and return (outputs, supplied).

    ``inputs`` holds this case's arrays from inputs.npz. Pass the backend copies and return them
    as ``supplied`` so mutation is detected; return every output array to compare with the oracle
    (e.g. {"logits": ...}) and validate output shapes here. Add probe_case() with the same
    signature for an untimed numerical call on numerical cases (see ports/nllb).
    """
    raise NotImplementedError("forward_case: call backend.<your method> and return its arrays")


def candidate(args):
    return portforge_eval.run_candidate(args, globals())


# Assessment (reference host): portforge_eval.run_assess loads both sides, checks identity and
# timing, calls check_case per case and quality() on the full stage, and writes the verdict.
def output_keys(case, spec):
    return ("logits",)  # TODO: the arrays forward_case (and probe_case) produce


def check_case(check, case, a):
    """TODO: the gates for one case: max per-row NRMSE <= GATES["nrmse"] against a.expected, exact
    tokens for decoders, compare_to consistency <= GATES["behavior_max_nrmse"]. Record numbers in
    ``check``, raise ValueError on a failed gate, and return {unit: count} for throughput rates."""
    raise NotImplementedError("check_case: implement the gates for this task")


def quality(a):
    """Optional: task metrics on the full stage's quality cases (a.labels() holds the ground truth);
    append to a.failures when a quality gate fails. Return {} when the task has none."""
    return {}


def assess(args):
    return portforge_eval.run_assess(args, globals())


if __name__ == "__main__":
    portforge_eval.main(globals())
