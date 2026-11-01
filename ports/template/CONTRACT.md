# <Model> backend contract

<!-- Replace every TODO. Agents read this file first; it is the acceptance envelope.
     Rules shared by every pack (precision, stages, sandbox, deliverable) are in
     PORTING_RULES.md; write here only what is specific to this model. -->

Write `backend.py` from scratch for the pinned `TODO/hf-repo-id` (model key
`my-model`, license TODO). PORTING_RULES.md holds the rules every port follows;
this file adds the model's interface, envelope and gates.

```python
def create_backend(weights_path, config, device, *, precision="bf16"):
    # Return an object with the methods below. The harness owns the TTNN device.
    ...

class Backend:
    # Required instance declaration; update it for the requested precision.
    precision_policy = {"mode": "bf16", "weights": "bf16", "activations": "bf16",
                        "accumulation": "fp32", "exceptions": []}

    def TODO_method(self, TODO_inputs):
        # TODO: exact input arrays (dtype, shape, masks) and returned arrays,
        # matching forward_case() in evaluate.py. Never mutate inputs.
        ...
```

The oracle is TODO: the official implementation and entry point it runs, and
which precisions it exposes (FP32 always; BF16 for the diagnostic study when it
can).

## Envelope and gates

TODO: required envelope (batch sizes, lengths, horizons, languages, ...), and
what is explicitly out of scope.

TODO: the numerical gates from `GATES` in evaluate.py, stated in words, for
example: max-row NRMSE ≤0.04 against the FP32 oracle on every numerical case;
consistency cases (single rows, reversed batch, revisit) agree within ≤0.01;
repeated identical calls are bit-identical; task metric on the full stage.

TODO: the suites in words: what `smoke` and `bringup` cover, the perf batch
sweep, and what `full` adds. Corpus provenance: BENCHMARK.md.

## Objective

TODO: the deployment scenario that matters for this model (single_stream,
offline or ar_interactive) and any hard constraints, matching `optimization` in
port.json.

TODO for autoregressive models: expose `prefill(...)` and `decode_step(...)` in
addition to the end-to-end method so TTFT and inter-token latency can be timed
separately.
