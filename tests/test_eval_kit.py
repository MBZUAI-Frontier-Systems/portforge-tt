"""The shared evaluator kit (portforge_eval.run_candidate / run_assess) with a fake TTNN and backend."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import support  # noqa: F401  (pack and registry setup)
try:
    import numpy as np
    import torch  # loaded once: run_candidate imports it inside a patched sys.modules
except ImportError:
    np = torch = None

import portforge_eval

KEY = "parakeet"
BACKEND = '''
import time
import ttnn

class Backend:
    precision_policy = dict(mode="bf16", weights="bf16", activations="bf16", accumulation="fp32 dest", exceptions=[])

    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, 0

    def double(self, x):
        self.calls += 1
        time.sleep(0.02)  # the timed span must include the call itself, not only the sync
        if self.behaviour != "no-tt":
            ttnn.matmul(x)
        if self.behaviour == "mutate":
            x += 1
        if self.behaviour == "drift":
            return x * 2 + self.calls
        return x * 2

def create_backend(weights, config, device, precision):
    return Backend(open("behaviour").read())
'''


def forward_case(backend, case, inputs, spec, config):
    supplied = {"x": inputs["x"].copy()}
    return {"y": backend.double(supplied["x"])}, supplied


def check_case(check, case, a):
    error = float(np.max(np.abs(a.actual[case["name"] + "__y"] - a.expected[case["name"] + "__y"])))
    check["y_max_row_nrmse"] = error
    if error > a.spec["gates"]["nrmse"]:
        raise ValueError("y differs")
    return {"rows": len(case["clips"])}


PACK = types.SimpleNamespace(KEY=KEY, GATES={"nrmse": .04}, PRECISIONS=("bf16", "fp32"), TIMING_PROTOCOL="kit-test",
                             FORBIDDEN_HOST_OPS=("aten.gelu.",), ROWS="clips", forward_case=forward_case,
                             output_keys=lambda case, spec: ("y",), check_case=check_case)


def fake_ttnn(calls):
    ttnn = types.ModuleType("ttnn")
    ttnn.bfloat16 = object()
    ttnn.matmul = lambda x: calls.append(x)
    ttnn.synchronize_device = lambda device: None
    ttnn.open_device = lambda device_id, **options: "device"
    ttnn.close_device = lambda device: None
    ttnn.BufferType = types.SimpleNamespace(DRAM=0)
    ttnn.device = types.SimpleNamespace(get_memory_view=lambda device, kind: types.SimpleNamespace(
        num_banks=2, total_bytes_allocated_per_bank=10))
    return ttnn


@unittest.skipIf(torch is None, "NumPy and PyTorch required")
class EvaluatorKitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.public, self.oracle, self.output, self.work = (root / n for n in ("input", "oracle", "output", "work"))
        for folder in (self.public, self.oracle, self.work):
            folder.mkdir()
        model, identity = portforge_eval.model_spec(KEY), portforge_eval.model_identity(KEY)
        self.spec = dict(version=2, stage="smoke", **identity, gates=PACK.GATES, timed_repeats=3,
                         cases=[dict(name="pair", clips=["a", "b"], numerical=True)])
        receipt = dict(model=KEY, repo_id=model["repo_id"], revision=model["revision"],
                       checkpoint_format=model["checkpoint_format"], verified_files=model["files"],
                       model_manifest_sha256=identity["model_manifest_sha256"],
                       suite_sha256=portforge_eval.suite_hash(self.spec))
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        portforge_eval.write_json(self.public / "suite.json", self.spec)
        portforge_eval.write_json(self.public / "config.json", model["architecture"])
        portforge_eval.write_json(self.public / "checkpoint_receipt.json", receipt)
        np.savez(self.public / "inputs.npz", pair__x=x)
        np.savez(self.oracle / "oracle.npz", pair__y=x * 2)
        portforge_eval.write_json(self.oracle / "reference.json", dict(
            **identity, suite_sha256=portforge_eval.suite_hash(self.spec), timing_protocol="kit-test",
            checkpoint_receipt=receipt, precision={"requested": "fp32"}))
        (self.work / "backend.py").write_text(BACKEND)
        self.calls = []
        self.verified = {name: model["files"][name] for name in model["weight_files"]}

    def tearDown(self):
        sys.modules.pop("backend", None)
        self.temp.cleanup()

    def candidate(self, behaviour="good"):
        (self.work / "behaviour").write_text(behaviour)
        sys.modules.pop("backend", None)
        args = types.SimpleNamespace(input=str(self.public), output=str(self.output), weights="w", precision="bf16")
        cwd, path = os.getcwd(), list(sys.path)
        try:
            os.chdir(self.work)
            with patch.dict(sys.modules, {"ttnn": fake_ttnn(self.calls)}), \
                 patch.object(portforge_eval, "verify_weights", return_value=self.verified), \
                 patch.object(portforge_eval, "smallest_weight_matrix", return_value=4):
                portforge_eval.run_candidate(args, PACK)
        finally:
            os.chdir(cwd)
            sys.path[:] = path

    def assess(self):
        args = types.SimpleNamespace(input=str(self.public), oracle=str(self.oracle), output=str(self.output))
        return portforge_eval.run_assess(args, PACK)

    def test_candidate_times_the_call_and_assessment_passes(self):
        self.candidate()
        measured = json.loads((self.output / "measurements.json").read_text())
        case = measured["cases"][0]
        self.assertEqual(len(case["seconds"]), 3)
        self.assertTrue(all(s >= .02 for s in case["seconds"] + [case["first_seconds"]]), case)
        self.assertEqual(case["items"], 2)
        self.assertEqual(measured["observed_tt_calls"], 4)
        self.assertEqual(measured["resident_dram_bytes"], 20)
        verdict = self.assess()
        self.assertTrue(verdict["passed"], verdict["failures"])
        check = verdict["checks"][0]
        self.assertEqual(check["rows_per_second"], 2 / check["p50_seconds"])
        self.assertEqual(verdict["max_nrmse"], 0.0)
        self.assertTrue((self.output / "verdict.json").is_file())

    def test_candidate_rejects_mutation_drift_and_host_only_compute(self):
        for behaviour, message in (("mutate", "mutated caller input"), ("drift", "nondeterminism"),
                                   ("no-tt", "missing TT compute")):
            with self.subTest(behaviour=behaviour), self.assertRaisesRegex(ValueError, message):
                self.candidate(behaviour)

    def test_assessment_fails_wrong_outputs_and_identity(self):
        self.candidate()
        np.savez(self.oracle / "oracle.npz", pair__y=np.zeros((2, 3), dtype=np.float32))
        self.assertIn("pair: y differs", self.assess()["failures"])
        measured = json.loads((self.output / "measurements.json").read_text())
        portforge_eval.write_json(self.output / "measurements.json", dict(measured, timing_protocol="other"))
        self.assertTrue(any(f.startswith("identity:") for f in self.assess()["failures"]))


    def test_reference_run_writes_public_inputs_and_private_oracle(self):
        model = portforge_eval.model_spec(KEY)
        pack = types.SimpleNamespace(**vars(PACK), make_suite=lambda corpus, stage, name: (self.spec, {"pair": ["ab"]}))
        out = Path(self.temp.name) / "reference"
        args = types.SimpleNamespace(command="reference", stage="smoke", model=KEY, corpus=str(self.public / "suite.json"),
                                     out=str(out))
        with patch.dict(os.environ, {"PORTFORGE_REFERENCE_EXECUTION": "direct"}), \
             patch("socket.gethostname", return_value="gpu-host"):
            run = portforge_eval.ReferenceRun(args, pack)
            torch_module, device = run.torch()
            run.receipt = dict(verified_files=model["files"])
            run.publish_config(json.dumps(model["architecture"]))
            run.loaded(parameter_dtypes=["float32"])
            for case in run.cases():
                x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
                outputs = run.time(case, lambda: x * 2, lambda y: {"y": y.numpy()})
                run.record(case, {"x": x.numpy()}, outputs)
            run.finish()
        self.assertEqual(device, "cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(np.load(out / "oracle/oracle.npz")["pair__y"].tolist(), [[0, 2, 4], [6, 8, 10]])
        self.assertEqual(sorted(np.load(out / "input/inputs.npz").files), ["pair__x"])
        meta = json.loads((out / "oracle/reference.json").read_text())
        self.assertEqual(len(meta["cases"][0]["seconds"]), 3)
        self.assertEqual(meta["precision"]["requested"], "fp32")
        self.assertEqual(json.loads((out / "oracle/labels.json").read_text()), {"pair": ["ab"]})
        self.assertEqual(json.loads((out / "input/suite.json").read_text()), self.spec)


class ProfileCaseTests(unittest.TestCase):
    SPEC = dict(stage="perf", timed_repeats=100, min_timed_seconds=20.0, max_timed_seconds=120.0, min_timed_repeats=10,
                cases=[dict(name="batch", numerical=True), dict(name="sweep-b1024", numerical=True, batch=1024)])

    def test_profile_keeps_one_perf_case_with_a_few_repeats(self):
        with patch.dict(os.environ, {"PORTFORGE_PROFILE": "1", "PORTFORGE_PROFILE_CASE": "sweep-b1024"}):
            spec = portforge_eval.profile_subset(self.SPEC)
        self.assertEqual([c["name"] for c in spec["cases"]], ["sweep-b1024"])
        self.assertEqual(spec["timed_repeats"], portforge_eval.PROFILE_REPEATS)
        self.assertNotIn("min_timed_seconds", spec)
        with patch.dict(os.environ, {"PORTFORGE_PROFILE": "1", "PORTFORGE_PROFILE_CASE": "sweep-b9"}), \
             self.assertRaisesRegex(ValueError, "not in the suite"):
            portforge_eval.profile_subset(self.SPEC)

    def test_evaluations_and_whole_suite_profiles_keep_the_suite(self):
        for env in ({}, {"PORTFORGE_PROFILE": "1"}, {"PORTFORGE_PROFILE_CASE": "sweep-b1024"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                self.assertIs(portforge_eval.profile_subset(self.SPEC), self.SPEC)

    def test_runner_accepts_a_perf_profile_only_with_one_safe_case(self):
        import runner
        good = ({"stage": "smoke"}, {"stage": "bringup"}, {"stage": "perf", "case": "sweep-b1024"})
        bad = ({"stage": "perf"}, {"stage": "smoke", "case": "batch"}, {"stage": "full"}, {"stage": "mesh", "case": "x"},
               {"stage": "perf", "case": "../x"}, {"stage": "perf", "case": "a b"}, {"stage": "perf", "case": 3})
        for request in good:
            self.assertTrue(runner.valid_profile(request), request)
        for request in bad:
            self.assertFalse(runner.valid_profile(request), request)


if __name__ == "__main__":
    unittest.main()


class ProfilerEntryTests(unittest.TestCase):
    def test_every_pack_command_line_runs_inside_another_main_module(self):
        # port_profile runs `python -m tracy /evaluate.py ...`: tracy execs the script with
        # __name__ == "__main__" while sys.modules["__main__"] is tracy itself.
        repo = Path(__file__).resolve().parents[1]
        script = ('import sys, types\n'
                  'import portforge_models; portforge_models.use_registry(sys.argv[2])\n'
                  'sys.modules["__main__"] = types.ModuleType("tracy_main")\n'
                  'sys.argv = [sys.argv[1], "--help"]\n'
                  'exec(compile(open(sys.argv[0]).read(), sys.argv[0], "exec"),'
                  ' {"__name__": "__main__", "__file__": sys.argv[0]})\n')
        for pack in sorted((repo / "ports").glob("*/evaluate.py")):
            with self.subTest(pack=pack.parent.name):
                done = subprocess.run([sys.executable, "-c", script, str(pack), str(pack.with_name("models.json"))],
                                      capture_output=True, text=True,
                                      env={**os.environ, "PYTHONPATH": str(repo / "portforge_tt" / "remote")})
                self.assertEqual(done.returncode, 0, done.stderr[-1500:])
                self.assertIn("candidate", done.stdout)
