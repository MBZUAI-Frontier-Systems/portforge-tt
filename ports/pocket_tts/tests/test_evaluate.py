"""Trusted evaluator regressions; no model weights, CUDA or TT devices required."""
# Run from the harness directory: python -m unittest discover -s ports/pocket_tts/tests
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK), str(PACK.parents[1] / "portforge_tt")]
import models  # noqa: E402

models.use_registry(PACK / "models.json")


def shipped_corpus(test):
    """The pack's frozen corpus.json; packs with large corpora ship none (pass --corpus)."""
    path = PACK / "corpus.json"
    if not path.is_file():
        test.skipTest("this pack does not ship its corpus")
    return path


import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import evaluate
try:
    import numpy as np
except ImportError:
    np = None


def synthetic_corpus():
    names = [f"hs-{i:04d}" for i in range(12)]
    cases = [dict(name=n, text=f"WORDS FOR CASE {i} TODAY.", voice=["alba", "vera", "michael"][i % 3],
                  label=f"WORDS FOR CASE {i} TODAY.", source="fixture") for i, n in enumerate(names)]
    return {"schema_version": 1, "task": "fixture", "protocol": "fixture", "sources": [],
            "selection": {"smoke": names[0],
                          "bringup": dict(short=names[1], long=names[2], one_word=names[3],
                                          digits=names[4], punct=names[5], upper=names[6],
                                          maxlen=names[7], revisit=names[1]),
                          "full": names[8:] + names[:4] + ["full-revisit"]},
            "cases": cases + [dict(name="full-revisit", text="WORDS FOR CASE 8 TODAY.", voice="alba",
                                   label="WORDS FOR CASE 8 TODAY.", source="fixture")]}


class PrecisionTests(unittest.TestCase):
    def policy(self, mode="bf16", **changes):
        return dict(mode=mode, weights=mode, activations=mode, accumulation="fp32", exceptions=[], **changes)

    def test_modes_and_mixed_exceptions(self):
        runtime = types.SimpleNamespace(bfloat16=object(), float32=object(), bfloat8_b=object(), float16=object())
        for mode in ("bf16", "fp32", "fp16", "bfp8_b"):
            self.assertEqual(evaluate.validate_precision(mode, self.policy(mode), runtime)["requested"], mode)
        self.assertIn("not IEEE FP8", evaluate.validate_precision("bfp8_b", self.policy("bfp8_b"))["format_note"])
        policy = self.policy("bfp8_b")
        policy["activations"] = "bf16"
        with self.assertRaises(ValueError):
            evaluate.validate_precision("bfp8_b", policy)
        policy["exceptions"] = ["flow head stays BF16 to bound AR drift"]
        evaluate.validate_precision("bfp8_b", policy)
        with self.assertRaises(ValueError):
            evaluate.validate_precision("bf16", None)

    def test_execution_modes(self):
        with patch("evaluate.socket.gethostname", return_value="dedicated-gpu"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                evaluate.reference_execution()
            with patch.dict(os.environ, {"PORTFORGE_REFERENCE_EXECUTION": "direct"}):
                self.assertEqual(evaluate.reference_execution(), "direct")
        with patch.dict(os.environ, {"PORTFORGE_REFERENCE_EXECUTION": "direct"}), \
             patch("evaluate.socket.gethostname", return_value="login-node"), self.assertRaises(RuntimeError):
            evaluate.reference_execution()


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.corpus = synthetic_corpus()

    def suite(self, stage):
        digest = hashlib.sha256(json.dumps(self.corpus, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        with patch.object(evaluate, "CORPUS_SHA256", digest):
            return evaluate.make_suite(self.corpus, stage, "pocket_tts")

    def test_stages_and_gates(self):
        for stage, count in (("smoke", 1), ("bringup", 8)):
            spec, labels = self.suite(stage)
            self.assertEqual(len(spec["cases"]), count)
            self.assertEqual(spec["gates"], evaluate.GATES)
            self.assertEqual(spec["model"], "kyutai/pocket-tts")
            self.assertEqual(spec["timed_repeats"], 1)
            self.assertTrue(all(c["anchors"] == [4, 12, 32] for c in spec["cases"]))
        spec, labels = self.suite("full")
        self.assertEqual(len(spec["cases"]), 9)
        self.assertEqual(spec["timed_repeats"], 3)
        self.assertTrue(all(c["anchors"] == [] for c in spec["cases"]))
        quality = [c for c in spec["cases"] if c.get("quality")]
        self.assertGreaterEqual(len(quality), 8)
        for case in spec["cases"]:
            self.assertEqual(labels[case["name"]], self.corpus["cases"][
                [i for i, c in enumerate(self.corpus["cases"]) if c["name"] == case["clip"]][0]]["label"])
        cases = {c["name"]: c for c in self.suite("bringup")[0]["cases"]}
        self.assertEqual(cases["revisit"]["compare_to"], "short")
        self.assertEqual(cases["revisit"]["compare_rows"], [0])
        self.assertTrue(all(c["numerical"] for c in spec["cases"]))

    def test_frozen_corpus_file_and_hash_rejection(self):
        path = shipped_corpus(self)
        corpus = json.loads(path.read_text())
        digest = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        self.assertEqual(digest, evaluate.CORPUS_SHA256)
        evaluate.make_suite(corpus, "smoke", "pocket_tts")
        corpus["cases"][0]["text"] = "tampered."
        with self.assertRaises(ValueError):
            evaluate.make_suite(corpus, "smoke", "pocket_tts")

    def test_case_noise_deterministic_and_scaled(self):
        rows = evaluate.case_noise("some-case", evaluate.CORPUS_SHA256, 64)
        again = evaluate.case_noise("some-case", evaluate.CORPUS_SHA256, 64)
        self.assertTrue((rows == again).all())
        self.assertAlmostEqual(float(rows.std()), evaluate.NOISE_STD, delta=0.01)
        self.assertEqual(rows.shape, (64, evaluate.LATENT_DIM))
        other = evaluate.case_noise("other-case", evaluate.CORPUS_SHA256, 64)
        self.assertFalse((rows == other).all())

    def test_max_frames_formula(self):
        self.assertEqual(evaluate.max_frames_for(10), 67)   # ceil((10/3 + 2) * 12.5)
        self.assertEqual(evaluate.max_frames_for(50), 234)

    def test_wer_and_normalization(self):
        self.assertEqual(evaluate.word_error_rate("hello world.", "Hello, world"), 0.0)
        self.assertAlmostEqual(evaluate.word_error_rate("hello there", "hello world"), 0.5)
        self.assertEqual(evaluate.word_error_rate("", "a b"), 1.0)
        self.assertEqual(evaluate.normalize_words("It's 2026!"), ["IT'S", "2026"])

    @unittest.skipIf(np is None, "NumPy required")
    def test_row_nrmse(self):
        b = np.ones((3, 32), dtype=np.float32)
        self.assertEqual(evaluate.max_row_nrmse(b.copy(), b), 0.0)
        bad = b.copy()
        bad[2] += 10.0
        self.assertGreater(evaluate.max_row_nrmse(bad, b), 0.04)
        with self.assertRaises(ValueError):
            evaluate.max_row_nrmse(np.full((3, 32), np.nan), b)

    @unittest.skipIf(np is None, "NumPy required")
    def test_audio_mel_nrmse_matches_identical_audio(self):
        rng = np.random.default_rng(0)
        audio = (np.sin(np.linspace(0, 200 * np.pi, 24000)) * 0.3).astype(np.float32)
        self.assertEqual(evaluate.audio_mel_nrmse(audio, audio), 0.0)
        noisy = audio + rng.normal(0, 0.05, len(audio)).astype(np.float32)
        self.assertGreater(evaluate.audio_mel_nrmse(noisy, audio), 0.0)
        with self.assertRaises(ValueError):
            evaluate.audio_mel_nrmse(np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32))

    def test_voice_state_export_import_round_trip(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch required")
        state = {}
        for i in range(evaluate.TRANSFORMER_LAYERS):
            state[f"transformer.layers.{i}.self_attn"] = {
                "cache": torch.arange(2 * 127 * 16 * 64, dtype=torch.float32).reshape(2, 1, 127, 16, 64),
                "offset": torch.tensor([127 * i], dtype=torch.int64),
                "pad": torch.tensor([0], dtype=torch.int64),
            }
        arrays = evaluate.export_voice_state(state)
        rebuilt = evaluate.import_voice_state(arrays, "cpu")
        self.assertEqual(sorted(rebuilt), sorted(state))
        for module in state:
            for field, tensor in state[module].items():
                self.assertTrue(torch.equal(rebuilt[module][field], tensor), module + "/" + field)
        with self.assertRaises(ValueError):
            evaluate.import_voice_state({}, "cpu")


@unittest.skipIf(np is None, "NumPy required")
class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = types.SimpleNamespace(input=self.root / "input", oracle=self.root / "oracle",
                                          output=self.root / "output")
        for path in vars(self.args).values():
            path.mkdir()
        self.cases = [dict(name="short", clip="c0", voice="alba", numerical=True, anchors=[4, 12, 32]),
                      dict(name="revisit", clip="c1", voice="vera", numerical=True, anchors=[4, 12, 32],
                           compare_to="short", compare_rows=[0])]
        self.stage = "bringup"
        self.strict_fixture()

    def strict_fixture(self, anchors=(4,)):
        model = evaluate.model_spec("pocket_tts")
        identity = evaluate.model_identity("pocket_tts")
        spec = dict(version=2, stage=self.stage, cases=copy.deepcopy(self.cases), gates=evaluate.GATES.copy(),
                    timed_repeats=1, sample_rate=evaluate.SAMPLE_RATE, frame_rate=evaluate.FRAME_RATE,
                    latent_dim=evaluate.LATENT_DIM, voices=list(evaluate.VOICES),
                    anchor_frames=list(evaluate.ANCHOR_FRAMES), **identity)
        receipt = dict(model="pocket_tts", repo_id=model["repo_id"], revision=model["revision"],
                       checkpoint_format=model["checkpoint_format"], verified_files=copy.deepcopy(model["files"]),
                       model_manifest_sha256=identity["model_manifest_sha256"], suite_sha256=evaluate.suite_hash(spec))
        self.spec, self.identity = spec, identity
        self.expected, self.actual, self.entries, self.inputs = {}, {}, [], {}
        frames, dim = 40, evaluate.LATENT_DIM
        for case in self.cases:
            name = case["name"]
            latents = np.tile(np.linspace(-1, 1, dim, dtype=np.float32), (frames, 1))
            audio = (np.sin(np.linspace(0, 100 * np.pi, frames * 1920)) * 0.2).astype(np.float32)
            self.expected[name + "__latents"] = latents
            self.expected[name + "__eos_frame"] = np.int64(20)
            self.expected[name + "__audio"] = audio
            self.actual[name + "__latents"] = latents.copy()
            self.actual[name + "__eos_frame"] = np.int64(20)
            self.actual[name + "__audio"] = audio.copy()
            self.entries.append(dict(name=name, seconds=[.5], first_seconds=.6))
            self.inputs[name + "__text_tokens"] = np.arange(8, dtype=np.int64)
            self.inputs[name + "__anchor_4"] = latents[:4].copy()
        base = dict(**identity, suite_sha256=evaluate.suite_hash(spec), timing_protocol=evaluate.TIMING_PROTOCOL,
                    verified_weight_files=model["files"], checkpoint_receipt=receipt, precision={"requested": "fp32"})
        evaluate.write_json(self.args.input / "suite.json", spec)
        evaluate.write_json(self.args.input / "config.json", model["architecture"])
        evaluate.write_json(self.args.input / "checkpoint_receipt.json", receipt)
        evaluate.write_json(self.args.output / "measurements.json", {**base, "cases": self.entries})
        evaluate.write_json(self.args.oracle / "reference.json",
                            {**base, "cases": [dict(name=c["name"], audio_seconds=3.0) for c in self.cases]})
        evaluate.write_json(self.args.oracle / "labels.json", {c["name"]: "WORDS FOR CASE ZERO" for c in self.cases})
        np.savez(self.args.input / "inputs.npz", **self.inputs)
        np.savez(self.args.output / "actual.npz", **self.actual)
        np.savez(self.args.oracle / "oracle.npz", **self.expected)
        for case in self.cases:
            self.actual[case["name"] + "__anchor_4"] = self.expected[case["name"] + "__latents"][4].copy()
        np.savez(self.args.output / "actual.npz", **self.actual)
        return spec

    def test_passing_bringup_not_accepted(self):
        result = evaluate.assess(self.args)
        self.assertTrue(result["passed"], result["failures"])
        self.assertFalse(result["accepted"])
        check = result["checks"][0]
        self.assertEqual(check["frames"], 40)
        self.assertEqual(check["eos_frame"], 20)
        self.assertAlmostEqual(check["real_time_factor"], 40 * 1920 / 24000 / 0.5)

    def test_eos_runaway_fails(self):
        self.actual["short__eos_frame"] = np.int64(-1)
        np.savez(self.args.output / "actual.npz", **self.actual)
        self.assertFalse(evaluate.assess(self.args)["passed"])

    def test_eos_within_tolerance_passes(self):
        self.actual["short__eos_frame"] = np.int64(22)
        self.actual["revisit__eos_frame"] = np.int64(22)
        np.savez(self.args.output / "actual.npz", **self.actual)
        self.assertTrue(evaluate.assess(self.args)["passed"])

    def test_anchor_nrmse_gate(self):
        self.actual["short__anchor_4"] += 0.5
        np.savez(self.args.output / "actual.npz", **self.actual)
        result = evaluate.assess(self.args)
        self.assertFalse(result["passed"])
        self.assertTrue(any("anchor NRMSE" in f for f in result["failures"]))

    def test_free_run_nrmse_gate(self):
        self.actual["short__latents"][30:] += 2.0
        np.savez(self.args.output / "actual.npz", **self.actual)
        result = evaluate.assess(self.args)
        self.assertFalse(result["passed"])
        self.assertTrue(any("free-run" in f for f in result["failures"]))

    def test_revisit_drift_fails(self):
        self.actual["revisit__latents"][8:] += 1.0
        np.savez(self.args.output / "actual.npz", **self.actual)
        result = evaluate.assess(self.args)
        self.assertFalse(result["passed"])
        self.assertTrue(any("revisit" in f for f in result["failures"]))

    def test_identity_tampering_fails(self):
        path = self.args.output / "measurements.json"
        original = json.loads(path.read_text())
        for field, value in (("suite_sha256", "0" * 64), ("model_key", "other"), ("timing_protocol", "x")):
            evaluate.write_json(path, {**original, field: value})
            self.assertFalse(evaluate.assess(self.args)["passed"])
            evaluate.write_json(path, original)

    def test_full_wer_inflation_gate(self):
        self.stage = "full"
        self.cases = [dict(name="w0", clip="c0", voice="alba", numerical=True, quality=True),
                      dict(name="revisit", clip="c1", voice="vera", numerical=True, quality=True,
                           compare_to="w0", compare_rows=[0])]
        self.strict_fixture(anchors=())
        # the scoring ASR transcribes the oracle audio perfectly and the candidate with one wrong word
        asr_texts = ["WORDS FOR CASE ZERO", "WORDS WRONG CASE ZERO"]

        def fake_transcribe(audios):
            return [asr_texts[1] if i == 0 else asr_texts[0] for i in range(len(audios))]

        with patch.object(evaluate, "transcribe", fake_transcribe):
            result = evaluate.assess(self.args)
        self.assertFalse(result["passed"])
        self.assertTrue(any("WER inflation" in f for f in result["failures"]))
        self.assertEqual(len(result["quality"]), 1)


if __name__ == "__main__":
    unittest.main()
