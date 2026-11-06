"""Trusted evaluator regressions; no model weights, CUDA or TT devices required."""
# Run from the harness directory: python -m unittest discover -s ports/<pack>/tests
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK), str(PACK.parents[1] / "portforge_tt" / "remote")]
import portforge_models  # noqa: E402

portforge_models.use_registry(PACK / "models.json")
import json  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
from unittest.mock import patch  # noqa: E402
import zlib  # noqa: E402

import evaluate  # noqa: E402
import portforge_eval  # noqa: E402
try:
    import numpy as np
    import build_corpus
except ImportError:
    np = None


def fixture_corpus(seed=7):
    """The real builder's schema and pool sizes, with flat-colour 'photographs' and cycling labels."""
    flat = lambda value: np.full((build_corpus.CROP, build_corpus.CROP, 3), value % 256, dtype=np.uint8)  # noqa: E731
    dev = [build_corpus.case_entry(f"dev-{i:02d}", flat(37 * i + 1), i % 1000, "fixture") for i in range(48)]
    heldout = [build_corpus.case_entry(f"ho-{i:04d}", flat(11 * i + 3), i % 1000, "fixture") for i in range(1024)]
    synthetic = [build_corpus.case_entry(name, build_corpus.synthetic_pixels(name, seed), None, "synthetic")
                 for name in build_corpus.SYNTHETIC]
    return build_corpus.assemble_corpus(dev, heldout, synthetic, seed, "fixture")


def oracle_row(clip):
    """Oracle logits depend only on the image, with a 1e-3 top-1 margin so a swap flips argmax."""
    values = np.random.default_rng(zlib.crc32(clip.encode())).normal(size=1000).astype(np.float32)
    order = np.argsort(-values)
    values[order[0]] = values[order[1]] + np.float32(1e-3)
    return values


def swap_top2(rows):
    rows = rows.copy()
    for row in rows:
        first, second = np.argsort(-row)[:2]
        row[first], row[second] = row[second], row[first]
    return rows


@unittest.skipIf(np is None, "NumPy required")
class SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = fixture_corpus()
        cls.patch = patch.object(evaluate, "CORPUS_SHA256", evaluate.corpus_identity(cls.corpus))
        cls.patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()

    def suite(self, stage):
        return evaluate.make_suite(self.corpus, stage)

    def test_stages(self):
        sizes = {stage: len(self.suite(stage)[0]["cases"]) for stage in portforge_eval.STAGES}
        self.assertEqual(sizes, {"smoke": 1, "bringup": 9, "perf": 13, "full": 68})
        full = {c["name"]: c for c in self.suite("full")[0]["cases"]}
        self.assertEqual(sum(len(c["clips"]) for c in full.values() if c.get("quality")), 1024)
        self.assertEqual(self.suite("full")[0]["timed_repeats"], 3)
        perf = self.suite("perf")[0]
        self.assertEqual(perf["timed_repeats"], portforge_eval.PERF_REPEATS)
        self.assertIn("min_timed_seconds", perf)
        self.assertNotIn("min_timed_seconds", self.suite("bringup")[0])

    def test_only_sweep_and_stream_cases_carry_a_batch(self):
        # A "batch" field puts a case on the throughput curve and takes it out of the latency aggregate.
        cases = {c["name"]: c for c in self.suite("perf")[0]["cases"]}
        curve = {name: (c["batch"], c.get("stream")) for name, c in cases.items() if "batch" in c}
        self.assertEqual(curve, {"stream-b16x4": (16, 4), "sweep-b16": (16, None), "sweep-b32": (32, None),
                                 "stream-b16x16": (16, 16), "stream-b32x8": (32, 8)})
        for name, (batch, stream) in curve.items():
            self.assertEqual(len(cases[name]["clips"]), batch * (stream or 1))

    def test_compare_rows_point_at_the_same_images(self):
        for stage in ("bringup", "perf", "full"):
            cases = {c["name"]: c for c in self.suite(stage)[0]["cases"]}
            for case in cases.values():
                if "compare_to" in case:
                    base = cases[case["compare_to"]]["clips"]
                    with self.subTest(stage=stage, case=case["name"]):
                        self.assertEqual(case["clips"][:len(case["compare_rows"])],
                                         [base[row] for row in case["compare_rows"]])

    def test_streams_hold_distinct_consecutive_batches(self):
        for case in self.suite("perf")[0]["cases"]:
            if case.get("stream"):
                batches = [case["clips"][i:i + case["batch"]] for i in range(0, len(case["clips"]), case["batch"])]
                self.assertTrue(all(len(set(b)) == case["batch"] for b in batches))
                self.assertTrue(all(a != b for a, b in zip(batches, batches[1:])))

    def test_labels_follow_the_clips_and_held_out_stays_out_of_development(self):
        spec, labels = self.suite("full")
        entries = {e["name"]: e for e in self.corpus["cases"]}
        for case in spec["cases"]:
            self.assertEqual(labels[case["name"]], [entries[c]["label"] for c in case["clips"]])
        self.assertIn(None, labels["synthetic"])
        for stage in ("smoke", "bringup", "perf"):
            clips = {clip for case in self.suite(stage)[0]["cases"] for clip in case["clips"]}
            self.assertFalse({c for c in clips if c.startswith("ho-")}, stage)

    def test_corpus_identity_and_pools_are_enforced(self):
        changed = json.loads(json.dumps(self.corpus))
        changed["cases"][0]["label"] = 5
        with self.assertRaisesRegex(ValueError, "identity"):
            evaluate.make_suite(changed, "smoke")
        changed = json.loads(json.dumps(self.corpus))
        changed["selection"]["dev"].pop()
        with patch.object(evaluate, "CORPUS_SHA256", evaluate.corpus_identity(changed)), \
             self.assertRaisesRegex(ValueError, "pools"):
            evaluate.make_suite(changed, "smoke")
        with self.assertRaises(ValueError):
            evaluate.make_suite(self.corpus, "mesh")

    def test_identity_matches_the_cluster_identity(self):
        sys.path.insert(0, str(PACK.parents[1] / "portforge_tt" / "control"))
        from cluster import identity
        self.assertEqual(identity(self.corpus), evaluate.corpus_identity(self.corpus))

    def test_normalization_and_decoding(self):
        spec = self.suite("smoke")[0]
        pixels = np.zeros((2, 224, 224, 3), dtype=np.uint8)
        pixels[1] = 255
        values = evaluate.normalize(pixels, spec)
        self.assertEqual((values.shape, values.dtype), ((2, 3, 224, 224), np.float32))
        self.assertTrue(values.flags.c_contiguous)
        mean, std = np.array(spec["mean"], np.float32), np.array(spec["std"], np.float32)
        np.testing.assert_array_equal(values[0, :, 0, 0], (np.float32(0) - mean) / std)
        np.testing.assert_array_equal(values[1, :, 0, 0], (np.float32(1) - mean) / std)
        with self.assertRaises(ValueError):
            evaluate.normalize(pixels.astype(np.float32), spec)
        try:
            import PIL  # noqa: F401  (the reference host decodes; the controller need not)
        except ImportError:
            return
        entry = self.corpus["cases"][3]
        self.assertEqual(evaluate.decode_case(entry).shape, (224, 224, 3))
        with self.assertRaisesRegex(ValueError, "pixels_sha256"):
            evaluate.decode_case(dict(entry, pixels_sha256="0" * 64))

    def test_candidate_inputs_are_read_only_and_streams_are_split_in_order(self):
        spec = self.suite("perf")[0]
        case = next(c for c in spec["cases"] if c["name"] == "stream-b32x8")
        pixels = np.random.default_rng(0).integers(0, 255, (256, 224, 224, 3), dtype=np.uint8)
        given = evaluate.case_inputs({case["name"] + "__pixels": pixels}, case, spec)
        with self.assertRaises(ValueError):
            given["pixel_values"][0, 0, 0, 0] = 1.0
        seen = []

        class Backend:
            def classify_stream(self, batches):
                seen.extend(batches)
                return {"logits": np.concatenate([b.reshape(len(b), -1)[:, :1000] for b in batches])}
        outputs, supplied = evaluate.forward_case(Backend(), case, given, spec, {})
        self.assertEqual([b.shape[0] for b in seen], [32] * 8)
        np.testing.assert_array_equal(np.concatenate(seen), given["pixel_values"])
        self.assertEqual(outputs["logits"].shape, (256, 1000))
        portforge_eval.validate_input_preservation(given, supplied)

        class Wrong:
            def classify(self, values):
                return {"logits": np.zeros((values.shape[0], 999), np.float32)}
        smoke = self.suite("smoke")[0]
        with self.assertRaisesRegex(ValueError, "num_classes"):
            evaluate.forward_case(Wrong(), smoke["cases"][0], {"pixel_values": np.zeros((16, 3, 224, 224), np.float32)},
                                  smoke, {})

    def test_host_guard_covers_the_network_ops(self):
        for op in ("aten.convolution.default", "aten.max_pool2d_with_indices.default", "aten.relu_.default",
                   "aten.add.Tensor", "aten.mean.dim", "aten._native_batch_norm_legit_no_training.default",
                   "aten._adaptive_avg_pool2d.default"):
            self.assertTrue(op.startswith(evaluate.FORBIDDEN_HOST_OPS), op)
        self.assertFalse("aten._to_copy.default".startswith(evaluate.FORBIDDEN_HOST_OPS))
        self.assertFalse("aten.addmm.default".startswith(evaluate.FORBIDDEN_HOST_OPS))  # weight-tainted only


@unittest.skipIf(np is None, "NumPy required")
class QualityTests(unittest.TestCase):
    def summary(self, candidate_correct, agree, images=1000):
        return dict(images=images, oracle_top1_correct=760, candidate_top1_correct=candidate_correct,
                    top1_agreement_count=agree)

    def test_gate_boundaries_are_exact(self):
        self.assertEqual(evaluate.quality_failures(self.summary(750, 950)), [])
        self.assertEqual(len(evaluate.quality_failures(self.summary(749, 950))), 1)
        self.assertEqual(len(evaluate.quality_failures(self.summary(750, 949))), 1)

    def test_counts_by_hand(self):
        oracle = np.eye(4, 1000, dtype=np.float32)
        candidate = oracle.copy()
        candidate[3] = np.eye(1, 1000, 7)[0]
        candidate[3, 3] = .5  # second place: top-5 agreement but not top-1
        summary = evaluate.quality_summary(candidate, oracle, [0, 1, 9, 3])
        self.assertEqual((summary["oracle_top1_correct"], summary["candidate_top1_correct"]), (3, 2))
        self.assertEqual((summary["top1_agreement_count"], summary["top5_agreement_count"]), (3, 4))
        self.assertAlmostEqual(summary["top1_drop_pp"], 25.0)
        with self.assertRaises(ValueError):
            evaluate.quality_summary(candidate, oracle, [0, 1, 2])
        with self.assertRaises(ValueError):
            evaluate.quality_summary(candidate, oracle, [0, 1, 2, 1000])


@unittest.skipIf(np is None, "NumPy required")
class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = types.SimpleNamespace(input=root / "input", oracle=root / "oracle", output=root / "output")
        for path in vars(self.args).values():
            path.mkdir()
        self.corpus = fixture_corpus()
        digest = patch.object(evaluate, "CORPUS_SHA256", evaluate.corpus_identity(self.corpus))
        digest.start()
        self.addCleanup(digest.stop)

    def write(self, stage, edit=None):
        """Oracle, labels and a candidate equal to the oracle (then edited), as the kit writes them."""
        spec, labels = evaluate.make_suite(self.corpus, stage)
        identity = {k: spec[k] for k in ("model_key", "model", "revision", "model_manifest_sha256")}
        selected = portforge_models.model_spec(evaluate.KEY)
        receipt = dict(model=evaluate.KEY, repo_id=spec["model"], revision=spec["revision"],
                       checkpoint_format=selected["checkpoint_format"],
                       model_manifest_sha256=spec["model_manifest_sha256"], suite_sha256=portforge_eval.suite_hash(spec),
                       verified_files={k: selected["files"][k] for k in selected["files"]})
        expected = {c["name"] + "__logits": np.stack([oracle_row(clip) for clip in c["clips"]]) for c in spec["cases"]}
        actual = {key: value.copy() for key, value in expected.items()}
        if edit:
            edit(actual)
        timing = dict(timing_protocol=evaluate.TIMING_PROTOCOL, **identity,
                      suite_sha256=portforge_eval.suite_hash(spec))
        samples = lambda: [0.01] * spec["timed_repeats"]  # noqa: E731
        portforge_eval.write_json(self.args.input / "suite.json", spec)
        portforge_eval.write_json(self.args.input / "config.json", selected["architecture"])
        portforge_eval.write_json(self.args.input / "checkpoint_receipt.json", receipt)
        portforge_eval.write_json(self.args.oracle / "labels.json", labels)
        portforge_eval.write_json(self.args.oracle / "reference.json", dict(
            timing, checkpoint_receipt=receipt, precision={"requested": "fp32"}))
        portforge_eval.write_json(self.args.output / "measurements.json", dict(
            timing, verified_weight_files={k: selected["files"][k] for k in selected["weight_files"]},
            cases=[dict(name=c["name"], seconds=samples(), first_seconds=.5) for c in spec["cases"]]))
        np.savez(self.args.oracle / "oracle.npz", **expected)
        np.savez(self.args.output / "actual.npz", **actual)
        return evaluate.assess(self.args)

    def test_identical_candidate_passes_every_stage(self):
        for stage in ("smoke", "bringup", "perf", "full"):
            with self.subTest(stage=stage):
                verdict = self.write(stage)
                self.assertTrue(verdict["passed"], verdict["failures"])
                self.assertEqual(verdict["accepted"], stage == "full")
        self.assertEqual(verdict["quality"]["images"], 1024)
        self.assertEqual(verdict["quality"]["top1_agreement"], 1.0)

    def test_throughput_curve_counts_every_streamed_image(self):
        verdict = self.write("perf")
        curve = {row["case"]: row for row in verdict["scaling"]}
        self.assertEqual(curve["stream-b32x8"]["items"], 256)
        self.assertAlmostEqual(curve["stream-b32x8"]["items_per_second"], 25600)
        self.assertAlmostEqual(curve["sweep-b16"]["items_per_second"], 1600)
        self.assertEqual(verdict["latency"]["measured_cases"], 8)  # the per-call bringup cases only

    def test_nrmse_gate(self):
        def noisy(actual):
            row = actual["batch16-b__logits"][3]
            row += np.float32(.31) * np.sqrt(np.mean(row ** 2))
        verdict = self.write("bringup", noisy)
        self.assertFalse(verdict["passed"])
        self.assertTrue(any(f.startswith("batch16-b: logits NRMSE") for f in verdict["failures"]), verdict["failures"])

    def test_behaviour_gate_on_a_stream_that_drifts_from_its_single_call(self):
        def drift(actual):
            actual["stream-b16x4__logits"][2] *= np.float32(1.06)
        verdict = self.write("bringup", drift)
        self.assertTrue(any(f.startswith("stream-b16x4: batch/order") for f in verdict["failures"]), verdict["failures"])

    def test_top1_gate_counts_labelled_cases_only(self):
        def flip(rows):
            def edit(actual):
                for key in ("batch16-c__logits", "synthetic__logits"):
                    actual[key][:rows] = swap_top2(actual[key][:rows])
            return edit
        self.assertTrue(self.write("bringup", flip(2))["passed"])  # 14/16 holds; synthetic is not gated
        verdict = self.write("bringup", flip(3))
        self.assertEqual([f.split(":")[0] for f in verdict["failures"]], ["batch16-c"])

    def test_held_out_agreement_gate(self):
        def flip(count):
            def edit(actual):
                for index in range(count // 16):
                    key = f"ho-batch-{index + 10:02d}__logits"
                    actual[key] = swap_top2(actual[key])
            return edit
        self.assertTrue(self.write("full", flip(48))["passed"])
        verdict = self.write("full", flip(64))  # 960/1024 < 0.95 agreement, and the drop exceeds 1 pp
        self.assertFalse(verdict["passed"])
        self.assertTrue(any("agreement 960/1024" in f for f in verdict["failures"]), verdict["failures"])

    def test_missing_labels_fail_the_full_stage(self):
        verdict = self.write("full")
        (self.args.oracle / "labels.json").unlink()
        verdict = evaluate.assess(self.args)
        self.assertFalse(verdict["passed"])


if __name__ == "__main__":
    unittest.main()
