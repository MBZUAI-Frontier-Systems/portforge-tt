"""portforge_tt.remote: parsers and source transformations, no hardware or containers."""
import json
from pathlib import Path
import tempfile
import textwrap
import unittest

from portforge_tt.remote import config, kernelsrc, xnnpack
from portforge_tt.verify.cvc5 import equiv

NEON = textwrap.dedent("""\
    #include <arm_neon.h>
    #include "src/xnnpack/vbinary.h"
    void xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16(size_t batch, const int8_t* a, const int8_t* b,
        int8_t* out, const struct xnn_qs8_add_minmax_params* restrict params) XNN_OOB_READS
    { vst1_s8(out, vld1_s8(a)); }
    """)
RVV = textwrap.dedent("""\
    #include <riscv_vector.h>
    struct xnn_qs8_add_minmax_params;
    #define XNN_OOB_READS
    struct xnn_f32_minmax_params {
      struct { float min; float max; } scalar;
    };
    void test_rvv(size_t batch, const int8_t* a, const int8_t* b, int8_t* out,
                  const struct xnn_qs8_add_minmax_params* restrict params)
    { size_t vl = __riscv_vsetvl_e8m2(batch); }
    """)
INC = textwrap.dedent("""\
    #if XNN_ARCH_ARM || XNN_ARCH_ARM64
    XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16, 16, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
    #endif
    #if XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)
    XNN_UKERNEL(xnn_arch_riscv_vector, xnn_qs8_vadd_minmax_ukernel__rvv_u2v, 2, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
    #endif
    """)


class KernelSourceTests(unittest.TestCase):
    def test_strips_only_file_scope_xnn_declarations(self):
        text, removed = kernelsrc.strip_local_xnn_types(RVV)
        self.assertEqual(removed, ["struct xnn_qs8_add_minmax_params", "struct xnn_f32_minmax_params",
                                   "#define XNN_OOB_READS"])
        self.assertNotIn("float min", text)
        self.assertIn("const struct xnn_qs8_add_minmax_params* restrict params", text)
        self.assertIn("__riscv_vsetvl_e8m2", text)

    def test_configuration_macros_survive(self):
        text, removed = kernelsrc.strip_local_xnn_types("#define XNN_ARCH_ARM64 1\n#define XNN_OOB_READS\nvoid f(void);\n")
        self.assertIn("#define XNN_ARCH_ARM64 1", text)
        self.assertEqual(removed, ["#define XNN_OOB_READS"])

    def test_prepare_renames_and_drops_includes(self):
        text, removed = equiv.prepare(NEON, "neon")
        self.assertIn("void test_neon(", text)
        self.assertNotIn("#include", text)
        self.assertEqual(removed, [])
        text, removed = equiv.prepare(RVV, "rvv")
        self.assertIn("void test_rvv(", text)
        self.assertEqual(len(removed), 3)

    def test_prepare_rejects_files_without_a_kernel(self):
        with self.assertRaises(ValueError):
            equiv.prepare("int x;", "rvv")


class EquivLogTests(unittest.TestCase):
    PARTIAL = textwrap.dedent("""\
        14:04:51 [INFO]   config=a_zero_point=0,shift=0 — 120s budget spent at batch=62; verified up to batch=61
        14:04:51 [INFO]   config=a_zero_point=0,shift=0: verified up to batch=61
        14:04:51 [INFO]   config=default — 120s budget spent at batch=1; verified up to batch=0
        14:04:51 [INFO] PARTIAL: qs8-vadd-minmax — 61 batches verified, max=61 (120.1s)
        {
          "kernel": "qs8-vadd-minmax",
          "verdict": "PARTIAL",
          "max_verified_batch": 61,
          "num_verified": 61
        }
        """)
    COUNTER = textwrap.dedent("""\
        14:27:30 [ERROR] FAILED: qs8-vadd-minmax — counterexample at batch=1 config=shift=30,output_zero_point=127
        {
          "verdict": "COUNTEREXAMPLE",
          "counterexample": {"status": "COUNTEREXAMPLE", "batch": 1, "fail_index": 0}
        }
        """)

    def test_partial(self):
        result, configs, failed, seconds = equiv.parse_log(self.PARTIAL)
        self.assertEqual(result["verdict"], "PARTIAL")
        self.assertEqual(result["max_verified_batch"], 61)
        self.assertEqual(configs, [{"config": "a_zero_point=0,shift=0", "verified_up_to": 61}])
        self.assertEqual(seconds, 120.1)
        self.assertEqual(failed, "")

    def test_counterexample(self):
        result, _, failed, _ = equiv.parse_log(self.COUNTER)
        self.assertEqual(result["verdict"], "COUNTEREXAMPLE")
        self.assertEqual(failed, "shift=30,output_zero_point=127")

    def test_missing_intrinsics_are_not_kernel_errors(self):
        log = ("x.cpp:118:14: error: ‘__riscv_vwadd_wv_i16m4_tu’ was not declared in this scope\n"
               "x.cpp:792:9: error: ‘vfloat32m1x6_t’ was not declared in this scope\n"
               "x.cpp:824:9: error: ‘v_px’ was not declared in this scope\n"
               '{\n "verdict": "COMPILE_ERROR"\n}\n')
        result, _, _, _ = equiv.parse_log(log)
        self.assertEqual(result["verdict"], "MISSING_INTRINSIC")
        self.assertEqual(result["missing_intrinsics"], ["__riscv_vwadd_wv_i16m4_tu", "vfloat32m1x6_t"])

    def test_no_json_is_no_result(self):
        result, *_ = equiv.parse_log("Traceback (most recent call last):\nboom\n")
        self.assertEqual(result["verdict"], "NO_RESULT")


class XnnpackRegistryTests(unittest.TestCase):
    def test_injected_block_stays_inside_file_local_macros(self):
        inc = ("#ifndef XNN_QUANTIZED\n#define XNN_QUANTIZED(T) T\n#endif\n"
               "#if XNN_ARCH_ARM\nXNN_UKERNEL(xnn_arch_arm_neon, xnn_t, 8, false, XNN_QUANTIZED(int8_t))\n#endif\n"
               "XNN_UKERNEL(xnn_arch_none, xnn_s, 1, false, XNN_QUANTIZED(int8_t))\n"
               "#ifdef XNN_DEFINED_QUANTIZED\n#undef XNN_QUANTIZED\n#endif\n")
        out = xnnpack.insert_registry_lines(inc, "xnn_t", ["XNN_UKERNEL(xnn_arch_riscv_vector, xnn_new, 8)"])
        self.assertLess(out.index("xnn_new"), out.index("#undef XNN_QUANTIZED"))
        self.assertGreater(out.index("xnn_new"), out.index("#define XNN_QUANTIZED"))

    def test_registry_lines_and_clone(self):
        entries = xnnpack.registry_lines(INC)
        self.assertEqual([a[1] for _, a, _ in entries],
                         ["xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16", "xnn_qs8_vadd_minmax_ukernel__rvv_u2v"])
        line = xnnpack.clone_registry_line(INC, "xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16", "xnn_new")
        self.assertTrue(line.startswith("XNN_UKERNEL(xnn_arch_riscv_vector, xnn_new, 16, false, int8_t,"))
        with self.assertRaises(ValueError):
            xnnpack.clone_registry_line(INC, "xnn_missing", "x")
        tiled = xnnpack.clone_registry_line(INC, "xnn_qs8_vadd_minmax_ukernel__rvv_u2v", "xnn_v", overrides={2: "4 * vl"})
        self.assertTrue(tiled.startswith("XNN_UKERNEL(xnn_arch_riscv_vector, xnn_v, 4 * vl, false,"))

    def test_split_args_keeps_nested_commas(self):
        self.assertEqual(xnnpack._split_args("a, f(b, c), d"), ["a", " f(b, c)", " d"])

    def test_kernel_source_takes_template_includes_minus_arm(self):
        text = xnnpack.kernel_source(RVV, "xnn_new", NEON)
        self.assertIn("#include <riscv_vector.h>", text)
        self.assertIn('#include "src/xnnpack/vbinary.h"', text)
        self.assertNotIn("arm_neon", text)
        self.assertIn("void xnn_new(", text)
        self.assertNotIn("float min", text)  # local struct copy removed

    def test_inject_writes_source_registry_and_cmake(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp)
            (tree / "src/qs8-vadd/gen").mkdir(parents=True)
            (tree / "cmake/gen").mkdir(parents=True)
            (tree / "src/qs8-vadd/qs8-vadd-minmax.inc").write_text(INC)
            (tree / "src/qs8-vadd/gen/neon.c").write_text(NEON)
            (tree / xnnpack.CMAKE_LIST).write_text("SET(PROD_RVV_MICROKERNEL_SRCS\n  a.c)\n\n"
                                                    "SET(NON_PROD_RVV_MICROKERNEL_SRCS\n  b.c)\n")
            kernel = tree / "k.c"
            kernel.write_text(RVV)
            added = xnnpack.inject(tree, [xnnpack.Injection(
                source=kernel, inc="src/qs8-vadd/qs8-vadd-minmax.inc",
                template="xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16", name="xnn_qs8_vadd_minmax_ukernel__rvv_x")])
            self.assertEqual(added, ["src/qs8-vadd/portforge/xnn_qs8_vadd_minmax_ukernel__rvv_x.c"])
            inc = (tree / "src/qs8-vadd/qs8-vadd-minmax.inc").read_text()
            self.assertIn(xnnpack.RVV_GUARD + "\nXNN_UKERNEL(xnn_arch_riscv_vector, xnn_qs8_vadd_minmax_ukernel__rvv_x, 16,", inc)
            self.assertLess(inc.index("rvv_x"), inc.index("#if XNN_ARCH_ARM"))  # before the template's section
            cmake = (tree / xnnpack.CMAKE_LIST).read_text()
            self.assertIn("SET(NON_PROD_RVV_MICROKERNEL_SRCS\n  src/qs8-vadd/portforge/", cmake)
            self.assertIn("void xnn_qs8_vadd_minmax_ukernel__rvv_x(", (tree / added[0]).read_text())

    def test_targets_for_scans_includes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp)
            (tree / "test").mkdir()
            (tree / "bench").mkdir()
            (tree / "test/qs8-vadd-minmax.cc").write_text('#include "src/qs8-vadd/qs8-vadd-minmax.inc"\n')
            (tree / "bench/vbinary.cc").write_text('#include "src/qs8-vadd/qs8-vadd-minmax.inc"\n')
            (tree / "bench/other.cc").write_text('#include "src/x/y.inc"\n')
            self.assertEqual(xnnpack.targets_for(tree, "src/qs8-vadd/qs8-vadd-minmax.inc"),
                             {"test": ["qs8-vadd-minmax-test"], "bench": ["vbinary-bench"]})


class ReportParserTests(unittest.TestCase):
    def test_gtest(self):
        report = {"testsuites": [{"name": "xnn_k", "testsuite": [
            {"name": "batch_eq", "result": "COMPLETED"},
            {"name": "qmin", "result": "COMPLETED", "failures": [{"failure": "Value of: y\n  Actual: 1"}]},
            {"name": "skipped", "result": "SKIPPED"}]}]}
        status, messages = xnnpack.parse_gtest(report)
        self.assertEqual(status, {"xnn_k.batch_eq": "pass", "xnn_k.qmin": "fail", "xnn_k.skipped": "skip"})
        self.assertIn("Actual: 1", messages["xnn_k.qmin"])

    def test_bench_aggregates(self):
        report = {"benchmarks": [
            {"name": "vbinary/xnn_a/N:8192/real_time_mean", "run_name": "vbinary/xnn_a/N:8192/real_time",
             "run_type": "aggregate", "aggregate_name": "mean", "real_time": 9.0, "time_unit": "us"},
            {"name": "vbinary/xnn_a/N:8192/real_time_median", "run_name": "vbinary/xnn_a/N:8192/real_time",
             "run_type": "aggregate", "aggregate_name": "median", "real_time": 8.0, "time_unit": "us"},
            {"name": "vbinary/xnn_a/N:8192/real_time_cv", "run_name": "vbinary/xnn_a/N:8192/real_time",
             "run_type": "aggregate", "aggregate_name": "cv", "real_time": 0.01, "time_unit": "us"},
            {"name": "vbinary/xnn_a/N:8192/real_time", "run_type": "iteration", "real_time": 1.0}]}
        self.assertEqual(xnnpack.parse_bench(report), {"vbinary/xnn_a/N:8192/real_time": {"ns": 8000.0, "cv": 0.01}})


class BenchCompareTests(unittest.TestCase):
    def test_names(self):
        self.assertEqual(xnnpack.ukernel_name("xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16"),
                         "xnn_qs8_vadd_minmax_ukernel__rvv_saltyrn")
        self.assertEqual(xnnpack.ukernel_name("xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld64", "rvv_pf"),
                         "xnn_f32_gemm_minmax_ukernel_4x8__rvv_pf")
        self.assertEqual(xnnpack.family_token("xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16"), "qs8_vadd_minmax")

    def test_compare_bench_uses_shapes_present_for_both(self):
        rows = {
            "vbinary/xnn_qs8_vadd_minmax_ukernel__rvv_saltyrn/N:8192/real_time": {"ns": 50.0},
            "vbinary/xnn_qs8_vadd_minmax_ukernel__rvv_saltyrn/N:65536/real_time": {"ns": 400.0},
            "vbinary/xnn_qs8_vadd_minmax_ukernel__rvv_u1v/N:8192/real_time": {"ns": 100.0},
            "vbinary/xnn_qs8_vadd_minmax_ukernel__rvv_u2v/N:8192/real_time": {"ns": 90.0},
            "vbinary/xnn_qs8_vadd_minmax_ukernel__rvv_u2v/N:65536/real_time": {"ns": 720.0},
            "vbinary/xnn_qs8_vaddc_minmax_ukernel__rvv_u2v/N:8192/real_time": {"ns": 1.0},
            "vbinary/xnn_qs8_vadd_minmaxc_ukernel__rvv_u2v/N:8192/real_time": {"ns": 1.0},  # prefix lookalike
        }
        cmp = xnnpack.compare_bench(rows, "xnn_qs8_vadd_minmax_ukernel__rvv_saltyrn", "qs8_vadd_minmax")
        self.assertEqual(cmp["shapes"], 2)
        self.assertAlmostEqual(cmp["speedup_vs_best"], (1.8 * 1.8) ** 0.5)
        self.assertAlmostEqual(cmp["vs"]["xnn_qs8_vadd_minmax_ukernel__rvv_u1v"], 2.0)

    def test_research_registry_is_consistent(self):
        registry = json.loads((Path(__file__).resolve().parents[1]
                               / "ports/xnnpack-rvv/research/saltyrn_kernels.json").read_text())["kernels"]
        self.assertEqual(len(registry), 40)
        for name, entry in registry.items():
            with self.subTest(kernel=name):
                self.assertTrue(entry["neon_ukernel"].startswith("xnn_"))


class ConfigTests(unittest.TestCase):
    def test_defaults_and_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cpu.yaml"
            path.write_text(f"workdir: {tmp}\njobs: 4\nx280:\n  port: 2300\n  boot_python: /usr/bin/python3\n")
            cfg = config.load(path)
            self.assertEqual((cfg.jobs, cfg.x280.port, cfg.x280.user), (4, 2300, "debian"))
            self.assertEqual(cfg.src, Path(tmp) / "src")
            path.write_text("x280:\n  prot: 1\n")
            with self.assertRaises(ValueError):
                config.load(path)

    def test_sources_pins_are_full_commits(self):
        repos = json.loads((config.DATA / "sources.json").read_text())["repos"]
        for name, spec in repos.items():
            with self.subTest(repo=name):
                self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$")


class GtestGateTests(unittest.TestCase):
    """Gate 1 passes only on a full, crash-free run of exactly the listed tests."""

    def run_(self, **kw):
        return {"target": "t-test", "where": "qemu:128", "passed": 15, "failed": 0, **kw}

    def test_full_clean_runs_pass(self):
        self.assertEqual(xnnpack.test_verdict([self.run_(), self.run_(where="x280")], 15), (True, []))

    def test_crash_with_zero_failures_is_a_failure(self):
        ok, problems = xnnpack.test_verdict([self.run_(passed=0, error="no gtest report (crash or timeout)")], 15)
        self.assertFalse(ok)
        self.assertIn("crash", problems[0])

    def test_empty_filter_and_short_runs_fail(self):
        self.assertFalse(xnnpack.test_verdict([self.run_(passed=0)], 0)[0])
        ok, problems = xnnpack.test_verdict([self.run_(passed=9)], 15)
        self.assertFalse(ok)
        self.assertIn("ran 9 of 15", problems[0])

    def test_failures_fail(self):
        self.assertFalse(xnnpack.test_verdict([self.run_(passed=14, failed=1)], 15)[0])

    def test_list_tests_parses_gtest_listing(self):
        listing = ("Running main() from gtest_main.cc\n"
                   "xnn_k__rvv_x.\n  batch_eq\n  qmin  # GetParam() = 3\n"
                   "Param/Suite.\n  case/0\n")
        from unittest import mock
        with mock.patch.object(xnnpack.host, "run", return_value=mock.Mock(stdout=listing)):
            cfg = config.Config(workdir=Path("/tmp/pf"))
            names = xnnpack.list_tests(cfg, "tree", "t-test", "*x*")
        self.assertEqual(names, ["xnn_k__rvv_x.batch_eq", "xnn_k__rvv_x.qmin", "Param/Suite.case/0"])


class FamilyMetadataTests(unittest.TestCase):
    def tree(self, tmp):
        tree = Path(tmp)
        (tree / "src/qs8-vadd").mkdir(parents=True)
        (tree / "test").mkdir()
        (tree / "src/qs8-vadd/qs8-vadd-minmax.inc").write_text(INC)
        (tree / "test/qs8-vadd-minmax.cc").write_text(
            '#include "test/vbinary-microkernel-tester.h"\n#include "src/qs8-vadd/qs8-vadd-minmax.inc"\n')
        return tree

    def test_signature_and_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = self.tree(tmp)
            inc = "src/qs8-vadd/qs8-vadd-minmax.inc"
            self.assertEqual(xnnpack.signature_of(tree, inc), "vbinary")
            self.assertEqual(xnnpack.variants(tree, inc),
                             {"xnn_arch_arm_neon": ["xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16"],
                              "xnn_arch_riscv_vector": ["xnn_qs8_vadd_minmax_ukernel__rvv_u2v"]})
            self.assertIsNone(xnnpack.signature_of(tree, "src/other.inc"))

    def test_extract_function_keeps_preamble_and_one_variant(self):
        text = ('#include "x.h"\nstatic int helper;\n'
                "void xnn_k__neon_u4(size_t n) { a(); }\n"
                "void xnn_k__neon_u8(size_t n) { b(); }\n")
        one = kernelsrc.extract_function(text, "xnn_k__neon_u8")
        self.assertIn('#include "x.h"', one)
        self.assertIn("b();", one)
        self.assertNotIn("a();", one)
        with self.assertRaises(ValueError):
            kernelsrc.extract_function(text, "xnn_missing")


class RegimesTests(unittest.TestCase):
    SPEC_PATH = Path(__file__).resolve().parents[1] / "ports/xnnpack-rvv/kernels/qs8-vcvt/regimes.json"

    def test_qs8_vcvt_spec_is_valid_and_flattens(self):
        from portforge_tt.remote import regimes
        spec = regimes.load(self.SPEC_PATH)
        text = regimes.flat(spec, regimes.check_regimes(spec))
        lines = text.splitlines()
        self.assertEqual(len(lines), len(spec["regimes"]) + len(spec["check_extra"]))
        self.assertEqual(lines[0].split(), ["regime", "bench", "0.5", "1", "1.0", "0", "0.7", "-1",
                                            "0.0", "0.0", "full", "-1.0", "1.0"])
        self.assertEqual(len(regimes.cells(spec)), len(spec["regimes"]) * len(spec["bench_shapes"]))

    def test_heldout_draws_respect_accept_and_seed(self):
        from portforge_tt.remote import regimes
        spec = regimes.load(self.SPEC_PATH)
        a, b = regimes.draw_heldout(spec, 7), regimes.draw_heldout(spec, 7)
        self.assertEqual(a, b)
        self.assertNotEqual(a, regimes.draw_heldout(spec, 8))
        for r in a:
            self.assertTrue(1 / 256 <= r["a"]["scale"] / r["out"]["scale"] <= 128)
            self.assertIsInstance(r["a"]["zero_point"], int)

    def test_validation_errors(self):
        from portforge_tt.remote import regimes
        bad = {"family": "f", "reference": "r", "bench_shapes": [8], "check_batches": [1],
               "regimes": [{"id": "x"}, {"id": "x"}]}
        with self.assertRaises(ValueError):
            regimes.validate(bad)
        bad["regimes"] = [{"id": "x", "a": {"scal": 1}}]
        with self.assertRaises(ValueError):
            regimes.validate(bad)


class TransposeRegimesTests(unittest.TestCase):
    SPEC_PATH = Path(__file__).resolve().parents[1] / "ports/xnnpack-rvv/kernels/x32-transposec/regimes.json"

    def test_layout_lines_shapes_and_heldout(self):
        from portforge_tt.remote import regimes
        spec = regimes.load(self.SPEC_PATH)
        lines = regimes.flat(spec, regimes.check_regimes(spec)).splitlines()
        self.assertEqual(lines[2].split(), ["tregime", "op32-view", "32", "3", "5", "1"])
        self.assertEqual(len(regimes.check_sizes(spec)), 40 * 40 + len(spec["check_shapes"]["extra"]))
        self.assertEqual(regimes.cells(spec)[0], ("single", "32x32"))
        held = regimes.draw_heldout(spec, 3)
        self.assertEqual(held, regimes.draw_heldout(spec, 3))
        self.assertTrue(all(set(r) == {"id", "heldout", "tile", "in_pad", "out_pad", "offset"} for r in held))
        self.assertTrue(regimes.flat(spec, held).startswith("tregime heldout-0 "))

    def test_transpose_validation(self):
        from portforge_tt.remote import regimes
        spec = {"family": "f", "kind": "transpose", "reference": "r", "bench_shapes": ["8x8"],
                "check_shapes": ["1x1"], "regimes": [{"id": "x", "a": {"scale": 1}}]}
        with self.assertRaises(ValueError):
            regimes.validate(spec)
        spec["regimes"] = [{"id": "x", "tile": 32}]
        spec["bench_shapes"] = [64]
        with self.assertRaises(ValueError):
            regimes.validate(spec)
        spec["bench_shapes"] = ["64x7"]
        regimes.validate(spec)


class StatsTests(unittest.TestCase):
    def noisy(self, mean, n, seed, cv=0.01):
        import random
        rng = random.Random(seed)
        return [mean * (1 + rng.gauss(0, cv)) for _ in range(n)]

    def test_aa_is_neutral_and_shift_is_detected(self):
        from portforge_tt.remote import stats
        base = self.noisy(100.0, 15, 1)
        aa = stats.cell(base, self.noisy(100.0, 15, 2))
        self.assertLess(abs(aa["s"] - 1), 0.02)
        self.assertLessEqual(aa["lo"], 1.0)
        self.assertGreaterEqual(aa["hi"], 1.0)
        self.assertFalse(stats.flagged(aa, 0.01))
        slower = stats.cell(base, self.noisy(110.0, 15, 3))
        self.assertAlmostEqual(slower["s"], 1 / 1.1, delta=0.02)
        self.assertTrue(stats.flagged(slower, 0.01))

    def test_geomean_ci_and_improvement_rule(self):
        from portforge_tt.remote import stats
        paired = {"a": (self.noisy(100, 15, 1), self.noisy(80, 15, 2)),
                  "b": (self.noisy(50, 15, 3), self.noisy(40, 15, 4))}
        g = stats.geomean_ci(paired)
        self.assertAlmostEqual(g["g"], 1.25, delta=0.03)
        self.assertTrue(g["lo"] <= g["g"] <= g["hi"])
        self.assertTrue(stats.improves(g))

    def test_noise_floor_has_a_minimum(self):
        from portforge_tt.remote import stats
        self.assertEqual(stats.noise_floor([1.0, 1.0001]), stats.MIN_NOISE)
        self.assertAlmostEqual(stats.noise_floor([1.0, 1.02, 0.99, 1.01]), 0.02, delta=0.001)
        self.assertEqual(stats.summary([3, 1, 2])["p50"], 2)


class BenchAnalyzeTests(unittest.TestCase):
    def rows(self, cells):
        """cells: {(regime, shape): {slot: (ns, correct)}} -> pfharness-like rows."""
        out = []
        for (regime, shape), by_slot in cells.items():
            for rnd in range(9):
                for slot, (ns, _) in by_slot.items():
                    jitter = 1 + 0.002 * ((rnd * 7 + slot) % 5 - 2)
                    out.append({"regime": regime, "shape": shape, "round": rnd, "offset": 0, "slot": slot,
                                "kernel": f"k{slot}", "reps": 4, "ns": ns * jitter, "cycles": ns * 1.75 * jitter,
                                "instret": ns * jitter})
            for slot, (_, correct) in by_slot.items():
                out.append({"regime": regime, "shape": shape, "slot": slot, "kernel": f"k{slot}",
                            "hash": "aa" if correct else "bb", "reference_hash": "aa"})
        return out

    def test_fastest_correct_baseline_and_flags(self):
        from portforge_tt.remote import bench
        rows = self.rows({
            ("bench", 1000): {0: (100.0, True), 1: (90.0, True), 2: (60.0, True)},   # 1.5x over k1
            ("edge", 1000): {0: (100.0, False), 1: (50.0, False), 2: (80.0, True)},  # no correct upstream
            ("slow", 1000): {0: (100.0, True), 1: (100.0, True), 2: (120.0, True)},  # regression
        })
        v = bench.analyze(rows, ["k0", "k1", "k2"], 2, [0, 1], delta=0.01)
        self.assertEqual(v["cells"]["bench@1000"]["baseline"], "k1")
        self.assertAlmostEqual(v["cells"]["bench@1000"]["speedup"]["s"], 1.5, delta=0.01)
        self.assertEqual(v["correctness_wins"], ["edge@1000"])
        self.assertIn("excluded", v["cells"]["edge@1000"])
        self.assertEqual(v["flagged"], ["slow@1000"])
        self.assertFalse(v["feasible"])
        self.assertAlmostEqual(v["metrics"]["worst_cell"], 1 / 1.2, delta=0.01)
        self.assertAlmostEqual(v["metrics"]["speedup_geomean"], (1.5 / 1.2) ** 0.5, delta=0.02)

    def test_wrong_candidate_is_infeasible(self):
        from portforge_tt.remote import bench
        rows = self.rows({("bench", 64): {0: (100.0, True), 1: (10.0, False)}})
        v = bench.analyze(rows, ["k0", "k1"], 1, [0])
        self.assertEqual(v["wrong_cells"], ["bench@64"])
        self.assertFalse(v["feasible"])


class IdleParseTests(unittest.TestCase):
    TEXT = ("0.00 0.00 0.00 1/80 329\n"
            "cpu0 82 0 90 38950 38 0 10 0 0 0\ncpu1 216 0 112 38796 52 0 0 0 0 0\n"
            "cpu0 83 0 90 39051 38 0 10 0 0 0\ncpu1 266 0 112 38846 52 0 0 0 0 0\n")

    def test_busy_hart_is_not_idle(self):
        from portforge_tt.remote import x280
        v = x280.parse_idle(self.TEXT)
        self.assertAlmostEqual(v["busy"]["cpu0"], 1 / 102, places=3)
        self.assertAlmostEqual(v["busy"]["cpu1"], 0.5, places=3)
        self.assertFalse(v["ok"])
        quiet = self.TEXT.replace("cpu1 266 0 112 38846", "cpu1 216 0 112 38896")
        self.assertTrue(x280.parse_idle(quiet)["ok"])


class StaticProfileTests(unittest.TestCase):
    """profile.py on real objdump -dr output (qs8-vcvt pf2, gcc 14, rv64gcv)."""

    def setUp(self):
        from portforge_tt.remote import profile
        self.profile = profile
        text = (Path(__file__).parent / "fixtures" / "qs8-vcvt-pf2.dis").read_text()
        self.funcs = profile.parse(text)

    def test_loops_classes_and_calls(self):
        p = self.profile.function_profile(self.funcs["xnn_qs8_vcvt_ukernel__rvv_pf2"], 450)
        self.assertEqual(len(p["loops"]), 5)                  # one loop per parameter regime
        self.assertEqual(p["calls"], ["__ctzdi2"])            # libgcc ctz: no Zbb in -march=rv64gcv
        self.assertTrue(p["descriptors"]["dispatch"])
        self.assertEqual(p["descriptors"]["lmul"], 8)
        regime_a = min(p["loops"], key=lambda h: h["vector_ops"])
        self.assertEqual(regime_a["vector_ops"], 3)           # vle8, vssub, vse8
        self.assertEqual(regime_a["counts"]["fixed_point"], 1)
        self.assertEqual(regime_a["vsetvl"], ["e8,m4"])

    def test_classify(self):
        c = self.profile.classify
        self.assertEqual([c(m) for m in ("vle8.v", "vse8.v", "vsetvli", "vsadd.vx", "vnclip.wi", "vwsub.vx",
                                         "vmv4r.v", "vmul.vx", "vadd.vx", "vsll.vi", "addi")],
                         ["load", "store", "vsetvl", "fixed_point", "fixed_point", "widen",
                          "move", "mul", "alu", "alu", "scalar"])


class CostModelTests(unittest.TestCase):
    """The calibrated X280 model must reproduce measured cycles/element (bench@16320,
    calibration session 2026-10-28) within 8% for qs8-vcvt pf2's regime loops."""

    def test_pf2_loops_match_measurements(self):
        from portforge_tt.remote import profile
        calib = json.loads((config.DATA / "x280-calib.json").read_text())
        insns = profile.parse((Path(__file__).parent / "fixtures" / "qs8-vcvt-pf2.dis").read_text())[
            "xnn_qs8_vcvt_ukernel__rvv_pf2"]
        by_start = {hex(s): profile.predict(insns, s, e, calib) for s, e in profile.loops(insns)}
        measured = {"0x18c": 0.256, "0x110": 0.303 / 4.5, "0x66": 0.303}  # regime C, A, general chain
        for start, cpe in measured.items():
            with self.subTest(loop=start):
                self.assertAlmostEqual(by_start[start]["cycles_per_elem"] / cpe, 1.0, delta=0.08)
        self.assertEqual(by_start["0x110"]["bound"], "memory")

    def test_segment_forms_are_memory_and_costed_exactly(self):
        from portforge_tt.remote import profile
        self.assertEqual([profile.classify(m) for m in ("vlsseg8e32.v", "vssseg4e32.v", "vlseg2e32.v", "vsseg8e32.v")],
                         ["load", "store", "load", "store"])
        # the upstream transpose inner step: one strided segment load, eight unit stores
        insns = [(0, "vsetvli", "a5,zero,e8,mf4,ta,ma"), (4, "vlsseg8e32.v", "v8,(a0),a1")]  # GCC's idiom
        insns += [(8 + 4 * i, "vse32.v", f"v{8 + i},(a{i})") for i in range(8)]
        insns += [(40, "bnez", "a2,4 <x>")]
        calib = {"memops": {"vlsseg8e32.v": {"e32,m1": 32.0}, "vse32.v": {"e32,m1": 5.0}}}
        p = profile.predict(insns, 4, 40, calib, per="loads")
        self.assertEqual((p["memory_cycles"], p["elements_per_iteration"]), (72.0, 128.0))
        self.assertAlmostEqual(p["cycles_per_elem"], 72 / 128)

    def test_roofline_levels(self):
        from portforge_tt.remote import profile
        calib = {"memory": {"copy": {"L1": 32.0, "L2": 2.5, "DRAM": 0.67}}}
        r = profile.roofline(0.30, {"cycles_per_elem": 0.29}, 2, 4096, calib)
        self.assertEqual((r["level"], r["bound"]), ("L1", "compute"))
        self.assertAlmostEqual(r["efficiency"], 0.29 / 0.30)
        r = profile.roofline(4.0, {"cycles_per_elem": 0.29}, 2, 4194304, calib)
        self.assertEqual((r["level"], r["bound"]), ("DRAM", "memory"))


class CatalogueTests(unittest.TestCase):
    """RVV_TECHNIQUES.md is the catalogue: the controller parses its headings."""

    def test_catalogue_parses_and_family_ids_resolve(self):
        from portforge_tt.ranking import search
        known = search.catalogue()
        self.assertGreaterEqual(len(known), 25)
        self.assertTrue(all(item["category"] and item["title"] for item in known.values()))
        for path in sorted(Path(__file__).resolve().parents[1].glob("ports/*/kernels/*/TECHNIQUES.json")):
            if "template-kernel" in path.parts:
                continue
            with self.subTest(family=path.parent.name):
                doc = json.loads(path.read_text())
                for item in doc["candidates"]:
                    self.assertTrue(item["id"].startswith("k:") or item["id"] in known,
                                    f"{item['id']} is neither a catalogue id nor a k: family id")
                    self.assertTrue(item.get("application"), item["id"])


if __name__ == "__main__":
    unittest.main()
