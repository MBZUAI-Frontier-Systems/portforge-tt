"""portforge_tt.cpu: parsers and source transformations, no hardware or containers."""
import json
from pathlib import Path
import tempfile
import textwrap
import unittest

from portforge_tt.cpu import config, equiv, kernelsrc, xnnpack

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


class XdiffTests(unittest.TestCase):
    def test_kernel_unit_renames_and_swaps_headers(self):
        from portforge_tt.cpu import xdiff
        text = xdiff._kernel_unit(RVV, "rvv", "src/xnnpack/microparams.h")
        self.assertIn("void xdiff_kernel_rvv(", text)
        self.assertIn("#include <riscv_vector.h>", text)
        self.assertIn('#include "src/xnnpack/microparams.h"', text)
        self.assertNotIn("float min", text)
        self.assertIn("#include <arm_neon.h>", xdiff._kernel_unit(NEON, "neon", "x.h"))

    def test_compare_reports_differing_and_missing_cases(self):
        from portforge_tt.cpu import xdiff
        neon = "case 0 in=00 -> 01\ncase 1 in=01 -> 02\ncase 2 in=02 -> 03\nnoise\n"
        rvv = "case 0 in=00 -> 01\ncase 1 in=01 -> ff\n"
        d = xdiff.compare(neon, rvv, "qemu:128")
        self.assertEqual((d.cases, d.differing), (2, 2))  # one differs, one missing on rvv
        self.assertEqual(d.first, [("case 1 in=01 -> 02", "case 1 in=01 -> ff")])
