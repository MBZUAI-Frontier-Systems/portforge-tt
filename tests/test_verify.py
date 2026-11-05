"""portforge_tt.verify (the contract, the cbmc lifter and model) and the dav1d adapter: no hardware, no
CBMC, no cvc5.

The lifted C is compiled and run natively with the host compiler, the same way
verify.cbmc.run.cosim does on the x86 host, so these tests check the lifter's control
flow and the model's arithmetic on hand-written disassembly."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portforge_tt.verify import contract  # noqa: E402
from portforge_tt.verify.cbmc import lift  # noqa: E402

TOY = """
0000000000010000 <toy_sum>:
   10000:	vsetvli	t0,a2,e16,m2,ta,ma
   10004:	vle16.v	v8,(a0)
   10008:	vwmul.vx	v16,v8,a3
   1000c:	vsetvli	zero,zero,e32,m4,ta,ma
   10010:	vadd.vx	v16,v16,a4
   10014:	vse32.v	v16,(a1)
   10018:	sub	a2,a2,t0
   1001c:	sh1add	a0,t0,a0
   10020:	sh2add	a1,t0,a1
   10024:	bnez	a2,10000 <toy_sum>
   10028:	ret

0000000000010030 <toy_table>:
   10030:	auipc	t1,0x0
   10034:	addi	t1,t1,24 # 10048 <toy_table+0x18>
   10038:	slli	a0,a0,0x3
   1003c:	add	t1,t1,a0
   10040:	jr	t1
   10044:	j	10060 <toy_table+0x30>
   10048:	li	a0,10
   1004c:	ret
   10050:	li	a0,20
   10054:	ret
   10058:	li	a0,30
   1005c:	ret
   10060:	ret

0000000000010070 <toy_call>:
   10070:	addi	sp,sp,-16
   10074:	sd	ra,8(sp)
   10078:	jal	10030 <toy_table>
   1007c:	ld	ra,8(sp)
   10080:	addi	sp,sp,16
   10084:	ret
"""

DRIVER = r"""
#include "fv_runtime.h"
const fv_data FV_DATA[FV_MAX_DATA] = {{0, 0, 0}};
const int FV_NDATA = 0;
%(lifted)s
static int16_t in[100];
static int32_t out[100], ref[100];
int main(void)
{
    for (int i = 0; i < 100; i++) in[i] = (int16_t)(i * 7919 - 30000);
    fv_region_add("in", 0x100000, sizeof in, (uint8_t *)in, 0);
    fv_region_add("out", 0x200000, sizeof out, (uint8_t *)out, 1);
    for (int i = 0; i < 100; i++) ref[i] = (int32_t)in[i] * -3 + 1000;
    const uint64_t a[5] = {0x100000, 0x200000, 100, (uint64_t)-3, 1000};
    fv_call(fvf_toy_sum, a, 5);
    for (int i = 0; i < 100; i++) if (out[i] != ref[i]) { printf("sum %%d: %%d != %%d\n", i, out[i], ref[i]); return 1; }
    for (uint64_t k = 0; k < 3; k++) {
        const uint64_t b[1] = {k};
        fv_call(fvf_toy_call, b, 1);
        if (S.x[10] != 10 * (k + 1)) { printf("table %%d -> %%d\n", (int)k, (int)S.x[10]); return 2; }
    }
    printf("OK\n");
    return 0;
}
"""


@unittest.skipUnless(shutil.which("cc"), "needs a host C compiler")
class LiftTests(unittest.TestCase):
    def build_and_run(self, vlen):
        funcs = lift.parse(TOY)
        body = lift.Lifter(funcs, "toy_sum").lift(["toy_sum", "toy_call"])
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "t.c"
            src.write_text(f"#define FV_VLEN {vlen}\n" + DRIVER % {"lifted": body})
            exe = Path(tmp) / "t"
            subprocess.run(["cc", "-O1", "-w", f"-I{ROOT / 'portforge_tt/verify/cbmc'}", str(src), "-o", str(exe)], check=True)
            return subprocess.run([str(exe)], capture_output=True, text=True)

    def test_vector_loop_and_jump_table(self):
        for vlen in (128, 512):
            r = self.build_and_run(vlen)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("OK", r.stdout)

    def test_unknown_instruction_fails_closed(self):
        funcs = lift.parse("0000000000010000 <f>:\n   10000:\tvfadd.vv\tv8,v8,v16\n   10004:\tret\n")
        with self.assertRaises(lift.Unsupported):
            lift.Lifter(funcs, "f").lift()

    def test_jump_into_another_function_fails_closed(self):
        text = ("0000000000010000 <f>:\n   10000:\tj\t10014 <g+0x4>\n   10004:\tret\n"
                "0000000000010010 <g>:\n   10010:\tli\ta0,1\n   10014:\tret\n")
        with self.assertRaises(lift.Unsupported):
            lift.Lifter(lift.parse(text), "f").lift()


class CheckasmParseTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ROOT / "ports/dav1d-rvv"))
        import evaluate
        self.ev = evaluate

    def test_pass_needs_an_rvv_function(self):
        ok = ("checkasm:\n - CPU: x\nC:\n - refmvs.splat_mv [OK]\nRVV:\n - refmvs.splat_mv [OK]\n"
              "checkasm: all 6 tests passed\n")
        self.assertTrue(self.ev.parse_checkasm(ok, 0)["passed"])
        no_rvv = "C:\n - refmvs.splat_mv [OK]\ncheckasm: all 6 tests passed\n"
        r = self.ev.parse_checkasm(no_rvv, 0)
        self.assertFalse(r["passed"])
        self.assertIn("no RVV", self.ev._row(r)["error"])
        failed = "RVV:\n - refmvs.splat_mv [FAILED]\ncheckasm: 1 of 6 tests failed\n"
        self.assertFalse(self.ev.parse_checkasm(failed, 1)["passed"])

    def test_families_and_technique_files(self):
        for family, entry in self.ev.families().items():
            for key in ("test", "functions", "symbols", "asm_path", "wiring", "context", "verify"):
                self.assertIn(key, entry, f"{family}: {key}")
            self.assertIn(entry["wiring"], self.ev.WIRING)
            self.assertTrue(set(entry["verify"]) <= set(contract.METHODS), family)
            self.assertTrue((ROOT / "ports/dav1d-rvv/kernels" / family / "TECHNIQUES.json").exists())
            self.assertTrue((ROOT / "ports/dav1d-rvv/kernels" / family / "placeholder.S").exists())

    @unittest.skipUnless(os.environ.get("PORTFORGE_DAV1D_SRC"), "set PORTFORGE_DAV1D_SRC to a dav1d checkout at the pin")
    def test_context_holds_every_isa_implementation_of_the_functions(self):
        """Workers port from context/: an excerpt that stops before the function (load_tmvs did) hides it."""
        source = Path(os.environ["PORTFORGE_DAV1D_SRC"])
        own = {"refmvs-load_tmvs": {"arm": "load_tmvs_neon", "x86": "cglobal load_tmvs", "loongarch": "load_tmvs_lsx"},
               "refmvs-splat_mv": {"arm": "splat_mv_neon", "x86": "cglobal splat_mv"}}
        for family, needles in own.items():
            texts = {}
            for item in self.ev.families()[family]["context"]:
                text = (source / item["path"]).read_text(errors="replace")
                if item.get("lines"):
                    a, b = item["lines"]
                    text = "\n".join(text.splitlines()[a - 1:b])
                texts[item["name"]] = text
            for isa, needle in needles.items():
                with self.subTest(family=family, isa=isa):
                    self.assertTrue(any(needle in t for n, t in texts.items() if n.startswith(isa)))


class ContractTests(unittest.TestCase):
    def test_only_all_passed_passes(self):
        for verdict in contract.VERDICTS:
            r = contract.result("cbmc", verdict, proved=1, configs=2)
            self.assertEqual(r["passed"], verdict == "ALL_PASSED")
            self.assertEqual(r["method"], "cbmc")
        with self.assertRaises(ValueError):
            contract.result("cbmc", "PROVED")

    def test_run_never_passes_a_broken_or_unknown_method(self):
        self.assertEqual(contract.run("nope", None, "f", {}, {}, ".", 1)["verdict"], "UNSUPPORTED")
        from portforge_tt.verify import cbmc
        r = cbmc.certify(None, "f", {"project": "xnnpack"}, {}, ".", 1)
        self.assertEqual(r["verdict"], "UNSUPPORTED")
        from unittest import mock
        with mock.patch("portforge_tt.verify.cbmc.certify", side_effect=RuntimeError("boom")):
            r = contract.run("cbmc", None, "f", {"project": "dav1d"}, {}, ".", 1)
        self.assertEqual((r["verdict"], r["passed"]), ("ERROR", False))
        self.assertTrue(contract.passed([]))
        self.assertFalse(contract.passed([r]))

    def test_cvc5_without_params_is_unsupported(self):
        from portforge_tt.verify import cvc5
        self.assertEqual(cvc5.certify(None, "f", {}, {"configs": []}, ".", 1)["verdict"], "UNSUPPORTED")


if __name__ == "__main__":
    unittest.main()
