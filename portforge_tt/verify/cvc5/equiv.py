"""Bounded formal equivalence of a Neon kernel and an RVV kernel.

Wraps SALTyRN's verifier (intrinsic shims that build SMT terms, solved by
cvc5): both kernels run symbolically on the same symbolic inputs and the solver
looks for any input that makes their outputs differ. UNSAT for every swept
batch size means equivalent within those sizes, that VLEN and those params.

Modes:
  concrete   SALTyRN's own edge-value param configs
  symbolic   params as constrained symbols (one proof over SALTyRN's ranges);
             fails to compile when a kernel does scalar C on a param
  configs    whole param structs from a JSON list, e.g. the ones the library's
             own init function produces; batches are capped (max_batch) so every
             config is reached within the budget
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re

from ...remote import host
from ...remote.kernelsrc import strip_local_xnn_types

KERNEL_DEF = re.compile(r"\bvoid\s+((?:xnn_|test_)\w*)\s*\(")


def prepare(source_text, side):
    """Kernel text as SALTyRN expects it: ukernel renamed test_<side>, no #include lines,
    no local copies of XNNPACK types (see kernelsrc). Returns (text, removed)."""
    source_text, removed = strip_local_xnn_types(source_text)
    match = KERNEL_DEF.search(source_text)
    if not match:
        raise ValueError(f"no xnn_* or test_* function definition in the {side} kernel")
    text = source_text[:match.start(1)] + f"test_{side}" + source_text[match.end(1):]
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#include")) + "\n", removed


@dataclass
class Verdict:
    kernel: str
    mode: str
    vlen: int
    verdict: str                      # ALL_PASSED PARTIAL COUNTEREXAMPLE COMPILE_ERROR ...
    max_verified_batch: int = 0
    num_verified: int = 0
    configs: list = field(default_factory=list)       # [{"config", "verified_up_to"}]
    counterexample: dict = field(default_factory=dict)
    failed_config: str = ""
    error: str = ""
    seconds: float = 0.0
    normalized: list = field(default_factory=list)   # declarations stripped before verifying
    log: str = ""

    @property
    def proven(self):
        return self.verdict == "ALL_PASSED"


CONFIG_LINE = re.compile(r"config=(.+?)(?: — .*?)?: verified up to batch=(\d+)\s*$")
FAILED_LINE = re.compile(r"FAILED: .*? counterexample at batch=(\d+) config=(.+?)\s*$")
ELAPSED = re.compile(r"\((\d+(?:\.\d+)?)s\)\s*$")
UNDECLARED = re.compile(r"[‘'`](__riscv_\w+|v[a-z]+\d+m[f\d]+(?:x\d+)?_t)[’'`] was not declared")


def parse_log(text):
    """The orchestrator's final JSON object plus per-config progress lines."""
    result, start = {}, None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.rstrip() == "{":
            start = i
    if start is not None:
        try:
            result = json.loads("\n".join(lines[start:]))
        except json.JSONDecodeError:
            result = {}
    progress = {}  # a config is reported by a budget line and a summary line; keep the last
    for line in lines:
        m = CONFIG_LINE.search(line)
        if m and m.group(1) != "default":
            progress[m.group(1)] = int(m.group(2))
    configs = [{"config": c, "verified_up_to": b} for c, b in progress.items()]
    failed = ""
    for line in lines:
        m = FAILED_LINE.search(line)
        if m:
            failed = m.group(2)
    seconds = 0.0
    for line in lines:
        m = ELAPSED.search(line)
        if m and ("PARTIAL" in line or "VERIFIED" in line or "FAILED" in line):
            seconds = float(m.group(1))
    if not result:
        result = {"verdict": "NO_RESULT", "compile_error": "\n".join(lines[-15:])}
    if result.get("verdict") == "COMPILE_ERROR":
        missing = sorted(set(UNDECLARED.findall(text)))
        if missing:  # the verifier's intrinsic shims lack these; says nothing about the kernel
            result["verdict"] = "MISSING_INTRINSIC"
            result["missing_intrinsics"] = missing
    return result, configs, failed, seconds


def verify(cfg, neon_text, rvv_text, kernel, out_dir, mode="concrete", vlen=512, budget=120,
           configs=None, max_batch=0, input_range=None):
    """Run one verification; returns a Verdict and leaves the log under out_dir."""
    out_dir = Path(out_dir)
    tag = f"{kernel}.{mode}.vlen{vlen}"
    work = out_dir / tag
    work.mkdir(parents=True, exist_ok=True)
    # SALTyRN derives the kernel family (fixture, param struct) from the file stem.
    (work / "neon").mkdir(exist_ok=True)
    (work / "rvv").mkdir(exist_ok=True)
    neon_prepped, neon_removed = prepare(neon_text, "neon")
    rvv_prepped, rvv_removed = prepare(rvv_text, "rvv")
    (work / "neon" / f"{kernel}.c").write_text(neon_prepped)
    (work / "rvv" / f"{kernel}.c").write_text(rvv_prepped)
    args = ["--neon", f"{host.in_container(cfg, work)}/neon/{kernel}.c",
            "--rvv", f"{host.in_container(cfg, work)}/rvv/{kernel}.c",
            "--vlen", vlen, "--timeout", budget, "--backend", "cvc5"]
    entry = ["-m", "src.workflow.verification.orchestrator"]
    if mode == "symbolic":
        args.append("--symbolic-params")
    elif mode == "configs":
        if not configs:
            raise ValueError("mode 'configs' needs a list of param structs")
        (work / "configs.json").write_text(json.dumps(configs, indent=1))
        entry = [host.in_container(cfg, Path(__file__).resolve().parent / "saltyrn_configs.py"), f"{host.in_container(cfg, work)}/configs.json"]
    elif mode != "concrete":
        raise ValueError(f"unknown mode {mode!r}")
    if max_batch:
        args += ["--max-batch", max_batch]
    if input_range:
        args.append(f"--input-range={input_range[0]},{input_range[1]}")
    log = work / "verify.log"
    script = f"cd $SALTYRN && python {host.quote(entry + args)} > {host.in_container(cfg, log)} 2>&1"
    host.run(cfg, script, check=False, timeout=budget + 900)
    text = log.read_text() if log.exists() else ""
    result, per_config, failed, seconds = parse_log(text)
    verdict = Verdict(kernel=kernel, mode=mode, vlen=vlen, verdict=result.get("verdict", "NO_RESULT"),
                      max_verified_batch=result.get("max_verified_batch", 0),
                      num_verified=result.get("num_verified", 0), configs=per_config,
                      counterexample=result.get("counterexample", {}),
                      failed_config=failed or result.get("failed_config", ""),
                      error=(result.get("compile_error") or "")[:1500]
                      + (" missing intrinsics: " + ", ".join(result["missing_intrinsics"])
                         if result.get("missing_intrinsics") else ""),
                      seconds=seconds, log=str(log),
                      normalized=[f"neon: {r}" for r in neon_removed] + [f"rvv: {r}" for r in rvv_removed])
    (work / "verdict.json").write_text(json.dumps(asdict(verdict), indent=1))
    return verdict


def report(verdicts, title="Equivalence report"):
    """Markdown summary of a list of Verdicts."""
    rows = [f"# {title}", "",
            "| kernel | mode | VLEN | verdict | batches proven | configs | seconds | detail |",
            "|---|---|---|---|---|---|---|---|"]
    for v in verdicts:
        detail = ""
        if v.verdict == "COUNTEREXAMPLE":
            detail = f"batch {v.counterexample.get('batch', '?')}, config `{v.failed_config}`"
        elif v.error:
            detail = v.error.strip().splitlines()[0][:140].replace("|", "\\|")
        configs = len(v.configs) or ("-" if v.verdict != "ALL_PASSED" else 1)
        rows.append(f"| {v.kernel} | {v.mode} | {v.vlen} | **{v.verdict}** | 1-{v.max_verified_batch} "
                    f"| {configs} | {v.seconds:.0f} | {detail} |")
    rows += ["", "ALL_PASSED: no input differs for every swept batch, config and the VLEN shown. "
             "PARTIAL: proven up to the batch shown, then the budget ran out. COUNTEREXAMPLE: "
             "the solver found inputs whose outputs differ."]
    return "\n".join(rows) + "\n"
