"""MCP tools of one kernel worker: its workspace, compile and check. Untrusted.

The worker reads anything in its workspace (the parent kernel, the plan, the
profiles and catalogue entries the orchestrator put there), writes only its
deliverables, and asks the x86 host to compile (with a static profile and a
predicted cycles/element from the X280 calibration) or to run the bit-exact
check under QEMU. It never reaches the board, the held-out regimes or the
trusted harness: the orchestrator re-runs every gate itself on the snapshot.
"""
from pathlib import Path

from ..mcpserver import ToolServer

DELIVERABLES = ("TRANSFORMS.json", "NOTES.md")
MAX_BYTES = 400_000


def _profile_text(p):
    if not p:
        return "no profile"
    lines = [f"{len(p.get('loops', []))} hot loop(s); calls: {p.get('calls') or 'none'}; "
             f"prologue {p.get('prologue_instructions')} insns, {p.get('prologue_branches')} branches"]
    preds = {x["start"]: x for x in p.get("predicted", []) if x}
    for h in p.get("loops", []):
        pr = preds.get(h["start"], {})
        counts = ", ".join(f"{k} {v}" for k, v in sorted(h["counts"].items()) if k != "scalar")
        lines.append(f"  loop {h['start']}: {counts}, scalar {h['counts'].get('scalar', 0)}, vsetvl {h['vsetvl']}, "
                     f"spills {h['spills']}" + (f"; predicted {pr['cycles_per_elem']:.3f} cycles/elem "
                                                f"({pr['bound']}-bound)" if pr.get("cycles_per_elem") else ""))
    return "\n".join(lines)


class KernelTools(ToolServer):
    def setup(self, workspace, gates, pack, family, tag, source="kernel.c"):
        self.workspace = Path(workspace)
        self.gates, self.pack, self.family, self.tag, self.source = gates, pack, family, tag, source
        self.writable = (source, source + ".in") + DELIVERABLES
        for name in ("files", "read", "write", "compile", "check"):
            self.mcp.add_tool(getattr(self, name), name="kernel_" + name)

    def _path(self, path):
        target = (self.workspace / path).resolve()
        if self.workspace.resolve() not in target.parents and target != self.workspace.resolve():
            raise ValueError("outside the workspace")
        return target

    def files(self) -> list:
        """Files in your workspace."""
        return sorted(str(p.relative_to(self.workspace)) for p in self.workspace.rglob("*") if p.is_file())

    def read(self, path: str, offset: int = 0) -> str:
        """Read a workspace file: 60k characters from character `offset` (large context
        files need several calls; the reply says where the next chunk starts)."""
        text = self._path(path).read_text()
        chunk = text[offset:offset + 60000]
        more = len(text) - offset - len(chunk)
        return chunk + (f"\n[... {more} more characters: read again with offset={offset + len(chunk)}]" if more > 0 else "")

    def write(self, path: str, content: str) -> dict:
        """Write one of your deliverables: the kernel source, TRANSFORMS.json, NOTES.md."""
        if path not in self.writable:
            raise ValueError(f"writable files: {', '.join(self.writable)}")
        if len(content.encode()) > MAX_BYTES:
            raise ValueError("file too large")
        self._path(path).write_text(content)
        return {"saved": path}

    def compile(self) -> str:
        """Compile your kernel source for rv64gcv as the project's build would; returns
        errors, or the static profile of each hot loop with predicted X280 cycles per
        element."""
        remote = self.gates.put(self.workspace / self.source, f"{self.gates.run_rel}/workers/{self.tag}/{self.source}")
        r = self.gates.call("dev_compile", {"pack": self.pack, "family": self.family, "tag": self.tag,
                                            "source": remote}, timeout=600)
        if not r.get("ok"):
            return "tool error: " + r.get("error", "?")[:2000]
        if not r["compiled"]:
            return "compile errors:\n" + r["errors"][-3000:]
        return "compiled.\n" + _profile_text(r["profile"])

    def check(self) -> str:
        """Bit-exact check of your kernel against the reference under QEMU at VLEN 128
        and 1024 (the project's own test suite where it has one). Builds the project
        with your kernel: about 1-3 minutes."""
        remote = self.gates.put(self.workspace / self.source, f"{self.gates.run_rel}/workers/{self.tag}/{self.source}")
        r = self.gates.call("dev_check", {"pack": self.pack, "family": self.family, "tag": self.tag,
                                          "source": remote, "out": f"{self.gates.run_rel}/workers/{self.tag}/check"},
                            timeout=1800)
        if not r.get("ok"):
            return "tool error: " + r.get("error", "?")[:2000]
        if not r["compiled"]:
            return f"does not build: {str(r['errors'])[-3000:]}"
        c = r["check"]
        lines = [f"passed: {c['passed']}"]
        for where, row in c.items():
            if where != "passed":
                lines.append(f"  {where}: {row.get('mismatched_cases')} of {row.get('cases')} cases differ, "
                             f"guard violations {row.get('guard_violations')}, max err {row.get('max_err')}"
                             + (f"; first: {row['first_failure']}" if row.get("first_failure") else "")
                             + (f"; error: {row['error']}" if row.get("error") else ""))
                if row.get("output"):
                    lines.append("    output:\n" + row["output"][-2500:])
        return "\n".join(lines)
