"""The search tree of one campaign, rebuilt from an append-only event log.

runs/<id>/events.jsonl holds trusted events written by the orchestrator only:

  {"type": "node", "id": sha, "parents": [...], "operator": "seed|import|placeholder|port|mutate|combine|dispatch|knob|debug|novel",
   "declared": {"applied": [...], "knobs": {...}, "novel": "...", "note": "..."}, "source": "nodes/<sha>/kernel.c"}
  {"type": "gate", "id": sha, "gate": "G0|G1|G2|G2f|G3|G4|G5", "passed": bool, "detail": {...}}
  {"type": "profile", "id": sha, "profile": {...}}          # static profile + detect hits
  {"type": "perf", "id": sha, "session": "...", "verdict": {...}}   # bench.analyze output
  {"type": "status", "id": sha, "status": "...", "note": "..."}
  {"type": "declare", "id": sha, "applied": [...], "note": "..."}   # techniques found in a seed's code

Workers never write here. Outcomes of an edge (parent -> child) are derived
only from completed receipts, so a pending child is never a rejection, and a
child that compiled but lost on speed is recorded as regress/neutral/mixed
with per-cell numbers instead of disappearing.
"""
import json
import math
import statistics
import time
from pathlib import Path

from . import objective

OPERATORS = ("seed", "import", "placeholder", "port", "mutate", "combine", "dispatch", "knob", "debug", "novel")
OUTCOMES = ("impl_fail", "wrong", "regress", "neutral", "mixed", "improve")
CORRECTNESS_GATES = ("G1", "G2", "G2f")


class Tree:
    def __init__(self, run_dir, noise=0.005):
        self.run_dir = Path(run_dir)
        self.noise = noise  # the family's per-cell noise floor δ (outcome thresholds)
        self.path = self.run_dir / "events.jsonl"
        self.nodes = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._apply(json.loads(line))

    # ------------------------------------------------------------------ events

    def append(self, event):
        event = {**event, "t": round(time.time(), 3)}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        self._apply(event)
        return event

    def _apply(self, e):
        kind = e["type"]
        if kind == "node":
            if e["operator"] not in OPERATORS:
                raise ValueError(f"unknown operator {e['operator']}")
            self.nodes.setdefault(e["id"], {
                "id": e["id"], "parents": list(e.get("parents", [])), "operator": e["operator"],
                "declared": e.get("declared", {}), "source": e.get("source"), "gates": {},
                "profile": None, "perf": [], "status": "pending", "notes": [], "t": e.get("t")})
            return
        node = self.nodes.get(e["id"])
        if node is None:
            raise ValueError(f"event for unknown node {e['id']}")
        if kind == "gate":
            node["gates"][e["gate"]] = {"passed": bool(e["passed"]), "detail": e.get("detail", {})}
            if not e["passed"]:
                node["status"] = f"failed@{e['gate']}"
        elif kind == "profile":
            node["profile"] = e["profile"]
        elif kind == "perf":
            node["perf"].append({"session": e.get("session"), "verdict": e["verdict"]})
            if node["status"] == "pending":
                node["status"] = "measured"
        elif kind == "declare":
            node["declared"]["applied"] = sorted(set(node["declared"].get("applied", [])) | set(e["applied"]))
        elif kind == "status":
            node["status"] = e["status"]
            if e.get("note"):
                node["notes"].append(e["note"])

    def add_node(self, sha, parents, operator, declared, source):
        if sha in self.nodes:
            return self.nodes[sha], False
        self.append({"type": "node", "id": sha, "parents": parents, "operator": operator,
                     "declared": declared, "source": source})
        return self.nodes[sha], True

    # ------------------------------------------------------------------ derived views

    def applied(self, node_id):
        node = self.nodes.get(node_id)
        return set((node or {}).get("declared", {}).get("applied", []))

    def delta(self, child_id):
        """Techniques the child adds over its first parent."""
        child = self.nodes[child_id]
        parent = child["parents"][0] if child["parents"] else None
        return sorted(self.applied(child_id) - (self.applied(parent) if parent else set()))

    def latest_verdict(self, node_id):
        perf = self.nodes[node_id]["perf"]
        return perf[-1]["verdict"] if perf else None

    def confirmed_flags(self, node_id):
        """Cells flagged slower in the latest session AND the one before it.

        A single session's flag is a suspicion (cross-session drift at DRAM-sized
        shapes exceeds the within-session noise floor); the orchestrator runs a
        confirmation session for every flagged node, and only cells flagged in
        both count against the no-regression constraint."""
        perf = self.nodes[node_id]["perf"]
        if not perf:
            return []
        latest = set(perf[-1]["verdict"].get("flagged", []))
        if len(perf) < 2:
            return sorted(latest)
        return sorted(latest & set(perf[-2]["verdict"].get("flagged", [])))

    def metrics(self, node_id):
        node = self.nodes[node_id]
        check = node["gates"].get("G2", {}).get("detail")
        m = objective.cpu_metrics(self.latest_verdict(node_id), check, node.get("profile"))
        if node["perf"]:
            m["flagged_cells"] = len(self.confirmed_flags(node_id))
        return m

    def correct(self, node_id):
        gates = self.nodes[node_id]["gates"]
        return all(gates[g]["passed"] for g in CORRECTNESS_GATES if g in gates) and "G2" in gates

    def cell_speedups(self, node_id):
        v = self.latest_verdict(node_id)
        if not v:
            return {}
        return {k: c["speedup"]["s"] for k, c in v["cells"].items() if c.get("speedup")}

    def outcome(self, child_id, delta=None, delta_g=None):
        """Outcome of the edge first-parent -> child, or None while pending."""
        delta = self.noise if delta is None else delta
        delta_g = max(0.01, 2 * self.noise) if delta_g is None else delta_g
        child = self.nodes[child_id]
        gates = child["gates"]
        if "G0" in gates and not gates["G0"]["passed"]:
            return "impl_fail"
        if any(g in gates and not gates[g]["passed"] for g in CORRECTNESS_GATES):
            return "wrong"
        if not child["perf"] or not child["parents"]:
            return None
        parent = child["parents"][0]
        if parent not in self.nodes or not self.nodes[parent]["perf"]:
            return None
        mine, theirs = self.cell_speedups(child_id), self.cell_speedups(parent)
        shared = [k for k in mine if k in theirs]
        if not shared:
            return None
        ratios = [mine[k] / theirs[k] for k in shared]
        g = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        wins = sum(r > 1 + delta for r in ratios)
        losses = sum(r < 1 - delta for r in ratios)
        if wins and losses:
            return "mixed"
        if g > 1 + delta_g and not losses:
            return "improve"
        if g < 1 - delta_g or losses:
            return "regress"
        return "neutral"

    def edges(self):
        """[(parent, child, delta techniques, outcome)] for every child with a parent."""
        out = []
        for node in self.nodes.values():
            if node["parents"]:
                out.append((node["parents"][0], node["id"], self.delta(node["id"]), self.outcome(node["id"])))
        return out

    def ratios(self, child_id):
        """{cell: child speedup / first-parent speedup} for a measured edge, else {}."""
        child = self.nodes[child_id]
        if not child["parents"] or not child["perf"] or child["parents"][0] not in self.nodes:
            return {}
        mine, theirs = self.cell_speedups(child_id), self.cell_speedups(child["parents"][0])
        return {k: mine[k] / theirs[k] for k in mine if k in theirs and theirs[k]}

    def history(self, parent, top=4):
        """Markdown evidence for a worker on `parent`: its children with every cell that
        moved beyond the noise floor, then each technique's record across the whole
        tree (which cells it tends to win or lose, per edge that added it), then the
        leader of every cell. Rejected ideas stay visible with their numbers."""
        d = self.noise
        moved = lambda r: sorted(((k, v) for k, v in r.items() if abs(v - 1) > d), key=lambda kv: -kv[1])
        fmt = lambda kv: ", ".join(f"{k} {v:.2f}x" for k, v in kv)
        lines = [f"# History (cells: speedup of child / its parent; noise floor {d:.1%})", "",
                 f"## Tried on this parent ({parent[:10]})", ""]
        children = [(c, delta, o) for p, c, delta, o in self.edges() if p == parent]
        for c, delta, o in children:
            r = self.ratios(c)
            g = math.exp(sum(map(math.log, r.values())) / len(r)) if r else None
            lines.append(f"- {', '.join(delta) or '(no new technique)'} [{self.nodes[c]['operator']}] -> {o or 'pending'}"
                         + (f", geomean vs parent {g:.3f}" if g else "")
                         + (f"; failed {self.nodes[c]['status']}" if self.nodes[c]["status"].startswith("failed") else ""))
            m = moved(r)
            if m:
                wins, losses = [kv for kv in m if kv[1] > 1], [kv for kv in m if kv[1] < 1]
                lines.append(f"  wins: {fmt(wins) or 'none'}; losses: {fmt(losses[::-1]) or 'none'}")
        if not children:
            lines.append("nothing yet")
        lines += ["", "## Each technique across the tree (every edge that added it)", ""]
        per = {}
        for p, c, delta, o in self.edges():
            for t in delta:
                e = per.setdefault(t, {"outcomes": {}, "cells": {}, "best": None})
                e["outcomes"][o or "pending"] = e["outcomes"].get(o or "pending", 0) + 1
                for k, v in self.ratios(c).items():
                    e["cells"].setdefault(k, []).append(v)
                g = (self.metrics(c) if self.nodes[c]["perf"] else {}).get("speedup_geomean")
                if g and (e["best"] is None or g > e["best"][1]):
                    e["best"] = (c, g)
        for t, e in sorted(per.items()):
            med = {k: statistics.median(v) for k, v in e["cells"].items()}
            m = moved(med)
            wins, losses = [kv for kv in m if kv[1] > 1][:top], [kv for kv in m if kv[1] < 1][::-1][:top]
            lines.append(f"- {t}: {', '.join(f'{k} {n}' for k, n in sorted(e['outcomes'].items()))}"
                         + (f"; best node {e['best'][0][:10]} at {e['best'][1]:.3f}x vs upstream" if e["best"] else ""))
            if m:
                lines.append(f"  median per cell, tends to win: {fmt(wins) or 'none'}; to lose: {fmt(losses) or 'none'}")
        if not per:
            lines.append("nothing yet")
        lines += ["", "## Leader of every cell (speedup vs the fastest correct upstream kernel)", ""]
        best = {}
        for n in self.measured():
            for k, v in self.cell_speedups(n).items():
                if k not in best or v > best[k][1]:
                    best[k] = (n, v)
        lines += [f"- {k}: {n[:10]} {v:.3f}x ({', '.join(sorted(self.applied(n))) or 'seed'})"
                  for k, (n, v) in sorted(best.items())] or ["nothing measured yet"]
        return "\n".join(lines) + "\n"

    def tried_index(self):
        """{(parent, frozenset(applied delta), knobs json): child} for plan dedup."""
        index = {}
        for node in self.nodes.values():
            if node["parents"]:
                key = (node["parents"][0], frozenset(self.delta(node["id"])),
                       json.dumps(node["declared"].get("knobs", {}), sort_keys=True))
                index[key] = node["id"]
        return index

    def measured(self):
        """Correct nodes with a perf verdict."""
        return [n for n in self.nodes if self.correct(n) and self.nodes[n]["perf"]]

    def frontier(self, chosen=None):
        chosen = chosen or objective.spec()
        points = [{"id": n, "metrics": self.metrics(n)} for n in self.measured()]
        return objective.rank(points, chosen)

    def specialists(self):
        """{cell: node} best measured correct node per cell, feasible overall or not."""
        best = {}
        for n in self.measured():
            for cell, s in self.cell_speedups(n).items():
                if cell not in best or s > best[cell][1]:
                    best[cell] = (n, s)
        return {cell: n for cell, (n, _) in best.items()}

    def descriptor(self, node_id):
        p = self.nodes[node_id].get("profile") or {}
        d = p.get("descriptors", {})
        return (d.get("lmul"), d.get("sew"), d.get("unroll"), d.get("dispatch"))

    def archive(self, chosen=None):
        """MAP-Elites: {descriptor: best feasible node by the primary metric}."""
        chosen = chosen or objective.spec()
        primary = chosen["primary"] or "speedup_geomean"
        cells = {}
        for n in self.measured():
            m = self.metrics(n)
            if primary not in m or not objective.feasible(m, chosen["constraints"])[0]:
                continue
            key = self.descriptor(n)
            if key not in cells or m[primary] > self.metrics(cells[key])[primary]:
                cells[key] = n
        return cells

    def summary(self):
        status = {}
        for n in self.nodes.values():
            status[n["status"]] = status.get(n["status"], 0) + 1
        return {"nodes": len(self.nodes), "status": status}
