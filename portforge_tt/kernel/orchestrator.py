"""One kernel campaign: seeds -> rounds of plan / implement / gate / measure -> report.

Runs on the controller (where the agent CLIs are logged in); every trusted
gate runs on the x86 host through RemoteGates. A round:

  1. parents  policy.parents: frontier, archive elites, per-cell specialists
  2. menus    policy.menu (Thompson over contextual posteriors, dropout, floor)
              plus combination and dispatch offers
  3. plans    the "thompson" planner turns menus into plans mechanically; the
              "llm" planner asks an agent to choose and combine
  4. dedupe   against tried (parent, technique set, knobs)
  5. work     isolated implementer agents in fresh workspaces, in parallel
  6. novelty  identical source -> alias, no evaluation
  7. gates    G0 compile + profile, G1 gtests, G2 bit-exact (one batched build)
  8. measure  G3/G4 bench for everything correct, one board session
  9. debug    one debugger retry for impl_fail / wrong children

Everything trusted lands in events.jsonl (tree.Tree); status.json is rewritten
every round.
"""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import secrets
import shutil
import time

from . import catalogue, novelty, objective, policy, tree as tree_mod
from .devtools import KernelTools

REPO = Path(__file__).resolve().parents[2]

PROMPTS = Path(__file__).resolve().parent / "prompts"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


XNN_RULES_3_4 = """3. Keep the signature, the params struct and XNN_OOB_READS semantics; do not read or
   write outside the buffers. No _tu/_mu intrinsics (the formal gate has no shims for
   them). No calls into libgcc in the hot path (XNNPACK compiles RVV with
   -march=rv64gcv: no Zbb, so __builtin_ctz becomes a call).
4. The hot loop must be vector code; no scalar fallback for the bulk of the work.
"""
# The implementer contract, by kernel signature.
CONTRACTS = {
    "transposec": """1. Exact for EVERY block shape (block_width, block_height >= 1, tails included), every
   stride >= the block's row size (any multiple of 4 bytes, not just multiples of 64),
   4-byte-aligned pointers, and every VLEN from 128 to 1024. Bytes between
   block_height*4 and output_stride in each output row (padding of a sub-tensor view)
   must stay untouched. Held-out layouts (tile sizes, paddings, offsets) are checked.
2. Dispatch is allowed only on block_width, block_height, the strides and vlenb, never on
   data; write why each path is exact on its whole domain in NOTES.md.
""" + XNN_RULES_3_4,
    None: """1. Bit-exact with the scalar reference for EVERY parameter value XNNPACK's init can
   produce, every batch size (tails included), every input. Width bounds come from the
   init's domain, never from test values. Held-out parameter regimes are checked.
2. Parameter dispatch is allowed only on predicates you prove make each path exact on
   its whole domain; never special-case particular values. Write each predicate's
   proof sketch in NOTES.md (every narrowing needs a range argument).
""" + XNN_RULES_3_4,
    "dav1d": """1. Bit-exact with dav1d's C reference: checkasm (the project's own test, randomized
   inputs, sizes and edge flags) must pass under QEMU at VLEN 128, 256, 512 and 1024 AND
   on the X280 itself (emulation passing is not enough: strided accesses with offsets
   that are not a multiple of the element width are implementation-defined and have
   failed on this board). Held-out checkasm seeds are run before anything counts.
2. Implement every function named in SPEC.md, completely, in your assembly file. The
   init code (harness-owned) points dav1d's function table at your symbols; you cannot
   change it, the C side, or any other file.
3. No calls or jumps to anything outside your file (no C fallback, no libc, no other
   dav1d asm): the gate rejects any call/tail/jump to a symbol you do not define. Use
   the project's asm macros (function/endfunc from src/riscv/asm.S) and the RISC-V
   psABI (callee-saved s0-s11, fs0-fs11; stack 16-byte aligned).
4. The hot loops must be vector code; a scalar port is rejected. Never read or write
   outside the buffers the C reference touches (checkasm guards the destination).
""",
}


LANGUAGES = {
    "kernel.c": "C with RVV intrinsics (riscv_vector.h)",
    "kernel.S": "RISC-V assembly (RVV 1.0, GNU as syntax, the target project's asm macros)",
}


def prompt(name, **fields):
    text = (PROMPTS / f"{name}.md").read_text()
    for key, value in fields.items():
        text = text.replace("{" + key + "}", str(value))
    return text


class Campaign:
    def __init__(self, run_dir, pack, family, gates, entry, kernel_doc, workers=2, beam=4, plans=4,
                 planner="thompson", rounds=10, hours=12.0, board_seconds=4 * 3600, seed=None,
                 implementer_minutes=20, bench_rounds=15, delta=None, log=print):
        self.run_dir = Path(run_dir)
        self.pack, self.family, self.gates, self.entry = pack, family, gates, entry
        self.src = entry.get("source_name", "kernel.c")  # kernel.S for assembly packs (dav1d)
        self.cat = catalogue.load()
        self.doc = kernel_doc
        self.items = catalogue.kernel_menu(kernel_doc, self.cat)
        self.workers, self.beam, self.nplans, self.planner = workers, beam, plans, planner
        self.max_rounds, self.deadline = rounds, time.time() + hours * 3600
        self.board_budget = board_seconds
        if delta is None:  # the family's measured noise floor, else the qs8-vcvt A/A value
            spec = REPO / pack / "kernels" / family / "regimes.json"
            delta = json.loads(spec.read_text()).get("noise_floor", 0.005) if spec.exists() else 0.005
        self.implementer_minutes, self.bench_rounds, self.delta = implementer_minutes, bench_rounds, delta
        self.log = log
        self.tree = tree_mod.Tree(self.run_dir, noise=delta)
        self.index = novelty.Index()
        manifest = self.run_dir / "manifest.json"
        if manifest.exists():
            self.manifest = json.loads(manifest.read_text())
        else:
            self.manifest = {"pack": pack, "family": family, "created": dt.datetime.now(dt.timezone.utc).isoformat(),
                             "seed": seed if seed is not None else secrets.randbelow(2 ** 31),
                             "heldout_seed": secrets.randbelow(2 ** 31),  # never shown to workers
                             "planner": planner, "workers": workers, "beam": beam,
                             "catalogue_sha256": sha(catalogue.CATALOGUE.read_text()),
                             "techniques_sha256": sha(json.dumps(kernel_doc, sort_keys=True)), "rounds": 0,
                             "delta": delta}
            self.run_dir.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(self.manifest, indent=1) + "\n")
        # budgets survive a restart: board seconds and active hours already spent count
        self.board_used = float(self.manifest.get("board_seconds", 0.0))
        self.started, self.active_before = time.time(), float(self.manifest.get("active_seconds", 0.0))
        self.deadline = time.time() + hours * 3600 - self.active_before
        for node_id, node in self.tree.nodes.items():
            src = self.run_dir / node["source"]
            if src.exists():
                self.index.add(node_id, src.read_text(), node.get("profile"))

    # ------------------------------------------------------------------ helpers

    def ukernel(self, node_id):
        token = self.entry["neon_ukernel"].split("_ukernel")[0]
        return f"{token}_ukernel__rvv_n{node_id[:10]}"

    def store_source(self, text):
        node_id = sha(text)
        rel = f"nodes/{node_id}/{self.src}"
        path = self.run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return node_id, rel

    def save_manifest(self):
        self.manifest["board_seconds"] = round(self.board_used, 1)
        self.manifest["active_seconds"] = round(self.active_before + time.time() - self.started, 1)
        (self.run_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=1) + "\n")

    def status(self, note=""):
        front = self.tree.frontier(objective.spec())
        best = front["frontier"][0] if front.get("frontier") else None
        measured = {n: self.tree.metrics(n) for n in self.tree.measured()}
        top = max(measured, key=lambda n: measured[n].get("speedup_geomean") or 0, default=None)
        status = {"round": self.manifest["rounds"], "tree": self.tree.summary(), "best": best,
                  "best_metrics": self.tree.metrics(best) if best else None, "pareto": front.get("pareto"),
                  # highest geomean among correct measured nodes, feasible or not (progress
                  # while every node still loses some cell to the per-cell upstream best)
                  "top_geomean": {"node": top, **measured[top]} if top else None,
                  "board_seconds": round(self.board_used), "note": note,
                  "updated": dt.datetime.now(dt.timezone.utc).isoformat()}
        (self.run_dir / "status.json").write_text(json.dumps(status, indent=1, default=str) + "\n")
        return status

    # ------------------------------------------------------------------ seeding

    def seed(self, sources, operator="seed"):
        """Add seed/import kernels ({label: path}) as root nodes and evaluate them."""
        new = []
        for label, path in sources.items():
            text = Path(path).read_text()
            node_id, rel = self.store_source(text)
            _, created = self.tree.add_node(node_id, [], operator, {"applied": [], "note": label}, rel)
            if created:
                new.append(node_id)
                self.index.add(node_id, text)
        if new:
            self.evaluate(new)
            self.declare_detected(new)
        return new

    def declare_detected(self, node_ids):
        """Seeds and imports declare no techniques, so plans kept re-proposing what their
        code already does (x32-transposec seeds use segment loads). Record every menu
        technique whose detect predicate holds on the compiled code; predicates that
        are always true ("True") say nothing and are skipped."""
        for n in node_ids:
            profile = self.tree.nodes[n].get("profile")
            if not profile:
                continue
            found = sorted(i["id"] for i in self.items if i.get("detect") not in (None, "True", "False")
                           and catalogue.detects(i, profile))
            if found:
                self.tree.append({"type": "declare", "id": n, "applied": found, "note": "detected in the code"})

    # ------------------------------------------------------------------ trusted evaluation

    def evaluate(self, node_ids):
        """G0-G2 for a batch in one injected build, then one bench session for the correct ones."""
        if not node_ids:
            return
        names = {self.ukernel(n): n for n in node_ids}
        kernels = {}
        for name, n in names.items():
            kernels[name] = self.gates.put(self.run_dir / self.tree.nodes[n]["source"],
                                           f"{self.gates.run_rel}/nodes/{n}/{self.src}")
        tag = f"camp-{self.run_dir.name}"[:40]
        r = self.gates.call("gates", {"pack": self.pack, "family": self.family, "tag": tag, "kernels": kernels,
                                      "out": f"{self.gates.run_rel}/gates/{int(time.time())}"})
        if not r.get("ok"):
            self.log(f"gates failed: {r.get('error')}")
            for n in node_ids:
                self.tree.append({"type": "status", "id": n, "status": "infra-error", "note": r.get("error", "")[:500]})
            return
        correct = []
        for name, res in r["results"].items():
            n = names[name]
            if res.get("profile"):
                self.tree.append({"type": "profile", "id": n, "profile": res["profile"]})
                self.index.add(n, profile=res["profile"])
            for gate in ("G0", "G1", "G2"):
                if gate in res:
                    detail = res[gate].get("detail", {k: v for k, v in res[gate].items() if k != "passed"})
                    self.tree.append({"type": "gate", "id": n, "gate": gate, "passed": res[gate]["passed"],
                                      "detail": detail})
            if all(res.get(g, {}).get("passed") for g in ("G0", "G1", "G2")):
                correct.append(name)
        if not correct:
            return
        if self.board_used >= self.board_budget:
            self.log("board budget spent; correct nodes stay unmeasured")
            return
        t0 = time.time()
        b = self.gates.call("bench", {"pack": self.pack, "family": self.family, "tag": tag, "kernels": correct,
                                      "rounds": self.bench_rounds, "delta": self.delta,
                                      "seed": self.manifest["seed"] + self.manifest["rounds"],
                                      "out": f"{self.gates.run_rel}/bench/{int(time.time())}"})
        self.board_used += time.time() - t0
        if not b.get("ok"):
            self.log(f"bench failed: {b.get('error')}")
            return
        self._record_perf(names, b)
        # a flag is only a suspicion: confirm it in a new session with twice the rounds
        flagged = [name for name, v in b["verdicts"].items() if v.get("flagged")]
        if flagged and self.board_used < self.board_budget:
            self.log(f"confirming flags of {len(flagged)} node(s) in a new session")
            t0 = time.time()
            c = self.gates.call("bench", {"pack": self.pack, "family": self.family, "tag": tag, "kernels": flagged,
                                          "rounds": 2 * self.bench_rounds, "delta": self.delta,
                                          "seed": self.manifest["seed"] + 7919 + self.manifest["rounds"],
                                          "out": f"{self.gates.run_rel}/bench/{int(time.time())}-confirm"})
            self.board_used += time.time() - t0
            if c.get("ok"):
                self._record_perf(names, c)
        self.save_manifest()

    def _record_perf(self, names, b):
        session = b["receipt"].get("seed")
        for name, verdict in b["verdicts"].items():
            slim = {k: verdict[k] for k in ("metrics", "geomean", "flagged", "wrong_cells", "correctness_wins",
                                            "feasible", "reference_cell", "delta")}
            slim["views"] = verdict.get("views", {})
            slim["cells"] = {k: {kk: c.get(kk) for kk in ("speedup", "baseline", "cycles_per_elem", "excluded",
                                                         "candidate_correct", "flagged")}
                             for k, c in verdict["cells"].items()}
            self.tree.append({"type": "perf", "id": names[name], "session": session, "verdict": slim})
            if verdict["wrong_cells"]:
                self.tree.append({"type": "gate", "id": names[name], "gate": "G2", "passed": False,
                                  "detail": {"wrong_after_timing": verdict["wrong_cells"][:10]}})

    # ------------------------------------------------------------------ planning

    def menus(self, rng):
        picked = policy.parents(self.tree, self.beam, rng)
        sig = self.entry.get("signature")
        out = []
        for parent, source in picked:
            out.append({"parent": parent, "source": source,
                        "menu": policy.menu(self.tree, parent, self.items, rng, signature=sig),
                        "offers": policy.combination_offers(self.tree, parent, self.items)})
        return out, policy.dispatch_offers(self.tree)

    def plan_thompson(self, menus, dispatch, rng):
        """Dispatch merges of complementary specialists first (the highest-value
        offers), then one combination offer and the top Thompson picks per parent."""
        plans = []
        for d in dispatch[:1]:
            a, b = d["nodes"]
            plans.append({"parent": a, "operator": "dispatch", "merge_with": b,
                          "applied": sorted(self.tree.applied(a) | self.tree.applied(b) | {"regime-dispatch"}),
                          "knobs": {}, "rationale": f"merge: {a[:8]} wins {d['a_wins'][:3]}, {b[:8]} wins {d['b_wins'][:3]}; "
                                                    f"merged geomean estimate {d['estimated_geomean']}"})
        for m in menus:
            applied = sorted(self.tree.applied(m["parent"]))
            if m["offers"] and rng.random() < 0.3:
                offer = rng.choice(m["offers"])
                plans.append({"parent": m["parent"], "operator": "combine",
                              "applied": applied + [t for t in offer["techniques"] if t not in applied],
                              "knobs": {}, "rationale": f"{offer['kind']} offer"})
            for item in m["menu"][:2]:
                plans.append({"parent": m["parent"], "operator": "mutate", "applied": applied + [item["id"]],
                              "knobs": {}, "rationale": f"Thompson pick (theta {item['theta']}, mean {item['mean']})"})
        return policy.dedupe(plans, self.tree)[:self.nplans]

    def plan_llm(self, menus, dispatch, rnd):
        from ..agents import agent
        context = {"parents": [{**m, "metrics": self.tree.metrics(m["parent"]),
                                "cells": self.tree.cell_speedups(m["parent"]),
                                "profile": _profile_summary(self.tree.nodes[m["parent"]].get("profile")),
                                "tried": [{"applied": d, "outcome": o} for p, c, d, o in self.tree.edges()
                                          if p == m["parent"]]} for m in menus],
                   "dispatch_offers": dispatch}
        text = prompt("planner", family=self.family, plans=self.nplans) + "\n\n" + json.dumps(context, default=str)
        work = self.run_dir / "planner"
        work.mkdir(exist_ok=True)
        record = agent.query(str(work), str(self.run_dir), "supervisor-planner", rnd, text, [], False, 600)
        try:
            raw = record["response"]
            plans = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])["plans"]
        except (ValueError, KeyError, TypeError):
            self.log("planner answer unusable; falling back to Thompson plans")
            return self.plan_thompson(menus, dispatch, policy.rng_for(self.manifest["seed"], rnd))
        valid = [p for p in plans if p.get("parent") in self.tree.nodes]
        for p in valid:
            p["applied"] = sorted(self.tree.applied(p["parent"]) | set(p.get("applied", [])))
        return policy.dedupe(valid, self.tree)[:self.nplans]

    # ------------------------------------------------------------------ workers

    def workspace(self, plan, rnd, slot, failure=None):
        ws = self.run_dir / "workers" / f"r{rnd:03d}-w{slot}"
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True)
        parent = self.tree.nodes[plan["parent"]]
        (ws / self.src).write_text((self.run_dir / parent["source"]).read_text())
        by_id = catalogue.by_id(self.cat)
        menu_by_id = {i["id"]: i for i in self.items}
        new = [t for t in plan["applied"] if t not in self.tree.applied(plan["parent"])]
        (ws / "PLAN.md").write_text(
            f"# Plan\n\nparent: {plan['parent']}\noperator: {plan['operator']}\n"
            f"techniques to add: {', '.join(new) or '(none: see rationale)'}\n"
            f"already applied in the parent: {', '.join(sorted(self.tree.applied(plan['parent']))) or 'none'}\n"
            f"rationale: {plan.get('rationale', '')}\n"
            + (f"\nmerge with node {plan['merge_with']}: its kernel is in OTHER{Path(self.src).suffix}; build one kernel that "
               f"dispatches each regime to the faster path, by proven predicates.\n" if plan.get("merge_with") else ""))
        if plan.get("merge_with"):
            other = self.tree.nodes[plan["merge_with"]]
            (ws / ("OTHER" + Path(self.src).suffix)).write_text((self.run_dir / other["source"]).read_text())
        (ws / "TECHNIQUES.md").write_text("\n\n".join(
            json.dumps({**by_id.get(t, {}), **menu_by_id.get(t, {})}, indent=1) for t in new) or "none")
        spec = self.doc.get("spec", {})
        # Sources go in context/ one file each: inlined, SPEC.md outgrew what a worker
        # can read in one call (x32-transposec: 76 KB with eight upstream kernels).
        sources = sorted(p for p in (self.run_dir / "context").glob("*") if p.is_file())
        (ws / "context").mkdir(exist_ok=True)
        for p in sources:
            shutil.copy(p, ws / "context" / p.name)
        regimes_file = REPO / self.pack / "kernels" / self.family / "regimes.json"
        layout = {k: v for k, v in json.loads(regimes_file.read_text()).items() if k != "heldout"} \
            if regimes_file.exists() else {}
        (ws / "SPEC.md").write_text(
            "# Spec\n\n```json\n" + json.dumps(spec, indent=1) + "\n```\n\n"
            "# Benchmark and check matrix (cells are regime@shape)\n\n```json\n" + json.dumps(layout, indent=1)
            + "\n```\n\n# Reference and upstream kernels\n\n"
            + "".join(f"- context/{p.name} ({len(p.read_text().splitlines())} lines)\n" for p in sources))
        (ws / "PROFILE.md").write_text(_profile_summary(parent.get("profile"), full=True))
        cells = self.tree.cell_speedups(plan["parent"])
        baseline = self.entry.get("baseline", "the fastest correct upstream RVV kernel per cell")
        (ws / "CELLS.md").write_text(f"# Parent speedup per cell (vs {baseline})\n\n"
                                     + "\n".join(f"- {k}: {v:.3f}x" for k, v in sorted(cells.items())))
        (ws / "HISTORY.md").write_text(self.tree.history(plan["parent"]))
        if failure:
            (ws / "FAILURE.md").write_text(failure)
        return ws

    def implement(self, plan, rnd, slot, failure=None):
        from ..agents import agent
        ws = self.workspace(plan, rnd, slot, failure)
        tools = KernelTools(f"kernel_w{slot}", ws, self.gates, self.pack, self.family, f"r{rnd}w{slot}", self.src,
                            base_port=8600)
        try:
            text = prompt("porter" if plan.get("operator") == "port" else "implementer", family=self.family,
                          signature=self.entry.get("signature"),
                          contract=CONTRACTS.get(self.entry.get("signature"), CONTRACTS[None]),
                          parent=plan["parent"], minutes=self.implementer_minutes, source=self.src,
                          language=LANGUAGES.get(self.src, LANGUAGES["kernel.c"]))
            if failure:
                text += "\n\n" + prompt("debugger") + "\n\n" + failure
            record = agent.query(str(ws), str(self.run_dir), "worker", rnd * 100 + slot, text, [tools], True,
                                 self.implementer_minutes * 60)
        finally:
            tools.stop()
        return ws, record

    def harvest(self, plan, ws, record, operator=None):
        text = (ws / self.src).read_text()
        try:
            declared = json.loads((ws / "TRANSFORMS.json").read_text())
            if not isinstance(declared.get("applied"), list):
                raise ValueError
        except (OSError, ValueError, AttributeError):
            declared = {"applied": plan["applied"], "note": "TRANSFORMS.json missing or invalid; plan used"}
        declared["applied"] = sorted(set(declared["applied"]) | self.tree.applied(plan["parent"]))
        declared.setdefault("knobs", plan.get("knobs", {}))
        declared["plan"] = {k: plan.get(k) for k in ("operator", "rationale", "merge_with")}
        same = self.index.same_source(text)
        if same:
            self.log(f"duplicate of {same[:10]}: not evaluated")
            return None
        node_id, rel = self.store_source(text)
        parents = [plan["parent"]] + ([plan["merge_with"]] if plan.get("merge_with") else [])
        _, created = self.tree.add_node(node_id, parents, operator or plan["operator"], declared, rel)
        if (ws / "NOTES.md").exists():
            (self.run_dir / "nodes" / node_id / "NOTES.md").write_text((ws / "NOTES.md").read_text())
        self.index.add(node_id, text)
        return node_id if created else None

    # ------------------------------------------------------------------ rounds

    def round(self, plans=None):
        rnd = self.manifest["rounds"] + 1
        lost = [n for n, node in self.tree.nodes.items() if node["status"] == "infra-error"]
        if lost:  # a gate call died (host/network); the sources are stored, so evaluate again
            self.log(f"re-evaluating {len(lost)} node(s) lost to infra errors")
            for n in lost:
                self.tree.append({"type": "status", "id": n, "status": "pending", "note": "retry after infra error"})
            self.evaluate(lost)
        rng = policy.rng_for(self.manifest["seed"], rnd)
        menus, dispatch = ([], []) if plans is not None else self.menus(rng)
        if plans is None:
            plans = (self.plan_llm(menus, dispatch, rnd) if self.planner == "llm"
                     else self.plan_thompson(menus, dispatch, rng))
        (self.run_dir / "plans").mkdir(exist_ok=True)
        (self.run_dir / "plans" / f"round-{rnd:03d}.json").write_text(
            json.dumps({"menus": menus, "dispatch": dispatch, "plans": plans}, indent=1, default=str) + "\n")
        self.log(f"round {rnd}: {len(plans)} plans from {len(menus)} parents")
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            done = list(pool.map(lambda ip: (ip[1], *self.implement(ip[1], rnd, ip[0])), enumerate(plans)))
        new = [n for plan, ws, rec in done for n in [self.harvest(plan, ws, rec)] if n]
        self.evaluate(new)
        retry = [(plan, ws) for plan, ws, rec in done
                 for n in [self.tree.nodes.get(sha((ws / self.src).read_text()))]
                 if n and n["status"].startswith(("failed@G0", "failed@G1", "failed@G2"))]
        if retry:
            fixes = []
            for slot, (plan, ws) in enumerate(retry):
                node = self.tree.nodes[sha((ws / self.src).read_text())]
                failure = json.dumps({g: node["gates"][g] for g in node["gates"] if not node["gates"][g]["passed"]},
                                     indent=1, default=str)[:6000]
                fix_plan = {**plan, "parent": plan["parent"]}
                ws2, rec2 = self.implement(fix_plan, rnd, 50 + slot, failure=failure)
                n2 = self.harvest(fix_plan, ws2, rec2, operator="debug")
                if n2:
                    fixes.append(n2)
            self.evaluate(fixes)
        self.manifest["rounds"] = rnd
        self.save_manifest()
        return self.status(f"round {rnd} done: {len(new)} new nodes")

    def certify(self, top=3):
        """G5 for the best nodes: a second bench session in a new lease, held-out
        regimes (seed never shown to workers) under QEMU and on the X280, and SMT
        where the family has a prepped Neon source. Recorded as G5 gate events."""
        front = self.tree.frontier(objective.spec())
        targets = list(dict.fromkeys(front.get("frontier", []) + front.get("pareto", [])))[:top]
        for n in targets:
            sessions_before = len(self.tree.nodes[n]["perf"])
            self.evaluate([n])                                  # rebuilds the tree with n, second session
            tag = f"camp-{self.run_dir.name}"[:40]
            r = self.gates.call("certify", {"pack": self.pack, "family": self.family, "tag": tag,
                                            "kernel": self.ukernel(n), "heldout_seed": self.manifest["heldout_seed"],
                                            "source": f"{self.gates.run_rel}/nodes/{n}/{self.src}",
                                            "out": f"{self.gates.run_rel}/certify/{n[:10]}"})
            held = (r.get("heldout") or {}).get("passed", False)
            smt = r.get("smt")
            perf = self.tree.nodes[n]["perf"]
            stable = None
            if len(perf) > sessions_before and sessions_before:
                a = perf[sessions_before - 1]["verdict"]["metrics"]["speedup_geomean"]
                b = perf[-1]["verdict"]["metrics"]["speedup_geomean"]
                stable = abs(a / b - 1) <= max(self.delta * 2, 0.01)
            passed = bool(r.get("ok") and held and stable is not False and (smt is None or smt.get("passed")))
            self.tree.append({"type": "gate", "id": n, "gate": "G5", "passed": passed,
                              "detail": {"heldout": r.get("heldout"), "smt": smt, "sessions_agree": stable,
                                         "error": r.get("error")}})
            if passed:
                self.tree.append({"type": "status", "id": n, "status": "certified"})
        return self.status("certified" if targets else "nothing to certify")

    PORT_APPROACHES = [
        "follow the structure of the arm64 implementation in context/ (its decomposition into "
        "passes and helper functions), translated to length-agnostic RVV",
        "follow the structure of the x86 implementation in context/, translated to length-agnostic RVV",
        "design directly from the C reference for RVV: strip-mine each row with vsetvli, no other "
        "ISA's structure",
        "lowest risk first: the simplest correct vector structure (wide lanes, no tricks), then "
        "measure-driven tuning once kernel_check is green",
    ]

    def correct_nodes(self):
        return [n for n, node in self.tree.nodes.items() if node["operator"] != "placeholder"
                and all(node["gates"].get(g, {}).get("passed") for g in ("G0", "G1", "G2"))]

    def port(self, ports=4, max_rounds=2):
        """Porting stage for families with no RVV yet: independent ports of the
        placeholder, each told a different approach, until one passes G0-G2. The
        ports become the roots the search optimizes from."""
        placeholder = [n for n, node in self.tree.nodes.items() if node["operator"] == "placeholder"]
        if not placeholder:
            return None
        for _ in range(max_rounds):
            if self.correct_nodes() or time.time() >= self.deadline:
                break
            k = self.manifest.get("port_rounds", 0)
            plans = [{"parent": placeholder[0], "operator": "port", "applied": [], "knobs": {},
                      "rationale": self.PORT_APPROACHES[(k * ports + i) % len(self.PORT_APPROACHES)]}
                     for i in range(ports)]
            self.manifest["port_rounds"] = k + 1
            self.round(plans=plans)
        return self.status(f"ports: {len(self.correct_nodes())} correct")

    def run(self, patience=3):
        best, stale = None, 0
        while self.manifest["rounds"] < self.max_rounds and time.time() < self.deadline:
            status = self.round()
            score = (status.get("best_metrics") or {}).get("speedup_geomean")
            if best is None or (score and score > best * (1 + 0.01)):
                best, stale = score, 0
            else:
                stale += 1
            if stale >= patience:
                self.log(f"no frontier gain above 1% for {patience} rounds: stopping")
                break
            if self.board_used >= self.board_budget:
                self.log("board budget spent: stopping")
                break
            if (self.run_dir / "STOP").exists():  # graceful stop between rounds: nothing is lost
                (self.run_dir / "STOP").unlink()
                self.log("STOP file found: stopping after a complete round")
                return self.status("stopped (STOP file); resume with the same command")
        return self.status("finished")


def _profile_summary(p, full=False):
    if not p:
        return "no profile yet"
    lines = [f"calls: {p.get('calls') or 'none'}; text bytes {p.get('text_bytes')}; bound {p.get('bound')}; "
             f"descriptors {p.get('descriptors')}"]
    preds = {x.get("start"): x for x in p.get("predicted", []) if x}
    for h in p.get("loops", []):
        pr = preds.get(h["start"], {})
        lines.append(f"loop {h['start']}: {h['counts']}, vsetvl {h['vsetvl']}, spills {h['spills']}"
                     + (f", predicted {pr.get('cycles_per_elem', 0):.3f} c/elem" if pr.get("cycles_per_elem") else ""))
    return "\n".join(lines if full else lines[:4])
