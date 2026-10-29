"""portforge_tt.kernel: objective, tree, policy, novelty. No hardware or agents."""
import unittest

from portforge_tt.kernel import objective


class ObjectiveTests(unittest.TestCase):
    def test_default_is_primary_speedup_with_no_regression(self):
        chosen = objective.spec()
        self.assertEqual(chosen["primary"], "speedup_geomean")
        self.assertEqual(chosen["constraints"], [dict(metric="flagged_cells", op="<=", value=0.0, relative=False)])

    def test_rank_drops_regressions_and_reports_pareto(self):
        chosen = objective.spec()
        points = [
            {"id": "pf1", "metrics": {"speedup_geomean": 1.00, "worst_cell": 0.997, "flagged_cells": 0, "text_bytes": 124}},
            {"id": "pf2", "metrics": {"speedup_geomean": 1.26, "worst_cell": 0.816, "flagged_cells": 1, "text_bytes": 450}},
            {"id": "pf3", "metrics": {"speedup_geomean": 1.20, "worst_cell": 0.994, "flagged_cells": 0, "text_bytes": 500}},
        ]
        r = objective.rank(points, chosen)
        self.assertEqual(r["frontier"], ["pf3"])
        self.assertIn("pf2", r["infeasible"])
        self.assertEqual(sorted(r["pareto"]), ["pf1", "pf3"])   # pf1 is smaller, pf3 faster
        # a 0.994 point estimate is noise unless the protocol flagged the cell

    def test_metric_vector_from_receipts(self):
        verdict = {"metrics": {"speedup_geomean": 1.1, "worst_cell": 0.99, "cycles_per_elem": 0.3}, "flagged": []}
        check = {"qemu:128": {"max_err": "0.000000"}, "qemu:1024": {"max_err": "0.000000"}, "passed": True}
        m = objective.cpu_metrics(verdict, check, {"text_bytes": 124})
        self.assertEqual(m, {"speedup_geomean": 1.1, "worst_cell": 0.99, "cycles_per_elem": 0.3, "flagged_cells": 0,
                             "max_err": 0.0, "text_bytes": 124})
        self.assertNotIn("max_err", objective.cpu_metrics(verdict, {"x": {"max_err": "inf"}}))

    def test_bad_specs(self):
        with self.assertRaises(ValueError):
            objective.spec(primary="p99_latency")
        with self.assertRaises(ValueError):
            objective.spec(constraints=["worst_cell>>1"])


import random
import tempfile
import time

from portforge_tt.kernel import novelty, policy, tree as tree_mod


def verdict(cells, worst=None):
    speeds = list(cells.values())
    g = 1.0
    for s in speeds:
        g *= s
    g **= 1 / len(speeds)
    return {"cells": {k: {"speedup": {"s": s, "lo": s * .99, "hi": s * 1.01}} for k, s in cells.items()},
            "metrics": {"speedup_geomean": g, "worst_cell": worst if worst is not None else min(speeds),
                        "cycles_per_elem": 0.3},
            "flagged": [k for k, s in cells.items() if s * 1.01 < 1 and s < 0.995]}


PASS_CHECK = {"qemu:128": {"max_err": "0.000000"}, "passed": True}


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = tree_mod.Tree(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def node(self, sha, parents, applied, cells=None, gates=("G0", "G1", "G2"), fail=None, profile=None):
        self.t.add_node(sha, parents, "mutate" if parents else "seed", {"applied": applied}, f"nodes/{sha}/kernel.c")
        for g in gates:
            self.t.append({"type": "gate", "id": sha, "gate": g, "passed": g != fail,
                           "detail": PASS_CHECK if g == "G2" else {}})
        if profile:
            self.t.append({"type": "profile", "id": sha, "profile": profile})
        if cells and fail is None:
            self.t.append({"type": "perf", "id": sha, "session": "s1", "verdict": verdict(cells)})

    def test_outcomes_and_replay(self):
        self.node("p", [], ["narrow-sew-chain"], {"a@1": 1.0, "b@1": 1.0})
        self.node("up", ["p"], ["narrow-sew-chain", "regime-dispatch"], {"a@1": 1.2, "b@1": 1.1})
        self.node("mix", ["p"], ["narrow-sew-chain", "load-ahead"], {"a@1": 1.2, "b@1": 0.9})
        self.node("slow", ["p"], ["narrow-sew-chain", "max-lmul"], {"a@1": 0.95, "b@1": 0.97})
        self.node("bad", ["p"], ["narrow-sew-chain", "fold-bias-scalar"], {"a@1": 2.0, "b@1": 2.0}, fail="G2")
        self.node("nocompile", ["p"], ["vnclip-requant"], fail="G0", gates=("G0",))
        self.t.add_node("pending", ["p"], "mutate", {"applied": ["narrow-sew-chain", "avoid-libcalls"]}, "x")
        out = {c: o for _, c, _, o in self.t.edges()}
        self.assertEqual(out, {"up": "improve", "mix": "mixed", "slow": "regress", "bad": "wrong",
                               "nocompile": "impl_fail", "pending": None})
        self.assertEqual(self.t.delta("up"), ["regime-dispatch"])
        replay = tree_mod.Tree(self.tmp.name)
        self.assertEqual({c: o for _, c, _, o in replay.edges()}, out)
        self.assertEqual(replay.nodes["bad"]["status"], "failed@G2")

    def test_frontier_specialists_archive_and_dedup(self):
        prof = {"descriptors": {"lmul": 8, "sew": 16, "unroll": "1", "dispatch": False}}
        self.node("p", [], [], {"a@1": 1.0, "b@1": 1.0}, profile=prof)
        self.node("fast_a", ["p"], ["single-op-regime"], {"a@1": 2.0, "b@1": 0.9}, profile=prof)
        self.node("even", ["p"], ["fold-bias-scalar"], {"a@1": 1.1, "b@1": 1.1},
                  profile={"descriptors": {"lmul": 8, "sew": 16, "unroll": "1", "dispatch": True}})
        self.assertEqual(self.t.frontier()["frontier"], ["even"])    # fast_a breaks no-regression
        self.assertEqual(self.t.specialists(), {"a@1": "fast_a", "b@1": "even"})
        self.assertEqual(len(self.t.archive()), 2)
        plans = [{"parent": "p", "applied": ["fold-bias-scalar"]}, {"parent": "p", "applied": ["max-lmul"]},
                 {"parent": "p", "applied": ["max-lmul"]}]
        self.assertEqual(policy.dedupe(plans, self.t), [{"parent": "p", "applied": ["max-lmul"]}])
        offers = policy.dispatch_offers(self.t)
        self.assertEqual(offers[0]["nodes"], ["fast_a", "even"])
        picked = policy.parents(self.t, 3, random.Random(0))
        self.assertIn("even", [p for p, _ in picked])

    def test_flag_counts_only_when_confirmed(self):
        self.node("k", [], [], {"a@1": 1.5, "b@1": 0.99})
        self.assertEqual(self.t.metrics("k")["flagged_cells"], 1)          # suspicion until re-measured
        self.assertEqual(self.t.frontier()["infeasible"], {"k": ["flagged_cells=1 violates <= 0"]})
        self.t.append({"type": "perf", "id": "k", "session": "s2", "verdict": verdict({"a@1": 1.5, "b@1": 0.998})})
        self.assertEqual(self.t.metrics("k")["flagged_cells"], 0)          # not reproduced in a new session
        self.assertEqual(self.t.frontier()["frontier"], ["k"])
        self.t.append({"type": "perf", "id": "k", "session": "s3", "verdict": verdict({"a@1": 1.5, "b@1": 0.98})})
        self.t.append({"type": "perf", "id": "k", "session": "s4", "verdict": verdict({"a@1": 1.5, "b@1": 0.97})})
        self.assertEqual(self.t.confirmed_flags("k"), ["b@1"])

    def test_history_spans_the_tree_and_keeps_rejections(self):
        self.node("p", [], [], {"a@1": 1.0, "b@1": 1.0})
        self.node("x", ["p"], ["segment-stores"], {"a@1": 1.3, "b@1": 0.5})
        self.node("q", [], [], {"a@1": 1.0, "b@1": 1.0})
        self.node("y", ["q"], ["segment-stores"], {"a@1": 1.2, "b@1": 0.6})
        h = self.t.history("q")
        self.assertIn("segment-stores [mutate] -> mixed", h)
        self.assertIn("wins: a@1 1.20x; losses: b@1 0.60x", h)
        # the technique line pools both edges (x under p and y under q)
        self.assertIn("- segment-stores: mixed 2", h)
        self.assertIn("tends to win: a@1 1.25x; to lose: b@1 0.55x", h)
        self.assertIn("- a@1: x ", h)

    def test_declare_adds_detected_techniques(self):
        self.node("s", [], [], {"a@1": 1.0})
        self.t.append({"type": "declare", "id": "s", "applied": ["segment-loads"]})
        self.assertEqual(self.t.applied("s"), {"segment-loads"})
        self.assertEqual(tree_mod.Tree(self.tmp.name).applied("s"), {"segment-loads"})


class PolicyTests(unittest.TestCase):
    ITEMS = [{"id": t, "title": t, "prior": {"p_improve": 0.3, "strength": 2},
              "interactions": {"synergy": ["fold-bias-scalar"] if t == "regime-dispatch" else []}}
             for t in ("regime-dispatch", "fold-bias-scalar", "load-ahead", "max-lmul", "avoid-libcalls")]

    def tree_with(self, outcomes):
        tmp = tempfile.mkdtemp()
        t = tree_mod.Tree(tmp)
        t.add_node("p", [], "seed", {"applied": []}, "x")
        for g in ("G0", "G1", "G2"):
            t.append({"type": "gate", "id": "p", "gate": g, "passed": True, "detail": PASS_CHECK if g == "G2" else {}})
        t.append({"type": "perf", "id": "p", "session": "s", "verdict": verdict({"a@1": 1.0})})
        for i, (tid, s) in enumerate(outcomes):
            c = f"c{i}"
            t.add_node(c, ["p"], "mutate", {"applied": [tid]}, "x")
            for g in ("G0", "G1", "G2"):
                t.append({"type": "gate", "id": c, "gate": g, "passed": True, "detail": PASS_CHECK if g == "G2" else {}})
            t.append({"type": "perf", "id": c, "session": "s", "verdict": verdict({"a@1": s})})
        return t

    def test_rejected_technique_stays_offerable(self):
        t = self.tree_with([("load-ahead", 0.9)] * 4)
        counts = {}
        for seed in range(400):
            for item in policy.menu(t, "p", self.ITEMS, random.Random(seed)):  # rejected under p, 4 times
                counts[item["id"]] = counts.get(item["id"], 0) + 1
        self.assertGreater(counts.get("load-ahead", 0), 0)           # never thrown away
        ctx_counts, pooled = policy.posteriors(t, self.ITEMS)
        self.assertEqual(pooled["load-ahead"], [0.0, 4.0])

    def test_success_raises_offer_rate(self):
        good = self.tree_with([("max-lmul", 1.3)] * 3)
        bad = self.tree_with([("max-lmul", 0.8)] * 3)

        def rate(t):
            return sum(any(i["id"] == "max-lmul" for i in policy.menu(t, "p", self.ITEMS, random.Random(s), keep=0.2,
                                                                       minimum=1)) for s in range(300))
        # parent "p" applies nothing, so max-lmul is offered under it in both trees
        self.assertGreater(rate(good), rate(bad))

    def test_synergy_offer(self):
        t = self.tree_with([("regime-dispatch", 1.2)])
        offers = policy.combination_offers(t, "c0", self.ITEMS)
        self.assertIn({"kind": "synergy", "techniques": ["regime-dispatch", "fold-bias-scalar"]}, offers)


class NoveltyTests(unittest.TestCase):
    def test_source_key_ignores_comments_whitespace_and_name(self):
        a = "#include <x.h>\n// fast\nvoid xnn_a__rvv_pf1(int n) {  return;  }\n"
        b = "/* other */ void   xnn_b__rvv_pf9(int n) { return; }"
        self.assertEqual(novelty.source_key(a), novelty.source_key(b))
        self.assertNotEqual(novelty.source_key(a), novelty.source_key(a.replace("return;", "n++;")))

    def test_loop_key(self):
        p = {"loops": [{"counts": {"load": 1, "alu": 2}, "vsetvl": ["e8,m4"]}]}
        q = {"loops": [{"counts": {"alu": 2, "load": 1}, "vsetvl": ["e8,m4"]}]}
        idx = novelty.Index()
        idx.add("n1", "void xnn_k(void){}", p)
        self.assertEqual(idx.same_loops(q), "n1")
        self.assertIsNone(idx.same_loops({"loops": []}))


class FakeGates:
    run_rel = "runs/test"

    def put(self, local, remote_rel):
        return remote_rel

    def call(self, command, args, timeout=0):
        raise AssertionError("no remote calls in unit tests")


class CampaignTests(unittest.TestCase):
    def setUp(self):
        import json as _json
        from pathlib import Path as _Path
        from portforge_tt.kernel import orchestrator
        self.orch = orchestrator
        self.tmp = tempfile.TemporaryDirectory()
        root = _Path(__file__).resolve().parents[1]
        entry = _json.loads((root / "ports/xnnpack-rvv/families.json").read_text())["families"]["qs8-vcvt"]
        doc = _json.loads((root / "ports/xnnpack-rvv/kernels/qs8-vcvt/TECHNIQUES.json").read_text())
        self.camp = orchestrator.Campaign(_Path(self.tmp.name) / "run", "ports/xnnpack-rvv", "qs8-vcvt", FakeGates(),
                                          entry, doc, plans=3, seed=5, log=lambda *a: None)
        t = self.camp.tree
        for name, cells in (("a", {"x@1": 1.0, "y@1": 2.0}), ("b", {"x@1": 1.3, "y@1": 1.0})):
            node_id, rel = self.camp.store_source(f"void xnn_k_{name}(void) {{ /* {name} */ }}\n")
            t.add_node(node_id, [], "seed", {"applied": ["narrow-sew-chain"]}, rel)
            for g in ("G0", "G1", "G2"):
                t.append({"type": "gate", "id": node_id, "gate": g, "passed": True,
                          "detail": PASS_CHECK if g == "G2" else {}})
            t.append({"type": "perf", "id": node_id, "session": "s", "verdict": verdict(cells)})
            setattr(self, name, node_id)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dispatch_plan_comes_first(self):
        rng = random.Random(0)
        menus, dispatch = self.camp.menus(rng)
        plans = self.camp.plan_thompson(menus, dispatch, rng)
        self.assertEqual(plans[0]["operator"], "dispatch")
        self.assertEqual(sorted([plans[0]["parent"], plans[0]["merge_with"]]), sorted([self.a, self.b]))
        self.assertLessEqual(len(plans), 3)

    def test_flagged_node_gets_a_confirmation_session(self):
        calls = []

        class Scripted(FakeGates):
            def call(self, command, args, timeout=0):
                calls.append((command, args))
                if command == "gates":
                    return {"ok": True, "results": {k: {g: {"passed": True} for g in ("G0", "G1", "G2")}
                                                    for k in args["kernels"]}}
                cells = {"x@1": 1.2, "y@1": 0.98} if len(calls) == 2 else {"x@1": 1.2, "y@1": 1.0}
                return {"ok": True, "receipt": {"seed": args["seed"]},
                        "verdicts": {k: {**verdict(cells), "geomean": None, "wrong_cells": [], "correctness_wins": [],
                                         "feasible": True, "reference_cell": "x@1", "delta": 0.005}
                                     for k in args["kernels"]}}

        self.camp.gates = Scripted()
        node_id, rel = self.camp.store_source("void xnn_k_c(void) { /* c */ }\n")
        self.camp.tree.add_node(node_id, [self.a], "mutate", {"applied": []}, rel)
        self.camp.evaluate([node_id])
        benches = [a for c, a in calls if c == "bench"]
        self.assertEqual([a["rounds"] for a in benches], [15, 30])
        self.assertNotEqual(benches[0]["seed"], benches[1]["seed"])
        self.assertEqual(len(self.camp.tree.nodes[node_id]["perf"]), 2)
        self.assertEqual(self.camp.tree.metrics(node_id)["flagged_cells"], 0)

    def test_budgets_survive_a_restart_and_stop_file(self):
        self.camp.board_used = 120.0
        self.camp.save_manifest()
        again = self.orch.Campaign(self.camp.run_dir, "ports/xnnpack-rvv", "qs8-vcvt", FakeGates(), self.camp.entry,
                                   self.camp.doc, plans=3, hours=1.0, log=lambda *a: None)
        self.assertEqual(again.board_used, 120.0)
        self.assertLessEqual(again.deadline, time.time() + 3600)
        self.assertGreaterEqual(again.manifest["active_seconds"], 0.0)
        calls = []
        again.round = lambda: calls.append(1) or {"best_metrics": None}
        (again.run_dir / "STOP").write_text("")
        self.assertIn("STOP", again.run()["note"])
        self.assertEqual(calls, [1])
        self.assertFalse((again.run_dir / "STOP").exists())

    def test_workspace_and_harvest(self):
        plan = {"parent": self.a, "operator": "dispatch", "merge_with": self.b, "rationale": "r",
                "applied": ["narrow-sew-chain", "regime-dispatch"], "knobs": {}}
        ws = self.camp.workspace(plan, 1, 0)
        for f in ("kernel.c", "PLAN.md", "SPEC.md", "PROFILE.md", "CELLS.md", "TECHNIQUES.md", "HISTORY.md", "OTHER.c"):
            self.assertTrue((ws / f).exists(), f)
        self.assertLess(len((ws / "SPEC.md").read_text()), 20000)
        self.assertIn("Leader of every cell", (ws / "HISTORY.md").read_text())
        self.assertIn("regime-dispatch", (ws / "TECHNIQUES.md").read_text())
        self.assertIsNone(self.camp.harvest(plan, ws, {}))            # unchanged source: duplicate, not evaluated
        (ws / "kernel.c").write_text("void xnn_k_c(int n) { n++; }\n")
        (ws / "TRANSFORMS.json").write_text("not json")
        node = self.camp.harvest(plan, ws, {})
        n = self.camp.tree.nodes[node]
        self.assertEqual(n["parents"], [self.a, self.b])
        self.assertEqual(n["operator"], "dispatch")
        self.assertEqual(n["declared"]["applied"], ["narrow-sew-chain", "regime-dispatch"])
        self.assertIn("plan used", n["declared"]["note"])
