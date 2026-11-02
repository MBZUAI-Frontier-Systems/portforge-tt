"""Choosing what to try next: parents, technique menus, combination offers.

Nothing is thrown away. Every technique keeps a Beta posterior per context
(signature, bound class, LMUL bucket, dispatch) built from the catalogue prior
and the outcomes the tree records; menus are drawn by Thompson sampling with an
exploration floor, so a technique that lost under one parent is still offered
(less often) under others, and comes back in full when a new frontier parent
changes the context. On top of single techniques the policy offers:

  - synergy pairs from the catalogue (one applied in the parent, one not),
  - "mixed"/"neutral" techniques re-tried together with an untried one,
  - dispatch merges of two nodes whose winning cells are complementary
    (pf2's regime dispatch, found systematically instead of by hand).

Parents come from the frontier (60%), MAP-Elites archive elites (25%) and
per-cell specialists (15%), so a specialist that loses overall but wins some
cells stays a parent.
"""
import json
import math
import random

from . import catalogue, objective

OUTCOME_UPDATE = {  # (alpha, beta) increments for the technique the edge added
    "improve": (1.0, 0.0), "mixed": (0.5, 0.5), "neutral": (0.0, 1.0),
    "regress": (0.0, 1.0), "wrong": (0.0, 1.0), "impl_fail": (0.0, 0.25),
}
POOL_WEIGHT = 0.3     # how much evidence from other contexts counts here
FLOOR = 0.02          # every applicable technique stays offerable with at least this weight


def context(tree, node_id, signature=None):
    """Context key of a parent: (signature, bound, LMUL bucket, dispatch)."""
    p = (tree.nodes[node_id].get("profile") or {})
    d = p.get("descriptors", {})
    bound = p.get("bound")
    lmul = d.get("lmul")
    bucket = None if lmul is None else ("low" if lmul <= 2 else "high")
    return (signature, bound, bucket, bool(d.get("dispatch")))


def posteriors(tree, menu, signature=None):
    """{(technique, context): [alpha, beta]} and pooled {technique: [alpha, beta]} from outcomes."""
    ctx_counts, pooled = {}, {}
    for parent, child, delta, outcome in tree.edges():
        if outcome is None or parent not in tree.nodes:
            continue
        a, b = OUTCOME_UPDATE[outcome]
        ctx = context(tree, parent, signature)
        for tid in delta:
            for table, key in ((ctx_counts, (tid, ctx)), (pooled, tid)):
                cell = table.setdefault(key, [0.0, 0.0])
                cell[0] += a
                cell[1] += b
    return ctx_counts, pooled


def beta_params(item, ctx, ctx_counts, pooled):
    prior = item.get("prior", {"p_improve": 0.2, "strength": 1})
    a = prior["p_improve"] * prior["strength"]
    b = (1 - prior["p_improve"]) * prior["strength"]
    pa, pb = pooled.get(item["id"], (0.0, 0.0))
    ca, cb = ctx_counts.get((item["id"], ctx), (0.0, 0.0))
    return max(a + POOL_WEIGHT * pa + ca, 1e-3), max(b + POOL_WEIGHT * pb + cb, 1e-3)


def menu(tree, parent, items, rng, keep=0.3, minimum=3, signature=None):
    """Dropout-masked, Thompson-ranked menu for one parent.

    Items already applied in the parent are left out. Each remaining item draws
    theta ~ Beta(posterior); the top `keep` fraction plus random survivors of a
    keep-probability mask are offered (Autocomp-style dropout), never fewer than
    `minimum`, and every item has at least FLOOR chance to appear.
    """
    applied = tree.applied(parent)
    ctx = context(tree, parent, signature)
    ctx_counts, pooled = posteriors(tree, items, signature)
    profile = tree.nodes[parent].get("profile")
    scored = []
    for item in items:
        if item["id"] in applied or item.get("detect") == "False" and item.get("category") == "offload":
            continue
        # a fix whose detect already holds on the parent's code has nothing to fix
        # (avoid-libcalls on a kernel without calls); presence techniques such as
        # regime-dispatch stay offered, there may be more of them to gain
        if profile and item.get("kind") == "fix" and catalogue.detects(item, profile):
            continue
        a, b = beta_params(item, ctx, ctx_counts, pooled)
        scored.append((rng.betavariate(a, b), item, a / (a + b)))
    scored.sort(key=lambda s: -s[0])
    top = max(minimum, int(round(keep * len(scored))))
    chosen = scored[:top] + [s for s in scored[top:] if rng.random() < max(FLOOR, keep / 3)]
    return [{"id": item["id"], "title": item.get("title"), "theta": round(theta, 3), "mean": round(mean, 3)}
            for theta, item, mean in chosen]


def parents(tree, beam, rng, chosen=None):
    """Up to `beam` distinct parents: frontier 60%, archive 25%, specialists 15%."""
    chosen = chosen or objective.spec()
    front = tree.frontier(chosen)
    pool = {
        "frontier": list(front.get("frontier", [])) + list(front.get("pareto", [])),
        "archive": list(tree.archive(chosen).values()),
        "specialist": sorted(set(tree.specialists().values())),
    }
    if not any(pool.values()):
        seeds = [n for n, node in tree.nodes.items() if node["operator"] in ("seed", "import", "port")
                 and tree.correct(n)]
        pool["frontier"] = seeds or [n for n, node in tree.nodes.items() if node["operator"] == "seed"]
    picked = []
    for _ in range(beam * 4):
        if len(picked) >= beam:
            break
        r = rng.random()
        source = "frontier" if r < 0.6 else "archive" if r < 0.85 else "specialist"
        options = [n for n in pool[source] if n not in [p for p, _ in picked]] or \
                  [n for s in pool.values() for n in s if n not in [p for p, _ in picked]]
        if not options:
            break
        picked.append((rng.choice(options), source))
    return picked


def combination_offers(tree, parent, items, limit=4):
    """Pairs worth trying together on this parent."""
    by_id = {i["id"]: i for i in items}
    applied = tree.applied(parent)
    offers = []
    for item in items:
        if item["id"] not in applied:
            continue
        for other in item.get("interactions", {}).get("synergy", []):
            if other not in applied and other in by_id:
                offers.append({"kind": "synergy", "techniques": [item["id"], other]})
    soft = {tid for p, c, delta, out in tree.edges() if p == parent and out in ("mixed", "neutral") for tid in delta}
    untried = [i["id"] for i in items if i["id"] not in applied and i["id"] not in soft]
    for tid in sorted(soft):
        for other in untried[:2]:
            offers.append({"kind": "retry-in-combination", "techniques": [tid, other]})
    seen, out = set(), []
    for o in offers:
        key = frozenset(o["techniques"])
        if key not in seen:
            seen.add(key)
            out.append(o)
    return out[:limit]


def dispatch_offers(tree, delta=0.005, limit=3):
    """Pairs of measured correct nodes whose winning cells are complementary:
    a regime-dispatch merge of the two could win both sets."""
    measured = tree.measured()
    speed = {n: tree.cell_speedups(n) for n in measured}
    offers = []
    for i, a in enumerate(measured):
        for b in measured[i + 1:]:
            shared = [k for k in speed[a] if k in speed[b]]
            a_wins = [k for k in shared if speed[a][k] > speed[b][k] * (1 + delta)]
            b_wins = [k for k in shared if speed[b][k] > speed[a][k] * (1 + delta)]
            if a_wins and b_wins:
                # what a merged kernel taking each cell's better path would score
                merged = math.exp(sum(math.log(max(speed[a][k], speed[b][k])) for k in shared) / len(shared))
                offers.append({"kind": "dispatch", "nodes": [a, b], "a_wins": a_wins[:6], "b_wins": b_wins[:6],
                               "estimated_geomean": round(merged, 3)})
    offers.sort(key=lambda o: -o["estimated_geomean"])
    return offers[:limit]


def plan_key(parent, applied, knobs):
    return (parent, frozenset(applied), json.dumps(knobs or {}, sort_keys=True))


def dedupe(plans, tree):
    """Drop plans whose (parent, technique set, knobs) was already tried."""
    tried = tree.tried_index()
    out, seen = [], set()
    for plan in plans:
        key = plan_key(plan["parent"], plan.get("applied", []), plan.get("knobs"))
        if key in tried or key in seen:
            continue
        seen.add(key)
        out.append(plan)
    return out


def rng_for(run_seed, step):
    return random.Random(run_seed * 1000003 + step)
