"""Markdown report of a kernel campaign from its trusted events only."""
import json
from pathlib import Path

from . import objective, tree as tree_mod


def lineage(t, node_id):
    chain, seen = [], set()
    while node_id and node_id not in seen:
        seen.add(node_id)
        node = t.nodes[node_id]
        chain.append(node_id)
        node_id = node["parents"][0] if node["parents"] else None
    return list(reversed(chain))


def technique_table(t):
    rows = {}
    for parent, child, delta, outcome in t.edges():
        for tid in delta or ["(none)"]:
            row = rows.setdefault(tid, {"tries": 0, "outcomes": {}, "best": None})
            row["tries"] += 1
            row["outcomes"][outcome or "pending"] = row["outcomes"].get(outcome or "pending", 0) + 1
            if outcome in ("improve", "mixed", "neutral", "regress"):
                g = (t.metrics(child) or {}).get("speedup_geomean")
                if g and (row["best"] is None or g > row["best"]):
                    row["best"] = g
    return rows


def render(run_dir):
    run_dir = Path(run_dir)
    manifest = Path(run_dir) / "manifest.json"
    noise = json.loads(manifest.read_text()).get("delta", 0.005) if manifest.exists() else 0.005
    t = tree_mod.Tree(run_dir, noise=noise)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    chosen = objective.spec()
    front = t.frontier(chosen)
    views_why = {}
    for n in t.measured():
        for k, vv in (t.latest_verdict(n).get("views") or {}).items():
            views_why.setdefault(k, vv.get("why", k))
    lines = [f"# Kernel campaign {run_dir.name}: {manifest['family']}", "",
             f"Planner {manifest['planner']}, {manifest['rounds']} rounds, {len(t.nodes)} nodes "
             f"({t.summary()['status']}).", "",
             "## Measured correct nodes", "",
             "Geomean and worst cell are against the fastest correct upstream kernel per cell (the "
             "objective). Other baselines from the same timing rows: "
             + "; ".join(f"**{k}** = {w}" for k, w in views_why.items()) if views_why else "",
             "", "| node | operator | techniques | geomean (CI) | worst cell | "
             + "".join(f"vs {k} | " for k in views_why) + "cycles/elem | text bytes | feasible |",
             "|---|---|---|---|---|" + "---|" * len(views_why) + "---|---|---|"]
    for n in sorted(t.measured(), key=lambda n: -(t.metrics(n).get("speedup_geomean") or 0)):
        m, node = t.metrics(n), t.nodes[n]
        v = t.latest_verdict(n)
        vw = v.get("views", {})
        ci = v.get("metrics", {}).get("speedup_geomean_ci") or [None, None]
        ok = objective.feasible(m, chosen["constraints"])[0]
        star = " **frontier**" if n in front.get("frontier", []) else (" pareto" if n in front.get("pareto", []) else "")
        lines.append(f"| `{n[:10]}`{star} | {node['operator']} | {', '.join(node['declared'].get('applied', [])) or '-'} "
                     f"| {m.get('speedup_geomean', 0):.3f} ({(ci[0] or 0):.3f}-{(ci[1] or 0):.3f}) "
                     f"| {m.get('worst_cell', 0):.3f} | "
                     + "".join(f"{vw[k]['speedup_geomean']:.3f} ({len(vw[k]['flagged'])} slower) | " if k in vw else "- | "
                               for k in views_why)
                     + f"{m.get('cycles_per_elem') or 0:.3f} "
                     f"| {m.get('text_bytes', '-')} | {'yes' if ok else 'no'} |")
    best = front["frontier"][0] if front.get("frontier") else None
    label = "Best node"
    if not best and t.measured():  # nothing feasible yet: show the top geomean, labelled as such
        best = max(t.measured(), key=lambda n: t.metrics(n).get("speedup_geomean") or 0)
        label = f"Top geomean (not feasible: {', '.join(t.confirmed_flags(best)) or 'see flags'} confirmed slower)"
    if best:
        lines += ["", f"## {label}", "", "Lineage: " + " -> ".join(
            f"`{n[:8]}` ({', '.join(t.delta(n)) or t.nodes[n]['operator']})" for n in lineage(t, best)), "",
                  "| cell | speedup | baseline |", "|---|---|---|"]
        v = t.latest_verdict(best)
        for cell, c in sorted(v["cells"].items()):
            if c.get("speedup"):
                lines.append(f"| {cell} | {c['speedup']['s']:.3f} | {(c.get('baseline') or '').split('__')[-1]} |")
            else:
                lines.append(f"| {cell} | - | {c.get('excluded', '')} |")
    lines += ["", "## Techniques (outcome per edge that added them)", "", "| technique | tries | outcomes | best geomean |",
              "|---|---|---|---|"]
    for tid, row in sorted(technique_table(t).items(), key=lambda kv: -kv[1]["tries"]):
        lines.append(f"| {tid} | {row['tries']} | {', '.join(f'{k} {v}' for k, v in row['outcomes'].items())} "
                     f"| {row['best']:.3f} |" if row["best"] else
                     f"| {tid} | {row['tries']} | {', '.join(f'{k} {v}' for k, v in row['outcomes'].items())} | - |")
    failed = [n for n, node in t.nodes.items() if node["status"].startswith("failed")]
    if failed:
        lines += ["", "## Failed nodes", ""] + [f"- `{n[:10]}` {t.nodes[n]['status']}: "
                                                f"{json.dumps({g: v['detail'] for g, v in t.nodes[n]['gates'].items() if not v['passed']}, default=str)[:300]}"
                                                for n in failed]
    return "\n".join(lines) + "\n"
