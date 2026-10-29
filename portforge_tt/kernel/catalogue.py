"""The technique catalogue (guides/rvv_techniques.json): loading, checks, rendering.

The catalogue says what each technique is, when it applies, what it risks and
how to recognize it in compiled code (`detect`, a predicate over the static
profile). It never says whether a technique worked: that is the search tree's
job, per parent and combination, so a technique rejected once stays available.
"""
import json
from pathlib import Path

GUIDES = Path(__file__).resolve().parents[1] / "guides"
CATALOGUE = GUIDES / "rvv_techniques.json"
MARKDOWN = GUIDES / "RVV_TECHNIQUES.md"
REQUIRED = ("id", "category", "title", "mechanism", "helps_when", "applies_to", "bound", "knobs",
            "preconditions", "correctness_risk", "verifier_support", "interactions", "detect", "prior",
            "exemplars", "sources")
SAFE_BUILTINS = {"any": any, "all": all, "len": len, "min": min, "max": max, "True": True, "False": False,
                 "None": None}


def load(path=CATALOGUE):
    data = json.loads(Path(path).read_text())
    problems = validate(data)
    if problems:
        raise ValueError("catalogue: " + "; ".join(problems[:10]))
    return data


def by_id(data):
    return {t["id"]: t for t in data["techniques"]}


def validate(data):
    """Problems with a catalogue: missing fields, unknown categories or sources,
    interactions naming unknown ids, unparseable detect predicates."""
    problems = []
    ids = [t.get("id") for t in data["techniques"]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    known = set(ids)
    for t in data["techniques"]:
        tid = t.get("id", "?")
        missing = [k for k in REQUIRED if k not in t]
        if missing:
            problems.append(f"{tid}: missing {missing}")
            continue
        if t["category"] not in data["categories"]:
            problems.append(f"{tid}: unknown category {t['category']}")
        for kind in ("synergy", "conflicts", "requires"):
            unknown = set(t["interactions"].get(kind, [])) - known
            if unknown:
                problems.append(f"{tid}: {kind} names unknown ids {sorted(unknown)}")
        bad_sources = [s for s in t["sources"] if s not in data.get("sources", {})]
        if bad_sources:
            problems.append(f"{tid}: unknown sources {bad_sources}")
        try:
            compile(t["detect"], tid, "eval")
        except SyntaxError as error:
            problems.append(f"{tid}: detect does not parse: {error}")
        prior = t["prior"]
        if not 0 <= prior.get("p_improve", -1) <= 1 or prior.get("strength", 0) <= 0:
            problems.append(f"{tid}: prior needs p_improve in [0, 1] and strength > 0")
    return problems


def detect_context(profile):
    """Names a detect predicate may use, from a profile.function_profile result."""
    loops = profile.get("loops", [])
    counts = {}
    for h in loops:
        for k, v in h["counts"].items():
            counts[k] = counts.get(k, 0) + v
    hot = max(loops, key=lambda h: h["vector_ops"], default=None)
    return {
        "loops": len(loops),
        "max_lmul": profile.get("descriptors", {}).get("lmul"),
        "widest_sew": profile.get("descriptors", {}).get("sew"),
        "calls": list(profile.get("calls", [])),
        "mnemonics": set(profile.get("mnemonics", [])),
        "counts": counts,
        "vsetvl_per_iter": len(hot["vsetvl"]) if hot else 0,
        "spills": sum(h["spills"] for h in loops),
        "moves": counts.get("move", 0),
    }


def detects(technique, profile):
    """Does the compiled kernel show this technique? None when undecidable."""
    try:
        return bool(eval(technique["detect"], {"__builtins__": SAFE_BUILTINS}, detect_context(profile)))
    except Exception:
        return None


def applicable(data, family):
    """Techniques whose applies_to matches a family {signature, dtype, op_tags}."""
    out = []
    for t in data["techniques"]:
        a = t["applies_to"]
        if "*" in a.get("signatures", []):
            out.append(t)
            continue
        hits = [family.get("signature") in a.get("signatures", []),
                family.get("dtype") in a.get("dtypes", []),
                bool(set(family.get("op_tags", [])) & set(a.get("op_tags", [])))]
        if any(hits):
            out.append(t)
    return out


def render(data):
    """RVV_TECHNIQUES.md: the human/agent-readable view of the JSON."""
    lines = ["# RVV techniques", "",
             "Generated from `rvv_techniques.json` by `portforge_tt.kernel.catalogue.render`; edit the JSON.",
             "Whether a technique worked is recorded per parent and combination in each run's search tree,",
             "not here: a technique rejected once is still worth combining with others.", ""]
    for cat, desc in data["categories"].items():
        members = [t for t in data["techniques"] if t["category"] == cat]
        if not members:
            continue
        lines += [f"## {cat}", "", desc, ""]
        for t in members:
            risk = t["correctness_risk"]
            inter = t["interactions"]
            lines += [f"### `{t['id']}`: {t['title']}", "",
                      f"- **Mechanism.** {t['mechanism']}",
                      f"- **Helps when.** {t['helps_when']}",
                      f"- **Bound.** {t['bound']}" + (f"; knobs: {', '.join(k['name'] for k in t['knobs'])}" if t["knobs"] else ""),
                      f"- **Correctness risk.** {risk['level']}" + (f": {'; '.join(risk['failure_modes'])}" if risk["failure_modes"] else ""),
                      f"- **Preconditions.** {'; '.join(t['preconditions']) or 'none'}",
                      f"- **Verifier.** SMT {t['verifier_support'].get('smt')}"
                      + (f" ({t['verifier_support']['note']})" if t["verifier_support"].get("note") else ""),
                      f"- **X280.** {t.get('x280', 'not yet measured')}"]
            if any(inter.values()):
                lines.append("- **Interactions.** " + "; ".join(f"{k}: {', '.join(v)}" for k, v in inter.items() if v))
            if t["exemplars"]:
                lines.append("- **Exemplar.** " + ", ".join(f"{e['repo']}:`{e['path']}`" for e in t["exemplars"]))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(data=None):
    MARKDOWN.write_text(render(data or load()))


KERNEL_FIELDS = ("family", "spec", "candidates", "mined_from")
CANDIDATE_FIELDS = ("id", "application", "expected_gain", "preconditions", "sources")


def validate_kernel(doc, data):
    """Problems with a kernels/<family>/TECHNIQUES.json: candidates must name a
    catalogue id or a local `k:<name>` idea (with a title), and carry how it
    applies to this family and why it might win. No status fields: outcomes
    live in the search tree."""
    problems = [f"missing {k}" for k in KERNEL_FIELDS if k not in doc]
    known = set(by_id(data))
    ids = []
    for c in doc.get("candidates", []):
        cid = c.get("id", "?")
        ids.append(cid)
        problems += [f"{cid}: missing {k}" for k in CANDIDATE_FIELDS if k not in c]
        if cid.startswith("k:"):
            if not c.get("title"):
                problems.append(f"{cid}: local ideas need a title")
        elif cid not in known:
            problems.append(f"{cid}: not in the catalogue (prefix local ideas with k:)")
        for banned in ("status", "tried", "result", "outcome"):
            if banned in c:
                problems.append(f"{cid}: '{banned}' belongs in the search tree, not here")
    if len(ids) != len(set(ids)):
        problems.append("duplicate candidate ids")
    return problems


def kernel_menu(doc, data):
    """Catalogue entries plus local ideas, as one list of menu items for a family."""
    catalog = by_id(data)
    items = []
    for c in doc["candidates"]:
        base = dict(catalog.get(c["id"], {}))
        base.update({"id": c["id"], "title": c.get("title", base.get("title", c["id"])),
                     "application": c["application"], "expected_gain": c["expected_gain"],
                     "family_preconditions": c["preconditions"], "knobs": c.get("knobs", base.get("knobs", [])),
                     "prior": c.get("prior", base.get("prior", {"p_improve": 0.2, "strength": 1}))})
        items.append(base)
    return items
