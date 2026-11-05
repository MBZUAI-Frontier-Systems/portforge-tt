"""Search tree over candidate sources, rebuilt by the controller from trusted receipts.

A node is one evaluated source (code_sha256). What was evaluated, on which
stage, and whether it passed always comes from the run's job receipts. What the
source *is* comes from the worker's TRANSFORMS.json inside that source:

    {"parent": "<code_sha256 of the node it was derived from, or null>",
     "applied": ["max-lmul", "k:width-dispatch"],   # ids from RVV_TECHNIQUES.md and TECHNIQUES.md
     "novel": "free text for a transform outside the catalogue (optional)",
     "note": "one line"}

That declaration is labelled as worker-declared; the controller never trusts it
for gates. A rejection is recorded with its parent and combination, so the same
technique stays eligible under a different parent (e.g. a gather rejected at LMUL 1
can win once the parent works at LMUL 4). The supervisor sees
the frontier, recent rejections in context, and a randomly masked menu of
untried (parent, technique) pairs (Autocomp-style diversity).
"""
import json
from pathlib import Path
import random
import re

from . import objective

GATED_STAGES = ('check', 'perf')
TRANSFORMS_FILE = 'TRANSFORMS.json'
HERE = Path(__file__).resolve().parent
CATALOGUE = HERE.parent / 'guides' / 'RVV_TECHNIQUES.md'


def catalogue(path=CATALOGUE):
    """Techniques from the catalogue guide: `## <category>` sections hold `### \\`<id>\\`: <title>` entries."""
    found, category = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith('## '):
            category = line[3:].split(':', 1)[0].strip()
        entry = re.match(r'### `([^`]+)`(?::| —) (.+)', line)
        if entry and category:
            found[entry.group(1)] = dict(id=entry.group(1), category=category, title=entry.group(2).strip())
    return found


def known(port, path=CATALOGUE):
    """The campaign's technique menu: the catalogue plus the family's own (`k:`) techniques."""
    found = catalogue(path)
    techniques = json.loads((port['dir'] / 'kernels' / port['family'] / 'TECHNIQUES.json').read_text())
    for item in techniques.get('candidates', []):
        if item['id'] not in found:
            found[item['id']] = dict(id=item['id'], category='family',
                                     title=item.get('title') or item.get('application', '')[:120])
    return found


def declared(files, known_ids):
    """The worker's TRANSFORMS.json from a source snapshot, validated but untrusted."""
    try:
        raw = json.loads(files.get(TRANSFORMS_FILE, '{}'))
    except ValueError:
        return dict(valid=False, parent=None, applied=[], unknown=[], novel='', note='unparseable TRANSFORMS.json')
    if not isinstance(raw, dict):
        raw = {}
    applied = [x for x in raw.get('applied', []) if isinstance(x, str)] if isinstance(raw.get('applied'), list) else []
    parent = raw.get('parent') if isinstance(raw.get('parent'), str) else None
    text = lambda key: raw.get(key)[:500] if isinstance(raw.get(key), str) else ''
    return dict(valid=True, parent=parent, applied=sorted({x for x in applied if x in known_ids}),
                unknown=sorted({x for x in applied if x not in known_ids}), novel=text('novel'), note=text('note'))


def _failure_reason(result):
    if not result:
        return 'pending'
    if result.get('failure'):
        return result['failure']
    failures = (result.get('verdict') or {}).get('failures') or []
    return '; '.join(str(f) for f in failures)[:300] if failures else None


def build(run, known_ids=None):
    """Nodes keyed by code_sha256, in first-evaluation order."""
    known_ids = known_ids if known_ids is not None else set(catalogue())
    nodes = {}
    requests = sorted(Path(run).glob('jobs/*/request.json'), key=lambda p: p.stat().st_mtime)
    for path in requests:
        request = json.loads(path.read_text())
        if request.get('kind') != 'evaluate' or request.get('stage') not in GATED_STAGES:
            continue
        code = request.get('code_sha256')
        if not code:
            continue
        result_path = path.with_name('result.json')
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        node = nodes.get(code)
        if node is None:
            source = path.with_name('source.json')
            files = json.loads(source.read_text()) if source.exists() else {}
            node = nodes[code] = dict(code=code, order=len(nodes), declared=declared(files, known_ids), evaluations=[])
        verdict = (result or {}).get('verdict') or {}
        passed = bool(result) and not _failure_reason(result) and verdict.get('passed') is True
        node['evaluations'].append(dict(
            job=path.parent.name, stage=request['stage'],
            cluster_sha256=request.get('cluster_sha256'), suite_sha256=verdict.get('suite_sha256'),
            pending=result is None, passed=passed, reason=None if passed else _failure_reason(result),
            metrics=objective.metrics(result) if passed else {}))
    ordered = sorted(nodes.values(), key=lambda n: n['order'])
    for index, node in enumerate(ordered):
        parent = node['declared']['parent']
        if parent in nodes and parent != node['code']:
            node['parent'], node['parent_source'] = parent, 'declared'
        else:
            node['parent'] = ordered[index - 1]['code'] if index else None
            node['parent_source'] = 'previous evaluation (inferred)' if index else None
    return ordered


def latest(node, stage, precision=None):
    rows = [e for e in node['evaluations'] if e['stage'] == stage and not e['pending']
            and (precision is None or e['precision'] == precision)]
    return rows[-1] if rows else None


def frontier(nodes, chosen, stage='perf'):
    """Rank passing nodes per (suite, cluster) of one stage; baseline = first passing node, latest receipt per node.
    Speedups are already relative to the family's baseline kernel (bench.analyze), so "baseline" here only
    anchors relative constraints."""
    groups = {}
    for node in nodes:
        for row in node['evaluations']:
            if row['stage'] == stage and row['passed'] and row['metrics']:
                key = (row['suite_sha256'], row['cluster_sha256'])
                # A re-measured node keeps its place (the baseline stays first) with its latest receipt.
                groups.setdefault(key, {})[node['code']] = dict(id=node['code'], job=row['job'], metrics=row['metrics'])
    out = []
    for (suite, cluster), by_code in groups.items():
        points = list(by_code.values())
        baseline = points[0]['metrics']
        ranked = objective.rank(points, chosen, baseline)
        out.append(dict(stage=stage, suite_sha256=suite, cluster_sha256=cluster,
                        baseline=dict(id=points[0]['id'], job=points[0]['job'], metrics=baseline),
                        candidates=len(points), **ranked))
    return out


def tried(nodes):
    """(parent, technique) pairs already attempted, with outcome."""
    pairs = {}
    by_code = {n['code']: n for n in nodes}
    for node in nodes:
        parent = by_code.get(node.get('parent'))
        inherited = set(parent['declared']['applied']) if parent else set()
        for technique in set(node['declared']['applied']) - inherited:
            evaluations = [e for e in node['evaluations'] if not e['pending']]
            pairs[(node.get('parent'), technique)] = dict(
                child=node['code'], passed=any(e['passed'] for e in evaluations),
                reasons=sorted({e['reason'] for e in evaluations if e['reason']})[:3])
    return pairs


def summary(nodes, chosen, known, seed=0, mask=.7, beam=4, limit=12):
    """Bounded supervisor view: frontier, contextual rejections and a masked untried menu."""
    pairs = tried(nodes)
    fronts = frontier(nodes, chosen, 'perf') or frontier(nodes, chosen, 'check')
    beam_ids = []
    for front in fronts:
        beam_ids += [i for i in front['frontier'] if i not in beam_ids]
    if not beam_ids:  # nothing measured yet: grow from the newest passing source
        passing = [n['code'] for n in nodes if any(e['passed'] for e in n['evaluations'])]
        beam_ids = passing[-1:]
    beam_ids = beam_ids[:beam]
    by_code = {n['code']: n for n in nodes}
    rng = random.Random(seed)
    menu = []
    for code in beam_ids:
        have = set(by_code[code]['declared']['applied'])
        untried = [t for t in sorted(known) if t not in have and (code, t) not in pairs]
        shown = [t for t in untried if rng.random() >= mask] or untried[:1]
        menu.append(dict(parent=code, applied=sorted(have), untried_total=len(untried),
                         offered=[dict(id=t, category=known[t]['category'], title=known[t]['title']) for t in shown]))
    rejected = [dict(parent=parent, technique=technique, child=row['child'], reasons=row['reasons'])
                for (parent, technique), row in pairs.items() if not row['passed']][-limit:]
    return dict(nodes=len(nodes), objective=dict(primary=chosen['primary'],
                constraints=chosen['constraints'], pareto=chosen['pareto']),
                frontier=fronts, beam=menu, rejected_in_context=rejected,
                unknown_declared_ids=sorted({u for n in nodes for u in n['declared']['unknown']}),
                novel_claims=[dict(code=n['code'], novel=n['declared']['novel']) for n in nodes if n['declared']['novel']][-limit:],
                rules='Rejections are bound to (parent, combination). Offer the worker one beam parent and '
                      'one or two offered techniques (or a novel RVV transform backed by port_profile evidence), '
                      'and ask it to update TRANSFORMS.json (parent = that code_sha256). Rank only on trusted '
                      'perf receipts.')


def write(run, nodes, chosen, known):
    """Persist nodes.jsonl and frontier.json under runs/<run>/search/ (controller-owned)."""
    folder = Path(run) / 'search'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'nodes.jsonl').write_text(''.join(json.dumps(n, sort_keys=True) + '\n' for n in nodes))
    (folder / 'frontier.json').write_text(json.dumps(
        dict(perf=frontier(nodes, chosen, 'perf'), check=frontier(nodes, chosen, 'check')), indent=2))
