"""Multi-objective ranking of gated kernel candidates from trusted receipts only.

Correctness gates stay hard: a candidate is ranked only when its trusted verdict
passed. Among those, the default objective is a primary metric with a
no-regression constraint,

    {"primary": "speedup_geomean", "constraints": ["flagged_cells<=0"]}

and the Pareto front over (speedup_geomean, worst_cell, text_bytes) is always
reported beside it. A pack's port.json "optimization" block or the launcher's
--optimize / --constraint override these. "baseline" in a relative constraint
is the first passing result of the same stage and suite in the run. Metrics are
only compared between results of the same stage and suite.
"""
import math
import re

# name -> (direction, description). Every value comes from bench.analyze / profile receipts.
METRICS = {
    'speedup_geomean': ('max', 'geomean over regime x shape cells of t_baseline / t_candidate'),
    'worst_cell': ('max', 'smallest per-cell speedup point estimate'),
    'flagged_cells': ('min', 'cells slower than the baseline beyond the noise floor in two sessions'),
    'cycles_per_elem': ('min', 'median cycles per element at the reference cell (roofline input)'),
    'text_bytes': ('min', 'kernel .text size (icache and review cost)'),
}
DEFAULT_PARETO = ('speedup_geomean', 'worst_cell', 'text_bytes')
DEFAULT = {'primary': 'speedup_geomean', 'constraints': ['flagged_cells<=0'], 'pareto': list(DEFAULT_PARETO),
           'epsilon': 0.01}
CONSTRAINT = re.compile(r'^\s*([a-z0-9_]+)\s*(<=|>=)\s*(?:([0-9.eE+-]+)\s*\*\s*baseline|baseline|([0-9.eE+-]+))\s*$')


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def metrics(result):
    """Metric vector of one trusted result.json (missing measurements stay absent)."""
    verdict = result.get('verdict') or {}
    perf = verdict.get('perf') or {}
    measured = perf.get('metrics') or {}
    found = {k: _number(measured.get(k)) for k in ('speedup_geomean', 'worst_cell', 'cycles_per_elem')}
    if perf:
        found['flagged_cells'] = len(perf.get('confirmed_flags', perf.get('flagged')) or [])
    found['text_bytes'] = _number((result.get('profile') or {}).get('text_bytes'))
    return {k: v for k, v in found.items() if v is not None}


def parse_constraint(text):
    match = CONSTRAINT.match(text or '')
    if not match or match.group(1) not in METRICS:
        raise ValueError('constraint must look like "worst_cell>=0.995" or "text_bytes<=1.5*baseline" '
                         'with a metric from: ' + ', '.join(METRICS))
    metric, op, factor, absolute = match.groups()
    if absolute is not None:
        return dict(metric=metric, op=op, value=float(absolute), relative=False)
    return dict(metric=metric, op=op, value=float(factor) if factor else 1.0, relative=True)


def spec(optimization=None, primary=None, constraints=()):
    """Merge a pack's optimization block with launcher overrides into a validated spec."""
    base = dict(DEFAULT)
    base.update(optimization or {})
    unknown = set(base) - {'primary', 'constraints', 'pareto', 'epsilon', 'report'}
    if unknown:
        raise ValueError('unknown optimization fields: ' + ', '.join(sorted(unknown)))
    chosen = dict(primary=primary or base.get('primary'),
                  constraints=[parse_constraint(c) for c in (list(base.get('constraints', [])) + list(constraints))],
                  pareto=list(base.get('pareto', DEFAULT_PARETO)), epsilon=float(base.get('epsilon', .01)))
    if chosen['primary'] is not None and chosen['primary'] not in METRICS:
        raise ValueError('primary must be a metric from: ' + ', '.join(METRICS))
    if any(m not in METRICS for m in chosen['pareto']) or not chosen['pareto']:
        raise ValueError('pareto must list metrics from: ' + ', '.join(METRICS))
    if not 0 <= chosen['epsilon'] < .5:
        raise ValueError('epsilon is a relative noise band in [0, 0.5)')
    return chosen


def feasible(vector, constraints, baseline=None):
    """(ok, reasons). A missing metric or baseline makes the constraint unknown, not satisfied."""
    reasons = []
    for c in constraints:
        value = vector.get(c['metric'])
        bound = c['value'] * (baseline or {}).get(c['metric'], math.nan) if c['relative'] else c['value']
        if value is None or not math.isfinite(bound):
            reasons.append(c['metric'] + ' unknown')
        elif (c['op'] == '<=' and value > bound) or (c['op'] == '>=' and value < bound):
            reasons.append('%s=%.6g violates %s %.6g' % (c['metric'], value, c['op'], bound))
    return not reasons, reasons


def better(metric, a, b, epsilon):
    """True when a beats b on metric by more than the relative noise band."""
    direction = METRICS[metric][0]
    margin = epsilon * max(abs(a), abs(b))
    return a < b - margin if direction == 'min' else a > b + margin


def dominates(a, b, dims, epsilon=.01):
    """a dominates b: no worse on every shared metric, strictly better on one (beyond noise)."""
    shared = [m for m in dims if m in a and m in b]
    if not shared:
        return False
    return (not any(better(m, b[m], a[m], epsilon) for m in shared)
            and any(better(m, a[m], b[m], epsilon) for m in shared))


def rank(points, chosen, baseline=None):
    """points: [{'id':..., 'metrics': {...}}] of gated passes on one stage/suite.

    Returns {'mode', 'frontier': [ids], 'pareto': [ids], 'ranked': [...], 'infeasible': {...}}.
    The Pareto front is always computed, so it can be reported next to a primary ranking.
    """
    infeasible, ok = {}, []
    for point in points:
        good, reasons = feasible(point['metrics'], chosen['constraints'], baseline)
        (ok.append(point) if good else infeasible.__setitem__(point['id'], reasons))
    front = [p for p in ok if not any(dominates(q['metrics'], p['metrics'], chosen['pareto'], chosen['epsilon'])
                                      for q in ok if q is not p)]
    pareto = [p['id'] for p in front]
    if chosen['primary']:
        metric = chosen['primary']
        sign = 1 if METRICS[metric][0] == 'min' else -1
        known = [p for p in ok if metric in p['metrics']]
        ranked = sorted(known, key=lambda p: sign * p['metrics'][metric])
        return dict(mode='primary', metric=metric, frontier=[p['id'] for p in ranked[:1]], pareto=pareto,
                    ranked=[dict(id=p['id'], value=p['metrics'][metric]) for p in ranked], infeasible=infeasible)
    return dict(mode='pareto', dims=chosen['pareto'], frontier=pareto, pareto=pareto,
                ranked=[dict(id=p['id'], metrics={m: p['metrics'].get(m) for m in chosen['pareto']}) for p in front],
                infeasible=infeasible)
