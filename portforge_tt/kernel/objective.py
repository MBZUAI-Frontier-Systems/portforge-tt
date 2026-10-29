"""Multi-objective ranking of gated kernel candidates from trusted receipts only.

Ported from the TT harness's v3 objective.py (main worktree) with one change:
the metric table is a parameter, so TT and CPU targets share this code.
Correctness gates stay hard; among passing candidates the default for CPU
kernels is a primary metric with a no-regression constraint:

    {"primary": "speedup_geomean", "constraints": ["flagged_cells<=0"]}

and the Pareto front over (speedup_geomean, worst_cell, text_bytes) is always
reported. "baseline" in a relative constraint is the first passing result of
the same suite in the run.
"""
import math
import re

# name -> (direction, description). Every value comes from bench.analyze / profile receipts.
CPU_METRICS = {
    'speedup_geomean': ('max', 'geomean over regime x shape cells of t_upstream_best / t_candidate'),
    'worst_cell': ('max', 'smallest per-cell speedup point estimate (reported)'),
    'flagged_cells': ('min', 'cells slower than the baseline beyond the noise floor with a CI excluding 1 '
                             '(the no-regression constraint reads this, as bench.py flags it)'),
    'cycles_per_elem': ('min', 'median cycles per element at the reference cell (roofline input)'),
    'max_err': ('min', 'largest LSB/ULP difference from the reference over the check set (0 for integers)'),
    'text_bytes': ('min', 'kernel .text size (reported; icache and review cost)'),
}
CPU_PARETO = ('speedup_geomean', 'worst_cell', 'text_bytes')
CPU_DEFAULT = {'primary': 'speedup_geomean', 'constraints': ['flagged_cells<=0'], 'pareto': list(CPU_PARETO),
               'epsilon': 0.01}
CONSTRAINT = re.compile(r'^\s*([a-z0-9_]+)\s*(<=|>=)\s*(?:([0-9.eE+-]+)\s*\*\s*baseline|baseline|([0-9.eE+-]+))\s*$')


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def cpu_metrics(verdict, check=None, profile=None):
    """Metric vector of one candidate from bench.analyze, check and profile receipts."""
    m = verdict.get('metrics', {}) if verdict else {}
    found = {k: _number(m.get(k)) for k in ('speedup_geomean', 'worst_cell', 'cycles_per_elem')}
    if verdict and 'flagged' in verdict:
        found['flagged_cells'] = len(verdict['flagged'])
    if check:
        errs = [r.get('max_err') for w, r in check.items() if isinstance(r, dict)]
        errs = [float(e) if e != 'inf' else math.inf for e in errs if e is not None]
        found['max_err'] = max(errs) if errs and all(math.isfinite(e) for e in errs) else None
    if profile:
        found['text_bytes'] = _number(profile.get('text_bytes'))
    return {k: v for k, v in found.items() if v is not None}


def parse_constraint(text, table=CPU_METRICS):
    match = CONSTRAINT.match(text or '')
    if not match or match.group(1) not in table:
        raise ValueError('constraint must look like "worst_cell>=0.995" or "text_bytes<=1.5*baseline" '
                         'with a metric from: ' + ', '.join(table))
    metric, op, factor, absolute = match.groups()
    if absolute is not None:
        return dict(metric=metric, op=op, value=float(absolute), relative=False)
    return dict(metric=metric, op=op, value=float(factor) if factor else 1.0, relative=True)


def spec(optimization=None, primary=None, constraints=(), table=CPU_METRICS, default=CPU_DEFAULT):
    """Merge a pack's optimization block with launcher overrides into a validated spec."""
    base = dict(default)
    base.update(optimization or {})
    unknown = set(base) - {'primary', 'constraints', 'pareto', 'epsilon', 'report'}
    if unknown:
        raise ValueError('unknown optimization fields: ' + ', '.join(sorted(unknown)))
    chosen = dict(primary=primary or base.get('primary'),
                  constraints=[parse_constraint(c, table) for c in list(base.get('constraints', [])) + list(constraints)],
                  pareto=list(base.get('pareto', CPU_PARETO)), epsilon=float(base.get('epsilon', .01)))
    if chosen['primary'] is not None and chosen['primary'] not in table:
        raise ValueError('primary must be a metric from: ' + ', '.join(table))
    if any(m not in table for m in chosen['pareto']) or not chosen['pareto']:
        raise ValueError('pareto must list metrics from: ' + ', '.join(table))
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


def better(metric, a, b, epsilon, table=CPU_METRICS):
    """True when a beats b on metric by more than the relative noise band."""
    margin = epsilon * max(abs(a), abs(b))
    return a < b - margin if table[metric][0] == 'min' else a > b + margin


def dominates(a, b, dims, epsilon=.01, table=CPU_METRICS):
    """a dominates b: no worse on every shared metric, strictly better on one (beyond noise)."""
    shared = [m for m in dims if m in a and m in b]
    if not shared:
        return False
    return (not any(better(m, b[m], a[m], epsilon, table) for m in shared)
            and any(better(m, a[m], b[m], epsilon, table) for m in shared))


def rank(points, chosen, baseline=None, table=CPU_METRICS):
    """points: [{'id':..., 'metrics': {...}}] of gated passes on one suite.

    Returns {'mode', 'frontier': [ids], 'pareto': [ids], 'ranked': [...], 'infeasible': {...}}.
    The Pareto front is always computed, so it can be reported next to a primary ranking.
    """
    infeasible, ok = {}, []
    for point in points:
        good, reasons = feasible(point['metrics'], chosen['constraints'], baseline)
        (ok.append(point) if good else infeasible.__setitem__(point['id'], reasons))
    front = [p for p in ok if not any(dominates(q['metrics'], p['metrics'], chosen['pareto'], chosen['epsilon'], table)
                                      for q in ok if q is not p)]
    pareto = [p['id'] for p in front]
    if chosen['primary']:
        metric = chosen['primary']
        sign = 1 if table[metric][0] == 'min' else -1
        known = [p for p in ok if metric in p['metrics']]
        ranked = sorted(known, key=lambda p: sign * p['metrics'][metric])
        return dict(mode='primary', metric=metric, frontier=[p['id'] for p in ranked[:1]], pareto=pareto,
                    ranked=[dict(id=p['id'], value=p['metrics'][metric]) for p in ranked], infeasible=infeasible)
    return dict(mode='pareto', dims=chosen['pareto'], frontier=pareto, pareto=pareto,
                ranked=[dict(id=p['id'], metrics={m: p['metrics'].get(m) for m in chosen['pareto']}) for p in front],
                infeasible=infeasible)
