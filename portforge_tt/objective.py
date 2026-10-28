"""Multi-objective ranking of gated candidates from trusted receipts only.

Accuracy gates stay hard: a candidate is ranked only when its trusted verdict
passed. Among those, the default objective is the Pareto front over latency,
tail latency, throughput, energy and device memory. A user can instead name a
primary metric plus constraints (port.json "optimization", or the launcher's
--optimize / --constraint / --scenario), e.g.

    {"scenario": "offline", "primary": "throughput",
     "constraints": ["p99_latency<=1.2*baseline", "joules_per_item<=0.05"]}

"baseline" is the first passing result of the same stage in the run, so
relative constraints mean "no worse than the measured starting point".
Metrics are only compared between results of the same stage and suite.
"""
import math
import re

# name -> (direction, description). Every value comes from runner/evaluator receipts.
METRICS = {
    'p50_latency': ('min', 'median of per-case p50 seconds'),
    'p99_latency': ('min', 'worst per-case p99 seconds (perf stage: >=100 samples per case)'),
    'first_call': ('min', 'worst per-case first (untimed, compile/warm-up) call seconds'),
    'throughput': ('max', 'best items/second on the batch scaling curve'),
    'joules_per_item': ('min', 'mean J per item over sustained energy windows'),
    'mean_watts': ('min', 'mean board watts over sustained energy windows'),
    'dram_bytes': ('min', 'resident (or peak, when reported) device DRAM bytes'),
    'load_seconds': ('min', 'backend creation seconds'),
    'max_nrmse': ('min', 'worst NRMSE against the FP32 oracle'),
}
DEFAULT_PARETO = ('p50_latency', 'p99_latency', 'throughput', 'joules_per_item', 'dram_bytes')
SCENARIOS = {
    'single_stream': 'one request at a time; rank on per-request latency percentiles',
    'offline': 'all work available; rank on throughput across the batch sweep',
    'ar_interactive': 'autoregressive serving; TTFT and inter-token latency (needs prefill/decode_step)',
}
CONSTRAINT = re.compile(r'^\s*([a-z0-9_]+)\s*(<=|>=)\s*(?:([0-9.eE+-]+)\s*\*\s*baseline|baseline|([0-9.eE+-]+))\s*$')


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def metrics(result):
    """Metric vector of one trusted result.json (missing measurements stay absent)."""
    verdict, found = result.get('verdict') or {}, {}
    latency, timing = verdict.get('latency') or {}, verdict.get('timing') or {}
    found['p50_latency'] = _number(latency.get('median_case_p50_seconds', timing.get('median_case_p50_seconds')))
    found['p99_latency'] = _number(latency.get('max_case_p99_seconds'))
    found['first_call'] = _number(latency.get('max_first_seconds'))
    rates = [_number(row.get('items_per_second')) for row in verdict.get('scaling') or []]
    found['throughput'] = max((r for r in rates if r), default=None)
    sustained = [row for row in (result.get('energy') or {}).get('cases', []) if not row.get('short_window')]
    per_item = [_number(row.get('joules_per_item')) for row in sustained]
    per_item = [x for x in per_item if x]
    found['joules_per_item'] = sum(per_item) / len(per_item) if per_item else None
    watts = [_number(row.get('mean_watts')) for row in sustained]
    watts = [x for x in watts if x]
    found['mean_watts'] = sum(watts) / len(watts) if watts else None
    measured = result.get('measurements') or {}
    found['dram_bytes'] = _number(measured.get('peak_dram_bytes', measured.get('resident_dram_bytes')))
    found['load_seconds'] = _number(measured.get('load_seconds'))
    found['max_nrmse'] = _number(verdict.get('max_nrmse'))
    return {k: v for k, v in found.items() if v is not None}


def parse_constraint(text):
    match = CONSTRAINT.match(text or '')
    if not match or match.group(1) not in METRICS:
        raise ValueError('constraint must look like "p99_latency<=1.1*baseline" or "dram_bytes<=4e9" '
                         'with a metric from: ' + ', '.join(METRICS))
    metric, op, factor, absolute = match.groups()
    if absolute is not None:
        return dict(metric=metric, op=op, value=float(absolute), relative=False)
    return dict(metric=metric, op=op, value=float(factor) if factor else 1.0, relative=True)


def spec(optimization=None, primary=None, constraints=(), scenario=None):
    """Merge a pack's optimization block with launcher overrides into a validated spec."""
    base = dict(optimization or {})
    unknown = set(base) - {'scenario', 'primary', 'constraints', 'pareto', 'epsilon', 'batch_sweep', 'report'}
    if unknown:
        raise ValueError('unknown optimization fields: ' + ', '.join(sorted(unknown)))
    chosen = dict(scenario=scenario or base.get('scenario', 'single_stream'),
                  primary=primary or base.get('primary'),
                  constraints=[parse_constraint(c) for c in (list(base.get('constraints', [])) + list(constraints))],
                  pareto=list(base.get('pareto', DEFAULT_PARETO)), epsilon=float(base.get('epsilon', .02)))
    if chosen['scenario'] not in SCENARIOS:
        raise ValueError('scenario must be one of ' + ', '.join(SCENARIOS))
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


def dominates(a, b, dims, epsilon=.02):
    """a dominates b: no worse on every shared metric, strictly better on one (beyond noise)."""
    shared = [m for m in dims if m in a and m in b]
    if not shared:
        return False
    return (not any(better(m, b[m], a[m], epsilon) for m in shared)
            and any(better(m, a[m], b[m], epsilon) for m in shared))


def rank(points, chosen, baseline=None):
    """points: [{'id':..., 'metrics': {...}}] of gated passes on one stage/suite.

    Returns {'mode', 'frontier': [ids], 'ranked': [...], 'infeasible': {...}}.
    """
    infeasible, ok = {}, []
    for point in points:
        good, reasons = feasible(point['metrics'], chosen['constraints'], baseline)
        (ok.append(point) if good else infeasible.__setitem__(point['id'], reasons))
    if chosen['primary']:
        metric = chosen['primary']
        sign = 1 if METRICS[metric][0] == 'min' else -1
        known = [p for p in ok if metric in p['metrics']]
        ranked = sorted(known, key=lambda p: sign * p['metrics'][metric])
        return dict(mode='primary', metric=metric, frontier=[p['id'] for p in ranked[:1]],
                    ranked=[dict(id=p['id'], value=p['metrics'][metric]) for p in ranked], infeasible=infeasible)
    front = [p for p in ok if not any(dominates(q['metrics'], p['metrics'], chosen['pareto'], chosen['epsilon'])
                                      for q in ok if q is not p)]
    return dict(mode='pareto', dims=chosen['pareto'], frontier=[p['id'] for p in front],
                ranked=[dict(id=p['id'], metrics={m: p['metrics'].get(m) for m in chosen['pareto']}) for p in front],
                infeasible=infeasible)
