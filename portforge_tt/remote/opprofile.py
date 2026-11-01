"""Summarize a TT-Metal device profile (Tracy ops_perf_results_*.csv) for agents.

Produced by running one case under ``python -m tracy -r -p`` with
TT_METAL_PROFILER_DIR pointing at the job output. Rows between two signposts
(the evaluator's timed region) are grouped by op code, input shapes and core
count, the same grouping as chronos_forecast's sweeps/summarize.py. TT's own
performance model columns (PM IDEAL, PM FPU UTIL, DRAM BW UTIL) give the
roofline view: efficiency = ideal / measured device kernel time.

Standard library only; the controller has no pandas.
"""
import csv
import json
from pathlib import Path

DURATION = 'DEVICE KERNEL DURATION [ns]'
GAP = 'OP TO OP LATENCY [ns]'
IDEAL = 'PM IDEAL [ns]'
FPU = 'PM FPU UTIL (%)'
DRAM = 'DRAM BW UTIL (%)'
SUSPECT_EFFICIENCY = 1.05  # faster than TT's ideal model: measurement or op-attribution problem


def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN -> None


def _shape(row, index):
    dims = [row.get('INPUT_%d_%s_PAD[LOGICAL]' % (index, axis), '') for axis in 'WZYX']
    dims = [d.split('[')[0] for d in dims if d]
    return 'x'.join(dims)


def _operand(row, index):
    shape, dtype = _shape(row, index), row.get('INPUT_%d_DATATYPE' % index) or ''
    return (shape + ' ' + dtype).strip()


def rows_between(rows, start=None, stop=None):
    """Device-op rows inside every start..stop signpost pair (one per case), or all device ops.

    Keeps the report's own (execution) order: a trace replay's rows carry the host
    timestamps of its capture, so sorting by host time would move them out of the window.
    The first op of a window has its op-to-op gap zeroed: that gap spans time before the
    window (the untimed call and the profiler flush), not dispatch inside it.
    """
    if start or stop:
        names = [r.get('OP CODE') for r in rows]
        if start not in names or stop not in names:
            raise ValueError('signposts %r..%r not found in the profile' % (start, stop))
        inside, first, kept = False, False, []
        for row, name in zip(rows, names):
            if name in (start, stop):
                inside = first = name == start
            elif inside:
                if first and row.get('OP TYPE') == 'tt_dnn_device':
                    row, first = dict(row, **{GAP: '0'}), False
                kept.append(row)
        rows = kept
    return [r for r in rows if r.get('OP TYPE') == 'tt_dnn_device']


def summarize(rows, top=25):
    groups, total_ns, gap_ns, ideal_ns, ideal_known_ns = {}, 0.0, 0.0, 0.0, 0.0
    suspects = []
    for row in rows:
        duration = _float(row.get(DURATION)) or 0.0
        total_ns += duration
        gap_ns += _float(row.get(GAP)) or 0.0
        ideal = _float(row.get(IDEAL))
        if ideal is not None and duration > 0:
            ideal_ns += ideal
            ideal_known_ns += duration
            if ideal / duration > SUSPECT_EFFICIENCY:
                suspects.append(row.get('OP CODE'))
        # Precision and fidelity are part of the identity: the same matmul at LoFi/bfp8 is a different kernel.
        key = '%s %s | %s c%s %s' % (row.get('OP CODE'), _operand(row, 0), _operand(row, 1),
                                     row.get('CORE COUNT', '?'), row.get('MATH FIDELITY') or '')
        key = key.strip()
        group = groups.setdefault(key, dict(op=row.get('OP CODE'), key=key, count=0, device_ns=0.0, ideal_ns=0.0,
                                            fpu=[], dram=[], fidelity=row.get('MATH FIDELITY') or None))
        group['count'] += 1
        group['device_ns'] += duration
        group['ideal_ns'] += ideal or 0.0
        for name, column in (('fpu', FPU), ('dram', DRAM)):
            value = _float(row.get(column))
            if value is not None:
                group[name].append(value)
    table = []
    for group in sorted(groups.values(), key=lambda g: -g['device_ns'])[:top]:
        table.append(dict(op=group['op'], key=group['key'], count=group['count'],
                          device_ms=group['device_ns'] / 1e6, share=group['device_ns'] / total_ns if total_ns else None,
                          avg_us=group['device_ns'] / group['count'] / 1e3,
                          efficiency=group['ideal_ns'] / group['device_ns'] if group['device_ns'] and group['ideal_ns'] else None,
                          fpu_util_pct=sum(group['fpu']) / len(group['fpu']) if group['fpu'] else None,
                          dram_util_pct=sum(group['dram']) / len(group['dram']) if group['dram'] else None,
                          math_fidelity=group['fidelity']))
    span = total_ns + gap_ns
    return dict(ops=len(rows), device_ms=total_ns / 1e6, op_to_op_gap_ms=gap_ns / 1e6,
                gap_share=gap_ns / span if span else None,
                model_efficiency=ideal_ns / ideal_known_ns if ideal_known_ns else None,
                top=table, suspect_faster_than_ideal=sorted(set(s for s in suspects if s))[:20],
                reading=('device_ms is summed kernel time; a large gap_share means dispatch/host bound (trace, fewer '
                         'ops); low efficiency on a dominant op is the place for program configs, layout, '
                         'precision/fidelity or a custom kernel. Efficiency uses TT\'s PM IDEAL model.'))


def summarize_file(path, start=None, stop=None, top=25):
    path = Path(path)
    if path.is_dir():
        found = sorted(path.rglob('ops_perf_results_*.csv'))
        if not found:
            raise ValueError('no ops_perf_results_*.csv under ' + str(path))
        path = found[-1]
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return dict(csv=path.name, **summarize(rows_between(rows, start, stop), top))


if __name__ == '__main__':
    import sys
    print(json.dumps(summarize_file(*sys.argv[1:4]), indent=2))
