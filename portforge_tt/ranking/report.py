"""Multi-objective report of one or more campaign runs (any pack).

    python -m portforge_tt.ranking.report runs/<run> [runs/<other> ...] [--optimize METRIC]
        [--constraint EXPR ...] [--json out.json] [--markdown out.md]

Reads only trusted receipts (jobs/*/request.json, result.json, source.json) via
search.build, ranks each stage/suite group with objective.rank, and prints one
table per group: every gated candidate with its metric vector, its declared
transforms, and whether it is on the frontier. compare.py stays as the paired
Chronos-specific comparison.
"""
import argparse
import json
from pathlib import Path

if __package__:
    from . import objective, search
else:
    import objective, search

COLUMNS = ('p50_latency', 'p99_latency', 'first_call', 'throughput', 'joules_per_item', 'mean_watts',
           'dram_bytes', 'max_nrmse')


def _format(metric, value):
    if value is None:
        return '-'
    if metric.endswith('latency') or metric == 'first_call':
        return '%.3f ms' % (value * 1e3)
    if metric == 'dram_bytes':
        return '%.1f MiB' % (value / 2 ** 20)
    return '%.4g' % value


def build(runs, chosen):
    groups = []
    for run in runs:
        nodes = search.build(run)
        for stage in ('perf', 'mesh', 'bringup'):
            for front in search.frontier(nodes, chosen, stage):
                rows = []
                for node in nodes:
                    for evaluation in node['evaluations']:
                        if (evaluation['stage'] == stage and evaluation['passed']
                                and evaluation['suite_sha256'] == front['suite_sha256']
                                and evaluation['model'] == front['model']):
                            rows.append(dict(code=node['code'], job=evaluation['job'], precision=evaluation['precision'],
                                             applied=node['declared']['applied'], novel=node['declared']['novel'],
                                             on_frontier=node['code'] in front['frontier'],
                                             infeasible=front['infeasible'].get(node['code']),
                                             metrics=evaluation['metrics']))
                groups.append(dict(run=str(run), **front, rows=rows))
    return groups


def markdown(groups):
    lines = []
    for group in groups:
        lines += ['## %s · %s · %s' % (Path(group['run']).name, group['model'], group['stage']), '',
                  'Mode: %s. Baseline: `%s`. Candidates: %d.' % (group['mode'], group['baseline']['id'][:12],
                                                                  group['candidates']), '',
                  '| source | precision | frontier | ' + ' | '.join(COLUMNS) + ' | transforms |',
                  '|---|---|---|' + '---|' * len(COLUMNS) + '---|']
        for row in group['rows']:
            mark = 'yes' if row['on_frontier'] else ('infeasible' if row['infeasible'] else '')
            transforms = ', '.join(row['applied']) + ((' + novel: ' + row['novel'][:60]) if row['novel'] else '')
            lines.append('| `%s` | %s | %s | %s | %s |' % (row['code'][:12], row['precision'], mark,
                         ' | '.join(_format(m, row['metrics'].get(m)) for m in COLUMNS), transforms or '-'))
        lines.append('')
    return '\n'.join(lines) or 'No gated candidates with measurements yet.\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--optimize', help='primary metric: ' + ', '.join(objective.METRICS))
    parser.add_argument('--constraint', action='append', default=[])
    parser.add_argument('--json', type=Path)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    chosen = objective.spec(primary=args.optimize, constraints=args.constraint)
    groups = build(args.runs, chosen)
    text = markdown(groups)
    if args.json:
        args.json.write_text(json.dumps(groups, indent=2))
    if args.markdown:
        args.markdown.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
