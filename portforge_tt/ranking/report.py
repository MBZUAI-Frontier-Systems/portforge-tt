"""Multi-objective report of one or more kernel campaign runs.

    python -m portforge_tt.ranking.report runs/<run> [runs/<other> ...] [--optimize METRIC]
        [--constraint EXPR ...] [--json out.json] [--markdown out.md]

Reads only trusted receipts (jobs/*/request.json, result.json, source.json) via
search.build, ranks each stage/suite group with objective.rank, and prints one
table per group: every gated candidate with its metric vector, its declared
transforms, and whether it is on the frontier; then each certification.
"""
import argparse
import json
from pathlib import Path

from . import objective, search
from ..control.port import load_port

COLUMNS = ('speedup_geomean', 'worst_cell', 'flagged_cells', 'cycles_per_elem', 'text_bytes')


def _format(metric, value):
    if value is None:
        return '-'
    if metric in ('speedup_geomean', 'worst_cell'):
        return '%.3fx' % value
    if metric in ('flagged_cells', 'text_bytes'):
        return '%d' % value
    return '%.4g' % value


def _known(run):
    try:
        return search.known(load_port(Path(run) / 'harness'))
    except (OSError, ValueError, KeyError):  # an older or partial run: the shared catalogue only
        return search.catalogue()


def certifications(run):
    rows = []
    for path in sorted(Path(run).glob('jobs/*/request.json'), key=lambda p: p.stat().st_mtime):
        request = json.loads(path.read_text())
        result_path = path.with_name('result.json')
        if request.get('kind') != 'certify' or not result_path.is_file():
            continue
        verdict = json.loads(result_path.read_text()).get('verdict') or {}
        rows.append(dict(code=request.get('code_sha256'), job=path.parent.name, passed=verdict.get('passed') is True,
                         failures=verdict.get('failures', []),
                         verify=[(m.get('method'), m.get('verdict'), m.get('proved'), m.get('configs'))
                                 for m in verdict.get('verify') or []],
                         metrics=((verdict.get('perf') or {}).get('metrics') or {})))
    return rows


def build(runs, chosen):
    groups = []
    for run in runs:
        nodes = search.build(run, set(_known(run)))
        for stage in ('perf', 'check'):
            for front in search.frontier(nodes, chosen, stage):
                rows = []
                for node in nodes:
                    for evaluation in node['evaluations']:
                        if (evaluation['stage'] == stage and evaluation['passed']
                                and evaluation['suite_sha256'] == front['suite_sha256']):
                            rows.append(dict(code=node['code'], job=evaluation['job'],
                                             applied=node['declared']['applied'] + node['declared']['unknown'],
                                             novel=node['declared']['novel'],
                                             on_frontier=node['code'] in front['frontier'],
                                             on_pareto=node['code'] in front.get('pareto', []),
                                             infeasible=front['infeasible'].get(node['code']),
                                             metrics=evaluation['metrics']))
                groups.append(dict(run=str(run), **front, rows=rows, certifications=certifications(run)))
    return groups


def markdown(groups):
    lines = []
    for group in groups:
        lines += ['## %s · %s' % (Path(group['run']).name, group['stage']), '',
                  'Mode: %s. Baseline: `%s`. Candidates: %d.' % (group['mode'], group['baseline']['id'][:12],
                                                                  group['candidates']), '',
                  '| source | frontier | ' + ' | '.join(COLUMNS) + ' | transforms |',
                  '|---|---|' + '---|' * len(COLUMNS) + '---|']
        for row in group['rows']:
            mark = ('yes' if row['on_frontier'] else 'pareto' if row['on_pareto']
                    else 'infeasible' if row['infeasible'] else '')
            transforms = ', '.join(row['applied']) + ((' + novel: ' + row['novel'][:60]) if row['novel'] else '')
            lines.append('| `%s` | %s | %s | %s |' % (row['code'][:12], mark,
                         ' | '.join(_format(m, row['metrics'].get(m)) for m in COLUMNS), transforms or '-'))
        lines.append('')
        if group['stage'] == 'perf' and group['certifications']:
            lines += ['Certification:', '']
            for row in group['certifications']:
                lines.append('- `%s` (job %s): %s; verify %s%s' % (
                    (row['code'] or '?')[:12], row['job'][:8], 'certified' if row['passed'] else 'not certified',
                    ', '.join('%s %s %s/%s' % m for m in row['verify']) or 'none listed',
                    ('; ' + '; '.join(row['failures'])) if row['failures'] else ''))
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
