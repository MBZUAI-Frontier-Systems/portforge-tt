"""Operator baseline: one implementation through a campaign's evaluator and sandbox, with no agents.

    python -m portforge_tt.control.baseline --port ports/resnet50 --cluster CLUSTER.yaml \\
        --corpus CORPUS.json --source ports/resnet50/expert --precision bfp8_b \\
        --stages smoke,bringup,perf,full

Creates a run directory (runs/baseline-<UTC>, or --run-dir) with the frozen harness, the pinned
corpus and the cluster identity, runs the controller's prepare() (upload, weight check, card
preflights) and then the evaluate jobs a campaign submits: PortTools._start_job's request for the
gated stages (perf and mesh hold their cards the same way) and the controller's frozen full-stage
call for full. Each job directory holds request.json, source.json and result.json like a campaign
job; baseline-summary.json collects per-stage pass/fail, failures, NRMSE, latency, the throughput
curve and quality, plus the objective metrics the search ranks on.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import uuid

if __package__:
    from . import cluster, runtime
    from .campaign import check_suites
    from .cluster import identity as digest
    from .controller import prepare
    from .jobs import code_hash, experiment, snapshot, suite_identity
    from .port import PORT, configure, freeze_harness, gated_stages, harness_identity
    from ..ranking import objective
    from ..remote.portforge_models import use_registry
    from ..remote.runner import write_json
    from .runtime import get
else:  # flat: the frozen run harness
    import cluster, objective, runtime
    from campaign import check_suites
    from cluster import identity as digest
    from controller import prepare
    from jobs import code_hash, experiment, snapshot, suite_identity
    from port import PORT, configure, freeze_harness, gated_stages, harness_identity
    from portforge_models import use_registry
    from runner import write_json
    from runtime import get

ORDER = ('smoke', 'bringup', 'perf', 'mesh', 'full')


def parse_stages(text, allowed):
    """Comma-separated stages, run in campaign order, each at most once."""
    requested = [name.strip() for name in text.split(',') if name.strip()]
    if not requested or len(set(requested)) != len(requested) or any(name not in allowed for name in requested):
        raise ValueError('stages must be a comma-separated subset of ' + ','.join(allowed) + ' (no repeats): ' + text)
    return [name for name in ORDER if name in requested]


def copy_workspace(source, run):
    """The implementation under test: a copy of the source directory, which must hold backend.py."""
    source = Path(source)
    if not (source / 'backend.py').is_file():
        raise ValueError(str(source) + ' has no backend.py')
    workspace = run / 'workspace'
    shutil.copytree(source, workspace, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    return workspace


def submit(run, workspace, model, precision, stage):
    """One evaluate job, submitted the way a campaign submits it. Returns (job_id, result)."""
    files = snapshot(workspace)
    config = cluster.activated(run)
    job = uuid.uuid4().hex
    record = {'kind': 'evaluate', 'stage': stage, 'precision': precision, 'model': model,
              'cluster_sha256': digest(config)}
    if stage != 'full':  # PortTools._start_job: the suite definition pins the gated stage
        record['suite_definition_sha256'] = suite_identity(run, model, stage)
    write_json(run / 'jobs' / job / 'source.json', files)
    write_json(run / 'jobs' / job / 'request.json', {**record, 'code_sha256': code_hash(files)})
    if stage == 'full':  # the controller's frozen submission carries no cluster or suite fields
        request, submitter = {'kind': 'evaluate', 'stage': 'full', 'files': files, 'precision': precision,
                              'model': model}, experiment.submit
    elif stage in ('perf', 'mesh'):  # timed jobs hold every card of the host, as in _start_job
        cards = int(config['tt'].get('resources', {}).get('tt', 1)) if stage == 'perf' else PORT['mesh_cards']
        request = {**record, 'files': files}
        submitter = runtime.job({'mesh': 1, 'tt': cards})(experiment._original).submit
    else:
        request, submitter = {**record, 'files': files}, experiment.submit
    return job, get(submitter(str(run), job, request, config))


def summarize(run, job, stage, result):
    verdict = result.get('verdict') or {}
    failures = list(verdict.get('failures', []))
    if result.get('error'):
        failures.append(str(result['error'])[-400:])
    if not verdict and result.get('failure'):
        failures.append('execution failure: ' + str(result['failure']))
    return {'stage': stage, 'job': job, 'job_dir': str(Path(run) / 'jobs' / job),
            'returncode': result.get('returncode'), 'failure': result.get('failure'), 'timeout': result.get('timeout'),
            'pci': result.get('pci'), 'passed': bool(verdict.get('passed')) and result.get('returncode') == 0,
            'accepted': bool(verdict.get('accepted')), 'failures': failures,
            'max_nrmse': verdict.get('max_nrmse'), 'case_count': verdict.get('case_count'),
            'precision': (verdict.get('precision') or {}).get('requested'),
            'timing': verdict.get('timing'), 'latency': verdict.get('latency'),
            'scaling': [{k: row.get(k) for k in ('case', 'batch', 'items', 'p50_seconds', 'items_per_second')}
                        for row in verdict.get('scaling') or []],
            'metrics': objective.metrics(result) if verdict else {},
            'quality': verdict.get('quality') if stage == 'full' else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=Path, required=True, help='Port pack directory, e.g. ports/resnet50')
    parser.add_argument('--cluster', type=Path, required=True, help='Machine inventory (the campaign cluster file)')
    parser.add_argument('--corpus', type=Path, help="Frozen benchmark (default: the pack's corpus.json)")
    parser.add_argument('--source', type=Path, required=True, help='Directory with backend.py (copied as the workspace)')
    parser.add_argument('--precision', required=True, help="One of the pack's precisions, e.g. bf16 or bfp8_b")
    parser.add_argument('--stages', default='smoke,bringup,perf,full', help='Subset of smoke,bringup,perf[,mesh],full')
    parser.add_argument('--model', help="Registry key (default: the pack's key)")
    parser.add_argument('--run-dir', type=Path, help='New run directory (default runs/baseline-<UTC>)')
    parser.add_argument('--stop-on-fail', action='store_true', help='Skip later stages after a failed stage')
    args = parser.parse_args(argv)

    configure(args.port)
    use_registry(PORT['dir'] / 'models.json')
    try:
        stages = parse_stages(args.stages, gated_stages() + ['full'])
    except ValueError as exc:
        parser.error(str(exc))
    model = args.model or PORT['key']
    if model not in PORT['delivery_models']:
        parser.error('--model must be one of the pack delivery models: ' + ', '.join(PORT['delivery_models']))
    if args.precision not in PORT['precisions']:
        parser.error('--precision must be one of ' + ', '.join(PORT['precisions']))
    corpus_path = args.corpus or PORT['corpus_path']
    if not corpus_path:
        parser.error('this pack ships no corpus.json; pass --corpus')
    corpus = json.loads(Path(corpus_path).read_text())
    if digest(corpus) != PORT['corpus_sha256']:
        raise ValueError('corpus identity mismatch')
    config = cluster.load(args.cluster)
    if model not in config['tt'].get('weights_by_model', {}):
        parser.error('the cluster file has no TT checkpoint for ' + model)
    check_suites(corpus, model)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = (args.run_dir or Path.cwd() / 'runs' / ('baseline-' + stamp)).resolve()
    if run.exists():
        parser.error('run directory already exists: ' + str(run))
    run.mkdir(parents=True)
    write_json(run / 'corpus.json', corpus)
    freeze_harness(run / 'harness')
    workspace = copy_workspace(args.source, run)
    write_json(run / 'manifest.json', {
        'kind': 'baseline', 'created': stamp, 'model': model, 'precision': args.precision, 'stages': stages,
        'source': str(args.source.resolve()), 'workspace': str(workspace), 'code_sha256': code_hash(snapshot(workspace)),
        'corpus_sha256': PORT['corpus_sha256'], 'harness_sha256': harness_identity(run / 'harness'),
        'cluster_file': str(args.cluster.resolve()), 'cluster_sha256': digest(config)})

    cards = int(config['tt'].get('resources', {}).get('tt', 1))
    runtime.configure(agent_creds=1, reference=1, mesh=1, tt=cards)
    runtime.trace(str(run / 'trace'))
    summaries = []
    try:
        prepare(run, config)
        for stage in stages:
            job, result = submit(run, workspace, model, args.precision, stage)
            summaries.append(summarize(run, job, stage, result))
            print(json.dumps(summaries[-1], indent=1, default=str), flush=True)
            if args.stop_on_fail and not summaries[-1]['passed']:
                break
    finally:
        runtime.end_trace()
        runtime.shutdown()
    write_json(run / 'baseline-summary.json', {'run': str(run), 'precision': args.precision, 'stages': summaries})
    print(json.dumps({'run': str(run), 'stages': [(s['stage'], s['job'], s['passed']) for s in summaries]}))
    return 0 if summaries and all(s['passed'] for s in summaries) else 1


if __name__ == '__main__':
    sys.exit(main())
