"""PortForge-TT controller: supervisor -> worker -> isolated TT jobs -> frozen full test.

One campaign's loop: preflight the TT cards, check the FP32 reference, start the GPU precision
study, then alternate supervisor and worker turns until the budget ends or the worker submits;
a submission is graded once on the held-out full suite. Progress is ``runs/<run>/status.json``.

This module's ``main`` is the ``portforge`` command; it launches through ``campaign.start``, which
re-executes this file from the run's frozen harness with ``--controller``:

    portforge --port ports/chronos2 --hours 6 --detach
    portforge --port ports/chronos2 --hours 4 --detach --resume runs/<earlier-run> --resume-note "..."
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
import uuid

# A re-executed controller verifies its archived harness before importing the
# campaign modules. This is not a resume command.
if __name__ == '__main__' and '--controller' in sys.argv:
    if __package__:
        from .port import frozen_environment
    else:
        from port import frozen_environment
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument('--run-dir', required=True)
    archive = Path(check.parse_known_args()[0].run_dir).resolve()
    environment = frozen_environment(archive, json.loads((archive / 'manifest.json').read_text()))
    if Path(__file__).resolve() != archive / 'harness/controller.py':
        raise ValueError('controller must execute its archived controller.py')
    os.environ.update({k: environment[k] for k in ('PYTHONPATH', 'PORTFORGE_REPOSITORY_ROOT')})

if __package__:
    from . import campaign, cluster, runtime
    from ..ranking import objective, search
    from ..agents import BACKENDS, supervisor_node, worker_node
    from .cluster import identity as digest
    from .jobs import code_hash, experiment, history_runs, import_references, passing_evaluation, \
        precision_study, reference, snapshot, suite_identity
    from .port import PORT, agent_environment, gated_stages, pack_evaluator, same_evaluator
    from ..remote.portforge_eval import TT_HOST_FILES
    from ..remote.portforge_models import model_spec
    from ..remote.runner import write_json
    from .runtime import get
    from .tools import PortTools, tool_name
else:  # flat: the frozen run harness
    import campaign, cluster, objective, runtime, search
    from agents import BACKENDS, supervisor_node, worker_node
    from cluster import identity as digest
    from jobs import code_hash, experiment, history_runs, import_references, passing_evaluation, \
        precision_study, reference, snapshot, suite_identity
    from port import PORT, agent_environment, gated_stages, pack_evaluator, same_evaluator
    from portforge_eval import TT_HOST_FILES
    from portforge_models import model_spec
    from runner import write_json
    from runtime import get
    from tools import PortTools, tool_name


def supervisor_facts(workspace, run, config, model=None):
    model = model or PORT['key']
    files = snapshot(workspace)
    saved_state = files.get('STATE.md', '')
    state_excerpt = saved_state
    if len(saved_state) > 12000:
        marker = '\n[Middle omitted; worker can read the full STATE.md.]\n'
        state_excerpt = saved_state[:6000] + marker + saved_state[-(6000 - len(marker)):]
    source, jobs, passed, candidates = code_hash(files), [], [], {}
    signature = digest(config)
    delivery_models = tuple(PORT['delivery_models'])
    evidence = {name: {precision: {stage: {'status': 'missing', 'job': None,
                                         'historical_jobs': []}
                                  for stage in ('smoke', 'bringup')}
                       for precision in PORT['delivery_precisions']} for name in delivery_models}
    definitions = {(name, stage): suite_identity(run, name, stage)
                   for name in delivery_models for stage in ('smoke', 'bringup')}
    for path in sorted((Path(run) / 'jobs').glob('*/request.json'), key=lambda p: p.stat().st_mtime):
        request = json.loads(path.read_text())
        result_path = path.with_name('result.json')
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        failed = bool(result.get('returncode') or result.get('failure')
                      or result.get('timeout') or result.get('host_memory_limit'))
        background = request.get('kind') == 'reference-study'
        item = {'id': path.parent.name, 'kind': request.get('kind', 'evaluate'),
                'stage': request.get('stage', 'experiment'), 'model': request.get('model', PORT['key']),
                'cluster_sha256': request.get('cluster_sha256'),
                'background_nonblocking': background, 'pending': not result_path.exists(),
                'failure': result.get('failure'),
                'passed': False if failed else result.get('verdict', {}).get('passed')}
        jobs.append(item)
        precision, stage = request.get('precision'), item['stage']
        if (item['kind'] == 'evaluate' and item['model'] in evidence
                and precision in evidence[item['model']] and stage in ('smoke', 'bringup')):
            cell = evidence[item['model']][precision][stage]
            current = (request.get('code_sha256') == source
                       and request.get('cluster_sha256') == signature
                       and request.get('suite_definition_sha256') == definitions[item['model'], stage])
            if not current:
                cell['historical_jobs'] = (cell['historical_jobs'] + [item['id']])[-3:]
                if cell['job'] is None:
                    cell['status'] = 'historical'
            else:
                verdict = result.get('verdict', {})
                verified = passing_evaluation(result, item['id'], signature, item['model'], precision, stage)
                cell.update(job=item['id'], status=('pending' if item['pending'] else
                            'passed' if verified else 'failed' if failed
                            or verdict.get('passed') is False else 'unverified'))
        if (result and request.get('code_sha256') and request.get('stage') in ('smoke', 'bringup')
                and request.get('cluster_sha256') == digest(config)):
            group = candidates.setdefault((item['model'], request.get('precision')), {'sources': set(), 'passing': set()})
            group['sources'].add(request['code_sha256'])
            if item['passed']:
                group['passing'].add(request['code_sha256'])
        if (item['passed'] and request.get('code_sha256') == source
                and request.get('cluster_sha256') == digest(config)
                and not (item['model'] in evidence and precision in evidence[item['model']]
                         and stage in ('smoke', 'bringup'))):
            passed.append({'stage': item['stage'], 'model': item['model'], 'precision': request.get('precision')})
    # Keep the legacy list's shape, but use the same latest-current decisions as
    # the delivery matrix for its cells; an older pass cannot hide a later failure.
    passed.extend({'stage': stage, 'model': name, 'precision': precision}
                  for name, precisions in evidence.items() for precision, stages in precisions.items()
                  for stage, cell in stages.items() if cell['status'] == 'passed')
    chosen = campaign_objective(run)
    known = search.catalogue()
    nodes = search.build(run, set(known))
    tree = search.summary(nodes, chosen, known, seed=len(jobs), beam=search_beam(config, run))
    baselines = [json.loads(path.read_text()) for path in (Path(run) / 'baselines').rglob('*.json')]
    baselines = [value for value in baselines if value.get('model') == model
                 and value.get('cluster_sha256') == digest(config)]
    return {'objective': PORT['objective'],
             'optimization': tree,
             'selected_model': model, 'baselines': baselines[-12:],
            'delivery_scope': {'models': list(delivery_models), 'development_anchor': model,
                'artifact': 'One configuration-driven backend plus a tt-metal-style package (tt/, demo/, reference/, tests/, benchmarks/, docs/) and portable tests.',
                'precision_policy': PORT['precision_policy'],
                'evidence_scope': 'Latest matching-source/config/suite evaluation per cell, using trusted '
                    'job receipts only. Historical IDs are context, never current passes. Short gates '
                    'do not establish full quality, optimality or production readiness.',
                'short_gates': evidence},
            'worker_state': {'present': 'STATE.md' in files, 'text': state_excerpt,
                             'available_chars': len(saved_state), 'truncated': len(saved_state) > 12000,
                             'note': 'Worker-authored handoff, not verified evidence. Use job receipts for '
                                     'liveness and correctness; continue concrete work after a timeout.'},
            'environment': {'cluster_sha256': digest(config),
                            'tt_runtime': config['tt'].get('metal'),
                            'tt_python': config['tt'].get('python'),
                            'note': 'Jobs from a different cluster identity are historical. A runtime/machine '
                                    'switch requires revalidation, not repeating unchanged old-runtime diagnosis.'},
            'evaluated_candidates': [{'model': key[0], 'precision': key[1],
                'distinct_sources': len(value['sources']), 'passing_sources': len(value['passing'])}
                for key, value in candidates.items()], 'workspace': {'has_backend': 'backend.py' in files, 'source_sha256': source,
                          'source_sha256_scope': 'All non-Markdown workspace files; this is not backend.py alone.',
                          'backend_sha256': hashlib.sha256(files['backend.py'].encode()).hexdigest() if 'backend.py' in files else None,
                          'backend_sha256_scope': 'UTF-8 bytes of backend.py only; compare worker file hashes here.',
                          'file_names': sorted(files, key=lambda name: (name.endswith('.md'), name))[:64],
                          'implementation_files': sorted(name for name in files if not name.endswith('.md'))[:64],
                          'implementation_file_count': sum(not name.endswith('.md') for name in files),
                          'current_source_passes': passed},
            'recent_jobs': [j for j in jobs[:-12] if j['pending']] + jobs[-12:],
            'controls': {'run': {'public_input_stage': ['', 'smoke', 'bringup'],
                                  'timeout_seconds': 'bounded to 1..300 seconds',
                                  'model': list(delivery_models), 'default_model': model},
                         'log': 'Read own TT job stdout/stderr in byte pages up to 65536; no reference-study logs.',
                         'status': 'Use compact=True for bounded log tails; structured results remain complete.',
                         'evaluate': {'stage': gated_stages(), 'precision': list(PORT['precisions']),
                                      'model': list(delivery_models),
                                      'perf': 'Bringup cases plus a batch sweep, >=100 samples or >=20 s per case: '
                                              'latency percentiles (batch <= 4 cases), throughput curve and board '
                                              'energy. Gated like bringup; ranked on its own frontier.',
                                      **({'mesh': 'The perf workload data parallel over a 1x%d mesh of this host\'s '
                                                  'cards (all leased at once; waits until every card is free). '
                                                  'Gated like bringup; ranked on its own frontier.'
                                                  % PORT['mesh_cards']} if PORT.get('mesh_cards') else {})},
                         'profile': {'stage': ['smoke', 'bringup', 'perf (with case, e.g. sweep-b1024)'],
                                     'precision': list(PORT['precisions']),
                                     'returns': 'device op table between the timed signposts: op time, gap share, '
                                                'TT performance-model efficiency, FPU/DRAM utilization'},
                         'fixed_suites': suite_sizes(run, model),
                         'unsupported': ['case selection/count', 'context or horizon override', 'card selection',
                                         'evaluation timeout override'],
                         'full': 'Only submit(precision, model) after current-source bringup passes; '
                                 'check guides/UPSTREAM_PR.md for the deliverable package first.'}}



def suite_sizes(run, model):
    """Case counts of the fixed suites, from the pack's own evaluator (see CONTRACT.md for detail)."""
    try:
        corpus = json.loads((Path(run) / 'corpus.json').read_text())
        return {stage: {'cases': len(pack_evaluator().make_suite(corpus, stage, model)[0]['cases']),
                        'details': 'CONTRACT.md'} for stage in ('smoke', 'bringup')}
    except Exception as exc:  # the summary must not fail on an evaluator import problem
        return {'unavailable': type(exc).__name__, 'details': 'CONTRACT.md'}



def campaign_objective(run):
    """The run's validated objective spec (manifest), or the pack default for older runs."""
    manifest = Path(run) / 'manifest.json'
    saved = json.loads(manifest.read_text()).get('optimization') if manifest.is_file() else None
    if saved:
        return saved
    return objective.spec(PORT.get('optimization') if PORT else None)



def search_beam(config, run=None):
    """Beam width: explicit --beam, else one branch per configured TT card."""
    manifest = Path(run) / 'manifest.json' if run else None
    beam = json.loads(manifest.read_text()).get('beam') if manifest and manifest.is_file() else None
    return beam or int(config['tt'].get('resources', {}).get('tt', 1))



def capacity_wait(record, retries, deadline):
    """Seconds to wait after a provider capacity failure, or None when the budget cannot cover it.

    A provider-reported reset time is honoured (plus a small margin); otherwise back off
    exponentially from one minute to fifteen. Waiting is bounded by the campaign deadline.
    """
    reset = record.get('retry_after')
    wait = reset - time.time() + 30 if isinstance(reset, (int, float)) else 60 * 2 ** max(retries - 1, 0)
    wait = max(30, min(wait, 6 * 3600)) if isinstance(reset, (int, float)) else min(wait, 900)
    return None if retries > 30 or time.monotonic() + wait >= deadline else wait



def preflight_timeout(endpoint):
    """The runner's preflight deadline plus slack; the leak audit (audit_pattern) greps the checkout."""
    return 510 if endpoint.get('audit_pattern') else 330


def prepare(run, config):
    """Verify a replacement before activating it; never reset a machine."""
    endpoint = config['tt']
    root = endpoint['work_dir'] + '/' + Path(run).name
    cluster.upload(endpoint, root + '/harness', {
        **{name: (Path(run) / 'harness' / name).read_bytes() for name in TT_HOST_FILES},
        'runtime.json': json.dumps(endpoint)})
    q = shlex.quote
    commit = cluster.ssh(endpoint, 'git -C ' + q(endpoint['metal']) + ' rev-parse HEAD', timeout=120).decode().strip()
    if commit != endpoint['metal_commit']:
        raise ValueError('replacement TT runtime/weight identity mismatch')
    weights = endpoint.get('weights_by_model', {})
    if not weights or any(not path for path in weights.values()):
        raise ValueError('configured model checkpoint paths are required')
    for model, location in weights.items():
        spec = model_spec(model)
        names = spec['weight_files'] + ([spec['weight_index']] if spec['weight_index'] else [])
        paths = [str(Path(location) / name) for name in names]
        command = 'sha256sum ' + ' '.join(q(p) for p in paths)
        if len(names) == 1:
            command = 'if test -f ' + q(location) + '; then sha256sum ' + q(location) + '; else ' + command + '; fi'
        actual = cluster.ssh(endpoint, command, timeout=600).decode().splitlines()
        if [line.split()[0] for line in actual] != [spec['files'][name]['sha256'] for name in names]:
            raise ValueError('replacement TT runtime/weight identity mismatch: ' + model)
        write_json(Path(run) / 'checkpoints' / (digest(config) + '-' + model + '.json'),
                   {'model': model, 'revision': spec['revision'], 'verified_weight_files': names})
    found = cluster.remote(config, run, {'kind': 'health'})['cards']
    for card in found:
        job = 'preflight-' + uuid.uuid4().hex
        result = cluster.remote(config, run, {'kind': 'preflight', 'job': job, 'pci': card['pci'],
                                      'model': next(iter(weights))}, timeout=preflight_timeout(endpoint))
        write_json(Path(run) / 'preflights' / (job + '.json'), result)
    health = cluster.remote(config, run, {'kind': 'health'})
    if not any(c['state'] == 'healthy' for c in health['cards']):
        raise RuntimeError('replacement has no healthy TT cards')
    write_json(Path(run) / 'clusters' / (digest(config) + '.json'), config)
    write_json(Path(run) / 'cluster.json', config)
    return health



def refresh_cluster(run, path, current):
    replacement = cluster.load(path)
    if digest(replacement) == digest(current):
        return current, 'unchanged'
    pending = [p for p in (Path(run) / 'jobs').glob('*/request.json') if not p.with_name('result.json').exists()]
    if pending:
        return current, 'draining'
    prepare(run, replacement)
    return replacement, 'changed'



def recover_timed_out_card(run, config, health, model):
    """Try one real preflight per card/config after a recorded terminal timeout.

    Never reset hardware, recover memory-limit failures, or clear quarantine by
    editing health records. A saved submission intent prevents duplicate probes
    after transport ambiguity. Pending TT jobs must drain before this is called.
    """
    run = Path(run)
    requests = list((run / 'jobs').glob('*/request.json'))
    if any(not p.with_name('result.json').exists()
           and json.loads(p.read_text()).get('kind') != 'reference-study' for p in requests):
        return health
    signature = digest(config)
    for card in health['cards']:
        if card['state'] != 'quarantined':
            continue
        job = card.get('job', '')
        if not job or Path(job).name != job or job in ('.', '..'):
            continue
        folder = run / 'jobs' / job
        if not (folder / 'result.json').exists() or not (folder / 'request.json').exists():
            continue
        result = json.loads((folder / 'result.json').read_text())
        request = json.loads((folder / 'request.json').read_text())
        if (result.get('failure') != 'timeout' or not result.get('timeout')
                or result.get('host_memory_limit') or result.get('pci') != card['pci']
                or result.get('returncode') != 124 or result.get('job') != job
                or result.get('cluster_sha256') != signature
                or request.get('cluster_sha256') != signature):
            continue
        receipt = run / 'recoveries' / (digest([signature, card['pci']]) + '.json')
        if receipt.exists():
            continue
        probe = {'kind': 'preflight', 'job': 'preflight-' + uuid.uuid4().hex,
                 'pci': card['pci'], 'model': model}
        record = {'failed_job': job, 'request': probe, 'cluster_sha256': signature,
                  'state': 'submitted'}
        receipt.parent.mkdir(parents=True, exist_ok=True)
        try:
            with receipt.open('x') as stream:
                json.dump(record, stream, indent=2)
        except FileExistsError:
            continue
        checked = cluster.remote(config, str(run), probe, timeout=preflight_timeout(config['tt']))
        write_json(run / 'preflights' / (probe['job'] + '.json'), checked)
        write_json(receipt, {**record, 'state': 'complete', 'result': checked})
        # The runner owns card state; success is established by its actual probe.
        return cluster.remote(config, str(run), {'kind': 'health'})
    return health


def loop(args):
    """The controller loop for an archived run (``--controller``); see the module docstring."""
    run = args.run_dir.resolve()
    model = args.model or PORT['key']
    if model not in PORT['delivery_models']:
        raise ValueError('model must be one of the pack delivery models: ' + ', '.join(PORT['delivery_models']))
    manifest = json.loads((run / 'manifest.json').read_text())
    if (run / 'status.json').exists():
        raise ValueError('existing controller state requires explicit recovery; this is not resume')
    if model != manifest['model']:
        raise ValueError('controller model differs from archived selection')
    workspace = Path(manifest['workspace'])
    if not workspace.is_absolute() or not workspace.is_dir():
        raise ValueError('saved fresh workspace missing')
    corpus = json.loads((run / 'corpus.json').read_text())
    if digest(corpus) != manifest['corpus_sha256']:
        raise ValueError('archived corpus identity mismatch')
    config = cluster.load(args.cluster)
    if model not in config['tt'].get('weights_by_model', {}):
        raise ValueError('selected model has no configured TT checkpoint')
    def state(value, **extra):
        write_json(run / 'status.json', {'status': value, 'time': datetime.now(timezone.utc).isoformat(), **extra})
    state('preflight')
    os.environ.update(agent_environment(manifest))
    # Local lease capacities: one agent credential, one reference pipeline,
    # and the configured TT card count.
    cards = int(config['tt'].get('resources', {}).get('tt', 1))
    if (PORT.get('mesh_cards') or 0) > cards:
        raise ValueError('the mesh stage needs %d cards; the cluster leases %d' % (PORT['mesh_cards'], cards))
    runtime.configure(agent_creds=1, reference=1, mesh=1, tt=cards)
    os.environ.setdefault('PYTHONPATH', str(run / 'harness'))
    runtime.trace(str(run / 'trace'))
    runtime.event('campaign.start', model=model, run=str(run), hours=args.hours,
                  backends=manifest.get('agent_backends', {}), worker_web=manifest.get('worker_web', 'on'))
    tools = None
    try:
        health = prepare(run, config)
        if manifest.get('reference_run'):
            import_references(run, manifest['reference_run'], config, model)
        tools = PortTools(name=tool_name(run), workspace=str(workspace), run_dir=str(run), model=model)
        preflight = get(worker_node.submit(str(workspace), str(run), [tools], 0,
            'Infrastructure check only: list files, read CONTRACT.md, confirm port_read ../outside is rejected. '
            'Do not implement or evaluate yet. End with PREFLIGHT_OK only if scoped tools work.'))
        if not preflight['success'] or not preflight['response'].strip().endswith('PREFLIGHT_OK'):
            state('paused_agent_preflight', error=preflight['error'])
            return
        state('reference_preflight')
        try:
            get(reference.submit(str(run), 'smoke', config, model))
        except Exception as exc:
            state('paused_reference', error=str(exc)[-1500:])
            return
        study_job = 'gpu-precision-' + model + '-' + digest(config)[:12]
        earlier = next((root / 'jobs' / study_job for root in history_runs(run)
                        if (root / 'jobs' / study_job / 'result.json').is_file() and same_evaluator(root)
                        and not json.loads((root / 'jobs' / study_job / 'result.json').read_text()).get('failure')),
                       None)
        if earlier:
            # Controller-produced CUDA context for this exact cluster; no candidate data.
            for name in ('request.json', 'result.json'):
                write_json(run / 'jobs' / study_job / name, json.loads((earlier / name).read_text()))
            write_json(run / 'jobs' / study_job / 'import.json', {'source': str(earlier)})
        else:
            write_json(run / 'jobs' / study_job / 'request.json',
                       {'kind': 'reference-study', 'stage': 'precision-study', 'model': model, 'cluster_sha256': digest(config)})
            # Background CUDA context; its result is polled through port_status, never awaited.
            _study = precision_study.submit(str(run), study_job, config, model)
        end, turn, retries, timeouts = time.monotonic() + args.hours * 3600, 0, 0, 0
        previous = None
        while time.monotonic() < end:
            turn += 1
            if (run / 'submission.json').exists():
                break
            config, change = refresh_cluster(run, args.cluster, config)
            if change == 'draining':
                state('draining_for_cluster_change')
                time.sleep(10)
                continue
            if change == 'changed':
                state('checking_replacement_reference')
                get(reference.submit(str(run), 'smoke', config, model))
            health = cluster.remote(config, str(run), {'kind': 'health'})
            if not any(c['state'] == 'healthy' for c in health['cards']):
                state('checking_timeout_recovery', health=health)
                health = recover_timed_out_card(run, config, health, model)
            if not any(c['state'] == 'healthy' for c in health['cards']):
                state('paused_hardware', health=health)
                break
            try:
                known = search.catalogue()
                search.write(run, search.build(run, set(known)), campaign_objective(run), known)
            except (OSError, ValueError) as exc:
                runtime.event('search.write_failed', error=str(exc)[-500:])
            facts = supervisor_facts(workspace, run, config, model)
            state('supervising', turn=turn, health=health, recent_jobs=facts['recent_jobs'])
            review = get(supervisor_node.submit(str(workspace), str(run), turn,
                {**facts, 'health':health, 'worker_report':(previous or {}).get('response','')[-2000:],
                 'worker_error':(previous or {}).get('error',''),
                 'operator_note': manifest.get('operator_note', ''),
                 'remaining_minutes':int((end-time.monotonic())/60)}))
            if not review['success']:
                retries += 1
                wait = capacity_wait(review, retries, end) if review['error'] == 'capacity' else None
                if wait is None:
                    state('paused_supervisor', error=review['error']); break
                state('retry_wait', role='supervisor', reason='capacity', retry=retries, wait_seconds=int(wait))
                time.sleep(wait); continue
            if review['decision']['action'] == 'pause':
                state('paused_supervisor', instruction=review['decision']['instruction']); break
            state('working', turn=turn, instruction=review['decision']['instruction'], health=health)
            note = manifest.get('operator_note', '')
            previous = get(worker_node.submit(str(workspace), str(run), [tools], turn,
                review['decision']['instruction'] + '\nContinue from your saved STATE.md. Call port_status(compact=True) before repeating tests.'
                + ('\nOperator note (trusted; supplied by the controller, not an agent): ' + note if note else '')))
            if not previous['success']:
                if previous['error'] == 'timeout':
                    timeouts += 1
                    if timeouts <= 3:
                        state('worker_timeout_recovery', turn=turn, retry=timeouts)
                        continue
                retries += 1
                wait = capacity_wait(previous, retries, end) if previous['error'] == 'capacity' else None
                if wait is None:
                    state('paused_worker', error=previous['error']); break
                state('retry_wait', role='worker', reason='capacity', retry=retries, wait_seconds=int(wait))
                time.sleep(wait); continue
            retries = timeouts = 0
        else:
            state('budget_exhausted')
        if (run / 'submission.json').exists():
            state('full_validation', turn=turn)
            submission = json.loads((run / 'submission.json').read_text())
            frozen = json.loads((run / 'clusters' / (submission['cluster_sha256'] + '.json')).read_text())
            result = get(experiment.submit(str(run), 'final', {'kind':'evaluate','stage':'full',
                'files':submission['files'], 'precision':submission['precision'], 'model':submission.get('model', PORT['key'])}, frozen))
            state('accepted_research_port' if result.get('verdict',{}).get('passed') else 'full_failed')
    except BaseException as exc:
        state('controller_error', error=str(exc)[-1500:])
        raise
    finally:
        if tools:
            tools.stop()
        runtime.event('campaign.end')
        runtime.end_trace()
        runtime.shutdown()


def main(argv=None):
    """The ``portforge`` script: flags mirror ``campaign.start``'s keyword arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=Path, help='Port pack directory, e.g. ports/chronos2 (see ports/template)')
    parser.add_argument('--cluster', type=Path, help='Machine inventory (default: cluster.yaml in the working '
                                                     'directory or beside the package)')
    parser.add_argument('--corpus', type=Path, help="Frozen benchmark (default: the pack's corpus.json)")
    parser.add_argument('--hours', type=float, default=6)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--model', help="Registry key to port (default: the pack's key)")
    parser.add_argument('--seed-run', type=Path)
    parser.add_argument('--seed-job', help='Use this passing bringup snapshot rather than a changing workspace')
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--resume', type=Path, help='Continue this earlier run in a new run directory: '
                        'full workspace and STATE.md, its FP32 references and read-only job history')
    parser.add_argument('--resume-note', default='', help='Operator note prepended to the resumed STATE.md')
    parser.add_argument('--agent', choices=BACKENDS, default='opencode', help='Worker agent CLI')
    parser.add_argument('--agent-model', help='Worker model (default depends on --agent)')
    parser.add_argument('--worker-web', choices=('on', 'off'), default=None,
                        help='Worker web access (WebFetch/WebSearch): on or off. Default on; --resume '
                             'inherits the earlier run unless this is given')
    parser.add_argument('--supervisor-agent', choices=BACKENDS, help='Supervisor agent CLI (default: --agent)')
    parser.add_argument('--supervisor-model', help='Supervisor model (default depends on its agent)')
    parser.add_argument('--optimize', help='Primary metric to optimize after the gates (default: Pareto front): '
                        + ', '.join(objective.METRICS))
    parser.add_argument('--constraint', action='append', default=[],
                        help='Repeatable, e.g. "p99_latency<=1.1*baseline" or "dram_bytes<=4e9"')
    parser.add_argument('--scenario', choices=tuple(objective.SCENARIOS), help='Deployment scenario to report')
    parser.add_argument('--beam', type=int, help='Parallel search width (default: TT cards in the cluster)')
    args = parser.parse_args(argv)
    if args.controller:
        return loop(args)
    if not args.port:
        parser.error('--port is required (a directory under ports/)')
    options = {k: v for k, v in vars(args).items() if k not in ('port', 'cluster', 'controller')}
    try:
        launched = campaign.start(args.port, args.cluster, **options)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(launched))


if __name__ == '__main__':
    main()
