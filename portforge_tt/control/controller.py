"""PortForge controller: supervisor -> worker -> trusted gates on the x86 host -> certification.

One campaign's loop: upload the frozen harness to the host and check it, fetch the family's
context sources, evaluate the seed kernels, then alternate supervisor and worker turns until the
budget ends or the worker submits. The submitted source (or, without a submission, the best
measured one) is certified once: held-out regimes, a second bench session and the family's
verify methods. Progress is ``runs/<run>/status.json``.

This module's ``main`` is the ``portforge`` command; it launches through ``campaign.start``, which
re-executes this module from the run's frozen harness with ``--controller``:

    portforge --port ports/dav1d-rvv --family refmvs-load_tmvs --hours 6 --board-minutes 120 --detach
    portforge --port ports/xnnpack-rvv --family qs8-vcvt --seed-kernel pf2=k.c --dry-run
    portforge --port ports/dav1d-rvv --family refmvs-load_tmvs --hours 4 --detach --resume runs/<earlier> --resume-note "..."
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid

# A re-executed controller verifies its archived harness before importing the
# campaign modules. This is not a resume command.
if __name__ == '__main__' and '--controller' in sys.argv:
    from .port import frozen_environment
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument('--run-dir', required=True)
    archive = Path(check.parse_known_args()[0].run_dir).resolve()
    environment = frozen_environment(archive, json.loads((archive / 'manifest.json').read_text()))
    if Path(__file__).resolve() != archive / 'harness/portforge_tt/control/controller.py':
        raise ValueError('controller must execute its archived controller.py')
    os.environ.update({k: environment[k] for k in ('PYTHONPATH', 'PORTFORGE_REPOSITORY_ROOT')})

from . import campaign, cluster, runtime
from ..ranking import objective, search
from ..agents import BACKENDS, supervisor_node, worker_node
from .cluster import identity as digest
from .jobs import board_seconds, code_hash, context, passing_evaluation, snapshot, suite_identity
from .port import CONTEXT_DIR, PORT, agent_environment, family_techniques, gated_stages, host_files, render_spec
from ..remote.runner import write_json
from .runtime import get
from .tools import PortTools, submit as job_submit, tool_name


def current_cell(run, signature, definition, source, stage):
    """Latest receipt of `stage` for this source, cluster and suite: missing, pending, passed or failed."""
    cell = {'status': 'missing', 'job': None}
    for path in sorted((Path(run) / 'jobs').glob('*/request.json'), key=lambda p: p.stat().st_mtime):
        request = json.loads(path.read_text())
        if (request.get('kind') == 'evaluate' and request.get('stage') == stage
                and request.get('code_sha256') == source and request.get('cluster_sha256') == signature
                and request.get('suite_definition_sha256') == definition):
            result_path = path.with_name('result.json')
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            cell = {'job': path.parent.name,
                    'status': 'pending' if result is None else 'passed' if passing_evaluation(
                        result, path.parent.name, signature, stage) else 'failed',
                    'failures': ((result or {}).get('verdict') or {}).get('failures') or (result or {}).get('error')}
            if result and stage == 'perf':
                cell['metrics'] = (((result.get('verdict') or {}).get('perf') or {}).get('metrics') or {})
    return cell


def supervisor_facts(workspace, run, config):
    files = snapshot(workspace)
    saved_state = files.get('STATE.md', '')
    state_excerpt = saved_state
    if len(saved_state) > 12000:
        marker = '\n[Middle omitted; worker can read the full STATE.md.]\n'
        state_excerpt = saved_state[:6000] + marker + saved_state[-(6000 - len(marker)):]
    manifest = json.loads((Path(run) / 'manifest.json').read_text())
    signature, definition = digest(config), suite_identity(run)
    source = code_hash(files) if PORT['source_name'] in files else None
    jobs = []
    for path in sorted((Path(run) / 'jobs').glob('*/request.json'), key=lambda p: p.stat().st_mtime):
        request = json.loads(path.read_text())
        result_path = path.with_name('result.json')
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        verdict = result.get('verdict') or {}
        jobs.append({'id': path.parent.name, 'kind': request.get('kind'), 'stage': request.get('stage'),
                     'command': request.get('command'), 'code_sha256': (request.get('code_sha256') or '')[:12],
                     'pending': not result_path.exists(), 'failure': result.get('failure'),
                     'passed': verdict.get('passed') if verdict else None,
                     'speedup_geomean': ((verdict.get('perf') or {}).get('metrics') or {}).get('speedup_geomean')})
    chosen = campaign_objective(run)
    known = search.known(PORT)
    nodes = search.build(run, set(known))
    tree = search.summary(nodes, chosen, known, seed=len(jobs), beam=search_beam(run))
    if not tree['beam']:  # nothing passed yet (a port in progress): the family's whole menu, unmasked
        tree['family_menu'] = [{'id': c['id'], 'application': c.get('application', '')[:200]}
                               for c in family_techniques(PORT).get('candidates', [])]
    spent = board_seconds(run) / 60
    return {'objective': PORT['objective'], 'family': PORT['family'], 'baseline': PORT['baseline'],
            'optimization': tree,
            'current_source': {'present': source is not None, 'code_sha256': source,
                               'is_placeholder': bool(source and source == manifest.get('placeholder_sha256')),
                               'declared': search.declared(files, set(known)),
                               'check': current_cell(run, signature, definition, source, 'check'),
                               'perf': current_cell(run, signature, definition, source, 'perf'),
                               'scope': 'Latest matching-source/cluster/suite receipt per stage. Historical IDs '
                                        'are context, never current passes.'},
            'worker_state': {'present': 'STATE.md' in files, 'text': state_excerpt,
                             'available_chars': len(saved_state), 'truncated': len(saved_state) > 12000,
                             'note': 'Worker-authored handoff, not verified evidence. Use job receipts for '
                                     'liveness and correctness; continue concrete work after a timeout.'},
            'board': {'minutes_spent': round(spent, 1), 'minutes_budget': manifest.get('board_minutes', 0),
                      'note': 'perf evaluations close when the budget is spent; certification needs one more session'},
            'seeds': sorted(manifest.get('seeds', {})),
            'recent_jobs': [j for j in jobs[:-12] if j['pending']] + jobs[-12:],
            'controls': {'evaluate': {'stage': gated_stages(),
                                      'check': 'G0 compile + static profile, G1 project tests under QEMU at VLEN '
                                               '128-1024, G2 bit-exact over the regime matrix',
                                      'perf': 'check, then an X280 bench session over the matrix vs the baseline; '
                                              'the only receipts that rank candidates'},
                         'run': 'command="compile" (static profile) or "check" (QEMU 128/1024): quick, untrusted',
                         'profile': 'static profile with predicted X280 cycles per element per hot loop',
                         'status': 'compact=True bounds long fields; job_id reads one receipt',
                         'submit': 'Only after the current source passes perf; the controller then certifies it '
                                   '(held-out regimes, second session, formal verification) and the run ends.'}}


def campaign_objective(run):
    """The run's validated objective spec (manifest), or the pack default for older runs."""
    manifest = Path(run) / 'manifest.json'
    saved = json.loads(manifest.read_text()).get('optimization') if manifest.is_file() else None
    if saved:
        return saved
    return objective.spec(PORT.get('optimization') if PORT else None)


def search_beam(run=None):
    """Beam width: explicit --beam, else four parents."""
    manifest = Path(run) / 'manifest.json' if run else None
    beam = json.loads(manifest.read_text()).get('beam') if manifest and manifest.is_file() else None
    return beam or 4


def capacity_wait(record, retries, deadline):
    """Seconds to wait after a provider capacity failure, or None when the budget cannot cover it.

    A provider-reported reset time is honoured (plus a small margin); otherwise back off
    exponentially from one minute to fifteen. Waiting is bounded by the campaign deadline.
    """
    reset = record.get('retry_after')
    wait = reset - time.time() + 30 if isinstance(reset, (int, float)) else 60 * 2 ** max(retries - 1, 0)
    wait = max(30, min(wait, 6 * 3600)) if isinstance(reset, (int, float)) else min(wait, 900)
    return None if retries > 30 or time.monotonic() + wait >= deadline else wait


def prepare(run, config):
    """Upload the frozen harness to the host and check the host can gate (and whether the board answers)."""
    endpoint = config['host']
    cluster.upload(endpoint, endpoint['work_dir'] + '/' + cluster.run_root(config, run) + '/harness',
                   host_files(Path(run) / 'harness'))
    health = cluster.remote(config, run, {'kind': 'health'}, timeout=600)
    if not health.get('ok') or not (health.get('host') or {}).get('ok'):
        raise RuntimeError('host cannot run gates: ' + json.dumps(health)[-1500:])
    if health.get('workdir') and health['workdir'] != endpoint['work_dir']:
        raise RuntimeError('the host works in %s (its cpu.yaml) but cluster.yaml says work_dir %s; gate paths '
                           'are relative to it, so they must agree' % (health['workdir'], endpoint['work_dir']))
    write_json(Path(run) / 'clusters' / (digest(config) + '.json'), config)
    write_json(Path(run) / 'cluster.json', config)
    return health


def install_context(run, workspace, names):
    """Copy the fetched context sources into the workspace (read-only for agents) and index them in SPEC.md."""
    target = Path(workspace) / CONTEXT_DIR
    target.mkdir(exist_ok=True)
    for name in names:
        (target / name).write_text((Path(run) / CONTEXT_DIR / name).read_text())
    (Path(workspace) / 'SPEC.md').write_text(render_spec(PORT, names))


def start_job(run, files, request, config):
    """A controller-owned job (seeds, certification): same receipts as an agent's."""
    job = uuid.uuid4().hex
    write_json(Path(run) / 'jobs' / job / 'source.json', files)
    request = {**request, 'cluster_sha256': digest(config), 'family': PORT['family'],
               'suite_definition_sha256': suite_identity(run)}
    write_json(Path(run) / 'jobs' / job / 'request.json', {**request, 'code_sha256': code_hash(files)})
    return job, job_submit(request)(str(run), job, {**request, 'files': files}, config)


def evaluate_seeds(run, manifest, config):
    """Seed kernels become baseline nodes: check, then perf while the board budget lasts."""
    futures = []
    for path in sorted((Path(run) / 'seeds').glob('*.json')):
        files = json.loads(path.read_text())
        futures.append(start_job(run, files, {'kind': 'evaluate', 'stage': 'check', 'seed': 1}, config)[1])
        if manifest.get('board_minutes', 0) > 0:
            futures.append(start_job(run, files, {'kind': 'evaluate', 'stage': 'perf', 'seed': 1}, config)[1])
    return [get(f) for f in futures]


def certification_target(run, config):
    """The submission, else the best measured source on the perf frontier (None when nothing was measured)."""
    submission = Path(run) / 'submission.json'
    if submission.exists():
        saved = json.loads(submission.read_text())
        return saved['files'], saved.get('metrics') or {}, 'submission'
    nodes = search.build(run, set(search.known(PORT)))
    fronts = search.frontier(nodes, campaign_objective(run), 'perf')
    best = fronts[-1]['frontier'][0] if fronts and fronts[-1]['frontier'] else None
    if not best:
        return None, {}, None
    node = next(n for n in nodes if n['code'] == best)
    row = [e for e in node['evaluations'] if e['stage'] == 'perf' and e['passed']][-1]
    files = json.loads((Path(run) / 'jobs' / row['job'] / 'source.json').read_text())
    return files, row['metrics'], 'frontier'


def loop(args):
    """The controller loop for an archived run (``--controller``); see the module docstring."""
    run = args.run_dir.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    if (run / 'status.json').exists():
        raise ValueError('existing controller state requires explicit recovery; this is not resume')
    workspace = Path(manifest['workspace'])
    if not workspace.is_absolute() or not workspace.is_dir():
        raise ValueError('saved fresh workspace missing')
    config = cluster.load(args.cluster)

    def state(value, **extra):
        write_json(run / 'status.json', {'status': value, 'time': datetime.now(timezone.utc).isoformat(), **extra})
    state('preflight')
    os.environ.update(agent_environment(manifest))
    resources = config['host']['resources']
    runtime.configure(agent_creds=1, host=int(resources['host']), board=int(resources['board']), dev=2)
    runtime.trace(str(run / 'trace'))
    runtime.event('campaign.start', family=PORT['family'], run=str(run), hours=args.hours,
                  backends=manifest.get('agent_backends', {}))
    tools = None
    end = time.monotonic() + args.hours * 3600
    try:
        health = prepare(run, config)
        state('context', health=health)
        install_context(run, workspace, get(context.submit(str(run), config)))
        state('seeds')
        evaluate_seeds(run, manifest, config)
        if manifest.get('dry_run'):
            state('dry_run_done')
            return
        tools = PortTools(name=tool_name(run), workspace=str(workspace), run_dir=str(run))
        preflight = get(worker_node.submit(str(workspace), str(run), [tools], 0,
            'Infrastructure check only: list files, read CONTRACT.md, confirm port_read ../outside is rejected. '
            'Do not implement or evaluate yet. End with PREFLIGHT_OK only if scoped tools work.'))
        if not preflight['success'] or not preflight['response'].strip().endswith('PREFLIGHT_OK'):
            state('paused_agent_preflight', error=preflight['error'])
            return
        turn, retries, timeouts, previous = 0, 0, 0, None
        while time.monotonic() < end:
            turn += 1
            if (run / 'submission.json').exists():
                break
            try:
                known = search.known(PORT)
                search.write(run, search.build(run, set(known)), campaign_objective(run), known)
            except (OSError, ValueError) as exc:
                runtime.event('search.write_failed', error=str(exc)[-500:])
            facts = supervisor_facts(workspace, run, config)
            state('supervising', turn=turn, recent_jobs=facts['recent_jobs'])
            review = get(supervisor_node.submit(str(workspace), str(run), turn,
                {**facts, 'worker_report': (previous or {}).get('response', '')[-2000:],
                 'worker_error': (previous or {}).get('error', ''),
                 'operator_note': manifest.get('operator_note', ''),
                 'remaining_minutes': int((end - time.monotonic()) / 60)}))
            if not review['success']:
                retries += 1
                wait = capacity_wait(review, retries, end) if review['error'] == 'capacity' else None
                if wait is None:
                    state('paused_supervisor', error=review['error']); break
                state('retry_wait', role='supervisor', reason='capacity', retry=retries, wait_seconds=int(wait))
                time.sleep(wait); continue
            if review['decision']['action'] == 'pause':
                state('paused_supervisor', instruction=review['decision']['instruction']); break
            state('working', turn=turn, instruction=review['decision']['instruction'])
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
        files, metrics, origin = certification_target(run, config)
        if files is None:
            state('nothing_to_certify')
            return
        state('certifying', origin=origin, code_sha256=code_hash(files))
        _, future = start_job(run, files, {'kind': 'certify', 'stage': 'full', 'origin': origin,
                                           'heldout_seed': manifest['heldout_seed'],
                                           'submitted_metrics': metrics}, config)
        result = get(future)
        state('certified' if (result.get('verdict') or {}).get('passed') else 'certify_failed', origin=origin,
              code_sha256=code_hash(files), failures=(result.get('verdict') or {}).get('failures') or result.get('error'))
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
    parser.add_argument('--port', type=Path, help='Kernel pack directory, e.g. ports/dav1d-rvv (see ports/template-kernel)')
    parser.add_argument('--family', help="Kernel family to port and optimize (a key of the pack's families.json)")
    parser.add_argument('--cluster', type=Path, help='Machine inventory (default: cluster.yaml in the working '
                                                     'directory or beside the package)')
    parser.add_argument('--hours', type=float, default=6)
    parser.add_argument('--board-minutes', type=float, default=120, help='X280 time for perf and certification')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--seed-kernel', action='append', default=[],
                        help='label=path (repeatable): baselines evaluated first; the first starts the workspace')
    parser.add_argument('--placeholder', type=Path, help="Stub source to port from (default: the family's placeholder)")
    parser.add_argument('--dry-run', action='store_true', help='Evaluate the seeds, then stop (no agents)')
    parser.add_argument('--resume', type=Path, help='Continue this earlier run in a new run directory: '
                        'full workspace and STATE.md and read-only job history')
    parser.add_argument('--resume-note', default='', help='Operator note prepended to the resumed STATE.md')
    parser.add_argument('--agent', choices=BACKENDS, default='claude', help='Worker agent CLI')
    parser.add_argument('--agent-model', help='Worker model (default depends on --agent)')
    parser.add_argument('--supervisor-agent', choices=BACKENDS, help='Supervisor agent CLI (default: --agent)')
    parser.add_argument('--supervisor-model', help='Supervisor model (default depends on its agent)')
    parser.add_argument('--optimize', help='Primary metric after the gates (default speedup_geomean): '
                        + ', '.join(objective.METRICS))
    parser.add_argument('--constraint', action='append', default=[],
                        help='Repeatable, e.g. "worst_cell>=0.995" or "text_bytes<=1.5*baseline"')
    parser.add_argument('--beam', type=int, help='Parents offered to the supervisor per turn (default 4)')
    args = parser.parse_args(argv)
    if args.controller:
        return loop(args)
    if not args.port or not args.family:
        parser.error('--port and --family are required (a directory under ports/ and one of its families)')
    options = {k: v for k, v in vars(args).items() if k not in ('port', 'cluster', 'controller')}
    try:
        launched = campaign.start(args.port, args.cluster, **options)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(launched))


if __name__ == '__main__':
    main()
