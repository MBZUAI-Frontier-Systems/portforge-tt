"""Receipts and remote jobs: workspace snapshots, code identity and the trusted gates on the x86 host.

What an agent's request becomes. ``experiment`` runs one job on the host through the frozen
harness (remote/runner.py and the pack adapter):

  evaluate/check   G0 compile + static profile, G1 the project's tests under QEMU at every VLEN,
                   G2 the bit-exact check over the family's regime matrix
  evaluate/perf    check, then G3/G4: one X280 bench session over the matrix; a flagged cell is
                   re-measured in a new session at twice the rounds and only a flag seen in both counts
  run              untrusted conveniences: compile (static profile) or check (QEMU at VLEN 128/1024)
  certify          after port_submit: check + a second bench session, held-out regimes under QEMU and on
                   the X280, and the family's verify methods (verify/contract.py)

``context`` fetches the family's reference and other-ISA sources once per run. Every outcome is a
file: ``runs/<run>/jobs/<id>/{request,source,result}.json``.
"""
import hashlib
import json
from pathlib import Path
import time

from . import cluster
from .cluster import identity as digest, run_root
from .port import PORT, CONTEXT_DIR, is_trusted
from ..remote.runner import write_json
from .runtime import job

SOURCE_SUFFIXES = ('.c', '.S', '.h', '.json', '.md', '.txt')
GATE_TIMEOUT = 3 * 3600
CERTIFY_TIMEOUT = 8 * 3600
BENCH_ROUNDS = 15


def snapshot(workspace):
    files = {}
    for path in sorted(Path(workspace).rglob('*')):
        if path.is_symlink():
            raise ValueError('symlinks are not allowed')
        if path.is_file() and path.suffix in SOURCE_SUFFIXES and not any(p.startswith('.') for p in path.parts):
            if path.stat().st_size > 2_000_000:
                raise ValueError('file too large')
            files[str(path.relative_to(workspace))] = path.read_text()
    if sum(len(x.encode()) for x in files.values()) > 8_000_000:
        raise ValueError('workspace too large')
    return files


def candidate(files):
    """The worker-owned part of a snapshot (the kernel source, TRANSFORMS.json and notes); instructions
    and context/ are trusted copies and stay out of receipts."""
    return {name: text for name, text in files.items() if not is_trusted(name)}


def code_hash(files):
    """Identity of a candidate: its kernel source alone (notes and TRANSFORMS.json do not change what runs)."""
    return digest({PORT['source_name']: files.get(PORT['source_name'], '')})


def ukernel(code):
    """The symbol a candidate is registered under in the project's build and test registry."""
    token = PORT['entry'].get('neon_ukernel', 'xnn_' + PORT['family'].replace('-', '_') + '_ukernel').split('_ukernel')[0]
    return f"{token}_ukernel__rvv_n{code[:10]}"


def suite_identity(run):
    """What a verdict was measured against: the family's frozen regimes and techniques plus its adapter."""
    pack = Path(run) / 'harness' / 'ports' / PORT['dir'].name
    folder = pack / 'kernels' / PORT['family']
    return digest({'family': PORT['family'], 'evaluator': (pack / 'evaluate.py').read_text(),
                   'regimes': (folder / 'regimes.json').read_text() if (folder / 'regimes.json').is_file() else ''})


def passing_evaluation(result, job, signature, stage):
    """Require an unambiguous trusted terminal receipt for this exact evaluation."""
    if not isinstance(result, dict):
        return False
    verdict = result.get('verdict')
    if not isinstance(verdict, dict):
        return False
    return (type(result.get('returncode')) is int and result['returncode'] == 0
            and not any(result.get(key) for key in ('failure', 'timeout'))
            and result.get('job') == job and result.get('cluster_sha256') == signature
            and verdict.get('passed') is True and verdict.get('stage') == stage
            and isinstance(verdict.get('suite_sha256'), str) and bool(verdict['suite_sha256']))


def history_runs(run):
    """Earlier runs this campaign continues, newest first; read-only receipts only."""
    manifest = Path(run) / 'manifest.json'
    if not manifest.is_file():
        return []
    return [Path(p) for p in json.loads(manifest.read_text()).get('history_runs', [])]


def board_seconds(run):
    """Board time spent by this run's jobs (budgeted by --board-minutes)."""
    total = 0.0
    for path in Path(run).glob('jobs/*/result.json'):
        total += float(json.loads(path.read_text()).get('board_seconds') or 0)
    return total


def _seed(job_id, salt=0):
    return int(hashlib.sha256((job_id + str(salt)).encode()).hexdigest()[:8], 16) % 2_000_000_000


def _slim(verdict):
    """A bench verdict without per-round samples (megabytes for a large matrix)."""
    keep = ('metrics', 'geomean', 'flagged', 'wrong_cells', 'correctness_wins', 'feasible', 'reference_cell', 'delta',
            'views')  # views: the same rows against other baselines (portable upstream, the X280's dispatched kernel)
    slim = {k: verdict.get(k) for k in keep}
    slim['cells'] = {k: {kk: c.get(kk) for kk in ('speedup', 'baseline', 'cycles_per_elem', 'excluded',
                                                   'candidate_correct', 'flagged')}
                     for k, c in (verdict.get('cells') or {}).items()}
    return slim


def _gates(config, run, job, request, name, source):
    """G0-G2 for one candidate; returns (verdict fields, profile) or raises on an infrastructure failure."""
    root = run_root(config, run)
    r = cluster.remote(config, run, {'kind': 'gates', 'pack': 'ports/' + PORT['dir'].name, 'family': PORT['family'],
                                     'tag': ('camp-' + Path(run).name)[:40], 'kernels': {name: source},
                                     'seed': request.get('seed', 1), 'out': root + '/jobs/' + job + '/gates'},
                       timeout=GATE_TIMEOUT)
    if not r.get('ok'):
        raise RuntimeError('gates: ' + str(r.get('error'))[-1500:])
    res = r['results'][name]
    gates = {g: res[g] for g in ('G0', 'G1', 'G2') if g in res}
    passed = all(gates.get(g, {}).get('passed') for g in ('G0', 'G1', 'G2'))
    failed = next((g for g in ('G0', 'G1', 'G2') if not gates.get(g, {}).get('passed')), None)
    return {'gates': gates, 'passed': passed, 'failures': [] if passed else ['failed at ' + failed]}, \
        res.get('profile'), r.get('rpc')


def _bench(config, run, job, request, name, rounds, salt):
    root = run_root(config, run)
    start = time.time()
    b = cluster.remote(config, run, {'kind': 'bench', 'pack': 'ports/' + PORT['dir'].name, 'family': PORT['family'],
                                     'tag': ('camp-' + Path(run).name)[:40], 'kernels': [name], 'rounds': rounds,
                                     'delta': PORT['noise_floor'], 'seed': _seed(job, salt),
                                     'out': root + '/jobs/' + job + '/bench-' + str(salt)}, timeout=GATE_TIMEOUT)
    seconds = time.time() - start
    if not b.get('ok'):
        raise RuntimeError('bench: ' + str(b.get('error'))[-1500:])
    return b['verdicts'][name], seconds


def measure(config, run, job, request, name, verdict):
    """G3/G4 for a candidate that passed check: one session, plus a confirmation session at twice the
    rounds when a cell is flagged (cross-session drift at DRAM-sized shapes exceeds the in-session noise
    floor, so only cells flagged in both sessions count against the no-regression constraint)."""
    rounds = request.get('rounds', BENCH_ROUNDS)
    first, spent = _bench(config, run, job, request, name, rounds, 1)
    sessions = [first]
    if first.get('flagged'):
        second, more = _bench(config, run, job, request, name, 2 * rounds, 2)
        sessions.append(second)
        spent += more
    latest = _slim(sessions[-1])
    confirmed = sorted(set(sessions[-1].get('flagged') or []) & set(sessions[0].get('flagged') or [])) \
        if len(sessions) > 1 else list(first.get('flagged') or [])
    latest.update(confirmed_flags=confirmed,
                  sessions=[{'metrics': s.get('metrics'), 'flagged': s.get('flagged')} for s in sessions])
    wrong = latest.get('wrong_cells') or []
    verdict.update(perf=latest, passed=verdict['passed'] and not wrong,
                   failures=verdict['failures'] + (['wrong after timing: ' + ', '.join(map(str, wrong[:5]))]
                                                   if wrong else []))
    return spent


def _upload(config, run, job, files):
    """Put the candidate source on the host; returns its work_dir-relative path."""
    endpoint, root = config['host'], run_root(config, run)
    cluster.upload(endpoint, endpoint['work_dir'] + '/' + root + '/jobs/' + job,
                   {PORT['source_name']: files[PORT['source_name']]})
    return root + '/jobs/' + job + '/' + PORT['source_name']


@job(resources={'host': 1})
def experiment(run_dir, job, request, config):
    """One trusted job on the host; see the module docstring for the kinds."""
    run, kind, stage = Path(run_dir), request.get('kind', 'evaluate'), request.get('stage')
    result = {'returncode': 0, 'job': job, 'kind': kind, 'stage': stage}
    try:
        files = request['files']
        if PORT['source_name'] not in files:
            raise ValueError(PORT['source_name'] + ' missing from the candidate')
        code = code_hash(files)
        name, source = ukernel(code), _upload(config, run, job, files)
        if kind in ('run', 'profile'):
            command = 'dev_compile' if kind == 'profile' or request.get('command') == 'compile' else 'dev_check'
            r = cluster.remote(config, run, {'kind': command, 'pack': 'ports/' + PORT['dir'].name,
                                             'family': PORT['family'], 'tag': job[:12], 'source': source,
                                             'out': run_root(config, run) + '/jobs/' + job + '/check'},
                               timeout=1800)
            if not r.get('ok'):
                raise RuntimeError(command + ': ' + str(r.get('error'))[-1500:])
            result.update(rpc=r.pop('rpc', None), output={k: v for k, v in r.items() if k != 'ok'})
        elif kind in ('evaluate', 'certify'):
            verdict, profile, rpc = _gates(config, run, job, request, name, source)
            verdict.update(stage=stage, suite_sha256=suite_identity(run), kernel=name)
            result.update(rpc=rpc, profile=profile, verdict=verdict)
            if verdict['passed'] and stage in ('perf', 'full'):
                result['board_seconds'] = round(measure(config, run, job, request, name, verdict), 1)
            if kind == 'certify' and verdict['passed']:
                certify(config, run, job, request, name, source, verdict)
        else:
            raise ValueError('unknown job kind ' + repr(kind))
        result['cluster_sha256'] = digest(config)
    except Exception as exc:
        result = {'returncode': -1, 'job': job, 'kind': kind, 'stage': stage, 'failure': 'infrastructure',
                  'error': str(exc)[-2000:], 'cluster_sha256': digest(config)}
    write_json(run / 'jobs' / job / 'result.json', result)
    return result


def certify(config, run, job, request, name, source, verdict):
    """G5: held-out regimes (a seed never shown to agents) under QEMU and on the X280, the family's verify
    methods, and agreement of this bench session with the submitted perf receipt within max(2 delta, 1%)."""
    r = cluster.remote(config, run, {'kind': 'certify', 'pack': 'ports/' + PORT['dir'].name, 'family': PORT['family'],
                                     'tag': ('camp-' + Path(run).name)[:40], 'kernel': name, 'source': source,
                                     'heldout_seed': request['heldout_seed'],
                                     'out': run_root(config, run) + '/jobs/' + job + '/certify'},
                       timeout=CERTIFY_TIMEOUT)
    if not r.get('ok') and 'heldout' not in r:
        raise RuntimeError('certify: ' + str(r.get('error'))[-1500:])
    held = (r.get('heldout') or {}).get('passed') is True
    methods = r.get('verify') or []
    formal = all(m.get('passed') for m in methods)
    before = (request.get('submitted_metrics') or {}).get('speedup_geomean')
    now = ((verdict.get('perf') or {}).get('metrics') or {}).get('speedup_geomean')
    stable = None if not (before and now) else abs(before / now - 1) <= max(2 * PORT['noise_floor'], 0.01)
    failures = ([] if held else ['held-out regimes failed']) + ([] if formal else
        ['verify: ' + ', '.join(f"{m.get('method')} {m.get('verdict')}" for m in methods if not m.get('passed'))]) \
        + (['sessions disagree: %.4f then %.4f' % (before, now)] if stable is False else [])
    verdict.update(heldout=r.get('heldout'), verify=methods, sessions_agree=stable, certify_error=r.get('error'),
                   passed=verdict['passed'] and held and formal and stable is not False,
                   failures=verdict['failures'] + failures)


@job(resources={'host': 1})
def context(run_dir, config):
    """The family's reference, test and other-ISA sources (once per run), saved under runs/<run>/context/."""
    folder = Path(run_dir) / CONTEXT_DIR
    if folder.is_dir() and any(folder.iterdir()):
        return sorted(p.name for p in folder.iterdir())
    r = cluster.remote(config, run_dir, {'kind': 'context', 'pack': 'ports/' + PORT['dir'].name,
                                         'family': PORT['family']}, timeout=1800)
    if not r.get('ok'):
        raise RuntimeError('context: ' + str(r.get('error'))[-1500:])
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (r.get('sources') or {}).items():
        (folder / (name if '.' in name else name + '.c')).write_text(text)
    return sorted(p.name for p in folder.iterdir())
