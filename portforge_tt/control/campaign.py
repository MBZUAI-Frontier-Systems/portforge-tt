"""Start a campaign. ``portforge --port ports/chronos2 --cluster my-cluster.yaml --hours 6 --detach``
(controller.main) calls ``start``; from Python, ``campaign.start('ports/chronos2', 'my-cluster.yaml', hours=6)``.

``start`` validates the pack, the corpus and the cluster, creates ``runs/<UTC time>/`` with a fresh
workspace (pack docs plus shared guides), freezes the harness (shared code plus the pack) into
``runs/<run>/harness`` and re-executes the controller from that frozen copy, so a run always replays
the exact evaluator it was graded by. ``resume=<earlier run>`` continues an earlier campaign in a
new run: its full workspace and STATE.md, its FP32 references and read-only job history, plus an
optional trusted ``resume_note`` to both agents. Neither agent receives an earlier private port.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if __package__:
    from ..ranking import objective
    from ..agents import DEFAULT_MODELS, repository_root
    from .cluster import identity as digest, load as load_cluster
    from .jobs import code_hash, complete_references, snapshot
    from .port import (PACKAGE, PORT, agent_environment, configure, freeze_harness, frozen_environment,
                       gated_stages, harness_identity, load_evaluator, same_evaluator, seed_documents,
                       seed_policy, trusted_documents)
    from ..remote.portforge_eval import check_suite
    from ..remote.portforge_models import model_spec, use_registry
    from ..remote.runner import safe_file, write_json
else:  # flat: the frozen run harness
    import objective
    from agents import DEFAULT_MODELS, repository_root
    from cluster import identity as digest, load as load_cluster
    from jobs import code_hash, complete_references, snapshot
    from port import (PACKAGE, PORT, agent_environment, configure, freeze_harness, frozen_environment,
                      gated_stages, harness_identity, load_evaluator, same_evaluator, seed_documents,
                      seed_policy, trusted_documents)
    from portforge_eval import check_suite
    from portforge_models import model_spec, use_registry
    from runner import safe_file, write_json


def check_suites(corpus, model):
    """Build every stage with the pack's evaluator and reject impossible gates before a run starts."""
    # Load this pack's evaluator strictly by path so no other "evaluate" module
    # can shadow it at launch time.
    evaluator = load_evaluator('portforge_pack_evaluate_launch')
    for stage in gated_stages() + ['full']:
        suite = check_suite(evaluator.make_suite(corpus, stage, model)[0])
        if suite.get('mesh_cards') != (PORT['mesh_cards'] if stage == 'mesh' else None):
            raise ValueError('port.json mesh_cards and the evaluator mesh suite disagree')



def seed_workspace(workspace, seed_run, model, seed_job=None):
    """Copy an explicitly selected fresh campaign, never its acceptance records."""
    root = Path(seed_run).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    seed_model = manifest.get('model')
    trusted = trusted_documents()
    if (sorted(manifest.get('seed_files', [])) != sorted(trusted)
            and manifest.get('seed_policy') != seed_policy()):
        raise ValueError('seed must be a fresh public-source campaign of this port')
    submission = root / 'submission.json'
    if seed_job:
        if Path(seed_job).name != seed_job or seed_job in ('.', '..'):
            raise ValueError('seed job must be a single job identifier')
        job = safe_file(root, 'jobs/' + seed_job)
        files = json.loads((job / 'source.json').read_text())
        request = json.loads((job / 'request.json').read_text())
        seed_model = request.get('model', seed_model)
        result = json.loads((job / 'result.json').read_text())
        if (request.get('kind') != 'evaluate' or request.get('stage') != 'bringup'
                or result.get('returncode') != 0 or result.get('timeout') or result.get('failure')
                or result.get('verdict', {}).get('passed') is not True
                or code_hash(files) != request.get('code_sha256')):
            raise ValueError('seed job must contain a passing, intact bringup source')
    elif submission.exists():
        frozen = json.loads(submission.read_text())
        seed_model = frozen.get('model', seed_model)
        files = frozen['files']
        if code_hash(files) != frozen['code_sha256']:
            raise ValueError('seed submission source identity mismatch')
    else:
        files = snapshot(Path(manifest['workspace']))
    omitted, retained = [], []
    for name, content in files.items():
        if Path(name).as_posix() in trusted:
            continue
        if Path(name).suffix == '.md':
            evidence = seed_model == model and Path(name).name.startswith(
                ('BASELINE_', 'BF16_BASELINE_', 'CURRENT_BF16_', 'CANDIDATE_', 'BFP8_'))
            if name not in ('RESEARCH.md', 'REPORT.md') and not evidence:
                omitted.append(name)
                continue
            retained.append(name)
        target = safe_file(workspace, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (Path(workspace) / 'STATE.md').write_text('# New campaign: ' + model + '\n\n'
        'Executable source and selected evidence were carried from an explicitly '
        'selected earlier fresh campaign. Old job IDs and results are historical: this campaign '
        'has no inherited pending jobs or passes. Follow the current selected-model TASK.md and '
        'CONTRACT.md, adapt the checkpoint, and re-evaluate. Retained baselines and evidence '
        'are historical under the archived runtime, not current instructions or passes. '
        'After a runtime change, revalidate the source and its same-precision baseline; then '
        'advance optimization and standalone deliverables. Do not repeat resolved original-runtime '
        'investigations unless new evidence warrants them.\n')
    return {'seed_run': str(root), 'seed_job': seed_job, 'seed_code_sha256': code_hash(files),
            'omitted_seed_notes': sorted(omitted), 'retained_seed_notes': sorted(retained)}



def resume_workspace(workspace, previous_run, model, backends, note=''):
    """Carry an earlier campaign's complete workspace, including STATE.md, into a new run.

    Unlike seed_workspace this keeps every note, because the campaign continues. Job
    receipts stay in the earlier archive and are exposed read-only as history; no pass,
    pending job or submission is inherited, so every gate is re-established here. A
    source that already froze a submission may be continued: its workspace must still
    match the frozen code, and its full-corpus verdict stays operator-only (the operator
    note may summarize it) so the held-out corpus is not optimized against directly.
    """
    root = Path(previous_run).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('seed_policy') != seed_policy():
        raise ValueError('resume source must be a fresh public-source campaign of this port')
    if manifest.get('model') != model:
        raise ValueError('resume source targets a different model')
    trusted = trusted_documents()
    current = snapshot(Path(manifest['workspace']))
    submitted = root / 'submission.json'
    if submitted.exists() and code_hash(current) != json.loads(submitted.read_text())['code_sha256']:
        raise ValueError('resume source workspace differs from its frozen submission')
    files = {name: text for name, text in current.items()
             if name not in trusted and not any(part.startswith('.') for part in Path(name).parts)}
    for name, content in files.items():
        target = safe_file(workspace, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    previous = '/'.join(manifest.get('agent_backends', {}).get(role, 'opencode')
                        for role in ('worker', 'supervisor'))
    header = ('# Resumed campaign\n\n'
        'This run continues ' + root.name + ' in a new archive (agents: ' + previous + ' -> '
        + backends['worker'] + '/' + backends['supervisor'] + ', worker/supervisor). Source, notes '
        'and the STATE below are carried unchanged. Job IDs recorded below belong to earlier runs: '
        'port_status(job_id=...) and port_log still read them as historical receipts, but no '
        'pass, pending job or baseline acceptance is inherited. Re-establish gates in this run and '
        'do not repeat resolved investigations without new evidence.\n')
    if note.strip():
        header += '\n## Operator note\n\n' + note.strip() + '\n'
    (Path(workspace) / 'STATE.md').write_text(header + '\n---\n\n' + files.get('STATE.md', '(no STATE.md)\n'))
    return {'resumed_from': str(root), 'resumed_code_sha256': code_hash(files),
            'history_runs': [str(root)] + [str(p) for p in manifest.get('history_runs', [])]}


def start(port, cluster=None, *, hours=6, detach=True, corpus=None, model=None, run_dir=None,
          agent='opencode', agent_model=None, supervisor_agent=None, supervisor_model=None,
          optimize=None, constraint=(), scenario=None, beam=None,
          resume=None, resume_note='', seed_run=None, seed_job=None, reference_run=None):
    """Launch a campaign for the pack at ``port``; raises ValueError on a bad selection.

    cluster: machine inventory (default: cluster.yaml in the working directory or beside the package).
    hours: budget in (0, 24]. detach: run in the background and return {'pid', 'run'}; otherwise this
    process becomes the controller. corpus: frozen benchmark (default: the pack's corpus.json).
    model: registry key to port (default: the pack's key). run_dir: default ./runs/<UTC time>.
    agent / supervisor_agent: opencode, claude or codex, with optional model overrides.
    optimize / constraint / scenario: objective after the gates (see objective.py; default Pareto).
    beam: parallel search width (default: TT cards in the cluster).
    resume / resume_note: continue an earlier run. seed_run / seed_job: start from an earlier fresh
    run's workspace (or one of its passing bringup snapshots). reference_run: reuse its FP32 references.
    """
    path = lambda value: Path(value) if value is not None else None
    cluster, corpus, run_dir = path(cluster), path(corpus), path(run_dir)
    resume, seed_run, reference_run = path(resume), path(seed_run), path(reference_run)
    constraint = list(constraint or [])
    if cluster is None:
        cluster = next((c for c in (Path.cwd() / 'cluster.yaml', PACKAGE.parent / 'cluster.yaml') if c.is_file()), None)
        if cluster is None:
            raise ValueError('no cluster.yaml found; copy the template and pass cluster=')
    configure(port)
    use_registry(PORT['dir'] / 'models.json')
    corpus = corpus or PORT['corpus_path']
    if not corpus:
        raise ValueError('this pack ships no corpus.json; pass corpus= with the frozen benchmark')
    model = model or PORT['key']
    if model not in PORT['delivery_models']:
        raise ValueError('model must be one of the pack delivery models: ' + ', '.join(PORT['delivery_models']))
    if seed_job and not seed_run:
        raise ValueError('seed_job requires seed_run')
    if resume and seed_run:
        raise ValueError('resume and seed_run are exclusive')
    if resume_note and not resume:
        raise ValueError('resume_note requires resume')
    supervisor_agent = supervisor_agent or agent
    agents = {'worker': {'backend': agent, 'model': agent_model or DEFAULT_MODELS[agent]},
              'supervisor': {'backend': supervisor_agent, 'model': supervisor_model or
                             (agent_model if supervisor_agent == agent and agent_model
                              else DEFAULT_MODELS[supervisor_agent])}}
    model_spec(model)
    if not 0 < hours <= 24:
        raise ValueError('hours must be in (0,24]')
    # Run archives land under ./runs of the working directory (never inside an
    # installed package), or exactly where run_dir points.
    run = (run_dir or Path.cwd() / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    if detach:
        if run.exists():
            raise ValueError('run directory already exists')
        run.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'portforge_tt.control.controller', '--port', str(PORT['dir']),
                   '--corpus', str(Path(corpus).resolve()), '--hours', str(hours), '--run-dir', str(run),
                   '--cluster', str(cluster.resolve()), '--model', model]
        for flag, value in (('--seed-run', seed_run), ('--reference-run', reference_run), ('--resume', resume)):
            if value:
                command += [flag, str(value.resolve())]
        for flag, value in (('--seed-job', seed_job), ('--resume-note', resume_note), ('--optimize', optimize),
                            ('--scenario', scenario), ('--beam', beam)):
            if value:
                command += [flag, str(value)]
        for value in constraint:
            command += ['--constraint', value]
        command += ['--agent', agents['worker']['backend'], '--agent-model', agents['worker']['model'],
                    '--supervisor-agent', agents['supervisor']['backend'],
                    '--supervisor-model', agents['supervisor']['model']]
        with run.with_suffix('.log').open('wb') as log:
            environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, (str(PACKAGE.parent), os.environ.get('PYTHONPATH'))))}
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                                     env=environment)
        if sys.platform == 'darwin':
            subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(child.pid)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        launched = {'pid': child.pid, 'run': str(run)}
        write_json(run.with_suffix('.launch.json'), launched)
        return launched
    try:
        load_cluster(cluster)  # Validate before creating a campaign archive.
    except (ValueError, KeyError, OSError) as exc:
        raise ValueError(f'{cluster}: {exc!r}; fill every <placeholder> (see README.md, Machines)') from exc
    corpus_data = json.loads(Path(corpus).read_text())
    if digest(corpus_data) != PORT['corpus_sha256']:
        raise ValueError('corpus identity mismatch')
    optimization = objective.spec(PORT['optimization'], optimize, constraint, scenario)
    if beam is not None and not 1 <= beam <= 64:
        raise ValueError('beam must be in 1..64')
    check_suites(corpus_data, model)
    repository = repository_root()
    if repository == Path('/'):
        raise ValueError('explicit repository isolation root required')
    run.mkdir(parents=True)
    workspace = Path(tempfile.mkdtemp(prefix='portforge-fresh-'))
    for name, text in seed_documents(PORT, PACKAGE / 'guides').items():
        (workspace / name).write_text(text)
    task = workspace / 'TASK.md'
    task.write_text('# Selected campaign model\n\nThis campaign targets ' + model_spec(model)['repo_id']
                    + ' (model key ' + model + '). This selection overrides size-specific defaults below.\n\n'
                    + task.read_text())
    provenance = seed_workspace(workspace, seed_run, model, seed_job) if seed_run else {}
    if resume:
        provenance = resume_workspace(workspace, resume, model,
                                      {role: agents[role]['backend'] for role in agents}, resume_note)
        if not reference_run and complete_references(resume.resolve(), model) and same_evaluator(resume.resolve()):
            reference_run = resume
    freeze_harness(run / 'harness')
    write_json(run / 'corpus.json', corpus_data)
    manifest = {'workspace': str(workspace), 'pid': os.getpid(), 'repository_root': str(repository),
                'corpus_sha256': digest(corpus_data), 'harness_sha256': harness_identity(run / 'harness'),
                'seed_files': sorted(p.name for p in workspace.iterdir()), 'budget_hours': hours,
                'cluster_file': str(cluster.resolve()), 'model': model,
                'seed_policy': seed_policy(),
                'agent_backends': {role: agents[role]['backend'] for role in agents},
                'agent_models': {role: agents[role]['model'] for role in agents},
                'operator_note': resume_note.strip(),
                'optimization': optimization, 'beam': beam,
                'reference_run': str(reference_run.resolve()) if reference_run else None, **provenance}
    write_json(run / 'manifest.json', manifest)
    command = [sys.executable, str(run / 'harness/controller.py'), '--controller', '--run-dir', str(run),
               '--cluster', str(cluster.resolve()), '--corpus', str(run / 'corpus.json'),
               '--hours', str(hours), '--model', model]
    if reference_run:
        command += ['--reference-run', str(reference_run.resolve())]
    os.execve(sys.executable, command, {**frozen_environment(run, manifest), **agent_environment(manifest)})


def status(run):
    """A run's latest controller status, search frontier and submission (None until written)."""
    run = Path(run)
    read = lambda path: json.loads(path.read_text()) if path.is_file() else None
    return {'status': read(run / 'status.json'), 'frontier': read(run / 'search' / 'frontier.json'),
            'submission': read(run / 'submission.json')}
