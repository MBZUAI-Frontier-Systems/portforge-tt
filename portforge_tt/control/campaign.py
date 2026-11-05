"""Start a campaign. ``portforge --port ports/dav1d-rvv --family refmvs-splat_mv --cluster my-cluster.yaml
--hours 6 --detach`` (controller.main) calls ``start``; from Python,
``campaign.start('ports/dav1d-rvv', 'my-cluster.yaml', family='refmvs-splat_mv', hours=6)``.

``start`` validates the pack, the family and the cluster, creates ``runs/<family>-<UTC time>/`` with a
fresh workspace (pack docs, shared guides, the family's SPEC.md and TECHNIQUES.md, and the starting
source: a ``--placeholder`` to port from, or the first ``--seed-kernel``), freezes the harness (shared
code plus the pack) into ``runs/<run>/harness`` and re-executes the controller from that frozen copy,
so a run always replays the exact gates it was graded by. ``resume=<earlier run>`` continues an
earlier campaign in a new run: its full workspace and STATE.md and read-only job history, plus an
optional trusted ``resume_note`` to both agents.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

from ..ranking import objective, search
from ..agents import DEFAULT_MODELS, repository_root
from .cluster import identity as digest, load as load_cluster
from .jobs import code_hash, snapshot
from .port import (PACKAGE, PORT, agent_environment, configure, freeze_harness, frozen_environment,
                   harness_identity, is_trusted, seed_documents, seed_policy)
from ..remote.runner import safe_file, write_json


def _labelled(values):
    """--seed-kernel label=path pairs (a bare path is labelled by its file name)."""
    out = {}
    for value in values or ():
        label, sep, path = str(value).partition('=')
        label, path = (label, path) if sep else (Path(value).stem, value)
        if not label or label in out:
            raise ValueError('seed kernels need distinct labels: label=path')
        out[label] = Path(path).read_text()
    return out


def transforms(note, parent=None):
    return json.dumps({'parent': parent, 'applied': [], 'novel': '', 'note': note}, indent=1) + '\n'


def resume_workspace(workspace, previous_run, backends, note=''):
    """Carry an earlier campaign's complete workspace, including STATE.md, into a new run.

    Job receipts stay in the earlier archive and are exposed read-only as history; no pass,
    pending job or submission is inherited, so every gate is re-established here. A source
    that already froze a submission may be continued; its certification verdict stays
    operator-only (the operator note may summarize it), so the held-out regimes are not
    optimized against directly.
    """
    root = Path(previous_run).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('seed_policy') != seed_policy():
        raise ValueError('resume source must be a campaign of this pack and family')
    current = snapshot(Path(manifest['workspace']))
    files = {name: text for name, text in current.items() if not is_trusted(name)}
    for name, content in files.items():
        target = safe_file(workspace, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    previous = '/'.join(manifest.get('agent_backends', {}).get(role, 'claude') for role in ('worker', 'supervisor'))
    header = ('# Resumed campaign\n\n'
        'This run continues ' + root.name + ' in a new archive (agents: ' + previous + ' -> '
        + backends['worker'] + '/' + backends['supervisor'] + ', worker/supervisor). Source, notes '
        'and the STATE below are carried unchanged. Job IDs recorded below belong to earlier runs: '
        'port_status(job_id=...) and port_log still read them as historical receipts, but no '
        'pass, pending job or acceptance is inherited. Re-establish gates in this run and '
        'do not repeat resolved investigations without new evidence.\n')
    if note.strip():
        header += '\n## Operator note\n\n' + note.strip() + '\n'
    (Path(workspace) / 'STATE.md').write_text(header + '\n---\n\n' + files.get('STATE.md', '(no STATE.md)\n'))
    return {'resumed_from': str(root), 'resumed_code_sha256': code_hash(files),
            'history_runs': [str(root)] + [str(p) for p in manifest.get('history_runs', [])],
            'placeholder_sha256': manifest.get('placeholder_sha256'), 'seeds': manifest.get('seeds', {})}


def start(port, cluster=None, *, family, hours=6, board_minutes=120, detach=True, run_dir=None,
          agent='claude', agent_model=None, supervisor_agent=None, supervisor_model=None,
          optimize=None, constraint=(), beam=None, seed_kernel=(), placeholder=None, dry_run=False,
          resume=None, resume_note=''):
    """Launch a campaign for one family of the pack at ``port``; raises ValueError on a bad selection.

    cluster: machine inventory (default: cluster.yaml in the working directory or beside the package).
    hours: agent budget in (0, 24]. board_minutes: X280 time for perf and certification. detach: run in
    the background and return {'pid', 'run'}; otherwise this process becomes the controller.
    seed_kernel: label=path kernels evaluated as baselines; the first one is the workspace's starting
    source unless placeholder= gives a stub to port from (default: the family's placeholder.S, if any).
    dry_run: evaluate the seeds and stop (no agents). agent / supervisor_agent: claude, codex or opencode.
    optimize / constraint: objective after the gates (see ranking/objective.py). beam: menus per turn.
    resume / resume_note: continue an earlier run of this family.
    """
    path = lambda value: Path(value) if value is not None else None
    cluster, run_dir, resume, placeholder = path(cluster), path(run_dir), path(resume), path(placeholder)
    constraint, seed_kernel = list(constraint or []), list(seed_kernel or [])
    if cluster is None:
        cluster = next((c for c in (Path.cwd() / 'cluster.yaml', PACKAGE.parent / 'cluster.yaml') if c.is_file()), None)
        if cluster is None:
            raise ValueError('no cluster.yaml found; copy the template and pass cluster=')
    configure(port, family)
    if resume and (seed_kernel or placeholder):
        raise ValueError('resume carries its workspace; seed_kernel and placeholder start a new one')
    if resume_note and not resume:
        raise ValueError('resume_note requires resume')
    supervisor_agent = supervisor_agent or agent
    agents = {'worker': {'backend': agent, 'model': agent_model or DEFAULT_MODELS[agent]},
              'supervisor': {'backend': supervisor_agent, 'model': supervisor_model or
                             (agent_model if supervisor_agent == agent and agent_model
                              else DEFAULT_MODELS[supervisor_agent])}}
    if not 0 < hours <= 24:
        raise ValueError('hours must be in (0,24]')
    if not 0 <= board_minutes <= 24 * 60:
        raise ValueError('board_minutes must be in [0, 1440]')
    run = (run_dir or Path.cwd() / 'runs' / (family + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    if detach:
        if run.exists():
            raise ValueError('run directory already exists')
        run.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'portforge_tt.control.controller', '--port', str(PORT['dir']),
                   '--family', family, '--hours', str(hours), '--board-minutes', str(board_minutes),
                   '--run-dir', str(run), '--cluster', str(cluster.resolve())]
        for flag, value in (('--placeholder', placeholder), ('--resume', resume)):
            if value:
                command += [flag, str(value.resolve())]
        for flag, value in (('--resume-note', resume_note), ('--optimize', optimize), ('--beam', beam)):
            if value:
                command += [flag, str(value)]
        for value in constraint:
            command += ['--constraint', value]
        for value in seed_kernel:
            label, sep, file = str(value).partition('=')
            command += ['--seed-kernel', label + '=' + str(Path(file).resolve()) if sep else str(Path(value).resolve())]
        if dry_run:
            command.append('--dry-run')
        command += ['--agent', agents['worker']['backend'], '--supervisor-agent', agents['supervisor']['backend']]
        for flag, value in (('--agent-model', agents['worker']['model']), ('--supervisor-model', agents['supervisor']['model'])):
            if value:
                command += [flag, value]
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
        raise ValueError(f'{cluster}: {exc!r}; fill every <placeholder> (see cluster.yaml)') from exc
    optimization = objective.spec(PORT['optimization'], optimize, constraint)
    if beam is not None and not 1 <= beam <= 16:
        raise ValueError('beam must be in 1..16')
    seeds = _labelled(seed_kernel)
    default = PORT['dir'] / 'kernels' / family / ('placeholder' + Path(PORT['source_name']).suffix)
    placeholder = placeholder or (default if default.is_file() and not seeds and not resume else None)
    if not (resume or seeds or placeholder):
        raise ValueError('a new campaign needs a starting source: --placeholder or --seed-kernel label=path')
    repository = repository_root()
    if repository == Path('/'):
        raise ValueError('explicit repository isolation root required')
    run.mkdir(parents=True)
    workspace = Path(tempfile.mkdtemp(prefix='portforge-fresh-'))
    for name, text in seed_documents(PORT, PACKAGE / 'guides', search.known(PORT)).items():
        (workspace / name).write_text(text)
    backends = {role: agents[role]['backend'] for role in agents}
    if resume:
        provenance = resume_workspace(workspace, resume, backends, resume_note)
    else:
        if placeholder:
            text, note = placeholder.read_text(), 'placeholder: port from scratch'
        else:
            label, text = next(iter(seeds.items()))
            note = 'seed ' + label
        (workspace / PORT['source_name']).write_text(text)
        (workspace / 'TRANSFORMS.json').write_text(transforms(note))
        (workspace / 'STATE.md').write_text('# New campaign: ' + family + '\n\nStarting source: ' + note + '.\n')
        provenance = {'seeds': {label: code_hash({PORT['source_name']: text}) for label, text in seeds.items()},
                      'placeholder_sha256': code_hash({PORT['source_name']: text}) if placeholder else None}
        for label, text in seeds.items():
            write_json(run / 'seeds' / (label + '.json'), {PORT['source_name']: text,
                                                           'TRANSFORMS.json': transforms('seed ' + label)})
    freeze_harness(run / 'harness')
    manifest = {'workspace': str(workspace), 'pid': os.getpid(), 'repository_root': str(repository),
                'harness_sha256': harness_identity(run / 'harness'), 'pack': PORT['key'], 'family': family,
                'seed_files': sorted(p.name for p in workspace.iterdir()), 'budget_hours': hours,
                'board_minutes': board_minutes, 'cluster_file': str(cluster.resolve()),
                'seed_policy': seed_policy(), 'agent_backends': backends,
                'agent_models': {role: agents[role]['model'] for role in agents},
                'operator_note': (resume_note or '').strip(), 'optimization': optimization, 'beam': beam,
                'dry_run': bool(dry_run), 'heldout_seed': secrets.randbelow(2 ** 31),  # never shown to agents
                **provenance}
    write_json(run / 'manifest.json', manifest)
    command = [sys.executable, '-m', 'portforge_tt.control.controller', '--controller', '--run-dir', str(run),
               '--cluster', str(cluster.resolve()), '--hours', str(hours)]
    # From inside the harness, so `-m` resolves the frozen package, never a checkout in the working directory.
    os.chdir(run / 'harness')
    os.execve(sys.executable, command, {**frozen_environment(run, manifest), **agent_environment(manifest)})


def status(run):
    """A run's latest controller status, search frontier and submission (None until written)."""
    run = Path(run)
    read = lambda path: json.loads(path.read_text()) if path.is_file() else None
    return {'status': read(run / 'status.json'), 'frontier': read(run / 'search' / 'frontier.json'),
            'submission': read(run / 'submission.json')}
