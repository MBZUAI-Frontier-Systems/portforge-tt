"""Receipts and remote jobs: workspace snapshots, code identity, FP32 references and TT dispatch.

What an agent's request becomes. ``experiment`` runs one TT job (one card, or every mesh card):
it fetches the FP32 reference for its stage, runs the trusted runner on the TT host and, for a
gated stage, the trusted assessment. ``reference`` produces and caches the FP32 oracle on the
reference host; ``precision_study`` records the GPU precision context. Every outcome is a file:
``runs/<run>/jobs/<id>/{request,source,result}.json``.
"""
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile

if __package__:
    from . import cluster
    from ..remote import portforge_eval
    from .cluster import identity as digest
    from .port import PORT, harness_identity, pack_evaluator
    from ..remote.portforge_eval import EVALUATOR_FILES
    from ..remote.portforge_models import model_spec, unique_object
    from ..remote.runner import write_json, full_job_timeout
    from .runtime import job, get
else:  # flat: the frozen run harness
    import cluster, portforge_eval
    from cluster import identity as digest
    from port import PORT, harness_identity, pack_evaluator
    from portforge_eval import EVALUATOR_FILES
    from portforge_models import model_spec, unique_object
    from runner import write_json, full_job_timeout
    from runtime import job, get


def snapshot(workspace):
    files = {}
    for path in sorted(Path(workspace).rglob('*')):
        if path.is_symlink():
            raise ValueError('symlinks are not allowed')
        if path.is_file() and path.suffix in ('.py', '.md', '.json', '.cpp', '.hpp', '.h', '.txt'):
            if path.stat().st_size > 2_000_000:
                raise ValueError('file too large')
            files[str(path.relative_to(workspace))] = path.read_text()
    if sum(len(x.encode()) for x in files.values()) > 8_000_000:
        raise ValueError('workspace too large')
    return files



def code_hash(files):
    return digest({k: v for k, v in files.items() if not k.endswith('.md')})



def suite_identity(run, model, stage):
    return digest({'model': model_spec(model), 'stage': stage, 'corpus': PORT['corpus_sha256'],
                   'evaluator': (Path(run) / 'harness/evaluate.py').read_text()})



def passing_evaluation(result, job, signature, model, precision, stage):
    """Require an unambiguous trusted terminal receipt for this exact evaluation."""
    if not isinstance(result, dict):
        return False
    verdict = result.get('verdict')
    if not isinstance(verdict, dict) or not isinstance(verdict.get('precision'), dict):
        return False
    return (type(result.get('returncode')) is int and result['returncode'] == 0
            and not any(result.get(key) for key in ('failure', 'timeout', 'host_memory_limit'))
            and result.get('job') == job and result.get('cluster_sha256') == signature
            and result.get('model') == model and result.get('precision') == precision
            and verdict.get('passed') is True and not verdict.get('failures')
            and verdict.get('stage') == stage and verdict.get('model_key') == model
            and verdict['precision'].get('requested') == precision
            and isinstance(verdict.get('suite_sha256'), str) and bool(verdict['suite_sha256']))



def without_raw_samples(result):
    """Drop per-repeat timing arrays from a receipt shown to agents; the percentiles beside them stay.
    A perf verdict keeps thousands of samples per case (megabytes), more than a tool reply can carry."""
    checks = (result.get('verdict') or {}).get('checks') if isinstance(result, dict) else None
    if not isinstance(checks, list) or not any(isinstance(c, dict) and isinstance(c.get('seconds'), list) for c in checks):
        return result
    checks = [dict({k: v for k, v in c.items() if k != 'seconds'}, seconds_omitted=len(c['seconds']))
              if isinstance(c, dict) and isinstance(c.get('seconds'), list) else c for c in checks]
    return dict(result, verdict=dict(result['verdict'], checks=checks))



def history_runs(run):
    """Earlier runs this campaign continues, newest first; read-only receipts only."""
    manifest = Path(run) / 'manifest.json'
    if not manifest.is_file():
        return []
    return [Path(p) for p in json.loads(manifest.read_text()).get('history_runs', [])]



def reference_progress(config, job):
    records = sorted(Path(job).glob('*-job.json'), key=lambda p: p.stat().st_mtime)
    if not records:
        return {'phase': 'preparing_reference', 'background_nonblocking': True}
    saved = json.loads(records[-1].read_text())
    endpoint = config['reference']
    if saved['endpoint_sha256'] != digest(endpoint):
        raise ValueError('reference progress endpoint mismatch')
    target = shlex.quote(saved['directory'] + '/output/progress.json')
    payload = cluster.ssh(endpoint, 'if test -f ' + target + '; then cat ' + target
                  + "; else echo '{\"phase\":\"reference_starting\"}'; fi", timeout=10)
    progress = json.loads(payload)
    return {'phase': progress.get('phase', 'reference_running'), 'background_nonblocking': True}



def validate_reference_archive(path, corpus, stage, model):
    """Inspect immutable short-stage oracle bytes before any remote extraction."""
    evaluator = pack_evaluator()
    make_suite = evaluator.make_suite
    suite_model = lambda spec: portforge_eval.suite_model(spec, evaluator.KEY, evaluator.GATES)
    validate_artifact = lambda meta, spec, identity: portforge_eval.validate_artifact(
        meta, spec, identity, evaluator.TIMING_PROTOCOL)
    if stage not in ('smoke', 'bringup'):
        raise ValueError('only completed short reference stages can be imported')
    path = Path(path)
    if not 0 < path.stat().st_size <= 512 * 1024 ** 2:
        raise ValueError('reference archive exceeds the short-stage size limit')
    data, entries = path.read_bytes(), {}
    required = {'input/suite.json', 'input/config.json', 'input/inputs.npz',
                'oracle/oracle.npz', 'oracle/reference.json', 'oracle/labels.json'}
    allowed = required | {'input/checkpoint_receipt.json'}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for entry in archive:
            name = entry.name.rstrip('/')
            if (name in entries or name not in allowed | {'input', 'oracle'}
                    or not (entry.isfile() or entry.isdir())
                    or entry.isdir() != (name in ('input', 'oracle'))):
                raise ValueError('unsafe reference archive entry')
            entries[name] = entry
        if not required <= entries.keys():
            raise ValueError('incomplete reference archive')
        def read(name):
            if entries[name].size > 2_000_000:
                raise ValueError('reference metadata too large')
            return json.loads(archive.extractfile(entries[name]).read(), object_pairs_hook=unique_object)
        spec, meta = read('input/suite.json'), read('oracle/reference.json')
        expected, _ = make_suite(corpus, stage, model)
        pinned, identity = suite_model(spec)
        if (identity['model_key'] != model or spec.get('model') != pinned['repo_id']
                or spec.get('revision') != pinned['revision']):
            raise ValueError('reference model identity missing or different')
        comparable = lambda doc: {k: v for k, v in doc.items() if k != 'cases'}
        if comparable(spec) != comparable(expected):
            raise ValueError('reference corpus or fixed suite differs')
        if spec.get('cases', []) != expected['cases']:
            raise ValueError('reference cases differ from the frozen stage')
        if [case.get('name') for case in meta.get('cases', [])] != [case['name'] for case in spec['cases']]:
            raise ValueError('partial reference case results')
        policy = meta.get('precision', {})
        effective = policy.get('effective', {})
        if (policy.get('requested') != 'fp32' or meta.get('parameter_dtypes') not in
                (['float32'], ['torch.float32'], ['fp32'])
                or any(effective.get(k) != 'fp32' for k in ('mode', 'weights', 'activations'))):
            raise ValueError('import requires an explicit FP32 reference')
        portforge_eval.validate_config(read('input/config.json'), pinned)
        if 'input/checkpoint_receipt.json' not in entries:
            raise ValueError('missing checkpoint identity receipt')
        portforge_eval.validate_receipt(read('input/checkpoint_receipt.json'), identity, pinned, spec)
        portforge_eval.validate_receipt(meta.get('checkpoint_receipt'), identity, pinned, spec)
        validate_artifact(meta, spec, identity)
    return data


def import_references(run, source, config, model):
    """Explicit verified data reuse; never import candidate code or acceptance."""
    source, run = Path(source).resolve(), Path(run)
    manifest = json.loads((source / 'manifest.json').read_text())
    old_hash = digest({p.name: p.read_text() for p in (source / 'harness').glob('*.py')})
    if manifest.get('harness_sha256') not in (old_hash, harness_identity(source / 'harness')):
        raise ValueError('source harness identity missing or changed')
    corpus = json.loads((run / 'corpus.json').read_text())
    if manifest.get('corpus_sha256') != digest(corpus):
        raise ValueError('source corpus identity missing or changed')
    key, validated = digest(config['reference'])[:16], []
    for stage in ('smoke', 'bringup'):
        origin = source / 'references' / key / model / stage
        ready, archive = origin / 'ready.json', origin / 'reference.tar'
        if (not ready.is_file() or not archive.is_file()
                or json.loads(ready.read_text()).get('endpoint_sha256') != digest(config['reference'])):
            raise ValueError('source reference incomplete or endpoint identity differs')
        data = validate_reference_archive(archive, corpus, stage, model)
        validated.append((stage, archive, data))
    for stage, origin, data in validated:
        cache = run / 'references' / key / model / stage
        gpu = config['reference']['work_dir'] + '/' + run.name + '/references/' + key + '/' + model + '/' + stage
        tt = config['tt']['work_dir'] + '/' + run.name + '/references/' + model + '/' + stage
        cluster.upload(config['reference'], gpu, {name: (run / 'harness' / name).read_bytes()
                                         for name in EVALUATOR_FILES})
        for endpoint, directory in ((config['reference'], gpu), (config['tt'], tt)):
            cluster.ssh(endpoint, 'mkdir -p ' + shlex.quote(directory) + ' && tar -xf - -C ' + shlex.quote(directory), data, 300)
        cache.mkdir(parents=True, exist_ok=True)
        (cache / 'reference.tar').write_bytes(data)
        write_json(cache / 'import.json', {'source_archive': str(origin), 'source_run': str(source),
            'source_harness_sha256': manifest['harness_sha256'], 'archive_sha256': hashlib.sha256(data).hexdigest(),
            'scope': 'FP32 reference data only; no candidate passes inherited'})
        write_json(cache / (digest(config['tt']) + '.deployed.json'), {'tt_dir': tt})
        write_json(cache / 'ready.json', {'reference_dir': gpu, 'tt_dir': tt, 'model': model,
            'stage': stage, 'endpoint_sha256': digest(config['reference'])})



def cached_reference(run, stage, config, model):
    model_spec(model)
    endpoint, run = config['reference'], Path(run)
    key = digest(endpoint)[:16]
    cache = run / 'references' / key / model / stage
    ready, deployed = cache / 'ready.json', cache / (digest(config['tt']) + '.deployed.json')
    if not all(p.is_file() for p in (ready, deployed, cache / 'reference.tar')):
        return None
    expected = {'reference_dir': endpoint['work_dir'] + '/' + run.name + '/references/' + key + '/' + model + '/' + stage,
                'tt_dir': config['tt']['work_dir'] + '/' + run.name + '/references/' + model + '/' + stage,
                'endpoint_sha256': digest(endpoint), 'model': model, 'stage': stage}
    record = json.loads(ready.read_text())
    if (any(record.get(k) != value for k, value in expected.items())
            or json.loads(deployed.read_text()).get('tt_dir') != expected['tt_dir']):
        return None
    return record



def complete_references(run, model):
    """True when an earlier run holds reusable smoke and bringup FP32 references."""
    return any(all((folder / model / stage / name).is_file() for stage in ('smoke', 'bringup')
                   for name in ('ready.json', 'reference.tar'))
               for folder in (Path(run) / 'references').glob('*'))



@job(resources={'reference': 1})
def reference(run_dir, stage, config, model=None):
    model = model or PORT['key']
    model_spec(model)
    cached = cached_reference(run_dir, stage, config, model)
    if cached:
        return cached
    run, endpoint = Path(run_dir), config['reference']
    key = digest(endpoint)[:16]
    cache = run / 'references' / key / model / stage
    directory = endpoint['work_dir'] + '/' + run.name + '/references/' + key + '/' + model + '/' + stage
    q = shlex.quote
    if not (cache / 'reference.tar').exists():
        cluster.upload(endpoint, directory, {'corpus.json': (run / 'corpus.json').read_bytes(),
                                     **{name: (run / 'harness' / name).read_bytes() for name in EVALUATOR_FILES}})
        cluster.reference_job(endpoint, directory, shlex.join([endpoint['python'], 'evaluate.py', 'reference',
            '--stage', stage, '--model', model, '--corpus', 'corpus.json', '--out', '.']), True, cache / 'job.json')
        data = cluster.ssh(endpoint, 'tar -cf - -C ' + q(directory) + ' input oracle', timeout=300)
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cache / 'reference.tmp'
        temporary.write_bytes(data)
        temporary.replace(cache / 'reference.tar')
    target = config['tt']['work_dir'] + '/' + run.name + '/references/' + model + '/' + stage
    deployed = cache / (digest(config['tt']) + '.deployed.json')
    if not deployed.exists():
        cluster.ssh(config['tt'], 'mkdir -p ' + q(target) + ' && tar -xf - -C ' + q(target),
            (cache / 'reference.tar').read_bytes(), 300)
        write_json(deployed, {'tt_dir': target})
    result = {'reference_dir': directory, 'tt_dir': target, 'endpoint_sha256': digest(endpoint), 'model': model, 'stage': stage}
    write_json(cache / 'ready.json', result)
    return result



@job(resources={'tt': 1})
def experiment(run_dir, job, request, config):
    run, stage, endpoint = Path(run_dir), request.get('stage'), config['tt']
    try:
        model = request.get('model', PORT['key'])
        public_stage = request.get('public_input_stage', '')
        if public_stage and (request.get('kind') != 'run' or public_stage not in ('smoke', 'bringup')):
            raise ValueError('public_input_stage requires run and smoke/bringup')
        input_stage = stage or public_stage
        ref = (cached_reference(run_dir, input_stage, config, model)
               or get(reference.submit(run_dir, input_stage, config, model))) if input_stage else None
        transport_timeout = (full_job_timeout(endpoint) + 400
                             if request.get('kind') == 'evaluate' and stage == 'full' else 22000)
        result = cluster.remote(config, run_dir, {'job': job, **request}, timeout=transport_timeout)
        if stage and result['returncode'] == 0 and request.get('kind', 'evaluate') == 'evaluate':
            root = endpoint['work_dir'] + '/' + run.name
            output = root + '/jobs/' + job + '/output'
            if stage != 'full':
                command = shlex.join([endpoint['python'], root + '/harness/evaluate.py', 'assess',
                    '--input', ref['tt_dir'] + '/input', '--oracle', ref['tt_dir'] + '/oracle', '--output', output])
                cluster.ssh(endpoint, command + ' || test -f ' + shlex.quote(output + '/verdict.json'), timeout=300)
                result['verdict'] = json.loads(cluster.ssh(endpoint, 'cat ' + shlex.quote(output + '/verdict.json')))
            else:
                outputs = cluster.ssh(endpoint, 'tar -cf - -C ' + shlex.quote(root + '/jobs/' + job) + ' output', timeout=300)
                gpu, score = config['reference'], ref['reference_dir'] + '/scores/' + job
                cluster.ssh(gpu, 'mkdir -p ' + shlex.quote(score) + ' && tar -xf - -C ' + shlex.quote(score), outputs, 300)
                command = shlex.join([gpu['python'], '../../evaluate.py', 'assess', '--input', '../../input',
                    '--oracle', '../../oracle', '--output', 'output']) + ' || test -f output/verdict.json'
                cluster.reference_job(gpu, score, command, False, run / 'jobs' / job / 'score-job.json')
                result['verdict'] = json.loads(cluster.ssh(gpu, 'cat ' + shlex.quote(score + '/output/verdict.json')))
        result['cluster_sha256'] = digest(config)
        result['precision'] = request.get('precision')
        result['model'] = request.get('model', PORT['key'])
    except Exception as exc:
        result = {'returncode': -1, 'failure': 'infrastructure', 'error': str(exc)[-2000:]}
    write_json(run / 'jobs' / job / 'result.json', result)
    return result



@job(resources={'reference': 1})
def precision_study(run_dir, job, config, model=None):
    model = model or PORT['key']
    """Matched short CUDA comparisons; FP32 oracle remains immutable."""
    try:
        ref = reference(run_dir, 'bringup', config, model)
        endpoint, modes = config['reference'], {}
        for mode in ('fp32', 'bf16'):
            directory = ref['reference_dir'] + '/precision/' + mode
            command = shlex.join([endpoint['python'], '../../evaluate.py', 'reference-benchmark',
                '--input', '../../input', '--oracle', '../../oracle', '--output', 'output', '--precision', mode, '--model', model])
            cluster.reference_job(endpoint, directory, command + ' || test -f output/verdict.json', True,
                          Path(run_dir) / 'jobs' / job / (mode + '-job.json'))
            modes[mode] = {name: json.loads(cluster.ssh(endpoint, 'cat ' + shlex.quote(directory + '/output/' + name + '.json')))
                           for name in ('measurements', 'verdict')}
        result = {'returncode': 0, 'modes': modes, 'cluster_sha256': digest(config),
                  'model': model, 'scope': 'eight short cases; not full-corpus quality or TT performance'}
    except Exception as exc:
        result = {'returncode': -1, 'failure': 'infrastructure', 'error': str(exc)[-2000:]}
    write_json(Path(run_dir) / 'jobs' / job / 'result.json', result)
    return result
