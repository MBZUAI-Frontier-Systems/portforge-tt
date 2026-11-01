"""Shared fixture for the controller-side tests: a run with a frozen harness and a workspace."""
import json
import io
import tarfile
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import support  # noqa: F401  (pack and registry setup)
import campaign, cluster, controller, jobs, objective, port, runtime, tools
from cluster import identity as digest
from portforge_models import model_spec
from runner import write_json

support.configure_port()


class RunFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.tools = object.__new__(tools.PortTools)
        self.tools.workspace, self.tools.run_dir, self.tools.refs = root / 'work', root / 'run', {}
        self.tools.workspace.mkdir()
        self.tools.run_dir.mkdir()
        support.write_harness(self.tools.run_dir / 'harness')
        self.config = {'tt': {'host': '192.0.2.1'}, 'reference': {'host': '192.0.2.2'}}
        write_json(self.tools.run_dir / 'cluster.json', self.config)
        (self.tools.workspace / 'backend.py').write_text('# fresh implementation')

    def tearDown(self):
        self.temp.cleanup()

    def timeout_fixture(self):
        health = {'cards': [{'pci': '0000:c1:00.0', 'state': 'quarantined', 'job': 'timed-out'}]}
        job = self.tools.run_dir / 'jobs/timed-out'
        request = {'kind': 'run', 'cluster_sha256': digest(self.config)}
        result = {'returncode': 124, 'failure': 'timeout', 'timeout': True,
                  'pci': '0000:c1:00.0', 'host_memory_limit': False,
                  'job': 'timed-out', 'cluster_sha256': digest(self.config)}
        write_json(job / 'request.json', request)
        write_json(job / 'result.json', result)
        return health, job, request, result

    def make_previous_run(self, name='previous', **manifest):
        root = self.tools.run_dir.parent / name
        source = root / 'workspace'
        (source / 'baselines').mkdir(parents=True)
        (source / '.opencode').mkdir()
        write_json(root / 'manifest.json', {'seed_policy': 'fresh-public-parakeet', 'model': 'parakeet',
                                            'workspace': str(source), **manifest})
        (source / 'STATE.md').write_text('# STATE\nnext: read probe v7 e244a077\n')
        (source / 'RESEARCH.md').write_text('research notes')
        (source / 'TT_NOTES.md').write_text('control notes are kept on resume')
        (source / 'backend.py').write_text('# carried implementation')
        (source / 'baselines/smoke.json').write_text('{"digest": "abc"}')
        (source / '.opencode/state.json').write_text('{}')
        (source / 'TASK.md').write_text('superseded instructions')
        (self.tools.workspace / 'TASK.md').write_text('current instructions')
        return root

    def archive(self, stage, result=None, precision='bf16', config=None, model='parakeet'):
        job = self.tools.run_dir / 'jobs/one'
        write_json(job / 'request.json', {'stage': stage, 'code_sha256': jobs.code_hash(jobs.snapshot(self.tools.workspace)),
                                         'precision': precision, 'cluster_sha256': digest(config or self.config), 'model': model,
                                         'suite_definition_sha256': jobs.suite_identity(self.tools.run_dir, model, stage)})
        if result is not None:
            write_json(job / 'result.json', result)

    def delivery_receipt(self, job='delivery', model='parakeet', precision='bf16', stage='bringup'):
        request = {'kind': 'evaluate', 'model': model, 'precision': precision, 'stage': stage,
                   'code_sha256': jobs.code_hash(jobs.snapshot(self.tools.workspace)),
                   'cluster_sha256': digest(self.config),
                   'suite_definition_sha256': jobs.suite_identity(self.tools.run_dir, model, stage)}
        result = {'job': job, 'returncode': 0, 'cluster_sha256': digest(self.config),
                  'model': model, 'precision': precision,
                  'verdict': {'passed': True, 'stage': stage, 'model_key': model,
                              'precision': {'requested': precision}, 'suite_sha256': 'prepared-suite'}}
        folder = self.tools.run_dir / 'jobs' / job
        write_json(folder / 'request.json', request)
        write_json(folder / 'result.json', result)
        return folder, request, result

    def reference_fixture(self, stage='smoke'):
        import evaluate
        pinned = model_spec('parakeet')
        names = [f"libri-{i:04d}" for i in range(15)]
        corpus = {'schema_version': 1, 'task': 'fixture', 'protocol': 'fixture', 'sources': [],
                  'selection': {'smoke': names[0],
                                'bringup': dict(batch0=names[1], batch1=names[4], long=names[-1],
                                                short=names[0], silence='syn-silence'),
                                'full': names[2:] + names[0:2]},
                  'cases': [dict(name=n, transcript=f"T{i}", audio_b64='AAAA', seconds=2.0, source='fixture')
                            for i, n in enumerate(names)]
                           + [dict(name='syn-silence', transcript='', audio_b64='AAAA', seconds=2.0, source='fixture'),
                              dict(name='syn-tone', transcript='', audio_b64='AAAA', seconds=3.0, source='fixture')]}
        with patch.object(evaluate, 'CORPUS_SHA256', digest(corpus)):
            spec, _ = evaluate.make_suite(corpus, stage)
        identity = {k: spec[k] for k in ('model_key', 'model', 'revision', 'model_manifest_sha256')}
        receipt = {'model': 'parakeet', 'repo_id': pinned['repo_id'], 'revision': pinned['revision'],
                   'checkpoint_format': pinned['checkpoint_format'], 'verified_files': pinned['files'],
                   'model_manifest_sha256': spec['model_manifest_sha256'],
                   'suite_sha256': evaluate.suite_hash(spec)}
        meta = {'parameter_dtypes': ['torch.float32'], **identity,
                'precision': {'requested': 'fp32', 'effective': {'mode': 'fp32', 'weights': 'fp32', 'activations': 'fp32'}},
                'cases': [{'name': case['name']} for case in spec['cases']],
                'checkpoint_receipt': receipt, 'suite_sha256': evaluate.suite_hash(spec),
                'timing_protocol': evaluate.TIMING_PROTOCOL}
        entries = {'input/suite.json': json.dumps(spec), 'input/config.json': json.dumps(pinned['architecture']),
                   'input/checkpoint_receipt.json': json.dumps(receipt),
                   'input/inputs.npz': 'fixture-input', 'oracle/oracle.npz': 'fixture-oracle',
                   'oracle/reference.json': json.dumps(meta), 'oracle/labels.json': '{}'}
        return corpus, entries

    def reference_tar(self, path, entries, unsafe=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, 'w') as archive:
            for name, content in entries.items():
                payload = content.encode()
                entry = tarfile.TarInfo(name)
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))
            if unsafe:
                entry = tarfile.TarInfo(unsafe[0])
                entry.type = unsafe[1]
                entry.linkname = '/outside'
                archive.addfile(entry)
        return path
