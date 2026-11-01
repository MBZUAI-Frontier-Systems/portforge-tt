"""Receipts and remote jobs: references, reference archives, TT experiments."""
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

from run_fixture import RunFixture

support.configure_port()


class JobsTests(RunFixture):
    def test_complete_references_requires_both_stages(self):
        root = self.make_previous_run()
        stage = root / 'references/key/parakeet'
        for name in ('smoke/ready.json', 'smoke/reference.tar', 'bringup/ready.json'):
            (stage / name).parent.mkdir(parents=True, exist_ok=True)
            (stage / name).write_text('x')
        self.assertFalse(jobs.complete_references(root, 'parakeet'))
        (stage / 'bringup/reference.tar').write_text('x')
        self.assertTrue(jobs.complete_references(root, 'parakeet'))

    def test_snapshot_rejects_symlink_inputs(self):
        (self.tools.workspace / 'link.py').symlink_to(self.tools.workspace / 'backend.py')
        with self.assertRaises(ValueError):
            jobs.snapshot(self.tools.workspace)

    def test_profile_jobs_never_run_the_oracle_assessment(self):
        run = self.tools.run_dir
        endpoint = {'work_dir': '/srv/tt', 'python': '/env/python'}
        config = {'tt': endpoint, 'reference': {'host': '192.0.2.2'}}
        reference = {'tt_dir': '/srv/tt/ref', 'reference_dir': '/gpu/ref'}
        with patch.object(jobs, 'cached_reference', return_value=reference), \
             patch.object(cluster, 'remote', return_value={'returncode': 0, 'profile': {'ops': 3}}), \
             patch.object(cluster, 'ssh') as ssh:
            result = jobs.experiment(str(run), 'prof', {'kind': 'profile', 'stage': 'smoke', 'model': 'parakeet'}, config)
        ssh.assert_not_called()
        self.assertEqual(result['profile'], {'ops': 3})
        self.assertNotIn('verdict', result)

    def test_profile_model_reaches_cached_inputs_and_remote_runner_without_assessment(self):
        request = {'kind': 'run', 'command': 'python profile.py',
                   'public_input_stage': 'smoke', 'model': 'parakeet'}
        with patch.object(jobs, 'cached_reference', return_value={'tt_dir': '/prepared/parakeet/smoke'}) as cached, \
             patch.object(jobs.reference, 'submit') as gpu, \
             patch.object(cluster, 'remote', return_value={'returncode': 0}) as tt, patch.object(cluster, 'ssh') as ssh:
            result = jobs.experiment._original(str(self.tools.run_dir), 'profile-large', request, self.config)
        cached.assert_called_once_with(str(self.tools.run_dir), 'smoke', self.config, 'parakeet')
        self.assertEqual(tt.call_args.args[2]['model'], 'parakeet')
        self.assertEqual(tt.call_args.args[2]['public_input_stage'], 'smoke')
        self.assertEqual(result['model'], 'parakeet')
        self.assertNotIn('verdict', result)
        gpu.assert_not_called()
        ssh.assert_not_called()

    def test_baselines_are_separate_per_model_precision_and_actual_suite(self):
        for model, precision in (('parakeet', 'bf16'), ('parakeet', 'fp32'), ('parakeet', 'bfp8_b')):
            actual_suite = digest({'model': model, 'precision': precision})
            self.archive('bringup', {'verdict': {'passed': True, 'suite_sha256': actual_suite}},
                         precision=precision, model=model)
            self.tools.status()
            baseline = self.tools.run_dir / 'baselines' / model / precision / digest(self.config) / (actual_suite + '.json')
            saved = json.loads(baseline.read_text())
            self.assertEqual(saved['model'], model)
            self.assertEqual(saved['precision'], precision)
            self.assertEqual(saved['code_sha256'], jobs.code_hash(jobs.snapshot(self.tools.workspace)))
        self.assertEqual(len(list((self.tools.run_dir / 'baselines').rglob('*.json'))), 3)

    def test_reference_paths_and_cli_are_model_specific(self):
        run = self.tools.run_dir
        write_json(run / 'corpus.json', {})
        config = {'tt': {'work_dir': '/tt'}, 'reference': {'work_dir': '/gpu', 'python': '/env/python'}}
        for stage in ('smoke', 'bringup'):
            with patch.object(cluster, 'upload'), patch.object(cluster, 'reference_job') as job, \
                 patch.object(cluster, 'ssh', return_value=b'archive'):
                result = jobs.reference._original(str(run), stage, config, 'parakeet')
            self.assertIn('/parakeet/' + stage, result['reference_dir'])
            self.assertTrue(result['tt_dir'].endswith('/parakeet/' + stage))
            self.assertIn('--model parakeet', job.call_args.args[2])

    def test_ready_deployed_reference_skips_gpu_resource_queue(self):
        run = self.tools.run_dir
        config = {'tt': {'work_dir': '/tt'}, 'reference': {'work_dir': '/gpu', 'python': '/env/python'}}
        key = digest(config['reference'])[:16]
        cache = run / 'references' / key / 'parakeet/smoke'
        expected = {'reference_dir': '/gpu/' + run.name + '/references/' + key + '/parakeet/smoke',
                    'tt_dir': '/tt/' + run.name + '/references/parakeet/smoke', 'model': 'parakeet', 'stage': 'smoke',
                    'endpoint_sha256': digest(config['reference'])}
        write_json(cache / 'ready.json', expected)
        write_json(cache / (digest(config['tt']) + '.deployed.json'), {'tt_dir': expected['tt_dir']})
        (cache / 'reference.tar').write_bytes(b'archive')
        request = {'kind': 'evaluate', 'stage': 'smoke', 'model': 'parakeet'}
        with patch.object(jobs.reference, 'submit') as reference, \
             patch.object(cluster, 'remote', return_value={'returncode': 1}):
            jobs.experiment._original(str(run), 'test', request, config)
            reference.assert_not_called()
        for altered in ({**expected, 'model': 'other-model'}, {**expected, 'stage': 'full'},
                        {**expected, 'endpoint_sha256': 'wrong'}):
            write_json(cache / 'ready.json', altered)
            self.assertIsNone(jobs.cached_reference(run, 'smoke', config, 'parakeet'))
        write_json(cache / 'ready.json', expected)
        (cache / 'reference.tar').unlink()
        with patch.object(jobs.reference, 'submit', return_value=support.future('future')) as reference, \
             patch.object(jobs, 'get', return_value=expected), \
             patch.object(cluster, 'remote', return_value={'returncode': 1}):
            jobs.experiment._original(str(run), 'miss', request, config)
            reference.assert_called_once_with(str(run), 'smoke', config, 'parakeet')

    def test_reference_archive_rejects_wrong_identity_precision_cases_and_unsafe_entries(self):
        import evaluate
        corpus, entries = self.reference_fixture()
        path = self.tools.run_dir / 'reference.tar'
        good = self.reference_tar(path, entries).read_bytes()
        with patch.object(evaluate, 'CORPUS_SHA256', digest(corpus)):
            self.assertEqual(jobs.validate_reference_archive(path, corpus, 'smoke', 'parakeet'), good)
            with self.assertRaises(ValueError):
                jobs.validate_reference_archive(path, corpus, 'smoke', 'other-model')
            # evaluate.py records the dtype name without the torch prefix.
            plain = dict(entries)
            plain['oracle/reference.json'] = json.dumps(dict(json.loads(entries['oracle/reference.json']),
                                                             parameter_dtypes=['float32']))
            self.assertIsInstance(jobs.validate_reference_archive(self.reference_tar(path, plain), corpus,
                                                                  'smoke', 'parakeet'), bytes)
            for name, field, value in [('input/suite.json', 'model', 'wrong'),
                                        ('oracle/reference.json', 'parameter_dtypes', ['bfloat16']),
                                        ('input/suite.json', 'revision', None),
                                        ('input/suite.json', 'weight_sha256', None),
                                        ('input/suite.json', 'corpus_sha256', 'wrong'),
                                        ('input/suite.json', 'gates', {}),
                                        ('input/suite.json', 'cases', []),
                                        ('oracle/reference.json', 'precision', {}),
                                        ('oracle/reference.json', 'cases', [])]:
                changed = dict(entries)
                metadata = json.loads(changed[name])
                metadata[field] = value
                changed[name] = json.dumps(metadata)
                self.reference_tar(path, changed)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    jobs.validate_reference_archive(path, corpus, 'smoke', 'parakeet')
            for name, kind in (('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                               ('input/checkpoint_receipt.json', tarfile.SYMTYPE), ('input/checkpoint_receipt.json', tarfile.LNKTYPE)):
                self.reference_tar(path, entries, unsafe=(name, kind))
                with self.assertRaises(ValueError):
                    jobs.validate_reference_archive(path, corpus, 'smoke', 'parakeet')
            partial = dict(entries)
            del partial['oracle/oracle.npz']
            self.reference_tar(path, partial)
            with self.assertRaises(ValueError):
                jobs.validate_reference_archive(path, corpus, 'smoke', 'parakeet')

    def test_imported_reference_bytes_and_provenance_are_preserved_without_passes(self):
        import evaluate
        run, source = self.tools.run_dir, self.tools.run_dir / 'previous'
        corpus, _ = self.reference_fixture()
        config = {'tt': {'work_dir': '/tt'}, 'reference': {'work_dir': '/gpu'}}
        key = digest(config['reference'])[:16]
        (source / 'harness').mkdir(parents=True)
        (source / 'harness/controller.py').write_text('# previous verified harness')
        old_hash = digest({'controller.py': '# previous verified harness'})
        write_json(source / 'manifest.json', {'harness_sha256': old_hash, 'corpus_sha256': digest(corpus)})
        write_json(run / 'corpus.json', corpus)
        originals = {}
        for stage in ('smoke', 'bringup'):
            _, entries = self.reference_fixture(stage)
            folder = source / 'references' / key / 'parakeet' / stage
            originals[stage] = self.reference_tar(folder / 'reference.tar', entries).read_bytes()
            write_json(folder / 'ready.json', {'endpoint_sha256': digest(config['reference'])})
        with patch.object(evaluate, 'CORPUS_SHA256', digest(corpus)), \
             patch.object(cluster, 'ssh') as ssh, patch.object(cluster, 'upload'):
            jobs.import_references(run, source, config, 'parakeet')
            self.assertEqual(ssh.call_count, 4)
            for stage in originals:
                folder = run / 'references' / key / 'parakeet' / stage
                self.assertEqual((folder / 'reference.tar').read_bytes(), originals[stage])
                self.assertEqual(json.loads((folder / 'import.json').read_text())['source_harness_sha256'], old_hash)
                self.assertIsNotNone(jobs.cached_reference(run, stage, config, 'parakeet'))
            self.assertFalse((run / 'baseline.json').exists())
            self.assertFalse((run / 'submission.json').exists())
            # A partially completed source fails before either remote is touched.
            (source / 'references' / key / 'parakeet' / 'bringup/ready.json').unlink()
            ssh.reset_mock()
            with self.assertRaises(ValueError):
                jobs.import_references(run, source, config, 'parakeet')
            ssh.assert_not_called()
            write_json(source / 'references' / key / 'parakeet' / 'bringup/ready.json', {'endpoint_sha256': 'wrong'})
            with self.assertRaises(ValueError):
                jobs.import_references(run, source, config, 'parakeet')
            ssh.assert_not_called()

    def test_profile_run_resolves_cached_reference_without_assessment_or_gpu_queue(self):
        request = {'kind': 'run', 'command': 'python profile.py', 'public_input_stage': 'smoke'}
        with patch.object(jobs, 'cached_reference', return_value={'tt_dir': '/prepared'}) as cached, \
             patch.object(jobs.reference, 'submit') as gpu, \
             patch.object(cluster, 'remote', return_value={'returncode': 0}) as tt, patch.object(cluster, 'ssh') as ssh:
            result = jobs.experiment._original(str(self.tools.run_dir), 'profile', request, self.config)
        cached.assert_called_once_with(str(self.tools.run_dir), 'smoke', self.config, 'parakeet')
        gpu.assert_not_called()
        ssh.assert_not_called()
        self.assertNotIn('verdict', result)
        self.assertEqual(tt.call_args.args[2]['public_input_stage'], 'smoke')
        with patch.object(jobs, 'cached_reference', return_value=None), \
             patch.object(jobs.reference, 'submit', return_value=support.future('future')) as gpu, \
             patch.object(jobs, 'get', return_value={'tt_dir': '/prepared'}), \
             patch.object(cluster, 'remote', return_value={'returncode': 0}):
            jobs.experiment._original(str(self.tools.run_dir), 'uncached', request, self.config)
        gpu.assert_called_once_with(str(self.tools.run_dir), 'smoke', self.config, 'parakeet')

    def test_full_transport_outlives_configured_job_without_extending_other_calls(self):
        config = json.loads(json.dumps(self.config))
        config['tt']['full_timeout_seconds'] = 64800
        for kind, stage, expected in (('evaluate', 'full', 65200), ('evaluate', 'smoke', 22000),
                                      ('run', None, 22000)):
            request = {'kind': kind, 'stage': stage, 'full_timeout_seconds': 999999}
            with patch.object(jobs, 'cached_reference', return_value={'tt_dir': '/prepared'}), \
                 patch.object(cluster, 'remote', return_value={'returncode': 1}) as remote:
                jobs.experiment._original(str(self.tools.run_dir), kind + str(stage), request, config)
            self.assertEqual(remote.call_args.kwargs['timeout'], expected)
        config['tt']['full_timeout_seconds'] = 86401
        with patch.object(jobs, 'cached_reference', return_value={'tt_dir': '/prepared'}), \
             patch.object(cluster, 'remote') as remote:
            result = jobs.experiment._original(str(self.tools.run_dir), 'invalid-deadline',
                                                   {'kind': 'evaluate', 'stage': 'full'}, config)
        remote.assert_not_called()
        self.assertEqual(result['failure'], 'infrastructure')


if __name__ == "__main__":
    unittest.main()
