"""The agents' MCP tools: status, logs, runs, evaluations and submission rules."""
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


class ToolsTests(RunFixture):
    def test_status_exact_old_job_keeps_complete_gate_and_is_read_only(self):
        job = self.tools.run_dir / 'jobs/old-gate'
        request = {'kind': 'evaluate', 'stage': 'bringup', 'model': 'parakeet',
                   'precision': 'bfp8_b', 'code_sha256': 'source', 'cluster_sha256': 'cluster',
                   'suite_definition_sha256': 'definition', 'timeout': 300}
        result = {'returncode': 0, 'stdout': 'x' * 2500, 'stderr': 'y' * 2000,
                  'verdict': {'passed': True, 'accepted': False, 'suite_sha256': 'suite',
                              'checks': [{'name': 'long', 'values': list(range(1200))}]}}
        write_json(job / 'request.json', request)
        write_json(job / 'result.json', result)
        os.utime(job / 'request.json', (1, 1))
        saved = {p.name: p.read_bytes() for p in job.iterdir()}
        for index in range(14):
            write_json(self.tools.run_dir / f'jobs/new-{index}/request.json', {'kind': 'run'})
            write_json(self.tools.run_dir / f'jobs/new-{index}/result.json', {'returncode': 0})
        default = self.tools.status()
        self.assertEqual(len(default), 12)
        self.assertNotIn('old-gate', default)
        with patch.object(cluster, 'remote') as remote, patch.object(tools, 'reference_progress') as progress, \
             patch.object(tools, 'write_json') as write:
            exact = self.tools.status(job_id='old-gate')
            compact = self.tools.status(job_id='old-gate', compact=True)
            remote.assert_not_called(); progress.assert_not_called(); write.assert_not_called()
        self.assertEqual(exact, compact)
        self.assertEqual(set(exact), {'old-gate'})
        self.assertEqual(exact['old-gate']['request_identity'], request)
        self.assertEqual(exact['old-gate']['verdict'], result['verdict'])
        self.assertEqual(exact['old-gate']['stdout'], result['stdout'][-1000:])
        self.assertEqual(exact['old-gate']['stderr'], result['stderr'][-1000:])
        self.assertEqual(saved, {p.name: p.read_bytes() for p in job.iterdir()})
        self.assertFalse((self.tools.run_dir / 'baselines').exists())

    def test_status_omits_raw_timing_samples(self):
        job = self.tools.run_dir / 'jobs/perf'
        check = {'name': 'sweep-b4', 'p50_seconds': .002, 'p99_seconds': .003, 'seconds': [.002] * 5000}
        result = {'returncode': 0, 'verdict': {'passed': True, 'latency': {'max_case_p99_seconds': .003},
                                               'checks': [check, {'name': 'gate', 'seconds': 1.5}]}}
        write_json(job / 'request.json', {'kind': 'evaluate', 'stage': 'perf'})
        write_json(job / 'result.json', result)
        for shown in (self.tools.status()['perf'], self.tools.status(job_id='perf')['perf']):
            checks = shown['verdict']['checks']
            self.assertEqual(checks[0], {'name': 'sweep-b4', 'p50_seconds': .002, 'p99_seconds': .003,
                                         'seconds_omitted': 5000})
            self.assertEqual(checks[1], {'name': 'gate', 'seconds': 1.5})
            self.assertEqual(shown['verdict']['latency'], result['verdict']['latency'])
        self.assertEqual(json.loads((job / 'result.json').read_text()), result)

    def test_status_exact_pending_is_local_and_preserves_request(self):
        request = {'kind': 'reference-study', 'model': 'parakeet', 'cluster_sha256': 'missing'}
        job = self.tools.run_dir / 'jobs/pending'
        write_json(job / 'request.json', request)
        with patch.object(cluster, 'remote') as remote, patch.object(tools, 'reference_progress') as progress:
            result = self.tools.status(job_id='pending')['pending']
            remote.assert_not_called(); progress.assert_not_called()
        self.assertEqual(result['status'], 'pending')
        self.assertEqual(result['request_identity'], request)
        self.assertEqual(result['details']['job_id'], 'pending')
        self.assertFalse((job / 'result.json').exists())

    def test_status_exact_historical_dotted_reference_id(self):
        job_id = 'gpu-precision.parakeet-20260921T1020'
        request = {'kind': 'reference-study', 'model': 'parakeet'}
        result = {'returncode': 0, 'modes': {'bf16': {
            'verdict': {'passed': True, 'checks': [{'error': .001}]},
            'measurements': {'cases': [{'seconds': [1., 2.]}]}}}}
        job = self.tools.run_dir / 'jobs' / job_id
        write_json(job / 'request.json', request)
        write_json(job / 'result.json', result)
        with patch.object(cluster, 'remote') as remote, patch.object(tools, 'reference_progress') as progress:
            actual = self.tools.status(job_id=job_id)[job_id]
            remote.assert_not_called(); progress.assert_not_called()
        self.assertEqual(actual['modes'], result['modes'])
        self.assertEqual(actual['request_identity'], request)
        self.assertEqual(actual['details']['job_id'], job_id)
        # The TT log validator deliberately retains its separate contract.
        with self.assertRaises(ValueError):
            tools.validate_log_request(job_id, 'stdout', 0, 1)

    def test_status_exact_rejects_invalid_missing_and_symlink_paths(self):
        for job_id in (None, 1, False, [], '.', '..', '../other', '/tmp', 'x/y',
                       'x\\y', 'a' * 81, 'missing', 'trailing\n', '.hidden', '-leading', '_leading'):
            with self.subTest(job_id=job_id), self.assertRaises(ValueError):
                self.tools.status(job_id=job_id)
        jobs = self.tools.run_dir / 'jobs'
        job = jobs / 'own'
        write_json(job / 'request.json', {'kind': 'run'})
        write_json(job / 'result.json', {'returncode': 0})
        for path in (job / 'request.json', job / 'result.json', job, jobs):
            moved = path.with_name(path.name + '-saved')
            path.rename(moved)
            path.symlink_to(moved, target_is_directory=moved.is_dir())
            try:
                with self.subTest(path=str(path)), self.assertRaises(ValueError):
                    self.tools.status(job_id='own')
            finally:
                path.unlink(); moved.rename(path)
        (job / 'result.json').unlink()
        (job / 'result.json').symlink_to(job / 'absent')
        with self.assertRaises(ValueError):
            self.tools.status(job_id='own')

    def test_compact_status_preserves_failure_quality_identity_and_legacy_response(self):
        job = self.tools.run_dir / 'jobs/failure'
        request = {'kind': 'evaluate', 'stage': 'smoke', 'model': 'parakeet', 'precision': 'bf16',
                   'code_sha256': 'source', 'cluster_sha256': 'cluster', 'suite_definition_sha256': 'definition'}
        result = {'returncode': 125, 'failure': 'host_memory_limit', 'timeout': False,
                  'host_memory_limit': True, 'peak_host_rss_bytes': 12345,
                  'host_memory_accounting': 'sampled_sum_rss_shared_pages_counted_per_process',
                  'model': 'parakeet', 'precision': 'bf16', 'pci': '0000:41:00.0',
                  'stdout': 'old ' * 500 + 'last output', 'stderr': 'trace ' * 600 + 'memory failure',
                  'verdict': {'passed': False, 'accepted': False, 'production_certified': False,
                              'suite_sha256': 'actualsuite', 'model_manifest_sha256': 'modelidentity',
                              'precision': {'requested': 'bf16'}, 'failures': ['numerical'],
                              'checks': [{'name': 'one', 'seconds': [1, 2], 'rows': 1}],
                              'quality': {'threshold': 0.1, 'observed': 0.2}}}
        write_json(job / 'request.json', request)
        write_json(job / 'result.json', result)
        saved = (job / 'result.json').read_bytes()
        shown = jobs.without_raw_samples(result)
        self.assertEqual(shown['verdict']['checks'], [{'name': 'one', 'rows': 1, 'seconds_omitted': 2}])
        self.assertEqual(self.tools.status(), {'failure': shown})
        compact = self.tools.status(compact=True)['failure']
        self.assertEqual(compact['request_identity'], request)
        for key, value in shown.items():
            if key not in ('stdout', 'stderr'):
                self.assertEqual(compact[key], value)
        for stream in ('stdout', 'stderr'):
            self.assertEqual(compact[stream], result[stream][-1000:])
            self.assertEqual(compact['log_tails'][stream], {
                'available_chars': len(result[stream]), 'returned_chars': 1000, 'truncated': True})
        self.assertEqual(compact['details']['tool'], 'port_log')
        self.assertEqual(compact['details']['job_id'], 'failure')
        self.assertEqual((job / 'result.json').read_bytes(), saved)
        self.assertEqual(self.tools.status(), {'failure': shown})
        for invalid in ('false', 1, None):
            with self.subTest(compact=invalid), self.assertRaises(ValueError):
                self.tools.status(compact=invalid)

    def test_compact_status_preserves_baseline_registration_without_inventing_gates(self):
        result = {'returncode': 0, 'stdout': 'short', 'stderr': '',
                  'verdict': {'passed': True, 'accepted': False, 'suite_sha256': 'actualsuite'}}
        self.archive('bringup', result)
        compact = self.tools.status(compact=True)['one']
        self.assertEqual(compact['verdict'], result['verdict'])
        self.assertFalse(compact['log_tails']['stdout']['truncated'])
        self.assertEqual(compact['log_tails']['stderr']['returned_chars'], 0)
        baselines = list((self.tools.run_dir / 'baselines').rglob('*.json'))
        self.assertEqual(len(baselines), 1)
        self.assertEqual(json.loads(baselines[0].read_text())['job'], 'one')
        write_json(self.tools.run_dir / 'jobs/shell/request.json', {'kind': 'run'})
        write_json(self.tools.run_dir / 'jobs/shell/result.json', {'returncode': 0})
        shell = self.tools.status(compact=True)['shell']
        self.assertNotIn('verdict', shell)
        self.assertIsNone(shell['request_identity']['precision'])

    def test_compact_status_keeps_old_pending_and_reference_study_details(self):
        run = self.tools.run_dir
        key = digest(self.config)
        write_json(run / 'clusters' / (key + '.json'), self.config)
        for job, kind in (('pending-reference', 'reference-study'), ('pending-tt', 'run')):
            path = run / 'jobs' / job / 'request.json'
            write_json(path, {'kind': kind, 'cluster_sha256': key, 'code_sha256': job + '-source'})
            os.utime(path, (1, 1))
        for index in range(14):
            write_json(run / f'jobs/done-{index}/request.json', {'kind': 'run'})
            write_json(run / f'jobs/done-{index}/result.json', {'returncode': 0})
        study = {'returncode': 0, 'modes': {'fp16': {'verdict': {'passed': False},
                  'measurements': {'cases': [{'seconds': [1.0]}]}}}}
        write_json(run / 'jobs/study/request.json', {'kind': 'reference-study', 'model': 'parakeet'})
        write_json(run / 'jobs/study/result.json', study)
        with patch.object(tools, 'reference_progress', return_value={'phase': 'queued', 'background_nonblocking': True}), \
             patch.object(cluster, 'remote', return_value={'phase': 'decode'}):
            legacy, compact = self.tools.status(), self.tools.status(compact=True)
        self.assertEqual(set(compact), set(legacy))
        self.assertEqual(len(compact), 14)
        for job in ('pending-reference', 'pending-tt'):
            self.assertEqual(compact[job]['status'], 'pending')
            self.assertEqual(compact[job]['progress'], legacy[job]['progress'])
            self.assertEqual(compact[job]['request_identity']['code_sha256'], job + '-source')
        self.assertEqual(compact['study']['modes'], study['modes'])
        for job in ('study', 'pending-reference'):
            self.assertEqual(compact[job]['details']['tool'], 'port_status')
            self.assertIs(compact[job]['details']['compact'], False)

    def test_log_registered_and_uses_job_archived_endpoint(self):
        names = []
        self.tools.mcp = SimpleNamespace(add_tool=lambda fn, name: names.append(name))
        self.tools.setup(self.tools.workspace, self.tools.run_dir)
        self.assertIn('port_log', names)
        archived = {'tt': {'host': '192.0.2.99'}, 'reference': {'host': '192.0.2.2'}}
        key = digest(archived)
        write_json(self.tools.run_dir / 'clusters' / (key + '.json'), archived)
        write_json(self.tools.run_dir / 'jobs/own/request.json', {'kind': 'run', 'cluster_sha256': key})
        with patch.object(cluster, 'remote', return_value={'text': 'earlier RESULT', 'eof': False}) as remote:
            result = self.tools.log('own', 'stderr', 12, 1024)
        self.assertEqual(result['text'], 'earlier RESULT')
        remote.assert_called_once_with(archived, str(self.tools.run_dir),
            {'kind': 'log', 'job': 'own', 'stream': 'stderr', 'offset_bytes': 12, 'limit_bytes': 1024}, timeout=15)

    def test_log_rejects_unknown_reference_and_bad_bounds_before_remote(self):
        write_json(self.tools.run_dir / 'jobs/reference/request.json', {'kind': 'reference-study'})
        with patch.object(cluster, 'remote') as remote:
            for job, kwargs in [('unknown', {}), ('reference', {}), ('../escape', {}),
                                ('unknown', {'stream': 'output'}), ('unknown', {'limit_bytes': 65537}),
                                ('unknown', {'offset_bytes': True})]:
                with self.subTest(job=job, kwargs=kwargs), self.assertRaises(ValueError):
                    self.tools.log(job, **kwargs)
            remote.assert_not_called()

    def test_log_rejects_symlink_job_and_tampered_cluster(self):
        run = self.tools.run_dir
        key = digest(self.config)
        write_json(run / 'jobs/own/request.json', {'kind': 'evaluate', 'cluster_sha256': key})
        write_json(run / 'clusters' / (key + '.json'), {'tt': {'host': '192.0.2.99'}})
        with patch.object(cluster, 'remote') as remote:
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                self.tools.log('own')
            (run / 'jobs/link').symlink_to(run / 'jobs/own', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'unsafe job'):
                self.tools.log('link')
            (run / 'jobs/own/request.json').unlink()
            (run / 'jobs/own/request.json').symlink_to(run / 'clusters' / (key + '.json'))
            with self.assertRaisesRegex(ValueError, 'unsafe job'):
                self.tools.log('own')
            remote.assert_not_called()

    def test_run_punctuation_cannot_create_nested_mcp_config_keys(self):
        from types import SimpleNamespace
        from agents.agent import IsolatedOpenCode
        name = tools.tool_name(self.tools.run_dir / 'model-parakeet with spaces')
        self.assertRegex(name, r'^portforge_[a-f0-9]{16}$')
        tool = SimpleNamespace(name=name, hostname='127.0.0.1', port=8002)
        agent = object.__new__(IsolatedOpenCode)
        agent.web, agent.repository, agent.system_message = True, Path('/repo'), ''
        agent.agent_name, agent.config = 'portforge-worker', None
        agent.additional_providers = []
        config = agent._build_config([tool])
        self.assertEqual(list(config['mcp']), [name])
        self.assertEqual(config['mcp'][name]['url'],
                         f'http://127.0.0.1:8002/{name}/mcp')

    def test_tool_identity_is_stable_and_distinguishes_runs(self):
        self.assertEqual(tools.tool_name('/tmp/run/../run'), tools.tool_name('/tmp/run'))
        self.assertNotEqual(tools.tool_name('/tmp/one/model'), tools.tool_name('/tmp/two/model'))

    def test_historical_full_verdicts_are_operator_only(self):
        root = self.make_previous_run()
        write_json(self.tools.run_dir / 'manifest.json', {'history_runs': [str(root)]})
        write_json(root / 'jobs/final/request.json', {'kind': 'evaluate', 'stage': 'full'})
        write_json(root / 'jobs/final/result.json', {'verdict': {'failures': ['held-out case']}})
        with self.assertRaisesRegex(ValueError, 'operator-only'):
            self.tools.status(job_id='final')
        with self.assertRaisesRegex(ValueError, 'operator-only'):
            self.tools.log('final')
        write_json(self.tools.run_dir / 'jobs/final/request.json', {'kind': 'evaluate', 'stage': 'full'})
        self.assertIn('final', self.tools.status(job_id='final'))  # this run's own receipt stays visible

    def test_history_jobs_are_readable_but_not_listed_reused_or_passing(self):
        root = self.make_previous_run()
        write_json(self.tools.run_dir / 'manifest.json', {'history_runs': [str(root)]})
        archived = {'tt': {'host': '192.0.2.77'}, 'reference': {'host': '192.0.2.2'}}
        key = digest(archived)
        write_json(root / 'clusters' / (key + '.json'), archived)
        write_json(root / 'jobs/e244a077/request.json', {'kind': 'run', 'cluster_sha256': key})
        write_json(root / 'jobs/e244a077/result.json', {'returncode': 0, 'stdout': 'V7 none'})
        self.assertEqual(self.tools.status(), {})
        exact = self.tools.status(job_id='e244a077')['e244a077']
        self.assertEqual((exact['historical_run'], exact['stdout']), (root.name, 'V7 none'))
        with patch.object(cluster, 'remote', return_value={'text': 'V7 none'}) as remote:
            self.assertEqual(self.tools.log('e244a077')['text'], 'V7 none')
        self.assertEqual(remote.call_args.args[:2], (archived, str(root)))
        write_json(self.tools.run_dir / 'jobs/current/request.json', {'kind': 'run'})
        self.assertNotIn('historical_run', self.tools.status(job_id='current')['current'])
        with self.assertRaisesRegex(ValueError, 'unknown or unsafe job'):
            self.tools.status(job_id='missing')

    def test_old_pending_reference_remains_visible_after_many_completed_jobs(self):
        run = self.tools.run_dir
        key = digest(self.config)
        write_json(run / 'clusters' / (key + '.json'), self.config)
        old = run / 'jobs/old-reference/request.json'
        write_json(old, {'kind': 'reference-study', 'stage': 'precision-study', 'cluster_sha256': key})
        os.utime(old, (1, 1))
        for index in range(14):
            write_json(run / f'jobs/done-{index}/request.json', {'kind': 'run'})
            write_json(run / f'jobs/done-{index}/result.json', {'returncode': 0})
        with patch.object(tools, 'reference_progress', return_value={'phase': 'queued'}) as progress:
            results = self.tools.status()
        self.assertEqual(results['old-reference']['progress']['phase'], 'queued')
        self.assertEqual(len(results), 13)
        progress.assert_called_once()
        facts = controller.supervisor_facts(self.tools.workspace, run, self.config)
        self.assertTrue(any(j['id'] == 'old-reference' and j['pending'] for j in facts['recent_jobs']))

    def test_submission_requires_passing_current_executable_source(self):
        self.delivery_receipt(job='one', model='parakeet')
        self.tools.write('STATE.md', 'Saved progress')
        self.assertEqual(self.tools.submit()['status'], 'frozen_for_full_evaluation')
        submitted = json.loads((self.tools.run_dir / 'submission.json').read_text())
        self.assertEqual(submitted['cluster_sha256'], digest(self.config))
        self.assertEqual(submitted['precision'], 'bf16')
        with self.assertRaises(ValueError):
            self.tools.write('backend.py', '# changed')
        (self.tools.run_dir / 'submission.json').unlink()
        self.tools.write('backend.py', '# changed')
        with self.assertRaises(ValueError):
            self.tools.submit()

    def test_submission_rejects_contradictory_missing_or_mismatched_receipts(self):
        folder, request, result = self.delivery_receipt(model='parakeet')
        changes = [{'returncode': 124}, {'returncode': False}, {'returncode': None},
                   {'failure': 'infrastructure'}, {'timeout': True}, {'host_memory_limit': True},
                   {'job': 'another-job'}, {'cluster_sha256': 'another-runtime'},
                   {'model': 'other-model'}, {'precision': 'fp32'}, {'verdict': None},
                   {'verdict': {'passed': True}}]
        changes += [{'verdict': {**result['verdict'], **change}} for change in
                    ({'passed': 'true'}, {'passed': False}, {'failures': ['numerical']},
                     {'stage': 'smoke'}, {'model_key': 'other-model'}, {'suite_sha256': ''},
                     {'suite_sha256': None}, {'precision': {'requested': 'bfp8_b'}})]
        changes += [{key: None} for key in ('job', 'cluster_sha256', 'model', 'precision')]
        for change in changes:
            with self.subTest(change=change):
                write_json(folder / 'result.json', {**result, **change})
                with self.assertRaisesRegex(ValueError, 'independent bringup pass'):
                    self.tools.submit()
                self.assertFalse((self.tools.run_dir / 'submission.json').exists())
        write_json(folder / 'result.json', result)
        for change in ({'kind': 'run'}, {'kind': 'reference-study'}, {'code_sha256': None},
                       {'code_sha256': 'other-source'}, {'cluster_sha256': 'old-runtime'},
                       {'suite_definition_sha256': 'old-suite'}):
            with self.subTest(request=change):
                write_json(folder / 'request.json', {**request, **change})
                with self.assertRaises(ValueError):
                    self.tools.submit()
                self.assertFalse((self.tools.run_dir / 'submission.json').exists())

    def test_submission_latest_matching_receipt_cannot_fall_back_to_older_pass(self):
        first, _, _ = self.delivery_receipt(job='earlier', model='parakeet')
        later, request, result = self.delivery_receipt(job='later', model='parakeet')
        os.utime(first / 'request.json', (1, 1))
        os.utime(later / 'request.json', (2, 2))
        for receipt in ({**result, 'timeout': True}, {'verdict': {'passed': True}}, None):
            if receipt is None:
                (later / 'result.json').unlink()
            else:
                write_json(later / 'result.json', receipt)
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                self.tools.submit()
            self.assertFalse((self.tools.run_dir / 'submission.json').exists())
        # A genuinely different historical source does not invalidate this pass.
        write_json(later / 'request.json', {**request, 'code_sha256': 'other-source'})
        self.assertEqual(self.tools.submit()['status'], 'frozen_for_full_evaluation')

    def test_submission_accepts_identified_passes_without_changing_precision_interface(self):
        for precision in port.PORT['precisions']:
            with self.subTest(precision=precision):
                self.delivery_receipt(model='parakeet', precision=precision)
                self.assertEqual(self.tools.submit(precision, 'parakeet')['status'],
                                 'frozen_for_full_evaluation')
                submission = self.tools.run_dir / 'submission.json'
                saved = json.loads(submission.read_text())
                self.assertEqual((saved['model'], saved['precision']), ('parakeet', precision))
                submission.unlink()

    def test_smoke_pass_cannot_authorize_full_submission(self):
        self.archive('smoke', {'verdict': {'passed': True}})
        with self.assertRaises(ValueError):
            self.tools.submit()
        with self.assertRaises(ValueError):
            self.tools.evaluate('full')

    def test_pending_or_passing_identical_stage_reuses_job(self):
        for result in (None, {'verdict': {'passed': True}}):
            self.archive('smoke', result)
            with patch.object(jobs.experiment, 'submit') as remote:
                self.assertEqual(self.tools.evaluate('smoke'), {'job': 'one', 'reused': True})
                remote.assert_not_called()

    def test_profile_is_never_answered_by_an_evaluate_receipt(self):
        self.archive('smoke', {'verdict': {'passed': True}})
        with patch.object(jobs.experiment, 'submit') as remote:
            started = self.tools.profile('smoke')
        self.assertNotIn('reused', started)
        remote.assert_called_once()

    def test_instruction_protection_handles_equivalent_paths(self):
        for path in ('CONTRACT.md', './CONTRACT.md', './TT_GUIDE.md', './TASK.md'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.tools.write(path, 'weaken contract')

    def test_submission_rejects_changed_precision_or_endpoint(self):
        self.archive('bringup', {'verdict': {'passed': True}}, precision='fp32')
        with self.assertRaises(ValueError):
            self.tools.submit('bf16')
        replacement = {**self.config, 'tt': {'host': '192.0.2.3'}}
        write_json(self.tools.run_dir / 'cluster.json', replacement)
        with self.assertRaises(ValueError):
            self.tools.submit('fp32')

    def test_changed_precision_or_cluster_gets_new_job_with_pinned_config(self):
        self.archive('smoke', {'verdict': {'passed': True}})
        with patch.object(jobs.experiment, 'submit', return_value=support.future('future')) as remote:
            result = self.tools.evaluate('smoke', 'fp32')
            self.assertNotEqual(result['job'], 'one')
            self.assertEqual(remote.call_args.args[-1], self.config)
            request = json.loads((self.tools.run_dir / 'jobs' / result['job'] / 'request.json').read_text())
            self.assertEqual(request['precision'], 'fp32')
            self.assertEqual(request['cluster_sha256'], digest(self.config))
        replacement = {**self.config, 'tt': {'host': '192.0.2.3'}}
        write_json(self.tools.run_dir / 'cluster.json', replacement)
        with patch.object(jobs.experiment, 'submit', return_value=support.future('future')) as remote:
            result = self.tools.evaluate('smoke')
            self.assertNotEqual(result['job'], 'one')
            self.assertEqual(remote.call_args.args[-1], replacement)

    def test_pending_status_uses_original_endpoint_after_config_change(self):
        self.archive('smoke')
        write_json(self.tools.run_dir / 'clusters' / (digest(self.config) + '.json'), self.config)
        write_json(self.tools.run_dir / 'cluster.json', {'tt': {'host': 'replacement'}})
        with patch.object(cluster, 'remote', return_value={'phase': 'generate'}) as remote:
            result = self.tools.status()
        self.assertEqual(remote.call_args.args[0], self.config)
        self.assertEqual(result['one']['progress']['phase'], 'generate')

    def test_other_precision_pass_never_deduplicates_or_authorizes_submission(self):
        self.archive('bringup', {'verdict': {'passed': True}}, precision='bf16')
        with self.assertRaises(ValueError):
            self.tools.submit(precision='bfp8_b')
        with patch.object(jobs.experiment, 'submit', return_value=support.future('future')) as remote:
            result = self.tools.evaluate('bringup', precision='bfp8_b')
        self.assertNotEqual(result['job'], 'one')
        self.assertEqual(remote.call_args.args[2]['precision'], 'bfp8_b')

    def test_background_reference_status_never_polls_tt(self):
        job = self.tools.run_dir / 'jobs/study'
        write_json(job / 'request.json', {'kind': 'reference-study', 'cluster_sha256': digest(self.config)})
        write_json(self.tools.run_dir / 'clusters' / (digest(self.config) + '.json'), self.config)
        with patch.object(cluster, 'remote') as tt, patch.object(tools, 'reference_progress',
                return_value={'phase': 'generate', 'background_nonblocking': True}) as reference:
            status = self.tools.status()
        tt.assert_not_called()
        reference.assert_called_once_with(self.config, job)
        self.assertTrue(status['study']['progress']['background_nonblocking'])

    def test_perf_and_profile_jobs_use_the_trusted_job_path(self):
        with patch.object(jobs.experiment, 'submit', return_value=support.future('future')) as remote:
            perf = self.tools.evaluate('perf')
            profiled = self.tools.profile('bringup', precision='bfp8_b')
        requests = [json.loads((self.tools.run_dir / 'jobs' / r['job'] / 'request.json').read_text())
                    for r in (perf, profiled)]
        self.assertEqual([(r['kind'], r['stage']) for r in requests], [('evaluate', 'perf'), ('profile', 'bringup')])
        self.assertEqual(remote.call_count, 2)
        for bad in (lambda: self.tools.evaluate('full'), lambda: self.tools.profile('perf'),
                    lambda: self.tools.profile('smoke', precision='int8')):
            with self.assertRaises(ValueError):
                bad()

    def test_campaign_default_model_reaches_run_and_evaluate(self):
        self.tools.model = 'parakeet'
        with patch.object(jobs.experiment, 'submit', return_value=support.future('future')) as remote:
            self.tools.run('python probe.py')
            self.assertEqual(remote.call_args.args[2]['model'], 'parakeet')
            self.tools.evaluate()
            self.assertEqual(remote.call_args.args[2]['model'], 'parakeet')
            self.tools.evaluate(model='parakeet')
            self.assertEqual(remote.call_args.args[2]['model'], 'parakeet')

    def test_explicit_profile_model_selects_checkpoint_and_its_public_suite(self):
        self.tools.model = 'parakeet'
        with patch.object(jobs.experiment, 'submit', return_value=support.future('future')) as remote:
            for model in ('parakeet', 'parakeet'):
                with self.subTest(model=model):
                    job = self.tools.run('python profile.py', public_input_stage='bringup', model=model)
                    request = remote.call_args.args[2]
                    saved = json.loads((self.tools.run_dir / 'jobs' / job['job'] / 'request.json').read_text())
                    self.assertEqual(request['model'], model)
                    self.assertEqual(saved['model'], model)
                    self.assertEqual(saved['suite_definition_sha256'],
                                     jobs.suite_identity(self.tools.run_dir, model, 'bringup'))
                    self.assertNotIn('stage', request)
            self.tools.run('python profile.py')
            self.assertEqual(remote.call_args.args[2]['model'], 'parakeet')
        self.assertEqual(self.tools.model, 'parakeet')

    def test_mesh_run_requests_every_mesh_card_under_the_mesh_lease(self):
        with patch.object(jobs.experiment, 'submit') as single:
            with self.assertRaisesRegex(ValueError, 'mesh stage'):
                self.tools.run('python probe.py', mesh=True)  # this pack declares no mesh stage
            with patch.dict(port.PORT, {'mesh_cards': 4}), \
                 patch.object(runtime, "job", return_value=lambda fn: SimpleNamespace(
                     submit=lambda *a: support.future('future'))) as job:
                self.tools.run('python probe.py', mesh=True)
                job.assert_called_once_with({'mesh': 1, 'tt': 4})
                saved = [json.loads(p.read_text()) for p in (self.tools.run_dir / 'jobs').glob('*/request.json')]
                self.assertEqual([r.get('mesh_cards') for r in saved], [4])
                self.tools.run('python other.py')
                single.assert_called_once()
                self.assertNotIn('mesh_cards', single.call_args.args[2])

    def test_invalid_profile_model_is_rejected_before_submission(self):
        with patch.object(jobs.experiment, 'submit') as remote:
            with self.assertRaisesRegex(ValueError, 'Unknown model'):
                self.tools.run('python profile.py', model='unregistered-model')
        remote.assert_not_called()
        self.assertFalse((self.tools.run_dir / 'jobs').exists())

    def test_supervisor_tracks_distinct_candidate_outcomes_and_baseline_evidence(self):
        self.archive('bringup', {'verdict': {'passed': True, 'suite_sha256': 'suite'}})
        self.tools.status()
        old = json.loads((self.tools.run_dir / 'jobs/one/request.json').read_text())
        write_json(self.tools.run_dir / 'jobs/two/request.json', {**old, 'code_sha256': 'changed'})
        write_json(self.tools.run_dir / 'jobs/two/result.json', {'verdict': {'passed': False}})
        facts = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        self.assertEqual(facts['evaluated_candidates'], [{'model': 'parakeet', 'precision': 'bf16',
                                                         'distinct_sources': 2, 'passing_sources': 1}])
        self.assertEqual(facts['baselines'][0]['code_sha256'], old['code_sha256'])
        self.assertIn('optimization', facts['objective'])

    def test_profile_run_requests_public_inputs_without_evaluation_dedup(self):
        self.archive('bringup', {'verdict': {'passed': True}})
        with patch.object(jobs.experiment, 'submit', return_value=support.future('future')) as remote:
            first = self.tools.run('python profile.py', public_input_stage='bringup')
            second = self.tools.run('python profile.py', public_input_stage='bringup')
        self.assertNotEqual(first['job'], second['job'])
        self.assertEqual(remote.call_count, 2)
        request = remote.call_args.args[2]
        self.assertEqual(request['kind'], 'run')
        self.assertEqual(request['public_input_stage'], 'bringup')
        self.assertNotIn('stage', request)
        saved = json.loads((self.tools.run_dir / 'jobs' / first['job'] / 'request.json').read_text())
        self.assertEqual(saved['code_sha256'], jobs.code_hash(jobs.snapshot(self.tools.workspace)))
        self.assertEqual(saved['suite_definition_sha256'], jobs.suite_identity(self.tools.run_dir, 'parakeet', 'bringup'))
        with self.assertRaises(ValueError):
            self.tools.run('python profile.py', public_input_stage='full')

    def test_profile_accepts_one_perf_case_and_forwards_it(self):
        with patch.object(tools.PortTools, '_start_job', side_effect=lambda request: request) as start:
            self.assertEqual(self.tools.profile(stage='perf', case='sweep-b1024')['case'], 'sweep-b1024')
            self.assertNotIn('case', self.tools.profile(stage='bringup'))
            for stage, case in (('perf', ''), ('smoke', 'batch'), ('full', ''), ('perf', '../x')):
                with self.subTest(stage=stage, case=case), self.assertRaises(ValueError):
                    self.tools.profile(stage=stage, case=case)
        self.assertEqual(start.call_count, 2)


if __name__ == "__main__":
    unittest.main()
