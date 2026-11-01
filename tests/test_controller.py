"""The controller: supervisor facts, cluster refresh, timeout recovery, capacity waits."""
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


class ControllerTests(RunFixture):
    def test_timeout_recovery_requires_real_probe_and_cannot_repeat(self):
        health, job, request, result = self.timeout_fixture()
        healthy = {'cards': [{**health['cards'][0], 'state': 'healthy'}]}
        with patch.object(cluster, 'remote', side_effect=[{'returncode': 0}, healthy]) as remote:
            self.assertEqual(controller.recover_timed_out_card(
                self.tools.run_dir, self.config, health, 'parakeet'), healthy)
            probe = remote.call_args_list[0].args[2]
            self.assertEqual(probe['kind'], 'preflight')
            self.assertEqual(probe['pci'], result['pci'])
            self.assertEqual(probe['model'], 'parakeet')
            self.assertEqual(remote.call_args_list[1].args[2], {'kind': 'health'})
        receipt = json.loads(next((self.tools.run_dir / 'recoveries').glob('*.json')).read_text())
        self.assertEqual(receipt['state'], 'complete')
        self.assertEqual(receipt['failed_job'], job.name)
        with patch.object(cluster, 'remote') as remote:
            # Even another terminal timeout on the same card cannot reset the budget.
            health['cards'][0]['job'] = 'second-timeout'
            write_json(job.parent / 'second-timeout/request.json', request)
            write_json(job.parent / 'second-timeout/result.json', {**result, 'job': 'second-timeout'})
            self.assertEqual(controller.recover_timed_out_card(
                self.tools.run_dir, self.config, health, 'parakeet'), health)
            remote.assert_not_called()

    def test_timeout_recovery_refuses_pending_memory_failure_or_wrong_identity(self):
        health, job, request, result = self.timeout_fixture()
        variants = [({}, {'failure': 'host_memory_limit', 'host_memory_limit': True}),
                    ({}, {'timeout': False}), ({}, {'pci': 'different'}),
                    ({}, {'returncode': 0}), ({}, {'job': 'different'}),
                    ({}, {'cluster_sha256': 'old-environment'}),
                    ({'cluster_sha256': 'old-environment'}, {})]
        with patch.object(cluster, 'remote') as remote:
            for req, res in variants:
                write_json(job / 'request.json', {**request, **req})
                write_json(job / 'result.json', {**result, **res})
                self.assertEqual(controller.recover_timed_out_card(
                    self.tools.run_dir, self.config, health, 'parakeet'), health)
            write_json(job / 'request.json', request)
            write_json(job / 'result.json', result)
            write_json(job.parent / 'pending/request.json', {'kind': 'evaluate'})
            self.assertEqual(controller.recover_timed_out_card(
                self.tools.run_dir, self.config, health, 'parakeet'), health)
            remote.assert_not_called()

    def test_timeout_recovery_preserves_failed_or_ambiguous_probe(self):
        health, job, request, result = self.timeout_fixture()
        with patch.object(cluster, 'remote', side_effect=TimeoutError('transport ambiguous')):
            with self.assertRaises(TimeoutError):
                controller.recover_timed_out_card(self.tools.run_dir, self.config, health, 'parakeet')
        receipt = next((self.tools.run_dir / 'recoveries').glob('*.json'))
        self.assertEqual(json.loads(receipt.read_text())['state'], 'submitted')
        with patch.object(cluster, 'remote') as remote:
            self.assertEqual(controller.recover_timed_out_card(
                self.tools.run_dir, self.config, health, 'parakeet'), health)
            remote.assert_not_called()
        # A separate, fresh configuration gets its own single probe budget.
        replacement = {**self.config, 'tt': {'host': '192.0.2.3'}}
        write_json(job / 'request.json', {**request, 'cluster_sha256': digest(replacement)})
        write_json(job / 'result.json', {**result, 'cluster_sha256': digest(replacement)})
        with patch.object(cluster, 'remote', side_effect=[{'returncode': 124, 'timeout': True}, health]):
            self.assertEqual(controller.recover_timed_out_card(
                self.tools.run_dir, replacement, health, 'parakeet'), health)

    def test_timeout_recovery_claim_is_exclusive_even_with_stale_existence_check(self):
        health, job, request, result = self.timeout_fixture()
        receipt = self.tools.run_dir / 'recoveries' / (
            digest([digest(self.config), result['pci']]) + '.json')
        write_json(receipt, {'state': 'submitted', 'request': {'job': 'other-probe'}})
        original_exists = Path.exists
        with patch.object(Path, 'exists', lambda p: False if p == receipt else original_exists(p)), \
                patch.object(cluster, 'remote') as remote:
            self.assertEqual(controller.recover_timed_out_card(
                self.tools.run_dir, self.config, health, 'parakeet'), health)
            remote.assert_not_called()
        self.assertEqual(json.loads(receipt.read_text())['request']['job'], 'other-probe')

    def test_capacity_wait_honours_reset_time_and_budget(self):
        now, mono = 1_000_000.0, 500.0
        with patch.object(time, 'time', return_value=now), \
             patch.object(time, 'monotonic', return_value=mono):
            self.assertEqual(controller.capacity_wait({'retry_after': now + 600}, 1, mono + 3600), 630)
            self.assertEqual(controller.capacity_wait({'retry_after': now - 5}, 1, mono + 3600), 30)
            self.assertIsNone(controller.capacity_wait({'retry_after': now + 7200}, 1, mono + 3600))
            self.assertEqual([controller.capacity_wait({}, r, mono + 3600) for r in (1, 2, 5, 9)], [60, 120, 900, 900])
            self.assertIsNone(controller.capacity_wait({}, 31, mono + 36000))

    def test_refresh_unchanged_configuration_does_not_reprobe(self):
        with patch.object(cluster, 'load', return_value=self.config), patch.object(controller, 'prepare') as prepare:
            config, status = controller.refresh_cluster(self.tools.run_dir, 'cluster.yaml', self.config)
        self.assertIs(config, self.config)
        self.assertEqual(status, 'unchanged')
        prepare.assert_not_called()

    def test_refresh_waits_for_pending_jobs_then_prepares_replacement(self):
        self.archive('smoke')
        replacement = {**self.config, 'reference': {'host': '192.0.2.8'}}
        with patch.object(cluster, 'load', return_value=replacement), patch.object(controller, 'prepare') as prepare:
            self.assertEqual(controller.refresh_cluster(self.tools.run_dir, 'cluster.yaml', self.config),
                             (self.config, 'draining'))
            prepare.assert_not_called()
            write_json(self.tools.run_dir / 'jobs/one/result.json', {'returncode': 0})
            self.assertEqual(controller.refresh_cluster(self.tools.run_dir, 'cluster.yaml', self.config),
                             (replacement, 'changed'))
            prepare.assert_called_once_with(self.tools.run_dir, replacement)

    def test_failed_replacement_identity_or_health_keeps_active_cluster(self):
        run = self.tools.run_dir
        (run / 'harness').mkdir(exist_ok=True)
        for name in ('runner.py', 'evaluate.py'):
            (run / 'harness' / name).write_text('# frozen harness')
        replacement = {'tt': {'host': '192.0.2.3', 'work_dir': '/srv/tt', 'python': '/env/python',
                              'metal': '/metal', 'weights_by_model': {'parakeet': '/weights/model.bin'}, 'metal_commit': 'pinned'},
                       'reference': self.config['reference']}
        with patch.object(cluster, 'upload'), patch.object(cluster, 'ssh', return_value=b'wrong wrong'), \
             patch.object(cluster, 'remote') as remote:
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                controller.prepare(run, replacement)
            remote.assert_not_called()
        unhealthy = {'cards': [{'pci': '0000:02:00.0', 'state': 'quarantined'}]}
        with patch.object(cluster, 'upload'), \
             patch.object(cluster, 'ssh', side_effect=[b'pinned', (model_spec('parakeet')['files']['model.safetensors']['sha256'] + ' model.bin').encode()]), \
             patch.object(cluster, 'remote', side_effect=[unhealthy, {'returncode': 1}, unhealthy]):
            with self.assertRaisesRegex(RuntimeError, 'no healthy TT cards'):
                controller.prepare(run, replacement)
        self.assertEqual(json.loads((run / 'cluster.json').read_text()), self.config)
        self.assertFalse((run / 'clusters' / (digest(replacement) + '.json')).exists())

    def test_supervisor_facts_report_real_source_and_exact_exposed_controls(self):
        self.delivery_receipt(job='one', stage='smoke', model='parakeet')
        facts = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        self.assertTrue(facts['workspace']['has_backend'])
        # Suite sizes come from the pack evaluator; this fixture ships no corpus.
        self.assertEqual(facts['controls']['fixed_suites']['details'], 'CONTRACT.md')
        self.assertEqual(facts['workspace']['current_source_passes'][0]['model'], 'parakeet')
        (self.tools.workspace / 'backend.py').write_text('# changed')
        self.assertEqual(controller.supervisor_facts(self.tools.workspace, self.tools.run_dir,
                                             self.config)['workspace']['current_source_passes'], [])

    def test_supervisor_facts_carry_objective_and_search(self):
        self.delivery_receipt(job='one', stage='bringup', model='parakeet')
        facts = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        self.assertEqual(facts['controls']['evaluate']['stage'], ['smoke', 'bringup', 'perf'])
        self.assertIn('profile', facts['controls'])
        optimization = facts['optimization']
        self.assertEqual(optimization['objective']['pareto'], list(objective.DEFAULT_PARETO))
        self.assertEqual(optimization['nodes'], 1)
        self.assertTrue(optimization['beam'] and optimization['beam'][0]['offered'])
        self.assertNotIn('mergeability', facts)
        write_json(self.tools.run_dir / 'manifest.json', {'optimization': objective.spec(primary='p99_latency'),
                                                          'beam': 2})
        facts = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        self.assertEqual(facts['optimization']['objective']['primary'], 'p99_latency')

    def test_delivery_anchor_does_not_hide_other_models_or_precision_gaps(self):
        self.delivery_receipt(stage='smoke', job='anchor-smoke')
        self.delivery_receipt(job='anchor-bringup')
        (self.tools.workspace / 'STATE.md').write_text('All models and precisions passed!')
        facts = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config, 'parakeet')
        self.assertEqual(facts['selected_model'], 'parakeet')
        scope = facts['delivery_scope']
        self.assertEqual(scope['models'], ['parakeet'])
        self.assertEqual(scope['development_anchor'], facts['selected_model'])
        matrix = scope['short_gates']
        for stage, job in [('smoke', 'anchor-smoke'), ('bringup', 'anchor-bringup')]:
            self.assertEqual(matrix['parakeet']['bf16'][stage]['status'], 'passed')
            self.assertEqual(matrix['parakeet']['bf16'][stage]['job'], job)
        for stage in ('smoke', 'bringup'):
            self.assertEqual(matrix['parakeet']['bfp8_b'][stage]['status'], 'missing')
            self.assertEqual(matrix['parakeet']['bfp8_b'][stage]['job'], None)
        self.assertIn('not IEEE FP8', scope['precision_policy'])
        self.assertIn('ONE production-level reusable', facts['objective'])

    def test_delivery_never_inherits_source_cluster_suite_or_precision_passes(self):
        folder, original_request, original_result = self.delivery_receipt()
        cases = [({'code_sha256': 'old-source'}, {}, {}, 'historical'),
                 ({'cluster_sha256': 'old-runtime'}, {}, {}, 'historical'),
                 ({'suite_definition_sha256': 'old-suite'}, {}, {}, 'historical'),
                 ({}, {'job': 'wrong-job'}, {}, 'unverified'),
                 ({}, {'cluster_sha256': 'old-runtime'}, {}, 'unverified'),
                 ({}, {'model': 'other-model'}, {}, 'unverified'),
                 ({}, {'precision': 'bfp8_b'}, {}, 'unverified'),
                 ({}, {}, {'model_key': 'other-model'}, 'unverified'),
                 ({}, {}, {'stage': 'smoke'}, 'unverified'),
                 ({}, {}, {'precision': {'requested': 'bfp8_b'}}, 'unverified'),
                 ({}, {}, {'suite_sha256': ''}, 'unverified'),
                 ({}, {'returncode': 124}, {}, 'failed'),
                 ({}, {'failure': 'infrastructure'}, {}, 'failed'),
                 ({}, {'timeout': True}, {}, 'failed'),
                 ({}, {'host_memory_limit': True}, {}, 'failed'),
                 ({}, {}, {'passed': False}, 'failed')]
        for req, res, verdict, expected in cases:
            with self.subTest(req=req, res=res, verdict=verdict):
                write_json(folder / 'request.json', {**original_request, **req})
                write_json(folder / 'result.json', {**original_result, **res,
                    'verdict': {**original_result['verdict'], **verdict}})
                facts = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
                matrix = facts['delivery_scope']['short_gates']
                self.assertEqual(matrix['parakeet']['bf16']['bringup']['status'], expected)
                self.assertEqual(matrix['parakeet']['bfp8_b']['bringup']['status'], 'missing')
                self.assertEqual(facts['workspace']['current_source_passes'], [])

    def test_delivery_latest_current_failure_or_pending_supersedes_earlier_pass(self):
        first, _, _ = self.delivery_receipt(job='earlier')
        later, request, result = self.delivery_receipt(job='later')
        os.utime(first / 'request.json', (1, 1))
        os.utime(later / 'request.json', (2, 2))
        for status in ('failed', 'pending'):
            if status == 'failed':
                write_json(later / 'result.json', {**result, 'returncode': 1})
            else:
                (later / 'result.json').unlink()
            facts = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
            cell = facts['delivery_scope']['short_gates']['parakeet']['bf16']['bringup']
            self.assertEqual((cell['job'], cell['status']), ('later', status))
            self.assertEqual(facts['workspace']['current_source_passes'], [])
        write_json(later / 'request.json', {**request, 'code_sha256': 'historical'})
        cell = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir,
            self.config)['delivery_scope']['short_gates']['parakeet']['bf16']['bringup']
        self.assertEqual((cell['job'], cell['status']), ('earlier', 'passed'))
        self.assertEqual(cell['historical_jobs'], ['later'])

    def test_delivery_bfp8_evidence_is_separate_and_reference_study_is_not_a_tt_gate(self):
        self.delivery_receipt(job='tt-bfp8', model='parakeet', precision='bfp8_b')
        folder, request, _ = self.delivery_receipt(job='cuda-study', model='parakeet')
        write_json(folder / 'request.json', {**request, 'kind': 'reference-study'})
        matrix = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir,
                                      self.config)['delivery_scope']['short_gates']
        self.assertEqual(matrix['parakeet']['bfp8_b']['bringup']['status'], 'passed')
        self.assertEqual(matrix['parakeet']['bfp8_b']['bringup']['job'], 'tt-bfp8')
        self.assertEqual(matrix['parakeet']['bf16']['bringup']['status'], 'missing')
        self.assertEqual(matrix['parakeet']['bf16']['bringup']['status'], 'missing')

    def test_delivery_preserves_legacy_other_precision_pass_shape(self):
        self.archive('smoke', {'verdict': {'passed': True}}, precision='fp32')
        facts = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        self.assertEqual(facts['workspace']['current_source_passes'],
                         [{'stage': 'smoke', 'model': 'parakeet', 'precision': 'fp32'}])
        self.assertEqual(facts['delivery_scope']['short_gates']['parakeet']['bf16']['smoke']['status'], 'missing')

    def test_supervisor_distinguishes_backend_from_snapshot_hash(self):
        import hashlib
        self.archive('smoke', {'verdict': {'passed': True}})
        before = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)['workspace']
        expected = hashlib.sha256((self.tools.workspace / 'backend.py').read_bytes()).hexdigest()
        self.assertEqual(before['backend_sha256'], expected)
        self.assertNotEqual(before['backend_sha256'], before['source_sha256'])
        (self.tools.workspace / 'helper.py').write_text('# new inference dependency')
        after = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)['workspace']
        self.assertEqual(after['backend_sha256'], before['backend_sha256'])
        self.assertNotEqual(after['source_sha256'], before['source_sha256'])
        self.assertEqual(after['current_source_passes'], [])
        (self.tools.workspace / 'backend.py').unlink()
        missing = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)['workspace']
        self.assertIsNone(missing['backend_sha256'])

    def test_diagnostic_notes_do_not_hide_implementation_from_supervisor(self):
        for index in range(70):
            (self.tools.workspace / f'A_DIAGNOSTIC_{index:03}.md').write_text('historical note')
        (self.tools.workspace / 'forecast.py').write_text('# standalone entry point')
        facts = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        workspace = facts['workspace']
        self.assertIn('backend.py', workspace['file_names'])
        self.assertIn('forecast.py', workspace['implementation_files'])
        self.assertNotIn('A_DIAGNOSTIC_000.md', workspace['implementation_files'])
        self.assertEqual(workspace['implementation_file_count'], len(workspace['implementation_files']))
        self.assertLessEqual(len(workspace['file_names']), 64)
        self.assertIn('tt-metal-style package', facts['objective'])

    def test_supervisor_can_distinguish_historical_runtime_receipts(self):
        self.archive('bringup', {'verdict': {'passed': True}})
        replacement = json.loads(json.dumps(self.config))
        replacement['tt']['metal'] = '/srv/replacement-runtime'
        replacement['tt']['python'] = '/srv/tokenizer-wrapper/bin/python'
        facts = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, replacement)
        self.assertEqual(facts['environment']['tt_runtime'], '/srv/replacement-runtime')
        self.assertEqual(facts['environment']['tt_python'], '/srv/tokenizer-wrapper/bin/python')
        self.assertEqual(facts['environment']['cluster_sha256'], digest(replacement))
        self.assertEqual(facts['recent_jobs'][0]['cluster_sha256'], digest(self.config))
        self.assertEqual(facts['workspace']['current_source_passes'], [])

    def test_supervisor_handoff_survives_missing_worker_response_without_certifying_claims(self):
        path = self.tools.workspace / 'STATE.md'
        path.write_text('Pending job alpha. Next: collect its receipt; all tests reportedly pass.')
        facts = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        self.assertEqual(facts['worker_state']['text'], path.read_text())
        self.assertTrue(facts['worker_state']['present'])
        self.assertFalse(facts['worker_state']['truncated'])
        self.assertIn('not verified evidence', facts['worker_state']['note'])
        self.assertEqual(facts['workspace']['current_source_passes'], [])
        self.assertEqual(facts['recent_jobs'], [])
        path.write_text('Saved before timeout. Next: finish the checkpoint validation fix.')
        refreshed = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)
        self.assertEqual(refreshed['worker_state']['text'], path.read_text())

    def test_supervisor_handoff_bounds_large_notes_and_preserves_both_ends(self):
        text = 'CURRENT RESULT\n' + 'long historical detail\n' * 1000 + '\nNEXT: collect job beta'
        (self.tools.workspace / 'STATE.md').write_text(text)
        state = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)['worker_state']
        self.assertTrue(state['truncated'])
        self.assertEqual(state['available_chars'], len(text))
        self.assertLessEqual(len(state['text']), 12000)
        self.assertTrue(state['text'].startswith('CURRENT RESULT\n'))
        self.assertTrue(state['text'].endswith('\nNEXT: collect job beta'))
        self.assertIn('Middle omitted', state['text'])

    def test_supervisor_reports_missing_handoff_without_inventing_state(self):
        state = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config)['worker_state']
        self.assertFalse(state['present'])
        self.assertEqual(state['text'], '')
        self.assertEqual(state['available_chars'], 0)
        self.assertFalse(state['truncated'])

    def test_supervisor_can_see_explicit_profile_models_and_anchor_default(self):
        facts = controller.supervisor_facts(self.tools.workspace, self.tools.run_dir, self.config, 'parakeet')
        self.assertEqual(facts['controls']['run']['model'], ['parakeet'])
        self.assertEqual(facts['controls']['run']['default_model'], 'parakeet')


if __name__ == "__main__":
    unittest.main()
