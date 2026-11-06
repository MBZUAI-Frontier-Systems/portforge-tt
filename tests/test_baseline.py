"""Operator baseline: stage rules, workspace copy and the job records it writes (no hardware, no remote calls)."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support  # noqa: F401  (pack and registry setup)
import baseline, jobs
from cluster import identity as digest
from runner import write_json

support.configure_port()


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.run, self.workspace = root / 'run', root / 'ws'
        self.workspace.mkdir()
        (self.workspace / 'backend.py').write_text('x = 1\n')
        self.config = {'tt': {'host': '192.0.2.1', 'resources': {'tt': 4}}, 'reference': {'host': '192.0.2.2'}}
        write_json(self.run / 'cluster.json', self.config)

    def submit(self, stage, precision='bf16'):
        """baseline.submit with experiment stubbed; returns (job, result, request sent, submitted resources)."""
        sent = {}

        def original(run_dir, job, request, config):
            sent.update(request=request, config=config)
            return {'returncode': 0}

        experiment = SimpleNamespace(_original=original,
                                     submit=lambda *args: support.FakeFuture(original(*args)))
        with patch.object(baseline, 'experiment', experiment), \
             patch.object(baseline, 'suite_identity', return_value='suite-hash'), \
             patch.object(baseline.runtime, 'job', wraps=baseline.runtime.job) as leased:
            job, result = baseline.submit(self.run, self.workspace, 'parakeet', precision, stage)
        return job, result, sent, [call.args[0] for call in leased.call_args_list]

    def test_stages_keep_campaign_order_and_reject_bad_sets(self):
        allowed = ['smoke', 'bringup', 'perf', 'full']
        self.assertEqual(baseline.parse_stages('full,perf,smoke', allowed), ['smoke', 'perf', 'full'])
        for bad in ('', ' , ', 'smoke,smoke', 'smoke,final', 'mesh'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                baseline.parse_stages(bad, allowed)

    def test_workspace_needs_backend_and_skips_caches(self):
        source, run = Path(self.temp.name) / 'src', Path(self.temp.name) / 'copy-run'
        source.mkdir()
        run.mkdir()
        with self.assertRaises(ValueError):
            baseline.copy_workspace(source, run)
        (source / 'backend.py').write_text('x = 1\n')
        (source / '__pycache__').mkdir()
        (source / '__pycache__' / 'backend.cpython-310.pyc').write_bytes(b'0')
        workspace = baseline.copy_workspace(source, run)
        self.assertTrue((workspace / 'backend.py').is_file())
        self.assertFalse((workspace / '__pycache__').exists())

    def test_gated_stage_records_match_the_campaign_request(self):
        job, result, sent, leased = self.submit('smoke')
        record = json.loads((self.run / 'jobs' / job / 'request.json').read_text())
        files = jobs.snapshot(self.workspace)
        self.assertEqual(record, {'kind': 'evaluate', 'stage': 'smoke', 'precision': 'bf16', 'model': 'parakeet',
                                  'cluster_sha256': digest(self.config), 'suite_definition_sha256': 'suite-hash',
                                  'code_sha256': jobs.code_hash(files)})
        self.assertEqual(json.loads((self.run / 'jobs' / job / 'source.json').read_text()), files)
        self.assertEqual(sent['request'], {**{k: v for k, v in record.items() if k != 'code_sha256'}, 'files': files})
        self.assertEqual(sent['config'], self.config)
        self.assertEqual((result, leased), ({'returncode': 0}, []))

    def test_perf_holds_every_card_like_a_campaign_timed_job(self):
        baseline.runtime.configure(mesh=1, tt=4)  # main() sizes the leases from the cluster
        self.addCleanup(baseline.runtime.configure, mesh=1, tt=1)
        _, result, sent, leased = self.submit('perf')
        self.assertEqual(leased, [{'mesh': 1, 'tt': 4}])
        self.assertEqual((sent['request']['stage'], result), ('perf', {'returncode': 0}))

    def test_full_submission_matches_the_controller_request_shape(self):
        job, _, sent, _ = self.submit('full', 'bfp8_b')
        self.assertEqual(set(sent['request']), {'kind', 'stage', 'files', 'precision', 'model'})
        self.assertEqual((sent['request']['stage'], sent['request']['precision']), ('full', 'bfp8_b'))
        record = json.loads((self.run / 'jobs' / job / 'request.json').read_text())
        self.assertEqual(record['cluster_sha256'], digest(self.config))
        self.assertNotIn('suite_definition_sha256', record)

    def test_summary_reports_pass_fail_curve_and_quality_for_full_only(self):
        verdict = {'passed': True, 'accepted': True, 'failures': [], 'max_nrmse': 0.01, 'case_count': 3,
                   'timing': {'median_case_p50_seconds': 0.003}, 'precision': {'requested': 'bf16'},
                   'latency': {'median_case_p50_seconds': 0.003},
                   'scaling': [{'case': 'sweep-b16', 'batch': 16, 'items': 16, 'p50_seconds': 0.004,
                                'items_per_second': 4000.0, 'p99_seconds': 0.005}],
                   'quality': {'images': 1024}}
        passed = baseline.summarize('run', 'job1', 'full', {'returncode': 0, 'verdict': verdict, 'pci': '0000:42:00.0'})
        self.assertTrue(passed['passed'])
        self.assertEqual(passed['quality'], {'images': 1024})
        self.assertEqual(passed['scaling'], [{'case': 'sweep-b16', 'batch': 16, 'items': 16, 'p50_seconds': 0.004,
                                              'items_per_second': 4000.0}])
        self.assertEqual(passed['metrics']['throughput'], 4000.0)
        failed = baseline.summarize('run', 'job2', 'bringup',
                                    {'returncode': 0, 'verdict': dict(verdict, passed=False, failures=['x: NRMSE'])})
        self.assertFalse(failed['passed'])
        self.assertIsNone(failed['quality'])
        self.assertIn('x: NRMSE', failed['failures'])
        broken = baseline.summarize('run', 'job3', 'smoke', {'returncode': -1, 'failure': 'infrastructure', 'error': 'ssh'})
        self.assertEqual((broken['passed'], broken['failures'], broken['metrics']),
                         (False, ['ssh', 'execution failure: infrastructure'], {}))


if __name__ == '__main__':
    unittest.main()
