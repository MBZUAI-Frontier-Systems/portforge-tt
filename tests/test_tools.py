"""control/tools.py: the agents' scope, job dedup, board budget and the submission rule."""
import json
from pathlib import Path
import unittest
from unittest import mock

from run_fixture import CONFIG, RunFixture
import support
from portforge_tt.control import tools as tools_mod
from portforge_tt.control.cluster import identity


class ToolTests(RunFixture):
    def setUp(self):
        super().setUp()
        self.submitted = []

        def submit(request):
            def run(run_dir, job, payload, config):
                self.submitted.append((job, payload))
                return support.FakeFuture({'job': job})
            return run
        mock.patch.object(tools_mod, 'submit', submit).start()
        self.tools = tools_mod.PortTools.__new__(tools_mod.PortTools)
        self.tools.mcp = mock.Mock()
        self.tools.setup(str(self.workspace), str(self.run_dir))

    def receipt(self, job, stage, passed=True, metrics=None):
        verdict = {'passed': passed, 'stage': stage, 'suite_sha256': 'suite',
                   **({'perf': {'metrics': metrics}} if metrics else {})}
        (self.run_dir / 'jobs' / job / 'result.json').write_text(json.dumps(
            {'returncode': 0, 'job': job, 'cluster_sha256': identity(CONFIG), 'verdict': verdict}))

    def test_scope_instructions_and_context_are_read_only(self):
        for path in ('CONTRACT.md', 'TECHNIQUES.md', 'SPEC.md', 'RVV_TECHNIQUES.md', 'context/reference.c'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.tools.write(path, 'x')
        self.assertEqual(self.tools.write('NOTES.md', 'n'), {'saved': 'NOTES.md'})
        with self.assertRaises(ValueError):
            self.tools.read('../outside')
        self.assertIn('TECHNIQUES.md', self.tools.files())
        self.assertIn('context/reference.c', self.tools.files())

    def test_read_pages_large_files(self):
        self.tools.write('big.txt', 'a' * 70000)
        first = self.tools.read('big.txt')
        self.assertIn('offset=60000', first)
        self.assertEqual(self.tools.read('big.txt', 60000), 'a' * 10000)

    def test_evaluate_dedups_unchanged_source_and_records_only_the_candidate(self):
        first = self.tools.evaluate('check')
        again = self.tools.evaluate('check')
        self.assertEqual(again, {'job': first['job'], 'reused': True})
        self.assertEqual(len(self.submitted), 1)
        source = json.loads((self.run_dir / 'jobs' / first['job'] / 'source.json').read_text())
        self.assertEqual(set(source), {'kernel.S'})
        request = json.loads((self.run_dir / 'jobs' / first['job'] / 'request.json').read_text())
        self.assertEqual((request['stage'], request['family']), ('check', 'refmvs-splat_mv'))
        self.receipt(first['job'], 'check', passed=False)
        self.assertEqual(self.tools.evaluate('check')['status'], 'running')  # a failure is retried

    def test_unknown_stage_and_spent_board_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'certification is controller-only'):
            self.tools.evaluate('full')
        manifest = json.loads((self.run_dir / 'manifest.json').read_text())
        (self.run_dir / 'manifest.json').write_text(json.dumps({**manifest, 'board_minutes': 0}))
        with self.assertRaisesRegex(ValueError, 'board budget'):
            self.tools.evaluate('perf')

    def test_submit_needs_a_perf_pass_of_the_current_source(self):
        with self.assertRaisesRegex(ValueError, 'perf pass'):
            self.tools.submit()
        job = self.tools.evaluate('perf')['job']
        self.receipt(job, 'perf', metrics={'speedup_geomean': 2.5})
        original = (self.workspace / 'kernel.S').read_text()
        self.tools.write('kernel.S', '/* changed */')
        with self.assertRaisesRegex(ValueError, 'perf pass'):
            self.tools.submit()
        self.tools.write('kernel.S', original)
        self.assertEqual(self.tools.submit(), {'status': 'frozen_for_certification'})
        submission = json.loads((self.run_dir / 'submission.json').read_text())
        self.assertEqual((submission['job'], submission['metrics']), (job, {'speedup_geomean': 2.5}))
        with self.assertRaises(ValueError):
            self.tools.write('kernel.S', 'late edit')

    def test_status_bounds_and_names_the_log(self):
        job = self.tools.run('check')['job']
        (self.run_dir / 'jobs' / job / 'result.json').write_text(json.dumps(
            {'returncode': 0, 'rpc': 'runs/x/rpc/dev_check-0123456789ab', 'output': {'text': 'y' * 5000}}))
        item = self.tools.status(compact=True)[job]
        self.assertLess(len(item['output']['text']), 2000)
        self.assertEqual(item['details']['tool'], 'port_log')
        with self.assertRaises(ValueError):
            self.tools.status(job_id='../x')


if __name__ == '__main__':
    unittest.main()
