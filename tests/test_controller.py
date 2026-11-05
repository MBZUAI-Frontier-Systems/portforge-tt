"""control/controller.py: the supervisor's view, seeds, context and what gets certified."""
import json
import unittest
from unittest import mock

from run_fixture import CONFIG, RunFixture
from portforge_tt.control import controller, jobs, runtime


class ControllerTests(RunFixture):
    def setUp(self):
        super().setUp()
        runtime.configure(host=1, board=1, dev=2)

    def test_supervisor_sees_current_source_menu_board_and_jobs(self):
        self.host(speedup=1.8)
        self.run_job({'kind': 'evaluate', 'stage': 'check'})
        self.run_job({'kind': 'evaluate', 'stage': 'perf'})
        facts = controller.supervisor_facts(self.workspace, self.run_dir, CONFIG)
        self.assertEqual(facts['family'], 'refmvs-splat_mv')
        self.assertEqual(facts['current_source']['check']['status'], 'passed')
        self.assertEqual(facts['current_source']['perf']['metrics']['speedup_geomean'], 1.8)
        self.assertFalse(facts['current_source']['is_placeholder'])
        menu = facts['optimization']['beam']
        self.assertEqual(len(menu), 1)
        self.assertTrue(menu[0]['offered'])  # techniques reach the supervisor
        self.assertIn('max-lmul', menu[0]['applied'])
        self.assertEqual(facts['board']['minutes_budget'], self.board_minutes)
        self.assertEqual(len(facts['recent_jobs']), 2)
        json.dumps(facts)  # the supervisor receives it as JSON

    def test_a_port_in_progress_shows_the_whole_family_menu(self):
        facts = controller.supervisor_facts(self.workspace, self.run_dir, CONFIG)
        self.assertEqual(facts['optimization']['beam'], [])
        self.assertIn('k:width-dispatch', [t['id'] for t in facts['optimization']['family_menu']])

    def test_family_techniques_are_in_the_known_menu(self):
        from portforge_tt.ranking import search
        from portforge_tt.control.port import PORT
        known = search.known(PORT)
        self.assertIn('k:width-dispatch', known)
        self.assertIn('max-lmul', known)

    def test_seeds_are_evaluated_check_then_perf(self):
        host = self.host()
        (self.run_dir / 'seeds').mkdir()
        (self.run_dir / 'seeds' / 'upstream.json').write_text(json.dumps(self.files()))
        results = controller.evaluate_seeds(self.run_dir, self.manifest, CONFIG)
        self.assertEqual([r['stage'] for r in results], ['check', 'perf'])
        self.assertTrue(all(r['verdict']['passed'] for r in results))
        self.assertIn('bench', [r['kind'] for r in host.requests])

    def test_context_is_installed_read_only_and_indexed(self):
        self.host()
        names = jobs.context._original(str(self.run_dir), CONFIG)
        controller.install_context(self.run_dir, self.workspace, names)
        self.assertEqual((self.workspace / 'context' / 'reference_refmvs.c').read_text(), '/* C */')
        self.assertIn('context/riscv_asm.S', (self.workspace / 'SPEC.md').read_text())

    def test_certification_targets_the_submission_else_the_best_measured(self):
        self.host(speedup=1.2)
        self.run_job({'kind': 'evaluate', 'stage': 'perf'})
        self.host(speedup=2.4)
        faster = self.files('/* faster */\n' + self.files()['kernel.S'])
        self.run_job({'kind': 'evaluate', 'stage': 'perf'}, files=faster)
        files, metrics, origin = controller.certification_target(self.run_dir, CONFIG)
        self.assertEqual((origin, files['kernel.S'], metrics['speedup_geomean']), ('frontier', faster['kernel.S'], 2.4))
        (self.run_dir / 'submission.json').write_text(json.dumps({'files': self.files(), 'metrics': {'speedup_geomean': 1.2}}))
        files, metrics, origin = controller.certification_target(self.run_dir, CONFIG)
        self.assertEqual((origin, metrics['speedup_geomean']), ('submission', 1.2))

    def test_prepare_uploads_host_files_and_refuses_a_broken_host(self):
        with mock.patch('portforge_tt.control.cluster.remote', return_value={'ok': True, 'host': {'ok': True}}):
            controller.prepare(self.run_dir, CONFIG)
        uploaded = self.uploads.call_args.args[2]
        self.assertIn('portforge_tt/remote/runner.py', uploaded)
        self.assertTrue((self.run_dir / 'cluster.json').exists())
        with mock.patch('portforge_tt.control.cluster.remote', return_value={'ok': True, 'host': {'ok': False}}):
            with self.assertRaisesRegex(RuntimeError, 'cannot run gates'):
                controller.prepare(self.run_dir, CONFIG)

    def test_capacity_wait_is_bounded_by_the_deadline(self):
        import time
        self.assertEqual(controller.capacity_wait({}, 1, time.monotonic() + 3600), 60)
        self.assertIsNone(controller.capacity_wait({}, 1, time.monotonic() + 10))


if __name__ == '__main__':
    unittest.main()
