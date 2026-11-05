"""control/jobs.py: what each trusted job asks the host and what its receipt says."""
import json
import unittest

from run_fixture import CONFIG, RunFixture
from portforge_tt.control import jobs
from portforge_tt.control.cluster import identity


class JobTests(RunFixture):
    def test_code_identity_is_the_kernel_source_only(self):
        a = self.files()
        b = dict(a, **{'TRANSFORMS.json': '{}', 'NOTES.md': 'changed'})
        self.assertEqual(jobs.code_hash(a), jobs.code_hash(b))
        self.assertNotEqual(jobs.code_hash(a), jobs.code_hash(self.files(a['kernel.S'] + '\n')))
        snap = jobs.snapshot(self.workspace)
        self.assertIn('context/reference.c', snap)
        self.assertEqual(set(jobs.candidate(snap)), {'kernel.S'})  # instructions and context/ stay out

    def test_check_passes_on_g0_g2_and_never_touches_the_board(self):
        host = self.host()
        job, result = self.run_job({'kind': 'evaluate', 'stage': 'check'})
        self.assertTrue(jobs.passing_evaluation(result, job, identity(CONFIG), 'check'))
        self.assertEqual([r['kind'] for r in host.requests], ['gates'])
        self.assertEqual(result['profile']['text_bytes'], 640)
        self.assertNotIn('board_seconds', result)
        self.assertEqual(json.loads((self.run_dir / 'jobs' / job / 'result.json').read_text()), result)

    def test_a_failed_gate_is_a_failed_verdict_not_an_infrastructure_error(self):
        self.host(failed_at='G1')
        job, result = self.run_job({'kind': 'evaluate', 'stage': 'perf'})
        self.assertEqual(result['returncode'], 0)
        self.assertFalse(result['verdict']['passed'])
        self.assertEqual(result['verdict']['failures'], ['failed at G1'])
        self.assertNotIn('perf', result['verdict'])  # no bench for an incorrect kernel

    def test_perf_confirms_a_flag_in_a_second_session_at_twice_the_rounds(self):
        host = self.host(flagged=['a@1'])
        job, result = self.run_job({'kind': 'evaluate', 'stage': 'perf'})
        benches = [r for r in host.requests if r['kind'] == 'bench']
        self.assertEqual([b['rounds'] for b in benches], [jobs.BENCH_ROUNDS, 2 * jobs.BENCH_ROUNDS])
        self.assertNotEqual(benches[0]['seed'], benches[1]['seed'])
        perf = result['verdict']['perf']
        self.assertEqual(perf['confirmed_flags'], ['a@1'])
        self.assertEqual(len(perf['sessions']), 2)
        self.assertIn('board_seconds', result)

    def test_host_failure_is_infrastructure(self):
        self.host()
        from unittest import mock
        with mock.patch('portforge_tt.control.cluster.remote', return_value={'ok': False, 'error': 'ssh reset'}):
            _, result = self.run_job({'kind': 'evaluate', 'stage': 'check'})
        self.assertEqual((result['returncode'], result['failure']), (-1, 'infrastructure'))
        self.assertIn('ssh reset', result['error'])

    def test_certify_needs_heldout_verify_and_agreeing_sessions(self):
        cases = [({}, 2.0, True, []), ({'heldout': False}, 2.0, False, ['held-out regimes failed']),
                 ({'verify': ('PARTIAL',)}, 2.0, False, ['verify: cbmc PARTIAL']),
                 ({}, 1.5, False, ['sessions disagree: 2.0000 then 1.5000'])]
        for options, speedup, passed, failures in cases:
            with self.subTest(options=options, speedup=speedup):
                host = self.host(speedup=speedup, **options)
                _, result = self.run_job({'kind': 'certify', 'stage': 'full', 'heldout_seed': 99,
                                          'submitted_metrics': {'speedup_geomean': 2.0}})
                self.assertEqual(result['verdict']['passed'], passed)
                self.assertEqual(result['verdict']['failures'], failures)
                certify = [r for r in host.requests if r['kind'] == 'certify'][0]
                self.assertEqual(certify['heldout_seed'], 99)

    def test_quick_run_uses_its_own_build_slot(self):
        host = self.host()
        _, result = self.run_job({'kind': 'run', 'command': 'check'})
        self.assertEqual(host.requests[0]['kind'], 'dev_check')
        self.assertTrue(result['output']['check']['passed'])
        self.assertNotIn('verdict', result)

    def test_board_seconds_sum_over_results(self):
        self.host()
        self.run_job({'kind': 'evaluate', 'stage': 'perf'})
        self.run_job({'kind': 'evaluate', 'stage': 'check'})
        self.assertGreaterEqual(jobs.board_seconds(self.run_dir), 0)


if __name__ == '__main__':
    unittest.main()
