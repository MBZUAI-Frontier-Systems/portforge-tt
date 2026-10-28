"""Objective ranking, search tree, telemetry, profile parsing and mergeability without hardware."""
import support  # noqa: F401  (pack and registry setup)
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import merge_check
import objective
import opprofile
import portforge_eval
import search
import telemetry


def result(p50, p99=None, throughput=None, joules=None, dram=None, passed=True, suite='s' * 64, stage='perf'):
    verdict = dict(stage=stage, passed=passed, failures=[] if passed else ['case: NRMSE exceeds 0.04'],
                   suite_sha256=suite, max_nrmse=.01, latency=dict(median_case_p50_seconds=p50, max_case_p99_seconds=p99),
                   scaling=[dict(batch=4, items_per_second=throughput)] if throughput else [])
    out = dict(returncode=0, verdict=verdict, measurements=dict(resident_dram_bytes=dram) if dram else {})
    if joules is not None:
        out['energy'] = dict(cases=[dict(joules_per_item=joules, mean_watts=100.0, short_window=False)])
    return out


class ObjectiveTests(unittest.TestCase):
    def test_metrics_come_only_from_receipt_fields(self):
        found = objective.metrics(result(.01, .02, 400.0, .05, 2 ** 30))
        self.assertEqual(found, dict(p50_latency=.01, p99_latency=.02, throughput=400.0, joules_per_item=.05,
                                     mean_watts=100.0, dram_bytes=2 ** 30, max_nrmse=.01))
        short = result(.01, joules=.05)
        short['energy']['cases'][0]['short_window'] = True
        self.assertNotIn('joules_per_item', objective.metrics(short))

    def test_constraints_parse_absolute_and_relative_bounds(self):
        self.assertEqual(objective.parse_constraint('p99_latency<=1.1*baseline'),
                         dict(metric='p99_latency', op='<=', value=1.1, relative=True))
        self.assertEqual(objective.parse_constraint('dram_bytes <= 4e9')['value'], 4e9)
        self.assertTrue(objective.parse_constraint('throughput>=baseline')['relative'])
        for bad in ('speed<=1', 'p99_latency<1', 'p99_latency<=fast'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                objective.parse_constraint(bad)

    def test_spec_validates_and_merges_overrides(self):
        chosen = objective.spec({'scenario': 'offline', 'constraints': ['dram_bytes<=1e9']}, 'throughput',
                                ['p99_latency<=2*baseline'])
        self.assertEqual((chosen['scenario'], chosen['primary'], len(chosen['constraints'])), ('offline', 'throughput', 2))
        for bad in ({'primary': 'speed'}, {'scenario': 'batchy'}, {'pareto': []}, {'unknown': 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                objective.spec(bad)

    def test_unknown_metric_or_baseline_is_not_feasible(self):
        constraint = [objective.parse_constraint('p99_latency<=1.1*baseline')]
        self.assertFalse(objective.feasible({}, constraint, {'p99_latency': 1.0})[0])
        self.assertFalse(objective.feasible({'p99_latency': .5}, constraint, {})[0])
        self.assertTrue(objective.feasible({'p99_latency': 1.05}, constraint, {'p99_latency': 1.0})[0])
        self.assertFalse(objective.feasible({'p99_latency': 1.2}, constraint, {'p99_latency': 1.0})[0])

    def test_pareto_front_ignores_noise_and_keeps_tradeoffs(self):
        points = [dict(id='fast', metrics=dict(p50_latency=1.0, dram_bytes=10.0)),
                  dict(id='small', metrics=dict(p50_latency=2.0, dram_bytes=5.0)),
                  dict(id='worse', metrics=dict(p50_latency=2.5, dram_bytes=11.0)),
                  dict(id='noise', metrics=dict(p50_latency=1.01, dram_bytes=10.05))]
        ranked = objective.rank(points, objective.spec({'pareto': ['p50_latency', 'dram_bytes']}))
        self.assertEqual(sorted(ranked['frontier']), ['fast', 'noise', 'small'])

    def test_primary_ranking_respects_constraints(self):
        chosen = objective.spec(primary='p50_latency', constraints=['dram_bytes<=8'])
        points = [dict(id='fast', metrics=dict(p50_latency=1.0, dram_bytes=10.0)),
                  dict(id='ok', metrics=dict(p50_latency=2.0, dram_bytes=5.0))]
        ranked = objective.rank(points, chosen)
        self.assertEqual(ranked['frontier'], ['ok'])
        self.assertIn('fast', ranked['infeasible'])


class SearchTests(unittest.TestCase):
    def job(self, run, name, code, stage, outcome, transforms=None, at=0):
        folder = Path(run) / 'jobs' / name
        folder.mkdir(parents=True)
        (folder / 'request.json').write_text(json.dumps(dict(kind='evaluate', stage=stage, code_sha256=code,
                                                             model='parakeet', precision='bf16', cluster_sha256='c')))
        files = {'backend.py': 'x'}
        if transforms is not None:
            files['TRANSFORMS.json'] = json.dumps(transforms)
        (folder / 'source.json').write_text(json.dumps(files))
        if outcome is not None:
            (folder / 'result.json').write_text(json.dumps(outcome))
        for path in folder.iterdir():
            os.utime(path, (at, at))
        os.utime(folder / 'request.json', (at, at))

    def test_tree_frontier_and_contextual_rejections(self):
        known = search.catalogue()
        with tempfile.TemporaryDirectory() as run:
            self.job(run, 'a', 'A', 'perf', result(.010, .012, 300.0), dict(parent=None, applied=[]), 1)
            self.job(run, 'b', 'B', 'perf', result(.006, .008, 500.0),
                     dict(parent='A', applied=['trace-capture', 'not-a-technique'], note='trace'), 2)
            self.job(run, 'c', 'C', 'bringup', result(.005, passed=False, stage='bringup'),
                     dict(parent='B', applied=['trace-capture', 'native-norm-softmax']), 3)
            self.job(run, 'd', 'D', 'perf', None, dict(parent='B', applied=['trace-capture', 'bfp8-weights']), 4)
            nodes = search.build(run, set(known))
            self.assertEqual([n['code'] for n in nodes], ['A', 'B', 'C', 'D'])
            self.assertEqual(nodes[1]['parent_source'], 'declared')
            self.assertEqual(nodes[1]['declared']['unknown'], ['not-a-technique'])
            chosen = objective.spec()
            front = search.frontier(nodes, chosen, 'perf')[0]
            self.assertEqual(front['frontier'], ['B'])
            self.assertEqual(front['baseline']['id'], 'A')
            view = search.summary(nodes, chosen, known, seed=1, mask=0.0)
            rejected = view['rejected_in_context']
            self.assertIn(dict(parent='B', technique='native-norm-softmax', child='C',
                               reasons=['case: NRMSE exceeds 0.04']), rejected)
            offered = {item['id'] for item in view['beam'][0]['offered']}
            self.assertEqual(view['beam'][0]['parent'], 'B')
            self.assertNotIn('trace-capture', offered)          # already applied on B
            self.assertNotIn('native-norm-softmax', offered)    # tried under B
            self.assertIn('fp32-dest-acc', offered)
            search.write(run, nodes, chosen, known)
            self.assertEqual(len((Path(run) / 'search/nodes.jsonl').read_text().splitlines()), 4)
            # Under a different parent the rejected technique is still on the menu.
            self.assertNotIn(('A', 'native-norm-softmax'), search.tried(nodes))

    def test_masking_always_offers_something_and_is_seeded(self):
        known = search.catalogue()
        with tempfile.TemporaryDirectory() as run:
            self.job(run, 'a', 'A', 'perf', result(.01), dict(parent=None, applied=[]), 1)
            nodes = search.build(run, set(known))
            first = search.summary(nodes, objective.spec(), known, seed=7, mask=.99)
            again = search.summary(nodes, objective.spec(), known, seed=7, mask=.99)
            self.assertTrue(first['beam'][0]['offered'])
            self.assertEqual(first['beam'], again['beam'])

    def test_declared_transforms_tolerate_garbage(self):
        self.assertFalse(search.declared({'TRANSFORMS.json': '{nope'}, set())['valid'])
        self.assertEqual(search.declared({'TRANSFORMS.json': '[1, 2]'}, set())['applied'], [])


class TelemetryTests(unittest.TestCase):
    def test_hwmon_lookup_by_pci_and_energy_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / 'devices' / '0000:c1:00.0'
            device.mkdir(parents=True)
            sensor = root / 'hwmon' / 'hwmon3'
            sensor.mkdir(parents=True)
            (sensor / 'device').symlink_to(device)
            (sensor / 'power1_input').write_text('150000000\n')
            self.assertEqual(telemetry.hwmon_for_pci('0000:c1:00.0', root / 'hwmon'), sensor)
            self.assertIsNone(telemetry.hwmon_for_pci('0000:01:00.0', root / 'hwmon'))
            self.assertEqual(telemetry.read_watts(sensor), 150.0)
            sampler = telemetry.PowerSampler(sensor, interval=.01).start()
            time.sleep(.05)
            self.assertTrue(all(w == 150.0 for _, w in sampler.stop()))
        samples = [(t, 100.0) for t in range(0, 31)]
        measurements = dict(cases=[dict(name='a', window=[5.0, 25.0], items=4, seconds=[.1] * 200),
                                   dict(name='short', window=[1.0, 3.0], items=1, seconds=[.1] * 20),
                                   dict(name='no-window', window=None, items=1, seconds=[.1])])
        report = telemetry.energy_report(samples, measurements, idle=25.0)
        first = report['cases'][0]
        self.assertAlmostEqual(first['joules'], 2000.0)
        self.assertAlmostEqual(first['mean_watts'], 100.0)
        self.assertAlmostEqual(first['active_joules'], 1500.0)
        self.assertAlmostEqual(first['joules_per_item'], 2000.0 / 800)
        self.assertFalse(first['short_window'])
        self.assertTrue(report['cases'][1]['short_window'])
        self.assertEqual(len(report['cases']), 2)

    def test_sparse_sensor_still_covers_the_window(self):
        part = telemetry.integrate([(0.0, 50.0), (10.0, 70.0)], 2.0, 8.0)
        self.assertAlmostEqual(part['joules'], 6 * 60.0)


class ProfileTests(unittest.TestCase):
    def csv_rows(self):
        header = ['OP CODE', 'OP TYPE', 'HOST START TS', 'DEVICE KERNEL DURATION [ns]', 'OP TO OP LATENCY [ns]',
                  'PM IDEAL [ns]', 'PM FPU UTIL (%)', 'DRAM BW UTIL (%)', 'CORE COUNT', 'MATH FIDELITY',
                  'INPUT_0_Y_PAD[LOGICAL]', 'INPUT_0_X_PAD[LOGICAL]']
        rows = [['warmup', 'tt_dnn_device', 0, 999, 0, '', '', '', 1, '', '', ''],
                ['portforge_timed_start', 'signpost', 1, '', '', '', '', '', '', '', '', ''],
                ['MatmulDeviceOperation', 'tt_dnn_device', 2, 8000, 1000, 4000, 40, 10, 110, 'HiFi4', '64[64]', '1024[1024]'],
                ['MatmulDeviceOperation', 'tt_dnn_device', 3, 8000, 1000, 4000, 40, 10, 110, 'HiFi4', '64[64]', '1024[1024]'],
                ['BinaryNgDeviceOperation', 'tt_dnn_device', 4, 2000, 2000, 2400, '', 60, 110, '', '64[64]', '1024[1024]'],
                ['portforge_timed_stop', 'signpost', 5, '', '', '', '', '', '', '', '', ''],
                ['after', 'tt_dnn_device', 6, 999, 0, '', '', '', 1, '', '', '']]
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
        return list(csv.DictReader(io.StringIO(stream.getvalue())))

    def test_summary_between_signposts(self):
        rows = opprofile.rows_between(self.csv_rows(), 'portforge_timed_start', 'portforge_timed_stop')
        summary = opprofile.summarize(rows)
        self.assertEqual(summary['ops'], 3)
        self.assertAlmostEqual(summary['device_ms'], .018)
        self.assertAlmostEqual(summary['gap_share'], 3000 / 21000)  # the window's first gap is pre-window time
        top = summary['top'][0]
        self.assertEqual((top['op'], top['count'], top['math_fidelity']), ('MatmulDeviceOperation', 2, 'HiFi4'))
        self.assertAlmostEqual(top['efficiency'], .5)
        self.assertEqual(summary['suspect_faster_than_ideal'], ['BinaryNgDeviceOperation'])

    def test_every_case_window_counts_and_fidelity_splits_groups(self):
        rows = self.csv_rows()
        later = [dict(row, **{'HOST START TS': str(10 + float(row['HOST START TS']))}) for row in rows[1:6]]
        for row in later:
            if row['MATH FIDELITY'] == 'HiFi4':
                row['MATH FIDELITY'] = 'LoFi'
        rows = opprofile.rows_between(rows + later, 'portforge_timed_start', 'portforge_timed_stop')
        self.assertEqual(len(rows), 6)
        groups = {g['key']: g['count'] for g in opprofile.summarize(rows)['top']}
        self.assertEqual(sorted(groups.values()), [2, 2, 2])
        self.assertTrue(any(key.endswith('LoFi') for key in groups))

    def test_trace_replay_rows_count_although_their_host_time_is_the_capture(self):
        rows = self.csv_rows()
        for row in rows[2:5]:  # replayed ops report the capture's (earlier) host timestamps
            row['HOST START TS'] = str(float(row['HOST START TS']) - 100)
        self.assertEqual(len(opprofile.rows_between(rows, 'portforge_timed_start', 'portforge_timed_stop')), 3)

    def test_missing_signposts_are_an_error(self):
        with self.assertRaises(ValueError):
            opprofile.rows_between(self.csv_rows(), 'nope', 'portforge_timed_stop')


class MergeabilityTests(unittest.TestCase):
    GOOD = {
        'models/experimental/m/tt/model.py': '# SPDX-License-Identifier: Apache-2.0\nimport ttnn\n',
        'models/experimental/m/tests/test_model.py': ('# SPDX-License-Identifier: Apache-2.0\n'
                                                      'def test_logits(device):\n    assert True\n'),
        'models/experimental/m/tests/test_e2e.py': ('# SPDX-License-Identifier: Apache-2.0\n'
                                                    'def test_e2e(device, model_location):\n    assert True\n'),
        'models/experimental/m/demo/demo.py': '# SPDX-License-Identifier: Apache-2.0\nprint(1)\n',
        'models/experimental/m/README.md': '| batch | seq | latency (ms) |\n|---|---|---|\n| 1 | 512 | 6.8 |\n',
        'backend.py': 'WEIGHTS = "/weights"\n',   # sandbox adapter outside the package is fine
    }

    def test_good_package_passes(self):
        verdict = merge_check.check(self.GOOD)
        self.assertTrue(verdict['passed'], verdict['issues'])
        self.assertEqual(verdict['package_root'], 'models/experimental/m')

    def test_reviewer_objections_are_reported(self):
        bad = dict(self.GOOD)
        bad['models/experimental/m/tests/test_model.py'] = ('# SPDX-License-Identifier: Apache-2.0\nimport backend\n'
                                                           'import ttnn\nd = ttnn.open_device(device_id=0)\n'
                                                           'def test_logits():\n    print(1)\n')
        bad['models/experimental/m/tt/model.py'] = 'path = "/weights/model.safetensors"\n'
        bad['models/experimental/m/README.md'] = '# Model\n'
        del bad['models/experimental/m/tests/test_e2e.py']
        issues = ' '.join(merge_check.check(bad)['issues'])
        for expected in ('never assert', 'sandbox-only', 'own device', 'device fixture', 'end-to-end',
                         'hard-coded sandbox path /weights', 'SPDX', 'performance table'):
            self.assertIn(expected, issues)
        self.assertFalse(merge_check.check({'backend.py': ''})['passed'])


class EvaluatorHelperTests(unittest.TestCase):
    def test_repeat_schedule_matches_range_without_perf_fields(self):
        spec = dict(timed_repeats=3)
        self.assertEqual(list(portforge_eval.repeat_schedule(spec, dict(numerical=True))), [0, 1, 2, 3])
        self.assertEqual(list(portforge_eval.repeat_schedule(spec, dict(numerical=False))), [0, 1])

    def test_perf_schedule_runs_for_minimum_time_and_stops_slow_cases(self):
        spec = dict(timed_repeats=2, min_timed_seconds=.05)
        start, count = time.perf_counter(), 0
        for _ in portforge_eval.repeat_schedule(spec, dict(numerical=True)):
            count += 1
            time.sleep(.002)  # a real forward takes milliseconds; MAX_TIMED_REPEATS bounds a no-op body
        self.assertGreaterEqual(time.perf_counter() - start, .05)
        self.assertGreater(count, 3)
        slow = dict(timed_repeats=100, min_timed_seconds=0, max_timed_seconds=.03, min_timed_repeats=2)
        seen = []
        for repeat in portforge_eval.repeat_schedule(slow, dict(numerical=True)):
            seen.append(repeat)
            time.sleep(.02)
        self.assertLess(len(seen), 10)
        self.assertTrue(portforge_eval.repeat_count_ok(slow, dict(numerical=True), len(seen) - 1))
        self.assertFalse(portforge_eval.repeat_count_ok(slow, dict(numerical=True), 1))
        self.assertFalse(portforge_eval.repeat_count_ok(dict(timed_repeats=3), dict(numerical=True), 4))

    def test_latency_summary_and_verdict_fields(self):
        summary = portforge_eval.latency_summary([i / 1000 for i in range(1, 101)])
        self.assertEqual((summary['p50_seconds'], summary['p99_seconds'], summary['max_seconds']), (.05, .099, .1))
        self.assertIsNone(summary['tail_note'])
        checks = [dict(name='sweep-b2', seconds=[.02, .02, .03], first_seconds=1.5, p50_seconds=.02),
                  dict(name='batch', seconds=[.01, .01], first_seconds=.5)]
        latency = portforge_eval.verdict_latency(checks)
        self.assertEqual((latency['max_case_p99_seconds'], latency['max_first_seconds']), (.03, 1.5))
        curve = portforge_eval.scaling_curve(checks, [dict(name='sweep-b2', batch=2), dict(name='batch')])
        self.assertEqual(curve, [dict(case='sweep-b2', batch=2, items=2, p50_seconds=.02, p99_seconds=.03,
                                      items_per_second=100.0)])
        with self.assertRaises(ValueError):
            portforge_eval.latency_summary([0.0])

    def test_timed_window_records_wall_clock(self):
        window = portforge_eval.TimedWindow(3)
        self.assertEqual(window.record(), dict(window=None, items=3))
        window.open()
        window.close()
        start, end = window.record()['window']
        self.assertLessEqual(start, end)

    def test_precision_groups(self):
        policy = dict(mode='bf16', weights='bf16', activations='bf16', accumulation='fp32', exceptions=['ff in bfp8'],
                      groups={'ff': {'weights': 'bfp8_b', 'math_fidelity': 'LoFi'},
                              'attention': {'math_fidelity': 'HiFi4', 'fp32_dest_acc': True}})
        checked = portforge_eval.validate_precision('bf16', policy)
        self.assertEqual(checked['effective']['groups']['ff']['weights'], 'bfp8_b')
        for bad in ({'ff': {'weights': 'fp8'}}, {'ff': {'math_fidelity': 'HiFi9'}}, {'ff': {'speed': 1}},
                    {'ff': {'fp32_dest_acc': 'yes'}}, {'': {'weights': 'bf16'}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                portforge_eval.validate_precision('bf16', dict(policy, groups=bad))
        with self.assertRaises(ValueError):  # a departing group needs an exception
            portforge_eval.validate_precision('bf16', dict(policy, exceptions=[]))
        plain = dict(mode='bfp8_b', weights='bfp8_b', activations='bfp8_b', accumulation='fp32', exceptions=[])
        self.assertIn('not IEEE', portforge_eval.validate_precision('bfp8_b', plain)['format_note'])
        for bfp4 in (dict(plain, mode='bfp4_b', weights='bfp4_b'), dict(policy, groups={'ff': {'weights': 'bfp4_b'}})):
            with self.subTest(deferred=bfp4), self.assertRaises(ValueError):  # BFP4 waits until BF16/BFP8 runs are proven
                portforge_eval.validate_precision(bfp4['mode'], bfp4)

    def test_device_options_use_open_device_keywords(self):
        module = type('backend', (), {})
        self.assertEqual(portforge_eval.device_options(module), {})
        module.DEVICE_OPTIONS = {'num_command_queues': 2, 'trace_region_size': 64 * 1024 ** 2}
        self.assertEqual(portforge_eval.device_options(module)['num_command_queues'], 2)
        for bad in ({'num_hw_cqs': 2}, {'num_command_queues': 3}, {'l1_small_size': 1.5}, ['trace_region_size']):
            module.DEVICE_OPTIONS = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'num_command_queues'):
                portforge_eval.device_options(module)

    def test_smallest_weight_matrix_reads_safetensors_header_only(self):
        header = {'__metadata__': {}, 'norm.weight': {'dtype': 'F32', 'shape': [8], 'data_offsets': [0, 32]},
                  'shared.weight': {'dtype': 'F32', 'shape': [2, 16], 'data_offsets': [32, 160]},
                  'proj.weight': {'dtype': 'F32', 'shape': [16, 16], 'data_offsets': [160, 1184]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            encoded = json.dumps(header).encode()
            path.write_bytes(len(encoded).to_bytes(8, 'little') + encoded)  # no tensor bytes: never read
            self.assertEqual(portforge_eval.smallest_weight_matrix(path, 'parakeet'), 32)


try:
    import numpy as np
    import torch
except ImportError:  # the controller environment may lack torch
    torch = None


@unittest.skipIf(torch is None, 'needs torch')
class HostComputeGuardTests(unittest.TestCase):
    def setUp(self):
        class Backend:
            pass
        self.backend = Backend()
        self.backend.weight = torch.randn(16, 16)
        self.backend.layers = [{'host': np.ones((16, 16), dtype=np.float32)}]
        self.backend.inv_freq = torch.rand(8)  # held, but smaller than any weight matrix
        self.guard = portforge_eval.HostComputeGuard(('aten.native_layer_norm.',), 32, self.backend)
        self.x = torch.randn(4, 16)

    def rejects(self, call):
        with self.assertRaisesRegex(RuntimeError, 'CPU learned compute forbidden'), self.guard():
            call()

    def test_weight_free_host_matmul_passes(self):
        with self.guard():
            positions = torch.arange(12).float()
            angles = positions[:, None] @ self.backend.inv_freq[None, :]
            mixed = self.x @ self.x.T
        self.assertEqual(tuple(angles.shape), (12, 8))
        self.assertEqual(tuple(mixed.shape), (4, 4))

    def test_matmuls_on_held_weights_fail(self):
        backend = self.backend
        self.rejects(lambda: self.x @ backend.weight)
        self.rejects(lambda: torch.nn.functional.linear(self.x, backend.weight))
        self.rejects(lambda: self.x @ backend.weight.T[:, :16])                        # a view
        self.rejects(lambda: self.x @ (backend.weight * 0.5))                           # derived in the call
        self.rejects(lambda: self.x @ torch.from_numpy(backend.layers[0]['host']))     # a held NumPy array
        def copied():
            buffer = torch.zeros(16, 16)
            buffer.copy_(backend.weight)  # an in-place write taints its target
            return self.x @ buffer
        self.rejects(copied)
        self.rejects(lambda: torch.nn.functional.layer_norm(self.x, (16,)))           # forbidden even without weights

    def test_refresh_sees_weights_built_after_construction(self):
        self.backend.late = torch.randn(16, 16)
        with self.guard():
            self.x @ self.backend.late
        self.guard.refresh()
        self.rejects(lambda: self.x @ self.backend.late)


class CatalogueTests(unittest.TestCase):
    def test_catalogue_is_consistent_with_its_guide(self):
        data = json.loads((support.HARNESS / 'guides/optimizations.json').read_text())
        ids = [t['id'] for t in data['techniques']]
        self.assertEqual(len(ids), len(set(ids)))
        guide = (support.HARNESS / 'guides/TT_OPTIMIZATIONS.md').read_text()
        for technique in data['techniques']:
            with self.subTest(id=technique['id']):
                self.assertIn(technique['category'], data['categories'])
                self.assertTrue(set(technique['interactions']) <= set(ids))
                self.assertIn('### `%s`' % technique['id'], guide)
                for field in ('title', 'mechanism', 'helps_when', 'risk', 'exemplar', 'source'):
                    self.assertTrue(technique[field])


if __name__ == '__main__':
    unittest.main()
