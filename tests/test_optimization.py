"""ranking/: objective, search tree, frontier and the supervisor's masked technique menu."""
import json
import unittest

from run_fixture import RunFixture
from portforge_tt.ranking import objective, report, search


class ObjectiveTests(unittest.TestCase):
    def test_default_is_geomean_with_no_confirmed_regression(self):
        chosen = objective.spec()
        self.assertEqual(chosen['primary'], 'speedup_geomean')
        points = [{'id': 'a', 'metrics': {'speedup_geomean': 2.0, 'worst_cell': 1.1, 'flagged_cells': 0, 'text_bytes': 900}},
                  {'id': 'b', 'metrics': {'speedup_geomean': 2.6, 'worst_cell': .8, 'flagged_cells': 1, 'text_bytes': 700}},
                  {'id': 'c', 'metrics': {'speedup_geomean': 1.9, 'worst_cell': 1.2, 'flagged_cells': 0, 'text_bytes': 400}}]
        ranked = objective.rank(points, chosen)
        self.assertEqual(ranked['frontier'], ['a'])
        self.assertIn('b', ranked['infeasible'])
        self.assertEqual(sorted(ranked['pareto']), ['a', 'c'])

    def test_metrics_come_from_the_perf_receipt(self):
        result = {'verdict': {'perf': {'metrics': {'speedup_geomean': 3.0, 'worst_cell': 1.4},
                                       'flagged': ['x'], 'confirmed_flags': []}},
                  'profile': {'text_bytes': 512}}
        self.assertEqual(objective.metrics(result), {'speedup_geomean': 3.0, 'worst_cell': 1.4,
                                                     'flagged_cells': 0, 'text_bytes': 512})

    def test_bad_specs_are_refused(self):
        for bad in ({'primary': 'p50_latency'}, {'scenario': 'offline'}, {'pareto': []}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                objective.spec(bad)
        with self.assertRaises(ValueError):
            objective.parse_constraint('throughput>=1')


class SearchTests(RunFixture):
    def node_files(self, text, parent=None, applied=()):
        return {'kernel.S': text, 'TRANSFORMS.json': json.dumps({'parent': parent, 'applied': list(applied)})}

    def test_tree_frontier_and_masked_menu(self):
        from portforge_tt.control import jobs
        self.host(speedup=1.5)
        base = self.node_files('/* base */')
        self.run_job({'kind': 'evaluate', 'stage': 'perf'}, files=base)
        self.host(speedup=2.5)
        child = self.node_files('/* child */', parent=jobs.code_hash(base), applied=['max-lmul', 'k:width-dispatch'])
        self.run_job({'kind': 'evaluate', 'stage': 'perf'}, files=child)
        self.host(failed_at='G2')
        bad = self.node_files('/* bad */', parent=jobs.code_hash(child), applied=['max-lmul', 'k:width-dispatch',
                                                                                  'avoid-gather'])
        self.run_job({'kind': 'evaluate', 'stage': 'perf'}, files=bad)
        from portforge_tt.control.port import PORT
        known = search.known(PORT)
        nodes = search.build(self.run_dir, set(known))
        self.assertEqual([n['parent_source'] for n in nodes], [None, 'declared', 'declared'])
        fronts = search.frontier(nodes, objective.spec(), 'perf')
        self.assertEqual(fronts[0]['frontier'], [jobs.code_hash(child)])
        view = search.summary(nodes, objective.spec(), known, seed=1, beam=2)
        self.assertEqual(view['beam'][0]['parent'], jobs.code_hash(child))
        self.assertNotIn('max-lmul', [o['id'] for o in view['beam'][0]['offered']])
        self.assertEqual(view['rejected_in_context'][0]['technique'], 'avoid-gather')
        text = report.markdown(report.build([self.run_dir], objective.spec()))
        self.assertIn('2.500x', text)


if __name__ == '__main__':
    unittest.main()
