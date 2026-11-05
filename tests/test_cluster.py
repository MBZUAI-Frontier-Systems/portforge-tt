"""control/cluster.py: inventory parsing and the detached, polled remote call."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import support  # noqa: F401
from portforge_tt.control import cluster

CONFIG = {'host': {'host': '192.0.2.10', 'user': 'pf', 'key': '/keys/pf', 'work_dir': '/work/pf',
                   'python': '/usr/bin/python3', 'resources': {'host': 1, 'board': 1}}}


class InventoryTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cluster.yaml'
            path.write_text(text)
            return cluster.load(path)

    def test_host_role_and_default_resources(self):
        config = self.load('auth: {ssh_user: pf, ssh_private_key: /keys/pf}\n'
                           'available_node_types: {host: {compatible_ips: [192.0.2.10]}}\n'
                           'porting: {host: {work_dir: /work/pf, python: /usr/bin/python3}}\n')
        self.assertEqual(config['host']['resources'], {'host': 1, 'board': 1})
        self.assertEqual((config['host']['host'], config['host']['user']), ('192.0.2.10', 'pf'))

    def test_placeholders_and_relative_paths_are_rejected(self):
        for porting in ('{work_dir: <work>, python: /usr/bin/python3}', '{work_dir: pf, python: /usr/bin/python3}'):
            with self.subTest(porting=porting), self.assertRaises(ValueError):
                self.load('auth: {ssh_user: pf, ssh_private_key: /keys/pf}\n'
                          'available_node_types: {host: {compatible_ips: [192.0.2.10]}}\n'
                          'porting: {host: ' + porting + '}\n')


class RemoteTests(unittest.TestCase):
    def test_remote_detaches_then_polls_until_the_exit_marker(self):
        calls = []

        def ssh(endpoint, command, data=None, timeout=90):
            calls.append(command)
            if command.startswith('nohup'):
                return b''
            if len([c for c in calls if c.startswith('cat ')]) < 3:
                raise RuntimeError('not done')  # the .done file is missing while the job runs
            return b'0\n{"ok": true, "results": {}}\n@@@\n'

        with mock.patch.object(cluster, 'ssh', ssh), mock.patch.object(cluster, 'upload') as upload, \
                mock.patch.object(cluster.time, 'sleep'):
            result = cluster.remote(CONFIG, '/runs/r1', {'kind': 'gates', 'family': 'f'}, timeout=600)
        self.assertTrue(result['ok'])
        self.assertTrue(result['rpc'].startswith('runs/r1/rpc/gates-'))
        self.assertIn('-m portforge_tt.remote.runner gates', calls[0])
        self.assertIn('/work/pf/runs/r1/harness', calls[0])
        self.assertEqual(upload.call_args.args[1], '/work/pf/runs/r1/rpc')

    def test_a_crash_without_json_is_reported(self):
        def ssh(endpoint, command, data=None, timeout=90):
            return b'' if command.startswith('nohup') else b'1\nTraceback\n@@@\nboom\n'

        with mock.patch.object(cluster, 'ssh', ssh), mock.patch.object(cluster, 'upload'), \
                mock.patch.object(cluster.time, 'sleep'):
            result = cluster.remote(CONFIG, '/runs/r1', {'kind': 'bench'}, timeout=60)
        self.assertFalse(result['ok'])
        self.assertIn('boom', result['error'])


if __name__ == '__main__':
    unittest.main()
