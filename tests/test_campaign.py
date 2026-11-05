"""control/campaign.py: a launch builds the workspace, freezes the harness and hands over to the controller."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import support
from portforge_tt.control import campaign, port

CLUSTER = """
cluster_name: test
auth: {ssh_user: pf, ssh_private_key: /keys/pf}
available_node_types:
  host: {compatible_ips: [192.0.2.10], resources: {host: 1, board: 1}}
porting:
  host: {work_dir: /work/portforge-cpu, python: /usr/bin/python3}
"""


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='portforge-campaign-'))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(support.configure_port)
        (self.tmp / 'cluster.yaml').write_text(CLUSTER)
        self.execve = mock.patch('os.execve').start()
        mock.patch('os.chdir').start()
        self.addCleanup(mock.patch.stopall)

    def start(self, run, **options):
        campaign.start(support.PACK, self.tmp / 'cluster.yaml', family=support.FAMILY, detach=False,
                       run_dir=self.tmp / run, **options)
        manifest = json.loads((self.tmp / run / 'manifest.json').read_text())
        self.addCleanup(shutil.rmtree, manifest['workspace'], True)
        return manifest, Path(manifest['workspace'])

    def test_placeholder_campaign_seeds_docs_techniques_and_freezes(self):
        manifest, workspace = self.start('run1')
        placeholder = (support.PACK / 'kernels' / support.FAMILY / 'placeholder.S').read_text()
        self.assertEqual((workspace / 'kernel.S').read_text(), placeholder)
        self.assertTrue(manifest['placeholder_sha256'])
        for name in ('CONTRACT.md', 'TASK.md', 'SPEC.md', 'TECHNIQUES.md', 'RVV_TECHNIQUES.md', 'RVV_GUIDE.md',
                     'LESSONS.md', 'TRANSFORMS.json', 'STATE.md'):
            self.assertTrue((workspace / name).is_file(), name)
        self.assertIn('k:width-dispatch', (workspace / 'TECHNIQUES.md').read_text())  # porters see the menu
        self.assertEqual(port.harness_identity(self.tmp / 'run1' / 'harness'), manifest['harness_sha256'])
        self.assertIsInstance(manifest['heldout_seed'], int)
        self.assertNotIn(str(manifest['heldout_seed']), ''.join(p.read_text() for p in workspace.iterdir()))
        command = self.execve.call_args.args[1]
        self.assertEqual(command[1:4], ['-m', 'portforge_tt.control.controller', '--controller'])
        self.assertEqual(self.execve.call_args.args[2]['PYTHONPATH'], str((self.tmp / 'run1' / 'harness').resolve()))

    def test_seed_kernels_are_baselines_and_the_first_starts_the_workspace(self):
        a, b = self.tmp / 'a.S', self.tmp / 'b.S'
        a.write_text('/* a */')
        b.write_text('/* b */')
        manifest, workspace = self.start('run2', seed_kernel=[f'up={a}', f'mine={b}'], board_minutes=0)
        self.assertEqual((workspace / 'kernel.S').read_text(), '/* a */')
        self.assertEqual(sorted(manifest['seeds']), ['mine', 'up'])
        self.assertIsNone(manifest['placeholder_sha256'])
        self.assertEqual(json.loads((self.tmp / 'run2' / 'seeds' / 'mine.json').read_text())['kernel.S'], '/* b */')

    def test_resume_carries_the_workspace_and_note(self):
        first, workspace = self.start('run3')
        (workspace / 'kernel.S').write_text('/* progress */')
        (workspace / 'STATE.md').write_text('step 4 of the port')
        manifest, resumed = self.start('run4', resume=self.tmp / 'run3', resume_note='try k:width-dispatch')
        self.assertEqual((resumed / 'kernel.S').read_text(), '/* progress */')
        state = (resumed / 'STATE.md').read_text()
        self.assertIn('try k:width-dispatch', state)
        self.assertIn('step 4 of the port', state)
        self.assertEqual(manifest['history_runs'], [str((self.tmp / 'run3').resolve())])
        self.assertEqual(manifest['placeholder_sha256'], first['placeholder_sha256'])

    def test_bad_selections_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'unknown family'):
            campaign.start(support.PACK, self.tmp / 'cluster.yaml', family='nope', detach=False, run_dir=self.tmp / 'x')
        with self.assertRaisesRegex(ValueError, 'tt branch'):
            campaign.start(support.ROOT / 'ports' / 'esm2', self.tmp / 'cluster.yaml', family='x', detach=False)
        with self.assertRaisesRegex(ValueError, 'hours'):
            campaign.start(support.PACK, self.tmp / 'cluster.yaml', family=support.FAMILY, hours=30, detach=False)


if __name__ == '__main__':
    unittest.main()
