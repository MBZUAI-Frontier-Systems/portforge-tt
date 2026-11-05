"""Run creation: seeding, resuming and the frozen environment."""
import json
import io
import tarfile
import os
from pathlib import Path
import subprocess
import sys
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


class CampaignTests(RunFixture):
    def test_resume_rejects_other_model_submitted_or_non_fresh_sources(self):
        root = self.make_previous_run(model='other')
        with self.assertRaisesRegex(ValueError, 'different model'):
            campaign.resume_workspace(self.tools.workspace, root, 'parakeet', {'worker': 'claude', 'supervisor': 'claude'})
        root = self.make_previous_run('submitted')
        write_json(root / 'submission.json', {'code_sha256': 'other source'})
        with self.assertRaisesRegex(ValueError, 'differs from its frozen submission'):
            campaign.resume_workspace(self.tools.workspace, root, 'parakeet', {'worker': 'claude', 'supervisor': 'claude'})
        root = self.make_previous_run('private', seed_policy='private')
        with self.assertRaisesRegex(ValueError, 'fresh public-source'):
            campaign.resume_workspace(self.tools.workspace, root, 'parakeet', {'worker': 'claude', 'supervisor': 'claude'})

    def test_seed_job_uses_immutable_pass_instead_of_active_workspace(self):
        root = self.tools.run_dir / 'seed'
        write_json(root / 'manifest.json', {'seed_policy': 'fresh-public-parakeet', 'workspace': '/missing'})
        files = {'backend.py': '# fixed passing source', 'STATE.md': 'old pass'}
        write_json(root / 'jobs/passed/source.json', files)
        write_json(root / 'jobs/passed/request.json', {'kind': 'evaluate', 'stage': 'bringup',
                                                     'code_sha256': jobs.code_hash(files)})
        write_json(root / 'jobs/passed/result.json', {'returncode': 0, 'verdict': {'passed': True}})
        receipt = campaign.seed_workspace(self.tools.workspace, root, 'parakeet', 'passed')
        self.assertEqual(receipt['seed_job'], 'passed')
        self.assertEqual((self.tools.workspace / 'backend.py').read_text(), files['backend.py'])
        self.assertIn('no inherited pending jobs or passes', (self.tools.workspace / 'STATE.md').read_text())
        for changed in ({'returncode': 1, 'verdict': {'passed': True}},
                        {'returncode': 0, 'verdict': {'passed': False}},
                        {'returncode': 0, 'timeout': True, 'verdict': {'passed': True}}):
            write_json(root / 'jobs/passed/result.json', changed)
            with self.assertRaises(ValueError):
                campaign.seed_workspace(self.tools.workspace, root, 'parakeet', 'passed')

    def test_seed_job_tampering_and_traversal_rejected(self):
        root = self.tools.run_dir / 'seed'
        write_json(root / 'manifest.json', {'seed_policy': 'fresh-public-parakeet'})
        write_json(root / 'jobs/pass/source.json', {'backend.py': '# altered'})
        write_json(root / 'jobs/pass/request.json', {'kind': 'evaluate', 'stage': 'bringup', 'code_sha256': 'wrong'})
        write_json(root / 'jobs/pass/result.json', {'returncode': 0, 'verdict': {'passed': True}})
        for job in ('pass', '../escape', '/absolute', '..'):
            with self.subTest(job=job), self.assertRaises(ValueError):
                campaign.seed_workspace(self.tools.workspace, root, 'parakeet', job)

    def test_seed_preserves_code_but_replaces_instructions_and_does_not_copy_passes(self):
        root = self.tools.run_dir / 'seed'
        workspace = root / 'worker'
        workspace.mkdir(parents=True)
        (workspace / 'backend.py').write_text('# independently written previous candidate')
        (workspace / 'TASK.md').write_text('old target')
        (workspace / 'STATE.md').write_text('old job 123 claimed success; user forbids optimization')
        (workspace / 'TT_NOTES.md').write_text('user forbids optimization')
        write_json(root / 'manifest.json', {'workspace': str(workspace),
                   'seed_policy': 'fresh-public-parakeet'})
        write_json(root / 'jobs/123/result.json', {'verdict': {'passed': True}})
        (self.tools.workspace / 'TASK.md').write_text('trusted new target')
        provenance = campaign.seed_workspace(self.tools.workspace, root, 'parakeet')
        self.assertEqual((self.tools.workspace / 'backend.py').read_text(),
                         '# independently written previous candidate')
        self.assertEqual((self.tools.workspace / 'TASK.md').read_text(), 'trusted new target')
        self.assertIn('historical', (self.tools.workspace / 'STATE.md').read_text())
        self.assertNotIn('forbids optimization', (self.tools.workspace / 'STATE.md').read_text())
        self.assertFalse((self.tools.workspace / 'TT_NOTES.md').exists())
        self.assertEqual(provenance['omitted_seed_notes'], ['STATE.md', 'TT_NOTES.md'])
        self.assertEqual(provenance['seed_run'], str(root.resolve()))
        self.assertFalse((self.tools.workspace / 'jobs').exists())
        with self.assertRaises(ValueError):
            self.tools.submit(model='parakeet')
        (workspace / 'escape.py').symlink_to('/tmp/outside')
        with self.assertRaises(ValueError):
            campaign.seed_workspace(self.tools.workspace, root, 'parakeet')

    def test_seed_submission_hash_is_verified(self):
        root = self.tools.run_dir / 'seed'
        write_json(root / 'manifest.json', {'seed_policy': 'fresh-public-parakeet'})
        files = {'backend.py': '# frozen candidate'}
        write_json(root / 'submission.json', {'files': files, 'code_sha256': jobs.code_hash(files)})
        campaign.seed_workspace(self.tools.workspace, root, 'parakeet')
        self.assertEqual((self.tools.workspace / 'backend.py').read_text(), files['backend.py'])
        write_json(root / 'submission.json', {'files': files, 'code_sha256': 'tampered'})
        with self.assertRaises(ValueError):
            campaign.seed_workspace(self.tools.workspace, root, 'parakeet')

    def test_cross_model_and_unknown_model_seed_still_drop_baseline_evidence(self):
        root = self.tools.run_dir / 'seed'
        files = {'backend.py': '# source', 'BASELINE_BF16.md': 'old baseline',
                 'BF16_BASELINE_X.md': 'old baseline', 'CURRENT_BF16_SOURCE_1.md': 'old source',
                 'CURRENT_BF16_IDENTITY.md': 'old runtime', 'CANDIDATE_X.md': 'old candidate',
                 'BFP8_X.md': 'old diagnostic', 'RESEARCH.md': 'public research', 'REPORT.md': 'old report'}
        write_json(root / 'manifest.json', {'seed_policy': 'fresh-public-parakeet', 'model': 'parakeet'})
        evidence = ('BASELINE_BF16.md', 'BF16_BASELINE_X.md', 'CURRENT_BF16_SOURCE_1.md',
                    'CURRENT_BF16_IDENTITY.md', 'CANDIDATE_X.md', 'BFP8_X.md')
        for artifact_model, expected in (('parakeet', sorted(list(evidence) + ['RESEARCH.md', 'REPORT.md'])),
                                         ('other-model', ['REPORT.md', 'RESEARCH.md'])):
            frozen = {'files': files, 'code_sha256': jobs.code_hash(files), 'model': artifact_model}
            write_json(root / 'submission.json', frozen)
            for name in evidence:
                (self.tools.workspace / name).unlink(missing_ok=True)
            receipt = campaign.seed_workspace(self.tools.workspace, root, 'parakeet')
            self.assertEqual(receipt['retained_seed_notes'], expected)
            for name in evidence:
                self.assertEqual((self.tools.workspace / name).exists(), artifact_model == 'parakeet')

    def test_resume_continues_a_submitted_source_that_still_matches(self):
        root = self.make_previous_run('frozen')
        source = jobs.snapshot(root / 'workspace')
        write_json(root / 'submission.json', {'files': source, 'code_sha256': jobs.code_hash(source)})
        campaign.resume_workspace(self.tools.workspace, root, 'parakeet', {'worker': 'claude', 'supervisor': 'claude'})
        self.assertEqual((self.tools.workspace / 'backend.py').read_text(), '# carried implementation')
        self.assertFalse((self.tools.run_dir / 'submission.json').exists())

    def test_same_model_seed_preserves_baseline_dependency_but_not_control_notes_or_passes(self):
        root = self.tools.run_dir / 'seed'
        source = root / 'workspace'
        source.mkdir(parents=True)
        write_json(root / 'manifest.json', {'seed_policy': 'fresh-public-parakeet',
                   'model': 'parakeet', 'workspace': str(source)})
        evidence = ('BASELINE_BF16_3_3B.md', 'BF16_BASELINE_RESULTS.md',
                    'CURRENT_BF16_BASELINE.md', 'CURRENT_BF16_SOURCE_1.md',
                    'CURRENT_BF16_IDENTITY.md', 'CANDIDATE_REVIEW.md', 'BFP8_RESULTS.md')
        controls = ('STATE.md', 'REVIEW_REQUIRED.md', 'HASH_SCOPES.md', 'TT_NOTES.md')
        for name in evidence:
            (source / name).write_text('```python\noriginal_baseline = True\n```\n')
        for name in controls:
            (source / name).write_text('old pass; stale user forbids optimization directive')
        for name in ('TASK.md', 'CONTRACT.md', 'TT_GUIDE.md'):
            (source / name).write_text('superseded instructions')
            (self.tools.workspace / name).write_text('current instructions')
        (source / 'backend.py').write_text('# preserved implementation')
        test = 'from pathlib import Path\nbaseline = Path(__file__).with_name("BASELINE_BF16_3_3B.md").read_text()\n'
        (source / 'test_cross_kv.py').write_text(test)
        write_json(root / 'jobs/old/result.json', {'verdict': {'passed': True}})
        receipt = campaign.seed_workspace(self.tools.workspace, root, 'parakeet')
        for name in evidence:
            self.assertEqual((self.tools.workspace / name).read_text(), (source / name).read_text())
        namespace = {'__file__': str(self.tools.workspace / 'test_cross_kv.py')}
        exec(compile(test, namespace['__file__'], 'exec'), namespace)
        self.assertIn('original_baseline = True', namespace['baseline'])
        for name in controls[1:]:
            self.assertFalse((self.tools.workspace / name).exists())
        for name in ('TASK.md', 'CONTRACT.md', 'TT_GUIDE.md'):
            self.assertEqual((self.tools.workspace / name).read_text(), 'current instructions')
        state = (self.tools.workspace / 'STATE.md').read_text()
        for required in ('no inherited pending jobs or passes', 'historical under the archived runtime',
                         'revalidate the source and its same-precision baseline',
                         'advance optimization and standalone deliverables', 'Do not repeat resolved original-runtime'):
            self.assertIn(required, state)
        self.assertNotIn('forbids optimization', state)
        self.assertEqual(receipt['retained_seed_notes'], sorted(evidence))
        self.assertEqual(receipt['omitted_seed_notes'], sorted(controls))
        self.assertEqual(self.tools.status(), {})
        self.assertFalse((self.tools.run_dir / 'baselines').exists())
        self.assertFalse((self.tools.run_dir / 'submission.json').exists())
        with self.assertRaisesRegex(ValueError, 'needs an independent bringup pass'):
            self.tools.submit(model='parakeet')

    def test_resume_carries_full_workspace_and_state_without_passes(self):
        root = self.make_previous_run(history_runs=['/archive/older'])
        write_json(root / 'jobs/old/request.json', {'kind': 'evaluate', 'stage': 'bringup'})
        write_json(root / 'jobs/old/result.json', {'verdict': {'passed': True}})
        receipt = campaign.resume_workspace(self.tools.workspace, root, 'parakeet',
                                        {'worker': 'claude', 'supervisor': 'opencode'}, 'Probe v7 compares x@W to W.T.')
        work = self.tools.workspace
        for name in ('RESEARCH.md', 'TT_NOTES.md', 'backend.py', 'baselines/smoke.json'):
            self.assertEqual((work / name).read_text(), (root / 'workspace' / name).read_text())
        self.assertFalse((work / '.opencode').exists())
        self.assertEqual((work / 'TASK.md').read_text(), 'current instructions')
        state = (work / 'STATE.md').read_text()
        self.assertTrue(state.startswith('# Resumed campaign'))
        for required in ('continues previous', 'opencode/opencode -> claude/opencode',
                         'no pass, pending job or baseline acceptance is inherited',
                         '## Operator note\n\nProbe v7 compares x@W to W.T.',
                         '# STATE\nnext: read probe v7 e244a077'):
            self.assertIn(required, state)
        self.assertEqual(receipt['history_runs'], [str(root.resolve()), '/archive/older'])
        self.assertEqual(receipt['resumed_from'], str(root.resolve()))
        self.assertEqual(self.tools.status(), {})
        with self.assertRaisesRegex(ValueError, 'needs an independent bringup pass'):
            self.tools.submit(model='parakeet')

    def test_frozen_environment_uses_archive_and_checks_docs_and_registry(self):
        run = self.tools.run_dir
        (run / 'harness/TASK.md').write_text('frozen instruction')
        manifest = {'harness_sha256': port.harness_identity(run / 'harness'),
                    'repository_root': '/workspace/original-repository'}
        env = port.frozen_environment(run, manifest)
        self.assertEqual(env['PYTHONPATH'], str(run.resolve() / 'harness'))
        self.assertEqual(env['PORTFORGE_REPOSITORY_ROOT'], '/workspace/original-repository')
        for name in ('TASK.md', 'models.json'):
            path = run / 'harness' / name
            original = path.read_text()
            path.write_text('changed')
            with self.assertRaises(ValueError):
                port.frozen_environment(run, manifest)
            path.write_text(original)

    def test_agent_environment_defaults_older_manifests_to_opencode(self):
        self.assertEqual(port.agent_environment({}), {'PORTFORGE_WORKER_BACKEND': 'opencode',
                                                      'PORTFORGE_SUPERVISOR_BACKEND': 'opencode'})
        manifest = {'agent_backends': {'worker': 'claude', 'supervisor': 'opencode'},
                    'agent_models': {'worker': 'claude-opus-5-5', 'supervisor': 'zai-coding-plan/glm-5.3'}}
        self.assertEqual(port.agent_environment(manifest), {
            'PORTFORGE_WORKER_BACKEND': 'claude', 'PORTFORGE_WORKER_MODEL': 'claude-opus-5-5',
            'PORTFORGE_SUPERVISOR_BACKEND': 'opencode', 'PORTFORGE_SUPERVISOR_MODEL': 'zai-coding-plan/glm-5.3'})

    def test_references_are_reused_only_from_the_same_evaluator(self):
        earlier = self.tools.run_dir.parent / 'earlier'
        (earlier / 'harness').mkdir(parents=True)
        self.assertFalse(port.same_evaluator(earlier))  # no archived evaluator
        (earlier / 'harness/evaluate.py').write_bytes((support.PACK / 'evaluate.py').read_bytes())
        self.assertTrue(port.same_evaluator(earlier))
        (earlier / 'harness/evaluate.py').write_text('# older oracle')
        self.assertFalse(port.same_evaluator(earlier))


if __name__ == "__main__":
    unittest.main()


class PackSeedTests(unittest.TestCase):
    def test_pack_seed_source_is_copied_into_a_fresh_workspace(self):
        pack = support.ROOT / "ports" / "chronos2"
        with tempfile.TemporaryDirectory() as workspace, patch.dict(port.PORT, {"dir": pack}):
            provenance = campaign.seed_pack_source(workspace)
            copied = jobs.snapshot(workspace)
        self.assertEqual(copied, jobs.snapshot(pack / "seed"))
        self.assertIn("backend.py", copied)
        self.assertIn("models/experimental/chronos_forecast/tt/model.py", copied)
        self.assertEqual(provenance, {"pack_seed_sha256": jobs.code_hash(copied)})

    def test_packs_without_a_seed_start_empty(self):
        with tempfile.TemporaryDirectory() as workspace, patch.dict(port.PORT, {"dir": support.ROOT / "ports" / "esm2"}):
            self.assertEqual(campaign.seed_pack_source(workspace), {})
            self.assertEqual(list(Path(workspace).iterdir()), [])


class PackageModeTests(unittest.TestCase):
    def test_launcher_checks_every_pack_suite_in_package_mode(self):
        # The launcher runs as portforge_tt.control.*, with neither remote/ nor the pack on sys.path.
        repo = Path(__file__).resolve().parents[1]
        script = ('import json, sys; from pathlib import Path\n'
                  'from portforge_tt.control import campaign, port\n'
                  'from portforge_tt.remote.portforge_models import use_registry\n'
                  'for pack in sys.argv[1:]:\n'
                  '    sys.modules.pop("evaluate", None); port.configure(Path(pack))\n'
                  '    use_registry(Path(pack) / "models.json")\n'
                  '    corpus = json.loads((Path(pack) / "corpus.json").read_text())\n'
                  '    campaign.check_suites(corpus, port.PORT["key"])\n')
        packs = [str(p) for p in sorted((repo / 'ports').iterdir()) if (p / 'corpus.json').is_file()]
        done = subprocess.run([sys.executable, '-c', script, *packs], cwd=str(repo), capture_output=True, text=True,
                              env={**os.environ, 'PYTHONPATH': str(repo)})
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
