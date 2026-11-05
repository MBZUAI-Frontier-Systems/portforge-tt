"""A frozen run, a workspace and a fake x86 host: what the loop tests share. No ssh, no agents."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import support
from portforge_tt.control import cluster, jobs, port
from portforge_tt.control.cluster import identity

CONFIG = {'host': {'host': '192.0.2.10', 'user': 'pf', 'key': '/keys/pf', 'proxy': None,
                   'work_dir': '/work/portforge-cpu', 'python': '/usr/bin/python3',
                   'resources': {'host': 1, 'board': 1}}}
KERNEL = '#include "src/riscv/asm.S"\nfunction splat_mv_rvv, export=1, ext=v\n  ret\nendfunc\n'


def gate_result(name, passed=True, failed_at=None):
    gates = {'G0': {'passed': True}, 'G1': {'passed': True}, 'G2': {'passed': True}}
    if failed_at:
        gates[failed_at] = {'passed': False, 'detail': 'mismatch'}
    return {'ok': True, 'rpc': 'runs/r/rpc/gates-0123456789ab',
            'results': {name: {**gates, 'profile': {'text_bytes': 640, 'loops': []}}}}


def bench_result(name, speedup=2.0, flagged=()):
    return {'ok': True, 'rpc': 'runs/r/rpc/bench-0123456789ab', 'receipt': {'seed': 1},
            'verdicts': {name: {'metrics': {'speedup_geomean': speedup, 'worst_cell': speedup * .9,
                                            'cycles_per_elem': 1.5},
                                'flagged': list(flagged), 'wrong_cells': [], 'feasible': not flagged,
                                'cells': {'a@1': {'speedup': {'s': speedup}, 'flagged': bool(flagged)}}}}}


class FakeHost:
    """Answers cluster.remote like the host's runner would, and records every request."""

    def __init__(self, speedup=2.0, failed_at=None, flagged=(), heldout=True, verify=('ALL_PASSED',)):
        self.speedup, self.failed_at, self.flagged = speedup, failed_at, list(flagged)
        self.heldout, self.verify, self.requests = heldout, verify, []

    def __call__(self, config, run, request, timeout=90):
        self.requests.append(request)
        kind = request['kind']
        if kind == 'gates':
            return gate_result(next(iter(request['kernels'])), failed_at=self.failed_at)
        if kind == 'bench':
            return bench_result(request['kernels'][0], self.speedup, self.flagged)
        if kind == 'certify':
            return {'ok': True, 'heldout': {'passed': self.heldout},
                    'verify': [{'method': 'cbmc', 'verdict': v, 'passed': v == 'ALL_PASSED'} for v in self.verify]}
        if kind in ('dev_check', 'dev_compile'):
            return {'ok': True, 'rpc': 'runs/r/rpc/' + kind + '-0123456789ab', 'compiled': True,
                    'check': {'passed': True}}
        if kind == 'context':
            return {'ok': True, 'sources': {'reference_refmvs.c': '/* C */', 'riscv_asm.S': '/* asm */'}}
        if kind == 'health':
            return {'ok': True, 'host': {'ok': True}, 'board': {'reachable': True}}
        raise AssertionError('unexpected request ' + kind)


class RunFixture(unittest.TestCase):
    """runs/<run>/ with a frozen harness, manifest and activated cluster, plus a workspace."""

    board_minutes = 60

    def setUp(self):
        support.configure_port()
        self.tmp = Path(tempfile.mkdtemp(prefix='portforge-test-'))
        self.run_dir = self.tmp / 'runs' / 'refmvs-splat_mv-test'
        self.run_dir.mkdir(parents=True)
        self.workspace = self.tmp / 'workspace'
        self.workspace.mkdir()
        port.freeze_harness(self.run_dir / 'harness')
        for name, text in port.seed_documents(port.PORT, port.PACKAGE / 'guides', {}).items():
            (self.workspace / name).write_text(text)
        (self.workspace / 'kernel.S').write_text(KERNEL)
        (self.workspace / 'context').mkdir()
        (self.workspace / 'context' / 'reference.c').write_text('/* C */')
        self.manifest = {'workspace': str(self.workspace), 'board_minutes': self.board_minutes,
                         'heldout_seed': 1234, 'optimization': None, 'beam': None, 'seeds': {},
                         'placeholder_sha256': None}
        (self.run_dir / 'manifest.json').write_text(json.dumps(self.manifest))
        for path in (self.run_dir / 'cluster.json', self.run_dir / 'clusters' / (identity(CONFIG) + '.json')):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(CONFIG))
        self.uploads = mock.patch.object(cluster, 'upload').start()
        self.addCleanup(mock.patch.stopall)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def host(self, **kwargs):
        fake = FakeHost(**kwargs)
        mock.patch.object(cluster, 'remote', fake).start()
        return fake

    def files(self, text=KERNEL):
        return {'kernel.S': text, 'TRANSFORMS.json': json.dumps({'parent': None, 'applied': ['max-lmul']})}

    def run_job(self, request, files=None, job=None):
        """Run one experiment in-process (no leases) and write its request like the tools do."""
        files = files or self.files()
        job = job or 'job%04d' % len(list((self.run_dir / 'jobs').glob('*')))
        request = {**request, 'cluster_sha256': identity(CONFIG), 'suite_definition_sha256':
                   jobs.suite_identity(self.run_dir)}
        (self.run_dir / 'jobs' / job).mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'jobs' / job / 'source.json').write_text(json.dumps(files))
        (self.run_dir / 'jobs' / job / 'request.json').write_text(json.dumps({**request, 'code_sha256': jobs.code_hash(files)}))
        return job, jobs.experiment._original(str(self.run_dir), job, {**request, 'files': files}, CONFIG)
