"""The agents' tools: one MCP server per campaign with port_files/read/write/run/evaluate/profile/status/log/submit.

``ToolServer`` serves FastMCP over streamable HTTP from a local uvicorn thread (heavy imports are
lazy). ``PortTools`` scopes every tool to one workspace and one run archive; each host action becomes
a job (jobs.py), and agents never see the held-out seed, the board or another run's private files.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import threading
import time
import uuid

from . import cluster, runtime
from .cluster import identity as digest
from .jobs import (board_seconds, candidate, code_hash, experiment, history_runs, passing_evaluation, snapshot,
                   suite_identity)
from .port import PORT, gated_stages, is_trusted
from ..remote.runner import safe_file, validate_log_request, write_json


_taken_ports: set[int] = set()


class ToolServer:
    """Base class for MCP tool servers served from the controller process."""

    def __init__(self, name: str, *, host: str = "127.0.0.1",
                 base_port: int = 8000, max_tries: int = 100,
                 logging_level: int = logging.DEBUG):
        from mcp.server.fastmcp import FastMCP
        from mcp.server.transport_security import TransportSecuritySettings

        self.name = name
        self.hostname = host  # advertised to agent CLIs; agents are local too
        self.base_port, self.max_tries = base_port, max_tries
        # stateless HTTP: every request carries its own session, which is all
        # the scoped agent tools need (same settings the CHIA base used).
        self.mcp = FastMCP(
            name,
            stateless_http=True,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=False,
            ),
        )
        self.port = None  # set by start()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging_level)
        self._server = None
        self._thread = None

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def setup(self, *args, **kwargs):
        """Register tools with ``self.mcp.add_tool(...)``; override this."""
        raise NotImplementedError(
            f"{type(self).__name__} defines neither __init__ nor setup(); "
            "implement one of them."
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Opt-in convenience, mirroring the CHIA base: a subclass that defines
        # setup() but no __init__ is constructed as
        # ChiaTool.__init__ -> setup(*args, **kwargs) -> start().
        if "setup" in cls.__dict__ and "__init__" not in cls.__dict__:
            def _auto_init(self, name, *args, base_port=8000, max_tries=100,
                           logging_level=logging.DEBUG, **kwargs):
                ToolServer.__init__(self, name, host="127.0.0.1", base_port=base_port,
                                    max_tries=max_tries, logging_level=logging_level)
                self.setup(*args, **kwargs)
                self.start()

            _auto_init.__name__ = "__init__"
            _auto_init.__qualname__ = f"{cls.__qualname__}.__init__"
            cls.__init__ = _auto_init

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Serve the FastMCP app on a free local port; returns the port."""
        import uvicorn
        from contextlib import AsyncExitStack, asynccontextmanager
        from fastapi import FastAPI

        mcp = self.mcp

        @asynccontextmanager
        async def lifespan(app):
            # Runs every mounted session manager concurrently.
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(mcp.session_manager.run())
                yield

        for port in range(self.base_port, self.base_port + self.max_tries):
            if port in _taken_ports:
                continue
            # A session manager can only .run() once per instance, so create a
            # fresh app per attempt (identical to the CHIA start_router).
            mcp._session_manager = None
            app = FastAPI(lifespan=lifespan)
            app.mount(f"/{self.name}", mcp.streamable_http_app())
            config = uvicorn.Config(app, host=self.hostname, port=port, log_level="warning")
            server = uvicorn.Server(config)
            thread = threading.Thread(target=server.run, daemon=True,
                                      name=f"portforge-mcp-{self.name}")
            thread.start()
            for _ in range(50):  # wait up to five seconds for a bind
                time.sleep(0.1)
                if server.started:
                    _taken_ports.add(port)
                    self.port, self._server, self._thread = port, server, thread
                    self.logger.info("%s serving at %s", self.name, self.url)
                    return port
            server.should_exit = True
            thread.join(timeout=2)
        raise RuntimeError(
            f"no available port in range {self.base_port}-{self.base_port + self.max_tries - 1}"
        )

    @property
    def url(self) -> str:
        if self.port is None:
            raise RuntimeError(f"tool server {self.name!r} is not started")
        return f"http://{self.hostname}:{self.port}/{self.name}/mcp"

    def stop(self) -> None:
        """Stop the HTTP server thread (registered tools stay defined)."""
        server, thread, port = self._server, self._thread, self.port
        self._server = self._thread = None
        self.port = None
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                self.logger.warning("MCP server thread for %s still alive after 5s", self.name)
        if port is not None:
            _taken_ports.discard(port)


def tool_name(run):
    """Use a stable bare TOML key; run names may contain dots or other punctuation."""
    return 'portforge_' + hashlib.sha256(str(Path(run).resolve()).encode()).hexdigest()[:16]





def _job_id(job_id):
    if (not isinstance(job_id, str) or not job_id or len(job_id) > 80
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in job_id)
            or job_id[0] in '.-_'):
        raise ValueError('invalid job ID')
    return job_id


class PortTools(ToolServer):
    def setup(self, workspace, run_dir):
        self.workspace, self.run_dir = Path(workspace), Path(run_dir)
        self.refs = {}
        for name in ('files', 'read', 'write', 'run', 'evaluate', 'profile', 'status', 'log', 'submit'):
            self.mcp.add_tool(getattr(self, name), name='port_' + name)

    def files(self) -> list:
        """Every file in your workspace: instructions, SPEC.md, TECHNIQUES.md, context/ and your own files."""
        return sorted(str(p.relative_to(self.workspace)) for p in self.workspace.rglob('*')
                      if p.is_file() and not any(part.startswith('.') for part in p.relative_to(self.workspace).parts))

    def _job_owner(self, job_id):
        """Run archive holding job_id: this run first, then earlier runs it continues.

        Earlier runs' certification verdicts are operator-only: the held-out regimes are
        reported to agents through the operator note, never as per-case results.
        """
        for root in [self.run_dir] + history_runs(self.run_dir):
            request = root / 'jobs' / job_id / 'request.json'
            if request.is_file():
                if root != self.run_dir and json.loads(request.read_text()).get('kind') == 'certify':
                    raise ValueError('historical certification verdicts are operator-only')
                return root
        return self.run_dir

    def read(self, path: str, offset: int = 0) -> str:
        """Read a workspace file: 60000 characters from character `offset` (large context files need
        several calls; the reply says where the next chunk starts)."""
        if type(offset) is not int or offset < 0:
            raise ValueError('offset must be a nonnegative integer')
        text = safe_file(self.workspace, path).read_text()
        chunk = text[offset:offset + 60000]
        more = len(text) - offset - len(chunk)
        return chunk + (f'\n[... {more} more characters: read again with offset={offset + len(chunk)}]' if more > 0 else '')

    def write(self, path: str, content: str) -> dict:
        """Write a workspace file: your kernel source, TRANSFORMS.json, NOTES.md, STATE.md or other notes.
        Instructions (CONTRACT, TASK, SPEC, TECHNIQUES, the guides) and context/ are read-only."""
        if is_trusted(path) or (self.run_dir / 'submission.json').exists():
            raise ValueError('instructions, context/ and submitted source are read-only')
        if len(content.encode()) > 2_000_000:
            raise ValueError('file too large')
        target = safe_file(self.workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return {'saved': path}

    def _start_job(self, request):
        files = snapshot(self.workspace)
        if PORT['source_name'] not in files:
            raise ValueError(PORT['source_name'] + ' is required in the workspace')
        stage = request.get('stage')
        config = cluster.activated(self.run_dir)
        request = {**request, 'cluster_sha256': digest(config), 'family': PORT['family']}
        if stage:
            request['suite_definition_sha256'] = suite_identity(self.run_dir)
        code = code_hash(files)
        for path in (self.run_dir / 'jobs').glob('*/request.json'):
            old = json.loads(path.read_text())
            # Same kind too: a quick run must never be answered with an evaluation receipt.
            if (stage and old.get('stage') == stage and old.get('code_sha256') == code
                    and old.get('kind', 'evaluate') == request.get('kind', 'evaluate')
                    and old.get('suite_definition_sha256') == request.get('suite_definition_sha256')
                    and old.get('cluster_sha256') == digest(config)):
                result = path.with_name('result.json')
                if not result.exists() or (json.loads(result.read_text()).get('verdict') or {}).get('passed'):
                    return {'job': path.parent.name, 'reused': True}
        job = uuid.uuid4().hex
        write_json(self.run_dir / 'jobs' / job / 'source.json', candidate(files))
        write_json(self.run_dir / 'jobs' / job / 'request.json', {**request, 'code_sha256': code})
        self.refs[job] = submit(request)(str(self.run_dir), job, {**request, 'files': candidate(files),
                                                                 'seed': _seed(job)}, config)
        return {'job': job, 'status': 'running'}

    def run(self, command: str = 'check') -> dict:
        """Untrusted quick check of the current source on the x86 host. command="compile": assemble/compile
        as the project's build would and return errors or the static profile with predicted X280 cycles
        per element; command="check": build the project with your kernel and run its own test under QEMU
        at VLEN 128 and 1024 (a few minutes). Results via port_status; never a gate or a pass.
        """
        if command not in ('compile', 'check'):
            raise ValueError('command must be "compile" or "check"')
        return self._start_job({'kind': 'run', 'command': command})

    def evaluate(self, stage: str = 'check') -> dict:
        """Trusted gates on the current source. stage="check": G0 compile + static profile, G1 the project's
        tests under QEMU at VLEN 128/256/512/1024, G2 the bit-exact check over the regime matrix.
        stage="perf": the same gates, then an X280 bench session over the matrix against the baseline
        (speedup per cell, geomean; a flagged cell is re-measured in a second session). Only perf
        receipts rank candidates. Call port_status for results.
        """
        if stage not in gated_stages():
            raise ValueError('use ' + '/'.join(gated_stages()) + '; certification is controller-only after port_submit')
        if stage == 'perf':
            budget = json.loads((self.run_dir / 'manifest.json').read_text()).get('board_minutes', 0) * 60
            if board_seconds(self.run_dir) >= budget:
                raise ValueError('the campaign board budget is spent; perf evaluations are closed')
        return self._start_job({'kind': 'evaluate', 'stage': stage})

    def profile(self) -> dict:
        """Static profile of the current source (no gate): per hot loop the instruction mix, vsetvl count,
        spills and predicted X280 cycles per element from the board calibration, plus text size and calls.
        Results via port_status."""
        return self._start_job({'kind': 'profile'})

    def status(self, compact: bool = False, job_id: str = '') -> dict:
        """Collect recent results and every pending job. compact=True bounds long text fields and adds
        request provenance. job_id retrieves one own request/result regardless of age (also from earlier
        runs this campaign resumed, marked historical_run; those are receipts only, never passes here).
        """
        if type(compact) is not bool:
            raise ValueError('compact must be a boolean')
        results = {}
        if job_id:
            owner = self._job_owner(_job_id(job_id))
            path = owner / 'jobs' / job_id / 'request.json'
            if any(p.is_symlink() for p in (path.parent.parent, path.parent, path)) or not path.is_file():
                raise ValueError('unknown or unsafe job')
            visible = [path]
        else:
            owner = self.run_dir
            paths = sorted((self.run_dir / 'jobs').glob('*/request.json'), key=lambda p: p.stat().st_mtime)
            visible = [p for p in paths[:-12] if not p.with_name('result.json').exists()] + paths[-12:]
        for path in visible:
            request = json.loads(path.read_text())
            result_file = path.with_name('result.json')
            if result_file.exists():
                item = json.loads(result_file.read_text())
            elif path.parent.name in self.refs and not self.refs[path.parent.name].done():
                item = {'status': 'pending', 'progress': 'queued_or_running: waiting for the host (and the board, '
                                                        'for perf) or running; it will run, never resubmit it'}
            else:
                item = {'status': 'pending', 'progress': 'saved_request_without_result'}
            if compact or job_id:
                item = _bounded(item, 4000 if job_id else 1500)
                item['request_identity'] = {key: request.get(key) for key in
                    ('kind', 'stage', 'command', 'code_sha256', 'cluster_sha256', 'suite_definition_sha256')}
                if owner != self.run_dir:
                    item['historical_run'] = owner.name
                if item.get('rpc'):
                    item['details'] = {'tool': 'port_log', 'job_id': path.parent.name, 'streams': ['stdout', 'stderr']}
            results[path.parent.name] = item
        return results

    def log(self, job_id: str, stream: str = 'stdout', offset_bytes: int = 0, limit_bytes: int = 65536) -> dict:
        """Read an own job's host stdout/stderr without rerunning it, in byte pages (UTF-8 replacement
        decoding). Use next_offset_bytes to continue."""
        owner = self._job_owner(_job_id(job_id))
        result_file = owner / 'jobs' / job_id / 'result.json'
        if not result_file.is_file():
            raise ValueError('no result yet; logs exist once the job finishes')
        result = json.loads(result_file.read_text())
        rpc = result.get('rpc')
        validate_log_request(rpc, stream, offset_bytes, limit_bytes)
        config = json.loads((owner / 'clusters' / (result['cluster_sha256'] + '.json')).read_text())
        return cluster.remote(config, str(owner), {'kind': 'log', 'rpc': rpc, 'stream': stream,
                                                   'offset_bytes': offset_bytes, 'limit_bytes': limit_bytes},
                              timeout=60)

    def submit(self) -> dict:
        """Freeze the current source for certification once its latest perf evaluation passed (check when
        the campaign has no board budget). The controller then runs held-out regimes, a second bench
        session and the formal verification; the workspace becomes read-only."""
        files = snapshot(self.workspace)
        config = cluster.activated(self.run_dir)
        source, signature = code_hash(files), digest(config)
        definition = suite_identity(self.run_dir)
        board = json.loads((self.run_dir / 'manifest.json').read_text()).get('board_minutes', 0) > 0
        stage = 'perf' if board else 'check'
        requests = sorted((self.run_dir / 'jobs').glob('*/request.json'),
                          key=lambda path: path.stat().st_mtime, reverse=True)
        for path in requests:
            req = json.loads(path.read_text())
            result = path.with_name('result.json')
            if (req.get('kind') == 'evaluate' and req.get('stage') == stage and req.get('code_sha256') == source
                    and req.get('suite_definition_sha256') == definition and req.get('cluster_sha256') == signature):
                receipt = json.loads(result.read_text()) if result.exists() else None
                if passing_evaluation(receipt, path.parent.name, signature, stage):
                    write_json(self.run_dir / 'submission.json', {
                        'files': candidate(files), 'code_sha256': source, 'cluster_sha256': signature,
                        'suite_definition_sha256': definition, 'job': path.parent.name,
                        'metrics': ((receipt['verdict'].get('perf') or {}).get('metrics') or {})})
                    return {'status': 'frozen_for_certification'}
                # A later matching failure, pending job or incomplete receipt must
                # not be bypassed by an older pass for the same source/configuration.
                break
        raise ValueError(f'current source needs an independent {stage} pass before submission')


def submit(request):
    """The job's resource leases: perf and certification hold the board too; quick runs use their own slot."""
    if request.get('kind') in ('run', 'profile'):
        return runtime.job({'dev': 1})(experiment._original).submit
    if request.get('stage') in ('perf', 'full'):
        return runtime.job({'board': 1, 'host': 1})(experiment._original).submit
    return experiment.submit


def _seed(job_id):
    return int(hashlib.sha256(job_id.encode()).hexdigest()[:8], 16) % 2_000_000_000


def _bounded(value, limit):
    """Shorten long strings inside a result so one tool reply stays readable."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit // 2] + '\n[...]\n' + value[-limit // 2:]
    if isinstance(value, dict):
        return {k: _bounded(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_bounded(v, limit) for v in value[:40]] + ([f'[... {len(value) - 40} more]'] if len(value) > 40 else [])
    return value
