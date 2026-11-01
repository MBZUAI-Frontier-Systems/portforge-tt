"""The agents' tools: one MCP server per campaign with port_files/read/write/run/evaluate/profile/status/log/submit.

``ToolServer`` serves FastMCP over streamable HTTP from a local uvicorn thread (heavy imports are
lazy). ``PortTools`` scopes every tool to one workspace and one run archive; each TT action becomes
a job (jobs.py), and agents never see the oracle or another run's private files.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import threading
import time
import uuid

if __package__:
    from . import cluster, runtime
    from .cluster import identity as digest
    from .jobs import (code_hash, experiment, history_runs, passing_evaluation, reference_progress, snapshot,
                       suite_identity, without_raw_samples)
    from .port import PORT, gated_stages, trusted_documents
    from ..remote.runner import PROFILE_CASE, PROFILE_STAGES
    from ..remote.portforge_models import model_spec
    from ..remote.runner import safe_file, validate_log_request, write_json
else:  # flat: the frozen run harness
    import cluster, runtime
    from cluster import identity as digest
    from jobs import (code_hash, experiment, history_runs, passing_evaluation, reference_progress, snapshot,
                      suite_identity, without_raw_samples)
    from port import PORT, gated_stages, trusted_documents
    from runner import PROFILE_CASE, PROFILE_STAGES
    from portforge_models import model_spec
    from runner import safe_file, validate_log_request, write_json


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



class PortTools(ToolServer):
    def setup(self, workspace, run_dir, model=None):
        model = model or PORT['key']
        model_spec(model)
        self.model = model
        self.workspace, self.run_dir = Path(workspace), Path(run_dir)
        self.refs = {}
        for name in ('files', 'read', 'write', 'run', 'evaluate', 'profile', 'status', 'log', 'submit'):
            self.mcp.add_tool(getattr(self, name), name='port_' + name)

    def files(self) -> list:
        return list(snapshot(self.workspace))

    def _job_owner(self, job_id):
        """Run archive holding job_id: this run first, then earlier runs it continues.

        Earlier runs' full-corpus verdicts are operator-only: the held-out corpus is
        reported to agents through the operator note, never as per-case results.
        """
        for root in [self.run_dir] + history_runs(self.run_dir):
            request = root / 'jobs' / job_id / 'request.json'
            if request.is_file():
                if root != self.run_dir and json.loads(request.read_text()).get('stage') == 'full':
                    raise ValueError('historical full-corpus verdicts are operator-only')
                return root
        return self.run_dir

    def read(self, path: str) -> str:
        return safe_file(self.workspace, path).read_text()[:24000]

    def write(self, path: str, content: str) -> dict:
        if Path(path).as_posix() in trusted_documents() or (self.run_dir / 'submission.json').exists():
            raise ValueError('instructions or submitted source are read-only')
        if len(content.encode()) > 2_000_000:
            raise ValueError('file too large')
        target = safe_file(self.workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return {'saved': path}

    def _start_job(self, request):
        files = snapshot(self.workspace)
        stage = request.get('stage')
        config = cluster.activated(self.run_dir)
        request = {**request, 'model': request.get('model', getattr(self, 'model', PORT['key'])),
                   'cluster_sha256': digest(config)}
        input_stage = stage or request.get('public_input_stage')
        if input_stage:
            request['suite_definition_sha256'] = suite_identity(self.run_dir, request.get('model', PORT['key']), input_stage)
        for path in (self.run_dir / 'jobs').glob('*/request.json'):
            old = json.loads(path.read_text())
            # Same kind too: a profile must never be answered with an evaluate receipt.
            if (stage and old.get('stage') == stage and old['code_sha256'] == code_hash(files)
                    and old.get('kind', 'evaluate') == request.get('kind', 'evaluate')
                    and old.get('suite_definition_sha256') == request.get('suite_definition_sha256')
                    and old.get('model', PORT['key']) == request.get('model', PORT['key'])
                    and old.get('precision') == request.get('precision')
                    and old.get('cluster_sha256') == digest(config)):
                result = path.with_name('result.json')
                if not result.exists() or json.loads(result.read_text()).get('verdict', {}).get('passed'):
                    return {'job': path.parent.name, 'reused': True}
        job = uuid.uuid4().hex
        write_json(self.run_dir / 'jobs' / job / 'source.json', files)
        write_json(self.run_dir / 'jobs' / job / 'request.json', {**request, 'code_sha256': code_hash(files)})
        # A mesh job holds every card it leases; 'mesh' first (sorted order) so two never split the cards.
        submit = (runtime.job({'mesh': 1, 'tt': PORT['mesh_cards']})(experiment._original).submit
                  if request.get('stage') == 'mesh' or request.get('mesh_cards') else experiment.submit)
        self.refs[job] = submit(str(self.run_dir), job, {**request, 'files': files}, config)
        return {'job': job, 'status': 'running'}

    def run(self, command: str, timeout_seconds: int = 120, public_input_stage: str = '', model: str = '',
            mesh: bool = False) -> dict:
        """Async isolated TT shell. /opt/tt-metal public source; /weights pinned weights.
        model selects a registered checkpoint; empty uses the campaign's model.
        Optional public_input_stage=smoke/bringup mounts prepared public workload files
        for that model read-only at /input; no oracle or evaluation pass is exposed.
        mesh=True (packs with a mesh stage) exposes all mesh cards, leased together, so
        ttnn.open_mesh_device(ttnn.MeshShape(1, N)) works; it waits until every card is free.
        Only write() persists source edits. No network, old ports, or credentials.
        A timeout quarantines its card(s), so use short focused commands, never sleep.
        """
        if public_input_stage not in ('', 'smoke', 'bringup'):
            raise ValueError('public_input_stage must be empty, smoke or bringup')
        if type(mesh) is not bool or (mesh and not PORT.get('mesh_cards')):
            raise ValueError('mesh must be a boolean, and True only for packs with a mesh stage')
        model = model or getattr(self, 'model', PORT['key'])
        model_spec(model)
        return self._start_job({'kind': 'run', 'command': command, 'timeout': min(timeout_seconds, 300),
                                'public_input_stage': public_input_stage, 'model': model,
                                **({'mesh_cards': PORT['mesh_cards']} if mesh else {})})

    def evaluate(self, stage: str = 'smoke', precision: str = 'bf16', model: str = '') -> dict:
        """Independent CUDA-reference check: smoke (tiny), bringup (short behavior suite) or perf
        (bringup cases + batch sweep, many repeats: latency percentiles, throughput, energy).
        Full corpus is reserved for frozen submit(). Call status to collect results.
        """
        model = model or getattr(self, 'model', PORT['key'])
        model_spec(model)
        if precision not in PORT['precisions']:
            raise ValueError('unsupported precision')
        if stage not in gated_stages():
            raise ValueError('use ' + '/'.join(gated_stages()) + '; full is controller-only after submission')
        if not (self.workspace / 'backend.py').exists():
            raise ValueError('backend.py required')
        return self._start_job({'kind': 'evaluate', 'stage': stage, 'precision': precision, 'model': model})

    def profile(self, stage: str = 'smoke', precision: str = 'bf16', model: str = '', case: str = '') -> dict:
        """Device-profile one candidate pass (Tracy) over public smoke/bringup inputs, or over one perf
        case (stage="perf", case="sweep-b1024") with a few repeats, to see large-batch behaviour.
        Returns, via status, a per-op table between the timed signposts: device time, op-to-op
        gap share, TT performance-model efficiency, FPU/DRAM utilization. No gate, no oracle.
        """
        model = model or getattr(self, 'model', PORT['key'])
        model_spec(model)
        if precision not in PORT['precisions']:
            raise ValueError('unsupported precision')
        if stage not in PROFILE_STAGES or (stage == 'perf') != bool(case):
            raise ValueError('profile uses smoke or bringup inputs, or stage="perf" with one case')
        if case and not PROFILE_CASE.fullmatch(case):
            raise ValueError('invalid case name')
        if not (self.workspace / 'backend.py').exists():
            raise ValueError('backend.py required')
        return self._start_job({'kind': 'profile', 'stage': stage, 'precision': precision, 'model': model,
                                **({'case': case} if case else {})})

    def status(self, compact: bool = False, job_id: str = '') -> dict:
        """Collect recent results and every pending job, including older background studies.
        compact=True bounds stdout/stderr to 1000 characters each and adds request
        provenance and retrieval hints. Structured results/gates remain complete.
        job_id retrieves one own historical request/result regardless of age, with
        complete request identity and compact log tails. This exact lookup is local
        and read-only, including pending jobs; it never reruns or refreshes a job.
        It also finds jobs of earlier runs this campaign resumed (marked historical_run);
        those are receipts only and never count as passes here.
        """
        if type(compact) is not bool:
            raise ValueError('compact must be a boolean')
        if not isinstance(job_id, str):
            raise ValueError('job_id must be a string')
        results = {}
        if job_id:
            if (len(job_id) > 80 or job_id[0] not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in job_id)):
                raise ValueError('invalid job ID')
            owner = self._job_owner(job_id)
            job = owner / 'jobs' / job_id
            path = job / 'request.json'
            result_file = job / 'result.json'
            if (any(p.is_symlink() for p in (job.parent, job, path, result_file))
                    or not path.is_file() or (result_file.exists() and not result_file.is_file())):
                raise ValueError('unknown or unsafe job')
            visible = [path]
        else:
            paths = sorted((self.run_dir / 'jobs').glob('*/request.json'), key=lambda p: p.stat().st_mtime)
            visible = [p for p in paths[:-12] if not p.with_name('result.json').exists()] + paths[-12:]
        for path in visible:
            result_file = path.with_name('result.json')
            request = json.loads(path.read_text())
            if result_file.exists():
                result = json.loads(result_file.read_text())
                if not job_id and request.get('stage') == 'bringup' and result.get('verdict', {}).get('passed'):
                    baseline = (self.run_dir / 'baselines' / request.get('model', PORT['key'])
                                / request['precision'] / request['cluster_sha256'] / (result['verdict']['suite_sha256'] + '.json'))
                    if not baseline.exists():
                        write_json(baseline, {'job': path.parent.name, 'code_sha256': request['code_sha256'],
                                             'precision': request.get('precision'), 'cluster_sha256': request.get('cluster_sha256'),
                                             'model': request.get('model', PORT['key']), 'suite_sha256': result['verdict']['suite_sha256'],
                                             'suite_definition_sha256': request['suite_definition_sha256']})
                results[path.parent.name] = without_raw_samples(result)
            elif job_id:
                results[path.parent.name] = {'status': 'pending', 'progress': {'phase': 'saved_request_without_result'}}
            else:
                try:
                    config = json.loads((self.run_dir / 'clusters' / (request['cluster_sha256'] + '.json')).read_text())
                    progress = (reference_progress(config, path.parent) if request.get('kind') == 'reference-study' else
                                cluster.remote(config, str(self.run_dir), {'kind': 'status', 'job': path.parent.name}, timeout=10))
                except Exception:
                    progress = {'phase': 'reference_or_queue'}
                results[path.parent.name] = {'status': 'pending', 'progress': progress}
            if compact or job_id:
                item = dict(results[path.parent.name])
                if job_id and owner != self.run_dir:
                    item['historical_run'] = owner.name
                item['request_identity'] = {key: request.get(key) for key in
                    ('kind', 'stage', 'model', 'precision', 'code_sha256', 'cluster_sha256',
                     'suite_definition_sha256')}
                if job_id:
                    item['request_identity'] = request
                tails = {}
                for stream in ('stdout', 'stderr'):
                    if isinstance(item.get(stream), str):
                        text = item[stream]
                        item[stream] = text[-1000:]
                        tails[stream] = {'available_chars': len(text), 'returned_chars': len(item[stream]),
                                         'truncated': len(text) > 1000}
                item['log_tails'] = tails
                item['details'] = ({'tool': 'port_log', 'job_id': path.parent.name,
                                    'streams': ['stdout', 'stderr'],
                                    'note': 'Saved result excerpts may already be truncated; log reads original byte pages.'}
                                   if request.get('kind') in ('run', 'evaluate', 'profile') else
                                   {'tool': 'port_status', 'compact': False, 'job_id': path.parent.name,
                                    'note': 'Reference-study details use status; TT log access is unavailable.'})
                results[path.parent.name] = item
        return results

    def log(self, job_id: str, stream: str = 'stdout', offset_bytes: int = 0, limit_bytes: int = 65536) -> dict:
        """Read existing TT job stdout/stderr without rerunning it. No oracle access.
        Byte pages use UTF-8 replacement decoding (a split character can be replaced).
        Use next_offset_bytes to continue; eof and size_bytes describe this read's snapshot.
        """
        validate_log_request(job_id, stream, offset_bytes, limit_bytes)
        owner = self._job_owner(job_id)
        job = owner / 'jobs' / job_id
        path = job / 'request.json'
        if any(p.is_symlink() for p in (job.parent, job, path)) or not path.is_file():
            raise ValueError('unknown or unsafe job')
        request = json.loads(path.read_text())
        if request.get('kind') not in ('run', 'evaluate', 'profile'):
            raise ValueError('only own TT run/evaluate/profile job logs are exposed')
        key = request.get('cluster_sha256', '')
        if not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('missing archived cluster identity')
        archive = owner / 'clusters' / (key + '.json')
        if archive.parent.is_symlink() or archive.is_symlink() or not archive.is_file():
            raise ValueError('missing or unsafe archived cluster')
        config = json.loads(archive.read_text())
        if digest(config) != key:
            raise ValueError('archived cluster identity mismatch')
        return cluster.remote(config, str(owner), {'kind': 'log', 'job': job_id, 'stream': stream,
                      'offset_bytes': offset_bytes, 'limit_bytes': limit_bytes}, timeout=15)

    def submit(self, precision: str = 'bf16', model: str = '') -> dict:
        """Freeze source after bringup passes. Controller alone runs the full acceptance corpus."""
        model = model or getattr(self, 'model', PORT['key'])
        model_spec(model)
        if precision not in PORT['precisions']:
            raise ValueError('unsupported precision')
        files = snapshot(self.workspace)
        config = cluster.activated(self.run_dir)
        source, signature = code_hash(files), digest(config)
        definition = suite_identity(self.run_dir, model, 'bringup')
        requests = sorted((self.run_dir / 'jobs').glob('*/request.json'),
                          key=lambda path: path.stat().st_mtime, reverse=True)
        for path in requests:
            req = json.loads(path.read_text())
            result = path.with_name('result.json')
            if (req.get('kind') == 'evaluate' and req.get('stage') == 'bringup'
                    and req.get('code_sha256') == source
                    and req.get('suite_definition_sha256') == definition
                    and req.get('model', PORT['key']) == model and req.get('precision') == precision
                    and req.get('cluster_sha256') == signature):
                if result.exists() and passing_evaluation(json.loads(result.read_text()), path.parent.name,
                                                         signature, model, precision, 'bringup'):
                    write_json(self.run_dir / 'submission.json', {'files': files, 'code_sha256': source, 'precision': precision,
                        'cluster_sha256': signature, 'model': model, 'suite_definition_sha256': definition})
                    return {'status': 'frozen_for_full_evaluation'}
                # A later matching failure, pending job or incomplete receipt must
                # not be bypassed by an older pass for the same source/configuration.
                break
        raise ValueError('current source needs an independent bringup pass before submission')
