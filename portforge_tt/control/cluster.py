"""Cluster inventory and pinned SSH jobs; no SSH aliases or cloud creation.

The inventory parser reads the small subset of the (CHIA-compatible) cluster
YAML schema PortForge actually uses: ``auth`` (global and per-IP SSH
credentials), ``available_node_types`` (one host per role with its resource
counts) and the ``porting`` section, with ``${VAR}`` environment expansion
in values. Everything else in the file is ignored.

Kernel campaigns have one remote role, ``host``: the x86 machine with the
toolchain image, QEMU and the verifiers, which also reaches the X280 board
(its address and boot recipe live in the host's own ``cpu.yaml``, see
remote/config.py). Resources: ``host`` (concurrent trusted gate jobs) and
``board`` (concurrent board sessions, normally 1).
"""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time
import uuid

import yaml

from ..remote.runner import safe_file, write_json

POLL_SECONDS = 10


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _expand_env_vars(data):
    """Recursively expand ${VAR} references in values, leaving $VAR as-is."""
    if isinstance(data, str):
        return re.sub(r'\$\{([^}]+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), data)
    if isinstance(data, dict):
        return {_expand_env_vars(k): _expand_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    return data


def load_raw_config(path):
    """Load the inventory YAML and expand environment variables in values."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"cluster inventory not found: {path}")
    with open(path) as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("cluster inventory must be a YAML mapping")
    return _expand_env_vars(raw)


def _ssh_auth(raw, host):
    """Resolve SSH credentials for *host*: per-IP override, else the globals."""
    auth = raw.get('auth') or {}
    override = (auth.get('overrides') or {}).get(host) or {}
    user = override.get('ssh_user', auth.get('ssh_user'))
    key = override.get('ssh_private_key', auth.get('ssh_private_key'))
    proxy = override.get('ssh_proxy_command', auth.get('ssh_proxy_command'))
    if not user:
        raise ValueError(f"no SSH user configured for {host}")
    return user, key, proxy


def load(path):
    raw = load_raw_config(str(path))
    node_types = raw.get('available_node_types') or {}
    endpoints = {}
    for role in ('host',):
        options = dict((raw.get('porting') or {}).get(role) or {})
        if not options:
            raise ValueError(f"porting.{role} section missing from the inventory")
        node = node_types.get(role) or {}
        hosts = node.get('compatible_ips') or []
        if len(hosts) != 1:
            raise ValueError('select exactly one host per role; edit compatible_ips to replace it')
        host = hosts[0]
        user, key, proxy = _ssh_auth(raw, host)
        if not re.fullmatch(r'[A-Za-z0-9_.:-]+', host) or host.startswith('-'):
            raise ValueError('invalid host')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', user) or user.startswith('-'):
            raise ValueError('invalid SSH user')
        if not key:
            raise ValueError('explicit SSH key path required')
        for key_name in ('work_dir', 'python'):
            value = options.get(key_name)
            if not isinstance(value, str) or not Path(value).is_absolute() or '\n' in value or '<' in value:
                raise ValueError('runtime paths must be absolute: porting.%s.%s' % (role, key_name))
        resources = dict(node.get('resources') or {})
        resources.setdefault('host', 1)
        resources.setdefault('board', 1)
        endpoints[role] = dict(options, host=host, user=user,
            key=str(Path(key).expanduser()), proxy=proxy, resources=resources)
    return endpoints


def ssh(endpoint, command, data=None, timeout=90):
    args = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
            '-i', endpoint['key'], '-l', endpoint['user']]
    for key, option in (('known_hosts', 'UserKnownHostsFile'), ('host_key_alias', 'HostKeyAlias'), ('proxy', 'ProxyCommand')):
        if endpoint.get(key):
            value = str(Path(endpoint[key]).expanduser()) if key == 'known_hosts' else endpoint[key]
            args += ['-o', option + '=' + value]
    args += ['--', endpoint['host'], command]
    result = subprocess.run(args, input=data, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-1500:])
    return result.stdout


def upload(endpoint, directory, files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, data in files.items():
            safe_file('/tmp/portforge-upload', name)
            data = data.encode() if isinstance(data, str) else data
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o600
            archive.addfile(info, io.BytesIO(data))
    ssh(endpoint, 'mkdir -p ' + shlex.quote(directory) + ' && tar -xf - -C ' + shlex.quote(directory), stream.getvalue(), 300)


def activated(run):
    return json.loads((Path(run) / 'cluster.json').read_text())


def run_root(config, run):
    """The run's directory on the host, relative to its work_dir (gate paths are work_dir-relative)."""
    return 'runs/' + Path(run).name


def _retry(fn, attempts=6, delay=15):
    """Run fn, retrying transient ssh failures with backoff."""
    for i in range(attempts):
        try:
            return fn()
        except (RuntimeError, subprocess.TimeoutExpired):
            if i == attempts - 1:
                raise
            time.sleep(delay * (i + 1))


def remote(config, run, request, timeout=90):
    """Run one runner command on the host and return its JSON result.

    The request is ``{'kind': <command>, ...arguments}``. The command runs detached on the host and
    is polled with short ssh calls: a dropped connection (seen mid-campaign during a 20-minute gate
    batch) never kills or loses it. A command still running at ``timeout`` returns ok=False.
    """
    endpoint = config['host']
    work, root = endpoint['work_dir'], run_root(config, run)
    name = request['kind'] + '-' + uuid.uuid4().hex[:12]
    rpc = root + '/rpc/' + name
    arguments = {k: v for k, v in request.items() if k != 'kind'}
    _retry(lambda: upload(endpoint, work + '/' + root + '/rpc', {name + '.json': json.dumps(arguments)}))
    base, q = work + '/' + rpc, shlex.quote
    job = ('cd ' + q(work + '/' + root + '/harness') + ' && PYTHONPATH=. ' + q(endpoint['python'])
           + ' -m portforge_tt.remote.runner ' + q(request['kind']) + ' ' + q(base + '.json')
           + ' > ' + q(base + '.out') + ' 2> ' + q(base + '.err') + '; echo $? > ' + q(base + '.done'))
    _retry(lambda: ssh(endpoint, 'nohup setsid sh -c ' + q(job) + ' > /dev/null 2>&1 < /dev/null &', timeout=60))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(min(POLL_SECONDS, max(1, timeout / 20)))
        try:
            out = ssh(endpoint, 'cat ' + q(base + '.done') + ' 2>/dev/null && tail -n 3 ' + q(base + '.out')
                      + ' && echo @@@ && tail -c 1500 ' + q(base + '.err'), timeout=120).decode(errors='replace')
        except (RuntimeError, subprocess.TimeoutExpired):
            continue  # not finished yet (cat fails), or a transient ssh failure
        if not out.strip():
            continue
        head, _, err = out.partition('@@@')
        lines = [line for line in head.splitlines() if line.startswith('{')]
        if not lines:
            return {'ok': False, 'error': 'no result (exit %s): %s' % (head.splitlines()[0], err[-1500:]),
                    'rpc': rpc}
        return {**json.loads(lines[-1]), 'rpc': rpc}
    return {'ok': False, 'error': 'remote %s still running after %ss (%s)' % (request['kind'], timeout, rpc),
            'rpc': rpc}
