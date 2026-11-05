"""The SiFive X280 cores on a Blackhole card, running Linux via tt-bh-linux.

The board is reached over ssh from the x86 host (tt-bh-linux forwards a local
port to the X280's sshd). It is a single shared machine, so every run takes a
file lock on the x86 host for its duration.
"""
from contextlib import contextmanager
import fcntl
import re
import shlex
import socket
import subprocess

from .config import TOOLS

SSH_OPTS = ["-F", "/dev/null", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15"]


def _ssh(cfg):
    x = cfg.x280
    return ["ssh", *SSH_OPTS, "-p", str(x.port), f"{x.user}@{x.host}"]


def run(cfg, command, timeout=600, check=True):
    proc = subprocess.run(_ssh(cfg) + [command], text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"x280: {command!r} failed ({proc.returncode}): {proc.stderr[-1500:]}")
    return proc


def push(cfg, files, remote_dir=None):
    """Copy local files into remote_dir (default cfg.x280.remote_dir); returns remote paths."""
    x = cfg.x280
    remote_dir = remote_dir or x.remote_dir
    run(cfg, f"mkdir -p {shlex.quote(remote_dir)}")
    files = [str(f) for f in files]
    subprocess.run(["scp", "-q", *SSH_OPTS, "-P", str(x.port), *files, f"{x.user}@{x.host}:{remote_dir}/"],
                   check=True, capture_output=True, text=True)
    return [f"{remote_dir}/{f.rsplit('/', 1)[-1]}" for f in files]


@contextmanager
def lease(cfg):
    """Exclusive use of the board for timing; other runs wait."""
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    with open(cfg.workdir / ".x280.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


IDLE_SCRIPT = ("cat /proc/loadavg; grep '^cpu[0-9]' /proc/stat; sleep 1; grep '^cpu[0-9]' /proc/stat")


def parse_idle(text, max_busy=0.03):
    """{ok, loadavg, busy: {cpu: fraction}} from IDLE_SCRIPT output.

    Only per-hart busy time over the sampled second decides: the load average
    lags by minutes, so right after our own session it reads "busy" for nothing.
    """
    lines = text.strip().splitlines()
    load = float(lines[0].split()[0])
    stats = [l.split() for l in lines[1:]]
    half = len(stats) // 2
    busy = {}
    for before, after in zip(stats[:half], stats[half:]):
        b, a = list(map(int, before[1:])), list(map(int, after[1:]))
        total = sum(a) - sum(b)
        idle = (a[3] + a[4]) - (b[3] + b[4])  # idle + iowait
        busy[before[0]] = round((total - idle) / total, 4) if total else 0.0
    return {"ok": all(v <= max_busy for v in busy.values()), "loadavg": load, "busy": busy}


def idle(cfg, max_busy=0.03):
    """Is the shared board quiet enough to time on? Other agents use it too, so
    timing never starts (or counts) while anything else is running."""
    proc = run(cfg, IDLE_SCRIPT, timeout=60, check=False)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr[-300:]}
    return parse_idle(proc.stdout, max_busy)


def board_id(cfg):
    """A stable identity for receipts: uarch, hart count and kernel release."""
    proc = run(cfg, "grep -m1 uarch /proc/cpuinfo; nproc; uname -r", timeout=60, check=False)
    return " ".join(proc.stdout.split()[2:]) if proc.returncode == 0 else "unknown"


def port_open(cfg):
    try:
        with socket.create_connection((cfg.x280.host, cfg.x280.port), timeout=3):
            return True
    except OSError:
        return False


PROBE_C = TOOLS / "x280_probe.c"


def probe(cfg):
    """ISA, VLEN, hart count, clock and counter access of the running board."""
    if not port_open(cfg):
        return {"ok": False, "error": f"nothing listening on {cfg.x280.host}:{cfg.x280.port}; "
                                      f"boot with `portforge-cpu x280 boot`"}
    try:
        remote = push(cfg, [PROBE_C])[0]
        exe = remote[:-2]
        out = run(cfg, f"uname -r; grep -m1 '^isa' /proc/cpuinfo; grep -m1 uarch /proc/cpuinfo; nproc; "
                       f"gcc -dumpfullversion; gcc -O2 -march=rv64gcv -o {exe} {remote} && {exe}",
                  timeout=120).stdout.splitlines()
    except (RuntimeError, subprocess.SubprocessError) as error:
        return {"ok": False, "error": str(error)[-500:]}
    info = {"kernel": out[0], "isa": out[1].split(":", 1)[1].strip(), "uarch": out[2].split(":", 1)[1].strip(),
            "harts": int(out[3]), "gcc": out[4]}
    for line in out[5:]:
        if "=" in line:
            key, value = line.split("=", 1)
            info[key] = float(value) if re.fullmatch(r"-?[\d.]+", value) else value
    info["ok"] = "v" in info["isa"].split("_")[0][4:] and info.get("vlen_bits", 0) > 0
    return info


def boot_command(cfg):
    x = cfg.x280
    python = f" PYTHON={shlex.quote(x.boot_python)}" if x.boot_python else ""
    return f"cd {shlex.quote(x.boot_dir)} && make boot TTDEVICE={x.ttdevice} L2CPU={x.l2cpu}{python}"


def boot(cfg, session="tt-bh-linux"):
    """Start (reset and boot) the X280 Linux in a detached tmux session.

    This resets the Blackhole card, so anything else using it (TT-Metal) dies.
    """
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        if port_open(cfg):
            raise RuntimeError(f"tmux session {session!r} exists and the X280 answers on port "
                               f"{cfg.x280.port}; attach with `tmux attach -t {session}` or kill "
                               "the session to force a reboot")
        # A console whose `make boot` failed (or whose card was reset) still holds its
        # tmux session for a day; a session with no board behind it is a leftover.
        subprocess.run(["tmux", "kill-session", "-t", session], check=True)
    log = cfg.workdir / "x280-boot.log"
    # The console must keep running (it serves the board's disk and network), so it
    # lives in tmux; the trailing sleep keeps a failed boot's output readable.
    command = f"{boot_command(cfg)} 2>&1 | tee {shlex.quote(str(log))}; sleep 86400"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
    return session
