"""Controller-side client of the gate runner (remote.py) on the x86 host.

The controller holds the agent CLIs and workspaces; the x86 host holds the
toolchain image, QEMU, the XNNPACK builds and the board. Files go over with
rsync into <remote workdir>/runs/<run>/..., commands run as
`python3 -m portforge_tt.kernel.remote CMD args.json` in the synced repo, and
the last stdout line is the JSON result.
"""
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
SSH = ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]


POLL_SECONDS = 10


def _retry(fn, attempts=6, delay=15):
    """Run fn, retrying transient ssh/rsync failures with backoff."""
    for i in range(attempts):
        try:
            return fn()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError):
            if i == attempts - 1:
                raise
            time.sleep(delay * (i + 1))


class RemoteGates:
    def __init__(self, host="tt_box", workdir="portforge-cpu", run_id="run"):
        # workdir is relative to the remote home: ssh starts there, and a quoted "~" never expands
        self.host, self.workdir, self.run_id = host, workdir.rstrip("/"), run_id

    @property
    def run_rel(self):
        return f"runs/{self.run_id}"

    def _ssh(self, command, timeout=7200, stdin=None):
        return subprocess.run(SSH + [self.host, command], text=True, capture_output=True, timeout=timeout,
                              input=stdin)

    def sync_code(self, pack):
        """Copy the harness and the pack to the host's repo (no deletion: the
        host's repo holds earlier campaigns' artifacts)."""
        subprocess.run(["rsync", "-a", "--exclude", "__pycache__", "-R", "portforge_tt", pack, "pyproject.toml",
                        f"{self.host}:{self.workdir}/repo/", "-e", " ".join(SSH)],
                       cwd=REPO, check=True, capture_output=True, text=True)

    def put(self, local, remote_rel):
        """Upload one local file to <workdir>/<remote_rel>; returns remote_rel."""
        def once():
            self._ssh(f"mkdir -p {shlex.quote(self.workdir + '/' + str(Path(remote_rel).parent))}", timeout=60)
            subprocess.run(["rsync", "-a", str(local), f"{self.host}:{self.workdir}/{remote_rel}", "-e",
                            " ".join(SSH)], check=True, capture_output=True, text=True, timeout=600)
        _retry(once)
        return remote_rel

    def call(self, command, args, timeout=7200):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(args, handle)
        remote_args = f"{self.run_rel}/rpc/{command}-{Path(handle.name).stem}.json"
        self.put(handle.name, remote_args)
        Path(handle.name).unlink()
        root = f"$HOME/{self.workdir}"  # unquoted: the remote shell expands $HOME
        base = f"{root}/{remote_args}"
        # Detached on the host, so a dropped ssh connection (seen mid-campaign: "Connection
        # reset by peer" during a 20-minute gates batch) never kills or loses the job; the
        # controller polls for the exit marker with short, retried ssh calls.
        job = (f"cd {root}/repo && PYTHONPATH=. python3 -m portforge_tt.kernel.remote {command} {base} "
               f"> {base}.out 2> {base}.err; echo $? > {base}.done")
        _retry(lambda: self._checked(f"nohup setsid sh -c {shlex.quote(job)} > /dev/null 2>&1 < /dev/null &", 60))
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(POLL_SECONDS)
            try:
                proc = self._ssh(f"cat {base}.done 2>/dev/null && tail -n 3 {base}.out && echo @@@ && "
                                 f"tail -c 1500 {base}.err", timeout=120)
            except subprocess.TimeoutExpired:
                continue
            if proc.returncode != 0 or not proc.stdout.strip():
                continue  # not finished yet, or a transient ssh failure
            out, _, err = proc.stdout.partition("@@@")
            lines = [l for l in out.splitlines() if l.startswith("{")]
            if not lines:
                return {"ok": False, "error": f"no result (exit {out.splitlines()[0]}): {err[-1500:]}"}
            return json.loads(lines[-1])
        return {"ok": False, "error": f"remote {command} still running after {timeout}s ({base})"}

    def _checked(self, command, timeout):
        proc = self._ssh(command, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"ssh {self.host} failed ({proc.returncode}): {proc.stderr[-500:]}")
        return proc

    def fetch(self, remote_rel, local):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{self.host}:{self.workdir}/{remote_rel}", str(local), "-e", " ".join(SSH)],
                       check=False, capture_output=True, text=True)
