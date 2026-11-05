"""The x86 host: pinned sources, the toolchain image, and running tools in it.

Inside the container the workdir is mounted at /pf and this package's tools/
directory at /tools (read-only), so paths under the workdir translate by
prefix. Nothing here needs root; rootless podman works as ``docker``.
"""
import json
from pathlib import Path
import shlex
import subprocess

from .config import DATA, DOCKERFILE_DIR, TOOLS

MOUNT = "/pf"


def sources():
    return json.loads((DATA / "sources.json").read_text())


def in_container(cfg, path):
    """Translate a host path under the workdir to its path inside the container."""
    path = Path(path).resolve()
    rel = path.relative_to(cfg.workdir.resolve())
    return f"{MOUNT}/{rel}" if str(rel) != "." else MOUNT


def container_command(cfg, script, env=None, workdir=MOUNT):
    cmd = [cfg.container, "run", "--rm", "-v", f"{cfg.workdir.resolve()}:{MOUNT}",
           "-v", f"{TOOLS}:/tools:ro", "-w", workdir]
    for key, value in (env or {}).items():
        cmd += ["-e", f"{key}={value}"]
    return cmd + [cfg.image, "bash", "-c", script]


def run(cfg, script, env=None, workdir=MOUNT, timeout=None, check=True, capture=True):
    """Run a bash script in the toolchain image; returns CompletedProcess."""
    proc = subprocess.run(container_command(cfg, script, env, workdir), text=True,
                          capture_output=capture, timeout=timeout)
    if capture:  # podman's docker shim prints a banner on stderr
        proc.stderr = "\n".join(line for line in proc.stderr.splitlines()
                                if not line.startswith("Emulate Docker CLI"))
    if check and proc.returncode != 0:
        tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
        raise RuntimeError(f"container command failed ({proc.returncode}):\n{script}\n{tail}")
    return proc


def git(*args, cwd=None, check=True):
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=check).stdout.strip()


def fetch_sources(cfg, names=None, log=print):
    """Clone or check out every pinned repo (or only *names*) under <workdir>/src."""
    cfg.src.mkdir(parents=True, exist_ok=True)
    repos = sources()["repos"]
    done = {}
    for name, spec in repos.items():
        if names and name not in names:
            continue
        dest = cfg.src / name
        rev = spec["revision"]
        if "worktree_of" in spec:
            base = cfg.src / spec["worktree_of"]
            if not (base / ".git").exists():
                raise RuntimeError(f"{name} needs {spec['worktree_of']} first")
            git("fetch", "-q", "origin", rev, cwd=base, check=False)
            if not dest.exists():
                git("worktree", "add", "-q", "--detach", str(dest), rev, cwd=base)
        elif not (dest / ".git").exists():
            git("clone", "-q", "--filter=blob:none", spec["url"], str(dest))
        if git("rev-parse", "HEAD", cwd=dest) != rev:
            git("fetch", "-q", "origin", rev, cwd=dest, check=False)
            git("checkout", "-q", "--detach", rev, cwd=dest)
        done[name] = git("rev-parse", "HEAD", cwd=dest)
        log(f"{name:24} {done[name]}")
    return done


def build_image(cfg, log=print):
    proc = subprocess.run([cfg.container, "build", "-t", cfg.image, str(DOCKERFILE_DIR)],
                          text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-3000:] + proc.stderr[-3000:])
    log(f"built {cfg.image}")


def probe(cfg):
    """Versions of every tool in the image, or the error that stops it running."""
    script = "; ".join([
        "echo gcc=$(gcc-14 -dumpfullversion)",
        "echo riscv64_gcc=$(riscv64-linux-gnu-gcc -dumpfullversion)",
        "echo aarch64_gcc=$(aarch64-linux-gnu-gcc -dumpfullversion)",
        "echo qemu=$(qemu-riscv64 --version | head -1 | awk '{print $3}')",
        "echo cvc5=$(cvc5 --version | head -1 | awk '{print $5}')",
        "echo saltyrn=$(git -C $SALTYRN rev-parse HEAD)",
        "echo cmake=$(cmake --version | head -1 | awk '{print $3}')",
    ])
    try:
        out = run(cfg, script, timeout=120).stdout
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": str(error)[-500:]}
    info = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    info["ok"] = all(info.get(k) for k in ("gcc", "riscv64_gcc", "qemu", "cvc5", "saltyrn"))
    info["image"] = cfg.image
    return info


def qemu_prefix(vlen):
    """qemu-riscv64 command for a binary built for the image's riscv64 sysroot."""
    cpu = f"rv64,v=true,vlen={vlen},elen=64,zba=true,zbb=true,zbs=true,vext_spec=v1.0"
    return f"qemu-riscv64 -L /usr/riscv64-linux-gnu -cpu {cpu}"


def quote(args):
    return " ".join(shlex.quote(str(a)) for a in args)
