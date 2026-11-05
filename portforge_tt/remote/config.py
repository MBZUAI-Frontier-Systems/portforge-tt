"""Machine configuration for CPU kernel porting.

One YAML file (default ``<workdir>/cpu.yaml``, or ``$PORTFORGE_CPU_CONFIG``)
describes the two machines. Every key has a default, so a missing file means
"x86 host is this machine, X280 is reachable on localhost:2222".

    workdir: ~/portforge-cpu        # everything lives here: src/ build/ results/ repo/
    image: portforge-rvv-host:0     # x86 image with toolchains, QEMU and the verifier
    container: docker               # docker or podman
    jobs: 32                        # parallel builds and solver runs
    qemu_vlens: [128, 256, 512, 1024]
    x280:
      host: localhost               # as seen from the x86 host
      port: 2222                    # tt-bh-linux forwards this to the X280's sshd
      user: debian
      cpu: 1                        # hart pinned for timing
      remote_dir: /tmp/portforge
      boot_dir: /opt/tt-bh-linux
      boot_python: null             # a python with pyluwen and pylibfdt (make's PYTHON)
      ttdevice: 0
      l2cpu: 0
"""
from dataclasses import dataclass, field, fields
import os
from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE / "data"
TOOLS = PACKAGE / "tools"
DOCKERFILE_DIR = PACKAGE / "docker"
DEFAULT_WORKDIR = "~/portforge-cpu"


@dataclass
class X280:
    host: str = "localhost"
    port: int = 2222
    user: str = "debian"
    cpu: int = 1
    remote_dir: str = "/tmp/portforge"
    boot_dir: str = "/opt/tt-bh-linux"
    boot_python: str = None
    ttdevice: int = 0
    l2cpu: int = 0


@dataclass
class Config:
    workdir: Path = Path(DEFAULT_WORKDIR).expanduser()
    image: str = "portforge-rvv-host:0"
    container: str = "docker"
    jobs: int = os.cpu_count() or 8
    qemu_vlens: list = field(default_factory=lambda: [128, 256, 512, 1024])
    x280: X280 = field(default_factory=X280)

    @property
    def src(self):
        return self.workdir / "src"

    @property
    def build(self):
        return self.workdir / "build"

    @property
    def results(self):
        return self.workdir / "results"


def _known(cls, raw, where):
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown keys in {where}: {sorted(unknown)}")
    return raw


def load(path=None):
    """Load the config file, or defaults when there is none."""
    path = path or os.environ.get("PORTFORGE_CPU_CONFIG")
    if path is None:
        candidate = Path(DEFAULT_WORKDIR).expanduser() / "cpu.yaml"
        path = candidate if candidate.exists() else None
    raw = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).expanduser().read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must be a YAML mapping")
    raw = dict(_known(Config, raw, path))
    x280 = X280(**_known(X280, raw.pop("x280", None) or {}, f"{path}: x280"))
    if "workdir" in raw:
        raw["workdir"] = Path(str(raw["workdir"])).expanduser()
    return Config(**raw, x280=x280)
