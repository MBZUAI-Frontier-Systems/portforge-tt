"""Shared test setup: point the harness at a real kernel pack (dav1d-rvv, refmvs-splat_mv) without hardware.

Import this before any harness module. It puts the repository on sys.path and selects the pack and
family every controller-side module sees.
"""
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]          # the checkout / installed tree
HARNESS = ROOT / "portforge_tt"
PACK = ROOT / "ports" / "dav1d-rvv"
FAMILY = "refmvs-splat_mv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PORTFORGE_REPOSITORY_ROOT", str(ROOT.parents[1]))

from portforge_tt.control import port  # noqa: E402

PORT = port.configure(PACK, FAMILY)


class FakeFuture:
    """Minimal stand-in for concurrent.futures.Future in patched dispatch."""

    def __init__(self, value=None, error=None):
        self.value, self.error = value, error

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value

    def done(self):
        return True


def configure_port():
    """Select the test pack for every controller-side module."""
    return port.configure(PACK, FAMILY)


def write_harness(harness):
    """Populate a run harness the way the launcher freezes one."""
    harness = Path(harness)
    if harness.exists():
        shutil.rmtree(harness)
    port.freeze_harness(harness)
    return harness
