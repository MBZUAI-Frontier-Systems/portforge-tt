"""Board power for a leased card, sampled by the trusted runner outside the sandbox.

Blackhole's kernel driver exposes one hwmon device per card
(/sys/class/hwmon/hwmonN, name "blackhole") whose ``device`` link resolves to the
card's PCI address; ``power1_input`` is board power in microwatts and is readable
without root. The firmware refreshes it a few times per second at most, so energy
is only meaningful over sustained windows (the perf stage runs each case for a
minimum wall time, MLPerf Power style); short windows are reported as such.

Candidate code never touches these numbers: the runner samples them while the
job runs and integrates them over the timed windows the trusted evaluator
recorded in measurements.json (wall-clock ``window`` and ``items`` per case).
"""
import json
from pathlib import Path
import threading
import time

HWMON = Path('/sys/class/hwmon')
MIN_WINDOW_SECONDS = 10.0


def hwmon_for_pci(pci, root=HWMON):
    """The hwmon directory whose device is this PCI card, or None."""
    if not Path(root).is_dir():
        return None
    for path in sorted(Path(root).iterdir()):
        try:
            if (path / 'device').resolve().name == pci and (path / 'power1_input').is_file():
                return path
        except OSError:
            continue
    return None


def read_watts(hwmon):
    """Board power of one card, or the sum over a list of cards (a mesh job's leased cards)."""
    if isinstance(hwmon, (list, tuple)):
        return sum(read_watts(path) for path in hwmon)
    return int((Path(hwmon) / 'power1_input').read_text().strip()) / 1e6


class PowerSampler:
    """Background sampler of (unix_seconds, watts); tolerant of transient read errors."""

    def __init__(self, hwmon, interval=0.1):
        self.hwmon = [Path(p) for p in hwmon] if isinstance(hwmon, (list, tuple)) else Path(hwmon)
        self.interval, self.samples = interval, []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.time(), read_watts(self.hwmon)))
            except (OSError, ValueError):
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        return self.samples


def idle_watts(hwmon, seconds=2.0, interval=0.1):
    sampler = PowerSampler(hwmon, interval).start()
    time.sleep(seconds)
    samples = sampler.stop()
    return sum(w for _, w in samples) / len(samples) if samples else None


def integrate(samples, start, end):
    """Joules over [start, end] by trapezoids on the sampled step signal, plus mean watts."""
    points = [(t, w) for t, w in samples if start <= t <= end]
    before = [(t, w) for t, w in samples if t < start]
    after = [(t, w) for t, w in samples if t > end]
    # Clamp the neighbouring samples to the window edges so a slow sensor still covers it.
    if before:
        points.insert(0, (start, before[-1][1]))
    if after:
        points.append((end, after[0][1]))
    if len(points) < 2 or end <= start:
        return None
    joules = sum((t1 - t0) * (w0 + w1) / 2 for (t0, w0), (t1, w1) in zip(points, points[1:]))
    covered = points[-1][0] - points[0][0]
    return dict(joules=joules, mean_watts=joules / covered if covered > 0 else None,
                samples=sum(start <= t <= end for t, _ in samples), seconds=end - start)


def energy_report(samples, measurements, idle=None):
    """Per-case energy over the evaluator's timed windows; never a gate, always labelled."""
    cases = []
    for case in (measurements or {}).get('cases', []):
        window, items = case.get('window'), case.get('items')
        if (not isinstance(window, list) or len(window) != 2
                or any(type(x) not in (int, float) for x in window)):
            continue
        part = integrate(samples, window[0], window[1])
        if not part:
            continue
        runs = len(case.get('seconds') or []) or None
        row = dict(name=case.get('name'), **part, timed_runs=runs, items=items,
                   short_window=part['seconds'] < MIN_WINDOW_SECONDS)
        if idle is not None and part['mean_watts'] is not None:
            row['active_joules'] = max(0.0, part['joules'] - idle * part['seconds'])
        if runs:
            row['joules_per_run'] = part['joules'] / runs
            if items:
                row['joules_per_item'] = part['joules'] / (runs * items)
                row['items_per_joule'] = runs * items / part['joules'] if part['joules'] > 0 else None
        cases.append(row)
    watts = [w for _, w in samples]
    return dict(source='hwmon power1_input (board input power, firmware-updated)', idle_watts=idle,
                job_mean_watts=sum(watts) / len(watts) if watts else None,
                job_peak_watts=max(watts) if watts else None, job_samples=len(watts), cases=cases,
                note='Sensor refresh is a few Hz; trust only windows >= %gs.' % MIN_WINDOW_SECONDS)


def write_trace(path, samples):
    Path(path).write_text(json.dumps({'unit': 'watts', 'samples': samples}))
