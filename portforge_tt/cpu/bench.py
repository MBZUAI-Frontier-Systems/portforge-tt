"""The trusted benchmark protocol: candidate vs upstream on the regime matrix.

One session = one pfharness process on the X280, holding the board lease,
started only when the shared board is idle and re-checked afterwards:

  - every cell (regime x shape) runs R rounds; each round runs every kernel
    once in a seeded random order at a rotating buffer offset, as a timed block
    of calls lasting ~2 ms, with perf_event cycles and instret;
  - after timing, each kernel's output on the cell's canonical input is hashed
    next to the reference kernel's: a wrong kernel is wrong, not fast;
  - per cell the baseline is the fastest upstream variant whose output is
    correct; cells where no upstream variant is correct are excluded from the
    speed geomean and reported as correctness wins;
  - speedups are paired per round (stats.cell); a cell is flagged slower only
    beyond the noise floor δ (stats.flagged).

analyze() is pure (rows in, verdict out) so it is unit-tested without a board.
"""
import json
import time
from pathlib import Path

from . import harness, stats, x280


class Contended(RuntimeError):
    """The board stayed busy (another agent's job) for the whole wait budget."""


def session(cfg, exe, regimes_file, shapes, reference, kernels, out_dir, rounds=15, target_ns=2e6,
            seed=1, wait_seconds=1800, log=print):
    """One timed session on the X280; returns its receipt (rows saved alongside).

    Waits (backing off) while the board is busy, and retries a session whose
    closing idle check shows another job ran during it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + wait_seconds
    delay = 15
    with x280.lease(cfg):
        while True:
            before = x280.idle(cfg)
            if before.get("ok"):
                t0 = time.time()
                rows, err = harness.bench_raw(cfg, exe, regimes_file, shapes, rounds, target_ns, seed,
                                              reference, kernels, where="x280")
                after = x280.idle(cfg)
                if after.get("ok"):
                    break
                log(f"board got busy during the session ({after}); retrying")
            else:
                log(f"board busy ({before}); waiting {delay}s")
            if time.time() > deadline:
                raise Contended(f"board busy for {wait_seconds}s")
            time.sleep(delay)
            delay = min(delay * 2, 240)
    receipt = {"board": x280.board_id(cfg), "harness_sha256": harness.sha(), "kernels": kernels,
               "reference": reference, "shapes": shapes, "rounds": rounds, "target_ns": target_ns,
               "seed": seed, "idle_before": before, "idle_after": after, "seconds": round(time.time() - t0, 1),
               "counters": "perf_event" if any(r.get("cycles") for r in rows if "ns" in r) else "none",
               "stderr": err[-500:]}
    (out_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    return receipt, rows


def _series(rows):
    """{(regime, shape): {slot: [row per round, ordered by round]}} and hashes."""
    timing, hashes = {}, {}
    for r in rows:
        key = (r["regime"], r["shape"])
        if "ns" in r:
            timing.setdefault(key, {}).setdefault(r["slot"], []).append(r)
        elif "hash" in r:
            hashes.setdefault(key, {})[r["slot"]] = r["hash"] == r["reference_hash"]
    for cell in timing.values():
        for series in cell.values():
            series.sort(key=lambda r: r["round"])
    return timing, hashes


def analyze(rows, slots, candidate_slot, baseline_slots, delta=0.01, excluded=(), reference_cell=None, seed=0):
    """Verdict of one candidate against the upstream baselines in a session.

    slots: kernel name per slot (the order passed to pfharness).
    Returns {cells: {"regime@shape": {...}}, metrics: {...}, flagged: [...], ...}.
    """
    timing, hashes = _series(rows)
    cells, paired, wrong, wins = {}, {}, [], []
    for (regime, shape), by_slot in sorted(timing.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        key = f"{regime}@{shape}"
        cand_rows = by_slot.get(candidate_slot, [])
        if not cand_rows:
            continue
        cand_ok = hashes.get((regime, shape), {}).get(candidate_slot, False)
        correct = [s for s in baseline_slots if hashes.get((regime, shape), {}).get(s, False)]
        elems = cand_rows[0].get("elems", shape)  # transposes: shape is "WxH", elems W*H
        cycles = stats.summary([r["cycles"] / elems for r in cand_rows if r["cycles"]])
        entry = {"candidate_correct": cand_ok, "baseline_correct": bool(correct),
                 "candidate_ns": stats.summary([r["ns"] for r in cand_rows]).get("p50"),
                 "cycles_per_elem": cycles.get("p50"),
                 "ipc": (stats.statistics.median(r["instret"] / r["cycles"] for r in cand_rows if r["cycles"])
                         if any(r["cycles"] for r in cand_rows) else None)}
        if not cand_ok:
            wrong.append(key)
        if not correct or key in excluded:
            entry["excluded"] = "no correct upstream variant" if not correct else "excluded by spec"
            if not correct and cand_ok:
                wins.append(key)
            cells[key] = entry
            continue
        best = min(correct, key=lambda s: stats.statistics.median(r["ns"] for r in by_slot[s]))
        base_ns = [r["ns"] for r in by_slot[best]]
        cand_ns = [r["ns"] for r in cand_rows]
        n = min(len(base_ns), len(cand_ns))
        c = stats.cell(base_ns[:n], cand_ns[:n], seed=seed)
        entry.update(baseline=slots[best], baseline_ns=stats.summary(base_ns).get("p50"),
                     speedup=c, flagged=stats.flagged(c, delta))
        paired[key] = (base_ns[:n], cand_ns[:n])
        cells[key] = entry
    g = stats.geomean_ci(paired, seed=seed) if paired else None
    worst = min((e["speedup"]["s"] for e in cells.values() if "speedup" in e), default=None)
    ref = reference_cell or next(iter(cells), None)
    metrics = {
        "speedup_geomean": g["g"] if g else None,
        "speedup_geomean_ci": [g["lo"], g["hi"]] if g else None,
        "worst_cell": worst,
        "cycles_per_elem": cells.get(ref, {}).get("cycles_per_elem") if ref else None,
    }
    return {"cells": cells, "metrics": metrics, "geomean": g, "reference_cell": ref,
            "flagged": [k for k, e in cells.items() if e.get("flagged")],
            "wrong_cells": wrong, "correctness_wins": wins, "delta": delta,
            "feasible": not wrong and not any(e.get("flagged") for e in cells.values())}
