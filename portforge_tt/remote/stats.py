"""Statistics for kernel speed comparisons on a noisy, shared board.

Every comparison is paired: candidate and baseline run in the same process, in
the same round, at the same buffer offset, so one round gives one log ratio
d = log(t_baseline / t_candidate) and slow drift cancels. Per cell the speedup
estimate is exp(Hodges-Lehmann(d)) with a bootstrap confidence interval over
rounds; geomeans bootstrap over rounds inside each cell.

The acceptance rule turns these into decisions with a measured noise floor δ
(from A/A sessions: the baseline against itself): a cell is *flagged* only when
its estimate is below 1 - δ and its interval excludes 1. Flags are confirmed by
a re-measurement before they make a candidate infeasible (bench.py), because
with ~20 cells at 5% a neutral candidate would otherwise be rejected most of
the time.
"""
import math
import random
import statistics

MIN_NOISE = 0.005  # never claim more resolution than 0.5%


def log_ratios(baseline, candidate):
    """Paired log ratios; inputs are per-round times in the same order."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired samples need equal, non-zero lengths")
    return [math.log(b / c) for b, c in zip(baseline, candidate) if b > 0 and c > 0]


def hodges_lehmann(xs):
    """Median of all pairwise (Walsh) averages: a robust location estimate."""
    xs = list(xs)
    walsh = [(xs[i] + xs[j]) / 2 for i in range(len(xs)) for j in range(i, len(xs))]
    return statistics.median(walsh)


def bootstrap(xs, stat, n=2000, alpha=0.05, seed=0):
    """(lo, hi) percentile interval of stat over resamples of xs."""
    rng = random.Random(seed)
    xs = list(xs)
    if len(xs) < 2:
        v = stat(xs)
        return v, v
    reps = sorted(stat([rng.choice(xs) for _ in xs]) for _ in range(n))
    return reps[int(alpha / 2 * n)], reps[min(n - 1, int((1 - alpha / 2) * n))]


def cell(baseline, candidate, seed=0):
    """{s, lo, hi, n}: speedup of candidate over baseline in one cell."""
    d = log_ratios(baseline, candidate)
    lo, hi = bootstrap(d, hodges_lehmann, seed=seed)
    return {"s": math.exp(hodges_lehmann(d)), "lo": math.exp(lo), "hi": math.exp(hi), "n": len(d)}


def geomean(values):
    values = [v for v in values if v and v > 0]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def geomean_ci(paired, n=2000, alpha=0.05, seed=0):
    """Geomean speedup over cells with an interval: resample rounds inside each cell.

    paired: {cell: (baseline_times, candidate_times)}.
    """
    rng = random.Random(seed)
    per_cell = {k: log_ratios(*v) for k, v in paired.items()}
    point = math.exp(statistics.mean(hodges_lehmann(d) for d in per_cell.values()))
    reps = []
    for _ in range(n):
        est = [hodges_lehmann([rng.choice(d) for _ in d]) for d in per_cell.values()]
        reps.append(statistics.mean(est))
    reps.sort()
    return {"g": point, "lo": math.exp(reps[int(alpha / 2 * n)]),
            "hi": math.exp(reps[min(n - 1, int((1 - alpha / 2) * n))]), "cells": len(per_cell)}


def noise_floor(aa_estimates):
    """δ from A/A speedup estimates (baseline vs itself): max(0.5%, P95 |s - 1|)."""
    devs = sorted(abs(math.log(s)) for s in aa_estimates)
    if not devs:
        return MIN_NOISE
    p95 = devs[min(len(devs) - 1, int(0.95 * len(devs)))]
    return max(MIN_NOISE, math.expm1(p95))


def flagged(c, delta):
    """A cell that looks slower than the baseline beyond noise."""
    return c["s"] < 1 - delta and c["hi"] < 1


def improves(g, delta_g=0.01):
    """A geomean interval that clears the parent by more than delta_g."""
    return g["lo"] > 1 + delta_g


def summary(samples):
    """p50/p90/p99/min/max/cv of raw samples (the v3 latency summary, CPU-sized)."""
    xs = sorted(samples)
    if not xs:
        return {}

    def rank(q):
        return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]

    mean = statistics.mean(xs)
    return {"n": len(xs), "p50": rank(0.5), "p90": rank(0.9), "p99": rank(0.99), "min": xs[0],
            "max": xs[-1], "cv": statistics.pstdev(xs) / mean if mean else None}
