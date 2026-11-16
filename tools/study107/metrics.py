"""Recompute offline 30-second timing errors from flux and 1-based (frame, direction) events."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np


ALPHAS = (0.0, 0.25, 0.5)
TOL = 1e-9  # percentage points
SHIFTS = np.arange(1, 158) * 25


def _flux_array(flux):
    flux = np.asarray(flux, dtype=np.float64)
    if (flux.ndim != 2 or flux.shape[1] != 2 or len(flux) == 0 or
            not np.all(np.isfinite(flux)) or np.any(flux < 0)):
        raise ValueError("Flux must be a finite nonnegative (frames, 2) array.")
    if not np.all(np.isfinite(flux.sum(axis=0))):
        raise ValueError("Nonfinite integrated flux.")
    return flux


def event_mass(events, nframes):
    """One bin per original frame and direction; duplicate-frame events accumulate."""
    events = np.asarray(events)
    if events.shape == (0,):
        events = events.reshape(0, 2)
    if events.ndim != 2 or events.shape[1] != 2 or not np.all(np.isfinite(events)):
        raise ValueError("Expected (frame, +1 in / -1 out) pairs.")
    if not np.all(events == events.astype(np.int64)):
        raise ValueError("Expected integer event frames and directions.")
    frames, directions = events.astype(np.int64).T
    if np.any(frames < 1) or np.any(frames > nframes) or not np.all(np.isin(directions, [-1, 1])):
        raise ValueError("Invalid event frame or direction.")
    idx = (frames - 1) * 2 + (directions < 0)
    return np.bincount(idx, minlength=nframes * 2).reshape(nframes, 2)


@lru_cache(maxsize=16)
def _bounds(nframes):
    starts, stops = [], []
    for offset in range(0, 750, 25):
        interior = list(range(750 - offset, nframes, 750))
        starts.extend([0, *interior])
        stops.extend([*interior, nframes])
    return np.array(starts), np.array(stops)


def window_counts(frame_mass):
    """All 30 offset partitions, including initial and final partial windows."""
    starts, stops = _bounds(len(frame_mass))
    aggregate = np.vstack(([0.0, 0.0], np.cumsum(frame_mass, axis=0, dtype=np.float64)))
    return aggregate[stops] - aggregate[starts]


def _error(truth, prediction):
    return float(np.abs(truth - prediction).sum() * 100 / truth.sum())


def _panels(flux, gt, p0):
    truth = event_mass(gt, len(flux))
    if not truth.any():
        raise ValueError("Window percentage requires a nonzero GT total.")
    base = event_mass(p0, len(flux))
    flux_totals = flux.sum(axis=0)
    valid = flux_totals > 1e-12
    shape = np.divide(flux, flux_totals, out=np.zeros_like(flux), where=valid)
    return truth, base, shape, valid


def _shaped_counts(base_windows, predicted_totals, shape, valid):
    """Window counts at alpha=1; unsensed output directions retain P0 at every frame."""
    modeled = window_counts(shape) * predicted_totals
    modeled[:, ~valid] = base_windows[:, ~valid]
    starts, _ = _bounds(len(shape))
    totals = np.add.reduceat(modeled, np.flatnonzero(starts == 0), axis=0)
    if (not np.all(np.isfinite(modeled)) or np.any(modeled < 0) or
            not np.allclose(totals, predicted_totals, atol=1e-9, rtol=1e-12)):
        raise ValueError("A direction has invalid mass or differs from P0 in a window partition.")
    return modeled


def _seeds(seed_map, n):
    if set(map(str, seed_map)) != {str(seed) for seed in range(n)} or len(seed_map) != n:
        raise ValueError(f"Expected seeds 0 through {n - 1}.")
    return [seed_map[s] if s in seed_map else seed_map[str(s)] for s in range(n)]


def fit_source(source_clips):
    """Select one shared alpha: average seeds within clip, then weight clips equally."""
    if len(source_clips) != 2:
        raise ValueError("Source fitting needs the same source hive's head and tail.")
    per_clip = {}
    for name, clip in sorted(source_clips.items()):
        flux = _flux_array(clip["flux"])
        seed_scores = []
        for p0 in _seeds(clip["p0"], 3):
            truth, base, shape, valid = _panels(flux, clip["gt"], p0)
            g, h = window_counts(truth), window_counts(base)
            model = _shaped_counts(h, base.sum(axis=0), shape, valid)
            seed_scores.append([_error(g, (1 - a) * h + a * model) for a in ALPHAS])
        per_clip[name] = seed_scores
    scores = np.mean([np.mean(rows, axis=0) for rows in per_clip.values()], axis=0)
    alpha = next(a for a, score in zip(ALPHAS, scores) if score <= min(scores) + TOL)
    return dict(alpha=alpha, candidates=list(ALPHAS), candidate_scores=scores.tolist(),
                scores_by_clip_seed_candidate=per_clip)


def score_target(flux, gt, p0, alpha):
    """All preregistered target comparisons; phase controls roll flux, leaving P0 fixed."""
    if alpha not in ALPHAS:
        raise ValueError("Use the selected source alpha.")
    flux = _flux_array(flux)
    if len(flux) != 3950:
        raise ValueError("The 157 nonidentity phase shifts are registered for 3950-frame heads.")
    truth, base, shape, valid = _panels(flux, gt, p0)
    g, h = window_counts(truth), window_counts(base)
    gt_total, predicted_total = truth.sum(axis=0), base.sum(axis=0)
    density = _shaped_counts(h, predicted_total, shape, valid)
    uniform = _shaped_counts(h, predicted_total, np.full(shape.shape, 1 / len(shape)), np.ones(2, bool))
    swapped = _shaped_counts(h, predicted_total, shape[:, ::-1], valid[::-1])
    shifts = [_shaped_counts(h, predicted_total, np.roll(shape, int(k), axis=0), valid)
              for k in SHIFTS]
    baseline = _error(g, h)
    floor = float(100 * np.abs(predicted_total - gt_total).sum() / gt_total.sum())
    def arm(a):
        def error(counts):
            return _error(g, (1 - a) * h + a * counts)
        aligned = error(density)
        phase = np.array([error(panel) for panel in shifts])
        beaten = int(np.count_nonzero(phase > aligned + TOL))
        unif = error(uniform)
        swapped_error = error(swapped)
        if min(aligned, unif, swapped_error, phase.min(), baseline) < floor - TOL:
            raise AssertionError("A total-preserving arm fell below its theoretical floor.")
        return dict(alpha=a, aligned=aligned, uniform=unif, swapped=swapped_error,
                    phase=phase.tolist(), phase_median=float(np.median(phase)),
                    phase_strictly_beaten=beaten, phase_fraction_le=(len(phase)-beaten)/len(phase),
                    beats_uniform_and_phase=bool(unif > aligned + TOL and beaten * 10 >= 9 * len(phase)))
    return dict(p0=baseline, floor=floor, timing_headroom=baseline-floor,
                gt_totals=gt_total.tolist(), p0_totals=predicted_total.tolist(),
                fallback_aligned=(~valid).tolist(), fallback_swapped=(~valid[::-1]).tolist(),
                primary=arm(float(alpha)), fixed=arm(0.25))


def fold_summary(rows, alpha):
    records = _seeds(rows, 9)
    base = np.array([r["p0"] for r in records])
    def measure(name, expected):
        arms = [r[name] for r in records]
        if any(a["alpha"] != expected for a in arms):
            raise ValueError("Target arm does not match frozen weight.")
        wins = sum(a["beats_uniform_and_phase"] for a in arms)
        values = {k: float(np.mean([a[k] for a in arms])) for k in ("aligned", "uniform", "swapped")}
        gain = float(np.mean(base - [a["aligned"] for a in arms]))
        passes = bool(expected > 0 and gain >= 2 and wins >= 7)
        directional_differences = np.array([a["swapped"] - a["aligned"] for a in arms])
        directional_gain = float(directional_differences.mean())
        directional_wins = int(np.count_nonzero(directional_differences > TOL))
        return dict(values, gain_vs_p0=gain, joint_wins=wins, passes=passes,
                    gain_vs_swapped=directional_gain, directional_wins=directional_wins,
                    directional_passes=bool(passes and directional_gain >= 1 and directional_wins >= 7))
    return dict(p0_mean=float(base.mean()), floor_mean=float(np.mean([r["floor"] for r in records])),
                primary=measure("primary", alpha), fixed=measure("fixed", 0.25))


def replay(package):
    """Refit source selections, verify frozen weights, then score all nine target seeds."""
    fits = {}
    for hive in ("33", "35"):
        fits[hive] = fit_source(package["sources"][hive])
        if fits[hive]["alpha"] != package["frozen"][hive]:
            raise ValueError(f"Frozen weight disagrees with source replay for hive {hive}.")
    output = {"sources": fits, "targets": {}}
    for hive in ("33", "35"):
        data = package["targets"][hive]
        alpha = fits[hive]["alpha"]
        rows = {s: score_target(data["flux"], data["gt"], p, alpha)
                for s, p in enumerate(_seeds(data["p0"], 9))}
        output["targets"][hive] = dict(seeds=rows, summary=fold_summary(rows, alpha))
    return output


def self_check():
    n = 3950
    gt = [[1, 1], [25, 1], [750, 1], [751, 1], [3950, 1]]
    p0 = [[1, 1], [750, 1], [750, -1], [751, 1], [3925, -1]]
    frames = np.arange(n)
    test_mass = np.column_stack((frames + 1., (frames % 11) / 17))
    reference = []
    for offset in range(0, 750, 25):
        idx = (frames + offset) // 750
        reference.extend(np.stack([np.bincount(idx, weights=d) for d in test_mass.T], axis=1))
    np.testing.assert_allclose(window_counts(test_mass), reference, rtol=0, atol=1e-10)
    zeros = np.zeros((n, 2))
    clips = {name: dict(flux=zeros, gt=gt, p0={s: p0 for s in range(3)}) for name in ("head","tail")}
    fit = fit_source(clips)
    assert fit["alpha"] == 0 and len(set(fit["candidate_scores"])) == 1
    row = score_target(zeros, gt, p0, 0.)
    assert row["floor"] == 80 and row["gt_totals"] == [5, 0] and row["p0_totals"] == [3, 2]
    assert row["primary"]["phase_fraction_le"] == 1 and row["fixed"]["aligned"] == row["p0"]
    assert not fold_summary({s: row for s in range(9)}, 0.)["primary"]["passes"]
    flux = event_mass(gt, n).astype(float)
    exact = score_target(flux, gt, gt, 0.25)
    assert exact["fallback_aligned"] == [False, True] and exact["fallback_swapped"] == [True, False]
    assert exact["primary"]["aligned"] < TOL and exact["floor"] == 0
    assert exact["primary"]["phase_strictly_beaten"] == 157
    # Matching directional counts on different frames lets alignment strictly beat P0.
    p_late = [[(f + 1749) % n + 1,d] for f,d in gt]
    for clip in clips.values():
        clip.update(flux=flux, p0={s: p_late for s in range(3)})
    fit = fit_source(clips)
    assert fit["alpha"] == 0.5 and np.all(np.diff(fit["candidate_scores"]) < 0)
    limits = score_target(zeros, gt, [[f,1] for f in [1,750,751]] + [[300,-1],[300,-1]], 0.)
    assert abs(limits["p0"] - limits["floor"]) < TOL


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    run = sub.add_parser("replay", help="recompute both folds, printing JSON if no output is given")
    run.add_argument("package", type=Path)
    run.add_argument("--out", type=Path)
    args = cli.parse_args()
    if args.command == "self-check":
        self_check()
        print("Synthetic timing and source/target separation checks passed.")
    else:
        result = replay(json.loads(args.package.read_text(encoding="utf-8")))
        serialized = json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True)
        if args.out is None:
            print(serialized)
        else:
            temporary = args.out.with_suffix(args.out.suffix + ".tmp")
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(args.out)
