"""Run one registered §103 case on the immutable P0 observations, on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np

from paired_swaps import discover, exchange
from phase_control import phase_summary
from swap_bound import best_disjoint, earliest_disjoint
from swap_score import (check_baseline, counts_from_events, event_scores, event_times,
                        load_case, vector_of_counts, window_error)


def run(hive, seed):
    gt, original, polygon, nframes = load_case(hive, seed)
    gt_events = event_times(gt, polygon)
    p0_events = event_times(original, polygon)
    truth = counts_from_events(gt_events, nframes)
    base = counts_from_events(p0_events, nframes)
    baseline_scores = event_scores(gt, original, polygon)
    check_baseline(hive, seed, truth, base, baseline_scores)
    candidates, census = discover(original, gt, return_counts=True)
    deltas, pair_events = [], []
    census.update(same_windows=0, timing_only=0, changed_totals=0)
    for left, right, cut in candidates:
        rows = original[np.isin(original[:, 1], [left, right])]
        before = event_times(rows, polygon)
        after = event_times(exchange(rows, [(left, right, cut)]), polygon)
        delta = counts_from_events(after, nframes) - counts_from_events(before, nframes)
        totals = delta.sum(axis=0)
        if totals[0] != totals[1]:
            raise RuntimeError("A complete suffix exchange changed signed net flow")
        kind = "changed_totals" if totals.any() else "timing_only" if delta.any() else "same_windows"
        census[kind] += 1
        pair_events.append(dict(pair=[left, right, cut], before=before.tolist(), after=after.tolist(), kind=kind))
        deltas.append(delta)

    chronological = earliest_disjoint(candidates)
    optimal, bound = best_disjoint(base, truth, candidates, deltas, np.full(len(base), 100 / truth.sum()))
    arm_cache = {}

    def replay(positions):
        key = tuple(sorted(positions))
        if key not in arm_cache:
            if positions:
                changed = exchange(original, [candidates[i] for i in positions])
                observed = vector_of_counts(changed, polygon, nframes)
                np.testing.assert_array_equal(observed, base + sum(deltas[i] for i in positions))
                scores = event_scores(gt, changed, polygon)
            else:
                observed, scores = base, baseline_scores
            difference = (observed - truth).sum(axis=0)
            assert all(difference % 30 == 0)
            arm_cache[key] = dict(window=window_error(truth, observed), scores=scores,
                                  bias_in=int(difference[0] // 30), bias_out=int(difference[1] // 30))
        return dict(arm_cache[key], positions=positions)

    arms = dict(p0=replay([]), chronological=replay(chronological), oracle=replay(optimal))
    if not np.isclose(arms["oracle"]["window"], bound, rtol=0, atol=1e-7):
        raise RuntimeError("Complete track replay differs from optimum")
    phase = phase_summary(gt_events, p0_events, nframes)
    if not np.isclose(phase["original"], arms["p0"]["window"]) or phase["n_shifts"] != 157:
        raise RuntimeError("Phase comparison failed its zero-shift control")
    return dict(hive=hive, seed=seed, nframes=nframes, gt_events=gt_events.tolist(),
                p0_events=p0_events.tolist(), census=census, candidates=pair_events, arms=arms, phase=phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hive", type=int, choices=[33, 35])
    parser.add_argument("seed", type=int, choices=range(9))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    record = run(args.hive, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(args.hive, args.seed, record["census"], flush=True)
