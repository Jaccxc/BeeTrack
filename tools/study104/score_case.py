"""Score the §104 tracking arms or recompute the metrics from archived events."""

import argparse
import json
from pathlib import Path

import numpy as np

from event_overlap import archive_arm, compare_arms, replay_arm, score_arm, union_summary
from tools.study103.swap_score import check_baseline, counts_from_events, load_case, window_error


def timestamps(entries):
    return np.array([(abs(frame), 1 if frame > 0 else -1) for _, frame, *_ in entries],
                    dtype=int).reshape(-1, 2)


def evaluate(record):
    nframes = record["nframes"]
    truth = counts_from_events(timestamps(record["gt"]), nframes)
    windows, pairwise, scores = {}, {}, {}
    for threshold in ("0.5", "0.3"):
        arms = {}
        for name, raw in record["arms"].items():
            arms[name] = replay_arm(dict(gt=record["gt"], predictions=raw["predictions"],
                                         edges=raw["edges"][threshold]))
            if threshold == "0.5":
                counts = counts_from_events(timestamps(raw["predictions"]), nframes)
                delta = (counts - truth).sum(axis=0)
                assert np.all(delta % 30 == 0)
                windows[name] = dict(window=window_error(truth, counts), bias=(delta // 30).tolist())
        pairwise[threshold] = {name: compare_arms(arms["F"], arms[name]) for name in ("R", "RF")}
        scores[float(threshold)] = pairwise[threshold]["R"]["forward"]
        if threshold == "0.5":
            union = union_summary(arms["F"], arms["R"], nframes)
            gt_in, gt_out = truth.sum(axis=0) // 30
            windows["union"] = dict(window=union["window"], added=len(union["added"]),
                                    bias=[int(union["n_in"]-gt_in), int(union["n_out"]-gt_out)])
    forward = counts_from_events(timestamps(record["arms"]["F"]["predictions"]), nframes)
    check_baseline(record["hive"], record["seed"], truth, forward, scores)
    return dict(windows=windows, pairwise=pairwise)


def score_case(hive, seed, track_dir):
    from beetrack.evaluate import load_tracks
    gt, canonical, polygon, nframes = load_case(hive, seed)
    arms = {}
    truth = None
    for name in ("F", "R", "RF"):
        pred = load_tracks(track_dir / f"{hive}_{seed}" / (name + ".txt"))
        if name == "F":
            from beetrack.evaluate import by_track
            np.testing.assert_array_equal(by_track(pred), canonical)
        edges = {}
        predictions = None
        for threshold in (0.5, 0.3):
            raw = archive_arm(score_arm(gt, pred, polygon, threshold))
            if truth is None:
                truth = raw["gt"]
            assert truth == raw["gt"]
            if predictions is None:
                predictions = raw["predictions"]
            assert predictions == raw["predictions"]
            edges[str(threshold)] = raw["edges"]
        arms[name] = dict(predictions=predictions, edges=edges)
    record = dict(hive=hive, seed=seed, nframes=nframes, gt=truth, arms=arms)
    record["metrics"] = evaluate(record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hive", type=int, choices=[33, 35])
    parser.add_argument("seed", type=int, choices=range(9))
    parser.add_argument("track_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = score_case(args.hive, args.seed, args.track_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".tmp")
    partial.write_text(json.dumps(record, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    partial.replace(args.output)
    main = record["metrics"]["pairwise"]["0.5"]["R"]["total"]
    print(f"{args.hive} seed {args.seed}: R-only TP {main['reverse_only_true']}, R-only FP "
          f"{main['reverse_only_false']}, J {main['net']}", flush=True)
