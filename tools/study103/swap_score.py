"""Frozen P0 dev-head inputs and BeeTrack's directional count / event scores.

Uses repo-native MOT files, ZONES and the project's frame matching. GT was
already repaired by beetrack.prep. Windows follow §95: shift by 0..29 s,
date each event at its last same-direction crossing, retain partial windows.
"""
import configparser
import json
import re
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment

from beetrack.common import ROOT, RUNS
from beetrack.evaluate import ZONES, by_track, f1, frame_match, load_tracks, zone_tracks


def load_case(hive, seed):
    """Return GT and P0 sorted by resolved run ID / frame, polygon, frame count."""
    if str(hive) not in ("33", "35") or str(seed) not in tuple(map(str, range(9))):
        raise ValueError("Only dev head hives 33/35 and seeds 0..8 are registered")
    seq = f"BEE24-{hive}-head"
    gt_dir = RUNS / "gt" / "dev" / seq
    cfg = configparser.ConfigParser()
    cfg.read(gt_dir / "seqinfo.ini", encoding="utf-8")
    assert cfg.getint("Sequence", "frameRate") == 25
    nframes = cfg.getint("Sequence", "seqLength")
    pred = RUNS / "trackers" / "dev" / f"q_lk_loho_no{hive}_s{seed}" / "data" / f"{seq}.txt"
    poly = json.loads(ZONES.read_text(encoding="utf-8"))[seq]
    gt, pr = (by_track(load_tracks(p)) for p in (gt_dir / "gt" / "gt.txt", pred))
    return gt, pr, poly, nframes


def event_times(rows, poly):
    """Each row is (1-based frame, +1 in / -1 out)."""
    rows = by_track(rows)
    events = []
    for direction, _, _, crossings in zone_tracks(rows, poly).values():
        if direction:
            index = next(i for i, d in reversed(crossings) if d == direction)
            events.append((int(rows[index, 0]), 1 if direction == "in" else -1))
    return np.array(events, dtype=int).reshape(-1, 2)


def counts_from_events(events, nframes):
    """Stack the (in,out) bins of each of the 30 offsets into a single array."""
    if nframes < 1 or np.any((events[:, 0] < 1) | (events[:, 0] > nframes)):
        raise ValueError("Events must be inside the 1-based video bounds")
    offsets = []
    direction = (events[:, 1] < 0).astype(int)
    for shift in range(0, 750, 25):
        nbins = (nframes + shift + 749) // 750
        bins = (events[:, 0] - 1 + shift) // 750
        offsets.append(np.bincount(2 * bins + direction, minlength=2 * nbins).reshape(nbins, 2))
    return np.vstack(offsets)


def vector_of_counts(rows, poly, nframes):
    return counts_from_events(event_times(rows, poly), nframes)


def window_error(gt_counts, pred_counts):
    """Percent. Each real GT event occurs once at each of 30 offsets."""
    if not gt_counts.sum():
        raise ValueError("The normalized count error requires at least one GT event")
    return float(100 * np.abs(gt_counts - pred_counts).sum() / gt_counts.sum())


def event_scores(gt, pr, poly):
    """Return {IoU: {tp, fp, fn, f1}} under official ±5-frame matching.

    The candidate matrix follows evaluate.match_events. Explicitly passing
    frame_match's IoU threshold avoids modifying its global default for .3.
    """
    gt, pr = by_track(gt), by_track(pr)
    gt_events = {tid: info for tid, info in zone_tracks(gt, poly).items() if info[0]}
    pr_events = [info for info in zone_tracks(pr, poly).values() if info[0]]
    gt_column = {tid: i for i, tid in enumerate(gt_events)}
    scores = {}
    for threshold in (0.5, 0.3):
        under = frame_match(gt, pr, thr=threshold)
        possible = np.zeros((len(pr_events), len(gt_events)), dtype=int)
        for i, (direction, _, _, crossings) in enumerate(pr_events):
            for row, cross_direction in crossings:
                if cross_direction != direction:
                    continue
                for tid in {under[row - 1], under[row]}:
                    target = gt_events.get(tid)
                    if target and target[0] == direction and any(
                        d == direction and abs(gt[g, 0] - pr[row, 0]) <= 5 for g, d in target[3]
                    ):
                        possible[i, gt_column[tid]] = 1
        tp = int(possible[linear_sum_assignment(possible, maximize=True)].sum())
        scores[threshold] = dict(tp=tp, fp=len(pr_events) - tp, fn=len(gt_events) - tp,
                                 f1=f1(tp, len(pr_events), len(gt_events)))
    return scores


@lru_cache(maxsize=2)
def _reference_rows(hive):
    """The published §95 per-seed table; read only its two existing result tables."""
    text = (ROOT / "docs" / "experiment_log.md").read_text(encoding="utf-8")
    section = text.split("\n## 95. ", 1)[1].split("\n## 96. ", 1)[0]
    table = section.split(f"留 {hive}(評 {hive}-head;", 1)[1].split("| seed |", 1)[1].split("| 平均(全距)", 1)[0]
    rows = {}
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.strip("|").replace("−", "-").split("|")]
        if cells[0].isdigit():
            seed = int(cells[0])
            nums = re.fullmatch(r"([.\d]+)\((\d+) / (\d+) / (\d+)\)", cells[3])
            if not nums:
                raise ValueError(f"Unrecognized §95 event counts for {hive} seed {seed}")
            rows[seed] = (float(cells[1]), int(cells[2].split("(")[0]), *(int(n) for n in nums.groups()[1:]))
    if set(rows) != set(range(9)):
        raise ValueError(f"Incomplete §95 reference table for hive {hive}")
    return rows


def check_baseline(hive, seed, gt_counts, pred_counts, scores):
    """Validate each P0's 30-offset error, signed bias, and exact official TP/FP/FN."""
    delta = int((pred_counts - gt_counts).sum())
    assert delta % 30 == 0
    official = scores[0.5]
    actual = (round(window_error(gt_counts, pred_counts), 2), delta // 30,
              official["tp"], official["fp"], official["fn"])
    expected = _reference_rows(int(hive))[int(seed)]
    if actual != expected:
        raise ValueError(f"P0 differs from §95 for {hive} seed {seed}: {actual} != {expected}")


if __name__ == "__main__":
    # A return excursion does not count, the last same-direction crossing
    # timestamps an in event, and IoU .4 is accepted only by the loose scorer.
    from beetrack.evaluate import match_events

    polygon = [(0, 0), (10, 0), (10, 30), (0, 30)]
    rows = np.array([[f, 1, x, 5, 2, 7] for f, x in zip((747, 748, 749, 750), (14, 6, 14, 6))], float)
    pred = rows.copy()
    pred[:, 3] += 3
    assert np.array_equal(event_times(rows, polygon), [[750, 1]])
    assert vector_of_counts(rows, polygon, 790).sum(axis=0).tolist() == [30, 0]
    a = counts_from_events(np.array([[750, 1]]), 790)
    b = counts_from_events(np.array([[751, 1]]), 790)
    assert np.isclose(window_error(a, b), 200 / 30)
    result = event_scores(rows, pred, polygon)
    assert result[0.5]["tp"] == match_events(rows, pred, polygon)["tp"] == 0
    assert result[0.3]["tp"] == 1
    assert event_scores(rows, rows, polygon)[0.5]["tp"] == match_events(rows, rows, polygon)["tp"] == 1
    print("Synthetic window and event checks passed.")
