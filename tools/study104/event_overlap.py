"""Event-level complementarity of two tracks scored on the original clip clock."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from beetrack.evaluate import by_track, f1, frame_match, zone_tracks


def _maximum(edges):
    """SciPy tie order, identical to the official matcher for sorted track IDs."""
    rows, cols = linear_sum_assignment(edges.astype(int), maximize=True)
    valid = edges[rows, cols]
    return rows[valid], cols[valid]


def _variable_rows(edges, rows, cols):
    """Rows whose matched/unmatched status differs across maximum matchings.

    From an unmatched row with a neighbor, alternate an unmatched edge and a
    matched edge. Flipping such a path exchanges the status of its endpoint
    rows and preserves cardinality. Every status-changing symmetric difference
    between two maximum matchings contains such a path.
    """
    partner = dict(zip(cols, rows))
    seen = set(np.flatnonzero(edges.any(axis=1))) - set(rows)
    queue = deque(seen)
    while queue:
        start = queue.popleft()
        for col in np.flatnonzero(edges[start]):
            next_row = partner.get(col)
            if next_row is not None and next_row not in seen:
                seen.add(next_row)
                queue.append(next_row)
    return seen


@dataclass
class Arm:
    gt: dict
    gt_frames: dict
    events: dict
    matches: dict
    edges: list
    variable_gt: set
    variable_predictions: set


def score_arm(gt, pr, polygon, threshold=0.5):
    """Recover matched event identities as well as the official event score.

    All IDs are the protocol's gap-split IDs. The stored representative frame
    and box describe the last crossing in the event direction. Official TP
    matching still considers *every* crossing in that direction.
    """
    if threshold not in (0.5, 0.3):
        raise ValueError("expected a registered frame-matching IoU threshold")
    gt, pr = by_track(gt), by_track(pr)
    truth = {tid: e for tid, e in zone_tracks(gt, polygon).items() if e[0]}
    preds = {tid: e for tid, e in zone_tracks(pr, polygon).items() if e[0]}
    tids, pids = sorted(truth), sorted(preds)
    columns = {tid: i for i, tid in enumerate(tids)}
    under = frame_match(gt, pr, thr=threshold) if len(gt) and len(pr) else np.full(len(pr), -1)
    edges = np.zeros((len(pids), len(tids)), dtype=bool)
    events = {}
    for i, pid in enumerate(pids):
        direction, _, _, crossings = preds[pid]
        for row, crossing_dir in crossings:
            if crossing_dir != direction:
                continue
            for owner in {under[row - 1], under[row]}:
                target = truth.get(owner)
                if target and target[0] == direction and any(
                    d == direction and abs(gt[g, 0] - pr[row, 0]) <= 5 for g, d in target[3]
                ):
                    edges[i, columns[owner]] = True
            frame, _, x, y, w, h = pr[row, :6]
            events[pid] = (direction, int(frame), (x, y, x + w, y + h))
    gt_frames = {tid: int(gt[next(row for row, d in reversed(e[3]) if d == e[0]), 0])
                 for tid, e in truth.items()}
    return _arm({tid: truth[tid][0] for tid in tids}, gt_frames, events, edges)


def _arm(gt, gt_frames, events, edges):
    tids, pids = sorted(gt), sorted(events)
    rows, cols = _maximum(edges)
    return Arm(gt, gt_frames, events, {pids[i]: tids[j] for i, j in zip(rows, cols)},
               [[pids[i], tids[j]] for i, j in zip(*np.nonzero(edges))],
               {tids[j] for j in _variable_rows(edges.T, cols, rows)},
               {pids[i] for i in _variable_rows(edges, rows, cols)})


def archive_arm(arm):
    """Portable event descriptions and sparse official candidate graph for one IoU."""
    return dict(gt=[[tid, frame if arm.gt[tid] == "in" else -frame] for tid, frame in sorted(arm.gt_frames.items())],
                predictions=[[pid, frame if d == "in" else -frame, *map(float, xyxy)]
                             for pid, (d, frame, xyxy) in sorted(arm.events.items())],
                edges=arm.edges)


def replay_arm(record):
    """Recreate the canonical matching from archived events and valid edges."""
    gt = {tid: "in" if f > 0 else "out" for tid, f in record["gt"]}
    gt_frames = {tid: abs(f) for tid, f in record["gt"]}
    events = {pid: ("in" if f > 0 else "out", abs(f), tuple(box)) for pid, f, *box in record["predictions"]}
    rows = {pid: i for i, pid in enumerate(sorted(events))}
    cols = {tid: j for j, tid in enumerate(sorted(gt))}
    edges = np.zeros((len(rows), len(cols)), dtype=bool)
    for pid, tid in record["edges"]:
        edges[rows[pid], cols[tid]] = True
    return _arm(gt, gt_frames, events, edges)


def _pair_events(forward_events, reverse_events, f_ids, r_ids):
    """One-to-one matching by direction, canonical frame and canonical box."""
    if not f_ids or not r_ids:
        return []
    left, right = [forward_events[p] for p in f_ids], [reverse_events[p] for p in r_ids]
    boxes_f, boxes_r = np.array([e[2] for e in left]), np.array([e[2] for e in right])
    frames_f, frames_r = np.array([e[1] for e in left]), np.array([e[1] for e in right])
    close = np.abs(frames_f[:, None] - frames_r[None, :]) <= 5
    directions_f = np.array([e[0] for e in left])
    directions_r = np.array([e[0] for e in right])
    # Cross-arm IoU >= 1/2, including equality. Compare areas directly to
    # avoid an epsilon in the denominator moving an exact half below threshold.
    low = np.maximum(boxes_f[:, None, :2], boxes_r[None, :, :2])
    high = np.minimum(boxes_f[:, None, 2:], boxes_r[None, :, 2:])
    intersection = np.maximum(high-low, 0).prod(axis=2)
    area_f = (boxes_f[:, 2:]-boxes_f[:, :2]).prod(axis=1)
    area_r = (boxes_r[:, 2:]-boxes_r[:, :2]).prod(axis=1)
    union = area_f[:, None] + area_r[None, :] - intersection
    edges = close & (directions_f[:, None] == directions_r[None, :]) & (union > 0) & (2*intersection >= union)
    fi, ri = _maximum(edges)
    return [(f_ids[i], r_ids[j]) for i, j in zip(fi, ri)]


def union_events(forward_events, reverse_events):
    """Keep every forward event and append reverse events without a forward partner."""
    common = _pair_events(forward_events, reverse_events, sorted(forward_events), sorted(reverse_events))
    paired_reverse = {pid for _, pid in common}
    supplemented = sorted(set(reverse_events) - paired_reverse)
    events = list(forward_events.values()) + [reverse_events[pid] for pid in supplemented]
    times = np.array([(frame, 1 if d == "in" else -1) for d, frame, _ in events], dtype=int).reshape(-1, 2)
    return times, supplemented


def union_summary(forward, reverse, nframes):
    from tools.study103.swap_score import counts_from_events, window_error

    if forward.gt != reverse.gt or forward.gt_frames != reverse.gt_frames:
        raise ValueError("arms must share ground truth events")
    times, added = union_events(forward.events, reverse.events)
    gt_times = np.array([(frame, 1 if forward.gt[tid] == "in" else -1)
                         for tid, frame in forward.gt_frames.items()], dtype=int).reshape(-1, 2)
    score = window_error(counts_from_events(gt_times, nframes), counts_from_events(times, nframes))
    return dict(window=score, added=added, events=times.tolist(), n_in=int((times[:, 1] == 1).sum()),
                n_out=int((times[:, 1] == -1).sum()))


def compare_arms(forward, reverse):
    """Return JSON-ready counts, split by in/out, plus comparator-only net gain."""
    if forward.gt != reverse.gt:
        raise ValueError("arms must share the same GT identities and directions")
    by_direction = {}
    for direction in ("in", "out"):
        gt = {tid for tid, d in forward.gt.items() if d == direction}
        f_true = gt & set(forward.matches.values())
        r_true = gt & set(reverse.matches.values())
        f_false = sorted(pid for pid, e in forward.events.items() if e[0] == direction and pid not in forward.matches)
        r_false = sorted(pid for pid, e in reverse.events.items() if e[0] == direction and pid not in reverse.matches)
        shared_false = len(_pair_events(forward.events, reverse.events, f_false, r_false))
        r_only_true = len(r_true - f_true)
        r_only_false = len(r_false) - shared_false
        by_direction[direction] = dict(
            gt=len(gt), both_true=len(f_true & r_true), forward_only_true=len(f_true - r_true),
            reverse_only_true=r_only_true, both_missed=len(gt - f_true - r_true),
            paired_false=shared_false, forward_only_false=len(f_false) - shared_false,
            reverse_only_false=r_only_false, net=r_only_true-r_only_false,
        )
    total = {key: sum(x[key] for x in by_direction.values()) for key in by_direction["in"]}
    arms = {}
    for name, arm in (("forward", forward), ("reverse", reverse)):
        tp = len(arm.matches)
        arms[name] = dict(tp=tp, fp=len(arm.events)-tp, fn=len(arm.gt)-tp, f1=f1(tp, len(arm.events), len(arm.gt)),
                          variable_gt=len(arm.variable_gt), variable_predictions=len(arm.variable_predictions))
    return dict(total=total, by_direction=by_direction, **arms)


if __name__ == "__main__":
    from beetrack.evaluate import match_events

    polygon = [(0, 0), (10, 0), (10, 100), (0, 100)]
    def crossing(pid, t, y):
        return [[t, pid, 13, y, 2, 2], [t + 1, pid, 5, y, 2, 2]]

    gt = np.array(crossing(1, 10, 5) + crossing(2, 20, 15), dtype=float)
    f = np.array(crossing(10, 10, 5) + crossing(11, 30, 45), dtype=float)
    r = np.array(crossing(20, 20, 15) + crossing(21, 35, 45) + crossing(22, 35, 80), dtype=float)
    for threshold in (0.5, 0.3):
        a, b = (score_arm(gt, x, polygon, threshold) for x in (f, r))
        c = compare_arms(a, b)
        assert c["total"] == dict(gt=2, both_true=0, forward_only_true=1, reverse_only_true=1,
                                  both_missed=0, paired_false=1, forward_only_false=0, reverse_only_false=1, net=0)
        assert c["total"] == c["by_direction"]["in"]
        assert c["reverse"]["tp"] == c["forward"]["tp"] == match_events(gt, f, polygon)["tp"] == 1
        assert compare_arms(a, a)["total"]["forward_only_false"] == 0
        assert compare_arms(replay_arm(archive_arm(a)), replay_arm(archive_arm(b))) == c
        joined = union_summary(a, b, 100)
        assert joined["n_in"] == 4 and joined["n_out"] == 0 and joined["added"] == [20, 22]
    edges = np.array([[1, 1], [1, 0], [0, 1]], dtype=bool)
    rows, cols = _maximum(edges)
    assert _variable_rows(edges, rows, cols) == {0, 1, 2}
    assert not _variable_rows(edges.T, cols, rows)
    assert _pair_events({1: ("in", 10, (0, 0, 2, 2))}, {2: ("in", 15, (0, 0, 1, 2))}, [1], [2]) == [(1, 2)]
    print("Passed complementarity, direction, duplicate false events and tie-ambiguity checks.")
