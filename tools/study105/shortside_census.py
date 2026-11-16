"""Conditional short-side census. The command-line self-check uses synthetic arrays only.

Provide the repo root on PYTHONPATH. P0 and the
first pass are MOT xywh; detector caches are [frame,x1,y1,x2,y2,confidence].
"""
from collections import Counter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from beetrack.evaluate import by_track, zone_tracks
from beetrack.track import nms
from tools.study104.event_overlap import score_arm
from tools.study104.track_reverse import first_pass


def _groups(frames):
    keys = np.argsort(frames, kind="stable")
    values, starts = np.unique(frames[keys], return_index=True)
    return dict(zip(values, np.split(keys, starts[1:])))


def _xyxy(mot):
    return np.c_[mot[:, 2:4], mot[:, 2:4] + mot[:, 4:6]]


def _inside(boxes, poly):
    centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                     for x, y in centers], dtype=bool)


def _matrix(a, b):
    overlap = np.maximum(0, np.minimum(a[:, None, 2:], b[:, 2:]) -
                         np.maximum(a[:, None, :2], b[:, :2])).prod(axis=2)
    union = ((a[:, 2:] - a[:, :2]).prod(axis=1)[:, None]
             + (b[:, 2:] - b[:, :2]).prod(axis=1)[None, :] - overlap)
    return np.divide(overlap, union, out=np.zeros_like(overlap, dtype=float), where=union > 0)


def _match(iou):
    """Maximize eligible pair count first, then total IoU; all valid sides compete."""
    width = min(iou.shape)
    result = np.full(iou.shape[0], -1, dtype=int)
    if width:
        permitted = iou >= .3
        rows, cols = linear_sum_assignment(np.where(permitted, width + 1 + iou, 0), maximize=True)
        accepted = permitted[rows, cols]
        result[rows[accepted]] = cols[accepted]
    return result


def _audit_frame(gt_boxes, cached, first_boxes, poly):
    """One row per GT bee, describing the stages separately; no numerical score oracle."""
    gt_side = _inside(gt_boxes, poly)
    cached_side = _inside(cached[:, :4], poly)
    first_side = _inside(first_boxes, poly)
    raw_iou = _matrix(gt_boxes, cached[:, :4])
    survivors = nms(cached.astype(np.float32), .5) if len(cached) else np.empty(0, int)
    kept_iou = raw_iou[:, survivors]
    raw_owner = _match(raw_iou)
    kept_owner = _match(kept_iou)
    first_owner = _match(_matrix(gt_boxes, first_boxes))
    result = []
    for i, side in enumerate(gt_side):
        raw_ok = raw_iou[i] >= .3
        right = raw_ok & (cached_side == side)
        high = right & (cached[:, 4] >= .25)
        kept_high = high[survivors]
        r, k, t = raw_owner[i], kept_owner[i], first_owner[i]
        raw_hit = bool(r >= 0 and cached_side[r] == side)
        kept_hit = bool(k >= 0 and cached_side[survivors[k]] == side)
        observed = bool(t >= 0 and first_side[t] == side)
        if observed:
            cause = "observed"
        elif not raw_ok.any():
            cause = "absent_from_cache"
        elif not right.any():
            cause = "wrong_side_only"
        elif not raw_hit or (right[survivors].any() and not kept_hit):
            cause = "assignment_competition"
        elif kept_high.any():
            cause = "post_nms_high"
        elif high.any():
            cause = "high_removed_nms"
        else:
            cause = "low_only"
        result.append(dict(reason=cause, raw_hit=raw_hit, nms_hit=kept_hit, first_hit=observed,
                           any_raw=bool(raw_ok.any()), any_correct_side=bool(right.any()),
                           high_raw=bool(high.any()), high_kept=bool(kept_high.any()),
                           first_box=tuple(map(float, first_boxes[t])) if observed else None))
    return result


def _short_shape(frames, inside, crosses, start, nframes):
    """Reject codes are mutually exclusive in this order."""
    n_in, n_out = sum(inside), len(inside) - sum(inside)
    if min(n_in, n_out) > 2:
        return "long_both_sides", None
    if max(n_in, n_out) < 3:
        return "short_both_sides", None
    short = np.flatnonzero(inside == (n_in < n_out))
    if len(crosses) != 1:
        return "repeated_crossing", None
    cross = crosses[0][0] - start
    if frames[cross] != frames[cross-1] + 1 or np.any(np.diff(frames[short]) != 1):
        return "gap_at_short_side", None
    if not (short[-1] == cross-1 or short[0] == cross):
        raise AssertionError("Single crossing does not abut its full short side")
    if frames[short[0]] == 1 or frames[short[-1]] == nframes:
        return "clip_edge_short_side", None
    return "eligible_geometry", short


def census(gt, p0, detections, polygon, nframes, first_rows=None):
    """Return frozensets eligible_ids/eligible_missed_ids plus auditable event records.

    'eligible' is the intervention population; it currently changes no boxes.
    Prepared GT runs are split by the official ID-reuse rule. first_rows=None
    reruns only P0's first pass on the supplied detections.
    """
    gt, p0 = by_track(gt), by_track(p0)
    poly = np.asarray(polygon, np.float32)
    gt_boxes = _xyxy(gt)
    gt_side = _inside(gt_boxes, poly)
    events = zone_tracks(gt, poly)
    official = score_arm(gt, p0, poly, .5)
    matched = set(official.matches.values())
    counts = Counter()
    excluded_events = {}
    selected = []
    needed = set()
    for tid, (direction, a, b, crosses) in events.items():
        if not direction:
            counts["cross_and_return" if crosses else "never_crossed"] += 1
            continue
        reason, idx = _short_shape(gt[a:b, 0], gt_side[a:b], crosses, a, nframes)
        counts[reason] += 1
        if idx is not None:
            selected.append((tid, direction, a, b, idx+a, crosses[0][0]))
            needed.update(gt[a:b, 0].astype(int))
        else:
            excluded_events[tid] = reason
    gt_by_frame = _groups(gt[:, 0])
    raw_by_frame = _groups(detections[:, 0])
    if first_rows is None:
        first_rows = first_pass(detections, nframes)
    first_by_frame = _groups(first_rows[:, 0])
    first_boxes = _xyxy(first_rows)
    audit = {}
    empty = np.empty(0, dtype=int)
    for frame in sorted(needed):
        g = gt_by_frame[frame]
        c = detections[raw_by_frame.get(frame, empty), 1:6]
        t = first_boxes[first_by_frame.get(frame, empty)]
        audit.update(zip(g, _audit_frame(gt_boxes[g], c, t, poly)))
    entries = {}
    eligible = set()
    for tid, direction, a, b, idx, crossing in selected:
        frames = tuple(int(gt[j, 0]) for j in idx)
        is_short = set(idx)
        anchors = [j for j in range(a, b) if j not in is_short and audit[j]["first_hit"]]
        anchor = min(anchors, key=lambda j: (min(abs(gt[j, 0]-f) for f in frames), gt[j, 0])) if anchors else None
        no_raw = all(not audit[j]["any_raw"] for j in idx)
        qualifies = anchor is not None and no_raw and all(not audit[j]["first_hit"] for j in idx)
        if qualifies:
            eligible.add(tid)
        entries[tid] = dict(direction=direction, short_role="origin" if idx[0] == a else "destination",
                            crossing_frame=int(gt[crossing, 0]), short_frames=frames,
                            short_audit=tuple(audit[j] for j in idx), official_hit=tid in matched,
                            eligible=qualifies, anchor_frame=None if anchor is None else int(gt[anchor, 0]),
                            anchor_box=None if anchor is None else audit[anchor]["first_box"])
    return dict(eligible_ids=frozenset(eligible), eligible_missed_ids=frozenset(eligible - matched),
                gt_event_count=len(official.gt), geometry_counts=dict(counts), events=entries,
                excluded_events=excluded_events, gt_directions=dict(official.gt),
                official_matches=dict(official.matches), pred_event_count=len(official.events))


def self_check():
    polygon = np.array([[0, 0], [10, 0], [10, 30], [0, 30]], np.float32)
    # Last single frame outside is followed by three inside. No input cache
    # row at frame 2; the long side has an observed box.
    gt = np.array([[f, 7, x, 3, 2, 2] for f,x in [(2,12),(3,7),(4,6),(5,5)]],float)
    p0 = gt[1:].copy()
    p0[:,1] = 99
    raw = np.c_[p0[:,0], _xyxy(p0), np.ones(len(p0))]
    first = np.c_[p0, np.ones(len(p0))]
    r = census(gt, p0, raw, polygon, 8, first)
    assert r["eligible_ids"] == r["eligible_missed_ids"] == {7}
    ev = r["events"][7]
    assert ev["short_role"] == "origin" and ev["crossing_frame"] == 3 and ev["anchor_frame"] == 3
    assert ev["short_audit"][0]["reason"] == "absent_from_cache"
    assert census(gt, gt, raw, polygon, 8, first)["eligible_missed_ids"] == set()
    backward = gt.copy()
    backward[:,0] = 9-backward[:,0]
    reverse_raw, reverse_first = raw.copy(), first.copy()
    reverse_raw[:,0], reverse_first[:,0] = 9-raw[:,0], 9-first[:,0]
    out = census(backward, reverse_first[:,:6], reverse_raw, polygon, 8, reverse_first)
    assert out["events"][7]["short_role"] == "destination" and out["events"][7]["direction"] == "out"
    assert _short_shape(np.array([2,3,4,5,6,7]), np.array([1,0,1,0,0,0],bool),[(1,"out"),(2,"in"),(3,"out")],0,8)[0] == "repeated_crossing"
    assert _short_shape(np.array([2,4,5,6]), np.array([0,1,1,1],bool),[(1,"in")],0,8)[0] == "gap_at_short_side"
    assert _short_shape(np.array([1,2,3,4]), np.array([0,1,1,1],bool),[(1,"in")],0,8)[0] == "clip_edge_short_side"
    # Cardinality wins even when one excellent IoU could crowd out two matches.
    assert _match(np.array([[.3,1.0],[0,.3]])).tolist() == [0,1]
    box = np.array([[7,3,9,5]],float)
    empty = np.empty((0,4))
    for conf, reason in [(.15,"low_only"),(.8,"post_nms_high")]:
        assert _audit_frame(box, np.c_[box,[conf]], empty, polygon)[0]["reason"] == reason
    shared = _audit_frame(np.vstack([box,box]), np.c_[box,[.8]], empty, polygon)
    assert {r["reason"] for r in shared} == {"post_nms_high","assignment_competition"}
    edge_box = np.array([[9.1,3,11.1,5]], float)
    crossed = np.array([[8.7,3,10.7,5,.9]], float)
    assert _audit_frame(edge_box, crossed, empty, polygon)[0]["reason"] == "wrong_side_only"
    # Higher-score wrong-side box suppresses a valid high-score right-side box.
    suppressed = np.vstack([crossed,np.c_[edge_box,[.7]]])
    assert _audit_frame(edge_box, suppressed, empty, polygon)[0]["reason"] == "high_removed_nms"
    print("Passed short-side shape, official-hit, actual observation, assignment and cache-stage checks.")


if __name__ == "__main__":
    self_check()
