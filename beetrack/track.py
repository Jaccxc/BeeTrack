"""Trackers: Ultralytics ByteTrack (baseline) and BeeTracker (ours, motion-adaptive association).
run on cached detections (fair comparison: the detector output is identical for every tracker):
  uv run python -m beetrack.track --dets y11s_640 --split val --tracker bytetrack
  uv run python -m beetrack.track --dets y11s_640 --split val --tracker ours --set buffer=0.3 track_buffer=60 --name ours_b03
writes runs/trackers/<split>/<name>/data/<seq>.txt (MOTChallenge) + <name>/timing.json

BeeTracker changes three things inside ByteTrack, each switchable for the ablation:
  buffer  - buffered IoU: both boxes are enlarged by `buffer` x their size before IoU, so a fast bee whose
            boxes no longer overlap between frames can still match (C-BIoU idea)
  grow    - the track's buffer also grows by `grow` per frame it has been lost (search area widens with
            uncertainty) and with its own speed (|v| / box size, capped)
  damp    - a lost track's velocity decays by this factor per frame, so it does not fly off while hidden
  q_vel   - Kalman velocity process-noise weight (ByteTrack: 1/160); bees accelerate harder than pedestrians
plus a longer track_buffer (frames a lost track is kept).
"""
import argparse
import ast
import json
import time
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils import matching
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH

from .common import RUNS, split_specs, write_mot

BYTETRACK = dict(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25, track_buffer=30,
                 match_thresh=0.8, fuse_score=True)                 # ultralytics/cfg/trackers/bytetrack.yaml
# tuned on val only (runs/eval/val/sweep_*.json); second-stage BIoU and interpolation did not help, so they stay off
OURS = dict(BYTETRACK, new_track_thresh=0.5, track_buffer=60, buffer=0.1, grow=0.1, speed=0.5, damp=0.8, q_vel=1 / 80, second=False)
# interp (any tracker): fill gaps <= interp frames inside a track by linear interpolation (offline post-processing)


class Dets:
    """Results-like view of an (N, 5) array x1, y1, x2, y2, score, as BYTETracker.update expects."""

    def __init__(self, d):
        self.d = np.asarray(d, dtype=np.float32).reshape(-1, 5)

    conf = property(lambda s: s.d[:, 4])
    cls = property(lambda s: np.zeros(len(s.d), np.float32))
    xyxy = property(lambda s: s.d[:, :4])
    xywh = property(lambda s: np.c_[(s.d[:, 0] + s.d[:, 2]) / 2, (s.d[:, 1] + s.d[:, 3]) / 2, s.d[:, 2] - s.d[:, 0], s.d[:, 3] - s.d[:, 1]])

    def __len__(self):
        return len(self.d)

    def __getitem__(self, m):
        return Dets(self.d[m])


def iou_matrix(a, b):
    """Pairwise IoU of (n, 4) and (m, 4) xyxy boxes."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def enlarge(xyxy, r):
    w, h = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
    return np.c_[xyxy[:, 0] - r * w, xyxy[:, 1] - r * h, xyxy[:, 2] + r * w, xyxy[:, 3] + r * h]


class BeeTracker(BYTETracker):
    def get_kalmanfilter(self):
        kf = KalmanFilterXYAH()
        kf._std_weight_velocity = self.args.q_vel
        return kf

    def biou_dists(self, tracks, detections):
        """1 - buffered IoU; the track's buffer grows with its speed and with the frames it has been lost."""
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)), np.float32)
        a = np.array([t.xyxy for t in tracks])
        b = np.array([d.xyxy for d in detections])
        lost = np.array([self.frame_id - t.end_frame - 1 for t in tracks], np.float32).clip(0, 10)
        v = np.array([np.hypot(*t.mean[4:6]) / max(t.mean[3], 1.0) for t in tracks])   # speed in box heights / frame
        r = self.args.buffer + self.args.grow * lost + self.args.speed * np.minimum(v, 1.0)
        return 1 - iou_matrix(enlarge(a, r), enlarge(b, np.full(len(b), self.args.buffer)))

    def get_dists(self, tracks, detections):
        dists = self.biou_dists(tracks, detections)
        return matching.fuse_score(dists, detections) if self.args.fuse_score and dists.size else dists

    def _second_association(self, strack_pool, u_track, detections_second, activated, refind, lost):
        """ByteTrack's low-score stage, with buffered IoU instead of plain IoU when args.second is on."""
        if not self.args.second:
            return super()._second_association(strack_pool, u_track, detections_second, activated, refind, lost)
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        u = list(range(len(r_tracked)))
        if r_tracked and detections_second:
            matches, u, _ = matching.linear_assignment(self.biou_dists(r_tracked, detections_second), thresh=0.5)
            self._apply_matches(matches, r_tracked, detections_second, activated, refind)
        for it in u:
            r_tracked[it].mark_lost()
            lost.append(r_tracked[it])

    def multi_predict(self, tracks):
        for t in tracks:
            if t.state != TrackState.Tracked and t.mean is not None:
                t.mean[4:6] *= self.args.damp
        STrack.shared_kalman._std_weight_velocity = self.args.q_vel   # shared filter used by STrack.multi_predict
        STrack.multi_predict(tracks)


class Tracker:
    """Common interface: update(N x 5 xyxy+score) -> rows x1, y1, x2, y2, id, score."""

    def __init__(self, name="ours", **overrides):
        base = OURS if name == "ours" else BYTETRACK
        self.args = SimpleNamespace(**{**base, **overrides})
        STrack.shared_kalman._std_weight_velocity = getattr(self.args, "q_vel", 1 / 160)
        self.t = (BeeTracker if name == "ours" else BYTETracker)(self.args)

    def update(self, dets):
        out = self.t.update(Dets(dets))
        return out[:, :6] if len(out) else np.zeros((0, 6), np.float32)


def make_tracker(name, frame_rate=25, **overrides):
    return Tracker(name, **overrides)


def run_sequence(dets, n, tracker, step=1):
    """Track one sequence of cached detections (frame, x1, y1, x2, y2, score); every `step`-th frame only.
    Output frames are renumbered 1, 2, ... over the kept frames. Returns MOT rows and tracker seconds."""
    rows, secs = [], 0.0
    for k, f in enumerate(range(1, n + 1, step), start=1):
        d = dets[dets[:, 0] == f, 1:6]
        t0 = time.perf_counter()
        out = tracker.update(d)
        secs += time.perf_counter() - t0
        rows += [[k, tid, x1, y1, x2 - x1, y2 - y1, s] for x1, y1, x2, y2, tid, s in out]
    a = tracker.args
    if getattr(a, "stitch", 0):
        rows = stitch(rows, a.stitch, getattr(a, "stitch_dist", 2.0))
    if getattr(a, "min_len", 0) or getattr(a, "min_score", 0):
        rows = drop_short(rows, getattr(a, "min_len", 0), getattr(a, "min_score", 0.0))
    gap = getattr(a, "interp", 0)
    return (interpolate(rows, gap) if gap else rows), secs


def drop_short(rows, min_len, min_score):
    """Remove tracks shorter than min_len boxes whose mean score is below min_score (min_score 0: length alone).
    Offline post-processing (a live system would hold a track's output for min_len frames)."""
    if not rows:
        return rows
    a = np.array(rows, dtype=np.float64)
    ids, inv, cnt = np.unique(a[:, 1], return_inverse=True, return_counts=True)
    mean = np.bincount(inv, weights=a[:, 6]) / cnt
    bad = (cnt < min_len) & (mean < min_score if min_score else True)
    return [r for r, b in zip(rows, bad[inv]) if not b]


def stitch(rows, max_gap, max_dist):
    """Tracklet stitching: a track that starts <= max_gap frames after another one ended, within max_dist box
    heights of that track's last centre, continues its ID (greedy, closest first). Needs max_gap frames of delay."""
    if not rows:
        return rows
    a = np.array(rows, dtype=np.float64)
    ends, starts = {}, {}
    for tid in np.unique(a[:, 1]):
        t = a[a[:, 1] == tid]
        e, s = t[t[:, 0].argmax()], t[t[:, 0].argmin()]
        ends[tid] = (e[0], e[2] + e[4] / 2, e[3] + e[5] / 2, e[5])
        starts[tid] = (s[0], s[2] + s[4] / 2, s[3] + s[5] / 2)
    cand = []
    for i, (fe, xe, ye, he) in ends.items():
        for j, (fs, xs, ys) in starts.items():
            if i != j and 0 < fs - fe <= max_gap:
                d = np.hypot(xs - xe, ys - ye) / max(he, 1.0)
                if d <= max_dist:
                    cand.append((d, i, j))
    parent, used_end, used_start = {}, set(), set()
    for d, i, j in sorted(cand):
        if i not in used_end and j not in used_start:
            parent[j] = i
            used_end.add(i)
            used_start.add(j)

    def root(t):
        while t in parent:
            t = parent[t]
        return t
    for r in rows:
        r[1] = root(r[1])
    return rows


def interpolate(rows, max_gap):
    """Linearly fill missing frames (gap <= max_gap) between two boxes of the same track."""
    if not rows:
        return rows
    a = np.array(rows, dtype=np.float64)
    extra = []
    for tid in np.unique(a[:, 1]):
        t = a[a[:, 1] == tid]
        t = t[np.argsort(t[:, 0])]
        for p, q in zip(t[:-1], t[1:]):
            g = int(q[0] - p[0])
            if 1 < g <= max_gap + 1:
                for j in range(1, g):
                    w = j / g
                    extra.append([p[0] + j, tid, *(p[2:6] * (1 - w) + q[2:6] * w), min(p[6], q[6])])
    return rows + extra


def _check_interpolate():
    r = interpolate([[1, 7, 0, 0, 10, 10, 0.9], [4, 7, 30, 0, 10, 10, 0.8]], 5)
    assert [x[2] for x in r[2:]] == [10.0, 20.0] and len(r) == 4, r
    assert len(interpolate([[1, 7, 0, 0, 1, 1, 1], [9, 7, 0, 0, 1, 1, 1]], 5)) == 2   # gap too long: untouched



def _check_stitch():
    r = stitch([[1, 1, 0, 0, 10, 10, 1], [2, 1, 5, 0, 10, 10, 1], [5, 2, 12, 0, 10, 10, 1], [5, 3, 500, 0, 10, 10, 1]], 10, 2.0)
    assert [x[1] for x in r] == [1, 1, 1, 3], r


def _check_drop_short():
    r = [[1, 1, 0, 0, 1, 1, 0.3], [1, 2, 0, 0, 1, 1, 0.9], [2, 3, 0, 0, 1, 1, 0.3], [3, 3, 0, 0, 1, 1, 0.3]]
    assert [x[1] for x in drop_short(r, 2, 0.5)] == [2, 3, 3]
    assert [x[1] for x in drop_short(r, 2, 0)] == [3, 3]


_check_interpolate()
_check_stitch()
_check_drop_short()


def parse_set(items):
    return {k: ast.literal_eval(v) for k, v in (it.split("=") for it in items or [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="y11s_640", help="detection cache tag under runs/dets/")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--tracker", default="ours", choices=["ours", "bytetrack"])
    ap.add_argument("--set", nargs="*", help="parameter overrides, e.g. buffer=0 track_buffer=30")
    ap.add_argument("--step", type=int, default=1, help="keep every step-th frame (frame-drop robustness)")
    ap.add_argument("--name", help="output name, default <tracker>")
    a = ap.parse_args()
    over = parse_set(a.set)
    name = a.name or a.tracker
    osplit = a.split if a.step == 1 else f"{a.split}_step{a.step}"   # matches the GT folder prep.py writes
    secs = frames = 0
    for ev, _, s0, s1 in split_specs(a.split):
        dets = np.load(RUNS / "dets" / a.dets / a.split / f"{ev}.npy")
        rows, t = run_sequence(dets, s1 - s0 + 1, Tracker(a.tracker, **over), a.step)
        write_mot(RUNS / "trackers" / osplit / name / "data" / f"{ev}.txt", rows)
        secs, frames = secs + t, frames + len(range(1, s1 - s0 + 2, a.step))
    info = dict(tracker=a.tracker, dets=a.dets, step=a.step, params={**(OURS if a.tracker == "ours" else BYTETRACK), **over},
                tracker_ms_per_frame=1000 * secs / max(frames, 1))
    (RUNS / "trackers" / osplit / name / "timing.json").write_text(json.dumps(info, indent=1))
    print(f"{name}: {frames} frames, tracker {info['tracker_ms_per_frame']:.2f} ms/frame")


if __name__ == "__main__":
    main()
