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
OURS = dict(BYTETRACK, new_track_thresh=0.5, track_buffer=60, buffer=0.1, grow=0.1, speed=0.5, damp=0.8, q_vel=1 / 80, second=False,
            flight=0, fly_tol=1.0, fly_n=2, fly_min=0.5, link_gap=2, link_cos=0.8)   # FlightLink, off by default: flight = max jump in body lengths
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
    """Pairwise IoU of (n, 4) and (m, 4) xyxy boxes; b may be (n, m, 4): its own version of the m boxes for every a."""
    a, b = a[:, None], b if b.ndim == 3 else b[None]
    inter = np.prod(np.clip(np.minimum(a[..., 2:], b[..., 2:]) - np.maximum(a[..., :2], b[..., :2]), 0, None), axis=-1)
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def enlarge(xyxy, r):
    """Boxes grown by r x their size on every side. r: scalar, (m,) per box, or (n, 1) -> (n, m, 4), every box at n ratios."""
    wh = (xyxy[..., 2:] - xyxy[..., :2]) * np.asarray(r)[..., None]
    return np.concatenate([xyxy[..., :2] - wh, xyxy[..., 2:] + wh], -1)


class BeeTracker(BYTETracker):
    def __init__(self, args):
        super().__init__(args)
        self.late = []   # rows frame, x1, y1, x2, y2, id, score: boxes seen before their track was confirmed (args.retro)

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
        # sym: the detection grows as much as the track. Otherwise a perfectly placed detection reaches at most
        # ((1 + 2 buffer) / (1 + 2 r))^2 IoU, so a fast or long-lost track can match nothing (r 0.6 -> 0.30, r 1.1 -> 0.14)
        b = enlarge(b, r[:, None] if getattr(self.args, "sym", 0) else self.args.buffer)
        iou = iou_matrix(enlarge(a, r), b)
        if getattr(self.args, "obs", 0):   # stay-or-go: a bee that stops dead is still at its last detection, not where constant velocity puts it
            iou = np.maximum(iou, iou_matrix(enlarge(np.array([getattr(t, "last", t.xyxy) for t in tracks]), r), b))
        return 1 - iou

    def get_dists(self, tracks, detections):
        dists = self.biou_dists(tracks, detections)
        return matching.fuse_score(dists, detections) if self.args.fuse_score and dists.size else dists

    def _second_association(self, strack_pool, u_track, detections_second, activated, refind, lost):
        """ByteTrack's low-score stage, with buffered IoU instead of plain IoU when args.second is on."""
        self._lost_now = lost   # this frame's newly lost tracks: _hop takes a track out again when it finds its bee
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

    # --- FlightLink (args.flight > 0): a flying bee moves several body lengths per frame, so its boxes never overlap and
    # ByteTrack deletes the new track before it is confirmed. A new track may instead continue by centre distance, and is
    # confirmed once (a) three points lie on a line, or (b) two points extrapolate back onto a track that was just lost
    # (take-off): then it keeps that track's ID. A new slow track that appears where a lost flyer was heading (landing)
    # takes the flyer's ID the same way.
    def update(self, results, img=None, feats=None):
        out = super().update(results, img, feats)
        a = self.args
        for t in self.tracked_stracks:
            if t.frame_id != self.frame_id:
                continue
            t.last = results.xyxy[int(t.idx)]
            if getattr(a, "flight", 0):
                t.pts = (getattr(t, "pts", []) + [(self.frame_id, *t.mean[:2])])[-2:]
                t.n_obs = getattr(t, "n_obs", 0) + 1
            if getattr(a, "retro", 0):   # boxes a track had before it was confirmed are emitted once it is (self.late)
                if not t.is_activated:
                    t.pend = getattr(t, "pend", []) + [[self.frame_id, *results.xyxy[int(t.idx)], t.score]]
                elif getattr(t, "pend", None):
                    self.late += [[f, *b, t.track_id, s] for f, *b, s in t.pend]
                    t.pend = None
        return out

    def _link(self, t, p, f, v, removed):
        """Give track t (first seen at point p, frame f) the ID of the lost track it continues. v: t's velocity for a
        take-off (t is the flyer); None for a landing, where the lost track's own last velocity is used."""
        a, s, best = self.args, float(np.sqrt(np.prod(t.tlwh[2:]))), None
        for o in self.lost_stracks:
            pts = getattr(o, "pts", [])
            if o.state != TrackState.Lost or not pts or (v is None and len(pts) < 2):
                continue
            gap = f - pts[-1][0]
            vv = v if v is not None else (np.subtract(pts[-1][1:], pts[-2][1:]) / max(pts[-1][0] - pts[-2][0], 1))
            sp, d = float(np.hypot(*vv)), np.subtract(p, pts[-1][1:])
            dist = float(np.hypot(*d))
            if not 0 < gap <= a.link_gap or sp < a.fly_min * s or dist > 1.5 * sp * gap + s:
                continue
            if dist > 0.5 * s and d @ vv / (dist * sp) < a.link_cos:   # the jump must point along the flight
                continue
            if best is None or dist < best[0]:
                best = (dist, o)
        if best:
            t.track_id = best[1].track_id
            best[1].mark_removed()
            removed.append(best[1])
        return bool(best)

    def _hop(self, D, activated):
        """args.hop > 0: a confirmed track that lost its box <= hop_gap frames ago takes a leftover detection within `hop`
        body lengths per frame of where it was last detected or of where the filter puts it (a hop or a short dash moves a
        bee further than any overlap test allows). Returns the detections that are still free."""
        a = self.args
        T = [t for t in self.tracked_stracks + self.lost_stracks if t.state == TrackState.Lost and hasattr(t, "last")
             and self.frame_id - t.end_frame <= getattr(a, "hop_gap", 1)]
        cost = np.full((len(T), len(D)), 2.0, np.float32)
        for i, t in enumerate(T):
            s, c0 = float(np.sqrt(np.prod(t.tlwh[2:]))), (t.last[:2] + t.last[2:]) / 2
            for j, d in enumerate(D):
                c, (w, h) = d.tlwh[:2] + d.tlwh[2:] / 2, d.tlwh[2:]
                if 1 / 3 < w * h / s**2 < 3:
                    cost[i, j] = min(np.hypot(*(c - c0)), np.hypot(*(c - t.mean[:2]))) / (a.hop * s * (self.frame_id - t.end_frame))
        m, _, free = matching.linear_assignment(cost, thresh=1.0)
        for i, j in m:
            T[i].re_activate(D[j], self.frame_id)
            T[i].how = "hop"
            if T[i] in self._lost_now:
                self._lost_now.remove(T[i])
            activated.append(T[i])
        return [D[j] for j in free]

    def _unconfirmed_association(self, unconfirmed, u_detection, detections, activated, removed):
        a = self.args
        fly, hop, dup = (getattr(a, k, 0) for k in ("flight", "hop", "dup"))
        if dup and 0 < len(u_detection) < len(detections):   # a leftover box on top of a box that already has its track is a duplicate
            free = set(u_detection)
            o = iou_matrix(np.array([detections[i].xyxy for i in u_detection]),
                           np.array([d.xyxy for k, d in enumerate(detections) if k not in free])).max(1)
            u_detection = [i for i, x in zip(u_detection, o) if x < dup]
        if not fly and not hop:
            return super()._unconfirmed_association(unconfirmed, u_detection, detections, activated, removed)
        detections = [detections[i] for i in u_detection]
        matches, u_unc, u_det = matching.linear_assignment(self.get_dists(unconfirmed, detections), thresh=0.7)
        for it, idet in matches:                       # ByteTrack's own confirmation (overlapping boxes)
            t = unconfirmed[it]
            first = getattr(t, "pts", None) and getattr(t, "n_obs", 0) == 1
            t.update(detections[idet], self.frame_id)
            if first and self._link(t, t.pts[0][1:], t.pts[0][0], None, removed):
                t.how = "land"   # how a track was confirmed, when not by plain overlap (diagnostics only)
            activated.append(t)
        U, D = [unconfirmed[i] for i in u_unc], [detections[j] for j in u_det]
        if hop:
            D = self._hop(D, activated)
        cost = np.full((len(U), len(D)), 2.0, np.float32)
        for i, t in enumerate(U):
            if not fly or not getattr(t, "pts", None):
                continue
            c, s, v = np.array(t.pts[-1][1:]), float(np.sqrt(np.prod(t.tlwh[2:]))), getattr(t, "fly_v", None)
            gate = a.flight * s if v is None else a.fly_tol * s + 0.5 * np.hypot(*v)
            for j, d in enumerate(D):
                w, h = d.tlwh[2:]
                if d.score >= getattr(a, "fly_score", 0) and 1 / 3 < w * h / s**2 < 3:
                    cost[i, j] = np.hypot(*(d.tlwh[:2] + d.tlwh[2:] / 2 - c - (0 if v is None else v))) / gate
        fm, fu, fd = matching.linear_assignment(cost, thresh=1.0)
        for i, j in fm:
            t, d = U[i], D[j]
            t.fly_v = d.tlwh[:2] + d.tlwh[2:] / 2 - np.array(t.pts[-1][1:])
            t.mean, t.covariance = t.kalman_filter.initiate(t.convert_coords(d.tlwh))   # a jump this size would wreck a Kalman update
            t.mean[4:6] = t.fly_v
            t.frame_id, t.score, t.cls, t.idx, t.state = self.frame_id, d.score, d.cls, d.idx, TrackState.Tracked
            t.tracklet_len += 1
            first = t.pts[0] if t.n_obs == 1 else None
            if first and self._link(t, first[1:], first[0], t.fly_v, removed):
                t.is_activated, t.how = True, "takeoff"
            elif t.n_obs >= a.fly_n:                    # free chain: fly_n + 1 points on a line
                t.is_activated, t.how = True, "chain"
            activated.append(t)
        for i in fu:
            U[i].mark_removed()
            removed.append(U[i])
        return list(range(len(fd))), [D[j] for j in fd]

    def _init_new_tracks(self, u_detection, detections, activated, refind=None):
        n = len(activated)
        super()._init_new_tracks(u_detection, detections, activated, refind)
        if getattr(self.args, "all", 0):   # tracklet mode: a new track is output from its first box, confirmed or not
            for t in activated[n:]:
                t.is_activated = True

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
        if not len(out):
            return np.zeros((0, 6), np.float32)
        if getattr(self.args, "raw", 0):   # the matched detection itself: the Kalman-smoothed box lags behind a bee that accelerates
            out[:, :4] = np.asarray(dets, np.float32).reshape(-1, 5)[out[:, 7].astype(int), :4]
        return out[:, :6]


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
    rows += [[f, tid, x1, y1, x2 - x1, y2 - y1, s] for f, x1, y1, x2, y2, tid, s in getattr(tracker.t, "late", [])]
    rows.sort(key=lambda r: r[0])
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


def _check_flight():
    """A bee sits for 5 frames, then flies 3 body lengths per frame: one ID with FlightLink, lost without it."""
    box = lambda x: [[x, 100, x + 20, 120, 0.9]]
    seq = [box(100)] * 5 + [box(100 + 60 * k) for k in range(1, 5)]
    ids = lambda **kw: [[int(r[4]) for r in t.update(np.array(d, np.float32))] for t in [Tracker("ours", **kw)] for d in seq]
    assert ids(flight=5)[-2:] == [[1], [1]], ids(flight=5)
    assert ids()[-2:] == [[], []], ids()


_check_interpolate()
_check_flight()
_check_stitch()
_check_drop_short()


def parse_set(items):
    return {k: ast.literal_eval(v) for k, v in (it.split("=") for it in items or [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="y11s_640", help="detection cache tag under runs/dets/")
    ap.add_argument("--split", default="val", choices=["val", "test", "dev"])
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
