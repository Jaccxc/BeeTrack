"""Trackers: Ultralytics ByteTrack (baseline) and BeeTracker (ours, motion-adaptive association).
run on cached detections (fair comparison: the detector output is identical for every tracker):
  .venv\\Scripts\\python -m beetrack.track --dets y11s_640 --split val --tracker bytetrack
  .venv\\Scripts\\python -m beetrack.track --dets y11s_640 --split val --tracker ours --set buffer=0.3 track_buffer=60 --name ours_b03
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
OURS = dict(BYTETRACK, track_buffer=60, buffer=0.3, grow=0.1, speed=0.5, damp=0.9, q_vel=1 / 80)


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

    def get_dists(self, tracks, detections):
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)), np.float32)
        a = np.array([t.xyxy for t in tracks])
        b = np.array([d.xyxy for d in detections])
        lost = np.array([self.frame_id - t.end_frame - 1 for t in tracks], np.float32).clip(0, 10)
        v = np.array([np.hypot(*t.mean[4:6]) / max(t.mean[3], 1.0) for t in tracks])   # speed in box heights / frame
        r = self.args.buffer + self.args.grow * lost + self.args.speed * np.minimum(v, 1.0)
        dists = 1 - iou_matrix(enlarge(a, r), enlarge(b, np.full(len(b), self.args.buffer)))
        return matching.fuse_score(dists, detections) if self.args.fuse_score else dists

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
    return rows, secs


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
