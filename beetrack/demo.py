"""BeeTrack demo: hive-entrance video in, annotated video + tracks CSV out.
run: uv run python -m beetrack.demo --source <video file or image folder> [--tracker ours|bytetrack] [--out runs/demo]
draws boxes, IDs, 1-second trails, the counting line, live in/out counts and the measured FPS.
"""
import argparse
import csv
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .common import RUNS
from .evaluate import ZONES
from .track import make_tracker

PALETTE = [(0, 159, 230), (233, 180, 86), (115, 158, 0), (66, 228, 240), (178, 114, 0), (0, 94, 213), (167, 121, 204)]


def frames(source):
    p = Path(source)
    if p.is_dir():
        for f in sorted(p.glob("*.jpg")):
            yield cv2.imread(str(f))
    else:
        cap = cv2.VideoCapture(str(p))
        while True:
            ok, im = cap.read()
            if not ok:
                break
            yield im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, or a folder of .jpg frames (e.g. data/BEE24/test/BEE24-36/img1)")
    ap.add_argument("--weights", default=str(RUNS / "detect" / "y11s_640" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tracker", default="ours", choices=["ours", "bytetrack"])
    ap.add_argument("--line", type=float, default=0.5, help="counting line height as a fraction of the frame")
    ap.add_argument("--zone", default="", help="entrance polygon key in beetrack/zones.json (e.g. BEE24-36); replaces the line count")
    ap.add_argument("--fps", type=float, default=25)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=str(RUNS / "demo"))
    a = ap.parse_args()

    model = YOLO(a.weights)
    tracker = make_tracker(a.tracker, frame_rate=a.fps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.source).parent.name if Path(a.source).is_dir() else Path(a.source).stem
    writer, rows = None, []
    trail = defaultdict(lambda: deque(maxlen=int(a.fps)))
    last_side, n_in, n_out, t_hist = {}, 0, 0, deque(maxlen=30)
    zone = np.array(json.loads(ZONES.read_text())[a.zone], np.float32) if a.zone else None
    first_in, last_in, last_seen = {}, {}, {}
    buf = tracker.args.track_buffer

    def finalize(tid):   # same rule as evaluate.zone_crossings, applied once the track is gone for good
        nonlocal n_in, n_out
        n_in += int(last_in[tid] and not first_in[tid])
        n_out += int(first_in[tid] and not last_in[tid])
        for d in (first_in, last_in, last_seen):
            d.pop(tid)
    for i, im in enumerate(frames(a.source), start=1):
        if a.max_frames and i > a.max_frames:
            break
        h, w = im.shape[:2]
        line_y = a.line * h
        if writer is None:
            writer = cv2.VideoWriter(str(out / f"{stem}_{a.tracker}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(im, imgsz=a.imgsz, conf=0.05, half=True, verbose=False)[0]
        dets = np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()] if len(r.boxes) else np.zeros((0, 5))
        tracks = tracker.update(dets)   # rows: x1, y1, x2, y2, id, score
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_hist.append(time.perf_counter() - t0)

        for x1, y1, x2, y2, tid, sc in tracks:
            tid = int(tid)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if zone is not None:
                inside = cv2.pointPolygonTest(zone, (float(cx), float(cy)), False) >= 0
                first_in.setdefault(tid, inside)
                last_in[tid], last_seen[tid] = inside, i
            else:
                side = np.sign(cy - line_y)
                if side and last_side.get(tid, side) != side:   # plain line rule (evaluate.crossings, band 0)
                    n_in, n_out = (n_in + 1, n_out) if side < 0 else (n_in, n_out + 1)
                if side:
                    last_side[tid] = side
            trail[tid].append((int(cx), int(cy)))
            c = PALETTE[tid % len(PALETTE)]
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            cv2.putText(im, str(tid), (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
            if len(trail[tid]) > 1:
                cv2.polylines(im, [np.array(trail[tid], np.int32)], False, c, 2, cv2.LINE_AA)
            rows.append([i, tid, round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1), round(float(sc), 3)])
        if zone is not None:
            for tid in [t for t, f in last_seen.items() if i - f > buf]:   # lost longer than the buffer: track ended
                finalize(tid)
            cv2.polylines(im, [zone.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.line(im, (0, int(line_y)), (w, int(line_y)), (255, 255, 255), 2)
        fps = len(t_hist) / sum(t_hist)
        hud = f"IN {n_in}   OUT {n_out}   bees {len(tracks)}   {fps:.0f} FPS ({a.tracker})"
        cv2.putText(im, hud, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(im, hud, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(im)
    for tid in list(last_seen):
        finalize(tid)
    if writer is not None:
        writer.release()
    with open(out / f"{stem}_{a.tracker}.csv", "w", newline="") as f:
        csv.writer(f).writerows([["frame", "id", "x", "y", "w", "h", "score"], *rows])
    print(f"wrote {out / f'{stem}_{a.tracker}.mp4'}  frames={i}  in={n_in} out={n_out}  mean FPS={len(t_hist) / sum(t_hist):.1f}")


if __name__ == "__main__":
    main()
