"""Run the detector once on every frame of the val/test sequences and cache the boxes, so every tracker
setting is compared on identical detections (Experiment 1 changes only the tracker).
run: uv run python -m beetrack.detect --weights runs/detect/y11s_640/weights/best.pt --imgsz 640 --split test
writes runs/dets/<tag>/<split>/<seq>.npy  rows: frame, x1, y1, x2, y2, score   (+ <seq>.json with timing)
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .common import RUNS, frame_path, gt_slice, split_specs

MIN_SCORE = 0.05     # ByteTrack's second stage uses low-score boxes (>= 0.1 by default), so keep them
_rng = np.random.default_rng(0)
CORRUPT = {          # robustness: one synthetic image change at a time, applied to the test frames only
    "none": lambda im: im,
    "dark": lambda im: (im * 0.4).astype(np.uint8),                                    # dusk / under-exposure
    "bright": lambda im: np.clip(im * 1.6 + 20, 0, 255).astype(np.uint8),             # direct sun / over-exposure
    "blur": lambda im: cv2.filter2D(im, -1, np.full((1, 9), 1 / 9)),                  # 9-px horizontal motion blur
    "noise": lambda im: np.clip(im + _rng.normal(0, 15, im.shape), 0, 255).astype(np.uint8),  # sensor noise
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="best.pt, or 'gt' for the oracle detector (GT boxes, score 1)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="test", choices=["val", "test", "dev", "train"])
    ap.add_argument("--corrupt", default="none", choices=list(CORRUPT))
    ap.add_argument("--tag", help="output name, default y11s_<imgsz>[_<corrupt>]")
    a = ap.parse_args()
    tag = a.tag or ("oracle" if a.weights == "gt" else f"y11s_{a.imgsz}" + ("" if a.corrupt == "none" else f"_{a.corrupt}"))
    out = RUNS / "dets" / tag / a.split
    out.mkdir(parents=True, exist_ok=True)
    if a.weights == "gt":   # perfect detections isolate the tracker's own association errors
        for ev, src, s0, s1 in split_specs(a.split):
            g = gt_slice(src, s0, s1)
            np.save(out / f"{ev}.npy", np.c_[g[:, 0], g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5], np.ones(len(g))])
        return
    fx = CORRUPT[a.corrupt]
    model = YOLO(a.weights)
    for ev, src, s0, s1 in split_specs(a.split):
        rows, times = [], []
        for i, f in enumerate(range(s0, s1 + 1), start=1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            r = model.predict(fx(cv2.imread(str(frame_path(src, f)))), imgsz=a.imgsz, conf=MIN_SCORE, iou=0.7, max_det=300, half=True, verbose=False)[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            b = r.boxes
            if len(b):
                rows.append(np.c_[np.full(len(b), i), b.xyxy.cpu().numpy(), b.conf.cpu().numpy()])
        np.save(out / f"{ev}.npy", np.concatenate(rows) if rows else np.zeros((0, 6)))
        warm = times[10:] or times   # first frames include CUDA warm-up
        (out / f"{ev}.json").write_text(json.dumps(dict(frames=len(times), det_ms_mean=1000 * float(np.mean(warm)), weights=str(Path(a.weights)), imgsz=a.imgsz, corrupt=a.corrupt)))
        print(f"{ev}: {len(times)} frames, {1000 * np.mean(warm):.1f} ms/frame (detector, includes image decode)")


if __name__ == "__main__":
    main()
