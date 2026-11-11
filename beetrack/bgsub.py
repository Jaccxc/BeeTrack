"""MotionRescue: add detections for moving blobs the detector missed (motion-blurred flying bees), camera is static.
  uv run python -m beetrack.bgsub --dets yolox_x --split test      -> runs/dets/yolox_x_mr/test/<seq>.npy
Per chunk of frames the background is the temporal median; a foreground blob outside the entrance zone, about the size
of a bee and not overlapping any existing detection, becomes a detection with a fixed score.
Prints how many boxes were added and how many of them hit a ground-truth box (IoU >= 0.3)."""
import argparse
import json

import cv2
import numpy as np

from .common import RUNS, frame_path, split_specs
from .evaluate import ZONES, load_tracks
from .track import iou_matrix

CHUNK, SAMPLES = 200, 25   # background = median of SAMPLES frames spread over each CHUNK frames (handles slow light drift)


def rescue(gray, bg, dets, poly, area, thr, score):
    """New detections (x1, y1, x2, y2, score) for one frame. area: typical bee box area in this sequence."""
    m = (cv2.GaussianBlur(cv2.absdiff(gray, bg), (5, 5), 0) > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for (x, y, w, h, _), (cx, cy) in zip(st[1:], cen[1:]):
        if not 0.3 * area < w * h < 4 * area or cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
            continue
        b = np.array([[x, y, x + w, y + h]], np.float32)
        if len(dets) and iou_matrix(b, dets[:, :4]).max() > 0.1:
            continue
        out.append([x, y, x + w, y + h, score])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="yolox_x")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--thr", type=int, default=30, help="grey-level difference from the background")
    ap.add_argument("--score", type=float, default=0.5)
    a = ap.parse_args()
    zones = json.loads(ZONES.read_text())
    added = hit = 0
    for ev, src, s0, s1 in split_specs(a.split):
        dets = np.load(RUNS / "dets" / a.dets / a.split / f"{ev}.npy")
        gt = load_tracks(RUNS / "gt" / a.split / ev / "gt" / "gt.txt")
        poly = np.asarray(zones[ev], np.float32)
        area = float(np.median((dets[:, 3] - dets[:, 1]) * (dets[:, 4] - dets[:, 2])))
        new = []
        for c0 in range(s0, s1 + 1, CHUNK):
            fs = list(range(c0, min(c0 + CHUNK, s1 + 1)))
            grays = [cv2.imread(str(frame_path(src, f)), cv2.IMREAD_GRAYSCALE) for f in fs]
            bg = np.median(np.stack(grays[:: max(len(fs) // SAMPLES, 1)]), axis=0).astype(np.uint8)
            for f, g in zip(fs, grays):
                k = f - s0 + 1
                r = rescue(g, bg, dets[dets[:, 0] == k, 1:6], poly, area, a.thr, a.score)
                if r:
                    new += [[k, *x] for x in r]
                    b = gt[gt[:, 0] == k, 2:6].copy()
                    b[:, 2:] += b[:, :2]
                    hit += int((iou_matrix(np.array(r, np.float32)[:, :4], b).max(1) >= 0.3).sum()) if len(b) else 0
        added += len(new)
        out = RUNS / "dets" / f"{a.dets}_mr" / a.split / f"{ev}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, np.vstack([dets, np.array(new, dets.dtype).reshape(-1, 6)]))
        print(f"{ev}: +{len(new)} boxes on {len(dets)}")
    print(f"added {added} boxes, {hit} on a ground-truth bee (IoU >= 0.3): precision {hit / max(added, 1):.0%}")


if __name__ == "__main__":
    main()
