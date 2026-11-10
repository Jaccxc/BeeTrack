"""Failure-case figures for the report: (a) the frame with most false positives, (b) an identity switch inside the entrance zone.
run: uv run python -m beetrack.failures --tracker ours
writes runs/figs/failure_fp_<seq>.png and runs/figs/failure_idsw_<seq>.png (GT green, predictions coloured by ID)
"""
import argparse
import json

import cv2
import numpy as np

from .common import RUNS, frame_path
from .evaluate import ZONES, load_tracks
from .track import iou_matrix

PALETTE = [(0, 159, 230), (233, 180, 86), (115, 158, 0), (66, 228, 240), (178, 114, 0), (0, 94, 213), (167, 121, 204)]


def xyxy(r):
    return np.c_[r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]]


def match(g, p):
    """Greedy IoU>=0.5 matches between GT rows g and predicted rows p of one frame: list of (gi, pi)."""
    if not len(g) or not len(p):
        return []
    m = iou_matrix(xyxy(g), xyxy(p))
    out = []
    while m.size and m.max() >= 0.5:
        i, j = np.unravel_index(m.argmax(), m.shape)
        out.append((i, j))
        m[i, :], m[:, j] = -1, -1
    return out


def draw(seq, f, g, p, box=None):
    im = cv2.imread(str(frame_path(seq, int(f))))
    for x1, y1, x2, y2 in xyxy(g).astype(int):
        cv2.rectangle(im, (x1, y1), (x2, y2), (80, 220, 80), 1)
    for (x1, y1, x2, y2), tid in zip(xyxy(p).astype(int), p[:, 1].astype(int)):
        c = PALETTE[tid % len(PALETTE)]
        cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
        cv2.putText(im, str(tid), (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    cv2.putText(im, f"{seq} frame {int(f)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if box is not None:
        x1, y1, x2, y2 = box
        im = im[max(0, y1):y2, max(0, x1):x2]
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="ours")
    ap.add_argument("--fp-seq", default="BEE24-34")
    ap.add_argument("--sw-seq", default="BEE24-36")
    a = ap.parse_args()
    out = RUNS / "figs"
    out.mkdir(parents=True, exist_ok=True)
    zones = json.loads(ZONES.read_text())

    # (a) most unmatched predictions in one frame
    seq = a.fp_seq
    G = load_tracks(RUNS / "gt" / "test" / seq / "gt" / "gt.txt")
    P = load_tracks(RUNS / "trackers" / "test" / a.tracker / "data" / f"{seq}.txt")
    best = max(np.unique(P[:, 0])[::5], key=lambda f: (P[:, 0] == f).sum() - len(match(G[G[:, 0] == f], P[P[:, 0] == f])))
    g, p = G[G[:, 0] == best], P[P[:, 0] == best]
    cv2.imwrite(str(out / f"failure_fp_{seq}.png"), draw(seq, best, g, p))
    print(f"FP frame {seq} {int(best)}: {len(p) - len(match(g, p))} unmatched predictions, {len(g)} GT")

    # (b) first GT track whose matched predicted ID changes while its centre is inside the entrance zone
    seq = a.sw_seq
    G = load_tracks(RUNS / "gt" / "test" / seq / "gt" / "gt.txt")
    P = load_tracks(RUNS / "trackers" / "test" / a.tracker / "data" / f"{seq}.txt")
    poly = np.asarray(zones[seq], np.float32)
    last, hit = {}, None
    for f in np.unique(G[:, 0]):
        g, p = G[G[:, 0] == f], P[P[:, 0] == f]
        for i, j in match(g, p):
            gid, pid = g[i, 1], p[j, 1]
            cx, cy = g[i, 2] + g[i, 4] / 2, g[i, 3] + g[i, 5] / 2
            if gid in last and last[gid][1] != pid and f - last[gid][0] > 3 \
                    and cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                hit = (last[gid][0], f, cx, cy)
                break
            last[gid] = (f, pid)
        if hit:
            break
    f0, f1, cx, cy = hit
    box = (int(cx) - 160, int(cy) - 120, int(cx) + 160, int(cy) + 120)
    ims = [draw(seq, f, G[G[:, 0] == f], P[P[:, 0] == f], box) for f in (f0, f1)]
    ims = [cv2.copyMakeBorder(im, 0, 240 - im.shape[0], 0, 320 - im.shape[1], cv2.BORDER_CONSTANT) for im in ims]
    cv2.imwrite(str(out / f"failure_idsw_{seq}.png"), np.hstack([ims[0], np.full((240, 6, 3), 255, np.uint8), ims[1]]))
    print(f"ID switch {seq}: frames {int(f0)} -> {int(f1)} at ({cx:.0f}, {cy:.0f})")


if __name__ == "__main__":
    main()
