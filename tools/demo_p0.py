"""Annotated MP4 of P0 on one video, from saved P0 tracks (MOT txt as run125.py / tools/repro_table_a.py write them; the
detector is not run). Draws every track's box and id with a trail of its last 25 frames, the entrance polygon
(beetrack/zones.json) and a panel: cumulative in / out and the current 30 s window's in / out of P0, and of GT when the
video has GT (runs/gt/<split>/<video>/gt/gt.txt). Events and windows are the scorer's (p0.events, 30 s windows at offset 0,
25 fps). Also writes the middle frame as <out>.jpg.
  uv run --offline --no-sync python tools/demo_p0.py --video BEE24-35-tail \
      --tracks runs/trackers/val/q_t74_v0_seen_thr15/data/BEE24-35-tail.txt --out D:/beetrack_runs/t233r/demo_35tail.mp4 --end 500
--start / --end: frames of the video (1-based, inclusive), default the whole video."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from beetrack.common import RUNS, frame_path, split_specs  # noqa: E402
from beetrack.demo import PALETTE  # noqa: E402
from beetrack.evaluate import load_tracks  # noqa: E402
from beetrack.p0 import FPS, WIN, ZONE, events, gt  # noqa: E402

TRAIL = 25


def counts(E, a, b):
    """(in, out) of the events E dated in frames a..b."""
    e = E[(E[:, 0] >= a) & (E[:, 0] <= b), 1]
    return int((e > 0).sum()), int((e < 0).sum())


def main():
    ap = argparse.ArgumentParser(description="Annotated MP4 of saved P0 tracks.")
    ap.add_argument("--video", required=True, help="video name as in the splits, e.g. BEE24-35-tail or BEE24-36")
    ap.add_argument("--tracks", required=True, type=Path, help="P0 tracks, MOT txt (frame, id, x, y, w, h, ...)")
    ap.add_argument("--out", required=True, type=Path, help="output .mp4 (the still goes next to it as .jpg)")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0, help="last frame, default the video's last")
    a = ap.parse_args()
    sp, (ev, src, s0, s1) = next((sp, s) for sp in ("val", "dev", "test") for s in split_specs(sp) if s[0] == a.video)
    nf, poly = s1 - s0 + 1, np.array(ZONE[ev], np.float32)
    end = a.end or nf
    assert 1 <= a.start <= end <= nf, (a.start, end, nf)
    T = load_tracks(a.tracks)
    assert len(T) and T[:, 0].max() <= nf, f"{a.tracks} has frames beyond {ev}'s {nf}"
    T = T[np.argsort(T[:, 0], kind="stable")]
    C = (T[:, 2:4] + T[:, 4:6] / 2).astype(np.int32)
    P = events(T, poly)
    G = events(gt(sp, ev), poly) if (RUNS / "gt" / sp / ev / "gt" / "gt.txt").exists() else None
    w, mid, vw = WIN * FPS, (a.start + end) // 2, None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for f in range(a.start, end + 1):
        im = cv2.imread(str(frame_path(src, s0 + f - 1)))
        if vw is None:
            h, wd = im.shape[:2]
            fs, lh = max(0.5, wd / 1600), int(28 * max(0.5, wd / 1600) / 0.5)
            for cc in ("avc1", "mp4v"):     # H.264 where this OpenCV build can write it, MPEG-4 otherwise
                vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*cc), FPS, (wd, h))
                if vw.isOpened():
                    break
            assert vw.isOpened(), a.out
        cv2.polylines(im, [poly.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
        lo, now, hi = np.searchsorted(T[:, 0], [f - TRAIL + 1, f, f + 1])
        for i in range(now, hi):            # tracks with a box in this frame
            tid, (x, y, bw, bh) = int(T[i, 1]), T[i, 2:6].astype(int)
            col = PALETTE[tid % len(PALETTE)]
            tr = C[lo:hi][T[lo:hi, 1] == tid]
            if len(tr) > 1:
                cv2.polylines(im, [tr], False, col, 2, cv2.LINE_AA)
            cv2.rectangle(im, (x, y), (x + bw, y + bh), col, 2)
            cv2.putText(im, str(tid), (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 2, cv2.LINE_AA)
        k = (f - 1) // w
        cum, win = counts(P, 1, f), counts(P, k * w + 1, f)
        rows = [f"frame {f} / {nf}   {(f - 1) / FPS:.1f} s",
                f"cumulative      P0 in {cum[0]:3d}  out {cum[1]:3d}",
                f"window {k * WIN}-{(k + 1) * WIN} s  P0 in {win[0]:3d}  out {win[1]:3d}"]
        if G is not None:
            g, gw = counts(G, 1, f), counts(G, k * w + 1, f)
            rows[1] += f"    GT in {g[0]:3d}  out {g[1]:3d}"
            rows[2] += f"    GT in {gw[0]:3d}  out {gw[1]:3d}"
        pw = int(max(cv2.getTextSize(r, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] for r in rows)) + 20
        im[:lh * len(rows) + 10, :pw] //= 3       # darkened panel; no black outline: OpenCV 5 glyphs widen with thickness
        for j, r in enumerate(rows):
            cv2.putText(im, r, (10, lh * (j + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(im)
        if f == mid:
            cv2.imwrite(str(a.out.with_suffix(".jpg")), im)
    vw.release()
    pc, gc = counts(P, a.start, end), (counts(G, a.start, end) if G is not None else None)
    print(f"wrote {a.out} ({end - a.start + 1} frames, {FPS} fps, codec {cc}) and {a.out.with_suffix('.jpg')} (frame {mid}); "
          f"events in frames {a.start}-{end}: P0 in {pc[0]} out {pc[1]}" + (f", GT in {gc[0]} out {gc[1]}" if gc else ""))


if __name__ == "__main__":
    main()
