"""Speed of P0, the locked pipeline of log §125 (beetrack/p0.py), on one val video, for the report sentence
"the 640 px detector needs X ms per frame, the first stage Y ms, linking + filtering + counting Z ms, end to end F FPS".
Video: BEE24-35-tail (val, 1,000 frames, detector y11s_640).
  stage 1  detector, timed as beetrack/detect.py does: cv2.imread + model.predict (640 px, conf 0.05, iou 0.7, max_det 300,
           half), CUDA synchronised around each frame, the first 10 frames are warm-up and not counted
  stage 2  first stage = p0.stage1's tracker on the cached detections runs/dets/y11s_640/val/; Y = tracker seconds of
           run_sequence (the number track.py prints), the wall time with the per-frame box lookup is printed too
  stage 3  p0.p0 (TrackletLink 0.15 with the test link model m_dev_strict.json, drop tracks under 3 boxes) + p0.events
           + 30 s window counts; Z = total time / frames
  F = 1000 / (X + Y + Z), printed when all three stages run.
Self-check (stage 3): (a) with m_v0_val.json, the val link model of the val + dev P0 (link seed 0), the tracks are
byte-identical to the cached P0 file runs/trackers/val/q_t74_v0_seen_thr15/data/BEE24-35-tail.txt; (b) the timed
m_dev_strict.json tracks score as in t125/val125.txt ('P0 with m_dev_strict.json', y11s_640, BEE24-35-tail):
33 predicted / 37 GT events, 32 true at IoU 0.5, 10.81 % window error.
  uv run --offline --no-sync python tools/speed_p0.py                                   # all stages (GPU for stage 1)
  CUDA_VISIBLE_DEVICES=-1 uv run --offline --no-sync python tools/speed_p0.py --device cpu   # code check, CPU numbers
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from beetrack.common import RUNS, frame_path, split_specs, write_mot  # noqa: E402
from beetrack.detect import MIN_SCORE  # noqa: E402
from beetrack.evaluate import load_tracks  # noqa: E402
from beetrack.p0 import FPS, WIN, ZONE, dets, events, link_model, nframes, p0, per_window, score, stage1  # noqa: E402
from beetrack.track import Tracker, run_sequence  # noqa: E402

VIDEO, TAG = "BEE24-35-tail", "y11s_640"
WEIGHTS = RUNS / "detect" / TAG / "weights" / "best.pt"
CACHED = RUNS / "trackers" / "val" / "q_t74_v0_seen_thr15" / "data" / f"{VIDEO}.txt"
STRICT = (33, 37, 32, 10.81)      # t125/val125.txt line 173: pred, GT, true events at IoU 0.5, window error %
WARM = 10                         # detect.py: the first frames include CUDA warm-up


def detector_ms(device):
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    LOGGER.addFilter(lambda r: "'half' is deprecated" not in r.getMessage())   # detect.py's half=True warns on every frame
    _, src, s0, s1 = next(s for s in split_specs("val") if s[0] == VIDEO)
    model = YOLO(str(WEIGHTS))
    cuda = torch.cuda.is_available() and device != "cpu"
    times, n = [], 0
    for f in range(s0, s1 + 1):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        r = model.predict(cv2.imread(str(frame_path(src, f))), imgsz=640, conf=MIN_SCORE, iou=0.7, max_det=300, half=True,
                          verbose=False, device=device)[0]
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        n += len(r.boxes)
    where = torch.cuda.get_device_name(0) if cuda else "CPU (NOT a report number)"
    print(f"stage 1 detector: {1000 * np.mean(times[WARM:]):.2f} ms/frame on {where}, {len(times) - WARM} timed frames; "
          f"{n} boxes (cache {len(dets(TAG, 'val', VIDEO))})")
    return 1000 * float(np.mean(times[WARM:])), cuda


def main():
    ap = argparse.ArgumentParser(description="P0 speed on one val video, per stage.")
    ap.add_argument("--stages", default="1,2,3", help="comma list of 1 (detector), 2 (first stage), 3 (link + filter + count)")
    ap.add_argument("--device", default=None, help="stage 1 device, as Ultralytics takes it (default: what detect.py uses)")
    a = ap.parse_args()
    st = {int(x) for x in a.stages.split(",")}
    D, nf = dets(TAG, "val", VIDEO), nframes("val", VIDEO)
    print(f"{VIDEO} (val), {nf} frames, detector {TAG}")
    ms = {}
    if 1 in st:
        ms[1], gpu = detector_ms(a.device)
    if 2 in st:
        t = time.perf_counter()
        rows, secs = run_sequence(D, nf, Tracker("link", link=0, min_len=0))   # = p0.stage1, keeping run_sequence's seconds
        wall = time.perf_counter() - t
        R = np.array(rows, float).reshape(-1, 7)
        ms[2] = 1000 * secs / nf
        print(f"stage 2 first stage: {ms[2]:.2f} ms/frame (tracker update; {1000 * wall / nf:.2f} with the box lookup), "
              f"{len(R)} boxes, {len(np.unique(R[:, 1]))} tracklets")
    if 3 in st:
        if 2 not in st:
            R = stage1(D, nf)
        prob = link_model("m_dev_strict")
        t = time.perf_counter()
        tracks = p0(R, prob)
        E = events(tracks, ZONE[VIDEO])
        C = per_window(E, nf, WIN * FPS)
        ms[3] = 1000 * (time.perf_counter() - t) / nf
        print(f"stage 3 TrackletLink + drop short + counting: {ms[3]:.2f} ms/frame ({ms[3] * nf / 1000:.2f} s for the video); "
              f"{len(np.unique(np.array(tracks)[:, 1]))} tracks, in {int(C[:, 0].sum())} / out {int(C[:, 1].sum())}")
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "p0.txt"
            write_mot(f, p0(R, link_model("m_v0_val")))
            assert f.read_bytes() == CACHED.read_bytes(), f"P0 (m_v0_val) differs from {CACHED}"
            write_mot(f, tracks)
            s = score(load_tracks(f), "val", VIDEO)
        got = (s["pred"], s["gt"], s["tp"], round(100 * s["err"] / s["gt"], 2))
        assert got == STRICT, (got, STRICT)
        print(f"self-check: P0 with m_v0_val byte-identical to {CACHED.relative_to(ROOT).as_posix()}; P0 with m_dev_strict "
              f"pred / GT / true events {got[0]} / {got[1]} / {got[2]}, window error {got[3]} % = t125/val125.txt")
    if len(ms) == 3:
        F = 1000 / sum(ms.values())
        print(f"end to end: {F:.1f} FPS = 1000 / ({ms[1]:.2f} + {ms[2]:.2f} + {ms[3]:.2f})"
              + ("" if gpu else "   [stage 1 on CPU: NOT a report number]"))


if __name__ == "__main__":
    main()
