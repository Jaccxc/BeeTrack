"""Prepare BEE24 for training and evaluation.
run: uv run python -m beetrack.prep      (from the repository root)

- YOLO labels: one .txt next to every training image (Ultralytics looks there when the path has no 'images' folder)
- runs/data/{train,val}.txt image lists (frame stride on long videos) and runs/data/bee24.yaml
- runs/gt/{val,test,dev}/<seq>/: MOTChallenge ground truth + seqinfo.ini + seqmap for TrackEval
"""
import json

import cv2
import numpy as np

from .common import RUNS, frame_path, gt_slice, load_gt, n_frames, split_specs
from .evaluate import merge_handovers

STRIDE_LONG = 5      # long videos: every 5th frame (25 fps -> 5 fps); neighbours are near-duplicates
STRIDE_VAL_LONG = 10
DROP_STEPS = (2, 3, 5)  # 25 fps -> 12.5, 8.3, 5 fps


def yolo_labels(name):
    g = load_gt(name)
    h, w = cv2.imread(str(frame_path(name, 1))).shape[:2]
    by_frame = {}
    for f, _, x, y, bw, bh in g:
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
        if x1 - x0 > 1 and y1 - y0 > 1:
            by_frame.setdefault(int(f), []).append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    for f in range(1, n_frames(name) + 1):  # empty file = frame with no bees (a valid negative)
        frame_path(name, f).with_suffix(".txt").write_text("\n".join(by_frame.get(f, [])))
    return w, h


def write_eval_gt(split, ev, src, a, b, step=1, merge=False):
    """MOTChallenge folder TrackEval reads: <seq>/gt/gt.txt (conf, class, visibility all 1) + seqinfo.ini.
    step > 1 keeps every step-th frame, renumbered 1, 2, ... (same rule as track.run_sequence)."""
    d = RUNS / "gt" / split / ev
    (d / "gt").mkdir(parents=True, exist_ok=True)
    g = gt_slice(src, a, b)
    # BEE24-14/15/33/35 (train only) reuse an ID for two bees in one frame (993 rows); TrackEval rejects that,
    # so each extra occurrence keeps its box and gets a fresh ID. The test sequences have no such rows.
    _, first = np.unique(g[:, 0] * 1e6 + g[:, 1], return_index=True)
    dup = np.setdiff1d(np.arange(len(g)), first)
    g[dup, 1] = g[:, 1].max() + 1 + np.arange(len(dup))
    if merge:   # dev only: IDs that hand over to a new ID on the same box are one bee
        g = merge_handovers(g)
    g = g[(g[:, 0] - 1) % step == 0]
    g[:, 0] = (g[:, 0] - 1) // step + 1
    np.savetxt(d / "gt" / "gt.txt", np.c_[g, np.ones((len(g), 3))], fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d")
    h, w = cv2.imread(str(frame_path(src, a))).shape[:2]
    (d / "seqinfo.ini").write_text(f"[Sequence]\nname={ev}\nimDir=img1\nframeRate=25\nseqLength={len(range(a, b + 1, step))}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n")
    return len(g)


def main():
    out = RUNS / "data"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        specs = split_specs(split)
        imgs, n_box = [], 0
        for ev, src, a, b in specs:
            if a == 1:                      # labels are per source sequence; write them once
                yolo_labels(src)
            stride = 1 if n_frames(src) < 1000 else (STRIDE_LONG if split == "train" else STRIDE_VAL_LONG)
            imgs += [str(frame_path(src, f)) for f in range(a, b + 1, stride)]
            n_box += write_eval_gt(split, ev, src, a, b) if split != "train" else len(gt_slice(src, a, b))
        (out / f"{split}.txt").write_text("\n".join(imgs))   # test.txt is only used for detection mAP
        if split != "train":
            (RUNS / "gt" / f"{split}.seqmap").write_text("name\n" + "\n".join(e for e, *_ in specs) + "\n")
        summary[split] = dict(sequences=[e for e, *_ in specs], frames=sum(b - a + 1 for _, _, a, b in specs),
                              detector_images=len(imgs), gt_boxes=n_box)
    for k in DROP_STEPS:                    # frame-drop robustness ground truth: runs/gt/test_step<k>
        for ev, src, a, b in split_specs("test"):
            write_eval_gt(f"test_step{k}", ev, src, a, b, step=k)
        (RUNS / "gt" / f"test_step{k}.seqmap").write_text((RUNS / "gt" / "test.seqmap").read_text())
    for ev, src, a, b in split_specs("dev"):   # tracker-development GT: the training part of the long videos
        write_eval_gt("dev", ev, src, a, b, merge=True)
    (RUNS / "gt" / "dev.seqmap").write_text("name\n" + "\n".join(e for e, *_ in split_specs("dev")) + "\n")
    (out / "bee24.yaml").write_text(f"path: {out.as_posix()}\ntrain: {(out / 'train.txt').as_posix()}\nval: {(out / 'val.txt').as_posix()}\ntest: {(out / 'test.txt').as_posix()}\nnames:\n  0: bee\n")
    # the three splits never share a frame
    seen = {}
    for split in ("train", "val", "test"):
        for _, src, a, b in split_specs(split):
            for other, (oa, ob) in seen.get(src, []):
                assert b < oa or a > ob, f"{split} overlaps {other} in {src}"
            seen.setdefault(src, []).append((split, (a, b)))
    (out / "split.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
