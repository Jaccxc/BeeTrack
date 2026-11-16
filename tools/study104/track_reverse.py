"""§104 CPU tracking for forward (F), reversed (R), and forward-link control (RF).

Run with the repo on PYTHONPATH. Each output uses the original 1-based frames.
Use --self-check to test frame mapping and the pipeline without any hive data.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from beetrack.common import RUNS, write_mot
from beetrack.link import link, model
from beetrack.track import Tracker, drop_short, run_sequence


STRICT_SHA256 = "fdd788e0dba0e6fa1c2d06feec3127bf84d5e7b2d214cc0f793c452270403dcf"


def reverse_frames(rows, nframes):
    """Reverse the clock alone. Within each frame even tied-confidence rows keep their order."""
    result = rows.copy()
    result[:, 0] = nframes + 1 - result[:, 0]
    return result


def first_pass(detections, nframes):
    rows, _ = run_sequence(detections, nframes, Tracker("link", link=0, min_len=0))
    return np.asarray(rows, dtype=float).reshape(-1, 7)


def finish(tracklets, probability):
    rows = sorted(link(tracklets, 0.15, probability).tolist())
    return np.asarray(drop_short(rows, 3, 0.0), dtype=float).reshape(-1, 7)


def track_arms(detections, nframes, probability):
    """Yield named arrays suitable for write_mot. R and RF share identical first-pass boxes and IDs."""
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError("Expected cached rows [frame, x1, y1, x2, y2, score]")
    frames = detections[:, 0]
    if not np.all((frames >= 1) & (frames <= nframes) & (frames == np.floor(frames))):
        raise ValueError("Detection frame lies outside the integer frame grid")
    yield "F", finish(first_pass(detections, nframes), probability)
    backwards = first_pass(reverse_frames(detections, nframes), nframes)
    result = reverse_frames(finish(backwards, probability), nframes)
    yield "R", result[np.lexsort((result[:, 1], result[:, 0]))]
    yield "RF", finish(reverse_frames(backwards, nframes), probability)


def run_case(hive, seed, weights, output_dir):
    if hive not in (33, 35) or seed not in range(9):
        raise ValueError("§104 supports only dev heads 33 and 35, seeds 0 through 8")
    seq, tag = f"BEE24-{hive}-head", f"loho_no{hive}_s{seed}"
    baseline = RUNS / "trackers" / "dev" / f"q_lk_{tag}" / "data" / f"{seq}.txt"
    data = RUNS / "dets" / tag / "dev" / f"{seq}.npy"
    weight_bytes = Path(weights).read_bytes()
    if hashlib.sha256(weight_bytes).hexdigest() != STRICT_SHA256:
        raise ValueError("Expected the preregistered m_dev_strict.json")
    if not baseline.is_file():
        raise FileNotFoundError(baseline)
    destination = Path(output_dir) / f"{hive}_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    detections = np.load(data, allow_pickle=False)
    started = time.monotonic()
    for name, rows in track_arms(detections, 3950, model(json.loads(weight_bytes))):
        path = destination / f"{name}.txt"
        partial = path.with_suffix(".tmp")
        write_mot(partial, rows)
        if name == "F" and partial.read_bytes().splitlines() != baseline.read_bytes().splitlines():
            raise RuntimeError(f"Forward replay differs from archived P0 in {seq} seed {seed}")
        partial.replace(path)
        print(f"{seq} seed={seed} arm={name} rows={len(rows)} cumulative_s={time.monotonic()-started:.1f}",
              flush=True)


def self_check():
    # A gap, two equal-score competing boxes, and empty frames at opposite ends.
    detections = np.array([[frame, x, 5, x + 10, 15, 0.8]
                           for frame in (2, 3, 4, 6, 7, 8) for x in (8, 9, 60)], dtype=np.float32)
    nframes = 10
    reversed_dets = reverse_frames(detections, nframes)
    np.testing.assert_array_equal(reverse_frames(reversed_dets, nframes), detections)
    for frame in range(1, nframes + 1):
        np.testing.assert_array_equal(reversed_dets[reversed_dets[:, 0] == nframes + 1 - frame, 1:],
                                      detections[detections[:, 0] == frame, 1:])

    prob = lambda features: np.full(len(features), 0.9)
    arms = dict(track_arms(detections, nframes, prob))
    assert arms.keys() == {"F", "R", "RF"}
    assert len(arms["F"]) == 12
    for rows in arms.values():
        np.testing.assert_array_equal(np.unique(rows[:, 0]), [2, 3, 4, 6, 7, 8])
        assert set(rows[:, 2]) == {8, 60}
    empty = dict(track_arms(detections[:0], nframes, prob))
    assert all(rows.shape == (0, 7) for rows in empty.values())
    print("Synthetic CPU forward/reverse checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hive", type=int, choices=[33, 35], nargs="?")
    parser.add_argument("seed", type=int, choices=range(9), nargs="?")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif None in (args.hive, args.seed, args.weights, args.output_dir):
        parser.error("Provide HIVE SEED --weights PATH --output-dir PATH, or --self-check")
    else:
        run_case(args.hive, args.seed, args.weights, args.output_dir)
