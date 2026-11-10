"""Paths, the data split and MOT-format I/O shared by every BeeTrack script."""
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "BEE24"
RUNS = ROOT / "runs"

# validation = whole short sequences + the tail of each long training video (gap frames are used by neither split)
VAL_WHOLE = ["BEE24-10", "BEE24-15", "BEE24-24", "BEE24-28"]
LONG = 1000          # a sequence this long or longer is a "long video"
VAL_TAIL, GAP = 1000, 50


def seq_dir(name):
    for split in ("train", "test"):
        if (DATA / split / name).is_dir():
            return DATA / split / name
    raise FileNotFoundError(name)


@lru_cache(None)
def n_frames(name):
    return len(list((seq_dir(name) / "img1").glob("*.jpg")))


def frame_path(name, f):
    return seq_dir(name) / "img1" / f"{f:06d}.jpg"


@lru_cache(None)
def load_gt(name):
    """(N, 6) float array: frame, id, x, y, w, h (top-left boxes, 1-based frames)."""
    return np.loadtxt(seq_dir(name) / "gt" / "gt.txt", delimiter=",", usecols=range(6), ndmin=2)


def split_specs(split):
    """List of (eval_name, source_sequence, first_frame, last_frame) for 'train', 'val' or 'test'."""
    out = []
    names = sorted(p.name for p in (DATA / ("test" if split == "test" else "train")).iterdir())
    for s in names:
        n = n_frames(s)
        if split == "test":
            out.append((s, s, 1, n))
        elif s in VAL_WHOLE:
            if split == "val":
                out.append((s, s, 1, n))
        elif n >= LONG:
            if split == "train":
                out.append((s, s, 1, n - VAL_TAIL - GAP))
            else:
                out.append((f"{s}-tail", s, n - VAL_TAIL + 1, n))
        elif split == "train":
            out.append((s, s, 1, n))
    return out


def gt_slice(name, first, last):
    """GT rows of frames first..last, renumbered so the slice starts at frame 1."""
    g = load_gt(name)
    g = g[(g[:, 0] >= first) & (g[:, 0] <= last)].copy()
    g[:, 0] -= first - 1
    return g


def write_mot(path, rows):
    """rows: frame, id, x, y, w, h[, score] -> MOTChallenge txt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6] if len(r) > 6 else 1:.3f},-1,-1,-1\n")
