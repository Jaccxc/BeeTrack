"""P0, the locked counting pipeline (log §91, §92, §125), and its 30 s window scorer, on cached detections:
  detections -> stage 1 (track.PRESETS['link'] without linking and minimum length, i.e. PASS1 as it is)
  -> TrackletLink at threshold 0.15 -> drop tracks with fewer than 3 boxes -> in / out events of the entrance zone.
Also the two ByteTrack baselines of Table A: default ByteTrack, and its in / out counts scaled by k = true / predicted
totals fitted on other videos (calibrated ByteTrack, §93). tools/repro_table_a.py runs it on val + dev."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .common import RUNS, split_specs
from .evaluate import ZONES, by_track, load_tracks, match_events, zone_tracks
from .link import link, model
from .track import Tracker, drop_short, run_sequence

MODELS = Path(__file__).with_name("models")
THR, MIN_LEN = 0.15, 3
FPS, WIN = 25, 30                 # 30 s windows, scored at 30 start offsets one second apart
ZONE = json.loads(ZONES.read_text())


def link_model(name):
    """beetrack/models/<name>.json as a link probability function."""
    return model(json.loads((MODELS / f"{name}.json").read_text()))


def nframes(sp, ev):
    return next(s1 - s0 + 1 for e, _, s0, s1 in split_specs(sp) if e == ev)


def dets(tag, sp, ev):
    return np.load(RUNS / "dets" / tag / sp / f"{ev}.npy")


def stage1(D, nf):
    """Stage-1 tracklets as float rows (frame, id, x, y, w, h, score)."""
    rows, _ = run_sequence(D, nf, Tracker("link", link=0, min_len=0))
    return np.array(rows, float).reshape(-1, 7)


def p0(R, prob):
    """Stage-1 rows -> TrackletLink -> tracks of 3 or more boxes, in the row order write_mot gets."""
    return drop_short(sorted(link(R.tolist(), THR, prob).tolist()), MIN_LEN, 0.0)


def bytetrack(D, nf):
    """Default ByteTrack (Ultralytics bytetrack.yaml)."""
    return run_sequence(D, nf, Tracker("bytetrack"))[0]


def events(rows, poly):
    """(frame, +1 in / -1 out) of every event, dated at the track's last boundary crossing in the event's direction."""
    rows = by_track(rows)
    out = []
    for ev, s, e, cross in zone_tracks(rows, poly).values():
        if ev:
            k = [j for j, d in cross if d == ev]
            out.append((rows[k[-1] if k else e - 1, 0], 1 if ev == "in" else -1))
    return np.array(out).reshape(-1, 2)


def per_window(E, nf, w):
    """(windows, 2) in and out counts of the events E in windows of w frames."""
    nb = int(np.ceil(nf / w))
    b = np.minimum(((E[:, 0] - 1) // w).astype(int), nb - 1)
    return np.c_[np.bincount(b[E[:, 1] > 0], minlength=nb), np.bincount(b[E[:, 1] < 0], minlength=nb)]


def window_err(P, G, nf, k=(1, 1), win=WIN):
    """Sum over windows of |k x predicted - true| for in and out, mean over the win offsets (windows of win seconds).
    Offset o moves every event and the video end o seconds later, so the first window is o seconds shorter.
    win=None: the whole video as one window, no offsets."""
    if win is None:
        return float(np.abs(per_window(P, nf, nf) * k - per_window(G, nf, nf)).sum())
    w = win * FPS
    return float(np.mean([np.abs(per_window(P + [o * FPS, 0], nf + o * FPS, w) * k - per_window(G + [o * FPS, 0], nf + o * FPS, w)).sum()
                          for o in range(win)]))


assert (per_window(np.array([[1, 1], [30, -1], [60, 1], [61, 1]]), 70, 50) == [[1, 1], [2, 0]]).all()
assert window_err(np.array([[1, 1]]), np.array([[1, 1], [2, -1]]), 100) == 1.0
assert window_err(np.array([[1, 1], [90, -1]]), np.array([[50, 1], [60, 1]]), 100, (2, 1), None) == 1.0


@lru_cache(None)
def gt(sp, ev):
    return load_tracks(RUNS / "gt" / sp / ev / "gt" / "gt.txt")


def score(pr, sp, ev):
    """One video: window error (not normalised), GT and predicted events, true events at IoU 0.5 (match_events)."""
    nf, poly = nframes(sp, ev), ZONE[ev]
    P, G = events(pr, poly), events(gt(sp, ev), poly)
    m = match_events(gt(sp, ev), pr, poly)
    assert (m["pred"], m["gt"]) == (len(P), len(G)), (sp, ev)
    return dict(err=window_err(P, G, nf), gt=len(G), pred=len(P), tp=m["tp"], P=P, G=G, nf=nf)


def pooled(ss):
    """Over videos: window error % (summed errors / all GT events), bias (predicted - GT events), event F1@0.5."""
    g, p, e, tp = (sum(s[x] for s in ss) for x in ("gt", "pred", "err", "tp"))
    return dict(win=100 * e / g, bias=p - g, f1=2 * tp / max(g + p, 1))


def totals(E):
    return np.array([(E[:, 1] > 0).sum(), (E[:, 1] < 0).sum()])


def factor(ss):
    """k = (k_in, k_out) = true totals / predicted totals over the videos ss."""
    return np.sum([totals(s["G"]) for s in ss], 0) / np.maximum(1, np.sum([totals(s["P"]) for s in ss], 0))


def calibrated(S, win=WIN):
    """Calibrated ByteTrack on val + dev, S = {'val': scores, 'dev': scores}: val counts x k fitted on dev and dev
    counts x k fitted on val, pooled (win %, bias). Also returns k fitted on val + dev together (frozen for test)."""
    k = {"val": factor(S["dev"]), "dev": factor(S["val"])}
    e = sum(sum(window_err(s["P"], s["G"], s["nf"], k[sp], win) for s in S[sp]) for sp in S)
    n = sum(s["gt"] for sp in S for s in S[sp])
    b = sum(float((k[sp] * totals(s["P"])).sum()) - s["gt"] for sp in S for s in S[sp])
    return dict(win=100 * e / n, bias=b), factor(S["val"] + S["dev"])
