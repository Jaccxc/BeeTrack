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
from .evaluate import ZONES, by_track, inside, load_tracks, match_events, zone_tracks
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


def crossing_join(rows, poly):
    """Crossing join (CJ, log §161) on P0's linked rows before drop_short, sorted(link(...).tolist()) as in p0(): a track b
    whose first box comes dt = 1-3 frames after the last box of a track a, its centre on the other side of the polygon
    (inside(), zone_tracks' side test) and at most 2 x dt mean body lengths (sqrt(w h) of the two boxes) from a's last
    centre, takes a's id; neither track has an event of its own (zone_tracks). Greedy by (dt, distance in body lengths,
    a, b), every track in at most one join. Returns (the rows in their order, b's ids replaced; [(a, b, dt, distance)]).
    P0 + CJ = drop_short(sorted(rows), MIN_LEN, 0.0)."""
    R = np.array(rows, float).reshape(-1, 7)
    S = R[np.lexsort((R[:, 0], R[:, 1]))]
    ids, s = np.unique(S[:, 1], return_index=True)
    e = np.r_[s[1:], len(S)]
    A, B = S[e - 1], S[s]                               # every track's last and first box
    own = np.array([any(t[0] for t in zone_tracks(by_track(S[i:j]), poly).values()) for i, j in zip(s, e)], bool)
    ia, ib = inside(A, poly), inside(B, poly)
    c = lambda b: b[:, 2:4] + b[:, 4:6] / 2
    bl = lambda b: np.sqrt(b[:, 4] * b[:, 5])
    cand = []
    for i in np.flatnonzero(~own):
        dt = B[:, 0] - A[i, 0]                          # dt >= 1 also rules out b = a
        d = np.hypot(*(c(B) - c(A[i:i + 1])).T) / ((bl(B) + bl(A[i:i + 1])) / 2)
        cand += [(dt[j], d[j], ids[i], ids[j]) for j in np.flatnonzero((dt >= 1) & (dt <= 3) & (ib != ia[i]) & ~own & (d <= 2 * dt))]
    used, joins = set(), []
    for dt, d, a, b in sorted(cand):
        if a not in used and b not in used:
            used |= {a, b}
            joins.append((float(a), float(b), float(dt), float(d)))
    to = {b: a for a, b, _, _ in joins}
    return [[r[0], to.get(r[1], r[1]), *r[2:]] for r in R.tolist()], joins


_q = [[0, 0], [100, 0], [100, 100], [0, 100]]
_x = lambda f, i, x: [f, i, x - 5, 45, 10, 10, 0.9]               # 10 x 10 box (body length 10) centred on (x, 50)
_a = [_x(1, 1, 130), _x(2, 1, 120), _x(3, 1, 110)]                # track 1 ends outside at x = 110 in frame 3
_j = lambda *t: [(a, b) for a, b, _, _ in crossing_join(_a + list(t) + [_x(f, 9, 50) for f in range(1, 9)], _q)[1]]
assert _j(_x(4, 2, 95), _x(5, 2, 85)) == [(1, 2)]                 # out -> in, 1.5 body lengths: joined
assert [r[1] for r in crossing_join(_a + [_x(4, 2, 95)], _q)[0]] == [1, 1, 1, 1]
assert _j(_x(4, 2, 90)) == [(1, 2)] and _j(_x(4, 2, 85)) == []    # 2.0 <= 2 x dt joins, 2.5 does not
assert _j(_x(6, 2, 85)) == [(1, 2)] and _j(_x(7, 2, 95)) == []    # dt = 3 joins (2.5 <= 6), dt = 4 never
assert _j(_x(4, 2, 105)) == []                                    # same side
assert _j(_x(4, 2, 95), _x(5, 2, 120)) == []                      # b has its own event (out)
assert crossing_join([_x(1, 1, 120), _x(2, 1, 105), _x(3, 1, 95), _x(4, 2, 103)], _q)[1] == []   # a has its own event (in)
assert _j(_x(4, 2, 95), _x(4, 3, 92)) == [(1, 2)]                 # two b's for one a: the nearer one only
assert _j(_x(4, 2, 95), _x(5, 2, 90), _x(6, 3, 101), _x(7, 3, 110)) == [(2, 3)]   # chain 1 -> 2 -> 3: 2 only once
_r = np.array(sorted(_a + [_x(f, 9, 50) for f in range(1, 9)]), float)
_l = sorted(link(_r.tolist(), THR, lambda X: np.zeros(len(X))).tolist())   # no link, no join: P0 + CJ = p0()
assert crossing_join(_l, _q) == (_l, []) and drop_short(sorted(crossing_join(_l, _q)[0]), MIN_LEN, 0.0) == p0(_r, lambda X: np.zeros(len(X)))


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
