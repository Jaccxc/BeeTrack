"""Tracking metrics (TrackEval: HOTA, CLEAR/MOTA, Identity/IDF1), count error and event-level counting accuracy.
run: uv run python -m beetrack.evaluate --split test --trackers bytetrack ours
tracker outputs are read from runs/trackers/<split>/<name>/data/<seq>.txt

The count error |pred - gt| lets a missed event and a false event cancel. The event columns do not: a predicted in/out
event is true only if its track is on the right bee when it crosses the zone boundary (match_events).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import RUNS, split_specs
from .track import iou_matrix

ZONES = Path(__file__).with_name("zones.json")

for _k, _v in (("float", float), ("int", int), ("bool", bool)):   # TrackEval still uses aliases removed in numpy 1.24
    if not hasattr(np, _k):
        setattr(np, _k, _v)
import trackeval  # noqa: E402


def trackeval_metrics(split, names):
    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update(USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
              PLOT_CURVES=False, TIME_PROGRESS=False, DISPLAY_LESS_PROGRESS=True, PRINT_ONLY_COMBINED=True, OUTPUT_EMPTY_CLASSES=False)
    dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dc.update(GT_FOLDER=str(RUNS / "gt" / split), TRACKERS_FOLDER=str(RUNS / "trackers" / split), SKIP_SPLIT_FOL=True,
              SEQMAP_FILE=str(RUNS / "gt" / f"{split}.seqmap"), GT_LOC_FORMAT="{gt_folder}/{seq}/gt/gt.txt",
              TRACKERS_TO_EVAL=list(names), TRACKER_SUB_FOLDER="data", OUTPUT_FOLDER=str(RUNS / "eval" / split),
              DO_PREPROC=False, PRINT_CONFIG=False)
    res, _ = trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)],
                                              [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
    out = {}
    for name in names:
        per = {}
        for seq, r in res["MotChallenge2DBox"][name].items():
            p = r["pedestrian"]
            per[seq] = dict(HOTA=100 * float(np.mean(p["HOTA"]["HOTA"])), DetA=100 * float(np.mean(p["HOTA"]["DetA"])),
                            AssA=100 * float(np.mean(p["HOTA"]["AssA"])), MOTA=100 * float(p["CLEAR"]["MOTA"]),
                            IDF1=100 * float(p["Identity"]["IDF1"]), IDSW=int(p["CLEAR"]["IDSW"]),
                            FP=int(p["CLEAR"]["CLR_FP"]), FN=int(p["CLEAR"]["CLR_FN"]))
        out[name] = per
    return out


def load_tracks(path):
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return a[:, :6] if a.size else np.zeros((0, 6))


BAND = 0.05   # hysteresis half-width, fraction of frame height (about half a bee); fixed a priori, not tuned


def side_changes(cy, line_y, band):
    """Sides (+1 below, -1 above) a centre track is assigned to; inside the band the previous side is kept."""
    side, out = 0, []
    for y in cy:
        s = 1 if y > line_y + band else -1 if y < line_y - band else side
        if s and s != side:
            out.append(s)
        side = s
    return out


def crossings(rows, line_y, band=0.0):
    """Counting protocol (same for GT and predictions): a track's box centre moving from below the band
    [line_y - band, line_y + band] to above it is one 'in' (upward), the reverse one 'out'. band=0 is the
    plain line rule, which also counts every wiggle of a bee resting on the line."""
    n_in = n_out = 0
    rows = by_track(rows)
    for tid in np.unique(rows[:, 1]):
        t = rows[rows[:, 1] == tid]
        flips = np.diff(side_changes(t[:, 3] + t[:, 5] / 2, line_y, band))
        n_in += int(np.sum(flips < 0))
        n_out += int(np.sum(flips > 0))
    return n_in, n_out


MAX_GAP = 25   # frames (1 s), fixed a priori


def by_track(rows, max_gap=MAX_GAP):
    """Rows sorted by track, then frame. For counting, a track is one ID with no frame gap longer than max_gap: BEE24's
    long videos use the same ID number again for another bee (64 times in test BEE24-36, 704 in the training part of
    BEE24-33; the box comes back a median of 6-7 body lengths away), so an ID is not a bee. The k-th such run of an ID
    becomes id + k * 1e5. Applied to ground truth and predictions alike."""
    rows = np.array(rows, float)
    rows = rows[np.lexsort((rows[:, 0], rows[:, 1]))]
    if len(rows):
        same = rows[1:, 1] == rows[:-1, 1]
        cuts = np.cumsum(np.r_[False, same & (np.diff(rows[:, 0]) > max_gap)])
        rows[:, 1] += 1e5 * (cuts - np.maximum.accumulate(np.where(np.r_[True, ~same], cuts, 0)))
    return rows[np.lexsort((rows[:, 0], rows[:, 1]))]


def merge_handovers(rows):
    """One bee, one ID - ground-truth repair, used for the dev split only. Where a track ends and a new ID starts on
    the same box in the next frame (IoU >= 0.3) or two or three frames later (IoU >= 0.5), the new ID is the same bee.
    The training part of BEE24-33 changes IDs like this all the time (3,588 IDs for about 1,500 bees, most of it in
    its first 1,000 frames: a new ID starts within half a body length of a track's end 473 times one frame later, but
    only 13-42 times at each lag of 2-10 frames). Val has 49 such handovers and test 51; they are left as they are."""
    g = by_track(rows)
    ids, start = np.unique(g[:, 1], return_index=True)
    first, last = g[start], g[np.r_[start[1:], len(g)] - 1]
    xyxy = lambda b: np.c_[b[:, 2:4], b[:, 2:4] + b[:, 4:6]]
    parent = {}
    for gap, thr in ((1, 0.3), (2, 0.5), (3, 0.5)):
        for f in np.unique(last[:, 0]):
            a = last[(last[:, 0] == f) & ~np.isin(last[:, 1], list(parent.values()))]
            b = first[(first[:, 0] == f + gap) & ~np.isin(first[:, 1], list(parent))]
            if len(a) and len(b):
                m = iou_matrix(xyxy(a), xyxy(b))
                parent.update({b[j, 1]: a[i, 1] for i, j in zip(*linear_sum_assignment(-m)) if m[i, j] >= thr})

    def root(x):
        while x in parent:
            x = parent[x]
        return x
    g[:, 1] = np.array([root(x) for x in ids])[np.searchsorted(ids, g[:, 1])]
    return g[np.lexsort((g[:, 0], g[:, 1]))]


def zone_tracks(rows, poly):
    """Entrance-zone protocol (same for GT and predictions), per track of `rows` (sorted by id, then frame):
    id -> (event, first row, end row, [(row, direction) of every boundary crossing]). event: first box centre outside
    the polygon and last centre inside = 'in', the reverse = 'out', same side at both ends = None."""
    poly = np.asarray(poly, np.float32)
    ins = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in rows[:, 2:4] + rows[:, 4:6] / 2], bool)
    ids, start = np.unique(rows[:, 1], return_index=True)
    out = {}
    for tid, s, e in zip(ids, start, [*start[1:], len(rows)]):
        i = ins[s:e]
        ev = "in" if i[-1] and not i[0] else "out" if i[0] and not i[-1] else None
        out[int(tid)] = (ev, s, e, [(s + k + 1, "in" if i[k + 1] else "out") for k in np.flatnonzero(i[1:] != i[:-1])])
    return out


def zone_crossings(rows, poly):
    """Number of 'in' and 'out' events (zone_tracks)."""
    ev = [t[0] for t in zone_tracks(by_track(rows), poly).values()]
    return ev.count("in"), ev.count("out")


def frame_match(gt, pr, thr=0.5):
    """GT id under every prediction row (IoU >= thr, one-to-one within a frame), -1 where there is none."""
    xyxy = lambda b: np.c_[b[:, 2:4], b[:, 2:4] + b[:, 4:6]]

    def groups(a):
        o = np.argsort(a[:, 0], kind="stable")
        f, s = np.unique(a[o, 0], return_index=True)
        return dict(zip(f, np.split(o, s[1:])))
    G, out = groups(gt), np.full(len(pr), -1)
    for f, p in groups(pr).items():
        g = G.get(f)
        if g is not None:
            m = iou_matrix(xyxy(gt[g]), xyxy(pr[p]))
            i, j = linear_sum_assignment(-m)
            ok = m[i, j] >= thr
            out[p[j[ok]]] = gt[g[i[ok]], 1]
    return out


def match_events(gt, pr, poly, tol=5):
    """Event-level accuracy of the zone count in one sequence: dict(gt, pred, tp, tp_id).
    tp: predicted events that are true. A predicted event is true when, at one of its track's boundary crossings in the
        event's direction, the track is on a GT bee that crosses the same way within `tol` frames and whose own event has
        that direction; every GT event is matched at most once (a bee counted twice gives one true and one false event).
    tp_id: stricter - the track is on that one bee at its first and at its last matched frame (no identity change)."""
    gt, pr = by_track(gt), by_track(pr)
    G, P, under = zone_tracks(gt, poly), zone_tracks(pr, poly), frame_match(gt, pr)
    gev, pev = [g for g, t in G.items() if t[0]], [p for p, t in P.items() if t[0]]
    col, same = {g: k for k, g in enumerate(gev)}, set()
    C = np.zeros((len(pev), len(gev)), int)
    for r, p in enumerate(pev):
        d, s, e, cross = P[p]
        u = under[s:e][under[s:e] >= 0]
        if len(u) and u[0] == u[-1] and G[u[0]][0] == d:
            same.add(u[0])
        for j, dj in cross:
            for g in {under[j - 1], under[j]} - {-1} if dj == d else ():
                if G[g][0] == d and any(dg == d and abs(gt[i, 0] - pr[j, 0]) <= tol for i, dg in G[g][3]):
                    C[r, col[g]] = 1
    return dict(gt=len(gev), pred=len(pev), tp=int(C[linear_sum_assignment(C, maximize=True)].sum()), tp_id=len(same))


def event_metrics(split, names):
    """match_events for every sequence: {name: {seq: dict}}."""
    zones = json.loads(ZONES.read_text())
    out = {n: {} for n in names}
    for ev, *_ in split_specs(split.split("_")[0]):
        gt = load_tracks(RUNS / "gt" / split / ev / "gt" / "gt.txt")
        for n in names:
            out[n][ev] = match_events(gt, load_tracks(RUNS / "trackers" / split / n / "data" / f"{ev}.txt"), zones[ev])
    return out


def f1(tp, pred, gt):
    return 2 * tp / max(pred + gt, 1)


def count_metrics(split, names, band=BAND, rule="line"):
    """Absolute counting error per sequence. rule="line": line at mid-height, hysteresis band = band x frame height;
    rule="zone": entrance polygon from zones.json."""
    out = {n: {} for n in names}
    zones = json.loads(ZONES.read_text()) if rule == "zone" else None
    for ev, *_ in split_specs(split.split("_")[0]):   # test_step2 -> test sequences
        d = RUNS / "gt" / split / ev
        h = int(next(l for l in (d / "seqinfo.ini").read_text().split() if l.startswith("imHeight")).split("=")[1])
        f = (lambda r: zone_crossings(r, zones[ev])) if zones else (lambda r: crossings(r, h / 2, band * h))
        gi, go = f(load_tracks(d / "gt" / "gt.txt"))
        for n in names:
            pi, po = f(load_tracks(RUNS / "trackers" / split / n / "data" / f"{ev}.txt"))
            out[n][ev] = dict(gt_in=gi, gt_out=go, pred_in=pi, pred_out=po, abs_err=abs(pi - gi) + abs(po - go))
    return out


assert crossings(np.array([[1, 1, 0, 60, 1, 0], [2, 1, 0, 49, 1, 0], [3, 1, 0, 52, 1, 0], [4, 1, 0, 40, 1, 0]]), 50, 5) == (1, 0)
assert crossings(np.array([[1, 1, 0, 60, 1, 0], [2, 1, 0, 49, 1, 0], [3, 1, 0, 52, 1, 0], [4, 1, 0, 40, 1, 0]]), 50, 0) == (2, 1)
_sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
assert zone_crossings(np.array([[1, 1, 20, 5, 0, 0], [2, 1, 5, 5, 0, 0]]), _sq) == (1, 0)   # outside -> inside
assert zone_crossings(np.array([[1, 1, 5, 5, 0, 0], [2, 1, 20, 5, 0, 0]]), _sq) == (0, 1)   # inside -> outside
assert zone_crossings(np.array([[1, 1, 20, 5, 0, 0], [2, 1, 5, 5, 0, 0], [3, 1, 20, 5, 0, 0]]), _sq) == (0, 0)   # visit and leave
_r = np.array([[1, 1, 20, 5, 0, 0], [2, 1, 5, 5, 0, 0], [90, 1, 5, 5, 0, 0], [91, 1, 20, 5, 0, 0], [92, 2, 5, 5, 0, 0]], float)
assert zone_crossings(_r, _sq) == (1, 1) and list(by_track(_r)[:, 1]) == [1, 1, 2, 100001, 100001]   # ID 1 used again 88 frames later: two bees
_b = lambda f, i, x, y=4: [f, i, x - 1, y - 1, 2, 2]   # 2 x 2 box centred on (x, y)
_h = np.array([_b(1, 1, 5), _b(2, 1, 5), _b(3, 2, 5), _b(4, 2, 5), _b(3, 3, 50), _b(4, 3, 50)], float)   # ID 1 goes on as ID 2 on the same box
assert list(merge_handovers(_h)[:, 1]) == [1, 1, 1, 1, 3, 3]
_g = np.array([_b(1, 1, 20), _b(2, 1, 14), _b(3, 1, 6), _b(4, 1, 5), *[_b(f, 2, 5, 8) for f in (1, 2, 3, 4)]], float)   # bee 1 walks in, bee 2 sits inside
assert match_events(_g, _g, _sq) == dict(gt=1, pred=1, tp=1, tp_id=1)
_p = np.array([_b(1, 7, 20), _b(2, 7, 14), _b(3, 7, 6), _b(4, 7, 5, 8)], float)   # crosses on bee 1, then jumps onto bee 2
assert match_events(_g, _p, _sq) == dict(gt=1, pred=1, tp=1, tp_id=0)
_g = np.array([_b(1, 1, 20), _b(2, 1, 5), _b(3, 1, 20), _b(4, 1, 5)], float)   # in, out, in: one GT 'in'
_p = np.array([_b(1, 7, 20), _b(2, 7, 5), _b(3, 8, 20), _b(4, 8, 5)], float)   # two fragments, each counts an 'in'
assert match_events(_g, _p, _sq) == dict(gt=1, pred=2, tp=1, tp_id=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="val, test, or test_step<k> (frame-drop GT written by prep.py)")
    ap.add_argument("--trackers", nargs="+", required=True)
    a = ap.parse_args()
    m = trackeval_metrics(a.split, a.trackers)
    c = count_metrics(a.split, a.trackers)
    raw = count_metrics(a.split, a.trackers, band=0.0)   # plain-line rule, reported for comparison
    z = count_metrics(a.split, a.trackers, rule="zone")
    e = event_metrics(a.split, a.trackers)
    for n in a.trackers:
        for ev, v in c[n].items():
            m[n][ev].update(v, raw_gt=raw[n][ev]["gt_in"] + raw[n][ev]["gt_out"], raw_abs_err=raw[n][ev]["abs_err"],
                         **{"zone_" + k: x for k, x in z[n][ev].items()}, **{"ev_" + k: x for k, x in e[n][ev].items()})
        tot = m[n]["COMBINED_SEQ"]
        tot.update(gt_crossings=sum(v["gt_in"] + v["gt_out"] for v in c[n].values()),
                   count_abs_err=sum(v["abs_err"] for v in c[n].values()),
                   raw_gt_crossings=sum(v["gt_in"] + v["gt_out"] for v in raw[n].values()),
                   raw_count_abs_err=sum(v["abs_err"] for v in raw[n].values()),
                   zone_gt_crossings=sum(v["gt_in"] + v["gt_out"] for v in z[n].values()),
                   zone_count_abs_err=sum(v["abs_err"] for v in z[n].values()),
                   **{"ev_" + k: sum(v[k] for v in e[n].values()) for k in ("gt", "pred", "tp", "tp_id")})
        print(f"{n:28s} HOTA {tot['HOTA']:.1f}  IDF1 {tot['IDF1']:.1f}  MOTA {tot['MOTA']:.1f}  IDSW {tot['IDSW']}  "
              f"count err {tot['count_abs_err']}/{tot['gt_crossings']} (plain line {tot['raw_count_abs_err']}/{tot['raw_gt_crossings']})  "
              f"zone err {tot['zone_count_abs_err']}/{tot['zone_gt_crossings']}  "
              f"events F1 {f1(tot['ev_tp'], tot['ev_pred'], tot['ev_gt']):.2f} (true {tot['ev_tp']}, false {tot['ev_pred'] - tot['ev_tp']}, "
              f"missed {tot['ev_gt'] - tot['ev_tp']}; same bee end to end {tot['ev_tp_id']})")
        (RUNS / "eval" / a.split).mkdir(parents=True, exist_ok=True)
        (RUNS / "eval" / a.split / f"{n}.json").write_text(json.dumps(m[n], indent=1))


if __name__ == "__main__":
    main()
