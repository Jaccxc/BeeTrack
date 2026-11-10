"""Tracking metrics (TrackEval: HOTA, CLEAR/MOTA, Identity/IDF1) and line-crossing count error.
run: uv run python -m beetrack.evaluate --split test --trackers bytetrack ours
tracker outputs are read from runs/trackers/<split>/<name>/data/<seq>.txt
"""
import argparse
import json

import numpy as np

from .common import RUNS, split_specs

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
    rows = rows[np.lexsort((rows[:, 0], rows[:, 1]))]
    for tid in np.unique(rows[:, 1]):
        t = rows[rows[:, 1] == tid]
        flips = np.diff(side_changes(t[:, 3] + t[:, 5] / 2, line_y, band))
        n_in += int(np.sum(flips < 0))
        n_out += int(np.sum(flips > 0))
    return n_in, n_out


def count_metrics(split, names, band=BAND):
    """Absolute counting error per sequence, line at mid-height, hysteresis band = band x frame height."""
    out = {n: {} for n in names}
    for ev, *_ in split_specs(split.split("_")[0]):   # test_step2 -> test sequences
        d = RUNS / "gt" / split / ev
        h = int(next(l for l in (d / "seqinfo.ini").read_text().split() if l.startswith("imHeight")).split("=")[1])
        gi, go = crossings(load_tracks(d / "gt" / "gt.txt"), h / 2, band * h)
        for n in names:
            pi, po = crossings(load_tracks(RUNS / "trackers" / split / n / "data" / f"{ev}.txt"), h / 2, band * h)
            out[n][ev] = dict(gt_in=gi, gt_out=go, pred_in=pi, pred_out=po, abs_err=abs(pi - gi) + abs(po - go))
    return out


assert crossings(np.array([[1, 1, 0, 60, 1, 0], [2, 1, 0, 49, 1, 0], [3, 1, 0, 52, 1, 0], [4, 1, 0, 40, 1, 0]]), 50, 5) == (1, 0)
assert crossings(np.array([[1, 1, 0, 60, 1, 0], [2, 1, 0, 49, 1, 0], [3, 1, 0, 52, 1, 0], [4, 1, 0, 40, 1, 0]]), 50, 0) == (2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="val, test, or test_step<k> (frame-drop GT written by prep.py)")
    ap.add_argument("--trackers", nargs="+", required=True)
    a = ap.parse_args()
    m = trackeval_metrics(a.split, a.trackers)
    c = count_metrics(a.split, a.trackers)
    raw = count_metrics(a.split, a.trackers, band=0.0)   # plain-line rule, reported for comparison
    for n in a.trackers:
        for ev, v in c[n].items():
            m[n][ev].update(v, raw_gt=raw[n][ev]["gt_in"] + raw[n][ev]["gt_out"], raw_abs_err=raw[n][ev]["abs_err"])
        tot = m[n]["COMBINED_SEQ"]
        tot.update(gt_crossings=sum(v["gt_in"] + v["gt_out"] for v in c[n].values()),
                   count_abs_err=sum(v["abs_err"] for v in c[n].values()),
                   raw_gt_crossings=sum(v["gt_in"] + v["gt_out"] for v in raw[n].values()),
                   raw_count_abs_err=sum(v["abs_err"] for v in raw[n].values()))
        print(f"{n:28s} HOTA {tot['HOTA']:.1f}  IDF1 {tot['IDF1']:.1f}  MOTA {tot['MOTA']:.1f}  IDSW {tot['IDSW']}  "
              f"count err {tot['count_abs_err']}/{tot['gt_crossings']} (plain line {tot['raw_count_abs_err']}/{tot['raw_gt_crossings']})")
        (RUNS / "eval" / a.split).mkdir(parents=True, exist_ok=True)
        (RUNS / "eval" / a.split / f"{n}.json").write_text(json.dumps(m[n], indent=1))


if __name__ == "__main__":
    main()
