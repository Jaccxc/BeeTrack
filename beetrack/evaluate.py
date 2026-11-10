"""Tracking metrics (TrackEval: HOTA, CLEAR/MOTA, Identity/IDF1) and line-crossing count error.
run: .venv\\Scripts\\python -m beetrack.evaluate --split test --trackers bytetrack ours
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


def crossings(rows, line_y):
    """Counting protocol (same for GT and predictions): a track's box centre moving from one side of the
    horizontal line y = line_y to the other is one crossing; upward = 'in', downward = 'out'."""
    n_in = n_out = 0
    rows = rows[np.lexsort((rows[:, 0], rows[:, 1]))]
    for tid in np.unique(rows[:, 1]):
        cy = rows[rows[:, 1] == tid]
        cy = cy[:, 3] + cy[:, 5] / 2
        side = np.sign(cy - line_y)
        side = side[side != 0]
        flips = np.diff(side)
        n_in += int(np.sum(flips < 0))
        n_out += int(np.sum(flips > 0))
    return n_in, n_out


def count_metrics(split, names):
    """Absolute counting error per sequence with the line at mid-height (documented protocol)."""
    out = {n: {} for n in names}
    for ev, *_ in split_specs(split.split("_")[0]):   # test_step2 -> test sequences
        d = RUNS / "gt" / split / ev
        h = int(next(l for l in (d / "seqinfo.ini").read_text().split() if l.startswith("imHeight")).split("=")[1])
        gi, go = crossings(load_tracks(d / "gt" / "gt.txt"), h / 2)
        for n in names:
            pi, po = crossings(load_tracks(RUNS / "trackers" / split / n / "data" / f"{ev}.txt"), h / 2)
            out[n][ev] = dict(gt_in=gi, gt_out=go, pred_in=pi, pred_out=po, abs_err=abs(pi - gi) + abs(po - go))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="val, test, or test_step<k> (frame-drop GT written by prep.py)")
    ap.add_argument("--trackers", nargs="+", required=True)
    a = ap.parse_args()
    m = trackeval_metrics(a.split, a.trackers)
    c = count_metrics(a.split, a.trackers)
    for n in a.trackers:
        for ev, v in c[n].items():
            m[n][ev].update(v)
        tot = m[n]["COMBINED_SEQ"]
        tot.update(gt_crossings=sum(v["gt_in"] + v["gt_out"] for v in c[n].values()),
                   count_abs_err=sum(v["abs_err"] for v in c[n].values()))
        print(f"{n:28s} HOTA {tot['HOTA']:.1f}  IDF1 {tot['IDF1']:.1f}  MOTA {tot['MOTA']:.1f}  IDSW {tot['IDSW']}  "
              f"count err {tot['count_abs_err']}/{tot['gt_crossings']}")
        (RUNS / "eval" / a.split).mkdir(parents=True, exist_ok=True)
        (RUNS / "eval" / a.split / f"{n}.json").write_text(json.dumps(m[n], indent=1))


if __name__ == "__main__":
    main()
