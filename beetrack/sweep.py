"""Run many tracker settings on cached detections and print one metrics table (tuning and ablation, val only).
run: .venv\\Scripts\\python -m beetrack.sweep --dets y11s_640 --split val --grid ablation
writes runs/eval/<split>/sweep_<grid>_<dets>.json
"""
import argparse
import json
import shutil

import numpy as np

from .common import RUNS, split_specs, write_mot
from .evaluate import count_metrics, trackeval_metrics
from .track import BYTETRACK, OURS, Tracker, parse_set, run_sequence

# (name, tracker, overrides) - each ablation row switches exactly one component of ours back to ByteTrack's
GRIDS = {
    "ablation": [
        ("bytetrack", "bytetrack", {}),
        ("ours", "ours", {}),
        ("ours-no-buffer", "ours", dict(buffer=0.0, grow=0.0, speed=0.0)),
        ("ours-no-adaptive", "ours", dict(grow=0.0, speed=0.0)),
        ("ours-no-damp", "ours", dict(damp=1.0)),
        ("ours-default-Q", "ours", dict(q_vel=1 / 160)),
        ("ours-buffer30", "ours", dict(track_buffer=30)),
    ],
    "buffer": [(f"b{b}_g{g}_s{s}", "ours", dict(buffer=b, grow=g, speed=s))
               for b in (0.1, 0.2, 0.3, 0.5) for g in (0.0, 0.1) for s in (0.0, 0.5, 1.0)],
    "motion": [(f"d{d}_q{q}_tb{tb}", "ours", dict(damp=d, q_vel=1 / q, track_buffer=tb))
               for d in (0.8, 0.9, 1.0) for q in (40, 80, 160) for tb in (30, 60, 90)],
    "thresh": [(f"{t}_h{h}_n{n}_m{m}", t, dict(track_high_thresh=h, new_track_thresh=n, match_thresh=m))
               for t in ("bytetrack", "ours") for h in (0.25, 0.4, 0.5) for n in (0.25, 0.5, 0.6) for m in (0.8, 0.9)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="y11s_640")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--grid", default="ablation", choices=list(GRIDS))
    ap.add_argument("--base", nargs="*", default=[], help="overrides applied to every 'ours' row, e.g. buffer=0.2")
    a = ap.parse_args()
    base = parse_set(a.base)
    rows = GRIDS[a.grid]
    specs = split_specs(a.split)
    dets = {ev: np.load(RUNS / "dets" / a.dets / a.split / f"{ev}.npy") for ev, *_ in specs}
    names = []
    for name, kind, over in rows:
        tag = f"sweep_{name}"
        params = {**(base if kind == "ours" else {}), **over}
        for ev, _, s0, s1 in specs:
            out, _ = run_sequence(dets[ev], s1 - s0 + 1, Tracker(kind, **params))
            write_mot(RUNS / "trackers" / a.split / tag / "data" / f"{ev}.txt", out)
        names.append(tag)
    m = trackeval_metrics(a.split, names)
    c = count_metrics(a.split, names)
    table = []
    for (name, kind, over), tag in zip(rows, names):
        tot = m[tag]["COMBINED_SEQ"]
        err = sum(v["abs_err"] for v in c[tag].values())
        gt = sum(v["gt_in"] + v["gt_out"] for v in c[tag].values())
        params = {**(OURS if kind == "ours" else BYTETRACK), **(base if kind == "ours" else {}), **over}
        table.append(dict(name=name, tracker=kind, params=params, **{k: round(tot[k], 2) for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1")},
                          IDSW=tot["IDSW"], count_err=err, gt_crossings=gt))
        shutil.rmtree(RUNS / "trackers" / a.split / tag)
    print(f"{'setting':34s} {'HOTA':>6s} {'AssA':>6s} {'IDF1':>6s} {'MOTA':>6s} {'IDSW':>6s} {'count':>8s}")
    for r in sorted(table, key=lambda r: -r["HOTA"]):
        print(f"{r['name']:34s} {r['HOTA']:6.1f} {r['AssA']:6.1f} {r['IDF1']:6.1f} {r['MOTA']:6.1f} {r['IDSW']:6d} {r['count_err']:4d}/{r['gt_crossings']}")
    (RUNS / "eval" / a.split).mkdir(parents=True, exist_ok=True)
    (RUNS / "eval" / a.split / f"sweep_{a.grid}_{a.dets}.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
