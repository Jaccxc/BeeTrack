"""Run many tracker settings on cached detections and print one metrics table (tuning and ablation, val only).
run: uv run python -m beetrack.sweep --dets y11s_640 --split val --grid ablation
writes runs/eval/<split>/sweep_<grid>_<dets>.json
"""
import argparse
import json
import shutil
from multiprocessing import Pool

import numpy as np

from .common import RUNS, split_specs, write_mot
from .evaluate import count_metrics, trackeval_metrics
from .track import BYTETRACK, OURS, Tracker, parse_set, run_sequence

# (name, tracker, overrides) - each ablation row switches exactly one component of ours back to ByteTrack's
GRIDS = {
    "ablation": [
        ("bytetrack", "bytetrack", {}),
        ("bytetrack-tuned", "bytetrack", dict(new_track_thresh=0.5)),   # same threshold tuning as ours, no new parts
        ("ours", "ours", {}),
        ("ours-default-thresh", "ours", dict(new_track_thresh=0.25)),
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
    "extra": [("bytetrack-tuned", "bytetrack", dict(new_track_thresh=0.5)),
              ("bytetrack-tuned-interp10", "bytetrack", dict(new_track_thresh=0.5, interp=10)),
              ("ours", "ours", {}), ("ours-second", "ours", dict(second=True)),
              ("ours-interp5", "ours", dict(interp=5)), ("ours-interp10", "ours", dict(interp=10)),
              ("ours-interp20", "ours", dict(interp=20)), ("ours-second-interp10", "ours", dict(second=True, interp=10))],
    "stitch": [("ours", "ours", {}), ("bytetrack-tuned", "bytetrack", dict(new_track_thresh=0.5))]
              + [(f"ours-stitch{g}-d{d}", "ours", dict(stitch=g, stitch_dist=d)) for g in (10, 25, 50) for d in (1.0, 2.0, 4.0)]
              + [("bytetrack-tuned-stitch25-d2.0", "bytetrack", dict(new_track_thresh=0.5, stitch=25, stitch_dist=2.0))],
    "zstitch": [(f"ours-stitch{g}-d{d}", "ours", dict(stitch=g, stitch_dist=d)) for g in (0, 5, 10, 20) for d in (2.0, 4.0, 6.0)],
    "short": [("ours", "ours", {})] + [(f"ours-len{l}-s{s}", "ours", dict(min_len=l, min_score=s))
                                       for l in (5, 10, 25, 50) for s in (0, 0.5, 0.6, 0.7)],
    "thresh": [(f"{t}_h{h}_n{n}_m{m}", t, dict(track_high_thresh=h, new_track_thresh=n, match_thresh=m))
               for t in ("bytetrack", "ours") for h in (0.25, 0.4, 0.5) for n in (0.25, 0.5, 0.6) for m in (0.8, 0.9)],
}


def run_one(dets_tag, split, name, kind, over):
    """Track every sequence of the split with one setting, evaluate it, delete the track files, return a table row."""
    tag = f"sweep_{name}"
    for ev, _, s0, s1 in split_specs(split):
        out, _ = run_sequence(np.load(RUNS / "dets" / dets_tag / split / f"{ev}.npy"), s1 - s0 + 1, Tracker(kind, **over))
        write_mot(RUNS / "trackers" / split / tag / "data" / f"{ev}.txt", out)
    tot = trackeval_metrics(split, [tag])[tag]["COMBINED_SEQ"]
    c = count_metrics(split, [tag])[tag]
    z = count_metrics(split, [tag], rule="zone")[tag]
    shutil.rmtree(RUNS / "trackers" / split / tag)
    return dict(name=name, tracker=kind, params={**(OURS if kind == "ours" else BYTETRACK), **over},
                **{k: round(tot[k], 2) for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1")}, IDSW=tot["IDSW"],
                count_err=sum(v["abs_err"] for v in c.values()), gt_crossings=sum(v["gt_in"] + v["gt_out"] for v in c.values()),
                zone_err=sum(v["abs_err"] for v in z.values()), zone_gt=sum(v["gt_in"] + v["gt_out"] for v in z.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="y11s_640")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--grid", default="ablation", choices=list(GRIDS))
    ap.add_argument("--base", nargs="*", default=[], help="overrides applied to every 'ours' row, e.g. buffer=0.2")
    ap.add_argument("--jobs", type=int, default=1, help="settings evaluated in parallel processes (CPU only)")
    a = ap.parse_args()
    base = parse_set(a.base)
    jobs = [(a.dets, a.split, name, kind, {**(base if kind == "ours" else {}), **over}) for name, kind, over in GRIDS[a.grid]]
    with Pool(a.jobs) as pool:
        table = pool.starmap(run_one, jobs)
    print(f"{'setting':34s} {'HOTA':>6s} {'AssA':>6s} {'IDF1':>6s} {'MOTA':>6s} {'IDSW':>6s} {'count':>8s} {'zone':>8s}")
    for r in sorted(table, key=lambda r: -r["HOTA"]):
        print(f"{r['name']:34s} {r['HOTA']:6.1f} {r['AssA']:6.1f} {r['IDF1']:6.1f} {r['MOTA']:6.1f} {r['IDSW']:6d} {r['count_err']:4d}/{r['gt_crossings']} {r['zone_err']:4d}/{r['zone_gt']}")
    (RUNS / "eval" / a.split).mkdir(parents=True, exist_ok=True)
    (RUNS / "eval" / a.split / f"sweep_{a.grid}_{a.dets}.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
