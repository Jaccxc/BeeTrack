"""Select crossing witnesses from prepared source-tail GT and measure two-endpoint visibility."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from beetrack.evaluate import by_track, load_tracks, zone_tracks
from beetrack.track import iou_matrix
from inspector import sha, write_once


def save(path, value):
    return write_once(path, value)


def _polygon(points):
    poly = np.asarray(points, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all() or cv2.contourArea(poly) <= 0:
        raise ValueError("Invalid entrance polygon")
    return poly


def _inside(poly, box):
    x1, y1, x2, y2 = map(float, box)
    return cv2.pointPolygonTest(poly, ((x1 + x2) / 2, (y1 + y2) / 2), False) >= 0


def _rank(clip, event):
    key = f"beetrack-warp-pilot-v1|{clip}|{event['dir']}|{event['gt_tid']}|{event['arrival']}"
    return hashlib.sha256(key.encode("utf8")).hexdigest()


def prepare_sample(gt_mot, polygon, clip, nframes):
    """Return an unlabelled frame request and a separate private full-frame label package."""
    if not isinstance(clip, str) or not clip or "|" in clip or type(nframes) is not int or nframes < 1:
        raise ValueError("Invalid physical clip or frame count")
    poly = _polygon(polygon)
    raw = np.asarray(gt_mot, dtype=float)
    if raw.size == 0:
        raw = np.empty((0, 6))
    if raw.ndim != 2 or raw.shape[1] < 6:
        raise ValueError("Expected MOT [frame,gt_id,x,y,w,h,...]")
    raw = raw[:, :6]
    if (not np.isfinite(raw).all() or np.any(raw[:, :2] != np.floor(raw[:, :2]))
            or np.any(raw[:, 0] < 1) or np.any(raw[:, 0] > nframes)
            or np.any(raw[:, 1] < 0) or np.any(raw[:, 4:6] < 0)):
        raise ValueError("Invalid frame, GT ID, or box extent")
    rows = by_track(raw, max_gap=25)
    if len(np.unique(rows[:, :2], axis=0)) != len(rows):
        raise ValueError("Ambiguous duplicate resolved GT ID at one frame")
    candidates = []
    for tid, (direction, _, _, changes) in zone_tracks(rows, poly).items():
        if direction:
            j = next(j for j, d in reversed(changes) if d == direction)
            event = dict(gt_tid=tid, dir=direction, before=int(rows[j-1, 0]), arrival=int(rows[j, 0]))
            event["rank"] = _rank(clip, event)
            candidates.append(event)

    candidates.sort(key=lambda e: (e["rank"], e["gt_tid"], e["arrival"]))
    selected = []
    by_dir = {d: [e for e in candidates if e["dir"] == d] for d in ("in", "out")}
    for direction in by_dir:
        selected.extend(by_dir[direction][:8])
    taken = {e["gt_tid"] for e in selected}
    selected.extend([e for e in candidates if e["gt_tid"] not in taken][:16-len(selected)])
    selected.sort(key=lambda e: (e["arrival"], e["gt_tid"]))
    frames = sorted({e[f] for e in selected for f in ("before", "arrival")})
    requests = dict(clip=clip, nframes=nframes, frames=frames)
    gt_frames = {}
    for f in frames:
        gt_frames[str(f)] = [
            dict(gt_tid=int(tid), xywh=[x,y,w,h], xyxy=[x,y,x+w,y+h],
                 inside=_inside(poly, [x,y,x+w,y+h]))
            for _,tid,x,y,w,h in rows[rows[:,0] == f].tolist()
        ]
    counts = Counter(e["dir"] for e in selected)
    population = dict(planned=16, selected=len(selected), available=len(candidates),
                      available_by_direction={d: len(v) for d, v in by_dir.items()},
                      selected_by_direction={d: counts[d] for d in by_dir},
                      shortfall=16-len(selected))
    labels = dict(format="warp-pilot-private-labels-v1", clip=clip, nframes=nframes,
                  polygon=poly.tolist(), requested_frames=frames, events=selected,
                  gt_frames=gt_frames, population=population)
    return requests, labels


def _match_frame(frame, truth, detections, polygon):
    """Official maximum-sum IoU matching of *all* boxes before thresholding."""
    gt = np.array([[frame, t["gt_tid"], *t["xywh"]] for t in truth], dtype=float).reshape(-1, 6)
    survivors = sorted(((i,d) for i,d in enumerate(detections) if d[4] >= .25),
                       key=lambda q: (-q[1][4], *q[1][:4], q[0]))
    pr = np.array([[frame, i, d[0], d[1], d[2]-d[0], d[3]-d[1]] for i,d in survivors],
                  dtype=float).reshape(-1, 6)
    # Use frame_match's exact xywh round trip, IoU kernel and solver once per
    # frame. The two cutoffs share one fixed one-to-one assignment.
    gt_xyxy = np.c_[gt[:,2:4],gt[:,2:4]+gt[:,4:6]]
    pr_xyxy = np.c_[pr[:,2:4],pr[:,2:4]+pr[:,4:6]]
    overlaps = iou_matrix(gt_xyxy,pr_xyxy)
    gi, pj = linear_sum_assignment(-overlaps)
    assigned = {int(j):(int(gt[i,1]),float(overlaps[i,j])) for i,j in zip(gi,pj) if overlaps[i,j]>=.3}
    by_id = {t["gt_tid"]:t for t in truth}
    matches = {}
    pred_rows = []
    for j, (i,d) in enumerate(survivors):
        tid, overlap = assigned.get(j,(-1,0.))
        pred_rows.append(dict(index=i, xyxy=d[:4], score=d[4], gt_tid03=None if tid < 0 else tid))
        if tid >= 0:
            side = _inside(polygon, d[:4])
            matches[tid] = dict(index=i, xyxy=d[:4], score=d[4], iou=overlap,
                                owner05=overlap>=.5,
                                predicted_inside=side, correct_side=side == by_id[tid]["inside"])
    return matches, pred_rows


def score_dose(labels, predictions):
    """Private scoring after unlabelled detections have been frozen. No tracker association."""
    if (predictions["clip"] != labels["clip"] or predictions["nframes"] != labels["nframes"]
            or predictions["requested_frames"] != labels["requested_frames"]):
        raise ValueError("Unlabelled prediction request and labels differ")
    request = dict(clip=labels["clip"], nframes=labels["nframes"], frames=labels["requested_frames"])
    if predictions["requests_sha256"] != sha(request):
        raise ValueError("Invalid request digest")
    run, status = predictions["run"], predictions["status"]
    if status not in ("available", "unavailable"):
        raise ValueError("Unknown dose status")
    available = status == "available"
    frozen = predictions["predictions"]
    if available and (not isinstance(frozen, dict) or set(frozen) != set(labels["gt_frames"])):
        raise ValueError("Available dose must contain every requested frame, including zero-box frames")
    if available and predictions["predictions_sha256"] != sha(frozen):
        raise ValueError("Detector rows changed after freeze")
    if not available and (frozen is not None or not run.get("unavailable_reason")):
        raise ValueError("Unavailable dose must carry a reason and no observations")
    poly = _polygon(labels["polygon"])
    frames, owners = {}, {}
    for f, truth in labels["gt_frames"].items():
        if available:
            matches, pred_rows = _match_frame(int(f), truth, frozen[f], poly)
            owners[int(f)] = matches
            frames[f] = dict(predictions=pred_rows, matches={str(k):v for k,v in matches.items()})
    out = []
    for event in labels["events"]:
        tid = event["gt_tid"]
        ends = []
        for key in ("before", "arrival"):
            f = event[key]
            gt = next(t for t in labels["gt_frames"][str(f)] if t["gt_tid"] == tid)
            ends.append(dict(frame=f, gt_inside=gt["inside"], gt_xyxy=gt["xyxy"],
                             match=owners[f].get(tid) if available else None))
        row = dict(event, ends=ends)
        for name in ("visible05", "visible03"):
            row[name] = all(x["match"] is not None and x["match"]["correct_side"]
                            and (name == "visible03" or x["match"]["owner05"]) for x in ends) if available else None
        out.append(row)
    return dict(clip=labels["clip"], fold=run["fold"], seed=run["seed"], dose=run["dose"],
                status=status, reason=run.get("unavailable_reason"), population=labels["population"],
                predictions_sha256=predictions["predictions_sha256"], events=out, frames=frames)


PREREG = "c8fabf6c55c9737086b02050fabb3360bb5f1db2"


def source_roster(fold, output, runs, zones, lineage):
    clip = f"BEE24-{68-fold}-tail"
    gt_path = runs / "gt" / "val" / clip / "gt" / "gt.txt"
    gt_sha = hashlib.sha256(gt_path.read_bytes()).hexdigest()
    zones_sha = hashlib.sha256(zones.read_bytes()).hexdigest()
    for seed in range(3):
        ref = lineage / "source" / "labels" / f"{fold}_{seed}_tail.json"
        if json.loads(ref.read_bytes())["gt_mot_sha256"] != gt_sha:
            raise ValueError(f"Source GT differs from archived §108 seed {seed}")
    requests, labels = prepare_sample(load_tracks(gt_path), json.loads(zones.read_bytes())[clip], clip, 1000)
    labels["sources_sha256"] = dict(gt_mot=gt_sha, zones=zones_sha)
    labels["prereg"] = PREREG
    return dict(fold=fold, clip=clip, selected=labels["population"]["selected"],
                frames=len(requests["frames"]), sources_sha256=labels["sources_sha256"],
                labels_file=f"{clip}/labels.private.json", requests_file=f"{clip}/requests.json",
                labels_sha256=save(output / clip / "labels.private.json", labels),
                requests_sha256=save(output / clip / "requests.json", requests))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    src = sub.add_parser("sample-source")
    src.add_argument("fold", type=int, choices=(33,35))
    freeze = sub.add_parser("freeze-rosters")
    for part in (src, freeze):
        part.add_argument("output", type=Path)
        part.add_argument("--runs", type=Path, default=Path("runs"))
        part.add_argument("--zones", type=Path, default=Path("beetrack/zones.json"))
        part.add_argument("--lineage-dir", type=Path, required=True)
    score = sub.add_parser("score")
    for key in ("labels","predictions","output"):
        score.add_argument(key, type=Path)
    args = p.parse_args()
    if args.mode in ("sample-source","freeze-rosters"):
        folds = (args.fold,) if args.mode == "sample-source" else (33,35)
        entries = [source_roster(f, args.output, args.runs, args.zones, args.lineage_dir) for f in folds]
        if len(folds) == 2:
            path = args.output / "index.json"
            print(save(path, dict(format="warp-pilot-roster-index-v1", prereg=PREREG, cases=entries)), path)
        print(json.dumps(entries, sort_keys=True))
    else:
        labels = json.loads(args.labels.read_bytes())
        preds = json.loads(args.predictions.read_bytes())
        save(args.output, score_dose(labels, preds))


if __name__ == "__main__":
    main()
