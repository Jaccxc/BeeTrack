"""Same-detector ByteTrack whole-clip counts for BEE24-10 and BEE24-15."""
import argparse
import hashlib
import json
from pathlib import Path

from beetrack.common import ROOT, RUNS
from beetrack.evaluate import ZONES
from freeze109 import CLIPS, SEEDS, read_events, sha, write_once


def directional_counts(events, nframes):
    seen = set()
    counts = [0, 0]
    for event_id, frame, *_ in events:
        if event_id in seen or int(frame) != frame or not 1 <= abs(frame) <= nframes:
            raise ValueError("Duplicate event ID or invalid signed frame")
        seen.add(event_id)
        counts[0 if frame > 0 else 1] += 1
    return counts


def predict():
    polygons = json.loads(ZONES.read_bytes())
    clips = {}
    for number, nframes in CLIPS.items():
        name = f"BEE24-{number}"
        tracks = {}
        for s in SEEDS:
            tag = f"q_bt_loho_noW_s{s}"
            path = RUNS / "trackers" / "val" / tag / "data" / f"{name}.txt"
            checksum = sha(path)
            events = [row[:2] for row in read_events(path, polygons[name], nframes)]
            if sha(path) != checksum:
                raise ValueError(f"Input changed while reading {name}/{s}")
            tracks[str(s)] = dict(events=events, totals=directional_counts(events, nframes),
                                  mot_path=path.relative_to(ROOT).as_posix(), mot_sha256=checksum)
        means = [sum(t["totals"][d] for t in tracks.values())/3 for d in (0, 1)]
        clips[name] = dict(nframes=nframes, tracks=tracks, e3_totals=means)
    return dict(study=109, zones_sha256=sha(ZONES), clips=clips)


def score_bytetrack(predictions, labels):
    if predictions["zones_sha256"] != labels["zones_sha256"]:
        raise ValueError("Comparator and labels used different entrance polygons")
    clips = {}
    for number, nframes in CLIPS.items():
        name = f"BEE24-{number}"
        pred = predictions["clips"][name]
        gt = directional_counts(labels["clips"][name]["gt"], nframes)

        def score(totals):
            diff = [p-g for p,g in zip(totals, gt)]
            distances = [abs(d) for d in diff]
            return dict(totals=totals, bias=diff, abs_cells=distances, l1=sum(distances))

        if set(pred["tracks"]) != set(map(str, SEEDS)) or pred["nframes"] != nframes:
            raise ValueError(f"Wrong frame/seed coverage for {name}")
        seeds = {}
        for s, track in pred["tracks"].items():
            totals = directional_counts(track["events"], nframes)
            if totals != track["totals"]:
                raise ValueError("Saved counts disagree with the comparator event times")
            seeds[s] = score(totals)
        means = [sum(row["totals"][d] for row in seeds.values())/3 for d in (0,1)]
        if means != pred["e3_totals"]:
            raise ValueError("Saved E3 counts differ from the three detector seeds")
        clips[name] = dict(gt=gt, seeds=seeds, e3=score(means))
    denom = sum(sum(clip["gt"]) for clip in clips.values())
    if denom != 69:
        raise ValueError("Native comparator GT denominator failed the §84 cross-check")

    def aggregate(rows):
        loss = sum(row["l1"] for row in rows)
        return dict(l1=loss, error_pct=100*loss/denom)

    pooled = dict(denominator=denom, e3=aggregate(c["e3"] for c in clips.values()),
                  seeds={str(s): aggregate(c["seeds"][str(s)] for c in clips.values()) for s in SEEDS})
    return dict(clips=clips, pooled=pooled, calibrated_available=False)


def replay(package):
    data = package["predictions"]
    payload = (json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True)
               + "\n").encode()
    if hashlib.sha256(payload).hexdigest() != package["predictions_sha256"]:
        raise ValueError("ByteTrack prediction archive fingerprint mismatch")
    numbers = score_bytetrack(data, {"zones_sha256": data["zones_sha256"], "clips": package["truth"]})
    if any(package[k] != v for k, v in numbers.items()):
        raise ValueError("Whole-clip ByteTrack baseline differs from signed event records")
    print("clip      method    pred_in/out  true_in/out  bias_in/out  L1")
    for name, clip in numbers["clips"].items():
        for tag, row in {**clip["seeds"], "E3": clip["e3"]}.items():
            print(name, tag, row["totals"], clip["gt"], row["bias"], f"{row['l1']:.2f}")
    print("All four clip/direction cells, denominator", numbers["pooled"]["denominator"])
    for tag, row in {**numbers["pooled"]["seeds"], "E3": numbers["pooled"]["e3"]}.items():
        print(tag, f"{row['l1']:.4f} errors; {row['error_pct']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    pred = subs.add_parser("predict", help="export six native signed event tables; no labels")
    pred.add_argument("output", type=Path)
    scorer = subs.add_parser("score", help="apply frozen §109 labels to whole-clip counts")
    scorer.add_argument("predictions", type=Path)
    scorer.add_argument("labels", type=Path)
    scorer.add_argument("output", type=Path)
    scorer.add_argument("--mask-sha", required=True)
    reader = subs.add_parser("replay", help="recompute both clips from the public comparator evidence")
    reader.add_argument("package", type=Path)
    args = parser.parse_args()
    if args.command == "predict":
        report = predict()
    elif args.command == "score":
        answers = json.loads(args.labels.read_bytes())
        if answers["study"] != 109 or answers["mask_sha256"] != args.mask_sha:
            raise ValueError("Labels do not belong to the frozen whiteboard study")
        inputs = json.loads(args.predictions.read_bytes())
        report = dict(study=109, prereg=answers["prereg"], mask_sha256=answers["mask_sha256"],
                      predictions=inputs, predictions_sha256=sha(args.predictions), labels_sha256=sha(args.labels),
                      truth={name: {"gt": clip["gt"]} for name,clip in answers["clips"].items()},
                      **score_bytetrack(inputs, answers))
    else:
        replay(json.loads(args.package.read_bytes()))
        raise SystemExit(0)
    write_once(args.output, report)
    print(args.output, sha(args.output))
