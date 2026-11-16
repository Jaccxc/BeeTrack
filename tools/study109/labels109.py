"""Collect legal prediction-to-GT event edges for the frozen white-board masks."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from beetrack.common import RUNS
from beetrack.evaluate import ZONES, load_tracks
from study108.inputs108 import canonical_archive
from tools.study104.event_overlap import archive_arm, score_arm
from freeze109 import CLIPS, SEEDS, write_once


def read_mot(path, nframes, expected=None):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Prediction MOT differs from freeze: {path}")
    rows = load_tracks(BytesIO(data)) if data.strip() else np.empty((0, 6))
    if (rows.shape[1] != 6 or not np.isfinite(rows).all() or np.any(rows[:, 4:6] <= 0)
            or np.any(rows[:, :2] != np.rint(rows[:, :2]))
            or np.any((rows[:, 0] < 1) | (rows[:, 0] > nframes))):
        raise ValueError(f"Invalid clip-local MOT data: {path}")
    return rows, digest


def legal_tables(gt, pred, polygon, frozen):
    result, gt_events = {}, None
    for key, threshold in (("edges05", .5), ("edges03", .3)):
        # Full MOTs, including non-events and vetoed tracks, enter the frame matcher.
        graph = archive_arm(score_arm(gt, pred, polygon, threshold))
        if canonical_archive(graph["predictions"]) != frozen:
            raise ValueError("Full-MOT event anchors differ from prediction freeze")
        if gt_events is not None and gt_events != graph["gt"]:
            raise ValueError("GT event identities or canonical dates differ across thresholds")
        gt_events = graph["gt"]
        result[key] = graph["edges"]
    return gt_events, result


def collect(mask_path, expected_sha, gt_root, mot_root, zones_path):
    data = mask_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("Refusing to read labels: mask checksum mismatch")
    mask = json.loads(data)
    expected_clips = {f"BEE24-{key}" for key in CLIPS}
    if mask["study"] != 109 or set(mask["clips"]) != expected_clips:
        raise ValueError("Expected the two registered §109 clips")
    zones_data = zones_path.read_bytes()
    if hashlib.sha256(zones_data).hexdigest() != mask["zones_sha256"]:
        raise ValueError("Zone data differ from the prediction freeze")
    zones = json.loads(zones_data)
    clips, gt_sha = {}, {}
    for number, nframes in CLIPS.items():
        clip = f"BEE24-{number}"
        frozen = mask["clips"][clip]
        if (frozen["nframes"] != nframes or frozen["polygon"] != zones[clip]
                or set(frozen["tracks"]) != set(map(str, SEEDS))):
            raise ValueError(f"Invalid frame bounds, polygon or seed list for {clip}")
        # Read only the registered val clips, independent of any paths in the manifest.
        gt, gt_sha[clip] = read_mot(gt_root / clip / "gt" / "gt.txt", nframes)
        anchors, gt_events = {}, None
        for seed in SEEDS:
            track = frozen["tracks"][str(seed)]
            tag = f"loho_noW_s{seed}"
            relative = Path(f"q_lk_{tag}") / "data" / f"{clip}.txt"
            if track["tag"] != tag or track["seed"] != seed or track["mot_path"] != f"runs/trackers/val/{relative.as_posix()}":
                raise ValueError(f"Frozen prediction path or seed mismatch for {clip}/{seed}")
            pred, _ = read_mot(mot_root / relative, nframes, track["mot_sha256"])
            gt_arm, tables = legal_tables(gt, pred, zones[clip], track["predictions"])
            if gt_events is not None and gt_events != gt_arm:
                raise ValueError(f"GT event identity differs across anchor seeds for {clip}")
            gt_events, anchors[str(seed)] = gt_arm, tables
        clips[clip] = dict(gt=gt_events, anchors=anchors)
    return dict(study=109, prereg=mask["prereg"], mask_sha256=expected_sha,
                gt_sha256=gt_sha, zones_sha256=mask["zones_sha256"], clips=clips)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mask", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mask-sha", required=True)
    ap.add_argument("--gt-root", type=Path, default=RUNS/"gt"/"val", help="val GT parent containing BEE24-10 and BEE24-15")
    ap.add_argument("--mot-root", type=Path, default=RUNS/"trackers"/"val", help="val MOT parent containing q_lk_loho_noW_s*")
    ap.add_argument("--zones", type=Path, default=ZONES)
    args = ap.parse_args()
    labels = collect(args.mask, args.mask_sha, args.gt_root, args.mot_root, args.zones)
    write_once(args.output, labels)
    print("Label evidence SHA256", hashlib.sha256(args.output.read_bytes()).hexdigest(), flush=True)
