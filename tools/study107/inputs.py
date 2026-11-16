"""Build either the source-calibration events or the held-out development events for the flow study.

Run from the repository root with PYTHONPATH set to that root. Frame numbers in both
the prepared head and tail MOT files start at 1 for that clip.
"""

import argparse
import configparser
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from beetrack.common import RUNS, write_mot
from beetrack.evaluate import ZONES, load_tracks
from beetrack.link import model
from tools.study103.comparison_inputs import _reference_errors
from tools.study103.swap_score import _reference_rows, counts_from_events, event_times, window_error
from tools.study104.track_reverse import STRICT_SHA256, track_arms


@lru_cache(maxsize=64)
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=4)
def clip_truth(split, clip):
    folder = RUNS / "gt" / split / clip
    metadata = folder / "seqinfo.ini"
    config = configparser.ConfigParser()
    config.read(metadata, encoding="utf-8")
    if config.getint("Sequence", "frameRate") != 25:
        raise ValueError(f"Unexpected frame rate for {clip}")
    nframes = config.getint("Sequence", "seqLength")
    expected = 3950 if split == "dev" else 1000
    if nframes != expected:
        raise ValueError(f"Expected {expected} local frames for {clip}, found {nframes}")
    zone = json.loads(ZONES.read_text(encoding="utf-8"))[clip]
    gt_file = folder / "gt" / "gt.txt"
    truth = event_times(load_tracks(gt_file), zone)
    hashes = {p.relative_to(RUNS).as_posix(): digest(p) for p in (metadata, gt_file)}
    hashes["beetrack/zones.json"] = digest(ZONES)
    return nframes, zone, truth, hashes


def jobs(role):
    if role not in ("source", "target"):
        raise ValueError(role)
    for fold in (33, 35):
        hive = 68 - fold if role == "source" else fold
        for seed in range(3 if role == "source" else 9):
            yield fold, seed, "dev", f"BEE24-{hive}-head"
            if role == "source":
                yield fold, seed, "val", f"BEE24-{hive}-tail"


def build(role, output, weights=None):
    probability = None
    if role == "source":
        if weights is None or digest(weights) != STRICT_SHA256:
            raise ValueError("Supply the registered m_dev_strict.json for source-tail reconstruction")
        probability = model(json.loads(weights.read_bytes()))
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with TemporaryDirectory(prefix="flow-events-", dir=output.parent) as temp:
        for fold, seed, split, clip in jobs(role):
            nframes, zone, truth, hashes = clip_truth(split, clip)
            inputs = dict(hashes)
            tag = f"loho_no{fold}_s{seed}"
            if split == "val":
                dets = RUNS / "dets" / tag / split / f"{clip}.npy"
                inputs[dets.relative_to(RUNS).as_posix()] = digest(dets)
                inputs["m_dev_strict.json"] = STRICT_SHA256
                # Use the same rounded MOT serialization as the original P0 tracker.
                name, rows = next(track_arms(np.load(dets, allow_pickle=False), nframes, probability))
                assert name == "F"
                path = Path(temp) / f"{tag}-{clip}.txt"
                write_mot(path, rows)
                p0_origin = "forward replay from cached fixed-640 detections"
            else:
                path = RUNS / "trackers" / split / f"q_lk_{tag}" / "data" / f"{clip}.txt"
                inputs[path.relative_to(RUNS).as_posix()] = digest(path)
                p0_origin = "archived P0 MOT"
            predicted = event_times(load_tracks(path), zone)
            gt_counts = counts_from_events(truth, nframes)
            pred_counts = counts_from_events(predicted, nframes)
            row = dict(fold=fold, seed=seed, clip=clip, nframes=nframes, tag=tag, split=split,
                       gt=truth.tolist(), p0=predicted.tolist(), p0_origin=p0_origin, inputs=inputs)
            if role == "target":
                expected = _reference_rows(fold)[seed][0]
                measured = window_error(gt_counts, pred_counts)
                if round(measured, 2) != expected:
                    raise ValueError(f"{clip} seed {seed} differs from the published §95 P0")
                comparisons = _reference_errors(fold)
                row["references"] = {"P0": expected, **{method: value for (method, s), value
                                                        in comparisons.items() if s == seed}}
            records.append(row)
            print(f"Prepared {role} fold {fold}, seed {seed}, {clip} ({p0_origin})", flush=True)
        package = dict(role=role, cases=records, frames="1-based, clip-local; direction +1 in / -1 out")
        tmp = Path(temp) / "events.json"
        tmp.write_text(json.dumps(package, ensure_ascii=True, separators=(",", ":"), allow_nan=False),
                       encoding="utf-8", newline="\n")
        tmp.replace(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", choices=("source", "target"))
    p.add_argument("output", type=Path)
    p.add_argument("--weights", type=Path)
    args = p.parse_args()
    build(args.role, args.output, args.weights)
