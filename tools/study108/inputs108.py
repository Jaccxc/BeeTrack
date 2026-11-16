"""Extract prediction-only event locations or label previously frozen locations, CPU only."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from beetrack.common import ROOT, RUNS, write_mot
from beetrack.evaluate import ZONES, by_track, load_tracks, zone_tracks
from beetrack.link import model
from tools.study103.comparison_inputs import _reference_errors
from tools.study103.replay import bins, error
from tools.study103.swap_score import _reference_rows
from tools.study104.event_overlap import archive_arm, score_arm
from tools.study104.track_reverse import STRICT_SHA256, track_arms


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        part = Path(tmp) / path.name
        part.write_text(json.dumps(record, separators=(",", ":"), ensure_ascii=True, allow_nan=False),
                        encoding="utf-8", newline="\n")
        part.replace(path)


def describe(raw, poly):
    """One last same-direction crossing per event; only native prediction boxes are consulted."""
    rows = by_track(raw)
    out = []
    for tid, (direction, _, _, changes) in zone_tracks(rows, poly).items():
        if direction:
            index = next(i for i, d in reversed(changes) if d == direction)
            f, _, x, y, w, h = rows[index]
            signed = int(f) if direction == "in" else -int(f)
            out.append([int(tid), signed, round(float(x), 2), round(float(y), 2),
                        round(float(x + w), 2), round(float(y + h), 2)])
    return sorted(out)


@lru_cache(maxsize=1)
def zones():
    return json.loads(ZONES.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def strict_probability(weights):
    if weights is None or digest(weights) != STRICT_SHA256:
        raise ValueError("Source tail predictions need the archived strict model")
    return model(json.loads(weights.read_bytes()))


def case_paths(output, role, fold, seed, part):
    if fold not in (33, 35) or seed not in range(9) or role not in ("source", "target"):
        raise ValueError("Invalid study case")
    if part not in (("head", "tail") if role == "source" else ("head",)):
        raise ValueError("Invalid registered partition")
    hive = 68 - fold if role == "source" else fold
    split = "dev" if part == "head" else "val"
    clip = f"BEE24-{hive}-{part}"
    tag = f"loho_no{fold}_s{seed}"
    name = f"{fold}_{seed}_{part}"
    root = output / role
    track = (RUNS / "trackers" / split / f"q_lk_{tag}" / "data" / f"{clip}.txt"
             if part == "head" else root / "tail_mot" / f"{name}.txt")
    return (dict(fold=fold, seed=seed, clip=clip, split=split, tag=tag, nframes=3950 if part == "head" else 1000),
            track, root / "predictions" / f"{name}.json", root / "labels" / f"{name}.json")


def predictions(case, path, dest, weights):
    sources = {"beetrack/zones.json": digest(ZONES)}
    if case["split"] == "val":
        det = RUNS / "dets" / case["tag"] / "val" / (case["clip"] + ".npy")
        sources[det.relative_to(RUNS).as_posix()] = digest(det)
        sources["m_dev_strict.json"] = STRICT_SHA256
        prob = strict_probability(weights)
        # Reusing an earlier tail is safe only if the entire input manifest and MOT still match.
        if path.is_file() and dest.is_file():
            old = json.loads(dest.read_bytes())
            if old.get("sources") == sources and old.get("mot_sha256") == digest(path):
                return
        arm, rows = next(track_arms(np.load(det, allow_pickle=False), case["nframes"], prob))
        assert arm == "F"
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as tmp:
            part = Path(tmp) / path.name
            write_mot(part, rows)
            part.replace(path)
    else:
        sources[path.relative_to(RUNS).as_posix()] = digest(path)
    data = describe(load_tracks(path), zones()[case["clip"]])
    if any(not 0 < abs(e[1]) <= case["nframes"] or e[4] <= e[2] or e[5] <= e[3] for e in data):
        raise ValueError("Invalid canonical event in " + case["clip"])
    save(dest, dict(case, predictions=data, sources=sources, mot_sha256=digest(path)))


@lru_cache(maxsize=1)
def public_target_graphs():
    # Only used after both the source decision and the target prediction freeze.
    data = json.loads((ROOT / "docs" / "study104_results.json").read_bytes())
    assert data["study"] == 104
    return data["clips"], {(r["hive"], r["seed"]):r["arms"]["F"] for r in data["cases"]}


@lru_cache(maxsize=1)
def target_group_refs():
    """Reconstruct §95 three-seed window means from the §103 signed event series."""
    stream = json.loads((ROOT / "docs" / "study103_results.json").read_bytes())
    cases = {(r["hive"], r["seed"], r["method"]): r for r in stream["cases"]}
    section = (ROOT / "docs" / "experiment_log.md").read_text(encoding="utf-8").split("\n## 95. ", 1)[1]
    section = section.split("| 折 | 組 | P0(單一) | E3 |", 1)[1].split("**判準**", 1)[0]
    refs = {}
    for fold in (33, 35):
        clip = stream["clips"][str(fold)]
        nframes = clip["frames"]
        truth = bins(clip["gt"], nframes)
        for start in (0, 3, 6):
            seeds = range(start, start + 3)
            triple = "、".join(map(str, seeds))
            row = next(line for line in section.splitlines() if line.startswith(f"| 留 {fold} | {triple} |"))
            cells = [x.strip() for x in row.strip("|").split("|")]
            expected = dict(zip(("P0", "ByteTrack", "calibrated ByteTrack"),
                                [float(cells[3]), float(cells[6].split("/")[-1]), float(cells[7].split("/")[-1])]))
            reconstructed = {}
            for name, published in expected.items():
                member = []
                for seed in seeds:
                    case = cases[fold, seed, name]
                    if name == "calibrated ByteTrack":
                        counts = bins(cases[fold, seed, "ByteTrack"]["events"], nframes)
                        counts = counts * [p / q for p, q in case["scale_ratio"]]
                    else:
                        counts = bins(case["events"], nframes)
                    member.append(counts)
                score = error(np.mean(member, axis=0), truth)
                if round(score, 2) != published:
                    raise ValueError(f"{name} three-seed W disagrees with §95 for {fold}/{start}")
                reconstructed[name] = score
            refs[fold, start] = reconstructed
    return refs


def canonical_archive(preds):
    return [[pid, frame, *(round(float(v), 2) for v in box)] for pid, frame, *box in preds]


def labels(case, path, pred_path, dest, role):
    frozen = json.loads(pred_path.read_bytes())
    if any(frozen[k] != v for k, v in case.items()) or digest(path) != frozen["mot_sha256"]:
        raise ValueError("MOT or case changed after the prediction freeze")
    clip = case["clip"]
    poly = zones()[clip]
    if digest(ZONES) != frozen["sources"]["beetrack/zones.json"]:
        raise ValueError("Polygon set changed after the prediction freeze")
    gt_path = RUNS / "gt" / case["split"] / clip / "gt" / "gt.txt"
    gt, pred = load_tracks(gt_path), load_tracks(path)
    answers = {}
    for tau in (0.5, 0.3):
        arm = archive_arm(score_arm(gt, pred, poly, tau))
        if canonical_archive(arm["predictions"]) != frozen["predictions"]:
            raise ValueError("Label extraction disagrees with the prediction-only event descriptors")
        if "gt" in answers and answers["gt"] != arm["gt"]:
            raise ValueError("GT event set changed across IoU thresholds")
        answers["gt"] = arm["gt"]
        answers["edges05" if tau == 0.5 else "edges03"] = arm["edges"]
    refs = {}
    if role == "target":
        fold, seed = case["fold"], case["seed"]
        clips, arms = public_target_graphs()
        original = arms[fold, seed]
        if (clips[str(fold)]["gt"] != answers["gt"]
                or canonical_archive(original["predictions"]) != frozen["predictions"]
                or original["edges"]["0.5"] != answers["edges05"]
                or original["edges"]["0.3"] != answers["edges03"]):
            raise ValueError(f"New held-out native graph differs from archived §104 F for {fold}/{seed}")
        refs["P0"] = _reference_rows(fold)[seed][0]
        refs.update({m:v for (m,s),v in _reference_errors(fold).items() if s == seed})
    group_refs = target_group_refs()[fold, 3 * (seed // 3)] if role == "target" else {}
    save(dest, dict(case, **answers, predictions_sha256=digest(pred_path), gt_mot_sha256=digest(gt_path),
                    refs=refs, group_refs=group_refs))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("predictions", "labels"))
    p.add_argument("role", choices=("source", "target"))
    p.add_argument("output", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--fold", type=int, choices=(33, 35))
    p.add_argument("--seeds", type=int, choices=range(9), nargs="+", default=list(range(9)))
    args = p.parse_args()
    for fold in ((args.fold,) if args.fold else (33,35)):
        for seed in args.seeds:
            for part in (("head","tail") if args.role == "source" else ("head",)):
                case, path, pr, lab = case_paths(args.output, args.role, fold, seed, part)
                if args.stage == "predictions":
                    predictions(case, path, pr, args.weights)
                else:
                    labels(case, path, pr, lab, args.role)
                print(args.role, args.stage, fold, seed, part, flush=True)
