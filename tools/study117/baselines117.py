"""Frozen ByteTrack, fractional calibration and selected historical TOPICTrack on §117 tails.

Run from the BeeTrack root. The track stage does not open GT. The score stage
requires hashes of both completed track inventories. Replay needs no raw data.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


CASES = [(f, s) for f in (33, 35) for s in range(3)]
ARMS = ("U", "A", "Z", "Q", "Zm", "D")
METHODS = ("ByteTrack", "calibrated ByteTrack", "TOPICTrack selected")
TAUS = ("0.5", "0.3")
UNIT = [[1, 1], [1, 1]]
CALIBRATION_SHA = "fb1f424fb83fe7fb61e8a719a1682b0de84efd9871f63453e2399b3f3e26457f"
HISTORY_SHA = "39b2a95c1c85acebf11194967a495e536ddfb893b96f28c473d35f17a2672ecd"


def sha(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return path


def history(repo, prior):
    source = json.loads(checked(prior, HISTORY_SHA).read_bytes())["cases"]
    calibration = repo / "docs" / "study103_results.json"
    cases = json.loads(checked(calibration, CALIBRATION_SHA).read_bytes())["cases"]
    ratios = {(r["hive"], r["seed"]): r["scale_ratio"] for r in cases if r["method"] == METHODS[1]}
    out = {}
    for fold, seed in CASES:
        name = f"{fold}_{seed}"
        case = source[name + "_tail"]
        clip = f"BEE24-{68-fold}-tail"
        threshold = 30 if (fold, seed) == (35, 0) else 20
        tracker = f"p99_f2_t{threshold}_h1_loho_no{fold}_s{seed}_post"
        mot = f"runs/trackers/val/{tracker}/data/{clip}.txt"
        if (case["fold"], case["seed"], case["clip"], case["nframes"]) != (fold, seed, clip, 1000):
            raise ValueError(f"Unexpected old source metadata for {name}")
        out[name] = dict(fold=fold, seed=seed, clip=clip, nframes=1000,
                         calibration=ratios[fold, seed], topic_tracker=tracker,
                         topic_mot=mot, topic_mot_sha256=sha(repo/mot),
                         old_detector=case["paths"]["det"], old_detector_sha256=case["hashes"]["det"],
                         gt=case["paths"]["gt"], gt_sha256=case["hashes"]["gt"],
                         zones=case["paths"]["zones"], zones_sha256=case["hashes"]["zones"])
    return dict(study=117, calibration_sha256=CALIBRATION_SHA, history_sha256=HISTORY_SHA, cases=out)


def stock(dets, nframes, dest):
    from beetrack.track import run_sequence, Tracker
    from beetrack.common import write_mot

    if dets.dtype != np.float64 or dets.ndim != 2 or dets.shape[1] != 6 or not np.isfinite(dets).all():
        raise ValueError("Expected the complete finite float64 N×6 cache from CPU FP32")
    f = dets[:, 0]
    if np.any(np.diff(f) < 0) or np.any(f != np.floor(f)) or np.any((f < 1) | (f > nframes)):
        raise ValueError("Unordered or out-of-range detector frames")
    if not np.array_equal(dets, dets.astype(np.float32).astype(np.float64)):
        raise ValueError("CPU detections were not losslessly widened from float32")
    tracker = Tracker("bytetrack")
    rows, _ = run_sequence(dets, nframes, tracker)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as temp:
        partial = Path(temp)/"stock.txt"
        write_mot(partial, rows)
        try:
            os.link(partial, dest)
        except FileExistsError:
            checked(dest, sha(partial))
    return vars(tracker.args)


def detector_difference(current, old):
    def rows(a):
        return Counter(map(tuple, np.asarray(a, dtype=np.float32)))

    return dict(new_dtype=str(current.dtype), historical_dtype=str(old.dtype), new_rows=len(current),
                historical_rows=len(old), equal_as_ordered_float32=np.array_equal(current.astype(np.float32), old.astype(np.float32)),
                common_exact_float32_rows_with_multiplicity=sum((rows(current) & rows(old)).values()))


def track(repo, root, history_sha, prereg, detector_sha):
    from tools.study111.summary111 import read, write

    source = read(checked(root/"baselines"/"history.json", history_sha))
    detector = json.loads(checked(root/"detector_done.json", detector_sha).read_bytes())
    expected = {f"detections/{f}_{s}/{arm}.npy" for f, s in CASES for arm in ARMS}
    actual = {k for k in detector["files"] if k.startswith("detections/") and k.endswith(".npy")}
    if detector["study"] != 117 or detector["prereg"] != prereg or expected != actual:
        raise ValueError("Incomplete or mismatched 36-cache detector freeze")
    index = dict(study=117, prereg=prereg, history_index_sha256=history_sha,
                 detector_sha256=detector_sha, cases={})
    for name, info in source["cases"].items():
        key = f"detections/{name}/U.npy"
        dets = np.load(checked(root/key, detector["files"][key]), allow_pickle=False)
        dest = root/"baselines"/"mots"/(name+".txt")
        preset = stock(dets, info["nframes"], dest)
        old = np.load(checked(repo/info["old_detector"], info["old_detector_sha256"]), allow_pickle=False)
        index["cases"][name] = dict(**info, detector=key, detector_sha256=detector["files"][key],
                                    stock_mot=dest.relative_to(root).as_posix(), stock_mot_sha256=sha(dest),
                                    preset=preset, detector_comparison=detector_difference(dets, old))
        print(name, sha(dest), flush=True)
    write(root/"baselines"/"index.json", index)


def evidence(gt, prediction, polygon):
    from tools.study104.event_overlap import archive_arm, score_arm

    record = {"edges": {}}
    for tau in TAUS:
        result = archive_arm(score_arm(gt, prediction, polygon, float(tau)))
        for key in ("gt", "predictions"):
            if record.setdefault(key, result[key]) != result[key]:
                raise AssertionError("Event identity varies by IoU")
        record["edges"][tau] = result["edges"]
    return record


def metrics(graph, nframes, scales, fractional=False):
    from tools.study111.summary111 import count_scores, histogram, official

    p, g = graph["predictions"], graph["gt"]
    predictions, truth = [r[1] for r in p], [r[1] for r in g]
    counts = count_scores(predictions, truth, nframes, scales)
    pred_bins = histogram(predictions, nframes).reshape(30, -1, 2)
    truth_bins = histogram(truth, nframes).reshape(30, -1, 2)
    scaled_l1_numerators = []
    w = Fraction()
    for i, (num, den) in enumerate(scales):
        direction = np.abs(pred_bins[:, :, i]*num - truth_bins[:, :, i]*den).sum(axis=1)
        scaled_l1_numerators.append(direction.tolist())
        w += Fraction(int(direction.sum()), den)
    if 100*w/(30*len(g)) != Fraction(*counts["window_percent_fraction"]):
        raise AssertionError("Directional offset totals do not reconstruct W")
    scores = None if fractional else {tau: official(g, graph, tau) for tau in TAUS}
    return dict(**counts, scales=scales, event_scores=scores,
                per_offset_l1_numerators_in_out=list(map(list,zip(*scaled_l1_numerators))),
                l1_denominators_in_out=[den for _, den in scales], window_gt_denominator=30*len(g))


def report(records):
    out, means = {}, {}
    for name, case in records.items():
        bt, topic = case["bytetrack"], case["topic"]
        if bt["gt"] != topic["gt"]:
            raise ValueError("Same-clip ground truth mismatch")
        n = case["info"]["nframes"]
        out[name] = {METHODS[0]: metrics(bt, n, UNIT),
                     METHODS[1]: metrics(bt, n, case["info"]["calibration"], fractional=True),
                     METHODS[2]: metrics(topic, n, UNIT)}
    for fold in (33, 35):
        means[str(fold)] = {}
        for m in METHODS:
            r = [out[f"{fold}_{s}"][m] for s in range(3)]
            w = sum(Fraction(*a["window_percent_fraction"]) for a in r)/3
            f1 = None if m == METHODS[1] else {tau: sum(a["event_scores"][tau]["f1"] for a in r)/3 for tau in TAUS}
            means[str(fold)][m] = dict(window=float(w), window_percent_fraction=[w.numerator, w.denominator], f1=f1)
    return dict(cases=out, fold_means=means)


def score(repo, root, prereg, primary_sha, baseline_sha):
    from beetrack.evaluate import load_tracks
    from tools.study111.summary111 import write

    idx = json.loads(checked(root/"baselines"/"index.json", baseline_sha).read_bytes())
    primary = json.loads(checked(root/"tracking"/"index.json", primary_sha).read_bytes())
    expected = {(f,s,a) for f,s in CASES for a in ARMS}
    groups = {(r["fold"],r["seed"],r["arm"]) for r in primary["cases"]}
    if (primary["stage"] != "predictions_only" or len(primary["cases"]) != 36 or groups != expected or
            set(idx["cases"]) != {f"{f}_{s}" for f,s in CASES} or
            primary["prereg"] != prereg or idx["prereg"] != prereg or
            primary["detector_index_sha256"] != idx["detector_sha256"] or
            primary["study"] != 117 or idx["study"] != 117):
        raise ValueError("Incomplete or mismatched registered prediction freeze")
    records = {}
    for name, info in idx["cases"].items():
        bt = load_tracks(checked(root/info["stock_mot"], info["stock_mot_sha256"]))
        topic = load_tracks(checked(repo/info["topic_mot"], info["topic_mot_sha256"]))
        gt = load_tracks(checked(repo/info["gt"], info["gt_sha256"]))
        zones = json.loads(checked(repo/info["zones"], info["zones_sha256"]).read_bytes())
        poly = zones[info["clip"]]
        records[name] = dict(info=info, bytetrack=evidence(gt,bt,poly), topic=evidence(gt,topic,poly))
        print(f"Scored complete MOTs {name}",flush=True)
    result = report(records)
    provenance = dict(study=117, prereg=prereg, baseline_index_sha256=baseline_sha, primary_index_sha256=primary_sha)
    write(root/"baselines"/"replay.json.gz", dict(**provenance, records=records, results=result))
    write(root/"baselines"/"results.json", dict(**provenance, **result))


def self_check():
    from beetrack.evaluate import load_tracks

    frames = range(1,24)
    detections = np.array([[f,f,0,f+10,10,.875] for f in frames], dtype=np.float64)
    polygon = [[17,-5],[100,-5],[100,20],[17,20]]
    with TemporaryDirectory() as tmp:
        p = Path(tmp)/"mot.txt"
        stock(detections, 23, p)
        mot = load_tracks(p)
        gt = np.array([[f,101,f,0,10,10] for f in frames],dtype=float)
        gt[:,2] = mot[:,2]
        graph = evidence(gt, mot, polygon)
        a = metrics(graph, 23, UNIT)
        assert a["window"] == 0 and a["gt_totals"] == [1,0] and a["event_scores"]["0.5"]["tp"] == 1
        b = metrics(graph, 23, [[3,2],[1,1]], fractional=True)
        assert b["window"] == 50 and b["pred_totals"] == [1.5,0] and b["event_scores"] is None
        empty = Path(tmp)/"empty.txt"
        stock(np.empty((0,6)), 23, empty)
        assert empty.read_bytes() == b""
        c = metrics(evidence(gt, np.empty((0,6)), polygon),23,UNIT)
        assert c["window"] == 100 and c["event_scores"]["0.3"]["fn"] == 1
        assert detector_difference(detections,detections)["common_exact_float32_rows_with_multiplicity"]==len(frames)
    print("Synthetic default tracking, IoU, windows, fractional calibration and empty detections passed.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--root", type=Path, default=Path("runs/study117"))
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("self-check")
    h = sub.add_parser("history")
    h.add_argument("--prior",type=Path,required=True)
    t = sub.add_parser("track")
    for flag in ("--history-sha","--detector-sha","--prereg"):
        t.add_argument(flag,required=True)
    s = sub.add_parser("score")
    for flag in ("--baseline-sha","--primary-sha","--prereg"):
        s.add_argument(flag,required=True)
    r = sub.add_parser("replay")
    r.add_argument("archive",type=Path)
    a = p.parse_args()
    repo, root = a.repo.resolve(), a.root.resolve()
    sys.path.insert(0,str(repo))
    from tools.study111.summary111 import read,write

    if a.mode == "self-check":
        self_check()
    elif a.mode == "history":
        write(root/"baselines"/"history.json",history(repo,a.prior))
    elif a.mode == "track":
        track(repo,root,a.history_sha,a.prereg,a.detector_sha)
    elif a.mode == "score":
        score(repo,root,a.prereg,a.primary_sha,a.baseline_sha)
    else:
        archive = read(a.archive)
        if report(archive["records"]) != archive["results"]:
            raise ValueError("Baseline replay disagrees with stored results")
        print("All six matched-seed baseline cells replayed.")


if __name__ == "__main__":
    main()
