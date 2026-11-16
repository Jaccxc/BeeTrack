"""Reconstruct native tracker references for the 18 detector/clip cells of study 111.

Run from the BeeTrack repository root. Building the compact extra event graphs
uses the archived dev MOTs. Replaying references only needs public JSON files,
NumPy and SciPy. No study 111 eligibility or treatment data enters either step.
"""
import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


HIVES = (33, 35)
METHODS = ("P0", "ByteTrack", "calibrated ByteTrack", "TOPICTrack selected")
THRESHOLDS = ("0.5", "0.3")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def write(path, record):
    data = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        partial = Path(tmp) / "complete"
        partial.write_bytes(data)
        try:
            os.link(partial, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("Different evidence already exists at " + str(path))
    print(f"{path.name} {sha(path)}", file=sys.stderr)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build(root):
    sys.path.insert(0, str(root.resolve()))
    from beetrack.evaluate import ZONES, load_tracks
    from tools.study104.event_overlap import score_arm

    docs = root / "docs"
    archive = {"source_sha256": {f"study{k}": sha(docs / f"study{k}_results.json")
                                 for k in (103, 104)}, "clips": {}, "cases": []}
    archive["source_sha256"]["zones"] = sha(ZONES)
    zones = read(ZONES)
    for h in HIVES:
        seq = f"BEE24-{h}-head"
        gt_path = root / "runs" / "gt" / "dev" / seq / "gt" / "gt.txt"
        gt = load_tracks(gt_path)
        archive["clips"][str(h)] = {"gt_mot_sha256": sha(gt_path)}
        for seed in range(9):
            tag = f"loho_no{h}_s{seed}"
            trackers = {"ByteTrack": f"q_bt_{tag}"}
            if seed < 3:
                threshold = 30 if (h, seed) == (35, 0) else 20
                trackers["TOPICTrack selected"] = f"p99_f2_t{threshold}_h1_{tag}_post"
            for method, tracker in trackers.items():
                path = root / "runs" / "trackers" / "dev" / tracker / "data" / f"{seq}.txt"
                pred = load_tracks(path)
                case = dict(hive=h, seed=seed, method=method, tracker=tracker,
                            mot_sha256=sha(path), edges={})
                for threshold in THRESHOLDS:
                    arm = score_arm(gt, pred, zones[seq], float(threshold))
                    events = sorted([int(p), int(f) if d == "in" else -int(f)]
                                    for p, (d, f, _) in arm.events.items())
                    truth = sorted([int(g), int(f) if arm.gt[g] == "in" else -int(f)]
                                   for g, f in arm.gt_frames.items())
                    clip = archive["clips"][str(h)]
                    require(clip.setdefault("gt", truth) == truth, "GT event identity drift")
                    require(case.setdefault("predictions", events) == events, "prediction identity drift")
                    case["edges"][threshold] = [[int(p), int(g)] for p, g in arm.edges]
                archive["cases"].append(case)
            print(f"Historical event graphs {h} s{seed}", file=sys.stderr, flush=True)
    return archive


def histogram(stream, frames):
    events = np.asarray(stream, dtype=np.int64)
    index = np.abs(events) - 1
    require(not np.any((index < 0) | (index >= frames)), "Event outside clip")
    # Include partial first and last windows, even at nonzero offsets.
    nbin = (frames + 725 + 749) // 750
    offsets = np.arange(30)[:, None]
    bins = offsets * nbin + (index + 25 * offsets) // 750
    cells = 2 * bins + (events < 0)
    return np.bincount(cells.ravel(), minlength=60 * nbin).reshape(-1, 2)


def official(truth, record, threshold):
    gt, pred = dict(truth), {p[0]: p[1] for p in record["predictions"]}
    require(len(gt) == len(truth) and len(pred) == len(record["predictions"]), "Duplicate event ID")
    pcol, gcol = {p: i for i, p in enumerate(pred)}, {g: i for i, g in enumerate(gt)}
    pi, gi = [], []
    for p, g in record["edges"][threshold]:
        require((pred[p] > 0) == (gt[g] > 0), "Official edge crosses direction")
        pi.append(pcol[p])
        gi.append(gcol[g])
    edges = csr_matrix((np.ones(len(pi), dtype=bool), (pi, gi)), shape=(len(pred), len(gt)))
    tp = int(np.count_nonzero(maximum_bipartite_matching(edges, perm_type="column") >= 0))
    return dict(tp=tp, fp=len(pred)-tp, fn=len(gt)-tp, f1=2*tp/(len(pred)+len(gt)))


def count_scores(stream, truth, frames, ratios):
    p, g = histogram(stream, frames), histogram(truth, frames)
    gaps, totals = [], []
    for i, (num, den) in enumerate(ratios):
        total = Fraction(int(p[:, i].sum())*num, 30*den)
        gap = Fraction(int(np.abs(p[:, i]*num - g[:, i]*den).sum()), den)
        totals.append(total)
        gaps.append(gap)
    w = 100*sum(gaps)/int(g.sum())
    gt_counts = [sum(f > 0 for f in truth), sum(f < 0 for f in truth)]
    return dict(window=float(w), window_percent_fraction=[w.numerator, w.denominator],
                pred_totals=[float(t) for t in totals],
                bias=[float(t-c) for t, c in zip(totals, gt_counts)], gt_totals=gt_counts)


def published(log, hive):
    section = log.split("\n## 95. ", 1)[1].split("\n## 96. ", 1)[0]
    table = section.split(f"留 {hive}(評 {hive}-head;", 1)[1].split("| seed |", 1)[1].split("| 平均(全距)", 1)[0]
    window, p0_counts = {}, {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0].isdigit():
            s = int(cells[0])
            for j, m in zip((1, 4, 5), METHODS[:3]):
                window[s, m] = float(cells[j].split("(")[0])
            p0_counts[s] = [int(v) for v in cells[3].split("(")[1].rstrip(")").split("/")]
    section = log.split("\n## 99. ", 1)[1].split("\n## 100. ", 1)[0]
    table = section.split(f"| 留 {hive} | 挑到的設定 |", 1)[1]
    cells = next(line for line in table.splitlines() if line.startswith("| 公平的重選 |")).strip("|").split("|")
    for s, val in enumerate(cells[2].split("/")):
        window[s, METHODS[3]] = float(val)
    return window, p0_counts, [float(v) for v in cells[4].split("/")]


def references(root, graphs):
    docs = root / "docs"
    extra = read(graphs)
    for k in (103, 104):
        require(sha(docs / f"study{k}_results.json") == extra["source_sha256"][f"study{k}"],
                f"Study {k} archive fingerprint mismatch")
    a, b = (read(docs / f"study{k}_results.json") for k in (103, 104))
    log = (docs / "experiment_log.md").read_text(encoding="utf-8")
    lookup = {(c["hive"], c["seed"], c["method"]): c for c in a["cases"]}
    p0 = {(c["hive"], c["seed"]): c for c in b["cases"]}
    other = {(c["hive"], c["seed"], c["method"]): c for c in extra["cases"]}
    require(len(lookup) == 60 and len(p0) == 18 and len(other) == 24, "Incomplete native reference set")
    out = []
    for h in HIVES:
        clip = str(h)
        frames = a["clips"][clip]["frames"]
        truth = a["clips"][clip]["gt"]
        gt = b["clips"][clip]["gt"]
        require(frames == b["clips"][clip]["nframes"] and sorted(gt) == extra["clips"][clip]["gt"],
                "Native GT ID/frame disagreement")
        require(Counter(truth) == Counter(f for _, f in gt), "Native GT stream disagreement")
        window, p0_counts, topic_f1 = published(log, h)
        for seed in range(9):
            for method in METHODS if seed < 3 else METHODS[:3]:
                c = lookup[h, seed, method]
                source = "ByteTrack" if method == "calibrated ByteTrack" else method
                stream = lookup[h, seed, source]["events"]
                ratios = c.get("scale_ratio", [[1, 1], [1, 1]])
                scores = count_scores(stream, truth, frames, ratios)
                w = scores["window"]
                require(abs(c["phase"]["original"] - w) <= 0.0000051
                        and abs(window[seed, method] - w) <= 0.0050001, f"Archived W differs {h}/{seed}/{method}")
                scores["events"] = None
                if method != "calibrated ByteTrack":
                    record = p0[h, seed]["arms"]["F"] if method == "P0" else other[h, seed, method]
                    require(Counter(row[1] for row in record["predictions"]) == Counter(stream),
                            f"Signed event stream differs {h}/{seed}/{method}")
                    scores["events"] = {t: official(gt, record, t) for t in THRESHOLDS}
                    if method == "P0":
                        require(abs(w-p0[h, seed]["metrics"]["windows"]["F"]["window"]) <= 1e-9,
                                "Study 104 native W differs")
                        for t, match in scores["events"].items():
                            counts = [match[k] for k in ("tp", "fp", "fn")]
                            require(all(match[k] == c["arms"]["p0"]["scores"][t][k] for k in ("tp", "fp", "fn"))
                                    and (t != "0.5" or counts == p0_counts[seed]), "P0 official counts differ")
                    elif method == "TOPICTrack selected":
                        require(abs(scores["events"]["0.5"]["f1"] - topic_f1[seed]) <= .0005001,
                                "Study 99 native F1 differs")
                out.append(dict(hive=h, seed=seed, method=method, count_scale_ratio=ratios, **scores))
    return {"tracker_graphs_sha256": sha(graphs), "source_sha256": extra["source_sha256"], "cases": out}


def verify_native(reference, evidence, results):
    """Check a scored 111 case against the corresponding historical P0 row."""
    record = results["arms"]["native"]
    require(reference["method"] == "P0", "Expected the same hive and seed's native P0")
    for field in ("gt_totals", "pred_totals", "bias"):
        require(reference[field] == record[field], f"Native {field} differs")
    require(Fraction(*reference["window_percent_fraction"]) ==
            Fraction(100*record["window_numerator"], record["window_denominator"]), "Native W differs")
    for k, t in zip(("05", "03"), THRESHOLDS):
        require(all(record[k][f] == reference["events"][t][f] for f in ("tp", "fp", "fn", "f1")),
                "Native official matching differs at " + t)
    require(len(evidence["gt"]) == sum(reference["gt_totals"]), "Native GT total differs")


def matched_means(rows, h):
    averages = []
    for size in (9, 3):
        for m in METHODS if size == 3 else METHODS[:3]:
            selected = [r for r in rows if r["hive"] == h and r["seed"] < size and r["method"] == m]
            require({r["seed"] for r in selected} == set(range(size)), "Incomplete matched seed set")
            mean_w = sum(r["window"] for r in selected) / size
            f1 = None if m == "calibrated ByteTrack" else sum(r["events"]["0.5"]["f1"] for r in selected)/size
            averages.append(dict(hive=h, seeds=f"0-{size-1}", method=m, window=mean_w, f1_05=f1))
    return averages


def print_rows(rows, hive, seed):
    print("hive seed method | pred in/out | bias in/out | W (%) | TP/FP/FN (.5) | F1 .5/.3")
    for r in rows:
        if hive is not None and r["hive"] != hive or seed is not None and r["seed"] != seed:
            continue
        match = r["events"]
        event = "count calibration only"
        if match:
            m = match["0.5"]
            event = f"{m['tp']}/{m['fp']}/{m['fn']} | {m['f1']:.4f}/{match['0.3']['f1']:.4f}"
        counts = "/".join(f"{x:.3f}".rstrip("0").rstrip(".") for x in r["pred_totals"])
        bias = "/".join(f"{x:+.3f}".rstrip("0").rstrip(".") for x in r["bias"])
        print(f"{r['hive']} {r['seed']} {r['method']} | {counts} | {bias} | {r['window']:.4f} | {event}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="BeeTrack repository root")
    sub = ap.add_subparsers(dest="cmd", required=True)
    build_cli = sub.add_parser("build", help="Extract historical ByteTrack and TOPICTrack official graphs")
    build_cli.add_argument("graphs", type=Path)
    refs = sub.add_parser("refs", help="Replay all native references and show matched rows")
    refs.add_argument("graphs", type=Path)
    refs.add_argument("--hive", type=int, choices=HIVES)
    refs.add_argument("--seed", type=int, choices=range(9))
    refs.add_argument("--output", type=Path, help="Save all 60 rows and both matched seed means")
    args = ap.parse_args()
    if args.cmd == "build":
        write(args.graphs, build(args.root))
    else:
        table = references(args.root, args.graphs)
        table["means"] = [mean for h in HIVES for mean in matched_means(table["cases"], h)]
        print_rows(table["cases"], args.hive, args.seed)
        if args.output:
            write(args.output, table)


if __name__ == "__main__":
    main()
