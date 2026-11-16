"""Pack or verify §104's event lists and two official matching graphs per tracking arm."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import mean, median

from score_case import evaluate


HIVES = {33: 184, 35: 129}
ARMS = {"F", "R", "RF"}
THRESHOLDS = {"0.5", "0.3"}
CASES = {(h, s) for h in HIVES for s in range(9)}
PREREGISTRATION = "c65d5e9396f11d61c525d56ee1726b9e5c59d21d"


def check_metrics(stored, actual, loc="metrics"):
    """Verify the entire tree, including direction splits, ambiguity and union counts."""
    if isinstance(actual, dict):
        if not isinstance(stored, dict) or stored.keys() != actual.keys():
            raise ValueError(f"{loc}: inconsistent keys")
        for k, value in actual.items():
            check_metrics(stored[k], value, f"{loc}/{k}")
    elif isinstance(actual, list):
        if not isinstance(stored, list) or len(stored) != len(actual):
            raise ValueError(f"{loc}: inconsistent list")
        for i, value in enumerate(actual):
            check_metrics(stored[i], value, f"{loc}/{i}")
    elif isinstance(actual, float):
        if not math.isclose(stored, actual, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{loc}: {stored} != {actual}")
    elif stored != actual:
        raise ValueError(f"{loc}: {stored} != {actual}")


def short_metrics(value):
    if isinstance(value, dict):
        return {k: short_metrics(v) for k, v in value.items()}
    if isinstance(value, list):
        return [short_metrics(v) for v in value]
    return round(value, 10) if isinstance(value, float) else value


def cross_arm_edges(forward, reverse):
    """All admissible prediction pairs, before any GT decision or assignment tie."""
    pairs = set()
    for a, f, ax, ay, ax2, ay2 in forward:
        for b, g, bx, by, bx2, by2 in reverse:
            if (f > 0) != (g > 0) or abs(abs(f)-abs(g)) > 5:
                continue
            overlap = max(0, min(ax2, bx2)-max(ax, bx)) * max(0, min(ay2, by2)-max(ay, by))
            union = (ax2-ax)*(ay2-ay) + (bx2-bx)*(by2-by) - overlap
            if union > 0 and 2 * overlap >= union:
                pairs.add((a, b))
    return pairs


def compact_arms(source):
    rounded = deepcopy(source)
    for arm in rounded.values():
        for row in arm["predictions"]:
            row[2:] = [round(v, 2) for v in row[2:]]
    for name in ("R", "RF"):
        original = cross_arm_edges(source["F"]["predictions"], source[name]["predictions"])
        compact = cross_arm_edges(rounded["F"]["predictions"], rounded[name]["predictions"])
        if original != compact:
            # Image boxes in MOT use two decimals. A float sum x+w near the
            # exact IoU boundary can still differ from its rounded endpoint.
            return source, "original"
    return rounded, "two_decimals"


def validate(record):
    h, s = record["hive"], record["seed"]
    if (h, s) not in CASES or record["nframes"] != 3950 or set(record["arms"]) != ARMS:
        raise ValueError(f"Unexpected §104 case {h}/{s}")
    gt = record["gt"]
    tids = {t for t, _ in gt}
    if len(tids) != len(gt) or len(tids) != HIVES[h] or any(
        not isinstance(tid, int) or not isinstance(f, int) or not 0 < abs(f) <= 3950 for tid, f in gt
    ):
        raise ValueError(f"Inconsistent GT event list for hive {h}")
    for name, arm in record["arms"].items():
        preds = arm["predictions"]
        pids = {p[0] for p in preds}
        if len(preds) != len(pids) or set(arm["edges"]) != THRESHOLDS:
            raise ValueError(f"{h}/{s}/{name}: inconsistent events or graphs")
        for pid, f, x, y, x2, y2 in preds:
            if (not isinstance(pid, int) or not isinstance(f, int) or not 0 < abs(f) <= 3950
                    or not all(map(math.isfinite, (x, y, x2, y2))) or not (x < x2 and y < y2)):
                raise ValueError(f"{h}/{s}/{name}: invalid canonical event")
        for edges in arm["edges"].values():
            if len(edges) != len(set(map(tuple, edges))) or any(p not in pids or t not in tids for p, t in edges):
                raise ValueError(f"{h}/{s}/{name}: bad graph edge")
    actual = evaluate(record)
    check_metrics(record["metrics"], actual, f"{h}/{s}")
    return actual


def pack(directory, output):
    clips, cases = {}, []
    for h, s in sorted(CASES):
        raw = json.loads((directory / f"{h}_{s}.json").read_text(encoding="utf-8"))
        if (raw["hive"], raw["seed"]) != (h, s):
            raise ValueError("Case identity disagrees with filename")
        validate(raw)
        clip = dict(nframes=raw["nframes"], gt=raw["gt"])
        if clips.setdefault(str(h), clip) != clip:
            raise ValueError(f"Different GT across seeds for hive {h}")
        arms, precision = compact_arms(raw["arms"])
        case = dict(hive=h, seed=s, coordinates=precision, arms=arms, metrics=short_metrics(raw["metrics"]))
        validate(dict(case, **clip))
        cases.append(case)
    head = dict(study=104, preregistration=PREREGISTRATION, clips=clips,
                evidence="Signed original frame: +in, -out. Predicted events contain ID, signed frame, "
                         "canonical xyxy box. Sparse [prediction ID,GT ID] edges encode all valid official "
                         "event matches using any same-direction crossing. Original MOTs are required "
                         "to rebuild these graphs from observations.")
    # One line per case leaves searchable plain text and avoids one very long line.
    def encode(obj):
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    data = encode(head)[:-1] + ',\n"cases":[\n' + ",\n".join(map(encode, cases)) + "\n]}\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(data, encoding="utf-8", newline="\n")
    tmp.replace(output)


def replay(path, table=False):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["study"] != 104 or data["preregistration"] != PREREGISTRATION:
        raise ValueError("Unexpected study or preregistration")
    records = {(r["hive"], r["seed"]): r for r in data["cases"]}
    if len(data["cases"]) != len(CASES) or records.keys() != CASES or set(data["clips"]) != {"33", "35"}:
        raise ValueError("Expected both complete nine-seed folds")
    union_better = []
    if table:
        print("hive seed R-only-TP R-only-FP J F-window R-window RF-window union-window F-F1 R-F1 RF-F1")
    for h, n_gt in HIVES.items():
        fold = [validate(dict(records[h, s], **data["clips"][str(h)])) for s in range(9)]
        pairs = [m["pairwise"]["0.5"]["R"] for m in fold]
        nets = [p["total"]["net"] for p in pairs]
        true = [p["total"]["reverse_only_true"] for p in pairs]
        false = [p["total"]["reverse_only_false"] for p in pairs]
        windows = {arm: mean(m["windows"][arm]["window"] for m in fold) for arm in ("F", "R", "RF", "union")}
        if table:
            for s, m in enumerate(fold):
                pair = pairs[s]
                print(h, s, true[s], false[s], nets[s],
                      *(f"{m['windows'][a]['window']:.2f}" for a in windows),
                      *(f"{v:.3f}" for v in (pair["forward"]["f1"], pair["reverse"]["f1"],
                                             m["pairwise"]["0.5"]["RF"]["reverse"]["f1"])))
        positive = sum(j > 0 for j in nets)
        threshold = math.ceil(0.03*n_gt)
        recovery = math.ceil(0.05*n_gt)
        passed = positive >= 7 and median(nets) >= threshold
        print(f"Hive {h}: {positive}/9 positive J, median J {median(nets):g} (threshold {threshold}), "
              f"primary gate {'pass' if passed else 'fail'}. Median R-only TP {median(true):g} "
              f"(recovery threshold {recovery}); median R-only FP {median(false):g}.")
        print("Mean 30-s error " + ", ".join(f"{a} {v:.2f}%" for a, v in windows.items()))
        if not passed:
            print("Candidate selection follow-up:" if median(true) >= recovery else "Short-side detection follow-up:",
                  f"hive {h}")
        union_better.append(windows["union"] < windows["F"])
    print(f"Union improves mean window error in both folds: {all(union_better)}")
    print("Verified 18 cases, 54 tracking arms, both matching thresholds, and 18 event-list unions.")


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run without -O to verify all score invariants")
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    build = subs.add_parser("pack")
    build.add_argument("input_dir", type=Path)
    build.add_argument("output", type=Path)
    check = subs.add_parser("replay")
    check.add_argument("input", type=Path)
    check.add_argument("--table", action="store_true")
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.input_dir, args.output)
    else:
        replay(args.input, args.table)
