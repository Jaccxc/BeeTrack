"""Replay Study 126 window counts and official event F1 from archived graphs."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def window_counts(events: list[list[int]], nframes: int) -> list[list[list[int]]]:
    """Thirty offset grids, each containing [in, out] event counts per 750 frames."""
    result = []
    for shift in range(0, 750, 25):
        nbins = (nframes + shift + 749) // 750
        bins = [[0, 0] for _ in range(nbins)]
        for _, signed_frame in events:
            if not 1 <= abs(signed_frame) <= nframes:
                raise ValueError("event outside clip")
            idx = (abs(signed_frame) - 1 + shift) // 750
            bins[idx][int(signed_frame < 0)] += 1
        result.append(bins)
    return result


def maximum_matching(gt: list[list[int]], predictions: list[list[int]], edges: list[list[int]]) -> dict[int, int]:
    """Canonical maximum-cardinality matching; edge labels are [prediction ID, GT ID]."""
    gids = {row[0] for row in gt}
    pids = {row[0] for row in predictions}
    gsign = {row[0]: row[1] > 0 for row in gt}
    psign = {row[0]: row[1] > 0 for row in predictions}
    if len(gids) != len(gt) or len(pids) != len(predictions):
        raise ValueError("duplicate event ID")
    graph: dict[int, list[int]] = {p: [] for p in pids}
    seen = set()
    for p, g in edges:
        if p not in pids or g not in gids or psign[p] != gsign[g] or (p, g) in seen:
            raise ValueError("invalid, duplicate, or direction-mismatched graph edge")
        seen.add((p, g))
        graph[p].append(g)
    owner: dict[int, int] = {}

    def visit(p: int, seen: set[int]) -> bool:
        for g in sorted(graph[p]):
            if g in seen:
                continue
            seen.add(g)
            if g not in owner or visit(owner[g], seen):
                owner[g] = p
                return True
        return False

    for p in sorted(graph):
        visit(p, set())
    return {p: g for g, p in owner.items()}


def replay(e: dict) -> dict:
    if e.get("format") not in ("study126-official-graph-v1", "temporal117-official-graph-v1"):
        raise ValueError("unsupported evidence format")
    n = e["nframes"]
    gt, pred = e["gt"], e["predictions"]
    if type(n) is not int or n < 1:
        raise ValueError("invalid clip length")
    gt_counts = window_counts(gt, n)
    pred_counts = window_counts([[r[0], r[1]] for r in pred], n)
    diffs = []
    for a, b in zip(gt_counts, pred_counts, strict=True):
        if len(a) != len(b):
            raise ValueError("window grid mismatch")
        diffs.extend(abs(x - y) for ar, br in zip(a, b, strict=True) for x, y in zip(ar, br, strict=True))
    numerator = sum(diffs)
    denominator = 30 * len(gt)
    metrics = {}
    e05 = {tuple(x) for x in e.get("edges05", [])}
    e03 = {tuple(x) for x in e.get("edges03", [])}
    if not e05 and not e03 and "edges" in e:
        e05, e03 = {tuple(x) for x in e["edges"]["0.5"]}, {tuple(x) for x in e["edges"]["0.3"]}
    if not e05.issubset(e03):
        raise ValueError("IoU 0.5 graph must be a subset of the IoU 0.3 graph")
    for threshold, key in (("0.5", "edges05"), ("0.3", "edges03")):
        edges = e05 if threshold == "0.5" else e03
        matches = maximum_matching(gt, pred, edges)
        tp, fp, fn = len(matches), len(pred) - len(matches), len(gt) - len(matches)
        metrics[threshold] = {"tp": tp, "fp": fp, "fn": fn,
                              "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    return {"w30_numerator": numerator, "w30_denominator": denominator,
            "w30_percent": 100 * numerator / denominator if denominator else None,
            "f1": metrics}


def summarize(rows: list[dict]) -> dict:
    """Keep the registered comparison paired within each hive/head and seed."""
    cases = {}
    for case in sorted({r["case"] for r in rows}):
        gt = {r["arm"]: r for r in rows if r["case"] == case and r["family"] == "gt"}
        real = {arm: {r["seed"]: r for r in rows
                      if r["case"] == case and r["family"] == "real" and r["arm"] == arm}
                for arm in ("U", "M", "E", "T", "K")}
        real = {arm: values for arm, values in real.items() if values}
        def mean(values):
            return sum(values) / len(values) if values else None
        gt_summary = {arm: {"w30_percent": r["w30_percent"], "f1_05": r["f1"]["0.5"]["f1"],
                            "f1_03": r["f1"]["0.3"]["f1"]} for arm, r in gt.items()}
        real_summary = {}
        for arm, by_seed in real.items():
            ordered = [by_seed[s] for s in sorted(by_seed)]
            vals = [r["w30_percent"] for r in ordered]
            real_summary[arm] = {"seeds": sorted(by_seed), "w30_by_seed": vals,
                                 "mean_w30": mean(vals), "seed_range_w30": max(vals) - min(vals),
                                 "mean_f1_05": mean([r["f1"]["0.5"]["f1"] for r in ordered]),
                                 "mean_f1_03": mean([r["f1"]["0.3"]["f1"] for r in ordered])}
        gate = None
        if all(a in real for a in ("U", "T", "K")):
            shared = sorted(set(real["U"]) & set(real["T"]))
            if shared:
                u = [real["U"][s] for s in shared]
                t = [real["T"][s] for s in shared]
                delta = mean([a["w30_percent"] - b["w30_percent"] for a, b in zip(u, t)])
                max_range = max(real_summary["U"]["seed_range_w30"], real_summary["T"]["seed_range_w30"])
                mean_f1_delta = mean([b["f1"]["0.5"]["f1"] - a["f1"]["0.5"]["f1"]
                                      for a, b in zip(u, t)])
                non_worse = mean_f1_delta >= -1e-12
                gate = {"paired_seeds": shared, "mean_w30_improvement_points": delta,
                        "larger_seed_range_points": max_range,
                        "w_improvement_strictly_exceeds_range": delta > max_range,
                        "mean_f1_05_improvement": mean_f1_delta,
                        "mean_f1_05_non_worse": non_worse,
                        "k_worse_than_t": real_summary["K"]["mean_w30"] > real_summary["T"]["mean_w30"],
                        "head_pass": delta > max_range and non_worse and
                                     real_summary["K"]["mean_w30"] > real_summary["T"]["mean_w30"]}
        cases[case] = {"gt_frames": gt_summary, "real_detections": real_summary, "paired_gate": gate}
    heads = [x["paired_gate"] for x in cases.values()]
    overall = {"both_heads_pass": bool(heads) and all(x is not None and x["head_pass"] for x in heads),
               "heads": {case: data["paired_gate"] for case, data in cases.items()}}
    return {"cases": cases, "preregistered_gate": overall}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("evidence", type=Path, help="gzip JSON evidence bundle")
    args = ap.parse_args()
    with gzip.open(args.evidence, "rt", encoding="utf-8") as f:
        bundle = json.load(f)
    rows = []
    for r in bundle["records"]:
        actual = replay(r["graph_evidence"])
        expected_num = r["window"]["integer_absolute_error_total"]
        expected_den = r["window"]["gt_denominator"]
        if (actual["w30_numerator"] != expected_num or actual["w30_denominator"] != expected_den
                or any(any(actual["f1"][t][k] != r["official_iou"][t][k] for k in ("tp", "fp", "fn"))
                       for t in ("0.5", "0.3"))):
            raise ValueError(f"Replay mismatch for {r['case']} {r['family']}:{r['arm']} seed {r.get('seed')}")
        rows.append({"case": r["case"], "family": r["family"], "arm": r["arm"],
                     "seed": r.get("seed"), **actual})
    print(json.dumps({"records": rows, "summary": summarize(rows)}, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
