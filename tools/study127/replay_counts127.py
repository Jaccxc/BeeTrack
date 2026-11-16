"""Recompute compact per-event category, direction, and event-graph counts."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def derive_categories(cell: dict, event: dict) -> tuple[str, str]:
    ids = [x["gt_id"] for x in event["owners"]]
    endpoint_forced = [
        owner["gt_id"] is not None
        and owner["qualifying_gt_candidates"] == 1
        and owner["qualifying_prediction_candidates_for_assigned_gt"] == 1
        for owner in event["owners"]
    ]
    both_forced = all(endpoint_forced)
    if both_forced != event["both_forced"]:
        raise ValueError("both_forced does not match endpoint candidate counts")
    if any(x is None for x in ids):
        assigned = "01_no_owner"
    elif event["gt_center_sides"] != event["prediction_sides"]:
        assigned = "03_wrong_side_geometry"
    elif ids[0] != ids[1]:
        assigned = "04_different_bees"
    else:
        net = event["same_segment_net"]
        if net is None:
            assigned = ("05b_same_bee_net0_outside_C" if cell["arm"] == "C"
                        and event["owner_in_C"] == [False, False]
                        else "05_same_bee_genuine_transient_net0")
        elif net == event["direction"]:
            assigned = "06_same_bee_official_direction"
        else:
            assigned = "07_same_bee_opposite_net"
    category = "02_ambiguous_owner" if all(x is not None for x in ids) and not both_forced else assigned
    return category, assigned


def maximum_matching(events: list[dict]) -> int:
    owner = {}

    def visit(pred_index: int, seen: set[int]) -> bool:
        for gt_id in sorted(set(events[pred_index]["legal_gt_events"])):
            if gt_id in seen:
                continue
            seen.add(gt_id)
            if gt_id not in owner or visit(owner[gt_id], seen):
                owner[gt_id] = pred_index
                return True
        return False

    for i in range(len(events)):
        visit(i, set())
    return len(owner)


def count_cell(cell: dict) -> dict:
    events = cell["events"]
    for event in events:
        category, assigned = derive_categories(cell, event)
        if category != event["category"] or assigned != event["assigned_category"]:
            raise ValueError(f"stored endpoint fields do not reproduce attribution for {event['pred_id']}")
    cats = dict(sorted(Counter(e["category"] for e in events).items()))
    statuses = dict(sorted(Counter(e["graph_status"] for e in events).items()))
    buckets = dict(sorted(Counter(e["graph_bucket"] for e in events).items()))
    directions = {}
    for sign in (1, -1):
        group = [e for e in events if e["direction"] == sign]
        directions[str(sign)] = {"predicted_events": len(group),
                                 "no_legal_edge": sum(not e["legal_gt_events"] for e in group),
                                 "categories": dict(sorted(Counter(e["category"] for e in group).items()))}
    if len(events) != cell["predicted_events"] or cats != cell["categories"]:
        raise ValueError("event rows do not match declared event denominator/categories")
    if statuses != cell["graph_statuses"]:
        raise ValueError("event rows do not match graph status totals")
    if maximum_matching(events) != cell["official_graph_tp"]:
        raise ValueError("event adjacency does not reproduce official maximum matching TP")
    if sum(e["official_selected"] for e in events) != cell["official_graph_tp"]:
        raise ValueError("selected-edge flags do not reproduce official TP")
    for i, e in enumerate(events):
        no_edge = not e["legal_gt_events"]
        if no_edge != (e["graph_status"] == "no_legal_edge"):
            raise ValueError("no-edge status and adjacency disagree")
        if no_edge:
            expected_status = "no_legal_edge"
        elif maximum_matching(events[:i] + events[i + 1:]) < cell["official_graph_tp"]:
            expected_status = "always_matched_in_maximum"
        else:
            expected_status = "matching_ambiguous"
        if e["graph_status"] != expected_status:
            raise ValueError("stored graph status does not reproduce from adjacency")
        if e["graph_bucket"] == "eligible_unselected" and (no_edge or e["official_selected"]):
            raise ValueError("eligible unselected bucket is inconsistent")
        if e["graph_bucket"] == "eligible_selected" and (no_edge or not e["official_selected"]):
            raise ValueError("eligible selected bucket is inconsistent")
    if directions != cell["direction_counts"]:
        raise ValueError("per-direction counts do not reproduce")
    return {"arm": cell["arm"], "sequence": cell["sequence"], "seed": cell["seed"],
            "threshold": cell["threshold"], "denominator": cell["predicted_events"],
            "gt_events": cell["gt_events"], "official_tp": cell["official_graph_tp"],
            "categories": cats, "graph_statuses": statuses, "graph_buckets": buckets,
            "direction_counts": directions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    args = p.parse_args()
    with gzip.open(args.bundle, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "study127-event-attribution-v1":
        raise ValueError("unsupported artifact format")
    summary = [count_cell(cell) for cell in data["cells"]]
    print(json.dumps({"format": data["format"], "replayed_cells": len(summary), "summary": summary},
                     ensure_ascii=False, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
