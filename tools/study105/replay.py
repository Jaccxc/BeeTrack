"""Pack and replay the registered §105 census from its archived per-event decisions."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median


PREREG = "244b9ca5f1d82e3f72aa142863c295b8eafef8ce"
HIVES = {33: 184, 35: 129}
REASONS = {"observed", "absent_from_cache", "wrong_side_only", "assignment_competition",
           "post_nms_high", "high_removed_nms", "low_only"}


def analyze(clip, case):
    gt = clip["gt_directions"]
    shape = clip["short_events"]
    audit = case["events"]
    counts = Counter(clip["excluded_events"].values())
    counts["eligible_geometry"] = len(shape)
    non_events = {"cross_and_return", "never_crossed"}
    geometry = {key: n for key, n in clip["geometry_counts"].items() if key not in non_events and n}
    assert geometry == dict(counts)
    assert set(shape).isdisjoint(clip["excluded_events"])
    assert shape.keys() | clip["excluded_events"].keys() == gt.keys() and audit.keys() == shape.keys()
    assert Counter(gt.values()).keys() <= {"in", "out"}
    matched = set(case["officially_matched_gt"])
    assert len(matched) == len(case["officially_matched_gt"]) and matched <= gt.keys()
    assert case["pred_event_count"] >= len(matched)
    eligible, missed, short_missed, both_seen = set(), set(), set(), set()
    causes = Counter()
    for tid, prop in shape.items():
        frames = prop["short_frames"]
        checks = audit[tid]["frames"]
        assert len(frames) == len(checks) and len(frames) in (1, 2)
        assert all(b == a + 1 for a, b in zip(frames, frames[1:])) and all(1 < f < 3950 for f in frames)
        assert prop["role"] in ("origin", "destination")
        assert prop["crossing_frame"] == (frames[-1] + 1 if prop["role"] == "origin" else frames[0])
        anchor = audit[tid]["anchor_frame"]
        assert anchor is None or (1 <= anchor <= 3950 and anchor not in frames)
        for reason, any_raw, first_hit in checks:
            assert reason in REASONS and isinstance(any_raw, bool) and isinstance(first_hit, bool)
            assert (reason == "observed") == first_hit
            assert reason != "absent_from_cache" or not any_raw
            causes[reason] += 1
        if tid not in matched:
            short_missed.add(tid)
            if anchor is not None and any(first_hit for _, _, first_hit in checks):
                both_seen.add(tid)
        if anchor is not None and all(not any_raw and not first_hit for _, any_raw, first_hit in checks):
            eligible.add(tid)
            if tid not in matched:
                missed.add(tid)
    assert sorted(missed) == sorted(case["registered_missed_gt"])
    assert sorted(eligible) == sorted(case["registered_eligible_gt"])
    direction_counts = {}
    for name in ("all", "in", "out"):
        ids = {tid for tid, d in gt.items() if name == "all" or d == name}
        direction_counts[name] = [len(ids), len(ids & shape.keys()), len(ids & eligible),
                                  len(ids & (eligible - missed)), len(ids & missed)]
    assert direction_counts == case["registered_coverage"] and dict(causes) == case["registered_frame_causes"]
    return dict(eligible=eligible, missed=missed, coverage=direction_counts, reasons=causes,
                short_missed=len(short_missed), missed_both_seen=len(both_seen))


def pack(folder, output):
    clips, cases = {}, []
    for h, total in HIVES.items():
        for s in range(9):
            r = json.loads((folder / f"{h}_{s}.json").read_text(encoding="utf-8"))
            assert (r["study"], r["prereg"], r["hive"], r["seed"], r["nframes"]) == (105, PREREG, h, s, 3950)
            assert len(r["gt_directions"]) == total and r["gt_event_count"] == total
            clip = {name: r[name] for name in ("geometry_counts", "gt_directions", "excluded_events")}
            clip["short_events"] = {tid: dict(short_frames=e["short_frames"], role=e["short_role"],
                                                crossing_frame=e["crossing_frame"]) for tid,e in r["events"].items()}
            assert clips.setdefault(str(h), clip) == clip
            cov = r["summary"]["coverage"]
            names = ("gt", "short", "eligible", "eligible_hit", "eligible_missed")
            official = {str(tid) for tid in r["official_matches"].values()}
            assert len(official) == len(r["official_matches"])
            case = dict(hive=h, seed=s, officially_matched_gt=sorted(official, key=int),
                        pred_event_count=r["pred_event_count"],
                        sources=r["sources"], registered_eligible_gt=list(map(str,r["eligible_ids"])),
                        registered_missed_gt=list(map(str,r["eligible_missed_ids"])),
                        registered_coverage={k:[cov[k][n] for n in names] for k in cov},
                        registered_frame_causes=r["summary"]["short_frame_causes"], events={})
            for tid, e in r["events"].items():
                assert e["official_hit"] == (tid in official)
                case["events"][tid] = dict(anchor_frame=e["anchor_frame"],
                    frames=[[f["reason"], f["any_raw"], f["first_hit"]] for f in e["short_audit"]])
            analyze(clip, case)
            cases.append(case)
    head = dict(study=105, prereg=PREREG, clips=clips,
                units="Per short-side observation: [registered reason,any cached IoU>=0.3 candidate,"
                      "correct-side stage-1 match]. GT IDs are resolved dev counting IDs, not physical tags.")
    def encode(obj):
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(encode(head)[:-1] + ',\n"cases":[\n' + ",\n".join(map(encode,cases)) + "\n]}\n",
                   encoding="utf-8", newline="\n")
    tmp.replace(output)


def replay(path, table):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["study"] == 105 and data["prereg"] == PREREG
    cases = {(r["hive"], r["seed"]): r for r in data["cases"]}
    assert len(data["cases"]) == 18 and cases.keys() == {(h,s) for h in HIVES for s in range(9)}
    flags = []
    if table:
        print("hive seed eligible hit miss miss_in miss_out short_missed both_sides_observed_among_missed")
    for h, n in HIVES.items():
        clip = data["clips"][str(h)]
        assert len(clip["gt_directions"]) == n
        eligible, missed, reasons = Counter(), Counter(), Counter()
        totals, already_hit = [], []
        for s in range(9):
            stats = analyze(clip, cases[h,s])
            eligible.update(stats["eligible"])
            missed.update(stats["missed"])
            reasons.update(stats["reasons"])
            coverage = stats["coverage"]
            totals.append(coverage["all"][-1])
            already_hit.append(coverage["all"][-2])
            if table:
                print(h,s,*coverage["all"][2:],coverage["in"][-1],coverage["out"][-1],
                      stats["short_missed"], stats["missed_both_seen"])
        limit = math.ceil(n*.03)
        advance = median(totals) >= limit
        flags.append(advance)
        print(f"Hive {h}: {len(clip['short_events'])}/{n} GT events with eligible geometry; "
              f"median actual miss {median(totals):g}/{limit} required. "
              f"{sum(totals)} eligible miss instances + {sum(already_hit)} already matched across seeds.")
        print("Unique eligible/missed GT IDs:",len(eligible),len(missed),
              "| missed-ID multiplicity (seeds:number of IDs):",dict(sorted(Counter(missed.values()).items())))
        print("Geometry and non-event track counts:",clip["geometry_counts"])
        print("Pooled short-observation reasons:",dict(sorted(reasons.items())))
    print("Intervention branch:", "run both folds" if any(flags) else "stop after census")


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Replay requires assertions enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    p = commands.add_parser("pack")
    p.add_argument("folder", type=Path)
    p.add_argument("output", type=Path)
    v = commands.add_parser("replay")
    v.add_argument("input", type=Path)
    v.add_argument("--table", action="store_true")
    args = parser.parse_args()
    if args.cmd == "pack":
        pack(args.folder, args.output)
    else:
        replay(args.input, args.table)
