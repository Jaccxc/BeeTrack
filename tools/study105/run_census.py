"""§105 CPU census from fixed P0 outputs, raw cached detections and the exact P0 first pass."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from beetrack.common import RUNS
from tools.study103.swap_score import _reference_rows, load_case
from tools.study104.track_reverse import first_pass
from shortside_census import census


def summarize(record):
    events = {int(k): v for k,v in record["events"].items()}
    directions = {int(k): v for k,v in record["gt_directions"].items()}
    excludes = {int(k): v for k,v in record["excluded_events"].items()}
    matches = set(record["official_matches"].values())
    assert (events.keys() | excludes.keys()) == directions.keys()
    assert not (events.keys() & excludes.keys()) and matches <= directions.keys()
    assert len(matches) == len(record["official_matches"])
    coverage = {}
    observed, missing = [], []
    reasons = Counter()
    for tid, item in events.items():
        frames = item["short_frames"]
        audit = item["short_audit"]
        assert len(frames) in (1,2) and len(frames) == len(audit)
        assert item["direction"] == directions[tid] and item["official_hit"] == (tid in matches)
        assert item["eligible"] == (item["anchor_frame"] is not None and all(
            not x["any_raw"] and not x["first_hit"] for x in audit))
        for x in audit:
            reasons[x["reason"]] += 1
        if item["eligible"]:
            observed.append(tid)
            if tid not in matches:
                missing.append(tid)
    assert set(observed) == set(record["eligible_ids"]) and set(missing) == set(record["eligible_missed_ids"])
    for name, ids in (("all",directions), ("in",[g for g,d in directions.items() if d == "in"]),
                      ("out",[g for g,d in directions.items() if d == "out"])):
        eligible = sorted(set(ids) & set(observed))
        missed = sorted(set(ids) & set(missing))
        coverage[name] = dict(gt=len(ids), short=len(set(ids)&events.keys()), eligible=len(eligible),
                              eligible_hit=len(eligible)-len(missed), eligible_missed=len(missed),
                              eligible_hit_fraction=(len(eligible)-len(missed))/len(eligible) if eligible else None,
                              eligible_missed_fraction=len(missed)/len(eligible) if eligible else None,
                              missed_per_gt=len(missed)/len(ids) if ids else None)
    tp, ng, npred = len(matches),len(directions),record["pred_event_count"]
    official = dict(tp=tp, fp=npred-tp, fn=ng-tp, f1=2*tp/(ng+npred))
    expected = _reference_rows(record["hive"])[record["seed"]]
    if (tp, npred-tp, ng-tp) != expected[2:] or ng != {33:184,35:129}[record["hive"]]:
        raise ValueError("Official baseline differs from §95")
    return dict(coverage=coverage, short_frame_causes=dict(sorted(reasons.items())), official=official)


def run(hive, seed, out):
    gt,p0,poly,nframes = load_case(hive, seed)
    tag,seq = f"loho_no{hive}_s{seed}",f"BEE24-{hive}-head"
    det_path = RUNS / "dets" / tag / "dev" / f"{seq}.npy"
    deps = [det_path, RUNS / "gt"/"dev"/seq/"gt"/"gt.txt",
            RUNS/"trackers"/"dev"/f"q_lk_{tag}"/"data"/f"{seq}.txt"]
    source_hash = {}
    for p in deps:
        with p.open("rb") as stream:
            source_hash[p.relative_to(RUNS).as_posix()] = hashlib.file_digest(stream,"sha256").hexdigest()
    det = np.load(det_path,allow_pickle=False)
    first = first_pass(det,nframes)
    r = census(gt,p0,det,poly,nframes,first)
    for key in ("eligible_ids","eligible_missed_ids"):
        r[key] = sorted(r[key])
    r.update(study=105, prereg="244b9ca5f1d82e3f72aa142863c295b8eafef8ce", hive=hive, seed=seed,
             sequence=seq, nframes=nframes, sources=source_hash, raw_rows=len(det), first_rows=len(first),
             stage1="live §104 first_pass", gt_id_type="official repaired dev counting identifier")
    r["summary"] = summarize(r)
    out.parent.mkdir(exist_ok=True, parents=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(r,separators=(",",":"),allow_nan=False),encoding="utf-8")
    reread = json.loads(tmp.read_text(encoding="utf-8"))
    assert summarize(reread) == reread["summary"]
    tmp.replace(out)
    print(f"{hive}/{seed}: {r['summary']['coverage']['all']}", flush=True)


def summarize_hive(hive, folder):
    """Verify all nine records and count which repaired event IDs recur across seeds."""
    counts, indices, misses, causes = [], Counter(), Counter(), Counter()
    missed_geometry = []
    seen_geometry = None
    by_direction = {}
    for seed in range(9):
        r = json.loads((folder / f"{hive}_{seed}.json").read_text(encoding="utf-8"))
        assert (r["hive"], r["seed"]) == (hive, seed)
        assert summarize(r) == r["summary"]
        geometry = (r["geometry_counts"], r["gt_directions"], r["excluded_events"], sorted(r["events"]))
        if seen_geometry is None:
            seen_geometry = geometry
        else:
            assert seen_geometry == geometry
        counts.append(len(r["eligible_missed_ids"]))
        for identifier in r["eligible_ids"]:
            indices[identifier] += 1
        for identifier in r["eligible_missed_ids"]:
            misses[identifier] += 1
        causes.update(r["summary"]["short_frame_causes"])
        short_misses = [item for item in r["events"].values() if not item["official_hit"]]
        missing_origin = [item for item in short_misses if item["short_role"] == "origin"]
        both_seen = sum(item["anchor_frame"] is not None and
                        any(frame["first_hit"] for frame in item["short_audit"]) for item in short_misses)
        missed_geometry.append(dict(short_missed=len(short_misses), short_origin=len(missing_origin),
                                    short_destination=len(short_misses)-len(missing_origin),
                                    stage1_both_sides_seen=both_seen))
        for key in ("in","out"):
            stats = r["summary"]["coverage"][key]
            by_direction.setdefault(key, []).append([stats["short"],stats["eligible"],stats["eligible_missed"]])
    median = float(np.median(counts))
    threshold = {33:6, 35:4}[hive]
    return dict(hive=hive, missed_by_seed=counts, median=median, threshold=threshold,
                advance=median >= threshold, geometry_counts=seen_geometry[0],
                eligible_unique=len(indices), missed_unique=len(misses),
                eligible_frequency=dict(sorted(indices.items())), missed_frequency=dict(sorted(misses.items())),
                short_eligible_missed_by_direction=by_direction, pooled_short_frame_causes=dict(causes),
                missed_geometry_by_seed=missed_geometry)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hive", type=int, choices=[33,35])
    p.add_argument("outdir", type=Path)
    p.add_argument("--seeds", nargs="+", type=int, choices=range(9), default=list(range(9)))
    p.add_argument("--summary", action="store_true", help="verify the nine archives; do not read tracking or detections")
    a = p.parse_args()
    if a.summary:
        print(json.dumps(summarize_hive(a.hive, a.outdir),sort_keys=True))
    else:
        for s in a.seeds:
            run(a.hive,s,a.outdir/f"{a.hive}_{s}.json")
