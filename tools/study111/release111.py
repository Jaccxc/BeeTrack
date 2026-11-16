"""Score the sealed treatment MOTs; pack and independently replay study 111."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from summary111 import HIVES, read, references, require, sha, verify_native


CELLS = tuple(f"{h}_{s}" for h in HIVES for s in range(9))
GROUPS = {"all": "rows_all", "paired": "rows", "weak": "natural_weak_rows"}
PHASES = {"B0": (1,2,3), "B1": (4,5,6), "S0": (1,3,5), "S1": (2,4,6)}


def encode(record):
    # Source manifests were sorted at publication. Keep their original key order,
    # including numeric GT identifiers that became strings when read by JSON.
    return (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pinned(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, f"Evidence fingerprint mismatch at {path.name}")
    return data


def save(path, record):
    data = gzip.compress(encode(record), mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        p = Path(tmp) / "archive"
        p.write_bytes(data)
        try:
            os.link(p, path)
        except FileExistsError:
            require(path.read_bytes() == data, "Different content exists at " + str(path))
    return digest(data)


def mask_checks(frozen, paired_run):
    arms = {}
    sites = [("nearall", "near", "rows_all")]
    if paired_run:
        sites += [("near", "near", "rows"), ("far", "far", "rows")]
    for family, site, field in sites:
        rows = frozen[field]
        for phase, idx in PHASES.items():
            name = family + "_" + phase
            ids = sorted(r[site+"_indices"][i] for r in rows for i in idx)
            require(ids == frozen["masks"][name] and len(set(ids)) == 3*len(rows),
                    "Deletion budget or indices differ for " + name)
            arms[name] = ids
    return arms


def check_seals(global_freeze, completed, key, case):
    h, s = map(int, key.split("_"))
    f, d = case["selection"], case["tracking"]
    gate = global_freeze["folds"][str(h)]
    fsha, dsha = (digest(encode(x)) for x in (f, d))
    require(fsha == global_freeze["cases"][key] == d["case_sha256"] and
            dsha == completed["cases"][key], "Case fingerprint mismatch " + key)
    require(d["freeze_sha256"] == completed["freeze_sha256"] and
            f["prereg"] == global_freeze["prereg"] == completed["prereg"] == d["prereg"] and
            f["fold"] == d["fold"] == h and f["seed"] == d["seed"] == s, "Case identity mismatch")
    require(gate["nearall_counts"][s] == len(f["rows_all"]) and gate["paired_counts"][s] == len(f["rows"])
            and gate["paired_run"] == d["paired_run"] == (sum(i >= 10 for i in gate["paired_counts"]) >= 7),
            "Coverage branch differs from freeze")
    require(d["arms"]["native"] == f["hashes"]["p0"] and d["det_sha256"] == f["hashes"]["det"]
            and d["strict_sha256"] == f["hashes"]["strict"], "Native replay/input mismatch")
    masks = mask_checks(f, gate["paired_run"])
    require(set(d["arms"]) == ({"native"} | masks.keys()) and set(case["arm_records"]) == masks.keys(),
            "Registered treatment set is incomplete")
    for arm, rec in case["arm_records"].items():
        ids = masks[arm]
        mask_sha = digest(b"".join(i.to_bytes(8, "little", signed=True) for i in ids))
        require(digest(encode(rec)) == d["arm_records_sha256"][arm] and
                rec["arm"] == arm and rec["case_sha256"] == fsha and rec["freeze_sha256"] == d["freeze_sha256"]
                and rec["deleted_count"] == len(ids) and rec["deleted_sha256"] == mask_sha
                and rec["mot_sha256"] == d["arms"][arm], "Treatment receipt differs " + arm)
        if rec["reused"] is not None:
            same = rec["reused"]
            require(d["arms"][same] == rec["mot_sha256"] and masks.get(same, []) == ids,
                    "Masks differed between shared MOTs")


def load_seals(run, freeze_sha, tracks_sha, verify_mots):
    freeze = json.loads(pinned(run / "freeze111.json", freeze_sha))
    done = json.loads(pinned(run / "tracks_done.json", tracks_sha))
    require(freeze["study"] == done["study"] == 111 and done["freeze_sha256"] == freeze_sha
            and set(freeze["cases"]) == set(done["cases"]) == set(CELLS), "Incomplete global seal")
    cells = {}
    for key in CELLS:
        directory = run / "tracks" / key
        f = json.loads(pinned(run / "target" / (key+".json"), freeze["cases"][key]))
        d = json.loads(pinned(directory / "done.json", done["cases"][key]))
        recs = {k:json.loads(pinned(directory / (k+".json"), v)) for k,v in d["arm_records_sha256"].items()}
        row = dict(selection=f, tracking=d, arm_records=recs)
        check_seals(freeze, done, key, row)
        if verify_mots:
            for arm, expected in d["arms"].items():
                require(sha(directory / (arm+".txt")) == expected, "Output MOT changed " + key + "/" + arm)
        cells[key] = row
    return dict(study=111, freeze_sha256=freeze_sha, tracks_sha256=tracks_sha,
                freeze=freeze, tracks=done, cases=cells)


def score_cell(job):
    root, run, key, row = job
    sys.path.insert(0, str(root.resolve()))
    import cv2
    import torch
    from beetrack.evaluate import ZONES, load_tracks
    from score111 import score_case, _write

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    frozen, done = row["selection"], row["tracking"]
    gt = root / "runs" / "gt" / "dev" / frozen["clip"] / "gt" / "gt.txt"
    truth = load_tracks(BytesIO(pinned(gt, frozen["hashes"]["gt"])))
    poly = json.loads(pinned(ZONES, frozen["hashes"]["zones"]))[frozen["clip"]]
    mots = {}
    for name, fingerprint in done["arms"].items():
        data = pinned(run / "tracks" / key / (name + ".txt"), fingerprint)
        mots[name] = load_tracks(BytesIO(data))
    ids = {k:[r["gt_id"] for r in frozen[field]] for k,field in GROUPS.items()}
    pkg = score_case(truth, mots, poly, frozen["nframes"], ids["all"], ids["paired"], ids["weak"])
    pkg.update(freeze_sha256=done["case_sha256"], tracks_case_sha256=digest(encode(done)))
    path = run / "scored" / (key + ".json.gz")
    return key, _write(path, pkg)


def score(root, run, freeze_sha, tracks_sha, workers):
    for setting in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[setting] = "1"
    # No official treatment graph is opened until every registered MOT verifies.
    state = load_seals(run, freeze_sha, tracks_sha, verify_mots=True)
    jobs = [(root, run, key, row) for key,row in state["cases"].items()]
    with ProcessPoolExecutor(workers) as pool:
        for key, fingerprint in pool.map(score_cell, jobs):
            print(key, fingerprint, flush=True)


def unselected_changes(results):
    """Canonical and worst/best change in TP among all untreated GT events."""
    change = {}
    for arm, both in results["arms"].items():
        if arm == "native":
            continue
        group = "all" if arm.startswith("nearall_") else "paired"
        change[arm] = {}
        for key in ("05", "03"):
            ranges = []
            for row in (results["arms"]["native"][key], both[key]):
                b = row["target_bounds"][group]
                ranges.append([row["tp"]-b[k] for k in ("high", "low", "canonical")])
            before, after = ranges
            change[arm][key] = dict(lower=after[0]-before[1], upper=after[1]-before[0],
                                    canonical=after[2]-before[2])
    return change


def cohort_diagnostics(cases):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    per_seed, repeated = {}, {}
    for seed, case in cases.items():
        frozen = case["selection"]
        p = case["score"]
        ev = p["evidence"]
        arm = ev["arms"]["native"]
        # Reproduce the sorted canonical tie convention as a descriptive count.
        truth, pred = sorted(dict(ev["gt"])), sorted(dict(arm["predictions"]))
        ix, jx = {g:i for i,g in enumerate(truth)}, {p:j for j,p in enumerate(pred)}
        graph = np.zeros((len(pred), len(truth)), dtype=int)
        for pid, gid in arm["edges05"]:
            graph[jx[pid], ix[gid]] = 1
        rr, cc = linear_sum_assignment(graph, maximize=True)
        matched = {truth[c] for r,c in zip(rr,cc) if graph[r,c]}
        reached = {g for _,g in arm["edges05"]}
        data = {}
        for label, field in (("nearall","rows_all"), ("paired","rows"),
                             ("natural","natural_rows"), ("weak","natural_weak_rows")):
            rows = frozen[field]
            ids = [r["gt_id"] for r in rows]
            require(len(ids) == len(set(ids)) and set(ids) <= set(truth), "Invalid frozen cohort")
            repeated.setdefault(label, Counter()).update(ids)
            directions = {r["gt_id"]:r["dir"] for r in rows}
            bins = {}
            for subset, selected in (("all",ids), ("canonical_miss",set(ids)-matched),
                                     ("no_native_edge",set(ids)-reached)):
                counts = Counter(directions[g] for g in selected)
                bins[subset] = [counts["in"], counts["out"]]
            if label in ("natural","weak"):
                bins["gap_frame_reasons"] = dict(sorted(Counter(
                    d["reason"] for r in rows for d in r["gap_audit"]).items()))
            data[label] = bins
        per_seed[str(seed)] = data
    frequencies = {label:{str(n):sum(k == n for k in cnt.values()) for n in range(1,10)}
                   for label,cnt in repeated.items()}
    return {"seeds":per_seed, "number_of_unique_gt_ids_by_number_of_seeds":frequencies}


def verify_archive(root, package, graphs):
    from score111 import replay
    from verify111 import aggregate

    require(__debug__, "Run the independent verifier with Python assertions enabled")
    require(package["study"] == 111 and set(package["cases"]) == set(CELLS), "Incomplete release")
    global_freeze, done = package["freeze"], package["tracks"]
    require(digest(encode(global_freeze)) == package["freeze_sha256"] == done["freeze_sha256"]
            and digest(encode(done)) == package["tracks_sha256"], "Global record fingerprint mismatch")
    native_refs = references(root, graphs)
    require(native_refs["tracker_graphs_sha256"] == package["tracker_graphs_sha256"],
            "Native comparator archive fingerprint mismatch")
    table = {(r["hive"], r["seed"]):r for r in native_refs["cases"] if r["method"] == "P0"}
    historical = read(root / "docs" / "study104_results.json")
    forward = {(r["hive"],r["seed"]):r["arms"]["F"] for r in historical["cases"]}
    fold_summary = {}
    for h in HIVES:
        packages, cases = {}, {}
        for s in range(9):
            key = f"{h}_{s}"
            case = package["cases"][key]
            check_seals(global_freeze, done, key, case)
            p = case["score"]
            require(digest(encode(p)) == case["score_json_sha256"] and
                    p["freeze_sha256"] == global_freeze["cases"][key] and
                    p["tracks_case_sha256"] == done["cases"][key], "Score provenance mismatch")
            ev, result, f = p["evidence"], p["results"], case["selection"]
            for label, field in GROUPS.items():
                require(sorted(r["gt_id"] for r in f[field]) == ev["target_ids"][label],
                        "Score cohort differs from freeze: " + label)
            require(set(ev["arms"]) == set(case["tracking"]["arms"]), "Score/treatment arm mismatch")
            require(result == replay(ev), "Score recalculation differs " + key)
            verify_native(table[h,s], ev, result)
            old, new = forward[h,s], ev["arms"]["native"]
            require(ev["gt"] == sorted(historical["clips"][str(h)]["gt"]) and
                    new["predictions"] == sorted([r[:2] for r in old["predictions"]]) and
                    all(sorted(new["edges"+k]) == sorted(old["edges"][t])
                        for k,t in (("05","0.5"),("03","0.3"))), "Native §104 identity/edge mismatch")
            cases[s], packages[s] = case, p
        # Independent sparse-rank and direct-window checks include every treated arm.
        fold_summary[str(h)] = {"gates":aggregate(packages), "cohorts":cohort_diagnostics(cases),
                                "unselected_tp_change":{str(s):unselected_changes(p["results"])
                                                        for s,p in packages.items()}}
    return fold_summary


def pack(root, run, output, graphs, freeze_sha, tracks_sha):
    state = load_seals(run, freeze_sha, tracks_sha, verify_mots=False)
    state["tracker_graphs_sha256"] = sha(graphs)
    for key, case in state["cases"].items():
        raw = gzip.decompress((run / "scored" / (key+".json.gz")).read_bytes())
        case["score_json_sha256"] = digest(raw)
        case["score"] = json.loads(raw)
    state["fold_summary"] = verify_archive(root, state, graphs)
    checksum = save(output, state)
    print(output.name, checksum)
    print(json.dumps({k:r["gates"] for k,r in state["fold_summary"].items()}, indent=2))


def print_tables(pkg):
    print("hive seed | near N | paired N | weak N | weak no-edge N | D near all (.5) | I near-far (.5)")
    for key,case in pkg["cases"].items():
        p = case["score"]
        result = p["results"]
        counts = p["evidence"]["target_ids"]
        weak = result["arms"]["native"]["05"]["target_bounds"]["weak"]
        values = []
        for label in ("nearall", "near_minus_far"):
            contrast = result["contrasts"].get(label)
            if contrast and contrast["05"]["n"]:
                c = contrast["05"]
                values.append(f"[{c['lower']:+.3f},{c['upper']:+.3f}]")
            else:
                values.append("NA")
        print(f"{key} | {len(counts['all'])} | {len(counts['paired'])} | {len(counts['weak'])} | "
              f"{len(weak['always_unmatched_ids'])} | " + " | ".join(values))
    print("hive seed arm | pred in/out | bias in/out | W (%) | TP/FP/FN (.5) | TP/FP/FN (.3) | F1 .5/.3")
    for key,case in pkg["cases"].items():
        for name, r in case["score"]["results"]["arms"].items():
            counts = ["/".join(str(v) for v in r[k]) for k in ("pred_totals", "bias")]
            event = ["/".join(str(r[t][k]) for k in ("tp", "fp", "fn")) for t in ("05","03")]
            print(f"{key} {name} | {counts[0]} | {counts[1]} | {r['window']:.4f} | "
                  f"{event[0]} | {event[1]} | {r['05']['f1']:.4f}/{r['03']['f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="BeeTrack repository root")
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("score", "pack", "replay"):
        sub = modes.add_parser(mode)
        if mode != "replay":
            sub.add_argument("run", type=Path, help="Frozen native/treated run directory")
            sub.add_argument("--freeze-sha", required=True)
            sub.add_argument("--tracks-sha", required=True)
        if mode == "score":
            sub.add_argument("--workers", type=int, choices=(1,2,3), default=3)
        else:
            sub.add_argument("archive", type=Path)
            sub.add_argument("--graphs", type=Path, required=True, help="Historical comparator graph supplement")
            if mode == "replay":
                sub.add_argument("--table", action="store_true", help="Print per-seed contrasts and all arm scores")
    args = parser.parse_args()
    if args.mode == "score":
        score(args.root, args.run, args.freeze_sha, args.tracks_sha, args.workers)
    elif args.mode == "pack":
        pack(args.root, args.run, args.archive, args.graphs, args.freeze_sha, args.tracks_sha)
    else:
        pkg = read(args.archive)
        summary = verify_archive(args.root, pkg, args.graphs)
        require(summary == pkg["fold_summary"], "Stored fold decisions or diagnostics differ")
        print(json.dumps({k:r["gates"] for k,r in summary.items()}, indent=2))
        if args.table:
            print_tables(pkg)
        print("Verified all 18 score graphs, registered treatment branches, native references, and gates.")


if __name__ == "__main__":
    main()
