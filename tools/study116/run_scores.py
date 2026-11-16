"""Score both NMS spaces only after verifying the complete jointly frozen prediction inventory."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from inspector import freeze_predictions, write_once
from regions import region_counts
from sampling import PREREG, prepare_sample, score_dose
from transitions import DOSES, summarize


SPACES = {"native": "predictions", "wrapped": "wrapped_predictions"}


def checked_json(path, want):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != want:
        raise ValueError(f"Frozen SHA-256 mismatch: {path}")
    return json.loads(raw)


def expected_cases():
    return [(f"BEE24-{68-fold}-tail", fold, seed) for fold in (33,35) for seed in range(3)]


def run(root, prediction_index_sha, roster_index_sha, geometry_sha):
    prediction_index = checked_json(root / "detector_done.json", prediction_index_sha)
    roster_index = checked_json(root / "roster" / "index.json", roster_index_sha)
    geometry = checked_json(root / "geometry.json", geometry_sha)
    if prediction_index["prereg"] != PREREG or roster_index["prereg"] != PREREG:
        raise ValueError("Frozen artifacts belong to another preregistration")
    schedule = expected_cases()
    roster = {}
    for case in roster_index["cases"]:
        clip, fold = case["clip"], case["fold"]
        if (clip,fold,0) not in schedule or clip in roster:
            raise ValueError("Unexpected source roster")
        label = checked_json(root / "roster" / case["labels_file"], case["labels_sha256"])
        req = checked_json(root / "roster" / case["requests_file"], case["requests_sha256"])
        if (label["clip"] != clip or label["requested_frames"] != req["frames"]
                or req["clip"] != clip or label["nframes"] != req["nframes"]):
            raise ValueError("Roster request and private labels disagree")
        roster[clip] = label
    if set(roster) != {c[0] for c in schedule}:
        raise ValueError("Incomplete source roster")
    files = prediction_index["files"]
    paths = {(space,clip,fold,seed,dose): f"{prefix}/{clip}/{fold}_{seed}_{dose}.json"
             for space,prefix in SPACES.items() for clip,fold,seed in schedule for dose in DOSES}
    stored = {f for f in files if f.split("/")[0] in SPACES.values() and f.endswith(".json")}
    if stored != set(paths.values()):
        raise ValueError("Frozen prediction inventory must include exactly 48 expected arm outputs")
    # Do not open a single assignment until every member of the complete parent artifact is validated.
    packs = {}
    for (space,clip,fold,seed,dose), name in paths.items():
        pack = checked_json(root / name, files[name])
        info = pack["run"]
        if (pack["clip"], info["fold"], info["seed"], info["dose"]) != (clip,fold,seed,dose):
            raise ValueError(f"Dose identity differs from filename: {name}")
        packs[space,clip,fold,seed,dose] = pack
    dest = root / "scores"
    index = dict(format="warp-pilot-score-index-v1", prereg=PREREG,
                 prediction_index_sha256=prediction_index_sha, roster_index_sha256=roster_index_sha,
                 geometry_sha256=geometry_sha, cases=[], summaries={})
    for space in SPACES:
        records = []
        for clip,fold,seed in schedule:
            rect = geometry["cases"][clip]["L"]["rect"]
            for dose in DOSES:
                k = space,clip,fold,seed,dose
                result = score_dose(roster[clip], packs[k])
                result["regions"] = region_counts(roster[clip], result, rect)
                records.append(result)
                file = f"{space}/{clip}/{fold}_{seed}_{dose}.json"
                index["cases"].append(dict(space=space, clip=clip, fold=fold, seed=seed, dose=dose,
                                           file=file, sha256=write_once(dest/file,result),
                                           prediction_sha256=files[paths[k]]))
        report = summarize(records, schedule)
        report["registered_primary"] = space=="native"
        report["source_decisions"] = {}
        for clip in roster:
            rows = [c for c in report["cases"] if c["clip"] == clip]
            delta = geometry["cases"][clip]["dose_difference"]
            n = roster[clip]["population"]["selected"]
            flags = [c["thresholds"]["0.5"]["seed_contrast_if_complete_16"] for c in rows]
            dose_ok = delta is not None and math.isfinite(delta) and delta >= .10
            complete = all(c["all_doses_available"] for c in rows)
            met = n == 16 and dose_ok and flags.count(True) >= 2
            unresolved = n == 16 and dose_ok and not met and flags.count(False) < 2
            report["source_decisions"][clip] = dict(n=n, dose_difference=delta, dose_sufficient=dose_ok,
                                                    complete=complete, qualifying_seed_count=flags.count(True),
                                                    seed_conditions=flags,
                                                    outcome=("sensitivity_only" if space!="native" else
                                                             "advance" if met else "unresolved" if unresolved else "stop"),
                                                    advance=bool(space=="native" and met))
        file = f"{space}_summary.json"
        index["summaries"][space] = dict(file=file, sha256=write_once(dest/file,report))
    return write_once(dest/"index.json",index)


def self_check():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        indexes, files, geocases = [],{},{}
        for fold in (33,35):
            clip = f"BEE24-{68-fold}-tail"
            data = [[1,1,4,1,4,4],[3,1,9,1,4,4],[1,2,9,8,4,4],[3,2,4,8,4,4]]
            req,lab = prepare_sample(data,[[10,0],[30,0],[30,30],[10,30]],clip,3)
            lp = f"{clip}/labels.private.json"
            rp = f"{clip}/requests.json"
            indexes.append(dict(clip=clip,fold=fold,labels_file=lp,requests_file=rp,
                                labels_sha256=write_once(root/"roster"/lp,lab),
                                requests_sha256=write_once(root/"roster"/rp,req)))
            geocases[clip] = dict(L=dict(rect=[7,0,30,30]),dose_difference=.25)
            for _,prefix in SPACES.items():
                for seed in range(3):
                    for dose in DOSES:
                        det = {1:[[4,1,8,5,.9],[9,8,13,12,.9]],3:[[9,1,13,5,.9],[4,8,8,12,.9]]}
                        if dose in ("C","U"):
                            det[1] = []
                        p = freeze_predictions(req,dict(fold=fold,seed=seed,dose=dose),det)
                        path = f"{prefix}/{clip}/{fold}_{seed}_{dose}.json"
                        files[path] = write_once(root/path,p)
        r = write_once(root/"roster/index.json",dict(prereg=PREREG,cases=indexes))
        g = write_once(root/"geometry.json",dict(cases=geocases))
        p = write_once(root/"detector_done.json",dict(prereg=PREREG,files=files))
        digest = run(root,p,r,g)
        index = checked_json(root/"scores/index.json",digest)
        assert len(index["cases"])==48
        for s in SPACES:
            entry = index["summaries"][s]
            report = checked_json(root/"scores"/entry["file"],entry["sha256"])
            assert len(report["cases"])==6
            assert all(c["thresholds"]["0.5"]["four_bits"]["0101"]==2 for c in report["cases"])
            assert all(not c["advance"] for c in report["source_decisions"].values())
        assert digest == run(root,p,r,g)
        (root / next(iter(files))).write_bytes(b"{}")
        try:
            run(root,p,r,g)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed prediction accepted")
        print("48 synthetic matched files, two summaries, shortfall, deterministic replay and SHA failure verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    score = sub.add_parser("score")
    score.add_argument("root",type=Path)
    for key in ("prediction-index-sha", "roster-index-sha", "geometry-sha"):
        score.add_argument("--"+key,required=True)
    args = parser.parse_args()
    if args.command == "self-check":
        self_check()
    else:
        print(run(args.root,args.prediction_index_sha,args.roster_index_sha,args.geometry_sha))
