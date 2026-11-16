"""Certify the maximum count-error improvement available from §110 frozen donor events."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from oracle110 import score_oracle


PREREG = "fc9133e1f1ffe5dd345f0ad5664cfb3049986b78"
TOL = 1e-9


def wire(obj):
    return (json.dumps(obj, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data), sha(data)


def save(path, obj):
    data = wire(obj)
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        candidate = Path(temp)/path.name
        candidate.write_bytes(data)
        try:
            os.link(candidate, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Existing output differs: {path}")
    return sha(data)


def case_key(row):
    return f"{row['role']}_{row['fold']}_{row['anchor_seed']}_{row['part']}"


def historical(geometry, archives, reports):
    truth, native, references, provenance = {}, {}, {}, {}
    for role in ("source", "target"):
        old, arc_sha = read(archives/f"study108_{role}.json.gz")
        scores, result_sha = read(reports/f"{role}_results.json")
        if (result_sha != old["result_sha256"] or old["masks_sha256"] != geometry["inputs_sha256"][role]
                or sha(wire(old["masks"])) != old["masks_sha256"] or scores["masks_sha256"] != old["masks_sha256"]):
            raise ValueError(f"§108 reference fingerprint mismatch ({role})")
        provenance[role] = dict(archive_sha256=arc_sha, result_sha256=result_sha, mask_sha256=old["masks_sha256"])
        for row in geometry["cases"]:
            if row["role"] != role:
                continue
            name = case_key(row)
            part, fold, anchor, clip = row["part"], row["fold"], row["anchor_seed"], row["clip"]
            anchor_key = f"{fold}_{anchor}_{part}"
            hist = scores["cases"][anchor_key]
            if sha(wire(hist)) != old["case_sha256"][anchor_key]:
                raise ValueError(f"Historical case changed: {name}")
            streams = []
            for seed in range(anchor, anchor+3):
                key = f"{fold}_{seed}_{part}"
                descriptor = old["masks"]["predictions"][key]
                if (descriptor["clip"] != clip or descriptor["mot_sha256"] != row["mot_sha256"][str(seed)]
                        or old["masks"]["prediction_sha256"][key] != row["prediction_sha256"][str(seed)]):
                    raise ValueError(f"Input stream mismatch for {role}/{key}")
                times = [item[1] for item in descriptor["predictions"]]
                streams.append(times)
                gt = [g[1] for g in old["labels"][key]["gt"]]
                if truth.setdefault(clip, gt) != gt:
                    raise ValueError(f"Inconsistent GT stream for {clip}")
            native[name] = streams
            ref = dict(supported108=hist["kept"]["window"], e3_108=hist["e3"]["window"],
                       native_p0=hist["full"]["window"])
            if role == "target":
                ref["e3_95"] = old["labels"][anchor_key]["group_refs"]["P0"]
            references[name] = ref
    return dict(truth=truth, native=native, references=references, provenance=provenance)


def cells(signed, n):
    f = np.asarray(signed, dtype=int)
    if np.any(abs(f) > n) or np.any(f == 0):
        raise ValueError("Signed frame outside the clip")
    chunks = []
    for offset in range(0,750,25):
        width = (n + offset + 749)//750
        bins = (abs(f)-1+offset)//750
        chunks.append(np.bincount(2*bins+(f<0), minlength=2*width))
    return np.concatenate(chunks)


def validate_proof(base, donors, gt, n, answer):
    """Reconstruct the stored primal and rational dual without requiring the same winning subset."""
    picks = answer["selected_donor_indices"]
    if len(set(picks)) != len(picks) or any(i < 0 or i >= len(donors) for i in picks):
        raise ValueError("Invalid donor witness")
    residual = cells(gt,n) - cells(base,n)
    witness = cells([donors[i] for i in picks],n)
    upper = int(np.abs(residual-witness).sum())
    all_cells = cells(donors,n)
    lower = int((np.maximum(-residual,0)+np.maximum(residual-all_cells,0)).sum())
    proof = answer["proof"]
    if proof["kind"] == "rational_dual":
        scale = proof["scale"]
        weights = np.zeros(len(residual),dtype=np.int64)
        if scale < 1 or len({i for i,_ in proof["nonzero_weights"]}) != len(proof["nonzero_weights"]):
            raise ValueError("Invalid rational certificate weights")
        for i,v in proof["nonzero_weights"]:
            if not 0 <= i < len(weights) or abs(v) > scale:
                raise ValueError("Out-of-range dual coordinate")
            weights[i] = v
        dual = int(weights @ residual) - sum(max(0,int(weights @ cells([f],n))) for f in donors)
        if dual != proof["numerator"]:
            raise ValueError("Invalid dual numerator")
        lower = max(lower,-((-dual)//scale))
    elif proof["kind"] != "interval_floor":
        raise ValueError("Unknown certificate type")
    if ((lower,upper) != (answer["integer_lower"],answer["primal_upper"])
            or answer["certified"] != (upper == lower) or upper < lower):
        raise ValueError("Stored witness or optimality certificate failed")


def evaluate(row, evidence, saved=None):
    name = case_key(row)
    gt = evidence["truth"][row["clip"]]
    n = row["nframes"]
    if len(gt) == 0:
        raise ValueError("Undefined whole-clip percentage without GT")
    base = [x[1] for x in row["kept"]]
    donors = [x["b"][1] for x in row["donors"]]
    answer = score_oracle(base, donors, gt, n)
    validate_proof(base, donors, gt, n, answer)
    if saved is not None:
        stored = saved["oracle"]
        validate_proof(base, donors, gt, n, stored)
        # Equal-cost witnesses may differ across solver versions.
        for k in ("base_numerator","all_numerator","cell_floor_numerator","direction_floor_numerator",
                  "optimal_numerator","denominator","direction_totals","certified"):
            if answer[k] != stored[k]:
                raise ValueError(f"Solver replay differs for {name}/{k}")
        answer = stored
    den = answer["denominator"]
    counts = answer["direction_totals"]
    k,d,t = (sum(counts[x]) for x in ("base","donor","gt"))
    total_floor = 100*max(k-t,t-k-d,0)/t
    forced_floor = 100*sum(abs(a+b-c) for a,b,c in zip(counts["base"],counts["donor"],counts["gt"]))/t
    frozen_counts = [row["counts"][tag] for tag in ("K","D")]
    if [[c["in"],c["out"]] for c in frozen_counts] != [counts["base"],counts["donor"]]:
        raise ValueError(f"Frozen direction counts differ in {name}")
    target = cells(gt,n)
    histograms = [cells(seq,n) for seq in evidence["native"][name]]
    def error(hist):
        return float(100*np.abs(target-hist).sum()/den)
    e3, native = error(np.mean(histograms,axis=0)), error(histograms[0])
    ref = evidence["references"][name]
    for x,y in [(e3,ref["e3_108"]), (native,ref["native_p0"]), (100*answer["base_numerator"]/den,ref["supported108"])]:
        if abs(x-y)>TOL:
            raise ValueError(f"Mismatch against frozen §108 scores for {name}")
    if row["role"]=="target" and abs(e3 - ref["e3_95"])>TOL:
        raise ValueError(f"E3 differs from §95 for {name}")
    return dict(oracle=answer, total_interval_floor=total_floor, forced_direction_floor=forced_floor,
                e3=e3, original_p0=native, supported108=ref["supported108"],
                selected_b_ids=[row["donors"][i]["b"][0] for i in answer["selected_donor_indices"]])


def summaries(geometry, results):
    groups = {}
    for row in geometry["cases"]:
        group=f"{row['role']}_{row['fold']}_{row['part']}"
        groups.setdefault(group,[]).append(results[case_key(row)])
    summaries={}
    for key, group in groups.items():
        if len(group)!=3:
            raise ValueError(f"Expected 3 non-overlapping detector groups for {key}")
        means = {k:float(np.mean([r[k] for r in group])) for k in
                 ("total_interval_floor","forced_direction_floor","e3","original_p0","supported108")}
        for name in ("direction_floor","cell_floor","base","all","optimal"):
            values = [r["oracle"]["percent"][name] for r in group]
            means[name] = None if None in values else float(np.mean(values))
        for stage in ("integer_lower","primal_upper"):
            means[stage] = float(np.mean([100*r["oracle"][stage]/r["oracle"]["denominator"] for r in group]))
        for name in ("base","donor","gt"):
            means[f"{name}_count_sum"] = sum(sum(r["oracle"]["direction_totals"][name]) for r in group)
        summaries[key]=means
    decisions = {}
    for hive in (33,35):
        m=summaries[f"target_{hive}_head"]
        def excluded(q):
            return q > m["e3"]-2+TOL if hive==33 else q > m["supported108"]+2+TOL or q >= m["e3"]-TOL
        decisions[str(hive)] = {key:excluded(m[key]) for key in
            ("total_interval_floor","direction_floor","cell_floor","integer_lower","forced_direction_floor","all")}
        decisions[str(hive)]["oracle_witness_meets"] = not excluded(m["primal_upper"])
    return dict(groups=summaries, excluded_by=decisions,
                stop_joint=any(v["integer_lower"] for v in decisions.values()))


def analyze(geometry, evidence, previous=None):
    expected = {f"{role}_{fold}_{anchor}_{part}" for role in ("source","target") for fold in (33,35)
                for anchor in (0,3,6) for part in (("head","tail") if role=="source" else ("head",))}
    if len(geometry["cases"])!=18 or {case_key(r) for r in geometry["cases"]}!=expected:
        raise ValueError("Missing, duplicate or unregistered study case")
    results={}
    for row in geometry["cases"]:
        name=case_key(row)
        results[name]=evaluate(row,evidence,previous["cases"][name] if previous else None)
    output = dict(cases=results, summary=summaries(geometry,results))
    if previous is not None and previous != output:
        raise ValueError("Archived comparison values or decisions differ from replay")
    return output


def report_text(report):
    print("case    K(in/out) D(in/out) GT(in/out)   E3    base    all   add-all-totalLB   add-all-dirLB   dirLB   cellLB   exact")
    forced = {}
    for key,value in sorted(report["cases"].items()):
        opt = value["oracle"]
        t,p = opt["direction_totals"], opt["percent"]
        all_total_floor = 100 * abs(sum(t["base"]) + sum(t["donor"]) - sum(t["gt"])) / sum(t["gt"])
        role, fold, _, part = key.split("_")
        forced.setdefault((role, fold, part), []).append(all_total_floor)
        value_str = "unproved" if p["optimal"] is None else f"{p['optimal']:.2f}"
        print(f"{key:17s} {t['base']} {t['donor']} {t['gt']}  {value['e3']:.2f} {p['base']:.2f} "
              f"{p['all']:.2f} {all_total_floor:.2f} {value['forced_direction_floor']:.2f} "
              f"{p['direction_floor']:.2f} {p['cell_floor']:.2f} {value_str}")
    for parts, values in sorted(forced.items()):
        print(f"{'_'.join(parts)} forced-all direction-blind total floor: {np.mean(values):.2f}")
    print(json.dumps(report["summary"],indent=1,allow_nan=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest="command",required=True)
    score=modes.add_parser("score")
    score.add_argument("geometry",type=Path)
    score.add_argument("out",type=Path)
    score.add_argument("--geometry-sha",required=True)
    score.add_argument("--archive-dir",type=Path,default=Path("docs"))
    score.add_argument("--historical-dir",type=Path,default=Path("D:/beetrack_runs/study108"))
    replay=modes.add_parser("replay")
    replay.add_argument("bundle",type=Path)
    args=parser.parse_args()
    if args.command=="score":
        geometry,checksum=read(args.geometry)
        if geometry["prereg"]!=PREREG or checksum!=args.geometry_sha:
            raise ValueError("Geometry preregistration or prediction-freeze fingerprint mismatch")
        evidence=historical(geometry,args.archive_dir,args.historical_dir)
        report=analyze(geometry,evidence)
        result=dict(prereg=PREREG,geometry_sha256=checksum,**report)
        result_sha=save(args.out/"results110.json",result)
        bundle=dict(study=110,prereg=PREREG,geometry_sha256=checksum,geometry=geometry,
                    evidence=evidence,result=result,result_sha256=result_sha)
        package_sha=save(args.out/"study110_replay.json.gz",bundle)
        print("Result SHA256",result_sha,"replay gzip SHA256",package_sha)
    else:
        bundle,_=read(args.bundle)
        geometry,previous=bundle["geometry"],bundle["result"]
        if (bundle["prereg"]!=PREREG or geometry["prereg"]!=PREREG
                or sha(wire(geometry))!=bundle["geometry_sha256"] or sha(wire(previous))!=bundle["result_sha256"]
                or previous["geometry_sha256"]!=bundle["geometry_sha256"] or previous["prereg"]!=PREREG):
            raise ValueError("Replay fingerprints do not match the original study")
        report=analyze(geometry,bundle["evidence"],{k:previous[k] for k in ("cases","summary")})
    report_text(report)


if __name__=="__main__":
    main()
