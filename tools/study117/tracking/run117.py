"""Freeze 36 CPU P0 MOT outputs; in a separate invocation score complete 1,000-frame source tails."""
import argparse
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from beetrack.evaluate import by_track,load_tracks
from tools.study103.replay import bins
from tools.study108.inputs108 import describe


ARMS = ("U","A","Z","Q","Zm","D")
HISTORICAL_SHA = "2a7d8bf2d4b027dafa59d924e5b908ff2b88c0edbdf9fcded097c60383edd383"
THRESHOLDS = ("0.5","0.3")
EPS = Fraction(1,10**9)


def schedule():
    return [(fold,seed,arm) for fold in (33,35) for seed in range(3) for arm in ARMS]


def clip_name(fold):
    return f"BEE24-{68-fold}-tail"


def file_sha(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src,"sha256").hexdigest()


def load_checked(path, expected, zipped=False):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Frozen file differs from registered SHA-256: {path}")
    return json.loads(gzip.decompress(raw) if zipped else raw)


def write_once(path, value):
    raw = (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        partial = Path(temp)/"payload"
        partial.write_bytes(raw)
        try:
            os.link(partial,path)
        except FileExistsError:
            if path.read_bytes()!=raw:
                raise ValueError(f"Frozen artifact already exists with other bytes: {path}")
    return hashlib.sha256(raw).hexdigest()


def method_hashes(names):
    here = Path(__file__).resolve().parent
    return {name:hashlib.sha256((here/name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in names}


def prediction_counts(rows, polygon, nframes):
    events = describe(rows,polygon)
    stream = [r[1] for r in events]
    offsets = bins(stream,nframes).reshape(30,-1,2)
    return dict(predictions=events, totals={"in":sum(f>0 for f in stream),"out":sum(f<0 for f in stream)},
                windows=offsets.tolist())


def expected_npy(fold,seed,arm):
    return f"detections/{fold}_{seed}/{arm}.npy"


def track(root, prereg, detector_sha, zones_path, weight_path, selected_folds, selected_seeds, finalize,
          tracker=None, strict_linker=None):
    if not re.fullmatch("[0-9a-f]{40}",prereg):
        raise ValueError("Expected the pinned preregistration commit")
    detector = load_checked(root/"detector_done.json",detector_sha)
    if detector["study"]!=117 or detector["prereg"]!=prereg:
        raise ValueError("Mismatched detector study or preregistration")
    hashes = detector["files"]
    expected = {expected_npy(*case) for case in schedule()}
    actual = {k for k in hashes if k.startswith("detections/") and k.endswith(".npy")}
    if expected!=actual:
        raise ValueError("Expected exactly the registered 36 detector caches")
    for f in sorted(expected):
        if file_sha(root/f)!=hashes[f]:
            raise ValueError(f"Changed detector input: {f}")
    if tracker is None:
        from p0_117 import StrictLinker, track_detections
        tracker,strict_linker = track_detections,StrictLinker(weight_path)
    zones = json.loads(zones_path.read_bytes())
    zone_sha = file_sha(zones_path)
    tracking_code = method_hashes(("run117.py","p0_117.py"))
    dest = root/"tracking"
    for fold,seed,arm in schedule():
        if fold not in selected_folds or seed not in selected_seeds:
            continue
        clip = clip_name(fold)
        stem = f"{clip}/{fold}_{seed}_{arm}"
        cache = expected_npy(fold,seed,arm)
        mot_name = f"mot/{stem}.txt"
        file = f"counts/{stem}.json"
        output = dest/mot_name
        output.parent.mkdir(parents=True,exist_ok=True)
        # Fresh inference is avoided only on an exact, already-frozen cache+MOT+count tuple.
        if (dest/file).exists() and output.is_file():
            old = json.loads((dest/file).read_bytes())
            if (old["prereg"]!=prereg or old["detector_index_sha256"]!=detector_sha or
                    old["zone_sha256"]!=zone_sha or old["det_sha256"]!=hashes[cache]
                    or old["mot_sha256"]!=file_sha(output) or old["method_sha256"]!=tracking_code):
                raise ValueError(f"Cached P0 artifact failed lineage validation: {stem}")
            got = prediction_counts(load_tracks(output),zones[clip],1000)
            if any(old[k]!=v for k,v in got.items()):
                raise ValueError(f"Physical MOT differs from frozen counts: {stem}")
            continue
        started = perf_counter_ns()
        receipt = tracker(root/cache,1000,strict_linker,output)
        elapsed_ns = perf_counter_ns()-started
        loaded = load_tracks(output)
        counts = prediction_counts(loaded,zones[clip],1000)
        receipt.update(tracking_elapsed_ns=elapsed_ns,elapsed_scope="load-npy+P0+MOT-write-and-hash",
                       original_track_ids=len(set(loaded[:,1])),gap_split_track_ids=len(set(by_track(loaded)[:,1])))
        meta = dict(fold=fold,seed=seed,arm=arm,clip=clip,nframes=1000,prereg=prereg,
                    detector_index_sha256=detector_sha,det_sha256=hashes[cache],zone_sha256=zone_sha,
                    mot_file=mot_name,mot_sha256=file_sha(output),tracking_receipt=receipt,method_sha256=tracking_code)
        if receipt["mot_sha256"]!=meta["mot_sha256"] or receipt["input_file_sha256"]!=hashes[cache]:
            raise ValueError(f"Native tracker receipt disagrees with frozen inputs or output: {stem}")
        write_once(dest/file,{**meta,**counts})
        print(f"Frozen {stem}",flush=True)
    if not finalize:
        return None
    cases = []
    for fold,seed,arm in schedule():
        clip = clip_name(fold)
        stem = f"{clip}/{fold}_{seed}_{arm}"
        file = f"counts/{stem}.json"
        record = json.loads((dest/file).read_bytes())
        mot = f"mot/{stem}.txt"
        if (record["detector_index_sha256"]!=detector_sha or record["prereg"]!=prereg
                or record["zone_sha256"]!=zone_sha or record["det_sha256"]!=hashes[expected_npy(fold,seed,arm)]
                or (record["fold"],record["seed"],record["arm"],record["clip"])!=(fold,seed,arm,clip)
                or record["mot_file"]!=mot or file_sha(dest/mot)!=record["mot_sha256"]
                or record["method_sha256"]!=tracking_code):
            raise ValueError(f"Invalid pre-existing partial output: {file}")
        cases.append(dict(fold=fold,seed=seed,arm=arm,clip=clip,file=file,sha256=file_sha(dest/file),
                          mot_file=mot,mot_sha256=record["mot_sha256"]))
    return write_once(dest/"index.json",dict(study=117,prereg=prereg,stage="predictions_only",
                                            detector_index_sha256=detector_sha,zone_sha256=zone_sha,cases=cases,
                                            method_sha256=tracking_code))


def count_w(record):
    w = record["window"]
    return Fraction(100*w["numerator"],w["denominator"]) if w["denominator"] else None


def f1_at(record,tau):
    s = record["iou"][tau]
    return Fraction(2*s["tp"],max(1,2*s["tp"]+s["fp"]+s["fn"]))


def mean(values):
    return None if None in values else sum(values,Fraction())/len(values)


def subtract(a,b):
    return None if a is None or b is None else a-b


def portable(value):
    if isinstance(value,Fraction):
        return float(value)
    if isinstance(value,dict):
        return {k:portable(v) for k,v in value.items()}
    if isinstance(value,list):
        return [portable(v) for v in value]
    return value


def fold_summary(records):
    summary = {}
    for fold in (33,35):
        rows = {(r["seed"],r["arm"]):r for r in records if r["fold"]==fold}
        getw = lambda s,a:count_w(rows[s,a])
        means = {a:dict(w=mean([getw(s,a) for s in range(3)]),
                        f1={t:mean([f1_at(rows[s,a],t) for s in range(3)]) for t in THRESHOLDS}) for a in ARMS}
        seedwise = []
        for s in range(3):
            for a in ARMS:
                base, case = rows[s,"U"],rows[s,a]
                seedwise.append(dict(seed=s,arm=a,w_reduction_pp=subtract(getw(s,"U"),getw(s,a)),
                                     f1_delta={t:f1_at(case,t)-f1_at(base,t) for t in THRESHOLDS},
                                     direction_count_delta={d:case["pred_counts"][d]-base["pred_counts"][d] for d in ("in","out")}))
        gains = [subtract(getw(s,"U"),getw(s,"A")) for s in range(3)]
        wins = None if None in gains else sum(g>EPS for g in gains)
        improvement = subtract(means["U"]["w"],means["A"]["w"])
        f1_delta = means["A"]["f1"]["0.5"]-means["U"]["f1"]["0.5"]
        competitors = {a:dict(w_advantage_pp=subtract(means[a]["w"],means["A"]["w"]),
                              f1_delta=means["A"]["f1"]["0.5"]-means[a]["f1"]["0.5"]) for a in ("Z","Q")}
        summary[str(fold)] = dict(clip=clip_name(fold),means=means,against_new_cpu_u=seedwise,
                                  utility=dict(mean_w_gain_pp=improvement,positive_seeds=wins,mean_f1_delta=f1_delta,
                                               passed=None if improvement is None else improvement>=2 and wins>=2 and f1_delta>=-EPS),
                                  complexity=dict(comparisons=competitors,passed=None if improvement is None else all(
                                      v["w_advantage_pp"]>EPS and v["f1_delta"]>=-EPS for v in competitors.values())))
    return portable(summary)


def score(root,prereg,tracking_sha,runs,zones_path,archive_path,history_sha=HISTORICAL_SHA):
    from score117 import FORMAT,score_case,replay_case
    dest = root/"tracking"
    index = load_checked(dest/"index.json",tracking_sha)
    cases = index["cases"]
    if (index["prereg"]!=prereg or index["zone_sha256"]!=file_sha(zones_path) or index["stage"]!="predictions_only"
            or [(c["fold"],c["seed"],c["arm"]) for c in cases]!=schedule()):
        raise ValueError("Incomplete or mismatched prediction-only freeze")
    frozen = []
    for c in cases:
        record = load_checked(dest/c["file"],c["sha256"])
        if (file_sha(dest/c["mot_file"])!=c["mot_sha256"] or c["mot_sha256"]!=record["mot_sha256"]
                or record["zone_sha256"]!=index["zone_sha256"] or record["detector_index_sha256"]!=index["detector_index_sha256"]):
            raise ValueError(f"MOT or detector provenance changed: {c['file']}")
        frozen.append(record)
    # No GT, historical labels, or signed scoring graphs are read above this point.
    archive = load_checked(archive_path,history_sha,zipped=True)
    zones = json.loads(zones_path.read_bytes())
    gt, gt_sha, history = {},{},{}
    for fold,seed,arm in schedule():
        if arm!="U":
            continue
        clip = clip_name(fold)
        if clip not in gt:
            file = runs/"gt"/"val"/clip/"gt"/"gt.txt"
            gt[clip],gt_sha[clip] = load_tracks(file),file_sha(file)
        key = f"{fold}_{seed}_tail"
        label = archive["labels"][key]
        raw = archive["masks"]["predictions"][key]
        if (gt_sha[clip]!=label["gt_mot_sha256"] or raw["sources"]["beetrack/zones.json"]!=index["zone_sha256"]
                or raw["clip"]!=clip or label["predictions_sha256"]!=archive["masks"]["prediction_sha256"][key]):
            raise ValueError(f"Historical tail identity mismatch: {key}")
        graph = dict(format=FORMAT,nframes=1000,gt=label["gt"],predictions=raw["predictions"],
                     edges05=label["edges05"],edges03=label["edges03"])
        hist = replay_case(graph)
        history[key] = dict(fold=fold,seed=seed,clip=clip,mot_sha256=raw["mot_sha256"],
                            gt_sha256=gt_sha[clip],**{k:hist[k] for k in ("w30_percent","pred_counts","gt_counts","iou")})
    result_dir = root/"scores"
    output,results = [],[]
    for c,p in zip(cases,frozen):
        fold,seed,arm,clip = c["fold"],c["seed"],c["arm"],c["clip"]
        m = load_tracks(dest/c["mot_file"])
        data = score_case(gt[clip],m,zones[clip],1000)
        key = f"{fold}_{seed}_tail"
        ev = data["evidence"]
        descriptors = [[pid,frame,*[round(float(x),2) for x in box]] for pid,frame,*box in ev["predictions"]]
        if (descriptors!=p["predictions"] or data["pred_counts"]!=p["totals"]
                or ev["gt"]!=archive["labels"][key]["gt"]
                or bins([r[1] for r in ev["predictions"]],1000).reshape(30,-1,2).tolist()!=p["windows"]):
            raise ValueError(f"Full score does not reproduce the physical prediction freeze and historical GT: {key}/{arm}")
        result = dict(fold=fold,seed=seed,arm=arm,clip=clip,nframes=1000,mot_sha256=c["mot_sha256"],
                      gt_sha256=gt_sha[clip],**data)
        fn = f"{clip}/{fold}_{seed}_{arm}.json"
        output.append(dict(fold=fold,seed=seed,arm=arm,clip=clip,file=fn,sha256=write_once(result_dir/fn,result)))
        results.append(result)
    summary = dict(study=117,prereg=prereg,tracking_index_sha256=tracking_sha,folds=fold_summary(results),
                   historical_gpu_fp16=dict(archive_sha256=history_sha,cases=history))
    return write_once(result_dir/"index.json",dict(study=117,prereg=prereg,tracking_index_sha256=tracking_sha,
                                                 method_sha256=method_hashes(("run117.py","score117.py")),
                                                 cases=output,summary_file="summary.json",
                                                 summary_sha256=write_once(result_dir/"summary.json",summary)))


def main():
    from beetrack.common import RUNS
    from beetrack.evaluate import ZONES
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd",required=True)
    track_p,score_p = sub.add_parser("track"),sub.add_parser("score")
    for parser in (track_p,score_p):
        parser.add_argument("root",type=Path)
        parser.add_argument("--prereg",required=True)
        parser.add_argument("--zones",type=Path,default=ZONES)
    track_p.add_argument("--detector-index-sha",required=True)
    track_p.add_argument("--weights",required=True,type=Path)
    track_p.add_argument("--fold",type=int,choices=(33,35),action="append")
    track_p.add_argument("--seed",type=int,choices=range(3),action="append")
    track_p.add_argument("--finalize",action="store_true")
    score_p.add_argument("--tracking-index-sha",required=True)
    score_p.add_argument("--runs",type=Path,default=RUNS)
    score_p.add_argument("--history",type=Path,default=Path("docs/study108_source.json.gz"))
    a = p.parse_args()
    if a.cmd=="track":
        result = track(a.root,a.prereg,a.detector_index_sha,a.zones,a.weights,a.fold or (33,35),
                       a.seed or range(3),a.finalize)
    else:
        result = score(a.root,a.prereg,a.tracking_index_sha,a.runs,a.zones,a.history)
    if result is not None:
        print(result)


if __name__=="__main__":
    main()
