"""Score clip-local official event graphs and replay the paired whiteboard result."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


CLIPS = ("BEE24-10", "BEE24-15")
SEEDS = ("0", "1", "2")
IOUS = ("05", "03")


def canonical(record):
    return (json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True,
                       allow_nan=False) + "\n").encode("ascii")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data), sha(data)


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp) / path.name
        staged.write_bytes(data)
        try:
            os.link(staged, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Existing output differs: {path.name}")


def counts(rows, nframes):
    mapping = {}
    for tid, sf, *_ in rows:
        if type(tid) is not int or tid in mapping or type(sf) is not int or not 1 <= abs(sf) <= nframes:
            raise ValueError("An ID repeats or a signed event frame is invalid.")
        mapping[tid] = sf
    return dict(sorted(mapping.items()))


def direction_counts(events):
    return [sum(f > 0 for f in events.values()), sum(f < 0 for f in events.values())]


def event_metrics(tp_dir, pred_dir, gt_dir):
    fp_dir = [p-t for p,t in zip(pred_dir,tp_dir)]
    fn_dir = [g-t for g,t in zip(gt_dir,tp_dir)]
    tp, fp, fn = sum(tp_dir), sum(fp_dir), sum(fn_dir)
    if min(*tp_dir,*fp_dir,*fn_dir) < 0:
        raise ValueError("Invalid directional matching counts.")
    return dict(tp=tp, fp=fp, fn=fn, tp_by_direction=tp_dir, fp_by_direction=fp_dir,
                fn_by_direction=fn_dir, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def count_metrics(pred, truth, absolute):
    return dict(totals=pred, gt_totals=truth, absolute_count_error=absolute,
                whole_clip_error_pct=100*sum(absolute)/sum(truth) if sum(truth) else None)


def clip_counts(pred, truth):
    return count_metrics(pred, truth, [abs(a-b) for a,b in zip(pred,truth)])


def pool(records):
    """IDs from different acquisition files never enter the same bipartite graph."""
    def combined(key):
        return [sum(r[key][d] for r in records) for d in (0,1)]
    pred, gt = combined("totals"), combined("gt_totals")
    result = count_metrics(pred, gt, combined("absolute_count_error"))
    for threshold in IOUS:
        if threshold in records[0]:
            tp = [sum(r[threshold]["tp_by_direction"][d] for r in records) for d in (0,1)]
            result[threshold] = event_metrics(tp, pred, gt)
    return result


def median_nullable(values):
    present = [x for x in values if x is not None]
    if present and len(present) != len(values):
        raise ValueError("Direction budget caused inconsistent metric denominators.")
    return median(present) if present else None


def summarize(native, supported, quota, e3):
    if len(quota) != 99 or any(q["totals"] != supported["totals"] or
                                q["absolute_count_error"] != supported["absolute_count_error"] for q in quota):
        raise ValueError("Quota replicates have incorrect count budgets.")
    removed = {"total": sum(native["totals"])-sum(supported["totals"])}
    med = {"whole_clip_error_pct": supported["whole_clip_error_pct"]}
    for k in IOUS:
        med[k] = {field: median_nullable([q[k][field] for q in quota])
                  for field in ("tp","fp","fn","f1","precision","recall")}
        omitted = {f"lost_{field}": native[k][field] - supported[k][field] for field in ("tp","fp")}
        if min(omitted.values()) < 0 or sum(omitted.values()) != removed["total"]:
            raise ValueError("Subset matching violates expected TP/FP monotonicity.")
        removed[k] = omitted
    return dict(native=native, supported=supported, quota=quota, quota_median=med,
                removed=removed, e3=e3)


def graph_for(edges, pred, gt):
    px = {p:i for i,p in enumerate(pred)}
    gx = {g:i for i,g in enumerate(gt)}
    ij = set()
    for p,g in edges:
        if p not in px or g not in gx or (pred[p]>0) != (gt[g]>0):
            raise ValueError("Official candidate edge has incompatible endpoints.")
        ij.add((px[p],gx[g]))
    rows, cols = zip(*sorted(ij)) if ij else ([],[])
    return csr_matrix((np.ones(len(ij),dtype=np.int8),(rows,cols)),shape=(len(pred),len(gt)))


def score_clip(frozen, labels):
    nframes = frozen["nframes"]
    gt = counts(labels["gt"], nframes)
    ng = direction_counts(gt)
    tables = {s:counts(frozen["tracks"][s]["predictions"], nframes) for s in SEEDS}
    e3_totals = [sum(direction_counts(t)[d] for t in tables.values()) / 3 for d in (0,1)]
    e3 = clip_counts(e3_totals, ng)
    seeds = {}
    for s,pred in tables.items():
        anchor = frozen["anchors"][s]
        relations = labels["anchors"][s]
        graphs = {k:graph_for(relations["edges"+k], pred, gt) for k in IOUS}
        indexes = {pid:i for i,pid in enumerate(pred)}
        def score_subset(ids):
            if len(set(ids)) != len(ids) or not set(ids).issubset(pred):
                raise ValueError("Retained subset contains duplicate or missing IDs.")
            indices = np.array([indexes[p] for p in ids], dtype=np.int64)
            positive = np.array([pred[p] > 0 for p in ids], dtype=bool)
            npos = int(positive.sum())
            npred = [npos,len(ids)-npos]
            out = clip_counts(npred,ng)
            for k,graph in graphs.items():
                subset = graph[indices]
                if subset.nnz:
                    hit = maximum_bipartite_matching(subset,perm_type="column") >= 0
                    tpin = int(np.sum(positive & hit))
                    tpdir = [tpin,int(hit.sum())-tpin]
                else:
                    tpdir = [0,0]
                out[k] = event_metrics(tpdir,npred,ng)
            return out
        native = score_subset(list(pred))
        supported = score_subset(anchor["retained"])
        quota = [score_subset(ids) for ids in anchor["quota_masks"]]
        seeds[s] = summarize(native,supported,quota,e3)
    return {"seeds":seeds}


def decision(pooled):
    def gain(seed):
        r = pooled[seed]
        a,b = r["supported"]["05"]["f1"],r["native"]["05"]["f1"]
        if a is None or b is None:
            raise ValueError("Whiteboard micro F1 is not defined.")
        return a-b
    main = pooled["0"]
    difference = gain("0")
    tp_gain = main["supported"]["05"]["tp"] - main["quota_median"]["05"]["tp"]
    rotation = [gain("1"),gain("2")]
    native = main["native"]["05"]
    supported = main["supported"]["05"]
    dn = 2*native["tp"] + native["fp"] + native["fn"]
    ds = 2*supported["tp"] + supported["fp"] + supported["fn"]
    two_points = 100*(supported["tp"]*dn - native["tp"]*ds) >= dn*ds
    return dict(micro_f1_gain=difference, tp_gain_vs_quota_median=tp_gain,
                rotation_f1_gains=rotation, passes=bool(
                    two_points and tp_gain >= 2 and any(g > 1e-9 for g in rotation)))


def score(masks, labels):
    if set(masks["clips"]) != set(CLIPS) or set(labels["clips"]) != set(CLIPS):
        raise ValueError("Only the two preregistered whiteboard clips belong in this comparison.")
    clips = {name:score_clip(masks["clips"][name], labels["clips"][name]) for name in CLIPS}
    pooled = {}
    for seed in SEEDS:
        pair = [clips[c]["seeds"][seed] for c in CLIPS]
        pooled[seed] = summarize(pool([r["native"] for r in pair]),pool([r["supported"] for r in pair]),
                                 [pool([r["quota"][i] for r in pair]) for i in range(99)],
                                 pool([r["e3"] for r in pair]))
    return dict(study=109, clips=clips, pooled=pooled, decision=decision(pooled))


def public_inputs(masks, labels):
    """Select replay fields explicitly: hashes identify the private full files."""
    compact_masks, compact_labels = {}, {}
    for name in CLIPS:
        source, annotations = masks["clips"][name], labels["clips"][name]
        compact_masks[name] = dict(nframes=source["nframes"],tracks={},anchors={})
        for s in SEEDS:
            compact_masks[name]["tracks"][s] = {"predictions":source["tracks"][s]["predictions"]}
            orig = source["anchors"][s]
            compact_masks[name]["anchors"][s] = {k:orig[k] for k in
                ("retained","quota_masks","pairs","buddy_seeds") if k in orig}
        compact_labels[name] = dict(gt=annotations["gt"],anchors={
            s:{key:annotations["anchors"][s][key] for key in ("edges05","edges03")} for s in SEEDS})
    return {"clips":compact_masks},{"clips":compact_labels}


def self_check():
    # Duplicated integer IDs in different clips must not compete for GT.
    # Predicted counts [2,1] and [0,1] vs GT [1,1] and [1,1] have opposite inward errors.
    masks, labels = {"clips":{}},{"clips":{}}
    for i,c in enumerate(CLIPS):
        events = [[1,5],[2,-8],[3,10]] if i == 0 else [[2,-8]]
        retained = [1,2] if i == 0 else [2]
        edges = [[1,1],[2,2]] if i == 0 else [[2,2]]
        masks["clips"][c] = dict(nframes=10,tracks={s:dict(predictions=events) for s in SEEDS},
                                  anchors={s:dict(retained=retained,quota_masks=[retained]*99) for s in SEEDS})
        labels["clips"][c] = dict(gt=[[1,5],[2,-8]],anchors={
            s:dict(edges05=edges,edges03=edges) for s in SEEDS})
    result=score(masks,labels)
    main=result["pooled"]["0"]
    assert main["native"]["05"]["tp"] == 3 and main["native"]["05"]["f1"] == .75
    assert main["supported"]["05"]["f1"] == 6/7 and main["native"]["absolute_count_error"] == [2,0]
    assert main["e3"]["whole_clip_error_pct"] == 50 and not result["decision"]["passes"]
    assert score(*public_inputs(masks,labels))==result
    # This second case passes only with the two extra supported TP at equal budgets.
    # Per-clip medians are both zero, but the actual paired-replicate median is two.
    for i,c in enumerate(CLIPS):
        tracks = [[1,1],[2,2],[3,3],[4,4]]
        yes,no = [1,2],[3,4]
        quota = [no]*50+[yes]*49 if i==0 else [yes]*49+[no]*50
        masks["clips"][c] = dict(nframes=4,tracks={s:dict(predictions=tracks) for s in SEEDS},
                                  anchors={s:dict(retained=yes,quota_masks=quota) for s in SEEDS})
        labels["clips"][c] = dict(gt=[[1,1],[2,2]],anchors={
            s:dict(edges05=[[1,1],[2,2]],edges03=[[1,1],[2,2]]) for s in SEEDS})
    paired=score(masks,labels)
    assert all(paired["clips"][c]["seeds"]["0"]["quota_median"]["05"]["tp"]==0 for c in CLIPS)
    assert paired["pooled"]["0"]["quota_median"]["05"]["tp"]==2 and paired["decision"]["passes"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="cmd",required=True)
    sub.add_parser("self-check")
    run=sub.add_parser("score")
    run.add_argument("masks",type=Path)
    run.add_argument("labels",type=Path)
    run.add_argument("--mask-sha",required=True)
    run.add_argument("--out",type=Path,required=True)
    run.add_argument("--bundle",type=Path,required=True)
    rep=sub.add_parser("replay")
    rep.add_argument("bundle",type=Path)
    rep.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.cmd=="self-check":
        self_check()
        print("Synthetic whiteboard micro-count and quota checks passed.")
        return
    if args.cmd=="score":
        masks, mask_hash = read(args.masks)
        if mask_hash != args.mask_sha:
            raise ValueError("Input prediction freeze differs from supplied SHA-256.")
        labels, label_hash = read(args.labels)
        if labels["mask_sha256"] != mask_hash:
            raise ValueError("Official labels do not identify this prediction freeze.")
        compact, annotations = public_inputs(masks,labels)
        report = score(compact,annotations)
        evidence = dict(study=109, prereg=masks.get("prereg"), mask_sha256=mask_hash,
                        labels_sha256=label_hash,masks=compact,labels=annotations,
                        results_sha256=sha(canonical(report)))
        write_once(args.bundle,gzip.compress(canonical(evidence),mtime=0))
    else:
        evidence,_ = read(args.bundle)
        report = score(evidence["masks"],evidence["labels"])
        if sha(canonical(report)) != evidence["results_sha256"]:
            raise ValueError("Recomputed results differ from the sealed results SHA-256.")
    if args.out:
        write_once(args.out,canonical(report))
    print("Results SHA-256", sha(canonical(report)))
    print(json.dumps(report["decision"],allow_nan=False))


if __name__=="__main__":
    main()
