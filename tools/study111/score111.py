"""Replay paired observation deletions with tie-independent target-GT matching bounds."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import linear_sum_assignment


PHASE = {"S0":1, "S1":1, "B0":-1, "B1":-1}
THRESHOLDS = {"05":0.5, "03":0.3}


def _solve(valid, col_weight):
    """First maximize event cardinality, then sum the signed secondary weight."""
    if min(valid.shape)==0:
        return np.empty(0, dtype=int)
    bonus = int(np.abs(col_weight).sum()) + 1
    rows,cols = linear_sum_assignment(valid*(bonus + col_weight[None,:]), maximize=True)
    return cols[valid[rows,cols]]


def _window_counts(events, nframes):
    signed = np.array(list(events.values()),dtype=np.int64)
    if np.any(signed==0) or np.any(abs(signed)>nframes):
        raise ValueError("An event lies outside its clip.")
    offsets = []
    for shift in range(0,750,25):
        bins = (abs(signed)-1+shift)//750
        ncell = (nframes+shift+749)//750
        offsets.append(np.bincount(2*bins + (signed<0),minlength=2*ncell).reshape(-1,2))
    return np.vstack(offsets)


def _pairs(records):
    ret={}
    for tid,f in records:
        if type(tid) is not int or type(f) is not int or tid in ret:
            raise ValueError("Expected unique integer event IDs and signed frame numbers.")
        ret[tid]=f
    return dict(sorted(ret.items()))


def _graph_scores(truth, pred, groups, edges):
    targets, predictions = list(truth), list(pred)
    px = {pid:i for i,pid in enumerate(predictions)}
    gx = {gid:i for i,gid in enumerate(targets)}
    valid = np.zeros((len(px),len(gx)),dtype=bool)
    for p,g in edges:
        if p not in px or g not in gx or (truth[g]>0)!=(pred[p]>0):
            raise ValueError("Official edge endpoints invalid.")
        valid[px[p],gx[g]]=True
    canonical=_solve(valid,np.zeros(len(gx),dtype=int))
    tp=len(canonical)
    in_tp = sum(truth[targets[int(i)]]>0 for i in canonical)
    bounds = {}
    for name,selected in groups.items():
        mask = np.array([tid in selected for tid in targets],dtype=int)
        if mask.any():
            high,low = (_solve(valid,s*mask) for s in (1,-1))
            if len(high)!=tp or len(low)!=tp:
                raise ValueError("A secondary objective changed total event TP.")
            lo,hi=int(mask[low].sum()),int(mask[high].sum())
        else:
            hi=lo=0
        chosen=int(mask[canonical].sum())
        if not lo<=chosen<=hi<=len(selected):
            raise ValueError("Canonical matching outside tie-independent bounds.")
        isolated=[int(t) for j,t in enumerate(targets) if mask[j] and not valid[:,j].any()]
        bounds[name]=dict(n=len(selected),low=lo,high=hi,canonical=chosen,always_unmatched_ids=isolated)
    return dict(tp=tp,fp=len(pred)-tp,fn=len(truth)-tp,f1=2*tp/max(1,len(pred)+len(truth)),
                tp_by_direction=[in_tp,tp-in_tp],target_bounds=bounds)


def contrast(arms, weights, group, iou):
    """A shared native term cancels before extrema; separate arm matchings vary independently."""
    names = {name:int(w) for name,w in weights.items() if w}
    n = arms["native"][iou]["target_bounds"][group]["n"]
    lo=hi=canonical=0
    for name,w in names.items():
        bounds=arms[name][iou]["target_bounds"][group]
        if bounds["n"] != n:
            raise ValueError("Contrasted arms use different GT cohorts.")
        l,h=bounds["low"],bounds["high"]
        lo+=min(w*l,w*h)
        hi+=max(w*l,w*h)
        canonical+=w*bounds["canonical"]
    return dict(n=n,denominator=2*n,lower_numerator=lo,upper_numerator=hi,canonical_numerator=canonical,
                lower=lo/(2*n) if n else None,upper=hi/(2*n) if n else None,
                canonical=canonical/(2*n) if n else None)


def replay(evidence):
    nframes=int(evidence["nframes"])
    truth=_pairs(evidence["gt"])
    if nframes<1 or not truth:
        raise ValueError("A positive clip length and nonempty GT event set are required.")
    groups={}
    for name,ids in evidence["target_ids"].items():
        groups[name]=set(ids)
        if len(groups[name])!=len(ids) or not groups[name].issubset(truth):
            raise ValueError("Target GT must be unique official event IDs.")
    g=_window_counts(truth,nframes)
    den=int(g.sum())
    if den!=30*len(truth):
        raise ValueError("Invalid GT partition.")
    arms={}
    for name,item in evidence["arms"].items():
        pred=_pairs(item["predictions"])
        p=_window_counts(pred,nframes)
        err=abs(g-p).sum(axis=0)
        pred_totals=(p.sum(axis=0)//30)
        gt_totals=(g.sum(axis=0)//30)
        arms[name]=dict(window_numerator=int(err.sum()),window_denominator=den,window=100*int(err.sum())/den,
                        direction_window_numerators=err.tolist(),pred_totals=pred_totals.tolist(),
                        gt_totals=gt_totals.tolist(),bias=(pred_totals-gt_totals).tolist())
        for k in THRESHOLDS:
            arms[name][k]=_graph_scores(truth,pred,groups,item["edges"+k])
    if "native" not in arms:
        raise ValueError("Native P0 required.")
    comparisons={}
    for prefix,group in (("nearall","all"),("near","paired"),("far","paired")):
        keys={prefix+"_"+phase:w for phase,w in PHASE.items()}
        present=set(keys)&arms.keys()
        if present and present!=set(keys):
            raise ValueError("Incomplete two-phase group "+prefix)
        if present:
            comparisons[prefix]={k:contrast(arms,keys,group,k) for k in THRESHOLDS}
    if ("near" in comparisons) != ("far" in comparisons):
        raise ValueError("Matched near and far groups must appear together.")
    if "near" in comparisons:
        diff={prefix+"_"+phase:sgn*w for prefix,sgn in (("near",1),("far",-1)) for phase,w in PHASE.items()}
        comparisons["near_minus_far"]={k:contrast(arms,diff,"paired",k) for k in THRESHOLDS}
    return dict(arms=arms,contrasts=comparisons)


def score_case(gt, arm_mots, polygon, nframes, all_ids, paired_ids=(), weak_ids=()):
    """Extract compact evidence from all original, competition-preserving MOTs of one hive/seed."""
    from tools.study104.event_overlap import score_arm
    groups={k:sorted(map(int,v)) for k,v in (("all",all_ids),("paired",paired_ids),("weak",weak_ids))}
    evidence=dict(nframes=int(nframes),target_ids=groups,arms={})
    for name,mot in arm_mots.items():
        graph={}
        for k,thr in THRESHOLDS.items():
            arm=score_arm(gt,mot,polygon,thr)
            tids=sorted([int(t),int(f) if arm.gt[t]=="in" else -int(f)] for t,f in arm.gt_frames.items())
            pids=sorted([int(p),int(f) if d=="in" else -int(f)] for p,(d,f,_) in arm.events.items())
            if "gt" in evidence and tids!=evidence["gt"]:
                raise ValueError("GT stream differs between arms.")
            if "predictions" in graph and graph["predictions"]!=pids:
                raise ValueError("Prediction stream differs between IoU thresholds.")
            evidence["gt"]=tids
            graph["predictions"]=pids
            graph["edges"+k]=[[int(p),int(g)] for p,g in arm.edges]
        evidence["arms"][name]=graph
    return dict(evidence=evidence,results=replay(evidence))


def seed_summary(per_seed, label, iou="05"):
    """Average over qualified seeds with equal seed weights; excluded observed rows remain available."""
    rows={str(s):r["contrasts"][label][iou] for s,r in per_seed.items() if label in r["contrasts"]}
    eligible=[r for r in rows.values() if r["n"]>=10]
    return dict(observed=rows,qualified_count=len(eligible),coverage_pass=len(eligible)>=7,
                mean_lower=float(np.mean([r["lower"] for r in eligible])) if eligible else None,
                mean_upper=float(np.mean([r["upper"] for r in eligible])) if eligible else None,
                mean_canonical=float(np.mean([r["canonical"] for r in eligible])) if eligible else None)


def _read(path):
    data=Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith(".gz") else data)


def _write(path,record):
    data=(json.dumps(record,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    if path.suffix==".gz":
        data=gzip.compress(data,mtime=0)
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        part=Path(tmp)/"out"
        part.write_bytes(data)
        try:
            os.link(part,path)
        except FileExistsError:
            if path.read_bytes()!=data:
                raise ValueError("A different result exists at the output path.")
    return hashlib.sha256(data).hexdigest()


def _frozen_ids(frozen):
    # Each class chooses its own representative side/phase for the same GT event.
    return tuple([r["gt_id"] for r in frozen[name]]
                 for name in ("rows_all","rows","natural_weak_rows"))


def self_check():
    # A fixed target can exchange places with an untreated GT without changing global TP.
    gt=[[1,1],[2,2],[3,-750]]
    def item(edges):
        return dict(predictions=[[7,2],[8,-750]],edges05=edges,edges03=edges)
    arms={"native":item([[7,1],[7,2],[8,3]])}
    for prefix in ("nearall","near","far"):
        for phase in PHASE:
            yes=phase.startswith("S") and prefix!="far"
            arms[prefix+"_"+phase]=item([[7,1 if yes else 2],[8,3]])
    frozen=dict(rows_all=[{"gt_id":1}],rows=[{"gt_id":1}],
                natural_rows=[{"gt_id":2,"weak":False,"side":"origin"}],
                natural_weak_rows=[{"gt_id":2,"weak":True,"side":"destination"}])
    all_ids,paired_ids,weak_ids=_frozen_ids(frozen)
    assert (all_ids,paired_ids,weak_ids)==([1],[1],[2])
    e=dict(gt=gt,nframes=751,arms=arms,target_ids=dict(all=all_ids,paired=paired_ids,weak=weak_ids))
    result=replay(e)
    native=result["arms"]["native"]["05"]
    assert native["tp"]==2
    assert (native["target_bounds"]["all"]["low"],native["target_bounds"]["all"]["high"])==(0,1)
    assert result["contrasts"]["nearall"]["05"]["lower"]==result["contrasts"]["nearall"]["05"]["upper"]==1
    assert result["contrasts"]["near_minus_far"]["03"]["lower"]==1
    assert not result["arms"]["nearall_B0"]["05"]["target_bounds"]["weak"]["always_unmatched_ids"]
    assert result["arms"]["nearall_S0"]["05"]["target_bounds"]["weak"]["always_unmatched_ids"]==[2]
    assert seed_summary({0:result},"nearall")["qualified_count"]==0
    assert replay(dict(e,arms={"native":arms["native"]}))["contrasts"]=={}
    print("Synthetic tie, cohort, phase contrast, spatial control and coverage checks passed.")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="cmd",required=True)
    sub.add_parser("self-check")
    make=sub.add_parser("score")
    make.add_argument("manifest",type=Path)
    make.add_argument("output",type=Path)
    rep=sub.add_parser("replay")
    rep.add_argument("package",type=Path)
    args=ap.parse_args()
    if args.cmd=="self-check":
        self_check()
    elif args.cmd=="score":
        from beetrack.evaluate import load_tracks
        m=_read(args.manifest)
        frozen=_read(m["freeze_path"])
        if hashlib.sha256(Path(m["freeze_path"]).read_bytes()).hexdigest()!=m["freeze_sha256"]:
            raise ValueError("Frozen cohort digest differs.")
        package=score_case(load_tracks(m["gt_path"]),{k:load_tracks(v) for k,v in m["arm_paths"].items()},
                           m["polygon"],m["nframes"],*_frozen_ids(frozen))
        package["freeze_sha256"]=m["freeze_sha256"]
        print("Package SHA-256",_write(args.output,package))
    else:
        package=_read(args.package)
        if replay(package["evidence"])!=package["results"]:
            raise ValueError("Replay differs from the recorded impact results.")
        print("Event-window scores and target-GT tie bounds verified.")


if __name__=="__main__":
    main()
