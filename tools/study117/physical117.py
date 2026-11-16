"""Verify §117 whole physical MOT files against the original camera, full GT and 30 window origins."""
import argparse
from fractions import Fraction
import gzip
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from beetrack.evaluate import by_track,frame_match,match_events
from tools.study104.event_overlap import score_arm


ARMS=("U","A","Z","Q","Zm","D")


def require(ok,message):
    if not ok:
        raise ValueError(message)


def wire(x):
    return (json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def digest(raw):
    return sha256(raw).hexdigest()


def save(path,obj):
    raw=wire(obj)
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        f=Path(tmp)/"record"
        f.write_bytes(raw)
        try:
            os.link(f,path)
        except FileExistsError:
            require(path.read_bytes()==raw,"Immutable verification receipt differs")
    return digest(raw)


def load_mot(raw):
    return np.loadtxt(BytesIO(raw),delimiter=",",ndmin=2)[:,:6] if raw.strip() else np.empty((0,6))


def segment_events(rows,polygon,nframes):
    """Get signed events with all same-direction row pairs, after official gap-splitting."""
    require(np.isfinite(rows).all() and np.all(rows[:,:2]==np.rint(rows[:,:2])) and
            np.all((rows[:,0]>=1)&(rows[:,0]<=nframes)),"Invalid native times or IDs")
    ordered=by_track(rows)
    require(len(np.unique(ordered[:,:2],axis=0))==len(ordered),"Duplicate resolved ID at one frame")
    centers=ordered[:,2:4]+ordered[:,4:6]/2
    poly=np.array(polygon,dtype=np.float32)
    inside=np.array([cv2.pointPolygonTest(poly,(float(x),float(y)),False)>=0 for x,y in centers],dtype=bool)
    tids,starts=np.unique(ordered[:,1],return_index=True)
    events={}
    for tid,a,b in zip(tids,starts,[*starts[1:],len(ordered)]):
        flags=inside[a:b]
        if flags[0]==flags[-1]:
            continue
        direction=1 if flags[-1] else -1
        right=a+1+np.flatnonzero((flags[1:]!=flags[:-1])&(flags[1:]==flags[-1]))
        f,_,x,y,w,h=ordered[right[-1],:6]
        events[int(tid)]=(direction*int(f),[float(x),float(y),float(x+w),float(y+h)],right)
    return ordered,events


def grid(events,nframes):
    times=[event[0] for event in events.values()]
    grids=[]
    for phase in range(0,750,25):
        starts=range(-phase,nframes,750)
        cells=[]
        for start in starts:
            lo,hi=max(1,start+1),min(nframes,start+750)
            cells.append([sum(lo<=f<=hi for f in times),sum(lo<=-f<=hi for f in times)])
        grids.append(cells)
    return grids


def assigned_edges(gt,pred,G,P,threshold):
    owner=frame_match(gt,pred,thr=threshold) if len(gt) and len(pred) else np.full(len(pred),-1)
    edges=[]
    for pid,(pf,_,candidates) in P.items():
        compatible=set()
        for j in candidates:
            t=pred[j,0]
            for gid in (int(owner[j-1]),int(owner[j])):
                if gid not in G or G[gid][0]*pf<=0:
                    continue
                if np.any(np.abs(gt[G[gid][2],0]-t)<=5):
                    compatible.add(gid)
        edges.extend([int(pid),g] for g in sorted(compatible))
    return edges


def metrics(G,P,edges):
    rows=list(P); cols=list(G)
    g_col={k:i for i,k in enumerate(cols)}; p_row={k:i for i,k in enumerate(rows)}
    matrix=np.zeros((len(rows),len(cols)),dtype=bool)
    for p,g in edges:
        matrix[p_row[p],g_col[g]]=True
    ri,cj=linear_sum_assignment(matrix.astype(int),maximize=True)
    matches=[[rows[a],cols[b]] for a,b in zip(ri,cj) if matrix[a,b]]
    if len(rows) and len(cols):
        independent=maximum_bipartite_matching(csr_matrix(matrix),perm_type="column")
        require(len(matches)==np.count_nonzero(independent>=0),"Sparse maximum and official Hungarian disagree")
    counts=lambda tp,npred,ng:dict(tp=tp,fp=npred-tp,fn=ng-tp,f1=2*tp/max(1,npred+ng))
    out=counts(len(matches),len(P),len(G)); bydir={}
    for d,sgn in (("in",1),("out",-1)):
        bydir[d]=counts(sum(G[g][0]*sgn>0 for _,g in matches),
                        sum(e[0]*sgn>0 for e in P.values()),sum(e[0]*sgn>0 for e in G.values()))
    out.update(matching=matches,by_direction=bydir)
    return out


def verify_mot(raw,gt,poly,nframes):
    mot=load_mot(raw)
    truth,G=segment_events(gt,poly,nframes)
    native,P=segment_events(mot,poly,nframes)
    g_grid,p_grid=grid(G,nframes),grid(P,nframes)
    absolute=[int(np.abs(np.array(a)-np.array(b)).sum()) for a,b in zip(p_grid,g_grid)]
    denom=len(G)*30
    proof=dict(gtruth=g_grid,pred=p_grid,l1_by_offset=absolute,numerator=sum(absolute),denominator=denom)
    statuses={}
    result=dict(physical_mot_sha256=digest(raw),rows=len(mot),gt=[[g,e[0]] for g,e in G.items()],
                predictions=[[p,e[0],*e[1]] for p,e in P.items()],window=proof,thresholds=statuses,
                w30_percent=100*sum(absolute)/denom if denom else None)
    for threshold in (.5,.3):
        edges=assigned_edges(truth,native,G,P,threshold)
        m=metrics(G,P,edges)
        official=score_arm(gt,mot,poly,threshold)
        require(official.edges==edges and m["matching"]==[[p,g] for p,g in sorted(official.matches.items())],
                "Eligible edge graph or maximum-Matching identity differs from official full-MOT output")
        if threshold==.5:
            primary=match_events(gt,mot,poly)
            require((primary["gt"],primary["pred"],primary["tp"])==(len(G),len(P),m["tp"]),
                    "Original event-level scorer differs")
        statuses[str(threshold)]=dict(**m,edges=edges)
    return result


def assert_score(ours,theirs):
    require(ours["physical_mot_sha256"]==theirs["mot_sha256"],"MOT hash differs")
    a,b=ours["w30_percent"],theirs["w30_percent"]
    require(a==b or (a is not None and b is not None and abs(a-b)<1e-10),"Window error differs")
    for k in ("numerator","denominator"):
        require(ours["window"][k]==theirs["window"][k],"Window fraction differs")
    truth=ours["window"]["gtruth"]; pred=ours["window"]["pred"]
    expect=[[int(x) for x in np.abs(np.asarray(g)-np.asarray(p)).sum(axis=0)] for g,p in zip(truth,pred)]
    require(theirs["window"]["l1_by_offset_in_out"]==expect and theirs["window"]["bins_per_offset"]==list(map(len,truth))
            and theirs["window"]["offsets_frames"]==list(range(0,750,25)),"Per-phase/per-direction window table differs")
    for d,i in (("in",0),("out",1)):
        a=sum(r[i] for r in truth[0]); b=sum(r[i] for r in pred[0])
        require((theirs["gt_counts"][d],theirs["pred_counts"][d],theirs["bias"][d])==(a,b,b-a),
                "Whole-clip count or signed bias differs")
    evidence=theirs["evidence"]
    for k in ("gt","predictions"):
        np.testing.assert_allclose(ours[k],evidence[k],rtol=0,atol=1e-10)
    for tau,k in (("0.5","edges05"),("0.3","edges03")):
        m=ours["thresholds"][tau]; main=theirs["iou"][tau]
        require(m["edges"]==evidence[k],"Primary event edge graph differs")
        require(all(m[c]==main[c] for c in ("tp","fp","fn","matching")) and
                abs(m["f1"]-main["f1"])<1e-12,"Primary official scores differ")
        for d in ("in","out"):
            require(all(main["by_direction"][d][c]==m["by_direction"][d][c] for c in ("tp","fp","fn")),
                    "Directional matching differs")
            require(abs(main["by_direction"][d]["f1"]-m["by_direction"][d]["f1"])<1e-12,"Directional F1 differs")


def read_checked(path,expected):
    raw=path.read_bytes()
    require(digest(raw)==expected,f"Changed physical input: {path}")
    return raw


def verify_run(root,repo,prereg,index_sha):
    directory=root/"tracking"
    index=json.loads(read_checked(directory/"index.json",index_sha))
    expected={(f,s,a) for f in (33,35) for s in range(3) for a in ARMS}
    require(index["stage"]=="predictions_only" and index["prereg"]==prereg and len(index["cases"])==36 and
            {(c["fold"],c["seed"],c["arm"]) for c in index["cases"]}==expected,"Incomplete or wrong MOT freeze")
    zones=json.loads(read_checked(repo/"beetrack/zones.json",index["zone_sha256"]))
    public=read_checked(repo/"docs/study108_source.json.gz",
                        "2a7d8bf2d4b027dafa59d924e5b908ff2b88c0edbdf9fcded097c60383edd383")
    history=json.loads(gzip.decompress(public))
    summary={}; truths={}
    for c in index["cases"]:
        fold,seed,arm=c["fold"],c["seed"],c["arm"]
        clip=f"BEE24-{68-fold}-tail"
        name=f"{clip}/{fold}_{seed}_{arm}"
        require(c["clip"]==clip and c["mot_file"]==f"mot/{name}.txt" and c["file"]==f"counts/{name}.json",
                "MOT manifest refers to an unexpected source or naming convention")
        raw=read_checked(directory/c["mot_file"],c["mot_sha256"])
        counts=json.loads(read_checked(directory/c["file"],c["sha256"]))
        historical=history["labels"][f"{fold}_{seed}_tail"]
        gt_sha=historical["gt_mot_sha256"]
        if clip not in truths:
            truths[clip]=load_mot(read_checked(repo/"runs/gt/val"/clip/"gt/gt.txt",gt_sha)),gt_sha
        gt,current_sha=truths[clip]
        require(current_sha==gt_sha and counts["detector_index_sha256"]==index["detector_index_sha256"]
                and counts["mot_sha256"]==c["mot_sha256"],"Conflicting native provenance")
        score=verify_mot(raw,gt,zones[clip],1000)
        require(score["gt"]==historical["gt"],"Native GT descriptor stream differs from historical source truth")
        signed=[[i,f,*[round(x,2) for x in box]] for i,f,*box in score["predictions"]]
        require(signed==counts["predictions"],"Native freeze signed event or original box differs")
        require(len(counts["windows"])==30,"Frozen predicted bins omit a window phase")
        for actual,recorded in zip(score["window"]["pred"],counts["windows"]):
            require(actual==recorded[:len(actual)] and not np.any(recorded[len(actual):]),"Frozen predicted bins differ")
        score.update(prereg=prereg,fold=fold,seed=seed,arm=arm,clip=clip,gt_sha256=gt_sha,tracking_index_sha256=index_sha)
        summary[name]=save(root/"validation/physical"/(name+".json"),score)
        print(name,score["thresholds"]["0.5"]["tp"],score["window"]["numerator"],flush=True)
    checksum=save(root/"validation/physical_index.json",dict(study=117,prereg=prereg,tracking_sha256=index_sha,cases=summary))
    print("Whole physical MOTs checked",checksum,flush=True)


def gates(records):
    eps=Fraction(1,10**9)
    output={}
    for fold in (33,35):
        group={(r["seed"],r["arm"]):r for r in records if r["fold"]==fold}
        costs={k:Fraction(100*r["window"]["numerator"],r["window"]["denominator"]) for k,r in group.items()}
        event={k:r["thresholds"]["0.5"] for k,r in group.items()}
        exact_f1={k:Fraction(2*m["tp"],max(1,2*m["tp"]+m["fp"]+m["fn"])) for k,m in event.items()}
        avg=lambda a,table:sum(table[s,a] for s in range(3))/3
        wins=sum(costs[s,"U"]-costs[s,"A"]>eps for s in range(3))
        u=avg("U",costs)-avg("A",costs)
        f=avg("A",exact_f1)-avg("U",exact_f1)
        complexity=all(avg(a,costs)-avg("A",costs)>eps and avg("A",exact_f1)-avg(a,exact_f1)>=-eps for a in ("Z","Q"))
        output[str(fold)]=dict(mean_w_gain_pp=float(u),f1_gain=float(f),wins=wins,utility=(wins>=2 and u>=2 and f>=-eps),
                               complexity=complexity)
    return output


def compare_run(root,primary_sha):
    index=json.loads(read_checked(root/"scores/index.json",primary_sha))
    independent=json.loads((root/"validation/physical_index.json").read_bytes())
    require(index["tracking_index_sha256"]==independent["tracking_sha256"] and index["prereg"]==independent["prereg"],
            "Scoring and validation differ in provenance")
    records=[]
    for c in index["cases"]:
        pri=json.loads(read_checked(root/"scores"/c["file"],c["sha256"]))
        name=f'{pri["clip"]}/{pri["fold"]}_{pri["seed"]}_{pri["arm"]}'
        ours=json.loads(read_checked(root/"validation/physical"/(name+".json"),independent["cases"][name]))
        assert_score(ours,pri)
        records.append(ours)
    require(len(records)==36 and
            {(r["fold"],r["seed"],r["arm"]) for r in records}=={(f,s,a) for f in (33,35) for s in range(3) for a in ARMS},
            "Not all distinct 36 cases scored")
    gate=gates(records)
    primary_summary=json.loads(read_checked(root/"scores"/index["summary_file"],index["summary_sha256"]))
    require(primary_summary["tracking_index_sha256"]==independent["tracking_sha256"] and
            primary_summary["prereg"]==independent["prereg"],"Summary provenance differs")
    for fold,v in gate.items():
        other=primary_summary["folds"][fold]
        utility=other["utility"]
        require(v["utility"]==utility["passed"] and v["complexity"]==other["complexity"]["passed"]
                and v["wins"]==utility["positive_seeds"] and abs(v["mean_w_gain_pp"]-utility["mean_w_gain_pp"])<1e-10
                and abs(v["f1_gain"]-utility["mean_f1_delta"])<1e-12,"Primary fold statistics or decision differ")
    receipt=dict(study=117,prereg=independent["prereg"],scores_sha256=primary_sha,
                 physical_index_sha256=digest((root/"validation/physical_index.json").read_bytes()),
                 conditions=gate,cases=len(records),window_offsets=30*len(records),iou_graphs=2*len(records))
    print(save(root/"validation/physical_agreement.json",receipt),gate)


def self_check():
    poly=[[0,-10],[100,-10],[100,20],[0,20]]
    def make(tid,frames,positions,y=0):
        return [[f,tid,x,y,2,2] for f,x in zip(frames,positions)]
    # Net-in with earlier valid crossing, a zero-net-return neighbor and reused ID.
    gt=np.array(make(7,[1,2,3,4],[-3,1,-3,1])+make(8,[1,2,3],[-3,1,-3],6)
                +make(8,[29,30],[1,-3],6),float)
    pred=np.array(make(9,[1,2,3,4],[-3,1,-3,1])+make(11,[40,41],[1,-3],6),float)
    raw=("\n".join(",".join(map(str,r)) for r in pred)+"\n").encode()
    result=verify_mot(raw,gt,poly,1000)
    assert result["gt"]==[[7,4],[100008,-30]]
    assert result["thresholds"]["0.5"]["matching"]==[[9,7]]
    assert result["thresholds"]["0.5"]["fp"]==1 and result["window"]["denominator"]==60
    assert len(result["window"]["pred"])==30 and all(sum(a+b for a,b in origin)==2 for origin in result["window"]["pred"])
    assert result["w30_percent"]==100*sum(result["window"]["l1_by_offset"])/60
    empty=verify_mot(b"",np.empty((0,6)),poly,1000)
    assert empty["w30_percent"] is None and empty["thresholds"]["0.3"]["f1"]==0
    print("Synthetic physical whole-MOT ownership, sparse maximum matching, gap-25 split and per-offset count replay passed.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="stage",required=True)
    sub.add_parser("self-check")
    v=sub.add_parser("verify")
    v.add_argument("root",type=Path);v.add_argument("repo",type=Path)
    v.add_argument("--prereg",required=True);v.add_argument("--tracking-sha",required=True)
    c=sub.add_parser("compare");c.add_argument("root",type=Path);c.add_argument("--scores-sha",required=True)
    args=parser.parse_args()
    if args.stage=="self-check":
        self_check()
    elif args.stage=="verify":
        verify_run(args.root,args.repo,args.prereg,args.tracking_sha)
    else:
        compare_run(args.root,args.scores_sha)
