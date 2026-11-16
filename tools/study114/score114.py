"""Label confirmed crossings; learn and score the immediately following same-bee return."""
from bisect import bisect_right
from collections import Counter, defaultdict

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


METHODS=("distance","time","cv","native","model")
TOL=1e-9


def _overlaps(obs, used, gt, polygon):
    """Archive all same-frame GT candidates at the exclusion threshold; other GT boxes need not be published."""
    by_frame=defaultdict(list)
    for i,row in enumerate(gt):
        by_frame[int(row[0])].append(i)
    needed=defaultdict(list)
    for i in sorted(used):
        needed[int(obs[i,0])].append(i)
    recorded=[]
    xywh=np.asarray(obs[:,1:5],dtype=float)/100
    for f,indices in needed.items():
        truth=gt[by_frame.get(f,[])]
        if not len(truth):
            continue
        boxes=xywh[indices]
        gtboxes=truth[:,2:6]
        lo=np.maximum(boxes[:,None,:2],gtboxes[None,:,:2])
        hi=np.minimum((boxes[:,:2]+boxes[:,2:4])[:,None,:],(gtboxes[:,:2]+gtboxes[:,2:4])[None,:,:])
        intersection=np.maximum(0,hi-lo).prod(axis=2)
        union=boxes[:,2:4].prod(axis=1)[:,None]+gtboxes[:,2:4].prod(axis=1)[None,:]-intersection
        iou=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
        gt_centers=gtboxes[:,:2]+gtboxes[:,2:4]/2
        gt_side=[int(cv2.pointPolygonTest(polygon,tuple(p),False)>=0) for p in gt_centers]
        for j in np.flatnonzero((iou>=.3).any(axis=1)):
            options=[[int(truth[k,1]),float(iou[j,k]),gt_side[k]] for k in np.flatnonzero(iou[j]>=.3)]
            recorded.append([indices[j],sorted(options)])
    return sorted(recorded)


def geometry_labels(case, gt_rows, polygon):
    from beetrack.evaluate import by_track,zone_tracks
    gt=by_track(gt_rows)
    poly=np.asarray(polygon,np.float32)
    turns=[]
    for gid,(_,_,_,changes) in zone_tracks(gt,poly).items():
        frames=[int(gt[i,0]) for i,_ in changes]
        gid=int(gid)
        if len(frames)!=len(set(frames)):
            raise ValueError("Multiple annotated crossings for one GT ID at one frame.")
        for j,(i,d) in enumerate(changes):
            turns.append([gid,frames[j],1 if d=="in" else -1])
    obs=np.asarray([row[:5] for row in case["observations"]],dtype=np.int64).reshape(-1,5)
    used={int(i) for c in case["candidates"] for i in c[:2]}
    return dict(polygon=poly.tolist(),turns=sorted(turns),overlaps=_overlaps(obs,used,gt,poly))


def label_case(case, gt_rows, polygon):
    return label_from_proof(case,geometry_labels(case,gt_rows,polygon))


def label_from_proof(case, proof):
    times=defaultdict(list)
    turns={}
    by_gid=defaultdict(list)
    for gid,f,d in proof["turns"]:
        by_gid[gid].append((f,d))
    for gid,entries in by_gid.items():
        entries.sort()
        times[gid]=[f for f,_ in entries]
        for j,(f,d) in enumerate(entries):
            if (gid,f) in turns or d not in (1,-1):
                raise ValueError("Repeated frame or invalid direction in GT turn ledger.")
            turns[gid,f]=(d,entries[j+1][0] if j+1<len(entries) else None)
    obs=np.asarray([row[:5] for row in case["observations"]],dtype=np.int64).reshape(-1,5)
    candidates=case["candidates"]
    ends=np.asarray([row[:3] for row in candidates],dtype=np.int64).reshape(-1,3)
    left,right,dirs=ends.T
    u,t=obs[left,0],obs[right,0]
    if np.any(t<=u) or not np.all(np.isin(dirs,[-1,1])):
        raise ValueError("Invalid predicted crossing interval or direction.")
    xywh=np.asarray(obs[:,1:5],dtype=float)/100
    centers=xywh[:,:2]+xywh[:,2:4]/2
    areas=xywh[:,2:4].prod(axis=1)
    poly=np.asarray(proof["polygon"],np.float32)
    owners=np.full(len(obs),-1,dtype=np.int64)
    for i,overlap in proof["overlaps"]:
        if len(overlap)==1:
            gid,iou,side=overlap[0]
            if iou>=.5 and (cv2.pointPolygonTest(poly,tuple(centers[i]),False)>=0)==bool(side):
                owners[i]=gid
    if np.any(areas[np.unique(ends[:,:2])]<=0):
        raise ValueError("A candidate endpoint has nonpositive area.")
    keys={}
    reps={}
    for i,(a,b,d) in enumerate(ends):
        gid=int(owners[a])
        if gid<0 or gid!=owners[b]:
            continue
        xs=times[gid]
        j=bisect_right(xs,int(u[i]))
        if j>=len(xs) or xs[j]>t[i] or (j+1<len(xs) and xs[j+1]<=t[i]):
            continue
        key=gid,xs[j]
        if turns[key][0]!=d:
            continue
        keys[i]=key
        rank=int(t[i]-u[i]),int(t[i]),int(u[i]),i
        if key not in reps or rank<reps[key][0]:
            reps[key]=rank,i
    sided={}
    for d in (-1,1):
        pos=np.flatnonzero(dirs==d)
        order=np.argsort(t[pos],kind="stable")
        sided[d]=pos[order]
    groups=[]
    for (gid,frame),(_,anchor) in sorted(reps.items()):
        d,nxt=turns[(gid,frame)]
        ids=sided[-d]
        clock=t[ids]
        sl=np.searchsorted(clock,[t[anchor]+1,t[anchor]+51])
        ids=ids[sl[0]:sl[1]]
        ids=ids[u[ids]>=t[anchor]]
        start,end=left[anchor],right[anchor]
        middle=left[ids]
        dt=t[ids]-t[anchor]
        dx=centers[middle]-centers[end]
        unit=(np.sqrt(areas[middle])+np.sqrt(areas[end]))/2
        dist=np.linalg.norm(dx,axis=1)/unit
        valid=dist<=4
        ids,dt,middle,dx,unit,dist=[x[valid] for x in (ids,dt,middle,dx,unit,dist)]
        velocity=(centers[end]-centers[start])/(t[anchor]-u[anchor])
        residual=np.linalg.norm(dx-(u[ids]-t[anchor])[:,None]*velocity,axis=1)/unit
        features=np.column_stack((dist,dt/50,np.abs(np.log(areas[middle]/areas[end])),residual)).clip(0,10)
        pool={}
        for j,idx in enumerate(ids):
            i=int(idx)
            key=keys.get(i)
            if key is None:
                continue
            rank=(float(dist[j]),int(dt[j]),int(t[i]-u[i]),int(t[i]),int(u[i]),i)
            if key not in pool or rank<pool[key][0]:
                pool[key]=(rank,i,features[j])
        options=[]
        later=0
        for (other,f),(rank,i,x) in sorted(pool.items()):
            if other==gid and f!=nxt:
                later+=1
                continue
            options.append(dict(candidate=i,gt_id=other,frame=f,positive=(other==gid),
                                dt=rank[1],features=x.tolist(),
                                native=int(bool(set(candidates[anchor][3])&set(candidates[i][3])))))
        pos=any(v["positive"] for v in options)
        neg=any(not v["positive"] for v in options)
        status="mixed" if pos and neg else "positive_only" if pos else "negative_only" if neg else "neither"
        timing="ending_or_censored" if nxt is None else ("outside_50" if nxt-frame>50 else "inside_50")
        groups.append(dict(gt_id=gid,frame=frame,direction=d,next_frame=nxt,candidate=anchor,
                           gt_timing=timing,has_positive=pos,pool_status=status,options=options,
                           unknown_raw_pool_candidates=sum(int(i) not in keys for i in ids),
                           later_same_gt_turns_excluded=later,raw_pool_candidates=len(ids)))
    coverage=Counter(group["pool_status"] for group in groups)
    timing=Counter(group["gt_timing"] for group in groups)
    counts=dict(gt_crossings=len(turns),gt_crossing_bees=len({g for g,f in turns}),labeled_anchors=len(groups),
                labeled_anchor_bees=len({g["gt_id"] for g in groups}),labeled_candidate_rows=len(keys),
                total_candidate_rows=len(candidates),ending_or_censored=timing["ending_or_censored"],
                labeled_with_next=sum(g["next_frame"] is not None for g in groups),
                all_true_with_next=sum(nxt is not None for d,nxt in turns.values()),
                pool_has_positive=coverage["positive_only"]+coverage["mixed"],
                pool_has_different_gt=coverage["negative_only"]+coverage["mixed"],
                mixed_origin_bees=len({g["gt_id"] for g in groups if g["pool_status"]=="mixed"}),
                next_outside_50=timing["outside_50"],next_inside_50=timing["inside_50"],
                inside_50_missing_pool=sum(g["gt_timing"]=="inside_50" and not g["has_positive"] for g in groups),
                outside_50_found_in_pool=sum(g["gt_timing"]=="outside_50" and g["has_positive"] for g in groups),
                observed_next_missing_pool=sum(g["next_frame"] is not None and not g["has_positive"] for g in groups),
                **{k:coverage[k] for k in ("mixed","positive_only","negative_only","neither")})
    # Limits refer to observed CANDIDATE arrival frames; GT time lags are reported separately.
    return dict(counts=counts,turns=[[g,f,d,nxt] for (g,f),(d,nxt) in sorted(turns.items())],
                candidate_labels=[[i,g,f] for i,(g,f) in sorted(keys.items())],groups=groups)


def fit_ranker(head):
    mixed=[g for g in head["groups"] if g["pool_status"]=="mixed"]
    n=len({g["gt_id"] for g in mixed})
    report=dict(distinct_origin_gt=n,mixed_groups=len(mixed),coefficients=None)
    if n<10:
        return report
    parts,weights=[],[]
    for g in mixed:
        pos,=[o for o in g["options"] if o["positive"]]
        neg=[o["features"] for o in g["options"] if not o["positive"]]
        parts.append(np.asarray(pos["features"])-np.asarray(neg))
        weights.extend([1/(len(mixed)*len(neg))]*len(neg))
    design=np.vstack(parts)
    w=np.array(weights)
    def objective(beta):
        z=design@beta
        return float(w@np.logaddexp(0,-z)+.5*(beta@beta)),beta-design.T@(w*expit(-z))
    result=minimize(objective,np.zeros(4),jac=True,method="L-BFGS-B",
                    options={"ftol":1e-15,"gtol":1e-10,"maxiter":500})
    if not result.success or np.linalg.norm(result.jac,ord=np.inf)>1e-7:
        raise RuntimeError("Regularized source fit did not converge.")
    report.update(coefficients=result.x.tolist(),loss=float(result.fun),
                  grad_inf=float(np.linalg.norm(result.jac,ord=np.inf)),pairs=len(w))
    return report


def _aggregate(rows):
    if not rows:
        return dict(groups=0,bees=0,equal_group=None,equal_bee=None)
    by_bee=defaultdict(list)
    for gid,value in rows:
        by_bee[gid].append(value)
    return dict(groups=len(rows),bees=len(by_bee),equal_group=float(np.mean([v for _,v in rows])),
                equal_bee=float(np.mean([np.mean(vals) for vals in by_bee.values()])))


def score_ranks(labeled, coefficients=None):
    fitted=coefficients is not None
    beta=np.asarray(coefficients,float) if fitted else None
    if fitted and (beta.shape!=(4,) or not np.all(np.isfinite(beta))):
        raise ValueError("Expected four frozen finite coefficients.")
    methods=METHODS if fitted else METHODS[:-1]
    scored=[]
    for g in labeled["groups"]:
        if g["pool_status"]!="mixed":
            continue
        pos,=[v for v in g["options"] if v["positive"]]
        neg=[v for v in g["options"] if not v["positive"]]
        def scores(v):
            x=v["features"]
            values=dict(distance=-x[0],time=-v["dt"],cv=-x[3],native=v["native"])
            if fitted:
                values["model"]=float(beta@x)
            return values
        positives=scores(pos)
        negatives=[scores(v) for v in neg]
        comparisons={}
        for subset,indices in (("all",list(range(len(neg)))),
                               ("same_dt",[j for j,v in enumerate(neg) if v["dt"]==pos["dt"]])):
            comparison={}
            for method in methods:
                if indices:
                    j=max(indices,key=lambda j:(negatives[j][method],-neg[j]["candidate"]))
                    diff=positives[method]-negatives[j][method]
                    comparison[method]=dict(credit=1. if diff>TOL else 0. if diff < -TOL else .5,
                                           delta=float(diff),hardest_candidate=neg[j]["candidate"])
            comparisons[subset]=comparison
        scored.append(dict(gt_id=g["gt_id"],frame=g["frame"],candidate=g["candidate"],
                           positive_candidate=pos["candidate"],negative_turns=len(neg),
                           blind_choice=1/(1+len(neg)),comparisons=comparisons))
    summary={}
    for scope in ("all","same_dt"):
        summary[scope]={m:_aggregate([(g["gt_id"],g["comparisons"][scope][m]["credit"])
                            for g in scored if m in g["comparisons"][scope]]) for m in methods}
    return dict(counts=labeled["counts"],summary=summary,mixed_groups=len(scored),
                mixed_origin_bees=len({r["gt_id"] for r in scored}),group_ranks=scored,
                chance_uniform_selection=_aggregate([(r["gt_id"],r["blind_choice"]) for r in scored]))


def seed_gate(head_scores,tail_scores):
    enough=head_scores["mixed_origin_bees"]>=10 and tail_scores["mixed_origin_bees"]>=10
    methods=tail_scores["summary"]["all"]
    score=methods.get("model",{}).get("equal_bee")
    distance=methods["distance"]["equal_bee"]
    gain=score-distance if score is not None else None
    return dict(head_bees=head_scores["mixed_origin_bees"],tail_bees=tail_scores["mixed_origin_bees"],
                sufficient=enough,model=score,distance=distance,gain=gain,
                passes=bool(enough and score is not None and score>=.70 and gain>=.10))


def fold_gate(seed_gates):
    if len(seed_gates)!=3 or set(map(str,seed_gates))!={"0","1","2"}:
        raise ValueError("Expected the three matched detector seeds.")
    flags=[bool((seed_gates[s] if s in seed_gates else seed_gates[str(s)])["passes"]) for s in range(3)]
    return dict(seed_pass=flags,n_pass=sum(flags),passes=sum(flags)>=2)


def self_check():
    poly=[[10,0],[30,0],[30,30],[10,30]]
    gt,obs,idx=[],[],{}
    for gid,y,locations in ((1,10,[(1,8),(2,11),(3,12),(4,8),(5,8),(6,11),(7,8)]),
                            (2,15,[(2,12),(3,12),(4,8)])):
        for f,x in locations:
            idx[gid,f]=len(obs)
            gt.append([f,gid,x-1,y-1,2,2])
            obs.append([f,100*(x-1),100*(y-1),200,200,[]])
    def c(g1,f1,g2,f2,d,ids):
        return [idx[g1,f1],idx[g2,f2],d,ids,[]]
    candidates=[c(1,1,1,2,1,[1]),c(1,1,1,3,1,[]),c(1,3,1,4,-1,[1]),c(1,2,1,4,-1,[1]),
                c(2,3,2,4,-1,[2]),c(1,6,1,7,-1,[1]),c(1,2,1,7,-1,[]),c(1,2,2,4,-1,[])]
    case=dict(observations=obs,candidates=candidates)
    proof=geometry_labels(case,np.array(gt,float),poly)
    labeled=label_from_proof(case,proof)
    labels={i:(g,f) for i,g,f in labeled["candidate_labels"]}
    assert 6 not in labels and 7 not in labels and labels[2]==labels[3]==(1,4)
    group=next(g for g in labeled["groups"] if (g["gt_id"],g["frame"])==(1,2))
    assert group["candidate"]==0 and group["gt_timing"]=="inside_50" and group["pool_status"]=="mixed"
    assert group["later_same_gt_turns_excluded"]==1 and group["unknown_raw_pool_candidates"]==2
    assert [(v["candidate"],v["positive"]) for v in group["options"]]==[(3,True),(4,False)]
    assert group["options"][0]["features"]==[0,2/50,0,0]
    report=score_ranks(labeled,None)
    assert report["summary"]["same_dt"]["time"]["equal_bee"]==.5
    assert fit_ranker(labeled)["coefficients"] is None
    # Marginal competing true box at IoU=.3 disqualifies the high-IoU owner.
    ambiguous=dict(proof,overlaps=[[i,edges+([[99,.3,0]] if i==idx[1,1] else [])]
                                    for i,edges in proof["overlaps"]])
    assert 0 not in {i for i,g,f in label_from_proof(case,ambiguous)["candidate_labels"]}
    repeated=dict(counts={},groups=[dict(group,gt_id=k) for k in range(11)])
    fit=fit_ranker(repeated)
    assert len(fit["coefficients"])==4 and fit["grad_inf"]<1e-7
    out=score_ranks(repeated,fit["coefficients"])
    assert out["summary"]["all"]["model"]["equal_bee"]==1 and out["chance_uniform_selection"]["equal_bee"]==.5
    assert not seed_gate(out,out)["passes"]  # equal-best distance also ranks all returns first.
    assert not fold_gate({i:seed_gate(out,out) for i in range(3)})["passes"]
    empty=label_case(dict(observations=[],candidates=[]),np.array(gt,float),poly)
    assert empty["counts"]["gt_crossings"]==5 and not empty["groups"]
    print("Synthetic unique-owner, cutoff, multiple turns, pool deduplication and ridge/ranking controls passed.")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd",choices=["self-check"])
    p.parse_args()
    self_check()
