"""Independent source crossing identity, geometry and hardest-negative checks for §114."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.special import expit


def read_json(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if data.startswith(b"\x1f\x8b") else data)


def corrected_truth(prepared_mot, part):
    """The prepared dev head already received merge_handovers in beetrack/prep.py."""
    from beetrack.evaluate import by_track
    if part not in ("head", "tail"):
        raise ValueError("Expected head or tail")
    return by_track(np.asarray(prepared_mot, dtype=float))


def side_of(poly, x, y, w, h):
    return cv2.pointPolygonTest(poly, (float(x + w / 2), float(y + h / 2)), False) >= 0


def truth_index(corrected, polygon):
    """Build every change of side, including those on tracks that finish on their initial side."""
    poly = np.asarray(polygon, dtype=np.float32)
    frames, tracks = defaultdict(list), defaultdict(list)
    for frame, identity, x, y, w, h in np.asarray(corrected)[:, :6]:
        t, g = int(frame), int(identity)
        # Zero-width annotations retain their official track center, but their box IoU is zero.
        assert frame == t and identity == g and w >= 0 and h >= 0
        inside = side_of(poly, x, y, w, h)
        tracks[g].append((t,inside))
        frames[t].append((g,x,y,w,h,inside))
    changes, all_crossings = {}, {}
    for g, sample in tracks.items():
        rows = sorted(sample)
        assert len(rows)==len({t for t,_ in rows}), ("Repeated GT and frame",g)
        jumps = [(b,1 if new else -1) for (_,old),(b,new) in zip(rows,rows[1:]) if new != old]
        changes[g] = [t for t,_ in jumps]
        for j,(t,d) in enumerate(jumps):
            all_crossings[g,t] = (d,jumps[j+1][0] if j+1<len(jumps) else None)
    return frames, changes, all_crossings


def xywh_iou_table(boxes, truth):
    """Continuous xywh, without an inclusive-pixel adjustment."""
    a = np.asarray(boxes, dtype=np.float64)
    b = np.asarray(truth, dtype=np.float64)
    overlap = np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])
    inter = np.maximum(0,overlap).prod(axis=2)
    aa = a[:,2:].prod(axis=1)[:,None]
    bb = b[:,2:].prod(axis=1)[None,:]
    return inter, aa + bb - inter


def observation_evidence(observations, frames):
    """Return the count of >=.3 competitors and >=.5 GT IDs for every observation."""
    by_time = defaultdict(list)
    for i,row in enumerate(observations):
        by_time[row[0]].append(i)
    evidence = [(0,()) for _ in observations]
    for t,indices in by_time.items():
        gt = frames.get(t)
        if not gt:
            continue
        # Multiplication by 100 preserves exact boundaries for integer/centipixel ground truth.
        a = [observations[i][1:5] for i in indices]
        b = np.asarray([row[1:5] for row in gt],dtype=np.float64)*100
        inter, union = xywh_iou_table(a,b)
        legal, overlap = 2*inter >= union, 10*inter >= 3*union
        for i, both, count in zip(indices,legal,overlap.sum(axis=1)):
            evidence[i] = (int(count),tuple(int(gt[j][0]) for j in np.flatnonzero(both)))
    return evidence


def choose_next(scores, labels, tolerance=1e-9):
    """Compare the one confirmed positive against the best-scoring other identity."""
    assert len(labels)==len(scores) and labels.count(1)==1 and labels.count(0)>=1
    difference = scores[labels.index(1)] - max(s for s,y in zip(scores,labels) if y == 0)
    return 0.5 if abs(difference) <= tolerance else float(difference > 0)


def fit_differences(groups):
    """Fit via Newton updates with Hessian; each list contains one group's positive-minus-negative rows."""
    if not groups:
        raise ValueError("No eligible training group")
    diffs = np.vstack(groups)
    weights = np.concatenate([np.full(len(d), 1/(len(groups)*len(d))) for d in groups])
    beta = np.zeros(diffs.shape[1])
    for _ in range(100):
        p = expit(-diffs@beta)
        gradient = beta - diffs.T @ (weights*p)
        if max(abs(gradient)) < 2e-12:
            break
        curvature = (diffs.T*(weights*p*(1-p))) @ diffs + np.eye(len(beta))
        step = np.linalg.solve(curvature,gradient)
        objective = np.logaddexp(0, -diffs@beta)@weights + beta@beta/2
        for power in range(40):
            proposed = beta - step/(2**power)
            after = np.logaddexp(0,-diffs@proposed)@weights + proposed@proposed/2
            if after <= objective + 1e-16:
                beta = proposed
                break
        else:
            raise ValueError("Independent group-balanced Newton fit stalled")
    assert np.linalg.norm(beta - diffs.T@(weights*expit(-diffs@beta)),ord=np.inf)<1e-9
    return beta


def crossing_membership(candidates, observations, known, changes, crossings):
    """A candidate can name at most one directed physical crossing in (departure,arrival]."""
    hits = {}
    for i,(lhs,rhs,direction,_,_) in enumerate(candidates):
        a,b=known[lhs],known[rhs]
        if a[0]!=1 or b[0]!=1 or len(a[1])!=1 or a[1]!=b[1]:
            continue
        tid=a[1][0]
        f,t=observations[lhs][0],observations[rhs][0]
        times=changes[tid]
        lo,hi=bisect_right(times,f),bisect_right(times,t)
        if hi==lo+1 and crossings[tid,times[lo]][0]==direction:
            hits[i]=(tid,times[lo])
    return hits


def group_averages(groups):
    """Compare the main equal-bee result with a descriptive equal-anchor result."""
    by_bee=defaultdict(list)
    for identity,value in groups:
        by_bee[identity].append(float(value))
    if not by_bee:
        return None,None
    return (float(np.mean([np.mean(rows) for rows in by_bee.values()])),
            float(np.mean([v for rows in by_bee.values() for v in rows])))


def simple_check():
    polygon=[[0,0],[20,0],[20,20],[0,20]]
    # Four alternating sides for a single bee yield three ordered crossings, including its return.
    gt=[[1,17,-4,2,2,2],[2,17,2,2,2,2],[3,17,-4,2,2,2],[4,17,2,2,2,2],
        [1,55,2,8,2,2],[2,55,2,8,2,2],[3,55,-4,8,2,2]]
    lookup,changes,crossings=truth_index(gt,polygon)
    assert crossings=={(17,2):(1,3),(17,3):(-1,4),(17,4):(1,None),(55,3):(-1,None)}
    boxes=[[r[0],*(100*x for x in r[2:6]),[]] for r in gt]
    eid=observation_evidence(boxes,lookup)
    cand=[[0,1,1,[],[]],[0,3,1,[],[]],[1,2,-1,[],[]],[5,6,-1,[],[]],[0,5,1,[],[]]]
    assert crossing_membership(cand,boxes,eid,changes,crossings)=={0:(17,2),2:(17,3),3:(55,3)}
    # The four input groups include an exact tie, a unique hard negative, and unequal groups per bee.
    assert choose_next([.8,.2,.8],[1,0,0])==.5
    assert choose_next([.8,.2,.85],[1,0,0])==0
    a,b=group_averages([(7,0),(7,0),(10,1)])
    assert a==.5 and b==1/3
    # Duplicating every member of one competition should leave its loss weight unchanged.
    left=np.array([[1.,1.],[-1.,1.]])
    right=np.array([[2.,0.]])
    assert np.allclose(fit_differences([left,right]),fit_differences([np.tile(left,(3,1)),right]),atol=1e-12)
    case=dict(observations=boxes,candidates=cand)
    expected=corrected_proof(case,gt,polygon,"tail")
    ledger, owners, groups=labels_from_evidence(case,expected)
    assert owners=={0:(17,2),2:(17,3),3:(55,3)} and ledger==crossings
    assert groups[0]["pool_status"]=="mixed" and groups[0]["candidate"]==0
    winner=rank_results(groups,None)
    assert winner["chance_uniform_selection"]["equal_bee"]==.5
    assert winner["summary"]["all"]["distance"]["equal_bee"]==1
    assert winner["summary"]["all"]["time"]["equal_bee"]==.5
    assert winner["summary"]["same_dt"]["native"]["equal_bee"]==.5


def independently_group(case, assigned, turns):
    """Generate an anchor for each true turn and retain one representative per turn in its future pool."""
    obs=np.asarray([r[:5] for r in case["observations"]],dtype=float).reshape(-1,5)
    proposals=case["candidates"]
    ends=np.array([r[:3] for r in proposals],dtype=int).reshape(-1,3)
    times=obs[:,0].astype(int)
    depart,arrive=times[ends[:,0]],times[ends[:,1]]
    xywh=obs[:,1:]/100
    area=xywh[:,2]*xywh[:,3]
    centers=xywh[:,:2]+xywh[:,2:]/2
    eligible=defaultdict(list)
    for idx,(a,b,d) in enumerate(ends):
        assert d in (-1,1) and times[b]>times[a] and area[a]>0 and area[b]>0
        eligible[int(d),int(times[b])].append(idx)
    rep={}
    for i,truth in assigned.items():
        key=(arrive[i]-depart[i],arrive[i],depart[i],i)
        if truth not in rep or key<rep[truth][0]:
            rep[truth]=(key,i)
    groups=[]
    for (g,f),(_,i) in sorted(rep.items()):
        direction,next_time=turns[g,f]
        a,b=ends[i,:2]
        speed=(centers[b]-centers[a])/(arrive[i]-depart[i])
        ids=np.array([j for t in range(arrive[i]+1,arrive[i]+51) for j in eligible.get((-direction,t),())],dtype=int)
        subset=ids[depart[ids]>=arrive[i]]
        points=ends[subset,0]
        scales=(np.sqrt(area[points])+np.sqrt(area[b]))/2
        deltas=centers[points]-centers[b]
        ds=np.sqrt(np.einsum("ij,ij->i",deltas,deltas))/scales
        in_pool=np.flatnonzero(ds<=4)
        pool_count,unknown=len(in_pool),0
        best={}
        for position in in_pool:
            j=int(subset[position])
            if j not in assigned:
                unknown+=1
                continue
            k=points[position]
            d,delta,scale=ds[position],deltas[position],scales[position]
            other=assigned[j]
            dt=int(arrive[j]-arrive[i])
            key=(d,dt,arrive[j]-depart[j],arrive[j],depart[j],j)
            if other in best and best[other][0]<=key:
                continue
            residual=delta-(depart[j]-arrive[i])*speed
            residual=np.sqrt(np.dot(residual,residual))/scale
            features=np.minimum([d,dt/50,abs(np.log(area[k]/area[b])),residual],10).tolist()
            best[other]=(key,dict(candidate=j,gt_id=other[0],frame=other[1],positive=other[0]==g,
                                   dt=dt,features=features,native=int(not set(proposals[i][3]).isdisjoint(proposals[j][3])),
                                   raw_residual=float(residual)))
        options=[best[key][1] for key in sorted(best) if key[0]!=g or key[1]==next_time]
        p=sum(x["positive"] for x in options)
        neg=len(options)-p
        assert p<=1
        status=("mixed" if neg else "positive_only") if p else ("negative_only" if neg else "neither")
        timing="ending_or_censored" if next_time is None else ("inside_50" if next_time-f<=50 else "outside_50")
        groups.append(dict(gt_id=g,frame=f,direction=direction,next_frame=next_time,candidate=i,
                           gt_timing=timing,has_positive=bool(p),pool_status=status,options=options,
                           unknown_raw_pool_candidates=unknown,later_same_gt_turns_excluded=len(best)-len(options),
                           raw_pool_candidates=pool_count))
    return groups


def compare_values(actual, expected, tag="", tolerance=3e-10):
    if isinstance(expected,dict):
        for k,v in expected.items():
            compare_values(actual[k],v,str(tag)+"/"+str(k),tolerance)
    elif isinstance(expected,(list,tuple)):
        assert len(expected)==len(actual),(tag,len(expected),len(actual))
        for j,(a,b) in enumerate(zip(actual,expected)):
            compare_values(a,b,str(tag)+"/"+str(j),tolerance)
    elif isinstance(expected,(float,np.floating)):
        assert np.isclose(actual,expected,rtol=0,atol=tolerance),(tag,actual,expected)
    else:
        assert actual==expected,(tag,actual,expected)


def fitted(groups):
    mixed=[g for g in groups if g["pool_status"]=="mixed"]
    if len({g["gt_id"] for g in mixed})<10:
        return None
    diffs=[]
    for group in mixed:
        pos,=[o["features"] for o in group["options"] if o["positive"]]
        diffs.append([np.subtract(pos,o["features"]) for o in group["options"] if not o["positive"]])
    return fit_differences(diffs)


def corrected_proof(case,raw,polygon,part):
    frames,changes,turns=truth_index(corrected_truth(raw,part),polygon)
    relevant=defaultdict(list)
    obs=case["observations"]
    for i in sorted({i for row in case["candidates"] for i in row[:2]}):
        relevant[obs[i][0]].append(i)
    overlaps=[]
    for f,ix in relevant.items():
        gt=frames.get(f)
        if not gt:
            continue
        inter,union=xywh_iou_table([obs[i][1:5] for i in ix],np.asarray([r[1:5] for r in gt])*100)
        for i,ii,uu in zip(ix,inter,union):
            nearby=sorted([gt[j][0],ii[j]/uu[j],int(gt[j][5])] for j in np.flatnonzero(10*ii>=3*uu))
            if nearby:
                overlaps.append([i,nearby])
    return dict(turns=sorted([g,t,d] for (g,t),(d,_) in turns.items()),polygon=polygon,overlaps=sorted(overlaps))


def labels_from_evidence(case,proof):
    poly=np.asarray(proof["polygon"],np.float32)
    turns_by_id=defaultdict(list)
    for g,t,d in proof["turns"]:
        turns_by_id[g].append((t,d))
    turns,changes={},defaultdict(list)
    for g,rows in turns_by_id.items():
        pairs=sorted(rows)
        assert len({t for t,d in rows})==len(rows)
        for j,(t,d) in enumerate(pairs):
            assert d in (-1,1) and (j==0 or d==-pairs[j-1][1])
            turns[g,t]=(d,pairs[j+1][0] if j+1<len(rows) else None)
        changes[g]=[t for t,_ in pairs]
    obs=case["observations"]
    known=[(0,()) for _ in obs]
    seen=set()
    for i,records in proof["overlaps"]:
        assert i not in seen and all(.3<=r[1]<=1 and r[2] in (0,1) for r in records)
        assert len({r[0] for r in records})==len(records)
        seen.add(i)
        x,y,w,h=obs[i][1:5]
        side=side_of(poly,x/100,y/100,w/100,h/100)
        known[i]=(len(records),tuple(g for g,q,s in records if q>=.5 and s==side))
    assigned=crossing_membership(case["candidates"],obs,known,changes,turns)
    groups=independently_group(case,assigned,turns)
    return turns,assigned,groups


def rank_results(groups,beta):
    winners=[]
    scores=defaultdict(list)
    blind=[]
    for group in groups:
        if group["pool_status"]!="mixed":
            continue
        opts=group["options"]
        pos,=[o for o in opts if o["positive"]]
        neg=[o for o in opts if not o["positive"]]
        chance=1/len(opts)
        blind.append((group["gt_id"],chance))
        profiles={}
        for x in opts:
            a=x["features"]
            vals=dict(distance=-a[0],time=-x["dt"],cv=-a[3],native=x["native"])
            if beta is not None:
                vals["model"]=float(np.dot(beta,a))
            profiles[x["candidate"]]=vals
        comparisons={}
        for scope,pool in (("all",neg),("same_dt",[n for n in neg if n["dt"]==pos["dt"]])):
            answers={}
            if pool:
                for k,pv in profiles[pos["candidate"]].items():
                    high,neg_key=max((profiles[n["candidate"]][k],-n["candidate"]) for n in pool)
                    diff=pv-high
                    credit=.5 if abs(diff)<=1e-9 else float(diff>0)
                    scores[scope,k].append((group["gt_id"],credit))
                    answers[k]=dict(delta=diff,credit=credit,hardest_candidate=-neg_key)
            comparisons[scope]=answers
        winners.append(dict(gt_id=group["gt_id"],frame=group["frame"],candidate=group["candidate"],
                            positive_candidate=pos["candidate"],negative_turns=len(neg),blind_choice=chance,
                            comparisons=comparisons))
    def mean_record(pairs):
        mbe,mg=group_averages(pairs)
        return dict(groups=len(pairs),bees=len({g for g,_ in pairs}),equal_group=mg,equal_bee=mbe)
    methods=("distance","time","cv","native")+(("model",) if beta is not None else ())
    summary={scope:{k:mean_record(scores[scope,k]) for k in methods} for scope in ("all","same_dt")}
    return dict(group_ranks=winners,summary=summary,chance_uniform_selection=mean_record(blind),
                mixed_groups=len(winners),mixed_origin_bees=len({g["gt_id"] for g in winners}))


def check_case(case,proof,expected_labels,beta=None,expected_scores=None):
    turns,assigned,groups=labels_from_evidence(case,proof)
    led=sorted([g,t,d,nxt] for (g,t),(d,nxt) in turns.items())
    assert led==expected_labels["turns"]
    assert sorted([i,g,t] for i,(g,t) in assigned.items())==expected_labels["candidate_labels"]
    registered=expected_labels["groups"]
    assert len(groups)==len(registered)
    for g,s in zip(groups,registered):
        # Kept separately for examining clipping. Registered options store only the four clipped values.
        for o in g["options"]:
            o.pop("raw_residual")
        compare_values(s,g,"labels")
    stats=coverage(turns,assigned,groups,case["candidates"])
    compare_values(expected_labels["counts"],stats,"counts")
    results=rank_results(groups,beta)
    if expected_scores is not None:
        compare_values(expected_scores,results,"ranks",2e-8)
    return groups,results


def coverage(turns,assigned,groups,candidates):
    from collections import Counter
    counts=Counter(g["pool_status"] for g in groups)
    timing=Counter(g["gt_timing"] for g in groups)
    return dict(gt_crossings=len(turns),gt_crossing_bees=len({k[0] for k in turns}),labeled_anchors=len(groups),
                labeled_anchor_bees=len({g["gt_id"] for g in groups}),labeled_candidate_rows=len(assigned),
                total_candidate_rows=len(candidates),ending_or_censored=timing["ending_or_censored"],
                next_outside_50=timing["outside_50"],next_inside_50=timing["inside_50"],
                inside_50_missing_pool=sum(g["gt_timing"]=="inside_50" and not g["has_positive"] for g in groups),
                outside_50_found_in_pool=sum(g["gt_timing"]=="outside_50" and g["has_positive"] for g in groups),
                observed_next_missing_pool=sum(g["next_frame"] is not None and not g["has_positive"] for g in groups),
                **{status:counts[status] for status in ("mixed","positive_only","negative_only","neither")})


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def check_decisions(report):
    """Recompute the seed and fold branches from individual 0, 0.5 and 1 credits."""
    from fractions import Fraction
    for fold in (33,35):
        flags=[]
        for seed in range(3):
            head=report["scores"][f"{fold}_{seed}_head"]["group_ranks"]
            tail=report["scores"][f"{fold}_{seed}_tail"]["group_ranks"]
            nh,nt=(len({row["gt_id"] for row in rows}) for rows in (head,tail))
            def measure(name):
                by_id=defaultdict(list)
                for row in tail:
                    values=row["comparisons"]["all"]
                    if name not in values:
                        return None
                    by_id[row["gt_id"]].append(Fraction(values[name]["credit"]))
                return sum(sum(v)/len(v) for v in by_id.values())/len(by_id) if by_id else None
            distance,model=measure("distance"),measure("model")
            sufficient=min(nh,nt)>=10
            gain=None if model is None else model-distance
            passed=bool(sufficient and model is not None and model>=Fraction(7,10) and gain>=Fraction(1,10))
            gate=dict(head_bees=nh,tail_bees=nt,sufficient=sufficient,passes=passed,
                      distance=None if distance is None else float(distance),model=None if model is None else float(model),
                      gain=None if gain is None else float(gain))
            compare_values(report["seed_gates"][str(fold)][str(seed)],gate,"registered seed gate")
            flags.append(passed)
        compare_values(report["fold_gates"][str(fold)],dict(seed_pass=flags,n_pass=sum(flags),passes=sum(flags)>=2),"fold gate")


def assert_file(path,digest):
    data=path.read_bytes()
    if sha_bytes(data)!=digest:
        raise ValueError("Frozen input changed: "+str(path))
    return data


def run_checks(args):
    from io import BytesIO
    import sys
    sys.path.insert(0,str(args.root.resolve()))
    index=read_json(args.run/"source_inventory.json")
    assert_file(args.run/"source_inventory.json",args.freeze_sha)
    models=read_json(args.run/"models.json") if args.models_sha else {}
    if args.models_sha:
        assert_file(args.run/"models.json",args.models_sha)
        assert models["source_inventory_sha256"]==args.freeze_sha
    official=read_json(args.run/"results.json") if args.results_sha else {}
    if args.results_sha:
        assert_file(args.run/"results.json",args.results_sha)
        assert official["models_sha256"]==args.models_sha
        check_decisions(official)
    findings={}
    for f in args.fold:
        for s in args.seed:
            key=f"{f}_{s}_{args.part}"
            path=args.run/"source"/(key+".json.gz")
            raw=assert_file(path,index["cases"][key])
            case=json.loads(gzip.decompress(raw))
            desc,geo=case["source"],case["geometry"]
            assert (f,s,args.part)==(desc["fold"],desc["seed"],desc["part"])
            gt=assert_file(args.root/desc["paths"]["gt"],desc["hashes"]["gt"])
            poly=assert_file(args.root/desc["paths"]["zones"],desc["hashes"]["zones"])
            proof=corrected_proof(geo,np.loadtxt(BytesIO(gt),delimiter=",",usecols=range(6),ndmin=2),
                                  json.loads(poly)[desc["clip"]],args.part)
            if args.labels:
                lab=read_json(args.run/"labels"/(key+".json.gz"))
                assert lab["source_inventory_sha256"]==args.freeze_sha
                compare_values(lab["proof"],proof,key+"/raw",5e-12)
                groups,_=check_case(geo,proof,lab["labeled"])
                counts=lab["labeled"]["counts"]
            else:
                turns,assigned,groups=labels_from_evidence(geo,proof)
                counts=coverage(turns,assigned,groups,geo["candidates"])
            bkey=f"{f}_{s}_head"
            if args.part=="head":
                beta=fitted(groups)
                if models:
                    stored=models["fits"][bkey]["coefficients"]
                    if beta is None:
                        assert stored is None
                    else:
                        np.testing.assert_allclose(stored,beta,rtol=0,atol=1e-7)
            else:
                if not args.models_sha:
                    raise ValueError("Tail replay requires a sealed head model receipt")
                beta=models["fits"][bkey]["coefficients"]
            compared=rank_results(groups,beta)
            if official:
                compare_values(official["scores"][key],compared,key,2e-7)
            findings[key]=dict(counts=counts,coefficients=None if beta is None else list(beta),
                               overlaps=len(proof["overlaps"]),summary=compared["summary"],
                               blind=compared["chance_uniform_selection"])
            print(key,counts["mixed"],compared["mixed_origin_bees"],"verified",flush=True)
    record=dict(freeze_sha256=args.freeze_sha,part=args.part,cases=findings,models_sha256=args.models_sha,
                results_sha256=args.results_sha,compared_labels=args.labels)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n",encoding="utf8")
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest="action",required=True)
    modes.add_parser("self-check",help="Synthetic only")
    check=modes.add_parser("raw",help="Validate one partition against the original prepared GT files")
    check.add_argument("--root",type=Path,default=Path.cwd())
    check.add_argument("--run",required=True,type=Path)
    check.add_argument("--part",required=True,choices=("head","tail"))
    check.add_argument("--fold",nargs="+",type=int,choices=(33,35),default=(33,35))
    check.add_argument("--seed",nargs="+",type=int,choices=(0,1,2),default=(0,1,2))
    check.add_argument("--freeze-sha",required=True)
    check.add_argument("--models-sha")
    check.add_argument("--results-sha")
    check.add_argument("--labels",action="store_true",help="Compare each independent proof and labeled group")
    check.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.action=="self-check":
        simple_check()
        print("GT isolation, alternating turns, next-only labels, ties and equal-group fit verified")
    else:
        run_checks(args)


if __name__=="__main__":
    main()
