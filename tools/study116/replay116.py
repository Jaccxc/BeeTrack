"""Independent source-tail sample and full-frame visibility verification for §116."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


PREREG="c8fabf6c55c9737086b02050fabb3360bb5f1db2"
ARCHIVE="2a7d8bf2d4b027dafa59d924e5b908ff2b88c0edbdf9fcded097c60383edd383"


def require(condition,message):
    if not condition:
        raise ValueError(message)


def wire(x):
    return (json.dumps(x,ensure_ascii=False,separators=(",",":"),sort_keys=True,allow_nan=False)+"\n").encode()


def digest(x):
    return hashlib.sha256(x).hexdigest()


def store(path,obj):
    raw=wire(obj)
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        f=Path(tmp)/"record"
        f.write_bytes(raw)
        try:
            os.link(f,path)
        except FileExistsError:
            require(path.read_bytes()==raw,"Witness already exists with different content")
    return digest(raw)


def within(poly,x,y):
    return cv2.pointPolygonTest(poly,(float(x),float(y)),False)>=0


def sample(gt,polygon,clip,nframes):
    """Group observations by raw ID and independently resolve each gap before ranking the events."""
    by_id=defaultdict(list)
    for f,tid,x,y,w,h in gt[:,:6]:
        require(f==int(f) and 1<=f<=nframes and tid==int(tid) and w>=0 and h>=0,"Invalid source GT observation")
        by_id[int(tid)].append((int(f),float(x),float(y),float(w),float(h)))
    poly=np.asarray(polygon,dtype=np.float32)
    observations=defaultdict(list)
    candidates=[]
    seen=set()
    for original,values in sorted(by_id.items()):
        sorted_rows=sorted(values)
        splits=[0]+[i for i in range(1,len(sorted_rows)) if sorted_rows[i][0]-sorted_rows[i-1][0]>25]+[len(sorted_rows)]
        for k,(a,b) in enumerate(zip(splits,splits[1:])):
            resolved=original+100000*k
            require(resolved not in seen,"Resolved GT labels collide")
            seen.add(resolved)
            changes=[]
            previous=None
            for f,x,y,w,h in sorted_rows[a:b]:
                inside=within(poly,x+w/2,y+h/2)
                observations[f].append(dict(gt_tid=resolved,xywh=[x,y,w,h],xyxy=[x,y,x+w,y+h],inside=inside))
                if previous:
                    require(f>previous[0],"Duplicate GT frame for one resolved bee")
                    if inside!=previous[1]:
                        changes.append((previous[0],f,inside))
                previous=f,inside
            if len(changes)%2:
                u,t,end=changes[-1]
                d="in" if end else "out"
                prefix=f"beetrack-warp-pilot-v1|{clip}|{d}|{resolved}|{t}"
                candidates.append(dict(gt_tid=resolved,dir=d,before=u,arrival=t,rank=digest(prefix.encode())))
    candidates.sort(key=lambda e:(e["rank"],e["gt_tid"],e["arrival"]))
    counts=Counter()
    selected,others=[],[]
    for e in candidates:
        if counts[e["dir"]]<8:
            selected.append(e)
            counts[e["dir"]]+=1
        else:
            others.append(e)
    selected=sorted(selected+others[:16-len(selected)],key=lambda e:(e["arrival"],e["gt_tid"]))
    frames=sorted({f for e in selected for f in (e["before"],e["arrival"])})
    bydir=lambda rows:{d:sum(e["dir"]==d for e in rows) for d in ("in","out")}
    pop=dict(planned=16,selected=len(selected),available=len(candidates),available_by_direction=bydir(candidates),
             selected_by_direction=bydir(selected),shortfall=16-len(selected))
    labels=dict(format="warp-pilot-private-labels-v1",clip=clip,nframes=nframes,polygon=poly.tolist(),
                requested_frames=frames,events=selected,
                gt_frames={str(f):sorted(observations[f],key=lambda x:x["gt_tid"]) for f in frames},population=pop)
    return dict(clip=clip,nframes=nframes,frames=frames),labels


def real_samples(repo,out):
    historical=(repo/"docs/study108_source.json.gz").read_bytes()
    require(digest(historical)==ARCHIVE,"§108 source archive fingerprint mismatch")
    prior=json.loads(gzip.decompress(historical))
    zones=(repo/"beetrack/zones.json").read_bytes()
    receipts={}
    for fold in (33,35):
        clip=f"BEE24-{68-fold}-tail"
        key=f"{fold}_0_tail"
        raw=(repo/"runs/gt/val"/clip/"gt/gt.txt").read_bytes()
        gt_sha,zsha=digest(raw),digest(zones)
        require(gt_sha==prior["labels"][key]["gt_mot_sha256"] and
                zsha==prior["masks"]["predictions"][key]["sources"]["beetrack/zones.json"],"Historical GT/zone changed")
        pub,labels=sample(np.loadtxt(BytesIO(raw),delimiter=",",ndmin=2),json.loads(zones)[clip],clip,1000)
        labels["sources_sha256"]={"gt_mot":gt_sha,"zones":zsha}
        refs={k:store(out/clip/name,v) for k,name,v in (("requests","requests.json",pub),("labels","labels.private.json",labels))}
        receipts[clip]=dict(**refs,population=labels["population"],distinct_frames=len(pub["frames"]))
    receipt={"prereg":PREREG,"source108_sha256":ARCHIVE,"clips":receipts}
    print(store(out/"independent_samples.json",receipt),json.dumps(receipts,sort_keys=True))


def verify_rosters(location,independent):
    result={}
    for clip in ("BEE24-33-tail","BEE24-35-tail"):
        files={}
        for name in ("requests.json","labels.private.json"):
            ours=json.loads((independent/clip/name).read_bytes())
            if name.startswith("labels"):
                ours["prereg"]=PREREG
            raw=(location/clip/name).read_bytes()
            require(raw==wire(ours),f"Primary roster differs: {clip} / {name}")
            files[name]=digest(raw)
        result[clip]=files
    print(store(independent/"roster_agreement.json",dict(prereg=PREREG,clips=result)))


def assigned(frame_truth,raw,poly,rect):
    """Solve objective once; compute post-assignment threshold/side statuses in stable row order."""
    ordered=sorted(frame_truth,key=lambda r:r["gt_tid"])
    ids=[r["gt_tid"] for r in ordered]
    gt=np.asarray([r["xywh"] for r in ordered],float).reshape(-1,4)
    require(len(ids)==len(set(ids)),"Duplicate GT at one frame")
    choices=sorted((i for i,r in enumerate(raw) if r[4]>=.25),key=lambda i:(-raw[i][4],*raw[i][:4],i))
    xy=np.asarray([raw[i][:4] for i in choices],float).reshape(-1,4)
    gt_xy=gt.copy()
    gt_xy[:,2:]+=gt[:,:2]
    # frame_match reconstructs the far corner after converting source predictions to xywh.
    decoded=xy.copy()
    decoded[:,2:]=(xy[:,2:]-xy[:,:2])+xy[:,:2]
    common=np.maximum(0,np.minimum(gt_xy[:,None,2:],decoded[None,:,2:])-
                      np.maximum(gt_xy[:,None,:2],decoded[None,:,:2])).prod(axis=2)
    a=np.prod(gt_xy[:,2:]-gt_xy[:,:2],axis=1)
    b=np.prod(decoded[:,2:]-decoded[:,:2],axis=1)
    quality=common/(a[:,None]+b[None,:]-common+1e-9)
    gi,pj=linear_sum_assignment(-quality)
    truth_side=[within(poly,x+w/2,y+h/2) for x,y,w,h in gt]
    centers=(xy[:,:2]+xy[:,2:])/2
    pred_side=[within(poly,x,y) for x,y in centers]
    x0,y0,x1,y1=rect
    region=lambda x,y:"inside" if x0<=x<=x1 and y0<=y<=y1 else "outside"
    result={}
    for tau in (.5,.3):
        owners={}; indexed={}
        totals={zone:{"gt":0,"matched_gt":0,"pred":0,"unmatched_pred":0} for zone in ("inside","outside")}
        for i,j in zip(gi,pj):
            if quality[i,j]>=tau:
                owners[choices[j]]=ids[i]
                indexed[ids[i]]={"index":choices[j],"correct_side":pred_side[j]==truth_side[i]}
        for i,(x,y,w,h) in enumerate(gt):
            area=totals[region(x+w/2,y+h/2)]
            area["gt"]+=1
            area["matched_gt"]+=int(ids[i] in indexed)
        for i,(x,y) in enumerate(centers):
            cell=totals[region(x,y)]
            cell["pred"]+=1
            cell["unmatched_pred"]+=int(choices[i] not in owners)
        result[str(tau)]=dict(matches=indexed,region=totals,
                              owners=[[i,owners.get(i)] for i in choices])
    return result


def inspect_arm(labels,record,rect):
    req=dict(clip=labels["clip"],nframes=labels["nframes"],frames=labels["requested_frames"])
    data=record["predictions"]
    require(record["requests_sha256"]==digest(wire(req)) and req["clip"]==record["clip"]
            and req["frames"]==record["requested_frames"] and req["nframes"]==record["nframes"],"Wrong sampling frame list")
    if record["status"]!="available":
        require(data is None and record["run"].get("unavailable_reason"),"Unavailable arm needs an explicit reason")
        return None
    require(record["predictions_sha256"]==digest(wire(data)) and set(data)==set(labels["gt_frames"]),
            "Prediction digest or GT-frame domain differs")
    poly=np.asarray(labels["polygon"],dtype=np.float32)
    frame={f:assigned(rows,data[f],poly,rect) for f,rows in labels["gt_frames"].items()}
    run={}
    for tau in ("0.5","0.3"):
        events={}
        for e in labels["events"]:
            k=e["gt_tid"]
            ends=[frame[str(e[which])][tau]["matches"].get(k) for which in ("before","arrival")]
            events[k]=dict(ends=ends,visible=all(x and x["correct_side"] for x in ends))
        region={side:{k:sum(f[tau]["region"][side][k] for f in frame.values())
                      for k in ("gt","matched_gt","pred","unmatched_pred")} for side in ("inside","outside")}
        run[tau]=dict(events=events,region=region,
                      owners={f:r[tau]["owners"] for f,r in frame.items()})
    return run


def seed_summary(arms,labels,tau):
    if any(arms[k] is None for k in "ULCH"):
        return {"evaluable":False,"seed_pass":False}
    by_id={}
    frequencies={f"{i:04b}":0 for i in range(16)}
    for e in labels["events"]:
        tid=e["gt_tid"]
        bits="".join(str(int(arms[k][tau]["events"][tid]["visible"])) for k in "ULCH")
        by_id[str(tid)]=bits
        frequencies[bits]+=1
    counts={a:sum(b[i]=="1" for b in by_id.values()) for i,a in enumerate("ULCH")}
    high={tid for tid,b in by_id.items() if b[0]=="0" and b[3]=="1"}
    rescued={a:sorted(int(tid) for tid in high if by_id[tid][i]=="1") for a,i in (("L",1),("C",2))}
    return {"evaluable":True,"counts":counts,"patterns":frequencies,"by_id":by_id,"high_only":len(high),
            "rescued":rescued,"seed_pass":counts["L"]>max(counts["C"],counts["U"]) and bool(rescued["L"])}


def group_gate(n,dose,seed_rows):
    require(len(seed_rows)==3,"A source needs all three seeds")
    return n==16 and dose is not None and dose>=.10 and sum(r["seed_pass"] for r in seed_rows)>=2


def across_seeds(summaries):
    if any(not s["evaluable"] for s in summaries):
        return None
    require(all(summaries[0]["by_id"].keys()==s["by_id"].keys() for s in summaries),
            "Different physical bees across detector seeds")
    conditions={"L_gained":lambda b:b[0:2]=="01","L_lost":lambda b:b[0:2]=="10",
                "high_only":lambda b:b[0]=="0" and b[3]=="1",
                "high_rescued_L":lambda b:b[0:2]=="01" and b[3]=="1",
                "high_rescued_C":lambda b:b[0]=="0" and b[2:]=="11"}
    result={}
    for name,predicate in conditions.items():
        groups=[{int(tid) for tid,b in s["by_id"].items() if predicate(b)} for s in summaries]
        result[name]={"union":sorted(set.union(*groups)),"intersection":sorted(set.intersection(*groups))}
    return result


def check_outputs(root,detector,seal_sha):
    from verify_dose import tensor_gains,verify_dose
    frozen=(detector/"detector_done.json").read_bytes()
    require(digest(frozen)==seal_sha,"Detector index differs from the authorized digest")
    index=json.loads(frozen)
    require(index["study"]==116 and index["prereg"]==PREREG,"Wrong detector preregistration")
    roles=("predictions","wrapped_predictions")
    expected={f"{role}/BEE24-{68-fold}-tail/{fold}_{s}_{a}.json" for role in roles
              for fold in (33,35) for s in range(3) for a in "ULCH"}
    require(set(index["files"])==expected,"Detector seal is incomplete")
    geometry=json.loads((root/"geometry.json").read_bytes())["cases"]
    seals={}; results={}; forwards=0
    for fold in (33,35):
        clip=f"BEE24-{68-fold}-tail"
        capsule=geometry[clip]
        verified_dose=verify_dose(capsule)
        labels=json.loads((root/"validation/sample"/clip/"labels.private.json").read_bytes())
        require(capsule["polygon"]==labels["polygon"],"Entrance polygon differs from source labels")
        results[clip]={}
        for role in roles:
            seed_rows={}
            for s in range(3):
                arms={}; metadata={}
                for a in "ULCH":
                    key=f"{role}/{clip}/{fold}_{s}_{a}.json"
                    raw=(detector/key).read_bytes()
                    require(digest(raw)==index["files"][key],"Frozen prediction hash mismatch")
                    record=json.loads(raw)
                    require((record["run"]["fold"],record["run"]["seed"],record["run"]["dose"])==(fold,s,a),
                            "Prediction case metadata differs")
                    metadata[a]=record
                    arms[a]=inspect_arm(labels,record,capsule["L"]["rect"])
                    seals[key]=store(root/"validation/full_frames"/key,arms[a])
                if role=="predictions":
                    for f in labels["requested_frames"]:
                        quartet={}
                        for a,record in metadata.items():
                            rpath=detector/"receipts"/clip/f"{fold}_{s}_{a}"/f"{f:04}.json"
                            raw=rpath.read_bytes()
                            require(digest(raw)==record["run"]["frame_receipts_sha256"][str(f)],"Unbound receipt")
                            row=json.loads(raw)
                            require(row["predictions"]==record["predictions"][str(f)] and
                                    (row["fold"],row["seed"],row["dose"],row["frame"],row["original_frame"])==
                                    (fold,s,a,f,4000+f),"Receipt provenance differs")
                            quartet[a]=row
                            forwards+=row["forward_count"]
                        actual=tensor_gains(capsule["size"],quartet)
                        np.testing.assert_allclose(actual,verified_dose["H_xyj"],rtol=0,atol=1e-12)
                seed_rows[str(s)]={tau:seed_summary(arms,labels,tau) for tau in ("0.5","0.3")}
            contrasts=[seed_rows[str(s)]["0.5"] for s in range(3)]
            diff=verified_dose["dose_difference"]
            results[clip][role]={"n":len(labels["events"]),"delta_J":diff,"seeds":seed_rows,
                                 "cross_seed_physical_ids":{tau:across_seeds([seed_rows[str(s)][tau] for s in range(3)])
                                                            for tau in ("0.5","0.3")},
                                 "gate":group_gate(len(labels["events"]),diff,contrasts)}
    output=dict(study=116,prereg=PREREG,detector_sha256=seal_sha,forwards=forwards,checks=seals,sources=results)
    print(store(root/"validation/independent_visibility.json",output))
    for clip,row in results.items():
        for role,v in row.items():
            print(clip,role,v["gate"],[v["seeds"][str(s)]["0.5"]["counts"] for s in range(3)])


def compare_primary(root):
    raw=(root/"validation/independent_visibility.json").read_bytes()
    witness=json.loads(raw)
    checked=Counter()
    primary_files={}
    for rel,hashed in sorted(witness["checks"].items()):
        role,clip,name=rel.split("/")
        ours=(root/"validation/full_frames"/rel).read_bytes()
        require(digest(ours)==hashed,"Independent arm receipt differs")
        ours=json.loads(ours)
        key="native" if role=="predictions" else "wrapped"
        official_raw=(root/"scores"/key/clip/name).read_bytes()
        primary_files[f"{key}/{clip}/{name}"]=digest(official_raw)
        primary=json.loads(official_raw)
        for tau,record in ours.items():
            visibility="visible05" if tau=="0.5" else "visible03"
            for event in primary["events"]:
                e=record["events"][str(event["gt_tid"])]
                require(e["visible"]==event[visibility],"Primary V differs")
                for expected,raw_end in zip(e["ends"],event["ends"]):
                    m=raw_end["match"]
                    actual=None if m is None or (tau=="0.5" and not m["owner05"]) else {
                        "index":m["index"],"correct_side":m["correct_side"]}
                    require(expected==actual,"Primary endpoint or detection row differs")
                    checked["ends"]+=1
            for f,projected in record["owners"].items():
                native=primary["frames"][f]
                strict={p["index"]:int(k) for k,p in native["matches"].items() if p["owner05"]}
                ordered=[[p["index"],p["gt_tid03"] if tau=="0.3" else strict.get(p["index"])]
                         for p in native["predictions"]]
                require(projected==ordered,"Primary complete single-frame ownership differs")
                checked["frame_thresholds"]+=1
            for side,value in record["region"].items():
                m=primary["regions"][tau][side]
                require(value=={"gt":m["gt"],"matched_gt":m["tp"],"unmatched_pred":m["fp"],"pred":m["pred"]},
                        "Primary periphery or entrance counts differ")
        checked["arm_files"]+=1
    for role,key in (("predictions","native"),("wrapped_predictions","wrapped")):
        primary_raw=(root/"scores"/(key+"_summary.json")).read_bytes()
        primary_files[key+"_summary.json"]=digest(primary_raw)
        summary=json.loads(primary_raw)
        for row in summary["cases"]:
            clip,s,fold=row["clip"],row["seed"],row["fold"]
            evidence=witness["sources"][clip][role]["seeds"][str(s)]
            labels=json.loads((root/"validation/sample"/clip/"labels.private.json").read_bytes())
            for tau,entry in evidence.items():
                published=row["thresholds"][tau]
                require(entry["patterns"]==published["four_bits"],"Primary 16-cell pattern histogram differs")
                for i,dose in enumerate("ULCH"):
                    require(entry["counts"][dose]==published["doses"][dose]["visible"],"Primary dose total differs")
                    path=root/"validation/full_frames"/role/clip/f"{fold}_{s}_{dose}.json"
                    obs=json.loads(path.read_bytes())[tau]["events"]
                    for d in ("in","out"):
                        tids=[str(e["gt_tid"]) for e in labels["events"] if e["dir"]==d]
                        require(sum(obs[k]["visible"] for k in tids)==published["by_direction"][d]["doses"][dose]["visible"],
                                "Primary directional count differs")
                        for j,phase in enumerate(("u","t")):
                            good=sum(bool(obs[k]["ends"][j] and obs[k]["ends"][j]["correct_side"]) for k in tids)
                            require(good==published["endpoints"][dose][d][phase]["success"],
                                    "Primary marginal endpoint count differs")
        for clip,data in witness["sources"].items():
            point=data[role]
            primary=summary["source_decisions"][clip]
            require(point["gate"]==primary["advance"] and point["n"]==primary["n"]
                    and abs(primary["dose_difference"]-point["delta_J"])<1e-10
                    and primary["seed_conditions"]==[point["seeds"][str(s)]["0.5"]["seed_pass"] for s in range(3)]
                    and primary["qualifying_seed_count"]==sum(v["0.5"]["seed_pass"] for v in point["seeds"].values()),
                    "Primary source advancement differs")
    output=dict(prereg=PREREG,independent_sha256=digest(raw),detector_sha256=witness["detector_sha256"],
                checked=dict(checked),primary_files_sha256=primary_files)
    print(store(root/"validation/primary_agreement.json",output),checked)


def self_check():
    from beetrack.evaluate import frame_match
    poly=np.float32([[0,-30],[100,-30],[100,30],[0,30]])
    gt=lambda k,x,y=0:dict(gt_tid=k,xywh=[x,y,4,4])
    # Deliberately unsorted and tied duplicates. Stable prediction sorting gives the
    # first of duplicate raw rows priority; the lower-numbered competing GT wins.
    g=[gt(22,2),gt(4,2),gt(7,-3,8)]
    p=[[2,0,6,4,.6],[2,0,6,4,.9],[-3,8,1,12,.25],[-3,8,1,12,.2499]]
    out=assigned(g,p,poly,[0,-10,50,20])
    assert out["0.5"]["owners"]==[[1,4],[0,22],[2,7]]
    assert within(poly,0,0)
    # Exact union has the official epsilon; one 2x4 box inside a 4x4 GT misses at .5.
    trial=assigned([gt(1,2)],[[2,0,4,4,.8]],poly,[0,-10,50,20])
    assert not trial["0.5"]["matches"] and trial["0.3"]["matches"][1]["correct_side"]
    # The alternative assignment has two IoUs above .5 (.75 and 4/7). The official
    # maximum-sum assignment takes 1 and 3/7, giving only one retained owner at .5.
    collision=assigned([dict(gt_tid=1,xywh=[0,0,4,4]),dict(gt_tid=2,xywh=[0,0,7,4])],
                       [[0,0,4,4,.9],[1,0,4,4,.8]],poly,[0,0,50,20])
    assert len(collision["0.5"]["matches"])==1 and len(collision["0.3"]["matches"])==2
    flip=assigned([gt(5,-2.1)],[[-1.9,0,2.1,4,.95]],poly,[0,0,50,20])
    assert flip["0.5"]["matches"][5]["correct_side"] is False
    rng=np.random.default_rng(116)
    for _ in range(100):
        actual=[gt(k,int(rng.integers(-7,8)),int(rng.integers(-3,4))) for k in range(4)]
        raw=[]
        for i in range(5):
            x,y=rng.integers(-7,8,size=2)
            raw.append([x,y,x+4,y+4,float(rng.choice([.9,.25,.2]))])
        checked=assigned(actual,raw,poly,[-2,-2,6,6])
        for tau,stats in checked.items():
            order=[i for i,_ in stats["owners"]]
            gg=np.array([[1,r["gt_tid"],*r["xywh"]] for r in actual],float)
            pp=np.array([[1,i,*raw[i][:2],raw[i][2]-raw[i][0],raw[i][3]-raw[i][1]] for i in order],float).reshape(-1,6)
            official=frame_match(gg,pp,thr=float(tau))
            assert [None if x<0 else int(x) for x in official]==[v for _,v in stats["owners"]]
    # Missing opposite-side observation on the same bee cannot be rescued by another bee's detection.
    ids=sample(np.array([[1,1,-5,0,4,4],[2,1,1,0,4,4],[1,2,-5,10,4,4],[2,2,1,10,4,4]],float),poly,"synthetic",2)
    req,labels=ids
    def arm(frames):
        return inspect_arm(labels,{"status":"available","clip":req["clip"],"nframes":req["nframes"],
                "requested_frames":req["frames"],"requests_sha256":digest(wire(req)),"predictions":frames,
                "predictions_sha256":digest(wire(frames))},[0,-10,50,20])
    mix=arm({"1":[[-5,0,-1,4,.9]],"2":[[1,10,5,14,.9]]})
    has=arm({"1":[[-5,0,-1,4,.9]],"2":[[1,0,5,4,.9]]})
    assert not any(e["visible"] for e in mix["0.5"]["events"].values())
    combined=seed_summary({"U":mix,"L":has,"C":mix,"H":has},labels,"0.5")
    assert combined["seed_pass"] and combined["patterns"]["0101"]==1
    votes=[combined,combined,dict(seed_pass=False)]
    assert group_gate(16,.10,votes) and not group_gate(15,.11,votes) and not group_gate(16,.099,votes)
    print("Synthetic frame assignment, ties, threshold, two-box identity and gates agree with the official owner.")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    sub=cli.add_subparsers(dest="mode",required=True)
    s=sub.add_parser("sample")
    s.add_argument("repo",type=Path)
    s.add_argument("out",type=Path)
    sub.add_parser("self-check")
    v=sub.add_parser("compare-roster")
    v.add_argument("primary",type=Path)
    v.add_argument("independent",type=Path)
    f=sub.add_parser("verify")
    f.add_argument("study",type=Path)
    f.add_argument("detector",type=Path)
    f.add_argument("--seal-sha",required=True)
    c=sub.add_parser("compare")
    c.add_argument("study",type=Path)
    args=cli.parse_args()
    if args.mode=="sample":
        real_samples(args.repo,args.out)
    elif args.mode=="compare-roster":
        verify_rosters(args.primary,args.independent)
    elif args.mode=="verify":
        check_outputs(args.study,args.detector,args.seal_sha)
    elif args.mode=="compare":
        compare_primary(args.study)
    else:
        self_check()
