"""Pack and independently replay the six source-only seven-frame counting studies."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import bmat, csr_matrix, eye
from tools.study115.release115 import numerical


STUDY108="2a7d8bf2d4b027dafa59d924e5b908ff2b88c0edbdf9fcded097c60383edd383"
PREREG="f713b556be94b9eb669ae2565c1532a79cf1b854"
FORMAT="seven-frame-capacity-replay-v1"
KEYS=[(f"{fold}_{seed}_tail",fold,seed,f"BEE24-{68-fold}-tail") for fold in (33,35) for seed in range(3)]
TAUS=(("05","0.5"),("03","0.3"))


def require(ok, description):
    if not ok:
        raise ValueError(description)


def wire(x):
    return json.dumps(x,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def incidence(frames,n):
    """Select event times by explicit half-open intervals in each direction."""
    fs=np.asarray(frames,dtype=int)
    require((abs(fs)<=n).all() and (fs!=0).all(),"Bad signed time")
    rows=[]
    for origin in range(1,-749,-25):
        for start in range(origin,n+1,750):
            for sign in (1,-1):
                rows.append((sign*fs>=max(1,start)) & (sign*fs<min(n+1,start+750)))
    a=np.asarray(rows,dtype=np.int16)
    require((a.sum(axis=0)==30).all(),"Origin coverage lost a signed event")
    return a


def optimum(residual, groups, matrix):
    by_id=defaultdict(list)
    for j,g in enumerate(groups):
        by_id[g].append(j)
    if math.prod(len(v)+1 for v in by_id.values())<=4000:
        best=int(abs(residual).sum())
        for choices in product(*[[None,*v] for v in by_id.values()]):
            use=[j for j in choices if j is not None]
            candidate=int(abs(residual-matrix[:,use].sum(axis=1)).sum())
            best=min(best,candidate)
        return best
    h,w=matrix.shape
    # x_j selects a concrete arrival time and u_j majorizes the absolute residual.
    one_per_gt=csr_matrix(np.array([np.isin(np.arange(w),v) for v in by_id.values()],dtype=float))
    g=csr_matrix(matrix,dtype=float)
    system=bmat([[one_per_gt,None],[g,-eye(h)],[-g,-eye(h)]],format="csr")
    right=np.r_[np.ones(len(by_id)),residual,-residual]
    answer=milp(np.r_[np.zeros(w),np.ones(h)],integrality=np.r_[np.ones(w),np.zeros(h)],
                bounds=Bounds(0,np.r_[np.ones(w),np.full(h,np.inf)]),
                constraints=LinearConstraint(system,-np.inf,right),options={"mip_rel_gap":0})
    require(answer.success and answer.status==0,"Independent MILP did not finish")
    x=np.rint(answer.x[:w]).astype(int)
    actual=int(abs(residual-matrix@x).sum())
    require(np.allclose(answer.x[:w],x,atol=1e-8,rtol=0) and
            all(x[v].sum()<=1 for v in by_id.values()) and
            abs(answer.mip_dual_bound-actual)<1e-7 and abs(answer.fun-actual)<1e-7,
            "Independent solver did not prove the integer optimum")
    return actual


def graph_matches(native,gt_times,edges,check_variability=True):
    pids,gids=sorted(native),sorted(gt_times)
    pi={pid:k for k,pid in enumerate(pids)}
    gi={gid:k for k,gid in enumerate(gids)}
    a=np.zeros((len(pids),len(gids)),dtype=int)
    for p,g in edges:
        require(p in pi and g in gi and native[p]*gt_times[g]>0 and not a[pi[p],gi[g]],"Bad graph edge")
        a[pi[p],gi[g]]=1
    def matching(b):
        ix,jx=linear_sum_assignment(-b)
        return sorted((pids[i],gids[j]) for i,j in zip(ix,jx) if a[i,j])
    main=matching(a)
    change=set()
    if not check_variability:
        return main,change
    for j,g in enumerate(gids):
        if a[:,j].any():
            low=2*a.copy()
            low[:,j]-=a[:,j]
            high=2*a.copy()
            high[:,j]+=a[:,j]
            if any(h==g for _,h in matching(high)) and not any(h==g for _,h in matching(low)):
                change.add(g)
    return main,change


def flip_report(case,inventory,n):
    grouped=defaultdict(list)
    for row in case["adjacent_flips"]:
        grouped[row["id"]].append(row)
    counts,confusion,returns=Counter(),Counter(),Counter()
    for g,sequence in grouped.items():
        flips=sorted(sequence,key=lambda z:z["t"])
        require(len({z["t"] for z in flips})==len(flips),"Duplicate flip time")
        net=(len(flips)%2==1)
        require(net==(g in inventory),"Truth net occurrence differs from flip parity")
        if net:
            require((inventory[g]["direction"],inventory[g]["u"],inventory[g]["t"])==
                    (flips[-1]["direction"],flips[-1]["u"],flips[-1]["t"]),"Net event differs from final flip")
        for i,r in enumerate(flips):
            u,t,d=r["u"],r["t"],r["direction"]
            require(1<=u<t<=n and (not i or (flips[i-1]["direction"]!=d and flips[i-1]["t"]<=u)),
                    "Invalid true flip chronology")
            is_end=(net and i==len(flips)-1)
            kind=("final" if is_end else "no_net_event" if not net else
                  "superseded" if d==inventory[g]["direction"] else "reverse_of_net")
            complete=4<=t<=n-3
            later=flips[i+1]["t"] if i+1<len(flips) else None
            require((r["gap"],r["complete"],r["canonical"],r["kind"],r["next_flip"],r["return_visible"])==
                    (t-u,complete,is_end,kind,later,later is not None and later<=t+3),"Flip classification")
            counts[d,kind]+=1
            for horizon in (3,25,100):
                support=[z for z in flips if t-horizon<=z["u"] and z["t"]<=t+horizon]
                label=bool(len(support)%2 and support[-1]["t"]==t)
                require(r["local_labels"][str(horizon)]==label,"Local identity label disagrees with flip parity")
                if complete and t-u<=3:
                    confusion[d,horizon,is_end,label]+=1
            if complete and t-u<=3:
                returns[d,"end" if later is None else "inside" if later<=t+3 else "later"]+=1
    require(set(inventory)<=set(grouped),"Missing official GT flip")
    return {"kinds":[[*k,v] for k,v in sorted(counts.items())],
            "local":[[*k,v] for k,v in sorted(confusion.items())],
            "next_return":[[*k,v] for k,v in sorted(returns.items())]}


def physical(saved,base_mapping,native,truth,addition):
    target={int(k):v for k,v in saved["added_targets"].items()}
    require(len(addition)==len(target) and set(target.values())==set(addition)
            and not (set(target)&set(native)),"Physical ID allocation")
    stream=dict(native)
    stream.update({p:addition[g] for p,g in target.items()})
    ng,npred=len(truth),len(stream)
    tp,fp,fn=saved["tp"],saved["fp"],saved["fn"]
    matches=[tuple(x) for x in saved["matching"]]
    require(len({p for p,g in matches})==len({g for p,g in matches})==len(matches)==tp and
            (fp,fn)==(npred-tp,ng-tp) and saved["f1"]==2*tp/max(1,npred+ng),"Physical event totals")
    strict=saved["strict_same_bees"]
    reproduce,_=graph_matches(stream,truth,saved["edges"],False)
    owners={int(k):v for k,v in saved["added_frame_owners"].items()}
    require(matches==reproduce and set(owners)==set(target) and all(len(v)==2 for v in owners.values()),
            "Augmented event matching or frame-owner inventory")
    edge_set=set(map(tuple,saved["edges"]))
    chosen=dict(matches)
    recovered=sorted(g for p,g in target.items() if owners[p]==[g,g] and chosen.get(p)==g)
    require(strict==recovered and saved["added_matched_to_own"]==sum(chosen.get(p)==g for p,g in target.items()) and
            saved["added_with_own_edge"]==sum((p,g) in edge_set for p,g in target.items()),"Physical owner tallies")
    require(sorted(set(dict(base_mapping).values())-set(dict(matches).values()))==saved["prior_matched_bees_lost"]
            and sorted(set(dict(base_mapping))-set(dict(matches)))==saved["prior_pred_ids_lost"],"Prior ID losses")
    return Fraction(2*tp,max(1,ng+npred))


def check_observations(record,poly,n):
    g,t0=record["id"],record["t"]
    seq={}
    direction=record["signed_t"]>0
    for signed,x,y,w,h in record["local_observations"]:
        t=abs(signed)
        inside=cv2.pointPolygonTest(poly,(float(x+w/2),float(y+h/2)),False)>=0
        require(t not in seq and 1<=t<=n and t0-8<=t<=t0+5 and w>=0 and h>=0 and inside==(signed>0),
                "Camera-space truth observation")
        seq[t]=(inside,[t,g,x,y,w,h])
    require(t0 in seq and seq[t0][0]==direction,"Missing official arrival observation")
    expected={}
    for t,(side,rt) in seq.items():
        if not (4<=t<=n-3 and abs(t-t0)<=5 and side==direction):
            continue
        for u in range(t-3,t):
            if u in seq and seq[u][0]!=side:
                expected[u,t]=(seq[u][1],rt)
    seen={}
    for v in record["witnesses"]:
        pk=v["u"],v["t"]
        require(pk not in seen,"Repeated local witness")
        seen[pk]=(v["first"],v["last"])
    require(seen==expected,"Exported witnesses differ from signed observations around the official event")


def replay_case(case):
    n=case["nframes"]
    require(n==1000,"Expected full source tail")
    truth=dict(case["gt_events"])
    native={r[0]:r[1] for r in case["native_events"]}
    roster={r["id"]:r for r in case["event_inventory"]}
    require(len(roster)==len(case["event_inventory"])==len(case["gt_events"])==len(truth) and
            len(native)==len(case["native_events"]) and truth=={g:r["signed_t"] for g,r in roster.items()},
            "Repeated or missing native/GT event")
    labels=flip_report(case,roster,n)
    poly=np.asarray(case["polygon"],dtype=np.float32)
    for item in roster.values():
        check_observations(item,poly,n)
    matches,category={},{}
    for key,tau in TAUS:
        edges=case["edges"+key]
        matches[tau],var=graph_matches(native,truth,edges)
        incident={g for p,g in edges}; hit={g for p,g in matches[tau]}
        states={g:"matched" if g in hit else "edge_unmatched" if g in incident else "no_edge" for g in truth}
        require(states=={g:r["category"+key] for g,r in roster.items()} and
                var=={g for g,r in roster.items() if r["variable_match"+key]} and
                dict(Counter(states.values()))==case["categories"+key],"Canonical matching or variable GT identities")
        require(matches[tau]==list(map(tuple,case["baseline"][tau]["matching"])),"Baseline graph discrepancy")
        physical(case["baseline"][tau],matches[tau],native,truth,{})
        category[tau]=states
    want=incidence(list(truth.values()),n).sum(axis=1)
    got=incidence(list(native.values()),n).sum(axis=1)
    residual=want-got
    denominator=30*len(truth); base=int(abs(residual).sum())
    require(denominator>0,"Need positive source GT denominator")
    table={}
    for fam,group in product(("canonical","tolerant"),("no_edge","all_unmatched")):
        name=f"{fam}_{group}"
        bound=case["bounds"][name]
        opts=[]
        for g,r in sorted(roster.items()):
            require(r["signed_t"]==r["t"]*(1 if r["direction"]=="in" else -1),"GT direction")
            eligible=(category["0.5"][g]=="no_edge" or (group=="all_unmatched" and category["0.5"][g]=="edge_unmatched"))
            alternatives=defaultdict(list)
            for w in r["witnesses"]:
                u,t=w["u"],w["t"]
                require(1<=u<t<=n and 1<=t-u<=3 and abs(t-r["t"])<=5 and 4<=t<=n-3
                        and w["id"]==g and w["direction"]==r["direction"] and
                        w["first"][:2]==[u,g] and w["last"][:2]==[t,g] and
                        w["canonical"]==((u,t)==(r["u"],r["t"])) and
                        w["signed_t"]==t*(1 if r["direction"]=="in" else -1),"Illegal GT witness")
                if eligible and (fam=="tolerant" or w["canonical"]):
                    alternatives[t].append(w)
            for t,values in sorted(alternatives.items()):
                opts.append(max(values,key=lambda z:(z["canonical"],z["u"])))
        frame=[q["signed_t"] for q in opts]; ids=[q["id"] for q in opts]
        picks=bound["selected"]
        require(all(type(j) is int and 0<=j<len(opts) for j in picks) and len(set(picks))==len(picks),"Invalid selections")
        chosen=[opts[j] for j in picks]
        require(len({q["id"] for q in chosen})==len(chosen) and
                [{k:q[k] for k in ("id","direction","u","t","canonical")} for q in chosen]==bound["chosen"] and
                (len(opts),len(set(ids)))==(bound["options"],bound["eligible_bees"]),"Selection chronology or eligibility")
        receipt=bound["physical_mot"]
        require(receipt["file"]==f"{name}.txt" and re.fullmatch("[a-f0-9]{64}",receipt["sha256"])
                and receipt["native_rows"]>=0 and receipt["added_rows"]==2*len(chosen),"Physical two-box MOT receipt")
        m=incidence(frame,n)
        primal=int(abs(residual-m[:,picks].sum(axis=1)).sum())
        best=optimum(residual,ids,m)
        proof=bound["proof"]
        if "integer_lower" in proof:
            low,upper=proof["integer_lower"],proof["primal_upper"]
        else:
            low,upper=proof["lower"],proof["upper"]
        require(low==upper==best==primal==bound["min_numerator"] and base==bound["base_numerator"] and
                denominator==bound["denominator"],"Oracle or solver certificate contradiction")
        gain=Fraction(100*(base-best),denominator)
        require(abs(float(gain)-bound["max_gain_pp"])<1e-12,"W gain discrepancy")
        for key,tau in TAUS:
            saved=bound["official_f1_of_this_witness"][tau]
            physical(saved,matches[tau],native,truth,{q["id"]:q["signed_t"] for q in chosen})
        seen=bound["official_f1_of_this_witness"]["0.5"]
        table[name]=dict(options=len(opts),eligible=len(set(ids)),gain=gain,base=base,best=best,den=denominator,
                         f1=Fraction(2*seen["tp"],max(1,2*seen["tp"]+seen["fn"]+seen["fp"])),
                         strict=seen["strict_same_bees"])
    q=case["baseline"]["0.5"]
    return dict(table=table,gt_directions=dict(Counter("in" if t>0 else "out" for t in truth.values())),
                base_f1=Fraction(2*q["tp"],max(1,2*q["tp"]+q["fp"]+q["fn"])),local=labels)


def summary_fields(report):
    q=report["table"]["canonical_no_edge"]
    return dict(w_gain_pp=float(q["gain"]),base_numerator=q["base"],opt_numerator=q["best"],denominator=q["den"],
                eligible=q["eligible"],strict_ids=q["strict"],f1=float(q["f1"]),base_f1=float(report["base_f1"]))


def decisions(reports):
    result={}; eps=Fraction(1,10**9)
    for fold in (33,35):
        rm=[reports[f"{fold}_{s}_tail"] for s in range(3)]
        q=[r["table"]["canonical_no_edge"] for r in rm]
        gain=sum(r["gain"] for r in q)/3
        positive=sum(r["gain"]>eps for r in q)
        hits=sum(bool(r["strict"]) for r in q)
        df1=sum(r["f1"]-b["base_f1"] for r,b in zip(q,rm))/3
        capacity=gain>=2 and positive>=2
        physical_ok=hits>=2 and df1>=-eps
        result[str(fold)]=dict(clip=f"BEE24-{68-fold}-tail",mean_w_gain_pp=float(gain),positive_seeds=positive,
                               capacity=capacity,strict_seeds=hits,delta_f1=float(df1),physical=physical_ok,
                               advance=capacity and physical_ok)
    return result


def check_history(case,record,labels,key):
    raw=record[key]; lab=labels[key]
    pairs=[[pid,f,*[round(float(v),2) for v in xyxy]] for pid,f,*xyxy in case["native_events"]]
    require(pairs==raw["predictions"] and case["gt_events"]==lab["gt"] and
            all(case[k]==lab[k] for k in ("edges05","edges03")),"Historical GT, P0 or graph differs")
    m=case["meta"]; det=f"dets/loho_no{m['fold']}_s{m['seed']}/val/{m['clip']}.npy"
    require(m["hashes"]==dict(gt=lab["gt_mot_sha256"],mot=raw["mot_sha256"],zone=raw["sources"]["beetrack/zones.json"],
                             det=raw["sources"][det]),"Historical source fingerprints changed")


def replay(data,history=None):
    index=data["index"]; cases=data["cases"]
    require(data["format"]==FORMAT and sha(wire(index))==data["index_sha256"] and index["prereg"]==PREREG
            and index["history_sha256"]==STUDY108,"Wrong study or preregistration identity")
    expected={key for key,*_ in KEYS}
    require(set(cases)==set(index["cases"])==expected,"Expected six registered source cases")
    reports={}; cross={}
    for key,fold,seed,clip in KEYS:
        summary=index["cases"][key]; c=cases[key]; m=c["meta"]
        require(summary["file"]==f"case_{key}.json" and summary["clip"]==clip and
                (m["fold"],m["seed"],m["clip"],m["prereg"])==(fold,seed,clip,index["prereg"]),"Source schedule changed")
        require(sha(wire(c))==summary["sha256"] and set(m["hashes"])=={"det","gt","mot","zone"} and
                all(re.fullmatch("[a-f0-9]{64}",h) for h in m["hashes"].values()),"Source receipt mismatch")
        if history is not None:
            check_history(c,history["masks"]["predictions"],history["labels"],key)
        identity=[c["gt_events"],c["adjacent_flips"],c["polygon"],m["hashes"]["gt"],
                  [{k:v for k,v in r.items() if not k.startswith(("category","variable_match"))}
                   for r in c["event_inventory"]]]
        require(cross.setdefault(clip,identity)==identity,"Three seeds do not share the same truth")
        r=replay_case(c)
        s=summary_fields(r)
        require(all(v==summary[k] for k,v in s.items()),"Source index summary differs from replay")
        reports[key]=r
    decision=decisions(reports)
    for f,values in decision.items():
        official=index["gates"][f]
        require(set(values)==set(official) and all(abs(v-official[k])<=1e-12 if isinstance(v,float) else v==official[k]
                                                  for k,v in values.items()),"Study gate differs from replay")
    return reports,decision


def collect(directory,pinned):
    raw=(directory/"index.json").read_bytes()
    require(sha(raw)==pinned,"Unexpected complete score index SHA-256")
    index=json.loads(raw)
    require(wire(index)==raw,"Index bytes must use the research release encoding")
    cases={}
    for key,*_ in KEYS:
        s=index["cases"][key]
        require(s["file"]==f"case_{key}.json","Unexpected physical case location")
        b=(directory/s["file"]).read_bytes()
        require(sha(b)==s["sha256"],"Changed physical score file")
        case=json.loads(b)
        for name,bound in case["bounds"].items():
            require(name in {"canonical_no_edge","canonical_all_unmatched","tolerant_no_edge","tolerant_all_unmatched"},
                    "Unexpected MOT arm")
            receipt=bound["physical_mot"]
            require(receipt["file"]==name+".txt","Unexpected physical MOT filename")
            mot=(directory/"physical"/key/receipt["file"]).read_bytes()
            require(sha(mot)==receipt["sha256"] and len(mot.splitlines())==receipt["native_rows"]+receipt["added_rows"],
                    "Physical track rows or fingerprint changed")
            if receipt["added_rows"]==0:
                require(sha(mot)==case["meta"]["hashes"]["mot"],"Unmodified MOT bytes changed")
        cases[key]=case
    return dict(format=FORMAT,index_sha256=pinned,index=index,cases=cases)


def table_rows(reports):
    rows=[]
    for key,fold,seed,clip in KEYS:
        r=reports[key]
        row=dict(key=key,clip=clip,seed=seed,gt_in=r["gt_directions"].get("in",0),gt_out=r["gt_directions"].get("out",0),
                 base_f1=float(r["base_f1"]),local=r["local"])
        row["families"]={name:{**d, "gain":float(d["gain"]),"f1":float(d["f1"])} for name,d in r["table"].items()}
        rows.append(row)
    return rows


def emit(reports,gates):
    print("source seed family | eligible times | W baseline optimum gain(pp) | F1 baseline witness | strict bees")
    for row in table_rows(reports):
        for name,q in row["families"].items():
            print(f"{row['clip']} {row['seed']} {name} | {q['eligible']} {q['options']} | "
                  f"{100*q['base']/q['den']:.3f} {100*q['best']/q['den']:.3f} {q['gain']:.3f} | "
                  f"{row['base_f1']:.4f} {q['f1']:.4f} | {q['strict']}")
    print(json.dumps(gates,sort_keys=True))


def self_check():
    from tempfile import TemporaryDirectory
    from capacity import analyze
    from run_capacity import physical_sink
    def box(t,i,x,y=4):
        return [t,i,x-1,y-1,2,2]
    poly=[[0,0],[20,0],[20,20],[0,20]]
    t=PREREG
    with TemporaryDirectory() as tmp:
        d=Path(tmp); reports={}; idx={}
        for key,fold,seed,clip in KEYS:
            export=[box(10,1,-2),box(11,1,2),box(100,2,-2,15),box(102,2,2,15),
                    box(45,5,2),box(46,5,-2),box(70,5,2)]
            gt=np.asarray(export,float)
            prior=(export[:4] if fold==35 else [])+[box(400,7,8)]
            if fold==33 and seed==2:
                prior.extend([box(10,10,-2),box(11,11,2)])
            p0_path=d/(key+".txt")
            np.savetxt(p0_path,np.asarray([row+[1,-1,-1,-1] for row in prior],float),fmt="%g",delimiter=",")
            hashes=dict.fromkeys(("gt","det","zone"),"f"*64)
            hashes["mot"]=sha(p0_path.read_bytes())
            r=analyze(gt,np.asarray(prior,float),poly,1000,
                      physical_sink=physical_sink(p0_path,d/"physical"/key))
            r.update(meta=dict(fold=fold,seed=seed,clip=clip,hashes=hashes,prereg=t))
            filename=f"case_{key}.json"; b=wire(r)
            (d/filename).write_bytes(b)
            reports[key]=replay_case(r)
            idx[key]=dict(file=filename,sha256=sha(b),clip=clip,**summary_fields(reports[key]))
        index=dict(cases=idx,prereg=t,history_sha256=STUDY108,gates=decisions(reports))
        raw=wire(index); (d/"index.json").write_bytes(raw)
        packed=collect(d,sha(raw))
        archive=gzip.compress(wire(numerical(packed)),mtime=0)
        rr,gates=replay(numerical(json.loads(gzip.decompress(archive)),decode=True))
        require(gates["33"]["advance"] and not gates["35"]["advance"],"Synthetic gates")
        broken=numerical(json.loads(gzip.decompress(archive)),decode=True)
        del broken["cases"]["33_0_tail"]["event_inventory"][0]["witnesses"][0]
        badcase=broken["cases"]["33_0_tail"]
        broken["index"]["cases"]["33_0_tail"]["sha256"]=sha(wire(badcase))
        broken["index_sha256"]=sha(wire(broken["index"]))
        try:
            replay(broken)
        except ValueError as ex:
            require("signed observations" in str(ex),"Unexpected negative control")
        else:
            raise AssertionError("Dropped witness survived source replay")
        pts=[1,25,26,50,51,500,501,749,750,751,752,999,1000]
        matrix=incidence(pts,1000)
        missing=incidence([25,26,51,500,749,751],1000).sum(axis=1)
        require(optimum(missing,list(range(len(pts))),matrix)==0,"Independent MILP boundary case")
        print(f"Six synthetic source cases, 24 independent optima, a larger MILP and positive/negative gates passed; archive {len(archive)} bytes.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    cmd=p.add_subparsers(dest="mode",required=True)
    cmd.add_parser("self-check")
    pack=cmd.add_parser("pack"); pack.add_argument("source",type=Path); pack.add_argument("--index-sha",required=True)
    run=cmd.add_parser("replay"); run.add_argument("--sha256")
    for action in (run,pack):
        action.add_argument("archive",type=Path)
        action.add_argument("--history",type=Path)
        action.add_argument("--report",type=Path)
        action.add_argument("--table",action="store_true")
    args=p.parse_args()
    if args.mode=="self-check":
        self_check()
    else:
        hist=None
        if args.history is not None:
            b=args.history.read_bytes()
            require(sha(b)==STUDY108,"Wrong published §108 archive")
            hist=json.loads(gzip.decompress(b))
        if args.mode=="pack":
            data=collect(args.source,args.index_sha)
        else:
            payload=args.archive.read_bytes()
            if args.sha256:
                require(sha(payload)==args.sha256,"Published replay bundle SHA-256 mismatch")
            data=numerical(json.loads(gzip.decompress(payload)),decode=True)
        reports,gates=replay(data,hist)
        if args.mode=="pack":
            payload=gzip.compress(wire(numerical(data)),mtime=0)
            if args.archive.exists():
                require(args.archive.read_bytes()==payload,"Existing replay bundle has different contents")
            else:
                args.archive.parent.mkdir(parents=True,exist_ok=True)
                args.archive.write_bytes(payload)
            print(sha(payload))
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_bytes(wire(dict(rows=table_rows(reports),gates=gates)))
        if args.table:
            emit(reports,gates)
