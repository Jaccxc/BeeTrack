"""Independent whole-MOT, competition graph and count checks for a frozen §115 role."""
import argparse
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from beetrack.common import ROOT,RUNS
from beetrack.evaluate import by_track, frame_match
from tools.study103.replay import bins
from tools.study104.event_overlap import score_arm,archive_arm
from verify_e3_counts import verify_counts


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def wire(record):
    return (json.dumps(record,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def load_checked(path, digest):
    raw=Path(path).read_bytes()
    check(sha(raw)==digest,f"Fingerprint mismatch: {path}")
    return raw


def table(raw,ncol):
    return np.loadtxt(BytesIO(raw),delimiter=",",ndmin=2) if raw.strip() else np.empty((0,ncol))


def parent_pool(streams, polygon):
    """Split native reused IDs; reserve one interval per parent in (seed,resolved ID) order."""
    all_parents={}
    for s,mot in streams.items():
        ordered=mot[np.lexsort((mot[:,0],mot[:,1]))]
        tids,starts=np.unique(ordered[:,1].astype(int),return_index=True)
        for tid,i,j in zip(tids,starts,[*starts[1:],len(ordered)]):
            part=ordered[i:j]
            check(np.all(np.diff(part[:,0])>0),"Duplicate native parent row in one frame")
            for ordinal,segment in enumerate(np.split(part,np.flatnonzero(np.diff(part[:,0])>25)+1)):
                key=(s,int(tid)+100000*ordinal)
                check(key not in all_parents,"Native resolved IDs collide")
                all_parents[key]=segment
    poly=np.asarray(polygon,dtype=np.float32)
    layout, desc, pool, suppressed, ranges=[],{s:[] for s in streams},[],[],{}
    base,first=1,0
    for (s,tid),rows in sorted(all_parents.items()):
        inside=np.array([cv2.pointPolygonTest(poly,(float(x+w/2),float(y+h/2)),False)>=0
                         for x,y,w,h in rows[:,2:6]])
        cuts=np.r_[0,np.cumsum(inside[1:]!=inside[:-1])]
        reserve=int(cuts[-1])+1
        layout.append([s,tid,base,reserve])
        copied=rows[:,:7].copy()
        copied[:,1]=base
        pool.append(copied)
        if inside[0]!=inside[-1]:
            i=int(np.flatnonzero(np.diff(cuts))[-1]+1)
            f,_,x,y,w,h=rows[i,:6]
            desc[s].append([tid,int(f)*(1 if inside[-1] else -1),round(float(x),2),round(float(y),2),
                            round(float(x+w),2),round(float(y+h),2)])
            suppressed.append(base+cuts)
            ranges[base]=slice(first,first+len(rows))
        else:
            suppressed.append(np.full(len(rows),base))
        first+=len(rows)
        base+=reserve
    return (np.concatenate(pool) if pool else np.empty((0,7)),layout,desc,
            np.concatenate(suppressed) if suppressed else np.empty((0,),dtype=int),ranges)


def physical_rows(folder,pack,pool,suppressed,ranges,layout):
    parent_id={(s,i):base for s,i,base,_ in layout}
    fixed_order=np.lexsort((pool[:,1],pool[:,0]))
    physical={}
    for arm in ("canonical","geometry"):
        raw=load_checked(folder/(arm+".txt"),pack["mot_sha256"][arm])
        actual=table(raw,10)
        check(actual.shape==(len(pool),10) and np.all(actual[:,7:]==-1),"Invalid physical 10-column MOT")
        ids=suppressed.copy()
        for key in pack["selectors"][arm]:
            base=parent_id[tuple(key)]
            ids[ranges[base]]=base
        expected=pool.copy()
        expected[:,1]=ids
        np.testing.assert_array_equal(actual[:,:7],expected[fixed_order],
            err_msg="Missing/reordered/duplicated box, changed score or incorrect selected/rejected/non-event ID")
        check(len(actual)==0 or np.all((actual[1:,0]>actual[:-1,0]) |
            ((actual[1:,0]==actual[:-1,0])&(actual[1:,1]>actual[:-1,1]))),"Physical order is not (frame,new ID)")
        physical[arm]=actual[:,:7]
    return physical


def sparse_rank(rows,gt_ids,edges):
    """Hopcroft-Karp on a newly restricted graph; independent of the Hungarian event matching."""
    if not rows or not gt_ids or not edges:
        return 0
    rid={r:i for i,r in enumerate(rows)}
    cid={g:i for i,g in enumerate(gt_ids)}
    ij=[(rid[p],cid[g]) for p,g in edges if p in rid]
    if not ij:
        return 0
    i,j=zip(*ij)
    a=csr_matrix((np.ones(len(i),dtype=np.uint8),(i,j)),shape=(len(rid),len(cid)))
    return int(np.count_nonzero(maximum_bipartite_matching(a,perm_type="column")>=0))


def scores(rows,truth,edges):
    tp=sparse_rank(rows,truth,edges)
    return [tp,len(rows)-tp,len(truth)-tp]


def count_metrics(proposed,truth):
    scalar=int(truth.sum())
    return {"count_in_out":(proposed.sum(axis=0)/30).tolist(),
            "bias_in_out":((proposed-truth).sum(axis=0)/30).tolist(),
            "w_percent":float(100*np.abs(proposed-truth).sum()/scalar) if scalar else None}


def check_group(folder,pack,streams,poly,gt):
    pool,layout,desc,ids,ranges=parent_pool(streams,poly)
    check(layout==pack["manifest"]["id_blocks"],"Parent reserved ID ranges differ")
    count_receipt=verify_counts(pack,desc)
    physical=physical_rows(folder,pack,pool,ids,ranges,layout)
    base={(s,i):b for s,i,b,_ in layout}
    selected={k:[base[tuple(x)] for x in pack["selectors"][k]] for k in ("canonical","geometry")}
    controls=[[base[tuple(k)] for k in row] for row in pack["selectors"]["controls"]]
    eligible=[[base[tuple(e["member"])] for e in block["evidence"]] for direction in ("in","out")
              for block in pack["manifest"]["blocks"][direction] if block["canonical"] is not None]
    slot={p:i for i,group in enumerate(eligible) for p in group}
    all_events=set(ranges)
    receipt={"counts":count_receipt,"rows":len(pool),"parents":len(layout),"native_events":len(all_events),
             "reserved_ids":sum(row[3] for row in layout),"thresholds":{}}
    for tau in (.5,.3):
        graph=archive_arm(score_arm(gt,pool,poly,tau))
        lookup={row[0]:row for row in graph["predictions"]}
        truth=[g for g,_ in graph["gt"]]
        edges=graph["edges"]
        stat={"pool_sha256":sha(wire(graph)),"pool_edges":len(edges),"native":{},"physical":{},"lean":{}}
        owners=frame_match(by_track(gt),pool,thr=tau) if len(gt) and len(pool) else np.full(len(pool),-1)
        start=0
        for s,native in sorted(streams.items()):
            native_graph=archive_arm(score_arm(gt,native,poly,tau))
            native_ids=[base[s,k[0]] for k in desc[s]]
            solo=frame_match(by_track(gt),by_track(native),thr=tau) if len(gt) and len(native) else np.full(len(native),-1)
            stat["native"][str(s)]={
                "standalone":scores([r[0] for r in native_graph["predictions"]],truth,native_graph["edges"]),
                "in_full_pool":scores(native_ids,truth,edges),
                "changed_box_owners":int(np.count_nonzero(solo!=owners[start:start+len(native)]))}
            start+=len(native)
        for arm,chosen in selected.items():
            keep=set(chosen)
            actual=archive_arm(score_arm(gt,physical[arm],poly,tau))
            check(actual["gt"]==graph["gt"] and actual["edges"]==[e for e in edges if e[0] in keep]
                  and actual["predictions"]==[lookup[p] for p in sorted(chosen)],
                  "Actual exported MOT differs from the projected full-frame ownership graph")
            stat["physical"][arm]=scores(chosen,truth,edges)
            lean=pool[~np.isin(pool[:,1],list(all_events-keep))]
            lean_graph=archive_arm(score_arm(gt,lean,poly,tau))
            stat["lean"][arm]=scores(chosen,truth,lean_graph["edges"])
        stat["controls"]=[scores(row,truth,edges) for row in controls]
        block_edges={(slot[p],g) for p,g in edges if p in slot}
        stat["ceiling_tp"]=sparse_rank(list(range(len(eligible))),truth,block_edges)
        receipt["thresholds"][str(tau)]=stat
    n=pack["meta"]["nframes"]
    gt_grid=bins([f for _,f in graph["gt"]],n)
    hist={s:bins([r[1] for r in records],n) for s,records in desc.items()}
    pooled=sum(hist.values())
    integer=bins([lookup[p][1] for p in selected["canonical"]],n)
    denominator=int(gt_grid.sum())
    receipt["window"]={"integer":count_metrics(integer,gt_grid),"e3":count_metrics(pooled/3,gt_grid),
                       "native":{str(s):count_metrics(v,gt_grid) for s,v in hist.items()},
                       "w_delta_bound_pp":float(100*abs(3*integer-pooled).sum()/(3*denominator)) if denominator else None}
    return receipt


def counts3(metrics):
    tp,fp,fn=[metrics[k] for k in ("tp","fp","fn")]
    check(abs(metrics["f1"]-2*tp/max(1,2*tp+fp+fn))<1e-12,"Reported F1 does not match TP/FP/FN")
    return [tp,fp,fn]


def compare(primary,receipt,pack):
    def close(a,b):
        if a is None or b is None:
            check(a==b,"Missing normalization differs")
        else:
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)
    win=receipt["window"]
    for k,item in [("integer",primary["integer"]),("e3",primary["e3"])]:
        for field,value in win[k].items():
            close(value,item[field])
    close(primary["w_delta_bound_pp"],win["w_delta_bound_pp"])
    i,e=win["integer"]["w_percent"],win["e3"]["w_percent"]
    close(primary["w_delta_vs_e3_pp"],None if i is None else i-e)
    for s,summary in win["native"].items():
        for field,value in summary.items():
            close(value,primary["native"][s]["count"][field])
    base={(s,i):b for s,i,b,_ in pack["manifest"]["id_blocks"]}
    blocks=[{tuple(e["member"]) for e in b["evidence"]} for d in ("in","out")
            for b in pack["manifest"]["blocks"][d] if b["canonical"] is not None]
    for tau,stat in receipt["thresholds"].items():
        graph=primary["pool_graphs"][tau]
        check(sha(wire(graph))==stat["pool_sha256"],"Primary full-competition graph differs")
        for k in ("canonical","geometry"):
            check(stat["physical"][k]==counts3(primary[k][tau]),"Primary physical event score differs")
            check(stat["lean"][k]==counts3(primary["lean"][k][tau]),"Primary lean score differs")
            selected={base[tuple(x)] for x in pack["selectors"][k]}
            for direction,sign in (("in",1),("out",-1)):
                pids=[row[0] for row in graph["predictions"] if row[0] in selected and row[1]*sign>0]
                tids=[g for g,f in graph["gt"] if f*sign>0]
                c=primary[k][tau]["by_direction"][direction]
                check(scores(pids,tids,graph["edges"])==[c[f] for f in ("tp","fp","fn")],
                      "Primary directional physical score differs")
        for s,record in stat["native"].items():
            native=primary["native"][s][tau]
            for k in ("standalone","in_full_pool"):
                check(record[k]==counts3(native[k]),"Primary native score differs")
            check(record["changed_box_owners"]==native["changed_box_owners"],"Competition count differs")
        check(stat["controls"]==[counts3(row[tau]) for row in primary["controls"]],"Primary quota controls differ")
        ceiling=primary["selector_ceiling"][tau]
        witness=[tuple(x) for x in ceiling["selected"]]
        check(len(witness)==len(blocks) and all(k in b for k,b in zip(witness,blocks)),
              "Selector ceiling witness is ineligible")
        got=scores([base[k] for k in witness],[g for g,_ in graph["gt"]],graph["edges"])
        check(got==counts3(ceiling) and stat["ceiling_tp"]==got[0],"Ceiling or attaining witness differs")


def run_case(role,folder,key,packsha,indexsha,lineage_dir,runs_dir,zones,archives):
    from inputs import read_group
    from score_e3 import ARCHIVES
    pack=json.loads(load_checked(folder/role/key/"freeze.json",packsha))
    meta=pack["meta"]
    streams,poly,expected=read_group(role,meta["fold"],meta["part"],meta["first_seed"],
                                    lineage_dir,runs_dir,zones)
    check(expected==meta,"Native stream provenance differs")
    archive=json.loads(gzip.decompress(load_checked(archives/f"study108_{role}.json.gz",ARCHIVES[role])))
    gt_sha=archive["labels"][key]["gt_mot_sha256"]
    gt=table(load_checked(runs_dir/"gt"/("dev" if meta["part"]=="head" else "val")/meta["clip"]/"gt/gt.txt",gt_sha),6)[:,:6]
    result=check_group(folder/role/key,pack,streams,poly,gt)
    dest=folder/"validation"/role/(key+".json")
    dest.parent.mkdir(parents=True,exist_ok=True)
    raw=wire({"role":role,"group":key,"prereg":pack["prereg"],"index_sha256":indexsha,
              "pack_sha256":packsha,"gt_sha256":gt_sha,"check":result})
    with TemporaryDirectory(dir=dest.parent) as tmp:
        record=Path(tmp)/"record.json"
        record.write_bytes(raw)
        try:
            os.link(record,dest)
        except FileExistsError:
            check(dest.read_bytes()==raw,"Previous verification differs")
    print(key,sha(raw),flush=True)


def self_check():
    from evaluate import synthetic_case,score_frozen
    from freeze_e3 import freeze_group
    streams,poly,meta,gt=synthetic_case()
    scene={s:np.array([[f,7,x,3,2,2,.8] for f,x in
                         ((1,11),(2,8),(3,11),(4,8),(30,8),(31,11),(32,8))],dtype=float)
           for s in range(3)}
    whole,layout,_,split_id,_=parent_pool(scene,poly)
    assert layout[:2]==[[0,7,1,4],[0,100007,5,3]]
    assert whole[:7,1].tolist()==[1,1,1,1,5,5,5]
    assert split_id[:7].tolist()==[1,2,3,4,5,5,5]
    with TemporaryDirectory() as tmp:
        folder=Path(tmp)/"group"
        freeze_group(streams,poly,meta,folder,"0"*40)
        pack=json.loads((folder/"freeze.json").read_bytes())
        report=check_group(folder,pack,streams,poly,gt)
        assert report["thresholds"]["0.5"]["physical"]=={"canonical":[0,2,2],"geometry":[1,1,1]}
        compare(score_frozen(streams,poly,meta,gt,pack),report,pack)
        for name in ("canonical","geometry"):
            original=(folder/(name+".txt")).read_bytes()
            bad=original.splitlines(keepends=True)
            bad[0],bad[1]=bad[1],bad[0]
            (folder/(name+".txt")).write_bytes(b"".join(bad))
            pack["mot_sha256"][name]=sha(b"".join(bad))
            try:
                check_group(folder,pack,streams,poly,gt)
            except (ValueError,AssertionError):
                pass
            else:
                raise AssertionError("Reordered physical frames passed")
            (folder/(name+".txt")).write_bytes(original)
            pack["mot_sha256"][name]=sha(original)
    print("Synthetic physical MOTs, 99 controls, independent count and selector ceilings verified.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    cmd=p.add_subparsers(dest="stage",required=True)
    cmd.add_parser("self-check")
    for sub in ("verify","compare"):
        opt=cmd.add_parser(sub)
        opt.add_argument("role",choices=("source","target"))
        opt.add_argument("folder",type=Path)
        opt.add_argument("--index-sha",required=True)
        opt.add_argument("--groups",nargs="+")
        if sub=="verify":
            opt.add_argument('--lineage-dir',type=Path,default=RUNS/'study108')
            opt.add_argument('--runs-dir',type=Path,default=RUNS)
            opt.add_argument('--zones',type=Path,default=ROOT/'beetrack/zones.json')
            opt.add_argument('--archives',type=Path,default=ROOT/'docs')
    a=p.parse_args()
    if a.stage=="self-check":
        self_check()
    else:
        index=json.loads(load_checked(a.folder/(a.role+"_index.json"),a.index_sha))
        expected={f"{fold}_{base}_{part}" for fold in (33,35) for base in (0,3,6)
                  for part in (("head","tail") if a.role=="source" else ("head",))}
        check(set(index["groups"])==expected and index["role"]==a.role,"Incomplete or wrong role index")
        keys=a.groups or sorted(expected)
        for key in keys:
            digest=index["groups"][key]
            if a.stage=="verify":
                run_case(a.role,a.folder,key,digest,a.index_sha,a.lineage_dir,a.runs_dir,a.zones,a.archives)
            else:
                receipt=json.loads((a.folder/"validation"/a.role/(key+".json")).read_bytes())
                pack=json.loads(load_checked(a.folder/a.role/key/"freeze.json",digest))
                summary=json.loads((a.folder/(a.role+"_scores.json")).read_bytes())
                primary=json.loads(load_checked(a.folder/"scores"/a.role/(key+".json"),summary["group_reports_sha256"][key]))
                check(receipt["index_sha256"]==primary["index_sha256"]==a.index_sha and
                      receipt["pack_sha256"]==primary["group_sha256"]==digest,"Case verification lineage differs")
                compare(primary["results"],receipt["check"],pack)
                print("Agrees",key,flush=True)
