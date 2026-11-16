"""Pack and independently replay the frozen E3 event selections and official event graphs."""
import argparse
from base64 import b64decode,b64encode
from fractions import Fraction
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import mean,median
import struct
import sys
from tempfile import TemporaryDirectory


TAUS=("0.5","0.3")
ARMS=("canonical","geometry")
ARCHIVES={"source":"2a7d8bf2d4b027dafa59d924e5b908ff2b88c0edbdf9fcded097c60383edd383",
          "target":"301993f5112951984cc747fb6a0466dc59bf92170ec383cbe9db7387dea9049c"}
F64_PREFIX="f64le:"
PREREG="33ae06e5646fe7eb03d7b4f02e7845c2fd3cefe6"
FROZEN={"source":"e5af5c611487d534f65e22425a489f0cb51f2c51f7583e560810289ac5365636",
        "target":"3ea1f7e2e55f0031277fa2cd3a03e97d742cecefe3ff197812d7291423c8f921"}
SCORES={"source":"dd0fac185dd6749cfa5566b8a591edc45480218e6a912b72af3e418a8e7105f8",
        "target":"772f4e444063c782ca3d052e4b54fd04590d229c35400f5df52e9e07bb486089"}


def numerical(record,decode=False):
    """Losslessly store scientific floating-point values while keeping all keys, counts and IDs visible."""
    if isinstance(record,dict):
        return {k:numerical(v,decode) for k,v in record.items()}
    if isinstance(record,list):
        return [numerical(v,decode) for v in record]
    if not decode and isinstance(record,float):
        require(math.isfinite(record),"Nonfinite scientific value")
        return F64_PREFIX+b64encode(struct.pack("<d",record)).decode("ascii")
    if decode and isinstance(record,str) and record.startswith(F64_PREFIX):
        raw=b64decode(record[len(F64_PREFIX):],validate=True)
        require(len(raw)==8,"Truncated binary64 value")
        x,=struct.unpack("<d",raw)
        require(math.isfinite(x),"Invalid binary64 value")
        return x
    return record


def load_archive(path):
    """Read the complete public archive as ordinary Python numbers for replay or plotting."""
    return numerical(read(Path(path)),decode=True)


def wire(record):
    return (json.dumps(record,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def digest(raw):
    return sha256(raw).hexdigest()


def require(ok,message):
    if not ok:
        raise ValueError(message)


def read(path,expected=None):
    raw=path.read_bytes()
    if expected is not None:
        require(digest(raw)==expected,"File receipt mismatch: "+path.name)
    return json.loads(gzip.decompress(raw) if path.suffix==".gz" else raw)


def save(path,record):
    raw=gzip.compress(wire(record),compresslevel=9,mtime=0)
    raw=raw[:9]+bytes([255])+raw[10:]
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        part=Path(tmp)/"archive"
        part.write_bytes(raw)
        try:
            os.link(part,path)
        except FileExistsError:
            require(path.read_bytes()==raw,"Different archive already exists at "+str(path))
    return digest(raw)


def specs(role):
    return [f"{f}_{s}_{p}" for f in (33,35) for p in (("head","tail") if role=="source" else ("head",))
            for s in (0,3,6)]


def close(actual,expected,what):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),"Different fields: "+what)
        for k in expected:
            close(actual[k],expected[k],what+"."+k)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),"Different row count: "+what)
        for i,(a,b) in enumerate(zip(actual,expected)):
            close(a,b,f"{what}[{i}]")
    elif isinstance(expected,float):
        require(isinstance(actual,(float,int)) and math.isclose(actual,expected,rel_tol=0,abs_tol=1e-10),
                "Numerical score mismatch: "+what)
    else:
        require(actual==expected,"Value mismatch: "+what)


def evidence(group,report,history):
    """Remove repeated GT streams and box tables; preserve exact parent-identity graph edges."""
    result=report["results"]
    baseline=result["pool_graphs"]["0.5"]
    gt,preds=baseline["gt"],baseline["predictions"]
    compare=group["meta"]
    require(compare==report["meta"],"Frozen case differs from scored case")
    standalone={}
    for seed in compare["seeds"]:
        key=f"{compare['fold']}_{seed}_{compare['part']}"
        label=history["labels"][key]
        native=history["masks"]["predictions"][key]
        desc=[item["row"] for item in group["manifest"]["observations"] if item["seed"]==seed]
        require(label["gt"]==gt and native["predictions"]==desc and
                native["mot_sha256"]==compare["mot_sha256"][str(seed)] and
                label["gt_mot_sha256"]==report["gt_sha256"] and
                label["predictions_sha256"]==compare["prediction_sha256"][str(seed)],
                "Historical native identity or source fingerprint mismatch: "+key)
        standalone[str(seed)]={t:label[k] for t,k in zip(TAUS,("edges05","edges03"))}
    pooled,lean={},{}
    for t in TAUS:
        graph=result["pool_graphs"][t]
        require(graph["gt"]==gt and graph["predictions"]==preds,"Pooled descriptor table differs across IoUs")
        pooled[t]=graph["edges"]
    layout={(s,i):base for s,i,base,_ in group["manifest"]["id_blocks"]}
    for arm in ARMS:
        ids={layout[tuple(k)] for k in group["selectors"][arm]}
        filtered=[r for r in preds if r[0] in ids]
        lean[arm]={}
        for t in TAUS:
            graph=result["lean_graphs"][arm][t]
            require(graph["gt"]==gt and graph["predictions"]==filtered,"Lean control changed the event set")
            lean[arm][t]=graph["edges"]
    graphdata=dict(pool=pooled,lean=lean,native=standalone)
    stored={k:v for k,v in result.items() if k not in ("pool_graphs","lean_graphs")}
    return gt,dict(freeze=group,report_meta={k:v for k,v in report.items() if k not in ("results","contrasts")},
                   pool_predictions=preds,graphs=graphdata,reported=stored)


def restore_report(cell,gt,contrasts):
    result=dict(cell["reported"])
    pack=cell["freeze"]
    layout={(s,i):base for s,i,base,_ in pack["manifest"]["id_blocks"]}
    predictions=cell["pool_predictions"]
    pool=cell["graphs"]["pool"]
    result["pool_graphs"]={t:dict(gt=gt,predictions=predictions,edges=pool[t]) for t in TAUS}
    result["lean_graphs"]={}
    for arm in ARMS:
        ids={layout[tuple(k)] for k in pack["selectors"][arm]}
        keep=[row for row in predictions if row[0] in ids]
        result["lean_graphs"][arm]={t:dict(gt=gt,predictions=keep,edges=cell["graphs"]["lean"][arm][t])
                                    for t in TAUS}
    return dict(cell["report_meta"],results=result,contrasts=contrasts)


def pack(args):
    clips,roles={},{}
    for role in ("source","target"):
        index_sha=getattr(args,role+"_index")
        scores_sha=getattr(args,role+"_scores")
        index=read(args.run/(role+"_index.json"),index_sha)
        scores=read(args.run/(role+"_scores.json"),scores_sha)
        history=read(args.archives/f"study108_{role}.json.gz",ARCHIVES[role])
        require(index_sha==FROZEN[role] and scores_sha==SCORES[role] and index["prereg"]==PREREG
                and set(index["groups"])==set(specs(role))==set(scores["group_reports_sha256"])
                and scores["role"]==index["role"]==role and scores["index_sha256"]==index_sha
                and scores["prereg"]==index["prereg"] and scores["archive108_sha256"]==ARCHIVES[role],
                "Role manifests or historical receipts differ")
        groups={}
        for key in specs(role):
            frozen=read(args.run/role/key/"freeze.json",index["groups"][key])
            report=read(args.run/"scores"/role/(key+".json"),scores["group_reports_sha256"][key])
            gt,cell=evidence(frozen,report,history)
            clip=frozen["meta"]["clip"]
            require(clips.setdefault(clip,gt)==gt,"GT event identities changed across groups for "+clip)
            groups[key]=cell
        roles[role]=dict(index=index,index_sha256=index_sha,scores=scores,scores_sha256=scores_sha,groups=groups)
    payload=dict(study=115,format=1,clips=clips,roles=roles)
    return save(args.archive,numerical(payload))


def numeric_view(scored):
    """Alternative maximum matches may use different IDs while attaining identical metrics."""
    if isinstance(scored,dict):
        return {k:numeric_view(v) for k,v in scored.items() if k not in ("matching","selected","block_gt_matching")}
    if isinstance(scored,list):
        return [numeric_view(v) for v in scored]
    return scored


def paired(r):
    native=list(r["native"].values())
    average=lambda xs: None if any(x is None for x in xs) else mean(xs)
    out={"count":dict(e3_w_percent=r["e3"]["w_percent"],integer_w_percent=r["integer"]["w_percent"],
          standalone_mean_w_percent=average([m["count"]["w_percent"] for m in native]),
          integer_minus_e3_pp=r["w_delta_vs_e3_pp"],max_abs_delta_pp=r["w_delta_bound_pp"])}
    for t in TAUS:
        row={}
        for k in ("tp","f1"):
            a,b=(r[arm][t][k] for arm in ARMS)
            solo=mean(m[t]["standalone"][k] for m in native)
            pool=mean(m[t]["in_full_pool"][k] for m in native)
            lean_a,lean_b=(r["lean"][arm][t][k] for arm in ARMS)
            mid=median(v[t][k] for v in r["controls"])
            cap=r["selector_ceiling"][t][k]
            row[k]=dict(canonical=a,geometry=b,controls_median=mid,standalone_mean=solo,
                        standalone_in_pool_mean=pool,lean_canonical=lean_a,lean_geometry=lean_b,
                        geometry_minus_canonical=b-a,geometry_minus_controls_median=b-mid,
                        canonical_minus_controls_median=a-mid,geometry_minus_native_mean=b-solo,
                        selector_ceiling=cap,ceiling_minus_geometry=cap-b,ceiling_minus_canonical=cap-a,
                        lean_minus_complete_geometry=lean_b-b,lean_minus_complete_canonical=lean_a-a)
        row["changed_box_owners_mean"]=mean(m[t]["changed_box_owners"] for m in native)
        out[t]=row
    return out


def tree_mean(items):
    first=items[0]
    if isinstance(first,dict):
        return {k:tree_mean([x[k] for x in items]) for k in first}
    return None if any(x is None for x in items) else mean(items)


def descriptors(pack):
    d={s:[] for s in pack["meta"]["seeds"]}
    for row in pack["manifest"]["observations"]:
        d[row["seed"]].append(row["row"])
    return d


def time_details(frozen,scores,gt):
    """Directional integer, fractional and signed difference for each second and official window."""
    frames={(r["seed"],r["row"][0]):r["row"][1] for r in frozen["manifest"]["observations"]}
    selected=[frames[tuple(k)] for k in frozen["selectors"]["canonical"]]
    def bins(values):
        rows=[[0,0] for _ in range((frozen["meta"]["nframes"]+24)//25)]
        for f in values:
            rows[(abs(f)-1)//25][int(f<0)]+=1
        return rows
    def thirds(observed,native):
        return [[3*a-c,3*b-d] for (a,b),(c,d) in zip(observed,native)]
    integer,pooled=bins(selected),bins(frames.values())
    proof=frozen["counts_proof"]
    return dict(seconds=dict(integer=integer,pooled_thirds=pooled,gt=bins(f for _,f in gt),
                             integer_minus_e3_thirds=thirds(integer,pooled)),
                windows=dict(integer=proof["integer_window_counts"],pooled_thirds=proof["pooled_window_counts"],
                             integer_minus_e3_thirds=thirds(proof["integer_window_counts"],proof["pooled_window_counts"])),
                full_clip_directional_absolute_error={k:list(map(abs,scores[k]["bias_in_out"])) for k in ("integer","e3")})


def verify_group(cell,gt):
    from graph115 import score_graphs,check_witness
    from verify_e3_counts import verify_counts
    f=cell["freeze"]
    d=descriptors(f)
    freeze_stats=verify_counts(f,d)
    quota=f["quota"]["pool_sizes"]
    canon,geom=(f["selectors"][k] for k in ARMS)
    freeze_stats["groups_with_multiple_same_second_members"]=sum(n>1 for n in quota)
    freeze_stats["geometry_changed_seed"]=sum(a[0]!=b[0] for a,b in zip(canon,geom))
    r=cell["reported"]
    graf=dict(cell["graphs"],gt=gt)
    graf["changed_box_owners"]={s:{t:m[t]["changed_box_owners"] for t in TAUS} for s,m in r["native"].items()}
    for src,dest in (("selected","oracle_selections"),("matching","oracle_matchings"),
                     ("block_gt_matching","oracle_block_matchings")):
        graf[dest]={t:r["selector_ceiling"][t][src] for t in TAUS}
    scores=score_graphs(f,graf)
    score_view={k:v for k,v in scores.items() if k!="offset_proof"}
    close(score_view,numeric_view(r),"group metrics")
    layout={(s,i):base for s,i,base,_ in f["manifest"]["id_blocks"]}
    signed={layout[s,row[0]]:row[1] for s,rows in d.items() for row in rows}
    require(signed=={row[0]:row[1] for row in cell["pool_predictions"]},"Pooled ID/frame disagrees with freeze")
    for t in TAUS:
        for arm in ARMS:
            ids={layout[tuple(k)] for k in f["selectors"][arm]}
            for m,edges in ((r[arm][t],graf["pool"][t]),(r["lean"][arm][t],graf["lean"][arm][t])):
                check_witness(m["matching"],edges,ids,m["tp"])
        for s,rows in d.items():
            seed=str(s)
            ids={row[0] for row in rows}
            for name,edges,mask in (("standalone",graf["native"][seed][t],ids),
                                    ("in_full_pool",graf["pool"][t],{layout[s,i] for i in ids})):
                m=r["native"][seed][t][name]
                check_witness(m["matching"],edges,mask,m["tp"])
    return scores,freeze_stats


def fold_gate(rows):
    def f1(row):
        return Fraction(2*row["tp"],max(1,2*row["tp"]+row["fp"]+row["fn"]))
    f1_gains=[]
    w_gaps=[]
    beats=0
    for r in rows:
        g,a=(r[k]["0.5"] for k in ("geometry","canonical"))
        f1_gains.append(f1(g)-f1(a))
        beats += int(g["tp"]>median(v["0.5"]["tp"] for v in r["controls"]))
        p=r["offset_proof"]
        g_count=sum(r["gt_in_out"])
        if g_count:
            num=lambda k:sum(map(sum,p[k]["abs_error_in_out_by_offset"]))
            w_gaps.append(Fraction(100*(3*num("integer")-num("e3")),90*g_count))
    mean_f1=mean(f1_gains)
    positive=sum(g>Fraction(1,10**9) for g in f1_gains)
    w=mean(w_gaps) if len(w_gaps)==len(rows) else None
    return dict(mean_f1_gain=float(mean_f1),f1_positive_groups=positive,above_quota_median_groups=beats,
                selector_pass=(mean_f1>=Fraction(2,100) and positive>=2 and beats>=2),
                mean_integer_minus_e3_pp=None if w is None else float(w),count_pass=w is not None and w<=2)


def verify(payload):
    require(payload["study"]==115 and payload["format"]==1 and set(payload["roles"])==set(ARCHIVES),
            "Expected both registered roles")
    gates,details={},{}
    for role,bundle in payload["roles"].items():
        scores,index=bundle["scores"],bundle["index"]
        require(index["prereg"]==PREREG and digest(wire(index))==bundle["index_sha256"]==scores["index_sha256"]==FROZEN[role] and
                digest(wire(scores))==bundle["scores_sha256"]==SCORES[role] and scores["archive108_sha256"]==ARCHIVES[role]
                and set(bundle["groups"])==set(index["groups"])==set(scores["groups"])==set(specs(role)),
                "Role scores, predictions or historical fingerprint mismatch: "+role)
        summaries,full,parts={},{},{}
        for key in specs(role):
            cell=bundle["groups"][key]
            frozen=cell["freeze"]
            meta=frozen["meta"]
            fid,base,part=key.split("_")
            expected_hive=68-int(fid) if role=="source" else int(fid)
            require(meta["role"]==role and (meta["fold"],meta["first_seed"],meta["part"])==(int(fid),int(base),part)
                    and meta["hive"]==expected_hive and meta["clip"]==f"BEE24-{expected_hive}-{part}"
                    and meta["seeds"]==list(range(int(base),int(base)+3))
                    and meta["nframes"]==(3950 if part=="head" else 1000)
                    and frozen["prereg"]==index["prereg"]==scores["prereg"]
                    and digest(wire(frozen))==index["groups"][key],"Freeze receipt mismatch: "+role+"/"+key)
            gt=payload["clips"][meta["clip"]]
            report=restore_report(cell,gt,scores["groups"][key])
            require(digest(wire(report))==scores["group_reports_sha256"][key] and
                    report["group_sha256"]==index["groups"][key] and report["index_sha256"]==bundle["index_sha256"]
                    and report["archive108_sha256"]==ARCHIVES[role],
                    "Score reconstruction digest mismatch: "+role+"/"+key)
            observed,receipt=verify_group(cell,gt)
            summaries[key]=paired(observed)
            close(summaries[key],scores["groups"][key],role+"/"+key+" contrasts")
            parts.setdefault(f"no{fid}_hive{expected_hive}_{part}",[]).append(summaries[key])
            full[key]=observed
            details[role+"/"+key]=dict(predictions=receipt,offsets=observed["offset_proof"],
                                       time=time_details(frozen,observed,gt))
        close({k:tree_mean(v) for k,v in parts.items()},scores["equal_group_means"],role+" equal group means")
        close(tree_mean(list(summaries.values())),scores["equal_group_mean_all"],role+" overall descriptive mean")
        if role=="target":
            gates={str(f):fold_gate([full[f"{f}_{s}_head"] for s in (0,3,6)]) for f in (33,35)}
    return gates,details


def print_table(payload):
    print("role hive fold part seeds | W E3 integer delta bound(pp) | F1 native canonical geometry quota oracle lean (.5)")
    for role,bundle in payload["roles"].items():
        for key in specs(role):
            r=bundle["scores"]["groups"][key]
            m=bundle["groups"][key]["freeze"]["meta"]
            w,f=r["count"],r["0.5"]["f1"]
            values=[w["e3_w_percent"],w["integer_w_percent"],w["integer_minus_e3_pp"],w["max_abs_delta_pp"],
                    f["standalone_mean"],f["canonical"],f["geometry"],f["controls_median"],f["selector_ceiling"],
                    f["lean_geometry"]]
            nums=" ".join("NA" if x is None else f"{x:.4f}" for x in values)
            print(f"{role} {m['hive']} {m['fold']} {m['part']} {m['seeds']} | {nums}")


def main():
    for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[var]="1"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir",type=Path,help="Optional directory containing verify_e3_counts.py")
    commands=parser.add_subparsers(dest="cmd",required=True)
    make=commands.add_parser("pack")
    make.add_argument("run",type=Path)
    make.add_argument("archive",type=Path)
    make.add_argument("--archives",type=Path,default=Path("docs"))
    for role in ("source","target"):
        for stage in ("index","scores"):
            make.add_argument(f"--{role}-{stage}",required=True)
    rep=commands.add_parser("replay")
    rep.add_argument("archive",type=Path)
    rep.add_argument("--table",action="store_true")
    rep.add_argument("--details",type=Path,help="Write exact offset numerators and recomputed gates to JSON")
    args=parser.parse_args()
    if args.engine_dir:
        sys.path.insert(0,str(args.engine_dir.resolve()))
    if args.cmd=="pack":
        print(args.archive.name,pack(args))
    else:
        payload=load_archive(args.archive)
        gates,details=verify(payload)
        print(json.dumps(gates,indent=2))
        joint={k:all(r[k] for r in gates.values()) for k in ("selector_pass","count_pass")}
        print("Both held-out hives",json.dumps(joint))
        if args.table:
            print_table(payload)
        if args.details:
            require(args.details != args.archive,"Report and archive paths must differ")
            with args.details.open("xb") as out:
                out.write(wire(dict(gates=gates,both_hives=joint,groups=details)))


if __name__=="__main__":
    main()
