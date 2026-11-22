"""Pure §210 disjoint, no-common-date whole-run unions and native scoring.

K retains the original208/209 rows bit-for-bit. A nonempty union configuration
uses immutable original canonical IDs, maps each selected pair to its smaller
ID, and stably sorts dates. Every original row survives exactly once. No GT
is accepted by slate construction, model ranking or materialization.
"""
import os
for _name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import importlib.util
import json
from pathlib import Path
import numpy as np

ARMS=("K","N","G","X","M","ALL")
MODELS=("G","X","M")
_spec=importlib.util.spec_from_file_location("interleave210_physical_base",Path(__file__).parents[1]/"study209"/"physical209.py")
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
plain,file_sha,verified_manifest=base.plain,base.file_sha,base.verified_manifest
native=base.native_base


def pair_table(pairs):
    table={}
    for p in pairs:
        ids=p["canonical_ids"]
        if (len(ids)!=2 or any(not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_)) for i in ids)
                or not 0<ids[0]<ids[1] or p["pair_id"]!=f"{ids[0]}|{ids[1]}" or p["pair_id"] in table):
            raise ValueError("Unique numeric original canonical pair A<B required")
        delta=np.asarray(p["delta_windows"])
        if delta.ndim!=2 or delta.shape[1]!=2 or not np.issubdtype(delta.dtype,np.integer):
            raise ValueError("Frozen signed integer full30phase pair delta required")
        if not isinstance(p["active"],(bool,np.bool_)) or bool(p["active"])!=bool(delta.any()):
            raise ValueError("Active flag must mean a nonzero frozen window vector")
        if not np.isfinite(p["min_offset_norm"]) or not 0<=p["min_offset_norm"]<=1:
            raise ValueError("Eligible local offset norm must lie in[0,1]")
        table[p["pair_id"]]=p
    return table


def fixed_slate(pairs):
    """All eligible pairs: active first, smaller norm, numeric IDs; disjoint."""
    pair_table(pairs)
    ordered=sorted(pairs,key=lambda p:(not p["active"],float(p["min_offset_norm"]),*p["canonical_ids"]))
    used=set();slate=[];suppressed=[]
    for p in ordered:
        ids=set(p["canonical_ids"])
        if used&ids: suppressed.append(p["pair_id"])
        else: slate.append(p["pair_id"]);used.update(ids)
    table=pair_table(pairs)
    active=[key for key in slate if table[key]["active"]]
    a=len(active)
    return dict(slate=slate,active_slate=active,a=a,k=(a+3)//4 if a else 0,suppressed=suppressed)


def select_states(pairs,model_scores):
    """Fixed intervention budget; zero/fallback G logits still select k."""
    if set(model_scores)!=set(MODELS): raise ValueError("Complete G/X/M frozen scores required")
    table=pair_table(pairs);audit=fixed_slate(pairs);active=audit["active_slate"];k=audit["k"]
    selected=dict(K=[],N=active[:k],ALL=list(active))
    for arm in MODELS:
        if not set(active)<=model_scores[arm].keys() or any(not np.isfinite(model_scores[arm][key]) for key in active):
            raise ValueError("Finite model/fallback score for every active-slate pair required")
        selected[arm]=sorted(active,key=lambda key:(-float(model_scores[arm][key]),*table[key]["canonical_ids"]))[:k]
    return dict(audit,selected={arm:selected[arm] for arm in ARMS})


def materialize(rows,canon,pairs,selected_ids):
    """No deletion, no interpolation and no merge without truly disjoint dates."""
    rows,canon,runs=base.original_runs(rows,canon)
    table=pair_table(pairs);ids=list(selected_ids)
    if any(not isinstance(k,str) for k in ids) or len(ids)!=len(set(ids)) or not set(ids)<=table.keys():
        raise ValueError("Complete unique frozen selected pair IDs required")
    if not ids:
        return rows.copy(),dict(selected_pair_ids=[],actions=[],output_raw_indices=list(range(len(rows))),
            deleted_raw_indices=[],all_original_rows_once=True,original_K_exact=True,
            immutable_original_canonical_mapping=True)
    out=rows.copy();out[:,1]=canon
    used=set();actions=[]
    for key in sorted(ids,key=lambda key:tuple(table[key]["canonical_ids"])):
        p=table[key];a,b=map(int,p["canonical_ids"])
        if a not in runs or b not in runs or used&{a,b}: raise ValueError("Foreign/reused endpoint; no multiway chains")
        ia,ib=runs[a],runs[b]
        if np.intersect1d(rows[ia,0],rows[ib,0]).size:
            raise ValueError("Whole-run union has common native dates")
        ix=np.r_[ia,ib]
        ix=ix[np.argsort(rows[ix,0],kind="stable")]
        if np.any(np.diff(rows[ix,0])>25): raise ValueError("Union would split again under official >25frame gap rule")
        out[ix,1]=a
        used.update((a,b))
        actions.append(dict(pair_id=key,canonical_ids=[a,b],merged_canonical_id=a,
            original_raw_indices_A=ia.tolist(),original_raw_indices_B=ib.tolist(),merged_date_order_raw_indices=ix.tolist(),
            no_common_dates=True,no_rows_deleted=True,official_gap_partition_consistent=True,
            delta_windows=p["delta_windows"]))
    order=np.lexsort((np.arange(len(rows)),out[:,0]))
    physical=out[order]
    if (len(order)!=len(rows) or not np.array_equal(np.sort(order),np.arange(len(rows)))
            or physical[:,[0,2,3,4,5,6]].tobytes()!=rows[order][:,[0,2,3,4,5,6]].tobytes()):
        raise AssertionError("Original frame/box/score row bijection failed")
    if len(np.unique(physical[:,:2],axis=0))!=len(physical): raise AssertionError("Duplicate native track/date after union")
    # Original unselected runs were already checked; selected unions were
    # checked above. Every output ID is an immutable, collision-free canonical
    # ID, so no legacy raw-ID gap ordinal can silently change another run.
    return physical,dict(selected_pair_ids=sorted(ids),actions=actions,output_raw_indices=order.tolist(),
        deleted_raw_indices=[],all_original_rows_once=True,immutable_original_canonical_mapping=True,
        original_K_exact=False,date_order="frame ascending, original raw index ascending at ties",
        ID_policy="original canonical IDs for every row; selected pair maps whole runs to smaller canonical ID")


def evaluate_clip(gt,rows,canon,pairs,selections,polygon,nframes,*,score_native,reference_metrics,
                  signature,out=None,write_mot=None):
    """Independent GT entry after the common prediction closure; full native F1."""
    import re
    if (not re.fullmatch("[0-9a-f]{40}",signature.get("prereg",""))
            or set(signature.get("folds",[]))!={"35","33","W","G"}
            or signature.get("heldout") not in ("35","33","W","G") or not signature.get("clip")
            or any(not re.fullmatch("[0-9a-f]{64}",signature.get(name,"")) for name in
                ("input208_manifest_sha256","candidate_manifest_sha256",
                 "label_manifest_sha256","prediction_manifest_sha256","reference209_score_seal"))):
        raise ValueError("Prereg/four-fold/complete artifact signature required")
    if set(selections)!=set(ARMS) or selections["K"]: raise ValueError("Exactly six states with original K required")
    table=pair_table(pairs)
    if out is not None:
        out=Path(out)
        if out.exists(): raise FileExistsError("Immutable native clip output already exists")
        if write_mot is None: raise ValueError("Exact native MOT writer required")
        out.mkdir(parents=True)
    result=dict(schema="interleave210-physical-clip-v1",signature=plain(signature),physical_code_sha256=file_sha(__file__),states={})
    baseline=None
    for arm in ARMS:
        physical,proof=materialize(rows,canon,pairs,selections[arm])
        if out is not None:
            folder=out/arm;folder.mkdir();write_mot(physical,folder/"mot.txt")
            reread=np.loadtxt(folder/"mot.txt",delimiter=",",usecols=range(7),ndmin=2) if len(physical) else np.empty((0,7))
            if reread.tobytes()!=physical.tobytes(): raise AssertionError("Exact physical MOT roundtrip changed rows")
            physical=reread
            np.save(folder/"output_raw_indices.npy",np.asarray(proof["output_raw_indices"],np.int64),allow_pickle=False)
        scored=native.score_physical(gt,physical,polygon,nframes,score_native)
        metric=scored["metrics"]
        if arm=="K":
            native.assert_baseline(metric,reference_metrics)
            if plain(metric)!=plain(reference_metrics): raise ValueError("Full K official evidence differs from209")
            baseline=np.asarray(metric["pred_bins"],np.int64)
        expected=baseline.copy()
        for key in selections[arm]:
            delta=np.asarray(table[key]["delta_windows"],np.int64)
            if delta.shape!=expected.shape: raise ValueError("Candidate delta clock/shape differs")
            expected+=delta
        np.testing.assert_array_equal(expected,np.asarray(metric["pred_bins"],np.int64),err_msg="FullMOT windows != globalK + sum frozen pair delta")
        counts={t:dict(tp=int(metric["iou"][t]["tp"]),fp=int(metric["iou"][t]["pred"]-metric["iou"][t]["tp"]),
            fn=int(metric["iou"][t]["gt"]-metric["iou"][t]["tp"]),pred=int(metric["iou"][t]["pred"]),
            gt=int(metric["iou"][t]["gt"]),f1=float(metric["iou"][t]["f1"])) for t in ("0.5","0.3")}
        record=dict(scored,proof=proof,selected_pair_ids=sorted(selections[arm]),selected_pairs=len(selections[arm]),
            nG=int(metric["nG"]),nP=int(metric["nP"]),TP_FP_FN=counts,all_window_delta_closure_exact=True)
        if out is not None:
            (folder/"score.json").write_text(json.dumps(plain(record),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
            record=dict(numerator=scored["numerator"],denominator=scored["denominator"],W=scored["W"],iou=counts,
                nG=record["nG"],nP=record["nP"],selected_pair_ids=record["selected_pair_ids"],selected_pairs=len(selections[arm]),
                path=arm,all_window_delta_closure_exact=True)
        result["states"][arm]=record
    if out is not None:
        (out/"summary.json").write_text(json.dumps(plain(result),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
        manifest={p.relative_to(out).as_posix():file_sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        (out/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
        (out/"SEALED.sha256").write_text(file_sha(out/"manifest.json")+"\n")
    return result


def stage_guard(root,prereg):
    """Verify all blind lists and numeric model/slate replay before target GT."""
    import fit210
    root=Path(root)
    if not all((root/k/"SEALED.sha256").is_file() for k in ("catalog","labels","predictions")):
        raise ValueError("All catalog/labels/common four-fold prediction seals required")
    closures={k:verified_manifest(root/k) for k in ("catalog","labels","predictions")}
    loaded=fit210.load_predictions(root/"predictions")
    receipt=loaded["receipt"]
    if (receipt["prereg"]!=prereg or receipt["candidate_manifest_sha256"]!=closures["catalog"][0]
            or receipt["label_manifest_sha256"]!=closures["labels"][0]
            or receipt["input208_manifest_sha256"]!="1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94"):
        raise ValueError("Blind prediction/catalog/label/source208 lineage differs")
    cr=dict(manifest_sha256=closures["catalog"][0],files=closures["catalog"][1])
    numeric,slates,support=fit210.load_numeric(root/"catalog",cr,prereg)
    catalog=json.loads((root/"catalog/deployment.json").read_text())
    predictions=loaded["predictions"]
    specs={("35","BEE24-35-head",3),("33","BEE24-33-head",3),("W","BEE24-10",0),
           ("W","BEE24-15",0),("G","BEE24-24",0),("G","BEE24-28",0)}
    if len(catalog["clips"])!=6 or {(r["domain"],r["clip"],r["seed"]) for r in catalog["clips"]}!=specs:
        raise ValueError("Exactly the original six fixed members required")
    if set(predictions["arms"])!=set(ARMS) or len(predictions["clips"])!=6:
        raise ValueError("All four-fold six-state blind lists required")
    models={}
    for h in ("35","33","W","G"):
        f=predictions["folds"][h]
        model=json.loads((root/"predictions"/f["model_file"]).read_text())
        if model["heldout"]!=h or set(model["source_domains"])!={"35","33","W","G"}-{h}:
            raise ValueError("Model heldout/source boundary differs")
        for field,head in (("source_counts","G"),("full_support_source_counts","correction")):
            counts=model[field]
            if set(counts)!={"35","33","W","G"}-{h}: raise ValueError("Source class domain whitelist differs")
            both=all(sum(v[c] for v in counts.values())>0 for c in ("S","D"))
            if (model[head]["status"]=="fitted")!=both: raise ValueError("Missing-class zero-logit/correction fallback differs")
        models[h]=model
    audits={}
    for (domain,clip),slate in slates.items():
        rows=[r for r in numeric if (r["domain"],r["clip"])==(domain,clip)]
        stored=predictions["clips"][domain+'/'+clip]
        expected=dict(domain=domain,clip=clip,**fit210.deploy_clip(rows,models[domain],slate))
        if plain(stored)!=plain(expected): raise ValueError("Frozen all-target logits/wrong-sign/support/budget replay differs")
        ref=next(r for r in catalog["clips"] if (r["domain"],r["clip"])==(domain,clip))
        data=base.read_gz(root/"catalog"/ref["path"]/"catalog.json.gz")
        scores={a:{r["pair_id"]:r["logits"][a] for r in stored["records"]} for a in MODELS}
        replay=select_states(data["pairs"],scores)
        if (replay["slate"]!=stored["slate"] or replay["active_slate"]!=stored["active_slate"]
                or replay["a"]!=stored["a"] or replay["k"]!=stored["k"]
                or any(replay["selected"][a]!=stored["selected"][a] for a in ARMS)):
            raise ValueError("Independent active-first/min-local-offset fixed slate/ranking differs")
        for r in stored["records"]:
            if not r["full_support"] or models[domain]["correction"]["status"]!="fitted":
                if any(np.float64(r["logits"][a]).tobytes()!=np.float64(r["logits"]["G"]).tobytes() for a in ("X","M")):
                    raise ValueError("Unsupported X/M must be exactly frozen G, not refitted/abstained")
        audits[domain+'/'+clip]=replay
    return dict(seals={k:r[0] for k,r in closures.items()},catalog=catalog,predictions=loaded,
                models=models,audits=audits,support=support)


def run_stage(args):
    """Explicit sole-CPU field entry; model/feature construction never occurs."""
    import gc
    import re
    import subprocess
    import sys
    import tempfile
    root,repo=args.root.resolve(),args.repo.resolve()
    if root.drive.upper()!="D:" or not re.fullmatch("[0-9a-f]{40}",args.prereg or ""):
        raise ValueError("D: root and full published prereg SHA required")
    subprocess.run(["git","merge-base","--is-ancestor",args.prereg,"origin/main"],cwd=repo,check=True,capture_output=True)
    text=subprocess.run(["git","show",args.prereg+":docs/experiment_log.md"],cwd=repo,check=True,capture_output=True).stdout.decode("utf8")
    if text.count("[alternating-offset-v1]")!=1: raise ValueError("Unique published210 marker required")
    closed=stage_guard(root,args.prereg)
    input208=args.inputs208.resolve();inputsha=file_sha(input208/"manifest.json")
    if (inputsha!="1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94"
            or (input208/"SEALED.sha256").read_text().strip()!=inputsha): raise ValueError("Wrong original208 source")
    input_manifest=json.loads((input208/"manifest.json").read_text())
    reference_sha,_=verified_manifest(args.reference209_score)
    if reference_sha!="3670fa9c892c8d1a918adfdc253013bf6837b0685897befb1f66af39e5e13381":
        raise ValueError("Wrong original209 K physical reference")
    # All four folded blind lists were decoded and replayed above before this
    # first target-evidence/GT metadata entry.
    label_top=json.loads((root/"labels/deployment.json").read_text())
    if label_top["prereg"]!=args.prereg or label_top["candidate_manifest_sha256"]!=closed["seals"]["catalog"]:
        raise ValueError("Target labels refer to different immutable candidates")
    sys.path[:0]=[str(repo),str(repo/"tools/study197")]
    from tools.study195.run import score_member_native,_gt_provider
    from tools.study195.freeze import write_mot_exact
    out=root/"score"
    if (out/"SEALED.sha256").exists(): raise FileExistsError("Completed210 score already sealed")
    out.mkdir(parents=True,exist_ok=True)
    identity=dict(prereg=args.prereg,input208_manifest_sha256=inputsha,artifact_seals=closed["seals"],
        reference209_score_seal=reference_sha,physical_code_sha256=file_sha(__file__),
        inherited209_code_sha256=file_sha(base.__file__),inherited208_code_sha256=file_sha(native.__file__))
    running=out/"RUNNING.json"
    if running.exists() and json.loads(running.read_text())!=identity: raise ValueError("Different scoring round cannot resume")
    running.write_text(json.dumps(identity,sort_keys=True,separators=(",",":"))+"\n")
    clips=[]
    for ref in closed["catalog"]["clips"]:
        domain,clip=ref["domain"],ref["clip"]
        data=base.read_gz(root/"catalog"/ref["path"]/"catalog.json.gz");meta=data["metadata"]
        signature=dict(prereg=args.prereg,folds=["35","33","W","G"],heldout=domain,clip=clip,
            input208_manifest_sha256=inputsha,candidate_manifest_sha256=closed["seals"]["catalog"],
            label_manifest_sha256=closed["seals"]["labels"],prediction_manifest_sha256=closed["seals"]["predictions"],
            reference209_score_seal=reference_sha,numeric_features_embedded_in_catalog=True)
        dest=out/"clips"/domain/clip
        if dest.exists():
            verified_manifest(dest);result=json.loads((dest/"summary.json").read_text())
            if result["signature"]!=signature or result["physical_code_sha256"]!=file_sha(__file__):
                raise ValueError("Cached clip input/physical code signature differs")
        else:
            source=(input208/meta["source208_path"]).resolve()
            if not source.is_relative_to(input208): raise ValueError("Raw208 source escapes original closure")
            for name,key in (("rows.npy","rows_sha256"),("canon.npy","canon_sha256")):
                rel=(source/name).relative_to(input208).as_posix()
                if file_sha(source/name)!=meta[key] or input_manifest[rel]!=meta[key]: raise ValueError("Original208 raw/canonical bytes differ")
            rows=np.load(source/"rows.npy",mmap_mode="r",allow_pickle=False)
            canon=np.load(source/"canon.npy",mmap_mode="r",allow_pickle=False)
            prior=args.reference209_score/"clips"/domain/clip/"K"
            historical=np.loadtxt(prior/"mot.txt",delimiter=",",usecols=range(7),ndmin=2) if len(rows) else np.empty((0,7))
            if historical.tobytes()!=rows.tobytes(): raise ValueError("Original K raw rows differ from209 physical baseline")
            reference=json.loads((prior/"score.json").read_text())["metrics"]
            header=dict(domain=domain,video=clip,split=meta["split"],nf=meta["nf"])
            truths,proof=_gt_provider(repo,[header],("35","33","W","G"),"evaluation")
            if any(label_top["GT_sha256"][p["path"]]!=p["sha256"] for p in proof.values()): raise ValueError("Native/label GT fingerprint differs")
            dest.parent.mkdir(parents=True,exist_ok=True)
            temporary=Path(tempfile.mkdtemp(prefix="."+clip+"-",dir=dest.parent))/"payload"
            result=evaluate_clip(truths[clip],rows,canon,data["pairs"],closed["predictions"]["decisions"][domain][clip],meta["polygon"],meta["nf"],
                score_native=score_member_native,reference_metrics=reference,signature=signature,out=temporary,write_mot=write_mot_exact)
            temporary.rename(dest);temporary.parent.rmdir()
            del rows,canon,historical,reference,truths
        clips.append(dict(result,domain=domain,clip=clip,seed=ref["seed"],member=meta["member"],
            selection_audit=closed["audits"][domain+'/'+clip],populations=data["denominator"]))
        del data
        print("PHYSICAL210_SCORED",domain,clip,flush=True);gc.collect()
    report=dict(schema="interleave210-native-stage-v1",input_receipt=identity,clips=clips,domains=native.pool_clips(clips),
        physical_scope=dict(clip_requests=6,state_requests=36,states_per_clip=6),numeric_support=closed["support"],
        source_models={h:{key:m[key] for key in ("source_counts","full_support_source_counts")} for h,m in closed["models"].items()},
        matching_contract="Every whole scene freshly scored at .5/.3; all original rows retained; full windows=K+sum fixed local deltas",independent_replay="pending")
    (out/"summary.json").write_text(json.dumps(plain(report),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
    manifest={p.relative_to(out).as_posix():file_sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p not in (out/"manifest.json",out/"SEALED.sha256")}
    (out/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
    (out/"SEALED.sha256").write_text(file_sha(out/"manifest.json")+"\n")
    print("PHYSICAL210_SEALED",file_sha(out/"manifest.json"),flush=True)
    return report


def selftest():
    def row(f,p,x=0,score=.9): return [f,p,x,0,4,4,score]
    rows=np.asarray([row(5,2),row(1,1),row(4,1),row(2,2),row(3,1),row(6,2)],float)
    canon=rows[:,1].astype(np.int64)
    delta=np.asarray([[1,0]]*30,np.int64).tolist()
    pair=dict(pair_id="1|2",canonical_ids=[1,2],min_offset_norm=.2,delta_windows=delta,active=True)
    joined,proof=materialize(rows,canon,[pair],["1|2"])
    assert len(joined)==len(rows) and len(set(joined[:,1]))==1 and joined[0,1]==1
    np.testing.assert_array_equal(joined[:,0],np.arange(1,7))
    np.testing.assert_array_equal(joined[:,[0,2,3,4,5,6]],rows[proof["output_raw_indices"]][:,[0,2,3,4,5,6]])
    kept,_=materialize(rows,canon,[pair],[]);np.testing.assert_array_equal(kept,rows)
    bad=rows.copy();bad[0,0]=4
    try: materialize(bad,canon,[pair],["1|2"])
    except ValueError: pass
    else: raise AssertionError("Common-date pair was accepted")
    candidates=[dict(pair,pair_id=f"{2*i+1}|{2*i+2}",canonical_ids=[2*i+1,2*i+2],min_offset_norm=.1*i) for i in range(3)]
    choices=select_states(candidates,{a:{p["pair_id"]:0. for p in candidates} for a in MODELS})
    assert choices["a"]==3 and choices["k"]==1 and choices["selected"]["G"]==["1|2"]
    assert len(choices["selected"]["ALL"])==3
    zero=dict(candidates[0],delta_windows=[[0,0]]*30,active=False)
    inactive=select_states([zero],{a:{zero["pair_id"]:0.} for a in MODELS})
    assert inactive["a"]==0 and all(not ids for ids in inactive["selected"].values())
    legacy=np.asarray([row(f,1) for f in (1,2,4,100,102,104,200,202,204)]+[row(f,2) for f in (101,103)],float)
    cc=np.asarray([1]*3+[100001]*3+[200001]*3+[2]*2,np.int64)
    p=dict(pair,pair_id="2|100001",canonical_ids=[2,100001])
    preserved,pp=materialize(legacy,cc,[p],[p["pair_id"]])
    assert set(preserved[preserved[:,0]>=200,1])=={200001}
    base.original_runs(preserved,preserved[:,1].astype(np.int64))
    return dict(status="PASS",tests="all rows/date/box/score once; mincanonical whole union; stable date order; no common dates; exact K; a<4 still quarter budget; zero-logit ties select; ALL active only; unselected legacy-gap alias preserved")


def synthetic(repo,out):
    import sys
    repo=Path(repo).resolve();sys.path[:0]=[str(repo),str(repo/"tools/study197"),str(Path(__file__).parent)]
    import catalog210
    from tools.study195.run import score_member_native
    from tools.study195.freeze import write_mot_exact
    rows=np.asarray([[750,1,0,0,4,4,.9],[751,2,-4,0,4,4,.8],
                     [748,1,-4,0,4,4,.9],[749,2,0,0,4,4,.8]],float)
    canon=rows[:,1].astype(np.int64)
    polygon=np.asarray([[0,-10],[100,-10],[100,100],[0,100]],np.float32)
    catalog=catalog210.build(rows,canon,800,polygon)
    assert len(catalog["pairs"])==1 and catalog["pairs"][0]["active"]
    choices=select_states(catalog["pairs"],{a:{p["pair_id"]:0. for p in catalog["pairs"]} for a in MODELS})
    assert choices["a"]==choices["k"]==1
    for arm in ARMS:
        ours,proof=materialize(rows,canon,catalog["pairs"],choices["selected"][arm])
        shared,other=catalog210.merge_rows(rows,canon,catalog["pairs"],choices["selected"][arm])
        np.testing.assert_array_equal(ours,shared)
        assert proof["output_raw_indices"]==other["output_raw_indices"]
    signature=dict(prereg="a"*40,folds=["35","33","W","G"],heldout="35",clip="synthetic",
        **{k:"b"*64 for k in ("input208_manifest_sha256","candidate_manifest_sha256","label_manifest_sha256",
                              "prediction_manifest_sha256","reference209_score_seal")})
    gt=rows[:,:6].copy()
    result=evaluate_clip(gt,rows,canon,catalog["pairs"],choices["selected"],polygon,800,
        score_native=score_member_native,reference_metrics=score_member_native(gt,rows,polygon,800),
        signature=signature,out=out,write_mot=write_mot_exact)
    assert len(result["states"])==6 and result["states"]["K"]["nP"]==2
    assert result["states"]["ALL"]["nP"]==0
    assert all(s["all_window_delta_closure_exact"] for s in result["states"].values())
    verified_manifest(out)
    print("PHYSICAL210_SYNTHETIC_PASS",file_sha(Path(out)/"manifest.json"),flush=True)
    return result


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--selftest",action="store_true")
    p.add_argument("--synthetic",action="store_true");p.add_argument("--score",action="store_true")
    p.add_argument("--repo",type=Path);p.add_argument("--out",type=Path);p.add_argument("--root",type=Path)
    p.add_argument("--inputs208",type=Path);p.add_argument("--reference209-score",type=Path);p.add_argument("--prereg")
    args=p.parse_args()
    if sum((args.selftest,args.synthetic,args.score))!=1: p.error("Choose --selftest, --synthetic or --score")
    if args.selftest: print(json.dumps(selftest(),sort_keys=True))
    elif args.synthetic:
        if args.repo is None or args.out is None: p.error("--synthetic needs --repo --out")
        synthetic(args.repo,args.out)
    else:
        if any(v is None for v in (args.repo,args.root,args.inputs208,args.reference209_score,args.prereg)):
            p.error("--score needs --repo --root --inputs208 --reference209-score --prereg")
        run_stage(args)
