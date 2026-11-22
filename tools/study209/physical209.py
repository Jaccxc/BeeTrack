"""GT-free pair selection/materialization, followed by explicit native scoring.

No field IO on import. One action merges two indivisible ORIGINAL canonical
gap runs; selected actions have disjoint endpoints, so multiway chains cannot
arise. Every output row is an identified original row, in original row order.
"""
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ARMS = ("K","J","U","G","F","X")
MODELS = ("G","F","X")
_spec = importlib.util.spec_from_file_location("common209_native_base",Path(__file__).parents[1]/"study208"/"physical208.py")
native_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native_base)
plain,file_sha,verified_manifest = native_base.plain,native_base.file_sha,native_base.verified_manifest


def original_runs(rows,canon):
    """Validate the exact native by_track ID+100000*gap-run mapping (>25)."""
    rows = np.asarray(rows,dtype=np.float64)
    canon = np.asarray(canon)
    if (rows.ndim!=2 or rows.shape[1]!=7 or not np.isfinite(rows).all() or np.any(rows[:,4:6]<=0)
            or not np.array_equal(rows[:,:2],np.rint(rows[:,:2])) or np.any(rows[:,:2]<1)):
        raise ValueError("Finite native Mx7 rows with positive integer date/ID required")
    if canon.shape!=(len(rows),) or not np.issubdtype(canon.dtype,np.integer) or np.any(canon<1):
        raise ValueError("One original canonical-run integer ID per raw row required")
    order = np.lexsort((np.arange(len(rows)),rows[:,0],rows[:,1]))
    boundaries = np.r_[0,np.flatnonzero(np.diff(rows[order,1]))+1,len(rows)] if len(rows) else [0]
    expected = np.empty(len(rows),np.int64)
    runs = {}
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        track = order[a:b]
        dates = rows[track,0]
        if np.any(np.diff(dates)<=0): raise ValueError("Original native ID has duplicate dates")
        cuts = np.r_[0,np.flatnonzero(np.diff(dates)>25)+1,len(track)]
        for run,(left,right) in enumerate(zip(cuts[:-1],cuts[1:])):
            cid = int(rows[track[0],1])+100000*run
            if cid in runs: raise ValueError("Original official gap alias collides")
            ix = track[left:right]
            runs[cid] = ix
            expected[ix] = cid
    if not np.array_equal(canon,expected): raise ValueError("canon differs from original official gap-run mapping")
    return rows,canon.astype(np.int64,copy=False),runs


def pair_table(pairs):
    table = {}
    for pair in pairs:
        a,b = pair["canonical_ids"]
        if not isinstance(a,(int,np.integer)) or not isinstance(b,(int,np.integer)) or not 0<a<b:
            raise ValueError("Pair IDs must be ascending distinct canonical integers")
        key = pair["pair_id"]
        if key!=f"{a}|{b}" or key in table: raise ValueError("Duplicate or ambiguous original pair ID")
        if not np.isfinite(pair["mean_iou"]) or not 0<=pair["mean_iou"]<=1:
            raise ValueError("Finite mean IoU in[0,1] required")
        if bool(pair["geometric_eligible"])!=(pair["mean_iou"]>=.20):
            raise ValueError("Geometric eligibility differs from frozen .20 threshold")
        table[key] = pair
    return table


def geometric_greedy(pairs,valid_by_pair=None):
    """J has no motion/representative-date gate; common slate adds true.valid."""
    table = pair_table(pairs)
    if valid_by_pair is not None and set(valid_by_pair)!=set(table):
        raise ValueError("Complete pair_id-keyed true.valid mask required")
    if valid_by_pair is not None and any(not isinstance(v,(bool,np.bool_)) for v in valid_by_pair.values()):
        raise ValueError("true.valid must be explicit boolean, never an unknown label")
    ranked = sorted((p for p in pairs if p["geometric_eligible"] and
        (valid_by_pair is None or valid_by_pair[p["pair_id"]])),
        key=lambda p:(-p["mean_iou"],*p["canonical_ids"]))
    used,selected,suppressed = set(),[],[]
    for pair in ranked:
        ids = set(pair["canonical_ids"])
        if ids&used: suppressed.append(pair["pair_id"])
        else: selected.append(pair["pair_id"]);used.update(ids)
    return dict(selected=selected,suppressed=suppressed,eligible=len(ranked))


def select_states(pairs,valid_by_pair,model_scores,*,models_active=True):
    """One true.valid disjoint slate and one fixed quarter-budget for U/G/F/X."""
    if set(model_scores)!=set(MODELS): raise ValueError("G/F/X score records required")
    if not isinstance(models_active,bool): raise ValueError("Explicit source-class status required")
    table = pair_table(pairs)
    common = geometric_greedy(pairs,valid_by_pair)
    slate = common["selected"]
    n = len(slate)
    k = (n+3)//4 if n>=4 else 0
    selected = dict(K=[],J=geometric_greedy(pairs)["selected"],U=slate[:k])
    for arm in MODELS:
        if not models_active:
            selected[arm] = []
            continue
        if not set(slate)<=model_scores[arm].keys(): raise ValueError("Model omits a common-slate score")
        if any(not np.isfinite(model_scores[arm][key]) for key in slate): raise ValueError("Nonfinite common-slate score")
        selected[arm] = sorted(slate,key=lambda key:(-float(model_scores[arm][key]),*table[key]["canonical_ids"]))[:k]
    return dict(selected=selected,common_slate=slate,n=n,k=k,models_active=models_active,
        common_suppressed=common["suppressed"],ordering="descending meanIoU or model score; ascending original(A,B)")


def materialize(rows,canon,pairs,selected_ids):
    """Whole-run pair union, score-priority same-date deletion, no chaining."""
    rows,canon,runs = original_runs(rows,canon)
    table = pair_table(pairs)
    ids = list(selected_ids)
    if any(not isinstance(key,str) for key in ids) or len(ids)!=len(set(ids)) or not set(ids)<=table.keys():
        raise ValueError("Unique frozen selected pair IDs required")
    used = set()
    keep = np.ones(len(rows),bool)
    output = rows.copy()
    actions = []
    ranked = sorted(table,key=lambda key:tuple(table[key]["canonical_ids"]))
    rank = {key:i for i,key in enumerate(ranked)}
    base = max(max(runs,default=0),int(rows[:,1].max()) if len(rows) else 0)+1
    if base+len(table)>=2**53: raise ValueError("Merged native ID exceeds exact numeric range")
    for key in sorted(ids,key=lambda key:tuple(table[key]["canonical_ids"])):
        pair = table[key]
        a,b = pair["canonical_ids"]
        if a not in runs or b not in runs or {a,b}&used:
            raise ValueError("Foreign/reused original run; chained merges are forbidden")
        if not pair["geometric_eligible"]: raise ValueError("Merge outside fixed meanIoU eligibility")
        left = {int(rows[i,0]):int(i) for i in runs[a]}
        right = {int(rows[i,0]):int(i) for i in runs[b]}
        dates = sorted(left.keys()&right.keys())
        actual = [[left[f],right[f]] for f in dates]
        if not dates: raise ValueError("Span overlap without actual shared dates is not a merge")
        if pair["shared_dates"]!=dates or pair["row_indices"]!=actual:
            raise ValueError("Frozen shared dates/raw row indices differ")
        ix = np.asarray(actual,np.int64)
        A,B = rows[ix[:,0],2:6],rows[ix[:,1],2:6]
        wh = np.maximum(0,np.minimum(A[:,:2]+A[:,2:],B[:,:2]+B[:,2:])-np.maximum(A[:,:2],B[:,:2]))
        intersection = np.prod(wh,axis=1)
        ious = intersection/(np.prod(A[:,2:],axis=1)+np.prod(B[:,2:],axis=1)-intersection)
        if float(ious.mean())!=pair["mean_iou"]: raise ValueError("Mean IoU differs from all shared dates including zeros")
        deleted,winners = [],[]
        for i,j in actual:
            winner,loser = (i,j) if rows[i,6]>rows[j,6] or rows[i,6]==rows[j,6] and i<j else (j,i)
            keep[loser] = False
            deleted.append(loser);winners.append(winner)
        merged_id = base+rank[key]
        output[np.r_[runs[a],runs[b]],1] = merged_id
        merged_dates = np.sort(np.r_[rows[runs[a],0],rows[runs[b],0]])
        if np.any(np.diff(np.unique(merged_dates))>25): raise ValueError("Merge would create an official gap split")
        used.update((a,b))
        actions.append(dict(pair_id=key,canonical_ids=[int(a),int(b)],merged_native_id=merged_id,
            original_rows_A=runs[a].tolist(),original_rows_B=runs[b].tolist(),
            shared_dates=dates,winner_raw_indices=winners,deleted_raw_indices=deleted,
            all_nonshared_rows_retained=True,indivisible_original_runs=True))
    retained = np.flatnonzero(keep)
    physical = output[keep]
    if physical[:,[0,2,3,4,5,6]].tobytes()!=rows[retained][:,[0,2,3,4,5,6]].tobytes():
        raise AssertionError("A retained observation changed date/geometry/score")
    if len(np.unique(physical[:,:2],axis=0))!=len(physical): raise AssertionError("Duplicate output native ID/date")
    # Fresh merged IDs are above every original official alias; they need no
    # new >25 gap split. Unselected original raw IDs/runs remain unchanged.
    if not ids and physical.tobytes()!=rows.tobytes(): raise AssertionError("K must be bitwise original208")
    return physical,dict(selected_pair_ids=sorted(ids),actions=actions,retained_raw_indices=retained.tolist(),
        deleted_raw_indices=np.flatnonzero(~keep).tolist(),raw_row_subset_bijection=True,
        native_row_order_stable=True,unselected_native_IDs_exact=True,official_gap_partition_consistent=True,
        ID_policy="one fresh ID per fixed numeric catalog-pair rank, above all original canonical/raw IDs")


def action_diagnostics(pair_labels,selected_ids):
    """Target-only diagnostic; an unconfirmed collision is never called absent."""
    records=[]
    for key in selected_ids:
        evidence = pair_labels[key]
        counts=evidence["common_counts"]
        confirmed=bool(evidence["any_verified_collision"])
        if confirmed!=(counts["D"]>0): raise ValueError("Collision evidence/count mismatch")
        records.append(dict(pair_id=key,any_verified_collision=confirmed,collision_dates=evidence["collision_dates"],
            common_counts=counts,unconfirmed_not_a_negative=not confirmed))
    return dict(selected_pairs=len(records),pairs_with_at_least_one_confirmed_distinct_GT=sum(r["any_verified_collision"] for r in records),
        remaining_unconfirmed_pairs=sum(not r["any_verified_collision"] for r in records),records=records,
        label_scope="Each box uniquely compatible with an original GT rectangle at IoU.30; not pixel identity truth")


def evaluate_clip(gt,rows,canon,pairs,selections,polygon,nframes,*,score_native,reference_metrics,
                  signature,pair_labels,out=None,write_mot=None):
    """Explicit downstream GT entry; upstream models/slates cannot accept GT.

    Caller verifies all four folded X/G/F selection closures before invoking.
    Every state is a complete scene, freshly rematched at official .5 and .3.
    Unknown common-date evidence is retained as unconfirmed, never a negative.
    """
    import re
    if (not re.fullmatch("[0-9a-f]{40}",signature.get("prereg",""))
            or set(signature.get("folds",[]))!={"35","33","W","G"}
            or signature.get("heldout") not in ("35","33","W","G") or not signature.get("clip")
            or any(not re.fullmatch("[0-9a-f]{64}",signature.get(name,"")) for name in
                ("input208_manifest_sha256","candidate_manifest_sha256","feature_manifest_sha256",
                 "label_manifest_sha256","prediction_manifest_sha256","reference208_score_seal"))):
        raise ValueError("Complete prereg/artifact/common-four-fold signature required")
    if set(selections)!=set(ARMS) or selections["K"]:
        raise ValueError("Exactly six states with original K required")
    table=pair_table(pairs)
    if set(pair_labels)!=set(table): raise ValueError("Complete posthoc common-date evidence required")
    if out is not None:
        out=Path(out)
        if out.exists(): raise FileExistsError("Immutable clip output already exists")
        if write_mot is None: raise ValueError("Exact full MOT writer required")
        out.mkdir(parents=True)
    result=dict(schema="common209-physical-clip-v1",signature=plain(signature),physical_code_sha256=file_sha(__file__),
        states={},all_action_diagnostics=action_diagnostics(pair_labels,sorted(table)),
        denominators=dict(original_rows=len(rows),original_runs=len(np.unique(canon)),catalog_pairs=len(pairs),
                          geometric_eligible_pairs=sum(p["geometric_eligible"] for p in pairs)))
    for arm in ARMS:
        physical,proof=materialize(rows,canon,pairs,selections[arm])
        if out is not None:
            folder=out/arm;folder.mkdir()
            write_mot(physical,folder/"mot.txt")
            replay=np.loadtxt(folder/"mot.txt",delimiter=",",usecols=range(7),ndmin=2) if len(physical) else np.empty((0,7))
            if replay.tobytes()!=physical.tobytes(): raise AssertionError("MOT roundtrip changed physical rows")
            physical=replay
            np.save(folder/"retained_raw_indices.npy",np.asarray(proof["retained_raw_indices"],np.int64),allow_pickle=False)
            np.save(folder/"deleted_raw_indices.npy",np.asarray(proof["deleted_raw_indices"],np.int64),allow_pickle=False)
        scored=native_base.score_physical(gt,physical,polygon,nframes,score_native)
        if arm=="K":
            native_base.assert_baseline(scored["metrics"],reference_metrics)
            if plain(scored["metrics"])!=plain(reference_metrics):
                raise ValueError("Full K official metrics/event graph differs from208")
        collision=action_diagnostics(pair_labels,selections[arm])
        record=dict(scored,proof=proof,selected_pair_ids=sorted(selections[arm]),
            selected_pairs=len(selections[arm]),removed_rows=len(proof["deleted_raw_indices"]),
            action_diagnostics=collision)
        if out is not None:
            (folder/"score.json").write_text(json.dumps(plain(record),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
            record=dict(numerator=scored["numerator"],denominator=scored["denominator"],W=scored["W"],
                iou={t:{k:scored["metrics"]["iou"][t][k] for k in ("tp","pred","gt","f1")} for t in ("0.5","0.3")},
                selected_pair_ids=sorted(selections[arm]),selected_pairs=len(selections[arm]),
                removed_rows=len(proof["deleted_raw_indices"]),action_diagnostics=collision,path=arm)
        result["states"][arm]=record
    if out is not None:
        (out/"summary.json").write_text(json.dumps(plain(result),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
        manifest={p.relative_to(out).as_posix():file_sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        (out/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
        (out/"SEALED.sha256").write_text(file_sha(out/"manifest.json")+"\n")
    return result


def read_gz(path):
    import gzip
    with gzip.open(path,"rt",encoding="utf8") as stream: return json.load(stream)


def stage_guard(root,prereg):
    """Verify/replay all four folded blind lists before target evidence access."""
    import fit209
    root=Path(root)
    if not all((root/k/"SEALED.sha256").is_file() for k in ("catalog","features","labels","predictions")):
        raise ValueError("All frozen feature/label/common-four-fold prediction seals required")
    closures={k:verified_manifest(root/k) for k in ("catalog","features","labels","predictions")}
    loaded=fit209.load_predictions(root/"predictions")
    receipt=loaded["receipt"]
    if receipt["prereg"]!=prereg or any(receipt[field]!=closures[k][0] for k,field in
        (("catalog","candidate_manifest_sha256"),("features","feature_manifest_sha256"),("labels","label_manifest_sha256"))):
        raise ValueError("Prediction/artifact lineage differs")
    expected208="1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94"
    if receipt["input208_manifest_sha256"]!=expected208: raise ValueError("Wrong original208 input source")
    candidates=json.loads((root/"catalog/deployment.json").read_text())
    features=json.loads((root/"features/deployment.json").read_text())
    if (candidates["prereg"]!=prereg or features["prereg"]!=prereg or candidates["source208"]!=expected208
            or features["candidate_catalog_manifest_sha256"]!=closures["catalog"][0]):
        raise ValueError("Frozen catalog/features provenance differs")
    specs={("35","BEE24-35-head",3),("33","BEE24-33-head",3),("W","BEE24-10",0),
           ("W","BEE24-15",0),("G","BEE24-24",0),("G","BEE24-28",0)}
    if len(candidates["clips"])!=6 or {(c["domain"],c["clip"],c["seed"]) for c in candidates["clips"]}!=specs:
        raise ValueError("Exactly original six fixed members required")
    numeric,clips,support=fit209.load_numeric(root/"catalog",root/"features",
        dict(manifest_sha256=closures["catalog"][0],files=closures["catalog"][1]),
        dict(manifest_sha256=closures["features"][0],files=closures["features"][1]),prereg)
    predictions=loaded["predictions"]
    if set(predictions["arms"])!=set(ARMS) or len(predictions["clips"])!=6:
        raise ValueError("Common six-arm/six-clip blind closure differs")
    models={}
    for h in ("35","33","W","G"):
        fold=predictions["folds"][h]
        model=json.loads((root/"predictions"/fold["model_file"]).read_text())
        if model["heldout"]!=h or set(model["source_domains"])!={"35","33","W","G"}-{h}:
            raise ValueError("Model source/heldout boundary differs")
        active=model["status"]=="fitted"
        if any((model["heads"][a]["status"]=="fitted")!=active for a in MODELS):
            raise ValueError("All three models must abstain together on missing source class")
        totals={c:sum(count[c] for count in model["source_counts"].values()) for c in ("S","D")}
        if active!=(totals["S"]>0 and totals["D"]>0): raise ValueError("No-op status differs from pooled source classes")
        models[h]=model
    audits={}
    for domain,clip in clips:
        rows=[r for r in numeric if (r["domain"],r["clip"])==(domain,clip)]
        stored=predictions["clips"][domain+'/'+clip]
        expected=dict(domain=domain,clip=clip,**fit209.deploy_clip(rows,models[domain]))
        if plain(stored)!=plain(expected): raise ValueError("Blind score/true-valid/sham/slate/budget replay differs")
        ref=next(c for c in candidates["clips"] if (c["domain"],c["clip"])==(domain,clip))
        catalog=read_gz(root/"catalog"/ref["path"]/"catalog.json.gz")
        valid={r["pair_id"]:bool(r["true"]["valid"]) for r in rows}
        scores={a:{r["pair_id"]:r["logits"][a] for r in stored["records"]} for a in MODELS}
        replay=select_states(catalog["pairs"],valid,scores,models_active=models[domain]["status"]=="fitted")
        if (replay["common_slate"]!=stored["slate"] or replay["n"]!=stored["n_slate"] or replay["k"]!=stored["k"]
                or any(replay["selected"][a]!=stored["selected"][a] for a in ARMS)):
            raise ValueError("Independent pair greedy/model ranking does not reproduce all blind lists")
        audits[domain+'/'+clip]=replay
        del catalog
    return dict(seals={k:value[0] for k,value in closures.items()},catalog=candidates,
                prediction=loaded,selection_audits=audits,models=models,numeric_support=support)


def run_stage(args):
    """Only field entry; no feature extraction, model training, or LK calls."""
    import gc
    import re
    import subprocess
    import sys
    import tempfile
    root,repo=args.root.resolve(),args.repo.resolve()
    if root.drive.upper()!="D:" or not re.fullmatch("[0-9a-f]{40}",args.prereg or ""):
        raise ValueError("D: root and full published prereg SHA required")
    subprocess.run(["git","merge-base","--is-ancestor",args.prereg,"origin/main"],cwd=repo,check=True,capture_output=True)
    log=subprocess.run(["git","show",args.prereg+":docs/experiment_log.md"],cwd=repo,check=True,capture_output=True).stdout.decode("utf8")
    if log.count("[cross-part-motion-v1]")!=1: raise ValueError("Unique published209 marker required")
    closed=stage_guard(root,args.prereg)
    input208=args.inputs208.resolve()
    inputsha=file_sha(input208/"manifest.json")
    if (inputsha!="1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94"
            or (input208/"SEALED.sha256").read_text().strip()!=inputsha): raise ValueError("208 named input closure differs")
    input_manifest=json.loads((input208/"manifest.json").read_text())
    reference_sha,_=verified_manifest(args.reference208_score)
    if reference_sha!="10e1a746d0a01b71d8f5b5aa1a9c1e15bc5572c354e1f8fb34d43b5ac2d5c895":
        raise ValueError("Wrong registered208 physical reference")
    # No GT/evidence file is decoded before the full blind-list replay above.
    label_top=json.loads((root/"labels/deployment.json").read_text())
    if (label_top["prereg"]!=args.prereg or label_top["candidate_manifest_sha256"]!=closed["seals"]["catalog"]
            or label_top["feature_manifest_sha256"]!=closed["seals"]["features"]):
        raise ValueError("Target evidence belongs to different frozen artifacts")
    sys.path[:0]=[str(repo),str(repo/"tools/study197")]
    from tools.study195.run import score_member_native,_gt_provider
    from tools.study195.freeze import write_mot_exact
    out=root/"score"
    if (out/"SEALED.sha256").exists(): raise FileExistsError("Complete209 score already sealed")
    out.mkdir(parents=True,exist_ok=True)
    identity=dict(prereg=args.prereg,input208_manifest_sha256=inputsha,artifact_seals=closed["seals"],
                  reference208_score_seal=reference_sha,physical_code_sha256=file_sha(__file__))
    running=out/"RUNNING.json"
    if running.exists() and json.loads(running.read_text())!=identity: raise ValueError("Different scoring round cannot resume")
    running.write_text(json.dumps(identity,sort_keys=True,separators=(",",":"))+"\n")
    summaries=[]
    for ref in closed["catalog"]["clips"]:
        domain,clip=ref["domain"],ref["clip"]
        catalog=read_gz(root/"catalog"/ref["path"]/"catalog.json.gz")
        meta=catalog["metadata"]
        signature=dict(prereg=args.prereg,folds=["35","33","W","G"],heldout=domain,clip=clip,
            input208_manifest_sha256=inputsha,candidate_manifest_sha256=closed["seals"]["catalog"],
            feature_manifest_sha256=closed["seals"]["features"],label_manifest_sha256=closed["seals"]["labels"],
            prediction_manifest_sha256=closed["seals"]["predictions"],reference208_score_seal=reference_sha)
        dest=out/"clips"/domain/clip
        if dest.exists():
            verified_manifest(dest)
            result=json.loads((dest/"summary.json").read_text())
            if result["signature"]!=signature or result["physical_code_sha256"]!=file_sha(__file__):
                raise ValueError("Cached clip has a different physical/input signature")
        else:
            rawfolder=(input208/meta["input208_root_relative"]).resolve()
            if not rawfolder.is_relative_to(input208): raise ValueError("Raw source escapes208 closure")
            for name,key in (("rows.npy","rows_sha256"),("canon.npy","canon_sha256")):
                relative=(rawfolder/name).relative_to(input208).as_posix()
                if file_sha(rawfolder/name)!=meta[key] or input_manifest[relative]!=meta[key]: raise ValueError("Original208 row/canonical artifact differs")
            rows=np.load(rawfolder/"rows.npy",mmap_mode="r",allow_pickle=False)
            canon=np.load(rawfolder/"canon.npy",mmap_mode="r",allow_pickle=False)
            prior=args.reference208_score/"clips"/domain/clip/"K"
            historical=np.loadtxt(prior/"mot.txt",delimiter=",",usecols=range(7),ndmin=2) if len(rows) else np.empty((0,7))
            if historical.tobytes()!=rows.tobytes(): raise ValueError("K is not bitwise original208 physical scene")
            reference=json.loads((prior/"score.json").read_text())["metrics"]
            evidence=read_gz(root/"labels"/domain/clip/"evidence.json.gz")
            labels={p["pair_id"]:p for p in evidence["records"]}
            header=dict(domain=domain,video=clip,split=meta["split"],nf=meta["nf"])
            truths,GTproof=_gt_provider(repo,[header],("35","33","W","G"),"evaluation")
            if any(label_top["GT_sha256"][p["path"]]!=p["sha256"] for p in GTproof.values()): raise ValueError("Weak-label/native GT fingerprints differ")
            dest.parent.mkdir(parents=True,exist_ok=True)
            temporary=Path(tempfile.mkdtemp(prefix="."+clip+"-",dir=dest.parent))/"payload"
            result=evaluate_clip(truths[clip],rows,canon,catalog["pairs"],closed["prediction"]["decisions"][domain][clip],
                meta["polygon"],meta["nf"],score_native=score_member_native,reference_metrics=reference,
                signature=signature,pair_labels=labels,out=temporary,write_mot=write_mot_exact)
            temporary.rename(dest);temporary.parent.rmdir()
            del rows,canon,historical,evidence,labels,truths,reference
        result=dict(result,domain=domain,clip=clip,seed=ref["seed"],member=ref["member"],selection_audit=closed["selection_audits"][domain+'/'+clip])
        summaries.append(result)
        print("PHYSICAL209_SCORED",domain,clip,flush=True)
        del catalog
        gc.collect()
    report=dict(schema="common209-native-stage-v1",input_receipt=identity,clips=summaries,
        domains=native_base.pool_clips(summaries),physical_scope=dict(clip_requests=6,state_requests=36,states_per_clip=6),
        source_support=closed["numeric_support"],source_models={h:{k:m[k] for k in ("status","source_counts","source_rows","main_valid_source_rows")} for h,m in closed["models"].items()},
        matching_contract="All36 whole scenes freshly matched at .5/.3; no per-pair TP delta interpolation",independent_replay="pending")
    (out/"summary.json").write_text(json.dumps(plain(report),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
    manifest={p.relative_to(out).as_posix():file_sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p not in (out/"manifest.json",out/"SEALED.sha256")}
    (out/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
    (out/"SEALED.sha256").write_text(file_sha(out/"manifest.json")+"\n")
    print("PHYSICAL209_SEALED",file_sha(out/"manifest.json"),flush=True)
    if args.attainable:
        import attainable209
        diagnostic=dict(schema="common209-posthoc-triangle-v1",score_seal=file_sha(out/"manifest.json"),
            prereg=args.prereg,code_sha256=file_sha(attainable209.__file__),clips=[],
            scope="Posthoc optimistic triangle bound for fixed common slate and k only; not J, not attainable, not a registered decision rule; no F1 claim")
        for ref,scored in zip(closed["catalog"]["clips"],summaries):
            catalog=read_gz(root/"catalog"/ref["path"]/"catalog.json.gz");meta=catalog["metadata"]
            source=input208/meta["input208_root_relative"]
            rows=np.load(source/"rows.npy",mmap_mode="r",allow_pickle=False)
            canon=np.load(source/"canon.npy",mmap_mode="r",allow_pickle=False)
            audit=closed["selection_audits"][ref["domain"]+'/'+ref["clip"]]
            value=attainable209.analyze(rows,canon,catalog["pairs"],audit["common_slate"],meta["polygon"],meta["nf"],
                audit["k"],GTevents=int(scored["states"]["K"]["iou"]["0.5"]["gt"]))
            diagnostic["clips"].append(dict(domain=ref["domain"],clip=ref["clip"],**value))
            del rows,canon,catalog
        sidecar=root/"attainable_diagnostic.json"
        if sidecar.exists(): raise FileExistsError("Posthoc sidecar already exists")
        sidecar.write_text(json.dumps(plain(diagnostic),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
        print("PHYSICAL209_POSTHOC",file_sha(sidecar),flush=True)
    return report


def selftest():
    def row(f,p,x=0,score=.8): return [f,p,x,0,4,4,score]
    raw=np.asarray([row(3,1,0,.8),row(1,2,1,.9),row(2,1,0,.7),row(3,2,1,.8),row(1,1,0,.8),row(4,2,1,.9)],float)
    canon=raw[:,1].astype(np.int64)
    shared=[[4,1],[0,3]]
    mean=.6
    pair=dict(pair_id="1|2",canonical_ids=[1,2],shared_dates=[1,3],row_indices=shared,mean_iou=mean,geometric_eligible=True)
    merged,proof=materialize(raw,canon,[pair],["1|2"])
    assert proof["deleted_raw_indices"]==[3,4]  # score wins f1; smaller raw index wins f3.
    assert proof["retained_raw_indices"]==[0,1,2,5] and len(set(merged[:,1]))==1
    np.testing.assert_array_equal(merged[:,[0,2,3,4,5,6]],raw[[0,1,2,5]][:,[0,2,3,4,5,6]])
    kept,proof=materialize(raw,canon,[pair],[])
    np.testing.assert_array_equal(kept,raw)
    zero=raw.copy();zero[3,2]=10
    merged,proof=materialize(zero,canon,[dict(pair,mean_iou=.3)],["1|2"])
    assert proof["deleted_raw_indices"]==[3,4]  # common date remains a single row even at IoU0.
    overlap=np.asarray([row(1,1),row(3,1),row(2,2),row(4,2)],float)
    try: materialize(overlap,overlap[:,1].astype(int),[dict(pair,shared_dates=[],row_indices=[])],["1|2"])
    except ValueError: pass
    else: raise AssertionError("Span-only overlap accepted")
    eight=[dict(pair,pair_id=f"{2*i+1}|{2*i+2}",canonical_ids=[2*i+1,2*i+2],mean_iou=.6-i*.01) for i in range(4)]
    active={p["pair_id"]:True for p in eight}
    scores={a:{p["pair_id"]:i for i,p in enumerate(eight)} for a in MODELS}
    choices=select_states(eight,active,scores)
    assert choices["n"]==4 and choices["k"]==1 and choices["selected"]["U"]==["1|2"]
    assert choices["selected"]["X"]==["7|8"] and len(choices["selected"]["J"])==4
    no_op=select_states(eight,active,scores,models_active=False)
    assert no_op["selected"]["U"]==["1|2"] and all(not no_op["selected"][a] for a in MODELS)
    return dict(status="PASS",tests="whole-run merge; full shared dates incl zero IoU; native row/score tie; span-only rejection; exact K; stable row subset; common disjoint quarter budget; three-model no-op")


def synthetic(repo,out):
    import sys
    repo=Path(repo).resolve()
    sys.path[:0]=[str(repo),str(repo/"tools/study197"),str(Path(__file__).parent)]
    import catalog209
    import labels209
    from tools.study195.run import score_member_native
    from tools.study195.freeze import write_mot_exact
    raw=[]
    for date,x in ((1,-8),(2,-4),(3,2)):
        for i in range(4):
            raw.extend([[date,2*i+1,x,30*i,5,4,.8],[date,2*i+2,x+3,30*i,5,4,.9]])
    rows=np.asarray(raw,float);canon=rows[:,1].astype(np.int64)
    polygon=np.asarray([[0,-10],[100,-10],[100,150],[0,150]],np.float32)
    catalog=catalog209.build(rows,canon,6)
    assert len(catalog["pairs"])==4
    table=pair_table(catalog["pairs"])
    scores={a:{key:float(i) for i,key in enumerate(table)} for a in MODELS}
    choices=select_states(catalog["pairs"],{key:True for key in table},scores)
    complete=dict(catalog,metadata=dict(domain="35",clip="synthetic"),rows=rows,
                  source_keys=[["synthetic",i] for i in range(len(rows))])
    truth=labels209.label_clip(complete,rows[:,:6])
    labels={r["pair_id"]:r for r in truth["records"]}
    assert all(r["any_verified_collision"] for r in labels.values())
    signature=dict(prereg="a"*40,folds=["35","33","W","G"],heldout="35",clip="synthetic",
        **{name:"b"*64 for name in ("input208_manifest_sha256","candidate_manifest_sha256","feature_manifest_sha256",
            "label_manifest_sha256","prediction_manifest_sha256","reference208_score_seal")})
    result=evaluate_clip(rows[:,:6],rows,canon,catalog["pairs"],choices["selected"],polygon,6,
        score_native=score_member_native,reference_metrics=score_member_native(rows[:,:6],rows,polygon,6),
        signature=signature,pair_labels=labels,out=out,write_mot=write_mot_exact)
    assert len(result["states"])==6 and result["states"]["J"]["removed_rows"]==12
    assert result["states"]["K"]["W"]==0 and result["states"]["J"]["W"]>0
    assert result["states"]["U"]["selected_pairs"]==1
    verified_manifest(out)
    print("PHYSICAL209_SYNTHETIC_PASS",file_sha(Path(out)/"manifest.json"),flush=True)
    return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest",action="store_true")
    parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--score",action="store_true")
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--inputs208",type=Path)
    parser.add_argument("--reference208-score",type=Path)
    parser.add_argument("--prereg")
    parser.add_argument("--attainable",action="store_true")
    args=parser.parse_args()
    if sum((args.selftest,args.synthetic,args.score))!=1: parser.error("Choose --selftest, --synthetic or --score")
    if args.selftest: print(json.dumps(selftest(),sort_keys=True))
    elif args.synthetic:
        if args.repo is None or args.out is None: parser.error("--synthetic needs --repo --out")
        synthetic(args.repo,args.out)
    else:
        if any(v is None for v in (args.repo,args.root,args.inputs208,args.reference208_score,args.prereg)):
            parser.error("--score needs --repo --root --inputs208 --reference208-score --prereg")
        run_stage(args)
