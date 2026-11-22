"""Pure §208 cut/relink operations and explicit downstream native scoring.

No field IO on import. Candidate rows retain data208's frozen ordinal: four
native raw row indices, with the split between indices[1] and indices[2].
Canonical IDs describe the official independent gap runs, not raw ID equality.
K preserves every native value. Other arms change only column1 (track ID),
while retaining every box, date, score and native row order exactly.
"""
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS = ("geometry_interval","relative_cross_exact","relative_cross_interval")
MAIN_ARM = "relative_cross_interval"
THRESHOLDS = ("0.5","0.3")


def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()


def verified_manifest(root):
    """Explicit artifact IO, never called at import or by materialization."""
    root = Path(root).resolve()
    manifest = json.loads((root/"manifest.json").read_text())
    stamp = file_sha(root/"manifest.json")
    if (root/"SEALED.sha256").read_text().strip()!=stamp: raise ValueError("Complete common seal required")
    for name,value in manifest.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or file_sha(path)!=value: raise ValueError("Artifact drift: "+name)
    return stamp,manifest


def inputs(rows, canon, indices):
    rows = np.asarray(rows,dtype=np.float64)
    canon = np.asarray(canon)
    indices = np.asarray(indices)
    if (rows.ndim!=2 or rows.shape[1]!=7 or not np.isfinite(rows).all()
            or np.any(rows[:,4:6]<=0) or np.any(rows[:,:2]<1)
            or not np.array_equal(rows[:,:2],np.rint(rows[:,:2]))):
        raise ValueError("Finite native Mx7 rows with legal dates/IDs/boxes required")
    if canon.shape!=(len(rows),) or not np.issubdtype(canon.dtype,np.integer) or np.any(canon<1):
        raise ValueError("canon must be one positive independent canonical-run ID per native row")
    if indices.ndim!=2 or indices.shape[1]!=4 or not np.issubdtype(indices.dtype,np.integer):
        raise ValueError("Frozen Nx4 integer raw row indices required")
    if np.any(indices<0) or np.any(indices>=len(rows)):
        raise ValueError("Candidate outside native row table")
    runs = {}
    ordinal = np.empty(len(rows),dtype=np.int64)
    order = np.lexsort((np.arange(len(rows)),rows[:,0],canon))
    boundaries = np.r_[0,np.flatnonzero(np.diff(canon[order]))+1,len(rows)] if len(rows) else [0]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        rr = order[a:b]
        cid = int(canon[rr[0]])
        if np.any(np.diff(rows[rr,0])<=0):
            raise ValueError("Canonical run has repeated/nonincreasing date")
        runs[cid] = rr
        ordinal[rr] = np.arange(len(rr))
    centers = set()
    metadata = []
    for i,ix in enumerate(indices):
        cid = int(canon[ix[0]])
        if not np.all(canon[ix]==cid) or not np.array_equal(ordinal[ix],np.arange(ordinal[ix[0]],ordinal[ix[0]]+4)):
            raise ValueError("Candidate must contain four consecutive observations of one canonical run")
        if np.any((np.diff(rows[ix,0])<1)|(np.diff(rows[ix,0])>3)):
            raise ValueError("Frozen candidate gaps must all lie in1..3frames")
        center = (cid,int(ordinal[ix[2]]))
        if center in centers:
            raise ValueError("Duplicate frozen cut boundary")
        centers.add(center)
        metadata.append(dict(candidate=i,canonical_id=cid,right_ordinal=center[1],
            left_raw=int(ix[1]),right_raw=int(ix[2]),left_frame=int(rows[ix[1],0]),right_frame=int(rows[ix[2],0])))
    return rows,canon.astype(np.int64,copy=False),indices.astype(np.int64,copy=False),runs,metadata


def select_cuts(rows,canon,indices,probabilities,threshold,valid=None,*,active=True):
    """Strict source threshold, then deterministic frame-distance NMS per run."""
    rows,canon,indices,runs,meta = inputs(rows,canon,indices)
    p = np.asarray(probabilities)
    valid = np.ones(len(indices),bool) if valid is None else np.asarray(valid)
    if p.shape!=(len(indices),) or valid.shape!=p.shape or valid.dtype!=np.bool_:
        raise ValueError("One probability and frozen boolean valid flag per candidate")
    if not np.isfinite(threshold) or not .5<=threshold<=1:
        raise ValueError("Frozen source threshold must lie in [.5,1]")
    if not active:
        return dict(selected=[],threshold=float(threshold),active=False,eligible=int(valid.sum()),
                    above_threshold=0,suppressed=[],reason="source arm declared no-op")
    if (np.any(~np.isfinite(p[valid])) or np.any((p[valid]<0)|(p[valid]>1))
            or np.any(np.isinf(p[~valid]))):
        raise ValueError("Valid probabilities must be finite [0,1]; invalid may be NaN")
    proposed = [m for m in meta if valid[m["candidate"]] and p[m["candidate"]]>threshold]
    proposed.sort(key=lambda m:(-float(p[m["candidate"]]),m["right_frame"],m["candidate"]))
    chosen,accepted,suppressed = [],defaultdict(list),[]
    for m in proposed:
        conflicts = [old["candidate"] for old in accepted[m["canonical_id"]]
                     if abs(old["right_frame"]-m["right_frame"])<=3]
        if conflicts:
            suppressed.append(dict(candidate=m["candidate"],by=conflicts))
        else:
            chosen.append(m["candidate"])
            accepted[m["canonical_id"]].append(m)
    return dict(selected=sorted(chosen),threshold=float(threshold),active=True,eligible=int(valid.sum()),
        above_threshold=len(proposed),suppressed=suppressed,ordering="probability desc/right frame asc/frozen ordinal asc")


def displace_scores(rows,canon,indices,probabilities,valid=None):
    """Forward one eligible cut within each unbroken observation-ordinal stretch.

    The old last score moves to the first cut. An omitted/invalid cut breaks a
    stretch; native date gaps do not merge distinct canonical runs.
    """
    _,_,indices,_,meta = inputs(rows,canon,indices)
    p = np.asarray(probabilities)
    valid = np.ones(len(indices),bool) if valid is None else np.asarray(valid)
    if p.shape!=(len(indices),) or valid.shape!=p.shape or valid.dtype!=np.bool_:
        raise ValueError("One score and frozen boolean valid flag per candidate")
    output = p.copy()
    by_run = defaultdict(list)
    for m in meta:
        if valid[m["candidate"]]: by_run[m["canonical_id"]].append(m)
    audit = []
    for cid,mm in sorted(by_run.items()):
        mm.sort(key=lambda m:m["right_ordinal"])
        stretches = []
        for m in mm:
            if not stretches or m["right_ordinal"]!=stretches[-1][-1]["right_ordinal"]+1:
                stretches.append([])
            stretches[-1].append(m)
        for stretch in stretches:
            ids = [m["candidate"] for m in stretch]
            output[ids] = np.roll(p[ids],1)
            audit.append(dict(canonical_id=cid,candidates=ids,source_candidates=ids[-1:]+ids[:-1]))
    return output,audit


def cut_rows(rows,canon,indices,selected):
    """Split at frozen boundaries. Stable segment IDs preserve canonical order."""
    rows,canon,indices,runs,meta = inputs(rows,canon,indices)
    ids = list(selected)
    if any(not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_)) for i in ids):
        raise ValueError("Frozen cut ordinals must be integers")
    if len(ids)!=len(set(ids)) or any(i<0 or i>=len(indices) for i in ids):
        raise ValueError("Duplicate/foreign cut ordinal")
    cuts = defaultdict(set)
    for i in ids: cuts[meta[i]["canonical_id"]].add(meta[i]["right_ordinal"])
    output = rows.copy()
    segments = []
    for cid,rr in runs.items():
        boundaries = [0,*sorted(cuts[cid]),len(rr)]
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            ix = rr[a:b]
            sid = len(segments)+1
            segments.append(dict(segment_id=sid,original_run=cid,raw_indices=ix.tolist(),
                first_raw=int(ix[0]),last_raw=int(ix[-1]),first_frame=int(rows[ix[0],0]),last_frame=int(rows[ix[-1],0]),
                new_start=a in cuts[cid],new_end=b in cuts[cid]))
            if ids: output[ix,1] = sid
    return output,dict(selected=sorted(ids),cuts=[meta[i] for i in sorted(ids)],segments=segments,
        original_K_exact=not ids,ID_policy="K unchanged; otherwise stable canonical-run/segment ascending positive IDs")


def relink_rows(cut,proof):
    """Fixed mutual-unique nearest proposals; null on ties, conflicts or self-run."""
    cut = np.asarray(cut,dtype=np.float64)
    segments = proof["segments"]
    by_id = {s["segment_id"]:s for s in segments}
    if len(by_id)!=len(segments): raise ValueError("Duplicate segment ID")
    eligible = []
    def center(row): return row[2:4]+row[4:6]/2
    starts = defaultdict(list)
    for b in segments: starts[b["first_frame"]].append(b)
    for a in segments:
        for b in (b for frame in range(a["last_frame"]+1,a["last_frame"]+4) for b in starts[frame]):
            gap = b["first_frame"]-a["last_frame"]
            if (a["original_run"]==b["original_run"] or not (a["new_end"] or b["new_start"])
                    or not 1<=gap<=3): continue
            ra,rb = cut[a["last_raw"]],cut[b["first_raw"]]
            scale = (float(np.hypot(ra[4],ra[5]))+float(np.hypot(rb[4],rb[5])))/2
            distance = float(np.linalg.norm(center(ra)-center(rb))/scale)
            if distance<=.75:
                eligible.append(dict(left=a["segment_id"],right=b["segment_id"],gap=gap,distance=distance,
                    left_frame=a["last_frame"],right_frame=b["first_frame"]))
    outgoing,incoming = defaultdict(list),defaultdict(list)
    for edge in eligible:
        outgoing[edge["left"]].append(edge)
        incoming[edge["right"]].append(edge)
    def unique_min(edges):
        value = min(e["distance"] for e in edges)
        matches = [e for e in edges if e["distance"]==value]
        return matches[0] if len(matches)==1 else None
    forward = {sid:unique_min(edges) for sid,edges in outgoing.items()}
    backward = {sid:unique_min(edges) for sid,edges in incoming.items()}
    proposed = [e for e in eligible if forward[e["left"]] is e and backward[e["right"]] is e]
    proposed.sort(key=lambda e:(e["left_frame"],e["right_frame"],e["left"],e["right"]))
    parent = {sid:sid for sid in by_id}
    def find(sid):
        while parent[sid]!=sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid
    accepted,rejected = [],[]
    for edge in proposed:
        a,b = find(edge["left"]),find(edge["right"])
        if a==b:
            rejected.append(dict(edge,reason="cycle"))
            continue
        parent[b] = a
        accepted.append(edge)
    output = cut.copy()
    components = defaultdict(list)
    for sid in by_id: components[find(sid)].append(sid)
    returns = []
    for sids in components.values():
        ordered = sorted(sids,key=lambda sid:by_id[sid]["first_frame"])
        original_runs = [by_id[sid]["original_run"] for sid in ordered]
        dates = [int(cut[i,0]) for sid in ordered for i in by_id[sid]["raw_indices"]]
        if len(set(dates))!=len(dates): raise AssertionError("Relink path has overlapping dates")
        if len(set(original_runs))<len(original_runs):
            returns.append(dict(segment_ids=ordered,original_runs=original_runs))
    if accepted:
        for sids in components.values():
            new_id = min(sids)
            for sid in sids: output[by_id[sid]["raw_indices"],1] = new_id
    return output,dict(eligible_edges=eligible,mutual_proposals=proposed,accepted_edges=accepted,rejected_edges=rejected,
        outgoing_ties=sorted(sid for sid,e in forward.items() if e is None),
        incoming_ties=sorted(sid for sid,e in backward.items() if e is None),
        paths_acyclic=True,indirect_original_run_returns=returns,
        indirect_original_run_return_paths=len(returns))


def materialize(rows,canon,indices,probabilities,threshold,valid=None,*,active=True,relink=False,displaced=False):
    p,audit = displace_scores(rows,canon,indices,probabilities,valid) if displaced else (np.asarray(probabilities),[])
    nms = select_cuts(rows,canon,indices,p,threshold,valid,active=active)
    output,cut = cut_rows(rows,canon,indices,nms["selected"])
    link = None
    if relink: output,link = relink_rows(output,cut)
    raw = np.asarray(rows,dtype=np.float64)
    if output[:,[0,2,3,4,5,6]].tobytes()!=raw[:,[0,2,3,4,5,6]].tobytes() or len(output)!=len(raw):
        raise AssertionError("Physical operation changed original measurements/date/row order")
    if len(np.unique(output[:,:2],axis=0))!=len(output): raise AssertionError("Duplicate track/date after linking")
    return output,dict(nms=nms,cut=cut,relink=link,displacement=audit,rows_exact_except_ID=True)


def score_physical(gt,rows,polygon,nframes,score_native):
    """Explicit independent GT entry; callers must verify all4 fold seals first."""
    metrics = score_native(gt,np.asarray(rows,dtype=np.float64),polygon,int(nframes))
    expected = {"nG","nP","pred_bins","gt_bins","iou"}
    if not expected<=metrics.keys() or not set(THRESHOLDS)<=metrics["iou"].keys():
        raise ValueError("Complete official two-IoU native metric/evidence required")
    pred,truth = np.asarray(metrics["pred_bins"]),np.asarray(metrics["gt_bins"])
    if not all(np.issubdtype(v.dtype,np.integer) for v in (pred,truth)) or pred.shape!=truth.shape:
        raise ValueError("Native 30phase integer bins required")
    numerator = int(np.abs(pred-truth).sum())
    denominator = 30*int(metrics["nG"])
    return dict(metrics=metrics,numerator=numerator,denominator=denominator,
        W=None if denominator==0 else 100*numerator/denominator)


def configurations(rows,canon,indices,predictions,calibration,valid):
    """Yield one physical state at a time; no GT accepted by deployment."""
    if set(predictions)!=set(ARMS) or set(calibration)!=set(ARMS):
        raise ValueError("Exactly the three frozen model arms required")
    yield "K",np.asarray(rows,dtype=np.float64).copy(),dict(original_K_exact=True)
    for arm in ARMS:
        setting = calibration[arm]
        if not {"threshold","active"}<=setting.keys() or not isinstance(setting["active"],bool):
            raise ValueError("Explicit frozen threshold and source-trained/no-op status required")
        cut,proof = materialize(rows,canon,indices,predictions[arm],setting["threshold"],valid,active=setting["active"])
        yield arm+"_CUT",cut,proof
        linked,links = relink_rows(cut,proof["cut"])
        if len(np.unique(linked[:,:2],axis=0))!=len(linked):
            raise AssertionError("Relink has duplicate track/date")
        yield arm+"_CUT_RELINK",linked,dict(proof,relink=links)
    setting = calibration[MAIN_ARM]
    shifted,proof = materialize(rows,canon,indices,predictions[MAIN_ARM],setting["threshold"],valid,
        active=setting["active"],displaced=True)
    yield MAIN_ARM+"_DISPLACED_CUT",shifted,proof
    linked,links = relink_rows(shifted,proof["cut"])
    yield MAIN_ARM+"_DISPLACED_CUT_RELINK",linked,dict(proof,relink=links)


def assert_baseline(actual,reference):
    for key in ("nG","nP","pred_bins","gt_bins"):
        np.testing.assert_array_equal(actual[key],reference[key],err_msg="Historical206 K "+key)
    for threshold in THRESHOLDS:
        for key in ("tp","pred","gt"):
            if actual["iou"][threshold][key]!=reference["iou"][threshold][key]:
                raise ValueError("Historical206 K official metric changed")


def cut_diagnostics(rows,canon,indices,valid,labels,probabilities,selected):
    """Posthoc only; adjacency AP and frame-resolution timing have distinct truth."""
    rows,canon,indices = np.asarray(rows),np.asarray(canon),np.asarray(indices)
    valid,labels,p = np.asarray(valid,bool),np.asarray(labels),np.asarray(probabilities)
    if valid.shape!=labels.shape or p.shape!=labels.shape or len(labels)!=len(indices) or not set(map(int,labels))<={-1,0,1}:
        raise ValueError("One explicit adjacency label(-1 unknown) per frozen cut required")
    known = valid&(labels>=0)&np.isfinite(p)
    positives = int(np.sum(known&(labels==1)))
    order = sorted(np.flatnonzero(known),key=lambda i:(-float(p[i]),int(i)))
    tp=fp=0
    ap=0.
    cursor=0
    while cursor<len(order):
        end=cursor+1
        while end<len(order) and p[order[end]]==p[order[cursor]]: end+=1
        added = sum(int(labels[i]==1) for i in order[cursor:end])
        tp+=added;fp+=end-cursor-added
        if positives: ap+=(added/positives)*(tp/(tp+fp))
        cursor=end
    chosen = sorted(set(map(int,selected)))
    if any(i<0 or i>=len(indices) or not valid[i] for i in chosen): raise ValueError("Invalid chosen cut")
    ctp=sum(labels[i]==1 for i in chosen);cfp=sum(labels[i]==0 for i in chosen)
    unknown=sum(labels[i]<0 for i in chosen)
    truth = np.flatnonzero(valid&(labels==1)&((rows[indices[:,2],0]-rows[indices[:,1],0])==1))
    right = rows[indices[:,2],0].astype(np.int64)
    cid = canon[indices[:,2]]
    matches=[];unmatched=[]
    for run in sorted(set(map(int,cid[chosen]))|set(map(int,cid[truth]))):
        pp=sorted((i for i in chosen if cid[i]==run),key=lambda i:(right[i],i))
        tt=sorted((int(i) for i in truth if cid[i]==run),key=lambda i:(right[i],i))
        a=b=0
        while a<len(pp) and b<len(tt):
            if right[pp[a]]<right[tt[b]]-1: unmatched.append(pp[a]);a+=1
            elif right[pp[a]]>right[tt[b]]+1: b+=1
            else: matches.append(dict(prediction=pp[a],truth=tt[b],canonical_id=run,
                    prediction_right_frame=int(right[pp[a]]),truth_frame=int(right[tt[b]])));a+=1;b+=1
        unmatched.extend(pp[a:])
    timing_unknown=[i for i in unmatched if labels[i]<0 or labels[i]==1 and rows[indices[i,2],0]-rows[indices[i,1],0]!=1]
    timing_fp=len(unmatched)-len(timing_unknown)
    timing_tp=len(matches)
    return dict(adjacency=dict(known_positive=positives,known_negative=int(np.sum(known&(labels==0))),
        unknown_valid=int(np.sum(valid&(labels<0))),invalid=int(np.sum(~valid)),
        AP=None if not positives else float(ap),AP_ties="equal probabilities grouped",
        nms=dict(tp=int(ctp),fp=int(cfp),fn=positives-int(ctp),selected_unknown=int(unknown),
            precision=None if not ctp+cfp else float(ctp/(ctp+cfp)),recall=None if not positives else float(ctp/positives))),
        timing_pm1=dict(truth=len(truth),tp=timing_tp,fp=timing_fp,fn=len(truth)-timing_tp,
            precision=None if not timing_tp+timing_fp else timing_tp/(timing_tp+timing_fp),
            recall=None if not len(truth) else timing_tp/len(truth),matches=matches,
            unmatched_unknown_burden=len(timing_unknown),all_selected_burden=len(chosen),
            rule="maximum one-to-one within same original canonical run; truth L1+observed gap1 only; selected right date is a cut proxy, not an interpolated true switch date"))


def evaluate_clip(gt,rows,canon,indices,predictions,calibration,valid,polygon,nframes,*,
                  score_native,reference_metrics,signature,out=None,write_mot=None,posthoc_labels=None):
    """Downstream scoring only: caller supplies a verified common four-fold seal.

    signature binds prereg, inputs_manifest_sha256,predictions_manifest_sha256,
    folds, heldout and clip. All materialization functions remain GT-free.
    A stage wrapper must verify the actual manifests before calling this API.
    Optional output keeps original raw row order and includes full native event
    graphs, each cut/right-frame/gap, mutual proposals and accepted link edges.
    """
    import re
    if (not re.fullmatch("[0-9a-f]{40}",signature.get("prereg",""))
            or any(not re.fullmatch("[0-9a-f]{64}",signature.get(k,""))
                for k in ("inputs_manifest_sha256","predictions_manifest_sha256"))
            or set(signature.get("folds",[]))!={"35","33","W","G"}
            or signature.get("heldout") not in ("35","33","W","G") or not signature.get("clip")):
        raise ValueError("Complete prereg/input/common four-fold prediction signature required")
    if out is not None:
        out = Path(out)
        if out.exists(): raise FileExistsError("Immutable clip score destination exists")
        if write_mot is None: raise ValueError("Exact native MOT writer required")
        out.mkdir(parents=True)
    result = dict(schema="switch208-physical-v1",signature=plain(signature),states={},
        physical_code_sha256=file_sha(__file__),
        denominators=dict(native_rows=len(rows),canonical_runs=len(set(map(int,canon))),
            frozen_candidates=len(indices),valid_candidates=int(np.asarray(valid,bool).sum())))
    for name,physical,proof in configurations(rows,canon,indices,predictions,calibration,valid):
        if out is not None:
            folder = out/name
            folder.mkdir()
            write_mot(physical,folder/"mot.txt")
            reread = np.loadtxt(folder/"mot.txt",delimiter=",",usecols=range(7),ndmin=2) if len(physical) else np.empty((0,7))
            if reread.tobytes()!=physical.tobytes(): raise AssertionError("Exact MOT roundtrip changed rows")
            physical = reread
        scored = score_physical(gt,physical,polygon,nframes,score_native)
        if name=="K": assert_baseline(scored["metrics"],reference_metrics)
        record = dict(scored,proof=proof,physical_rows=len(physical),raw_row_bijection=True,
            cuts=len(proof.get("cut",{}).get("selected",[])),
            relink_edges=len((proof.get("relink") or {}).get("accepted_edges",[])))
        if name!="K" and posthoc_labels is not None:
            arm = next(a for a in ARMS if name.startswith(a+"_"))
            probabilities = predictions[arm]
            if "_DISPLACED_" in name:
                probabilities,_ = displace_scores(rows,canon,indices,probabilities,valid)
            record["cut_diagnostics"] = cut_diagnostics(rows,canon,indices,valid,posthoc_labels,probabilities,proof["cut"]["selected"])
        if out is not None:
            (folder/"score.json").write_text(json.dumps(plain(record),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
            record = dict(numerator=scored["numerator"],denominator=scored["denominator"],W=scored["W"],
                iou={t:{k:scored["metrics"]["iou"][t][k] for k in ("tp","pred","gt","f1")} for t in THRESHOLDS},
                cuts=record["cuts"],relink_edges=record["relink_edges"],path=name)
            if name!="K" and posthoc_labels is not None:
                record["cut_diagnostics"] = cut_diagnostics(rows,canon,indices,valid,posthoc_labels,probabilities,proof["cut"]["selected"])
        result["states"][name] = record
    if out is not None:
        (out/"summary.json").write_text(json.dumps(plain(result),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
        manifest = {p.relative_to(out).as_posix():file_sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        (out/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
        (out/"SEALED.sha256").write_text(file_sha(out/"manifest.json")+"\n")
    return result


def stage_guard(root,prereg):
    """Read/verify all closure bytes before returning any target-score access."""
    root = Path(root)
    if not all((root/k/"SEALED.sha256").is_file() for k in ("inputs","labels","predictions")):
        raise ValueError("All input/label/common four-fold prediction seals required before scoring")
    closures = {k:verified_manifest(root/k) for k in ("inputs","labels","predictions")}
    seals = {k:value[0] for k,value in closures.items()}
    inputs_top = json.loads((root/"inputs/deployment.json").read_text())
    labels_top = json.loads((root/"labels/deployment.json").read_text())
    meta = json.loads((root/"predictions/metadata.json").read_text())
    receipt = json.loads((root/"predictions/receipt.json").read_text())
    if (any(m["prereg"]!=prereg for m in (inputs_top,labels_top,meta,receipt))
            or labels_top["input_manifest_sha256"]!=seals["inputs"]
            or meta["input_manifest_sha256"]!=seals["inputs"] or meta["label_manifest_sha256"]!=seals["labels"]
            or receipt["input_manifest_sha256"]!=seals["inputs"] or receipt["label_manifest_sha256"]!=seals["labels"]):
        raise ValueError("Common four-fold predictions/input/label lineage differs")
    if inputs_top["source206"]!="066aaa245d78ee5d926cafeb709d200746e9ea9f736f820606270c35a2188d5d":
        raise ValueError("Inputs do not belong to the registered206 K source")
    domains = {"35","33","W","G"}
    if set(meta["folds"])!=domains or set(meta["arms"])!=set(ARMS) or len(meta["clips"])!=6:
        raise ValueError("All twelve models/eighteen target probability files must share one seal")
    for domain,fold in meta["folds"].items():
        if fold["heldout"]!=domain or set(fold["source_domains"])!=domains-{domain} or set(fold["models"])!=set(ARMS):
            raise ValueError("Held-out/source boundary differs")
        for arm,model in fold["models"].items():
            cal = model["calibration"]
            if cal["quantile_probability"]!=.999 or cal["quantile_method"]!="higher":
                raise ValueError("Source calibration rule differs")
            status = model["status"]
            q = cal["quantile"]
            score_file = cal["scores_file"]
            if score_file not in closures["predictions"][1]: raise ValueError("Unsealed source calibration scores")
            with np.load(root/"predictions"/score_file,allow_pickle=False) as bank:
                source_scores = bank["probabilities"]
                observed_q = float(np.quantile(source_scores,.999,method="higher")) if len(source_scores) else None
                if observed_q!=q or len(source_scores)!=cal["exact_negative_bags"]:
                    raise ValueError("Source quantile does not reproduce frozen calibration scores")
            if any(c.split('/')[0]==domain for c in cal["clip_order"]):
                raise ValueError("Held-out clip entered calibration")
            expected = 1. if status!="trained" or q is None else max(.5,float(q))
            if model["threshold"]!=expected or cal["threshold"]!=expected:
                raise ValueError("Threshold is not the frozen source higher .999 quantile")
    for filename,value in receipt["code_receipts"].items():
        if file_sha(filename)!=value: raise ValueError("Frozen training implementation changed")
    specs = {("35","BEE24-35-head",3),("33","BEE24-33-head",3),("W","BEE24-10",0),
             ("W","BEE24-15",0),("G","BEE24-24",0),("G","BEE24-28",0)}
    if {(r["domain"],r["clip"],r["seed"]) for r in inputs_top["clips"]}!=specs or len(inputs_top["clips"])!=6:
        raise ValueError("Fixed six historical members required")
    if {(r["domain"],r["clip"],r["seed"]) for r in meta["clips"]}!=specs:
        raise ValueError("Probability clip roster differs")
    return dict(seals=seals,inputs=inputs_top,labels=labels_top,predictions=meta)


def pool_clips(clips):
    """One native member; W/G actual clips pool counts before percentages/F1."""
    from fractions import Fraction
    result = {}
    for domain in ("35","33","W","G"):
        vv = [r for r in clips if r["domain"]==domain]
        if not vv: continue
        if len(vv)!=(1 if domain in ("35","33") else 2): raise ValueError("Actual domain clip scope differs")
        methods = {}
        for name in vv[0]["states"]:
            numerator = sum(v["states"][name]["numerator"] for v in vv)
            denominator = sum(v["states"][name]["denominator"] for v in vv)
            methods[name] = dict(numerator=numerator,denominator=denominator,
                W=None if not denominator else float(Fraction(100*numerator,denominator)),iou={})
            for threshold in THRESHOLDS:
                counts = {k:sum(v["states"][name]["iou"][threshold][k] for v in vv) for k in ("tp","pred","gt")}
                counts.update(fp=counts["pred"]-counts["tp"],fn=counts["gt"]-counts["tp"],
                    f1=float(Fraction(2*counts["tp"],max(1,counts["pred"]+counts["gt"]))))
                methods[name]["iou"][threshold] = counts
        result[domain] = dict(methods=methods,clips=[r["clip"] for r in vv],
            count_contract="One actual frozen K member per clip; N/(30*GT), never a three-member ensemble average")
    return result


def registered_gate(domains):
    from fractions import Fraction
    def W(h,name):
        m=domains[h]["methods"][name]
        return Fraction(100*m["numerator"],m["denominator"])
    plus=MAIN_ARM+"_CUT_RELINK"
    checks=dict(head35_gain_at_least1=W("35","K")-W("35",plus)>=1,
        head35_strictly_beats_geometry_exact_displaced=all(W("35",plus)<W("35",name) for name in
            ("geometry_interval_CUT_RELINK","relative_cross_exact_CUT_RELINK",MAIN_ARM+"_DISPLACED_CUT_RELINK")),
        head33_no_harm_halfpoint=W("33",plus)-W("33","K")<=Fraction(1,2),
        head_F1_no_harm_onehundredth=all(Fraction(2*domains[h]["methods"][plus]["iou"]["0.5"]["tp"],
            max(1,domains[h]["methods"][plus]["iou"]["0.5"]["pred"]+domains[h]["methods"][plus]["iou"]["0.5"]["gt"])) >=
            Fraction(2*domains[h]["methods"]["K"]["iou"]["0.5"]["tp"],
                max(1,domains[h]["methods"]["K"]["iou"]["0.5"]["pred"]+domains[h]["methods"]["K"]["iou"]["0.5"]["gt"]))-Fraction(1,100)
            for h in ("35","33")))
    return dict(checks=checks,all_checks=all(checks.values()),independent_replay="pending; not an effect conclusion")


def run_stage(args):
    """Authorized FIFO entry; no model training, crop loading or image decoding."""
    import gc
    import re
    import subprocess
    import sys
    import tempfile
    repo,root = args.repo.resolve(),args.root.resolve()
    if not re.fullmatch("[0-9a-f]{40}",args.prereg or "") or root.drive.upper()!="D:":
        raise ValueError("D: root and full published prereg SHA required")
    subprocess.run(["git","merge-base","--is-ancestor",args.prereg,"origin/main"],cwd=repo,check=True,capture_output=True)
    registration = subprocess.run(["git","show",args.prereg+":docs/experiment_log.md"],cwd=repo,check=True,capture_output=True).stdout.decode("utf8")
    if registration.count("[natural-switch-hazard-v1]")!=1: raise ValueError("Unique published208 marker required")
    # Target owner arrays and GT are inaccessible until this complete guard ends.
    closed = stage_guard(root,args.prereg)
    reference_seal,reference_manifest = verified_manifest(args.reference206_score)
    if reference_seal!="cbedb72a4e2c833203d6ea2fa929bdd9989acb0cdc06a1b54ae7c598d1cb709a":
        raise ValueError("Wrong registered206 K reference")
    old = json.loads((args.reference206_score/"native_scores.json").read_text())["K"]
    baseline = {(v["unit"],str(m["member"])):m["metrics"] for v in old for m in v["members"]}
    del old
    sys.path[:0] = [str(repo),str(repo/"tools/study197")]
    from tools.study195.run import score_member_native,_gt_provider
    from tools.study195.freeze import write_mot_exact
    out = root/"score"
    if (out/"SEALED.sha256").exists(): raise FileExistsError("Completed208 score already sealed")
    out.mkdir(parents=True,exist_ok=True)
    identity = dict(prereg=args.prereg,input_seals=closed["seals"],reference206_score_seal=reference_seal,
                    physical_code_sha256=file_sha(__file__))
    running = out/"RUNNING.json"
    if running.exists() and json.loads(running.read_text())!=identity: raise ValueError("Different score inputs/code cannot resume")
    running.write_text(json.dumps(identity,sort_keys=True,separators=(",",":"))+"\n")
    pred_index = {(r["domain"],r["clip"]):r for r in closed["predictions"]["clips"]}
    summaries = []
    for ref in closed["inputs"]["clips"]:
        folder = (root/"inputs"/ref["path"]).resolve()
        if not folder.is_relative_to((root/"inputs").resolve()): raise ValueError("Input clip path escapes closure")
        if file_sha(folder/"manifest.json")!=ref["seal"]: raise ValueError("Input clip seal differs")
        meta = json.loads((folder/"metadata.json").read_text())
        signature = dict(prereg=args.prereg,inputs_manifest_sha256=closed["seals"]["inputs"],
            predictions_manifest_sha256=closed["seals"]["predictions"],labels_manifest_sha256=closed["seals"]["labels"],
            folds=sorted(closed["predictions"]["folds"]),heldout=ref["domain"],clip=ref["clip"],input_clip_seal=ref["seal"])
        dest = out/"clips"/ref["domain"]/ref["clip"]
        if dest.exists():
            verified_manifest(dest)
            scored = json.loads((dest/"summary.json").read_text())
            if scored["signature"]!=signature or scored["physical_code_sha256"]!=file_sha(__file__):
                raise ValueError("Completed clip belongs to different inputs/code")
        else:
            arrays = {name:np.load(folder/(name+".npy"),mmap_mode="r",allow_pickle=False)
                      for name in ("rows","canon","indices","valid")}
            prediction = pred_index[(ref["domain"],ref["clip"])]
            probabilities,calibration = {},{}
            for arm in ARMS:
                paths = prediction["outputs"][arm]
                probabilities[arm] = np.load(root/"predictions"/paths["probabilities"],mmap_mode="r",allow_pickle=False)
                model = closed["predictions"]["folds"][ref["domain"]]["models"][arm]
                calibration[arm] = dict(threshold=model["threshold"],active=model["status"]=="trained")
                if paths["threshold"]!=model["threshold"]: raise ValueError("Prediction calibration differs")
                selected = np.load(root/"predictions"/paths["selected"],allow_pickle=False)
                if not np.array_equal(selected,arrays["valid"]&(probabilities[arm]>model["threshold"])):
                    raise ValueError("Frozen pre-NMS threshold selection differs")
            # Exact official prepared GT: preserve even historical anomalous rows.
            header = dict(domain=meta["domain"],video=meta["clip"],split=meta["split"],nf=meta["nf"])
            truths,proof = _gt_provider(repo,[header],("35","33","W","G"),"evaluation")
            for value in proof.values():
                if closed["labels"]["GT_sha256"][value["path"]]!=value["sha256"]:
                    raise ValueError("Label and official scoring GT differ")
            dest.parent.mkdir(parents=True,exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix="."+ref["clip"]+"-",dir=dest.parent))/"payload"
            import label208
            owners = label208.load_posthoc_owners(root/"labels",ref["domain"],ref["clip"])
            adjacent = label208.represented_exact_labels(arrays["indices"],arrays["valid"],owners)
            scored = evaluate_clip(truths[ref["clip"]],arrays["rows"],arrays["canon"],arrays["indices"],probabilities,
                calibration,arrays["valid"],meta["polygon"],meta["nf"],score_native=score_member_native,
                reference_metrics=baseline[(meta["unit"],str(meta["member"]))],signature=signature,
                out=temporary,write_mot=write_mot_exact,posthoc_labels=adjacent)
            temporary.rename(dest)
            temporary.parent.rmdir()
            del arrays,probabilities,truths,owners,adjacent
        summaries.append(dict(domain=ref["domain"],clip=ref["clip"],seed=ref["seed"],member=ref["member"],**scored))
        print("PHYSICAL208_SCORED",ref["domain"],ref["clip"],flush=True)
        gc.collect()
    result = dict(schema="switch208-native-stage-v1",input_receipt=identity,clips=summaries,domains=pool_clips(summaries),
        physical_scope=dict(clip_requests=6,state_requests=54,states_per_clip=9),
        timing_contract="Selected raw cut positions are not exact switch dates across observation gaps2/3")
    result["registered_gate"] = registered_gate(result["domains"])
    (out/"summary.json").write_text(json.dumps(plain(result),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
    manifest = {p.relative_to(out).as_posix():file_sha(p) for p in sorted(out.rglob("*"))
                if p.is_file() and p not in (out/"manifest.json",out/"SEALED.sha256")}
    (out/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
    (out/"SEALED.sha256").write_text(file_sha(out/"manifest.json")+"\n")
    print("PHYSICAL208_SEALED",file_sha(out/"manifest.json"),flush=True)
    return result


def selftest():
    def row(f,p,x=0): return [f,p,x,0,4,4,.9]
    a = np.asarray([row(f,7) for f in range(1,11)],float)
    ix = np.asarray([range(i,i+4) for i in range(7)],np.int64)
    canon = np.full(len(a),7,np.int64)
    p = np.asarray([.8,.9,.9,.7,.6,.95,.9],np.float32)
    selected = select_cuts(a,canon,ix,p,.5)
    assert selected["selected"]==[1,5]
    rotated,audit = displace_scores(a,canon,ix,p)
    np.testing.assert_array_equal(rotated,np.roll(p,1))
    valid = np.ones(len(ix),bool);valid[2]=False
    rotated,audit = displace_scores(a,canon,ix,p,valid)
    np.testing.assert_array_equal(rotated[:2],np.roll(p[:2],1))
    assert rotated[2]==p[2]
    np.testing.assert_array_equal(rotated[3:],np.roll(p[3:],1))
    raw = np.asarray([row(f,1,0) for f in (1,2,3,4)]+[row(f,2,0) for f in (4,5,6)],float)
    canon = np.asarray([1]*4+[2]*3,np.int64)
    ix = np.asarray([[0,1,2,3]],np.int64)
    cut,proof = materialize(raw,canon,ix,np.asarray([.9]),.5,relink=True)
    assert proof["nms"]["selected"]==[0]
    assert len(proof["relink"]["accepted_edges"])==1
    assert cut[0,1]==cut[4,1] and cut[2,1]!=cut[0,1]
    np.testing.assert_array_equal(cut[:,[0,2,3,4,5,6]],raw[:,[0,2,3,4,5,6]])
    kept,proof = materialize(raw,canon,ix,np.asarray([.5]),.5,relink=True)
    np.testing.assert_array_equal(kept,raw)
    assert not proof["relink"]["accepted_edges"]
    twin = np.vstack([raw,[row(f,3,0) for f in (4,5,6)]])
    _,proof = materialize(twin,np.r_[canon,[3]*3],ix,np.asarray([.9]),.5,relink=True)
    assert not proof["relink"]["accepted_edges"]
    # A native ID reused after a long gap is a separate canonical run.
    split = np.asarray([row(f,1,0) for f in (1,2,3,4,100,101,102,103)],float)
    cc = np.asarray([1]*4+[100001]*4,np.int64)
    ii = np.asarray([[0,1,2,3],[4,5,6,7]],np.int64)
    shifted,audit = displace_scores(split,cc,ii,np.asarray([.6,.9]))
    np.testing.assert_array_equal(shifted,[.6,.9])
    returning = np.asarray([row(f,1) for f in (1,2,5,6)]+[row(f,2) for f in (3,4)],float)
    joined,proof = materialize(returning,np.asarray([1]*4+[2]*2,np.int64),np.asarray([[0,1,2,3]],np.int64),
        np.asarray([.9]),.5,relink=True)
    assert len(proof["relink"]["accepted_edges"])==2
    assert proof["relink"]["indirect_original_run_return_paths"]==1
    assert len(set(joined[:,1]))==1
    dd=cut_diagnostics(a,np.full(len(a),7,np.int64),np.asarray([range(i,i+4) for i in range(7)],np.int64),
        np.ones(7,bool),np.asarray([1,-1,0,1,0,-1,0]),np.full(7,.5),[1,4,5])
    assert dd["adjacency"]["AP"]==.4
    assert dd["timing_pm1"]["tp"]==2 and dd["timing_pm1"]["fp"]==0
    assert dd["timing_pm1"]["unmatched_unknown_burden"]==1
    return dict(status="PASS",tests="strict threshold; probability/earlier-right NMS; eligible-stretch cyclic shift; native gap independence; exact K; cut-only ID changes; new-endpoint link; exact nearest ties abstain; indirect same-run return allowed/reported; grouped-tie AP; one-to-one timing/unknown burden")


def synthetic(repo,out):
    """Nine tiny real official scoring calls; no natural footage or GT files."""
    import sys
    repo = Path(repo).resolve()
    sys.path[:0] = [str(repo),str(repo/"tools/study197")]
    from tools.study195.run import score_member_native
    from tools.study195.freeze import write_mot_exact
    def row(f,p,x): return [f,p,x,0,4,4,.9]
    raw = np.asarray([row(f,1,x) for f,x in ((1,-6),(2,-5),(5,0),(6,1))]
        +[row(3,2,-4),row(4,2,-3),row(1,9,-5),row(2,9,5)],float)
    canon = np.asarray([1]*4+[2]*2+[9]*2,np.int64)
    indices = np.asarray([[0,1,2,3]],np.int64)
    polygon = np.asarray([[0,-10],[100,-10],[100,100],[0,100]],np.float32)
    gt = raw[:,:6].copy()
    reference = score_member_native(gt,raw,polygon,10)
    signature = dict(prereg="a"*40,inputs_manifest_sha256="b"*64,predictions_manifest_sha256="c"*64,
        folds=["35","33","W","G"],heldout="35",clip="synthetic")
    result = evaluate_clip(gt,raw,canon,indices,{a:np.asarray([.9],np.float32) for a in ARMS},
        {a:dict(threshold=.5,active=True) for a in ARMS},np.asarray([True]),polygon,10,
        score_native=score_member_native,reference_metrics=reference,signature=signature,out=out,write_mot=write_mot_exact)
    assert len(result["states"])==9
    assert result["states"][MAIN_ARM+"_CUT"]["W"] > result["states"]["K"]["W"]
    assert result["states"][MAIN_ARM+"_CUT_RELINK"]["W"]==result["states"]["K"]["W"]
    verified_manifest(out)
    for folder in Path(out).iterdir():
        if folder.is_dir():
            replay = np.loadtxt(folder/"mot.txt",delimiter=",",usecols=range(7),ndmin=2)
            np.testing.assert_array_equal(replay[:,[0,2,3,4,5,6]],raw[:,[0,2,3,4,5,6]])
    print("PHYSICAL208_SYNTHETIC_PASS",file_sha(Path(out)/"manifest.json"),flush=True)
    return result


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest",action="store_true")
    parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--score",action="store_true")
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--reference206-score",type=Path)
    parser.add_argument("--prereg")
    args = parser.parse_args()
    if sum((args.selftest,args.synthetic,args.score))!=1: parser.error("Choose --selftest, --synthetic or --score")
    if args.selftest: print(json.dumps(selftest(),sort_keys=True))
    elif args.synthetic:
        if args.repo is None or args.out is None: parser.error("--synthetic requires --repo --out")
        synthetic(args.repo,args.out)
    else:
        if any(v is None for v in (args.repo,args.root,args.reference206_score,args.prereg)):
            parser.error("--score requires --repo --root --reference206-score --prereg")
        run_stage(args)
