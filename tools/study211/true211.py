"""Private true-GT event graphs and fixed-K left-vertex necessity labels.

No field IO on import. All true graphs, original GT identities, event dates,
state outcomes and singleton witnesses remain in assessment/, never frozen/
or the minimal per-domain source-label shards. Necessity y is a fixed-graph
matching marginal, not a statement that an event or bee is intrinsically true.
"""
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import graphs211

DOMAINS=("35","33","W","G")
THRESHOLDS=("0.5","0.3")


def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    return value


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(value),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def seal(root):
    root=Path(root)
    manifest={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob("*")) if p.is_file()
              and p not in (root/"manifest.json",root/"SEALED.sha256")}
    save(root/"manifest.json",manifest);value=sha(root/"manifest.json")
    (root/"SEALED.sha256").write_text(value+"\n")
    return value


def verify(root):
    root=Path(root).resolve()
    value=sha(root/"manifest.json")
    if (root/"SEALED.sha256").read_text().strip()!=value:raise ValueError("Complete immutable seal required")
    for name,expected in json.loads((root/"manifest.json").read_text()).items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha(path)!=expected:raise ValueError("Sealed artifact changed")
    return value


def event_key(domain,clip,pid): return f"{domain}/{clip}/{int(pid)}"


def graph_from_metrics(metrics,threshold,*,deletion=False):
    """All legal GT neighbors, not just one chosen maximum matching."""
    if threshold not in THRESHOLDS: raise ValueError("Official threshold .5 or .3 required")
    evidence=metrics["evidence"]
    left=[int(v[0]) for v in evidence["predictions"]]
    right=[int(v[0]) for v in evidence["gt"]]
    if len(set(left))!=len(left) or len(set(right))!=len(right): raise ValueError("Duplicate event vertex")
    edges=evidence["edges05" if threshold=="0.5" else "edges03"]
    graph=graphs211.summarize_graph(left,right,edges,deletion=deletion)
    expected=metrics["iou"][threshold]
    if (graph["nPred"],graph["nReference"],graph["R"])!=(int(metrics["nP"]),int(metrics["nG"]),int(expected["tp"])):
        raise ValueError("True legal graph does not reproduce full native official cardinality")
    graph.update(left_vertices=[dict(pid=int(v[0]),signed_frame=int(v[1])) for v in evidence["predictions"]],
                 right_vertices=[dict(gt_event_id=int(v[0]),signed_frame=int(v[1])) for v in evidence["gt"]],
                 frame_iou=float(threshold),event_tolerance_frames=5,reference_is_true_GT=True,
                 fixed_graph_left_delete_only=bool(deletion))
    return graph


def necessity_records(domain,clip,Kmetrics):
    graph=graph_from_metrics(Kmetrics,"0.5",deletion=True)
    records=[dict(domain=domain,clip=clip,pid=int(f["canonical_id"]),
        event_key=event_key(domain,clip,f["canonical_id"]),y=int(f["drop_left_marginal"])) for f in graph["flags"]]
    if len(records)!=int(Kmetrics["nP"]): raise AssertionError("Every K left event needs a necessity label")
    return records,graph


def fixed_families(active_pairs):
    """One family namespace for K and EVERY subset, not state-local numeric IDs."""
    family={}
    for pair in active_pairs:
        a,b=map(int,pair["canonical_ids"])
        if not a<b or a in family or b in family: raise ValueError("Frozen active pairs must be disjoint")
        family[a]=family[b]=(a,b)
    return family


def family_events(rows,output_raw_indices,original_canon,polygon,active_pairs,track_structure):
    """Event provenance through raw-row bijection, immune to renumbered IDs."""
    rows=np.asarray(rows,float);order=np.asarray(output_raw_indices)
    original_canon=np.asarray(original_canon)
    if (order.shape!=(len(rows),) or not np.issubdtype(order.dtype,np.integer)
            or not np.array_equal(np.sort(order),np.arange(len(original_canon)))):
        raise ValueError("All original rows once in each211 state required")
    structure=track_structure(rows,polygon)
    families=fixed_families(active_pairs);result=[]
    for pid,(frame,direction) in sorted(structure["events"].items()):
        sign,start,end,crosses=structure["tracks"][pid]
        raw=order[np.asarray(structure["index"])[start:end]]
        parents=tuple(sorted(set(map(int,original_canon[raw]))))
        normalized=set()
        for parent in parents: normalized.update(families.get(parent,(parent,)))
        chosen=next(j for j,d in reversed(crosses) if d==direction)
        chosen_raw=int(order[int(structure["index"][chosen])])
        result.append(dict(pid=int(pid),frame=int(frame),direction=int(direction),signed_frame=int(frame)*int(direction),
            original_parent_ids=list(parents),family=list(sorted(normalized)),selected_crossing_raw_index=chosen_raw,
            crossing_raw_indices=[dict(before_raw=int(order[int(structure["index"][j-1])]),
                after_raw=int(order[int(structure["index"][j])]),frame=int(structure["rows"][j,0]),direction=int(d)) for j,d in crosses]))
    return result


def signed_family_diff(before,after):
    """Multiset accounting; unchanged signature need not mean unchanged GT edge."""
    def key(e): return (tuple(e["family"]),int(e["signed_frame"]))
    a,b=Counter(map(key,before)),Counter(map(key,after))
    def expand(counter):
        return [dict(family=list(family),signed_frame=frame,multiplicity=n) for (family,frame),n in sorted(counter.items()) if n]
    return dict(unchanged=expand(a&b),removed=expand(a-b),added=expand(b-a),
        identity_claim=False,scope="Fixed original-run-family signed-event multiset only; not event/bee identity or matching invariance")


def local_event_records(events,graph,parents):
    parents=set(map(int,parents))
    neighbors={int(v["pid"]):[] for v in graph["left_vertices"]}
    for p,g in graph["edges"]: neighbors[int(p)].append(int(g))
    return [dict(e,legal_GT_neighbors=sorted(set(neighbors[e["pid"]]))) for e in events
            if set(e["original_parent_ids"])<=parents and set(e["original_parent_ids"])&parents]


def singleton_diagnostic(pair,K,current):
    """Strict official-event redundancy certificate, never true bee identity."""
    parents=pair["canonical_ids"];answer={}
    for threshold in THRESHOLDS:
        old=local_event_records(K["family_events"],K["true_graphs"][threshold],parents)
        new=local_event_records(current["family_events"],current["true_graphs"][threshold],parents)
        by_parent={int(parent):[e for e in old if e["original_parent_ids"]==[int(parent)]] for parent in parents}
        two_original=all(len(v)==1 for v in by_parent.values())
        left=[by_parent[int(p)][0] for p in parents] if two_original else []
        same_direction=two_original and left[0]["direction"]==left[1]["direction"]
        singleton_neighbors=(same_direction and len(left[0]["legal_GT_neighbors"])==1
                             and left[0]["legal_GT_neighbors"]==left[1]["legal_GT_neighbors"])
        target=left[0]["legal_GT_neighbors"][0] if singleton_neighbors else None
        survivor=(len(new)==1 and set(new[0]["original_parent_ids"])==set(parents))
        survivor_edge=survivor and target is not None and target in new[0]["legal_GT_neighbors"]
        km=K["metrics"]["iou"][threshold];cm=current["metrics"]["iou"][threshold]
        tp_non_decrease=int(cm["tp"])>=int(km["tp"])
        pred_drops_one=int(cm["pred"])==int(km["pred"])-1
        def counts(m):return dict(tp=int(m["tp"]),fp=int(m["pred"])-int(m["tp"]),fn=int(m["gt"])-int(m["tp"]))
        a,b=counts(km),counts(cm)
        delta={key:b[key]-a[key] for key in a}
        delta["E"]=(b["fp"]+b["fn"])-(a["fp"]+a["fn"])
        conditions=dict(two_original_parent_events=two_original,same_direction=bool(same_direction),
            identical_singleton_K_GT_neighbors=bool(singleton_neighbors),one_merged_survivor_with_that_edge=bool(survivor_edge),
            whole_scene_TP_not_lower=tp_non_decrease,whole_scene_pred_exactly_minus1=pred_drops_one)
        answer[threshold]=dict(before_parent_events=old,new_family_events=new,singleton_GT_event_id=target,
            conditions=conditions,strict_event_redundancy_witness=all(conditions.values()),delta=delta,
            delta_pred=int(cm["pred"])-int(km["pred"]),true_bee_duplicate_claim=False)
    answer.update(pair_id=pair["pair_id"],delta_W=current["W"]-K["W"] if current["W"] is not None and K["W"] is not None else None,
        delta_window_numerator=int(current["numerator"])-int(K["numerator"]),
        scope="A fixed native official event-graph redundancy witness; not whole-graph physical bee identity")
    return answer


def assess_state(domain,clip,rows,output_raw_indices,original_canon,active_pairs,GT,polygon,nframes,
                 score_native,track_structure):
    """Only this explicit API accepts true GT; no deployment/model inputs here."""
    metrics=score_native(GT,np.asarray(rows,float),polygon,int(nframes))
    pred=np.asarray(metrics["pred_bins"],np.int64);truth=np.asarray(metrics["gt_bins"],np.int64)
    numerator=int(np.abs(pred-truth).sum());denominator=30*int(metrics["nG"])
    graphs={t:graph_from_metrics(metrics,t,deletion=False) for t in THRESHOLDS}
    events=family_events(rows,output_raw_indices,original_canon,polygon,active_pairs,track_structure)
    if {(e["pid"],e["signed_frame"]) for e in events}!={(int(v[0]),int(v[1])) for v in metrics["evidence"]["predictions"]}:
        raise ValueError("Raw provenance events differ from native official event table")
    return dict(schema="match211-private-true-state-v1",domain=domain,clip=clip,metrics=metrics,true_graphs=graphs,
        family_events=events,numerator=numerator,denominator=denominator,W=None if not denominator else 100*numerator/denominator,
        TP_FP_FN={t:dict(tp=int(metrics["iou"][t]["tp"]),fp=int(metrics["nP"])-int(metrics["iou"][t]["tp"]),
            fn=int(metrics["nG"])-int(metrics["iou"][t]["tp"]),pred=int(metrics["nP"]),gt=int(metrics["nG"]),
            tp_id=int(metrics["iou"][t]["tp_id"]),f1=float(metrics["iou"][t]["f1"])) for t in THRESHOLDS},
        true_outcomes_private_until_all_models_and_choices_sealed=True)


def prepare(args):
    """Sole-CPU phase: sealed GT-free22+peers -> private graphs -> source shards.

    Prints engineering status only. No numeric target outcome is returned to
    callers, fit input files, global source deployments or stdout.
    """
    import gc
    import re
    import subprocess
    import sys
    import tempfile
    repo,root=args.repo.resolve(),args.root.resolve()
    if root.drive.upper()!="D:" or not re.fullmatch("[0-9a-f]{40}",args.prereg or ""):
        raise ValueError("D: root and full published prereg SHA required")
    if not (root/"frozen/SEALED.sha256").is_file():raise ValueError("Complete GT-free22+peer seal must precede true GT")
    subprocess.run(["git","merge-base","--is-ancestor",args.prereg,"origin/main"],cwd=repo,check=True,capture_output=True)
    log=subprocess.run(["git","show",args.prereg+":docs/experiment_log.md"],cwd=repo,check=True,capture_output=True).stdout.decode("utf8")
    if log.count("[peer-event-capacity-v1]")!=1:raise ValueError("Unique211 published marker required")
    import freeze211
    frozen_sha=verify(root/"frozen")
    if frozen_sha!=args.frozen_sha:raise ValueError("GT-free frozen seal differs from CPU handoff")
    top=json.loads((root/"frozen/deployment.json").read_text())
    if top["prereg"]!=args.prereg or top["states"]!=22 or len(top["clips"])!=6 or top["GT_access"]:
        raise ValueError("Original-family sixclip22state GT-free closure required")
    if sha(args.reference210_score/"manifest.json")!=(args.reference210_score/"SEALED.sha256").read_text().strip():
        raise ValueError("Historical210 reference must be sealed")
    if sha(args.reference210_score/"manifest.json")!="090f56cc8014893ff3626ae8887d69368092ecb2efc69b021bb3c26acb5bbae1":
        raise ValueError("Wrong registered210 native reference")
    reference_manifest=json.loads((args.reference210_score/"manifest.json").read_text())
    sys.path[:0]=[str(repo),str(repo/"tools/study197")]
    from tools.study195.run import score_member_native,_gt_provider
    from tools.study195.freeze import write_mot_exact
    from tools.study197.verify import track_structure
    import inspect
    code=dict(true211=sha(__file__),graphs211=sha(Path(graphs211.__file__)),
        official_scorer=sha(inspect.getsourcefile(score_member_native)),track_structure=sha(inspect.getsourcefile(track_structure)))
    identity=dict(prereg=args.prereg,frozen_manifest_sha256=frozen_sha,
        reference210_score_manifest_sha256=sha(args.reference210_score/"manifest.json"),code=code)
    assessment=root/"assessment"
    labels=root/"labels"
    if (labels/"SEALED.sha256").exists():raise FileExistsError("Source-label stage already sealed")
    assessment.mkdir(parents=True,exist_ok=True)
    if (assessment/"RUNNING.json").exists() and json.loads((assessment/"RUNNING.json").read_text())!=identity:
        raise ValueError("Different true-graph input/code round cannot resume")
    save(assessment/"RUNNING.json",identity)
    shards={h:dict(schema="match211-source-labels-v1",domain=h,prereg=args.prereg,
        frozen_manifest_sha256=frozen_sha,records=[],
        label_rule="Official true-GT IoU.5 fixed-K graph maximum-cardinality loss after deleting one left event; ownership and other graph vertices/edges fixed") for h in DOMAINS}
    private_index=[];total=0;fresh=0;resumed=0
    for ref in top["clips"]:
        folder=(root/"frozen"/ref["path"]).resolve()
        if not folder.is_relative_to((root/"frozen").resolve()) or sha(folder/"manifest.json")!=ref["seal"]:
            raise ValueError("Frozen clip path/seal differs")
        clip=freeze211.load_clip(folder);meta=clip["metadata"]
        domain,name=meta["domain"],meta["clip"]
        if (domain,name)!=(ref["domain"],ref["clip"]):raise ValueError("Frozen clip identity differs")
        Kstates=[s for s in clip["states"] if s["is_K"]]
        if len(Kstates)!=1 or Kstates[0]["selected_pair_ids"]:raise ValueError("Exactly one empty K state required")
        GT,GTproof=_gt_provider(repo,[dict(domain=domain,video=name,split=meta["split"],nf=meta["nf"])],DOMAINS,"evaluation")
        truth=GT[name]
        active=clip["active_pairs"]
        if len(clip["states"])!=2**len(active):raise ValueError("Every disjoint active subset must be represented")
        frozen_predictors=clip["K_event_features"]
        outcomes={};state_records=[]
        for state in clip["states"]:
            state_id=state["state_id"]
            if not re.fullmatch("[01]+",state_id) or state_id in outcomes:raise ValueError("Unique binary state ID required")
            target=assessment/domain/name/state_id
            sig=dict(identity,domain=domain,clip=name,state_id=state_id,rows_sha256=state["rows_sha256"],GT_entry=GTproof[name])
            if target.exists():
                verify(target)
                if json.loads((target/"receipt.json").read_text())!=sig:raise ValueError("Cached true state signature differs")
                result=json.loads((target/"score.json").read_text());resumed+=1
            else:
                rows=freeze211.state_rows(clip,state)
                with np.load(folder/state["materialization_file"],allow_pickle=False) as z:order=z["output_raw_indices"]
                receiver=clip["native"]["receiver"]
                if (len(rows)!=len(receiver) or not np.array_equal(np.sort(order),np.arange(len(receiver)))
                    or rows[:,[0,2,3,4,5,6]].tobytes()!=receiver[order][:,[0,2,3,4,5,6]].tobytes()):
                    raise ValueError("Frozen state is not a full original-row bijection")
                target.parent.mkdir(parents=True,exist_ok=True)
                tmp=Path(tempfile.mkdtemp(prefix="."+state_id+"-",dir=target.parent))
                write_mot_exact(rows,tmp/"mot.txt")
                disk=np.loadtxt(tmp/"mot.txt",delimiter=",",usecols=range(7),ndmin=2) if len(rows) else np.empty((0,7))
                if disk.tobytes()!=rows.tobytes():raise ValueError("Serialized complete MOT changed rows")
                result=assess_state(domain,name,disk,order,clip["native"]["receiver_canon"],active,truth,meta["polygon"],meta["nf"],score_member_native,track_structure)
                result.update(state_id=state_id,selected_pair_ids=state["selected_pair_ids"],frozen_materialization=state["materialization_file"],rows_sha256=state["rows_sha256"])
                if result["metrics"]["nP"]!=state["nP"] or not np.array_equal(result["metrics"]["pred_bins"],state["pred_windows"]):
                    raise ValueError("True scoring altered the frozen GT-free prediction event ledger")
                for peer in state["peer_graphs"]:
                    if {(v["canonical_id"],v["signed_frame"]) for v in peer["prediction_vertices"]}!={(v["pid"],v["signed_frame"]) for v in result["family_events"]}:
                        raise ValueError("Peer/true scoring use different frozen prediction vertices")
                if state["is_K"]:
                    oldfile=f'clips/{domain}/{name}/K/score.json'
                    if sha(args.reference210_score/oldfile)!=reference_manifest[oldfile]:raise ValueError("Historical K reference changed")
                    old=json.loads((args.reference210_score/oldfile).read_text())["metrics"]
                    if plain(result["metrics"])!=plain(old):raise ValueError("Full K official metrics/evidence differs from210")
                np.save(tmp/"output_raw_indices.npy",order,allow_pickle=False)
                save(tmp/"score.json",result);save(tmp/"receipt.json",sig);seal(tmp);tmp.rename(target)
                fresh+=1
            outcomes[state_id]=result
            state_records.append(dict(state_id=state_id,selected_pair_ids=state["selected_pair_ids"],path=f'{domain}/{name}/{state_id}',seal=(target/"SEALED.sha256").read_text().strip()))
            total+=1
        K=outcomes[Kstates[0]["state_id"]]
        records,Kdeletion=necessity_records(domain,name,K["metrics"])
        if {r["pid"] for r in records}!={int(f["canonical_id"]) for f in frozen_predictors}:
            raise ValueError("Source labels do not cover every frozen K-event predictor")
        shards[domain]["records"].extend(records)
        singles={}
        for pair in active:
            s=[state for state in clip["states"] if state["selected_pair_ids"]==[pair["pair_id"]]]
            if len(s)!=1:raise ValueError("Every active action needs exactly one singleton state")
            singles[pair["pair_id"]]=singleton_diagnostic(pair,K,outcomes[s[0]["state_id"]])
        diagnostic=dict(schema="match211-private-clip-diagnostics-v1",domain=domain,clip=name,K_state_id=Kstates[0]["state_id"],
            K_fixed_graph_deletion=Kdeletion,singletons=singles,
            signed_family_diffs={sid:signed_family_diff(K["family_events"],value["family_events"]) for sid,value in outcomes.items()},
            outcomes_private_until_model_choice_common_seal=True)
        save(assessment/domain/name/"diagnostics.json",diagnostic)
        private_index.append(dict(domain=domain,clip=name,GT_entry=GTproof[name],states=state_records))
        del outcomes,clip,truth,GT,K,diagnostic
        print("TRUE211_CLIP_READY",domain,name,flush=True);gc.collect()
    if total!=22:raise ValueError("Original-family true22-state scope differs")
    save(assessment/"index.json",dict(schema="match211-private-assessment-v1",input_receipt=identity,clips=private_index,
        fresh_native_calls=fresh,resumed_states=resumed,source_label_scope="Fixed K official true graph .5 left-vertex deletion only",private=True))
    assessment_sha=seal(assessment)
    if labels.exists():raise FileExistsError("Unsealed source-label destination already exists")
    labels.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix=".labels211-",dir=labels.parent))
    for h,shard in shards.items():save(tmp/(h+".json"),shard)
    save(tmp/"receipt.json",dict(prereg=args.prereg,frozen_manifest_sha256=frozen_sha,
        graph_assessment_manifest_sha256=assessment_sha,producer_code_sha256=sha(__file__),domains=list(DOMAINS),
        source_loader_rule="Validate other-three whitelist before opening any source shard; assessment outcomes and true GT IDs are not fit inputs"))
    label_sha=seal(tmp);tmp.rename(labels)
    print("TRUE211_ASSESSMENT_SEALED",assessment_sha,"LABELS",label_sha,flush=True)
    return dict(assessment_manifest_sha256=assessment_sha,label_manifest_sha256=label_sha)


def selftest():
    def metric(pred,gt,edges):
        tp=len(graphs211.maximum_matching(pred,gt,edges))
        return dict(nP=len(pred),nG=len(gt),evidence=dict(predictions=[[p,3,0,0,1,1] for p in pred],gt=[[g,3] for g in gt],edges05=edges,edges03=edges),
            iou={t:dict(tp=tp,pred=len(pred),gt=len(gt),tp_id=tp,f1=2*tp/max(len(pred)+len(gt),1)) for t in THRESHOLDS})
    redundant=metric([1,2],[9],[[1,9],[2,9]])
    records,graph=necessity_records("35","toy",redundant)
    assert [r["y"] for r in records]==[0,0]
    essential=metric([1,2],[9,10],[[1,9],[2,10]])
    assert all(r["y"]==1 for r in necessity_records("35","toy",essential)[0])
    def ev(pid,parents):return dict(pid=pid,frame=3,direction=1,signed_frame=3,original_parent_ids=parents,family=[1,2])
    before=dict(metrics=redundant,true_graphs={t:graph_from_metrics(redundant,t) for t in THRESHOLDS},family_events=[ev(1,[1]),ev(2,[2])],W=0.,numerator=0)
    merged=metric([1],[9],[[1,9]])
    after=dict(metrics=merged,true_graphs={t:graph_from_metrics(merged,t) for t in THRESHOLDS},family_events=[ev(1,[1,2])],W=0.,numerator=0)
    proof=singleton_diagnostic(dict(pair_id="1|2",canonical_ids=[1,2]),before,after)
    assert all(proof[t]["strict_event_redundancy_witness"] for t in THRESHOLDS)
    up_before=metric([1,2,3],[9,10],[[1,9],[2,9]])
    up_after=metric([1,3],[9,10],[[1,9],[3,10]])
    a=dict(before,metrics=up_before,true_graphs={t:graph_from_metrics(up_before,t) for t in THRESHOLDS},
        family_events=before["family_events"]+[dict(ev(3,[3]),family=[3])])
    b=dict(after,metrics=up_after,true_graphs={t:graph_from_metrics(up_after,t) for t in THRESHOLDS},
        family_events=after["family_events"]+[dict(ev(3,[3]),family=[3])])
    assert singleton_diagnostic(dict(pair_id="1|2",canonical_ids=[1,2]),a,b)["0.5"]["strict_event_redundancy_witness"]
    different=metric([1,2],[9,10],[[1,9],[2,10]])
    before["metrics"]=different;before["true_graphs"]={t:graph_from_metrics(different,t) for t in THRESHOLDS}
    assert not singleton_diagnostic(dict(pair_id="1|2",canonical_ids=[1,2]),before,after)["0.5"]["strict_event_redundancy_witness"]
    d=signed_family_diff([ev(1,[1]),ev(2,[2])],[ev(1,[1,2])])
    assert d["unchanged"][0]["multiplicity"]==1 and d["removed"][0]["multiplicity"]==1
    return dict(passed=True,scope="synthetic fixed-graph labels/strict-event witness/family multiplicity; no natural outcomes")


def synthetic(repo,out):
    import sys
    repo=Path(repo).resolve();sys.path[:0]=[str(repo),str(repo/"tools/study197")]
    from tools.study195.run import score_member_native
    from tools.study197.verify import track_structure
    def row(f,p,x):return [f,p,x,0,2,2,.9]
    raw=np.asarray([row(1,1,-3),row(2,1,1),row(3,2,-3),row(4,2,1)],float)
    GT=raw[:,:6].copy();GT[:,1]=9
    polygon=np.asarray([[0,-10],[50,-10],[50,10],[0,10]],np.float32)
    merged=raw.copy();merged[:,1]=1
    canon=np.asarray([1,1,2,2],np.int64);pair=dict(pair_id="1|2",canonical_ids=[1,2])
    K=assess_state("35","synthetic",raw,np.arange(4),canon,[pair],GT,polygon,20,score_member_native,track_structure)
    changed=assess_state("35","synthetic",merged,np.arange(4),canon,[pair],GT,polygon,20,score_member_native,track_structure)
    labels,deletion=necessity_records("35","synthetic",K["metrics"])
    assert [r["y"] for r in labels]==[0,0]
    witness=singleton_diagnostic(pair,K,changed)
    assert all(witness[t]["strict_event_redundancy_witness"] for t in THRESHOLDS)
    difference=signed_family_diff(K["family_events"],changed["family_events"])
    assert len(difference["removed"])==1 and len(difference["unchanged"])==1
    out=Path(out)
    if out.exists():raise FileExistsError("Immutable synthetic output exists")
    out.mkdir(parents=True)
    save(out/"K.json",K);save(out/"merged.json",changed)
    save(out/"labels.json",dict(synthetic=True,records=labels))
    save(out/"proof.json",dict(synthetic=True,witness=witness,fixed_K_deletion=deletion,signed_diff=difference))
    value=seal(out)
    print("TRUE211_SYNTHETIC_PASS",value,flush=True)
    return value


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--selftest",action="store_true")
    p.add_argument("--synthetic",action="store_true");p.add_argument("--prepare",action="store_true")
    p.add_argument("--repo",type=Path);p.add_argument("--out",type=Path);p.add_argument("--root",type=Path)
    p.add_argument("--reference210-score",type=Path);p.add_argument("--prereg");p.add_argument("--frozen-sha")
    args=p.parse_args()
    if sum((args.selftest,args.synthetic,args.prepare))!=1:p.error("Choose --selftest, --synthetic or --prepare")
    if args.selftest:print(json.dumps(selftest(),sort_keys=True))
    elif args.synthetic:
        if args.repo is None or args.out is None:p.error("--synthetic needs --repo --out")
        synthetic(args.repo,args.out)
    else:
        if any(v is None for v in (args.repo,args.root,args.reference210_score,args.prereg,args.frozen_sha)):
            p.error("--prepare needs --repo --root --reference210-score --prereg --frozen-sha")
        prepare(args)
