"""Private two-stage endpoint-peer evaluation; synthetic default, no field IO.

Stage1 count capacity gates training. Stage2 uses sealed K/A full event graphs,
physically rechecks all I members and one member/clip per fold for each control.
GT appears only in the explicit final physical-score callback.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fractions import Fraction
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

ARMS = ("K", "T197", "Tg", "N", "B", "I", "F", "Y", "M", "O")
DOMAINS = ("35", "33", "W", "G")
GROUPS = {"35":4, "33":4, "W":3, "G":3}
PROBES = ("N", "B", "F", "Y", "M", "Tg")


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [_plain(v) for v in value]
    return value


def combine(videos):
    """One physical seed group: clips pool first; three member F1 stay separate."""
    if not videos:
        raise ValueError("Nonempty group required")
    domain, group = videos[0]["domain"], videos[0]["group"]
    order = [m["member"] for m in videos[0]["members"]]
    if len(order) != 3 or len(set(order)) != 3 or len({v["video"] for v in videos}) != len(videos):
        raise ValueError("Three distinct members and distinct clips required")
    numerator, denominator, clips = 0, 0, []
    for video in videos:
        if video["domain"] != domain or video["group"] != group or [m["member"] for m in video["members"]] != order:
            raise ValueError("Mixed group or member order")
        mm = [m["metrics"] for m in video["members"]]
        gt = np.asarray(mm[0]["gt_bins"], dtype=np.int64)
        if any(m["nG"] != mm[0]["nG"] or not np.array_equal(m["gt_bins"],gt) for m in mm):
            raise ValueError("Three GT copies disagree")
        pred = sum((np.asarray(m["pred_bins"],dtype=np.int64) for m in mm),np.zeros_like(gt))
        raw = int(abs(pred-3*gt).sum())
        numerator += raw; denominator += 90*int(mm[0]["nG"])
        clips.append(dict(video=video["video"],numerator=raw,denominator=90*int(mm[0]["nG"]),
                          gt_bins=gt.tolist(),sum_pred_bins=pred.tolist()))
    iou = {}
    for threshold in ("0.5","0.3"):
        members = []
        for index, identity in enumerate(order):
            totals = {k:sum(int(v["members"][index]["metrics"]["iou"][threshold][k]) for v in videos) for k in ("tp","pred","gt")}
            totals.update(member=identity,f1=2*totals["tp"]/max(totals["pred"]+totals["gt"],1))
            members.append(totals)
        iou[threshold] = dict(members=members,mean_member_f1=sum(m["f1"] for m in members)/3)
    return dict(domain=domain,group=group,numerator=numerator,denominator=denominator,
                E3=100*numerator/denominator if denominator else None,iou=iou,clips=clips)


def summarize(scored, field_layout=True):
    if set(scored) != set(ARMS):
        raise ValueError("All ten registered arms required")
    keys = [{(v["domain"],v["group"],v["video"]) for v in scored[a]} for a in ARMS]
    if any(k != keys[0] or len(k) != len(scored[a]) for a,k in zip(ARMS,keys)):
        raise ValueError("Unmatched actual clips")
    domains = {}
    for domain in sorted({k[0] for k in keys[0]}):
        ids = sorted({k[1] for k in keys[0] if k[0] == domain})
        if field_layout and len(ids) != GROUPS[domain]:
            raise ValueError("Expected 4/4/3/3 groups")
        methods = {}
        for arm in ARMS:
            groups = [combine([v for v in scored[arm] if v["domain"] == domain and v["group"] == g]) for g in ids]
            if any(g["denominator"] == 0 for g in groups):
                raise ValueError("Entire group GT empty")
            if field_layout and any(len(g["clips"]) != (2 if domain in ("W","G") else 1) for g in groups):
                raise ValueError("Wrong actual clip count")
            methods[arm] = dict(groups=groups,
                E3=float(sum(Fraction(100*g["numerator"],g["denominator"]) for g in groups)/len(groups)),
                F1={t:float(sum((sum(Fraction(2*m["tp"],max(m["pred"]+m["gt"],1)) for m in g["iou"][t]["members"])/3
                                for g in groups),Fraction())/len(groups)) for t in ("0.5","0.3")})
        domains[domain] = dict(methods=methods,differences_vs_K={a:dict(E3=methods[a]["E3"]-methods["K"]["E3"],
            F1_05=methods[a]["F1"]["0.5"]-methods["K"]["F1"]["0.5"]) for a in ARMS})
    result = dict(schema="endpoint-peer-score-v1",domains=domains,strict_identity_metric_reported=False)
    if set(domains) == set(DOMAINS):
        delta = {h:domains[h]["differences_vs_K"]["I"] for h in DOMAINS}
        result["investment_checks"] = dict(head35_gain=delta["35"]["E3"] < -1.4,
            head33_gain=delta["33"]["E3"] < -1.3,W_no_harm=delta["W"]["E3"] <= 1.4,
            F1_no_harm=all(delta[h]["F1_05"] >= -.005 for h in ("35","33","W")))
        result["investment_pass"] = all(result["investment_checks"].values())
        result["identity_attribution_pass"] = all(domains[h]["methods"]["I"]["E3"] < domains[h]["methods"][other]["E3"]
                                                  for h in ("35","33") for other in ("B","M"))
    return result


def capacity_gate(groups):
    """Exact source-free stage1 gate, groups carry K/O raw counts and denominator.

    groups=[{domain,group,K_numerator,O_upper,O_lower,denominator,...}].
    Physical/F1 feasibility is reported separately and never changes this gate.
    """
    heads = {}
    for hive, line in (("35",Fraction(14,10)),("33",Fraction(13,10))):
        rows = [g for g in groups if g["domain"] == hive]
        if len(rows) != 4 or len({g["group"] for g in rows}) != 4:
            raise ValueError("Four complete groups per head required")
        for g in rows:
            if not 0 <= g["O_lower"] <= g["O_upper"] <= g["K_numerator"] or g["denominator"] <= 0:
                raise ValueError("Invalid count certificate or original-option witness")
        guaranteed = sum(Fraction(100*(g["K_numerator"]-g["O_upper"]),g["denominator"]) for g in rows)/4
        possible = sum(Fraction(100*(g["K_numerator"]-g["O_lower"]),g["denominator"]) for g in rows)/4
        heads[hive] = dict(guaranteed_gain=float(guaranteed),maximum_possible_gain=float(possible),
                           required_strict_gain=float(line),pass_gate=guaranteed > line,
                           excluded=possible <= line,all_optimal=all(g["O_lower"] == g["O_upper"] for g in rows))
    return dict(heads=heads,train_allowed=all(h["pass_gate"] for h in heads.values()),
                f1_note="Count gate only; physical O F1 is one witness, not an F1-constrained infeasibility proof")


def assert_metrics(actual, expected):
    for key in ("nG","nP","gt_bins","pred_bins"):
        np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)
    for t in ("0.5","0.3"):
        for k in ("tp","pred","gt"):
            if int(actual["iou"][t][k]) != int(expected["iou"][t][k]):
                raise ValueError(f"Graph/native discrepancy {t}/{k}")


def evaluate_explicit(groups, decisions, graphs, oracle_metrics, gt_provider, out,
                      graph_module, physical_module, references=None, field_layout=True, eligible=None,
                      resume_identity=None):
    """All four prediction seals MUST be verified by the orchestration caller.

    decisions[unit][str(member)][K/T197/Tg/N/B/I/F/Y/M] -> global cut IDs.
    graphs[unit][member] is labels185's sealed K/A full-graph model.
    oracle_metrics[unit][member] reuses physically verified stage1 O metrics.
    """
    # All eligibility and mask checks precede the first physical GT callback.
    if field_layout and eligible is None:
        raise ValueError("Sealed eligible candidate sets required")
    if set(decisions) != {g["unit"] for g in groups}:
        raise ValueError("Complete target masks required")
    for group in groups:
        for member in group["members"]:
            unit,identity = group["unit"],str(member["member"])
            masks = decisions[unit][identity]
            if set(masks) != set(ARMS)-{"O"} or masks["K"]:
                raise ValueError("Wrong arm mask inventory")
            candidates = {c["candidate_id"]:c for c in member["bundle"]["candidates"]}
            if any(len(ids) != len(set(ids)) or not set(ids) <= set(candidates) for ids in masks.values()):
                raise ValueError("Unknown or duplicate selected candidate")
            if eligible is not None:
                allowed = set(eligible[unit][identity])
                if any(not set(masks[a]) <= allowed for a in ("Tg","N","B","I","F","Y","M")):
                    raise ValueError("New model selected an ineligible candidate")
                if set(masks["Tg"]) != set(masks["T197"]) & allowed:
                    raise ValueError("Tg differs from gated frozen T197")
            strata = lambda arm:Counter((candidates[c]["delta"],candidates[c]["direction"]) for c in masks[arm])
            if strata("I") != strata("M"):
                raise ValueError("I/M member-clip directional add/drop cut counts differ")
    if resume_identity is None:
        raw_digest = hashlib.sha256()
        for group in groups:
            raw_digest.update(json.dumps([group[k] for k in ("unit","domain","group","video","nf","polygon")]).encode())
            for member in group["members"]:
                raw = np.asarray(member["raw"])
                raw_digest.update(str(raw.shape).encode()); raw_digest.update(raw.tobytes())
        resume_identity = dict(raw=raw_digest.hexdigest(),masks=object_sha(decisions),graphs=object_sha(graphs),
                               oracle=object_sha(oracle_metrics),references=object_sha(references),eligible=object_sha(eligible),
                               graph_code=sha(graph_module.__file__),physical_code=sha(physical_module.__file__))
    prepare_output(out,resume_identity)
    out = Path(out)
    (out/"physical").mkdir(exist_ok=True)
    scored = {a:[] for a in ARMS}; receipts = []; probed = set()
    for gi,group in enumerate(groups):
        unit = group["unit"]; domain = group["domain"]
        records = {a:dict(unit=unit,domain=domain,group=group["group"],video=group["video"],members=[]) for a in ARMS}
        gt = None
        for mi,member in enumerate(group["members"]):
            identity = str(member["member"])
            masks = decisions[unit][identity]
            if set(masks) != set(ARMS)-{"O"} or masks["K"]:
                raise ValueError("Wrong arm mask inventory")
            cache = {}
            for arm in ARMS:
                if arm == "O":
                    metrics = oracle_metrics[unit][identity]
                else:
                    key = tuple(sorted(masks[arm]))
                    if key not in cache:
                        cache[key] = graph_module.evaluate(graphs[unit][identity], masks[arm])
                    metrics = cache[key]
                if references is not None and arm in ("K","T197"):
                    assert_metrics(metrics,references[unit][identity][arm])
                must_check = arm == "I" or arm in PROBES and domain not in probed and mi == 0
                if must_check:
                    if gt is None: gt = gt_provider(group)
                    path = out/"physical"/f"u{gi:02d}_m{mi}_{arm}"
                    physical = physical_module.score_member(gt,member,masks[arm],group["polygon"],group["nf"],path,
                                                             expected_graph=metrics)
                    actual = physical.get("metrics",physical)
                    assert_metrics(actual,metrics)
                    receipts.append(dict(unit=unit,member=identity,arm=arm,path=path.relative_to(out).as_posix(),proof=physical))
                records[arm]["members"].append(dict(member=identity,metrics=metrics,
                                                       selected_ids=masks[arm] if arm != "O" else None))
        probed.add(domain)
        for arm in ARMS: scored[arm].append(records[arm])
    summary = summarize(scored,field_layout)
    for domain,data in summary["domains"].items():
        for arm,method in data["methods"].items():
            if arm != "O":
                method["cut_count"] = sum(len(decisions[g["unit"]][str(m["member"])][arm]) for g in groups if g["domain"] == domain for m in g["members"])
    summary["physical_checks"] = dict(I=sum(r["arm"] == "I" for r in receipts),controls=sum(r["arm"] != "I" for r in receipts))
    summary["physical_checks"]["reused"] = sum(bool(r["proof"].get("reused")) for r in receipts)
    for name,value in (("native_scores.json",scored),("summary.json",summary),("physical_receipts.json",receipts)):
        (out/name).write_text(json.dumps(_plain(value),indent=2,allow_nan=False))
    return summary


def selfcheck():
    rows = [dict(domain=h,group=g,K_numerator=400,O_upper=200,O_lower=200,denominator=10000)
            for h in ("35","33") for g in range(4)]
    assert capacity_gate(rows)["train_allowed"]
    # Strict boundary is not a pass; insufficient capacity does not start fit.
    for r in rows:
        if r["domain"] == "35": r["O_upper"] = r["O_lower"] = 260
    assert not capacity_gate(rows)["train_allowed"]
    metric = lambda ng,p,tp:dict(nG=ng,nP=p,gt_bins=[[ng,0]]*30,pred_bins=[[p,0]]*30,
        iou={t:dict(tp=tp,pred=p,gt=ng) for t in ("0.5","0.3")})
    a = dict(domain="W",group=1,video="a",members=[dict(member=str(i),metrics=metric(1,p,min(p,1))) for i,p in enumerate((2,0,1))])
    b = dict(domain="W",group=1,video="b",members=[dict(member=str(i),metrics=metric(9,9,9)) for i in range(3)])
    pooled = combine([a,b])
    assert pooled["numerator"] == 0 and pooled["denominator"] == 900
    assert pooled["iou"]["0.5"]["members"][0]["f1"] == 20/21
    report = summarize({arm:[a,b] for arm in ARMS},field_layout=False)
    assert report["domains"]["W"]["methods"]["I"]["E3"] == 0
    return dict(synthetic_pass=True,training_called=False,strict_boundary_passed=False)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def object_sha(value):
    digest = hashlib.sha256()
    for part in json.JSONEncoder(sort_keys=True,separators=(",",":"),allow_nan=False).iterencode(_plain(value)):
        digest.update(part.encode())
    return digest.hexdigest()


def prepare_output(out,identity):
    """Only an exact same-round partial output is resumable; never delete files."""
    out = Path(out); identity = _plain(identity)
    marker = out/"RUNNING.json"
    if out.exists():
        if (out/"SEALED.sha256").exists():
            raise FileExistsError("Completed score already sealed")
        if not marker.exists() or json.loads(marker.read_text()) != identity:
            raise ValueError("Existing score belongs to different inputs or algorithms")
    else:
        out.mkdir(parents=True)
        with marker.open("x") as stream: json.dump(identity,stream,sort_keys=True,allow_nan=False)


def seal(out):
    out = Path(out)
    manifest = {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
    digest = sha(out/"manifest.json")
    (out/"SEALED.sha256").write_text(digest+"\n")
    return digest


def field_score(args):
    """Verify stage1 first; a failed guard returns before loading any models/GT."""
    import sys
    if (args.out/"SEALED.sha256").exists():
        raise FileExistsError("Completed evaluation already sealed; no model or GT reopened")
    stage = args.stage1
    stage_manifest_sha = sha(stage/"manifest.json")
    if (stage/"SEALED.sha256").read_text().strip() != stage_manifest_sha:
        raise ValueError("Final stage1 manifest required, not a count checkpoint")
    manifest = json.loads((stage/"manifest.json").read_text())
    if sha(stage/"capacity.json") != manifest["capacity.json"]:
        raise ValueError("Capacity changed")
    capacity = json.loads((stage/"capacity.json").read_text())
    if not capacity.get("all60_native_graph_verified") or capacity.get("physical_verified_members") != 60:
        raise ValueError("All60 native/graph validations required")
    guard = capacity_gate(capacity["group_records"])
    if guard["train_allowed"] != capacity["stage2_authorized_by_count_gate"]:
        raise ValueError("Capacity resource gate disagrees with exact count aggregation")
    if not guard["train_allowed"]:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.mkdir(parents=True)
        report = dict(status="stage1_stop",stage2_started=False,models_loaded=False,target_GT_opened=False,
            stage1_manifest_sha256=stage_manifest_sha,capacity_sha256=sha(stage/"capacity.json"),
            guard=guard,hives=capacity["hives"],group_records=capacity["group_records"],
            scope="Same60 §197 F members; eligible endpoint ID-cut family only; all30 phases share masks",
            f1_note="Observed count-optimal O F1 is one witness; failed witness does not prove no F1-feasible mask")
        (args.out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False))
        print("STAGE1_STOP "+seal(args.out),flush=True)
        return report
    if any(getattr(args,k) is None for k in ("repo","peers","source197_frozen","source197_score","predictions","prereg")):
        raise ValueError("Passed stage1 requires all registered stage2 paths")
    if not re.fullmatch(r"[0-9a-f]{40}",args.prereg):
        raise ValueError("Full prereg SHA required")
    repo = args.repo.resolve(); sys.path.insert(0,str(repo))
    subprocess.run(["git","merge-base","--is-ancestor",args.prereg,"origin/main"],cwd=repo,check=True,capture_output=True)
    import freeze as peerfreeze
    import model198
    import graph198
    import physical198
    # The archived197 freezer intentionally imports its sibling candidates.
    sys.path.insert(0,str(repo/"tools"/"study197"))
    from tools.study197 import freeze as sourcefreeze
    from tools.study195 import run as previous
    predictions = model198.load_predictions(args.predictions)
    receipt = predictions["receipt"]
    peer_receipt = peerfreeze.verify(args.peers)
    if receipt["prereg"] != args.prereg or receipt["peerfreeze_receipt"] != peer_receipt:
        raise ValueError("Stage2 predictions/input registration changed")
    if receipt["guard_receipt"]["guard_sha256"] != sha(stage/"capacity.json"):
        raise ValueError("Model fitted under another stage1 guard")
    peer_groups = peerfreeze.load_groups(args.peers)
    peer_meta = json.loads((args.peers/"deployment.json").read_text())
    source_receipt = sourcefreeze.verify(args.source197_frozen)
    if peer_meta["source197_receipt"] != source_receipt:
        raise ValueError("Peer/source197 binding differs")
    groups = sourcefreeze.load_groups(args.source197_frozen)
    if {g["unit"] for g in groups} != {g["unit"] for g in peer_groups}:
        raise ValueError("Peer/source197 roster differs")
    eligible = {g["unit"]:{str(m["member"]):[r["candidate_id"] for r in m["records"] if r["eligible"]]
                           for m in g["members"]} for g in peer_groups}
    for file in ("graph_data.json","physical_batch.json"):
        if sha(stage/file) != manifest[file]: raise ValueError("Stage1 replay artifact changed")
    graphs = json.loads((stage/"graph_data.json").read_text())["models"]
    audit = json.loads((stage/"physical_batch.json").read_text())
    if audit["complete_members"] != 60: raise ValueError("O physical roster incomplete")
    by_group = {(g["domain"],g["group"],g["video"]):g for g in groups}
    oracle_metrics = {}
    for video in audit["videos"]:
        key = (video["domain"],video["group"],video["video"])
        unit = by_group[key]["unit"]
        oracle_metrics[unit] = {str(m["member"]):m["metrics"] for m in video["members"]}
    old_root = args.source197_score
    if (old_root/"SEALED.sha256").read_text().strip() != sha(old_root/"manifest.json"):
        raise ValueError("Original197 score seal differs")
    old_manifest = json.loads((old_root/"manifest.json").read_text())
    if sha(old_root/"native_scores.json") != old_manifest["native_scores.json"]:
        raise ValueError("Original197 numeric reference changed")
    old_scores = json.loads((old_root/"native_scores.json").read_text())
    references = {}
    for name,old_arm in (("K","K"),("T197","T")):
        for group in old_scores[old_arm]:
            for member in group["members"]:
                references.setdefault(group["unit"],{}).setdefault(str(member["member"]),{})[name] = member["metrics"]
    del old_scores
    # First actual GT entrance follows passed count guard and all4 prediction seals.
    truth,gt_receipts = previous._gt_provider(repo,groups,DOMAINS,"evaluation")
    summary = evaluate_explicit(groups,predictions["decisions"],graphs,oracle_metrics,
        lambda g:truth[g["video"]],args.out,graph198,physical198,references=references,eligible=eligible,
        resume_identity=dict(prereg=args.prereg,stage1_manifest_sha256=stage_manifest_sha,
            prediction_manifest_sha256=sha(args.predictions/"manifest.json"),peerfreeze_receipt=peer_receipt,
            source197_receipt=source_receipt,source197_score_sha256=sha(old_root/"manifest.json"),
            graph_code_sha256=sha(graph198.__file__),physical_code_sha256=sha(physical198.__file__),GT_entry=gt_receipts))
    for domain,data in summary["domains"].items():
        data["methods"]["O"]["cut_count"] = sum(len(ids) for row in capacity["units"].values() if row["hive"] == domain for ids in row["member_masks"].values())
        data["eligibility_coverage"] = capacity["hives"][domain]["coverage"]
        gain = data["methods"]["K"]["E3"]-data["methods"]["O"]["E3"]
        assert np.isclose(gain,capacity["hives"][domain]["mean_witness_gain"])
    support = {}
    for hive in DOMAINS:
        fitted = json.loads((args.predictions/f"{hive}.model.json").read_text())
        support[hive] = {k:fitted[k] for k in ("source_hives","source_domains","source_rows","eligible_source_rows")}
    summary.update(prereg=args.prereg,stage1_manifest_sha256=stage_manifest_sha,
        stage1_guard=guard,prediction_receipt=receipt,prediction_manifest_sha256=sha(args.predictions/"manifest.json"),
        source_support=support,peerfreeze_receipt=peer_receipt,
        GT_entry=gt_receipts,source197_receipt=source_receipt,
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")})
    (args.out/"summary.json").write_text(json.dumps(_plain(summary),indent=2,allow_nan=False))
    print("STAGE2_SCORES_SEALED "+seal(args.out),flush=True)
    return summary


def synthetic_integration(repo, out):
    """Normal graph/physical entry points, artificial RGB-free MOT only."""
    import sys
    sys.path[:0] = [str(Path(repo).resolve()),str(Path(__file__).resolve().parent.parent)]
    from endpoint_cut import candidates
    from tools.study197 import physical as old_physical
    import graph198
    import physical198
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    groups, decisions, graphs, oracle_metrics, truths, references, eligible = [], {}, {}, {}, {}, {}, {}
    poly = [[0,0],[10,0],[10,30],[0,30]]
    for clip in ("synthetic_a","synthetic_b"):
        raw = np.asarray([[f,7,x,5,2,2,.8] for f,x in enumerate((14,7,6,5,4),1)],float)
        gt = raw.copy(); gt[:,1] = 101
        bundle = candidates.build_candidates(raw,poly,75)
        ids = [c["candidate_id"] for c in bundle["candidates"]]
        groups.append(dict(unit=clip,domain="W",group=1,video=clip,polygon=poly,nf=75,
            members=[dict(member=str(i),member_index=i,raw=raw.copy(),bundle=bundle) for i in range(3)]))
        decisions[clip],graphs[clip],oracle_metrics[clip],references[clip],eligible[clip] = {},{},{},{},{}
        truths[clip] = gt
        for member in groups[-1]["members"]:
            identity = member["member"]
            k = old_physical.score_configuration(gt,raw,bundle,[],poly,75)["metrics"]
            a = old_physical.score_configuration(gt,raw,bundle,ids,poly,75)["metrics"]
            graph = graph198.build_member(bundle,k,a,75)
            graphs[clip][identity] = graph
            decisions[clip][identity] = {arm:([] if arm in ("K","N") else ids) for arm in ARMS if arm != "O"}
            eligible[clip][identity] = ids
            references[clip][identity] = dict(K=k,T197=a)
            oracle = physical198.score_member(gt,member,[],poly,75,out/"stage1_O"/f"{clip}_{identity}",
                                              expected_graph=graph198.evaluate(graph,[]))
            oracle_metrics[clip][identity] = oracle["metrics"]
    from types import SimpleNamespace
    calls = [0]
    def interrupted(*args,**kwargs):
        calls[0] += 1
        if calls[0] == 5: raise RuntimeError("injected synthetic interruption")
        return physical198.score_member(*args,**kwargs)
    interrupted_module = SimpleNamespace(__file__=physical198.__file__,score_member=interrupted)
    try:
        evaluate_explicit(groups,decisions,graphs,oracle_metrics,lambda g:truths[g["video"]],out/"stage2",
                          graph198,interrupted_module,references=references,field_layout=False,eligible=eligible)
    except RuntimeError as error:
        assert "injected synthetic" in str(error)
    else:
        raise AssertionError("Synthetic interruption missing")
    original_seals = {p:sha(p) for p in (out/"stage2"/"physical").rglob("SEALED.sha256")}
    assert len(original_seals) == 4
    result = evaluate_explicit(groups,decisions,graphs,oracle_metrics,lambda g:truths[g["video"]],out/"stage2",
                               graph198,physical198,references=references,field_layout=False,eligible=eligible)
    assert result["physical_checks"] == dict(I=6,controls=6,reused=4)
    assert all(sha(p) == digest for p,digest in original_seals.items())
    assert result["domains"]["W"]["methods"]["K"]["E3"] == 0
    assert result["domains"]["W"]["methods"]["I"]["E3"] == 100
    resumed = evaluate_explicit(groups,decisions,graphs,oracle_metrics,lambda g:truths[g["video"]],out/"stage2",
                               graph198,physical198,references=references,field_layout=False,eligible=eligible)
    assert resumed["physical_checks"] == dict(I=6,controls=6,reused=12)
    changed = json.loads(json.dumps(decisions))
    changed["synthetic_a"]["0"]["N"] = list(changed["synthetic_a"]["0"]["I"])
    try:
        evaluate_explicit(groups,changed,graphs,oracle_metrics,lambda g:truths[g["video"]],out/"stage2",
                          graph198,physical198,references=references,field_layout=False,eligible=eligible)
    except ValueError as error:
        assert "different inputs" in str(error)
    else:
        raise AssertionError("Changed-mask resume accepted")
    return dict(synthetic_integration_pass=True,physical_checks=result["physical_checks"],
                interrupted_after_sealed_members=4,resume_reused=12,changed_mask_resume_rejected=True,training_called=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--score",action="store_true")
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--stage1",type=Path)
    parser.add_argument("--peers",type=Path)
    parser.add_argument("--source197-frozen",type=Path)
    parser.add_argument("--source197-score",type=Path)
    parser.add_argument("--predictions",type=Path)
    parser.add_argument("--prereg")
    args = parser.parse_args()
    if args.score:
        if args.synthetic or args.stage1 is None or args.out is None:
            parser.error("--score requires --stage1 --out")
        field_score(args)
    elif args.synthetic:
        if args.repo is None or args.out is None:
            parser.error("--synthetic requires --repo --out")
        print(json.dumps(synthetic_integration(args.repo,args.out)))
    else:
        print(json.dumps(selfcheck()))
