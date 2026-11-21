"""Private §199 identity-cut scoring; startup/synthetic only until explicit GO.

Portable replay contract:
native_scores[arm] -> actual clip-group records with member metrics (integer
pred/GT bins, TP/pred/GT at .5/.3). Apply published198.combine: pool clip raw
numerators/GT and each physical member's TP/pred/GT before F1; groups equal.
classification_inputs -> candidate ID, domain, eligible, D/S/U, delta/direction,
each selected bit and continuous pD. These suffice to reproduce every aggregate,
classification rate/AUROC and exact count-line comparison without pixels or GT.
Graph/native proofs and immutable member paths identify the physical witnesses.
"""
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

ARMS = ("K","I198","O_D","O_DU","H_N","H_B","H_I","M")
MODEL_ARMS = ("K","I198","H_N","H_B","H_I","M")
PROBES = ("H_N","H_B","M")
DOMAINS = ("35","33","W","G")
GROUPS = {"35":4,"33":4,"W":3,"G":3}


def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    return value


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()


def object_sha(value):
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True,separators=(",",":"),allow_nan=False).iterencode(plain(value)):
        digest.update(chunk.encode())
    return digest.hexdigest()


def auroc(labels,scores):
    """Exact tied-rank Mann–Whitney area; missing a class -> undefined."""
    if len(labels) != len(scores): raise ValueError("Label/score sizes differ")
    if any(y not in (0,1) for y in labels) or not all(np.isfinite(s) for s in scores):
        raise ValueError("Finite binary AUROC inputs required")
    positives,negatives = sum(labels),len(labels)-sum(labels)
    if not positives or not negatives: return None
    bins = {}
    for y,score in zip(labels,scores):
        bins.setdefault(float(score),[0,0])[int(y)] += 1
    below,twice_wins = 0,0
    for score in sorted(bins):
        neg,pos = bins[score]
        twice_wins += 2*pos*below+pos*neg
        below += neg
    return float(Fraction(twice_wins,2*positives*negatives))


def classification(records,selected,scores=None):
    """One explicitly stated candidate scope. D positive; U never a known label.

    scores[cid] is {pD,abstain}; M uses binary selected bit as its score.
    AUROC is undefined if any known label has no nonabstaining score, rather
    than silently dropping observations or treating abstention as evidence.
    """
    selected = set(selected)
    ids = {r["candidate_id"] for r in records}
    if len(ids) != len(records) or not selected <= ids:
        raise ValueError("Candidate scope or mask mismatch")
    totals = Counter(r["class"] for r in records)
    if set(totals)-{"D","S","U"}: raise ValueError("Unknown identity class")
    cuts = Counter(r["class"] for r in records if r["candidate_id"] in selected)
    recall = cuts["D"]/totals["D"] if totals["D"] else None
    false_split = cuts["S"]/totals["S"] if totals["S"] else None
    known = [r for r in records if r["class"] in ("D","S")]
    values,truth,abstentions = [],[],0
    for r in known:
        cid = r["candidate_id"]
        item = dict(pD=float(cid in selected),abstain=False) if scores is None else scores.get(cid)
        if item is None or item.get("abstain") or item.get("pD") is None:
            abstentions += 1
            continue
        value = float(item["pD"])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("pD must be a finite probability")
        values.append(value);truth.append(int(r["class"] == "D"))
    directional = {}
    for delta,direction in ((d,q) for d in (-1,1) for q in (-1,1)):
        subset = [r for r in records if r["delta"] == delta and r["direction"] == direction]
        directional[f"{delta}/{direction}"] = dict(total=len(subset),cuts=sum(r["candidate_id"] in selected for r in subset),
            classes={c:sum(r["class"] == c for r in subset) for c in ("D","S","U")},
            cut_classes={c:sum(r["class"] == c and r["candidate_id"] in selected for r in subset) for c in ("D","S","U")})
    return dict(total=len(records),class_denominators={c:totals[c] for c in ("D","S","U")},
        cut_classes={c:cuts[c] for c in ("D","S","U")},cut_count=len(selected),
        D_recall=recall,S_false_split_rate=false_split,U_cut_count=cuts["U"],U_cut_rate=cuts["U"]/totals["U"] if totals["U"] else None,
        balanced_accuracy=(recall+1-false_split)/2 if recall is not None and false_split is not None else None,
        AUROC=auroc(truth,values) if not abstentions else None,
        AUROC_score_type="continuous_pD" if scores is not None else "binary_mask",
        known_score_count=len(values),known_abstentions=abstentions,by_delta_direction=directional)


def summarize(scored,combine,field_layout=True):
    """Reuse published198 actual-clip/member aggregation, never member-as-video."""
    if set(scored) != set(ARMS): raise ValueError("Eight complete arms required")
    roster = {(v["domain"],v["group"],v["video"]) for v in scored["K"]}
    if any(len(scored[a]) != len(roster) or {(v["domain"],v["group"],v["video"]) for v in scored[a]} != roster for a in ARMS):
        raise ValueError("Missing/duplicate actual clip groups")
    domains = {}
    for domain in sorted({r[0] for r in roster}):
        groups = sorted({r[1] for r in roster if r[0] == domain})
        if field_layout and len(groups) != GROUPS[domain]: raise ValueError("Changed 4/4/3/3 group scope")
        methods = {}
        for arm in ARMS:
            values = [combine([v for v in scored[arm] if v["domain"] == domain and v["group"] == g]) for g in groups]
            if any(v["denominator"] <= 0 for v in values): raise ValueError("Whole group GT empty")
            if field_layout and any(len(v["clips"]) != (2 if domain in ("W","G") else 1) for v in values):
                raise ValueError("Actual clip count changed")
            methods[arm] = dict(groups=values,
                E3=float(sum(Fraction(100*v["numerator"],v["denominator"]) for v in values)/len(values)),
                F1={t:float(sum((sum(Fraction(2*m["tp"],max(m["pred"]+m["gt"],1)) for m in v["iou"][t]["members"])/3 for v in values),Fraction())/len(values)) for t in ("0.5","0.3")})
        domains[domain] = dict(methods=methods,differences_vs_K={a:dict(E3=methods[a]["E3"]-methods["K"]["E3"],
            F1_05=methods[a]["F1"]["0.5"]-methods["K"]["F1"]["0.5"]) for a in ARMS})
    result = dict(schema="endpoint-identity-score-v1",domains=domains,strict_TP_ID_reported=False)
    if set(domains) == set(DOMAINS):
        delta = {h:domains[h]["differences_vs_K"]["H_I"] for h in DOMAINS}
        result["investment_checks"] = dict(head35_gain=delta["35"]["E3"] < -1.4,
            head33_gain=delta["33"]["E3"] < -1.3,W_no_harm=delta["W"]["E3"] <= 1.4,
            F1_no_harm=all(delta[h]["F1_05"] >= -.005 for h in ("35","33","W")))
        result["investment_pass"] = all(result["investment_checks"].values())
        result["identity_attribution_pass"] = all(domains[h]["methods"]["H_I"]["E3"] < domains[h]["methods"][a]["E3"] for h in ("35","33") for a in ("H_B","M"))
    return result


class NativeCache:
    """Immutable198 physical member checks, shared ONLY by unit/member/mask."""
    def __init__(self,out,physical_module):
        self.out = Path(out);self.physical = physical_module;self.memory = {};self.calls = []

    def check(self,group,member,mask,expected,gt_provider,arm):
        mask = sorted(mask)
        if len(mask) != len(set(mask)): raise ValueError("Duplicate cut ID")
        key = (group["unit"],str(member["member"]),tuple(mask))
        token = object_sha(key)
        within_job = key in self.memory
        if not within_job:
            gt = gt_provider(group)
            result = self.physical.score_member(gt,member,mask,group["polygon"],group["nf"],
                self.out/token,expected_graph=expected)
            self.memory[key] = result
        result = self.memory[key]
        self.physical.compare_graph(result["metrics"],expected)
        path = self.out/token
        self.calls.append(dict(unit=group["unit"],member=str(member["member"]),member_index=member["member_index"],
            arm=arm,selected_ids=mask,path=token,seal=(path/"SEALED.sha256").read_text().strip(),
            within_job_cache=within_job,reused_from_disk=bool(result.get("reused"))))
        return result["metrics"]


def prepare_output(out,identity):
    out = Path(out); identity = plain(identity)
    marker = out/"RUNNING.json"
    if out.exists():
        if (out/"SEALED.sha256").exists(): raise FileExistsError("Completed score already sealed")
        if not marker.exists() or json.loads(marker.read_text()) != identity:
            raise ValueError("Different inputs cannot resume this score")
    else:
        out.mkdir(parents=True)
        with marker.open("x") as stream: json.dump(identity,stream,sort_keys=True,allow_nan=False)


def matched_mask(records,selected):
    """Independent199 deterministic within-member/clip/delta/direction null."""
    eligible = [r for r in records if r["eligible"]]
    selected = set(selected)
    want = Counter((r["delta"],r["direction"]) for r in eligible if r["candidate_id"] in selected)
    result = []
    for (delta,direction),count in sorted(want.items()):
        pool = [r["candidate_id"] for r in eligible if (r["delta"],r["direction"]) == (delta,direction)]
        pool.sort(key=lambda cid:(hashlib.sha256(("endpoint-identity-null-v1|"+cid).encode()).hexdigest(),cid))
        result.extend(pool[:count])
    return sorted(result)


def evaluate_explicit(groups,decisions,probabilities,records,graphs,oracles,head_witnesses,gt_provider,out,
                      graph_module,physical_module,combine,references=None,field_layout=True,resume_identity=None):
    """Pure normalized orchestration; all model seals precede target records.

    decisions[unit][global member][MODEL_ARMS] selected global candidate IDs.
    probabilities[H_N/H_B/H_I][cid]={pD,abstain}; records flat normalized D/S/U.
    oracles[unit][O_D/O_DU]={member_masks:{local index:IDs},oracle:{...}}.
    head_witnesses[35/33]={feasible,unit_masks:{unit:{local index:IDs}},...}.
    """
    units = {g["unit"]:g for g in groups}
    by_id = {r["candidate_id"]:r for r in records}
    if len(units) != len(groups) or len(by_id) != len(records): raise ValueError("Duplicate unit/candidate")
    expected = {c["candidate_id"] for g in groups for m in g["members"] for c in m["bundle"]["candidates"]}
    if expected != set(by_id) or set(decisions) != set(units) or set(oracles) != set(units):
        raise ValueError("Complete frozen candidate/mask/oracle scopes required")
    all_masks = {}
    for group in groups:
        unit = group["unit"]; all_masks[unit] = {}
        for member in group["members"]:
            identity,slot = str(member["member"]),str(member["member_index"])
            candidates = {c["candidate_id"]:c for c in member["bundle"]["candidates"]}
            rr = [by_id[cid] for cid in candidates]
            masks = dict(decisions[unit][identity])
            if set(masks) != set(MODEL_ARMS) or masks["K"]: raise ValueError("Wrong model arms")
            for arm in ("O_D","O_DU"):
                masks[arm] = oracles[unit][arm]["member_masks"][slot]
            if any(len(mask) != len(set(mask)) or not set(mask) <= set(candidates) for mask in masks.values()):
                raise ValueError("Unknown/repeated cut")
            for arm in ("I198","H_N","H_B","H_I","M"):
                if any(not by_id[cid]["eligible"] for cid in masks[arm]): raise ValueError("Ineligible deployment cut")
            if any(by_id[cid]["class"] != "D" for cid in masks["O_D"]): raise ValueError("D-only oracle cut another class")
            if any(by_id[cid]["class"] == "S" for cid in masks["O_DU"]): raise ValueError("DU oracle split confirmed same track")
            if sorted(masks["M"]) != matched_mask(rr,masks["H_I"]): raise ValueError("Matched SHA control differs")
            for arm in ("H_N","H_B","H_I"):
                predicted = {r["candidate_id"] for r in rr if r["eligible"]
                    and not probabilities[arm][r["candidate_id"]]["abstain"]
                    and probabilities[arm][r["candidate_id"]]["pD"] > .5}
                if predicted != set(masks[arm]): raise ValueError("Hard mask differs from frozen pD/abstention")
            all_masks[unit][identity] = masks
    if resume_identity is None:
        raw_sha = hashlib.sha256()
        for g in groups:
            raw_sha.update(json.dumps([g[k] for k in ("unit","domain","group","video","polygon","nf")]).encode())
            for m in g["members"]:
                raw = np.asarray(m["raw"]);raw_sha.update(str(raw.shape).encode());raw_sha.update(raw.tobytes())
        resume_identity = dict(raw=raw_sha.hexdigest(),masks=object_sha(all_masks),oracles=object_sha(oracles),records=object_sha(records),
            probabilities=object_sha(probabilities),graphs=object_sha(graphs),heads=object_sha(head_witnesses),
            graph_code=sha(graph_module.__file__),physical_code=sha(physical_module.__file__))
    prepare_output(out,resume_identity)
    out = Path(out);cache = NativeCache(out/"physical",physical_module)
    scored = {arm:[] for arm in ARMS};probed = set();graph_cache = {}
    def metric(group,member,mask):
        key = (group["unit"],str(member["member"]),tuple(sorted(mask)))
        if key not in graph_cache:
            graph_cache[key] = graph_module.evaluate(graphs[key[0]][key[1]],mask)
        return graph_cache[key]
    for group in groups:
        unit,domain = group["unit"],group["domain"]
        video = {arm:dict(unit=unit,domain=domain,group=group["group"],video=group["video"],nf=group["nf"],members=[]) for arm in ARMS}
        for mi,member in enumerate(group["members"]):
            identity = str(member["member"]);masks = all_masks[unit][identity]
            for arm in ARMS:
                value = metric(group,member,masks[arm])
                if references is not None and arm in ("K","I198"):
                    physical_module.compare_graph(references[unit][identity][arm],value)
                if arm in ("O_D","O_DU","H_I") or arm in PROBES and domain not in probed and mi == 0:
                    cache.check(group,member,masks[arm],value,gt_provider,arm)
                video[arm]["members"].append(dict(member=identity,metrics=value,selected_ids=masks[arm]))
        probed.add(domain)
        for arm in ARMS: scored[arm].append(video[arm])
        for arm in ("O_D","O_DU"):
            actual = combine([video[arm]])
            oracle = oracles[unit][arm]["oracle"]
            assert actual["numerator"] == oracle["upper_bound"]
            assert actual["denominator"] == oracle["denominator"]
            np.testing.assert_array_equal(actual["clips"][0]["sum_pred_bins"],oracle["witness_pred_sum"])
    summary = summarize(scored,combine,field_layout)
    summary["restricted_capacity"] = {}
    for domain,data in summary["domains"].items():
        lines = {"35":Fraction(14,10),"33":Fraction(13,10)}
        comparisons = {}
        for arm in ("O_D","O_DU"):
            bounds = {}
            for group in groups:
                if group["domain"] != domain: continue
                row = oracles[group["unit"]][arm]["oracle"]
                b = bounds.setdefault(group["group"],dict(lower=0,upper=0,denominator=0))
                b["lower"] += row["lower_bound"];b["upper"] += row["upper_bound"];b["denominator"] += row["denominator"]
            lower = sum(Fraction(100*b["lower"],b["denominator"]) for b in bounds.values())/len(bounds)
            upper = sum(Fraction(100*b["upper"],b["denominator"]) for b in bounds.values())/len(bounds)
            baseline = sum(Fraction(100*g["numerator"],g["denominator"]) for g in data["methods"]["K"]["groups"])/len(bounds)
            assert np.isclose(float(upper),data["methods"][arm]["E3"])
            comparisons[arm] = dict(groups=bounds,mean_lower_E3=float(lower),mean_witness_E3=float(upper),
                guaranteed_gain=float(baseline-upper),maximum_possible_gain=float(baseline-lower),
                all_optimal=lower==upper,
                witness_crosses_line=baseline-upper > lines[domain] if domain in lines else None,
                excluded_by_lower_bound=baseline-lower <= lines[domain] if domain in lines else None)
        if domain in lines:
            comparisons["interpretation"] = ("confirmed_D_witness_crosses_count_line" if comparisons["O_D"]["witness_crosses_line"] else
                "a_confirmed_S_cut_is_required_within_this_candidate_family" if comparisons["O_DU"]["excluded_by_lower_bound"] else
                "unidentified_between_D_and_DU")
        summary["restricted_capacity"][domain] = comparisons
    summary["classification"] = {}
    summary["taxonomy_migration"] = {}
    summary["observed_truncation"] = {}
    replay = []
    for domain,data in summary["domains"].items():
        rr = [r for r in records if r["domain"] == domain]
        summary["taxonomy_migration"][domain] = {}
        summary["observed_truncation"][domain] = {}
        for scope,subset in (("all_candidates",rr),("deployment_eligible",[r for r in rr if r["eligible"]])):
            summary["taxonomy_migration"][domain][scope] = (dict(Counter(r["class"]+"->"+r["class03"] for r in subset))
                if all("class03" in r for r in subset) else None)
            table = {}
            for direction in (-1,1):
                same_deletions = [r for r in subset if r["class"] == "S" and r["delta"] == -1 and r["direction"] == direction]
                table[str(direction)] = dict(S_deletion_denominator=len(same_deletions),
                    supported_types=dict(Counter(r["censor_type"] for r in same_deletions if r.get("censor_supported"))),
                    unconfirmed=sum(not r.get("censor_supported",False) for r in same_deletions))
            summary["observed_truncation"][domain][scope] = table
        summary["classification"][domain] = {}
        for arm in ARMS:
            selected = [cid for g in groups if g["domain"] == domain for m in g["members"]
                        for cid in all_masks[g["unit"]][str(m["member"])][arm]]
            score = probabilities.get(arm)
            qualified = [r for r in rr if r["eligible"]]
            result = dict(all_candidates=classification(rr,selected,score),
                deployment_eligible=classification(qualified,[c for c in selected if by_id[c]["eligible"]],score))
            summary["classification"][domain][arm] = result
            data["methods"][arm]["cut_count"] = len(selected)
    for r in records:
        item = dict(r,selected={a:r["candidate_id"] in all_masks[r["unit"]][str(r["member"])][a] for a in ARMS},
                    probabilities={a:probabilities[a].get(r["candidate_id"]) for a in ("H_N","H_B","H_I")})
        replay.append(item)
    heads = {}
    for domain,witness in head_witnesses.items():
        if domain not in ("35","33"): raise ValueError("Min-S witness must be head-only")
        if not witness["feasible"]:
            heads[domain] = dict(witness=witness,physical_requested=0)
            continue
        videos = [];selected = []
        for group in groups:
            if group["domain"] != domain: continue
            item = dict(unit=group["unit"],domain=domain,group=group["group"],video=group["video"],nf=group["nf"],members=[])
            for member in group["members"]:
                mask = witness["unit_masks"][group["unit"]][str(member["member_index"])]
                if not set(mask) <= {c["candidate_id"] for c in member["bundle"]["candidates"]}: raise ValueError("Foreign min-S cut")
                value = metric(group,member,mask)
                cache.check(group,member,mask,value,gt_provider,"min_S_"+domain)
                item["members"].append(dict(member=str(member["member"]),metrics=value,selected_ids=mask));selected.extend(mask)
            videos.append(combine([item]))
        raw = sum(v["numerator"] for v in videos)
        cost = sum(by_id[cid]["class"] == "S" for cid in selected)
        assert cost == witness["S_upper"]
        if "budget" in witness: assert raw <= witness["budget"]
        if "count_numerator" in witness: assert raw == witness["count_numerator"]
        baseline = {g["group"]:g for g in summary["domains"][domain]["methods"]["K"]["groups"]}
        gain = sum(Fraction(100*(baseline[v["group"]]["numerator"]-v["numerator"]),v["denominator"]) for v in videos)/len(videos)
        assert gain > {"35":Fraction(14,10),"33":Fraction(13,10)}[domain]
        heads[domain] = dict(witness=witness,physical_requested=sum(len(g["members"]) for g in groups if g["domain"] == domain),
                            groups=videos,raw_numerator=raw,S_selected=cost,
                            mean_E3=float(np.mean([v["E3"] for v in videos])),count_gain=float(gain),
                            F1={t:float(np.mean([v["iou"][t]["mean_member_f1"] for v in videos])) for t in ("0.5","0.3")})
    summary["head_min_S"] = heads
    summary["physical_scope"] = dict(requested_arm_members=len(cache.calls),fresh_unique_physical=sum(not r.get("reused") for r in cache.memory.values()),
        historical_reused=0,resumed_unique_physical=sum(bool(r.get("reused")) for r in cache.memory.values()),
        same_job_shared=sum(c["within_job_cache"] for c in cache.calls),unique_streams=len(cache.memory))
    summary["interpretation"] = "Restricted oracle bounds address the count line; witness F1 is not an F1-constrained certificate. D/S describe only the two uniquely owned cut rows."
    oracle_replay = {u:{arm:dict(oracle=value["oracle"],member_masks=value["member_masks"]) for arm,value in arms.items()} for u,arms in oracles.items()}
    for name,value in (("native_scores.json",scored),("classification_inputs.json",replay),("oracle_inputs.json",oracle_replay),
                       ("physical_receipts.json",cache.calls),("summary.json",summary)):
        (out/name).write_text(json.dumps(plain(value),indent=2,allow_nan=False))
    return summary


def verify_manifest(root,names=()):
    root = Path(root)
    digest = sha(root/"manifest.json")
    if (root/"SEALED.sha256").read_text().strip() != digest: raise ValueError("Unsealed input")
    manifest = json.loads((root/"manifest.json").read_text())
    for name in names:
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != manifest[name]: raise ValueError("Input artifact changed: "+name)
    return manifest,digest


def seal(out):
    out = Path(out)
    manifest = {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
    digest = sha(out/"manifest.json");(out/"SEALED.sha256").write_text(digest+"\n")
    return digest


def field_score(args):
    """All four deployed folds are verified BEFORE target taxonomy is read."""
    import sys
    if (args.out/"SEALED.sha256").exists(): raise FileExistsError("Completed score already sealed")
    if not re.fullmatch(r"[0-9a-f]{40}",args.prereg): raise ValueError("Full prereg SHA required")
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo),str(repo/"tools/study198"),str(repo/"tools/study197")]
    subprocess.run(["git","merge-base","--is-ancestor",args.prereg,"origin/main"],cwd=repo,check=True,capture_output=True)
    log = subprocess.run(["git","show",args.prereg+":docs/experiment_log.md"],cwd=repo,check=True,capture_output=True).stdout.decode("utf-8")
    if log.count("[endpoint-identity-v1]") != 1: raise ValueError("Unique199 registration marker required")
    import model199
    import owners199
    from tools.study197 import freeze as sourcefreeze
    from tools.study198 import freeze as peerfreeze,graph198,physical198
    from tools.study198.run198 import combine
    from tools.study195 import run as previous
    predictions = model199.load_predictions(args.predictions)
    receipt = predictions["receipt"]
    owner_manifest,owner_sha = verify_manifest(args.owners)
    peer_receipt = peerfreeze.verify(args.peer198/"frozen")
    _,reference_mask_sha = verify_manifest(args.peer198/"predictions")
    if (receipt["prereg"] != args.prereg or receipt["owners_receipt"]["manifest_sha256"] != owner_sha
            or receipt["peerfreeze_receipt"] != peer_receipt
            or receipt["reference198_masks_sha256"] != reference_mask_sha):
        raise ValueError("Prediction/source registration binding differs")
    # First target label entrance follows verification of ALL four model/mask folds.
    owner_shards = {h:owners199.load_domain(args.owners,h) for h in DOMAINS}
    if any(s["prereg"] != args.prereg for s in owner_shards.values()): raise ValueError("Owner prereg differs")
    source_receipt = sourcefreeze.verify(args.source197_frozen)
    if any(s["source197_receipt"] != source_receipt for s in owner_shards.values()):
        raise ValueError("Owner/native source binding differs")
    groups = sourcefreeze.load_groups(args.source197_frozen)
    peer_groups = peerfreeze.load_groups(args.peer198/"frozen")
    peer = {r["candidate_id"]:r for g in peer_groups for m in g["members"] for r in m["records"]}
    records = []
    owner_units = {}
    for h,shard in owner_shards.items():
        for unit in shard["units"]:
            if unit["unit"] in owner_units: raise ValueError("Duplicate owner unit")
            owner_units[unit["unit"]] = unit
            for member in unit["members"]:
                for r in member["records"]:
                    cid = r["candidate_id"];p = peer[cid];censor = r["labels"]["0.5"]["censor"]
                    records.append(dict(candidate_id=cid,unit=unit["unit"],domain=h,group=unit["group"],video=unit["video"],
                        member=str(member["member"]),member_index=int(member["member_index"]),eligible=bool(p["eligible"]),
                        **{"class":r["labels"]["0.5"]["class"]},class03=r["labels"]["0.3"]["class"],
                        delta=r["delta"],direction=r["direction"],frame=r["frame"],
                        censor_supported=bool(censor["supported"]),censor_type=censor.get("type"),censor_reason=censor["reason"]))
    if len(records) != 1196 or sum(r["eligible"] for r in records) != 1076: raise ValueError("1196/1076 support changed")
    probabilities = {a:{} for a in ("H_N","H_B","H_I")}
    for fold in predictions["folds"].values():
        for item in fold["member_clips"]:
            for arm in probabilities:
                for cid,value in zip(item["candidate_ids"],item["scores"][arm]):
                    if cid in probabilities[arm]: raise ValueError("Duplicate predicted candidate")
                    probabilities[arm][cid] = value
    stage = args.peer198/"stage1"
    _,stage_sha = verify_manifest(stage,("graph_data.json",))
    graph_sha = sha(stage/"graph_data.json")
    graphs = json.loads((stage/"graph_data.json").read_text())["models"]
    _,capacity_sha = verify_manifest(args.capacity,("capacity199.json",))
    capacity = json.loads((args.capacity/"capacity199.json").read_text())
    cap_input = capacity["input_receipt"]
    if cap_input["prereg"] != args.prereg or cap_input["owners_seal"] != owner_sha or cap_input["graphs_sha"] != graph_sha:
        raise ValueError("Oracle/source binding differs")
    oracles = {u:row["families"] for u,row in capacity["units"].items()}
    heads = {h:dict(w,S_lower=w.get("protected_lower"),S_upper=w.get("protected_upper"),
                    unit_masks=w.get("unit_member_masks",{})) for h,w in capacity["minimum_S"].items()}
    old_root = args.peer198/"score"
    _,old_score_sha = verify_manifest(old_root,("native_scores.json","summary.json"))
    old_summary = json.loads((old_root/"summary.json").read_text())
    if old_summary["prediction_manifest_sha256"] != reference_mask_sha: raise ValueError("198 score/reference model differs")
    old = json.loads((old_root/"native_scores.json").read_text());references = {}
    for name,source in (("K","K"),("I198","I")):
        for group in old[source]:
            for m in group["members"]:
                references.setdefault(group["unit"],{}).setdefault(str(m["member"]),{})[name] = m["metrics"]
    del old
    # Native GT entrance is explicit, preserves official24 anomalous rows.
    truth,gt_receipts = previous._gt_provider(repo,groups,DOMAINS,"evaluation")
    expected_GT = owner_shards[DOMAINS[0]]["GT_sha256"]
    for proof in gt_receipts.values():
        if expected_GT[proof["path"]] != proof["sha256"]: raise ValueError("Taxonomy/native GT differs")
    identity = dict(prereg=args.prereg,source197_receipt=source_receipt,owners_seal=owner_sha,
        peer_receipt=peer_receipt,model_seal=sha(args.predictions/"manifest.json"),capacity_seal=capacity_sha,
        graph_data_sha256=graph_sha,reference198_score_sha=old_score_sha,GT_entry=gt_receipts,
        graph_code_sha=sha(graph198.__file__),physical_code_sha=sha(physical198.__file__))
    result = evaluate_explicit(groups,predictions["decisions"],probabilities,records,graphs,oracles,heads,
        lambda g:truth[g["video"]],args.out,graph198,physical198,combine,references,resume_identity=identity)
    support = {}
    for h in DOMAINS:
        fitted = json.loads((args.predictions/f"{h}.model.json").read_text())
        support[h] = {k:fitted[k] for k in ("source_hives","source_domains","source_rows","known_source_rows") if k in fitted}
    result.update(prereg=args.prereg,input_receipt=identity,stage198_seal=stage_sha,source_support=support,
                  code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")})
    (args.out/"summary.json").write_text(json.dumps(plain(result),indent=2,allow_nan=False))
    print("SCORE199_SEALED "+seal(args.out),flush=True)
    return result


def selfcheck():
    assert auroc([1,0],[.5,.5]) == .5
    assert auroc([1,0],[1.,0.]) == 1
    assert auroc([1,1],[1.,0.]) is None
    records = [dict(candidate_id=c,domain="35",eligible=True,**{"class":label},delta=-1,direction=1)
               for c,label in (("d","D"),("s","S"),("u","U"))]
    perfect = classification(records,["d"],{c:dict(pD=p,abstain=False) for c,p in (("d",.9),("s",.1),("u",.5))})
    assert perfect["balanced_accuracy"] == 1 and perfect["AUROC"] == 1
    abstain = classification(records,[],{c:dict(pD=.5,abstain=True) for c in ("d","s","u")})
    assert abstain["balanced_accuracy"] == .5 and abstain["AUROC"] is None
    assert classification(records,["s"])["AUROC"] == 0
    return dict(synthetic_classification_pass=True,no_field_read=True)


def synthetic_integration(repo,out):
    """Artificial two-track scene: D-cut and S-cut tie in counts but differ in TP."""
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,str(Path(repo).resolve()))
    # Archived197 modules retain sibling imports.
    sys.path.insert(0,str(Path(repo).resolve()/"tools/study197"))
    from tools.study197 import candidates,physical as physical197,oracle as oracle197
    from tools.study198 import graph198,physical198
    from tools.study198.run198 import combine
    out = Path(out);out.mkdir(parents=True,exist_ok=False)
    groups,decisions,records,graphs,oracles,references,truths = [],{},{},{},{},{},{}
    records = [];probabilities = {a:{} for a in ("H_N","H_B","H_I")}
    poly = [[0,0],[10,0],[10,30],[0,30]]
    for clip in ("toy_a","toy_b"):
        raw = np.asarray([[f,tid,x,y,2,2,.8] for tid,y in ((7,5),(11,18))
                          for f,x in enumerate((14,7,6,5,4),1)],float)
        gt = raw.copy();gt[:5,1] = 101;gt[5,1] = 201;gt[6:,1] = 202
        group = dict(unit=clip,domain="W",group=1,video=clip,polygon=poly,nf=75,members=[])
        groups.append(group);truths[clip] = gt
        decisions[clip],graphs[clip],references[clip] = {},{},{}
        baselines,allowed = [],[]
        for mi in range(3):
            identity = str(mi)
            bundle = candidates.build_candidates(raw,poly,75)
            for c in bundle["candidates"]:
                c["candidate_id"] = f"{clip}|{mi}|{c['parent_id']}"
                label = "S" if c["parent_id"] == 7 else "D"
                record = dict(candidate_id=c["candidate_id"],unit=clip,member=identity,member_index=mi,
                              domain="W",eligible=True,**{"class":label},delta=c["delta"],direction=c["direction"])
                records.append(record)
                for arm in probabilities:
                    p = .5 if arm == "H_N" else float((label == "D") == (arm == "H_I"))
                    probabilities[arm][c["candidate_id"]] = dict(pD=p,abstain=False)
            member = dict(member=identity,member_index=mi,raw=raw.copy(),bundle=bundle)
            group["members"].append(member)
            k = physical197.score_configuration(gt,raw,bundle,[],poly,75)["metrics"]
            a = physical197.score_configuration(gt,raw,bundle,[c["candidate_id"] for c in bundle["candidates"]],poly,75)["metrics"]
            graph = graph198.build_member(bundle,k,a,75)
            graphs[clip][identity] = graph;references[clip][identity] = dict(K=k,I198=k)
            baseline = graph198.evaluate(graph,[]);baselines.append(baseline)
            own = [r for r in records if r["unit"] == clip and r["member"] == identity]
            good = [r["candidate_id"] for r in own if r["class"] == "D"]
            bad = [r["candidate_id"] for r in own if r["class"] == "S"]
            decisions[clip][identity] = dict(K=[],I198=[],H_N=[],H_B=bad,H_I=good,M=matched_mask(own,good))
            allowed.extend(c for c in bundle["candidates"] if c["parent_id"] == 11)
        optimum = oracle197.solve_clip(sum(np.asarray(b["pred_bins"]) for b in baselines),baselines[0]["gt_bins"],allowed,75)
        masks = {str(mi):[cid for cid in optimum["selected_ids"] if cid.startswith(f"{clip}|{mi}|")] for mi in range(3)}
        oracles[clip] = {arm:dict(oracle=optimum,member_masks=masks) for arm in ("O_D","O_DU")}
    calls = [0]
    def interrupted(*args,**kwargs):
        calls[0] += 1
        if calls[0] == 5: raise RuntimeError("injected interruption")
        return physical198.score_member(*args,**kwargs)
    wrapper = SimpleNamespace(__file__=physical198.__file__,score_member=interrupted,compare_graph=physical198.compare_graph)
    try:
        evaluate_explicit(groups,decisions,probabilities,records,graphs,oracles,{},lambda g:truths[g["video"]],out/"score",
                          graph198,wrapper,combine,references,field_layout=False)
    except RuntimeError as error:
        assert "injected" in str(error)
    else: raise AssertionError("Missing synthetic interruption")
    sealed = {p:sha(p) for p in (out/"score/physical").glob("*/SEALED.sha256")}
    assert len(sealed) == 4
    result = evaluate_explicit(groups,decisions,probabilities,records,graphs,oracles,{},lambda g:truths[g["video"]],out/"score",
                              graph198,physical198,combine,references,field_layout=False)
    assert all(sha(p) == value for p,value in sealed.items())
    domain = result["domains"]["W"]["methods"]
    assert domain["H_I"]["E3"] == domain["H_B"]["E3"] == 0
    assert domain["H_I"]["F1"]["0.5"] == 1 and domain["H_B"]["F1"]["0.5"] == 0
    assert result["classification"]["W"]["H_I"]["deployment_eligible"]["balanced_accuracy"] == 1
    scope = result["physical_scope"]
    assert scope["requested_arm_members"] == 21 and scope["unique_streams"] == 8
    assert scope["resumed_unique_physical"] == 4 and scope["fresh_unique_physical"] == 4
    return dict(synthetic_integration_pass=True,physical_scope=scope,
                identity_cut_E3=0,same_track_split_E3=0,identity_F1=1,same_track_split_F1=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--score",action="store_true")
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--source197-frozen",type=Path)
    parser.add_argument("--peer198",type=Path)
    parser.add_argument("--owners",type=Path)
    parser.add_argument("--capacity",type=Path)
    parser.add_argument("--predictions",type=Path)
    parser.add_argument("--prereg")
    args = parser.parse_args()
    if args.score:
        if args.synthetic or any(getattr(args,k) is None for k in ("repo","out","source197_frozen","peer198","owners","capacity","predictions","prereg")):
            parser.error("--score requires registered repo/input/output paths")
        field_score(args)
    elif args.synthetic:
        if args.repo is None or args.out is None: parser.error("--synthetic requires --repo --out")
        print(json.dumps(synthetic_integration(args.repo,args.out)))
    else:
        print(json.dumps(selfcheck()))
