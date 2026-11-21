"""Full-scene native orphan-continuation scoring. No field IO on import.

New boxes can change Hungarian owners. Every unique cut+assignment state gets
fresh official scoring; no198 event graph supplies its F1. Native member dirs
are immutable and reusable only for identical unit/member/full action masks,
source rows/provenance, GT, polygon, clock and scorer/materializer versions.
Catalogs are loaded one member at a time. Main ledgers retain compact metrics
and pointers; each unique state's own score.json contains dated event graphs.
"""
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import numpy as np

MODEL_ARMS = ("K","G","L","M","J")
ORACLE_ARMS = ("E","DE","PE","ALL")
DOMAINS = ("35","33","W","G")
GROUPS = {"35":4,"33":4,"W":3,"G":3}


def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain(v) for v in value]
    return value


def dump(path,value):
    with Path(path).open("w",encoding="utf-8") as stream:
        json.dump(plain(value),stream,sort_keys=True,separators=(",",":"),allow_nan=False)
        stream.write("\n")


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()


def object_sha(value):
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True,separators=(",",":"),allow_nan=False).iterencode(plain(value)):
        digest.update(chunk.encode())
    return digest.hexdigest()


def array_sha(rows):
    rows = np.asarray(rows,dtype=np.float64)
    return object_sha([str(rows.dtype),list(rows.shape)])+":"+hashlib.sha256(rows.tobytes()).hexdigest()


def seal(root):
    root = Path(root)
    manifest = {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    dump(root/"manifest.json",manifest)
    value = sha(root/"manifest.json");(root/"SEALED.sha256").write_text(value+"\n")
    return value


def verify(root):
    root = Path(root)
    if (root/"SEALED.sha256").read_text().strip() != sha(root/"manifest.json"):
        raise ValueError("Unsealed physical state")
    manifest = json.loads((root/"manifest.json").read_text())
    for name,value in manifest.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != value:
            raise ValueError("Physical state artifact changed")
    return json.loads((root/"receipt.json").read_text())


def compact_metric(metrics):
    return dict(nG=int(metrics["nG"]),nP=int(metrics["nP"]),pred_bins=metrics["pred_bins"],gt_bins=metrics["gt_bins"],
        iou={t:{k:metrics["iou"][t][k] for k in ("tp","pred","gt","f1")} for t in ("0.5","0.3")})


def assert_K(actual,expected):
    for key in ("nG","nP","pred_bins","gt_bins"):
        np.testing.assert_array_equal(actual[key],expected[key],err_msg="Historical K "+key)
    for t in ("0.5","0.3"):
        for key in ("tp","pred","gt"):
            if actual["iou"][t][key] != expected["iou"][t][key]:
                raise ValueError("Historical K official metric changed")


class NativeCache:
    def __init__(self,out,materialize,score_native,write_mot,code_identity):
        self.out = Path(out);self.materialize = materialize;self.score_native = score_native
        self.write_mot = write_mot;self.code_identity = code_identity
        self.current_member = None;self.memory = {};self.seen = set();self.calls = []
        self.fresh = 0;self.resumed = 0

    def score(self,group,member,state,gt_provider,arm):
        cuts,edges = sorted(state["cut_ids"]),sorted(state["edge_ids"])
        if len(cuts) != len(set(cuts)) or len(edges) != len(set(edges)): raise ValueError("Duplicate actions")
        context = (group["unit"],str(member["member"]))
        if self.current_member != context:
            self.memory.clear();self.current_member = context
        key = (*context,tuple(cuts),tuple(edges));token = object_sha(key)
        shared = token in self.seen
        if token not in self.memory:
            rows,proof = self.materialize(group,member,cuts,edges)
            rows = np.asarray(rows,dtype=np.float64)
            raw = np.asarray(member.get("raw",member.get("rows")),dtype=np.float64)
            catalog = member["catalog"]
            if rows.ndim != 2 or rows.shape[1] != 7 or not np.isfinite(rows).all(): raise ValueError("Invalid physical rows")
            if len(rows) < len(raw) or rows[:len(raw),[0,2,3,4,5,6]].tobytes() != raw[:,[0,2,3,4,5,6]].tobytes():
                raise ValueError("Existing native measurements/date/order changed")
            if (not np.array_equal(rows[:,:2],np.rint(rows[:,:2])) or np.any(rows[:,0]<1)
                    or np.any(rows[:,0]>group["nf"]) or np.any(rows[:,1]<1) or np.any(rows[:,4:6]<=0)
                    or len(np.unique(rows[:,:2],axis=0)) != len(rows)):
                raise ValueError("Invalid clock/geometry/duplicate track-date")
            source = [tuple(s) for s in proof["sources"]]
            expected_source = [tuple(s) for s in catalog["baseline_sources"]]
            by_edge = {e["edge_id"]:e for e in catalog["edges"]}
            expected_rows = []
            for eid in edges:
                e = by_edge[eid];orphan = catalog["orphans"][int(e["orphan_index"])]
                expected_source.extend(map(tuple,orphan["sources"]))
                add = np.asarray(orphan["rows"],dtype=np.float64).copy();add[:,1] = e["recipient"]
                expected_rows.append(add)
            added = np.concatenate(expected_rows) if expected_rows else np.empty((0,7))
            if source != expected_source or len(source) != len(rows) or len(set(source)) != len(source):
                raise ValueError("Nonunique, missing or invented row provenance")
            if rows[len(raw):].tobytes() != added.tobytes(): raise ValueError("Orphan measurements/assignment changed")
            if not cuts:
                np.testing.assert_array_equal(rows[:len(raw)],np.asarray(catalog["baseline"]))
            gt = gt_provider(group)
            identity = dict(unit=context[0],member=context[1],cut_ids=cuts,edge_ids=edges,
                raw_sha256=array_sha(raw),rows_sha256=array_sha(rows),GT_sha256=array_sha(gt),
                polygon_sha256=array_sha(group["polygon"]),nf=int(group["nf"]),
                source_map_sha256=object_sha(source),code=self.code_identity)
            path = self.out/token
            if path.exists():
                receipt = verify(path)
                if receipt["identity"] != identity: raise ValueError("Cached native state belongs to different inputs")
                reread = np.loadtxt(path/"mot.txt",delimiter=",",usecols=range(7),ndmin=2) if len(rows) else np.empty((0,7))
                if reread.tobytes() != rows.tobytes(): raise ValueError("Cached physical MOT differs from current realization")
                compact = json.loads((path/"compact.json").read_text());reused = True
                self.resumed += int(not shared)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix="."+token+"-",dir=path.parent))
                try:
                    mot_receipt = self.write_mot(rows,temporary/"mot.txt")
                    reread = np.loadtxt(temporary/"mot.txt",delimiter=",",usecols=range(7),ndmin=2) if len(rows) else np.empty((0,7))
                    if reread.tobytes() != rows.tobytes(): raise ValueError("MOT roundtrip changed original floats")
                    metrics = self.score_native(gt,reread,group["polygon"],group["nf"])
                    compact = compact_metric(metrics)
                    dump(temporary/"score.json",metrics);dump(temporary/"compact.json",compact)
                    with gzip.open(temporary/"row_sources.jsonl.gz","wt",encoding="utf-8") as stream:
                        for item in source: stream.write(json.dumps(plain(item),separators=(",",":"))+"\n")
                    receipt = dict(identity=identity,mot=mot_receipt,rows=len(rows),original_rows=len(raw),
                        added_rows=len(rows)-len(raw),provenance_complete=True,source_rows_unique=True,
                        materialization_proof_sha256=object_sha(proof),new_fullscene_official_matching=True,
                        no198_graph_F1=True)
                    dump(temporary/"receipt.json",receipt);seal(temporary);temporary.rename(path)
                    reused = False;self.fresh += 1
                finally:
                    if temporary.exists(): shutil.rmtree(temporary)
            self.memory[token] = dict(metrics=compact,path=token,reused=reused,added_rows=len(rows)-len(raw))
            del rows,raw,source,expected_source,expected_rows,added,gt
        result = self.memory[token];self.seen.add(token)
        self.calls.append(dict(unit=context[0],member=context[1],member_index=int(member["member_index"]),arm=arm,
            cut_ids=cuts,edge_ids=edges,path=result["path"],within_job_shared=shared,
            resumed_from_disk=result["reused"],seal=(self.out/token/"SEALED.sha256").read_text().strip()))
        return result["metrics"]


def summarize(scored,combine,field_layout=True):
    """Every state's metrics came from actual complete MOT matching, not old edges."""
    if set(scored) != set(MODEL_ARMS+ORACLE_ARMS): raise ValueError("Five models and four oracle families required")
    report = {}
    for domain in sorted({v["domain"] for v in scored["K"]}):
        group_ids = sorted({v["group"] for v in scored["K"] if v["domain"] == domain})
        if field_layout and len(group_ids) != GROUPS[domain]: raise ValueError("4/4/3/3 group scope changed")
        methods = {}
        arms = MODEL_ARMS+ORACLE_ARMS if domain in ("35","33") else MODEL_ARMS
        for arm in arms:
            vv = [v for v in scored[arm] if v["domain"] == domain]
            if len({(v["group"],v["video"]) for v in vv}) != len(vv): raise ValueError("Duplicate actual clip")
            grouped = [combine([v for v in vv if v["group"] == group]) for group in group_ids]
            if field_layout and any(len(g["clips"]) != (2 if domain in ("W","G") else 1) for g in grouped):
                raise ValueError("Actual clip count changed")
            if any(g["denominator"] <= 0 for g in grouped): raise ValueError("Whole group GT empty")
            methods[arm] = dict(groups=grouped,
                E3=float(sum(Fraction(100*g["numerator"],g["denominator"]) for g in grouped)/len(grouped)),
                F1={t:float(sum((sum(Fraction(2*m["tp"],max(m["pred"]+m["gt"],1)) for m in g["iou"][t]["members"])/3
                    for g in grouped),Fraction())/len(grouped)) for t in ("0.5","0.3")})
        report[domain] = dict(methods=methods,differences_vs_K={a:dict(E3=v["E3"]-methods["K"]["E3"],
            F1_05=v["F1"]["0.5"]-methods["K"]["F1"]["0.5"]) for a,v in methods.items()})
    result = dict(schema="orphan-continuation-native-score-v1",domains=report,
                  F1_contract="Fresh full-scene matching for every unique physical state; no198 event-graph interpolation")
    if set(report) == set(DOMAINS):
        def gate(arm):
            d = {h:report[h]["differences_vs_K"][arm] for h in DOMAINS}
            return dict(head35_gain=d["35"]["E3"] < -1.4,head33_gain=d["33"]["E3"] < -1.3,
                W_no_harm=d["W"]["E3"] <= 1.4,F1_no_harm=all(d[h]["F1_05"] >= -.005 for h in ("35","33","W")))
        result["investment_checks"] = {a:gate(a) for a in ("L","J")}
        result["investment_pass"] = {a:all(v.values()) for a,v in result["investment_checks"].items()}
        result["L_identity_ranking_pass"] = all(report[h]["methods"]["L"]["E3"] < report[h]["methods"][other]["E3"]
                                                 for h in ("35","33") for other in ("G","M"))
        result["J_strictly_beats_L_and_G_both_heads"] = all(report[h]["methods"]["J"]["E3"] < report[h]["methods"][other]["E3"]
                                                              for h in ("35","33") for other in ("L","G"))
    return result


def prepare_output(out,identity):
    out = Path(out);identity = plain(identity)
    if out.exists():
        if (out/"SEALED.sha256").exists(): raise FileExistsError("Complete score already sealed")
        if not (out/"RUNNING.json").exists() or json.loads((out/"RUNNING.json").read_text()) != identity:
            raise ValueError("Cannot resume different input/action/scorer round")
    else:
        out.mkdir(parents=True);dump(out/"RUNNING.json",identity)


def normalized_state(state):
    if set(state) == {"cuts","edges"}: state = dict(cut_ids=state["cuts"],edge_ids=state["edges"])
    if set(state) != {"cut_ids","edge_ids"}: raise ValueError("Complete cut+edge mask required")
    result = {k:sorted(state[k]) for k in ("cut_ids","edge_ids")}
    if any(len(ids) != len(set(ids)) for ids in result.values()): raise ValueError("Duplicate action")
    return result


def action_table(member):
    actions = {}
    for c in member["bundle"]["candidates"]:
        actions["cut:"+c["candidate_id"]] = dict(type="cut",id=c["candidate_id"],parent=c["parent_id"],
            delta=np.asarray(c["delta_windows"],np.int64),effect="added" if c["delta"] == 1 else "removed")
    for e in member["catalog"]["edges"]:
        actions[e["edge_id"]] = dict(type="extension",id=e["edge_id"],parent=e["parent_canonical_id"],
            orphan=e["orphan_index"],delta=np.asarray(e["delta_windows"],np.int64),effect=e["event_effect"])
    return actions


def selected_ids(state): return ["cut:"+cid for cid in state["cut_ids"]]+state["edge_ids"]


def validate_M(audit,member):
    """All recipient scores move within the same orphan, not a selected-orphan constraint."""
    if audit is None: return
    edges = {e["edge_id"]:e for e in member["catalog"]["edges"]}
    L,M = audit["scores"]["L"],audit["scores"]["M"]
    if set(L) != set(edges) or set(M) != set(edges): raise ValueError("M/L score inventory changed")
    buckets = {}
    for eid,e in edges.items(): buckets.setdefault(e["orphan_index"],[]).append(eid)
    expected = {}
    for ids in buckets.values():
        ids.sort()
        targets = sorted(ids,key=lambda x:(hashlib.sha256(("orphan-recipient-null-v1|"+x).encode()).hexdigest(),x))
        for src,dst in zip(ids,targets): expected[dst] = L[src]
    if M != expected: raise ValueError("Within-orphan L score permutation changed")


def evaluate_stream(groups,load_member,decisions,oracles,owners,gt_provider,out,materialize,score_native,write_mot,
                    combine,code_identity,resume_identity,reference_provider,model_audit=None,field_layout=True):
    """Only one raw/catalog member is resident; retained results are compact.

    groups is a GT-free descriptor roster, load_member(group,descriptor) loads
    that one member. owners[unit][global member] maps actionID->S/D/U record.
    decisions[unit][global member][K/G/L/M/J] has complete cut_ids/edge_ids.
    oracles[unit][E/DE/PE/ALL] is each exact count-family solver record.
    """
    if set(decisions) != {g["unit"] for g in groups}: raise ValueError("All model clip groups must be sealed")
    expected_heads = {g["unit"] for g in groups if g["domain"] in ("35","33")}
    if set(oracles) != expected_heads: raise ValueError("Exactly eight head oracle units required")
    prepare_output(out,resume_identity)
    out = Path(out);(out/"members").mkdir(exist_ok=True)
    cache = NativeCache(out/"physical",materialize,score_native,write_mot,code_identity)
    scored = {a:[] for a in MODEL_ARMS+ORACLE_ARMS};stats = {}
    for group in groups:
        arms = MODEL_ARMS+ORACLE_ARMS if group["domain"] in ("35","33") else MODEL_ARMS
        current = {arm:dict(unit=group["unit"],domain=group["domain"],group=group["group"],video=group["video"],nf=group["nf"],members=[]) for arm in arms}
        for descriptor in group["members"]:
            member = load_member(group,descriptor)
            identity = str(member["member"]);slot = str(member["member_index"])
            actions = action_table(member);labels = owners[group["unit"]][identity]
            if set(actions) != set(labels): raise ValueError("Frozen action/owner inventory mismatch")
            if model_audit is not None: validate_M(model_audit[group["unit"]][identity],member)
            states = {a:normalized_state(decisions[group["unit"]][identity][a]) for a in MODEL_ARMS}
            if states["K"]["cut_ids"] or states["K"]["edge_ids"]: raise ValueError("K must keep original")
            if any(states[a]["cut_ids"] for a in ("G","L","M")): raise ValueError("Extension-only arm made a cut")
            for arm in ORACLE_ARMS if group["domain"] in ("35","33") else ():
                states[arm] = normalized_state(oracles[group["unit"]][arm]["member_masks"][slot])
            baseline = None;member_rows = {}
            for arm in arms:
                state = states[arm];ids = selected_ids(state)
                if not set(ids) <= set(actions): raise ValueError("Foreign action ID")
                for aid in ids:
                    label = labels[aid]["labels"]["0.5"]["class"];kind = actions[aid]["type"]
                    if label not in ("S","D","U"): raise ValueError("Unknown owner label")
                    if arm in ("E","DE","PE") and kind == "extension" and label != "S": raise ValueError("E extension is not confirmed S")
                    if arm == "E" and kind == "cut": raise ValueError("E made a cut")
                    if arm == "DE" and kind == "cut" and label != "D": raise ValueError("DE cut outside D")
                    if arm == "PE" and kind == "cut" and label == "S": raise ValueError("PE split S")
                metric = cache.score(group,member,state,gt_provider,arm)
                if baseline is None:
                    baseline = metric
                    assert_K(metric,reference_provider(group,descriptor))
                expected = np.asarray(baseline["pred_bins"],np.int64).copy()
                for aid in ids: expected += actions[aid]["delta"]
                np.testing.assert_array_equal(expected,np.asarray(metric["pred_bins"],np.int64),err_msg="Whole-phase single-action delta closure")
                role = dict(member=identity,member_index=int(slot),metrics=metric,state=state,
                    physical=cache.calls[-1]["path"],seal=cache.calls[-1]["seal"],delta_closure_exact=True)
                current[arm]["members"].append(role);member_rows[arm] = role
                key = (group["domain"],arm)
                stat = stats.setdefault(key,dict(cuts=Counter(),extensions=Counter(),effects=Counter(),extension_denominators=Counter(),cut_denominators=Counter()))
                for aid,action in actions.items():
                    cls = labels[aid]["labels"]["0.5"]["class"]
                    stat["extension_denominators" if action["type"] == "extension" else "cut_denominators"][cls] += 1
                for aid in ids:
                    action = actions[aid];cls = labels[aid]["labels"]["0.5"]["class"]
                    stat["extensions" if action["type"] == "extension" else "cuts"][cls] += 1
                    stat["effects"][action["effect"]] += 1
            dump(out/"members"/(object_sha([group["unit"],identity])+".json"),dict(unit=group["unit"],member=identity,arms=member_rows))
            cache.memory.clear()
            del member,actions,member_rows,labels
        for arm in arms: scored[arm].append(current[arm])
        for arm in ORACLE_ARMS if group["domain"] in ("35","33") else ():
            actual = combine([current[arm]]);cap = oracles[group["unit"]][arm]
            assert actual["numerator"] == cap["upper_bound"] and actual["denominator"] == cap["denominator"]
            np.testing.assert_array_equal(actual["clips"][0]["sum_pred_bins"],cap["witness_pred_sum"])
        print("SCORED "+group["unit"],flush=True)
    summary = summarize(scored,combine,field_layout)
    for (domain,arm),stat in stats.items():
        item = {k:dict(v) for k,v in stat.items()}
        total = sum(stat["extensions"].values())
        item["selected_extension_proportions"] = {c:stat["extensions"][c]/total if total else None for c in ("S","D","U")}
        summary["domains"][domain]["methods"][arm]["action_stats"] = item
    for domain in ("35","33"):
        if domain not in summary["domains"]: continue
        bounds = {}
        for arm in ORACLE_ARMS:
            rr = [oracles[g["unit"]][arm] for g in groups if g["domain"] == domain]
            lower = sum(Fraction(100*r["lower_bound"],r["denominator"]) for r in rr)/len(rr)
            upper = sum(Fraction(100*r["upper_bound"],r["denominator"]) for r in rr)/len(rr)
            baseline = sum(Fraction(100*r["baseline_numerator"],r["denominator"]) for r in rr)/len(rr)
            line = Fraction(14 if domain == "35" else 13,10)
            bounds[arm] = dict(mean_lower_E3=float(lower),mean_witness_E3=float(upper),
                guaranteed_gain=float(baseline-upper),maximum_possible_gain=float(baseline-lower),
                all_optimal=all(r["lower_bound"] == r["upper_bound"] for r in rr),
                crosses_count_line=baseline-upper > line,excluded_by_lower_bound=baseline-lower <= line)
        summary["domains"][domain]["capacity_bounds"] = bounds
    summary["physical_scope"] = dict(requested_arm_members=len(cache.calls),unique_streams=len(cache.seen),
        fresh_unique_physical=cache.fresh,resumed_unique_physical=cache.resumed,
        same_job_shared=sum(c["within_job_shared"] for c in cache.calls),historical_reused=0)
    dump(out/"native_scores.json",scored);dump(out/"physical_receipts.json",cache.calls);dump(out/"summary.json",summary)
    return summary


def headers_from_freeze(data,field_layout=True):
    groups = {}
    for ref in data["members"]:
        unit = ref["unit"]
        group = groups.setdefault(unit,{k:ref[k] for k in ("unit","domain","group","video","split","nf","polygon")})
        if any(group[k] != ref[k] for k in ("domain","group","video","split","nf","polygon")):
            raise ValueError("Member clip metadata differs")
        group.setdefault("members",[]).append(ref)
    for group in groups.values():
        group["members"].sort(key=lambda m:m["member_index"])
        if [m["member_index"] for m in group["members"]] != [0,1,2] or len({str(m["member"]) for m in group["members"]}) != 3:
            raise ValueError("Exactly three original members required")
    if field_layout:
        if len(groups) != 20 or len(data["members"]) != 60 or len({g["video"] for g in groups.values()}) != 6:
            raise ValueError("Exact60 members/20 units/6 clips required")
        if any(m["seed"] > 14 or m["split"] == "test" for m in data["members"]): raise ValueError("Forbidden source")
    return list(groups.values())


def member_loader(root,canonicalize,verify_child):
    """A closure over paths only; never captures all decoded catalogs."""
    root = Path(root)
    def load(group,ref):
        directory = (root/ref["path"]).resolve()
        if not directory.is_relative_to(root.resolve()): raise ValueError("Catalog escapes frozen root")
        verify_child(directory)
        if sha(directory/"manifest.json") != ref["seal"]: raise ValueError("Member freeze seal differs")
        with np.load(directory/"rows.npz",allow_pickle=False) as bank: raw = bank["original_K"]
        catalog = json.loads((directory/"catalog.json").read_text())
        bundle = canonicalize(raw)
        reference = catalog["cut_catalog"]
        if not isinstance(reference,dict) or not isinstance(reference["candidates"],list): raise ValueError("Complete197 cut bundle required")
        parent_keys = ("rank","parent_id","start","end")
        if bundle["parent_blocks"] != [{k:b[k] for k in parent_keys} for b in reference["parent_blocks"]]:
            raise ValueError("Canonical197 parent allocation differs")
        bundle["candidates"] = reference["candidates"]
        catalog["baseline"] = np.asarray(catalog["baseline"],dtype=np.float64)
        for edge in catalog["edges"]: edge["delta_windows"] = np.asarray(edge["delta_windows"],np.int64)
        for cut in bundle["candidates"]: cut["delta_windows"] = np.asarray(cut["delta_windows"],np.int64)
        # Only the candidate references are needed beyond this point.
        catalog.pop("cut_catalog")
        return dict(ref,raw=raw,rows=raw,bundle=bundle,catalog=catalog)
    return load


def verify_manifest(root,names=()):
    root = Path(root);digest = sha(root/"manifest.json")
    if (root/"SEALED.sha256").read_text().strip() != digest: raise ValueError("Complete top seal required")
    manifest = json.loads((root/"manifest.json").read_text())
    for name in names:
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != manifest[name]: raise ValueError("Input artifact changed")
    return manifest,digest


def field_score(args):
    import sys
    import inspect
    import gc
    if (args.out/"SEALED.sha256").exists(): raise FileExistsError("Completed score already sealed")
    if not re.fullmatch(r"[0-9a-f]{40}",args.prereg or ""): raise ValueError("Full prereg SHA required")
    repo = args.repo.resolve();root = args.root.resolve()
    required = ("frozen","owners","predictions","capacity")
    if not all((root/k/"SEALED.sha256").is_file() for k in required): raise ValueError("All four complete stage seals required")
    sys.path[:0] = [str(repo),str(repo/"tools/study197")]
    subprocess.run(["git","merge-base","--is-ancestor",args.prereg,"origin/main"],cwd=repo,capture_output=True,check=True)
    registration = subprocess.run(["git","show",args.prereg+":docs/experiment_log.md"],cwd=repo,capture_output=True,check=True).stdout.decode("utf-8")
    if registration.count("[orphan-continuation-v1]") != 1: raise ValueError("Unique orphan registration marker required")
    import freeze_orphans as frozen_module
    import fit_orphans
    import owners_orphans
    from tools.study197.physical import canonicalize
    from tools.study195.run import score_member_native,_gt_provider
    from tools.study195.freeze import write_mot_exact
    from tools.study198.run198 import combine
    import beetrack.evaluate as official
    from tools.study193 import source193
    predictions = fit_orphans.load_predictions(root/"predictions")
    receipt = predictions["receipt"]
    frozen_manifest,frozen_sha = verify_manifest(root/"frozen",("deployment.json",))
    _,owner_sha = verify_manifest(root/"owners")
    _,pred_sha = verify_manifest(root/"predictions")
    _,cap_sha = verify_manifest(root/"capacity",("compact_input.json","capacity.json"))
    if receipt["prereg"] != args.prereg or receipt["freeze_receipt"]["manifest_sha256"] != frozen_sha or receipt["owners_manifest_sha256"] != owner_sha:
        raise ValueError("Prediction inputs/prereg differ")
    compact_input = json.loads((root/"capacity/compact_input.json").read_text())
    expected_seals = dict(frozen=frozen_sha,owners=owner_sha,predictions=pred_sha)
    if compact_input["source_seals"] != expected_seals or compact_input["prereg"] != args.prereg:
        raise ValueError("Capacity was computed for another closure")
    capacity = json.loads((root/"capacity/capacity.json").read_text())
    if capacity["input_receipt"]["input_sha256"] != sha(root/"capacity/compact_input.json") or capacity["input_receipt"]["prereg"] != args.prereg:
        raise ValueError("Capacity input receipt differs")
    del compact_input
    data = json.loads((root/"frozen/deployment.json").read_text())
    if data["prereg"] != args.prereg: raise ValueError("Frozen prereg differs")
    groups = headers_from_freeze(data)
    if data["source197_seal"] != "e7fd48078f8d0a32bb9018f2bac93dc4b5270d374d5bcd0a631fa97dd4d790e4":
        raise ValueError("Registered197 source differs")
    # All four model/mask seals and all capacity closures passed before target labels.
    owners = {};GT_hashes = None
    for domain in DOMAINS:
        shard = owners_orphans.load_domain(root/"owners",domain)
        if shard["prereg"] != args.prereg or shard["frozen_manifest_sha256"] != frozen_sha: raise ValueError("Owner input differs")
        if GT_hashes is None: GT_hashes = shard["GT_sha256"]
        elif GT_hashes != shard["GT_sha256"]: raise ValueError("Owner GT files differ across shards")
        for unit in shard["units"]:
            owners[unit["unit"]] = {}
            for member in unit["members"]:
                owners[unit["unit"]][str(member["member"])] = {
                    a["action_id"]:dict(labels={t:dict(**{"class":a["labels"][t]["class"]}) for t in ("0.5","0.3")})
                    for a in [*member["edges"],*member["cuts"]]}
        del shard
    _,old_sha = verify_manifest(args.source197_score,("native_scores.json",))
    old = json.loads((args.source197_score/"native_scores.json").read_text())["K"]
    references = {(g["unit"],str(m["member"])):compact_metric(m["metrics"]) for g in old for m in g["members"]}
    del old
    model_audit = {item["unit"]:{} for f in predictions["folds"].values() for item in f["member_clips"]}
    for fold in predictions["folds"].values():
        for item in fold["member_clips"]: model_audit[item["unit"]][str(item["member"])] = item
    # Full unfiltered official prepared GT, including24 negative-extent rows.
    truths,GT_receipts = _gt_provider(repo,groups,DOMAINS,"evaluation")
    for proof in GT_receipts.values():
        if GT_hashes[proof["path"]] != proof["sha256"]: raise ValueError("Taxonomy/native GT differs")
    code = dict(materialize_sha256=sha(frozen_module.__file__),official_evaluate_sha256=sha(official.__file__),
        scorer_sha256=sha(inspect.getsourcefile(score_member_native)),source193_sha256=sha(source193.__file__),
        writer_sha256=sha(inspect.getsourcefile(write_mot_exact)),native_cache_sha256=hashlib.sha256(inspect.getsource(NativeCache).encode()).hexdigest(),
        numpy_version=np.__version__)
    identity = dict(prereg=args.prereg,source_seals=expected_seals,capacity_seal=cap_sha,source197_score_seal=old_sha,GT_entry=GT_receipts,code=code)
    gc.collect()
    result = evaluate_stream(groups,member_loader(root/"frozen",canonicalize,frozen_module.verify),predictions["decisions"],
        capacity["units"],owners,lambda g:truths[g["video"]],args.out,frozen_module.materialize,score_member_native,write_mot_exact,
        combine,code,identity,lambda g,m:references[g["unit"],str(m["member"])],model_audit)
    result.update(prereg=args.prereg,input_receipt=identity,source_support={h:json.loads((root/"predictions"/(h+".model.json")).read_text())["source_domains"] for h in DOMAINS})
    dump(args.out/"summary.json",result)
    print("ORPHAN_SCORES_SEALED "+seal(args.out),flush=True)
    return result


def selfcheck():
    assert object_sha(dict(edge_ids=["x"],cut_ids=[])) == object_sha(dict(cut_ids=[],edge_ids=["x"]))
    return dict(startup_pass=True,field_read=False,old_graph_F1_used=False)


def synthetic_integration(repo,out):
    import sys
    from types import SimpleNamespace
    sys.path[:0] = [str(Path(repo).resolve()),str(Path(repo).resolve()/"tools/study197")]
    import freeze_orphans as frozen
    import owners_orphans as owner
    import fit_orphans as fit
    import cap_orphans as cap
    from tools.study197.candidates import build_candidates
    from tools.study197.physical import canonicalize
    from tools.study199.owners199 import analyze_member
    from tools.study195.run import score_member_native
    from tools.study195.freeze import write_mot_exact
    from tools.study198.run198 import combine
    out = Path(out);out.mkdir(parents=True,exist_ok=False)
    poly = [[0,0],[20,0],[20,20],[0,20]]
    track = lambda pid,xs,y:np.array([[f,pid,x,y,2,2,.9] for f,x in enumerate(xs,1)],float)
    turn = track(1,[-4,1,3,1,-3],2);other = track(2,[-5,-3,1,3,4],14)
    raw = np.vstack((turn[:4],other));gt = np.vstack((turn,other))[:,:6]
    metadata=[];labels={};truths={};refs={};options=[];skeleton=[];original_members={}
    for hive in DOMAINS:
        unit = "toy_"+hive;truths[unit] = gt;labels[unit]={}
        for mi in range(3):
            bundle = build_candidates(raw,poly,75)
            context = dict(unit=unit,domain=hive,hive=hive,group=1,video=unit,clip=unit,member=str(mi),member_index=mi,tag="synthetic")
            for c in bundle["candidates"]:
                c.update(context,candidate_id=f"{unit}|{mi}|{c['parent_id']}")
            part = dict(retained_sources=[("synthetic",i) for i in range(len(raw))],orphans=[dict(rows=turn[4:].copy(),
                sources=[("synthetic",9)],source_s1_index=np.array([9]))])
            part["orphans"][0]["rows"][:,1] = 90
            catalog = frozen.build_catalog(raw,part,poly,75,unit,mi)
            reference = {k:v for k,v in bundle.items() if k not in ("raw_rows","canonical_rows","source_raw_index","polygon","nframes")}
            catalog.update(cut_catalog=reference,feature_names_extension=list(frozen.EXTENSION_FEATURES))
            folder = out/"frozen"/f"u{len(metadata):02d}";folder.mkdir(parents=True)
            np.savez(folder/"rows.npz",original_K=raw);dump(folder/"catalog.json",catalog);dump(folder/"receipt.json",dict(synthetic=True));child_seal=seal(folder)
            ref = dict(context,nf=75,polygon=poly,split="val",seed=mi,path=folder.name,seal=child_seal)
            metadata.append(ref);skeleton.append(context)
            member = dict(ref,raw=raw.copy(),bundle=bundle,catalog=catalog);original_members[unit,mi] = member
            edge_labels = owner.label_member(gt,catalog,poly,context)["edges"]
            cuts = owner.copy_cut_labels(analyze_member(gt,catalog["baseline"],bundle,poly,75)["records"])
            labels[unit][str(mi)] = {a["action_id"]:a for a in [*edge_labels,*cuts]}
            for e in catalog["edges"]:
                options.append(dict(context,action_id=e["edge_id"],type="extension",features=e["features10"],
                    recipient=e["recipient_cut_join_key"],orphan_key=f"{unit}|{mi}|orphan:{e['orphan_linked_id']}",edge_id=e["edge_id"],
                    parent_canonical_id=e["parent_canonical_id"],gap=e["gap"],max_error=e["normalized_error"]))
            for c in bundle["candidates"]:
                options.append(dict(context,action_id="cut:"+c["candidate_id"],type="cut",candidate_id=c["candidate_id"],
                    recipient=c["candidate_id"],features=list(c["features_G"])+[float(c["delta"]),float(c["direction"]==1)],
                    delta=c["delta"],direction=c["direction"],parent_canonical_id=c["parent_id"]))
            refs[unit,str(mi)] = compact_metric(score_member_native(gt,catalog["baseline"],poly,75))
    def provider(target):
        source = [r for r in options if r["hive"] != target]
        return dict(source_hives=[h for h in DOMAINS if h != target],labels={r["action_id"]:labels[r["unit"]][str(r["member"])][r["action_id"]]["labels"]["0.5"]["class"] for r in source})
    fit.train_and_seal(options,provider,out/"predictions",dict(prereg="synthetic"),skeleton)
    predicted = fit.load_predictions(out/"predictions")
    audits = {u:{} for u in labels}
    for fold in predicted["folds"].values():
        for m in fold["member_clips"]: audits[m["unit"]][str(m["member"])] = m
    groups = headers_from_freeze(dict(members=metadata),field_layout=False)
    oracle = {}
    for g in groups:
        if g["domain"] not in ("35","33"): continue
        acts=[];base=[]
        for m in g["members"]:
            acts.extend(labels[g["unit"]][str(m["member"])].values());base.append(refs[g["unit"],str(m["member"])])
        problem = dict(unit=g["unit"],hive=g["domain"],group=1,video=g["video"],nf=75,
            pred_sum=sum(np.asarray(m["pred_bins"],np.int64) for m in base),gt_bins=np.asarray(base[0]["gt_bins"],np.int64),
            actions=acts,member_indices=[0,1,2])
        oracle[g["unit"]] = {family:cap.solve_unit(problem,family) for family in ORACLE_ARMS}
    loader = member_loader(out/"frozen",canonicalize,frozen.verify)
    calls=[0]
    def fault(*args):
        calls[0]+=1
        if calls[0]==5:raise RuntimeError("injected synthetic interruption")
        return score_member_native(*args)
    kwargs=dict(groups=groups,load_member=loader,decisions=predicted["decisions"],oracles=oracle,owners=labels,
        gt_provider=lambda g:truths[g["unit"]],out=out/"score",materialize=frozen.materialize,write_mot=write_mot_exact,
        combine=combine,code_identity=dict(synthetic="same-scorer"),resume_identity=dict(synthetic="same-round"),
        reference_provider=lambda g,m:refs[g["unit"],str(m["member"])],model_audit=audits,field_layout=False)
    try:evaluate_stream(score_native=fault,**kwargs)
    except RuntimeError as error:assert "injected" in str(error)
    else:raise AssertionError("No synthetic interruption")
    preserved={p:sha(p) for p in (out/"score/physical").glob("*/SEALED.sha256")};assert len(preserved)==4
    result=evaluate_stream(score_native=score_member_native,**kwargs)
    assert all(sha(p)==s for p,s in preserved.items())
    assert result["physical_scope"]["requested_arm_members"]==84
    assert result["physical_scope"]["resumed_unique_physical"]==4
    assert all(d["methods"]["K"]["E3"]==100 for d in result["domains"].values())
    assert all(d["methods"]["L"]["E3"]==0 for d in result["domains"].values())
    assert all(d["methods"]["L"]["F1"]["0.5"]==1 for d in result["domains"].values())
    first=original_members["toy_35",0];edge=first["catalog"]["edges"][0]
    cut=next(c["candidate_id"] for c in first["bundle"]["candidates"] if c["parent_id"]==edge["parent_canonical_id"])
    try:frozen.materialize(groups[0],first,[cut],[edge["edge_id"]])
    except ValueError:pass
    else:raise AssertionError("Cut/extension parent conflict accepted")
    return dict(synthetic_pass=True,full_fit_cap_owner_materialize_native=True,cache=result["physical_scope"],
                interrupted_then_resumed=True,new_frame_matching_used=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score",action="store_true")
    parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--source197-score",type=Path)
    parser.add_argument("--prereg")
    args = parser.parse_args()
    if args.score:
        if args.synthetic:parser.error("Choose score or synthetic")
        if any(getattr(args,k) is None for k in ("repo","root","out","source197_score","prereg")):
            parser.error("--score requires --repo --root --out --source197-score --prereg")
        field_score(args)
    elif args.synthetic:
        if args.repo is None or args.out is None:parser.error("--synthetic requires --repo --out")
        print(json.dumps(synthetic_integration(args.repo,args.out)))
    else:
        print(json.dumps(selfcheck()))
