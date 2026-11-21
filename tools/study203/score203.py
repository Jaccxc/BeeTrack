"""Native complete-scene suffix scoring; importing this module reads no data.

Scoring is downstream of jointly sealed four-fold choices. A whole unit's
allocation must be validated before any member is materialized. All official
matching is rerun on complete physical scenes; single-action deltas validate
counts only, never supply event F1.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np

_spec = importlib.util.spec_from_file_location("suffix203_score_common", Path.cwd()/"tools/study202/score_orphans.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
plain, dump, sha, object_sha, array_sha = common.plain, common.dump, common.sha, common.object_sha, common.array_sha
seal, verify, compact_metric, assert_K = common.seal, common.verify, common.compact_metric, common.assert_K
ARMS = ("K", "G", "I", "L", "T", "M")
FAMILIES = ("S", "P", "A")
DOMAINS = ("35", "33", "W", "G")


def allocation(group, ids):
    """Replay all host/source/inserted-IoU resources jointly across three members."""
    import capacity203
    table = {c["candidate_id"]:c for c in group["candidates"]}
    if len(set(ids)) != len(ids) or not set(ids) <= table.keys():
        raise ValueError("Foreign or duplicated group allocation")
    active = [dict(table[cid],action_id=cid) for cid in ids]
    used = set()
    for resources in capacity203.compile_resources(active):
        for resource in resources:
            if resource in used:
                raise ValueError("Group host/source/IoU resource allocated twice")
            used.add(resource)
    import freeze203
    freeze203.validate_allocation(group,ids)


def materialize(group, member, ids):
    """Use the supply writer on one host after complete group allocation checks."""
    import freeze203
    one_host = dict(group,members=[dict(member,rows=member["raw"])])
    return freeze203.materialize_group(one_host,str(member["member"]),ids)


def event_effect(candidate):
    before,after = candidate["events_before"],candidate["events_after"]
    if len(after) > len(before): return "added"
    if len(after) < len(before): return "removed"
    return "unchanged" if before == after else "date_only"


def selection(value):
    """A complete immutable candidate mask, including the empty K mask."""
    if not isinstance(value, dict) or set(value) != {"candidate_ids"}:
        raise ValueError("Complete candidate_ids mask required")
    ids = value["candidate_ids"]
    if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique strings")
    return sorted(ids)


class NativeCache:
    """Safe resume for an identical host, full suffix mask and full row sources."""
    def __init__(self, out, materialize, score_native, write_mot, code_identity):
        self.out = Path(out)
        self.materialize, self.score_native, self.write_mot = materialize, score_native, write_mot
        self.code_identity = code_identity
        self.seen, self.calls = set(), []
        self.fresh = self.resumed = 0

    def score(self, group, member, ids, gt, arm):
        ids = selection(dict(candidate_ids=ids))
        rows, proof = self.materialize(group, member, ids)
        rows = np.asarray(rows, dtype=np.float64)
        raw = np.asarray(member["raw"], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 7 or not np.isfinite(rows).all():
            raise ValueError("Invalid complete physical scene")
        if len(rows) < len(raw) or rows[:len(raw)].tobytes() != raw.tobytes():
            raise ValueError("Original K row IDs/date/order/measurements changed")
        if (not np.array_equal(rows[:, :2], np.rint(rows[:, :2])) or np.any(rows[:, 0] < 1)
                or np.any(rows[:, 0] > group["nf"]) or np.any(rows[:, 1] < 1)
                or np.any(rows[:, 4:6] <= 0) or len(np.unique(rows[:, :2], axis=0)) != len(rows)):
            raise ValueError("Invalid clock/boxes/duplicate track-date")
        sources = plain(proof["sources"])
        if len(sources) != len(rows) or len({object_sha(s) for s in sources}) != len(rows):
            raise ValueError("Missing or duplicated physical source rows")
        table = {c["candidate_id"]:c for c in group["candidates"]}
        expected_sources = plain(member["sources"])
        added = []
        for cid in ids:
            c = table[cid]
            native_id = raw[c["host_anchor_row_indices"][-1],1]
            for suffix in c["suffix"]:
                expected_sources.append(plain(suffix["source"]))
                row = np.asarray(suffix["row"],dtype=np.float64).copy()
                row[1] = native_id
                added.append(row)
        expected_added = np.asarray(added,dtype=np.float64).reshape(-1,7)
        if sources!=expected_sources or rows[len(raw):].tobytes()!=expected_added.tobytes():
            raise ValueError("Materialized suffix changes source measurements or assignment")
        identity = dict(unit=group["unit"], member=str(member["member"]), candidate_ids=ids,
                        raw_sha256=array_sha(raw), rows_sha256=array_sha(rows), GT_sha256=array_sha(gt),
                        sources_sha256=object_sha(sources), polygon_sha256=array_sha(group["polygon"]),
                        nf=int(group["nf"]), code=self.code_identity)
        token = object_sha([identity["unit"], identity["member"], ids])
        path = self.out/token
        shared = token in self.seen
        from_disk = path.exists()
        if path.exists():
            receipt = verify(path)
            if receipt["identity"] != identity:
                raise ValueError("Cached state differs in complete source/input/selection")
            reread = np.loadtxt(path/"mot.txt", delimiter=",", usecols=range(7), ndmin=2) if len(rows) else np.empty((0,7))
            if reread.tobytes() != rows.tobytes():
                raise ValueError("Cached physical MOT differs")
            metric = json.loads((path/"compact.json").read_text())
            self.resumed += int(not shared)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix="."+token+"-", dir=path.parent))
            try:
                mot_receipt = self.write_mot(rows, tmp/"mot.txt")
                reread = np.loadtxt(tmp/"mot.txt", delimiter=",", usecols=range(7), ndmin=2) if len(rows) else np.empty((0,7))
                if reread.tobytes() != rows.tobytes():
                    raise ValueError("Physical MOT float roundtrip changed rows")
                native = self.score_native(gt, reread, group["polygon"], group["nf"])
                metric = compact_metric(native)
                dump(tmp/"score.json", native)
                dump(tmp/"compact.json", metric)
                with gzip.open(tmp/"row_sources.jsonl.gz", "wt", encoding="utf-8") as stream:
                    for source in sources:
                        stream.write(json.dumps(source, separators=(",", ":"))+"\n")
                dump(tmp/"receipt.json", dict(identity=identity, mot=mot_receipt, original_rows=len(raw),
                    added_rows=len(rows)-len(raw), materialization_proof_sha256=object_sha(proof),
                    new_fullscene_official_matching=True, original_K_rows_unchanged=True))
                seal(tmp)
                tmp.rename(path)
                self.fresh += 1
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp)
        self.seen.add(token)
        self.calls.append(dict(unit=group["unit"], member=str(member["member"]),
            member_index=int(member["member_index"]), arm=arm, candidate_ids=ids, path=token,
            within_job_shared=shared, resumed_from_disk=from_disk and not shared,
            seal=(path/"SEALED.sha256").read_text().strip()))
        return metric


def summarize(scored, combine, field_layout=True):
    report = {}
    if set(scored) != set(ARMS+FAMILIES):
        raise ValueError("Exactly six deployment and three capacity arms required")
    for hive in DOMAINS:
        videos = [v for v in scored["K"] if v["domain"] == hive]
        if not videos:
            continue
        gids = sorted({v["group"] for v in videos})
        if field_layout and len(gids) != common.GROUPS[hive]:
            raise ValueError("Registered group scope changed")
        methods = {}
        for arm in ARMS+(FAMILIES if hive in ("35", "33") else ()):
            vv = [v for v in scored[arm] if v["domain"] == hive]
            if len({(v["group"],v["video"]) for v in vv}) != len(vv):
                raise ValueError("Duplicate member group/clip")
            groups = [combine([v for v in vv if v["group"] == gid]) for gid in gids]
            if any(g["denominator"] <= 0 for g in groups):
                raise ValueError("Empty whole GT group")
            methods[arm] = dict(groups=groups,
                E3=float(sum(Fraction(100*g["numerator"],g["denominator"]) for g in groups)/len(groups)),
                F1={t:float(sum((sum(Fraction(2*m["tp"],max(m["pred"]+m["gt"],1))
                    for m in g["iou"][t]["members"])/3 for g in groups),Fraction())/len(groups))
                    for t in ("0.5","0.3")})
            exact = sum((Fraction(100*g["numerator"],g["denominator"]) for g in groups),Fraction())/len(groups)
            methods[arm]["E3_fraction"] = [exact.numerator,exact.denominator]
            methods[arm]["F1_fraction"] = {}
            for threshold in ("0.5","0.3"):
                exact_f1 = sum((sum((Fraction(2*m["tp"],max(m["pred"]+m["gt"],1))
                    for m in g["iou"][threshold]["members"]),Fraction())/3 for g in groups),Fraction())/len(groups)
                methods[arm]["F1_fraction"][threshold] = [exact_f1.numerator,exact_f1.denominator]
        report[hive] = dict(methods=methods)
    result = dict(schema="peer-suffix203-native-score-v1", domains=report,
        F1_contract="Fresh official full-scene matching; integer deltas validate counts only")
    if set(report) == set(DOMAINS):
        def improvement(h, a):
            return Fraction(*report[h]["methods"]["K"]["E3_fraction"])-Fraction(*report[h]["methods"][a]["E3_fraction"])
        def checks(a):
            return dict(head35_gain=improvement("35",a)>Fraction(14,10),head33_gain=improvement("33",a)>Fraction(13,10),
                W_no_harm=improvement("W",a)>=-Fraction(14,10),
                F1_no_harm=all(Fraction(*report[h]["methods"][a]["F1_fraction"]["0.5"]) >=
                    Fraction(*report[h]["methods"]["K"]["F1_fraction"]["0.5"])-Fraction(5,1000) for h in ("35","33","W")))
        result["investment_checks"] = {a:checks(a) for a in ("L","T")}
        result["investment_pass"] = {a:all(v.values()) for a,v in result["investment_checks"].items()}
        result["attribution"] = dict(
            T_strictly_beats_L_both_heads=all(report[h]["methods"]["T"]["E3"] < report[h]["methods"]["L"]["E3"] for h in ("35","33")),
            T_strictly_beats_I_G_M_both_heads=all(report[h]["methods"]["T"]["E3"] < report[h]["methods"][a]["E3"]
                for h in ("35","33") for a in ("I","G","M")),
            interpretation="Configuration/event learning and ranking controls; not identification of individual pipeline steps")
    return result


def audit_predictions(predictions):
    """Replay sealed logits, within-host T permutation and outcome/global choices."""
    import fit203
    for hive,fold in predictions["folds"].items():
        contexts = fold["contexts"]
        scores = {a:{} for a in ("G","I","L","T","M")}
        for member in fold["member_clips"]:
            for a in scores:
                if set(scores[a]) & set(member["scores"][a]):
                    raise ValueError("Repeated logit ID across host members")
                scores[a].update(member["scores"][a])
        permuted,audit = fit203.permute(contexts,scores["T"])
        if permuted!=scores["M"] or audit!=fold["M_permutation"]:
            raise ValueError("M does not permute all T logits within the same host")
        for arm in scores:
            replay = fit203.outcome_choices(contexts,scores[arm])
            if replay!=fold["outcomes"][arm]:
                raise ValueError("Sealed outcome/global allocation replay differs")
            selected = sorted(cid for m in fold["member_clips"] for cid in m["selected"][arm]["candidate_ids"])
            if selected!=replay["selected"]:
                raise ValueError("Member masks differ from global outcome allocation")


def frozen_loader(root):
    """Capture paths only; decode and release one complete three-member unit."""
    root = Path(root).resolve()
    def load(descriptor):
        import freeze203
        folder = (root/descriptor["path"]).resolve()
        if not folder.is_relative_to(root) or sha(folder/"manifest.json")!=descriptor["seal"]:
            raise ValueError("Frozen group path/seal differs")
        group = freeze203.load_group(folder)
        for key in ("unit","domain","group","video"):
            if group[key]!=descriptor[key]:
                raise ValueError("Frozen group descriptor mismatch")
        for member in group["members"]:
            member["raw"] = member["rows"]
        return group
    return load


def evaluate(groups, decisions, capacities, owners, gt_provider, out, *, materialize,
             validate_allocation, score_native, write_mot, combine, reference_provider,
             code_identity, resume_identity, field_layout=True, load_group=None):
    """Score fully decoded synthetic groups; field adapters must verify seals first.

    Each group has members(raw,member,member_index), candidates(candidate_id,
    host_member,delta_windows,event_effect). The shared allocation validator is
    called on the complete three-member union before materializing any host.
    """
    units = {g["unit"] for g in groups}
    if set(decisions) != units or len(units) != len(groups):
        raise ValueError("Exactly one sealed deployment record per group/clip")
    heads = {g["unit"] for g in groups if g["domain"] in ("35","33")}
    if set(capacities) != heads:
        raise ValueError("Exactly registered head capacity units required")
    common.prepare_output(out, resume_identity)
    out = Path(out)
    cache = NativeCache(out/"physical", materialize, score_native, write_mot, code_identity)
    scored = {a:[] for a in ARMS+FAMILIES}
    stats = {}
    for descriptor in groups:
        group = load_group(descriptor) if load_group is not None else descriptor
        members = group["members"]
        if sorted(m["member_index"] for m in members) != [0,1,2] or len({str(m["member"]) for m in members}) != 3:
            raise ValueError("Three distinct original member scenes required")
        candidates = {c["candidate_id"]:c for c in group["candidates"]}
        if len(candidates) != len(group["candidates"]):
            raise ValueError("Duplicate frozen candidate IDs")
        arms = ARMS+(FAMILIES if group["unit"] in heads else ())
        selections = {}
        for arm in arms:
            masks = {}
            for m in members:
                mask = (decisions[group["unit"]][str(m["member"])][arm] if arm in ARMS else
                    capacities[group["unit"]][arm]["member_masks"][str(m["member_index"])] )
                ids = selection(mask)
                if arm == "K" and ids:
                    raise ValueError("K must preserve the complete original scene")
                if any(cid not in candidates or str(candidates[cid]["host_member"]) != str(m["member"]) for cid in ids):
                    raise ValueError("Foreign candidate or wrong host")
                if arm in ("S","P"):
                    allowed = {"S"} if arm=="S" else {"S","U"}
                    if any(owners[group["unit"]][cid]["labels"]["0.5"]["class"] not in allowed for cid in ids):
                        raise ValueError("Capacity mask selected outside formal family")
                masks[str(m["member"])] = ids
            all_ids = [cid for ids in masks.values() for cid in ids]
            if len(set(all_ids)) != len(all_ids):
                raise ValueError("Duplicate cross-member candidate allocation")
            validate_allocation(group, all_ids)
            selections[arm] = masks
        current = {a:dict(unit=group["unit"],domain=group["domain"],group=group["group"],
            video=group["video"],nf=group["nf"],members=[]) for a in arms}
        for member in members:
            host = str(member["member"])
            gt = gt_provider(group)
            baseline = None
            for arm in arms:
                ids = selections[arm][host]
                metric = cache.score(group,member,ids,gt,arm)
                if arm == "K":
                    baseline = metric
                    assert_K(metric,reference_provider(group,member))
                expected = np.asarray(baseline["pred_bins"],dtype=np.int64).copy()
                for cid in ids:
                    delta = np.asarray(candidates[cid]["delta_windows"])
                    if not np.issubdtype(delta.dtype,np.integer):
                        raise ValueError("Action delta must be an integer vector")
                    expected += delta
                np.testing.assert_array_equal(expected,np.asarray(metric["pred_bins"],dtype=np.int64),
                    err_msg="Complete physical 30-phase single-action delta closure")
                pooled_member = str(member.get("source_record",{}).get("member",host))
                current[arm]["members"].append(dict(member=pooled_member,host_member=host,member_index=int(member["member_index"]),
                    metrics=metric,candidate_ids=ids,physical=cache.calls[-1]["path"],
                    seal=cache.calls[-1]["seal"],delta_closure_exact=True))
                stat = stats.setdefault((group["domain"],arm),dict(selected=Counter(),available=Counter(),effects=Counter(),
                    no_window_effect=0,available_no_window_effect=0,selected_rows=0,
                    class_by_iou={t:dict(selected=Counter(),available=Counter()) for t in ("0.5","0.3")}))
                for cid,c in candidates.items():
                    if str(c["host_member"]) != host:
                        continue
                    label = owners[group["unit"]][cid]["labels"]["0.5"]["class"]
                    if label not in ("S","D","U"):
                        raise ValueError("Unknown native owner class")
                    stat["available"][label] += 1
                    stat["available_no_window_effect"] += int(not np.asarray(c["delta_windows"]).any())
                    for threshold in ("0.5","0.3"):
                        cls = owners[group["unit"]][cid]["labels"][threshold]["class"]
                        if cls not in ("S","D","U"): raise ValueError("Invalid sensitivity owner class")
                        stat["class_by_iou"][threshold]["available"][cls] += 1
                        if cid in ids: stat["class_by_iou"][threshold]["selected"][cls] += 1
                    if cid in ids:
                        stat["selected"][label] += 1
                        stat["effects"][event_effect(c)] += 1
                        stat["no_window_effect"] += int(not np.asarray(c["delta_windows"]).any())
                        stat["selected_rows"] += len(c["suffix"])
            del gt
        for arm in arms:
            scored[arm].append(current[arm])
        for arm in FAMILIES if group["unit"] in heads else ():
            actual, cap = combine([current[arm]]), capacities[group["unit"]][arm]
            if actual["numerator"] != cap["upper_bound"] or actual["denominator"] != cap["denominator"]:
                raise ValueError("Capacity native witness objective differs")
            np.testing.assert_array_equal(actual["clips"][0]["sum_pred_bins"],cap["witness_pred_sum"])
            import capacity203
            problem = dict(nf=group["nf"],pred_sum=np.sum([m["metrics"]["pred_bins"] for m in current["K"]["members"]],axis=0).tolist(),
                gt_bins=current["K"]["members"][0]["metrics"]["gt_bins"],member_indices=[0,1,2],
                actions=[dict(c,action_id=c["candidate_id"],labels=owners[group["unit"]][c["candidate_id"]]["labels"])
                    for c in group["candidates"]])
            capacity203.verify_answer(problem,arm,cap)
        print("SCORED "+group["unit"],flush=True)
    report = summarize(scored,combine,field_layout)
    for hive in ("35","33"):
        if hive not in report["domains"]: continue
        bounds = {}
        for family in FAMILIES:
            rr = [capacities[g["unit"]][family] for g in groups if g["domain"]==hive]
            lower = sum((Fraction(100*r["lower_bound"],r["denominator"]) for r in rr),Fraction())/len(rr)
            upper = sum((Fraction(100*r["upper_bound"],r["denominator"]) for r in rr),Fraction())/len(rr)
            base = Fraction(*report["domains"][hive]["methods"]["K"]["E3_fraction"])
            line = Fraction(14 if hive=="35" else 13,10)
            bounds[family] = dict(mean_lower_E3=float(lower),mean_witness_E3=float(upper),
                guaranteed_gain=float(base-upper),maximum_possible_gain=float(base-lower),
                crosses_count_line=base-upper>line,excluded_by_lower_bound=base-lower<=line,
                all_optimal=all(r["lower_bound"]==r["upper_bound"] for r in rr),
                F1_claim="Physical witness only; count optimum has no F1 constraint")
        report["domains"][hive]["capacity_bounds"] = bounds
    for (h,a),stat in stats.items():
        item = {k:dict(v) if isinstance(v,Counter) else v for k,v in stat.items()}
        total = sum(stat["selected"].values())
        item["selected_proportions"] = {c:stat["selected"][c]/total if total else None for c in ("S","D","U")}
        item["class_coverage"] = {c:stat["selected"][c]/stat["available"][c] if stat["available"][c] else None for c in ("S","D","U")}
        report["domains"][h]["methods"][a]["action_stats"] = item
    report["physical_scope"] = dict(requested_arm_members=len(cache.calls),unique_streams=len(cache.seen),
        fresh_unique_physical=cache.fresh,resumed_unique_physical=cache.resumed,
        same_job_shared=sum(c["within_job_shared"] for c in cache.calls),historical_reused=0,
        study_fresh_unique_physical=cache.fresh+cache.resumed)
    dump(out/"native_scores.json",scored)
    dump(out/"physical_receipts.json",cache.calls)
    dump(out/"summary.json",report)
    return report


def synthetic(repo, out):
    """Actual supply, owner labels, count solver, .17g MOT and official matching."""
    import sys
    sys.path[:0] = [str(repo),str(repo/"tools/study197"),str(Path(__file__).parent),str(Path(__file__).parents[1])]
    from peer_suffix_supply import catalog
    import freeze203
    import owners203
    import capacity203
    import fit203
    from tools.study197.candidates import event_times,window_counts
    from tools.study195.run import score_member_native
    from tools.study195.freeze import write_mot_exact
    from tools.study198.run198 import combine
    polygon = np.asarray([[0,-10],[100,-10],[100,100],[0,100]],np.float32)
    def row(f,x): return [f,1,x,0,4,4,.9]
    full = np.asarray([row(f,x) for f,x in [(747,-6),(748,-5),(749,-4),(750,-3),(751,-2),(752,-1),(753,1)]],float)
    gt = full[:,:6]
    groups, decisions, caps, labels, references = [], {}, {}, {}, {}
    contexts,supervision,skeleton = [],{},[]
    for hive,clip_index in (("35",0),("33",0),("W",0),("W",1),("G",0),("G",1)):
        unit = "synthetic_"+hive+("_second" if clip_index else "")
        members = [dict(member=unit+"|"+str(i),member_index=i,rows=full[:3].copy() if i==0 else full.copy(),
            sources=[(unit+"-detector"+str(i),j) for j in range(3 if i==0 else len(full))],
            source_record=dict(member=hive+"|"+str(i))) for i in range(3)]
        context = dict(unit=unit,domain=hive,group=1,video=unit,nf=800,polygon=polygon)
        candidates = freeze203.enrich(catalog(members,polygon,800),members,context)
        assert len(candidates)==2
        group = dict(context,members=[dict(m,raw=m["rows"]) for m in members],candidates=candidates)
        groups.append(group)
        prepared = owners203.prepare_gt(gt,polygon)
        scenes = [owners203.scene(m["rows"],m["sources"]) for m in members]
        labels[unit] = {}
        for c in candidates:
            rows,proof = freeze203.materialize_single(members,c)
            inserted = owners203.scene(rows,proof["sources"])
            labels[unit][c["candidate_id"]] = owners203.label_candidate(prepared,scenes[c["host_member_index"]],
                scenes[c["donor_member_index"]],inserted,c)
        assert all(r["labels"]["0.5"]["class"]=="S" for r in labels[unit].values())
        chosen = candidates[0]["candidate_id"]
        decisions[unit] = {m["member"]:{a:dict(candidate_ids=[chosen] if a!="K" and m["member_index"]==0 else [])
            for a in ARMS} for m in members}
        option_resources = {}
        for resource in freeze203.resources(candidates):
            for cid in resource["candidate_ids"]:
                option_resources.setdefault(cid,[]).append(resource["resource"])
        options = []
        for c in candidates:
            freeze203.features(c,members,polygon)
            options.append(dict(action_id=c["candidate_id"],features10=c["features10"],
                outcome=window_counts(c["events_after"],800).tolist(),suffix_source_keys=c["suffix_source_keys"],
                resources=option_resources[c["candidate_id"]],min_anchor_iou=c["features10"][0],
                first_extrapolation_error=c["features10"][2]))
        host_key = unit+"|host1"
        contexts.append(dict(host_key=host_key,hive=hive,group=1,clip=context["video"],unit=unit,
            host_member=members[0]["member"],stratum="00000",options=options,
            baseline_outcome=window_counts(candidates[0]["events_before"],800).tolist()))
        supervision[host_key] = dict(valid=True,target_outcome=window_counts(event_times(gt,polygon),800).tolist(),
            classes={cid:r["labels"]["0.5"]["class"] for cid,r in labels[unit].items()})
        skeleton.extend(dict(unit=unit,host_member=m["member"],hive=hive,clip=context["video"]) for m in members)
        kk = [score_member_native(gt,m["rows"],polygon,800) for m in members]
        references[unit] = {m["member"]:compact_metric(k) for m,k in zip(members,kk)}
        if hive in ("35","33"):
            actions = [dict(c,labels=labels[unit][c["candidate_id"]]["labels"]) for c in candidates]
            problem = dict(nf=800,pred_sum=np.sum([k["pred_bins"] for k in kk],axis=0).tolist(),
                gt_bins=kk[0]["gt_bins"],actions=actions,member_indices=[0,1,2])
            caps[unit] = {f:capacity203.solve_unit(problem,f) for f in FAMILIES}
        try:
            allocation(group,[c["candidate_id"] for c in candidates])
        except ValueError:
            pass
        else:
            raise AssertionError("Conflicting host choices were accepted")
        c = dict(candidates[0],candidate_id="duplicate_donor",action_id="duplicate_donor",host_member=members[1]["member"])
        try:
            allocation(dict(group,candidates=candidates+[c]),[chosen,"duplicate_donor"])
        except ValueError:
            pass
        else:
            raise AssertionError("Cross-host donor row reuse was accepted")
    def source_provider(h):
        return dict(source_hives=[d for d in DOMAINS if d!=h],supervision={c["host_key"]:supervision[c["host_key"]]
            for c in contexts if c["hive"]!=h})
    prediction_root = Path(out).with_name(Path(out).name+"_predictions")
    fit203.train_and_seal(contexts,source_provider,prediction_root,dict(prereg="synthetic"),skeleton=skeleton)
    fitted = fit203.load_predictions(prediction_root)
    audit_predictions(fitted)
    decisions = fitted["decisions"]
    count = [0]
    def fail_once(*args):
        count[0] += 1
        if count[0]==4:
            raise RuntimeError("synthetic interrupted native state")
        return score_member_native(*args)
    kwargs = dict(materialize=materialize,validate_allocation=allocation,write_mot=write_mot_exact,combine=combine,
        reference_provider=lambda g,m:references[g["unit"]][m["member"]],code_identity={"synthetic":1},
        resume_identity={"synthetic":1},field_layout=False)
    try:
        evaluate(groups,decisions,caps,labels,lambda g:gt,out,score_native=fail_once,**kwargs)
    except RuntimeError as error:
        if str(error)!="synthetic interrupted native state": raise
    else:
        raise AssertionError("Synthetic interruption did not execute")
    result = evaluate(groups,decisions,caps,labels,lambda g:gt,out,score_native=score_member_native,**kwargs)
    assert result["physical_scope"]["requested_arm_members"]==126
    assert result["physical_scope"]["resumed_unique_physical"]==3
    assert all(result["domains"][h]["methods"]["L"]["E3"] < result["domains"][h]["methods"]["K"]["E3"] for h in DOMAINS)
    ledger = json.loads((Path(out)/"native_scores.json").read_text())
    w = [v for v in ledger["K"] if v["domain"]=="W"]
    assert [m["member"] for m in w[0]["members"]]==[m["member"] for m in w[1]["members"]]
    swapped = dict(w[1],members=list(reversed(w[1]["members"])))
    try:
        combine([w[0],swapped])
    except ValueError:
        pass
    else:
        raise AssertionError("Permuted single-clip global member alignment was accepted")
    result["synthetic_audits"] = dict(shared_supply=True,real_owner_labels=True,actual_capacity=True,
        four_source_only_folds_trained_sealed=True,logits_outcomes_and_M_replayed=True,
        fresh_official_matching=True,all30_integer_delta_closure=True,donor_reuse_rejected=True,
        conflicting_host_choices_rejected=True,interrupted_state_safely_resumed=True)
    result["synthetic_audits"]["two_clip_global_members_pooled_per_member"] = True
    result["synthetic_audits"]["single_clip_member_order_permutation_rejected"] = True
    dump(Path(out)/"summary.json",result)
    digest = seal(out)
    print("SCORE203_SYNTHETIC_PASS "+digest,flush=True)
    return result


def field_score(args):
    """The only field entry; all four common seals precede target label/GT IO."""
    import gc
    import inspect
    import re
    import subprocess
    import sys
    repo,root,out = args.repo.resolve(),args.root.resolve(),args.out.resolve()
    if out.drive.upper()!="D:" or not re.fullmatch("[0-9a-f]{40}",args.prereg or ""):
        raise ValueError("D: output and full published prereg SHA required")
    if (out/"SEALED.sha256").exists():
        raise FileExistsError("Complete score already sealed")
    required = ("frozen","owners","predictions","capacity")
    if not all((root/k/"SEALED.sha256").is_file() for k in required):
        raise ValueError("All four complete stage seals required before target evaluation")
    sys.path[:0] = [str(repo),str(repo/"tools/study197"),str(Path(__file__).parent),str(Path(__file__).parents[1])]
    subprocess.run(["git","merge-base","--is-ancestor",args.prereg,"origin/main"],cwd=repo,check=True,capture_output=True)
    registration = subprocess.run(["git","show",args.prereg+":docs/experiment_log.md"],cwd=repo,check=True,capture_output=True).stdout.decode("utf-8")
    if registration.count("[peer-suffix-observed-v1]")!=1:
        raise ValueError("Unique published peer suffix registration required")
    import freeze203
    import fit203
    import capacity203
    import peer_suffix_supply
    import beetrack.evaluate as official
    from tools.study195.run import score_member_native,_gt_provider
    from tools.study195.freeze import write_mot_exact
    from tools.study198.run198 import combine
    from tools.study193 import source193
    # Hash closures before decoding any owner classes or official target GT.
    freeze203.verify(root/"frozen")
    _,frozen_sha = common.verify_manifest(root/"frozen",("deployment.json",))
    _,owner_sha = common.verify_manifest(root/"owners",("classes.json",))
    predictions = fit203.load_predictions(root/"predictions")
    _,pred_sha = common.verify_manifest(root/"predictions")
    cap_manifest,cap_sha = common.verify_manifest(root/"capacity",("capacity.json","compact_input.json","input_receipt.json"))
    for name in cap_manifest:
        if name.endswith(".json") and name not in ("compact_input.json","capacity.json","input_receipt.json"):
            common.verify_manifest(root/"capacity",(name,))
    frozen_header = json.loads((root/"frozen/deployment.json").read_text())
    receipt = predictions["receipt"]
    if (frozen_header["prereg"]!=args.prereg or receipt["prereg"]!=args.prereg
            or receipt["freeze_receipt"]["manifest_sha256"]!=frozen_sha
            or receipt["owners_manifest_sha256"]!=owner_sha):
        raise ValueError("Common prediction/freeze/owner lineage differs")
    if set(receipt["arms"])!=set(ARMS) or set(receipt["folds"])!=set(DOMAINS):
        raise ValueError("All four folds/six arms must share one seal")
    capacity = json.loads((root/"capacity/capacity.json").read_text())
    cr = capacity["receipt"]
    if cr["prereg"]!=args.prereg or cr["prediction_seal"]!=pred_sha or cr["compact_input_sha256"]!=sha(root/"capacity/compact_input.json"):
        raise ValueError("Capacity does not follow these frozen predictions")
    expected_seals = dict(frozen=frozen_sha,owners=owner_sha,predictions=pred_sha)
    if "source_receipts" in cr:
        lineage = cr["source_receipts"]
    else:
        compact = json.loads((root/"capacity/compact_input.json").read_text())
        lineage = compact["source_receipts"]
        del compact
    if {k:v["manifest_sha256"] for k,v in lineage.items()}!=expected_seals:
        raise ValueError("Capacity complete input lineage differs")
    audit_predictions(predictions)
    descriptors = frozen_header["groups"]
    if len(descriptors)!=20 or len({g["unit"] for g in descriptors})!=20:
        raise ValueError("Exactly twenty old clip groups required")
    # Derive lightweight clocks/splits/roster without retaining all catalogs.
    headers = []
    for descriptor in descriptors:
        catalog = json.loads((root/"frozen"/descriptor["path"]/"catalog.json").read_text())
        old = catalog["members"][0]["source_record"]
        original_members = [str(m["source_record"]["member"]) for m in catalog["members"]]
        if len(original_members)!=3 or len(set(original_members))!=3:
            raise ValueError("Three one-to-one historical source members required")
        header = dict(descriptor,nf=catalog["nf"],polygon=catalog["polygon"],split=old["split"],source_members=original_members)
        if set(predictions["decisions"][descriptor["unit"]])!={str(m["member"]) for m in catalog["members"]}:
            raise ValueError("Sealed prediction member roster differs")
        if any(m["source_record"]["seed"]>=15 or m["source_record"]["split"]=="test" for m in catalog["members"]):
            raise ValueError("Source outside fixed old seed/clip whitelist")
        headers.append(header)
        del catalog,old
    if len({g["video"] for g in headers})!=6:
        raise ValueError("Exactly original six clips required")
    roster = {}
    for header in headers:
        key = (header["domain"],header["group"])
        if key in roster and roster[key]!=header["source_members"]:
            raise ValueError("Historical member/slot alignment differs across actual clips")
        roster[key] = header["source_members"]
    # Every seal/choice boundary above passed before loading the target taxonomy.
    classes = json.loads((root/"owners/classes.json").read_text())
    if classes["prereg"]!=args.prereg or classes["freeze_receipt"]["manifest_sha256"]!=frozen_sha:
        raise ValueError("Target classes refer to different frozen inputs")
    owners = {u["unit"]:{a["action_id"]:dict(labels=a["labels"]) for a in u["actions"]} for u in classes["units"]}
    GT_hashes = classes["GT_sha256"]
    del classes
    _,reference_sha = common.verify_manifest(args.source202_score,("native_scores.json",))
    if reference_sha!="90c8ca299d5bcee5078c298f471a355924b17e1e5808c710e97f682f09213800":
        raise ValueError("Wrong registered historical202 K reference")
    historical = json.loads((args.source202_score/"native_scores.json").read_text())["K"]
    references = {(g["unit"],str(m["member"])):compact_metric(m["metrics"]) for g in historical for m in g["members"]}
    del historical
    truths,GT_receipts = _gt_provider(repo,headers,DOMAINS,"evaluation")
    for proof in GT_receipts.values():
        if GT_hashes[proof["path"]]!=proof["sha256"]:
            raise ValueError("Formal target classes and native GT differ")
    code = dict(native_cache_sha256=hashlib.sha256(inspect.getsource(NativeCache).encode()).hexdigest(),
        materialize_adapter_sha256=hashlib.sha256(inspect.getsource(materialize).encode()).hexdigest(),
        allocation_adapter_sha256=hashlib.sha256(inspect.getsource(allocation).encode()).hexdigest(),
        materialize_sha256=sha(freeze203.__file__),
        supply_sha256=sha(peer_suffix_supply.__file__),capacity_sha256=sha(capacity203.__file__),
        official_evaluate_sha256=sha(official.__file__),scorer_sha256=sha(inspect.getsourcefile(score_member_native)),
        source193_sha256=sha(source193.__file__),writer_sha256=sha(inspect.getsourcefile(write_mot_exact)),
        numpy_version=np.__version__)
    identity = dict(prereg=args.prereg,source_seals=expected_seals,capacity_seal=cap_sha,
        source202_score_seal=reference_sha,GT_entry=GT_receipts,code=code)
    source_support = {}
    for hive in DOMAINS:
        model = json.loads((root/"predictions"/(hive+".model.json")).read_text())
        source_support[hive] = dict(source_domains=model["source_domains"],models={
            a:{k:model["models"][a][k] for k in ("status","coverage","artificial_coverage","weight_audit","source_hives")}
            for a in ("I","L","T")})
    gc.collect()
    prior_completed = len(list((out/"physical").glob("*/SEALED.sha256"))) if (out/"physical").exists() else 0
    result = evaluate(headers,predictions["decisions"],capacity["units"],owners,lambda g:truths[g["video"]],out,
        materialize=materialize,validate_allocation=allocation,score_native=score_member_native,write_mot=write_mot_exact,
        combine=combine,reference_provider=lambda g,m:references[(g["unit"],str(m["source_record"]["member"]))],
        code_identity=code,resume_identity=identity,load_group=frozen_loader(root/"frozen"))
    if result["physical_scope"]["requested_arm_members"]!=432:
        raise ValueError("Registered 360 deployment +72 capacity native requests required")
    result["physical_scope"]["prior_attempt_completed_unique_physical"] = prior_completed
    result.update(prereg=args.prereg,input_receipt=identity,source_support=source_support,
        natural_denominators=receipt["natural_denominators"],score_source_sha256=sha(__file__))
    dump(out/"summary.json",result)
    print("SCORE203_SEALED "+seal(out),flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--score",action="store_true")
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--source202-score",type=Path)
    parser.add_argument("--prereg")
    args = parser.parse_args()
    if args.synthetic==args.score:
        parser.error("Choose exactly one of --synthetic or --score")
    if args.synthetic:
        synthetic(args.repo.resolve(),args.out.resolve())
    else:
        if args.root is None or args.source202_score is None or args.prereg is None:
            parser.error("--score needs --root --source202-score --prereg")
        field_score(args)


if __name__ == "__main__":
    main()
