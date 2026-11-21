"""Private endpoint-cut scoring integration; no field IO occurs on import.

Physical scorer, frozen candidates, predicted masks and source197 oracle are
provided by their owners. GT is accepted only at score entry, after a caller's
four-fold prediction seal check. Main currently runs synthetic aggregation only.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fractions import Fraction
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np

ARMS = ("K", "A", "H", "G", "T", "M", "O")
DOMAINS = ("35", "33", "W", "G")
EXPECTED_GROUPS = {"35": 4, "33": 4, "W": 3, "G": 3}


def combine(videos):
    """Pool clip raw error and each member TP/pred/GT before calculating F1.

    Input video has domain,group,video,members[{member,metrics}], where metrics
    is the shared physical scorer's metrics dictionary (integer native bins).
    """
    if not videos:
        raise ValueError("Nonempty actual clips required")
    domain, group = videos[0]["domain"], videos[0]["group"]
    order = [m["member"] for m in videos[0]["members"]]
    if len(order) != 3 or len(set(order)) != 3:
        raise ValueError("Exactly three distinct physical members required")
    if len({v["video"] for v in videos}) != len(videos):
        raise ValueError("Duplicate physical video in one group")
    numerator, denominator, clip_records = 0, 0, []
    for video in videos:
        if (video["domain"] != domain or video["group"] != group
                or [m["member"] for m in video["members"]] != order):
            raise ValueError("Domain, group and physical member order must match across clips")
        metrics = [m["metrics"] for m in video["members"]]
        truth = np.asarray(metrics[0]["gt_bins"], dtype=np.int64)
        ng = int(metrics[0]["nG"])
        if any(m["nG"] != ng or not np.array_equal(m["gt_bins"], truth) for m in metrics):
            raise ValueError("Three copies of actual clip GT must agree")
        pred = sum((np.asarray(m["pred_bins"], dtype=np.int64) for m in metrics), np.zeros_like(truth))
        if any(np.asarray(m["pred_bins"]).shape != truth.shape for m in metrics):
            raise ValueError("Phase/bin/direction layouts differ")
        raw = int(np.abs(pred-3*truth).sum())
        numerator += raw
        denominator += 90*ng
        clip_records.append(dict(video=video["video"], numerator=raw, denominator=90*ng,
                                 gt_bins=truth.tolist(), sum_pred_bins=pred.tolist()))
    iou = {}
    for threshold in ("0.5", "0.3"):
        members = []
        for index, identity in enumerate(order):
            pooled = {key: sum(int(v["members"][index]["metrics"]["iou"][threshold][key]) for v in videos)
                      for key in ("tp", "pred", "gt", "tp_id")}
            pooled.update(member=identity, f1=2*pooled["tp"]/max(pooled["pred"]+pooled["gt"], 1))
            members.append(pooled)
        iou[threshold] = dict(members=members,
                              mean_member_f1=sum(m["f1"] for m in members)/3)
    return dict(domain=domain, group=group, videos=[v["video"] for v in videos],
                numerator=numerator, denominator=denominator,
                E3=100*numerator/denominator if denominator else None,
                iou=iou, clips=clip_records)


def summarize(scored, *, enforce_field_layout=False):
    """scored[arm] lists actual clip scores; groups and then four domains equal."""
    if set(scored) != set(ARMS):
        raise ValueError("Exactly K/A/H/G/T/M/O scores required")
    summaries = {}
    expected_keys = None
    for arm in ARMS:
        keys = {(v["domain"], v["group"], v["video"]) for v in scored[arm]}
        if len(keys) != len(scored[arm]) or expected_keys is not None and keys != expected_keys:
            raise ValueError("Duplicate or unmatched physical clips across arms")
        expected_keys = keys
    for domain in sorted({v["domain"] for v in scored["K"]}):
        group_ids = sorted({v["group"] for v in scored["K"] if v["domain"] == domain})
        if enforce_field_layout:
            if domain not in EXPECTED_GROUPS or len(group_ids) != EXPECTED_GROUPS[domain]:
                raise ValueError("Fixed 4/4/3/3 group layout changed")
        methods = {}
        for arm in ARMS:
            grouped = [combine([v for v in scored[arm] if v["domain"] == domain and v["group"] == group])
                       for group in group_ids]
            if enforce_field_layout and any(len(g["videos"]) != (2 if domain in ("W", "G") else 1) for g in grouped):
                raise ValueError("Wrong physical clip count")
            if any(g["denominator"] == 0 for g in grouped):
                raise ValueError("Whole group GT empty")
            methods[arm] = dict(groups=grouped,
                E3=float(sum(Fraction(100*g["numerator"], g["denominator"]) for g in grouped)/len(grouped)),
                F1={t:float(sum((sum(Fraction(2*m["tp"], max(m["pred"]+m["gt"], 1)) for m in g["iou"][t]["members"])/3
                                for g in grouped), Fraction())/len(grouped)) for t in ("0.5", "0.3")})
        summaries[domain] = dict(methods=methods,
            differences_vs_K={arm:dict(E3=methods[arm]["E3"]-methods["K"]["E3"],
                                      F1_05=methods[arm]["F1"]["0.5"]-methods["K"]["F1"]["0.5"])
                              for arm in ARMS})
    if enforce_field_layout and set(summaries) != set(DOMAINS):
        raise ValueError("All four heldout folds required")
    return dict(domains=summaries,
                oracle_note="O is count-optimal physical selection; its measured F1 is not an F1-constrained optimality certificate")


def assert_K_reference(actual, reference):
    """Compare every member/clip to the exact §195 F native score, not rounded W."""
    for key in ("nG", "nP", "pred_bins", "gt_bins"):
        np.testing.assert_array_equal(actual[key], reference[key], err_msg=f"K/F mismatch {key}")
    for threshold in ("0.5", "0.3"):
        for key in ("tp", "pred", "gt", "tp_id"):
            if actual["iou"][threshold][key] != reference["iou"][threshold][key]:
                raise ValueError(f"K/F official mismatch {threshold}/{key}")


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_plain(v) for v in value]
    return value


def score_explicit(groups, decisions, gt_provider, reference_provider, out,
                   physical_module=None, oracle_module=None, enforce_field_layout=True):
    """Score already sealed masks; all inputs explicit, final GT via callback.

    groups follow §195 clip-group schema with member.raw/member.bundle added.
    decisions[group.unit][member.member][K/A/H/G/T/M] are local candidate IDs.
    gt_provider(group) returns unfiltered official GT (including §24 anomalies).
    reference_provider(group,member) returns exact §195 F score. Caller MUST
    verify all four prediction/input seals before invoking this GT entrance.
    Output contains every mask and full physical MOT, including selected O.
    """
    if physical_module is None:
        import physical as physical_module
    if oracle_module is None:
        import oracle as oracle_module
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    physical_dir = out / "physical"
    physical_dir.mkdir()
    if set(decisions) != {g["unit"] for g in groups}:
        raise ValueError("All clip-group decisions required before GT")
    # Validate masks before the first GT callback, including all four domains.
    for group in groups:
        mm = group["members"]
        if len(mm) != 3 or len({str(m["member"]) for m in mm}) != 3:
            raise ValueError("Exactly three distinct members required")
        if set(decisions[group["unit"]]) != {str(m["member"]) for m in mm}:
            raise ValueError("Prediction member set mismatch")
        for member in mm:
            masks = decisions[group["unit"]][str(member["member"])]
            ids = {c["candidate_id"] for c in member["bundle"]["candidates"]}
            if set(masks) != set(ARMS)-{"O"} or masks["K"]:
                raise ValueError("K/A/H/G/T/M masks required, K must be empty")
            if any(len(mask) != len(set(mask)) or not set(mask) <= ids for mask in masks.values()):
                raise ValueError("Invalid local candidate mask")
            candidates_by_id = {c["candidate_id"]:c for c in member["bundle"]["candidates"]}
            if set(masks["A"]) != ids or set(masks["H"]) != {cid for cid,c in candidates_by_id.items() if c["H"]}:
                raise ValueError("All-cut or fixed heuristic mask changed")
            strata = lambda arm:Counter((candidates_by_id[c]["delta"],candidates_by_id[c]["direction"]) for c in masks[arm])
            if strata("T") != strata("M"):
                raise ValueError("T/M member/clip/add-drop/direction counts differ")
    scored = {arm:[] for arm in ARMS}
    proofs, realized = [], []
    for group_index, group in enumerate(groups):
        gt = gt_provider(group)
        caches = [{} for _ in range(3)]
        local_masks = [dict(decisions[group["unit"]][str(m["member"])]) for m in group["members"]]
        records = {arm:dict(domain=group["domain"], group=group["group"], video=group["video"],
                                  unit=group["unit"], members=[]) for arm in ARMS}
        def score_member(index, arm):
            member = group["members"][index]
            mask = local_masks[index][arm]
            key = tuple(sorted(mask))
            path = physical_dir / f"u{group_index:02d}_m{index}_{arm}.txt"
            if key not in caches[index]:
                result = physical_module.score_configuration(gt, member["raw"], member["bundle"], mask,
                    group["polygon"], group["nf"], physical_path=path)
                caches[index][key] = result
            else:
                original = caches[index][key]
                rows, meta = physical_module.realize(member["raw"], member["bundle"], mask)
                _, receipt = physical_module.write_physical(rows, path)
                result = dict(original, physical_receipt=receipt, realization=meta)
            records[arm]["members"].append(dict(member=str(member["member"]), metrics=result["metrics"]))
            realized.append(dict(unit=group["unit"], member=str(member["member"]), arm=arm,
                                 mask=mask, path=path.relative_to(out).as_posix(), proof=result))
            return result["metrics"]
        baseline = []
        for index, member in enumerate(group["members"]):
            metrics = score_member(index, "K")
            assert_K_reference(metrics, reference_provider(group, member))
            baseline.append(metrics)
        gt_bins = np.asarray(baseline[0]["gt_bins"])
        assert all(m["nG"] == baseline[0]["nG"] and np.array_equal(m["gt_bins"], gt_bins) for m in baseline)
        pred_sum = sum(np.asarray(m["pred_bins"]) for m in baseline)
        candidates, lookup = [], {}
        for index, member in enumerate(group["members"]):
            for candidate in member["bundle"]["candidates"]:
                global_id = f"{index}|{candidate['candidate_id']}"
                candidates.append(dict(candidate, candidate_id=global_id))
                lookup[global_id] = (index, candidate["candidate_id"])
            local_masks[index]["O"] = []
        oracle = oracle_module.solve_clip(pred_sum, gt_bins, candidates, group["nf"])
        for selected in oracle["selected_ids"]:
            index, local = lookup[selected]
            local_masks[index]["O"].append(local)
        proofs.append(dict(unit=group["unit"], oracle=oracle, global_to_local=lookup))
        for arm in ARMS[1:]:
            for index in range(3):
                score_member(index, arm)
        for arm in ARMS:
            scored[arm].append(records[arm])
        actual = combine([records["O"]])
        assert actual["numerator"] == oracle["upper_bound"]
        assert actual["denominator"] == oracle["denominator"]
        np.testing.assert_array_equal(actual["clips"][0]["sum_pred_bins"], oracle["witness_pred_sum"])
        # M and T match per-member whole-clip directional event totals.
        for index in range(3):
            a, b = (np.asarray(records[arm]["members"][index]["metrics"]["pred_bins"]) for arm in ("M", "T"))
            np.testing.assert_array_equal(a.sum(axis=0), b.sum(axis=0))
        print("SCORED "+group["unit"], flush=True)
    summary = summarize(scored, enforce_field_layout=enforce_field_layout)
    summary["oracle_certificates"] = proofs
    units = {g["unit"]:g for g in groups}
    bounds = {}
    for proof in proofs:
        group = units[proof["unit"]]
        key = (group["domain"], group["group"])
        total = bounds.setdefault(key, dict(lower=0, upper=0, denominator=0))
        oracle = proof["oracle"]
        total["lower"] += oracle["lower_bound"]
        total["upper"] += oracle["upper_bound"]
        total["denominator"] += oracle["denominator"]
    for domain, result in summary["domains"].items():
        bb = [dict(group=group, **b) for (hive,group),b in bounds.items() if hive == domain]
        result["oracle_bounds"] = dict(groups=bb,
            mean_lower_E3=float(np.mean([100*b["lower"]/b["denominator"] for b in bb])),
            mean_upper_E3=float(np.mean([100*b["upper"]/b["denominator"] for b in bb])),
            all_optimal=all(b["lower"] == b["upper"] for b in bb))
        assert np.isclose(result["oracle_bounds"]["mean_upper_E3"], result["methods"]["O"]["E3"])
        for arm in ARMS:
            result["methods"][arm]["candidate_count"] = sum(len(m["bundle"]["candidates"]) for g in groups if g["domain"] == domain for m in g["members"])
            result["methods"][arm]["cut_count"] = sum(len(r["mask"]) for r in realized if r["arm"] == arm and units[r["unit"]]["domain"] == domain)
    if set(summary["domains"]) == set(DOMAINS):
        d = summary["domains"]
        summary["investment_checks"] = dict(
            head35_gain=d["35"]["differences_vs_K"]["T"]["E3"] < -1.4,
            head33_gain=d["33"]["differences_vs_K"]["T"]["E3"] < -1.3,
            W_no_harm=d["W"]["differences_vs_K"]["T"]["E3"] <= 1.4,
            F1_no_harm=all(d[h]["differences_vs_K"]["T"]["F1_05"] >= -.005 for h in ("35", "33", "W")))
        summary["investment_pass"] = all(summary["investment_checks"].values())
        summary["motion_attribution_pass"] = all(d[h]["methods"]["T"]["E3"] < d[h]["methods"][other]["E3"] for h in ("35", "33") for other in ("G", "M"))
        summary["oracle_family_excluded"] = {h:d[h]["oracle_bounds"]["mean_lower_E3"] >= d[h]["methods"]["K"]["E3"]-gain for h,gain in (("35",1.4),("33",1.3))}
    for name, value in (("native_scores.json", scored), ("physical_receipts.json", realized), ("summary.json", summary)):
        (out/name).write_text(json.dumps(_plain(value), indent=2, allow_nan=False))
    return summary


def synthetic_check():
    def metric(gt, pred, tp):
        iou = {t:dict(tp=tp, pred=pred, gt=gt, tp_id=tp) for t in ("0.5", "0.3")}
        return dict(nG=gt, nP=pred, gt_bins=[[gt, 0]]*30, pred_bins=[[pred, 0]]*30, iou=iou)
    a = dict(domain="W", group=1, video="a", members=[dict(member=i, metrics=metric(1, p, min(p, 1))) for i,p in enumerate((2,0,1))])
    b = dict(domain="W", group=1, video="b", members=[dict(member=i, metrics=metric(9, 9, 9)) for i in range(3)])
    result = combine([a, b])
    assert result["numerator"] == 0 and result["denominator"] == 900
    assert result["E3"] == 0  # 2/0/1 votes cancel before taking absolute error.
    assert result["iou"]["0.5"]["members"][0]["f1"] == 20/21
    assert result["iou"]["0.5"]["members"][1]["f1"] == 18/19
    scored = {arm:[a,b] for arm in ARMS}
    report = summarize(scored)
    assert report["domains"]["W"]["methods"]["K"]["E3"] == 0
    assert_K_reference(a["members"][0]["metrics"], a["members"][0]["metrics"])
    return dict(passed=True, pooled=result)


def synthetic_physical(repo, out):
    """Small artificial MOTs through actual candidate/oracle/physical interfaces."""
    import sys
    sys.path.insert(0, str(Path(repo).resolve()))
    import candidates
    from tools.study195.run import score_member_native
    poly = [[0,0], [10,0], [10,30], [0,30]]
    groups, decisions, truths, references = [], {}, {}, {}
    for clip, xs in (("synthetic_a", [14,7,6,5,4]), ("synthetic_b", [4,14,14,14,7])):
        raw = np.asarray([[f,7,x,5,2,2,.8] for f,x in enumerate(xs,1)], float)
        truth = raw.copy()
        truth[:,1] = 101
        bundle = candidates.build_candidates(raw, poly, 75)
        group = dict(unit=clip, domain="W", group=1, video=clip, polygon=poly, nf=75,
                     members=[dict(member=str(i), raw=raw.copy(), bundle=bundle) for i in range(3)])
        groups.append(group)
        truths[clip] = truth
        decisions[clip] = {}
        for member in group["members"]:
            ids = [c["candidate_id"] for c in bundle["candidates"]]
            h = [c["candidate_id"] for c in bundle["candidates"] if c["H"]]
            decisions[clip][member["member"]] = dict(K=[], A=ids, H=h, G=[], T=ids, M=ids)
            references[clip, member["member"]] = score_member_native(truth, raw, poly, 75)
    result = score_explicit(groups, decisions, lambda g:truths[g["video"]],
                            lambda g,m:references[g["unit"],m["member"]], out,
                            enforce_field_layout=False)
    assert result["domains"]["W"]["methods"]["K"]["E3"] == 0
    return result


def field_score(args):
    """Registered final evaluation only; caller waits for root GO to invoke CLI."""
    import sys
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    if not re.fullmatch(r"[0-9a-f]{40}", args.prereg or ""):
        raise ValueError("Full preregistration commit required")
    subprocess.run(["git","merge-base","--is-ancestor",args.prereg,"origin/main"], cwd=repo, check=True, capture_output=True)
    log = subprocess.run(["git","show",args.prereg+":docs/experiment_log.md"], cwd=repo, check=True, capture_output=True).stdout.decode("utf-8")
    if log.count("[endpoint-cut-v1]") != 1:
        raise ValueError("Unique endpoint-cut preregistration marker required")
    import freeze
    import model
    from tools.study195 import run as previous
    groups = freeze.load_groups(args.frozen)
    frozen_receipt = freeze.verify(args.frozen)
    frozen_metadata = json.loads((args.frozen/"deployment.json").read_text())
    if frozen_metadata["prereg"] != args.prereg:
        raise ValueError("Freeze prereg mismatch")
    predictions = model.load_predictions(args.predictions)
    receipt = predictions["receipt"]
    if receipt["prereg"] != args.prereg:
        raise ValueError("Prediction prereg mismatch")
    # model.load_predictions validates all four heldout/source-fold boundaries
    # and frozen-input binding; verify the exposed binding once more here.
    if receipt["freeze_receipt"] != frozen_receipt:
        raise ValueError("Prediction frozen-input mismatch")
    if set(predictions["folds"]) != set(DOMAINS):
        raise ValueError("All four model folds must be sealed before evaluation")
    for heldout, fold in predictions["folds"].items():
        if fold["heldout"] != heldout:
            raise ValueError("Heldout fold changed")
    if args.out.exists():
        raise FileExistsError(args.out)
    # Only read old numeric scores, not the other60 arm's physical MOT files.
    old_score_sha = previous._sha(args.source195_score/"manifest.json")
    if (args.source195_score/"SEALED.sha256").read_text().strip() != old_score_sha:
        raise ValueError("Prior score manifest seal mismatch")
    old_manifest = json.loads((args.source195_score/"manifest.json").read_text())
    if previous._sha(args.source195_score/"native_scores.json") != old_manifest["native_scores.json"]:
        raise ValueError("Prior native numeric scores changed")
    original = json.loads((args.source195_score/"native_scores.json").read_text())
    references = {}
    for group in original:
        fixed = "t50" if group["domain"] in ("35","33") else "t40"
        for member in group["members"]:
            references[group["unit"], str(member["member"])] = member["arms"][fixed]
    if set(references) != {(g["unit"],str(m["member"])) for g in groups for m in g["members"]}:
        raise ValueError("All60 prior F references required")
    # This is the first GT read: source/prediction manifests have all passed.
    arrays, gt_proofs = previous._gt_provider(repo, groups, DOMAINS, "evaluation")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent, prefix=".endpoint-score-") as temporary:
        stage = Path(temporary)/"payload"
        summary = score_explicit(groups, predictions["decisions"], lambda g:arrays[g["video"]],
             lambda g,m:references[g["unit"],str(m["member"])], stage)
        summary.update(prereg=args.prereg, freeze_receipt=frozen_receipt,
                       prediction_receipt=receipt, source195_score_manifest_sha256=old_score_sha,
                       GT_entry=gt_proofs, single_cpu=True,
                       code_sha256={p.name:previous._sha(p) for p in Path(__file__).parent.glob("*.py")})
        (stage/"summary.json").write_text(json.dumps(_plain(summary), indent=2, allow_nan=False))
        import hashlib
        manifest = {}
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    manifest[path.relative_to(stage).as_posix()] = hashlib.file_digest(stream,"sha256").hexdigest()
        (stage/"manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":"))+"\n")
        sha = previous._sha(stage/"manifest.json")
        (stage/"SEALED.sha256").write_text(sha+"\n")
        stage.rename(args.out)
    print("SCORES_SEALED "+str(args.out)+" "+sha, flush=True)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--source195-score", type=Path)
    parser.add_argument("--prereg")
    args = parser.parse_args()
    if args.score:
        if args.synthetic or any(getattr(args,key) is None for key in ("repo","out","frozen","predictions","source195_score","prereg")):
            parser.error("--score requires --repo --out --frozen --predictions --source195-score --prereg")
        field_score(args)
    elif args.synthetic:
        if args.repo is None or args.out is None:
            parser.error("Physical synthetic requires --repo and --out")
        check = synthetic_physical(args.repo, args.out)
        print(json.dumps(dict(synthetic_pass=True, summary=str(args.out / "summary.json"))))
    else:
        check = synthetic_check()
        print(json.dumps(dict(synthetic_aggregation_pass=check["passed"])))
