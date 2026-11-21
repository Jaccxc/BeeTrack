"""Post-seal numeric aggregation only; never reads images or raw GT/MOT.

Models/all blind decisions must verify before the physical result file opens.
Abstentions count0 for S; physical tables use their actual blind-given fallback.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import model


def physical_swap(pair, arm):
    return (arm["physical_choice"] == "cross") ^ bool(pair["order_bit"])


def vote(pair, arm):
    return 0 if arm["decision"] == "tie" else -1 if physical_swap(pair, arm) else 1


def _vote_stats(pairs, variant, method):
    clips = {}
    votes = []
    missing = 0
    for pair in pairs:
        arm = pair["variants"][variant][method]
        value = vote(pair, arm)
        votes.append(value)
        missing += int(not arm["valid"])
        clips.setdefault(pair["clip"], []).append(value)
    return dict(S=float(np.mean([np.mean(v) for v in clips.values()])) if clips else None,
        pairs=len(pairs), clips=len(clips), positive=votes.count(1), negative=votes.count(-1),
        abstained=votes.count(0), missing=missing,
        per_clip={clip: dict(N=len(v), sum_vote=sum(v), S=float(np.mean(v))) for clip, v in sorted(clips.items())})


def _identity_keys(pairs, with_run):
    keys = set()
    for pair in pairs:
        if model._cohort(pair) != "old":
            continue
        for n in (1, 2):
            key = (pair["clip"], int(pair[f"id{n}"]))
            keys.add(key+(int(pair[f"run{n}"]),) if with_run else key)
    return keys


def _physical_stats(nodes):
    N = sum(int(node["N"]) for node in nodes)
    GT = sum(int(node["nGT"]) for node in nodes)
    P = sum(int(node["nPred"]) for node in nodes)
    iou = {}
    for threshold in ("0.5", "0.3"):
        tp = sum(int(node["iou"][threshold]["tp"]) for node in nodes)
        if tp > min(P, GT):
            raise ValueError("invalid aggregate physical TP")
        iou[threshold] = dict(TP=tp, F1=2*tp/(P+GT) if P+GT else None)
    return dict(pairs=len(nodes), N=N, nGT=GT, nPred=P, TP=iou["0.5"]["TP"],
                W_percent=100*N/(30*GT) if GT else None, F1=iou["0.5"]["F1"], iou=iou,
                zero_GT_pairs=sum(node["nGT"] == 0 for node in nodes))


def _gates(domains):
    mechanism, physical = {}, {}
    for hive in ("W", "G"):
        d = domains[hive]
        s = d["S"]["all"]
        get = lambda variant, arm: s[variant][arm]["S"]
        R_T, R_C, Z_T, Q_T, R_V, Q_V = get("R", "T"), get("R", "C"), get("Z", "T"), get("Q", "T"), get("R", "V"), get("Q", "V")
        available = all(v is not None for v in (R_T, R_C, Z_T, Q_T, R_V, Q_V))
        mechanism[hive] = dict(new_pairs_at_least16=d["pairs"] >= 16, clips_at_least3=d["clips"] >= 3,
            T_R_minus_C_R=R_T-R_C if available else None,
            T_R_minus_T_Z=R_T-Z_T if available else None,
            T_R_minus_T_Q=R_T-Q_T if available else None,
            V_R_minus_V_Q=R_V-Q_V if available else None,
            T_over_C=bool(available and R_T-R_C >= .05),
            T_over_shift=bool(available and R_T-Z_T >= .05),
            T_over_shuffle=bool(available and R_T-Q_T >= .05),
            V_over_shuffle=bool(available and R_V > Q_V))
        c, t = d["physical"]["C"], d["physical"]["T"]
        physical[hive] = dict(GT_at_least10=c["nGT"] >= 10,
            W_nonincrease=bool(c["W_percent"] is not None and t["W_percent"] is not None and t["W_percent"] <= c["W_percent"]),
            W_strict_decrease=bool(c["W_percent"] is not None and t["W_percent"] is not None and t["W_percent"] < c["W_percent"]),
            F1_guard=bool(c["F1"] is not None and t["F1"] is not None and t["F1"] >= c["F1"]-.005))
    mechanism_pass = all(all(row[k] for k in ("new_pairs_at_least16", "clips_at_least3", "T_over_C", "T_over_shift", "T_over_shuffle", "V_over_shuffle")) for row in mechanism.values())
    physical_pass = all(all(row[k] for k in ("GT_at_least10", "W_nonincrease", "F1_guard")) for row in physical.values()) and any(row["W_strict_decrease"] for row in physical.values())
    return dict(mechanism=dict(by_hive=mechanism, passed=mechanism_pass),
                physical=dict(by_hive=physical, passed=physical_pass), both_passed=mechanism_pass and physical_pass)


def aggregate(predictions, physical, roster_pairs):
    pairs, results = predictions["pairs"], physical["results"]
    if len({p["key"] for p in pairs}) != len(pairs) or set(results) != {p["key"] for p in pairs}:
        raise ValueError("physical/new prediction key sets differ")
    old_ids, old_runs = _identity_keys(roster_pairs, False), _identity_keys(roster_pairs, True)
    domains, per_pair = {}, []
    for hive in ("W", "G"):
        hp = [p for p in pairs if p["hive"] == hive]
        subsets = {name: [] for name in ("all", "near_boundary", "P_wrong", "P_tie", "P_wrong_or_tie", "old_id_any_reuse", "old_run_any_reuse")}
        changed = dict(any=0, window=0, TP05=0, TP03=0, predicted_count=0)
        selected = {arm: [] for arm in ("identity", "P", "C", "T", "V")}
        mismatches = []
        for p in hp:
            trial = results[p["key"]]
            if trial["hive"] != hive or trial["pair"]["clip"] != p["clip"]:
                raise ValueError("numeric physical metadata differs")
            meta = p["metadata"]
            pv = vote(p, p["variants"]["R"]["P"])
            reused_id = any((p["clip"], int(meta[f"id{n}"])) in old_ids for n in (1, 2))
            reused_run = any((p["clip"], int(meta[f"id{n}"]), int(meta[f"run{n}"])) in old_runs for n in (1, 2))
            flags = dict(all=True, near_boundary=bool(meta.get("near_boundary", False)), P_wrong=pv < 0,
                         P_tie=pv == 0, P_wrong_or_tie=pv <= 0, old_id_any_reuse=reused_id, old_run_any_reuse=reused_run)
            for group, flag in flags.items():
                if flag:
                    subsets[group].append(p)
            identity, cross = trial["identity"], trial["cross"]
            selected["identity"].append(identity)
            choices = {}
            for arm in ("P", "C", "T", "V"):
                choice = "cross" if physical_swap(p, p["variants"]["R"][arm]) else "identity"
                choices[arm] = choice
                selected[arm].append(trial[choice])
            delta = dict(window=cross["N"] != identity["N"],
                TP05=cross["iou"]["0.5"]["tp"] != identity["iou"]["0.5"]["tp"],
                TP03=cross["iou"]["0.3"]["tp"] != identity["iou"]["0.3"]["tp"],
                predicted_count=cross["nPred"] != identity["nPred"])
            for key, flag in delta.items():
                changed[key] += int(flag)
            changed["any"] += int(any(delta.values()))
            if not trial.get("identity_official_check", {}).get("passed", True):
                mismatches.append(p["key"])
            per_pair.append(dict(key=p["key"], hive=hive, clip=p["clip"], flags=flags,
                votes={variant: {arm: vote(p, p["variants"][variant][arm]) for arm in ("P", "C", "T", "V")} for variant in model.VARIANTS},
                actual_R_choices=choices))
        S = {}
        for group, subset in subsets.items():
            S[group] = {variant: {arm: _vote_stats(subset, variant, arm) for arm in ("P", "C", "T", "V")} for variant in model.VARIANTS}
            S[group]["Q"] = {}
            for arm in ("P", "C", "T", "V"):
                stats = [S[group][f"Q{q}"][arm] for q in range(4)]
                S[group]["Q"][arm] = dict(S=float(np.mean([r["S"] for r in stats])) if stats[0]["S"] is not None else None,
                    definition="arithmetic mean of4 complete clip-balanced S values", replicates=[f"Q{q}" for q in range(4)])
        domains[hive] = dict(pairs=len(hp), clips=len({p["clip"] for p in hp}), S=S,
            physical={arm: _physical_stats(nodes) for arm, nodes in selected.items()}, physical_changed=changed,
            identity_official_mismatch_keys=mismatches)
    return dict(schema="pair-texture-score-v1", prereg=predictions["prereg"], domains=domains,
                gates=_gates(domains), per_pair=per_pair,
                interpretation="S uses abstention0; physical uses actual given fallback. Reused bees/frames are repeated controlled trials, not independent samples.")


def _verify_bundle(root):
    root = Path(root)
    if (root/"SEALED.sha256").read_text().strip() != model._sha(root/"manifest.json"):
        raise ValueError("numeric input bundle not sealed")
    for name, sha in json.loads((root/"manifest.json").read_text()).items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or model._sha(path) != sha:
            raise ValueError("numeric input file hash mismatch")


def score_sealed(prediction_dir, physical_dir, feature_dir, output_dir):
    # This gate is deliberately before opening physical truth/numeric results.
    predictions = model.verify_prediction_seal(prediction_dir)
    _verify_bundle(physical_dir); _verify_bundle(feature_dir)
    physical = json.loads((Path(physical_dir)/"physical.json").read_text())
    roster = json.loads((Path(feature_dir)/"roster.json").read_text())
    if physical["prereg"] != predictions["prereg"] or roster["prereg"] != predictions["prereg"]:
        raise ValueError("preregistration differs across sealed inputs")
    receipts = physical["input_receipts"]
    if (receipts["predictions_manifest_sha256"] != model._sha(Path(prediction_dir)/"manifest.json")
            or receipts["features_manifest_sha256"] != model._sha(Path(feature_dir)/"manifest.json")
            or predictions["features_sha256"] != model._sha(Path(feature_dir)/"features.json")):
        raise ValueError("physical/descriptor/prediction freeze mismatch")
    result = aggregate(predictions, physical, roster["pairs"])
    result["input_receipts"] = dict(predictions_manifest_sha256=model._sha(Path(prediction_dir)/"manifest.json"),
        physical_manifest_sha256=model._sha(Path(physical_dir)/"manifest.json"), features_manifest_sha256=model._sha(Path(feature_dir)/"manifest.json"),
        evaluator_sha256=model._sha(__file__))
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".texture-score-", dir=root.parent) as tmp:
        work = Path(tmp)/"payload"; work.mkdir()
        model._save(work/"score.json", result)
        model._save(work/"manifest.json", {"score.json": model._sha(work/"score.json")})
        (work/"SEALED.sha256").write_text(model._sha(work/"manifest.json")+"\n")
        work.rename(root)
    return result


def self_test():
    pairs = []
    physical = dict(results={})
    roster = []
    for hive in ("W", "G"):
        for n in range(4):
            clip = hive+str(n % 2)
            bit = n % 2
            correct = dict(score=1., decision="cross" if bit else "given", physical_choice="cross" if bit else "given", valid=True)
            tie = dict(score=0., decision="tie", physical_choice="given", valid=False)
            p = dict(key=hive+str(n), hive=hive, clip=clip, order_bit=bit,
                metadata=dict(near_boundary=n == 0, id1=1, run1=0, id2=2, run2=0),
                variants={v: {a: dict(correct) for a in ("P", "C", "T", "V")} for v in model.VARIANTS})
            if n == 1:
                p["variants"]["R"]["T"] = tie
            pairs.append(p)
            def metric(N, gt, pred, tp):
                return dict(N=N, nGT=gt, nPred=pred, TP=tp, iou={"0.5": dict(tp=tp), "0.3": dict(tp=tp)})
            gt = 0 if n == 1 else 1
            physical["results"][p["key"]] = dict(hive=hive, pair=dict(clip=clip),
                identity=metric(0, gt, gt, gt), cross=metric(30, gt, 1 if gt == 0 else 0, 0))
        for c in range(2):
            roster.append(dict(hive=hive, clip=hive+str(c), cohort="old", id1=1, run1=0, id2=2, run2=0))
    predictions = dict(prereg="synthetic", pairs=pairs)
    result = aggregate(predictions, physical, roster)
    for hive in ("W", "G"):
        d = result["domains"][hive]
        assert d["S"]["all"]["R"]["C"]["S"] == 1
        assert d["S"]["all"]["R"]["T"]["S"] == .75
        assert d["S"]["all"]["R"]["T"]["abstained"] == 1
        assert d["physical"]["T"]["N"] == 30 and d["physical"]["T"]["nPred"] == 4
        assert d["physical"]["T"]["zero_GT_pairs"] == 1
        assert d["physical"]["T"]["W_percent"] == 100/3
        assert d["S"]["all"]["Q"]["T"]["S"] == 1
        assert d["S"]["old_id_any_reuse"]["R"]["C"]["pairs"] == 4
    assert not result["gates"]["both_passed"]
    # Clip-balanced S is not pooled pair accuracy, and missing stays in N.
    unbalanced = [dict(pairs[0], clip="a"), dict(pairs[0], clip="b"), dict(pairs[0], clip="b")]
    unbalanced[0]["variants"] = {v: {a: dict(score=0, decision="tie", physical_choice="given", valid=False) for a in ("P", "C", "T", "V")} for v in model.VARIANTS}
    assert _vote_stats(unbalanced, "R", "C")["S"] == .5
    zero = _physical_stats([dict(N=30, nGT=0, nPred=1, iou={"0.5": dict(tp=0), "0.3": dict(tp=0)})])
    assert zero["W_percent"] is None and zero["F1"] == 0
    return dict(clip_balanced_S="passed", abstention_vs_actual_given_fallback="passed", zero_GT_false_positive="retained",
                shuffled_Q="mean of4 S", reuse_and_P_subsets="passed", input_scope="pure numeric fixtures only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--physical-dir", type=Path)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif all((args.predictions_dir, args.physical_dir, args.feature_dir, args.out)):
        result = score_sealed(args.predictions_dir, args.physical_dir, args.feature_dir, args.out)
        print(json.dumps(dict(gates=result["gates"], score_manifest_sha256=model._sha(args.out/"manifest.json")), indent=2))
    else:
        parser.error("use --self-test or all four input/output directories")
