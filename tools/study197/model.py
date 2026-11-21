"""Source-only endpoint-cut ridge heads and four-fold blind mask seal.

No field reads occur at import/default self-test. Model inputs are only frozen
G9/T12 features. Labels are singleton group-error improvements, not independent
batch utilities: every deployed complete mask must be officially rescored.
"""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np

DOMAINS = ("35", "33", "W", "G")
ARMS = ("K", "A", "H", "G", "T", "M")
RIDGE, THRESHOLD = .20, .10
SCHEMA = "endpoint-cut-model-v1"
FROZEN_SHA = "e7fd48078f8d0a32bb9018f2bac93dc4b5270d374d5bcd0a631fa97dd4d790e4"
G_NAMES = ["k_minus1", "span_over12", "is_end", "short_inside", "parent_has_event",
           "short_minus_core_score", "log_parent_length", "short_abs_distance", "core_abs_distance"]
T_NAMES = G_NAMES+["short_extrapolation_residual", "signed_core_side_prediction", "core_fit_residual"]


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(v) for v in value]
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n", encoding="utf-8")


def _features(G, T):
    G, T = np.asarray(G, float), np.asarray(T, float)
    if G.size == 0: G = G.reshape(0, 9)
    if T.size == 0: T = T.reshape(0, 12)
    if G.ndim != 2 or G.shape[1] != 9 or T.shape != (len(G), 12) or not np.isfinite(G).all() or not np.isfinite(T).all():
        raise ValueError("finite aligned Nx9/Nx12 frozen feature matrices required")
    G, T = G.copy(), T.copy()
    G[:, 7:9] = np.clip(G[:, 7:9], -4, 4)
    T[:, 7:12] = np.clip(T[:, 7:12], -4, 4)
    if not np.array_equal(G, T[:, :9]):
        raise ValueError("T's first9 features differ from G after specified clipping")
    return G, T


def _feature_rows(records):
    return _features([r.get("features_G9", r.get("features_G")) for r in records],
                     [r.get("features_T12", r.get("features_T")) for r in records])


def _record_key(record):
    hive = record.get("hive", record.get("domain"))
    group = record.get("group_id", record.get("group"))
    clip = record.get("clip", record.get("video"))
    key = record.get("key", record.get("candidate_id"))
    if hive not in DOMAINS or not isinstance(clip, str) or not isinstance(key, str) or not key or group is None:
        raise ValueError("candidate requires hive/group/clip/globalkey")
    return hive, json.dumps(_plain(group), sort_keys=True), clip, key


def source_weights(records):
    keys = [_record_key(r) for r in records]
    if not keys: return np.empty(0)
    if len({k[3] for k in keys}) != len(keys):
        raise ValueError("global candidate keys must be distinct")
    hives = {k[0] for k in keys}
    groups = {h: {k[1] for k in keys if k[0] == h} for h in hives}
    clips = {(h, g): {k[2] for k in keys if k[:2] == (h, g)} for h in hives for g in groups[h]}
    count = {k[:3]: sum(v[:3] == k[:3] for v in keys) for k in keys}
    weights = np.array([1/(len(hives)*len(groups[h])*len(clips[h, g])*count[h, g, c]) for h, g, c, key in keys])
    if abs(weights.sum()-1) > 1e-12:
        raise RuntimeError("source weights do not sum to1")
    return weights


def fit(records, heldout=None):
    """Weighted ridge, unpenalized intercept; empty/constant sources explicit."""
    records = sorted(list(records), key=_record_key)
    if heldout is not None and heldout not in DOMAINS:
        raise ValueError("unknown heldout domain")
    if any(_record_key(r)[0] == heldout for r in records):
        raise ValueError("heldout labels/features entered source fit")
    G, T = _feature_rows(records)
    y = np.array([r["y"] for r in records], float)
    if not np.isfinite(y).all() or np.any(np.abs(y) > 1):
        raise ValueError("singleton source labels must be finite in[-1,1]")
    for r, label in zip(records, y):
        if "improvement_numerator" in r and r["improvement_numerator"]/30 != label:
            raise ValueError("source y differs from raw integer improvement/30")
    weights = source_weights(records)
    heads = {}
    for name, X, names in (("G", G, G_NAMES), ("T", T, T_NAMES)):
        dimension = X.shape[1]
        mean, scale, beta, intercept = np.zeros(dimension), np.ones(dimension), np.zeros(dimension), 0.
        status, objective, gradient_size = "empty", 0., 0.
        if len(records):
            mean = weights @ X
            constant_columns = np.ptp(X, axis=0) == 0
            mean[constant_columns] = X[0, constant_columns]
            scale = np.sqrt(weights @ ((X-mean)**2))
            scale[(scale == 0) | constant_columns] = 1
            Z = (X-mean)/scale
            if np.ptp(y) == 0:
                status, intercept = "constant", float(y[0])
            else:
                status = "ridge"
                design = np.column_stack([np.ones(len(Z)), Z])
                penalty = np.diag(np.r_[0., np.full(dimension, RIDGE)])
                coefficient = np.linalg.solve(design.T @ (weights[:, None]*design)+penalty,
                                              design.T @ (weights*y))
                intercept, beta = float(coefficient[0]), coefficient[1:]
            error = intercept+Z @ beta-y
            objective = float(weights @ (error**2)+RIDGE*(beta @ beta))
            gradient = np.r_[2*(weights @ error), 2*(Z.T @ (weights*error)+RIDGE*beta)]
            gradient_size = float(np.max(np.abs(gradient), initial=0))
            if gradient_size > 1e-8 or not np.isfinite(beta).all():
                raise RuntimeError("weighted ridge/intercept stationarity failed")
        heads[name] = dict(feature_names=names, mean=mean.tolist(), scale=scale.tolist(), beta=beta.tolist(),
                           intercept=intercept, status=status, ridge=RIDGE, objective=objective, max_abs_gradient=gradient_size)
    audit = [dict(key=_record_key(r)[3], hive=_record_key(r)[0], group_id=r.get("group_id", r.get("group")),
                  clip=_record_key(r)[2], y=float(label), improvement_numerator=r.get("improvement_numerator"),
                  weight=float(weight), features_G9=gg.tolist(), features_T12=tt.tolist())
             for r, label, weight, gg, tt in zip(records, y, weights, G, T)]
    return dict(schema=SCHEMA, heldout=heldout, heads=heads, threshold=THRESHOLD,
                source_hives=sorted({_record_key(r)[0] for r in records}), training_records=audit,
                weighting="nonempty hive -> nonempty seed group -> nonempty clip -> all candidates equal")


def predict(model, features_G9, features_T12):
    """Numeric blind deployment only: strictly score>.10 cuts; no GT/IDs."""
    if model.get("schema") != SCHEMA:
        raise ValueError("unknown endpoint model schema")
    G, T = _features(features_G9, features_T12)
    result = {}
    for name, X in (("G", G), ("T", T)):
        head = model["heads"][name]
        mean, scale, beta = (np.array(head[k], float) for k in ("mean", "scale", "beta"))
        if any(v.shape != (X.shape[1],) or not np.isfinite(v).all() for v in (mean, scale, beta)) or np.any(scale <= 0) or not np.isfinite(head["intercept"]):
            raise ValueError("invalid frozen standardization/coefficients")
        score = head["intercept"]+((X-mean)/scale) @ beta
        result[name] = dict(scores=score.tolist(), cut=(score > THRESHOLD).tolist())
    return result


def matched_null(records, t_mask):
    """GT-free M: per member/video match T counts within(delta,direction)."""
    if len(records) != len(t_mask):
        raise ValueError("one T decision per candidate required")
    selected = [False]*len(records)
    strata = {}
    hashes = []
    parents = set()
    for i, r in enumerate(records):
        member = r["member_index"]
        parent = r.get("parent_canonical_id", r.get("parent_id"))
        if isinstance(member, bool) or int(member) != member or isinstance(parent, bool) or int(parent) != parent:
            raise ValueError("integral canonical parent/member_index required")
        parent = int(parent); member = int(member)
        identity = (r["unit"], member, _record_key(r)[2], parent)
        if identity in parents: raise ValueError("at most one candidate per canonical parent")
        parents.add(identity)
        if r["delta"] not in (-1, 1) or r["direction"] not in (-1, 1):
            raise ValueError("single signed directional event required")
        digest = hashlib.sha256(f"endpoint-null-v1|{r['unit']}|{member}|{parent}".encode("utf-8")).hexdigest()
        hashes.append(digest)
        key = (r["unit"], member, _record_key(r)[2], int(r["delta"]), int(r["direction"]))
        strata.setdefault(key, []).append(i)
    audit = []
    for key, indices in sorted(strata.items()):
        count = sum(bool(t_mask[i]) for i in indices)
        ordered = sorted(indices, key=lambda i: (hashes[i], int(records[i].get("parent_canonical_id", records[i].get("parent_id")))))
        for i in ordered[:count]: selected[i] = True
        audit.append(dict(unit=key[0], member_index=key[1], clip=key[2], delta=key[3], direction=key[4],
                          available=len(indices), T_count=count, M_count=sum(selected[i] for i in indices)))
    return dict(cut=selected, hashes=hashes, strata=audit)


def deploy_records(records, model, member_skeleton=None):
    records = sorted(records, key=lambda r: _record_key(r)[3])
    if any(_record_key(r)[0] != model["heldout"] for r in records):
        raise ValueError("deploy_records must contain the heldout domain only")
    G, T = _feature_rows(records)
    numerical = predict(model, G, T)
    masks = dict(K=[False]*len(records), A=[True]*len(records), H=[bool(r["H"]) for r in records],
                 G=numerical["G"]["cut"], T=numerical["T"]["cut"])
    null = matched_null(records, masks["T"])
    masks["M"] = null["cut"]
    grouped = {}
    skeleton = {}
    for item in member_skeleton or []:
        if item["hive"] == model["heldout"]:
            key = (item["unit"], int(item["member_index"]), item["clip"])
            if key in skeleton: raise ValueError("duplicate frozen member-clip")
            skeleton[key] = item
            grouped[key] = []
    for i, r in enumerate(records):
        key = (r["unit"], int(r["member_index"]), _record_key(r)[2])
        grouped.setdefault(key, []).append(i)
    outputs = []
    for (unit, member, clip), indices in sorted(grouped.items()):
        base = skeleton.get((unit, member, clip), records[indices[0]] if indices else {})
        metadata = [{k: _plain(v) for k, v in records[i].items() if k not in ("y", "improvement_numerator", "features_G9", "features_T12", "features_G", "features_T")} for i in indices]
        outputs.append(dict(unit=unit, member_index=member, member=str(base.get("member", member)), clip=clip,
            group_id=base.get("group_id", base.get("group")), hive=model["heldout"], candidates=metadata,
            masks={arm: [masks[arm][i] for i in indices] for arm in ARMS},
            selected={arm: [_record_key(records[i])[3] for i in indices if masks[arm][i]] for arm in ARMS},
            scores={arm: [numerical[arm]["scores"][i] for i in indices] for arm in ("G", "T")},
            M_hashes=[null["hashes"][i] for i in indices]))
    return dict(heldout=model["heldout"], member_clips=outputs, M_strata=null["strata"],
                candidates=len(records), cut_counts={arm: sum(mask) for arm, mask in masks.items()})


def train_and_seal(records, label_provider, output_dir, input_receipt, member_skeleton=None):
    """provider(heldout) returns labels for exactly the other3 domains only."""
    records = list(records)
    for record in records:
        if "y" in record or "improvement_numerator" in record:
            raise ValueError("deployment records must not contain source/target labels")
    if len({_record_key(r)[3] for r in records}) != len(records):
        raise ValueError("duplicate frozen candidate key")
    root = Path(output_dir)
    if root.exists(): raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".endpoint-masks-", dir=root.parent) as tmp:
        work = Path(tmp)/"payload"; work.mkdir()
        for target in DOMAINS:
            source = [r for r in records if _record_key(r)[0] != target]
            supplied = label_provider(target)
            if set(supplied["source_hives"]) != set(DOMAINS)-{target} or set(supplied["labels"]) != {_record_key(r)[3] for r in source}:
                raise ValueError("strict other-three source label whitelist mismatch")
            training = []
            for record in source:
                label = supplied["labels"][_record_key(record)[3]]
                if not isinstance(label, dict): label = dict(y=label)
                training.append(dict(record, **label))
            fitted = fit(training, heldout=target)
            fitted["source_receipts"] = supplied.get("receipts", {})
            fitted["source_domains"] = sorted(supplied["source_hives"])
            deployed = deploy_records([r for r in records if _record_key(r)[0] == target], fitted, member_skeleton)
            _save(work/f"{target}.model.json", fitted)
            _save(work/f"{target}.masks.json", deployed)
        _save(work/"receipt.json", dict(input_receipt=input_receipt,
                                      prereg=input_receipt.get("prereg") if isinstance(input_receipt, dict) else "synthetic",
                                      freeze_receipt=input_receipt.get("freeze_receipt") if isinstance(input_receipt, dict) else "synthetic",
                                      model_code_sha256=_sha(__file__),
                                      schema="endpoint-cut-all-fold-masks-v1", arms=ARMS))
        manifest = {p.name: _sha(p) for p in sorted(work.iterdir())}
        _save(work/"manifest.json", manifest)
        (work/"SEALED.sha256").write_text(_sha(work/"manifest.json")+"\n")
        verify_seal(work)
        work.rename(root)
    return dict(manifest_sha256=_sha(root/"manifest.json"), candidates=len(records), folds=list(DOMAINS))


def verify_seal(root):
    root = Path(root)
    if (root/"SEALED.sha256").read_text().strip() != _sha(root/"manifest.json"):
        raise ValueError("four-fold mask seal mismatch")
    manifest = json.loads((root/"manifest.json").read_text())
    expected = {f"{target}.{kind}.json" for target in DOMAINS for kind in ("model", "masks")}|{"receipt.json"}
    if set(manifest) != expected:
        raise ValueError("all four models and masks must be sealed together")
    for name, digest in manifest.items():
        if _sha(root/name) != digest: raise ValueError("frozen model/mask changed")
    receipt = json.loads((root/"receipt.json").read_text())
    if receipt["model_code_sha256"] != _sha(__file__): raise ValueError("model code changed after seal")
    return receipt


def load_predictions(root):
    """Verify all four folds before returning scorer-ready complete masks."""
    receipt = verify_seal(root)
    folds, decisions = {}, {}
    for target in DOMAINS:
        fitted = json.loads((Path(root)/f"{target}.model.json").read_text())
        if target in fitted["source_hives"] or set(fitted["source_domains"]) != set(DOMAINS)-{target}:
            raise ValueError("source-only model boundary differs")
        fold = json.loads((Path(root)/f"{target}.masks.json").read_text())
        if fold["heldout"] != target: raise ValueError("wrong heldout mask fold")
        folds[target] = fold
        for item in fold["member_clips"]:
            members = decisions.setdefault(item["unit"], {})
            member = str(item["member"])
            if member in members: raise ValueError("duplicate member deployment")
            members[member] = item["selected"]
    return dict(receipt=receipt, folds=folds, decisions=decisions)


def flatten_groups(groups):
    records, members = [], []
    for group in groups:
        for member in group["members"]:
            base = dict(unit=group["unit"], hive=group["domain"], domain=group["domain"],
                        group_id=group["group"], clip=group["video"], member=str(member["member"]),
                        member_index=int(member["member_index"]))
            members.append(base)
            for candidate in member["candidates"]:
                record = dict(candidate, **base)
                key = candidate.get("key", candidate["candidate_id"])
                expected = f"{group['unit']}|{base['member_index']}|{int(candidate['parent_canonical_id'])}"
                if key != expected or candidate["candidate_id"] != expected:
                    raise ValueError("global candidate/member namespace differs from frozen protocol")
                record["key"] = key
                records.append(record)
    return records, members


def field_train(args):
    """Published-protocol gate, frozen inputs, explicit other-three GT loader."""
    if not args.prereg or not re.fullmatch("[0-9a-f]{40}", args.prereg):
        raise ValueError("full published preregistration SHA required")
    sys.path.insert(0, str(args.repo.resolve()))
    import freeze
    import oracle
    from tools.study195.run import _gt_provider
    from tools.study193.source193 import event_descriptor
    frozen = args.frozen or args.root/"frozen"
    groups = freeze.load_groups(frozen)
    payload = json.loads((frozen/"deployment.json").read_text())
    if payload["prereg"] != args.prereg or payload["marker"] != "[endpoint-cut-v1]":
        raise ValueError("frozen candidates use a different preregistration")
    if len(payload['groups'])!=20 or sum(len(g['members']) for g in payload['groups'])!=60 or payload['authorized_arms']!=60:
        raise ValueError('registered20group60member scope required')
    allowed={'35':{'BEE24-35-head'},'33':{'BEE24-33-head'},'W':{'BEE24-10','BEE24-15'},'G':{'BEE24-24','BEE24-28'}}
    if any(g['video'] not in allowed.get(g['domain'],set()) or len(g['members'])!=3 for g in payload['groups']):
        raise ValueError('registered source clip/member whitelist required')
    receipt = freeze.verify(frozen)
    if getattr(args,"require_original_freeze",False) and receipt["manifest_sha256"] != FROZEN_SHA:
        raise ValueError("candidate freeze differs from authorized immutable input")
    records, skeleton = flatten_groups(groups)

    def provider(target):
        allowed = tuple(h for h in DOMAINS if h != target)
        source_groups = [g for g in groups if g["domain"] in allowed and any(m["candidates"] for m in g["members"])]
        arrays, proofs = _gt_provider(args.repo, source_groups, allowed, "source")
        labels = {}
        for group in source_groups:
            gt = event_descriptor(arrays[group["video"]], group["polygon"], group["nf"])["bins"]
            baseline = sum((np.asarray(m["baseline_windows"], np.int64) for m in group["members"]), np.zeros_like(gt))
            candidates = [c for member in group["members"] for c in member["candidates"]]
            for item in oracle.singleton_labels(baseline, gt, candidates, group["nf"]):
                key = item["candidate_id"]
                if key in labels: raise ValueError("source candidate key repeated")
                labels[key] = dict(y=item["y"], improvement_numerator=item["improvement_numerator"])
        for proof in proofs.values():
            if _sha(args.repo/proof["path"]) != proof["sha256"]:
                raise ValueError("source GT changed during label creation")
        print("source labels ready:", target, len(labels), flush=True)
        return dict(source_hives=allowed, labels=labels, receipts=proofs)

    bound = dict(prereg=args.prereg, freeze_receipt=receipt,
                 source195_seal=payload["source195_seal"],
                 code_receipts={Path(module.__file__).name: _sha(module.__file__) for module in (freeze, oracle)})
    output = args.out or args.root/"predictions"
    result = train_and_seal(records, provider, output, bound, skeleton)
    print(json.dumps(result, indent=2))


def self_test():
    rng = np.random.default_rng(19703)
    records = []
    for hive in DOMAINS:
        for group in (1, 2):
            for clip in ("a", "b") if group == 1 else ("a",):
                for i in range(1 if clip == "a" else 3):
                    key = f"{hive}-{group}-{clip}-{i}"
                    G = rng.normal(size=9); T = np.r_[G, rng.normal(size=3)]
                    records.append(dict(key=key, candidate_id=key, hive=hive, group_id=group, clip=hive+clip,
                        unit=f"{hive}_g{group}_{hive+clip}", member_index=0, parent_canonical_id=i,
                        features_G9=G.tolist(), features_T12=T.tolist(), H=i % 2 == 0,
                        delta=1 if i % 2 else -1, direction=1 if clip == "a" else -1,
                        y=float(rng.choice([-1, -.5, 0, .5, 1]))))
    source = [r for r in records if r["hive"] != "35"]
    fitted = fit(source, heldout="35")
    audit = fitted["training_records"]
    for hive in ("33", "W", "G"):
        assert abs(sum(r["weight"] for r in audit if r["hive"] == hive)-1/3) < 1e-15
        for group in (1, 2):
            assert abs(sum(r["weight"] for r in audit if r["hive"] == hive and r["group_id"] == group)-1/6) < 1e-15
    for head in fitted["heads"].values(): assert head["max_abs_gradient"] < 1e-8
    G, T = _feature_rows(records)
    before = predict(fitted, G, T)
    assert predict(json.loads(json.dumps(fitted)), G, T) == before
    empty = fit([], heldout="35")
    assert not any(predict(empty, G, T)["T"]["cut"])
    constant = fit([dict(r, y=.10) for r in source], heldout="35")
    assert constant["heads"]["T"]["intercept"] == .10 and not any(predict(constant, G, T)["T"]["cut"])
    positive = fit([dict(r, y=.10000001) for r in source], heldout="35")
    assert all(predict(positive, G, T)["T"]["cut"])
    smaller = fit([dict(r, y=r["y"]*.5) for r in source], heldout="35")
    shifted_labels = fit([dict(r, y=r["y"]*.5+.1) for r in source], heldout="35")
    for name in ("G", "T"):
        np.testing.assert_allclose(smaller["heads"][name]["beta"], shifted_labels["heads"][name]["beta"], atol=1e-13)
        assert abs(shifted_labels["heads"][name]["intercept"]-smaller["heads"][name]["intercept"]-.1) < 1e-13
    extreme_G = np.zeros((1, 9)); extreme_G[:, 7:] = [-99, 99]
    extreme_T = np.c_[extreme_G, [[-99, 99, 0]]]
    clipped_G, clipped_T = _features(extreme_G, extreme_T)
    np.testing.assert_array_equal(clipped_G[:, 7:], [[-4, 4]])
    np.testing.assert_array_equal(clipped_T[:, 9:], [[-4, 4, 0]])
    assert fit(reversed(source), heldout="35") == fitted
    try: fit(records, heldout="35")
    except ValueError: pass
    else: raise AssertionError("heldout labels accepted")
    clip_records = [dict(records[0], key=str(i), parent_canonical_id=i, candidate_id=str(i), delta=1 if i < 4 else -1,
                         direction=1 if i % 2 else -1) for i in range(8)]
    tmask = [True, False, True, False, False, True, True, False]
    null = matched_null(clip_records, tmask)
    assert all(s["M_count"] == s["T_count"] for s in null["strata"])
    assert sum(null["cut"]) == sum(tmask)
    assert matched_null(clip_records, [False]*8)["cut"] == [False]*8
    assert matched_null(clip_records, [True]*8)["cut"] == [True]*8
    deployment = [{k: v for k, v in r.items() if k != "y"} for r in records]

    def provider(target):
        allowed = set(DOMAINS)-{target}
        return dict(source_hives=sorted(allowed), labels={r["key"]: r["y"] for r in records if r["hive"] in allowed}, receipts={})

    with TemporaryDirectory(prefix="endpoint-model-") as tmp:
        root = Path(tmp)/"masks"
        train_and_seal(deployment, provider, root, "synthetic")
        verify_seal(root)
        assert set(load_predictions(root)["folds"]) == set(DOMAINS)
        assert len(json.loads((root/"manifest.json").read_text())) == 9
    return dict(seed=19703, dimensions=[9, 12], source_hierarchy="hive/group/clip/candidate equal",
                ridge_intercept="stationarity passed; intercept unpenalized", threshold="strict>.10",
                empty_source="keeps all", constant_point10="keeps all exactly", JSON_reload="exact",
                target_label_guard="passed", M="per member-clip signed direction counts exactly match T",
                all_four_seal="passed", scope="synthetic only; no field data opened")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-field", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--require-original-freeze", action="store_true")
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--prereg")
    args = parser.parse_args()
    if args.run_field:
        if args.repo is None or args.root is None: parser.error("--repo and --root required for field fit")
        field_train(args)
    else:
        print(json.dumps(self_test(), indent=2))
