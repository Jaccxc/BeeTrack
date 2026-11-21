"""Direct source-only identity ridge. Synthetic default; no capacity gate.

All three heads use the same eligible known D/S source cohort, never U. The
target's labels are not accepted by deployment. A strict p(D)>.5 cuts; an
empty known source returns neutral.5 with an explicit abstention flag.
"""
import os
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np

DOMAINS = ("35", "33", "W", "G")
DIMENSIONS = dict(N=11, B=15, I=22)
ARMS = ("K", "I198", "H_N", "H_B", "H_I", "M")
RIDGE, THRESHOLD = .20, .5
SCHEMA = "endpoint-identity-model199-v1"
PEER_FREEZE_SHA = "d53e70b757984aa42a7d2e8b6c6ea0f287bedfe564d3a6a1fde685dc2d85765f"
REFERENCE198_SHA = "9adafc2ecc6dff3ba85b674653bd3a55ce91183bf0917ce34bfea2e5c14c5843"
SOURCE197_SHA = "e7fd48078f8d0a32bb9018f2bac93dc4b5270d374d5bcd0a631fa97dd4d790e4"


def _plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [_plain(v) for v in x]
    return x


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n", encoding="utf-8")


def _key(r):
    hive, group, clip, key = r.get("hive", r.get("domain")), r.get("group_id", r.get("group")), r.get("clip", r.get("video")), r.get("key", r.get("candidate_id"))
    if hive not in DOMAINS or group is None or not isinstance(clip, str) or not isinstance(key, str) or not key:
        raise ValueError("candidate requires hive/group/clip/197globalkey")
    if r.get("candidate_id", key) != key: raise ValueError("candidate key aliases differ")
    return hive, json.dumps(_plain(group), sort_keys=True), clip, key


def _eligible(r):
    value = r["eligible"]
    if not isinstance(value, (bool, np.bool_)) or r.get("qualifies", value) != value:
        raise ValueError("one frozen198 eligibility flag required")
    return bool(value)


def _matrices(records):
    matrices = {name: np.asarray([r[f"features_{name}{dim}"] for r in records], float).reshape(-1, dim) for name, dim in DIMENSIONS.items()}
    if any(not np.isfinite(v).all() for v in matrices.values()): raise ValueError("known eligible source features must be finite")
    if not np.array_equal(matrices["N"], matrices["B"][:, :11]) or not np.array_equal(matrices["B"], matrices["I"][:, :15]):
        raise ValueError("N/B/I ablation prefixes differ")
    return matrices


def source_weights(records):
    if not records: return np.empty(0)
    if len({_key(r)[3] for r in records}) != len(records): raise ValueError("duplicate source candidates")
    if any(r["delta"] not in (-1, 1) for r in records): raise ValueError("add/drop class must be+1/-1")
    keys = [(*_key(r)[:3], int(r["delta"])) for r in records]
    hives = {k[0] for k in keys}
    groups = {h: {k[1] for k in keys if k[0] == h} for h in hives}
    clips = {(h, g): {k[2] for k in keys if k[:2] == (h, g)} for h in hives for g in groups[h]}
    classes = {k[:3]: {j[3] for j in keys if j[:3] == k[:3]} for k in keys}
    counts = {k: keys.count(k) for k in keys}
    weights = np.array([1/(len(hives)*len(groups[h])*len(clips[h, g])*len(classes[h, g, c])*counts[h, g, c, d]) for h, g, c, d in keys])
    if abs(weights.sum()-1) > 1e-12: raise RuntimeError("source weights do not sum to1")
    return weights


def fit(records, heldout=None):
    """Unpenalized-intercept weighted ridge; source eligible D/S only."""
    records = sorted(records, key=_key)
    if heldout is not None and heldout not in DOMAINS: raise ValueError("unknown heldout domain")
    if any(_key(r)[0] == heldout for r in records): raise ValueError("heldout labels entered source fit")
    if any(r["identity_class"] not in ("D", "S", "U") for r in records): raise ValueError("source label must be D/S/U")
    eligible = [r for r in records if _eligible(r)]
    known = [r for r in eligible if r["identity_class"] in ("D", "S")]
    Xs = _matrices(known)
    y = np.array([int(r["identity_class"] == "D") for r in known], float)
    weights = source_weights(known)
    heads = {}
    for name, X in Xs.items():
        dim = DIMENSIONS[name]
        mean, scale, beta, intercept = np.zeros(dim), np.ones(dim), np.zeros(dim), .5
        status, objective, gradient_size = "empty", 0., 0.
        if known:
            mean = weights @ X
            constant = np.ptp(X, axis=0) == 0
            mean[constant] = X[0, constant]
            scale = np.sqrt(weights @ ((X-mean)**2)); scale[(scale == 0) | constant] = 1
            Z = (X-mean)/scale
            if np.ptp(y) == 0:
                status, intercept = "constant", float(y[0])
            else:
                status = "ridge"
                design = np.column_stack([np.ones(len(Z)), Z])
                penalty = np.diag(np.r_[0., np.full(dim, RIDGE)])
                coefficients = np.linalg.solve(design.T @ (weights[:, None]*design)+penalty, design.T @ (weights*y))
                intercept, beta = float(coefficients[0]), coefficients[1:]
            error = intercept+Z @ beta-y
            gradient = np.r_[2*weights @ error, 2*(Z.T @ (weights*error)+RIDGE*beta)]
            objective = float(weights @ (error**2)+RIDGE*(beta @ beta))
            gradient_size = float(np.max(np.abs(gradient), initial=0))
            if gradient_size > 1e-8: raise RuntimeError("identity ridge stationarity failed")
        heads[name] = dict(mean=mean.tolist(), scale=scale.tolist(), beta=beta.tolist(), intercept=intercept,
                           status=status, ridge=RIDGE, objective=objective, max_abs_gradient=gradient_size)
    audit = []
    for i, r in enumerate(known):
        item = dict(key=_key(r)[3], hive=_key(r)[0], group_id=r.get("group_id", r.get("group")), clip=_key(r)[2],
                    delta=int(r["delta"]), identity_class=r["identity_class"], y=int(y[i]), weight=float(weights[i]))
        for name, dim in DIMENSIONS.items(): item[f"features_{name}{dim}"] = Xs[name][i].tolist()
        audit.append(item)
    return dict(schema=SCHEMA, heldout=heldout, heads=heads, source_hives=sorted({_key(r)[0] for r in known}),
        source_rows=len(records), eligible_source_rows=len(eligible), known_source_rows=len(known),
        unknown_eligible_source_rows=sum(r["identity_class"] == "U" for r in eligible), training_records=audit,
        threshold=THRESHOLD, weighting="known eligible hive/group/clip/add-drop/candidate equal; never DS class balancing",
        empty_prediction="neutral pD.5 with abstain; no cuts", outputs="D1,S0;U never fitted")


def predict(model, features_N11, features_B15, features_I22):
    if model.get("schema") != SCHEMA: raise ValueError("wrong direct identity model")
    Xs = {name: np.asarray(v, float).reshape(-1, DIMENSIONS[name]) for name, v in zip(DIMENSIONS, (features_N11, features_B15, features_I22))}
    if not np.array_equal(Xs["N"], Xs["B"][:, :11]) or not np.array_equal(Xs["B"], Xs["I"][:, :15]): raise ValueError("feature prefixes differ")
    result = {}
    for name, X in Xs.items():
        head = model["heads"][name]
        mean, scale, beta = (np.asarray(head[k], float) for k in ("mean", "scale", "beta"))
        if (not np.isfinite(X).all() or any(v.shape != (DIMENSIONS[name],) or not np.isfinite(v).all() for v in (mean, scale, beta))
                or np.any(scale <= 0) or not np.isfinite(head["intercept"])): raise ValueError("invalid features or frozen coefficients")
        raw = head["intercept"]+((X-mean)/scale) @ beta
        probability = np.clip(raw, 0, 1)
        abstain = np.full(len(X), head["status"] == "empty")
        result["H_"+name] = dict(raw=raw.tolist(), pD=probability.tolist(), abstain=abstain.tolist(),
                                cut=((probability > THRESHOLD) & ~abstain).tolist())
    return result


def matched_null(records, mask):
    if len(mask) != len(records): raise ValueError("one H_I decision per candidate required")
    chosen, hashes, strata = [False]*len(records), [], {}
    for i, r in enumerate(records):
        hashes.append(hashlib.sha256(("endpoint-identity-null-v1|"+_key(r)[3]).encode("utf-8")).hexdigest())
        if not _eligible(r):
            if mask[i]: raise ValueError("H_I cut outside frozen support")
            continue
        bucket = (r["unit"], int(r["member_index"]), _key(r)[2], int(r["delta"]), int(r["direction"]))
        strata.setdefault(bucket, []).append(i)
    audit = []
    for bucket, indices in sorted(strata.items()):
        count = sum(bool(mask[i]) for i in indices)
        ordered = sorted(indices, key=lambda i: (hashes[i], _key(records[i])[3]))
        for i in ordered[:count]: chosen[i] = True
        audit.append(dict(unit=bucket[0], member_index=bucket[1], clip=bucket[2], delta=bucket[3], direction=bucket[4],
                          available=len(indices), H_I_count=count, M_count=sum(chosen[i] for i in indices)))
    return dict(cut=chosen, hashes=hashes, strata=audit)


def deploy_records(records, model, member_skeleton=None):
    records = sorted(records, key=lambda r: _key(r)[3])
    if any(_key(r)[0] != model["heldout"] for r in records): raise ValueError("wrong heldout candidates")
    eligible = [_eligible(r) for r in records]
    X = _matrices([r for r in records if _eligible(r)])
    output = predict(model, X["N"], X["B"], X["I"])
    masks = {arm: [False]*len(records) for arm in ARMS}
    scores = {arm: [None]*len(records) for arm in ("H_N", "H_B", "H_I")}
    j = 0
    for i, r in enumerate(records):
        old = r["I198_cut"]
        if not isinstance(old, (bool, np.bool_)) or old and not eligible[i]: raise ValueError("198I reference/support differs")
        masks["I198"][i] = bool(old)
        if eligible[i]:
            for arm in scores:
                masks[arm][i] = output[arm]["cut"][j]
                scores[arm][i] = {field: output[arm][field][j] for field in ("raw", "pD", "abstain")}
            j += 1
    null = matched_null(records, masks["H_I"]); masks["M"] = null["cut"]
    groups, bases = {}, {}
    for base in member_skeleton or []:
        if base["hive"] == model["heldout"]:
            key = (base["unit"], int(base["member_index"]), base["clip"]); groups[key] = []; bases[key] = base
    for i, r in enumerate(records):
        key = (r["unit"], int(r["member_index"]), _key(r)[2]); groups.setdefault(key, []).append(i); bases.setdefault(key, r)
    members = []
    for (unit, slot, clip), indices in sorted(groups.items()):
        members.append(dict(unit=unit, member_index=slot, member=str(bases[unit, slot, clip]["member"]),
            clip=clip, hive=model["heldout"], candidate_ids=[_key(records[i])[3] for i in indices], eligible=[eligible[i] for i in indices],
            masks={arm: [masks[arm][i] for i in indices] for arm in ARMS},
            selected={arm: [_key(records[i])[3] for i in indices if masks[arm][i]] for arm in ARMS},
            scores={arm: [scores[arm][i] for i in indices] for arm in scores}, M_hashes=[null["hashes"][i] for i in indices]))
    return dict(heldout=model["heldout"], member_clips=members, candidates=len(records), eligible=sum(eligible),
                M_strata=null["strata"], cut_counts={arm: sum(values) for arm, values in masks.items()})


def train_and_seal(records, label_provider, output_dir, input_receipt, member_skeleton=None):
    """Always fit four folds. No capacity/strict-oracle decision is accepted."""
    records = list(records)
    if any("identity_class" in r or "labels" in r for r in records): raise ValueError("deployment records must be unlabelled")
    if len({_key(r)[3] for r in records}) != len(records): raise ValueError("duplicate frozen candidates")
    root = Path(output_dir)
    if root.exists(): raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".identity199-", dir=root.parent) as tmp:
        work = Path(tmp)/"payload"; work.mkdir()
        for target in DOMAINS:
            source = [r for r in records if _key(r)[0] != target and _eligible(r)]
            supplied = label_provider(target)
            if set(supplied["source_hives"]) != set(DOMAINS)-{target} or set(supplied["labels"]) != {_key(r)[3] for r in source}:
                raise ValueError("source labels must name eligible candidates from exactly the other3 domains")
            training = [dict(r, identity_class=supplied["labels"][_key(r)[3]]) for r in source]
            fitted = fit(training, heldout=target)
            fitted["source_domains"] = sorted(supplied["source_hives"])
            fitted["source_receipts"] = supplied.get("receipts", {})
            deployed = deploy_records([r for r in records if _key(r)[0] == target], fitted, member_skeleton)
            _save(work/f"{target}.model.json", fitted); _save(work/f"{target}.masks.json", deployed)
        _save(work/"receipt.json", dict(schema="endpoint-identity-masks199-v1", input_receipt=input_receipt,
            prereg=input_receipt.get("prereg") if isinstance(input_receipt, dict) else "synthetic",
            peerfreeze_receipt=input_receipt.get("peerfreeze_receipt") if isinstance(input_receipt, dict) else "synthetic",
            owners_receipt=input_receipt.get("owners_receipt") if isinstance(input_receipt, dict) else "synthetic",
            reference198_masks_sha256=input_receipt.get("reference198_masks_sha256") if isinstance(input_receipt, dict) else "synthetic",
            model_code_sha256=_sha(__file__), arms=ARMS, capacity_gate="none; always trained"))
        _save(work/"manifest.json", {p.name: _sha(p) for p in sorted(work.iterdir())})
        (work/"SEALED.sha256").write_text(_sha(work/"manifest.json")+"\n")
        verify_seal(work); work.rename(root)
    return dict(manifest_sha256=_sha(root/"manifest.json"), candidates=len(records), folds=list(DOMAINS))


def verify_seal(root):
    root = Path(root)
    if (root/"SEALED.sha256").read_text().strip() != _sha(root/"manifest.json"): raise ValueError("identity prediction seal differs")
    manifest = json.loads((root/"manifest.json").read_text())
    expected = {f"{h}.{kind}.json" for h in DOMAINS for kind in ("model", "masks")}|{"receipt.json"}
    if set(manifest) != expected: raise ValueError("four models/all masks must be sealed together")
    for name, digest in manifest.items():
        if _sha(root/name) != digest: raise ValueError("sealed prediction changed")
    receipt = json.loads((root/"receipt.json").read_text())
    if receipt["model_code_sha256"] != _sha(__file__): raise ValueError("model code changed after seal")
    return receipt


def load_predictions(root):
    receipt = verify_seal(root)
    folds, decisions = {}, {}
    for target in DOMAINS:
        fitted = json.loads((Path(root)/f"{target}.model.json").read_text())
        if target in fitted["source_hives"] or set(fitted["source_domains"]) != set(DOMAINS)-{target}: raise ValueError("source-only fold boundary differs")
        fold = json.loads((Path(root)/f"{target}.masks.json").read_text())
        if fold["heldout"] != target: raise ValueError("wrong heldout masks")
        folds[target] = fold
        for member in fold["member_clips"]:
            current = decisions.setdefault(member["unit"], {})
            if member["member"] in current: raise ValueError("duplicate member output")
            current[member["member"]] = member["selected"]
    return dict(receipt=receipt, folds=folds, decisions=decisions)


def load_source_labels(owners_root, heldout, records, *, source_domains, domain_loader, prereg, manifest_receipt):
    """Validate other-three scope BEFORE any domain-label payload is opened."""
    allowed = tuple(source_domains)
    if heldout not in DOMAINS or len(allowed) != 3 or set(allowed) != set(DOMAINS)-{heldout}:
        raise ValueError("explicit other-three owner source whitelist required")
    expected_all = {h: {_key(r)[3] for r in records if _key(r)[0] == h} for h in allowed}
    eligible = {_key(r)[3] for r in records if _key(r)[0] in allowed and _eligible(r)}
    labels, receipts = {}, {}
    for hive in allowed:
        data = domain_loader(owners_root, hive)
        if data["domain"] != hive or data["prereg"] != prereg or data["source197_seal"] != SOURCE197_SHA:
            raise ValueError("source owner domain/registration/inventory differs")
        found = set()
        for unit in data["units"]:
            if unit["domain"] != hive: raise ValueError("owner shard has another domain")
            for member in unit["members"]:
                for row in member["records"]:
                    key = row["candidate_id"]
                    if key in found: raise ValueError("duplicate owner candidate")
                    found.add(key)
                    if key in eligible:
                        label = row["labels"]["0.5"]["class"]
                        if label not in ("D", "S", "U"): raise ValueError("invalid isolated .5 class")
                        labels[key] = label
        if found != expected_all[hive]: raise ValueError("source owner/feature candidate inventory differs")
        receipts[hive] = dict(file=hive+".json", sha256=manifest_receipt["files"][hive+".json"],
                              candidates=len(found), source197_receipt=data.get("source197_receipt"))
    if set(labels) != eligible: raise ValueError("source eligible owner labels incomplete")
    return dict(source_hives=allowed, labels=labels,
                receipts=dict(owner_manifest_sha256=manifest_receipt["manifest_sha256"], source_shards=receipts,
                              label_threshold="0.5", target_label_access="heldout shard never opened"))


def _manifest(root):
    root = Path(root)
    digest = _sha(root/"manifest.json")
    if (root/"SEALED.sha256").read_text().strip() != digest: raise ValueError("input manifest seal differs")
    return json.loads((root/"manifest.json").read_text()), digest


def field_train(args):
    """Source owner shards only. Always train; no capacity artifact is read."""
    if not args.prereg or not re.fullmatch("[0-9a-f]{40}", args.prereg): raise ValueError("full published199 preregistration required")
    if not args.owners_sha or not re.fullmatch("[0-9a-f]{64}", args.owners_sha): raise ValueError("pinned owners manifest SHA required")
    sys.path.insert(0, str(args.repo.resolve()))
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location("identity199_source_owners", here/"owners199.py")
    owners = importlib.util.module_from_spec(spec); sys.modules[spec.name] = owners; spec.loader.exec_module(owners)
    files, owner_sha = _manifest(args.owners)
    if owner_sha != args.owners_sha: raise ValueError("owners manifest differs from authorized input")
    # Do not open summary.json or any domain-label shard at this stage.
    owner_receipt = dict(manifest_sha256=owner_sha, files=files)
    feature_root = args.peer198/"frozen"
    peer_files, peer_sha = _manifest(feature_root)
    if peer_sha != PEER_FREEZE_SHA: raise ValueError("fixed198 feature support differs")
    if _sha(feature_root/"deployment.json") != peer_files["deployment.json"]: raise ValueError("unlabelled feature file changed")
    feature_data = json.loads((feature_root/"deployment.json").read_text())
    if feature_data["schema"] != "endpoint-peer-freeze-v1" or feature_data["source197_seal"] != SOURCE197_SHA:
        raise ValueError("unexpected frozen feature family")
    records = [r for group in feature_data["groups"] for member in group["members"] for r in member["records"]]
    if len(records) != 1196 or sum(_eligible(r) for r in records) != 1076: raise ValueError("frozen1196/eligible1076 support changed")
    old_root = args.peer198/"predictions"
    old_files, old_sha = _manifest(old_root)
    if old_sha != REFERENCE198_SHA: raise ValueError("fixed198 I reference differs")
    old_bits = {}
    for hive in DOMAINS:
        name = hive+".masks.json"
        if _sha(old_root/name) != old_files[name]: raise ValueError("198 unlabelled mask changed")
        data = json.loads((old_root/name).read_text())
        for member in data["member_clips"]:
            selected = set(member["selected"]["I"])
            for key in member["candidate_ids"]:
                if key in old_bits: raise ValueError("duplicate198 candidate")
                old_bits[key] = key in selected
    if set(old_bits) != {_key(r)[3] for r in records}: raise ValueError("feature/reference inventory differs")
    records = [dict(r, I198_cut=old_bits[_key(r)[3]]) for r in records]
    skeleton = [dict(unit=g["unit"], hive=g["domain"], clip=g["video"], member=str(m["member"]), member_index=int(m["member_index"]))
                for g in feature_data["groups"] for m in g["members"]]
    def provider(target):
        result = load_source_labels(args.owners, target, records,
            source_domains=tuple(h for h in DOMAINS if h != target), domain_loader=owners.load_domain,
            prereg=args.prereg, manifest_receipt=owner_receipt)
        known = sum(label in ("D", "S") for label in result["labels"].values())
        print("source identity labels:", target, "eligible", len(result["labels"]), "known DS", known, flush=True)
        return result
    receipt = dict(prereg=args.prereg, peerfreeze_receipt=dict(manifest_sha256=peer_sha, deployment_sha256=peer_files["deployment.json"]),
        owners_receipt=dict(manifest_sha256=owner_sha), reference198_masks_sha256=old_sha,
        source197_seal=SOURCE197_SHA, labels_code_sha256=_sha(owners.__file__))
    result = train_and_seal(records, provider, args.out or args.root/"predictions", receipt, skeleton)
    print(json.dumps(result, indent=2))


def self_test():
    rng = np.random.default_rng(19903)
    records = []
    for hive in DOMAINS:
        for group in (1, 2):
            for clip in ("a", "b") if group == 1 else ("a",):
                for i, (label, delta) in enumerate((("D", 1), ("S", 1), ("U", 1), ("D", -1), ("S", -1), ("S", 1), ("D", -1))):
                    N = rng.normal(size=11); B = np.r_[N, rng.normal(size=4)]; I = np.r_[B, rng.normal(size=7)]
                    key = f"{hive}-g{group}-{clip}|0|{i}"
                    records.append(dict(key=key, candidate_id=key, hive=hive, group_id=group, clip=hive+clip,
                        unit=f"{hive}_g{group}_{clip}", member_index=0, member=hive+"_m0", delta=delta, direction=1 if i % 2 else -1,
                        eligible=i != 6, I198_cut=i == 0, features_N11=N.tolist(), features_B15=B.tolist(), features_I22=I.tolist(), identity_class=label))
    source = [r for r in records if r["hive"] != "35"]
    fitted = fit(source, heldout="35")
    assert fitted["known_source_rows"] == 45 and fitted["unknown_eligible_source_rows"] == 9
    assert all(r["identity_class"] in ("D", "S") for r in fitted["training_records"])
    for hive in ("33", "W", "G"):
        assert abs(sum(r["weight"] for r in fitted["training_records"] if r["hive"] == hive)-1/3) < 1e-15
        for group in (1, 2):
            for clip in (hive+"a", hive+"b") if group == 1 else (hive+"a",):
                values = [sum(r["weight"] for r in fitted["training_records"] if r["hive"] == hive and r["group_id"] == group and r["clip"] == clip and r["delta"] == d) for d in (-1, 1)]
                np.testing.assert_allclose(values, [sum(values)/2]*2, atol=1e-15)
    assert abs(sum(r["weight"] for r in fitted["training_records"] if r["identity_class"] == "D")-5/12) < 1e-15
    eligible = [r for r in source if _eligible(r)]
    X = _matrices(eligible)
    prediction = predict(fitted, X["N"], X["B"], X["I"])
    assert predict(json.loads(json.dumps(fitted)), X["N"], X["B"], X["I"]) == prediction
    assert predict({k: fitted[k] for k in ("schema", "heads")}, X["N"], X["B"], X["I"]) == prediction
    unknown = fit([dict(r, identity_class="U") for r in source], heldout="35")
    empty = predict(unknown, X["N"], X["B"], X["I"])
    assert all(all(head["abstain"]) and not any(head["cut"]) and all(p == .5 for p in head["pD"]) for head in empty.values())
    for label, probability in (("D", 1.), ("S", 0.)):
        constant = fit([dict(r, identity_class=label) for r in source], heldout="35")
        out = predict(constant, X["N"], X["B"], X["I"])
        assert all(all(p == probability for p in head["pD"]) and not any(head["abstain"]) for head in out.values())
    manual = json.loads(json.dumps(fitted))
    for head in manual["heads"].values(): head.update(intercept=.5, beta=[0.]*len(head["beta"]))
    tie = predict(manual, X["N"], X["B"], X["I"])
    assert all(not any(h["cut"]) and not any(h["abstain"]) for h in tie.values())
    for head in manual["heads"].values(): head["intercept"] = 2
    assert all(all(p == 1 for p in h["pD"]) for h in predict(manual, X["N"], X["B"], X["I"]).values())
    corrupted = [dict(r) for r in source]
    for r in corrupted:
        if r["identity_class"] == "U":
            r.update(features_N11=[1e9]*11, features_B15=[1e9]*15, features_I22=[1e9]*22)
    assert fit(corrupted, heldout="35") == fitted
    assert fit(reversed(source), heldout="35") == fitted
    zero_features = []
    for r in source:
        zero_features.append(dict(r, features_N11=[0.]*11, features_B15=[0.]*15, features_I22=[0.]*22))
    zero_fit = fit(zero_features, heldout="35")
    for h in zero_fit["heads"].values():
        assert h["scale"] == [1.]*len(h["scale"]) and h["beta"] == [0.]*len(h["beta"])
        assert abs(h["intercept"]-5/12) < 1e-15
    target = [r for r in records if r["hive"] == "35"]
    deployed = deploy_records(target, fitted)
    assert all(s["H_I_count"] == s["M_count"] for s in deployed["M_strata"])
    for member in deployed["member_clips"]:
        for i, eligible in enumerate(member["eligible"]):
            if not eligible: assert all(not member["masks"][arm][i] for arm in ARMS)
    clean = [{k: v for k, v in r.items() if k != "identity_class"} for r in records]
    calls = []
    def provider(target):
        calls.append(target)
        return dict(source_hives=[h for h in DOMAINS if h != target], labels={_key(r)[3]: r["identity_class"] for r in records if r["hive"] != target and _eligible(r)})
    with TemporaryDirectory(prefix="identity199-") as tmp:
        root = Path(tmp)/"predictions"
        train_and_seal(clean, provider, root, dict(prereg="synthetic", prior_capacity_passed=False))
        assert calls == list(DOMAINS) and set(load_predictions(root)["folds"]) == set(DOMAINS)
    owner_data = {}
    for hive in DOMAINS:
        owner_data[hive] = dict(domain=hive, prereg="synthetic", source197_seal=SOURCE197_SHA,
            units=[dict(domain=hive, members=[dict(records=[dict(candidate_id=_key(r)[3], labels={"0.5": {"class": r["identity_class"]}})
                                                         for r in records if r["hive"] == hive])])])
    receipt = dict(manifest_sha256="synthetic", files={h+".json": "synthetic" for h in DOMAINS})
    for target in DOMAINS:
        opened = []
        def spy(root, domain):
            assert domain != target
            opened.append(domain)
            return owner_data[domain]
        loaded = load_source_labels("unused", target, clean, source_domains=tuple(h for h in DOMAINS if h != target),
            domain_loader=spy, prereg="synthetic", manifest_receipt=receipt)
        assert set(opened) == set(DOMAINS)-{target}
        assert set(loaded["labels"]) == {_key(r)[3] for r in clean if r["hive"] != target and _eligible(r)}
    opened = []
    try:
        load_source_labels("unused", "35", clean, source_domains=("35", "W", "G"), domain_loader=lambda *a: opened.append(a),
                           prereg="synthetic", manifest_receipt=receipt)
    except ValueError: pass
    else: raise AssertionError("heldout owner shard allowed")
    assert not opened
    try: fit(records, heldout="35")
    except ValueError: pass
    else: raise AssertionError("heldout source labels accepted")
    return dict(seed=19903, dimensions=DIMENSIONS, source_cohort="same eligible D/S for all heads; U inert",
                hierarchical_weights="known hive/group/clip/add-drop/candidate", ridge="unpenalized intercept;stationarity passed",
                constant_DS="exact0/1", empty_known="abstain at.5; no cuts", strict_threshold="nonempty.5 keeps S",
                clipped_probability="passed", M="eligible count matched H_I", four_fold_seal="always runs;no capacity gate",
                owner_shard_spy="exactlyother3; heldout rejected before any label payload",
                source_guard_reload_order="passed", scope="synthetic only; no field inputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-field", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--owners", type=Path)
    parser.add_argument("--owners-sha")
    parser.add_argument("--peer198", type=Path)
    parser.add_argument("--prereg")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.run_field:
        if args.repo is None: parser.error("--repo required for authorized field training")
        args.owners = args.owners or args.root/"owners"
        field_train(args)
    else:
        print(json.dumps(self_test(), indent=2))
