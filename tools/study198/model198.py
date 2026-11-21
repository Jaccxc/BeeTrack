"""Phase2 peer endpoint models. Default execution is synthetic only.

N11/B15/I22 share the SAME eligible sources. y is singleton count improvement,
z is the source event-ON minus event-OFF whole-member maximum-TP increment.
Raw ridge outputs are clipped only for deployment. A failed capacity gate
stops before the source label provider is ever invoked.
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
ARMS = ("K", "T197", "Tg", "N", "B", "I", "F", "Y", "M")
RIDGE, THRESHOLD = .2, .10
SCHEMA = "endpoint-peer-model198-v1"
PEER_FREEZE_SHA = "d53e70b757984aa42a7d2e8b6c6ea0f287bedfe564d3a6a1fde685dc2d85765f"
SOURCE197_MASK_SHA = "9a726ff7c097e0f2c595c63e2a50bd4b05390006439bd12c0c539eac4a6f752f"


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(v) for v in value]
    return value


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n", encoding="utf-8")


def _key(record):
    hive = record.get("hive", record.get("domain"))
    group = record.get("group_id", record.get("group"))
    clip = record.get("clip", record.get("video"))
    key = record.get("key", record.get("candidate_id"))
    if hive not in DOMAINS or group is None or not isinstance(clip, str) or not isinstance(key, str) or not key:
        raise ValueError("candidate requires hive/group/clip/global197key")
    return hive, json.dumps(_plain(group), sort_keys=True), clip, key


def _eligible(record):
    value = record["eligible"]
    if not isinstance(value, (bool, np.bool_)) or record.get("qualifies", value) != value:
        raise ValueError("one consistent observable/core-ID eligibility flag required")
    return bool(value)


def _matrices(records):
    result = {}
    for name, dimension in DIMENSIONS.items():
        rows = [record[f"features_{name}{dimension}"] for record in records]
        X = np.asarray(rows, float).reshape(-1, dimension)
        if not np.isfinite(X).all(): raise ValueError("eligible features must be finite")
        result[name] = X
    if not np.array_equal(result["N"], result["B"][:, :11]) or not np.array_equal(result["B"], result["I"][:, :15]):
        raise ValueError("nested ablation feature prefixes differ")
    return result


def source_weights(records):
    if not records: return np.empty(0)
    keys = [(*_key(r)[:3], int(r["delta"])) for r in records]
    if any(r["delta"] not in (-1, 1) for r in records): raise ValueError("add/drop delta must be+1/-1")
    if len({_key(r)[3] for r in records}) != len(records): raise ValueError("duplicate source candidate")
    hives = {k[0] for k in keys}
    groups = {h: {k[1] for k in keys if k[0] == h} for h in hives}
    clips = {(h, g): {k[2] for k in keys if k[:2] == (h, g)} for h in hives for g in groups[h]}
    classes = {k[:3]: {v[3] for v in keys if v[:3] == k[:3]} for k in keys}
    counts = {k: keys.count(k) for k in keys}
    weights = np.array([1/(len(hives)*len(groups[h])*len(clips[h, g])*len(classes[h, g, c])*counts[h, g, c, delta])
                        for h, g, c, delta in keys])
    if abs(weights.sum()-1) > 1e-12: raise RuntimeError("source weights do not sum to1")
    return weights


def fit(records, heldout=None):
    records = sorted(records, key=_key)
    if heldout is not None and heldout not in DOMAINS: raise ValueError("unknown heldout")
    if any(_key(r)[0] == heldout for r in records): raise ValueError("heldout labels entered source fit")
    selected = [r for r in records if _eligible(r)]
    Xs = _matrices(selected)
    labels = np.array([[r["y"], r["z"]] for r in selected], float).reshape(-1, 2)
    if (not np.isfinite(labels).all() or np.any(np.abs(labels[:, 0]) > 1)
            or np.any(~np.isin(labels[:, 1], (0, 1)))):
        raise ValueError("source y in[-1,1] and z in{0,1} required")
    for r in selected:
        if "improvement_numerator" in r and r["improvement_numerator"]/30 != r["y"]:
            raise ValueError("y differs from singleton raw gain/30")
        if "event_on_TP" in r and r["event_on_TP"]-r["event_off_TP"] != r["z"]:
            raise ValueError("z differs from event-ON/OFF maximum-TP increment")
    weights = source_weights(selected)
    heads = {}
    for name, X in Xs.items():
        dimension = DIMENSIONS[name]
        mean, scale = np.zeros(dimension), np.ones(dimension)
        beta, intercept = np.zeros((dimension, 2)), np.array([0., .5])
        status = ["empty", "empty"]
        objective, gradient_size = 0., 0.
        if len(selected):
            mean = weights @ X
            constant = np.ptp(X, axis=0) == 0
            mean[constant] = X[0, constant]
            scale = np.sqrt(weights @ ((X-mean)**2)); scale[(scale == 0) | constant] = 1
            Z = (X-mean)/scale
            design = np.column_stack([np.ones(len(Z)), Z])
            penalty = np.diag(np.r_[0., np.full(dimension, RIDGE)])
            normal = design.T @ (weights[:, None]*design)+penalty
            for output in (0, 1):
                if np.ptp(labels[:, output]) == 0:
                    status[output], intercept[output] = "constant", labels[0, output]
                else:
                    coefficients = np.linalg.solve(normal, design.T @ (weights*labels[:, output]))
                    status[output], intercept[output], beta[:, output] = "ridge", coefficients[0], coefficients[1:]
            error = intercept+Z @ beta-labels
            objective = float(np.sum(weights[:, None]*error**2)+RIDGE*np.sum(beta**2))
            gradient = np.vstack([2*weights @ error, 2*(Z.T @ (weights[:, None]*error)+RIDGE*beta)])
            gradient_size = float(np.max(np.abs(gradient), initial=0))
            if gradient_size > 1e-8: raise RuntimeError("multi-output ridge stationarity failed")
        heads[name] = dict(mean=mean.tolist(), scale=scale.tolist(), beta=beta.tolist(), intercept=intercept.tolist(),
                           output_status=status, outputs=["y", "z"], ridge=RIDGE, objective=objective,
                           max_abs_gradient=gradient_size)
    audit = []
    for i, r in enumerate(selected):
        row = dict(key=_key(r)[3], hive=_key(r)[0], group_id=r.get("group_id", r.get("group")),
                   clip=_key(r)[2], delta=int(r["delta"]), weight=float(weights[i]), y=float(r["y"]), z=int(r["z"]))
        for name, dimension in DIMENSIONS.items(): row[f"features_{name}{dimension}"] = Xs[name][i].tolist()
        for field in ("improvement_numerator", "event_on_TP", "event_off_TP"):
            if field in r: row[field] = r[field]
        audit.append(row)
    return dict(schema=SCHEMA, heldout=heldout, heads=heads, source_hives=sorted({_key(r)[0] for r in selected}),
                source_rows=len(records), eligible_source_rows=len(selected), training_records=audit,
                threshold=THRESHOLD, weighting="eligible hive/group/clip/add-drop class/candidate equal",
                empty_prediction="neutral y0,z.5, no learned cuts")


def predict(model, features_N11, features_B15, features_I22, delta):
    """Blind numeric prediction; caller passes eligible rows only."""
    if model.get("schema") != SCHEMA: raise ValueError("unknown peer model")
    Xs = {name: np.asarray(X, float).reshape(-1, DIMENSIONS[name])
          for name, X in zip(DIMENSIONS, (features_N11, features_B15, features_I22))}
    if not np.array_equal(Xs["N"], Xs["B"][:, :11]) or not np.array_equal(Xs["B"], Xs["I"][:, :15]):
        raise ValueError("deployment ablation prefixes differ")
    delta = np.asarray(delta)
    if delta.shape != (len(Xs["N"]),) or not np.isin(delta, (-1, 1)).all(): raise ValueError("one add/drop sign per row required")
    result = {}
    for name, X in Xs.items():
        if not np.isfinite(X).all(): raise ValueError("finite deployment features required")
        head = model["heads"][name]
        mean, scale, beta, intercept = (np.asarray(head[k], float) for k in ("mean", "scale", "beta", "intercept"))
        dim = DIMENSIONS[name]
        if (mean.shape != (dim,) or scale.shape != mean.shape or beta.shape != (dim, 2) or intercept.shape != (2,)
                or any(not np.isfinite(v).all() for v in (mean, scale, beta, intercept)) or np.any(scale <= 0)):
            raise ValueError("invalid frozen peer head")
        raw = intercept+((X-mean)/scale) @ beta
        y, z = np.clip(raw[:, 0], -1, 1), np.clip(raw[:, 1], 0, 1)
        identity = delta*(2*z-1)
        utility = (y+identity)/2
        result[name] = dict(raw=raw.tolist(), yhat=y.tolist(), zhat=z.tolist(),
                           identity_utility=identity.tolist(), utility=utility.tolist(), cut=(utility > THRESHOLD).tolist())
    result["F"] = dict(cut=(np.array(result["I"]["identity_utility"]) > THRESHOLD).tolist())
    result["Y"] = dict(cut=(np.array(result["I"]["yhat"]) > THRESHOLD).tolist())
    return result


def matched_null(records, i_mask):
    if len(records) != len(i_mask): raise ValueError("one I mask entry per candidate required")
    selected, hashes, strata = [False]*len(records), [], {}
    for i, r in enumerate(records):
        key = _key(r)[3]
        hashes.append(hashlib.sha256(("endpoint-peer-null-v1|"+key).encode("utf-8")).hexdigest())
        if not _eligible(r):
            if i_mask[i]: raise ValueError("I selected an ineligible candidate")
            continue
        bucket = (r["unit"], int(r["member_index"]), _key(r)[2], int(r["delta"]), int(r["direction"]))
        strata.setdefault(bucket, []).append(i)
    audit = []
    for bucket, indices in sorted(strata.items()):
        count = sum(bool(i_mask[i]) for i in indices)
        ordered = sorted(indices, key=lambda i: (hashes[i], _key(records[i])[3]))
        for i in ordered[:count]: selected[i] = True
        audit.append(dict(unit=bucket[0], member_index=bucket[1], clip=bucket[2], delta=bucket[3], direction=bucket[4],
                          available=len(indices), I_count=count, M_count=sum(selected[i] for i in indices)))
    return dict(cut=selected, hashes=hashes, strata=audit)


def deploy_records(records, model, member_skeleton=None):
    records = sorted(records, key=lambda r: _key(r)[3])
    if any(_key(r)[0] != model["heldout"] for r in records): raise ValueError("wrong target candidates")
    eligible = [_eligible(r) for r in records]
    subset = [r for r in records if _eligible(r)]
    X = _matrices(subset)
    numerical = predict(model, X["N"], X["B"], X["I"], [r["delta"] for r in subset])
    masks = {arm: [False]*len(records) for arm in ARMS}
    scores = {head: [None]*len(records) for head in DIMENSIONS}
    j = 0
    for i, r in enumerate(records):
        old = r["T197_cut"]
        if not isinstance(old, (bool, np.bool_)): raise ValueError("frozen T197 mask bit required")
        masks["T197"][i], masks["Tg"][i] = bool(old), bool(old and eligible[i])
        if eligible[i]:
            for name in DIMENSIONS:
                masks[name][i] = numerical[name]["cut"][j]
                scores[name][i] = {field: numerical[name][field][j] for field in ("raw", "yhat", "zhat", "identity_utility", "utility")}
            for name in ("F", "Y"): masks[name][i] = numerical[name]["cut"][j]
            j += 1
    null = matched_null(records, masks["I"]); masks["M"] = null["cut"]
    grouped, bases = {}, {}
    for base in member_skeleton or []:
        if base["hive"] == model["heldout"]:
            key = (base["unit"], int(base["member_index"]), base["clip"])
            grouped[key], bases[key] = [], base
    for i, r in enumerate(records):
        key = (r["unit"], int(r["member_index"]), _key(r)[2])
        grouped.setdefault(key, []).append(i); bases.setdefault(key, r)
    out = []
    for (unit, slot, clip), indices in sorted(grouped.items()):
        base = bases[unit, slot, clip]
        out.append(dict(unit=unit, member_index=slot, member=str(base["member"]), clip=clip, hive=model["heldout"],
            candidate_ids=[_key(records[i])[3] for i in indices], eligible=[eligible[i] for i in indices],
            masks={arm: [masks[arm][i] for i in indices] for arm in ARMS},
            selected={arm: [_key(records[i])[3] for i in indices if masks[arm][i]] for arm in ARMS},
            scores={name: [scores[name][i] for i in indices] for name in DIMENSIONS},
            M_hashes=[null["hashes"][i] for i in indices]))
    return dict(heldout=model["heldout"], member_clips=out, M_strata=null["strata"], candidates=len(records),
                eligible=sum(eligible), cut_counts={arm: sum(values) for arm, values in masks.items()})


def train_and_seal(records, label_provider, output_dir, input_receipt, *, phase1_passed, guard_receipt, member_skeleton=None):
    if type(phase1_passed) is not bool: raise ValueError("explicit boolean finalized phase1 decision required")
    root = Path(output_dir)
    if root.exists(): raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if any("y" in r or "z" in r for r in records): raise ValueError("deployment roster must be unlabelled")
    if len({_key(r)[3] for r in records}) != len(records): raise ValueError("duplicate frozen candidate IDs")
    with TemporaryDirectory(prefix=".peer198-", dir=root.parent) as tmp:
        work = Path(tmp)/"payload"; work.mkdir()
        if not phase1_passed:
            _save(work/"STOPPED.json", dict(status="stopped", reason="phase1 guarded capacity did not pass", guard_receipt=guard_receipt))
        else:
            for target in DOMAINS:
                all_source = [r for r in records if _key(r)[0] != target]
                source = [r for r in all_source if _eligible(r)]
                supplied = label_provider(target)
                if set(supplied["source_hives"]) != set(DOMAINS)-{target} or set(supplied["labels"]) != {_key(r)[3] for r in source}:
                    raise ValueError("strict other-three source labels required")
                training = [dict(r, **supplied["labels"][_key(r)[3]]) for r in source]
                fitted = fit(training, heldout=target)
                fitted["source_candidate_rows"] = len(all_source)
                fitted["source_domains"] = sorted(supplied["source_hives"])
                fitted["source_receipts"] = supplied.get("receipts", {})
                deployed = deploy_records([r for r in records if _key(r)[0] == target], fitted, member_skeleton)
                _save(work/f"{target}.model.json", fitted); _save(work/f"{target}.masks.json", deployed)
        _save(work/"receipt.json", dict(schema="endpoint-peer-masks198-v1", input_receipt=input_receipt,
            prereg=input_receipt.get("prereg") if isinstance(input_receipt, dict) else "synthetic",
            freeze_receipt=input_receipt.get("freeze_receipt") if isinstance(input_receipt, dict) else "synthetic",
            peerfreeze_receipt=input_receipt.get("freeze_receipt") if isinstance(input_receipt, dict) else "synthetic",
            source197_receipt=input_receipt.get("source197_receipt") if isinstance(input_receipt, dict) else "synthetic",
            guard_receipt=guard_receipt, stopped=not phase1_passed, model_code_sha256=_sha(__file__), arms=ARMS))
        _save(work/"manifest.json", {p.name: _sha(p) for p in sorted(work.iterdir())})
        (work/"SEALED.sha256").write_text(_sha(work/"manifest.json")+"\n")
        verify_seal(work); work.rename(root)
    return dict(stopped=not phase1_passed, manifest_sha256=_sha(root/"manifest.json"))


def verify_seal(root):
    root = Path(root)
    if (root/"SEALED.sha256").read_text().strip() != _sha(root/"manifest.json"): raise ValueError("peer mask seal differs")
    manifest = json.loads((root/"manifest.json").read_text())
    for name, digest in manifest.items():
        if _sha(root/name) != digest: raise ValueError("peer frozen file changed")
    receipt = json.loads((root/"receipt.json").read_text())
    expected = {"STOPPED.json", "receipt.json"} if receipt["stopped"] else {f"{h}.{k}.json" for h in DOMAINS for k in ("model", "masks")}|{"receipt.json"}
    if set(manifest) != expected or receipt["model_code_sha256"] != _sha(__file__): raise ValueError("incomplete or changed model freeze")
    return receipt


def load_predictions(root):
    receipt = verify_seal(root)
    if receipt["stopped"]: raise RuntimeError("phase1 failed: no stage2 model or masks were produced")
    folds, decisions = {}, {}
    for target in DOMAINS:
        fitted = json.loads((Path(root)/f"{target}.model.json").read_text())
        if target in fitted["source_hives"] or set(fitted["source_domains"]) != set(DOMAINS)-{target}: raise ValueError("heldout label boundary differs")
        fold = json.loads((Path(root)/f"{target}.masks.json").read_text())
        if fold["heldout"] != target: raise ValueError("heldout mask differs")
        folds[target] = fold
        for item in fold["member_clips"]:
            members = decisions.setdefault(item["unit"], {})
            if item["member"] in members: raise ValueError("duplicate member deployment")
            members[item["member"]] = item["selected"]
    return dict(receipt=receipt, folds=folds, decisions=decisions)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _verify_bundle(root):
    root = Path(root)
    if (root/"SEALED.sha256").read_text().strip() != _sha(root/"manifest.json"):
        raise ValueError("input manifest not sealed")
    manifest = json.loads((root/"manifest.json").read_text())
    for name, digest in manifest.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or _sha(path) != digest:
            raise ValueError("input artifact/hash differs")
    return _sha(root/"manifest.json")


def guarded_capacity(capacity, peer_receipt, prereg):
    """A count-only/incomplete capacity file cannot authorize source fitting."""
    if capacity.get("prereg") != prereg or capacity.get("input_sha256") != peer_receipt["deployment_sha256"]:
        raise ValueError("capacity decision belongs to another registration/peer freeze")
    if capacity.get("all60_native_graph_verified") is not True or capacity.get("physical_verified_members") != 60:
        raise ValueError("phase1 not finalized: all60 native/graph checks required")
    flag = capacity.get("stage2_authorized_by_count_gate")
    if not isinstance(flag, bool): raise ValueError("explicit completed phase1 resource decision required")
    return flag


def field_train(args):
    """Actual source-only CLI; root GO plus finalized pinned capacity required."""
    if (not args.prereg or not re.fullmatch("[0-9a-f]{40}", args.prereg)
            or not args.guard_sha or not re.fullmatch("[0-9a-f]{64}", args.guard_sha)):
        raise ValueError("full published preregistration and finalized guard SHA required")
    if _sha(args.guard) != args.guard_sha: raise ValueError("finalized guard SHA differs")
    sys.path.insert(0, str(args.repo.resolve()))
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    peer_freezer = _load_module(here/"freeze.py", "peer198_model_freezer")
    peer_receipt = peer_freezer.verify(args.root/"frozen")
    if peer_receipt["manifest_sha256"] != PEER_FREEZE_SHA: raise ValueError("authorized peer feature freeze differs")
    peer_data = json.loads((args.root/"frozen/deployment.json").read_text())
    if peer_data["prereg"] != args.prereg or peer_data["marker"] != "[endpoint-peer-v1]":
        raise ValueError("peer input preregistration differs")
    capacity = json.loads(args.guard.read_text())
    passed = guarded_capacity(capacity, peer_receipt, args.prereg)
    guard_receipt = dict(guard_sha256=args.guard_sha, peerfreeze_receipt=peer_receipt,
                         physical_audit_receipt=capacity.get("physical_audit_receipt"),
                         resource_gate=capacity.get("resource_gate"))
    input_receipt = dict(prereg=args.prereg, freeze_receipt=peer_receipt,
                         source197_receipt=peer_data["source197_receipt"],
                         code_receipts={"freeze.py": _sha(peer_freezer.__file__)})
    output = args.out or args.root/"predictions"
    if not passed:
        result = train_and_seal([], lambda target: (_ for _ in ()).throw(AssertionError("source entrance after failed guard")),
            output, input_receipt, phase1_passed=False, guard_receipt=guard_receipt)
        print(json.dumps(result, indent=2)); return
    # The independent splitter alone opened the all-hive phase1 graphs. This
    # process opens only its three source shards per fold; no GT/native replay.
    if args.shards is None or args.shards_sha is None:
        raise ValueError("sealed independently split source shards and manifest SHA required")
    import split_inputs198 as shards
    import graph198
    shard_manifest, shard_sha = shards.manifest_only(args.shards)
    if shard_sha != args.shards_sha or shard_manifest["prereg"] != args.prereg:
        raise ValueError("source shard manifest/registration differs")
    shard_stage1 = shard_manifest["stage1_receipt"]
    if (shard_stage1["capacity_sha256"] != args.guard_sha
            or shard_stage1["graph_data_sha256"] != capacity["physical_audit_receipt"]["graph_data_sha256"]
            or shard_stage1["peer_data_sha256"] != peer_receipt["deployment_sha256"]):
        raise ValueError("source shards do not bind the finalized checked capacity")
    groups = peer_freezer.load_groups(args.root/"frozen")
    # T197 masks are unlabelled references. Do NOT hash/open its model files,
    # which contain all four folds' historical source label audit tables.
    old_root = args.source197/"predictions"
    source197_mask_sha = _sha(old_root/"manifest.json")
    if (old_root/"SEALED.sha256").read_text().strip() != source197_mask_sha:
        raise ValueError("T197 mask manifest not sealed")
    if source197_mask_sha != SOURCE197_MASK_SHA: raise ValueError("original T197 mask freeze differs")
    old_manifest = json.loads((old_root/"manifest.json").read_text())
    for target in DOMAINS:
        name = target+".masks.json"
        if _sha(old_root/name) != old_manifest[name]: raise ValueError("unlabelled T197 mask changed")
    old_folds = {h: json.loads((args.source197/f"predictions/{h}.masks.json").read_text()) for h in DOMAINS}
    old_bits = {}
    for fold in old_folds.values():
        for member in fold["member_clips"]:
            selected = set(member["selected"]["T"])
            for c in member["candidates"]:
                key = c.get("key", c["candidate_id"])
                if key in old_bits: raise ValueError("duplicate T197 candidate")
                old_bits[key] = key in selected
    records = peer_freezer.load_records(args.root/"frozen")
    if {_key(r)[3] for r in records} != set(old_bits): raise ValueError("T197/peer candidate inventory differs")
    records = [dict(r, T197_cut=old_bits[_key(r)[3]]) for r in records]
    skeleton = [dict(unit=g["unit"], hive=g["domain"], clip=g["video"],
                     member_index=int(m["member_index"]), member=str(m["member"])) for g in groups for m in g["members"]]

    def provider(target):
        allowed = tuple(h for h in DOMAINS if h != target)
        loaded = shards.load_source_shards(args.shards, target, source_domains=allowed)
        def loader(group):
            if group["domain"] not in allowed: raise ValueError("heldout graph entrance blocked")
            saved = loaded["models"][group["unit"]]
            models = [saved[str(member["member"])] for member in group["members"]]
            for member, model in zip(group["members"], models):
                if {c["candidate_id"] for c in model["candidates"]} != {r["candidate_id"] for r in member["records"]}:
                    raise ValueError("source graph candidate inventory differs from peer features")
            return models
        labels = graph198.load_source_labels(groups, target, source_domains=allowed, group_loader=loader)
        eligible_keys = {_key(r)[3] for r in records if _key(r)[0] in allowed and _eligible(r)}
        labels["labels"] = {key: value for key, value in labels["labels"].items() if key in eligible_keys}
        labels["receipts"] = dict(graphs=labels["receipts"], shard_manifest_sha256=loaded["manifest_sha256"],
                                   source_shards=loaded["shard_receipts"], stage1_receipt=loaded["stage1_receipt"])
        print("eligible source labels:", target, len(labels["labels"]), flush=True)
        return labels

    input_receipt.update(source197_masks_sha256=source197_mask_sha,
        graph_code_sha256=_sha(graph198.__file__), source_shards_manifest_sha256=shard_sha,
        source_splitter_code_sha256=_sha(shards.__file__))
    result = train_and_seal(records, provider, output, input_receipt,
        phase1_passed=True, guard_receipt=guard_receipt, member_skeleton=skeleton)
    print(json.dumps(result, indent=2))


def self_test():
    rng = np.random.default_rng(19803)
    records = []
    for hive in DOMAINS:
        for group in (1, 2):
            for clip in ("a", "b") if group == 1 else ("a",):
                for i, delta in enumerate((1, 1, 1, -1)):
                    N = rng.normal(size=11); B = np.r_[N, rng.normal(size=4)]; I = np.r_[B, rng.normal(size=7)]
                    key = f"{hive}_g{group}_{clip}|0|{i}"
                    records.append(dict(key=key, candidate_id=key, hive=hive, group_id=group, clip=hive+clip,
                        unit=f"{hive}_g{group}_{clip}", member_index=0, member=hive+"_m0", delta=delta, direction=1,
                        eligible=i != 1, T197_cut=i < 2, features_N11=N.tolist(), features_B15=B.tolist(), features_I22=I.tolist(),
                        y=float(rng.choice([-.5, 0, .5])), z=int(rng.integers(0, 2))))
    source = [r for r in records if r["hive"] != "35"]
    fitted = fit(source, heldout="35")
    audit = fitted["training_records"]
    for hive in ("33", "W", "G"):
        assert abs(sum(r["weight"] for r in audit if r["hive"] == hive)-1/3) < 1e-15
        for group in (1, 2):
            assert abs(sum(r["weight"] for r in audit if r["hive"] == hive and r["group_id"] == group)-1/6) < 1e-15
            for clip in (hive+"a", hive+"b") if group == 1 else (hive+"a",):
                values = [sum(r["weight"] for r in audit if r["hive"] == hive and r["group_id"] == group and r["clip"] == clip and r["delta"] == delta) for delta in (-1, 1)]
                np.testing.assert_allclose(values, [sum(values)/2]*2, atol=1e-15)
    eligible = [r for r in source if r["eligible"]]
    X = _matrices(eligible)
    pred = predict(fitted, X["N"], X["B"], X["I"], [r["delta"] for r in eligible])
    assert predict(json.loads(json.dumps(fitted)), X["N"], X["B"], X["I"], [r["delta"] for r in eligible]) == pred
    empty = fit([], heldout="35")
    neutral = predict(empty, X["N"], X["B"], X["I"], [r["delta"] for r in eligible])
    assert all(not any(neutral[arm]["cut"]) for arm in ("N", "B", "I", "F", "Y"))
    const = fit([dict(r, y=.2, z=1) for r in source], heldout="35")
    constant = predict(const, X["N"], X["B"], X["I"], [r["delta"] for r in eligible])
    assert all(y == .2 for y in constant["I"]["yhat"]) and all(z == 1 for z in constant["I"]["zhat"])
    assert constant["I"]["cut"] == [r["delta"] == 1 for r in eligible]
    assert constant["F"]["cut"] == [r["delta"] == 1 for r in eligible]
    assert all(constant["Y"]["cut"])
    manual = json.loads(json.dumps(empty))
    for head in manual["heads"].values(): head["intercept"] = [2., -1.]
    clipped = predict(manual, X["N"], X["B"], X["I"], [r["delta"] for r in eligible])
    assert all(v == 1 for v in clipped["I"]["yhat"]) and all(v == 0 for v in clipped["I"]["zhat"])
    for head in manual["heads"].values(): head["intercept"] = [.2, .5]
    equality = predict(manual, X["N"], X["B"], X["I"], [r["delta"] for r in eligible])
    assert not any(equality["I"]["cut"])  # jointutility exactly.10.
    assert predict({k: fitted[k] for k in ("schema", "heads")}, X["N"], X["B"], X["I"], [r["delta"] for r in eligible]) == pred
    target = [r for r in records if r["hive"] == "35"]
    deployed = deploy_records(target, fitted)
    assert all(s["I_count"] == s["M_count"] for s in deployed["M_strata"])
    for item in deployed["member_clips"]:
        for index, okay in enumerate(item["eligible"]):
            if not okay:
                assert all(not item["masks"][a][index] for a in ("N", "B", "I", "F", "Y", "M", "Tg"))
                assert item["masks"]["T197"][index]
    clean = [{k: v for k, v in r.items() if k not in ("y", "z")} for r in records]
    calls = []
    def provider(target):
        calls.append(target)
        return dict(source_hives=[h for h in DOMAINS if h != target],
            labels={_key(r)[3]: dict(y=r["y"], z=r["z"]) for r in records if r["hive"] != target and _eligible(r)})
    with TemporaryDirectory(prefix="peer198-model-") as tmp:
        stop = Path(tmp)/"stopped"
        train_and_seal(clean, provider, stop, "synthetic", phase1_passed=False, guard_receipt="failed synthetic guard")
        assert not calls and verify_seal(stop)["stopped"]
        full = Path(tmp)/"full"
        train_and_seal(clean, provider, full, "synthetic", phase1_passed=True, guard_receipt="passed synthetic guard")
        assert calls == list(DOMAINS) and set(load_predictions(full)["folds"]) == set(DOMAINS)
    try: fit(records, heldout="35")
    except ValueError: pass
    else: raise AssertionError("heldout labels accepted")
    assert fit(reversed(source), heldout="35") == fitted
    peer_receipt = dict(deployment_sha256="synthetic")
    completed = dict(prereg="synthetic", input_sha256="synthetic", all60_native_graph_verified=True,
                     physical_verified_members=60, stage2_authorized_by_count_gate=False)
    assert not guarded_capacity(completed, peer_receipt, "synthetic")
    for incomplete in (dict(completed, all60_native_graph_verified=False), dict(completed, physical_verified_members=59)):
        try: guarded_capacity(incomplete, peer_receipt, "synthetic")
        except ValueError: pass
        else: raise AssertionError("unfinished guard allowed source training")
    return dict(seed=19803, dimensions=DIMENSIONS, source_weights="eligible hive/group/clip/add-drop/candidate",
                shared_N_B_I_cohort="passed", multioutput_ridge="unpenalized intercept; stationarity passed",
                empty_source="neutral no cut", clipping_and_constant_outputs="passed", ineligible="learned arms never cut",
                T197_and_Tg="preserved and eligibility-restricted", M="eligible count-matched by member/delta/direction",
                phase1_failed="stopped before any provider call", four_fold_masks="sealed before evaluation",
                target_guard_reload_order="passed", scope="synthetic only; no new field reads")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-field", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--source197", type=Path)
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--shards-sha")
    parser.add_argument("--guard", type=Path)
    parser.add_argument("--guard-sha")
    parser.add_argument("--prereg")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.run_field:
        if any(v is None for v in (args.repo,args.root,args.source197,args.guard)): parser.error("explicit --repo --root --source197 and finalized --guard required")
        field_train(args)
    else:
        print(json.dumps(self_test(), indent=2))
