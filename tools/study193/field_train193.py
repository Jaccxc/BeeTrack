"""Private §193 source-only trainer; default CLI uses synthetic contracts only.

No field data are opened here. After public preregistration, a caller may pass
freeze193.load_groups() units and a source193 whitelisted label provider.
Each fold receives only the other three domains' labels. All four target
prediction/model documents are sealed before a scoring process may open GT.
"""
import os
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import harm_budget as H
import identity_rank as R


DOMAINS = ("35", "33", "W", "G")
BUDGET_FEATURES = ["member_count_low", "member_count_middle", "member_count_high",
                   "member_count_range", "cross_member_support_fraction", "window_seconds", "is_in"]
RANK_FEATURES = ["in", "boxes", "frames", "missing", "tracklets", "start_d", "end_d",
                 "straight", "speed", "near", "cross_member_support"]
PREREG = "f161c64a6310c663d6cb079e83abee335d90e324"
PINNED_FREEZE = "e885efcd3952abb988f385b00681f22550faf2259ec3f3e496cb26e4bbacc9e5"


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False), encoding="utf-8")


def validate_field_inputs(units, frozen_root, repo=None, *, require_original=False):
    """Complete sealed freeze, native source hashes and excluded linkers.

    A fresh reconstruction has its own receipt. Historical byte equality is an
    optional extra check; every reconstruction still validates all sixty units.
    """
    import freeze193 as F
    frozen = Path(frozen_root)
    receipt = F.seal_receipt(frozen)
    if require_original and receipt["manifest_sha256"] != PINNED_FREEZE:
        raise ValueError("unexpected immutable §193 input manifest")
    manifest = json.loads((frozen / 'manifest.json').read_text())
    expected_files = {'deployment.json'} | {f'rows/u{i:02d}.npz' for i in range(60)}
    if set(manifest) != expected_files:
        raise ValueError('complete deployment plus60 native row files required')
    for name, digest in manifest.items():
        path = (frozen / name).resolve()
        if not path.is_relative_to(frozen.resolve()) or _digest(path) != digest:
            raise ValueError('unsafe or changed frozen artifact')
    deployment = json.loads((frozen / "deployment.json").read_text())
    if deployment['schema'] != 'budget193-deployment-v1' or deployment["prereg"] != PREREG or len(units) != 20 or sum(len(u["members"]) for u in units) != 60:
        raise ValueError("preregistration or ALL60 contract mismatch")
    if len(deployment['units']) != 60 or len(deployment['members']) != 42 or len(deployment['clips']) != 6 or len(deployment['groups']) != 20:
        raise ValueError('native deployment bookkeeping incomplete')
    for name, digest in F.MODEL_SHA.items():
        if deployment['source_hashes'].get(f'beetrack/models/{name}.json') != digest:
            raise ValueError('model provenance differs from registered clean linker')
    if repo is not None:
        repo = Path(repo)
        for name, digest in deployment['source_hashes'].items():
            path = repo / 'beetrack/zones.json' if name == 'zones' else Path(F.__file__) if name == 'extractor' else repo / name
            if _digest(path) != digest:
                raise ValueError('native builder/source digest mismatch')
    expected = {}
    members = F.members()
    for domain, split, video, nf in F.CLIPS:
        groups = sorted({member["group"] for member in members if member["hive"] == domain})
        for group in groups:
            expected[(domain, video, group)] = (split, nf, [m for m in members if m["hive"] == domain and m["group"] == group])
    if {(u["domain"], u["video"], u["group"]) for u in units} != set(expected):
        raise ValueError("native domain/video/seed-group hard allowlist mismatch")
    for unit in units:
        split, nf, seeds = expected[unit["domain"], unit["video"], unit["group"]]
        linker = "m_no" + unit["domain"] if unit["domain"] in ("W", "G") else "m_dev_strict"
        excluded = [unit["domain"]] if unit["domain"] in ("W", "G") else ["33", "35"]
        if unit["nf"] != nf or unit["split"] != split:
            raise ValueError("native frame/split allowlist mismatch")
        if len(unit['members']) != 3:
            raise ValueError('exactly three native members per group required')
        for member, seed in zip(unit["members"], seeds):
            if str(member["member"]) != str(seed["member"]) or member["tag"] != seed["tag"]:
                raise ValueError("native seed order mismatch")
            proof = member["linker_receipt"]
            if member["linker_name"] != linker or proof != dict(name=linker, sha256=F.MODEL_SHA[linker], excluded_domains=excluded):
                raise ValueError("domain-excluded linker mismatch")
            kinds = {"native_clean_rebuild"} if unit["domain"] in ("W", "G") else {"native_clean_rebuild", "section189_head_cache"}
            if member["raw_receipt"]["source"]["kind"] not in kinds or member["raw_receipt"]["source"]["linker"] != proof:
                raise ValueError("raw asset provenance mismatch")
            path = frozen / member["raw_receipt"]["file"]
            if member['raw_receipt']['file'] not in expected_files or manifest[member['raw_receipt']['file']] != member['raw_receipt']['sha256'] or _digest(path) != member["raw_receipt"]["sha256"]:
                raise ValueError("native row digest mismatch")
            native = deployment['units'][member['unit_index']]
            if (native['member'], native['clip'], native['linker_name'], native['frozen_rows_sha256']) != (int(member['member']), unit['video'], linker, member['raw_receipt']['sha256']):
                raise ValueError('normalized member differs from sealed native unit')
            if native['detector_sha256'] != member['raw_receipt']['source']['detector_sha256']:
                raise ValueError('detector provenance mismatch')
            if repo is not None:
                detector = repo / 'runs/dets' / member['tag'] / split / (unit['video'] + '.npy')
                if _digest(detector) != native['detector_sha256']:
                    raise ValueError('source detector changed after native reconstruction')
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved['rows'], member['rows']) or saved['s1'].shape != (len(member['rows']),):
                    raise ValueError('normalized native rows differ from sealed asset')
    return receipt


def source_coverage(model, target_X=None):
    """Record actual-video sample size and exact-duration support, not seeds."""
    training = model["budget_training"]
    x = np.asarray(training["raw_X"])
    sources, videos = np.asarray(training["source"]), np.asarray(training["video"])
    result = {}
    for source in sorted(set(sources)):
        names = sorted(set(videos[sources == source]))
        item = dict(videos=names, video_count=len(names), effective_k=min(5, len(names)),
                    full_30second_videos=sum(bool(np.isclose(x[videos == video, 5], 30).any()) for video in names),
                    duration_ranges={video: [float(x[videos == video, 5].min()), float(x[videos == video, 5].max())] for video in names})
        if target_X is not None:
            lengths, counts = np.unique(np.asarray(target_X)[:, 5], return_counts=True)
            item["target_duration_coverage"] = [dict(seconds=float(length), target_windows=int(count),
                source_video_count=sum(bool(np.isclose(x[videos == video, 5], length, rtol=0, atol=1e-12).any()) for video in names))
                for length, count in zip(lengths, counts)]
        result[str(source)] = item
    return result


def unit_features(unit):
    """GT-free source/deployment features in the preregistered fixed order."""
    if unit["domain"] not in DOMAINS or len(unit["members"]) != 3 or unit["nf"] < 1:
        raise ValueError("one native three-member unit and positive frame count required")
    nf = int(unit["nf"])
    T = (nf + 24) // 25
    windows = H.intervals(T)
    A = H.incidence(windows, T)
    events, owners, predictions, feature10 = [], [], [], []
    buckets = {}
    for owner, member in enumerate(unit["members"]):
        member_name = str(member["member"])
        ordered = sorted(member["event_records"], key=lambda event: int(event["pid"]))
        for event in ordered:
            pid, frame, direction = event["pid"], event["frame"], event["direction"]
            if int(pid) != pid or int(frame) != frame or not 1 <= frame <= nf or direction not in (-1, 1):
                raise ValueError("invalid native event identity/frame/direction")
            pred = (member_name, int(pid))
            if pred in predictions:
                raise ValueError("repeated member/native ID")
            x = np.asarray(event["features10"], float)
            box = np.asarray(event["box"], float)
            if x.shape != (10,) or box.shape != (4,) or not np.isfinite(x).all() or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                raise ValueError("finite 10 native features and positive xyxy representative box required")
            events.append(event)
            owners.append(owner)
            predictions.append(pred)
            feature10.append(x)
            bucket = (member_name, (int(frame) - 1) // 25, int(direction))
            buckets.setdefault(bucket, []).append(pred)
    for bucket in buckets:
        buckets[bucket].sort(key=lambda pred: pred[1])
    count = len(events)
    supported = np.zeros(count, bool)
    if count:
        box = np.array([e["box"] for e in events], float)
        area = np.prod(box[:, 2:] - box[:, :2], axis=1)
        frames = np.array([e["frame"] for e in events])
        directions = np.array([e["direction"] for e in events])
        owners = np.array(owners)
        for start in range(0, count, 128):
            stop = min(start + 128, count)
            extent = np.maximum(0, np.minimum(box[start:stop, None, 2:], box[:, 2:])
                                - np.maximum(box[start:stop, None, :2], box[:, :2]))
            overlap = extent.prod(axis=2)
            iou = overlap / (area[start:stop, None] + area - overlap)
            match = ((owners[start:stop, None] != owners) & (directions[start:stop, None] == directions)
                     & (np.abs(frames[start:stop, None] - frames) <= 5) & (iou >= .5))
            supported[start:stop] = match.any(axis=1)
    rank_features = {pred: np.r_[x, int(support)].tolist()
                     for pred, x, support in zip(predictions, feature10, supported)}
    by_direction = {}
    for direction in (1, -1):
        member_ids = [[[] for _ in range(3)] for _ in range(T)]
        support_per_second = np.zeros(T, int)
        for i, event in enumerate(events):
            if event["direction"] != direction:
                continue
            t = (int(event["frame"]) - 1) // 25
            owner = int(owners[i])
            member_ids[t][owner].append(predictions[i])
            support_per_second[t] += int(supported[i])
        for members in member_ids:
            for ids in members:
                ids.sort(key=lambda pred: pred[1])
        capacity = np.array([[len(ids) for ids in members] for members in member_ids], int)
        window_counts = np.asarray(A @ capacity).astype(int)
        pooled = window_counts.sum(axis=1)
        duration = np.array([(min(25 * hi, nf) - 25 * lo) / 25 for lo, hi in windows])
        X = np.column_stack([np.sort(window_counts, axis=1), np.ptp(window_counts, axis=1),
                             np.asarray(A @ support_per_second).ravel() / np.maximum(pooled, 1),
                             duration, np.full(len(windows), int(direction == 1))])
        by_direction[direction] = dict(X=X, pooled_counts=pooled, member_ids=member_ids,
                                       capacity=capacity, A=A)
    return dict(windows=windows, seconds=T, buckets=buckets, features=rank_features,
                directions=by_direction, supported=dict(zip(predictions, supported.tolist())))


def fit_fold(heldout, units, labels):
    """labels={video:{domain,gt_window_counts,rank_edges,gt_events}}; no heldout."""
    if heldout not in DOMAINS:
        raise ValueError("unknown heldout domain")
    allowed = set(DOMAINS) - {heldout}
    if not labels or {label["domain"] for label in labels.values()} != allowed:
        raise ValueError("fold labels must contain exactly the three source domains")
    unit_by_video = {}
    for unit in units:
        if unit["domain"] in allowed:
            unit_by_video.setdefault(unit["video"], []).append(unit)
    if set(labels) != set(unit_by_video):
        raise ValueError("source video label whitelist does not match native units")
    X, residuals, sources, videos = [], [], [], []
    rank_groups = []
    for video in sorted(unit_by_video):
        label = labels[video]
        rank = dict(video=video, buckets={}, quotas={}, features={}, edges=label["rank_edges"])
        for unit in unit_by_video[video]:
            if label["domain"] != unit["domain"]:
                raise ValueError("source video domain mismatch")
            f = unit_features(unit)
            for direction, d in f["directions"].items():
                gt = np.asarray(label["gt_window_counts"][direction])
                if gt.shape != d["pooled_counts"].shape or not np.isfinite(gt).all() or np.any(gt < 0) or np.any(gt != np.floor(gt)):
                    raise ValueError("invalid source GT window count vector")
                X.extend(d["X"])
                residuals.extend(d["pooled_counts"] - 3 * gt)
                sources.extend([unit["domain"]] * len(gt))
                videos.extend([video] * len(gt))
            if set(rank["buckets"]) & set(f["buckets"]):
                raise ValueError("same source member repeated across seed groups")
            rank["buckets"].update(f["buckets"])
            rank["features"].update(f["features"])
        rank["quotas"] = {bucket: len(ids) - int(len(ids) >= 2) for bucket, ids in rank["buckets"].items()}
        if rank["buckets"]:
            if set(rank["features"]) != set(rank["edges"]):
                raise ValueError("complete source legal graph required for all native events")
            rank_groups.append(rank)
    budget = H.fit(X, residuals, sources, videos, k=5)
    identity = R.fit(rank_groups, l2=1.)
    rank_examples = []
    for group in rank_groups:
        pairs, audit = R.source_pairs(group)
        rank_examples.append(dict(video=group["video"],
            buckets=[dict(member=b[0], second=b[1], direction=b[2],
                          pred_ids=[int(p[1]) for p in ids], keep=group["quotas"][b])
                     for b, ids in group["buckets"].items()],
            events=[dict(member=p[0], pid=int(p[1]), features=x, legal_gt=group["edges"][p])
                    for p, x in group["features"].items()],
            pairs=[dict(a=list(a), b=list(b), delta=delta,
                        weight_each_reversal=1 / (len(rank_groups) * 2 * len(pairs)))
                   for a, b, delta in pairs], audit=audit))
    result = dict(heldout=heldout, source_domains=sorted(allowed), source_videos=sorted(labels),
                budget=budget, identity=identity, budget_feature_names=BUDGET_FEATURES,
                identity_feature_names=RANK_FEATURES,
                budget_training=dict(raw_X=np.asarray(X), residual=np.asarray(residuals),
                                     source=sources, video=videos), rank_training=rank_examples,
                source_label_receipts={video: labels[video].get("source_receipt") for video in sorted(labels)})
    result["source_coverage"] = source_coverage(result)
    return result


def predict_unit(unit, model):
    """No labels/graphs accepted. Q and R share every per-member keep quota."""
    if unit["domain"] != model["heldout"]:
        raise ValueError("unit must belong to heldout fold")
    f = unit_features(unit)
    all_keep = {bucket: len(ids) for bucket, ids in f["buckets"].items()}
    quotas = {"B": dict(all_keep), "I": {}, "Q": {}}
    direction_proof = {}
    member_names = [str(m["member"]) for m in unit["members"]]
    for direction, d in f["directions"].items():
        pooled_capacity = d["capacity"].sum(axis=1)
        maximum = int(max(d["A"] @ pooled_capacity, default=0))
        probability = H.survival(model["budget"], d["X"], maximum)
        costs = 1 - 2 * probability
        candidates = [[pred for ids in members for pred in ids] for members in d["member_ids"]]
        coherent = H.decode(f["windows"], candidates, costs)
        independent_q = H.independent(d["A"], pooled_capacity, costs)
        for arm, q in (("I", independent_q), ("Q", coherent["q"])):
            prepared = H.prepare_member_quotas(q, d["member_ids"])
            for t, capacities in enumerate(d["capacity"]):
                for i, capacity in enumerate(capacities):
                    if capacity:
                        quotas[arm][(member_names[i], t, direction)] = int(prepared["keep"][t, i])
        direction_proof[str(direction)] = dict(q_Q=coherent["q"], q_I=independent_q.tolist(),
            objective=coherent["objective"], dual=coherent["dual"],
            window_features=d["X"].tolist(), pooled_window_counts=d["pooled_counts"].tolist(),
            source_coverage=source_coverage(model, d["X"]))
    fixed = {arm: R.fixed(f["buckets"], q) for arm, q in quotas.items()}
    ranked, abstained = R.select(model["identity"], f["buckets"], quotas["Q"], f["features"])
    fixed["R"] = ranked
    quotas["R"] = quotas["Q"]
    output_members = []
    for member in unit["members"]:
        name = str(member["member"])
        pool = sorted(int(event["pid"]) for event in member["event_records"])
        arms = {}
        for arm, selected in fixed.items():
            cells = [dict(second=b[1], direction=b[2], capacity=len(ids), keep=quotas[arm][b],
                          delete=len(ids) - quotas[arm][b]) for b, ids in f["buckets"].items() if b[0] == name]
            kept = sorted(int(pred[1]) for b, ids in selected.items() if b[0] == name for pred in ids)
            if sum(cell["keep"] for cell in cells) != len(kept):
                raise RuntimeError("member keep count and selected identities differ")
            arms[arm] = dict(kept_ids=kept, flexible_pool_ids=pool, quotas=cells)
        if arms["Q"]["quotas"] != arms["R"]["quotas"]:
            raise RuntimeError("Q/R member quotas differ")
        output_members.append(dict(member=name, tag=member.get("tag"),
                                   raw_receipt=member.get("raw_receipt"),
                                   linker_name=member.get("linker_name"), linker_receipt=member.get("linker_receipt"), arms=arms))
    return dict(schema="field193-prediction-unit-v1", unit=unit["unit"], domain=unit["domain"],
                video=unit["video"], group=unit["group"], nf=unit["nf"], members=output_members,
                freeze_receipt=unit.get("freeze_receipt"), linker_model=unit.get("linker_model"),
                linker_sha256=unit.get("linker_sha256"), linker_excluded_domains=unit.get("linker_excluded_domains"),
                direction_proof=direction_proof, abstained_buckets=[list(b) for b in abstained],
                rank_features=[dict(member=p[0], pid=p[1], features=x) for p, x in f["features"].items()])


def train_and_freeze(units, label_provider, output_dir, input_receipt):
    """label_provider(heldout) must open ONLY other-three-domain source labels.

    Writes a new or untouched empty private directory; seal hashes all four prediction and
    model documents. There is deliberately no target evaluation callback.
    """
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("refusing to overwrite an existing freeze")
    if {unit["domain"] for unit in units} != set(DOMAINS):
        raise ValueError("all four domains must be frozen before training")
    output.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for target in DOMAINS:
        print(f"fold {target}: loading explicitly whitelisted sources", flush=True)
        labels = label_provider(target)
        print(f"fold {target}: fitting budget and identity heads", flush=True)
        model = fit_fold(target, units, labels)
        del labels
        predictions = []
        for unit in units:
            if unit["domain"] == target:
                print(f"fold {target}: predicting {unit['unit']}", flush=True)
                predictions.append(predict_unit(unit, model))
        for kind, document in (("model", model), ("predictions", predictions)):
            path = output / f"{target}.{kind}.json"
            _write(path, document)
            receipts[path.name] = _digest(path)
        print(f"fold {target}: documents frozen; target evaluation still forbidden", flush=True)
    code_names = ("field_train193.py", "harm_budget.py", "identity_rank.py", "identity_flow.py",
                  "source193.py", "freeze193.py")
    code_receipts = {name: _digest(Path(__file__).with_name(name)) for name in code_names}
    seal = dict(schema="field193-all-four-seal-v1", input_receipt=input_receipt, code_receipts=code_receipts,
                domains=list(DOMAINS), files=receipts,
                protocol=dict(k=5, l2=1, candidate_pool="all C.40 native events",
                              source_groups="actual video; seed groups and directions merged",
                              budget_features=BUDGET_FEATURES, rank_features=RANK_FEATURES))
    _write(output / "ALL_FOUR_SEAL.json", seal)
    (output / "ALL_FOUR_SEAL.sha256").write_text(_digest(output / "ALL_FOUR_SEAL.json") + "\n")
    return seal


def synthetic_contract():
    """Four domains, two videos/domain, two seed groups/video; no field reads."""
    units, labels = [], {}
    for domain in DOMAINS:
        for clip in range(2):
            video = f"synthetic-{domain}-{clip}"
            edges = {}
            for group in range(2):
                members = []
                for i in range(3):
                    name = f"{domain}-seed{group * 3 + i}"
                    event_records = []
                    for j, direction in enumerate((1, -1)):
                        for good in (False, True):
                            pid = j * 2 + int(good) + 1
                            event_records.append(dict(pid=pid, frame=51 + 250*j, direction=direction,
                                features10=[int(direction == 1), 2, 4 if good else 2, 0, 1, 1, 1, 1, 1, 0],
                                box=([0, 0, 2, 2] if good else [10*i+10, 10, 10*i+12, 12])))
                            edges[(name, pid)] = [f"gt-{j}"] if good else []
                    members.append(dict(member=name, tag=name, event_records=event_records,
                                        raw_receipt="synthetic-no-file", rows=np.empty((0, 7))))
                units.append(dict(unit=f"{video}/g{group}", domain=domain, video=video, group=group,
                                  nf=998, polygon=[[0, 0], [2, 0], [2, 2]], members=members))
            windows = H.intervals(40)
            A = H.incidence(windows, 40)
            gt = {}
            for direction, t in ((1, 2), (-1, 12)):
                counts = np.zeros(40, int)
                counts[t] = 1
                gt[direction] = np.rint(A @ counts).astype(int)
            labels[video] = dict(domain=domain, gt_window_counts=gt, rank_edges=edges,
                                 gt_events={member: ["gt-0", "gt-1"] for member, _ in edges})
    return units, labels


def selfcheck():
    units, labels = synthetic_contract()
    called = []

    def provider(target):
        source = {video: value for video, value in labels.items() if value["domain"] != target}
        assert {v["domain"] for v in source.values()} == set(DOMAINS) - {target}
        called.append(target)
        return source

    with TemporaryDirectory(prefix="synthetic193_") as tmp:
        output = Path(tmp) / "freeze"
        seal = train_and_freeze(units, provider, output, "synthetic-input-receipt")
        assert called == list(DOMAINS) and len(seal["files"]) == 8
        assert _digest(output / "ALL_FOUR_SEAL.json") == (output / "ALL_FOUR_SEAL.sha256").read_text().strip()
        for name, sha in seal["files"].items():
            assert _digest(output / name) == sha
        summary = {}
        for domain in DOMAINS:
            predictions = json.loads((output / f"{domain}.predictions.json").read_text())
            model = json.loads((output / f"{domain}.model.json").read_text())
            assert domain not in model["source_domains"]
            assert len(model["source_videos"]) == 6
            assert len({r[1] for r in model["budget"]["records"]}) == 6
            counts = {arm: 0 for arm in ("B", "I", "Q", "R")}
            for prediction in predictions:
                for member in prediction["members"]:
                    for arm, detail in member["arms"].items():
                        counts[arm] += len(detail["kept_ids"])
                    assert member["arms"]["Q"]["quotas"] == member["arms"]["R"]["quotas"]
                    assert member["arms"]["R"]["kept_ids"] == [2, 4]
                f = unit_features(next(u for u in units if u["unit"] == prediction["unit"]))
                assert all(x.shape[1] == 7 for x in [d["X"] for d in f["directions"].values()])
                assert any(np.isclose(d["X"][:, 5], .92).any() for d in f["directions"].values())
            summary[domain] = counts
    try:
        fit_fold("35", units, labels)
    except ValueError:
        pass
    else:
        raise AssertionError("heldout label contamination accepted")
    return dict(scope="synthetic contract only; no field files opened", domains=summary,
                video_grouping="6 source videos per fold, not12 seed units", k=5, rank_l2=1,
                heldout_label_guard="passed", all_four_seal="passed",
                exact_partial_tail_seconds="passed", Q_R_member_quotas="identical")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-field", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--prereg")
    parser.add_argument('--require-original-freeze', action='store_true', help='Also require the historical freeze manifest bytes')
    args = parser.parse_args()
    if args.run_field:
        if args.prereg != PREREG or args.repo is None or args.root is None:
            parser.error("published --prereg, --repo and explicit --root required")
        sys.path.insert(0, str(args.repo.resolve()))
        import freeze193 as F
        import source193 as S
        frozen = args.root / "frozen"
        units = F.load_groups(frozen)
        print("ALL60 freeze validation:", validate_field_inputs(units, frozen, args.repo, require_original=args.require_original_freeze), flush=True)
        provider = S.make_label_provider(args.repo, frozen, PREREG, evidence_dir=args.root / "source_labels")
        receipt = (frozen / "SEALED.sha256").read_text().strip()
        seal = train_and_freeze(units, provider, args.root / "deploy", receipt)
        print("ALL_FOUR SEALED", _digest(args.root / "deploy/ALL_FOUR_SEAL.json"), flush=True)
    else:
        print(json.dumps(selfcheck(), indent=2))
