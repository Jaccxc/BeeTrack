"""Source-only pair texture ridge models and blind fit/prediction seal.

No pixels or GT files are opened: inputs are the sealed descriptor bundle.
predict() sees only four xywh boxes and optional four12-dimensional descriptor
rows. It never sees IDs, dates, hive, pair digest or physical truth.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

SCHEMA = "pair-texture-model-v1"
SELECTION_SHA = "027e127674826303042df7074b3b855bb282b7caaf2e7205da3030cb6a4943a4"
INDICES = dict(P=list(range(3)), C=list(range(9)), T=list(range(15)), V=list(range(9, 15)))
NAMES = ["centre_distance", "log_area_distance", "log_aspect_distance"] + [f"color{i}" for i in range(6)] + [f"texture{i}" for i in range(6)]
RIDGE = .20
THRESHOLD = 1e-10
VARIANTS = ("R", "Z", "Q0", "Q1", "Q2", "Q3")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n", encoding="utf-8")


def _boxes(rows):
    rows = np.asarray(rows, float)
    boxes = rows[:, 2:6] if rows.ndim == 2 and rows.shape[1] >= 6 else rows
    if boxes.shape != (4, 4) or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= 0):
        raise ValueError("four finite positive xywh boxes required")
    return boxes.copy()


def distances(boxes, crops=None):
    """2x2x15 edge feature distances; all-four mean sqrt(area) is fixed."""
    b = _boxes(boxes)
    area = b[:, 2]*b[:, 3]
    centre = b[:, :2]+b[:, 2:]/2
    size = np.sqrt(area).mean()
    aspect = np.log(b[:, 2]/b[:, 3])
    edges = np.zeros((2, 2, 15))
    edges[:, :, 0] = np.linalg.norm(centre[:2, None]-centre[None, 2:], axis=2)/size
    edges[:, :, 1] = np.abs(np.log(area[:2, None]/area[None, 2:]))
    edges[:, :, 2] = np.abs(aspect[:2, None]-aspect[None, 2:])
    if crops is not None:
        crops = np.asarray(crops, float)
        if crops.shape != (4, 12) or not np.isfinite(crops).all():
            raise ValueError("four finite color6+texture6 descriptor rows required")
        edges[:, :, 3:] = np.abs(crops[:2, None]-crops[None, 2:])
    return edges


def difference(edges):
    return (edges[0, 1]+edges[1, 0])-(edges[0, 0]+edges[1, 1])


def _cohort(pair):
    cohort = pair.get("cohort", pair.get("subset"))
    if cohort not in ("old", "new") or pair.get("subset", cohort) != cohort:
        raise ValueError("consistent old/new cohort metadata required")
    return cohort


def pair_key(pair):
    key = pair["clip"]+"|"+pair["digest"]
    if pair.get("pair_key", key) != key:
        raise ValueError("pair key differs from clip|digest")
    return key


def _rows(pair):
    return pair["endpoint_rows"] if "endpoint_rows" in pair else pair["rows"]


def fit(pairs, source_hive):
    """Only source old93 valid original RGB; P uses these SAME source pairs."""
    if source_hive not in ("W", "G"):
        raise ValueError("source hive must be W/G")
    selected = [p for p in pairs if p["hive"] == source_hive and _cohort(p) == "old" and p["primary"]["valid"]]
    selected.sort(key=pair_key)
    if not selected or len({pair_key(p) for p in selected}) != len(selected):
        raise ValueError("nonempty distinct source-valid old pairs required")
    clips = sorted({p["clip"] for p in selected})
    count = {clip: sum(p["clip"] == clip for p in selected) for clip in clips}
    weights = np.array([1/(len(clips)*count[p["clip"]]) for p in selected])
    d = np.stack([difference(distances(_rows(p), p["primary"]["crops"])) for p in selected])
    heads = {}
    for arm, columns in INDICES.items():
        raw = d[:, columns]
        scale = np.sqrt(weights @ (raw**2)); scale[scale == 0] = 1
        X = raw/scale
        beta = np.linalg.solve(X.T @ (weights[:, None]*X)+RIDGE*np.eye(len(columns)), X.T @ weights)
        error = X @ beta-1
        gradient = 2*(X.T @ (weights*error)+RIDGE*beta)
        if not np.isfinite(beta).all() or np.max(np.abs(gradient), initial=0) > 1e-8:
            raise RuntimeError("ridge validation failed")
        heads[arm] = dict(columns=columns, scale=scale.tolist(), beta=beta.tolist(), ridge=RIDGE,
            intercept=0, objective=float(weights @ (error**2)+RIDGE*(beta @ beta)),
            max_abs_gradient=float(np.max(np.abs(gradient), initial=0)))
    audit = [dict(key=pair_key(p), clip=p["clip"], digest=p["digest"], weight=float(weight), d=vector.tolist())
             for p, weight, vector in zip(selected, weights, d)]
    return dict(schema=SCHEMA, source_hive=source_hive, feature_names=NAMES, heads=heads,
                training_records=audit, training_clips=clips, source_pairs=len(selected),
                weighting="nonempty source clips equal; valid old primary pairs equal within clip", threshold=THRESHOLD)


def predict(model, boxes, crops=None, *, valid=True):
    """Return blind2x2 learned costs and score; no ID/GT/hive is accepted.

    Linear learned costs may be negative (not a mathematical metric). Score is
    off-diagonal cost minus diagonal cost. Missing RGB forces C/T/V score0.
    """
    if model.get("schema") != SCHEMA:
        raise ValueError("wrong model schema")
    rgb_valid = bool(valid and crops is not None)
    edge = distances(boxes, crops if rgb_valid else None)
    out = {}
    for arm, columns in INDICES.items():
        head = model["heads"][arm]
        scale, beta = np.array(head["scale"]), np.array(head["beta"])
        if head["columns"] != columns or scale.shape != (len(columns),) or beta.shape != scale.shape or np.any(scale <= 0) or not np.isfinite(scale).all() or not np.isfinite(beta).all():
            raise ValueError("invalid frozen coefficients")
        usable = arm == "P" or rgb_valid
        costs = (edge[:, :, columns]/scale) @ beta if usable else np.zeros((2, 2))
        score = float((costs[0, 1]+costs[1, 0])-(costs[0, 0]+costs[1, 1]))
        decision = "given" if score > THRESHOLD else "cross" if score < -THRESHOLD else "tie"
        out[arm] = dict(costs2x2=costs.tolist(), score=score, decision=decision,
                        physical_choice="cross" if decision == "cross" else "given", valid=bool(usable))
    return out


def blind_order(digest):
    if not isinstance(digest, str) or not digest:
        raise ValueError("frozen pair digest required")
    bit = hashlib.sha256(("texture-order-v1|"+digest).encode("utf-8")).digest()[0] & 1
    return bit, [0, 1, 2+bit, 3-bit]


def predict_record(pair, model):
    bit, permutation = blind_order(pair["digest"])
    boxes = _boxes(_rows(pair))[permutation]
    variants = {}
    arms = [pair["primary"], pair["shifted"]]+list(pair["shuffles"])
    if len(arms) != 6:
        raise ValueError("R/Z and exactly4 shuffle records required")
    for name, descriptors in zip(VARIANTS, arms):
        valid = bool(descriptors["valid"])
        crops = np.asarray(descriptors["crops"], float)[permutation] if valid else None
        variants[name] = predict(model, boxes, crops, valid=valid)
    excluded = {"primary", "shifted", "shuffles", "rows", "endpoint_rows"}
    metadata = {k: v for k, v in pair.items() if k not in excluded}
    return dict(key=pair_key(pair), hive=pair["hive"], clip=pair["clip"], digest=pair["digest"],
                order_bit=bit, permutation=permutation, metadata=metadata, variants=variants)


def verify_prediction_seal(root):
    root = Path(root)
    if (root/"SEALED.sha256").read_text().strip() != _sha(root/"manifest.json"):
        raise ValueError("prediction manifest seal mismatch")
    manifest = json.loads((root/"manifest.json").read_text())
    expected = {"model_for_W.json", "model_for_G.json", "predictions.json", "receipt.json"}
    if set(manifest) != expected:
        raise ValueError("both models and all predictions must be sealed together")
    for file, digest in manifest.items():
        if _sha(root/file) != digest:
            raise ValueError("sealed prediction file changed: "+file)
    receipt = json.loads((root/"receipt.json").read_text())
    if receipt["model_code_sha256"] != _sha(__file__):
        raise ValueError("model code changed after prediction freeze")
    return json.loads((root/"predictions.json").read_text())


def seal_predictions(bundle, output_dir, input_receipt, *, expected_old=93):
    if bundle["schema"] != "pair-texture-features-v1" or bundle["selection_sha256"] != SELECTION_SHA:
        raise ValueError("wrong feature/old selection protocol")
    pairs = bundle["pairs"]
    keys = [pair_key(p) for p in pairs]
    if len(keys) != len(set(keys)) or any(p["hive"] not in ("W", "G") for p in pairs):
        raise ValueError("duplicate pairs or unexpected source domain")
    old = [p for p in pairs if _cohort(p) == "old"]
    new = [p for p in pairs if _cohort(p) == "new"]
    if expected_old is not None and len(old) != expected_old:
        raise ValueError("old93 cohort size changed")
    for clip in {p["clip"] for p in new}:
        if sum(p["clip"] == clip for p in new) > 12:
            raise ValueError("new cohort exceeds preselected12/clip")
    models = {target: fit(pairs, "G" if target == "W" else "W") for target in ("W", "G")}
    decisions = [predict_record(pair, models[pair["hive"]]) for pair in sorted(new, key=pair_key)]
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pair-texture-pred-", dir=root.parent) as tmp:
        work = Path(tmp)/"payload"; work.mkdir()
        for target, fitted in models.items():
            _save(work/f"model_for_{target}.json", fitted)
        _save(work/"predictions.json", dict(schema="pair-texture-predictions-v1", prereg=bundle["prereg"],
            features_sha256=input_receipt["features_sha256"], roster_sha256=bundle["roster_sha256"],
            selection_sha256=bundle["selection_sha256"], pairs=decisions))
        _save(work/"receipt.json", dict(input_receipt=input_receipt, model_code_sha256=_sha(__file__),
            old_pairs=len(old), new_pairs=len(new), folds={h: models[h]["source_hive"] for h in models}))
        manifest = {p.name: _sha(p) for p in sorted(work.iterdir())}
        _save(work/"manifest.json", manifest)
        (work/"SEALED.sha256").write_text(_sha(work/"manifest.json")+"\n")
        verify_prediction_seal(work)
        work.rename(root)
    return dict(prediction_seal_sha256=_sha(root/"manifest.json"), pairs=len(new), models={h: m["source_pairs"] for h, m in models.items()})


def _load_features(path, prereg):
    path = Path(path); root = path.parent
    if not re.fullmatch("[0-9a-f]{40}", prereg):
        raise ValueError("full published preregistration SHA required")
    if (root/"SEALED.sha256").read_text().strip() != _sha(root/"manifest.json"):
        raise ValueError("descriptor input not sealed")
    for name, digest in json.loads((root/"manifest.json").read_text()).items():
        file = (root/name).resolve()
        if not file.is_relative_to(root.resolve()) or _sha(file) != digest:
            raise ValueError("descriptor input file/hash mismatch")
    bundle = json.loads(path.read_text())
    if bundle["prereg"] != prereg or bundle["roster_sha256"] != _sha(root/"roster.json"):
        raise ValueError("descriptor preregistration/roster differs")
    return bundle, dict(features_sha256=_sha(path), manifest_sha256=_sha(root/"manifest.json"),
                        roster_sha256=_sha(root/"roster.json"), prereg=prereg)


def _synthetic():
    pairs = []
    rng = np.random.default_rng(19604)
    for hive in ("W", "G"):
        for clip in range(2):
            for n in range(4):
                rows = np.array([[5, 7, 4, 4, 8, 8], [5, 9, 22, 4, 8, 8],
                                 [13, 7, 5, 4, 8, 8], [13, 9, 21, 4, 8, 8]], float)
                first = rng.uniform(.1, .9, (2, 12))
                crops = np.r_[first, first+rng.normal(0, .005, (2, 12))]
                arm = dict(valid=True, crops=crops.tolist())
                pair = dict(hive=hive, clip=f"synthetic-{hive}-{clip}", digest=hashlib.sha256(f"{hive}-{clip}-{n}".encode()).hexdigest(),
                    cohort="old" if n < 2 else "new", subset="old" if n < 2 else "new", endpoint_rows=rows.tolist(),
                    primary=arm, shifted=arm, shuffles=[arm]*4, near_boundary=n % 2 == 0)
                pairs.append(pair)
    return pairs


def self_test():
    pairs = _synthetic()
    fitted = fit(pairs, "W")
    assert fitted["source_pairs"] == 4
    np.testing.assert_allclose([sum(r["weight"] for r in fitted["training_records"] if r["clip"] == clip)
                               for clip in fitted["training_clips"]], [.5, .5])
    p = pairs[0]; boxes = _boxes(_rows(p)); crops = np.array(p["primary"]["crops"])
    original = predict(fitted, boxes, crops)
    swapped = predict(fitted, boxes[[0, 1, 3, 2]], crops[[0, 1, 3, 2]])
    d = difference(distances(boxes, crops))
    np.testing.assert_array_equal(d, -difference(distances(boxes[[0, 1, 3, 2]], crops[[0, 1, 3, 2]])))
    for arm, columns in INDICES.items():
        head = fitted["heads"][arm]
        np.testing.assert_allclose(original[arm]["score"], (d[columns]/head["scale"]) @ head["beta"], atol=1e-12)
        assert original[arm]["score"] == -swapped[arm]["score"] and original[arm]["decision"] == "given"
        X = np.array([r["d"] for r in fitted["training_records"]])[:, columns]/head["scale"]
        weights = np.array([r["weight"] for r in fitted["training_records"]])
        symX = np.r_[X, -X]; symy = np.r_[np.ones(len(X)), -np.ones(len(X))]; symw = np.r_[weights/2, weights/2]
        symbeta = np.linalg.solve(symX.T @ (symw[:, None]*symX)+RIDGE*np.eye(len(columns)), symX.T @ (symw*symy))
        np.testing.assert_allclose(symbeta, head["beta"], atol=1e-12)
    missing = predict(fitted, boxes, None, valid=False)
    assert missing["P"] == predict(fitted, boxes, crops)["P"]
    assert all(missing[a]["score"] == 0 and missing[a]["physical_choice"] == "given" for a in ("C", "T", "V"))
    # All four-box areas contribute to the common geometry denominator.
    uneven = boxes.copy(); uneven[3, 2:] = [16, 8]
    edge = distances(uneven)
    centres = uneven[:, :2]+uneven[:, 2:]/2
    assert edge[0, 0, 0] == np.linalg.norm(centres[0]-centres[2])/np.sqrt(uneven[:, 2]*uneven[:, 3]).mean()
    # Target/new descriptors, IDs and dates do not enter source fit or prediction.
    perturbed = json.loads(json.dumps(pairs))
    for record in perturbed:
        if record["hive"] != "W" or record["cohort"] == "new":
            record["primary"]["crops"] = np.full((4, 12), 12345.).tolist()
    assert fit(perturbed, "W") == fitted
    metadata_free = dict(schema=SCHEMA, heads=fitted["heads"])
    assert predict(metadata_free, boxes, crops) == original
    assert predict(json.loads(json.dumps(fitted)), boxes, crops) == original
    no_signal = json.loads(json.dumps(pairs))
    for record in no_signal:
        record["endpoint_rows"] = np.tile([5, 1, 4, 4, 8, 8], (4, 1)).tolist()
        record["primary"]["crops"] = np.zeros((4, 12)).tolist()
    zero_model = fit(no_signal, "W")
    assert all(all(s == 1 for s in h["scale"]) and all(b == 0 for b in h["beta"]) for h in zero_model["heads"].values())
    bundle = dict(schema="pair-texture-features-v1", prereg="0"*40, selection_sha256=SELECTION_SHA,
                  roster_sha256="synthetic", pairs=pairs)
    with TemporaryDirectory(prefix="pair-texture-model-") as tmp:
        root = Path(tmp)/"predictions"
        seal = seal_predictions(bundle, root, dict(features_sha256="synthetic"), expected_old=None)
        frozen = verify_prediction_seal(root)
        assert len(frozen["pairs"]) == 8
        for record in frozen["pairs"]:
            assert set(record["variants"]) == set(VARIANTS)
            assert record["order_bit"] == blind_order(record["digest"])[0]
    return dict(source_pairs_per_model=4, new_pairs_sealed=seal["pairs"], heads=list(INDICES),
                source_clip_weighting="equal", symmetric_ridge="equivalent", blind_matrix_replay="passed",
                all_four_box_geometry_scale="passed", missing_RGB="P available; C/T/V zero and given fallback",
                source_only_fit="target/new perturbation inert", no_ID_or_truth_predictor="passed",
                numerical_reload="exact", all_models_predictions_one_seal="passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--prereg")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif args.features and args.out and args.prereg:
        bundle, receipt = _load_features(args.features, args.prereg)
        print(json.dumps(seal_predictions(bundle, args.out, receipt), indent=2))
    else:
        parser.error("use --self-test or --features/--out/--prereg")
