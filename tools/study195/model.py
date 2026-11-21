"""Source-trained whole-video three-member routing, numpy only.

counts: (member=3,threshold=2,official_window=J,direction=2). Threshold0 is
low(.40),1 high(.50); keep EVERY supplied official bin including all30 offsets.
Deployment accepts counts and a frozen numeric model only, never a hive/label.
Running this module uses synthetic event dates only and opens no field data.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


STATES = tuple(itertools.product((0, 1), repeat=3))
FEATURE_NAMES = ["A", "D", "s_in", "s_out", "A_wide", "D_wide", "k_high_fraction"]
D_INDICES = (1, 2, 3, 5, 6)
RIDGE = .01
TIE_ATOL = 1e-12
SCHEMA = "whole-member-router-model-v1"


def _integer_array(value, shape, name):
    a = np.asarray(value)
    if (a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all()
            or np.any(a < 0) or np.any(a != np.floor(a)) or np.any(a >= 2**63)):
        raise ValueError(f"{name} must be finite nonnegative integer counts with shape{shape}")
    return a.astype(np.int64)


def _counts(counts):
    a = np.asarray(counts)
    if a.ndim != 4 or a.shape[:2] != (3, 2) or a.shape[3] != 2:
        raise ValueError("counts must have shape(3,2,J,2)")
    return _integer_array(a, a.shape, "counts")


def _sha(a):
    a = np.asarray(a, dtype="<i8", order="C")
    return hashlib.sha256(json.dumps(list(a.shape)).encode() + b"\n" + a.tobytes()).hexdigest()


def selected_counts(counts, state):
    """Return the three complete member streams for an integer state0..7."""
    a = _counts(counts)
    if isinstance(state, bool) or int(state) != state or not 0 <= state < 8:
        raise ValueError("state must be an integer0..7")
    return a[np.arange(3), STATES[int(state)]].copy()


def state_features(counts):
    c = _counts(counts)
    six = c.reshape(6, c.shape[2], 2)
    u = np.median(six, axis=0)
    B = max(90., 3 * float(u.sum()))
    wide = np.ptp(six, axis=0) >= 2
    chosen = np.stack([c[np.arange(3), bits] for bits in STATES]).astype(float)
    t = chosen - u[None, None]
    s = t.sum(axis=1)
    z = np.column_stack([np.abs(s).sum(axis=(1, 2)), np.abs(t).sum(axis=(1, 2, 3)),
                         s[:, :, 0].sum(axis=1), s[:, :, 1].sum(axis=1),
                         (np.abs(s)*wide[None]).sum(axis=(1, 2)),
                         (np.abs(t)*wide[None, None]).sum(axis=(1, 2, 3))]) / B
    z = np.column_stack([z, np.sum(STATES, axis=1) / 3])
    return dict(states=STATES, z=z, centered_z=z-z.mean(axis=0), B=B, u=u, wide=wide)


def true_costs(counts, gw):
    c = _counts(counts)
    gt = _integer_array(gw, (c.shape[2], 2), "gw")
    return np.array([np.abs(c[np.arange(3), bits].sum(axis=0)-3*gt).sum()
                     for bits in STATES], dtype=np.int64)


def _key(record):
    hive, group, clip = record["hive"], record["group_id"], record["clip"]
    if (not isinstance(hive, str) or not hive or not isinstance(clip, str) or not clip
            or isinstance(group, bool) or not isinstance(group, (str, int, np.integer))):
        raise ValueError("record requires nonempty hive/clip and scalar group_id")
    group = int(group) if isinstance(group, np.integer) else group
    return hive, json.dumps(group), clip


def fit(records, heldout=None):
    """Fit both source-only ridge heads; returns JSON-safe model/audit numbers.

    Equal hives -> equal groups in each hive -> clip weight B/sum(group B).
    Each video weight is shared by its8 states. Weighted squared loss plus
    .01*sum(beta**2), no intercept. J prior=A-centred; D prior=D-centred.
    heldout is an optional explicit source-boundary rejection, not a feature.
    """
    records = sorted(list(records), key=_key)
    if not records or len({_key(r) for r in records}) != len(records):
        raise ValueError("nonempty distinct(hive,group_id,clip) source records required")
    if heldout is not None and any(r["hive"] == heldout for r in records):
        raise ValueError("heldout hive must not occur in source training records")
    features = [state_features(r["counts"]) for r in records]
    losses = [true_costs(r["counts"], r["gw"]) for r in records]
    hives = sorted({r["hive"] for r in records})
    groups = {h: {_key(r)[1] for r in records if r["hive"] == h} for h in hives}
    B_totals = {}
    for record, f in zip(records, features):
        hive, group, _ = _key(record)
        B_totals[hive, group] = B_totals.get((hive, group), 0.) + f["B"]
    video_weights = np.array([f["B"] / B_totals[_key(r)[:2]] / len(groups[r["hive"]]) / len(hives)
                              for r, f in zip(records, features)])
    if abs(video_weights.sum()-1) > 1e-12:
        raise RuntimeError("source video weights failed to sum to1")
    Z = np.stack([f["centered_z"] for f in features])
    y = np.stack([(loss-loss.mean()) / f["B"] for loss, f in zip(losses, features)])
    state_weights = np.repeat(video_weights/8, 8)
    flat_Z = Z.reshape(-1, 7)
    heads = {}
    for name, indices, prior in (("J", tuple(range(7)), 0), ("D", D_INDICES, 1)):
        raw = flat_Z[:, indices]
        scale = np.sqrt(state_weights @ (raw**2))
        scale[scale == 0] = 1.
        X = raw / scale
        residual = (y-Z[:, :, prior]).reshape(-1)
        normal = X.T @ (state_weights[:, None]*X) + RIDGE*np.eye(len(indices))
        rhs = X.T @ (state_weights*residual)
        beta = np.linalg.solve(normal, rhs)
        error = X @ beta-residual
        objective = float(state_weights @ (error**2) + RIDGE*(beta @ beta))
        gradient = 2*(X.T @ (state_weights*error) + RIDGE*beta)
        if np.max(np.abs(gradient), initial=0) > 1e-8 or not np.isfinite(beta).all():
            raise RuntimeError("ridge normal-equation validation failed")
        heads[name] = dict(feature_indices=list(indices), feature_names=[FEATURE_NAMES[i] for i in indices],
                           prior_index=prior, prior=FEATURE_NAMES[prior], scale=scale.tolist(), beta=beta.tolist(),
                           ridge=RIDGE, intercept=0., objective=objective, max_abs_gradient=float(np.max(np.abs(gradient), initial=0)))
    audit = []
    for record, f, loss, yy, weight in zip(records, features, losses, y, video_weights):
        audit.append(dict(hive=record["hive"], group_id=int(record["group_id"]) if isinstance(record["group_id"], np.integer) else record["group_id"],
                          clip=record["clip"], B=f["B"], video_weight=float(weight), state_weight=float(weight/8),
                          raw_z=f["z"].tolist(), centered_z=f["centered_z"].tolist(),
                          true_cost=loss.tolist(), y=yy.tolist(),
                          counts_sha256=_sha(_counts(record["counts"])),
                          gw_sha256=_sha(_integer_array(record["gw"], (_counts(record["counts"]).shape[2], 2), "gw"))))
    audit_hash = hashlib.sha256(json.dumps(audit, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return dict(schema=SCHEMA, states=[list(s) for s in STATES], feature_names=FEATURE_NAMES.copy(),
                J=heads["J"], D=heads["D"], heldout=heldout, training_sources=hives,
                training_records=audit, records_hash=audit_hash,
                weighting="equal hive -> equal group -> clip B/sum(group B); states /8", tie_atol=TIE_ATOL)


def choose(scores):
    scores = np.asarray(scores, float)
    if scores.shape != (8,) or not np.isfinite(scores).all():
        raise ValueError("eight finite scores required")
    minimum = float(scores.min())
    tied = np.flatnonzero(np.abs(scores-minimum) <= TIE_ATOL)
    state = min(tied, key=lambda k: (sum(STATES[k]), STATES[k]))
    return dict(state=int(state), bits=list(STATES[state]), scores=scores.tolist(),
                score=float(scores[state]), tied_states=tied.tolist())


def predict(counts, model):
    """Pure deployment: only counts and frozen numbers, no source labels/IDs."""
    if model.get("schema") != SCHEMA:
        raise ValueError("unknown whole-member model schema")
    f = state_features(counts)
    result = dict(M=choose(f["z"][:, 0]))
    for name, expected_indices, prior in (("J", tuple(range(7)), 0), ("D", D_INDICES, 1)):
        head = model[name]
        indices = tuple(head["feature_indices"])
        scale, beta = np.asarray(head["scale"], float), np.asarray(head["beta"], float)
        if (indices != expected_indices or head["prior_index"] != prior or scale.shape != (len(indices),)
                or beta.shape != scale.shape or not np.isfinite(scale).all() or not np.isfinite(beta).all()
                or np.any(scale <= 0)):
            raise ValueError("invalid frozen head coefficients/standardization")
        score = f["centered_z"][:, prior] + (f["centered_z"][:, indices]/scale) @ beta
        result[name] = choose(score)
    return result


def save(model, path):
    Path(path).write_text(json.dumps(model, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def load(path):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("schema") != SCHEMA:
        raise ValueError("unknown model schema")
    return model


def _windows(events, nf=1000):
    """Independent synthetic native-date30-offset bin construction."""
    rows = []
    for offset in range(30):
        counts = np.zeros(((nf+25*offset+749)//750, 2), int)
        for frame, direction in events:
            counts[(frame-1+25*offset)//750, direction] += 1
        rows.extend(counts)
    return np.array(rows)


def selfcheck():
    from tempfile import TemporaryDirectory
    truth = [(200, 0), (800, 1)]
    streams = [[truth+[(225, 0)], truth], [truth, [(800, 1)]], [truth+[(825, 1)], truth]]
    counts = np.array([[_windows(events) for events in member] for member in streams])
    gw = _windows(truth)
    f = state_features(counts)
    costs = true_costs(counts, gw)
    independent = []
    for state, bits in enumerate(STATES):
        direct = sum((_windows(streams[m][bits[m]]) for m in range(3)), np.zeros_like(gw))
        independent.append(int(sum(abs(int(p)-3*int(g)) for p, g in zip(direct.ravel(), gw.ravel()))))
        np.testing.assert_array_equal(selected_counts(counts, state).sum(axis=0), direct)
    np.testing.assert_array_equal(costs, independent)
    assert costs[5] == 0 and (costs == 0).sum() == 1
    # Independent hand computation of every fixed feature on all8 states.
    for state, bits in enumerate(STATES):
        t = counts[np.arange(3), bits]-f["u"]
        s = t.sum(axis=0)
        expected = [np.abs(s).sum()/f["B"], np.abs(t).sum()/f["B"], s[:, 0].sum()/f["B"],
                    s[:, 1].sum()/f["B"], np.abs(s)[f["wide"]].sum()/f["B"],
                    np.abs(t)[:, f["wide"]].sum()/f["B"], sum(bits)/3]
        np.testing.assert_allclose(f["z"][state], expected, atol=1e-15)
    zeros = np.zeros_like(counts)
    ones = np.ones_like(counts)
    half = zeros.copy(); half[:, 1] = 1
    half_features = state_features(half)
    np.testing.assert_array_equal(half_features["u"], np.full_like(gw, .5, dtype=float))
    assert not half_features["wide"].any()
    records = [dict(hive="W", group_id=1, clip="w-a", counts=zeros, gw=np.zeros_like(gw)),
               dict(hive="W", group_id=1, clip="w-b", counts=ones, gw=np.zeros_like(gw)),
               dict(hive="W", group_id=2, clip="w-c", counts=counts, gw=gw),
               dict(hive="G", group_id=1, clip="g-a", counts=counts, gw=gw),
               dict(hive="33", group_id=1, clip="h33", counts=counts, gw=gw)]
    model = fit(records, heldout="35")
    assert model["training_sources"] == ["33", "G", "W"]
    total_by_hive = {h: sum(r["video_weight"] for r in model["training_records"] if r["hive"] == h) for h in ("33", "G", "W")}
    np.testing.assert_allclose(list(total_by_hive.values()), [1/3]*3)
    w1 = [r for r in model["training_records"] if r["hive"] == "W" and r["group_id"] == 1]
    assert abs(sum(r["video_weight"] for r in w1)-1/6) < 1e-15
    np.testing.assert_allclose([r["video_weight"]/r["B"] for r in w1], [w1[0]["video_weight"]/w1[0]["B"]]*2)
    raw = np.array([r["centered_z"] for r in model["training_records"]])
    weight = np.array([r["video_weight"] for r in model["training_records"]])
    expected_rms = np.sqrt(np.sum(raw**2 * weight[:, None, None]/8, axis=(0, 1)))
    expected_rms[expected_rms == 0] = 1
    np.testing.assert_allclose(model["J"]["scale"], expected_rms)
    for name in ("D", "J"):
        assert model[name]["max_abs_gradient"] < 1e-8
    try:
        fit(records+[dict(hive="35", group_id=1, clip="blocked", counts=counts, gw=gw)], heldout="35")
    except ValueError:
        pass
    else:
        raise AssertionError("heldout labels entered source fit")
    prediction = predict(counts, model)
    assert predict(counts, {key: model[key] for key in ("schema", "J", "D")}) == prediction
    perm = [2, 0, 1]
    permuted = predict(counts[perm], model)
    for arm in ("M", "D", "J"):
        scores = sorted(prediction[arm]["scores"])
        assert scores[1]-scores[0] > TIE_ATOL
        assert permuted[arm]["bits"] == [prediction[arm]["bits"][i] for i in perm]
    # D is additive: zero every pairwise mixed finite difference.
    d_scores = np.array(prediction["D"]["scores"]).reshape(2, 2, 2)
    for i, j in itertools.combinations(range(3), 2):
        np.testing.assert_allclose(np.diff(np.diff(d_scores, axis=i), axis=j), 0, atol=1e-12)
    assert predict(zeros, model)["M"]["bits"] == [0, 0, 0]
    assert choose(np.zeros(8))["state"] == 0
    assert choose([1, 0, 0, 1, 1, 1, 1, 1])["state"] == 1  #001 before010.
    with TemporaryDirectory(prefix="whole_member_model_") as tmp:
        path = Path(tmp)/"model.json"
        save(model, path)
        assert predict(counts, load(path)) == prediction
    reversed_fit = fit(reversed(records), heldout="35")
    assert reversed_fit == model
    return dict(scope="synthetic counts with native event dates; no field reads/training", states=[list(s) for s in STATES],
                true_cost_all_states=costs.tolist(), predicted_bits={k: prediction[k]["bits"] for k in ("M", "D", "J")},
                zero_GT="included; finite without denominator exception", member_permutation="equivariant without ties",
                hive_group_clip_weights="passed", source_only_RMS_fit="passed", heldout_guard="passed",
                additive_D="mixed differences zero", JSON_reload="exact predictions", ridge=RIDGE)


if __name__ == "__main__":
    print(json.dumps(selfcheck(), indent=2))
