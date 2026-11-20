"""CPU-only weak-box selector; running this module uses synthetic data only."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


FEATURE_NAMES = ["confidence", "log_sqrt_area", "same_frame_iou",
                 "past_cost", "future_cost", "past_log_area_ratio",
                 "future_log_area_ratio", "interpolation_error"]
BLOCK = 256


def _detections(D):
    a = np.asarray(D, dtype=float)
    if a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all():
        raise ValueError("D must be finite Nx6: frame,x1,y1,x2,y2,score")
    eligible = a[:, 5] >= .25
    if np.any(a[:, 0] != np.floor(a[:, 0])) or np.any(a[eligible, 3:5] <= a[eligible, 1:3]):
        raise ValueError("frames must be integer and eligible boxes have positive area")
    return a


def _group(indices, frames):
    order = indices[np.argsort(frames[indices], kind="stable")]
    if not len(order):
        return {}
    cuts = np.flatnonzero(np.diff(frames[order])) + 1
    return {frames[g[0]]: g for g in np.split(order, cuts)}


def features(D):
    """Return native weak row indices and eight features in FEATURE_NAMES order.

    Anchor cost is centre distance / mean sqrt(area) / actual frame gap.
    Minimize untruncated cost, then gap, then native row index. Missing
    anchors have cost 4 and absolute log area ratio 2.
    """
    a = _detections(D)
    weak = np.flatnonzero((a[:, 5] >= .25) & (a[:, 5] < .4))
    high = np.flatnonzero(a[:, 5] >= .4)
    X = np.zeros((len(weak), 8))
    if not len(weak):
        return weak, X
    eligible = a[:, 5] >= .25
    area = np.ones(len(a))
    area[eligible] = np.prod(a[eligible, 3:5] - a[eligible, 1:3], axis=1)
    size = np.sqrt(area)
    centre = (a[:, 1:3] + a[:, 3:5]) / 2
    X[:, 0] = a[weak, 5]
    X[:, 1] = np.log(size[weak])
    X[:, 3:5] = 4
    X[:, 5:7] = 2
    X[:, 7] = 4
    hg = _group(high, a[:, 0])
    # Group feature-row positions, preserving their native order within frame.
    wg = _group(np.arange(len(weak)), a[weak, 0])
    for frame, positions in wg.items():
        same = hg.get(frame, np.empty(0, dtype=int))
        candidates = []
        for direction in (-1, 1):
            rows = np.concatenate([hg.get(frame + direction * dt,
                                           np.empty(0, dtype=int))
                                   for dt in (1, 2, 3)])
            if len(rows):
                gap = np.abs(a[rows, 0] - frame)
                rows = rows[np.lexsort((rows, gap))]
            candidates.append(rows)
        for start in range(0, len(positions), BLOCK):
            p = positions[start:start + BLOCK]
            w = weak[p]
            if len(same):
                best = np.zeros(len(w))
                for j in range(0, len(same), BLOCK):
                    s = same[j:j + BLOCK]
                    extent = np.maximum(0, np.minimum(a[w, None, 3:5], a[s, 3:5])
                                        - np.maximum(a[w, None, 1:3], a[s, 1:3]))
                    intersection = extent.prod(axis=2)
                    iou = intersection / (area[w, None] + area[s] - intersection)
                    best = np.maximum(best, iou.max(axis=1))
                X[p, 2] = best
            chosen = []
            for side, rows in enumerate(candidates):
                best_cost = np.full(len(w), np.inf)
                anchor = np.full(len(w), -1, dtype=int)
                # Candidates are already ordered by gap and native index, so
                # strict improvement preserves the prescribed exact ties.
                for j in range(0, len(rows), BLOCK):
                    s = rows[j:j + BLOCK]
                    gap = np.abs(a[s, 0] - frame)
                    cost = np.linalg.norm(centre[w, None] - centre[s], axis=2)
                    cost /= (size[w, None] + size[s]) / 2
                    cost /= gap
                    k = cost.argmin(axis=1)
                    value = cost[np.arange(len(w)), k]
                    improve = value < best_cost
                    best_cost[improve] = value[improve]
                    anchor[improve] = s[k[improve]]
                present = anchor >= 0
                X[p, 3 + side] = np.minimum(best_cost, 4)
                X[p[present], 5 + side] = np.minimum(
                    np.abs(np.log(area[anchor[present]] / area[w[present]])), 2)
                chosen.append(anchor)
            present = (chosen[0] >= 0) & (chosen[1] >= 0)
            if present.any():
                past, future = chosen[0][present], chosen[1][present]
                t = (frame - a[past, 0]) / (a[future, 0] - a[past, 0])
                predicted = centre[past] + t[:, None] * (centre[future] - centre[past])
                error = np.linalg.norm(centre[w[present]] - predicted, axis=1)
                X[p[present], 7] = np.minimum(error / size[w[present]], 4)
    return weak, X


def fit(X, y, groups):
    """Fit weighted ridge logistic regression, balancing groups then classes."""
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    if X.ndim != 2 or X.shape[1] != 8 or not len(X) or not np.isfinite(X).all():
        raise ValueError("X must be nonempty finite Nx8")
    if y.shape != (len(X),) or groups.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("y must be binary and groups must match its length")
    names = np.unique(groups)
    weights = np.empty(len(y))
    for group in names:
        for label in (0, 1):
            selected = (groups == group) & (y == label)
            if not selected.any():
                raise ValueError("each group must contain both classes")
            weights[selected] = 1 / (len(names) * 2 * selected.sum())
    mean = weights @ X
    scale = np.sqrt(weights @ ((X - mean) ** 2))
    scale[scale < 1e-12] = 1
    Z = (X - mean) / scale

    def objective(theta):
        beta = theta[1:]
        score = theta[0] + Z @ beta
        residual = weights * (expit(score) - y)
        loss = weights @ (np.logaddexp(0, score) - y * score) + .005 * (beta @ beta)
        gradient = np.r_[residual.sum(), Z.T @ residual + .01 * beta]
        return loss, gradient

    result = minimize(objective, np.zeros(9), jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise RuntimeError("logistic fit failed: " + result.message)
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "intercept": float(result.x[0]), "coef": result.x[1:].tolist(),
            "feature_names": FEATURE_NAMES.copy(), "ridge": .01}


def predict(X, model):
    X = np.asarray(X, float)
    return expit(((X - np.asarray(model["mean"])) / np.asarray(model["scale"]))
                 @ np.asarray(model["coef"]) + model["intercept"])


def arms(D, model):
    """Return H/L/M/C detection arrays, retaining exact input rows and order."""
    a = _detections(D)
    original = np.asarray(D)
    weak, X = features(a)
    high = a[:, 5] >= .4
    learned = high.copy()
    learned[weak] = predict(X, model) >= .5
    matched = high.copy()
    for _, rows in _group(weak, a[:, 0]).items():
        count = int(learned[rows].sum())
        ranked = rows[np.lexsort((rows, -a[rows, 5]))]
        matched[ranked[:count]] = True
    return {"H": original[high].copy(), "L": original[a[:, 5] >= .25].copy(),
            "M": original[learned].copy(), "C": original[matched].copy()}


def _self_test():
    import json

    def box(frame, x, score=.5, width=2):
        return [frame, x - width / 2, -width / 2, x + width / 2, width / 2, score]

    D = np.array([box(4, 0, .3), box(3, -2), box(2, 4), box(5, 2),
                  box(4, 0), box(4, 10, .2), box(4, 0, .25)])
    rows, X = features(D)
    assert rows.tolist() == [0, 6] and X.shape == (2, 8)
    # Past costs tie at 1; dt=1 wins. Anchors interpolate exactly.
    np.testing.assert_allclose(X[0], [.3, np.log(2), 1, 1, 1, 0, 0, 0])
    # Equal cost/gap: original row wins; asymmetry reveals chosen anchor.
    tie = np.array([box(4, 0, .3), box(3, -2), box(3, 2), box(5, 2)])
    assert features(tie)[1][0, 7] == 0
    tie[[1, 2]] = tie[[2, 1]]
    assert features(tie)[1][0, 7] == 1
    # Candidate choice uses uncapped cost; a later block can still improve it.
    blocked = np.array([box(4, 0, .3)] + [box(3, 40)] * BLOCK
                       + [box(3, 20, width=4), box(5, -20, width=4)])
    bx = features(blocked)[1][0]
    np.testing.assert_allclose(bx[3:7], [4, 4, np.log(4), np.log(4)])
    assert bx[7] == 0
    # Unequal actual frame gaps must be used for interpolation.
    uneven = np.array([box(4, 0, .3), box(1, -6), box(5, 2)])
    assert features(uneven)[1][0, 7] == 0
    gap = np.array([box(4, 0, .3), box(1, -6), box(8, 0)])
    np.testing.assert_allclose(features(gap)[1][0, 3:], [1, 4, 0, 2, 4])
    isolated = features(np.array([box(4, 0, .3)]))[1][0]
    np.testing.assert_allclose(isolated[2:], [0, 4, 4, 2, 2, 4])
    assert features(np.array([box(1, 0)]))[1].shape == (0, 8)
    assert features(np.empty((0, 6)))[1].shape == (0, 8)
    rng = np.random.default_rng(185)
    train = rng.normal(size=(80, 8))
    y = np.tile([0, 1], 40)
    train[:, 0] += 2 * y
    groups = np.repeat(["W", "G"], 40)
    model = json.loads(json.dumps(fit(train, y, groups)))
    assert (predict(train, model)[y == 1].mean() > predict(train, model)[y == 0].mean())
    # Repeating a whole group does not change its total training weight.
    idx = np.r_[np.arange(80), np.arange(40)]
    repeated = fit(train[idx], y[idx], groups[idx])
    np.testing.assert_allclose(model["coef"], repeated["coef"], atol=1e-6)
    try:
        fit(train[:40], np.zeros(40), groups[:40])
    except ValueError:
        pass
    else:
        raise AssertionError("single-class group accepted")
    controlled = {"mean": [0.] * 8, "scale": [1.] * 8,
                  "coef": [0., 0., 0., -1., 0., 0., 0., 0.], "intercept": 1.5}
    data = np.array([box(4, 0, .30), box(4, 20, .39), box(4, 30, .39),
                     box(3, 0), box(4, 50, .1)])
    result = arms(data, controlled)
    # Subthreshold degenerate geometry is inert, including inserted native rows.
    inert = np.array([[4, 0, 0, 2, 0, .1], [3, 0, 0, 2, -1, .249]])
    augmented = np.concatenate([inert[:1], data, inert[1:]])
    with np.errstate(all="raise"):
        before_rows, before_X = features(data)
        after_rows, after_X = features(augmented)
        np.testing.assert_array_equal(after_rows, before_rows + 1)
        np.testing.assert_array_equal(after_X, before_X)
        augmented_result = arms(augmented, controlled)
    for arm in result:
        np.testing.assert_array_equal(augmented_result[arm], result[arm])
    np.testing.assert_array_equal(result["M"], data[[0, 3]])
    np.testing.assert_array_equal(result["C"], data[[1, 3]])
    for frame in np.unique(data[:, 0]):
        assert (result["C"][:, 0] == frame).sum() == (result["M"][:, 0] == frame).sum()
    for empty in (np.empty((0, 6)), np.array([box(1, 0)]), np.array([box(1, 0, .3)])):
        assert all(v.shape[1] == 6 for v in arms(empty, model).values())
    print("selector synthetic self-test passed")


if __name__ == "__main__":
    _self_test()
