"""Synthetic-only first-head residual model and coherent deletion-budget decoder.

Run with numpy/scipy installed. No field files are read or written. A caller
supplies deployment features, source residual labels, and native candidate IDs.
Directions are decoded separately. This is not a representative-track ranker.
"""
import itertools
import json

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack


def intervals(seconds):
    """All thirty offset partitions, retaining duplicate short/tail windows."""
    result = []
    for offset in range(30):
        for j in range((seconds + offset + 29) // 30):
            result.append((max(0, 30 * j - offset), min(seconds, 30 * (j + 1) - offset)))
    return result


def incidence(windows, seconds):
    rows, cols = [], []
    for j, (lo, hi) in enumerate(windows):
        if not 0 <= lo <= hi <= seconds:
            raise ValueError("windows must be contiguous half-open intervals")
        rows.extend([j] * (hi - lo))
        cols.extend(range(lo, hi))
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(windows), seconds))


def allocate_member_quota(q, n, t):
    """Allocate pooled DELETIONS q among three members in fixed seed order.

    Use proportional floors and largest integer remainders; remainder ties
    cycle from member t%3. n contains actual native candidate capacities.
    This function reads no identity, score, GT, or ranking information.
    The second head must KEEP K=n-d identities within each member separately.
    """
    n = np.asarray(n)
    if (n.shape != (3,) or not np.issubdtype(n.dtype, np.number)
            or np.iscomplexobj(n) or not np.isfinite(n).all()
            or np.any(n != np.floor(n)) or np.any(n < 0)):
        raise ValueError("n must contain three nonnegative integer capacities")
    if (isinstance(q, (bool, np.bool_)) or isinstance(t, (bool, np.bool_))
            or not np.isscalar(q) or not np.isscalar(t)
            or not np.isfinite(q) or not np.isfinite(t)
            or q != int(q) or t != int(t) or t < 0):
        raise ValueError("q and t must be integers, with t>=0")
    # Python integers keep proportional products exact, even for large counts.
    counts = [int(v) for v in n]
    total = sum(counts)
    q, t = int(q), int(t)
    if not 0 <= q <= total:
        raise ValueError("q must lie between zero and pooled capacity")
    if not total:
        return np.zeros(3, dtype=np.int64)
    deletion = [q * count // total for count in counts]
    remainder = [q * count % total for count in counts]
    priority = sorted(range(3), key=lambda i: (-remainder[i], (i - t % 3) % 3))
    for member in priority[:q - sum(deletion)]:
        deletion[member] += 1
    result = np.array(deletion, dtype=np.int64)
    if int(result.sum()) != q or any(d > count for d, count in zip(deletion, counts)):
        raise RuntimeError("member allocation violated total or capacity")
    return result


def prepare_member_quotas(q, member_candidate_ids):
    """Second-head entry: per-second, per-member keep/deletion quotas and IDs.

    Input IDs have shape [second][member][native candidate]. Decode each
    direction/group independently, then call this with its q. A representative
    head may permute identities ONLY inside a cell, selecting exactly keep[t,i].
    """
    q = np.asarray(q)
    if q.shape != (len(member_candidate_ids),):
        raise ValueError("one pooled deletion count per second required")
    if any(len(members) != 3 for members in member_candidate_ids):
        raise ValueError("exactly three members per second required")
    flat = [event for members in member_candidate_ids for ids in members for event in ids]
    if len(set(flat)) != len(flat):
        raise ValueError("native IDs must be globally unique within direction/group")
    capacity = np.array([[len(ids) for ids in members] for members in member_candidate_ids], dtype=int).reshape(-1, 3)
    deletion = np.array([allocate_member_quota(value, capacity[t], t) for t, value in enumerate(q)], dtype=int).reshape(-1, 3)
    return dict(delete=deletion, keep=capacity - deletion, capacity=capacity,
                candidate_ids=member_candidate_ids)


def fit(X, residual, sources, groups, k=5):
    """Fit source-only standardization and store distinct video exemplars.

    Each source has equal total weight, each video equal within source, each
    distinct (feature,residual) row equal within video. At prediction, choose
    k distinct nearest videos per source; never treat phases as independent
    neighbours. A video belongs to one source. k is fixed before evaluation.
    """
    X, residual = np.asarray(X, float), np.asarray(residual)
    sources, groups = np.asarray(sources, str), np.asarray(groups, str)
    if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
        raise ValueError("finite nonempty feature matrix required")
    if residual.shape != (len(X),) or sources.shape != residual.shape or groups.shape != residual.shape:
        raise ValueError("row labels must match X")
    if not np.isfinite(residual).all() or np.any(residual != np.floor(residual)) or int(k) != k or k < 1:
        raise ValueError("integer residuals and positive integer k required")
    records, weights = [], []
    for source in np.unique(sources):
        source_groups = np.unique(groups[sources == source])
        for group in source_groups:
            if len(np.unique(sources[groups == group])) != 1:
                raise ValueError("video group must belong to one source")
            use = (sources == source) & (groups == group)
            distinct = np.unique(np.column_stack([X[use], residual[use]]), axis=0)
            records.append((source, group, distinct[:, :-1], distinct[:, -1].astype(int)))
            weights.append(np.full(len(distinct), 1 / (len(np.unique(sources)) * len(source_groups) * len(distinct))))
    raw = np.concatenate([r[2] for r in records])
    weight = np.concatenate(weights)
    mean = weight @ raw
    scale = np.sqrt(weight @ ((raw - mean) ** 2))
    scale[scale < 1e-12] = 1
    return dict(mean=mean, scale=scale, k=int(k),
                records=[(s, g, (x - mean) / scale, r) for s, g, x, r in records])


def survival(model, X, maximum):
    """Return P(r>=j|x), j=1..maximum, a monotone empirical survival curve.

    Exact nearest-instance distance ties within a video are averaged. Videos
    tie by stable group name at the fixed-k boundary. All chosen videos have
    equal weights within a source; all sources have equal total weights.
    """
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != len(model["mean"]) or not np.isfinite(X).all():
        raise ValueError("invalid prediction feature matrix")
    if int(maximum) != maximum or maximum < 0:
        raise ValueError("maximum must be a nonnegative integer")
    Z = (X - model["mean"]) / model["scale"]
    out = np.zeros((len(X), maximum))
    source_names = sorted({r[0] for r in model["records"]})
    thresholds = np.arange(1, maximum + 1)
    for i, z in enumerate(Z):
        for source in source_names:
            neighbours = []
            for s, group, x, r in model["records"]:
                if s != source:
                    continue
                distance = np.sum((x - z) ** 2, axis=1)
                minimum = distance.min()
                labels = r[distance == minimum]
                neighbours.append((float(minimum), group, labels))
            chosen = sorted(neighbours, key=lambda row: (row[0], row[1]))[:model["k"]]
            for _, _, labels in chosen:
                out[i] += (labels[:, None] >= thresholds).mean(axis=0) / (len(source_names) * len(chosen))
    return np.clip(out, 0, 1)


def objective(A, q, costs):
    counts = np.rint(A @ q).astype(int)
    return float(sum(costs[w, :count].sum() for w, count in enumerate(counts)))


def decode(windows, candidate_ids, costs):
    """Exact separable convex interval LP; q counts DELETIONS, not keeps.

    y_wj is a unit segment of the window cost curve. Aq=sum_j y_wj.
    Increasing marginal costs make the optimal segments a prefix. The
    interval incidence matrix is totally unimodular; appended unit columns
    and integral bounds preserve integrality. HiGHS dual simplex returns a
    vertex, and primal/dual/integer checks certify the returned optimum.
    Within-second IDs use fixed string order, without a learned ranker.
    """
    seconds = len(candidate_ids)
    flat = [event for ids in candidate_ids for event in ids]
    if len(flat) != len(set(flat)):
        raise ValueError("native candidate IDs must be unique")
    cap = np.array([len(ids) for ids in candidate_ids], int)
    A = incidence(windows, seconds)
    bounds_per_window = np.rint(A @ cap).astype(int)
    costs = np.asarray(costs, float)
    if costs.ndim != 2 or costs.shape[0] != len(windows) or costs.shape[1] < max(bounds_per_window, default=0):
        raise ValueError("one cost curve per window, covering its actual capacity, required")
    if not np.isfinite(costs).all() or np.any(np.diff(costs, axis=1) < -1e-12):
        raise ValueError("finite nondecreasing marginal costs required")
    segment_rows = np.repeat(np.arange(len(windows)), bounds_per_window)
    segment_count = len(segment_rows)
    if not segment_count:
        return dict(q=[0] * seconds, removed_ids=[], kept_ids=flat, objective=0., dual=0.)
    S = csr_matrix((np.ones(segment_count), (segment_rows, np.arange(segment_count))),
                   shape=(len(windows), segment_count))
    B = hstack([A, -S], format="csr")
    segment_cost = np.concatenate([costs[w, :n] for w, n in enumerate(bounds_per_window)])
    c = np.r_[np.zeros(seconds), segment_cost]
    upper = np.r_[cap, np.ones(segment_count)]
    sol = linprog(c, A_eq=B, b_eq=np.zeros(len(windows)),
                  bounds=list(zip(np.zeros(len(upper)), upper)), method="highs-ds")
    if not sol.success:
        raise RuntimeError(sol.message)
    primary = sol
    # The primary-optimal face still has integral vertices. Prefer fewer
    # deletions on it; a zero expected benefit never forces a gratuitous edit.
    tie = linprog(np.r_[np.ones(seconds), np.zeros(segment_count)],
                  A_eq=vstack([B, csr_matrix(c[None, :])], format="csr"),
                  b_eq=np.r_[np.zeros(len(windows)), primary.fun],
                  bounds=list(zip(np.zeros(len(upper)), upper)), method="highs-ds")
    if not tie.success:
        raise RuntimeError("minimum-deletion tie solve: " + tie.message)
    sol = tie
    if np.max(np.abs(sol.x - np.rint(sol.x)), initial=0) > 1e-7:
        raise RuntimeError("LP did not return an integral vertex")
    q = np.rint(sol.x[:seconds]).astype(int)
    value = objective(A, q, costs)
    dual = float(upper @ primary.upper.marginals)
    if (q < 0).any() or (q > cap).any() or abs(value - primary.fun) > 1e-7 or abs(value - dual) > 1e-7:
        raise RuntimeError("primal/dual/capacity certificate failed")
    removed = [event for t, ids in enumerate(candidate_ids) for event in sorted(ids)[:q[t]]]
    removed_set = set(removed)
    return dict(q=q.tolist(), removed_ids=removed, kept_ids=[v for v in flat if v not in removed_set],
                objective=value, dual=dual)


def independent(A, cap, costs):
    """Each candidate uses its isolated deletion cost, ignoring joint harm."""
    return np.asarray(cap) * (np.asarray(A.T @ costs[:, 0]).ravel() < 0) if costs.shape[1] else np.zeros(len(cap), int)


def synthetic_clip(rng, name, morphology, seconds=40):
    """Two directions. Labels exist only inside this synthetic generator."""
    windows = intervals(seconds)
    A = incidence(windows, seconds)
    directions = []
    for direction in (0, 1):
        truth = rng.binomial(1, .20, seconds)
        members = np.array([rng.binomial(truth, .90) for _ in range(3)])
        # The held-out morphology changes the false-event process. Its label
        # is not an input feature and no target label fits the first head.
        false_rate = (.08, .16, .24)[morphology]
        false = rng.binomial(1, false_rate, (3, seconds))
        observed = members + false
        support = members * (members.sum(axis=0)[None, :] >= 2)
        if morphology == 2:
            support = support + false * (false.sum(axis=0)[None, :] >= 2)
        counts = np.asarray(A @ observed.T)
        total = counts.sum(axis=1)
        support_count = np.asarray(A @ support.sum(axis=0)).ravel()
        duration = np.array([hi - lo for lo, hi in windows], float)
        X = np.column_stack([counts, counts.max(axis=1) - counts.min(axis=1),
                             support_count / np.maximum(total, 1), duration])
        residual = (total - 3 * np.asarray(A @ truth).ravel()).astype(int)
        ids = [[f"{name}/d{direction}/m{m}/t{t}/e{j}" for m in range(3)
                for j in range(int(observed[m, t]))] for t in range(seconds)]
        directions.append(dict(X=X, residual=residual, ids=ids, truth_total=int(truth.sum())))
    return windows, directions


def selfcheck():
    rng = np.random.default_rng(19001)
    allocation_cases = 0
    for n_tuple in itertools.product(range(4), repeat=3):
        n = np.array(n_tuple)
        total = int(n.sum())
        for q in range(total + 1):
            possible = [d for d in itertools.product(*(range(c + 1) for c in n_tuple)) if sum(d) == q]
            for t in range(3):
                d = allocate_member_quota(q, n, t)
                # Independent exhaustive criterion: nearest proportional
                # allocation in squared error, then cyclic member priority.
                cyclic = [(t + j) % 3 for j in range(3)]
                best = min(possible, key=lambda v: (
                    sum((total * v[i] - q * n_tuple[i]) ** 2 for i in range(3)),
                    tuple(-v[i] for i in cyclic)))
                np.testing.assert_array_equal(d, best)
                np.testing.assert_array_equal(d, allocate_member_quota(q, n, t + 3))
                assert int(d.sum()) == q and np.all(d >= 0) and np.all(d <= n)
                allocation_cases += 1
    for t in range(3):
        expected = np.zeros(3, int)
        expected[t] = 1
        np.testing.assert_array_equal(allocate_member_quota(1, [1, 1, 1], t), expected)
    for bad in ((-1, [1, 1, 1], 0), (4, [1, 1, 1], 0), (1, [0, 0, 0], 0),
                (0, [1, -1, 1], 0), (1.5, [1, 1, 1], 0), (0, [1, 1, 1], -1)):
        try:
            allocate_member_quota(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid allocation accepted")
    # A second head exchanges identities only inside fixed member/second K.
    member_ids = [[[f"t{t}/m{m}/e{j}" for j in range(3)] for m in range(3)] for t in range(3)]
    prepared = prepare_member_quotas([3, 3, 3], member_ids)
    member_counts = []
    for reverse in (False, True):
        selected = [[sorted(ids, reverse=reverse)[:prepared["keep"][t, m]]
                     for m, ids in enumerate(members)] for t, members in enumerate(member_ids)]
        member_counts.append(np.array([[len(ids) for ids in members] for members in selected]))
    assert any(sorted(ids)[:2] != sorted(ids, reverse=True)[:2] for members in member_ids for ids in members)
    np.testing.assert_array_equal(member_counts[0], member_counts[1])
    member_A = incidence(intervals(3), 3)
    truth = np.array([1, 0, 2])
    for m in range(3):
        a, b = (counts[:, m] for counts in member_counts)
        # Member W uses its own unchanged GT denominator; F1's denominator
        # is N_pred+N_GT, fixed even though identity-dependent TP may change.
        assert np.abs(member_A @ a - member_A @ truth).sum() == np.abs(member_A @ b - member_A @ truth).sum()
        assert int(a.sum() + truth.sum()) == int(b.sum() + truth.sum())
    assert prepare_member_quotas([], [])["keep"].shape == (0, 3)
    brute_count = 0
    for seconds in range(1, 6):
        windows = intervals(seconds)
        A = incidence(windows, seconds)
        for trial in range(8):
            cap = rng.integers(0, 3, seconds)
            maximum = int(max(A @ cap, default=0))
            costs = np.sort(rng.uniform(-1, 1, (len(windows), maximum)), axis=1)
            ids = [[f"t{t}/e{j}" for j in range(n)] for t, n in enumerate(cap)]
            result = decode(windows, ids, costs)
            best = min(objective(A, np.array(q), costs)
                       for q in itertools.product(*(range(n + 1) for n in cap)))
            assert abs(result["objective"] - best) < 1e-7
            brute_count += 1
    # Three native events, one window, exact residual=1: delete one, not three.
    candidates = [["a", "b", "c"]]
    costs = np.array([[-1., 1., 1.]])
    result = decode([(0, 1)], candidates, costs)
    assert result["q"] == [1]
    A = incidence([(0, 1)], 1)
    assert independent(A, [3], costs).tolist() == [3]
    assert abs(1 - result["q"][0]) == 0 and abs(1 - 3) == 2
    # Same-second identity exchanges leave all window counts and W identical.
    windows = intervals(3)
    A = incidence(windows, 3)
    q1 = np.array([1, 0, 0])
    remove_a, remove_b = {"a"}, {"b"}
    events = [(0, "a"), (0, "b"), (2, "c")]
    count = lambda removed: np.bincount([t for t, event in events if event not in removed], minlength=3)
    np.testing.assert_array_equal(A @ count(remove_a), A @ count(remove_b))
    r = rng.integers(-2, 3, len(windows))
    assert np.abs(r - A @ q1).sum() == np.abs(r - A @ (count(set()) - count(remove_b))).sum()
    zero = decode(windows, [[], [], []], np.empty((len(windows), 0)))
    assert zero["q"] == [0, 0, 0] and not zero["removed_ids"]
    assert decode([(0, 1)], [["a", "b"]], np.zeros((1, 2)))["q"] == [0]
    # Source/group balancing: duplicate phases and entire copies are inert.
    X = np.array([[0.], [1.], [0.], [1.]])
    r = np.array([1, 2, -1, 0])
    sources = np.array(["W", "W", "G", "G"])
    groups = np.array(["wv", "wv", "gv", "gv"])
    m1 = fit(X, r, sources, groups, k=1)
    idx = np.r_[np.arange(4), np.zeros(20, int)]
    m2 = fit(X[idx], r[idx], sources[idx], groups[idx], k=1)
    np.testing.assert_allclose(m1["mean"], m2["mean"])
    np.testing.assert_array_equal(survival(m1, X, 4), survival(m2, X, 4))
    # Exact same distance: sources with unequal video counts have equal mass.
    m3 = fit(np.zeros((4, 1)), [1, 1, 1, -1], ["W", "W", "W", "G"],
             ["w1", "w2", "w3", "g1"], k=5)
    assert survival(m3, [[0.]], 1)[0, 0] == .5
    return dict(enumeration_cases=brute_count, member_allocation_exhaustive_cases=allocation_cases,
                member_rotation_zero_full_capacity="passed", fixed_member_K_W_and_F1_denominator="passed",
                r1_three_events_deleted=result["q"][0],
                isolated_three_events_deleted=3, same_second_exchange="identical W",
                zero_capacity="passed", duplicate_phase_and_source_balance="passed")


def benchmark():
    train_rng, test_rng = np.random.default_rng(19002), np.random.default_rng(19003)
    train_X, train_r, source_rows, group_rows = [], [], [], []
    for source, morphology in (("W", 0), ("G", 1)):
        for clip in range(12):
            group = f"source_{source}_{clip}"
            _, directions = synthetic_clip(train_rng, group, morphology)
            for d in directions:
                train_X.extend(d["X"])
                train_r.extend(d["residual"])
                source_rows.extend([source] * len(d["X"]))
                group_rows.extend([group] * len(d["X"]))
    model = fit(train_X, train_r, source_rows, group_rows, k=5)
    # Break feature/label association while retaining each source's labels.
    shuffle_rng = np.random.default_rng(19004)
    shuffled_r = np.array(train_r)
    for source in ("W", "G"):
        take = np.flatnonzero(np.array(source_rows) == source)
        shuffled_r[take] = shuffle_rng.permutation(shuffled_r[take])
    shuffled_model = fit(train_X, shuffled_r, source_rows, group_rows, k=5)
    arm_names = ("keep", "independent", "coherent", "coherent_shuffled_labels")
    rows = []
    for clip in range(12):
        group = f"independent_unseen_{clip}"
        assert group not in group_rows
        windows, directions = synthetic_clip(test_rng, group, 2)
        totals = dict.fromkeys(arm_names, 0.)
        broken_totals = dict.fromkeys(arm_names, 0.)
        deleted = dict.fromkeys(arm_names, 0)
        truth = 0
        broken_truth = 0
        for d in directions:
            cap = np.array([len(ids) for ids in d["ids"]])
            A = incidence(windows, len(cap))
            maximum = int(max(A @ cap, default=0))
            probability = survival(model, d["X"], maximum)
            assert np.all(np.diff(probability, axis=1) <= 1e-12)
            costs = 1 - 2 * probability
            result = decode(windows, d["ids"], costs)
            shuffled_costs = 1 - 2 * survival(shuffled_model, d["X"], maximum)
            shuffled_result = decode(windows, d["ids"], shuffled_costs)
            q = {"keep": np.zeros(len(cap), int), "independent": independent(A, cap, costs),
                 "coherent": np.array(result["q"]),
                 "coherent_shuffled_labels": np.array(shuffled_result["q"])}
            assert len(result["removed_ids"]) == sum(result["q"])
            assert len(result["kept_ids"]) + len(result["removed_ids"]) == sum(cap)
            for arm, removed in q.items():
                totals[arm] += float(np.abs(d["residual"] - A @ removed).sum())
                deleted[arm] += int(removed.sum())
                # Adversarial alternate world with identical deployable inputs:
                # every native event is a different true bee, seen by only one
                # member. GT_t=cap_t; pooled residual=-2*cap, so any deletion hurts.
                broken_r = -2 * np.asarray(A @ cap).ravel()
                broken_totals[arm] += float(np.abs(broken_r - A @ removed).sum())
            truth += d["truth_total"]
            broken_truth += int(cap.sum())
        denominator = 30 * 3 * truth
        rows.append(dict(clip=group, truth_events=truth, deleted=deleted,
                         W_percent={arm: 100 * value / denominator for arm, value in totals.items()},
                         broken_relation_W_percent={arm: 100 * value / (30 * 3 * broken_truth)
                                                    for arm, value in broken_totals.items()}))
    return dict(seed_train=19002, seed_holdout=19003, seed_label_shuffle=19004, k_per_source=5,
                source_videos=24, holdout_videos=12, directions_per_video=2,
                morphology="holdout false-event process absent from source generator",
                truth_design="two directions; true Bernoulli(.20)/second; each of three members detects with .90 probability",
                interference_design="independent member false events .08/.16 per second for source W/G; unseen holdout .24 and coincident false events contribute apparent support",
                negative_control="permute source training residual labels within source, retaining feature rows and source label marginals; never fit target labels",
                broken_relation_control="same frozen holdout features/candidates; alternate GT_t=pooled native capacity_t, each native event is a distinct true bee seen by one member; all deletions now harmful; no refit",
                W_definition="100 * sum_{direction,all 30 phases,windows}|sum_member_count - deletion_count - 3*GT| / (30*3*sum_direction_GT)",
                mean_clip_W_percent={arm: float(np.mean([r["W_percent"][arm] for r in rows]))
                                     for arm in arm_names},
                mean_broken_relation_W_percent={arm: float(np.mean([r["broken_relation_W_percent"][arm] for r in rows]))
                                                for arm in arm_names},
                total_deleted={arm: sum(r["deleted"][arm] for r in rows)
                               for arm in arm_names},
                coherent_better_than_keep=sum(r["W_percent"]["coherent"] < r["W_percent"]["keep"] for r in rows),
                coherent_better_than_independent=sum(r["W_percent"]["coherent"] < r["W_percent"]["independent"] for r in rows),
                per_clip=rows)


if __name__ == "__main__":
    print(json.dumps(dict(checks=selfcheck(), synthetic_holdout=benchmark(),
                         limitation="Synthetic only; no GT-free scene-generalization claim. Native ID ranking is fixed."),
                     indent=2))
