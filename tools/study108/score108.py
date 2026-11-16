"""Score retained anchor events, with new maximum matchings and direction-budget controls."""
import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from tools.study103.swap_score import counts_from_events


TRIOS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))


def _event_table(records, nframes):
    if not len(records):
        return {}
    pairs = np.array([r[:2] for r in records])
    values = pairs.astype(np.int64)
    if values.shape != (len(records), 2) or not np.array_equal(pairs, values):
        raise ValueError("Expected unique integer IDs and signed, 1-based frames.")
    ids, signed = values.T
    if len(set(ids)) != len(ids) or np.any(signed == 0) or np.any(abs(signed) > nframes):
        raise ValueError("An event ID was duplicated or its frame was outside this clip.")
    return dict(sorted((int(i), int(f)) for i, f in values))


def _windows(signed, nframes):
    frames = np.asarray(signed, dtype=np.int64)
    return counts_from_events(np.c_[np.abs(frames), np.sign(frames)], nframes)


def _window_metric(totals, bins, gt_totals, gt_bins):
    err = np.abs(bins - gt_bins).sum(axis=0) * (100 / gt_bins.sum())
    delta = totals - gt_totals
    return dict(window=float(err.sum()), window_by_direction=err.tolist(),
                totals=totals.tolist(), bias=delta.tolist(),
                floor=float(100 * abs(delta).sum() / gt_totals.sum()))


def _graph(edges, predictions, truth):
    px = {p: i for i,p in enumerate(predictions)}
    gx = {g: i for i,g in enumerate(truth)}
    ij = set()
    for p, g in edges:
        if p not in px or g not in gx or (predictions[p] > 0) != (truth[g] > 0):
            raise ValueError("Official edge refers to a missing or opposite-direction event.")
        ij.add((px[p], gx[g]))
    rows, cols = zip(*sorted(ij)) if ij else ([], [])
    return csr_matrix((np.ones(len(ij), dtype=np.int8), (rows, cols)),
                      shape=(len(px), len(gx)))


def _match(graph, indices, signed, totals, gt_totals):
    selected = graph[indices]
    if selected.nnz:
        assigned = maximum_bipartite_matching(selected, perm_type="column")
        hit = signed[indices][assigned >= 0]
        tp_dir = np.array([(hit > 0).sum(), (hit < 0).sum()])
    else:
        tp_dir = np.zeros(2, dtype=np.int64)
    tp = int(tp_dir.sum())
    return dict(tp=tp, fp=int(totals.sum())-tp, fn=int(gt_totals.sum())-tp,
                f1=float(2 * tp / (totals.sum() + gt_totals.sum())),
                tp_by_direction=tp_dir.tolist(), fp_by_direction=(totals-tp_dir).tolist(),
                fn_by_direction=(gt_totals-tp_dir).tolist())


def quota_masks(predictions, retained):
    """Freeze the 99 predicted-ID selections before any official edges or GT labels are read."""
    by_id = dict(sorted((int(row[0]), int(row[1])) for row in predictions))
    keep = set(retained)
    if len(by_id) != len(predictions) or len(keep) != len(retained) or not keep.issubset(by_id):
        raise ValueError("Quota needs distinct predicted IDs and a known retained subset.")
    pools = [np.array([p for p,f in by_id.items() if sign*f > 0], dtype=np.int64)
             for sign in (1, -1)]
    budget = [sum(int(p) in keep for p in pool) for pool in pools]
    rng = np.random.Generator(np.random.PCG64(108))
    masks = []
    for _ in range(99):
        selected = [rng.choice(pool, size=k, replace=False) for pool,k in zip(pools,budget)]
        masks.append(np.sort(np.concatenate(selected)).tolist())
    return masks


def score_anchor(case, trio, support, quotas=None):
    """Score one anchor. Direction order is in/out. All 99 retained subsets are archived."""
    nframes = case["nframes"]
    pred = _event_table(case["predictions"], nframes)
    gt = _event_table(case["gt"], nframes)
    if not gt:
        raise ValueError("Window error requires at least one ground-truth event.")
    gt_bins = _windows(list(gt.values()), nframes)
    gt_totals = gt_bins.sum(axis=0) // 30
    ids = np.asarray(list(pred), dtype=np.int64)
    signed = np.asarray(list(pred.values()), dtype=np.int64)
    graphs = {name: _graph(case["edges" + name], pred, gt) for name in ("05", "03")}
    def evaluate(indices):
        frames = signed[indices]
        totals = np.array([(frames > 0).sum(), (frames < 0).sum()])
        out = _window_metric(totals, _windows(frames, nframes), gt_totals, gt_bins)
        for name, graph in graphs.items():
            out[name] = _match(graph, indices, signed, totals, gt_totals)
        return out
    all_indices = np.arange(len(ids))
    full = evaluate(all_indices)
    retained = set(support["retained"])
    if len(retained) != len(support["retained"]) or not retained.issubset(pred):
        raise ValueError("Mask retains duplicate or unknown event IDs.")
    kept_indices = np.array([i for i, pid in enumerate(ids) if pid in retained], dtype=np.int64)
    kept = evaluate(kept_indices)
    size = len(kept_indices)
    removed = len(ids) - size
    losses = {}
    for name in graphs:
        orig, filt = full[name], kept[name]
        lost_tp = orig["tp"] - filt["tp"]
        lost_fp = orig["fp"] - filt["fp"]
        if lost_tp < 0 or lost_fp < 0 or lost_tp + lost_fp != removed:
            raise ValueError("Retaining a prediction subset violated matching monotonicity.")
        # Compare integer cross-products; either official maximum can reassign surviving edges.
        f1_differs = removed * orig["tp"] - (len(gt) + len(ids)) * lost_tp
        losses[name] = dict(lost_tp=lost_tp, lost_fp=lost_fp,
                            loss_fraction=lost_tp / removed if removed else None,
                            f1_improved=bool(f1_differs > 0),
                            f1_equal=bool(f1_differs == 0))
    groups = [_event_table(table, nframes) for table in trio]
    if len(groups) != 3 or pred not in groups:
        raise ValueError("E3 requires the anchor and both buddies from the same clip.")
    mean_bins = np.mean([_windows(list(group.values()), nframes) for group in groups], axis=0)
    e3 = _window_metric(mean_bins.sum(axis=0) / 30, mean_bins, gt_totals, gt_bins)
    expected_masks = quota_masks(case["predictions"], retained)
    if quotas is None:
        quotas = support.get("quota_masks", expected_masks)
    if quotas != expected_masks:
        raise ValueError("Supplied quota subsets do not reproduce the prediction-only budget.")
    by_id = {int(pid): i for i, pid in enumerate(ids)}
    quota = []
    for mask in quotas:
        indices = np.array([by_id[pid] for pid in mask], dtype=np.int64)
        row = evaluate(indices)
        row["ids"] = ids[indices].tolist()
        quota.append(row)
    shifted = support["phase_counts"]
    if len(shifted) != 99 or any(type(k) not in (int, np.int64, np.int32) or
                                 not 0 <= k <= len(ids) for k in shifted):
        raise ValueError("Expected 99 integer phase-support counts within the anchor size.")
    phase_wins = sum(size > int(k) for k in shifted)
    median_tp = int(np.median([q["05"]["tp"] for q in quota]))
    quota_medians = dict(tp05=median_tp, tp03=int(np.median([q["03"]["tp"] for q in quota])),
                         window=float(np.median([q["window"] for q in quota])),
                         f105=float(np.median([q["05"]["f1"] for q in quota])),
                         f103=float(np.median([q["03"]["f1"] for q in quota])))
    return dict(full=full, kept=kept, e3=e3, quota=quota, quota_medians=quota_medians,
                gt_totals=gt_totals.tolist(), removed=removed, retained=size,
                loss05=losses["05"], loss03=losses["03"], quota_median05_tp=median_tp,
                phase_counts=[int(v) for v in shifted], phase_wins=phase_wins,
                source_anchor_pass=bool(removed >= 10 and size >= 10 and phase_wins >= 95 and
                                        kept["05"]["tp"] >= median_tp + 2))


def _ordered(rows):
    if len(rows) != 9 or set(map(str, rows)) != set(map(str, range(9))):
        raise ValueError("A fold requires all 9 named anchor seeds.")
    return [rows[s] if s in rows else rows[str(s)] for s in range(9)]


def source_gate(rows):
    anchors = [bool(r["source_anchor_pass"]) for r in _ordered(rows)]
    group_votes = [sum(anchors[s] for s in group) for group in TRIOS]
    return dict(anchor_pass=anchors, group_votes=group_votes,
                group_pass=[v >= 2 for v in group_votes],
                passes=sum(v >= 2 for v in group_votes) >= 2)


def target_summary(rows):
    anchors = _ordered(rows)
    def contrasts(group):
        def means(key, other):
            diffs = [other(r) - r["kept"][key] for r in group]
            return dict(gain=float(np.mean(diffs)), wins=sum(d > 1e-9 for d in diffs))
        def f1(other):
            diffs = [r["kept"]["05"]["f1"] - other(r) for r in group]
            return dict(gain=float(np.mean(diffs)), wins=sum(d > 1e-9 for d in diffs))
        return dict(window_vs_p0=means("window", lambda r:r["full"]["window"]),
                    window_vs_e3=means("window", lambda r:r["e3"]["window"]),
                    window_vs_quota=means("window", lambda r:r["quota_medians"]["window"]),
                    f1_vs_p0=f1(lambda r:r["full"]["05"]["f1"]),
                    f1_vs_quota=f1(lambda r:r["quota_medians"]["f105"]))
    designated = [anchors[group[0]] for group in TRIOS]
    primary = contrasts(designated)
    tp_wins = sum(r["kept"]["05"]["tp"] > r["quota_median05_tp"] for r in designated)
    window_pass = (primary["window_vs_e3"]["gain"] >= 2 and
                   primary["window_vs_e3"]["wins"] >= 2 and primary["window_vs_p0"]["gain"] > 1e-9)
    f1_pass = (primary["f1_vs_p0"]["gain"] >= 0.02 and
               primary["f1_vs_p0"]["wins"] >= 2 and tp_wins >= 2)
    primary.update(tp_wins_vs_quota=tp_wins, window_pass=window_pass, f1_pass=f1_pass)
    return dict(primary=primary, passes=bool(window_pass and f1_pass),
                rotations=[contrasts([anchors[s] for s in group]) for group in TRIOS])


def self_check():
    table = [[2,-751,0,0,3,3],[5,1,0,0,3,3],[9,750,0,0,3,3],[17,752,0,0,3,3]]
    case = dict(predictions=table, gt=[[11,1],[12,750],[13,-751]],
                edges05=[[2,13],[5,11],[5,12],[9,11]], edges03=[[2,13],[5,11],[5,12],[9,11],[17,12]],
                nframes=1000)
    support = dict(retained=[2,5,17], phase_counts=[2]*99)
    out = score_anchor(case, [table]*3, support)
    assert out["full"]["05"]["tp"] == 3 and out["kept"]["05"]["tp"] == 2
    assert out["kept"]["03"]["tp"] == 3 and out["loss03"]["f1_improved"]
    assert out["loss05"]["lost_tp"] == 1 and not out["loss05"]["f1_improved"]
    assert out["loss03"]["lost_fp"] == 1 and out["phase_wins"] == 99 and not out["source_anchor_pass"]
    assert all(r["totals"] == [2,1] and len(set(r["ids"])) == 3 for r in out["quota"])
    assert out["full"]["window"] == out["e3"]["window"]
    assert out["quota"] == score_anchor(case, [table]*3, support)["quota"]
    empty = score_anchor(dict(case, predictions=[], edges05=[], edges03=[]), [[],[],[]],
                         dict(retained=[], phase_counts=[0]*99))
    assert empty["full"]["05"]["f1"] == 0 and empty["loss05"]["f1_equal"]
    example = {i: dict(source_anchor_pass=i in (0,1,3,4)) for i in range(9)}
    assert source_gate(example)["passes"]
    example[0]["source_anchor_pass"] = False
    assert not source_gate(example)["passes"]
    json.dumps(out, allow_nan=False)


if __name__ == "__main__":
    self_check()
    print("Synthetic subset matching, loss identity, direction budgets, E3 and source gates passed.")
