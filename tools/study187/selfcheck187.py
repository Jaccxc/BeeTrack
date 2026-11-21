"""§187 synthetic self-checks (no data, no GT): extraction, rewrites, joint = base + sum of group diffs, bound checks.
From final_project: ... uv run --offline --no-sync python tools/study187/selfcheck187.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import lib187 as X

SQ = [[0, 0], [100, 0], [100, 100], [0, 100]]     # inside = centre x in [0, 100]


def crossing(shift, ids):
    """P comes from outside, Q from inside; one box for frames 12-13 on the edge; then C goes inside, D outside.
    The tracker keeps P on the merged box and on D: no event. Swapping the after part gives one in and one out."""
    box = lambda f, i, x: [f + shift, i, x - 5, 45, 10, 10, 0.9]
    rows = []
    for f in range(1, 12):
        rows += [box(f, ids[0], 160 - 5 * (f - 1)), box(f, ids[1], 40 + 5 * (f - 1))]
    rows += [box(f, ids[0], 100) for f in (12, 13)]
    for f in range(14, 31):
        rows += [box(f, ids[2], 90 - 2 * (f - 14)), box(f, ids[0], 110 + 2 * (f - 14))]
    return rows


def video():
    rows = np.asarray(sorted(crossing(0, (1, 2, 3)) + crossing(1000, (11, 12, 13))), float)
    found, tally, _ = X.extract(rows, rows[:, 1].copy())
    assert [(c['t0'], c['t1'], c['status']) for c in found] == [(12, 13, 'ok'), (1012, 1013, 'ok')], (found, tally)
    S = X.labelled(rows)
    scenes = [X.scene(dict(c, cid=n), (100 + 2 * n, 101 + 2 * n)) for n, c in enumerate(found)]
    return rows, S, scenes


def run():
    rows, S, scenes = video()
    E0 = X.events(S, SQ)
    assert len(E0) == 0
    # one scene: some option is exactly +1 in and +1 out
    one = [sc for sc in scenes if sc['t0'] == 12]
    changes = []
    for o in X.options(one[0])[1:]:
        U = X.apply(S, one[0], o)
        if U is not None:
            changes.append(sorted(map(tuple, X.events(U, SQ).tolist())))
    assert [(12, 1), (14, -1)] in changes, changes
    # (a) no candidates: bound = W; (b) every enumerated W >= bound
    G = np.array([[12, 1], [14, -1], [2000, 1]])
    gc, pc, n = X.counts(G), X.counts(E0), len(G)
    assert X.lower_bound(pc, gc, n, []) == X.W(pc, gc, n)
    (sc0,), ids0 = X.groups(S, one)[0]
    rng = X.group_range(S, [sc0], ids0, SQ)[:2]
    lb = X.lower_bound(pc, gc, n, [rng])
    Ws = [X.W(X.counts(X.events(U, SQ)), gc, n) for o in X.options(sc0)
          if (U := X.apply(S, sc0, o)) is not None]
    assert min(Ws) >= lb - 1e-9 and abs(min(Ws) - lb) < 1e-9, (Ws, lb)
    # (c) both scenes jointly = base + sum of the per-group diffs
    gs = X.groups(S, scenes)
    assert len(gs) == 2
    choice = {sc['cid']: (1, 'keep') for sc in scenes}   # c is D here (row order), so pairing 1 = P-C, Q-D
    joint = X.counts(X.events(X.apply_all(S, scenes, choice), SQ))
    summed = pc.copy()
    for g, ids in gs:
        Sl = X.local(S, ids)
        summed += X.counts(X.events(X.apply_all(Sl, g, choice), SQ)) - X.counts(X.events(Sl, SQ))
    assert np.array_equal(joint, summed)
    assert sorted(map(tuple, X.events(X.apply_all(S, scenes, choice), SQ).tolist())) == [(12, 1), (14, -1), (1012, 1), (1014, -1)]
    # a GT-box option: two boxes per middle frame, legal, same pairing
    sc = dict(sc0, gt_mid=[[12, 'a', 90, 45, 10, 10], [12, 'b', 100, 45, 10, 10]])
    U = X.apply(S, sc, (0, 'gt'))
    assert U is not None and len(U) == len(S) - 2 + 2
    # illegal: giving the middle box to a track that already has a box there
    bad = dict(sc0, mid=[int(np.flatnonzero((rows[:, 0] == 5) & (rows[:, 1] == 2))[0])])
    assert X.apply(S, bad, (0, 'a')) is None
    return dict(extraction='two synthetic 2->1->2 found, nothing else', one_in_one_out=True,
                bound_a=True, bound_b=dict(Ws=sorted(set(round(w, 6) for w in Ws)), bound=lb), joint_c=True)


if __name__ == '__main__':
    print(run())
