"""GT-assisted diagnostic for synchronous reciprocal P0 identity switches.

Input rows are (frame, id, x, y, width, height, ...) after the protocol's
ID-reuse repair. This module never splits IDs or changes observed boxes.
"""

from collections import defaultdict

import numpy as np


def _frames(rows):
    order = np.argsort(rows[:, 0], kind="stable")
    frames, starts = np.unique(rows[order, 0], return_index=True)
    return dict(zip(frames, np.split(order, starts[1:])))


def _owners(pr, gt):
    """Per observation, use its unique best overlap. Multiple boxes may share a bee."""
    owners = np.full(len(pr), -1, dtype=np.int64)
    gt_frames = _frames(gt)
    for frame, p in _frames(pr).items():
        g = gt_frames.get(frame)
        if g is None:
            continue
        pbox, gbox = pr[p, 2:6], gt[g, 2:6]
        low = np.maximum(pbox[:, None, :2], gbox[None, :, :2])
        high = np.minimum((pbox[:, :2] + pbox[:, 2:])[:, None],
                          (gbox[:, :2] + gbox[:, 2:])[None])
        intersection = np.maximum(high - low, 0).prod(axis=2)
        union = pbox[:, 2:].prod(axis=1)[:, None] + gbox[:, 2:].prod(axis=1) - intersection
        overlaps = np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union > 0)
        best = overlaps.argmax(axis=1)
        score = overlaps[np.arange(len(p)), best]
        unique = (overlaps == score[:, None]).sum(axis=1) == 1
        accepted = unique & (score >= 0.3)
        owners[p[accepted]] = gt[g[best[accepted]], 1].astype(np.int64)
    return owners


def discover(pr, gt, *, return_counts=False):
    """Return sorted (smaller P0 id, larger P0 id, cut), optionally with counts."""
    counts = dict(single_switches=0, reciprocal_pairs=0, close_pairs=0)
    if len(pr) < 6 or len(gt) == 0:
        return ([], counts) if return_counts else []
    pr = pr[np.lexsort((pr[:, 0], pr[:, 1]))]
    owners = _owners(pr, gt)
    # A duplicate observation for one P0 ID and frame cannot be a unique anchor.
    duplicate = (np.diff(pr[:, 0]) == 0) & (np.diff(pr[:, 1]) == 0)
    owners[:-1][duplicate] = -1
    owners[1:][duplicate] = -1

    # With sorted rows, a six-row span in a single track covers six consecutive
    # frames iff its endpoints differ by five and all six frames are unique.
    six = np.lib.stride_tricks.sliding_window_view(owners, 6)
    continuous = (pr[:-5, 1] == pr[5:, 1]) & (pr[5:, 0] - pr[:-5, 0] == 5)
    stable = (six[:, :3] == six[:, :1]).all(axis=1) & (six[:, 3:] == six[:, 3:4]).all(axis=1)
    changes = (six[:, 0] >= 0) & (six[:, 3] >= 0) & (six[:, 0] != six[:, 3])
    centers = pr[:, 2:4] + pr[:, 4:6] / 2
    lengths = np.sqrt(pr[:, 4] * pr[:, 5])
    transitions = defaultdict(list)
    candidates = []
    switches = np.flatnonzero(continuous & stable & changes)
    counts["single_switches"] = len(switches)
    for index in switches:
        left = index + 2
        frame, track = map(int, pr[left, :2])
        old, new = owners[left:left + 2]
        for peer in transitions.get((frame, new, old), ()):
            counts["reciprocal_pairs"] += 1
            for offset in (0, 1):
                a, b = left + offset, peer + offset
                if np.linalg.norm(centers[a] - centers[b]) <= lengths[a] + lengths[b]:
                    candidates.append((int(pr[peer, 1]), track, frame))
                    break
        transitions[frame, old, new].append(left)
    counts["close_pairs"] = len(candidates)
    candidates.sort()
    return (candidates, counts) if return_counts else candidates


def exchange(pr, swaps):
    """Exchange entire suffixes strictly after each cut. Pairs must be disjoint."""
    changed = pr.copy()
    used = set()
    for a, b, frame in swaps:
        if a == b or a in used or b in used:
            raise ValueError("suffix exchanges must use disjoint tracks")
        used.update((a, b))
        after = pr[:, 0] > frame
        changed[after & (pr[:, 1] == a), 1] = b
        changed[after & (pr[:, 1] == b), 1] = a
    assert len(changed) == len(pr)
    return changed[np.lexsort((changed[:, 0], changed[:, 1]))]


def _self_check():
    gt = np.array([[f, bee, x, 0, 10, 10] for f in range(1, 9)
                   for bee, x in [(7, 0), (8, 16)]], dtype=float)
    pr = gt.copy()
    pr[:, 1] = np.where((pr[:, 1] == 7) == (pr[:, 0] <= 3), 1, 2)
    pair = [(1, 2, 3)]
    assert discover(pr[::-1], gt) == pair
    assert discover(pr, gt, return_counts=True) == (pair, dict(single_switches=2, reciprocal_pairs=1, close_pairs=1))
    assert discover(pr[pr[:, 1] == 1], gt, return_counts=True)[1]["single_switches"] == 1
    repaired = exchange(pr, pair)
    assert np.all(repaired[repaired[:, 1] == 1, 2] == 0)
    assert np.all(repaired[repaired[:, 1] == 2, 2] == 16)
    assert np.array_equal(exchange(repaired, pair), pr[np.lexsort((pr[:, 0], pr[:, 1]))])
    assert not discover(pr[~((pr[:, 0] == 5) & (pr[:, 1] == 2))], gt)
    assert not discover(pr, np.vstack([gt, gt[:1]]))
    far_gt, far_pr = gt.copy(), pr.copy()
    far_gt[gt[:, 2] == 16, 2] = 21
    far_pr[pr[:, 2] == 16, 2] = 21
    assert not discover(far_pr, far_gt)
    far_gt[(gt[:, 0] == 4) & (gt[:, 2] == 16), 2] = 20
    far_pr[(pr[:, 0] == 4) & (pr[:, 2] == 16), 2] = 20
    assert discover(far_pr, far_gt) == pair
    try:
        exchange(pr, pair + pair)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping exchanges were accepted")


if __name__ == "__main__":
    _self_check()
    print("paired swap self-check passed")
