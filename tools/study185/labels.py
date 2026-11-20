"""Offline bridge labels using the official frame-wise Hungarian assignment."""

import numpy as np

from beetrack.evaluate import by_track, frame_match


def bridge_labels(D, GT):
    """Return original weak row indices, uint8 labels, and nested-filter counts.

    D is [frame, x1, y1, x2, y2, score]; prepared GT is MOT
    [frame, id, x, y, width, height]. Frames retain their native numbers.
    ``two_sided`` counts only assigned weak rows whose owner is uncovered now.
    """
    D, GT = np.asarray(D), np.asarray(GT)
    weak_idxs = np.flatnonzero((D[:, 5] >= .25) & (D[:, 5] < .40))
    y = np.zeros(len(weak_idxs), dtype=np.uint8)
    counts = dict(weak=len(weak_idxs), assigned=0, uncovered_now=0,
                  two_sided=0, positives=0)
    gt = by_track(GT)

    def owners(indices):
        if not len(indices) or not len(gt):
            return np.full(len(indices), -1, dtype=np.int64)
        rows = D[indices]
        mot = np.column_stack((rows[:, 0], indices, rows[:, 1:3],
                               rows[:, 3:5] - rows[:, 1:3]))
        return frame_match(gt, mot, .5)

    eligible = np.flatnonzero(D[:, 5] >= .25)
    full_owner = owners(eligible)
    weak_owner = full_owner[np.searchsorted(eligible, weak_idxs)]
    strong_idxs = np.flatnonzero(D[:, 5] >= .40)
    strong_owner = owners(strong_idxs)
    strong_pairs = {(D[i, 0], int(owner))
                    for i, owner in zip(strong_idxs, strong_owner) if owner >= 0}
    for k, (i, owner) in enumerate(zip(weak_idxs, weak_owner)):
        if owner < 0:
            continue
        counts['assigned'] += 1
        frame = D[i, 0]
        if (frame, int(owner)) in strong_pairs:
            continue
        counts['uncovered_now'] += 1
        before = any((frame - lag, int(owner)) in strong_pairs for lag in (1, 2, 3))
        after = any((frame + lag, int(owner)) in strong_pairs for lag in (1, 2, 3))
        if before and after:
            counts['two_sided'] += 1
            y[k] = 1
    counts['positives'] = int(y.sum())
    return weak_idxs, y, counts


def _self_check():
    def det(f, score=.5, x=0):
        return [f, x, 0, x + 10, 10, score]

    def gt(frames, identity=1, x=0):
        return [[f, identity, x, 0, 10, 10] for f in frames]

    def check(d, g, expected):
        idx, y, counts = bridge_labels(np.array(d, float).reshape(-1, 6),
                                       np.array(g, float).reshape(-1, 6))
        assert y.dtype == np.uint8
        assert y.tolist() == expected, (idx, y, counts)
        assert counts['positives'] == sum(expected)
        return counts

    # Native frame distances, missing intermediate detections, and exact cutoffs.
    check([det(4, .25), det(1, .40), det(7, .40), det(4, .249)],
          gt(range(1, 8)), [1])
    check([det(1), det(5, .3), det(9)], gt([1, 5, 9]), [0])
    check([det(1), det(2, .3), det(3)], gt([1, 2, 3]), [1])
    # A below-threshold perfect box cannot steal the eligible weak match.
    check([det(1), det(2, .3, 1), det(2, .249), det(3)],
          gt([1, 2, 3]), [1])
    # A better weak duplicate owns the full match, but the strong-only match
    # still covers this identity now; weak evidence must not create a positive.
    c = check([det(1), det(2, .3), det(2, .4, 1), det(3)],
              gt([1, 2, 3]), [0])
    assert c['assigned'] == 1 and c['uncovered_now'] == 0
    # Different identities on the two sides cannot bridge one another.
    check([det(1), det(2, .3), det(3, .5, 30)],
          gt([1, 2, 3]) + gt([1, 2, 3], 2, 30), [0])
    check([det(1), det(2, .3), det(3), det(2, .3, 30)],
          gt([1, 2, 3]) + gt([1, 2, 3], 2, 30), [1, 0])
    check([det(2, .3)], gt([2]), [0])
    check([det(2, .3)], [], [0])
    check([], [], [])
    check([det(2, .4)], gt([2]), [])
    # Canonical reused-ID splitting stays delegated to by_track.
    check([det(31), det(32, .3), det(33)], gt([1, 31, 32, 33]), [1])


if __name__ == '__main__':
    _self_check()
    print('bridge_labels synthetic checks passed')
