"""GT-free canonical P0 endpoint emergence candidates; native row order retained."""
import numpy as np


def canonicalize(raw, by_track):
    a = np.asarray(raw, dtype=float).reshape(-1, 7)
    if (not np.isfinite(a).all() or not np.equal(a[:, :2], np.floor(a[:, :2])).all()
            or (a[:, 4:6] <= 0).any()):
        raise ValueError('Invalid native MOT rows')
    if len(np.unique(a[:, :2], axis=0)) != len(a):
        raise ValueError('Duplicate native frame/ID makes row membership ambiguous')
    canonical = a.copy()
    membership = np.zeros(len(a), dtype=np.int64)
    segments = []
    for native_id in np.unique(a[:, 1]):
        indices = np.flatnonzero(a[:, 1] == native_id)
        tagged = np.column_stack((a[indices], indices))
        tagged[:, 1] = 1  # avoid native_id + 100000 gap-split collisions
        split = by_track(tagged)
        for official_id in np.unique(split[:, 1]):
            chosen = split[split[:, 1] == official_id]
            row_indices = chosen[:, 7].astype(np.int64)
            cid = len(segments) + 1
            membership[row_indices] = cid
            canonical[row_indices, 1] = cid
            segments.append(dict(id=cid, native_id=int(native_id), rows=row_indices.tolist(),
                                 first=int(chosen[0, 0]), last=int(chosen[-1, 0])))
    assert not len(a) or (membership > 0).all()
    return canonical, membership, segments


def _center(row):
    return row[2:4] + row[4:6] / 2


def _iou(a, b):
    lo = np.maximum(a[2:4], b[2:4])
    hi = np.minimum(a[2:4] + a[4:6], b[2:4] + b[4:6])
    inter = np.maximum(hi - lo, 0).prod()
    return inter / (a[4:6].prod() + b[4:6].prod() - inter)


def candidates(rows, nframes=3950):
    """Unique local two-frame starts and time-reversed endings, without ROI gates."""
    R = np.asarray(rows, dtype=float).reshape(-1, 7)
    if (not np.isfinite(R).all() or not np.equal(R[:, :2], np.floor(R[:, :2])).all()
            or (R[:, 0] < 1).any() or (R[:, 0] > nframes).any() or (R[:, 4:6] <= 0).any()
            or len(np.unique(R[:, :2], axis=0)) != len(R)):
        raise ValueError('Invalid canonical rows')
    tracks = {int(cid): {int(R[i, 0]): int(i) for i in np.flatnonzero(R[:, 1] == cid)}
              for cid in np.unique(R[:, 1])}
    frames = {int(f): np.flatnonzero(R[:, 0] == f) for f in np.unique(R[:, 0])}
    found, abstain = [], []
    for B, bt in sorted(tracks.items()):
        for kind, sign in (('start', 1), ('end', -1)):
            f = min(bt) if sign == 1 else max(bt)
            if f + sign not in bt:
                continue
            legal = []
            bidx = [bt[f], bt[f + sign]]
            for A, at in sorted(tracks.items()):
                if A == B or not all(t in at for t in (f - sign, f, f + sign)):
                    continue
                aidx = [at[f - sign], at[f], at[f + sign]]
                L = float(np.median(np.sqrt(R[[aidx[1], aidx[2], *bidx], 4:6].prod(axis=1))))
                if not all(.5 * L <= np.linalg.norm(_center(R[ai]) - _center(R[bi])) <= 2 * L
                           and _iou(R[ai], R[bi]) < .5 for ai, bi in zip(aidx[1:], bidx)):
                    continue
                isolated = True
                for ai, bi in zip(aidx[1:], bidx):
                    others = [int(i) for i in frames[int(R[ai, 0])] if int(R[i, 1]) not in (A, B)]
                    if any(min(np.linalg.norm(_center(R[i]) - _center(R[ai])),
                               np.linalg.norm(_center(R[i]) - _center(R[bi]))) <= 2 * L for i in others):
                        isolated = False
                        break
                if not isolated:
                    continue
                previous = [int(i) for i in frames[f - sign]
                            if min(np.linalg.norm(_center(R[i]) - _center(R[aidx[1]])),
                                   np.linalg.norm(_center(R[i]) - _center(R[bidx[0]]))) <= 2 * L]
                if previous != [aidx[0]]:
                    continue
                legal.append(dict(kind=kind, sign=sign, boundary_frame=f, A=A, B=B,
                                  anchor_frames=[f, f + sign], row_indices=dict(A=aidx, B=bidx),
                                  row_keys=dict(A=[[A, int(R[i, 0])] for i in aidx],
                                                B=[[B, int(R[i, 0])] for i in bidx]),
                                  L=L, center=_center(R[bidx[0]]).tolist(), radius=2 * L,
                                  permitted_frames=[f - sign * t for t in range(1, 13)
                                                    if 1 <= f - sign * t <= nframes]))
            if len(legal) == 1:
                found.append(dict(candidate_id=len(found) + 1, **legal[0]))
            elif len(legal) > 1:
                abstain.append(dict(B=B, kind=kind, boundary_frame=f, residents=[c['A'] for c in legal]))
    return dict(candidates=found, ambiguous_abstentions=abstain)


def self_test(by_track):
    def row(f, cid, x):
        return [f, cid, x, 0, 10, 10, .8]
    a = np.asarray([row(f, 70, 0) for f in range(1, 7)] + [row(f, 90, 10) for f in (3, 4, 5)], float)
    # Deliberately scramble raw rows: membership must refer back to native order.
    a = a[[6, 0, 8, 3, 1, 7, 2, 4, 5]]
    C, ids, segments = canonicalize(a, by_track)
    assert np.array_equal(C[:, [0, 2, 3, 4, 5, 6]], a[:, [0, 2, 3, 4, 5, 6]])
    assert np.array_equal(C[:, 1], ids)
    out = candidates(C, 10)['candidates']
    assert [(c['kind'], c['boundary_frame']) for c in out] == [('start', 3), ('end', 5)]
    assert out[0]['permitted_frames'] == [2, 1] and out[1]['permitted_frames'] == [6, 7, 8, 9, 10]
    assert all(c['L'] == 10 and c['radius'] == 20 and c['center'] == [15, 5] for c in out)
    for distance, expected in ((5, 2), (20, 2), (20.01, 0), (4.99, 0)):
        b = C.copy(); b[b[:, 1] == 2, 2] = distance
        assert len(candidates(b, 10)['candidates']) == expected
    third = np.vstack((C, row(3, 3, 25)))
    assert [c['kind'] for c in candidates(third, 10)['candidates']] == ['end']
    prior_third = np.vstack((C, row(2, 3, 25)))
    assert [c['kind'] for c in candidates(prior_third, 10)['candidates']] == ['end']
    hole = C[~((C[:, 1] == 2) & (C[:, 0] == 4))]
    assert not candidates(hole, 10)['candidates']
    reused = np.asarray([row(1, 1, 0), row(100, 1, 0), row(2, 100001, 0)], float)
    _, membership, segments = canonicalize(reused, by_track)
    assert len(set(membership)) == 3 and len(segments) == 3
    assert not candidates(np.empty((0, 7)), 10)['candidates']
    print('Synthetic canonical membership, geometry, unique-neighbour, mirrored endpoints checks passed')
