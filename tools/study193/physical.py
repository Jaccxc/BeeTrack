"""Synthetic-verified MOT realization of an arbitrary retained event subset.

No rows are removed, no geometry is rounded, and frame-Hungarian tie order is
preserved. Fresh prediction IDs describe fragments; they do not link tracks.
"""
import os

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np

from beetrack.evaluate import by_track, frame_match, inside, match_events, zone_tracks
from tools.study104.event_overlap import archive_arm, replay_arm, score_arm


def realize_subset(raw, polygon, retained):
    """Return (MOT, metadata) retaining exactly these official canonical event IDs.

    Legal domain: float64-compatible finite Nx6+ MOT, positive box dimensions,
    positive integral native frames, nonnegative integral IDs, no repeated
    (frame, ID). Integer IDs/frames and by_track's possible 1e5 offsets must
    remain exactly representable below 2**53. Nonintegral IDs are rejected
    because official zone_tracks casts its dictionary keys to int.
    Extra columns (including scores) retain their native values and precision.
    ``retained`` names IDs AFTER official by_track, including gap splits.
    No drop_short or rounding/export step is applied here.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 6 or not np.isfinite(raw).all():
        raise ValueError('expected finite Nx6+ MOT')
    if (np.any(raw[:, 4:6] <= 0) or np.any(raw[:, :2] != np.rint(raw[:, :2]))
            or np.any(raw[:, 0] < 1) or np.any(raw[:, 1] < 0)
            or np.any(raw[:, 0] >= 2**53)
            or np.any(raw[:, 1] >= 2**53 - 100000 * len(raw))):
        raise ValueError('invalid geometry or unsafe/nonintegral frame/ID')
    if len(np.unique(raw[:, :2], axis=0)) != len(raw):
        raise ValueError('repeated native frame/ID')
    base = by_track(raw)
    if len(np.unique(base[:, :2], axis=0)) != len(base):
        raise ValueError('official gap-split IDs collide within a frame')
    # Idempotence is necessary because the evaluator canonicalizes again.
    if not np.array_equal(by_track(base), base):
        raise ValueError('official canonicalization is not idempotent')
    tracks = zone_tracks(base, polygon) if len(base) else {}
    event_ids = {tid for tid, value in tracks.items() if value[0]}
    kept = set(retained)
    if not kept <= event_ids:
        raise ValueError('retained IDs must be official canonical event IDs')
    out = base.copy()
    side = inside(base, polygon) if len(base) else np.empty(0, bool)
    segments, fresh = [], 0
    for tid, (event, start, end, _) in tracks.items():
        cut = [] if not event or tid in kept else (np.flatnonzero(side[start + 1:end] != side[start:end - 1]) + start + 1).tolist()
        bounds = [start, *cut, end]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            fresh += 1
            out[lo:hi, 1] = fresh
            segments.append(dict(source_id=tid, output_id=fresh, start=lo, end=hi,
                                 event_retained=bool(event and tid in kept)))
    if not np.array_equal(by_track(out), out):
        raise ValueError('unexpected evaluator re-splitting of fresh fragments')
    assert np.array_equal(out[:, 0], base[:, 0])
    assert np.array_equal(out[:, 2:], base[:, 2:])
    return out, dict(segments=segments, retained=sorted(kept),
                     suppressed=sorted(event_ids - kept), source_ids=base[:, 1].copy())


def verify_subset(gt, raw, polygon, retained):
    """Synthetic/test helper: verify exact official edges and all event metrics."""
    base = by_track(raw)
    out, meta = realize_subset(raw, polygon, retained)
    new_to_old = {s['output_id']: s['source_id'] for s in meta['segments'] if s['event_retained']}
    old_owner = frame_match(by_track(gt), base)
    new_owner = frame_match(by_track(gt), by_track(out))
    assert np.array_equal(old_owner, new_owner), 'frame assignments changed (including ties)'
    comparisons = {}
    for threshold in (.5, .3):
        original, realized = (score_arm(gt, r, polygon, threshold) for r in (base, out))
        expected_edges = {(p, g) for p, g in original.edges if p in retained}
        actual_edges = {(new_to_old[p], g) for p, g in realized.edges}
        assert expected_edges == actual_edges
        assert {new_to_old[p]: event for p, event in realized.events.items()} == {
            p: event for p, event in original.events.items() if p in retained}
        archive = archive_arm(original)
        archive['predictions'] = [r for r in archive['predictions'] if r[0] in retained]
        archive['edges'] = [edge for edge in archive['edges'] if edge[0] in retained]
        expected = replay_arm(archive)
        assert len(expected.matches) == len(realized.matches)
        comparisons[str(threshold)] = dict(original_pred=len(original.events),
                                           retained_pred=len(realized.events),
                                           original_tp=len(original.matches), retained_tp=len(realized.matches),
                                           edges=len(actual_edges))
    official = match_events(gt, out, polygon)
    assert official['pred'] == len(retained)
    assert official['tp'] == comparisons['0.5']['retained_tp']
    # Official tp_id additionally requires first/last matched frame same bee.
    truth = zone_tracks(by_track(gt), polygon)
    strict = set()
    for tid, (direction, lo, hi, _) in zone_tracks(base, polygon).items():
        if tid in retained:
            owners = old_owner[lo:hi][old_owner[lo:hi] >= 0]
            if len(owners) and owners[0] == owners[-1] and truth[owners[0]][0] == direction:
                strict.add(owners[0])
    assert official['tp_id'] == len(strict)
    return dict(original_official=match_events(gt, base, polygon),
                realized_official=official, thresholds=comparisons)


def _self_check():
    from itertools import combinations
    poly = [[0, 0], [10, 0], [10, 30], [0, 30]]
    def track(tid, frames, xs, y=5):
        return [[f, tid, x + .0000001, y, 2, 2, .812345678901]
                for f, x in zip(frames, xs)]
    # Both directions, same-frame identical-box competition, repeated crossing,
    # non-event track, and reuse of native ID across >25 frame gap.
    raw = np.array(track(3, range(1, 5), [14, 4, 14, 4]) +
                   track(7, range(1, 5), [14, 4, 14, 4]) +
                   track(11, range(1, 5), [4, 14, 4, 14], 15) +
                   track(13, range(1, 5), [4, 14, 14, 4], 23) +
                   track(3, (40, 41), [4, 14], 5), float)
    gt = np.array(track(101, range(1, 5), [14, 4, 14, 4]) +
                  track(102, range(1, 5), [4, 14, 4, 14], 15) +
                  track(103, (40, 41), [4, 14], 5), float)
    ids = [i for i, value in zone_tracks(by_track(raw), poly).items() if value[0]]
    results = []
    rng = np.random.default_rng(7)
    for order in (np.arange(len(raw)), np.arange(len(raw))[::-1], rng.permutation(len(raw))):
        for size in range(len(ids) + 1):
            for keep in combinations(ids, size):
                results.append(verify_subset(gt, raw[order], poly, set(keep)))
    fractional, unsafe = raw.copy(), raw.copy()
    fractional[0, 1], unsafe[0, 1] = .5, 2**53
    for bad in (np.concatenate((raw, raw[:1])), fractional, unsafe):
        try:
            realize_subset(bad, poly, set())
        except ValueError:
            pass
        else:
            raise AssertionError('invalid MOT accepted')
    empty, _ = realize_subset(np.empty((0, 7)), poly, set())
    assert empty.shape == (0, 7)
    loser = verify_subset(gt, raw, poly, {7})
    assert loser['realized_official']['pred'] == 1 and loser['realized_official']['tp'] == 0
    return dict(checks=len(results), first=results[0], last=results[-1], tied_loser_only=loser)


if __name__ == '__main__':
    import json
    print(json.dumps(_self_check(), indent=2))
