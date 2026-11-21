"""Synthetic-only first-head -> quota -> identity -> physical MOT roundtrip.

Features and source pairs are generated from these synthetic MOT geometries;
this checks integration/attainability, not performance or field generalization.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import harm_budget as H
import identity_rank as R
from identity_flow import identity_flow
from physical import realize_subset, verify_subset
from beetrack.evaluate import by_track, match_events, zone_tracks
from tools.study104.event_overlap import score_arm
from tools.study103.swap_score import vector_of_counts, window_error

POLY = [[0, 0], [10, 0], [10, 40], [0, 40]]
SECONDS = 40
NF = 25 * SECONDS


def fixture(seed, video):
    rng = np.random.default_rng(seed)
    members, buckets, features, edges, universe = {}, {}, {}, {}, {}
    for m in range(3):
        member = f'{video}:m{m}'
        rows, gt = [], []
        for j, t in enumerate((2, 10, 31, 37)):
            direction = 1 if j % 2 == 0 else -1
            xs = [14, 4] if direction == 1 else [4, 14]
            for good in (False, True):
                pid, y = 2*j + 1 + int(good), 5 if good else 25
                score = (.88 if good else .42) + rng.uniform(-.02, .02)
                rows.extend([[25*t, pid, xs[0] + 1e-8, y, 2, 2, score],
                             [25*t + 1, pid, xs[1] + 1e-8, y, 2, 2, score]])
            gid = seed * 1000 + 100*m + j
            gt.extend([[25*t, gid, xs[0] + 1e-8, 5, 2, 2],
                       [25*t + 1, gid, xs[1] + 1e-8, 5, 2, 2]])
        rows, gt = by_track(np.asarray(rows)), by_track(np.asarray(gt))
        members[member] = dict(rows=rows, gt=gt)
        graph = score_arm(gt, rows, POLY)
        universe[member] = list(graph.gt)
        for pid, (direction, frame, _) in graph.events.items():
            key = (member, (frame - 1)//25, 1 if direction == 'in' else -1)
            pred = (member, pid)
            buckets.setdefault(key, []).append(pred)
            tr = rows[rows[:, 1] == pid]
            features[pred] = [float(tr[:, 6].mean()), float(abs(tr[-1, 2] - tr[0, 2])), len(tr)]
            edges[pred] = [g for p, g in graph.edges if p == pid]
    return dict(video=video, members=members, buckets=buckets, features=features,
                edges=edges, gt_events=universe)


def direction_inputs(group, direction):
    ids = [[] for _ in range(SECONDS)]
    for (_, sec, d), preds in group['buckets'].items():
        if d == direction:
            ids[sec].extend(preds)
    return ids


def window_features(group, direction, windows):
    cap = np.array([len(v) for v in direction_inputs(group, direction)])
    A = H.incidence(windows, SECONDS)
    return np.column_stack((np.asarray(A @ cap), [hi-lo for lo, hi in windows]))


def run():
    # Source models freeze before target generation; source/target random seeds,
    # videos, GT identities and MOT arrays are distinct.
    source = [fixture(101, 'source-a'), fixture(202, 'source-b')]
    windows = H.intervals(SECONDS)
    X, residual, source_names, videos = [], [], [], []
    for group in source:
        group['quotas'] = {b: 1 for b in group['buckets']}
        for direction in (1, -1):
            xx = window_features(group, direction, windows)
            # Source labels explicitly come from source MOT/GT window counts.
            pred = sum((vector_of_counts(v['rows'], POLY, NF) for v in group['members'].values()))
            truth = sum((vector_of_counts(v['gt'], POLY, NF) for v in group['members'].values()))
            dd = 0 if direction == 1 else 1
            rr = (pred[:, dd] - truth[:, dd]).astype(int)
            assert len(rr) == len(windows)
            X.extend(xx)
            residual.extend(rr)
            source_names.extend([group['video']] * len(xx))
            videos.extend([group['video']] * len(xx))
    budget_model = H.fit(X, residual, source_names, videos, k=1)
    rank_model = R.fit(source)
    target = fixture(909, 'target-only')
    assert not set(rank_model['source_videos']) & {target['video']}
    assert not set(g for u in target['gt_events'].values() for g in u) & {
        g for s in source for u in s['gt_events'].values() for g in u}
    quotas, decoded = {}, {}
    member_order = list(target['members'])
    for direction in (1, -1):
        ids = direction_inputs(target, direction)
        xx = window_features(target, direction, windows)
        maximum = int(xx[:, 0].max())
        costs = 1 - 2 * H.survival(budget_model, xx, maximum)
        result = H.decode(windows, ids, costs)
        decoded[direction] = result['q']
        for t, q in enumerate(result['q']):
            n = [len(target['buckets'].get((m, t, direction), [])) for m in member_order]
            deletion = H.allocate_member_quota(q, n, t)
            assert int(sum(deletion)) == q
            for m, capacity, d in zip(member_order, n, deletion):
                b = (m, t, direction)
                if b in target['buckets']:
                    quotas[b] = int(capacity-d)
    fixed = R.fixed(target['buckets'], quotas)
    ranked, abstained = R.select(rank_model, target['buckets'], quotas, target['features'])
    oracle = identity_flow(target['buckets'], quotas, target['edges'], target['gt_events'])
    arms = dict(fixed=fixed, scored=ranked, oracle=oracle['retained'])
    all_kept = {b: len(ids) for b, ids in target['buckets'].items()}
    native_flow = identity_flow(target['buckets'], all_kept, target['edges'], target['gt_events'])
    reports = {}
    for name, selected in arms.items():
        assert {b: len(v) for b, v in selected.items()} == quotas
        reports[name] = {}
        for member, data in target['members'].items():
            kept = {p[1] for b, ps in selected.items() if b[0] == member for p in ps}
            mot, _ = realize_subset(data['rows'], POLY, kept)
            # 17 significant digits roundtrip float64 exactly; repository
            # two-decimal writers intentionally do not meet this guarantee.
            with TemporaryDirectory(prefix='synthetic_mot_') as tmp:
                path = Path(tmp) / 'predictions.txt'
                np.savetxt(path, mot, delimiter=',', fmt='%.17g')
                loaded = np.loadtxt(path, delimiter=',', ndmin=2)
            assert np.array_equal(loaded, mot)
            verified = verify_subset(data['gt'], data['rows'], POLY, kept)
            official = match_events(data['gt'], loaded, POLY)
            assert official == verified['realized_official']
            pc = vector_of_counts(loaded, POLY, NF)
            gc = vector_of_counts(data['gt'], POLY, NF)
            reports[name][member] = dict(**official, f1=2*official['tp']/(official['pred']+official['gt']),
                W=window_error(gc, pc), window_counts=pc.tolist(),
                W_denominator=int(gc.sum()), F1_denominator=official['pred']+official['gt'])
    for member, data in target['members'].items():
        rows = [reports[name][member] for name in arms]
        assert len({r['pred'] for r in rows}) == len({r['W_denominator'] for r in rows}) == len({r['F1_denominator'] for r in rows}) == 1
        assert rows[0]['window_counts'] == rows[1]['window_counts'] == rows[2]['window_counts']
        assert len({r['W'] for r in rows}) == 1
        assert max(rows[0]['f1'], rows[1]['f1']) <= rows[2]['f1'] + 1e-12
        assert rows[2]['tp'] == oracle['per_member'][member]['tp']
        native = match_events(data['gt'], data['rows'], POLY)
        assert native['tp'] == native_flow['per_member'][member]['tp']
        physical, _ = realize_subset(data['rows'], POLY, {p[1] for b, ps in target['buckets'].items() if b[0] == member for p in ps})
        assert match_events(data['gt'], physical, POLY) == native
    for values in reports.values():
        for r in values.values():
            del r['window_counts']
    return dict(scope='synthetic geometry/MOT source pairs; no field generalization claim',
                source_seeds=[101, 202], target_seed=909, rank_pairs=rank_model['pairs'],
                rank_abstained=len(abstained), q=decoded, quotas=[(str(k), v) for k, v in quotas.items()],
                native_attainable=True, reports=reports)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
