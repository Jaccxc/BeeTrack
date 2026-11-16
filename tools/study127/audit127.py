"""Study 127 crossing endpoint ownership audit. --self-test never opens research data."""
import argparse
import sys
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from beetrack.evaluate import by_track, frame_match, match_events, merge_handovers, zone_tracks
from beetrack.track import iou_matrix

OFF = {fn.__name__: fn for fn in (by_track, frame_match, match_events, merge_handovers, zone_tracks, iou_matrix)}
SOURCES = {name: hashlib.sha256((REPO / name).read_text(encoding="utf-8").encode()).hexdigest()
           for name in ("beetrack/track.py", "beetrack/evaluate.py")}


def side(row, poly):
    x, y = row[2:4] + row[4:6] / 2
    return bool(cv2.pointPolygonTest(np.asarray(poly, np.float32), (float(x), float(y)), False) >= 0)


def xyxy(rows):
    return np.c_[rows[:, 2:4], rows[:, 2:4] + rows[:, 4:6]]


def owners(gt, pr, threshold, detail_indices):
    under = np.full(len(pr), -1, int)
    details = [None] * len(pr)
    for frame in np.unique(pr[:, 0]):
        pi, gi = np.flatnonzero(pr[:, 0] == frame), np.flatnonzero(gt[:, 0] == frame)
        matrix = OFF['iou_matrix'](xyxy(gt[gi]), xyxy(pr[pi]))
        if len(gi):
            ii, jj = linear_sum_assignment(-matrix)
            for i, j in zip(ii, jj):
                if matrix[i, j] >= threshold:
                    under[pi[j]] = gi[i]
        for j, p in enumerate(pi):
            if p not in detail_indices:
                continue
            candidates = np.flatnonzero(matrix[:, j] >= threshold)
            owner = under[p]
            local = int(np.flatnonzero(gi == owner)[0]) if owner >= 0 else None
            reverse = int(np.count_nonzero(matrix[local] >= threshold)) if local is not None else None
            details[p] = dict(gt_row=int(owner), gt_id=int(gt[owner, 1]) if owner >= 0 else None,
                              prediction_row=pr[p].tolist(),
                              assigned_gt_row=gt[owner].tolist() if owner >= 0 else None,
                              alternatives=[int(gt[gi[k], 1]) for k in candidates],
                              alternative_evidence=[dict(gt_row=gt[gi[k]].tolist(), iou=float(matrix[k, j])) for k in candidates],
                              reverse_evidence=[dict(prediction_row=pr[pi[q]].tolist(), iou=float(matrix[local, q]))
                                                for q in np.flatnonzero(matrix[local] >= threshold)] if local is not None else [],
                              above_threshold_count=len(candidates), reverse_multiplicity=reverse,
                              forced=bool(local is not None and len(candidates) == 1 and reverse == 1))
    expected = OFF['frame_match'](gt, pr, threshold)
    actual = np.array([gt[g, 1] if g >= 0 else -1 for g in under])
    assert np.array_equal(expected, actual)
    return under, details


def cardinality(graph):
    return int(graph[linear_sum_assignment(graph, maximize=True)].sum()) if graph.size else 0


def audit(gt, pr, poly, threshold, c_segment_ids=None):
    gt, pr = OFF['by_track'](gt), OFF['by_track'](pr)
    G, P = OFF['zone_tracks'](gt, poly), OFF['zone_tracks'](pr, poly)
    canonical = {p: [j for j, direction in P[p][3] if direction == P[p][0]][-1]
                 for p in P if P[p][0]}
    detail_indices = {k for j in canonical.values() for k in (j - 1, j)}
    under, detail = owners(gt, pr, threshold, detail_indices)
    gev = [g for g in G if G[g][0]]
    pev = [p for p in P if P[p][0]]
    graph = np.zeros((len(pev), len(gev)), int)
    witnesses = [[] for _ in pev]
    for a, p in enumerate(pev):
        direction, _, _, crossings = P[p]
        for j, d in crossings:
            if d != direction:
                continue
            for k in (j - 1, j):
                if under[k] < 0:
                    continue
                g = int(gt[under[k], 1])
                if G[g][0] == d and any(dg == d and abs(gt[q, 0] - pr[j, 0]) <= 5 for q, dg in G[g][3]):
                    graph[a, gev.index(g)] = 1
                    witnesses[a].append(dict(gt_id=g, prediction_crossing_frame=int(pr[j, 0]),
                                             matched_endpoint_frame=int(pr[k, 0]),
                                             gt_crossing_frames=[int(gt[q, 0]) for q, dg in G[g][3]
                                                                 if dg == d and abs(gt[q, 0] - pr[j, 0]) <= 5]))
    maximum = cardinality(graph)
    selected = {int(a) for a, b in zip(*linear_sum_assignment(graph, maximize=True)) if graph[a, b]} if graph.size else set()
    if threshold == .5:
        assert maximum == OFF['match_events'](gt, pr, poly)['tp']
    records = []
    for a, p in enumerate(pev):
        d, _, _, crossings = P[p]
        j = [j for j, dj in crossings if dj == d][-1]
        left, right = j - 1, j
        gl, gr = int(under[left]), int(under[right])
        forced = detail[left]['forced'] and detail[right]['forced']
        sides = [side(pr[k], poly) for k in (left, right)]
        gt_sides = [side(gt[g], poly) if g >= 0 else None for g in (gl, gr)]
        net = None
        if min(gl, gr) < 0:
            assigned = '01_no_owner'
        elif gt_sides != sides:
            assigned = '03_wrong_side_geometry'
        elif gt[gl, 1] != gt[gr, 1]:
            assigned = '04_different_bees'
        else:
            net = G[int(gt[gl, 1])][0]
            assigned = ('05_same_bee_genuine_transient_net0' if net is None else
                        '06_same_bee_official_direction' if net == d else '07_same_bee_opposite_net')
            if net is None and c_segment_ids is not None and int(gt[gl, 1]) not in c_segment_ids:
                assigned = '05b_same_bee_net0_outside_C'
        if gl >= 0 and gr >= 0 and gt[gl, 1] == gt[gr, 1]:
            net = G[int(gt[gl, 1])][0]
        category = '02_ambiguous_owner' if min(gl, gr) >= 0 and not forced else assigned
        legal = [gev[k] for k in np.flatnonzero(graph[a])]
        graph_status = ('no_legal_edge' if not legal else 'always_matched_in_maximum' if
                        cardinality(np.delete(graph, a, axis=0)) < maximum else 'matching_ambiguous')
        records.append(dict(pred_id=p, direction=d, frames=[int(pr[k, 0]) for k in (left, right)],
                            prediction_sides=sides, gt_center_sides=gt_sides,
                            owners=[detail[k] for k in (left, right)], same_segment_net=net,
                            owner_in_C=[int(gt[g, 1]) in c_segment_ids if g >= 0 else None for g in (gl, gr)] if c_segment_ids is not None else None,
                            both_forced=forced, assigned_category=assigned, category=category,
                            legal_gt_events=legal, legal_edge_witnesses=witnesses[a], graph_status=graph_status,
                            official_selected=a in selected,
                            graph_bucket='no_legal_edge' if not legal else 'eligible_selected' if a in selected else 'eligible_unselected'))
    return dict(threshold=threshold, gt_events=len(gev), predicted_events=len(pev), official_graph_tp=maximum,
                categories=dict(sorted(Counter(r['category'] for r in records).items())),
                assigned_categories=dict(sorted(Counter(r['assigned_category'] for r in records).items())),
                graph_statuses=dict(sorted(Counter(r['graph_status'] for r in records).items())),
                categories_by_graph_bucket={bucket: dict(sorted(Counter(r['category'] for r in records if r['graph_bucket'] == bucket).items()))
                                            for bucket in ('no_legal_edge', 'eligible_selected', 'eligible_unselected')},
                both_forced=sum(r['both_forced'] for r in records), records=records)


POLY = [[0, -10], [10, -10], [10, 10], [0, 10]]


def rows(tid, centers):
    return [[frame, tid, x - 1, -1, 2, 2] for frame, x in centers]


def self_test():
    cases = {}
    for direction, sign in [('in', 1), ('out', -1)]:
        # Every case is mirrored to exercise both official directions.
        outside, inside = (-2, 2) if sign == 1 else (2, -2)
        p = rows(100, [(1, outside), (2, inside)])
        fixtures = {
            'net0': (rows(1, [(1, outside), (2, inside), (3, outside)]), p, '05_same_bee_genuine_transient_net0'),
            'official': (rows(1, [(1, outside), (2, inside)]), p, '06_same_bee_official_direction'),
            'different': (rows(1, [(1, outside)]) + rows(2, [(2, inside)]), p, '04_different_bees'),
            'missing': (rows(1, [(1, outside)]), p, '01_no_owner'),
            'ambiguous_gt': (rows(1, [(1, outside), (2, inside)]) + rows(2, [(1, outside), (2, inside)]), p, '02_ambiguous_owner'),
            'ambiguous_reverse': (rows(1, [(1, outside), (2, inside)]), p + rows(101, [(1, outside), (2, inside)]), None),
            'wrong_side': (rows(1, [(1, outside), (2, -.1 if sign == 1 else .1)]),
                           rows(100, [(1, outside), (2, .1 if sign == 1 else -.1)]), '03_wrong_side_geometry'),
            'opposite': (rows(1, [(0, inside), (1, outside), (2, inside), (3, outside)]), p, '07_same_bee_opposite_net'),
        }
        for name, (g, predicted, category) in fixtures.items():
            for threshold in (.5, .3):
                result = audit(np.array(g, float), np.array(predicted, float), POLY, threshold)
                assert sum(result['categories'].values()) == result['predicted_events']
                if category:
                    assert result['records'][0]['category'] == category, (direction, name, result)
                if name == 'ambiguous_reverse':
                    assert result['both_forced'] == 0
                    assert result['categories'] == {'01_no_owner': 1, '02_ambiguous_owner': 1}
                cases[f'{direction}/{name}/{threshold}'] = dict(categories=result['categories'], graph=result['graph_statuses'])
        g = np.array(rows(1, [(1, outside), (2, inside), (3, outside), (4, inside)]), float)
        p = np.array(rows(100, [(1, outside), (2, inside)]) + rows(101, [(3, outside), (4, inside)]), float)
        for threshold in (.5, .3):
            r = audit(g, p, POLY, threshold)
            assert r['graph_statuses'] == {'matching_ambiguous': 2} and r['official_graph_tp'] == 1
            cases[f'{direction}/legal_competition/{threshold}'] = {'graph': r['graph_statuses']}
    # Canonical means LAST crossing in the net direction, with adjacent actual observations.
    g = np.array(rows(1, [(1, -2), (2, 2), (3, -2), (9, 2)]), float)
    r = audit(g, g, POLY, .5)['records'][0]
    assert r['frames'] == [3, 9] and r['both_forced']
    cases['last_crossing_gap'] = {'frames': r['frames']}
    g = np.array(rows(1, [(1, -2), (2, 2)]), float)
    p = g.copy()
    p[:, 1] = 100
    p[:, 3] += .8
    a, b = audit(g, p, POLY, .5), audit(g, p, POLY, .3)
    assert a['categories'] == {'01_no_owner': 1}
    assert b['categories'] == {'06_same_bee_official_direction': 1}
    cases['threshold_control'] = {'.5': a['categories'], '.3': b['categories']}
    boundary = np.array(rows(1, [(1, -2), (2, 0)]), float)
    r = audit(boundary, boundary, POLY, .5)['records'][0]
    assert r['prediction_sides'] == [False, True]
    cases['boundary_inclusive'] = {'sides': r['prediction_sides']}
    g = np.array(rows(1, [(1, -2), (2, 2), (3, -2)]), float)
    p = np.array(rows(100, [(1, -2), (2, 2)]), float)
    for threshold in (.5, .3):
        inside_c = audit(g, p, POLY, threshold, [1])
        outside_c = audit(g, p, POLY, threshold, [2])
        assert inside_c['categories'] == {'05_same_bee_genuine_transient_net0': 1}
        assert outside_c['categories'] == {'05b_same_bee_net0_outside_C': 1}
        cases[f'C_membership/{threshold}'] = {'selected': inside_c['categories'], 'outside': outside_c['categories']}
        # Even with two different owners, geometry has priority before identity.
        g = np.array(rows(1, [(1, -2)]) + rows(2, [(2, -.1)]), float)
        p = np.array(rows(100, [(1, -2), (2, .1)]), float)
        r = audit(g, p, POLY, threshold)
        assert r['categories'] == {'03_wrong_side_geometry': 1}
        cases[f'different_owner_wrong_side_priority/{threshold}'] = r['categories']
        g = np.array(rows(1, [(1, -2), (2, 2), (3, -2)]), float)
        p = np.array(rows(100, [(1, -2), (2, 2)]), float)
    return dict(synthetic_cases=len(cases), cases=cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--confirmed-prereg', help='Published prereg commit; required before opening real files')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        if not args.manifest or not args.confirmed_prereg:
            parser.error('Real audit requires manifest and confirmed published prereg')
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        result = {'prereg': args.confirmed_prereg, 'outputs': []}
        for item in manifest:
            if item['arm'] == 'C' and 'c_segment_ids' not in item:
                raise ValueError('C arm requires full prepared segment IDs selected by §126 C')
            hashes = {k: hashlib.sha256(Path(item[k]).read_bytes()).hexdigest() for k in ('gt', 'prediction', 'polygon')}
            polygon = json.loads(Path(item['polygon']).read_text(encoding='utf-8'))
            if isinstance(polygon, dict):
                polygon = polygon[item['sequence']]
            arrays = [np.loadtxt(item[k], delimiter=',', ndmin=2) for k in ('gt', 'prediction')]
            g, p = [a[:, :6] if a.size else np.zeros((0, 6)) for a in arrays]
            for threshold in (.5, .3):
                result['outputs'].append(dict(arm=item['arm'], sequence=item['sequence'], seed=item.get('seed'),
                                              input_sha256=hashes, **audit(g, p, polygon, threshold, item.get('c_segment_ids'))))
    result['official_source_sha256'] = SOURCES
    result['audit_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases' and k != 'outputs'}))


if __name__ == '__main__':
    main()
