"""Independent CPU verifier for registered endpoint-inventory relaxation.

Only run after the six native float arrays and runner scores are sealed.
The independent path does not use official track splitting/events/window bins.
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json
import sys
import hashlib
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path.cwd()
assert (ROOT/'beetrack/p0.py').is_file(), 'Run from the BeeTrack repository root.'
sys.path.insert(0, str(ROOT))
from beetrack.evaluate import by_track, match_events, zone_tracks
from beetrack.p0 import events, window_err
from permutation_fixture import exhaustive, row, POLY, timing_rows, neutral_rows


def rebuild(raw, polygon):
    """Rebuild official run IDs, with an explicit per-original-ID gap scan."""
    groups = defaultdict(list)
    for r in np.asarray(raw, float):
        groups[float(r[1])].append(r.copy())
    rows = []
    for identity, group in groups.items():
        group.sort(key=lambda r: r[0])
        run = 0
        previous = None
        for r in group:
            if previous is not None and r[0]-previous > 25:
                run += 1
            previous = r[0]
            r[1] = identity + 100000*run
            rows.append(r)
    rows.sort(key=lambda r: (r[1], r[0]))
    rows = np.asarray(rows, float).reshape(-1, np.asarray(raw).shape[1])
    poly = np.asarray(polygon, np.float32)
    groups = defaultdict(list)
    for j, r in enumerate(rows):
        groups[r[1]].append(j)
    records, event_array = {}, []
    a = b = 0
    for identity, indices in groups.items():
        sides = [int(cv2.pointPolygonTest(poly, (float(rows[j, 2]+rows[j, 4]/2),
                                                    float(rows[j, 3]+rows[j, 5]/2)), False) >= 0)
                 for j in indices]
        a += sides[0]
        b += sides[-1]
        direction = sides[-1]-sides[0]
        crossings = [(indices[k], sides[k]-sides[k-1]) for k in range(1, len(indices))
                     if sides[k] != sides[k-1]]
        records[identity] = dict(direction=direction, indices=indices, crossings=crossings)
        if direction:
            last = [j for j, d in crossings if d == direction][-1]
            event_array.append([int(rows[last, 0]), direction])
    return rows, records, np.asarray(event_array, int).reshape(-1, 2), (len(groups), a, b)


def bins(event_array, nf):
    """Explicit event insertion, independently from official bincount routines."""
    all_bins = []
    for offset in range(30):
        shift = 25*offset
        phase = [[0, 0] for _ in range((nf+shift+749)//750)]
        for frame, direction in event_array:
            assert 1 <= frame <= nf
            phase[(int(frame)+shift-1)//750][0 if direction == 1 else 1] += 1
        all_bins.extend(phase)
    return np.asarray(all_bins, int)


def minimum_floor(K, a, b, gi, go, bounds=None):
    """Project the median interval onto the feasible integer s interval.

    |b-s-gi|+|a-s-go| is flat between its two targets, linear outside.
    Thus clamping either target gives an exact minimizer; no enumeration.
    K,a,b may be sums over seeds, and gi/go then use threefold GT.
    """
    lo, hi = bounds if bounds is not None else (max(0, a+b-K), min(a, b))
    target1, target2 = sorted((b-gi, a-go))
    s = min(hi, max(lo, target1))
    value = abs(b-s-gi)+abs(a-s-go)
    signed = abs((b-a)-(gi-go))
    assert signed <= value
    return dict(lo=lo, hi=hi, s=s, numerator=value, signed_numerator=signed,
                median_interval=[target1, target2])


def independent_f1(grows, grecords, prows, precords, threshold):
    """Own IoU assignment, event graph, and augmenting-path cardinality match."""
    gf, pf = defaultdict(list), defaultdict(list)
    for j, r in enumerate(grows):
        gf[int(r[0])].append(j)
    for j, r in enumerate(prows):
        pf[int(r[0])].append(j)
    under = {}
    for frame, pind in pf.items():
        gind = gf.get(frame, [])
        if not gind:
            continue
        g, p = grows[gind], prows[pind]
        gl, pl = g[:, 2:4], p[:, 2:4]
        gr, pr = gl+g[:, 4:6], pl+p[:, 4:6]
        extent = np.maximum(0, np.minimum(gr[:, None, :], pr[None, :, :])-
                            np.maximum(gl[:, None, :], pl[None, :, :]))
        overlap = extent[:, :, 0]*extent[:, :, 1]
        iou = overlap / (np.prod(g[:, 4:6], axis=1)[:, None]+
                         np.prod(p[:, 4:6], axis=1)[None, :]-overlap+1e-9)
        rr, cc = linear_sum_assignment(-iou)
        for i, j in zip(rr, cc):
            if iou[i, j] >= threshold:
                under[pind[j]] = grows[gind[i], 1]
    gevents = {i: r for i, r in grecords.items() if r['direction']}
    pevents = {i: r for i, r in precords.items() if r['direction']}
    graph = {}
    for identity, record in pevents.items():
        candidates = set()
        for j, direction in record['crossings']:
            if direction != record['direction']:
                continue
            for gj in {under.get(j-1), under.get(j)}:
                target = gevents.get(gj)
                if target and target['direction'] == direction and any(
                        gd == direction and abs(grows[k, 0]-prows[j, 0]) <= 5
                        for k, gd in target['crossings']):
                    candidates.add(gj)
        graph[identity] = candidates
    owner = {}

    def augment(identity, seen):
        for candidate in sorted(graph[identity]):
            if candidate not in seen:
                seen.add(candidate)
                if candidate not in owner or augment(owner[candidate], seen):
                    owner[candidate] = identity
                    return True
        return False

    tp = sum(augment(i, set()) for i in graph)
    return dict(tp=tp, fp=len(pevents)-tp, fn=len(gevents)-tp,
                f1=Fraction(2*tp, max(1, len(pevents)+len(gevents))))


def synthetic_extended():
    exhaustive()
    # Reuse the same-totals/different-last-crossing and local-neutral fixtures.
    ta, tb = [rebuild(timing_rows(swap), POLY)[2] for swap in (False, True)]
    assert ta.tolist() == [[1000, 1], [700, 1], [1100, -1]]
    assert tb.tolist() == [[100, 1], [1000, 1], [1100, -1]]
    assert int(np.abs(bins(ta,1600)-bins(tb,1600)).sum()) == 48
    local_events = []
    for swap in (False, True):
        nr = neutral_rows(swap)
        assert rebuild(nr, POLY)[2].tolist() == []
        local_events.append(rebuild([r for r in nr if 499 <= r[0] <= 503], POLY)[2].tolist())
    assert local_events == [[], [[502, 1], [502, -1]]]
    # Before stitching: outside->inside fragment and outside->inside fragment.
    # Stitch deletes an inside endpoint and an outside start: inventory changes.
    before = [row(1, 1, 0), row(2, 1, 1), row(3, 2, 0), row(4, 2, 1)]
    after = [r.copy() for r in before]
    for r in after:
        r[1] = 1
    _, _, eb, invb = rebuild(before, POLY)
    _, _, ea, inva = rebuild(after, POLY)
    assert (eb[:, 1].sum(), ea[:, 1].sum()) == (2, 1)
    # GT=the stitched trajectory: old signed-flow bound is 1 but new W=0.
    old_bound = abs((invb[2]-invb[1])-int(ea[:, 1].sum()))
    assert old_bound == 1 and window_err(ea, ea, 4) == 0
    # Arbitrary suffix rotation preserves inventory if every new junction gap<=25.
    # Swap at t=25 creates junction gaps of 1 and 49.
    # One swapped run is newly split, adding opposite-side endpoints and changing K.
    # A ends at 1 then suffix at 26 (gap25); B prefix at25 then suffix at50(gap25).
    irregular = [row(1, 1, 0), row(26, 1, 1), row(25, 2, 1), row(50, 2, 1)]
    rotated = [r.copy() for r in irregular]
    for r in rotated:
        if r[0] > 25:
            r[1] = 3-r[1]
    _, _, ex, ix = rebuild(irregular, POLY)
    _, _, ey, iy = rebuild(rotated, POLY)
    assert ix != iy and int(ex[:, 1].sum()) != int(ey[:, 1].sum())
    safe = [row(f, i, (f+i)%2, i-1) for i in (1, 2, 3) for f in range(1, 10)]
    _, _, e0, i0 = rebuild(safe, POLY)
    for permutation in __import__('itertools').permutations((1, 2, 3)):
        rotated = [r.copy() for r in safe]
        for r in rotated:
            if r[0] >= 5:
                r[1] = permutation[int(r[1])-1]
        _, _, ep, ip = rebuild(rotated, POLY)
        assert ip == i0 and ep[:, 1].sum() == e0[:, 1].sum()
    return dict(timing=dict(direct_events=ta.tolist(), swapped_events=tb.tolist(),
                            phase_absolute_error_sum=48, raw_W=Fraction(48,30),
                            W_percent=Fraction(100*48,30*3)),
                local_events=local_events,
                stitch=dict(before_events=eb.tolist(), after_events=ea.tolist(),
                            before_inventory=invb, after_inventory=inva,
                            obsolete_signed_bound=old_bound, after_W=0),
                unsafe_rotation=dict(before_events=ex.tolist(), after_events=ey.tolist(),
                                     before_inventory=ix, after_inventory=iy),
                safe_three_suffix_rotations_checked=6)


def fraction_json(value):
    if isinstance(value, Fraction):
        return dict(numerator=value.numerator, denominator=value.denominator, value=float(value))
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(type(value).__name__)


def verify(directory):
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    seal = json.loads((directory/'scored_seal.json').read_text())
    freeze = json.loads((directory/'freeze.json').read_text())
    evidence = json.loads((directory/'evidence.json').read_text())
    assert freeze['prereg_sha'] == '01fb9c21a3565c16df7884f66a03d2dc1b9e4073'
    assert seal['freeze_sha256'] == digest(directory/'freeze.json')
    assert seal['evidence_sha256'] == digest(directory/'evidence.json')
    assert len(freeze['cells']) == len(evidence['cells']) == 6
    report = dict(prereg_sha=freeze['prereg_sha'], synthetic=synthetic_extended(), cells=[], summary={})
    # Load only allowlisted dev GT, after verifying all native arrays' checksums.
    for frozen in freeze['cells']:
        assert digest(directory/frozen['mot']) == frozen['mot_sha256']
    zones = json.loads((ROOT/'beetrack/zones.json').read_text())
    for frozen, reference in zip(freeze['cells'], evidence['cells']):
        h, seed = frozen['h'], frozen['seed']
        assert h in (33, 35) and seed in (0, 1, 2)
        assert (h, seed) == (reference['h'], reference['seed'])
        raw = np.load(directory/frozen['mot'])
        polygon = zones[f'BEE24-{h}-head']
        gt = np.loadtxt(ROOT/f'runs/gt/dev/BEE24-{h}-head/gt/gt.txt', delimiter=',', ndmin=2)[:, :6]
        gr, grec, ge, _ = rebuild(gt, polygon)
        pr, prec, pe, inventory = rebuild(raw, polygon)
        assert np.array_equal(pr, by_track(raw))
        assert np.array_equal(gr, by_track(gt))
        assert np.array_equal(pe, events(raw, polygon))
        assert np.array_equal(ge, events(gt, polygon))
        assert pe.tolist() == reference['events'] and ge.tolist() == reference['gt_events']
        assert dict(zip(('K', 'a', 'b'), inventory)) == reference['endpoints']
        pb, gb = bins(pe, 3950), bins(ge, 3950)
        assert pb.tolist() == reference['phase_counts']
        numerator = int(np.abs(pb-gb).sum())
        N = len(ge)
        gi, go = int(sum(ge[:, 1] == 1)), int(sum(ge[:, 1] == -1))
        W = Fraction(100*numerator, 30*N)
        assert abs(float(W)-reference['W']) < 1e-10
        assert abs(float(W)-100*window_err(pe, ge, 3950)/N) < 1e-10
        floor = minimum_floor(*inventory, gi, go)
        assert floor['numerator'] == reference['floor']['numerator']
        assert floor['signed_numerator'] == reference['floor']['signed']
        assert floor['s'] == reference['floor']['best_s']
        assert floor['lo'] == reference['floor']['lo'] and floor['hi'] == reference['floor']['hi']
        f1 = {}
        for threshold in ((0.5, 0.3) if h == 33 else (0.5,)):
            metric = independent_f1(gr, grec, pr, prec, threshold)
            official = reference['official'][str(threshold)]
            for key in ('tp', 'fp', 'fn'):
                assert metric[key] == official[key], (h, seed, threshold, key)
            assert abs(float(metric['f1'])-official['f1']) < 1e-12
            if threshold == 0.5:
                m = match_events(gt, raw, polygon)
                assert metric['tp'] == m['tp']
                assert metric['fp'] == m['pred']-m['tp']
                assert metric['fn'] == m['gt']-m['tp']
            f1[str(threshold)] = metric
        # Manual segment endpoint examples accompany the full machine replay.
        examples = []
        for identity, rec in prec.items():
            if rec['direction'] and len(examples) < 3:
                start, end = pr[rec['indices'][0]], pr[rec['indices'][-1]]
                examples.append(dict(id=identity, first_row=start.tolist(), last_row=end.tolist(),
                                     direction=rec['direction'],
                                     crossing_frames=[[int(pr[j, 0]), d] for j, d in rec['crossings']]))
        report['cells'].append(dict(h=h, seed=seed, endpoints=dict(zip(('K','a','b'), inventory)),
                                   gt_in=gi, gt_out=go, W=W, phase_L1_integer=numerator,
                                   floor=floor, floor_percent=Fraction(100*floor['numerator'], N),
                                   f1=f1, endpoint_examples=examples))
    for h in (33, 35):
        cells = [c for c in report['cells'] if c['h'] == h]
        refs = [c for c in evidence['cells'] if c['h'] == h]
        N = cells[0]['gt_in']+cells[0]['gt_out']
        total_bounds = (sum(c['floor']['lo'] for c in cells), sum(c['floor']['hi'] for c in cells))
        inv = [sum(c['endpoints'][key] for c in cells) for key in ('K','a','b')]
        ef = minimum_floor(*inv, 3*cells[0]['gt_in'], 3*cells[0]['gt_out'], bounds=total_bounds)
        e3_floor = Fraction(100*ef['numerator'], 3*N)
        # Fractions retain the 1/3 counts before absolute error; no float rounding.
        gb = bins(np.asarray(refs[0]['gt_events'], int).reshape(-1,2), 3950)
        e3_count_error = sum(abs(sum(Fraction(c['phase_counts'][j][d], 3) for c in refs)-int(gb[j,d]))
                             for j in range(len(gb)) for d in (0,1))
        e3 = 100*e3_count_error/(30*N)
        mean = sum(c['W'] for c in cells)/3
        mean_floor = sum(c['floor_percent'] for c in cells)/3
        target = Fraction('20.95' if h == 33 else '5.05')
        result = dict(single=mean, single_floor=mean_floor, maximum_mean_gain=mean-mean_floor,
                      e3=e3, e3_floor=e3_floor, e3_s_range=list(total_bounds), e3_best_s=ef['s'],
                      mean_2pp_precluded=mean-mean_floor < 2,
                      e3_strict_target_precluded=e3_floor >= target,
                      e3_floor_detail=ef)
        for key in ('single','single_floor','maximum_mean_gain','e3','e3_floor','e3_best_s',
                    'mean_2pp_precluded','e3_strict_target_precluded'):
            assert abs(float(result[key])-float(evidence['summary'][str(h)][key])) < 1e-10, (h,key)
        assert result['e3_s_range'] == evidence['summary'][str(h)]['e3_s_range']
        report['summary'][str(h)] = result
    report['checks'] = 'All six native arrays, independent endpoint/events/windows/F1 graph/floors/E3 passed'
    report['source_sha256'] = digest(Path(__file__))
    report['evidence_sha256'] = digest(directory/'evidence.json')
    destination = directory/'independent.json'
    destination.write_text(json.dumps(report, default=fraction_json, indent=2)+'\n')
    print(json.dumps(report['summary'], default=fraction_json, indent=2))
    print(str(destination))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        verify(Path(sys.argv[1]))
    else:
        print(json.dumps(synthetic_extended(), default=fraction_json, indent=2))
