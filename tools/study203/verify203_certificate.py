"""Independent, stdlib-only count witness/certificate checks for section 203.

No solver, field loader, producer imports, filesystem reads, or GPU access.
Inputs are already-loaded mappings. This verifies count arithmetic and suffix
capacity only; native event extraction, owner labels and provenance require
separate verification. Run this file directly for tiny synthetic checks only.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

from fractions import Fraction
from itertools import combinations, product
from numbers import Integral
import math


def _int(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError('integer required')
    return int(value)


def _vector(values, size=None):
    result = [_int(v) for v in values]
    if size is not None and len(result) != size:
        raise ValueError('vector dimension differs')
    return result


def _box(values):
    if len(values) != 4:
        raise ValueError('four box coordinates required')
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
            raise ValueError('numeric box coordinate required')
        if not math.isfinite(value):
            raise ValueError('finite box required')
        result.append(Fraction(str(value)))
    if min(result[2:]) <= 0:
        raise ValueError('positive box size required')
    return result


def _overlap(a, b):
    # Registered supplier uses binary64 xywh IoU without an epsilon. Preserve
    # that geometry convention; dual/count arithmetic below remains exact.
    a, b = [float(v) for v in a], [float(v) for v in b]
    x, y, w, h = a
    u, v, s, t = b
    intersection = max(0, min(x+w, u+s)-max(x, u)) * max(0, min(y+h, v+t)-max(y, v))
    return intersection / (w*h + s*t - intersection) >= .5


def compile_resources(actions):
    """Reconstruct canonical capacity rows without trusting supplied incidence."""
    ids = [a['action_id'] for a in actions]
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError('unique nonempty action IDs required')
    resources, dates = [], {}
    for index, action in enumerate(actions):
        member = str(action['host_member'])
        host = _int(action['host_track_id'])
        row = [('host', member, host)]
        sources, frames = set(), set()
        for suffix in action['suffix']:
            source = tuple(suffix['source'])
            if len(source) != 2 or not isinstance(source[0], str) or not source[0] or _int(source[1]) < 0:
                raise ValueError('source SHA/index required')
            frame = _int(suffix['frame'])
            if frame < 1 or source in sources or frame in frames:
                raise ValueError('unique source rows and positive dates required')
            sources.add(source)
            frames.add(frame)
            row.append(('row', *source))
            dates.setdefault((member, frame), []).append((index, _box(suffix['box'])))
        if not sources:
            raise ValueError('nonempty suffix required')
        resources.append(row)
    conflicts = set()
    for entries in dates.values():
        for (i, a), (j, b) in combinations(entries, 2):
            if i != j and _overlap(a, b):
                conflicts.add(tuple(sorted((i, j))))
    for i, j in sorted(conflicts):
        key = ('overlap', *sorted((ids[i], ids[j])))
        resources[i].append(key)
        resources[j].append(key)
    names = list(dict.fromkeys(key for row in resources for key in row))
    lookup = {key: index for index, key in enumerate(names)}
    return names, [sorted({lookup[key] for key in row}) for row in resources]


def _bins(values, rows):
    if len(values) != rows or any(len(row) != 2 for row in values):
        raise ValueError('full30 two-direction bin shape required')
    return _vector([v for row in values for v in row])


def compile_problem(unit, family):
    if family not in ('S', 'P', 'A'):
        raise ValueError('unknown family')
    nf = _int(unit['nf'])
    if nf < 1:
        raise ValueError('positive clip length required')
    rows = sum((nf + 25*offset + 749)//750 for offset in range(30))
    pred, gt = _bins(unit['pred_sum'], rows), _bins(unit['gt_bins'], rows)
    if min(pred + gt) < 0 or sum(gt) % 30:
        raise ValueError('nonnegative counts and phase-consistent GT total required')
    actions, columns, inactive, pool_size = [], [], [], 0
    all_ids = [a['action_id'] for a in unit['actions']]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('duplicate action IDs')
    for action in unit['actions']:
        label = action['labels']['0.5']['class']
        if label not in ('S', 'D', 'U'):
            raise ValueError('formal owner label required')
        if family != 'A' and label != 'S' and not (family == 'P' and label == 'U'):
            continue
        pool_size += 1
        if any(not 1 <= _int(s['frame']) <= nf for s in action['suffix']):
            raise ValueError('suffix date outside clip')
        delta = _bins(action['delta_windows'], rows)
        if not any(delta):
            inactive.append(action['action_id'])
            continue
        actions.append(action)
        columns.append({i: v for i, v in enumerate(delta) if v})
    names, incidence = compile_resources(actions)
    return dict(r=[p-3*g for p, g in zip(pred, gt)], pred=pred, gt=gt,
                actions=actions, columns=columns, indices=incidence,
                resource_names=names, rows=rows, pool_size=pool_size, inactive=inactive)


def witness(problem, selected):
    table = {a['action_id']: i for i, a in enumerate(problem['actions'])}
    if len(set(selected)) != len(selected) or any(key not in table for key in selected):
        raise ValueError('duplicate or unknown selected action')
    used, delta = set(), [0] * len(problem['r'])
    bits = [0] * len(table)
    for key in selected:
        index = table[key]
        bits[index] = 1
        for resource in problem['indices'][index]:
            if resource in used:
                raise ValueError('selected actions violate Ax<=1')
            used.add(resource)
        for coordinate, value in problem['columns'][index].items():
            delta[coordinate] += value
    pred = [p+d for p, d in zip(problem['pred'], delta)]
    if any(v < 0 for v in pred):
        raise ValueError('negative physical prediction count')
    value = sum(abs(r+d) for r, d in zip(problem['r'], delta))
    return value, [pred[i:i+2] for i in range(0, len(pred), 2)], bits


def verify_l1_certificate(residual, columns, incidence, resource_count, proof):
    """Check a rational dual bound for arbitrary binary incidence Ax<=1.

    With |p_i|<=q and nonnegative prices z, require p.V_j+(A^T z)_j>=0.
    Then ||r+Vx||_1 >= (p.r-sum(z))/q for every nonnegative feasible x.
    Integer residual/columns and binary x permit rounding upward. No graph,
    matching, total-unimodularity, or solver optimality assumption is used.
    """
    r = _vector(residual)
    count = _int(resource_count)
    q = _int(proof['q'])
    p = _vector(proof['p'], len(r))
    prices = _vector(proof['prices'], count)
    if count < 0 or q <= 0 or any(abs(v) > q for v in p) or any(v < 0 for v in prices):
        raise ValueError('invalid dual domain')
    if len(columns) != len(incidence):
        raise ValueError('action dimensions differ')
    dots = []
    for column, resources in zip(columns, incidence):
        coords = {_int(i): _int(v) for i, v in column.items()}
        if any(not 0 <= i < len(r) for i in coords):
            raise ValueError('delta coordinate outside residual')
        resources = _vector(resources)
        if len(set(resources)) != len(resources) or any(not 0 <= k < count for k in resources):
            raise ValueError('invalid binary incidence')
        dot = sum(p[i]*v for i, v in coords.items())
        if dot + sum(prices[k] for k in resources) < 0:
            raise ValueError('negative reduced action cost')
        dots.append(dot)
    scaled = sum(a*b for a, b in zip(r, p)) - sum(prices)
    lower = max(0, -((-scaled)//q))
    _vector(proof['edge_dot_products'], len(columns))
    _int(proof['scaled_bound'])
    _int(proof['lower_bound'])
    expected = dict(p=p, q=q, prices=prices, edge_dot_products=dots,
                    scaled_bound=scaled, lower_bound=lower)
    if proof != expected:
        raise ValueError('certificate arithmetic or schema differs')
    return lower


def verify_certificate(problem, proof):
    return verify_l1_certificate(problem['r'], problem['columns'], problem['indices'],
                                 len(problem['resource_names']), proof)


def verify_answer(unit, family, answer):
    """Check count answer plus serialized incidence/order and member masks."""
    problem = compile_problem(unit, family)
    lower = verify_certificate(problem, answer['certificate'])
    upper, pred, bits = witness(problem, answer['selected_ids'])
    baseline = sum(abs(v) for v in problem['r'])
    masks = {str(_int(i)): dict(candidate_ids=[]) for i in unit['member_indices']}
    for action, bit in zip(problem['actions'], bits):
        if bit:
            masks[str(_int(action['host_member_index']))]['candidate_ids'].append(action['action_id'])
    expected = dict(family=family, baseline_numerator=baseline, lower_bound=lower,
                    upper_bound=upper, gap=upper-lower, optimal=lower == upper,
                    denominator=3*sum(problem['gt']), witness_pred_sum=pred,
                    member_masks=masks, action_order=[a['action_id'] for a in problem['actions']],
                    resource_names=[list(k) for k in problem['resource_names']],
                    action_resources=problem['indices'], family_actions=problem['pool_size'],
                    active_actions=len(problem['actions']), zero_delta_count=len(problem['inactive']))
    if lower > upper:
        raise ValueError('lower certificate exceeds witness')
    for key, value in expected.items():
        actual = answer[key]
        if key == 'resource_names':
            actual = [list(k) for k in actual]
        if actual != value:
            raise ValueError('answer differs: ' + key)
    return dict(lower_bound=lower, upper_bound=upper, optimal=lower == upper)


def selftest():
    def action(key, member, host, source, box=(0, 0, 2, 2), frame=2):
        return dict(action_id=key, host_member=member, host_track_id=host,
                    suffix=[dict(source=['synthetic', source], frame=frame, box=box)])
    def conflicts(a, b):
        _, incidence = compile_resources([a, b])
        return bool(set(incidence[0]) & set(incidence[1]))
    a = action('a', 'H', 1, 1)
    assert conflicts(a, action('b', 'Q', 2, 1))  # source is global
    assert conflicts(a, action('b', 'H', 1, 2, (20, 0, 2, 2)))
    assert conflicts(a, action('b', 'H', 2, 2))
    assert not conflicts(a, action('b', 'Q', 1, 2))
    assert not conflicts(a, action('b', 'H', 2, 2, frame=3))
    assert _overlap(_box([0, 0, 3, 1]), _box([1, 0, 3, 1]))  # exactly .5
    # Odd cycle capacity: three pair resources, not a bipartite matching model.
    incidence = [[0, 2], [0, 1], [1, 2]]
    columns = [{0: -1}, {0: -1}, {0: -1}]
    proof = dict(p=[2], q=2, prices=[1, 1, 1], edge_dot_products=[-2]*3,
                 scaled_bound=1, lower_bound=1)
    lower = verify_l1_certificate([2], columns, incidence, 3, proof)
    feasible = []
    for bits in product((0, 1), repeat=3):
        if all(sum(bits[j] for j in range(3) if k in incidence[j]) <= 1 for k in range(3)):
            feasible.append(abs(2-sum(bits)))
    assert lower == min(feasible) == 1
    # Signed columns, rational rounding, negative bound clamp, and tampering.
    assert verify_l1_certificate([-3], [{0: 2}], [[0]], 1,
        dict(p=[-3], q=3, prices=[6], edge_dot_products=[-6], scaled_bound=3, lower_bound=1)) == 1
    assert verify_l1_certificate([0], [], [], 0,
        dict(p=[0], q=1, prices=[], edge_dot_products=[], scaled_bound=0, lower_bound=0)) == 0
    for bad in (dict(proof, prices=[0, 0, 0]), dict(proof, lower_bound=2), dict(proof, q=True)):
        try:
            verify_l1_certificate([2], columns, incidence, 3, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('tampered certificate accepted')
    # Exercise normalized input compilation and physical-count witness using
    # the minimum full30 phase shape, without importing the producer.
    full = dict(a, host_member_index=0, labels={'0.5': {'class': 'S'}},
                delta_windows=[[-1, 0] for _ in range(30)])
    unit = dict(nf=3, actions=[full], member_indices=[0, 1, 2],
                pred_sum=[[4, 0] for _ in range(30)],
                gt_bins=[[1, 0] for _ in range(30)])
    problem = compile_problem(unit, 'S')
    assert witness(problem, ['a'])[:2] == (0, [[3, 0] for _ in range(30)])
    assert verify_certificate(problem, dict(p=[0]*60, q=1,
        prices=[0]*len(problem['resource_names']), edge_dot_products=[0],
        scaled_bound=0, lower_bound=0)) == 0
    return dict(status='PASS', scope='tiny synthetic only; no field inputs',
                checks='host/source/member-date overlap; odd cycle exhaustive; signed rational dual; tampering')


if __name__ == '__main__':
    import json
    print(json.dumps(selftest(), sort_keys=True))
