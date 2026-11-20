"""Independent stdlib-only event replay. CLI: python replay179.py input.json.

JSON input: {"nf": frame_count, "pred": [[frame,cell,dir],...],
             "gt": [[frame,cell,dir],...]}; direction 0=in, 1=out.
Frames are sequence-relative, one-based. Run --self-test without reading data.
"""
import argparse
import gzip
import json
from collections import defaultdict


def _validate(events, nf):
    result = []
    for row in events:
        if len(row) != 3 or any(not isinstance(x, int) or isinstance(x, bool) for x in row):
            raise ValueError('events must be integer frame/cell/direction triples')
        frame, cell, direction = row
        if not 1 <= frame <= nf or not 0 <= cell < 32 or direction not in (0, 1):
            raise ValueError('event outside frame/cell/direction range')
        result.append(tuple(row))
    return result


def _match(p, g):
    # Sparse exact-time/cell bins retain duplicate GT as separate vertices.
    bins = defaultdict(list)
    for j, (frame, cell, _) in enumerate(g):
        bins[frame, cell].append(j)
    edges = []
    for frame, cell, _ in p:
        edges.append([j for t in range(frame - 5, frame + 6)
                      for c in ((cell - 1) % 32, cell, (cell + 1) % 32)
                      for j in bins[t, c]])
    owner = [-1] * len(g)
    # Explicit DFS stack is equivalent to recursive Kuhn augmentation, without
    # sequence-size-dependent recursion depth limits.
    for root in range(len(p)):
        seen = set()
        stack = [(root, iter(edges[root]))]
        route = []
        while stack:
            i, choices = stack[-1]
            j = next(choices, None)
            if j is None:
                stack.pop()
                if route:
                    route.pop()
                continue
            if j in seen:
                continue
            seen.add(j)
            if owner[j] == -1:
                owner[j] = i
                for prior_i, prior_j in reversed(route):
                    owner[prior_j] = prior_i
                break
            route.append((i, j))
            stack.append((owner[j], iter(edges[owner[j]])))
    assert len([i for i in owner if i >= 0]) == len(set(i for i in owner if i >= 0))
    return sum(i >= 0 for i in owner)


def independent(pred, gt, nf):
    """Return independent maximum matches and standard 30-phase W percent.

    in/out error is signed whole-sequence count error; window errors are sums
    of absolute directional errors, averaged over 30 phases, then normalized
    by total GT. Empty GT gives W=None and empty/empty F1=0.
    """
    if not isinstance(nf, int) or isinstance(nf, bool) or nf <= 0:
        raise ValueError('nf must be a positive integer')
    pred, gt = _validate(pred, nf), _validate(gt, nf)
    result = {}
    directional_window = []
    for direction, name in enumerate(('in', 'out')):
        p = [e for e in pred if e[2] == direction]
        g = [e for e in gt if e[2] == direction]
        tp = _match(p, g)
        err_sum = 0
        for phase in range(30):
            shift = phase * 25
            # Count each sequence independently; final partial window included.
            last_window = (nf - 1 + shift) // 750
            pc, gc = [0] * (last_window + 1), [0] * (last_window + 1)
            for frame, _, _ in p:
                pc[(frame - 1 + shift) // 750] += 1
            for frame, _, _ in g:
                gc[(frame - 1 + shift) // 750] += 1
            err_sum += sum(abs(a - b) for a, b in zip(pc, gc))
        raw = err_sum / 30
        directional_window.append(raw)
        result[name] = dict(tp=tp, pred=len(p), gt=len(g), error=len(p) - len(g),
                            f1=2 * tp / (len(p) + len(g)) if p or g else 0,
                            err=raw)
    result.update(tp=sum(result[d]['tp'] for d in ('in', 'out')),
                  pred=len(pred), gt=len(gt), err=sum(directional_window))
    result['f1'] = 2 * result['tp'] / (len(pred) + len(gt)) if pred or gt else 0
    result['W'] = 100 * result['err'] / len(gt) if gt else None
    result['directional_abs_error'] = sum(abs(result[d]['error']) for d in ('in', 'out'))
    return result


def _self_test():
    assert independent([], [], 1)['f1'] == 0
    assert independent([], [], 1)['W'] is None
    assert independent([(6, 0, 0)], [(1, 31, 0)], 6)['tp'] == 1
    assert independent([(7, 0, 0)], [(1, 31, 0)], 7)['tp'] == 0
    assert independent([(1, 0, 0)], [(1, 0, 1)], 1)['tp'] == 0
    assert independent([(1, 0, 0)] * 2, [(1, 0, 0)], 1)['tp'] == 1
    assert independent([(5, 0, 0), (1, 0, 0)], [(1, 0, 0), (10, 0, 0)], 10)['tp'] == 2
    r = independent([(751, 0, 0)], [(750, 0, 0)], 800)
    assert r['err'] == 2 / 30 and r['W'] == 100 * (2 / 30)
    # Opposite-direction whole-video errors must not cancel.
    r = independent([(1, 0, 0)], [(1, 0, 1)], 1)
    assert r['err'] == 2 and r['directional_abs_error'] == 2
    # Exhaustive tiny graph comparison uses subset enumeration, not matching.
    from itertools import product
    universe = [(1, 0, 0), (5, 1, 0), (10, 0, 0), (1, 31, 0)]
    for n in range(4):
        for p in product(universe, repeat=n):
            g = universe[:3]
            best = 0
            for choices in product(range(-1, len(g)), repeat=n):
                used = [j for j in choices if j >= 0]
                if len(set(used)) != len(used):
                    continue
                if all(j < 0 or (abs(p[i][0] - g[j][0]) <= 5 and
                       min(abs(p[i][1] - g[j][1]), 32 - abs(p[i][1] - g[j][1])) <= 1)
                       for i, j in enumerate(choices)):
                    best = max(best, len(used))
            assert _match(p, g) == best
    print('replay179: all synthetic checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    elif args.input:
        opener = gzip.open if args.input.endswith('.gz') else open
        with opener(args.input, 'rt', encoding='utf-8') as stream:
            data = json.load(stream)
        if 'folds' in data:
            for fold in data['folds']:
                for arm, frozen in fold['arms'].items():
                    metrics = independent(frozen['pred'], fold['gt'], data['nf'])
                    print(fold['target'], arm, json.dumps(metrics, sort_keys=True))
        else:
            print(json.dumps(independent(data['pred'], data['gt'], data['nf']), indent=2))
    else:
        parser.error('provide input JSON or --self-test')
