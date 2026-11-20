"""CPU scoring for boundary events (one-based frame, cell 0..31, dir 0=in/1=out).

Run this file directly for synthetic regression checks. No data is read.
"""
import numpy as np

THRESHOLDS = tuple(i / 10 for i in range(1, 10)) + (0.95, 1.01)


def decode(scores, threshold, start=1):
    """Descending score greedy NMS; ties use ascending frame, cell, direction.

    Scores must be probabilities. The >1 calibration arm is therefore empty.
    A retained event suppresses its own direction within 5 frames and 1 cell.
    """
    s = np.asarray(scores)
    if s.ndim != 3 or s.shape[1:] != (32, 2):
        raise ValueError('scores must have shape (T,32,2)')
    if not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError('scores must be finite probabilities in [0,1]')
    if int(start) != start or start < 1:
        raise ValueError('start must be a positive integer')
    candidates = np.argwhere(s >= threshold)
    if not len(candidates):
        return []
    t, c, d = candidates.T
    order = np.lexsort((d, c, t, -s[t, c, d]))
    blocked = np.zeros(s.shape, dtype=bool)
    events = []
    for i in order:
        t, c, d = map(int, candidates[i])
        if blocked[t, c, d]:
            continue
        events.append((t + int(start), c, d))
        for cell in ((c - 1) % 32, c, (c + 1) % 32):
            blocked[max(0, t - 5):t + 6, cell, d] = True
    return events


def _events(events):
    out = []
    for event in events:
        if len(event) != 3:
            raise ValueError('events must be (frame,cell,dir) triples')
        f, c, d = event
        if any(int(x) != x for x in event) or f < 1 or not 0 <= c < 32 or d not in (0, 1):
            raise ValueError('invalid event frame, cell, or direction')
        out.append((int(f), int(c), int(d)))
    return out


def counts(pred, gt):
    """Maximum-cardinality tolerant event matching, with empty/empty F1=0.

    Return tp, pred, gt, f1 plus in/out dictionaries with the same fields.
    Duplicate event rows remain distinct events.
    """
    pred, gt = _events(pred), _events(gt)
    result = {}
    for direction, name in enumerate(('in', 'out')):
        p = [e for e in pred if e[2] == direction]
        g = [e for e in gt if e[2] == direction]
        edges = [[j for j, b in enumerate(g)
                  if abs(a[0] - b[0]) <= 5 and min(abs(a[1] - b[1]), 32 - abs(a[1] - b[1])) <= 1]
                 for a in p]
        # Iterative augmenting paths avoid recursion limits on long sequences.
        matched = {}
        for root in range(len(p)):
            queue, seen_p, seen_g, parent = [root], {root}, set(), {}
            end = None
            for i in queue:
                for j in edges[i]:
                    if j in seen_g:
                        continue
                    seen_g.add(j)
                    parent[j] = i
                    if j not in matched:
                        end = j
                        break
                    nxt = matched[j]
                    if nxt not in seen_p:
                        seen_p.add(nxt)
                        queue.append(nxt)
                if end is not None:
                    break
            if end is not None:
                reverse = {i: j for j, i in matched.items()}
                while True:
                    i = parent[end]
                    old = reverse.get(i)
                    matched[end] = i
                    if old is None:
                        break
                    end = old
        tp = len(matched)
        result[name] = dict(tp=tp, pred=len(p), gt=len(g), f1=2 * tp / max(len(p) + len(g), 1))
    for key in ('tp', 'pred', 'gt'):
        result[key] = sum(result[name][key] for name in ('in', 'out'))
    result['f1'] = 2 * result['tp'] / max(result['pred'] + result['gt'], 1)
    return result


def window_err(pred, gt, nf, start=1):
    """Standard P0 30-phase absolute count error, before GT normalization.

    Each phase shifts event frames and sequence end by 0..29 seconds (25fps).
    Input frames are absolute; start is the first frame of this sequence.
    """
    if int(nf) != nf or nf < 1:
        raise ValueError('nf must be a positive frame count')
    p, g = _events(pred), _events(gt)
    if any(not start <= e[0] < start + nf for e in p + g):
        raise ValueError('event outside sequence')
    errors = []
    for offset in range(30):
        nb = int(np.ceil((nf + offset * 25) / 750))
        hist = []
        for events in (p, g):
            h = np.zeros((nb, 2), dtype=int)
            for frame, _, direction in events:
                h[(frame - start + offset * 25) // 750, direction] += 1
            hist.append(h)
        errors.append(np.abs(hist[0] - hist[1]).sum())
    return float(np.mean(errors))


def score_cube(scores, gt, threshold, start=1):
    """Score exactly one sequence; W is percent of GT, None when GT is empty."""
    gt = _events(gt)
    pred = decode(scores, threshold, start)
    result = counts(pred, gt)
    result.update(events=pred, threshold=float(threshold), err=window_err(pred, gt, len(scores), start))
    result['W'] = 100 * result['err'] / len(gt) if gt else None
    return result


def calibrate(scores, gt, start=1):
    """Source-only threshold selection: F1, directional count error, threshold.

    Returns the selected score_cube dictionary. Never calibrate on target data.
    """
    gt = _events(gt)
    trials = [score_cube(scores, gt, threshold, start) for threshold in THRESHOLDS]
    def key(r):
        error = sum(abs(r[d]['pred'] - r[d]['gt']) for d in ('in', 'out'))
        return r['f1'], -error, r['threshold']
    return max(trials, key=key)


def _self_test():
    s = np.zeros((15, 32, 2))
    s[0, 0, 0] = s[0, 31, 0] = s[0, 0, 1] = .9
    s[5, 1, 0] = .8
    s[6, 0, 0] = .7
    assert decode(s, .5, 10) == [(10, 0, 0), (10, 0, 1), (16, 0, 0)]
    # Suppressed middle peak must not suppress a lower surviving peak.
    s = np.zeros((11, 32, 2))
    s[0, 0, 0], s[5, 0, 0], s[10, 0, 0] = .9, .8, .7
    assert len(decode(s, .5)) == 2
    assert counts([(1, 0, 0), (1, 0, 0)], [(1, 31, 0)])['tp'] == 1
    assert counts([(6, 0, 0)], [(1, 31, 0)])['tp'] == 1
    assert counts([(7, 0, 0)], [(1, 31, 0)])['tp'] == 0
    assert counts([(1, 0, 0)], [(1, 0, 1)])['tp'] == 0
    # First prediction can match both GT; second can only match the first.
    assert counts([(5, 0, 0), (1, 0, 0)], [(1, 0, 0), (10, 0, 0)])['tp'] == 2
    assert counts([], [])['f1'] == 0
    assert calibrate(np.zeros((1, 32, 2)), [])['threshold'] == 1.01
    s = np.zeros((1, 32, 2)); s[0, 0, 0] = .9
    assert calibrate(s, [(1, 0, 0)])['threshold'] == .9
    assert window_err([], [(1, 0, 0)], 100) == 1
    assert window_err([(751, 0, 0)], [(750, 0, 0)], 800) == 2 / 30
    assert window_err([(850, 0, 0)], [(849, 0, 0)], 800, 100) == 2 / 30
    print('strip_score: all synthetic checks passed')


if __name__ == '__main__':
    _self_test()
