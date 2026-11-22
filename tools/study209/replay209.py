"""Small numeric replay for common209; never opens images, GT, or native tracks.

The bundle is a path-free JSON projection, not a replacement for source
verification. It replays frozen model scores and slate choices, recomputes W
from the 30 offset-bin arrays, and recomputes event F1 from anonymous vertices
and legal-edge lists. It cannot reproduce LK features or GT association.

Schema common209.replay.v1 (adapter may be updated to the sealed producer):
  models[heldout][G|F|X] = {status, feature_names, mean, scale, beta, intercept}
  clips[] = {domain, clip, nframes, candidates[], arms{K,J,U,G,F,X}}
candidate = {pair_id:str, canonical_ids:[int,int], geometry4:[4], true_valid,
               true_features:[4]|null, sham_valid, sham_features:[4]|null}
  each arm = {selected:[pair_id], bins:{gt,pred:[[window_count_in,window_count_out],...]},
              events:{gt:[[opaque_id,signed_frame],...], pred:[[opaque_id,signed_frame],...],
                      edges05:[[pred_id,gt_id]], edges03:[...]}}

Only opaque per-state event IDs, integer bins, and legal edges are needed;
no individual GT identity, boxes, image pixels, or absolute paths belong here.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import gzip
import json
import math
from pathlib import Path


SCHEMA = 'common209.replay.v1'
ARMS = ('K', 'J', 'U', 'G', 'F', 'X')
HEADS = ('G', 'F', 'X')
DIMENSIONS = {'G': 4, 'F': 5, 'X': 8}


def _finite_vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(name + ' has wrong length')
    out = [float(x) for x in value]
    if not all(math.isfinite(x) for x in out):
        raise ValueError(name + ' must be finite')
    return out


def _pair_key(row):
    ids = row.get('canonical_ids')
    if not isinstance(ids, list) or len(ids) != 2 or any(type(x) is not int for x in ids) or ids[0] >= ids[1]:
        raise ValueError('canonical_ids must be two increasing anonymous integers')
    return tuple(ids)


def _score(model, row, head, sham=False):
    if model.get('status') != 'fitted' or not row['true_valid']:
        return None
    if sham and (head != 'X' or not row['sham_valid']):
        return None
    features = row['sham_features'] if sham else row['true_features']
    if not isinstance(features, list) or len(features) != 4:
        raise ValueError('valid feature branch requires four numeric feature values')
    x = row['geometry4'] + features
    d = DIMENSIONS[head]
    mean = _finite_vector(model['mean'], d, head + '.mean')
    scale = _finite_vector(model['scale'], d, head + '.scale')
    beta = _finite_vector(model['beta'], d, head + '.beta')
    intercept = float(model['intercept'])
    if not math.isfinite(intercept) or any(s <= 0 for s in scale):
        raise ValueError('invalid model intercept or scale')
    return sum(((x[i] - mean[i]) / scale[i]) * beta[i] for i in range(d)) + intercept


def refit_models(bundle):
    """Refit the four source-only logistic folds from compact labeled rows."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    fitted = {}
    for heldout in ('35', '33', 'W', 'G'):
        cohort = []
        for clip in bundle['clips']:
            if clip['domain'] == heldout:
                continue
            for row in clip['candidates']:
                if row['geometry4'][0] >= .20 and row['true_valid'] and row['weak'] in ('S', 'D'):
                    order = row.get('source_order', {}).get(heldout)
                    weight = row.get('source_weight', {}).get(heldout)
                    if type(order) is not int or not isinstance(weight, (int, float)):
                        raise ValueError('source training row lacks sealed order/weight')
                    cohort.append((order, clip['domain'], row, float(weight)))
        cohort.sort(key=lambda x: x[0])
        if [item[0] for item in cohort] != list(range(len(cohort))):
            raise ValueError('source training order is incomplete or duplicated')
        counts = {}
        for _, domain, row, _ in cohort:
            cls = int(row['weak'] == 'S')
            counts[cls, domain] = counts.get((cls, domain), 0) + 1
        active_domains = {cls: {domain for (label, domain), n in counts.items() if label == cls and n}
                          for cls in (0, 1)}
        if cohort and all(active_domains.values()):
            weights = np.asarray([.5 / (len(active_domains[int(row['weak'] == 'S')]) *
                counts[int(row['weak'] == 'S'), domain]) for _, domain, row, _ in cohort], dtype=np.float64)
            supplied_weights = np.asarray([item[3] for item in cohort], dtype=np.float64)
            if not np.allclose(weights, supplied_weights, rtol=0, atol=1e-14):
                raise ValueError('sealed fold class/domain weights differ')
            X = np.asarray([row['geometry4'] + row['true_features'] for _, _, row, _ in cohort], dtype=np.float64)
            y = np.asarray([row['weak'] == 'S' for _, _, row, _ in cohort], dtype=np.float64)
            mean = weights @ X
            scale = np.sqrt(weights @ ((X - mean) ** 2))
            scale[scale < 1e-6] = 1.
            heads = {}
            for head, d in DIMENSIONS.items():
                Z = (X[:, :d] - mean[:d]) / scale[:d]
                def objective(theta):
                    beta, bias = theta[:-1], theta[-1]
                    logits = Z @ beta + bias
                    value = float(weights @ (np.logaddexp(0., logits) - y * logits) + .1 / 2 * (beta @ beta))
                    residual = weights * (expit(logits) - y)
                    grad = np.r_[Z.T @ residual + .1 * beta, residual.sum()]
                    return value, grad
                opt = minimize(objective, np.zeros(d + 1), jac=True, method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12})
                if not opt.success or not np.isfinite(opt.x).all():
                    raise ValueError('source refit failed for ' + heldout + '/' + head)
                trained = bundle['models'][heldout][head]
                for name, actual in (('mean', mean[:d]), ('scale', scale[:d]),
                                     ('beta', opt.x[:d]), ('intercept', opt.x[-1])):
                    if not np.allclose(np.asarray(trained[name]), actual, rtol=2e-6, atol=2e-6):
                        raise ValueError('refit differs from frozen producer model: ' + heldout + '/' + head + '/' + name)
                heads[head] = {'status': 'fitted', 'feature_names': trained['feature_names'],
                    'mean': mean[:d].tolist(), 'scale': scale[:d].tolist(),
                    'beta': opt.x[:d].tolist(), 'intercept': float(opt.x[-1])}
            fitted[heldout] = heads
        else:
            if cohort:
                raise ValueError('incomplete source classes need explicit no-op receipt')
            heads = {}
            for head in HEADS:
                trained = bundle['models'][heldout][head]
                if trained['status'] != 'noop_missing_class':
                    raise ValueError('empty cohort does not match frozen model status')
                heads[head] = {'status': 'noop_missing_class'}
            fitted[heldout] = heads
    return fitted


def replay_selection(clip, models):
    """Recompute geometry/common slates, fixed budget and each deployed choice."""
    rows = clip['candidates']
    by_id = {}
    for row in rows:
        pid = row.get('pair_id')
        if not isinstance(pid, str) or not pid or pid in by_id:
            raise ValueError('pair_id must be a unique opaque string within clip')
        _pair_key(row)
        g = _finite_vector(row['geometry4'], 4, 'geometry4')
        if not 0 <= g[0] <= g[1] <= 1 or g[2] < 0 or not 0 <= g[3] <= 1:
            raise ValueError('invalid geometry4')
        if type(row.get('true_valid')) is not bool or type(row.get('sham_valid')) is not bool:
            raise ValueError('branch validity must be boolean')
        if row['true_valid']:
            _finite_vector(row.get('true_features'), 4, 'true_features')
        if row['sham_valid']:
            _finite_vector(row.get('sham_features'), 4, 'sham_features')
        by_id[pid] = row

    def greedy(eligible):
        eligible = sorted(eligible, key=lambda r: (-r['geometry4'][0], _pair_key(r)))
        used, result = set(), []
        for row in eligible:
            ids = set(_pair_key(row))
            if not used.intersection(ids):
                result.append(row)
                used.update(ids)
        return result

    geom = greedy([r for r in rows if r['geometry4'][0] >= .20])
    slate = greedy([r for r in rows if r['geometry4'][0] >= .20 and r['true_valid']])
    n = len(slate)
    k = (n + 3) // 4 if n >= 4 else 0
    expected = {'K': [], 'J': [r['pair_id'] for r in geom],
                'U': [r['pair_id'] for r in slate[:k]]}
    scored = {}
    for head in HEADS:
        scored[head] = {r['pair_id']: _score(models[head], r, head) for r in rows}
        for r in rows:
            expected_score = r.get('expected_logits', {}).get(head)
            actual_score = scored[head][r['pair_id']]
            if expected_score is None:
                if actual_score is not None:
                    raise ValueError('frozen prediction unexpectedly lacks a score')
            elif actual_score is None or not math.isclose(float(expected_score), actual_score, rel_tol=0, abs_tol=2e-10):
                raise ValueError('refit inference differs from sealed fold logits')
        ranked = sorted(slate, key=lambda r: (
            -(scored[head][r['pair_id']] if scored[head][r['pair_id']] is not None else -math.inf),
            _pair_key(r)))
        if models[head].get('status') == 'fitted':
            expected[head] = [r['pair_id'] for r in ranked[:k]]
        else:
            expected[head] = []
    supplied = clip['arms']
    if set(supplied) != set(ARMS):
        raise ValueError('all six scored arms required')
    for arm in ARMS:
        stated = supplied[arm].get('selected')
        if (not isinstance(stated, list) or len(stated) != len(set(stated))
                or set(stated) != set(expected[arm])):
            raise ValueError('frozen selection differs from replay for arm ' + arm)
    return {'n_slate': n, 'k': k, 'slate': [r['pair_id'] for r in slate],
            'selected': expected, 'logits': scored}


def _maximum_matching(pred_ids, gt_ids, edges):
    pred, gt = set(pred_ids), set(gt_ids)
    adjacency = {p: [] for p in pred}
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError('event edge must have [pred_id, gt_id]')
        p, g = edge
        if p not in pred or g not in gt:
            raise ValueError('edge references an unknown event vertex')
        adjacency[p].append(g)
    owner = {}

    def augment(p, seen):
        for g in adjacency[p]:
            if g in seen:
                continue
            seen.add(g)
            if g not in owner or augment(owner[g], seen):
                owner[g] = p
                return True
        return False

    return sum(augment(p, set()) for p in sorted(pred))


def _event_vertices(values, name):
    if not isinstance(values, list):
        raise ValueError(name + ' must be a list')
    ids = []
    for row in values:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(name + ' vertices require [opaque_id,signed_frame]')
        event_id, frame = row
        if not isinstance(event_id, str) or not event_id or type(frame) is not int:
            raise ValueError(name + ' vertex ID/time has wrong type')
        ids.append(event_id)
    return ids


def score_arm(arm):
    """Recompute window W and official-style event F1 from sealed integers."""
    bins = arm['bins']
    gt_bins, pred_bins = bins['gt'], bins['pred']
    if not isinstance(gt_bins, list) or not isinstance(pred_bins, list):
        raise ValueError('integer bins must be lists')
    if not gt_bins or len(gt_bins) != len(pred_bins):
        raise ValueError('window bin shapes differ')
    # Producer stores the 30 phase histograms concatenated as [B,2]. Their
    # lengths vary with clip duration; each GT event must occur once per phase.
    error, gt_total = 0, 0
    for g, p in zip(gt_bins, pred_bins):
        if len(g) != 2 or len(p) != 2:
            raise ValueError('each flattened bin needs [in,out] counts')
        for x in (*g, *p):
            if type(x) is not int or x < 0:
                raise ValueError('window counts must be nonnegative integers')
        gt_total += sum(g)
        error += abs(p[0] - g[0]) + abs(p[1] - g[1])
    if gt_total % 30:
        raise ValueError('GT event counts must repeat across all 30 offsets')
    ngt_from_bins = gt_total // 30
    if ngt_from_bins == 0:
        raise ValueError('W undefined for zero GT events')

    ev = arm['events']
    gt_ids, pred_ids = _event_vertices(ev['gt'], 'gt'), _event_vertices(ev['pred'], 'pred')
    if len(set(gt_ids)) != len(gt_ids) or len(set(pred_ids)) != len(pred_ids):
        raise ValueError('event vertex IDs must be unique within each side')
    if len(gt_ids) != ngt_from_bins:
        raise ValueError('GT event denominator differs between windows and event vertices')
    tp05 = _maximum_matching(pred_ids, gt_ids, ev['edges05'])
    tp03 = _maximum_matching(pred_ids, gt_ids, ev['edges03'])
    def f1(tp):
        return 2 * tp / (len(pred_ids) + len(gt_ids)) if pred_ids or gt_ids else 0.0
    result = {'W_percent': 100.0 * error / gt_total, 'window_error_sum': error,
            'window_gt_repeated': gt_total, 'gt': len(gt_ids), 'pred': len(pred_ids),
            'tp05': tp05, 'fp05': len(pred_ids) - tp05, 'fn05': len(gt_ids) - tp05,
            'f1_05': f1(tp05), 'tp03': tp03, 'fp03': len(pred_ids) - tp03,
            'fn03': len(gt_ids) - tp03, 'f1_03': f1(tp03)}
    declared = arm.get('declared')
    if declared:
        for metric, actual in (('nG', result['gt']), ('nP', result['pred']),
                               ('tp05', result['tp05']), ('tp03', result['tp03']),
                               ('numerator', result['window_error_sum']),
                               ('denominator', result['window_gt_repeated'])):
            if int(declared[metric]) != actual:
                raise ValueError('replayed count differs from sealed scorer: ' + metric)
        for key, actual in (('W', result['W_percent']), ('f105', result['f1_05']), ('f103', result['f1_03'])):
            if not math.isclose(float(declared[key]), actual, rel_tol=0, abs_tol=1e-10):
                raise ValueError('replayed metric differs from sealed scorer: ' + key)
    return result


def replay(bundle):
    if bundle.get('schema') != SCHEMA:
        raise ValueError('unexpected replay bundle schema')
    refitted = refit_models(bundle)
    results = {}
    grouped = {}
    for clip in bundle['clips']:
        key = (clip['domain'], clip['clip'])
        if key in results:
            raise ValueError('duplicate clip')
        if set(clip['arms']) != set(ARMS):
            raise ValueError('each clip must include all six arms')
        selection = replay_selection(clip, refitted[clip['domain']])
        scored = {a: score_arm(clip['arms'][a]) for a in ARMS}
        results['/'.join(key)] = {'selection': selection, 'scores': scored}
        grouped.setdefault(key[0], []).append(scored)

    domains = {}
    for domain, clips in grouped.items():
        pooled = {}
        for arm in ARMS:
            rows = [c[arm] for c in clips]
            den = sum(r['window_gt_repeated'] for r in rows)
            err = sum(r['window_error_sum'] for r in rows)
            counts = {name: sum(r[name] for r in rows) for name in ('gt', 'pred', 'tp05', 'tp03')}
            pooled[arm] = {'W_percent': 100.0 * err / den if den else None,
                           'F1_05': 2 * counts['tp05'] / (counts['pred'] + counts['gt']) if counts['pred'] + counts['gt'] else 0.0,
                           'F1_03': 2 * counts['tp03'] / (counts['pred'] + counts['gt']) if counts['pred'] + counts['gt'] else 0.0,
                           **counts, 'window_error_sum': err, 'window_denominator': den}
        domains[domain] = pooled

    gates = {}
    for domain in ('35', '33'):
        if domain not in domains:
            continue
        s = domains[domain]
        if domain == '35':
            gates[domain] = {
                'X_improves_K_by_at_least_1pp': s['K']['W_percent'] - s['X']['W_percent'] >= 1.0,
                'X_strictly_beats_G_F_U': all(s['X']['W_percent'] < s[a]['W_percent'] for a in ('G', 'F', 'U')),
                'X_F1_05_nonworse_than_K_by_more_than_.01': s['X']['F1_05'] >= s['K']['F1_05'] - .01,
            }
        else:
            gates[domain] = {
                'X_W_not_worse_than_K_by_more_than_.5pp': s['X']['W_percent'] <= s['K']['W_percent'] + .5,
                'X_F1_05_nonworse_than_K_by_more_than_.01': s['X']['F1_05'] >= s['K']['F1_05'] - .01,
            }
    gates['investment_pass'] = bool(gates.get('35') and gates.get('33') and
        all(gates['35'].values()) and all(gates['33'].values()))
    return {'schema': SCHEMA,
            'scope': 'numeric replay only; no pixel LK extraction or GT rectangle association; event matching recomputed from supplied legal edges',
            'clips': results, 'domains': domains, 'preregistered_gates': gates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.bundle.suffix == '.gz':
        with gzip.open(args.bundle, 'rt', encoding='utf8') as stream:
            bundle = json.load(stream)
    else:
        bundle = json.loads(args.bundle.read_text(encoding='utf8'))
    result = replay(bundle)
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if args.out:
        if args.out.suffix == '.gz':
            with gzip.open(args.out, 'wt', encoding='utf8', compresslevel=9) as stream:
                stream.write(rendered)
        else:
            args.out.write_text(rendered, encoding='utf8')
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
