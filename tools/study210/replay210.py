"""Path-free numeric replay for §210 interleaved-run association.

The pack schema is intentionally small: four source folds, anonymous eligible
pair rows, a sparse active-slate delta ledger, and six complete anonymous event
graphs/bin summaries per clip. Source rows contain only G6/X2/Xminus2 and weak
S/D/U labels. No frames, boxes, raw MOT, images, or original GT IDs are stored.

This module independently refits the G and frozen-G-offset correction models,
recomputes G/X/M scores, N/G/X/M choices, equal-k slate closure, weak grouped
AP, event matching/W via the shared §209 numeric core, and active-slate exact
capacity via capacity210.py.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import gzip
import importlib.util
import json
import math
import sys
from pathlib import Path


SCHEMA = 'interleave210.numeric.v1'
DOMAINS = ('35', '33', 'W', 'G')
ARMS = ('K', 'N', 'G', 'X', 'M', 'ALL')
HEADS = ('G', 'X', 'M')
G_NAMES = ('min_offset_norm', 'median_offset_norm', 'log1p_brackets',
           'orientation_balance', 'full_support', 'suffix_zero_rms')
X_NAMES = ('prefix_offset_norm', 'suffix_zero_minus_offset_rms')
ALL_DISJOINT_REGISTRATION = '4685f0b9702978c3b083c7a1e9fa698d54bd28ae'


def _core_modules(repo_root):
    repo_root = Path(repo_root).resolve()
    if not (repo_root / 'beetrack' / 'p0.py').is_file():
        raise ValueError('--repo must point to the final_project root')
    sys.path.insert(0, str(repo_root / 'tools' / 'study209'))
    from replay209 import score_arm
    from replay_release209 import bins
    sibling = Path(__file__).with_name('capacity210.py')
    spec = importlib.util.spec_from_file_location('interleave210_capacity_core', sibling)
    capacity = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(capacity)
    return score_arm, bins, capacity


def _fit_head(X, y, weights, offset, names):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    if weights is None:
        return {'status': 'zero_missing_class', 'feature_names': list(names), 'mean': None,
                'scale': None, 'beta': [0.] * len(names), 'intercept': 0.}
    mean = weights @ X
    scale = np.sqrt(weights @ ((X - mean) ** 2))
    scale[scale < 1e-6] = 1.
    Z = (X - mean) / scale
    offset = np.asarray(offset, dtype=np.float64)
    def objective(theta):
        beta, bias = theta[:-1], theta[-1]
        z = offset + Z @ beta + bias
        residual = weights * (expit(z) - y)
        value = float(weights @ (np.logaddexp(0., z) - y * z) + .1 / 2 * (beta @ beta))
        gradient = np.r_[Z.T @ residual + .1 * beta, residual.sum()]
        return value, gradient
    result = minimize(objective, np.zeros(X.shape[1] + 1), jac=True, method='L-BFGS-B',
                      options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('source logistic fit failed')
    return {'status': 'fitted', 'feature_names': list(names), 'mean': mean.tolist(),
            'scale': scale.tolist(), 'beta': result.x[:-1].tolist(),
            'intercept': float(result.x[-1]), 'objective': float(result.fun),
            'gradient': objective(result.x)[1].tolist()}


def _weights(rows, labels):
    import numpy as np
    from collections import Counter
    counts = Counter((int(y), r['domain']) for r, y in zip(rows, labels))
    hives = {y: {h for (c, h), n in counts.items() if c == y and n} for y in (0, 1)}
    if not all(hives.values()):
        return None
    return np.asarray([.5 / (len(hives[int(y)]) * counts[int(y), r['domain']])
                       for r, y in zip(rows, labels)], dtype=np.float64)


def _logit(head, features):
    import numpy as np
    if head['status'] != 'fitted':
        return 0.
    x = np.asarray(features, dtype=np.float64)
    return float(((x - head['mean']) / head['scale']) @ head['beta'] + head['intercept'])


def refit(bundle):
    """Rebuild G6 and supported X2 offset-correction models for all heldouts."""
    import numpy as np
    folds = {}
    for heldout in DOMAINS:
        g_rows = []
        for clip in bundle['clips']:
            if clip['domain'] == heldout:
                continue
            for row in clip['candidates']:
                if row['eligible'] and row['weak'] in ('S', 'D'):
                    order = row['source_order_G'][heldout]
                    weight = row['source_weight_G'][heldout]
                    if type(order) is not int or (weight is not None and not isinstance(weight, (int, float))):
                        raise ValueError('G source order/weight malformed')
                    g_rows.append((order, clip['domain'], row, float(weight)))
        g_rows.sort(key=lambda t: t[0])
        if [v[0] for v in g_rows] != list(range(len(g_rows))):
            raise ValueError('G source order is not complete and unique')
        rows = [v[2] for v in g_rows]
        g_weights = [v[3] for v in g_rows]
        y = np.asarray([r['weak'] == 'S' for r in rows], dtype=np.float64)
        w = _weights(rows, y)
        if w is None and any(v is not None for v in g_weights):
            raise ValueError('saved G fold contains only one label class')
        if w is not None and not np.allclose(w, g_weights, rtol=0, atol=1e-14):
            raise ValueError('G class/domain weights differ from frozen fit')
        G = np.asarray([r['G6'] for r in rows], dtype=np.float64).reshape(-1, 6)
        Ghead = _fit_head(G, y, w, np.zeros(len(rows)), G_NAMES)

        c_rows = []
        for _, domain, row, _ in g_rows:
            if row['full_support']:
                order = row['source_order_correction'][heldout]
                weight = row['source_weight_correction'][heldout]
                if type(order) is not int or (weight is not None and not isinstance(weight, (int, float))):
                    raise ValueError('correction source order/weight malformed')
                c_rows.append((order, domain, row, None if weight is None else float(weight)))
        c_rows.sort(key=lambda t: t[0])
        if [v[0] for v in c_rows] != list(range(len(c_rows))):
            raise ValueError('correction source order is not complete and unique')
        cdata = [v[2] for v in c_rows]
        cy = np.asarray([r['weak'] == 'S' for r in cdata], dtype=np.float64)
        cw = _weights(cdata, cy)
        saved_cw = [v[3] for v in c_rows]
        if cw is None and any(v is not None for v in saved_cw):
            raise ValueError('saved correction fold contains only one label class')
        if cw is not None and not np.allclose(cw, np.asarray(saved_cw, dtype=np.float64), rtol=0, atol=1e-14):
            raise ValueError('correction class/domain weights differ from frozen fit')
        offset = np.asarray([_logit(Ghead, r['G6']) for r in cdata], dtype=np.float64)
        X = np.asarray([r['X2'] for r in cdata], dtype=np.float64).reshape(-1, 2)
        correction = _fit_head(X, cy, cw, offset, X_NAMES)

        frozen = bundle['models'][heldout]
        _compare_head(Ghead, frozen['G'], heldout + '/G')
        _compare_head(correction, frozen['correction'], heldout + '/correction')
        folds[heldout] = {'G': Ghead, 'correction': correction,
                          'source_rows_G': len(g_rows), 'source_rows_correction': len(c_rows)}
    return folds


def _compare_head(actual, saved, name):
    import numpy as np
    if actual['status'] != saved['status']:
        raise ValueError('frozen model status differs: ' + name)
    if actual['status'] != 'fitted':
        return
    for field in ('mean', 'scale', 'beta', 'intercept'):
        if not np.allclose(np.asarray(actual[field]), np.asarray(saved[field]), rtol=2e-6, atol=2e-6):
            raise ValueError('frozen coefficients differ: ' + name + '/' + field)


def _scores(model, row):
    base = _logit(model['G'], row['G6'])
    if not row['full_support'] or model['correction']['status'] != 'fitted':
        return {'G': base, 'X': base, 'M': base}
    return {'G': base,
            'X': base + _logit(model['correction'], row['X2']),
            'M': base + _logit(model['correction'], row['Xminus2'])}


def _plan_clip(clip, model):
    rows = clip['candidates']
    by_id = {}
    for row in rows:
        if row['pair_id'] in by_id or not isinstance(row['pair_id'], str):
            raise ValueError('duplicate or malformed candidate pair id')
        ids = row['canonical_ids']
        if (len(ids) != 2 or any(type(x) is not int for x in ids) or ids[0] >= ids[1]
                or row['pair_id'] != f"{ids[0]}|{ids[1]}"):
            raise ValueError('anonymous numeric endpoint pair is not canonical')
        if row['weak'] not in ('S', 'D', 'U') or type(row['active']) is not bool or type(row['eligible']) is not bool:
            raise ValueError('invalid weak label/eligibility flag')
        if type(row['full_support']) is not bool or type(row['co_visible']) is not bool:
            raise ValueError('invalid support or co-visible flag')
        if not row['eligible']:
            raise ValueError('numeric universe includes a noneligible pair')
        if not math.isfinite(float(row['min_offset_norm'])) or not 0 <= float(row['min_offset_norm']) <= 1:
            raise ValueError('local offset norm must be finite in [0,1]')
        for feature, size in (('G6', 6), ('X2', 2), ('Xminus2', 2)):
            values = row.get(feature)
            if not isinstance(values, list) or len(values) != size or not all(math.isfinite(float(v)) for v in values):
                raise ValueError('finite numeric feature vector required: ' + feature)
        by_id[row['pair_id']] = row
    ordered = sorted(rows, key=lambda r: (not r['active'], float(r['min_offset_norm']),
                                           r['canonical_ids'][0], r['canonical_ids'][1]))
    used, slate = set(), []
    for row in ordered:
        ids = set(row['canonical_ids'])
        if not used & ids:
            slate.append(row); used.update(ids)
    active = [r for r in slate if r['active']]
    a = len(active); k = (a + 3) // 4 if a else 0
    logits = {r['pair_id']: _scores(model, r) for r in rows}
    for row in rows:
        saved = row.get('expected_logits', {})
        for head, score in logits[row['pair_id']].items():
            expected = saved.get(head)
            if expected is None or not math.isclose(float(expected), score, rel_tol=0, abs_tol=2e-10):
                raise ValueError('target model score differs from sealed prediction')
    selected = {'K': [], 'N': [r['pair_id'] for r in active[:k]],
                'ALL': [r['pair_id'] for r in active]}
    selected['G'] = [r['pair_id'] for r in sorted(active,
        key=lambda r: (-logits[r['pair_id']]['G'], tuple(r['canonical_ids'])))[:k]]
    for head in ('X', 'M'):
        selected[head] = [r['pair_id'] for r in sorted(active,
            key=lambda r: (-logits[r['pair_id']][head], tuple(r['canonical_ids'])))[:k]]
    actual = clip['states']
    if set(actual) != set(ARMS):
        raise ValueError('complete K/N/G/X/M/ALL state ledger required')
    for arm in ARMS:
        stated = actual[arm]['selected']
        if len(stated) != len(set(stated)) or set(stated) != set(selected[arm]):
            raise ValueError('selected slate differs from replay: ' + arm)
    return {'slate': [r['pair_id'] for r in slate], 'active_slate': [r['pair_id'] for r in active],
            'a': a, 'k': k, 'selected': selected, 'logits': logits,
            'active_rows': active, 'all_rows': by_id}


def _average_precision(rows, model, head, min_each=5):
    # Equal-score groups enter together; do not use candidate ID to break ties.
    use = [r for r in rows if r['full_support'] and r['weak'] in ('S', 'D')]
    pos = sum(r['weak'] == 'S' for r in use)
    neg = sum(r['weak'] == 'D' for r in use)
    if min(pos, neg) < min_each:
        return {'AP': None, 'positive': pos, 'negative': neg, 'reason': 'requires at least 5 S and 5 D'}
    groups = {}
    for row in use:
        groups.setdefault(float(_scores(model, row)[head]), []).append(row['weak'] == 'S')
    tp = n = 0; ap = 0.
    for score in sorted(groups, reverse=True):
        new = sum(groups[score]); tp += new; n += len(groups[score])
        ap += (new / pos) * (tp / n)
    return {'AP': ap, 'positive': pos, 'negative': neg,
            'tie_rule': 'whole equal-score group added before precision at group end'}


def _all_disjoint_capacity(clip, plan, capacity, fixed_capacity):
    """Exact 0..a any-budget count family, separate from fixed-k capacity."""
    import numpy as np
    receipt = clip.get('all_disjoint_registration')
    if not isinstance(receipt, dict) or receipt.get('sha1') != ALL_DISJOINT_REGISTRATION:
        raise ValueError('registered all-disjoint addendum receipt missing')
    rows = clip['candidates']
    active = [r for r in rows if r['active']]
    actions = clip['all_disjoint_actions']
    active_ids = {r['pair_id'] for r in active}
    action_ids = {a['pair_id'] for a in actions}
    if len(action_ids) != len(actions) or action_ids != active_ids:
        raise ValueError('all eligible active edge deltas are not complete')
    slate_active = set(plan['active_slate'])
    if active_ids != slate_active:
        raise ValueError('not every eligible active edge enters the disjoint slate')
    used = set()
    for row in active:
        ids = set(row['canonical_ids'])
        if used & ids:
            raise ValueError('active graph edges are not mutually disjoint')
        used.update(ids)
    if len(active) > 3:
        raise ValueError('registered all-disjoint enumeration ceiling 2^3 exceeded')

    table = {a['pair_id']: a for a in actions}
    ids = sorted(table, key=lambda key: tuple(table[key]['canonical_ids']))
    baseline = np.asarray(clip['states']['K']['bins']['pred'], dtype=np.int64)
    truth = np.asarray(clip['states']['K']['bins']['gt'], dtype=np.int64)
    den = int(truth.sum())
    if den % 30:
        raise ValueError('all-disjoint GT count must repeat over 30 phases')
    bins_by_pair = {key: np.asarray(table[key]['delta_windows'], dtype=np.int64) for key in ids}
    if any(v.shape != baseline.shape for v in bins_by_pair.values()):
        raise ValueError('all-disjoint pair delta shape differs from native bin ledger')

    by_budget = {}
    all_vector_receipts = {}
    for budget in range(len(ids) + 1):
        witnesses = []
        unique_vectors = {}
        from itertools import combinations
        for chosen in combinations(ids, budget):
            predicted = baseline.copy()
            for key in chosen:
                predicted += bins_by_pair[key]
            if (predicted < 0).any():
                raise ValueError('all-disjoint sum produced negative event counts')
            numerator = int(np.abs(predicted - truth).sum())
            digest = capacity.digest(predicted)
            unique_vectors[digest] = unique_vectors.get(digest, 0) + 1
            whole = all_vector_receipts.setdefault(digest, {
                'active_subset_count': 0, 'numerator': numerator, 'by_active_budget': {}})
            if whole['numerator'] != numerator:
                raise ValueError('identical count-vector digest has inconsistent numerator')
            whole['active_subset_count'] += 1
            whole['by_active_budget'][str(budget)] = whole['by_active_budget'].get(str(budget), 0) + 1
            witnesses.append({'selected': list(chosen), 'numerator': numerator,
                              'vector_sha256': digest})
        def tie_order(w):
            return tuple(capacity.pair_key(key) for key in w['selected'])
        minimum = min(witnesses, key=lambda w: (w['numerator'], tie_order(w)))
        maximum = min(witnesses, key=lambda w: (-w['numerator'], tie_order(w)))
        by_budget[str(budget)] = {
            'subset_count': len(witnesses), 'unique_window_vectors': len(unique_vectors),
            'vector_multiplicity_histogram': {str(m): sum(v == m for v in unique_vectors.values())
                                              for m in sorted(set(unique_vectors.values()))},
            'min_numerator': minimum['numerator'], 'min_witness': minimum,
            'max_numerator': maximum['numerator'], 'max_witness': maximum,
            'W_min_percent': 100. * minimum['numerator'] / den if den else None,
            'W_max_percent': 100. * maximum['numerator'] / den if den else None,
        }
    # The endpoints of the family must equal K and the physical ALL state.
    all_delta = sum((bins_by_pair[k] for k in ids), np.zeros_like(baseline))
    all_pred = baseline + all_delta
    all_state = clip['states']['ALL']
    if (all_pred.tolist() != all_state['bins']['pred']
            or by_budget[str(len(ids))]['min_numerator'] != int(all_state['declared']['numerator'])
            or by_budget['0']['min_numerator'] != int(clip['states']['K']['declared']['numerator'])):
        raise ValueError('all-disjoint family endpoints do not close to K and ALL')
    if fixed_capacity['mode'] == 'exact':
        k = str(plan['k'])
        if (fixed_capacity['min_loss_interval'] != [by_budget[k]['min_numerator']] * 2
                or fixed_capacity['max_loss_interval'] != [by_budget[k]['max_numerator']] * 2):
            raise ValueError('fixed-k exact bounds disagree with all-disjoint budget slice')
    declared = clip.get('declared_all_disjoint')
    if not isinstance(declared, dict) or declared.get('global_scope_proved') is not True:
        raise ValueError('independent all-disjoint global-scope proof is required')
    expected_budget = declared['by_active_budget']
    actual_budget = [{
        'active_budget': int(b), 'active_subset_count': value['subset_count'],
        'unique_window_vectors': value['unique_window_vectors'],
        'min_numerator': value['min_numerator'], 'max_numerator': value['max_numerator'],
    } for b, value in sorted(by_budget.items(), key=lambda t: int(t[0]))]
    if expected_budget != actual_budget or declared['enumerated_active_subsets'] != 2 ** len(ids):
        raise ValueError('complete all-disjoint subset support differs from independent receipt')
    projected_vectors = [{
        'vector_sha256': key,
        'active_subset_count': value['active_subset_count'],
        'numerator': value['numerator'],
        'by_active_budget': value['by_active_budget'],
    } for key, value in sorted(all_vector_receipts.items())]
    if projected_vectors != declared['vectors']:
        raise ValueError('deduplicated all-disjoint window vectors differ from independent receipt')
    if (declared['active_support_pair_count'] != len(ids)
            or declared['denominator'] != den
            or declared['baseline_numerator'] != clip['states']['K']['declared']['numerator']):
        raise ValueError('all-disjoint active support or K denominator differs')
    return {
        'preconditions_passed': True,
        'eligible_active_count': len(ids),
        'active_ids_all_enter_disjoint_slate': True,
        'all_active_endpoints_mutually_disjoint': True,
        'families_enumerated': 2 ** len(ids),
        'by_budget': by_budget,
        'scope': 'All one-time disjoint active pair subsets only; inactive zero-delta edges are omitted. This is a W/count family, not F1, identity accuracy, or a deployment policy.',
    }


def _load_core(repo_root):
    return _core_modules(repo_root)


def replay(bundle, repo_root):
    import numpy as np
    if bundle.get('schema') != SCHEMA:
        raise ValueError('unexpected interleave210 numeric bundle schema')
    score_arm, bins, capacity = _load_core(repo_root)
    folds = refit(bundle)
    clip_results = {}
    for clip in bundle['clips']:
        domain, name = clip['domain'], clip['clip']
        if domain not in DOMAINS:
            raise ValueError('unknown target domain')
        plan = _plan_clip(clip, folds[domain])
        if type(clip.get('nframes')) is not int or clip['nframes'] < 1:
            raise ValueError('positive clip frame denominator required for phase replay')
        for arm in ARMS:
            for side in ('gt', 'pred'):
                if bins(clip['states'][arm]['events'][side], clip['nframes']) != clip['states'][arm]['bins'][side]:
                    raise ValueError('event signed-time to 30-phase histogram differs: ' + arm + '/' + side)
        scored = {arm: score_arm(clip['states'][arm]) for arm in ARMS}
        for arm, score in scored.items():
            declared = clip['states'][arm]['declared']
            if (score['gt'] != declared['nG'] or score['pred'] != declared['nP']
                    or score['tp05'] != declared['tp05'] or score['tp03'] != declared['tp03']
                    or not math.isclose(score['W_percent'], declared['W'], rel_tol=0, abs_tol=1e-10)):
                raise ValueError('official native score differs from event/bin replay: ' + arm)

        # Capacity is limited to the preregistered fixed active-slate subset.
        actions = clip['capacity']['actions']
        capacity_result = capacity.diagnose_clip(
            clip['states']['K']['bins']['pred'], clip['states']['K']['bins']['gt'],
            actions,
            {a: {'selected': plan['selected'][a], 'pred_bins': clip['states'][a]['bins']['pred'],
                 'numerator': int(clip['states'][a]['declared']['numerator'])} for a in ('N', 'G', 'X', 'M')},
            clip['capacity']['population'])
        fixed_declared = clip['capacity']['declared']
        for key in ('mode', 'a', 'k', 'combinations', 'enumerated', 'denominator',
                    'baseline_numerator', 'min_loss_interval', 'max_loss_interval'):
            if capacity_result[key] != fixed_declared[key]:
                raise ValueError('fixed-k capacity replay differs from sealed receipt: ' + key)
        if any(capacity_result['actual'][arm]['numerator'] != fixed_declared['actual_numerators'][arm]
               for arm in ('N', 'G', 'X', 'M')):
            raise ValueError('fixed-k actual arm numerator differs from sealed capacity receipt')
        all_disjoint_result = _all_disjoint_capacity(clip, plan, capacity, capacity_result)
        # The observed slate-loss delta must close exactly against all six MOTs.
        for arm in ARMS:
            if score_arm(clip['states'][arm])['window_error_sum'] != int(clip['states'][arm]['declared']['numerator']):
                raise ValueError('window numerator mismatch')
        clip_results[domain + '/' + name] = {
            'selection': {k: plan[k] for k in ('slate', 'active_slate', 'a', 'k', 'selected')},
            'scores': scored, 'capacity': capacity_result,
            'all_disjoint': all_disjoint_result,
        }

    pooled_capacity = capacity.pool([dict(domain=clip['domain'], capacity=clip_results[
        clip['domain'] + '/' + clip['clip']]['capacity']) for clip in bundle['clips']])
    all_disjoint_pooled = {}
    for domain in DOMAINS:
        clips = [clip_results[domain + '/' + c['clip']] for c in bundle['clips'] if c['domain'] == domain]
        denom = sum(c['scores']['K']['window_gt_repeated'] for c in clips)
        low_parts = [min(int(v['min_numerator']) for v in c['all_disjoint']['by_budget'].values()) for c in clips]
        high_parts = [max(int(v['max_numerator']) for v in c['all_disjoint']['by_budget'].values()) for c in clips]
        low, high = sum(low_parts), sum(high_parts)
        all_disjoint_pooled[domain] = {
            'scope': 'sum of per-clip extrema over independent one-time active-edge subsets; budget may vary by clip',
            'denominator': denom, 'min_numerator': low, 'max_numerator': high,
            'W_min_percent': 100. * low / denom if denom else None,
            'W_max_percent': 100. * high / denom if denom else None,
        }
    domains = {}
    for domain in DOMAINS:
        clips = [clip_results[domain + '/' + c['clip']] for c in bundle['clips'] if c['domain'] == domain]
        if not clips:
            raise ValueError('missing fixed target clip')
        pooled = {}
        for arm in ARMS:
            vals = [c['scores'][arm] for c in clips]
            den = sum(v['window_gt_repeated'] for v in vals)
            num = sum(v['window_error_sum'] for v in vals)
            gt = sum(v['gt'] for v in vals); pred = sum(v['pred'] for v in vals)
            tp05 = sum(v['tp05'] for v in vals); tp03 = sum(v['tp03'] for v in vals)
            pooled[arm] = {'W_percent': 100 * num / den if den else None,
                           'F1_05': 2 * tp05 / (gt + pred) if gt + pred else 0.,
                           'F1_03': 2 * tp03 / (gt + pred) if gt + pred else 0.,
                           'nG': gt, 'nP': pred, 'tp05': tp05, 'tp03': tp03,
                           'numerator': num, 'denominator': den}
        rows = [r for c in bundle['clips'] if c['domain'] == domain for r in c['candidates']]
        pooled['grouped_AP'] = {head: _average_precision(rows, folds[domain], head) for head in ('G', 'X', 'M')}
        domains[domain] = pooled

    gates = {}
    for domain in ('35', '33'):
        s = domains[domain]
        if domain == '35':
            gates[domain] = {
                'X_beats_K_by_at_least_1pp': s['K']['W_percent'] - s['X']['W_percent'] >= 1.,
                'X_strictly_beats_N_and_G': all(s['X']['W_percent'] < s[a]['W_percent'] for a in ('N', 'G')),
                'F1_05_loss_at_most_.01': s['X']['F1_05'] >= s['K']['F1_05'] - .01,
            }
        else:
            gates[domain] = {
                'X_loss_vs_K_at_most_.5pp': s['X']['W_percent'] <= s['K']['W_percent'] + .5,
                'F1_05_loss_at_most_.01': s['X']['F1_05'] >= s['K']['F1_05'] - .01,
            }
        ap = s['grouped_AP']
        enough = min(ap[h]['positive'] for h in ('G', 'X', 'M')) >= 5 and min(ap[h]['negative'] for h in ('G', 'X', 'M')) >= 5
        gates[domain]['attribution_sample_sufficient'] = enough
        gates[domain]['attribution_status'] = 'insufficient' if not enough else 'passed' if (
            ap['X']['AP'] is not None and ap['G']['AP'] is not None and ap['M']['AP'] is not None
            and ap['X']['AP'] > ap['G']['AP'] and ap['X']['AP'] > ap['M']['AP']) else 'failed'
        gates[domain]['X_AP_strictly_beats_G_and_M'] = None if not enough else (
            ap['X']['AP'] is not None and ap['G']['AP'] is not None and ap['M']['AP'] is not None
            and ap['X']['AP'] > ap['G']['AP'] and ap['X']['AP'] > ap['M']['AP'])
    gates['investment_pass'] = all(gates[d][k] for d in ('35', '33') for k in
        (('X_beats_K_by_at_least_1pp', 'X_strictly_beats_N_and_G', 'F1_05_loss_at_most_.01') if d == '35'
         else ('X_loss_vs_K_at_most_.5pp', 'F1_05_loss_at_most_.01')))
    attribution_checks = [gates[d]['X_AP_strictly_beats_G_and_M'] for d in ('35', '33')]
    gates['attribution_pass'] = None if any(v is None for v in attribution_checks) else all(attribution_checks)
    return {'schema': SCHEMA, 'scope': 'numeric replay; no image LK or GT box reassociation',
            'clips': clip_results, 'domains': domains, 'pooled_capacity': pooled_capacity,
            'all_disjoint_pooled': all_disjoint_pooled,
            'preregistered_gates': gates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repo', type=Path)
    args = parser.parse_args()
    if args.repo:
        repo = args.repo.resolve()
    else:
        candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
        repo = next((p.resolve() for p in candidates if (p / 'beetrack' / 'p0.py').is_file()), None)
        if repo is None:
            parser.error('run from final_project or pass --repo <final_project>')
    if args.bundle.suffix == '.gz':
        with gzip.open(args.bundle, 'rt', encoding='utf8') as stream:
            bundle = json.load(stream)
    else:
        bundle = json.loads(args.bundle.read_text(encoding='utf8'))
    result = replay(bundle, repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == '.gz':
        with gzip.open(args.out, 'wt', encoding='utf8', compresslevel=9) as stream:
            json.dump(result, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
    else:
        args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
