"""Synthetic-only second head: fixed-quota source pairwise linear identity rank.

No footage, dataset GT, filesystem data, or GPU access. Deployment select()
accepts only ordered native IDs, quotas and deployment feature vectors.
Source swaps and evaluation alone receive the complete frozen legal graph.
This conditional fixed-graph prototype is not an official event scorer.
"""
import json

import numpy as np
from scipy.optimize import minimize

from identity_flow import identity_flow


def _validate(buckets, quotas, features):
    if set(buckets) != set(quotas):
        raise ValueError('bucket/quota keys differ')
    ids = [p for preds in buckets.values() for p in preds]
    if len(set(ids)) != len(ids) or set(ids) != set(features):
        raise ValueError('IDs must be unique with exactly one feature row each')
    for bucket, preds in buckets.items():
        if len(bucket) != 3 or bucket[2] not in (-1, 1):
            raise ValueError('expected (member,second,+1/-1)')
        k = quotas[bucket]
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= len(preds):
            raise ValueError('invalid quota')
    x = np.asarray([features[p] for p in ids], dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('finite nonempty feature rows required')
    return ids, x


def fixed(buckets, quotas):
    """Input list order is the frozen baseline native ID order, never GT order."""
    return {b: list(ids[:quotas[b]]) for b, ids in buckets.items()}


def cardinality(retained, edges):
    """Whole-member maximum matching; local GT IDs are namespaced by member."""
    owner = {}
    for bucket, ids in retained.items():
        for pred in ids:
            owner[pred] = bucket[0]
    matched = {}

    def augment(pred, seen):
        for gt in edges.get(pred, ()):
            node = (owner[pred], gt)
            if node in seen:
                continue
            seen.add(node)
            if node not in matched or augment(matched[node], seen):
                matched[node] = pred
                return True
        return False

    for pred in owner:
        augment(pred, set())
    return len(matched)


def source_pairs(group):
    """Swap one kept a for one rejected b, all other buckets held fixed.

    No tie-assigned individual TP label is ever computed. Only whole-member
    cardinality differences of exactly +/-1 enter the pairwise training set.
    """
    buckets, quotas = group['buckets'], group['quotas']
    _validate(buckets, quotas, group['features'])
    baseline = fixed(buckets, quotas)
    before = cardinality(baseline, group['edges'])
    pairs, audit = [], dict(positive=0, negative=0, zero=0)
    for bucket, ids in buckets.items():
        for a in baseline[bucket]:
            for b in ids[quotas[bucket]:]:
                swapped = {key: list(value) for key, value in baseline.items()}
                swapped[bucket][swapped[bucket].index(a)] = b
                delta = cardinality(swapped, group['edges']) - before
                assert delta in (-1, 0, 1)
                audit[{1: 'positive', -1: 'negative', 0: 'zero'}[delta]] += 1
                if delta:
                    pairs.append((a, b, delta))
    return pairs, audit


def fit(groups, l2=1.0):
    """Fixed L2 logistic objective, source-only normalization, zero intercept.

    Every source video group has equal normalization and loss weight. Each
    retained pair is explicitly accompanied by its reversed +/- sample.
    l2=1 is fixed before synthetic evaluation; no target tuning is performed.
    """
    if len(groups) < 2 or len({g['video'] for g in groups}) != len(groups):
        raise ValueError('at least two distinct source video groups required')
    if not np.isfinite(l2) or l2 <= 0:
        raise ValueError('positive L2 required')
    rows = [_validate(g['buckets'], g['quotas'], g['features'])[1] for g in groups]
    mean = np.mean([x.mean(axis=0) for x in rows], axis=0)
    scale = np.sqrt(np.mean([((x - mean)**2).mean(axis=0) for x in rows], axis=0))
    scale[scale < 1e-12] = 1
    X, Y, weights, audits = [], [], [], []
    for group in groups:
        pairs, audit = source_pairs(group)
        audits.append(dict(video=group['video'], **audit))
        for a, b, delta in pairs:
            difference = (np.asarray(group['features'][b]) - group['features'][a]) / scale
            X.extend([difference, -difference])
            Y.extend([delta, -delta])
            weights.extend([1 / (len(groups) * 2 * len(pairs))] * 2)
    if not X:
        return dict(mean=mean, scale=scale, coef=np.zeros_like(mean), audit=audits,
                    source_videos=[g['video'] for g in groups], pairs=0)
    X, Y, weights = np.asarray(X), np.asarray(Y), np.asarray(weights)

    def objective(coef):
        margin = Y * (X @ coef)
        value = weights @ np.logaddexp(0, -margin) + l2 * (coef @ coef) / 2
        probability = np.exp(-np.logaddexp(0, margin))
        gradient = X.T @ (-weights * Y * probability) + l2 * coef
        return value, gradient

    result = minimize(objective, np.zeros_like(mean), jac=True, method='L-BFGS-B')
    if not result.success:
        raise RuntimeError(result.message)
    return dict(mean=mean, scale=scale, coef=result.x, audit=audits,
                source_videos=[g['video'] for g in groups], pairs=len(X)//2)


def select(model, buckets, quotas, features):
    """Deployment: no GT, graph, member outcome or labels can be passed here.

    Equal cutoff scores abstain for the entire bucket, retaining baseline IDs.
    Native input order deterministically breaks score ties away from cutoff.
    """
    ids, x = _validate(buckets, quotas, features)
    scores = ((x - model['mean']) / model['scale']) @ model['coef']
    score = dict(zip(ids, scores))
    retained, abstained = {}, []
    for bucket, candidates in buckets.items():
        k = quotas[bucket]
        ranked = sorted(candidates, key=lambda p: -score[p])
        if 0 < k < len(ranked) and abs(score[ranked[k-1]] - score[ranked[k]]) <= 1e-12:
            retained[bucket] = list(candidates[:k])
            abstained.append(bucket)
        else:
            retained[bucket] = ranked[:k]
    return retained, abstained


def _synthetic(seed, video):
    rng = np.random.default_rng(seed)
    buckets, quotas, features, edges, truth, kinds = {}, {}, {}, {}, {video: []}, {}
    # Signal occurs twice; competing-bee control has the same feature profiles
    # with reversed preference. A linear head cannot solve both from these X.
    for j, kind in enumerate(['signal', 'signal', 'indistinguishable', 'src-tie', 'competition'] * 6):
        sec, direction = 3*j, 1 if j % 2 == 0 else -1
        bucket = (video, sec, direction)
        a, b = f'{video}:{j}:a', f'{video}:{j}:b'
        g0, g1 = f'g{j}:0', f'g{j}:1'
        low, high = [0.0, float(rng.normal())], [1.0, float(rng.normal())]
        if kind in ('indistinguishable', 'src-tie'):
            low = high = [0.5, 0.0]
        features[a], features[b] = low, high
        buckets[bucket], quotas[bucket], kinds[bucket] = [a,b], 1, kind
        edges[a], edges[b] = [], [g1]
        truth[video].append((g1, sec, direction))
        if kind == 'src-tie':
            edges[a] = [g1]
        if kind == 'competition':
            # Baseline a is useful; high-score b only competes for an occupied
            # GT. A tie-assigned TP(b) label could incorrectly favor this swap.
            guard = f'{video}:{j}:guard'
            edges[a], edges[b], edges[guard] = [g1], [g0], [g0]
            guard_bucket = (video, sec+1, direction)
            buckets[guard_bucket], quotas[guard_bucket] = [guard], 1
            features[guard] = [0.5, 0.0]
            kinds[guard_bucket] = 'guard'
            truth[video].append((g0, sec+1, direction))
    return dict(video=video,buckets=buckets,quotas=quotas,features=features,edges=edges,
                gt_events={m:[e[0] for e in events] for m,events in truth.items()}, truth=truth,kinds=kinds)


def _window_error(group, retained):
    errors = []
    for phase in range(30):
        p, g = {}, {}
        for (member, sec, direction), ids in retained.items():
            key = member, (sec+phase)//30, direction
            p[key] = p.get(key,0) + len(ids)
        for member, events in group['truth'].items():
            for _, sec, direction in events:
                key = member, (sec+phase)//30, direction
                g[key] = g.get(key,0) + 1
        errors.append(sum(abs(p.get(k,0)-g.get(k,0)) for k in p.keys() | g.keys()))
    return float(np.mean(errors))


def evaluate(group, model):
    baseline = fixed(group['buckets'],group['quotas'])
    ranked, abstained = select(model,group['buckets'],group['quotas'],group['features'])
    ceiling = identity_flow(group['buckets'],group['quotas'],group['edges'],group['gt_events'])
    out = dict(video=group['video'],abstained_buckets=len(abstained),methods={})
    for name, retained in [('fixed_ID',baseline),('source_rank',ranked),('identity_flow',ceiling['retained'])]:
        assert all(len(retained[b]) == group['quotas'][b] for b in retained)
        tp = cardinality(retained,group['edges'])
        p,g = sum(map(len,retained.values())),sum(map(len,group['gt_events'].values()))
        err = _window_error(group,retained)
        out['methods'][name] = dict(tp=tp,pred=p,gt=g,f1=2*tp/(p+g),err=err,W=100*err/g)
    metrics = list(out['methods'].values())
    assert len({(r['pred'],r['gt'],r['W']) for r in metrics})==1
    assert metrics[-1]['tp'] == ceiling['maximum_tp']
    out['kind_changed_buckets'] = {kind:sum(ranked[b]!=baseline[b] for b in baseline if group['kinds'][b]==kind)
                                   for kind in set(group['kinds'].values())}
    return out


def selfcheck():
    sources = [_synthetic(11,'source_video_A'),_synthetic(22,'source_video_B')]
    model = fit(sources)
    # Any arbitrary maximum-matching assignment in a two-pred/one-GT tie must
    # not become a preference: swapping either identity has delta zero.
    tie = dict(video='tie',buckets={('tie',0,1):['a','b']},quotas={('tie',0,1):1},
               features={'a':[0.,0.],'b':[1.,0.]},edges={'a':['x','y'],'b':['y','x']})
    assert source_pairs(tie)[0] == []
    reversed_tie = dict(tie,edges={'a':['y','x'],'b':['x','y']})
    assert source_pairs(reversed_tie)[0] == []
    for group in sources:
        pairs,_ = source_pairs(group)
        kind_by_pred = {p:group['kinds'][bucket] for bucket,ids in group['buckets'].items() for p in ids}
        for a,b,delta in pairs:
            if kind_by_pred[a] == 'competition':
                assert delta == -1  # competition with other retained bee
            assert kind_by_pred[a] != 'src-tie'
    # No normalization or fitting accesses the independently seeded test group.
    unseen = _synthetic(991,'unseen_video_seed991')
    result = evaluate(unseen,model)
    assert result['kind_changed_buckets']['indistinguishable']==0
    assert result['kind_changed_buckets']['src-tie']==0
    assert result['methods']['source_rank']['tp'] < result['methods']['identity_flow']['tp']
    # Pair reversal is antisymmetric because scores are linear with intercept 0.
    a,b = np.array([0.,1.]),np.array([1.,0.])
    difference = (b-a)/model['scale']
    assert np.isclose(difference@model['coef'], -((-difference)@model['coef']))
    return dict(scope='synthetic fixed frozen graph only; no evidence for field benefit',
                feature_names=['synthetic_geometry_support_proxy','independent_nuisance'],
                l2=1,intercept=0,coef=model['coef'].tolist(),source_audits=model['audit'],
                sources=[evaluate(g,model) for g in sources],unseen=result,
                tie_cardinality_tests='passed; no tie-assigned TP labels',
                limitation='indistinguishable identities abstain; competition has reversed feature preference and remains partly unlearnable')


if __name__ == '__main__':
    print(json.dumps(selfcheck(),indent=2,sort_keys=True))
