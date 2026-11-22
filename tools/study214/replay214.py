"""Portable numeric replay core for §214 (source-only implementation).

This module intentionally handles only anonymized numerical inputs.  It does
not read images, source tracks, polygons, or raw GT identities.  Those remain
the independent producer audit scope.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

DOMAINS = ('35', '33', 'W', 'G')
ARMS = ('K', 'G', 'I', 'Z', 'O')
LIMIT = .25
RIDGE = .1
SCHEMA = 'center214.numeric.v1'


def event_bins(events, nframes):
    """Recreate the official 30-offset, 750-frame, two-direction window rows."""
    if isinstance(nframes, bool) or not isinstance(nframes, int) or nframes < 1:
        raise ValueError('nframes must be a positive integer')
    rows = []
    for offset in range(0, 750, 25):
        n = (nframes + offset + 749) // 750
        counts = np.zeros((n, 2), dtype=np.int64)
        for frame, direction in events:
            frame, direction = int(frame), int(direction)
            if not 1 <= frame <= nframes or direction not in (-1, 1):
                raise ValueError('event frame or direction out of range')
            counts[(frame - 1 + offset) // 750, 0 if direction == 1 else 1] += 1
        rows.extend(counts.tolist())
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def maximum_matching(left, right, edges):
    """Cardinality-only bipartite matching; IDs are opaque strings/integers."""
    left, right = list(left), list(right)
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError('duplicate graph vertex')
    ls, rs = set(left), set(right)
    adj = defaultdict(list)
    for p, g in edges:
        if p not in ls or g not in rs:
            raise ValueError('edge endpoint is absent from its declared vertex set')
        adj[p].append(g)
    owner = {}

    def visit(p, seen):
        for g in adj[p]:
            if g in seen:
                continue
            seen.add(g)
            if g not in owner or visit(owner[g], seen):
                owner[g] = p
                return True
        return False

    return sum(bool(visit(p, set())) for p in left)


def score_graph(state, nframes):
    """Recompute bins, legal-edge matchings, W and both event F1s."""
    def parse(rows):
        out=[]
        for row in rows:
            eid, signed = str(row[0]), int(row[1])
            if signed == 0 or abs(signed) > nframes:
                raise ValueError('signed event frame outside scene')
            out.append((eid, abs(signed), 1 if signed > 0 else -1))
        if len({r[0] for r in out}) != len(out):
            raise ValueError('one event vertex per canonical event ID required')
        return out
    pred, gt = parse(state['pred_events']), parse(state['gt_events'])
    pb, gb = event_bins([(f, s) for _, f, s in pred], nframes), event_bins([(f, s) for _, f, s in gt], nframes)
    declared = state['bins']
    if not np.array_equal(pb, np.asarray(declared['pred'], dtype=np.int64)) or not np.array_equal(gb, np.asarray(declared['gt'], dtype=np.int64)):
        raise ValueError('sealed event clocks do not reproduce integer bins')
    if pb.shape != gb.shape:
        raise ValueError('prediction and GT window shapes differ')
    numerator = int(np.abs(pb - gb).sum())
    ng, npred = len(gt), len(pred)
    result = dict(numerator=numerator, denominator=30 * ng,
                  W=100 * numerator / (30 * ng) if ng else None,
                  nGT=ng,nPred=npred,pred_bins=pb.tolist(), gt_bins=gb.tolist(), iou={})
    for threshold in ('0.5', '0.3'):
        graph = state['graphs'][threshold]
        edges = [(str(p), str(g)) for p, g in graph['edges']]
        tp = maximum_matching([p for p, _, _ in pred], [g for g, _, _ in gt], edges)
        result['iou'][threshold] = dict(TP=tp, Pred=npred, GT=ng, FP=npred-tp,
            FN=ng-tp, E=npred+ng-2*tp,
            F1=2*tp/(npred+ng) if npred+ng else 1.)
    return result


def ensemble_clip(member_scores):
    """Pool a clip's three member windows before absolute error (registered E3)."""
    if len(member_scores) != 3:
        raise ValueError('each clip must contain exactly three member scores')
    gt = np.asarray(member_scores[0]['gt_bins'], dtype=np.int64)
    if any(not np.array_equal(gt, np.asarray(s['gt_bins'], dtype=np.int64)) for s in member_scores):
        raise ValueError('the three clip members must share a GT window vector')
    total_pred = sum((np.asarray(s['pred_bins'], dtype=np.int64) for s in member_scores), np.zeros_like(gt))
    if any(np.asarray(s['pred_bins']).shape != gt.shape for s in member_scores):
        raise ValueError('member prediction window shape differs')
    numerator = int(np.abs(total_pred - 3*gt).sum())
    n_gt = int(member_scores[0]['iou']['0.5']['GT'])
    den = 90*n_gt
    return dict(numerator=numerator, denominator=den, E3=100*numerator/den if den else None,
                gt_bins=gt.tolist(), pred_sum_bins=total_pred.tolist(),
                mean_member_F1={t:sum(s['iou'][t]['F1'] for s in member_scores)/3 for t in ('0.5','0.3')},
                members=member_scores)


def pool_domain(clip_scores):
    """Pool raw E3 counts and member TP/P/GT across the two clips in a hive."""
    if not clip_scores:
        raise ValueError('each hive must contain a nonempty registered clip set')
    numerator = sum(int(c['numerator']) for c in clip_scores)
    denominator = sum(int(c['denominator']) for c in clip_scores)
    pooled_members=[]
    for slot in range(3):
        record={'slot':slot, 'iou':{}}
        for t in ('0.5','0.3'):
            q={k:sum(int(c['members'][slot]['iou'][t][k]) for c in clip_scores)
               for k in ('TP','Pred','GT','FP','FN','E')}
            q['F1']=2*q['TP']/(q['Pred']+q['GT']) if q['Pred']+q['GT'] else 1.
            record['iou'][t]=q
        pooled_members.append(record)
    return dict(numerator=numerator, denominator=denominator,
        E3=100*numerator/denominator if denominator else None,
        mean_member_F1={t:sum(m['iou'][t]['F1'] for m in pooled_members)/3 for t in ('0.5','0.3')},
        members=pooled_members)


def replay_member_bounds(bound_evidence, member_k, member_o):
    """Recompute separate member W intervals; E3 intervals are not substitutes."""
    from bound214 import envelope
    if bound_evidence.get('schema') != 'endpoint-side-bound-evidence-v1':
        raise ValueError('unknown endpoint-side bound evidence')
    if len(bound_evidence['members']) != 3:
        raise ValueError('bound evidence needs three original member run rosters')
    gt = np.asarray(bound_evidence['gt_windows'], dtype=np.int64)
    rows = []
    for slot, member in enumerate(bound_evidence['members']):
        fixed = np.asarray(member['fixed_windows'], dtype=np.int64)
        env = envelope(member['runs'], bound_evidence['nframes'])
        baseline = env['baseline'] + fixed
        gt_one = gt
        den = int(gt_one.sum())
        k = np.asarray(member_k[slot]['bins']['pred'], dtype=np.int64)
        o = np.asarray(member_o[slot]['bins']['pred'], dtype=np.int64)
        if not (k.shape == o.shape == gt_one.shape == baseline.shape):
            raise ValueError('member K/O/bound/GT clock shapes differ')
        if not np.array_equal(k, baseline):
            raise ValueError('side baseline does not reproduce official member K windows')
        fields = {}
        for mode in ('all_side', 'owner_side'):
            lo, hi = env[mode]['lower'] + fixed, env[mode]['upper'] + fixed
            if (lo > hi).any() or (lo > o).any() or (o > hi).any():
                raise ValueError('O is outside the registered physical member side interval')
            n = int((np.maximum(lo-gt_one, 0) + np.maximum(gt_one-hi, 0)).sum())
            fields[mode] = dict(lower_numerator=n, denominator=den,
                W_lower_bound=100*n/den if den else None,
                max_saving_vs_K_numerator=int(np.abs(k-gt_one).sum())-n,
                max_saving_vs_K_pp=(100*(int(np.abs(k-gt_one).sum())-n)/den) if den else None)
        kn, on = int(np.abs(k-gt_one).sum()), int(np.abs(o-gt_one).sum())
        rows.append(dict(member_slot=slot, gt_window_count=den, K=dict(numerator=kn, denominator=den,
            W=100*kn/den if den else None), O=dict(numerator=on, denominator=den,
            W=100*on/den if den else None), intervals=fields))
    return rows


def pool_member_bounds(clip_rows):
    """Pool the two clips per hive at integer numerator/denominator level."""
    if not clip_rows or any(len(x) != 3 for x in clip_rows):
        raise ValueError('each hive needs registered clips and the same three member slots')
    result=[]
    for slot in range(3):
        values=[clip[slot] for clip in clip_rows]
        row={'member_slot':slot, 'K':{}, 'O':{}, 'intervals':{}}
        for arm in ('K','O'):
            n=sum(x[arm]['numerator'] for x in values);d=sum(x[arm]['denominator'] for x in values)
            row[arm]=dict(numerator=n,denominator=d,W=100*n/d if d else None)
        for mode in ('all_side','owner_side'):
            n=sum(x['intervals'][mode]['lower_numerator'] for x in values)
            d=sum(x['intervals'][mode]['denominator'] for x in values)
            kn=sum(x['K']['numerator'] for x in values)
            row['intervals'][mode]=dict(lower_numerator=n,denominator=d,
                W_lower_bound=100*n/d if d else None,
                max_saving_vs_K_numerator=kn-n,
                max_saving_vs_K_pp=100*(kn-n)/d if d else None)
        result.append(row)
    return result


def endpoint_weights(rows):
    """Equal effective domain, clip, labelled run, then endpoint."""
    if not rows:
        return np.empty(0, dtype=np.float64)
    domains = {str(r['domain']) for r in rows}
    clips = defaultdict(set)
    runs = defaultdict(set)
    counts = defaultdict(int)
    for r in rows:
        h, c, run = str(r['domain']), str(r['clip']), str(r['run_id'])
        clips[h].add(c)
        runs[h, c].add(run)
        counts[h, c, run] += 1
    w = np.asarray([1 / (len(domains) * len(clips[str(r['domain'])]) *
                          len(runs[str(r['domain']), str(r['clip'])]) *
                          counts[str(r['domain']), str(r['clip']), str(r['run_id'])])
                     for r in rows], dtype=np.float64)
    return w / w.sum()


def fit_axis_features(rows, *, include_chi):
    """Refit the registered two-output weighted ridge head from anonymized rows."""
    ordered = sorted(rows, key=lambda r: int(r['source_order']))
    if not ordered:
        return dict(status='empty_source', mean=[], scale=[], beta=[], intercept=[0., 0.])
    if any(not r.get('target_known', False) for r in ordered):
        raise ValueError('all fit rows must have known assigned-owner targets')
    features = [list(r['G6']) + (list(r['chi24']) if include_chi else []) for r in ordered]
    X = np.asarray(features, dtype=np.float64)
    Y = np.asarray([r['target_clipped'] for r in ordered], dtype=np.float64)
    if X.shape[1] != (30 if include_chi else 6) or Y.shape != (len(X), 2):
        raise ValueError('feature/target shape differs')
    weights = endpoint_weights(ordered)
    mu = weights @ X
    const = np.all(X == X[0], axis=0)
    mu[const] = X[0, const]
    scale = np.sqrt(weights @ ((X-mu)**2))
    zero_scale_columns=np.flatnonzero(scale==0).tolist()
    scale[scale == 0] = 1.
    Z = (X - mu) / scale
    intercept = weights @ Y
    constant_y = np.all(Y == Y[0], axis=0)
    intercept[constant_y] = Y[0, constant_y]
    beta = np.linalg.solve(Z.T @ (weights[:, None] * Z) + RIDGE*np.eye(X.shape[1]),
                           Z.T @ (weights[:, None] * (Y-intercept)))
    beta[:,constant_y]=0.
    return dict(status='fitted', mean=mu.tolist(), scale=scale.tolist(), beta=beta.tolist(),
                intercept=intercept.tolist(), weights=weights.tolist(),
                zero_scale_columns=zero_scale_columns,training_rows=[r['endpoint_id'] for r in ordered])


def refit_folds(bundle):
    """Independent four-fold fit from the public anonymous receiver-K bank.

    Each row must be one eligible, owner-known receiver endpoint.  The package
    carries only clipped normalized targets; raw owner coordinates are neither
    needed nor accepted by this routine.
    """
    rows = bundle['source_rows']
    keys = [r['endpoint_id'] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('source endpoint aliases must be unique')
    if any(r['member_index'] != 0 or r['role'] != 'receiverK' or
           r['target_known'] is not True or r['eligible'] is not True for r in rows):
        raise ValueError('only eligible, owner-known source receiver-K rows may train')
    out = {}
    for heldout in DOMAINS:
        order = bundle['fold_orders'][heldout]
        source = [dict(r, source_order=order[r['endpoint_id']]) for r in rows if r['domain'] != heldout]
        if set(order) != {r['endpoint_id'] for r in source}:
            raise ValueError('fold training order must cover exactly the eligible known source rows')
        models = {}
        for head, chi in (('G', False), ('I', True)):
            models[head] = fit_axis_features(source, include_chi=chi)
        out[heldout] = models
    return out


def predict_shift(model, endpoint, arm):
    if arm == 'K':
        return [0., 0.]
    if arm not in ('G', 'I', 'Z'):
        raise ValueError('only model arms G/I/Z are predicted')
    head = model['G'] if arm == 'G' else model['I']
    if head['status'] == 'empty_source' or not endpoint['eligible']:
        return [0., 0.]
    chi = np.asarray(endpoint['chi24'], dtype=np.float64)
    if arm == 'Z':
        chi = -chi
    x = np.asarray(list(endpoint['G6']) + ([] if arm == 'G' else chi.tolist()), dtype=np.float64)
    pred = ((x - np.asarray(head['mean'])) / np.asarray(head['scale'])) @ np.asarray(head['beta']) + np.asarray(head['intercept'])
    return np.clip(pred, -LIMIT, LIMIT).tolist()


def self_test():
    # Counts exercise a long scene (more than 30 window rows) and signed clocks.
    ev = [(1, 1), (749, -1), (750, 1), (3950, -1)]
    b = event_bins(ev, 3950)
    assert b.shape[1] == 2 and b.shape[0] > 30 and int(b.sum()) == 4 * 30
    assert maximum_matching(['a', 'b'], ['x', 'y'], [('a', 'x'), ('b', 'x'), ('b', 'y')]) == 2
    tiny = dict(pred_events=[['p1', 12]], gt_events=[['g1', 12]],
                bins={'pred': event_bins([(12, 1)], 100).tolist(), 'gt': event_bins([(12, 1)], 100).tolist()},
                graphs={'0.5': {'edges': [['p1', 'g1']]}, '0.3': {'edges': []}})
    tiny_score = score_graph(tiny, 100)
    assert tiny_score['iou']['0.5']['TP'] == 1 and tiny_score['iou']['0.3']['FN'] == 1
    clip_score=ensemble_clip([tiny_score, tiny_score, tiny_score])
    assert clip_score['denominator']==90 and clip_score['numerator']==0
    from bound214 import window_counts, net_event
    gt1 = window_counts([(2, 1)], 10)
    run = dict(frames=[1, 2], sides=[False, True], eligible=[1], oracle_sides=[False, False])
    empty = window_counts([], 10)
    bound = dict(schema='endpoint-side-bound-evidence-v1', nframes=10, gt_windows=gt1.tolist(),
                 members=[dict(fixed_windows=empty.tolist(), runs=[run]) for _ in range(3)])
    krows=[dict(bins=dict(pred=gt1.tolist())) for _ in range(3)]
    orows=[dict(bins=dict(pred=empty.tolist())) for _ in range(3)]
    member_intervals = replay_member_bounds(bound, krows, orows)
    assert member_intervals[0]['K']['W']==0 and member_intervals[0]['O']['W']==100
    assert member_intervals[0]['intervals']['owner_side']['lower_numerator']==0
    rows = [dict(domain='35', clip='a', run_id='r1', clip_index=0, run_order=0, endpoint_order=i,source_order=i,
                 endpoint_id=str(i), G6=[1, 2, 3, 4, 5, 6], chi24=[0]*24,
                 target_clipped=[.1, -.2], target_known=True) for i in range(2)]
    np.testing.assert_allclose(endpoint_weights(rows), [.5, .5])
    m = {'G': fit_axis_features(rows, include_chi=False), 'I': fit_axis_features(rows, include_chi=True)}
    ep = dict(eligible=True, G6=[1, 2, 3, 4, 5, 6], chi24=[.01]*24)
    assert predict_shift(m, ep, 'Z') == predict_shift(m, dict(ep, chi24=[-.01]*24), 'I')
    # Explicitly verify output cap despite arbitrarily large coefficients.
    m['G']['beta'] = (np.ones((6, 2))*1e4).tolist()
    assert max(abs(x) for x in predict_shift(m, ep, 'G')) <= LIMIT
    return dict(passed=True, long_scene_rows=int(b.shape[0]), matching=2,
                weighted_refit=True, Z_reuses_I_with_raw_chi_sign_flip=True,
                shift_cap=LIMIT, field_reads=0)


def close_tree(a,b,*,atol=2e-10):
    if isinstance(a,dict) and isinstance(b,dict):
        return set(a)==set(b) and all(close_tree(a[k],b[k],atol=atol) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(close_tree(x,y,atol=atol) for x,y in zip(a,b))
    if isinstance(a,(int,float,np.number)) and not isinstance(a,(bool,np.bool_)) and isinstance(b,(int,float,np.number)) and not isinstance(b,(bool,np.bool_)):
        return math.isclose(float(a),float(b),rel_tol=0,abs_tol=atol)
    return a==b


def verify_model_refs(models, references):
    for heldout in DOMAINS:
        for head in ('G','I'):
            got=models[heldout][head];want=references[heldout][head]
            for key in ('status','mean','scale','beta','intercept','zero_scale_columns'):
                if not close_tree(got[key],want[key]):raise ValueError(f'refitted source model differs: {heldout}/{head}/{key}')


def _state_equal(actual, expected):
    for key in ('numerator','denominator','nGT','nPred'):
        if actual[key]!=expected[key]:return False
    if expected.get('W') is not None and not math.isclose(actual['W'],expected['W'],abs_tol=1e-11,rel_tol=0):return False
    for t in ('0.5','0.3'):
        for key in ('TP','Pred','GT','FP','FN','E'):
            if actual['iou'][t][key]!=expected['iou'][t][key]:return False
        if not math.isclose(actual['iou'][t]['F1'],expected['iou'][t]['F1'],abs_tol=1e-12,rel_tol=0):return False
    return True


def _endpoint_diagnostics(records):
    if not records:return dict(eligible=0,matched=0,unmatched=0,GT_inside=0,GT_outside=0)
    matched=[r for r in records if r['original_matched']]
    n=len(matched)
    out=dict(eligible=len(records),matched=n,unmatched=len(records)-n,
        GT_inside=sum(bool(r['GT_side']) for r in matched),GT_outside=sum(not bool(r['GT_side']) for r in matched),
        side_flips=sum(bool(r['side_flipped']) for r in records),flip_repaired=sum(bool(r.get('flip_repaired',False)) for r in matched),
        flip_damaged=sum(bool(r.get('flip_damaged',False)) for r in matched))
    for arm_name,key in (('boundary_MAE','boundary_MAE'),('center_error','center_error'),('side_error_rate','side_error')):
        for phase in ('before','after'):
            source=f'{phase}_{key}'
            field=source if source in matched[0] else None
            if field is None:
                # The compact endpoint rows store signed distances/diagonal and
                # booleans, so aggregate metrics are rebuilt below.
                vals=[]
                for r in matched:
                    if arm_name=='boundary_MAE':
                        dist=r['original_signed_polygon_distance'] if phase=='before' else r['corrected_signed_polygon_distance']
                        vals.append(abs(dist-r['GT_signed_polygon_distance'])/r['original_box_diagonal'])
                    elif arm_name=='center_error':vals.append(float(r[f'{phase}_center_error']))
                    else:vals.append(float(bool(r[f'{phase}_side_error'])))
            else:vals=[float(r[field]) for r in matched]
            out[f'{phase}_{arm_name}']=sum(vals)/n if n else None
    return out


def replay_bundle(bundle):
    if bundle.get('schema')!=SCHEMA:raise ValueError('unknown §214 evidence schema')
    if bundle.get('prereg')!='79dbed053bcddd4290300494ceff29c8130a5807' or bundle.get('supplement')!='21c6c803290fad2816ca472ec673b9f2eda76759':
        raise ValueError('registered primary/supplement marker differs')
    if len(bundle['endpoint_rows'])!=15026 or sum(bool(r['eligible']) for r in bundle['endpoint_rows'])!=867:
        raise ValueError('fixed 15,026 endpoint roster/867 eligible denominator required')
    if len(bundle['source_rows'])!=155 or len(bundle['clips'])!=6:
        raise ValueError('source model/clip inventory differs')
    models=refit_folds(bundle);verify_model_refs(models,bundle['model_references'])

    member_index={}
    for row in bundle['endpoint_rows']:
        member_index.setdefault((row['domain'],row['clip'],row['member_index']),[]).append(row)
    if len(member_index)!=18:raise ValueError('expected 18 member-clip endpoint rosters')
    shift_errors=[]
    for key,rows in member_index.items():
        heldout=key[0];model=models[heldout]
        for row in rows:
            for arm in ('K','G','I','Z'):
                observed=np.asarray(predict_shift(model,row,arm),float)
                expected=np.asarray(row['certified_shifts'][arm],float)
                if not np.allclose(observed,expected,atol=2e-10,rtol=0):
                    raise ValueError('refitted shift differs from sealed shift: '+repr(key)+'/'+row['endpoint_id']+'/'+arm)
                shift_errors.append(float(np.max(np.abs(observed-expected))))

    clip_results=[];member_clip_bounds=[];domain_clips={h:[] for h in DOMAINS}
    diag_cells=defaultdict(lambda:defaultdict(list))
    all_k212=True;all_events=0
    for clip in bundle['clips']:
        h,c,nf=clip['domain'],clip['clip'],int(clip['nframes'])
        if h not in DOMAINS or len(clip['members'])!=3:raise ValueError('clip domain/member schema differs')
        per_arm={a:[] for a in ARMS};member_scores=[];endpoint_checks={}
        for member in sorted(clip['members'],key=lambda m:int(m['member_index'])):
            slot=int(member['member_index']);rowmap={r['endpoint_id']:r for r in member_index[h,c,slot]}
            state_scores={};arm_diags={}
            for arm in ARMS:
                state=member['states'][arm]
                score=score_graph(state,nf)
                if not _state_equal(score,state['declared']):raise ValueError(f'graph replay differs from native declared metrics: {h}/{c}/m{slot}/{arm}')
                if arm=='K':
                    closure=member['k212_closure']
                    if not all(closure.get(k) is True for k in ('all18_original_K','signed_events_exact','both_IoU_edges_TP_exact','all30_integer_windows_exact')):
                        raise ValueError('K does not close against the registered §212 event/window receipt')
                    all_k212=all_k212 and True
                diag=state['endpoint_diagnostics'];
                if len(diag)!=sum(bool(r['eligible']) for r in rowmap.values()):raise ValueError('endpoint diagnostic denominator differs')
                if any(r['endpoint_id'] not in rowmap or not rowmap[r['endpoint_id']]['eligible'] for r in diag):raise ValueError('diagnostic endpoint alias outside eligible roster')
                aggregate=_endpoint_diagnostics(diag);arm_diags[arm]=aggregate
                state_scores[arm]=score;per_arm[arm].append(score);all_events+=len(state['pred_events'])+len(state['gt_events'])
                for r in diag:
                    if r['original_matched']:
                        if r['original_box_diagonal']<=0 or not all(k in r for k in ('GT_side','GT_signed_polygon_distance')):
                            raise ValueError('matched endpoint distance/normalizer missing')
                diag_cells[h,slot][arm].extend(diag)
            member_scores.append(dict(member_slot=slot,states=state_scores,diagnostics=arm_diags))
        arms={a:ensemble_clip(per_arm[a]) for a in ARMS}
        # K/O actual member W and two independent side-envelope W bounds.
        bound_rows=replay_member_bounds(clip['bound_evidence'],
            [m['states']['K'] for m in sorted(clip['members'],key=lambda x:int(x['member_index']))],
            [m['states']['O'] for m in sorted(clip['members'],key=lambda x:int(x['member_index']))])
        member_clip_bounds.append(dict(domain=h,clip=c,members=bound_rows))
        result=dict(domain=h,clip=c,arms=arms,members=member_scores,member_bounds=bound_rows)
        clip_results.append(result);domain_clips[h].append(result)

    if len(clip_results)!=6:raise ValueError('all six original clips must be present')
    domains={h:{a:pool_domain([c['arms'][a] for c in domain_clips[h]]) for a in ARMS} for h in DOMAINS}
    primary={}
    for h in ('35','33'):
        k,i=domains[h]['K'],domains[h]['I']
        primary[h+'_E3_gain_at_least_1pp']=(k['numerator']*i['denominator']-i['numerator']*k['denominator'])*100>=k['denominator']*i['denominator']
        primary[h+'_mean_member_F1_loss_at_most_0.01']=(
            domains[h]['I']['mean_member_F1']['0.5'] >= domains[h]['K']['mean_member_F1']['0.5']-.01)
    for h in ('W','G'):
        primary[h+'_E3_not_worse']=domains[h]['I']['numerator']*domains[h]['K']['denominator']<=domains[h]['K']['numerator']*domains[h]['I']['denominator']

    reserve=[];support_ok=True;spatial_ok=True
    for h in ('35','33'):
        for slot in (1,2):
            byarm={a:diag_cells[h,slot][a] for a in ARMS}
            matched=[r for r in byarm['I'] if r['original_matched']]
            inside=sum(bool(r['GT_side']) for r in matched);outside=len(matched)-inside
            sufficient=bool(matched and inside and outside)
            support_ok &= sufficient
            maes={}
            for arm in ('G','I','Z'):
                vals=[]
                for r in byarm[arm]:
                    if r['original_matched']:
                        vals.append(abs(float(r['corrected_signed_polygon_distance'])-float(r['GT_signed_polygon_distance']))/float(r['original_box_diagonal']))
                maes[arm]=sum(vals)/len(vals) if vals else None
            better=sufficient and maes['I'] is not None and maes['G'] is not None and maes['Z'] is not None and maes['I']<maes['G'] and maes['I']<maes['Z']
            spatial_ok &= better
            reserve.append(dict(domain=h,member_slot=slot,matched=len(matched),GT_inside=inside,GT_outside=outside,
                sufficient=sufficient,boundary_MAE=maes,I_strictly_better_G_Z=better))
    strict_head=any(domains[h]['I']['numerator']*domains[h]['G']['denominator']<domains[h]['G']['numerator']*domains[h]['I']['denominator'] and
                    domains[h]['I']['numerator']*domains[h]['Z']['denominator']<domains[h]['Z']['numerator']*domains[h]['I']['denominator'] for h in ('35','33'))
    mechanism_status='inconclusive_support' if not support_ok else ('passed' if spatial_ok and strict_head else 'not_passed')
    gates=dict(primary=primary,crop_spatial_information=dict(status=mechanism_status,
        passed=(spatial_ok and strict_head) if support_ok else None,sufficient=support_ok,
        reserve_cells=reserve,head_I_strictly_below_G_and_Z_in_any_registered_domain=strict_head))
    expected=bundle['independent_gate_references']
    if primary!=expected['primary']:
        raise ValueError('primary gate vector differs from independent audit')
    expected_mechanism=expected['crop_spatial_information']
    expected_status=('inconclusive_support' if not expected_mechanism['sufficient'] else
                     ('passed' if expected_mechanism['passed'] else 'not_passed'))
    if gates['crop_spatial_information']['status']!=expected_status or gates['crop_spatial_information']['sufficient']!=expected_mechanism['sufficient'] or gates['crop_spatial_information']['passed']!=expected_mechanism['passed']:
        raise ValueError('crop spatial information gate differs from independent audit')

    # Independently replay both W-bound families by member and compare the
    # integer domain pools to the independent audit's compact count receipt.
    member_bounds_by_domain={}
    from bound214 import replay_evidence,pool_bounds
    for h in DOMAINS:
        clip_rows=[c['member_bounds'] for c in domain_clips[h]]
        pooled=pool_member_bounds(clip_rows);member_bounds_by_domain[h]=pooled
        source_rows=bundle['independent_count_bounds'][h]['members']
        byslot={int(r['member_slot']):r for r in source_rows}
        for r in pooled:
            ref=byslot[r['member_slot']]
            if r['K']['denominator']!=ref['denominator'] or any(r['intervals'][m]['lower_numerator']!=ref['lower_bound_numerators'][m] for m in ('all_side','owner_side')):
                raise ValueError('per-member W bounds differ from independent audit receipt: '+h+'/m'+str(r['member_slot']))
        clip_evidence=[c['bound_evidence'] for c in bundle['clips'] if c['domain']==h]
        replayed=[replay_evidence(e) for e in clip_evidence]
        pooled_e3=pool_bounds(replayed)
        ref_e3=bundle['independent_count_bounds'][h]['E3']
        for mode in ('all_side','owner_side'):
            if pooled_e3['layers'][mode]['lower_bound_numerator']!=ref_e3[mode]['numerator'] or pooled_e3['denominator']!=ref_e3[mode]['denominator']:
                raise ValueError('independent E3 interval differs: '+h+'/'+mode)
    return dict(schema=SCHEMA,passed=True,receipts=bundle['receipts'],
        counts=dict(endpoint_rows=15026,eligible_endpoints=867,source_rows=155,member_clips=18,native_graph_states=90),
        models=models,shift_max_abs_error=max(shift_errors,default=0.),
        clips=clip_results,domains=domains,gates=gates,
        member_count_bounds=member_clip_bounds,pooled_member_count_bounds=member_bounds_by_domain,
        scope=bundle['scope'],K212_closure_all18=all_k212,scored_event_vertices=all_events,
        count_bound_scope='side-option relaxed interval lower bounds; not physically attainable witnesses or universal model bounds')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('evidence', nargs='?')
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return
    if not args.evidence or not args.out:raise SystemExit('usage: replay214.py evidence214.json.gz --out result214.json')
    path=Path(args.evidence)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf8') as f:bundle=json.load(f)
    result=replay_bundle(bundle)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(out),passed=result['passed'],sha256=__import__('hashlib').sha256(out.read_bytes()).hexdigest()),sort_keys=True))


if __name__ == '__main__':
    main()
