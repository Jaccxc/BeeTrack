"""Replay compact whole-member numerical evidence; never opens native data."""
import argparse
import gzip
import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path
import numpy as np
import model
import run


def equal(a, b, path='root'):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a: equal(a[k], b[k], path+'/'+k)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): equal(x, y, path+'/'+str(i))
    elif isinstance(a, float):
        assert isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10), path
    else:
        assert a == b, path


def slim(value):
    """Drop duplicated window matrices inside configuration summaries only."""
    if isinstance(value, dict):
        return {k: slim(v) for k, v in value.items() if k not in ('clip_routes',)}
    if isinstance(value, list): return [slim(v) for v in value]
    return value


def signed_bins(events,nf):
    result=[]
    assert len({e[0] for e in events})==len(events)
    for offset in range(30):
        rows=[[0,0] for _ in range((nf+25*offset+749)//750)]
        for _,date,*_ in events:
            assert isinstance(date,int) and 1<=abs(date)<=nf
            rows[(abs(date)-1+25*offset)//750][int(date<0)]+=1
        result.extend(rows)
    return result


def matching(predictions,truth,edges):
    ps={p[0] for p in predictions};gs={g[0] for g in truth};adj={p:[] for p in ps}
    for p,g in edges:
        assert p in ps and g in gs
        adj[p].append(g)
    owner={}
    def visit(p,seen):
        for g in adj[p]:
            if g in seen:continue
            seen.add(g)
            if g not in owner or visit(owner[g],seen):owner[g]=p;return True
        return False
    for p in sorted(ps):visit(p,set())
    return len(owner)


def replay(data, require_scope=True):
    assert data['schema'] == 'whole-member-compact-v1'
    videos=data['videos']; folds=data['folds']; summary=data['summary']
    if require_scope:
        assert len(videos)==20 and sum(len(v['members'])*2 for v in videos)==120
    sign_flips=0;first_order_optima=0
    for v in videos:
        losses=[run.route_state(v,s)['numerator'] for s in run.STATES]
        chosen=[]
        for member in range(3):
            differences=[]
            for bits in product((0,1),repeat=2):
                low=list(bits);low.insert(member,0);high=low.copy();high[member]=1
                differences.append(losses[run.STATES.index(tuple(high))]-losses[run.STATES.index(tuple(low))])
            sign_flips+=min(differences)<0<max(differences)
            chosen.append(int(sum(differences)<0))
        first_order_optima+=losses[run.STATES.index(tuple(chosen))]==min(losses)
        assert len(v['members'])==3
        for m in v['members']:
            assert set(m['arms'])==set(run.ARMS)
            for arm in m['arms'].values():
                equal(arm['gt_bins'],v['gt_bins'])
                assert arm['nG']==v['nG']
                p=np.asarray(arm['pred_bins']);g=np.asarray(v['gt_bins'])
                assert p.shape==g.shape and p.ndim==2 and p.shape[1]==2
                assert (p>=0).all() and (p==p.astype(np.int64)).all()
                assert math.isclose(arm['W'],100*int(np.abs(p-g).sum())/(30*v['nG']),abs_tol=1e-8)
                for metric in arm['iou'].values():
                    assert 0<=metric['tp']<=min(metric['pred'],metric['gt'])
                    assert metric['pred']==arm['nP'] and metric['gt']==arm['nG']
                    assert math.isclose(metric['f1'],run.f1(metric['tp'],metric['pred'],metric['gt']),abs_tol=1e-12)
                if 'evidence' in arm:
                    e=arm['evidence']
                    equal(signed_bins(e['gt'],v['nf']),v['gt_bins'])
                    equal(signed_bins(e['predictions'],v['nf']),arm['pred_bins'])
                    assert len(e['gt'])==arm['nG'] and len(e['predictions'])==arm['nP']
                    for t,key in (('0.5','edges05'),('0.3','edges03')):
                        assert matching(e['predictions'],e['gt'],e[key])==arm['iou'][t]['tp']
    choices={}
    for heldout, fold in folds.items():
        records=[dict(hive=v['domain'],group_id=v['group'],clip=v['video'],counts=counts(v),gw=v['gt_bins']) for v in videos if v['domain']!=heldout]
        assert set(fold['source_domains'])==set(run.DOMAINS)-{heldout}
        fitted=model.fit(records,heldout=heldout)
        equal(fitted,fold['model'],heldout+'/fit')
        byunit={c['unit']:c for c in fold['choices']}
        assert set(byunit)=={v['unit'] for v in videos if v['domain']==heldout}
        for v in videos:
            if v['domain']==heldout:
                prediction=model.predict(counts(v),fitted)
                equal(prediction,byunit[v['unit']]['prediction'],v['unit']+'/predict')
                choices[v['unit']]=prediction
    total=0
    for h, saved in summary['hives'].items():
        units=[]; methods={k:[] for k in ('L','H','F','M','D','J','U','O')}
        for gid in sorted({v['group'] for v in videos if v['domain']==h}):
            vs=[v for v in videos if v['domain']==h and v['group']==gid]
            enumerated=run.enumerate_group(vs);units.append(enumerated);total+=len(enumerated['candidates'])
            for name in ('L','H','F','M','D','J'):
                if name in ('M','D','J'): states=[choices[v['unit']][name]['bits'] for v in vs]
                else: states=[(1,1,1) if name=='H' or name=='F' and h in ('35','33') else (0,0,0)]*len(vs)
                methods[name].append(run.combine(vs,states))
            uniform=[run.combine(vs,states) for states in product(((0,0,0),(1,1,1)),repeat=len(vs))]
            for name,pool in (('U',uniform),('O',enumerated['candidates'])):
                minimum=min(c['numerator'] for c in pool);ties=[c for c in pool if c['numerator']==minimum]
                chosen=dict(min(ties,key=lambda c:c['states']),tie_count=len(ties),tie_F1_ranges={t:[float(min(run.exact_f1(c,t) for c in ties)),float(max(run.exact_f1(c,t) for c in ties))] for t in ('0.5','0.3')})
                methods[name].append(chosen)
        equal(slim(units),saved['groups'],h+'/all-configurations')
        for name in methods: equal(slim(methods[name]),saved['methods'][name],h+'/'+name)
        means={name:dict(E3=float(np.mean([c['E3'] for c in values])),F1_05=float(sum((run.exact_f1(c) for c in values),Fraction())/len(values)),F1_03=float(sum((run.exact_f1(c,'0.3') for c in values),Fraction())/len(values))) for name,values in methods.items()}
        equal(means,saved['means'],h+'/means')
        equal({name:dict(E3=means[name]['E3']-means['F']['E3'],F1_05=means[name]['F1_05']-means['F']['F1_05']) for name in methods},saved['matched_differences'],h+'/differences')
        equal(run.guarded_oracle(units,methods['F']),saved['guarded_oracle'],h+'/guarded')
    differences={h:summary['hives'][h]['matched_differences']['J'] for h in run.DOMAINS}
    gate=(differences['35']['E3'] < -1.4 and differences['33']['E3'] < -1.3 and differences['W']['E3'] <=1.4 and all(differences[h]['F1_05']>=-.005-1e-15 for h in ('35','33','W')))
    assert gate==summary['investment_pass']
    joint=all(summary['hives'][h]['means']['J']['E3']<summary['hives'][h]['means']['D']['E3'] for h in ('35','33'))
    assert joint==summary['joint_better_than_D_both_heads']==summary['joint_gain_condition']
    if require_scope:
        assert total==448 and sign_flips==0 and first_order_optima==20
    return dict(videos=len(videos),physical_configurations=total,folds=len(folds),refit=True,
                posthoc_truth_only=dict(strict_marginal_sign_flips=int(sign_flips),member_opportunities=3*len(videos),first_order_optima=int(first_order_optima)))


def counts(v):
    return np.asarray([[m['arms'][a]['pred_bins'] for a in run.ARMS] for m in v['members']],dtype=np.int64)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('evidence',type=Path);args=ap.parse_args()
    opener=gzip.open if args.evidence.suffix=='.gz' else open
    with opener(args.evidence,'rt',encoding='utf-8') as f: data=json.load(f)
    print(json.dumps(replay(data),sort_keys=True))
