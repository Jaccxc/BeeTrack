"""Numeric-only paired-texture replay primitives; field adapter awaits seals."""
import hashlib
import numpy as np

COLUMNS={'P':list(range(3)),'C':list(range(9)),'T':list(range(15)),'V':list(range(9,15))}

def difference(rows,descriptors):
    b=np.asarray(rows,float)[:,2:6];d=np.asarray(descriptors,float)
    assert b.shape==(4,4) and d.shape==(4,12) and (b[:,2:]>0).all()
    scale=np.sqrt(b[:,2]*b[:,3]).mean()
    def edge(i,j):
        a,c=b[i],b[j]
        return np.r_[np.linalg.norm((a[:2]+a[2:]/2)-(c[:2]+c[2:]/2))/scale,
                     abs(np.log(a[2]*a[3])-np.log(c[2]*c[3])),
                     abs(np.log(a[2]/a[3])-np.log(c[2]/c[3])),abs(d[i]-d[j])]
    return edge(0,3)+edge(1,2)-edge(0,2)-edge(1,3)

def fit(records,name):
    columns=COLUMNS[name];videos=sorted({r['clip'] for r in records});assert videos
    x=np.asarray([r['difference'] for r in records])[:,columns]
    w=np.asarray([1/len(videos)/sum(q['clip']==r['clip'] for q in records) for r in records])
    scale=np.sqrt(w@(x*x));scale[scale==0]=1
    z=x/scale;beta=np.linalg.solve(z.T@(w[:,None]*z)+.2*np.eye(len(columns)),z.T@w)
    assert np.max(abs(z.T@(w*(z@beta-1))+.2*beta),initial=0)<1e-10
    return dict(columns=columns,scale=scale.tolist(),beta=beta.tolist())

def choose(vector,digest,fitted):
    flip=hashlib.sha256(('texture-order-v1|'+digest).encode()).digest()[0]&1
    sign=1-2*flip
    score=float((sign*np.asarray(vector)[fitted['columns']]/fitted['scale'])@fitted['beta'])
    vote=1 if score>1e-10 else -1 if score < -1e-10 else 0
    return dict(flip=flip,score=score,vote=vote*sign,exchanged=bool(flip^(vote<0)))

def bins(signed,nf):
    out=[]
    for offset in range(30):
        rows=[[0,0] for _ in range((nf+25*offset+749)//750)]
        for _,date,*_ in signed:
            assert isinstance(date,int) and 1<=abs(date)<=nf
            rows[(abs(date)-1+25*offset)//750][int(date<0)]+=1
        out.extend(rows)
    return out

def maximum_matching(predictions,truth,edges):
    ps={x[0] for x in predictions};gs={x[0] for x in truth};adj={p:[] for p in ps};owner={}
    for p,g in edges:assert p in ps and g in gs;adj[p].append(g)
    def augment(p,seen):
        for g in adj[p]:
            if g in seen:continue
            seen.add(g)
            if g not in owner or augment(owner[g],seen):owner[g]=p;return True
        return False
    for p in sorted(ps):augment(p,set())
    return len(owner)

def physical(arm,nf):
    graph=arm['legal_graph'];g=bins(graph['gt'],nf);p=bins(graph['predictions'],nf)
    assert g==arm['integer_bins']['gt'] and p==arm['integer_bins']['pred']
    n=int(np.abs(np.asarray(g)-p).sum());assert n==arm['N']
    assert len(graph['gt'])==arm['nGT'] and len(graph['predictions'])==arm['nPred']
    for key,edges in (('0.5','edges05'),('0.3','edges03')):
        tp=maximum_matching(graph['predictions'],graph['gt'],graph[edges]);m=arm['iou'][key]
        assert tp==m['tp'] and arm['nPred']-tp==m['fp'] and arm['nGT']-tp==m['fn']
    return n

def pool(arms,threshold='0.5'):
    # Include zero-event trials: their N and false positives still contribute.
    ng=sum(a['nGT'] for a in arms);npred=sum(a['nPred'] for a in arms)
    tp=sum(a['iou'][threshold]['tp'] for a in arms);n=sum(a['N'] for a in arms)
    return dict(N=n,GT=ng,pred=npred,TP=tp,W=100*n/(30*ng) if ng else None,F1=2*tp/(npred+ng) if npred+ng else None)

def replay_models(features,models,predictions):
    """Adapter for pair-texture-features/model/predictions-v1 JSON objects."""
    assert features['schema']=='pair-texture-features-v1'
    pairs=features['pairs'];bykey={p['clip']+'|'+p['digest']:p for p in pairs}
    assert len(bykey)==len(pairs)
    expected={k for k,p in bykey.items() if p['subset']=='new'}
    assert {p['key'] for p in predictions['pairs']}==expected
    refit={}
    for target in ('W','G'):
        source='G' if target=='W' else 'W';saved=models[target]
        assert saved['source_hive']==source
        old=sorted([p for p in pairs if p['hive']==source and p['subset']=='old' and p['primary']['valid']],key=lambda p:p['clip']+'|'+p['digest'])
        records=[dict(clip=p['clip'],difference=difference(p['endpoint_rows'],p['primary']['crops'])) for p in old]
        assert len(old)==saved['source_pairs']
        refit[target]={name:fit(records,name) for name in COLUMNS}
        for name,f in refit[target].items():
            h=saved['heads'][name]
            assert h['columns']==f['columns'] and h['ridge']==.2 and h['intercept']==0
            np.testing.assert_allclose(h['scale'],f['scale'],rtol=1e-10,atol=1e-11)
            np.testing.assert_allclose(h['beta'],f['beta'],rtol=1e-9,atol=1e-10)
    votes=[]
    for record in predictions['pairs']:
        p=bykey[record['key']];assert p['hive']==record['hive']
        bit=hashlib.sha256(('texture-order-v1|'+p['digest']).encode()).digest()[0]&1
        assert bit==record['order_bit'] and record['permutation']==[0,1,2+bit,3-bit]
        for variant,arm in zip(('R','Z','Q0','Q1','Q2','Q3'),[p['primary'],p['shifted']]+p['shuffles']):
            for name in COLUMNS:
                usable=name=='P' or arm['valid']
                vector=difference(p['endpoint_rows'],arm['crops'] if arm['valid'] else np.zeros((4,12)))
                choice=choose(vector,p['digest'],refit[p['hive']][name]) if usable else dict(score=0,vote=0,exchanged=bool(bit))
                saved=record['variants'][variant][name]
                assert bool(saved['valid'])==bool(usable)
                np.testing.assert_allclose(saved['score'],choice['score'],rtol=1e-8,atol=1e-10)
                decision='given' if choice['score']>1e-10 else 'cross' if choice['score'] < -1e-10 else 'tie'
                assert saved['decision']==decision and saved['physical_choice']==('cross' if decision=='cross' else 'given')
                votes.append(dict(key=record['key'],clip=p['clip'],hive=p['hive'],near_boundary=p['near_boundary'],variant=variant,model=name,vote=choice['vote'],exchanged=choice['exchanged']))
        for shuffled in p['shuffles']:
            for a,b in zip(p['primary']['crops'],shuffled['crops']):
                if a is None:assert b is None
                else:np.testing.assert_allclose(a[:6],b[:6],rtol=0,atol=1e-12)
    return votes

def summarize_votes(votes):
    results={}
    for h in ('W','G'):
        for variant in ('R','Z','Q0','Q1','Q2','Q3'):
            for name in COLUMNS:
                rows=[r for r in votes if r['hive']==h and r['variant']==variant and r['model']==name]
                clips=sorted({r['clip'] for r in rows})
                results[h,variant,name]=dict(n=len(rows),clips=len(clips),positive=sum(r['vote']>0 for r in rows),negative=sum(r['vote']<0 for r in rows),abstain=sum(r['vote']==0 for r in rows),S=float(np.mean([np.mean([r['vote'] for r in rows if r['clip']==c]) for c in clips])) if clips else None)
    return results

def replay(data):
    assert data['schema']=='pair-texture-compact-v1'
    import json
    for pair in data['features']['pairs']:
        key=pair['clip']+'|'+pair['digest']
        signatures=[hashlib.sha256(json.dumps(q,sort_keys=True,separators=(',',':')).encode()).hexdigest() for q in pair['shuffles']]
        assert data['shuffle_aggregate_signatures'][key]==signatures
    votes=replay_models(data['features'],data['models'],data['predictions'])
    records=data['physical']['results'];bykey={p['clip']+'|'+p['digest']:p for p in data['features']['pairs'] if p['subset']=='new'}
    assert set(records)==set(bykey)
    for key,r in records.items():
        assert r['hive']==bykey[key]['hive']
        for a in ('identity','cross'):physical(r[a],r['nframes'])
    result={}
    for h in ('W','G'):
        result[h]={}
        for variant in ('R','Z','Q0','Q1','Q2','Q3'):
            result[h][variant]={}
            for name in COLUMNS:
                selected=[r for r in votes if r['hive']==h and r['variant']==variant and r['model']==name]
                chosen=[records[v['key']]['cross' if v['exchanged'] else 'identity'] for v in selected]
                vote_summary=summarize_votes(selected)[h,variant,name]
                result[h][variant][name]=dict(votes=vote_summary,physical05=pool(chosen),physical03=pool(chosen,'0.3'))
    if data.get('score'):
        verify_score(data,votes)
    if data.get('impact'):
        # Posthoc hard-choice decomposition, separate from vote/S mechanism.
        import impact
        recomputed=impact.completed_impact(data['predictions'],data['physical'])
        close(recomputed,data['impact'])
    return dict(pairs=len(records),decisions=len(votes),hives=result,score_checked=bool(data.get('score')))

def close(actual,saved):
    import math
    if isinstance(actual,dict):
        for k,v in actual.items():assert k in saved;close(v,saved[k])
    elif isinstance(actual,list):
        assert len(actual)==len(saved)
        for a,b in zip(actual,saved):close(a,b)
    elif isinstance(actual,float):assert saved is not None and math.isclose(actual,saved,rel_tol=1e-9,abs_tol=1e-10)
    else:assert actual==saved

def verify_score(data,votes):
    score=data['score'];new={p['clip']+'|'+p['digest']:p for p in data['features']['pairs'] if p['subset']=='new'}
    old=[p for p in data['roster']['pairs'] if p['subset']=='old']
    ids={(p['clip'],p[f'id{n}']) for p in old for n in (1,2)}
    runs={(p['clip'],p[f'id{n}'],p[f'run{n}']) for p in old for n in (1,2)}
    lookup={(v['key'],v['variant'],v['model']):v for v in votes};physical=data['physical']['results']
    flags={}
    for key,p in new.items():
        pv=lookup[key,'R','P']['vote']
        flags[key]=dict(all=True,near_boundary=bool(p['near_boundary']),P_wrong=pv<0,P_tie=pv==0,P_wrong_or_tie=pv<=0,
            old_id_any_reuse=any((p['clip'],p[f'id{n}']) in ids for n in (1,2)),old_run_any_reuse=any((p['clip'],p[f'id{n}'],p[f'run{n}']) in runs for n in (1,2)))
    for row in score['per_pair']:
        key=row['key'];close(flags[key],row['flags'])
        close({v:{m:lookup[key,v,m]['vote'] for m in COLUMNS} for v in ('R','Z','Q0','Q1','Q2','Q3')},row['votes'])
        close({m:'cross' if lookup[key,'R',m]['exchanged'] else 'identity' for m in COLUMNS},row['actual_R_choices'])
    mechanism={};physical_gates={}
    for h in ('W','G'):
        keys=[k for k,p in new.items() if p['hive']==h];domain=score['domains'][h]
        close(dict(pairs=len(keys),clips=len({new[k]['clip'] for k in keys})),domain)
        for subset in flags[keys[0]]:
            selected=[k for k in keys if flags[k][subset]]
            for variant in ('R','Z','Q0','Q1','Q2','Q3'):
                for m in COLUMNS:
                    rows=[lookup[k,variant,m] for k in selected];clips=sorted({r['clip'] for r in rows})
                    per={c:dict(N=sum(r['clip']==c for r in rows),sum_vote=sum(r['vote'] for r in rows if r['clip']==c),S=float(np.mean([r['vote'] for r in rows if r['clip']==c]))) for c in clips}
                    saved=domain['S'][subset][variant][m]
                    stats=dict(S=float(np.mean([x['S'] for x in per.values()])) if per else None,pairs=len(rows),clips=len(clips),positive=sum(r['vote']>0 for r in rows),negative=sum(r['vote']<0 for r in rows),abstained=sum(r['vote']==0 for r in rows),missing=sum(not next(p for p in data['predictions']['pairs'] if p['key']==r['key'])['variants'][variant][m]['valid'] for r in rows),per_clip=per)
                    close(stats,saved)
            for m in COLUMNS:
                ss=[domain['S'][subset][f'Q{i}'][m]['S'] for i in range(4)]
                close(dict(S=float(np.mean(ss)) if ss[0] is not None else None),domain['S'][subset]['Q'][m])
        for m in ('identity',*COLUMNS):
            arms=[physical[k]['identity' if m=='identity' or not lookup[k,'R',m]['exchanged'] else 'cross'] for k in keys]
            p=pool(arms);q=pool(arms,'0.3')
            close(dict(pairs=len(arms),N=p['N'],nGT=p['GT'],nPred=p['pred'],TP=p['TP'],W_percent=p['W'],F1=p['F1'],zero_GT_pairs=sum(a['nGT']==0 for a in arms),iou={'0.5':dict(TP=p['TP'],F1=p['F1']),'0.3':dict(TP=q['TP'],F1=q['F1'])}),domain['physical'][m])
        s=domain['S']['all'];rt=s['R']['T']['S'];rc=s['R']['C']['S'];zt=s['Z']['T']['S'];qt=s['Q']['T']['S'];rv=s['R']['V']['S'];qv=s['Q']['V']['S']
        mechanism[h]=dict(new_pairs_at_least16=len(keys)>=16,clips_at_least3=domain['clips']>=3,T_R_minus_C_R=rt-rc,T_R_minus_T_Z=rt-zt,T_R_minus_T_Q=rt-qt,V_R_minus_V_Q=rv-qv,T_over_C=rt-rc>=.05,T_over_shift=rt-zt>=.05,T_over_shuffle=rt-qt>=.05,V_over_shuffle=rv>qv)
        c=domain['physical']['C'];t=domain['physical']['T']
        physical_gates[h]=dict(GT_at_least10=c['nGT']>=10,W_nonincrease=c['W_percent'] is not None and t['W_percent'] is not None and t['W_percent']<=c['W_percent'],W_strict_decrease=c['W_percent'] is not None and t['W_percent'] is not None and t['W_percent']<c['W_percent'],F1_guard=c['F1'] is not None and t['F1'] is not None and t['F1']>=c['F1']-.005)
    mp=all(all(r[k] for k in ('new_pairs_at_least16','clips_at_least3','T_over_C','T_over_shift','T_over_shuffle','V_over_shuffle')) for r in mechanism.values())
    pp=all(all(r[k] for k in ('GT_at_least10','W_nonincrease','F1_guard')) for r in physical_gates.values()) and any(r['W_strict_decrease'] for r in physical_gates.values())
    close(dict(mechanism=dict(by_hive=mechanism,passed=mp),physical=dict(by_hive=physical_gates,passed=pp),both_passed=mp and pp),score['gates'])

def self_test():
    rows=[[1,1,0,0,4,4],[1,2,6,0,4,4],[9,1,1,0,4,4],[9,2,5,0,4,4]]
    desc=np.arange(48,dtype=float).reshape(4,12)/48
    v=difference(rows,desc);records=[dict(clip='a',difference=v),dict(clip='b',difference=-v)]
    for name in COLUMNS:
        f=fit(records,name);a=choose(v,'0'*64,f);assert len(f['columns'])==len(COLUMNS[name]) and a['vote']==0
    assert maximum_matching([[1,1],[2,2]],[[3,1],[4,2]],[[1,3],[1,4],[2,3]])==2
    assert np.asarray(bins([[1,1],[2,-75]],75)).sum()==60
    z=dict(nGT=0,nPred=1,N=30,iou={'0.5':dict(tp=0)})
    assert pool([z])==dict(N=30,GT=0,pred=1,TP=0,W=None,F1=0)
    return dict(synthetic=True,features=True,blind_order=True,matching=True,zero_event_pool=True)

if __name__=='__main__':
    import argparse,gzip,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('evidence',type=Path,nargs='?');a=parser.parse_args()
    if a.evidence:
        op=gzip.open if a.evidence.suffix=='.gz' else open
        with op(a.evidence,'rt',encoding='utf-8') as f:data=json.load(f)
        r=replay(data);print(json.dumps(dict(pairs=r['pairs'],decisions=r['decisions'],score_checked=r['score_checked'],gates=data['score']['gates'] if data.get('score') else None)))
    else:print(json.dumps(self_test()))
