"""Incremental206 integer replay; no native/GT/pixel IO on import."""
from collections import defaultdict, Counter
from fractions import Fraction

THRESHOLDS=('0.5','0.3')

def bins(events,nf):
    assert type(nf)is int and nf>0
    result=[]
    for offset in range(30):
        part=[[0,0]for _ in range((nf+25*offset+749)//750)]
        for frame,direction in events:
            assert type(frame)is int and 1<=frame<=nf and direction in (-1,1)
            part[(frame-1+25*offset)//750][int(direction<0)]+=1
        result.extend(part)
    return result

def flatten(matrix):
    assert all(len(row)==2 and all(type(v)is int for v in row)for row in matrix)
    return [v for row in matrix for v in row]

def residual(unit):
    pred,gt=flatten(unit['pred_sum']),flatten(unit['gt_bins'])
    assert len(pred)==len(gt)==len(flatten(bins([],unit['nf'])))
    assert all(v>=0 for v in pred+gt)and sum(gt)%30==0
    return [p-3*g for p,g in zip(pred,gt)]

def all_queries(unit):
    return [(m,q)for m in unit['members']for q in m['queries']]

def bound(unit,threshold=None,eligible=False):
    r=residual(unit);capacity=[0]*len(r);ids=[]
    for m,q in all_queries(unit):
        if eligible and not q['eligible']:continue
        label=None if threshold is None else q['labels'][threshold]
        if label is not None and not label['confirmed']:continue
        col=flatten(bins([(q['event']['frame'],q['event']['direction'])],unit['nf']))
        if label is not None:
            candidate=label['candidate'];assert candidate['event_delete_only']
            assert flatten(candidate['delta_windows'])==[-v for v in col]
        capacity=[a+b for a,b in zip(capacity,col)];ids.append(q['query_id'])
    assert len(ids)==len(set(ids))
    return dict(query_ids=ids,query_count=len(ids),baseline_numerator=sum(map(abs,r)),
                gain_upper_numerator=sum(min(max(v,0),c)for v,c in zip(r,capacity)),
                denominator=3*sum(flatten(unit['gt_bins'])))

def aggregate_bound(units,reports):
    groups=defaultdict(lambda:[0,0])
    for unit in units:
        r=reports[unit['unit']];v=groups[unit['domain'],unit['group']]
        v[0]+=r['gain_upper_numerator'];v[1]+=r['denominator']
    out=defaultdict(list)
    for (domain,group),(num,den)in groups.items():
        assert den>0;out[domain].append(Fraction(100*num,den))
    return {h:sum(values,Fraction())/len(values)for h,values in out.items()}

def replay_greedy(units,threshold,saved,joint_audits):
    table={u['unit']:u for u in units};r={k:residual(v)for k,v in table.items()};remaining={};selected={k:[]for k in table};used=set();steps=[];rejected=[]
    assert len({(u['domain'],u['group'])for u in units})==1
    for u in units:
        for m,q in all_queries(u):
            label=q['labels'][threshold]
            if not label['confirmed']:continue
            c=label['candidate'];delta=flatten(c['delta_windows']);expected=flatten(bins([(q['event']['frame'],q['event']['direction'])],u['nf']))
            assert delta==[-v for v in expected];key=q['query_id'];assert key not in remaining
            remaining[key]=(u['unit'],str(m['member']),q,c,delta)
    baseline=sum(sum(map(abs,v))for v in r.values())
    while remaining:
        ranked=[]
        for key,(unit,member,q,c,delta)in remaining.items():
            gain=sum(map(abs,r[unit]))-sum(abs(a+b)for a,b in zip(r[unit],delta))
            if gain>0:ranked.append((-gain,key))
        if not ranked:break
        neg,key=min(ranked);unit,member,q,c,delta=remaining.pop(key);resource=(member,tuple(c['GT_row_key']))
        state=tuple(sorted(selected[unit]+[key]))
        safe=resource not in used and joint_audits[unit,state]['safe']
        if not safe:rejected.append(key);continue
        before=sum(map(abs,r[unit]));r[unit]=[a+b for a,b in zip(r[unit],delta)];after=sum(map(abs,r[unit]))
        selected[unit].append(key);used.add(resource);steps.append((key,-neg,before,after))
    assert {k:sorted(v)for k,v in selected.items()}==saved['selected']
    assert [x[0]for x in steps]==[s['query_id']for s in saved['steps']]
    assert rejected==[s['query_id']for s in saved['rejected']]
    for actual,s in zip(steps,saved['steps']):assert actual==(s['query_id'],s['gain'],s['before'],s['after'])
    final=sum(sum(map(abs,v))for v in r.values());assert baseline-final==saved['gain_numerator']
    assert baseline==saved['baseline_numerator']and final==saved['final_numerator']
    for key,values in r.items():assert values==[p-3*g for p,g in zip(flatten(saved['witness_pred_sum'][key]),flatten(table[key]['gt_bins']))]
    return dict(steps=len(steps),rejected=len(rejected),gain_numerator=baseline-final)

def maximum_matching(edges):
    graph=defaultdict(list)
    for p,g in edges:graph[p].append(g)
    match={}
    def augment(p,seen):
        for g in graph[p]:
            if g in seen:continue
            seen.add(g)
            if g not in match or augment(match[g],seen):match[g]=p;return True
        return False
    return sum(augment(p,set())for p in graph)

def self_test():
    nf=30;gt=bins([(2,1)],nf);pred=[[3*a,3*b]for a,b in gt]
    unit=dict(unit='toy',domain='35',group=1,nf=nf,pred_sum=pred,gt_bins=gt,members=[dict(member='0',queries=[dict(query_id='q',event=dict(frame=2,direction=1),eligible=False,labels={})])])
    assert bound(unit)['gain_upper_numerator']==0 and bound(unit,eligible=True)['query_count']==0
    assert len(bins([],nf))==sum((nf+25*o+749)//750 for o in range(30)) and sum(flatten(gt))==30
    assert maximum_matching([(0,0),(0,1),(1,0)])==2
    column=bins([(2,1)],nf)
    c=dict(event_delete_only=True,delta_windows=[[-v for v in row]for row in column],GT_row_key=['toy',99,0,4,0])
    q=unit['members'][0]['queries'][0];q['eligible']=True;q['labels']={'0.5':dict(confirmed=True,candidate=c)}
    unit['pred_sum']=[[4*a,4*b]for a,b in gt]
    saved=dict(selected={'toy':['q']},steps=[dict(query_id='q',gain=30,before=30,after=0)],rejected=[],
               baseline_numerator=30,final_numerator=0,gain_numerator=30,witness_pred_sum={'toy':pred})
    assert bound(unit,'0.5')['gain_upper_numerator']==30
    assert replay_greedy([unit],'0.5',saved,{('toy',('q',)):dict(safe=True)})['steps']==1
    denied=dict(selected={'toy':[]},steps=[],rejected=[dict(query_id='q')],baseline_numerator=30,
                final_numerator=30,gain_numerator=0,witness_pred_sum={'toy':unit['pred_sum']})
    assert replay_greedy([unit],'0.5',denied,{('toy',('q',)):dict(safe=False)})['rejected']==1
    try:bins([(31,1)],nf)
    except AssertionError:pass
    else:raise AssertionError('invalid clock accepted')
    return 'PASS synthetic integer bins/bounds, eligibility denominator, augmenting matching; no field'
