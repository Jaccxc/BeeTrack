"""Small independent208 audit primitives; no field, image or tensor IO on import.

Exact adjacency classification and frame-resolution timing are separate.
Unrepresented, interval and unknown predictions remain burden, not false labels.
"""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction
from itertools import combinations
import json
import math
from numbers import Integral

HIVES=('35','33','W','G')


def calibration_threshold(source_negative_probabilities):
    """Registered pooled source-negative q.999, order statistic 'higher'."""
    scores=sorted(float(x) for x in source_negative_probabilities)
    assert all(math.isfinite(x) and 0<=x<=1 for x in scores)
    if not scores:return 1.
    index=(999*(len(scores)-1)+999)//1000
    return max(.5,scores[index])


def average_precision(labels,scores):
    """Threshold-group AP: equal-score positives are never tie-broken by ID."""
    pairs=sorted(zip(scores,labels),key=lambda x:-x[0]);positives=sum(labels)
    assert len(labels)==len(scores) and all(y in (0,1) for y in labels)
    assert all(math.isfinite(p) and 0<=p<=1 for p in scores)
    if not positives:return None
    tp=fp=0;area=Fraction();position=0
    while position<len(pairs):
        end=position+1
        while end<len(pairs) and pairs[end][0]==pairs[position][0]:end+=1
        increment=sum(y for _,y in pairs[position:end]);tp+=increment;fp+=(end-position)-increment
        area+=Fraction(increment,positives)*Fraction(tp,tp+fp);position=end
    return area


def binary_diagnostics(rows,threshold):
    """Rows use normalized kind/length/represented/valid/score fields."""
    assert math.isfinite(threshold) and 0<=threshold<=1
    counts=Counter();known=[]
    for row in rows:
        counts['all_candidates']+=1;kind=row['kind'];counts['truth_kind:'+kind]+=1
        exact=row['length']==1 and kind in ('exact_switch','exact_survival')
        if exact:counts['all_known_exact_positive' if kind=='exact_switch' else 'all_known_exact_same']+=1
        if not row['represented']:
            counts['unrepresented']+=1
            if exact:counts['unrepresented_exact_positive' if kind=='exact_switch' else 'unrepresented_exact_same']+=1
            continue
        if not row['valid'] or not math.isfinite(row['score']):
            counts['invalid_or_missing_score']+=1
            if exact:counts['unscored_represented_exact_positive' if kind=='exact_switch' else 'unscored_represented_exact_same']+=1
            continue
        score=float(row['score']);assert 0<=score<=1
        counts['all_valid_predictions']+=1;counts['all_predicted_positive']+=score>threshold
        if exact:
            y=int(kind=='exact_switch');known.append((y,score));counts['exact_positive' if y else 'exact_same']+=1
            counts['TP' if y else 'FP']+=score>threshold;counts['FN' if y else 'TN']+=score<=threshold
        else:counts['interval_or_unknown_valid']+=1
    n=len(known);positives=sum(y for y,_ in known);ap=average_precision([y for y,_ in known],[p for _,p in known])
    return dict(threshold=threshold,strict_greater=True,counts=dict(counts),exact_base_rate=[positives,n] if n else None,
        all_known_exact_base_rate=[counts['all_known_exact_positive'],counts['all_known_exact_positive']+counts['all_known_exact_same']],
        pipeline_FN=counts['all_known_exact_positive']-counts['TP'],pipeline_TN=counts['all_known_exact_same']-counts['FP'],
        exact_prediction_rate=[counts['TP']+counts['FP'],n] if n else None,
        full_valid_prediction_rate=[counts['all_predicted_positive'],counts['all_valid_predictions']] if counts['all_valid_predictions'] else None,
        AP_fraction=[ap.numerator,ap.denominator] if ap is not None else None,AP_scope='Represented valid finite-score L1 adjacency only; unscored known positives remain pipeline FN, not silently removed.',
        recall=[counts['TP'],positives] if positives else None,precision=[counts['TP'],counts['TP']+counts['FP']] if counts['TP']+counts['FP'] else None)


def maximum_matching(truth,predictions,tolerance=1):
    """Same original K run only; maximum cardinality rather than nearest-first."""
    assert len({r['key'] for r in truth})==len(truth) and len({r['key'] for r in predictions})==len(predictions)
    adjacency={p['key']:[t['key'] for t in truth if t['run_key']==p['run_key'] and abs(t['frame']-p['frame'])<=tolerance] for p in predictions}
    owner={}
    def visit(pred,seen):
        for target in adjacency[pred]:
            if target in seen:continue
            seen.add(target)
            if target not in owner or visit(owner[target],seen):owner[target]=pred;return True
        return False
    for pred in sorted(adjacency):visit(pred,set())
    return sorted((pred,target) for target,pred in owner.items())


def timing_diagnostics(rows,threshold):
    truths=[];predictions=[];known_date=set();counts=Counter()
    for row in rows:
        counts['all_candidates']+=1
        usable=row['represented'] and row['valid'] and math.isfinite(row['score'])
        date_exact=row['length']==1 and row['delta_frame']==1 and row['kind'] in ('exact_switch','exact_survival')
        if date_exact and row['kind']=='exact_switch':
            counts['all_frame_exact_positive']+=1
            if not row['represented']:counts['unrepresented_frame_exact_positive']+=1
            elif not usable:counts['unscored_represented_frame_exact_positive']+=1
        if row['represented'] and date_exact:
            known_date.add(row['key'])
            if row['kind']=='exact_switch':truths.append(dict(key=row['key'],run_key=row['run_key'],frame=row['right_frame']))
        if usable and row['score']>threshold:predictions.append(dict(key=row['key'],run_key=row['run_key'],frame=row['right_frame']))
        if row['kind']=='exact_switch' and row['length']==1 and row['delta_frame'] in (2,3):counts['adjacency_exact_but_not_frame_exact']+=1
    matching=maximum_matching(truths,predictions,1);matched={p for p,t in matching};truthmatched={t for p,t in matching}
    known_unmatched=[p['key'] for p in predictions if p['key'] not in matched and p['key'] in known_date]
    ungraded=[p['key'] for p in predictions if p['key'] not in matched and p['key'] not in known_date]
    return dict(exact_date_truth=len(truths),all_valid_predicted_cuts=len(predictions),matched=len(matching),FN=len(truths)-len(truthmatched),
        known_date_unmatched_predictions=known_unmatched,unconfirmed_unmatched_predictions=ungraded,matching=matching,counts=dict(counts),
        scope='L1+delta_frame1 positives at right-frame dates; same original K run, one-to-one max matching +/-1. Unmatched interval/unknown/gap2or3 predictions are not false positives.')


def source_fold_guard(heldout,source_domains,source_records,negative_pool):
    allowed=set(HIVES)-{heldout};assert heldout in HIVES and set(source_domains)==allowed
    assert all(r['hive'] in allowed for r in source_records)
    assert all(r['hive'] in allowed and r['kind']=='exact_survival' and r['length']==1 and r['valid'] for r in negative_pool)
    return calibration_threshold([r['score'] for r in negative_pool])


def audit_same_rows(base,output):
    assert len(base)==len(output)
    for old,new in zip(base,output):
        assert len(old)==len(new)==7 and old[0]==new[0] and tuple(old[2:])==tuple(new[2:])
        assert int(new[1])==new[1] and new[1]>=0
    assert len({(row[0],row[1]) for row in output})==len(output)
    return True


def pooled_physical(clips):
    numerator=sum(c['numerator'] for c in clips);denominator=sum(c['denominator'] for c in clips)
    tp=sum(c['TP'] for c in clips);pred=sum(c['Pred'] for c in clips);truth=sum(c['GT'] for c in clips)
    assert denominator==30*truth and denominator>0
    W=Fraction(100*numerator,denominator);F1=Fraction(2*tp,pred+truth) if pred+truth else Fraction()
    return dict(W_fraction=[W.numerator,W.denominator],F1_fraction=[F1.numerator,F1.denominator],TP=tp,Pred=pred,GT=truth,
        scope='One physical member per clip; pool clips before W/F1. Not a three-member E3.')


def candidate_table(rows,canonical,indices):
    assert len(rows)==len(canonical) and len(set((r[0],r[1]) for r in rows))==len(rows)
    runs=defaultdict(list)
    for i,parent in enumerate(canonical):runs[int(parent)].append(i)
    position={}
    for parent,ii in runs.items():
        ii.sort(key=lambda i:(rows[i][0],i))
        assert all(rows[b][0]>rows[a][0] for a,b in zip(ii,ii[1:]))
        for j,i in enumerate(ii):position[i]=j
    table=[];seen=set()
    for number,quad in enumerate(indices):
        assert len(quad)==4 and len(set(quad))==4
        assert all(isinstance(i,Integral) and not isinstance(i,bool) and 0<=i<len(rows) for i in quad)
        parent=int(canonical[quad[0]]);ordinals=[position[i] for i in quad]
        assert all(canonical[i]==parent for i in quad) and ordinals==list(range(ordinals[0],ordinals[0]+4))
        assert all(1<=rows[b][0]-rows[a][0]<=3 for a,b in zip(quad,quad[1:]))
        key=parent,ordinals[2];assert key not in seen;seen.add(key)
        table.append(dict(candidate=number,parent=parent,right_ordinal=ordinals[2],right_frame=int(rows[quad[2]][0])))
    return runs,table


def adjacent_census(rows,canonical,indices,valid,owners,clip):
    """All original adjacency labels, including no input and invalid crop."""
    runs,_=candidate_table(rows,canonical,indices);represented={(int(q[1]),int(q[2])):i for i,q in enumerate(indices)};records=[]
    for parent,ii in sorted(runs.items()):
        for left,right in zip(ii,ii[1:]):
            a,b=owners[left],owners[right];kind='unknown'
            if min(a)>=0 and min(b)>=0:
                if a[0]!=b[0]:kind='exact_switch'
                elif a[1]==b[1]:kind='exact_survival'
            ordinal=represented.get((left,right))
            records.append(dict(key=f'{clip}:{parent}:{left}:{right}',run_key=(clip,parent),parent=parent,left_raw=left,right_raw=right,
                left_frame=int(rows[left][0]),right_frame=int(rows[right][0]),delta_frame=int(rows[right][0]-rows[left][0]),length=1,kind=kind,
                input_ordinal=ordinal,represented=ordinal is not None,valid=bool(valid[ordinal]) if ordinal is not None else False))
    return records


def independent_bags(rows,canonical,indices,valid,owners):
    runs,_=candidate_table(rows,canonical,indices);represented={(int(q[1]),int(q[2])):i for i,q in enumerate(indices)};bags=[];excluded=Counter()
    for parent,ii in sorted(runs.items()):
        known=[position for position,i in enumerate(ii) if min(owners[i])>=0]
        for a,b in zip(known,known[1:]):
            length=b-a;left,right=ii[a],ii[b];x,y=owners[left],owners[right]
            if length>3:excluded['more_than_three_gaps']+=1;continue
            if x[0]==y[0]:
                if x[1]!=y[1]:excluded['same_ID_different_GT_run']+=1;continue
                if length!=1:excluded['same_ID_across_unknown']+=1;continue
                target=0
            else:target=1
            cutids=[represented.get((ii[j],ii[j+1])) for j in range(a,b)]
            if any(i is None for i in cutids):excluded['missing_input']+=1;continue
            if not all(valid[i] for i in cutids):excluded['invalid_input']+=1;continue
            bags.append(dict(parent=parent,anchors=[left,right],X=cutids+[-1]*(3-length),y=target,length=length,
                kind='exact_switch' if target and length==1 else 'interval_switch' if target else 'exact_survival'))
    return bags,dict(excluded)


def selected_cuts(rows,canonical,indices,probabilities,threshold,valid,active=True):
    _,table=candidate_table(rows,canonical,indices);assert len(table)==len(probabilities)==len(valid)
    assert .5<=threshold<=1 and math.isfinite(threshold)
    if not active:return [],[]
    proposed=[]
    for record,p,ok in zip(table,probabilities,valid):
        if ok:
            assert math.isfinite(p) and 0<=p<=1
            if p>threshold:proposed.append(record)
    proposed.sort(key=lambda r:(-probabilities[r['candidate']],r['right_frame'],r['candidate']))
    chosen=[];suppressed=[]
    for record in proposed:
        conflict=[r['candidate'] for r in chosen if r['parent']==record['parent'] and abs(r['right_frame']-record['right_frame'])<=3]
        if conflict:suppressed.append(dict(candidate=record['candidate'],by=conflict))
        else:chosen.append(record)
    return sorted(r['candidate'] for r in chosen),suppressed


def displaced_scores(rows,canonical,indices,probabilities,valid):
    _,table=candidate_table(rows,canonical,indices);output=list(probabilities);byparent=defaultdict(list)
    for record,ok in zip(table,valid):
        if ok:byparent[record['parent']].append(record)
    audit=[]
    for parent,values in sorted(byparent.items()):
        values.sort(key=lambda r:r['right_ordinal']);stretches=[]
        for r in values:
            if not stretches or r['right_ordinal']!=stretches[-1][-1]['right_ordinal']+1:stretches.append([])
            stretches[-1].append(r)
        for stretch in stretches:
            ids=[r['candidate'] for r in stretch];sources=ids[-1:]+ids[:-1]
            for dst,src in zip(ids,sources):output[dst]=probabilities[src]
            audit.append(dict(parent=parent,candidates=ids,source_candidates=sources))
    return output,audit


def split_rows(rows,canonical,indices,selected):
    runs,table=candidate_table(rows,canonical,indices);assert len(selected)==len(set(selected)) and all(isinstance(i,Integral) and not isinstance(i,bool) and 0<=i<len(table) for i in selected)
    boundaries=defaultdict(set)
    for i in selected:boundaries[table[i]['parent']].add(table[i]['right_ordinal'])
    output=[list(r) for r in rows];segments=[]
    for parent,ii in sorted(runs.items()):
        limits=[0,*sorted(boundaries[parent]),len(ii)]
        for lo,hi in zip(limits,limits[1:]):
            record=dict(id=len(segments)+1,parent=parent,rows=ii[lo:hi],new_start=lo in boundaries[parent],new_end=hi in boundaries[parent])
            segments.append(record)
            if selected:
                for i in record['rows']:output[i][1]=record['id']
    audit_same_rows(rows,output);return output,segments


def relink(cut,segments):
    import numpy as np
    byid={s['id']:s for s in segments};starts=defaultdict(list);edges=[]
    for s in segments:starts[int(cut[s['rows'][0]][0])].append(s)
    for left in segments:
        a=cut[left['rows'][-1]];end=int(a[0])
        for date in range(end+1,end+4):
            for right in starts[date]:
                if left['parent']==right['parent'] or not(left['new_end'] or right['new_start']):continue
                b=cut[right['rows'][0]];scale=(float(np.hypot(a[4],a[5]))+float(np.hypot(b[4],b[5])))/2
                ac=np.asarray(a[2:4])+np.asarray(a[4:6])/2;bc=np.asarray(b[2:4])+np.asarray(b[4:6])/2
                distance=float(np.linalg.norm(ac-bc)/scale)
                if distance<=.75:edges.append((left['id'],right['id'],distance))
    outgoing=defaultdict(list);incoming=defaultdict(list)
    for a,b,d in edges:outgoing[a].append((d,b));incoming[b].append((d,a))
    def unique(values):
        values=sorted(values)
        return values[0] if len(values)==1 or values[0][0]<values[1][0] else None
    forward={a:unique(values) for a,values in outgoing.items()};backward={b:unique(values) for b,values in incoming.items()}
    links={a:b for a,b,d in edges if forward[a]==(d,b) and backward[b]==(d,a)}
    assert len(set(links.values()))==len(links)
    output=[list(r) for r in cut];paths=[];visited=set()
    for start in sorted(set(byid)-set(links.values())):
        path=[];current=start
        while True:
            assert current not in visited;visited.add(current);path.append(current)
            if current not in links:break
            current=links[current]
        dates=[cut[i][0] for sid in path for i in byid[sid]['rows']];assert len(set(dates))==len(dates)
        if links:
            for sid in path:
                for i in byid[sid]['rows']:output[i][1]=min(path)
        paths.append(path)
    assert visited==set(byid)
    returns=[p for p in paths if len({byid[s]['parent'] for s in p})<len(p)]
    audit_same_rows(cut,output)
    return output,dict(links=sorted(links.items()),indirect_original_run_return_paths=returns)


def duplicate_bee_state(left_index,right_index,rows,canonical,owners,GT_presence):
    """Descriptive evidence only; absence of confirmed other track is not unilateral."""
    left,right=owners[left_index],owners[right_index]
    assert min(left)>=0 and min(right)>=0 and left[0]!=right[0] and canonical[left_index]==canonical[right_index]
    positions=sorted((i for i in range(len(rows)) if canonical[i]==canonical[left_index]),key=lambda i:(rows[i][0],i))
    assert positions.index(right_index)==positions.index(left_index)+1
    date=rows[left_index][0]
    presence=GT_presence(date,right[0])
    if presence!='present_known':return dict(state='unknown',reason=presence,other_rows=[])
    other=[i for i,row in enumerate(rows) if row[0]==date and canonical[i]!=canonical[left_index] and owners[i][0]==right[0]]
    return dict(state='yes' if other else 'not_confirmed_other_track',reason='mutual_unique_other_run' if other else 'no_confirmed_other_run',other_rows=other,other_GT_run_aliases=[owners[i][1] for i in other])


def self_test():
    assert calibration_threshold([])==1 and calibration_threshold([.1,.2,.7])==.7
    assert average_precision([1,0,1],[.8,.8,.2])==Fraction(7,12)
    truth=[dict(key='t0',run_key='one',frame=100),dict(key='t1',run_key='one',frame=102)]
    predicted=[dict(key='p0',run_key='one',frame=101),dict(key='p1',run_key='one',frame=103),dict(key='p2',run_key='other',frame=100)]
    assert len(maximum_matching(truth,predicted))==2
    def row(key,kind,frame,p=.8,delta=1,length=1):return dict(key=key,run_key='one',kind=kind,right_frame=frame,score=p,delta_frame=delta,length=length,represented=True,valid=True)
    rows=[row('positive','exact_switch',100),row('same','exact_survival',104,.5),row('interval','interval_switch',110,.9,length=3),row('gap3','exact_switch',120,.9,delta=3)]
    d=binary_diagnostics(rows,.5);assert d['counts']['TP']==2 and d['counts']['FP']==0 and d['counts']['TN']==1 and d['counts']['interval_or_unknown_valid']==1
    timing=timing_diagnostics(rows,.5);assert timing['exact_date_truth']==1 and timing['matched']==1 and len(timing['unconfirmed_unmatched_predictions'])==2
    invalid=row('invalid','exact_switch',130,float('nan'));invalid['valid']=False
    absent=row('absent','exact_switch',140,float('nan'));absent.update(valid=False,represented=False)
    full=binary_diagnostics(rows+[invalid,absent],.5);assert full['pipeline_FN']==2 and full['counts']['exact_positive']==2
    timing=timing_diagnostics(rows+[invalid,absent],.5);assert timing['exact_date_truth']==2 and timing['FN']==1 and timing['counts']['unrepresented_frame_exact_positive']==1
    base=[[1,1,0,0,2,2,.8],[2,1,1,0,2,2,.9]];assert audit_same_rows(base,[base[0],list(base[1][:1])+[2]+base[1][2:]])
    try:source_fold_guard('35',['33','W','G'],[{'hive':'35'}],[])
    except AssertionError:pass
    else:raise AssertionError('target source leakage accepted')
    R=[[1,1],[2,1],[1,2]];C=[1,1,2];O=[(1,0),(2,0),(2,0)]
    assert duplicate_bee_state(0,1,R,C,O,lambda *args:'present_known')['state']=='yes'
    assert duplicate_bee_state(0,1,R,C,O,lambda *args:'missing_GT_annotation')['state']=='unknown'
    pooled=pooled_physical([dict(numerator=30,denominator=30,TP=1,Pred=1,GT=1),dict(numerator=60,denominator=0,TP=0,Pred=3,GT=0)])
    assert pooled['W_fraction']==[300,1] and pooled['F1_fraction']==[2,5]
    return dict(passed=True,source_only=True,AP_ties=True,strict_threshold=True,frame_vs_adjacency_exact=True,max_cardinality_timing=True,unknown_not_negative=True,source_guard=True)


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--field-go',action='store_true');p.add_argument('--physical-finished',action='store_true')
    p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--root',type=Path,default=Path('D:/beetrack_runs/switch208'));p.add_argument('--source206',type=Path,default=Path('D:/beetrack_runs/return_query206/frozen'));p.add_argument('--reference206-score',type=Path,default=Path('D:/beetrack_runs/return_query206/score'))
    for name,default in [('inputs','1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94'),('labels','e73ba698916d5e60fda6289b04f7a9d53d75ef1767621250370bf9bb299a473a'),('predictions','aed515e2a35b365d5fc48f2e132eabd7b41539b9cbda9ab1a5685b8dcd8a1dcb'),('score','10e1a746d0a01b71d8f5b5aa1a9c1e15bc5572c354e1f8fb34d43b5ac2d5c895')]:p.add_argument('--'+name+'-sha',default=default)
    args=p.parse_args()
    if args.audit:
        assert args.field_go and args.physical_finished,'Coordinator GO and physical worker exit required'
        import audit208
        result=audit208.execute(args)
    else:result=self_test()
    print(json.dumps(result,indent=2))
