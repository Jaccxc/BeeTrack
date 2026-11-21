"""Independent return-query reachability replay; no field IO on import.

This module does not import the production query, owner, or score functions.
Field adapters remain gated until final score exit and the coordinator's GO.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(1)


def array_sha(a):
    a=np.ascontiguousarray(a,dtype='<f8')
    return hashlib.sha256(json.dumps(list(a.shape)).encode()+b'\n'+a.tobytes()).hexdigest()


def signed_events(rows,polygon):
    """Official net event, dated at the last crossing in its net direction."""
    a=np.asarray(rows,float);p=np.asarray(polygon,np.float32);result=[]
    for pid in sorted(set(a[:,1])):
        indices=np.flatnonzero(a[:,1]==pid);indices=indices[np.argsort(a[indices,0],kind='stable')]
        for run,ii in enumerate(np.split(indices,np.flatnonzero(np.diff(a[indices,0])>25)+1)):
            rr=a[ii];inside=[cv2.pointPolygonTest(p,(float(x[2]+x[4]/2),float(x[3]+x[5]/2)),False)>=0 for x in rr]
            if inside[0]==inside[-1]:continue
            direction=1 if inside[-1] else -1
            dates=[int(rr[j,0]) for j in range(1,len(rr)) if inside[j]!=inside[j-1] and inside[j]==inside[-1]]
            result.append((int(pid),run,ii,inside,int(dates[-1]),direction))
    return result


def query_roster(rows,sources,polygon,nf,unit,slot):
    a=np.asarray(rows,float);keys=[tuple(x) for x in sources]
    assert a.ndim==2 and a.shape[1]==7 and np.isfinite(a).all() and (a[:,4:6]>0).all()
    assert np.array_equal(a[:,:2],np.floor(a[:,:2])) and (a[:,0]>=1).all() and (a[:,0]<=nf).all() and (a[:,1]>=0).all()
    assert len(set(map(tuple,a[:,:2])))==len(a)==len(keys)==len(set(keys))
    aliases=set()
    for pid in sorted(set(a[:,1])):
        dates=np.sort(a[a[:,1]==pid,0])
        for r in range(1+sum(np.diff(dates)>25)):
            alias=int(pid+100000*r);assert alias<2**53 and alias not in aliases;aliases.add(alias)
    out=[]
    for pid,run,ii,sides,date,direction in signed_events(a,polygon):
        rr=a[ii];f=int(rr[-1,0]);why=[]
        if len(ii)<3:why.append('fewer_than_three_distinct_anchor_observations')
        if len(ii)<2 or rr[-1,0]-rr[-2,0]!=1:why.append('last_two_dates_not_consecutive')
        dates=list(range(f+1,min(f+25,nf)+1))
        if not dates:why.append('empty_clipped_search_range')
        anchors=[int(ii[i]) for i in dict.fromkeys((0,max(0,len(ii)-2),len(ii)-1))]
        length=float(np.median(np.hypot(rr[-2:,4],rr[-2:,5])))
        out.append(dict(query_id=f'{unit}|{slot}|{pid}|{run}|{f}',unit=unit,member_index=slot,native_id=pid,run_index=run,canonical_id=pid+100000*run,
            eligible=not why,ineligible_reasons=why,raw_run_indices=ii.tolist(),run_source_keys=[list(keys[i]) for i in ii],anchor_raw_indices=anchors,
            anchor_rows=a[anchors].tolist(),anchor_source_keys=[list(keys[i]) for i in anchors],event=dict(frame=date,direction=direction),endpoint_frame=f,
            search_frames=dates,search_first=f+1 if dates else None,search_last=dates[-1] if dates else None,L=length,
            center=(rr[-1,2:4]+rr[-1,4:6]/2).tolist(),radius=4*length,boundary_inclusive=True,starting_side=bool(sides[0]),raw_K_sha256=array_sha(a),immutable_native_ID=True))
    return sorted(out,key=lambda q:q['canonical_id'])


def in_fixed_search(query,row):
    row=np.asarray(row,float)
    return bool(row.shape==(6,) and np.isfinite(row).all() and (row[4:6]>0).all() and int(row[0])==row[0] and int(row[0]) in query['search_frames']
        and float(np.linalg.norm(row[2:4]+row[4:6]/2-np.asarray(query['center'])))<=query['radius'])


def native_membership(rows):
    a=np.asarray(rows,float);membership={};aliases=set()
    assert len(set(map(tuple,a[:,:2])))==len(a)
    for pid in sorted(set(a[:,1])):
        ii=np.flatnonzero(a[:,1]==pid);ii=ii[np.argsort(a[ii,0],kind='stable')]
        for run,part in enumerate(np.split(ii,np.flatnonzero(np.diff(a[ii,0])>25)+1)):
            alias=int(pid+100000*run);assert alias not in aliases and alias<2**53;aliases.add(alias)
            for j in part:membership[int(j)]=alias
    return membership


def insert_native(rows,query,GT_row,polygon,nf):
    """Physical/run guard only; same-bee/full-frame confirmation is separate."""
    a=np.asarray(rows,float);g=np.asarray(GT_row,float)
    assert array_sha(a)==query['raw_K_sha256']
    if not query['eligible'] or not in_fixed_search(query,g):return None,'ineligible_or_outside_search'
    poly=np.asarray(polygon,np.float32)
    side=cv2.pointPolygonTest(poly,tuple((g[2:4]+g[4:6]/2).tolist()),False)>=0
    if side!=query['starting_side']:return None,'wrong_return_side'
    row=np.r_[g[0],query['native_id'],g[2:6],1.]
    if not 1<=row[0]<=nf:return None,'invalid_date'
    if any(r[0]==row[0] and r[1]==row[1] for r in a):return None,'native_date_ID_conflict'
    augmented=np.vstack((a,row))
    try:before=native_membership(a);after=native_membership(augmented)
    except AssertionError:return None,'canonical_alias_collision'
    if any(after[i]!=alias for i,alias in before.items()):return None,'other_run_membership_changed'
    if after[len(a)]!=query['canonical_id']:return None,'addition_not_in_target_run'
    old={(pid,run):(date,direction) for pid,run,_,_,date,direction in signed_events(a,polygon)}
    new={(pid,run):(date,direction) for pid,run,_,_,date,direction in signed_events(augmented,polygon)}
    key=query['native_id'],query['run_index']
    if key not in old or key in new or {k:v for k,v in old.items() if k!=key}!=new:return None,'not_exactly_one_event_retraction'
    assert np.array_equal(augmented[:len(a)],a)
    assert np.array_equal(bins([(d,s) for d,s in new.values()],nf),bins([(d,s) for d,s in old.values()],nf)-bins([(query['event']['frame'],query['event']['direction'])],nf))
    return augmented,'safe_native_single_event_retraction'


def earliest_legal(query,GT_rows,predicate):
    """Never replace the frozen first legal row during subsequent allocation."""
    order=sorted(range(len(GT_rows)),key=lambda j:(GT_rows[j][0],j));tried=[]
    for j in order:
        row=GT_rows[j]
        if not in_fixed_search(query,row):continue
        legal,reason=predicate(j,row);tried.append(dict(GT_raw_index=j,legal=bool(legal),reason=reason))
        if legal:return j,tried
    return None,tried


def bins(events,nf):
    out=[]
    for phase in range(30):
        count=np.zeros(((nf+25*phase+749)//750,2),np.int64)
        for frame,direction in events:count[(int(frame)-1+25*phase)//750,int(direction<0)]+=1
        out.extend(count.tolist())
    return np.asarray(out,np.int64)


def loose_bound(pred_sum,gt_bins,events,nf):
    """Independent per-coordinate relaxation; unknown/ineligible queries retained."""
    residual=np.asarray(pred_sum,np.int64)-3*np.asarray(gt_bins,np.int64)
    capacity=bins(events,nf);assert capacity.shape==residual.shape
    improvement=int(np.minimum(np.maximum(residual,0),capacity).sum())
    denominator=3*int(np.asarray(gt_bins).sum())
    return dict(baseline_numerator=int(abs(residual).sum()),capacity=capacity.tolist(),upper_improvement=improvement,denominator=denominator,
        gain_fraction=[100*improvement,denominator] if denominator else None)


def greedy_trace(residual,options,legal):
    """Exact integer one-step gains; legality may depend on the whole selection."""
    residual=np.asarray(residual,np.int64).copy();chosen=[];trace=[];rejected=[];remaining=dict(options)
    while remaining:
        proposals=[]
        for key,column in remaining.items():
            column=np.asarray(column,np.int64)
            gain=int(abs(residual).sum()-abs(residual-column).sum())
            if gain>0:proposals.append((-gain,key,column))
        if not proposals:break
        minus,key,column=min(proposals,key=lambda x:(x[0],x[1]));del remaining[key]
        if not legal(chosen+[key]):rejected.append(dict(query_id=key,proposed_gain=-minus));continue
        before=int(abs(residual).sum());residual-=column;chosen.append(key)
        trace.append(dict(query_id=key,integer_gain=-minus,before=before,after=int(abs(residual).sum()),selected=list(chosen)))
    return dict(selected=chosen,steps=trace,rejected=rejected,remaining_nonpositive=sorted(remaining),residual=residual.tolist(),numerator=int(abs(residual).sum()))


def pooled_head(groups):
    """Each group pools clips per physical member, then means three F1s."""
    E=[];F=[]
    for clips in groups:
        numerator=denominator=0;members=defaultdict(lambda:[0,0,0])
        for clip in clips:
            gt=np.asarray(clip['gt_bins'],np.int64);pred=sum(np.asarray(m['pred_bins'],np.int64) for m in clip['members'])
            numerator+=int(abs(pred-3*gt).sum());denominator+=3*int(gt.sum())
            for m in clip['members']:
                counts=members[m['member']]
                for j,k in enumerate(('TP','nPred','nGT')):counts[j]+=int(m[k])
        assert len(members)==3 and denominator>0
        E.append(Fraction(100*numerator,denominator));F.append(sum((Fraction(2*t,p+g) if p+g else Fraction() for t,p,g in members.values()),Fraction())/3)
    return dict(E3=sum(E,Fraction())/len(E),F1=sum(F,Fraction())/len(F))


def self_test():
    poly=[[0,-20],[50,-20],[50,20],[0,20]]
    row=lambda f,p,x:[f,p,x,0,2,2,.9]
    a=np.asarray([row(1,7,-4),row(2,7,-3),row(3,7,1),row(30,7,-4),row(31,7,-3),row(32,7,1),row(4,8,-3),row(5,8,1),row(8,9,-4),row(9,9,-3),row(11,9,1)],float)
    q=query_roster(a,[('synthetic',i) for i in range(len(a))],poly,32,'toy',0)
    assert len(q)==4 and sum(x['eligible'] for x in q)==1
    assert [x['canonical_id'] for x in q]==sorted(x['canonical_id'] for x in q)
    assert any('empty_clipped_search_range' in x['ineligible_reasons'] for x in q)
    first=next(x for x in q if x['native_id']==7 and x['run_index']==0)
    assert insert_native(a,first,[4,17,-3,0,2,2],poly,32)[0] is not None
    assert insert_native(a,first,[5,17,-3,0,2,2],poly,32)[1]=='other_run_membership_changed'
    assert bins([(749,1),(750,-1)],800).sum()==60
    r=np.asarray([[2,-1],[0,1]]);options={'b':[[1,0],[0,0]],'a':[[1,0],[0,1]],'c':[[0,0],[0,1]]}
    trace=greedy_trace(r,options,lambda ids:not ('a' in ids and 'b' in ids))
    assert trace['selected']==['a'] and trace['numerator']==2
    # Upper bound is coordinatewise, so may exceed feasible joint improvement.
    best=max(int(abs(r).sum()-abs(r-sum((np.asarray(options[k]) for k in ids),np.zeros_like(r))).sum()) for ids in ([],['a'],['b'],['c'],['b','c'],['a','c']))
    assert best<=int(np.minimum(np.maximum(r,0),sum(map(np.asarray,options.values()))).sum())
    clip=lambda p,t:dict(gt_bins=[[1,0]],members=[dict(member=i,pred_bins=[[p,0]],TP=t,nPred=p,nGT=1) for i in range(3)])
    pooled=pooled_head([[clip(1,1),clip(3,0)]])
    assert pooled['F1']==Fraction(1,3) and pooled['E3']==100
    return dict(passed=True,field_opened=False,all_events_and_ineligible=True,integer_greedy=True,relaxed_bound=True,phase_event_mass=30,pooled_member_F1=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--phase',choices=('supply','owners','counts','scores','all'));p.add_argument('--resume-after',choices=('supply','owners','counts','scores'))
    p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--root',type=Path);p.add_argument('--source203',type=Path);p.add_argument('--source203-score',type=Path)
    p.add_argument('--field-go',action='store_true');p.add_argument('--score-worker-finished',action='store_true')
    p.add_argument('--frozen-sha',default='066aaa245d78ee5d926cafeb709d200746e9ea9f736f820606270c35a2188d5d');p.add_argument('--owners-sha',default='02820baa4689ee0318b4e53227450cb8d9200a8d9a83e116f356db05229f514d');p.add_argument('--score-sha')
    args=p.parse_args()
    if args.phase:
        assert args.root and args.source203 and args.source203_score, "explicit native roots required"
        assert args.field_go and args.score_worker_finished and args.score_sha,'Final coordinator GO, scorer exit, and score seal are required'
        import sys
        sys.path.insert(0,str(args.repo.resolve()));import verify206_field as field;import verify206_scores as score
        for stage in ('frozen','owners','score'):field.seal(args.root/stage,getattr(args,stage+'_sha'))
        assert field.read(args.root/'owners/deployment.json')['prereg']==field.PREREG
        steps=dict(supply=field.supply,owners=field.owners,counts=field.counts,scores=score.check)
        if args.phase=='all':
            reports={};skip=list(steps).index(args.resume_after) if args.resume_after else -1
            for index,(name,fn) in enumerate(steps.items()):
                if index<=skip:
                    path=args.root/'verification'/(name+'.json');assert field.sha(path)==path.with_suffix('.sha256').read_text().strip();reports[name]=field.read(path);assert reports[name]['passed']
                    assert reports[name]['input_seals']=={s:field.sha(args.root/s/'manifest.json') for s in ('frozen','owners','score')}
                else:reports[name]=fn(args)
            result=dict(schema='return-query206-independent-v1',passed=True,prereg=field.PREREG,input_seals={s:field.sha(args.root/s/'manifest.json') for s in ('frozen','owners','score')},
                sources={name:field.sha(Path(__file__).with_name(name)) for name in ('verify206.py','verify206_field.py','verify206_owners.py','verify206_scores.py')},
                phases={name:dict(path=name+'.json',sha256=field.sha(args.root/'verification'/(name+'.json')),scope=r['scope'],verifier_sources=r['verifier_sources']) for name,r in reports.items()})
            field.save(args.root/'verification/independent.json',result)
        else:result=steps[args.phase](args)
    else:result=self_test()
    print(json.dumps(result,indent=2))
