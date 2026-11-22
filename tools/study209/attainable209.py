"""Prediction-only count bound for ONE fixed disjoint209 common slate.

No field loader, GT rows, model fitting or rank/gate choice. GTevents is only
the scalar count already known to the authorized scorer. J has another slate
and is outside this bound. A large upper bound is NOT an attainable gain.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from fractions import Fraction
from numbers import Integral
from pathlib import Path
import sys
import numpy as np

def integer(x):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral):raise ValueError('integer required')
    return int(x)

def window_counts(events,nframes):
    nframes=integer(nframes)
    if nframes<1:raise ValueError('positive clip length required')
    out=[]
    for shift in range(0,750,25):
        part=np.zeros(((nframes+shift+749)//750,2),np.int64)
        for frame,direction in events:
            frame,direction=integer(frame),integer(direction)
            if not 1<=frame<=nframes or direction not in (-1,1):raise ValueError('event outside full clip or invalid direction')
            part[(frame-1+shift)//750,int(direction<0)]+=1
        out.append(part)
    return np.vstack(out)

def side_event(frames,sides):
    """One official gap run: first/last sides and LAST net-direction crossing."""
    frames=np.asarray(frames);sides=np.asarray(sides,dtype=bool)
    if frames.ndim!=1 or sides.shape!=frames.shape or not len(frames):raise ValueError('nonempty side sequence required')
    if not np.array_equal(frames,np.rint(frames)) or np.any(np.diff(frames)<=0) or np.any(np.diff(frames)>25):
        raise ValueError('unique chronological dates in ONE <=25-gap run required')
    if sides[0]==sides[-1]:return []
    direction=1 if sides[-1] else -1
    crossings=np.flatnonzero((sides[1:]!=sides[:-1]) & (sides[1:]==sides[-1]))+1
    if not len(crossings):raise AssertionError('different endpoints without crossing')
    return [[int(frames[crossings[-1]]),direction]]

def analyze(rows,canon,pairs,common_slate,polygon,nframes,k,GTevents=None,*,slate_kind='common'):
    """Per-pair local physical union + all <=k-subset triangle upper bound.

    API matches physical209: full pair catalog and fixed pair-ID slate. Same-date
    winners use confidence then ORIGINAL raw row index; global union cannot
    alter another action because no two slate pairs share an original run.
    This count-only helper never invokes native GT/event matching.
    """
    import physical209
    from beetrack.evaluate import inside
    if slate_kind!='common':raise ValueError('This diagnostic is not a bound for J or another slate')
    R,C,runs=physical209.original_runs(rows,canon);table=physical209.pair_table(pairs)
    slate=list(common_slate);k=integer(k);nframes=integer(nframes)
    if len(slate)!=len(set(slate)) or not set(slate)<=table.keys() or not 0<=k<=len(slate):raise ValueError('fixed slate/declared budget differs')
    if len(R) and (R[:,0].min()<1 or R[:,0].max()>nframes):raise ValueError('row outside clip')
    used=set();bits=inside(R,polygon);records=[]
    for key in slate:
        pair=table[key];a,b=pair['canonical_ids']
        if a not in runs or b not in runs or {a,b}&used:raise ValueError('slate must be disjoint original canonical runs')
        if not pair['geometric_eligible']:raise ValueError('pair outside frozen geometric eligibility')
        used.update((a,b));left={int(R[i,0]):int(i) for i in runs[a]};right={int(R[i,0]):int(i) for i in runs[b]}
        common=sorted(left.keys()&right.keys());actual=[[left[f],right[f]] for f in common]
        if not common or common!=pair['shared_dates'] or actual!=pair['row_indices']:raise ValueError('frozen shared raw dates differ')
        ix=np.asarray(actual,np.int64);A,B=R[ix[:,0],2:6],R[ix[:,1],2:6]
        wh=np.maximum(0,np.minimum(A[:,:2]+A[:,2:],B[:,:2]+B[:,2:])-np.maximum(A[:,:2],B[:,:2]))
        intersection=np.prod(wh,axis=1);ious=intersection/(np.prod(A[:,2:],axis=1)+np.prod(B[:,2:],axis=1)-intersection)
        if float(ious.mean())!=pair['mean_iou']:raise ValueError('frozen meanIoU differs actual all-shared dates')
        winners={};dropped=[]
        for i,j in actual:
            winner,loser=(i,j) if R[i,6]>R[j,6] or R[i,6]==R[j,6] and i<j else (j,i)
            winners[int(R[i,0])]=winner;dropped.append(loser)
        kept=[]
        for frame in sorted(left.keys()|right.keys()):
            kept.append(winners[frame] if frame in winners else left[frame] if frame in left else right[frame])
        kept=np.asarray(kept,np.int64)
        before=side_event(R[runs[a],0],bits[runs[a]])+side_event(R[runs[b],0],bits[runs[b]])
        after=side_event(R[kept,0],bits[kept])
        delta=window_counts(after,nframes)-window_counts(before,nframes);l1=sum(abs(int(v)) for v in delta.ravel())
        if l1>90:raise AssertionError('two original events + one merged event exceed90 incidence units')
        records.append(dict(pair_id=key,canonical_ids=[int(a),int(b)],before_events=before,after_events=after,
            retained_raw_indices=kept.tolist(),shared_winner_raw_indices=[winners[f] for f in common],deleted_raw_indices=dropped,
            delta_windows=delta.tolist(),delta_indices=np.flatnonzero(delta.ravel()).tolist(),
            delta_values=delta.ravel()[np.flatnonzero(delta.ravel())].tolist(),delta_l1=l1,
            changes_any_window=bool(l1),merged_max_gap=int(np.diff(R[kept,0]).max(initial=0))))
    ordered=sorted(records,key=lambda r:(-r['delta_l1'],*r['canonical_ids']));prefix=[0]
    for r in ordered:prefix.append(prefix[-1]+r['delta_l1'])
    maximum=max((r['delta_l1'] for r in records),default=0);changed=sum(r['changes_any_window'] for r in records)
    result=dict(slate_kind='fixed_disjoint_common_only',common_slate=slate,n=len(slate),declared_k=k,
        formal_quarter_budget=(len(slate)+3)//4 if len(slate)>=4 else 0,
        pairs=records,l1_descending_pair_ids=[r['pair_id'] for r in ordered],prefix_l1_upper=prefix,
        topk_l1_upper=prefix[k],max_single_pair_l1=maximum,any_window_pairs=changed,
        any_window_fraction=[changed,len(slate)],any_window_proportion=changed/len(slate) if slate else None,
        all_count_invariant=changed==0,GTevents=GTevents,normalization_denominator=None,W_gain_upper_percent=None,
        W_gain_upper_fraction=None,cannot_reach_1pp=None,
        proof='|P-G|_1-|P+sum(delta)-G|_1 <= |sum(delta)|_1 <= sum selected|delta|_1 <= sum topk|delta|_1',
        limits='No guarantee of achievable gain or F1; not for J; disjoint original runs and no new >25 official split required; fixed k not chosen here.')
    if GTevents is not None:
        GTevents=integer(GTevents)
        if GTevents<0:raise ValueError('nonnegative scalar GT census required')
        result['GTevents']=GTevents;denominator=30*GTevents;result['normalization_denominator']=denominator
        if denominator:
            bound=Fraction(100*prefix[k],denominator);required=(denominator+99)//100
            result.update(W_gain_upper_percent=float(bound),W_gain_upper_fraction=[bound.numerator,bound.denominator],
                cannot_reach_1pp=bound<Fraction(1),required_raw_gain_for_1pp=required,
                necessary_max_single_pair_l1_for_1pp=(required+k-1)//k if k else None,
                necessary_changed_pairs_for_1pp=(required+maximum-1)//maximum if maximum else None)
        else:result['undefined_normalization_reason']='Zero GT events: pool raw bounds with other registered clips; do not drop false positives.'
    return result

def selftest():
    import physical209,catalog209
    from beetrack.p0 import events as official_events
    def row(f,p,x,y=0,score=.9):return [f,p,x,y,4,4,score]
    polygon=[[0,-20],[100,-20],[100,100],[0,100]];nf=7
    A=[row(f,1,-3) for f in range(1,6)]
    B=[row(f,2,-1,score=.8) for f in range(3,8)]
    D=[row(f,3,-3 if f<6 else -1,y=20,score=.7) for f in range(1,8)]
    E=[row(f,4,-3 if f<6 else -1,y=20,score=.7) for f in range(1,8)]
    R=np.asarray(A+B+D+E,float);canon=R[:,1].astype(np.int64);pairs=catalog209.build(R,canon,nf)['pairs'];slate=['1|2','3|4']
    answer=analyze(R,canon,pairs,slate,polygon,nf,2,GTevents=1)
    assert [p['delta_l1'] for p in answer['pairs']]==[30,30] and answer['topk_l1_upper']==60
    delta=sum(np.asarray(p['delta_windows']) for p in answer['pairs']);assert not delta.any()
    merged,proof=physical209.materialize(R,canon,pairs,slate)
    actual=window_counts(official_events(merged,polygon).astype(int),nf)-window_counts(official_events(R,polygon).astype(int),nf)
    assert np.array_equal(delta,actual) # opposite effects cancel for EVERY truth
    baseline=window_counts(official_events(R,polygon).astype(int),nf)
    for truth_in in range(4):
        for truth_out in range(4):
            truth=window_counts([[1,1]]*truth_in+[[6,-1]]*truth_out,nf)
            for bits in ((0,0),(0,1),(1,0),(1,1)):
                change=sum((np.asarray(p['delta_windows']) for p,bit in zip(answer['pairs'],bits) if bit),np.zeros_like(baseline))
                gain=int(np.abs(baseline-truth).sum()-np.abs(baseline+change-truth).sum())
                assert gain<=answer['prefix_l1_upper'][sum(bits)]
    for p in answer['pairs']:
        out,_=physical209.materialize(R,canon,pairs,[p['pair_id']])
        observed=window_counts(official_events(out,polygon).astype(int),nf)-window_counts(official_events(R,polygon).astype(int),nf)
        assert np.array_equal(observed,p['delta_windows'])
    assert answer['pairs'][0]['after_events']==[[6,1]] # winner A on shared dates3..5; both have real +/-1 around median4
    assert answer['pairs'][1]['shared_winner_raw_indices']==list(range(10,17)) # ties use original raw index
    try:analyze(R,canon,pairs,['1|2','1|3'],polygon,nf,1)
    except ValueError:pass
    else:raise AssertionError('overlapping slate accepted')
    # Zero common-slate effect does not bound J: validity filter removes the
    # highest-IoU cross-side pair; J is allowed that DIFFERENT pair.
    R2=np.asarray([row(f,1,-3) for f in (1,2,3)]+[row(f,3,-3,y=2.5) for f in (1,2,3)]+[row(f,2,-1,score=.8) for f in (3,4,5)],float);C2=R2[:,1].astype(np.int64)
    P2=catalog209.build(R2,C2,nf)['pairs'];valid={p['pair_id']:p['pair_id']=='1|3' for p in P2}
    common=physical209.geometric_greedy(P2,valid)['selected'];J=physical209.geometric_greedy(P2)['selected']
    assert common==['1|3'] and J==['1|2']
    assert analyze(R2,C2,P2,common,polygon,nf,1)['all_count_invariant']
    out,_=physical209.materialize(R2,C2,P2,J)
    assert not np.array_equal(window_counts(official_events(out,polygon).astype(int),nf),window_counts(official_events(R2,polygon).astype(int),nf))
    try:analyze(R2,C2,P2,J,polygon,nf,1,slate_kind='J')
    except ValueError:pass
    else:raise AssertionError('J scope accepted')
    assert side_event([1,2,3,4],[False,True,False,True])==[[4,1]]
    assert np.abs(window_counts([[25,1]],26)-window_counts([[26,1]],26)).sum()==2
    try:side_event([1,27],[False,True])
    except ValueError:pass
    else:raise AssertionError('official gap split ignored')
    empty=analyze(R,canon,pairs,[],polygon,nf,0,GTevents=0)
    assert empty['topk_l1_upper']==0 and empty['W_gain_upper_percent'] is None
    print('PASS synthetic: local vs physical official deltas; disjoint additive; opposing cancellation; raw-index ties; last-crossing; >25 split refusal; common != J; zero GT')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()));selftest()
