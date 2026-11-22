"""Independent §210 verifier. SOURCE/SYNTHETIC ONLY until sole-CPU GO.

No data paths are opened on import. Bracket geometry is an interpolation
diagnostic, never a physical identity likelihood or an independent-bee count.
Field adapter is added against the final frozen producer contract.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from bisect import bisect_left
import itertools
import json
import numpy as np


def runs(rows,canonical):
    result={}
    for j,c in enumerate(canonical):result.setdefault(int(c),[]).append(j)
    for c,ii in result.items():
        ii.sort(key=lambda j:(rows[j,0],j))
        if any(rows[j,0]>=rows[k,0] for j,k in zip(ii,ii[1:])):raise ValueError('duplicate run/date')
    return result


def brackets(rows,canonical,a,b,rr=None):
    """All actual nearest opposite-run outer observations; singleton allowed."""
    rr=runs(rows,canonical) if rr is None else rr;aa,bb=rr[a],rr[b]
    da={int(rows[j,0]) for j in aa};db={int(rows[j,0]) for j in bb}
    if da.intersection(db):return []
    answer=[]
    for middle,outer,sign in ((bb,aa,1),(aa,bb,-1)):
        outer_dates=[rows[j,0] for j in outer]
        for mid in middle:
            date=rows[mid,0]
            position=bisect_left(outer_dates,date)
            if position==0 or position==len(outer):continue
            l,r=outer[position-1],outer[position];t0,t,t1=rows[[l,mid,r],0]
            if t-t0>3 or t1-t>3:continue
            centres=rows[[l,mid,r],2:4]+rows[[l,mid,r],4:6]/2
            estimate=centres[0]+(t-t0)/(t1-t0)*(centres[2]-centres[0])
            displacement=sign*(centres[1]-estimate)
            local_scale=sum(float(np.linalg.norm(rows[j,4:6])) for j in (l,mid,r))/3
            answer.append(dict(raw_indices=[l,mid,r],dates=[int(t0),int(t),int(t1)],middle_canonical=int(canonical[mid]),b_minus_a=displacement.tolist(),local_mean_diagonal=local_scale,normalized_offset=float(np.linalg.norm(displacement)/local_scale)))
    return sorted(answer,key=lambda x:(x['dates'][1],x['middle_canonical']))


def candidate_universe(rows,canonical):
    rr=runs(rows,canonical);answer=[];bydate={};possible=set()
    dates={pid:set(rows[ii,0]) for pid,ii in rr.items()}
    for j,row in enumerate(rows):bydate.setdefault(int(row[0]),[]).append(j)
    # Every legal bracket has one pair of consecutive outer observations.
    # Enumerate those intervals directly, including singleton middle tracks.
    for pid,ii in rr.items():
        for left,right in zip(ii,ii[1:]):
            t0,t1=map(int,rows[[left,right],0])
            for date in range(max(t0+1,t1-3),min(t1-1,t0+3)+1):
                for middle in bydate.get(date,[]):
                    other=int(canonical[middle])
                    if other!=pid:possible.add(tuple(sorted((pid,other))))
    for a,b in sorted(possible):
        if dates[a].intersection(dates[b]):continue
        support=brackets(rows,canonical,a,b,rr)
        if support and any(bracket['normalized_offset']<=1 for bracket in support):answer.append(dict(pair_id=f'{a}|{b}',canonical_ids=[a,b],brackets=support,min_offset_norm=min(bracket['normalized_offset'] for bracket in support)))
    return answer


def features(rows,canonical,a,b,rr=None):
    support=brackets(rows,canonical,a,b,rr)
    if not support:return None
    unique={j for bracket in support for j in bracket['raw_indices']}
    scale=float(np.median([np.linalg.norm(rows[j,4:6]) for j in sorted(unique)]))
    dates=[int(rows[j,0]) for j in unique];cutoff=(min(dates)+max(dates))//2
    before=[i for i,bracket in enumerate(support) if bracket['dates'][2]<=cutoff]
    after=[i for i,bracket in enumerate(support) if bracket['dates'][0]>cutoff]
    nb=sum(bracket['middle_canonical']==b for bracket in support);na=len(support)-nb
    full=bool(na and nb and len(before)>=2 and len(after)>=2)
    offsets=np.asarray([bracket['b_minus_a'] for bracket in support]);length=np.linalg.norm(offsets,axis=1)/scale
    g=[float(min(length)),float(np.median(length)),float(np.log1p(len(support))),min(na,nb)/len(support),float(full),0.]
    x=[0.,0.];xm=[0.,0.];estimate=None
    if full:
        prefixraw={j for i in before for j in support[i]['raw_indices']};suffixraw={j for i in after for j in support[i]['raw_indices']}
        assert not prefixraw.intersection(suffixraw)
        estimate=np.median(offsets[before],axis=0);suffix=offsets[after]
        rms=lambda xy:float(np.sqrt(np.mean(np.sum(xy*xy,axis=1))))
        zero=rms(suffix);correct=rms(suffix-estimate);wrong=rms(suffix+estimate)
        g[-1]=zero/scale;x=[float(np.linalg.norm(estimate)/scale),(zero-correct)/scale]
        xm=[x[0],(zero-wrong)/scale]
    return dict(G=g,X=x,Xminus=xm,full_support=full,common_scale=scale,
                brackets=support,prefix_indices=before,suffix_indices=after,
                cutoff_frame=cutoff,prefix_offset=None if estimate is None else estimate.tolist())


def merge(rows,canonical,selected_pairs,all_pairs=None,return_indices=False):
    """Union two original runs with no shared date; preserve EVERY raw row."""
    result=np.asarray(rows).copy();used=set()
    if selected_pairs:result[:,1]=canonical
    for a,b in selected_pairs:
        if a==b or a in used or b in used:raise ValueError('chained/reused run')
        ia=np.flatnonzero(canonical==a);ib=np.flatnonzero(canonical==b)
        if set(rows[ia,0]).intersection(rows[ib,0]):raise ValueError('common date forbidden')
        result[np.r_[ia,ib],1]=min(a,b);used.update((a,b))
    if not np.array_equal(result[:,[0,2,3,4,5,6]],rows[:,[0,2,3,4,5,6]]):raise AssertionError('raw measurements changed')
    if len(set(map(tuple,result[:,:2])))!=len(result):raise AssertionError('duplicate output ID/date')
    order=np.lexsort((np.arange(len(rows)),rows[:,0])) if selected_pairs else np.arange(len(rows))
    result=result[order]
    return (result,order.tolist()) if return_indices else result


def weak_pair(bracket_support,owners,rows,presence):
    """Three reliable old owners; unknown is retained, not a negative label."""
    labels=[];same_original=set();covisible=False
    for bracket in bracket_support:
        ix=bracket['raw_indices'];left,middle,right=[tuple(map(int,owners[j])) for j in ix]
        label='U'
        if all(min(owner)>=0 for owner in (left,middle,right)) and left==right:
            if middle==left:label='S';same_original.add(left[0])
            elif middle[0]!=left[0]:
                label='D';date=int(rows[ix[1],0])
                covisible|=bool(presence.get((date,left[0]))) and bool(presence.get((date,middle[0])))
        labels.append(label)
    S,D=labels.count('S'),labels.count('D')
    pair='S' if S and not D and len(same_original)==1 else 'D' if D and not S else 'U'
    return dict(label=pair,brackets=labels,co_visible=pair=='D' and covisible,
                counts={c:labels.count(c) for c in ('S','D','U')})


def clock(events,nframes):
    result=[]
    for phase in range(30):
        shift=25*phase
        for j in range((nframes+shift+749)//750):
            lo,hi=750*j-shift+1,750*(j+1)-shift
            result.append([sum(lo<=f<=hi and d==1 for f,d in events),sum(lo<=f<=hi and d==-1 for f,d in events)])
    return np.asarray(result,np.int64)


def self_test():
    rows=np.array([[3,1,2,0,2,2,.9],[2,2,6,0,2,2,.8],[1,1,0,0,2,2,.7]],float)
    canonical=rows[:,1].astype(int)
    support=brackets(rows,canonical,1,2)
    assert len(support)==1 and support[0]['raw_indices']==[2,1,0]
    assert support[0]['dates']==[1,2,3] and support[0]['b_minus_a']==[5.,0.]
    output=merge(rows,canonical,[(1,2)],[(1,2)])
    assert output.shape==rows.shape and len(set(output[:,1]))==1
    bad=np.r_[rows,[[2,1,1,0,2,2,.6]]]
    assert not brackets(bad,bad[:,1].astype(int),1,2)
    bins=clock([(1,1),(750,-1),(751,1)],751)
    assert bins.sum()==90 and bins[0].tolist()==[1,1] and bins[1].tolist()==[1,0]
    rows=[]
    for date in range(1,32):
        pid=1 if date%2 else 2;offset=np.array([0,0]) if pid==1 else np.array([3,-1])
        center=np.array([.4*date,.2*date])+offset
        rows.append([date,pid,*(center-1),2,2,.9])
    rows=np.asarray(rows);feat=features(rows,rows[:,1].astype(int),1,2)
    assert feat['full_support'] and feat['X'][1]>0 and feat['Xminus'][1]<0
    reverse=features(rows,rows[:,1].astype(int),2,1)
    assert np.allclose(feat['G'],reverse['G']) and np.allclose(feat['X'],reverse['X'])
    return dict(passed=True,scope='synthetic only',checks=['singleton bracket','signed B-minus-A interpolation','original-row order/full retention','common-date rejection','30phase native clock','unique-support scale/prefix-suffix separation','correct versus reversed offset sign'])


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--audit',action='store_true')
    p.add_argument('--repo',type=Path);p.add_argument('--root',type=Path,default=Path('D:/beetrack_runs/interleave210'))
    p.add_argument('--source208',type=Path,default=Path('D:/beetrack_runs/switch208/inputs'))
    p.add_argument('--labels208',type=Path,default=Path('D:/beetrack_runs/switch208/labels'))
    p.add_argument('--reference209',type=Path,default=Path('D:/beetrack_runs/common209/score'))
    p.add_argument('--seal-pins',type=Path);p.add_argument('--field-go',action='store_true');p.add_argument('--physical-finished',action='store_true');args=p.parse_args()
    if args.audit:
        if args.repo is None or args.seal_pins is None:p.error('--repo and --seal-pins required')
        import field210
        field210.run(args)
    else:print(json.dumps(self_test(),sort_keys=True))
