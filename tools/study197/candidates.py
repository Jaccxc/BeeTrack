"""Pure GT-free endpoint ID cuts and fixed G/T features; synthetic by default.

Every candidate cuts one canonical run, preserves every observation, and has
exactly one signed-event count addition/removal. No field IO, fitting or GT.
"""
from collections import Counter
import cv2
import numpy as np
from beetrack.evaluate import by_track,inside,zone_tracks

G_NAMES=('k_minus1','span_over12','is_end','short_inside','base_has_event',
         'short_minus_core_score','log_parent_size','abs_distance_short','abs_distance_core')
T_NAMES=G_NAMES+('short_prediction_residual','core_sign_prediction','core_fit_residual')


def event_times(rows,polygon):
    rows=by_track(rows)
    result=[]
    for pid,(direction,start,end,cross) in zone_tracks(rows,polygon).items():
        if direction:
            same=[j for j,d in cross if d==direction]
            result.append(dict(track_id=int(pid),frame=int(rows[same[-1] if same else end-1,0]),direction=1 if direction=='in' else -1))
    return result


def window_counts(events,nf):
    result=[]
    for offset in range(30):
        bins=np.zeros(((nf+25*offset+749)//750,2),dtype=np.int64)
        for e in events:
            if not 1<=e['frame']<=nf or e['direction'] not in (-1,1): raise ValueError('event clock/direction invalid')
            bins[(e['frame']+25*offset-1)//750,int(e['direction']<0)]+=1
        result.extend(bins)
    return np.asarray(result,dtype=np.int64).reshape(-1,2)


def _canonical(raw,nf):
    raw=np.asarray(raw,dtype=np.float64)
    if raw.ndim!=2 or raw.shape[1]<7 or not np.isfinite(raw).all(): raise ValueError('finite native Nx7+ rows required')
    if isinstance(nf,bool) or int(nf)!=nf or nf<1: raise ValueError('positive integer clip length required')
    if np.any(raw[:,:2]!=np.rint(raw[:,:2])) or np.any(raw[:,0]<1) or np.any(raw[:,0]>nf) or np.any(raw[:,1]<0):
        raise ValueError('invalid native frame/ID')
    if np.any(raw[:,1]>=2**53-100000*max(len(raw),1)): raise ValueError('unsafe ID range')
    if len(np.unique(raw[:,:2],axis=0))!=len(raw): raise ValueError('duplicate native frame/ID')
    aliases=set()
    for identity in np.unique(raw[:,1]):
        dates=np.sort(raw[raw[:,1]==identity,0])
        for run in range(1+int(np.sum(np.diff(dates)>25))):
            alias=int(identity)+100000*run
            if alias in aliases: raise ValueError('ambiguous official run alias')
            aliases.add(alias)
    indexed=by_track(np.c_[raw,np.arange(len(raw))])
    canonical=indexed[:,:-1];source=indexed[:,-1].astype(int)
    if not np.array_equal(canonical,by_track(raw)) or not np.array_equal(by_track(canonical),canonical):
        raise ValueError('non-idempotent canonical run mapping')
    if not np.array_equal(np.sort(source),np.arange(len(raw))) or len(np.unique(canonical[:,:2],axis=0))!=len(raw):
        raise ValueError('canonical mapping not a collision-free bijection')
    return raw,canonical,source


def _feature(window,parent,short_inside,is_end,k,span,has_event,polygon):
    scale=float(np.mean(np.sqrt(window[:,4]*window[:,5])))
    poly=np.asarray(polygon,dtype=np.float32)
    center=window[:,2:4]+window[:,4:6]/2
    d=np.array([cv2.pointPolygonTest(poly,(float(x),float(y)),True)/scale for x,y in center])
    tau=np.abs(window[:,0]-window[0,0])
    A=np.c_[np.ones(3),tau[k:k+3]]
    if len(set(tau[k:k+3]))!=3: raise ValueError('three unique core dates required')
    coef=np.linalg.lstsq(A,d[k:k+3],rcond=None)[0]
    pred=coef[0]+coef[1]*tau[:k]
    fitted=A@coef;core_inside=not short_inside;core_sign=1 if core_inside else -1
    heuristic=bool(np.all((pred>=0)==core_inside))
    G=[float(k-1),span/12,float(is_end),float(short_inside),float(has_event),
       float(window[:k,6].mean()-window[k:k+3,6].mean()),float(np.log1p(len(parent))/np.log(51)),
       float(np.mean(np.abs(d[:k]))),float(np.mean(np.abs(d[k:k+3])))]
    extra=[float(np.mean(np.abs(d[:k]-pred))),float(core_sign*pred.mean()),float(np.mean(np.abs(d[k:k+3]-fitted)))]
    G[-2:]=np.clip(G[-2:],-4,4).tolist();extra=np.clip(extra,-4,4).tolist()
    return dict(features_G=G,features_T=G+extra,H=heuristic,core_inside=core_inside,
                distance_scale=scale,d_normalized=d.tolist(),tau=tau.tolist(),core_fit_ab=coef.tolist(),
                pred_short=pred.tolist(),core_fitted=fitted.tolist())


def build_candidates(raw,polygon,nf):
    raw,canonical,source=_canonical(raw,nf)
    ids,starts=np.unique(canonical[:,1],return_index=True)
    ends=np.r_[starts[1:],len(canonical)]
    candidates=[];parents=[];exclusions=Counter()
    for rank,(pid,start,end) in enumerate(zip(ids,starts,ends)):
        pid=int(pid);start=int(start);end=int(end)
        parent=canonical[start:end]
        parents.append(dict(rank=rank,parent_id=pid,start=start,end=end,left_new_id=2*rank+1,right_new_id=2*rank+2))
        if len(parent)<4:
            exclusions['parents_fewer4_observations']+=1
            continue
        side=inside(parent,polygon);original=event_times(parent,polygon)
        options=[]
        for endpoint in ('start','end'):
            indices=np.arange(start,end) if endpoint=='start' else np.arange(end-1,start-1,-1)
            sides=side if endpoint=='start' else side[::-1]
            for k in (1,2):
                if len(indices)<k+3:
                    exclusions['end_k_insufficient_observations']+=1
                    continue
                chosen=indices[:k+3];window=canonical[chosen]
                span=int(abs(window[-1,0]-window[0,0]))
                if span>12:
                    exclusions['end_k_span_over12']+=1
                    continue
                if not np.all(window[:,4:6]>0):
                    exclusions['end_k_nonpositive_box']+=1
                    continue
                if not np.all(sides[:k]==sides[0]) or not np.all(sides[k:k+3]!=sides[0]):
                    exclusions['end_k_side_pattern_fail']+=1
                    continue
                cut=start+k-1 if endpoint=='start' else end-k-1
                options.append(dict(end=endpoint,k=k,span=span,window_indices=chosen.tolist(),
                                    short_inside=bool(sides[0]),cut_after_canonical_row=int(cut)))
        if not options:
            exclusions['parents_no_eligible_end']+=1
            continue
        choice=min(options,key=lambda x:(x['k'],x['span'],x['end']!='start'))
        cut=choice['cut_after_canonical_row'];local=cut-start+1
        left=parent[:local].copy();right=parent[local:].copy()
        left[:,1]=2*rank+1;right[:,1]=2*rank+2
        replacement=event_times(np.r_[left,right],polygon)
        old=Counter((e['frame'],e['direction']) for e in original)
        new=Counter((e['frame'],e['direction']) for e in replacement)
        added=list((new-old).elements());dropped=list((old-new).elements())
        if len(added)+len(dropped)!=1:
            raise ValueError('endpoint cut is not exactly one signed-event toggle')
        frame,direction=(added or dropped)[0];delta=1 if added else -1
        vector=window_counts(replacement,nf)-window_counts(original,nf)
        single=delta*window_counts([dict(frame=frame,direction=direction)],nf)
        if not np.array_equal(vector,single): raise ValueError('event delta clock mismatch')
        window=canonical[choice['window_indices']]
        feature=_feature(window,parent,choice['short_inside'],choice['end']=='end',choice['k'],choice['span'],bool(original),polygon)
        candidates.append(dict(candidate_id=f"cut:{pid}:{choice['end']}:{choice['k']}",parent_id=pid,parent_rank=rank,
                               **choice,**feature,base_has_event=bool(original),frame=int(frame),direction=int(direction),delta=delta,
                               original_events=original,replacement_events=replacement,delta_windows=vector,
                               left_range=[start,cut+1],right_range=[cut+1,end]))
    baseline=event_times(canonical,polygon)
    return dict(schema='endpoint-cut-candidates-v1',raw_rows=raw.copy(),canonical_rows=canonical,source_raw_index=source,
                parent_blocks=parents,candidates=candidates,baseline_events=baseline,baseline_windows=window_counts(baseline,nf),
                exclusions=dict(exclusions),
                polygon=polygon,nframes=int(nf),feature_names_G=list(G_NAMES),feature_names_T=list(T_NAMES),
                clock=dict(fps=25,window_frames=750,offset_seconds=list(range(30)),frame_origin=1))


def self_test():
    poly=[[0,0],[10,0],[10,10],[0,10]]
    row=lambda f,p,x,score=.8:[f,p,x-1,4,2,2,score]
    # Outside short prefix then inside core: removes exactly one original in.
    raw=np.array([row(f,9,x) for f,x in enumerate((12,9,8,7,6),1)],float)
    b=build_candidates(raw,poly,40);c=b['candidates'][0]
    assert (c['end'],c['k'],c['span'],c['delta'],c['direction'])==('start',1,3,-1,1)
    assert c['cut_after_canonical_row']==0 and len(c['features_G'])==9 and len(c['features_T'])==12
    assert c['delta_windows'].sum()==-30
    # No original net event: prefix removal creates an out at the last change.
    add=np.r_[raw,np.array([row(6,9,12)])]
    c=build_candidates(add,poly,40)['candidates'][0]
    assert c['delta']==1 and c['direction']==-1 and c['frame']==6
    # End-only short suffix and two-observation short block.
    tail=np.array([row(f,4,x) for f,x in enumerate((7,8,9,12,13),1)],float)
    c=build_candidates(tail,poly,40)['candidates'][0]
    assert c['end']=='end' and c['k']==2 and c['cut_after_canonical_row']==2
    # Span too long fails even when observation count passes; nonpositive blocks fail.
    slow=raw.copy();slow[:,0]*=5
    assert not build_candidates(slow,poly,40)['candidates']
    bad=raw.copy();bad[1,4]=0
    assert not build_candidates(bad,poly,40)['candidates']
    # Input order mapping remains bijective, and ID reuse becomes separate parents.
    reuse=np.r_[raw,np.array([row(f+30,9,x) for f,x in enumerate((12,9,8,7,6),1)],float)][::-1]
    bundle=build_candidates(reuse,poly,75)
    assert len(bundle['parent_blocks'])==2 and len(bundle['candidates'])==2
    assert np.array_equal(bundle['canonical_rows'][:,0],reuse[bundle['source_raw_index'],0])
    assert build_candidates(np.empty((0,7)),poly,40)['baseline_windows'].sum()==0
    print('Endpoint cuts synthetic tie/short/span/add-drop/clock/G9T12/mapping checks passed')


if __name__=='__main__':self_test()
