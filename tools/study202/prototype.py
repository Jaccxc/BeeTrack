"""SYNTHETIC action prototype, NOT a registered deployment rule.

Only a disjoint, complete linked-before-drop short track may be appended.
Sources are parallel (src_hash, raw_row_index) tuples, never geometry keys.
"""
import argparse
import json
import os
from pathlib import Path
import sys
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np


def _rows(rows):
    a=np.asarray(rows,float).reshape(-1,7)
    if not np.isfinite(a).all() or (a[:,4:6]<=0).any():raise ValueError('positive finite boxes required')
    if (a[:,:2]!=np.floor(a[:,:2])).any() or (a[:,0]<1).any():raise ValueError('integral frame/ID required')
    if len(set(map(tuple,a[:,:2])))!=len(a):raise ValueError('duplicate(track,date)')
    return a


def validate(base,base_sources,orphans):
    base=_rows(base);seen=set();ids=set(base[:,1])
    def take(rows,sources):
        if len(rows)!=len(sources):raise ValueError('source cardinality')
        for source in sources:
            if len(source)!=2 or not isinstance(source[0],str) or not source[0] or int(source[1])!=source[1] or source[1]<0:
                raise ValueError('source must be (hash, nonnegative integer raw index)')
            key=tuple(source)
            if key in seen:raise ValueError('duplicate source row')
            seen.add(key)
    take(base,base_sources)
    orphan_ids=set()
    for o in orphans:
        r=_rows(o['rows'])
        if len(r) not in (1,2) or len(set(r[:,1]))!=1 or (np.diff(r[:,0])<=0).any():
            raise ValueError('complete ordered one/two-row short track required')
        oid=int(r[0,1])
        if oid in ids or oid in orphan_ids:raise ValueError('orphan is not disjoint from retained tracks')
        orphan_ids.add(oid);take(r,o['sources'])
    return base


def candidates(base,base_sources,orphans,max_gap=12,max_error=2):
    """GT-free linear extrapolation from nearest three recipient endpoint rows."""
    base=validate(base,base_sources,orphans);out=[]
    for pid in sorted(set(base[:,1])):
        track=base[base[:,1]==pid];track=track[np.argsort(track[:,0],kind='stable')]
        if len(track)<3:continue
        for i,o in enumerate(orphans):
            short=_rows(o['rows'])
            if short[-1,0]<track[0,0]:side='before';core=track[:3];gap=track[0,0]-short[-1,0]
            elif short[0,0]>track[-1,0]:side='after';core=track[-3:];gap=short[0,0]-track[-1,0]
            else:continue
            if gap>max_gap:continue
            t=core[:,0];center=core[:,2:4]+core[:,4:6]/2
            design=np.column_stack((np.ones(3),t-t.mean()))
            beta=np.linalg.lstsq(design,center,rcond=None)[0]
            predicted=np.column_stack((np.ones(len(short)),short[:,0]-t.mean()))@beta
            scale=float(np.median(np.hypot(core[:,4],core[:,5])))
            error=float(np.max(np.linalg.norm(predicted-(short[:,2:4]+short[:,4:6]/2),axis=1))/scale)
            if error<=max_error:
                out.append(dict(recipient=int(pid),orphan_index=i,orphan_id=int(short[0,1]),side=side,
                                gap=int(gap),normalized_error=error))
    return sorted(out,key=lambda c:(c['normalized_error'],c['gap'],c['recipient'],c['orphan_id'],c['side']))


def realize(base,base_sources,orphans,selected):
    """Preserve baseline rows/IDs exactly; append original orphan boxes under recipient."""
    base=validate(base,base_sources,orphans);used_rec=set();used_orphan=set()
    parts=[base.copy()];sources=list(map(tuple,base_sources))
    legal={(c['recipient'],c['orphan_index'],c['side']) for c in candidates(base,base_sources,orphans)}
    for c in selected:
        pid,oi=c['recipient'],c['orphan_index']
        if (pid,oi,c['side']) not in legal:raise ValueError('not a legal demonstration candidate')
        if pid in used_rec or oi in used_orphan:raise ValueError('recipient/orphan capacity exceeded')
        used_rec.add(pid);used_orphan.add(oi)
        rows=_rows(orphans[oi]['rows']).copy();rows[:,1]=pid
        parts.append(rows);sources.extend(map(tuple,orphans[oi]['sources']))
    result=np.concatenate(parts)
    _rows(result)
    if len(set(sources))!=len(sources):raise ValueError('source reused')
    assert np.array_equal(result[:len(base)],base)
    return result,sources


def select(catalog):
    """Demonstration greedy arbitration; formal policy must be preregistered."""
    chosen=[];rec=set();short=set()
    for c in catalog:
        if c['recipient'] not in rec and c['orphan_index'] not in short:
            chosen.append(c);rec.add(c['recipient']);short.add(c['orphan_index'])
    return chosen


def event_change(before,after,polygon):
    from tools.study197.candidates import event_times
    a={e['track_id']:e for e in event_times(before,polygon)}
    b={e['track_id']:e for e in event_times(after,polygon)}
    changes=[]
    for pid in sorted(a.keys()|b.keys()):
        if a.get(pid)==b.get(pid):continue
        kind='added' if pid not in a else 'removed' if pid not in b else 'date_only' if a[pid]['direction']==b[pid]['direction'] else 'direction_changed'
        changes.append(dict(track_id=pid,kind=kind,before=a.get(pid),after=b.get(pid)))
    return dict(before=list(a.values()),after=list(b.values()),changes=changes,
                net_events=len(b)-len(a))


def self_test():
    from tools.study195.run import score_member_native
    poly=[(0,0),(20,0),(20,20),(0,20)]
    def track(pid,xs):return np.array([[f,pid,x,2,2,2,.9] for f,x in enumerate(xs,1)],float)
    turn=track(1,[-4,1,3,1,-3]);other=track(2,[-5,-3,1,3,4]);other[:,3]=14
    base=np.vstack((turn[:4],other));sources=[('synthetic',i) for i in range(len(base))]
    orphan=[dict(rows=turn[4:].copy(),sources=[('synthetic',99)])];orphan[0]['rows'][:,1]=90
    catalog=candidates(base,sources,orphan);assert len(catalog)==1
    extended,extsources=realize(base,sources,orphan,select(catalog))
    cut=base.copy();cut[0,1]=100
    gt=np.vstack((turn,other))[:,:6]
    scores={name:score_member_native(gt,r,poly,75) for name,r in [('K',base),('cut_only',cut),('raw_extension',extended)]}
    assert scores['raw_extension']['window']['numerator']==0
    assert scores['raw_extension']['iou']['0.5']['tp']==scores['raw_extension']['nG']
    assert event_change(base,extended,poly)['changes'][0]['kind']=='removed'
    # Existing non-event becomes an event by actual new measurement.
    non=track(3,[-5,-4,-3]);o=[dict(rows=np.array([[4,91,0,2,2,2,.9]]),sources=[('s',4)])]
    cs=candidates(non,[('s',i) for i in range(3)],o)
    out,_=realize(non,[('s',i) for i in range(3)],o,cs)
    assert event_change(non,out,poly)['net_events']==1
    dated=track(7,[-4,1,2])
    dated_after=np.vstack((dated,[[4,7,-3,2,2,2,.9],[5,7,1,2,2,2,.9]]))
    assert event_change(dated,dated_after,poly)['changes'][0]['kind']=='date_only'
    # Mirror the five-frame return: extension is before rather than after.
    mirrored=base.copy();mirrored[:,0]=6-mirrored[:,0]
    mo=[dict(rows=orphan[0]['rows'].copy(),sources=[('synthetic',99)])];mo[0]['rows'][:,0]=1
    mc=candidates(mirrored,sources,mo);assert len(mc)==1 and mc[0]['side']=='before'
    realize(mirrored,sources,mo,mc)
    # Two endpoints compete for the same orphan; no duplication permitted.
    competition=np.vstack((base,base[:4]+np.array([0,10,0,0,0,0,0])))
    sc=[('competition',i) for i in range(len(competition))]
    cc=candidates(competition,sc,orphan);assert len(cc)==2 and len(select(cc))==1
    try:realize(competition,sc,orphan,cc)
    except ValueError:pass
    else:raise AssertionError('orphan reused')
    bad=[dict(rows=orphan[0]['rows'],sources=[sources[0]])]
    try:candidates(base,sources,bad)
    except ValueError:pass
    else:raise AssertionError('duplicate source accepted')
    return dict(label='SYNTHETIC ONLY: controlled ground-truth boxes, not detector performance',
                scores={k:dict(N=v['window']['numerator'],W=v['W'],nGT=v['nG'],nPred=v['nP'],iou=v['iou']) for k,v in scores.items()},
                events=event_change(base,extended,poly),tests='five-frame return; only-cut; true raw extension; new event; before mirror; competition; duplicate source rejection')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--self-test',action='store_true',required=True)
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    print(json.dumps(self_test(),indent=2))
