"""Unregistered GT-free native-event return queries; synthetic/source only.

No field loader or GT API. Never renumber native K IDs. The immutable event
roster includes ineligible anchors and empty clipped search ranges.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def array_sha(a):
    a=np.ascontiguousarray(a,dtype='<f8')
    return hashlib.sha256(json.dumps(list(a.shape)).encode()+b'\n'+a.tobytes()).hexdigest()


def prepare(rows,sources,nframes):
    """Official >25 gap runs with exact raw row indices, no bbox matching."""
    from beetrack.evaluate import by_track
    if isinstance(nframes,bool) or int(nframes)!=nframes or nframes<1:raise ValueError('positive integer clip length required')
    a=np.asarray(rows,float)
    if a.ndim!=2 or a.shape[1]!=7 or not np.isfinite(a).all() or (a[:,4:6]<=0).any():raise ValueError('finite positive native Nx7 required')
    if (a[:,:2]!=np.floor(a[:,:2])).any() or (a[:,0]<1).any() or (a[:,0]>nframes).any() or (a[:,1]<0).any():raise ValueError('invalid native clock/ID')
    if len(set(map(tuple,a[:,:2])))!=len(a):raise ValueError('duplicate native date/ID')
    keys=[tuple(s) for s in sources]
    if len(keys)!=len(a) or len(set(keys))!=len(keys):raise ValueError('missing or repeated source key')
    if any(len(s)!=2 or not isinstance(s[0],str) or not s[0] or int(s[1])!=s[1] or s[1]<0 for s in keys):raise ValueError('source=(SHA,integer row) required')
    aliases={}
    for pid in np.unique(a[:,1]):
        dates=np.sort(a[a[:,1]==pid,0])
        for run in range(1+int(np.sum(np.diff(dates)>25))):
            alias=int(pid)+100000*run
            if alias>=2**53 or alias in aliases:raise ValueError('official run alias collision')
            aliases[alias]=(int(pid),run)
    tagged=by_track(np.column_stack((a,np.arange(len(a)))))
    canonical=tagged[:,:7];indices=tagged[:,-1].astype(int)
    if not np.array_equal(np.sort(indices),np.arange(len(a))):raise ValueError('canonical index nonbijective')
    runs=[]
    for alias in np.unique(canonical[:,1]):
        ix=np.flatnonzero(canonical[:,1]==alias);rawindices=indices[ix]
        pid,run=aliases[int(alias)]
        runs.append(dict(native_id=pid,run_index=run,canonical_id=int(alias),raw_indices=rawindices,
                         rows=canonical[ix],sources=[keys[j] for j in rawindices]))
    return dict(raw=a,sources=keys,canonical=canonical,index=indices,runs=runs,
                raw_to_canonical={int(j):int(canonical[i,1]) for i,j in enumerate(indices)})


def generate(rows,sources,polygon,nframes,unit='synthetic',slot=0):
    """One frozen query per official native net event, including eligible=False."""
    from beetrack.evaluate import inside,zone_tracks
    data=prepare(rows,sources,nframes);queries=[]
    for run in data['runs']:
        R=run['rows'];zone=zone_tracks(R,polygon)[run['canonical_id']]
        direction,start,end,cross=zone
        if not direction:continue
        sign=1 if direction=='in' else -1
        same=[j for j,d in cross if d==direction]
        event_frame=int(R[same[-1] if same else len(R)-1,0]);f=int(R[-1,0])
        reasons=[]
        if len(R)<3:reasons.append('fewer_than_three_distinct_anchor_observations')
        if len(R)<2 or R[-1,0]-R[-2,0]!=1:reasons.append('last_two_dates_not_consecutive')
        dates=list(range(f+1,min(f+25,int(nframes))+1))
        if not dates:reasons.append('empty_clipped_search_range')
        local=list(dict.fromkeys([0,max(0,len(R)-2),len(R)-1]))
        rawanchors=[int(run['raw_indices'][i]) for i in local]
        L=float(np.median(np.hypot(R[-2:,4],R[-2:,5])))
        center=(R[-1,2:4]+R[-1,4:6]/2).tolist()
        key=f'{unit}|{slot}|{run["native_id"]}|{run["run_index"]}|{f}'
        queries.append(dict(query_id=key,unit=unit,member_index=int(slot),native_id=run['native_id'],
            run_index=run['run_index'],canonical_id=run['canonical_id'],eligible=not reasons,ineligible_reasons=reasons,
            raw_run_indices=run['raw_indices'].tolist(),run_source_keys=[list(s) for s in run['sources']],
            anchor_raw_indices=rawanchors,anchor_rows=data['raw'][rawanchors].tolist(),
            anchor_source_keys=[list(data['sources'][j]) for j in rawanchors],
            event=dict(frame=event_frame,direction=sign),endpoint_frame=f,search_frames=dates,
            search_first=f+1 if dates else None,search_last=dates[-1] if dates else None,
            L=L,center=center,radius=4*L,boundary_inclusive=True,
            starting_side=bool(inside(R[:1],polygon)[0]),raw_K_sha256=array_sha(data['raw']),
            immutable_native_ID=True))
    official_count=sum(bool(z[0]) for z in zone_tracks(data['canonical'],polygon).values())
    if len(queries)!=official_count:raise AssertionError('native event denominator lost')
    return dict(schema='return-query-roster-prototype-v1',unit=unit,member_index=int(slot),nframes=int(nframes),
                queries=queries,total_native_events=official_count,eligible=sum(q['eligible'] for q in queries),
                ineligible=sum(not q['eligible'] for q in queries),raw_K_sha256=array_sha(data['raw']),
                gap_rule='official by_track gap strictly greater than25',status='unregistered synthetic/source only')


def in_corridor(query,frame,box):
    """Deploy-visible fixed disk/date check, not GT candidate creation."""
    b=np.asarray(box,float)
    if b.shape!=(4,) or not np.isfinite(b).all() or (b[2:]<=0).any():raise ValueError('positive finite xywh required')
    return (int(frame)==frame and int(frame) in query['search_frames'] and
            float(np.linalg.norm(b[:2]+b[2:]/2-np.asarray(query['center'])))<=query['radius'])


def insertion_safety(rows,sources,query,new_row,new_source,nframes):
    """Pure native run/provenance guard; NOT a GT or same-bee confirmation.

    Reject rather than re-ID when the appended real row bridges another run,
    shifts an official alias, or collides in date/ID/source. No modification.
    """
    from beetrack.evaluate import by_track
    old=prepare(rows,sources,nframes);row=np.asarray(new_row,float)
    if row.shape!=(7,) or not np.isfinite(row).all() or (row[4:6]<=0).any():return dict(safe=False,reason='invalid_added_geometry')
    if array_sha(old['raw'])!=query['raw_K_sha256']:raise ValueError('query is not for this immutable K')
    if not query['eligible']:return dict(safe=False,reason='query_ineligible')
    if row[1]!=query['native_id']:return dict(safe=False,reason='native_ID_changed')
    if not in_corridor(query,row[0],row[2:6]):return dict(safe=False,reason='outside_fixed_search')
    if tuple(new_source) in old['sources']:return dict(safe=False,reason='physical_source_reused')
    if any(r[0]==row[0] and r[1]==row[1] for r in old['raw']):return dict(safe=False,reason='native_date_ID_conflict')
    proposed=np.vstack((old['raw'],row))
    try:new=prepare(proposed,old['sources']+[tuple(new_source)],nframes)
    except ValueError as exc:return dict(safe=False,reason='canonical_conflict',detail=str(exc))
    for j,alias in old['raw_to_canonical'].items():
        if new['raw_to_canonical'][j]!=alias:return dict(safe=False,reason='other_run_membership_changed',raw_row=j)
    if new['raw_to_canonical'][len(old['raw'])]!=query['canonical_id']:
        return dict(safe=False,reason='addition_not_in_target_run')
    if not np.array_equal(proposed[:len(old['raw'])],old['raw']):raise AssertionError('original K changed')
    return dict(safe=True,reason='native_run_safe_only',raw_prefix_exact=True,
                added_canonical_id=new['raw_to_canonical'][len(old['raw'])],
                GT_identity_and_scene_matching_checked=False)


def self_test():
    def row(f,p,x):return [f,p,x,0,2,2,.9]
    poly=[(0,-20),(50,-20),(50,20),(0,20)]
    # RawID7 has two >25gap runs; do not silently allocate safe fresh IDs.
    R=np.array([row(1,7,-4),row(2,7,-3),row(3,7,1),row(30,7,-4),row(31,7,-3),row(32,7,1),
                row(4,8,-3),row(5,8,1),row(8,9,-4),row(9,9,-3),row(11,9,1)],float)
    S=[('synthetic-K',i) for i in range(len(R))]
    roster=generate(R,S,poly,32)
    assert roster['total_native_events']==4 and roster['eligible']==1
    q=next(q for q in roster['queries'] if q['native_id']==7 and q['run_index']==0)
    assert q['search_frames']==list(range(4,29)) and len(q['anchor_raw_indices'])==3
    assert any('empty_clipped_search_range' in q['ineligible_reasons'] for q in roster['queries'])
    # Date4 is26frames before next run: safe. Date5 is25: it merges oldrun1.
    safe=insertion_safety(R,S,q,row(4,7,-3),('synthetic-new',0),32)
    bridge=insertion_safety(R,S,q,row(5,7,-3),('synthetic-new',1),32)
    assert safe['safe'] and not bridge['safe'] and bridge['reason']=='other_run_membership_changed'
    center=np.asarray(q['center']);r=q['radius']
    b=[center[0]+r-1,center[1]-1,2,2]
    assert in_corridor(q,4,b) and not in_corridor(q,29,b)
    assert not in_corridor(q,4,[b[0]+1e-8,b[1],2,2])
    # Exactly25 remains a single run, strictly26 splits.
    A=np.array([row(1,2,-3),row(26,2,1)],float)
    assert len(prepare(A,[('a',0),('a',1)],30)['runs'])==1
    A[1,0]=27;assert len(prepare(A,[('a',0),('a',1)],30)['runs'])==2
    try:generate(R,S[:-1]+[S[0]],poly,32)
    except ValueError:pass
    else:raise AssertionError('source duplicate accepted')
    # Same-frame alias collision must fail, never be disambiguated by geometry.
    alias=np.array([row(1,1,-3),row(30,1,1),row(30,100001,10)],float)
    try:prepare(alias,[('a',i) for i in range(3)],40)
    except ValueError:pass
    else:raise AssertionError('alias collision accepted')
    return dict(status='PASS synthetic only',native_events=4,eligible=1,
                tests=['all events retained','3distinct anchors','last2 consecutive','25vs26 gap',
                       'fixed inclusive4L disk','clipped empty dates','exact source index provenance',
                       'safe date4 vs bridge date5','source/alias collision refusal'],
                field_gate='No field CLI. Requires future published preregistration and root GO.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--self-test',action='store_true',required=True)
    a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()));print(json.dumps(self_test(),indent=2))
