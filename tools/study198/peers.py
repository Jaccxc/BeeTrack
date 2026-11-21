"""GT-free peer identity evidence for frozen endpoint candidates; pure APIs.

Core identities are chosen on exactly three core dates, never short dates.
Missing boxes never become identity votes. No source field access on import.
"""
from collections import defaultdict
import cv2
import numpy as np
from beetrack.evaluate import by_track


def iou(box,others):
    """Finite float64 xywh IoU, exact union division, invalid dimensions→0."""
    box=np.asarray(box,float);others=np.asarray(others,float).reshape(-1,4)
    if not len(others):return np.empty(0,float)
    if not np.isfinite(box).all() or np.any(box[2:]<=0):return np.zeros(len(others))
    valid=np.isfinite(others).all(1)&(others[:,2:]>0).all(1)
    overlap=np.maximum(0,np.minimum(box[:2]+box[2:],others[:,:2]+others[:,2:])-np.maximum(box[:2],others[:,:2]))
    inter=overlap.prod(1);union=box[2]*box[3]+others[:,2]*others[:,3]-inter
    return np.divide(inter,union,out=np.zeros(len(others)),where=valid&(union>0))


def prepare_peer(raw,polygon,dates=None):
    a=np.asarray(raw,dtype=float)
    if a.ndim!=2 or a.shape[1]<6 or not np.isfinite(a).all():raise ValueError('finite peer MOT required')
    if np.any(a[:,:2]!=np.rint(a[:,:2])) or np.any(a[:,0]<1) or np.any(a[:,1]<0):raise ValueError('peer dates/IDs invalid')
    rows=by_track(a)
    if dates is not None:rows=rows[np.isin(rows[:,0],list(dates))]
    indexed=defaultdict(dict)
    poly=np.asarray(polygon,np.float32)
    for row in rows:
        f,pid=int(row[0]),int(row[1])
        if pid in indexed[f]:raise ValueError('peer canonical frame/ID collision')
        valid=bool(np.all(row[4:6]>0))
        center=row[2:4]+row[4:6]/2
        side=bool(cv2.pointPolygonTest(poly,(float(center[0]),float(center[1])),False)>=0) if valid else None
        indexed[f][pid]=dict(box=row[2:6].copy(),side=side,valid=valid)
    return dict(rows=rows,dates=dict(indexed))


def _at_date(peer,date,source_box):
    boxes=peer['dates'].get(int(date),{})
    eligible=[pid for pid,b in boxes.items() if b['valid']]
    scores=iou(source_box,[boxes[p]['box'] for p in eligible])
    ordered=sorted(zip(eligible,scores),key=lambda z:(-float(z[1]),z[0]))
    return boxes,ordered


def compare_peer(short,core,peer):
    """short/core are own canonical rows in inward order; returns compact."""
    if len(core)!=3 or len(short) not in (1,2):raise ValueError('fixed k=1/2 and3core observations required')
    dates=[int(r[0]) for r in core]
    ids=sorted(set(pid for f in dates for pid in peer['dates'].get(f,{})))
    qualified=[]
    for pid in ids:
        values=[]
        for row in core:
            entry=peer['dates'].get(int(row[0]),{}).get(pid)
            values.append(float(iou(row[2:6],[entry['box']])[0]) if entry is not None and entry['valid'] else 0.0)
        if sum(v>=.5 for v in values)>=2:
            qualified.append(dict(track_id=int(pid),iou=values,mean=float(np.mean(values))))
    qualified.sort(key=lambda x:(-x['mean'],x['track_id']))
    best=qualified[0] if qualified else None
    second=qualified[1] if len(qualified)>1 else None
    margin=float(best['mean']-(second['mean'] if second else 0)) if best else 0.0
    identified=bool(best and margin>=.10)
    fixed=best['track_id'] if identified else None
    short_records=[]
    for own in short:
        boxes,ordered=_at_date(peer,int(own[0]),own[2:6])
        winner=ordered[0] if ordered else None
        nextbest=ordered[1] if len(ordered)>1 else None
        entry=boxes.get(fixed) if fixed is not None else None
        fixed_iou=float(iou(own[2:6],[entry['box']])[0]) if entry is not None and entry['valid'] else None
        short_records.append(dict(frame=int(own[0]),any_best_id=int(winner[0]) if winner else None,
             any_iou=float(winner[1]) if winner else 0.0,any_second_id=int(nextbest[0]) if nextbest else None,
             any_second_iou=float(nextbest[1]) if nextbest else 0.0,any_side=boxes[winner[0]]['side'] if winner else None,
             observable=bool(winner and winner[1]>=.5),missing=not bool(ordered),
             fixed_core_id=fixed,core_row_present=entry is not None,core_valid=bool(entry and entry['valid']),
             core_iou=fixed_iou,core_side=entry['side'] if entry is not None else None))
    free_core=[]
    for row in core:
        boxes,ordered=_at_date(peer,int(row[0]),row[2:6]);winner=ordered[0] if ordered else None
        free_core.append(dict(frame=int(row[0]),best_id=int(winner[0]) if winner else None,
                              iou=float(winner[1]) if winner else 0.0,side=boxes[winner[0]]['side'] if winner else None,
                              missing=not bool(ordered)))
    return dict(core_dates=dates,qualified=qualified,best_id=best['track_id'] if best else None,
                best_mean=best['mean'] if best else 0.0,second_id=second['track_id'] if second else None,
                second_mean=second['mean'] if second else 0.0,margin=margin,identified=identified,
                identified_id=fixed,short=short_records,free_core=free_core,
                short_observable=any(s['observable'] for s in short_records))


def peer_evidence(candidate,member_bundle,peer_rows,polygon):
    """Two peers may be raw MOT arrays or prepare_peer results, no GT fields."""
    if len(peer_rows)!=2:raise ValueError('exactly two other members required')
    window=member_bundle['canonical_rows'][candidate['window_indices']]
    k=candidate['k'];short=window[:k];core=window[k:k+3]
    peers=[p if isinstance(p,dict) and 'dates' in p else prepare_peer(p,polygon) for p in peer_rows]
    evidence=[compare_peer(short,core,p) for p in peers]
    gate_same=any(p['identified'] and p['short_observable'] for p in evidence)
    gate_any=any(p['identified'] for p in evidence) and any(p['short_observable'] for p in evidence)
    available=[p for p in evidence if p['identified'] and p['short_observable']]
    extra_u=[];extra_i=[]
    for p in available:
        obs=[s for s in p['short'] if s['observable']]
        core_side=bool(candidate['core_inside']);short_side=bool(candidate['short_inside'])
        extra_u.append([float(np.mean([x['iou'] for x in p['free_core']])),len(obs)/k,
                        float(np.mean([x['any_iou'] for x in obs])),float(np.mean([x['any_side']==core_side for x in obs]))])
        extra_i.append([p['best_mean'],p['margin'],float(np.mean([x['core_row_present'] for x in obs])),
                        float(np.mean([x['core_iou'] if x['core_iou'] is not None else 0 for x in obs])),
                        float(np.mean([(x['core_iou'] or 0)>=.5 and x['core_side']==short_side for x in obs])),
                        float(np.mean([(x['core_iou'] or 0)>=.5 and x['core_side']==core_side for x in obs])),
                        float(np.mean([x['any_best_id']==p['identified_id'] for x in obs]))])
    U4=np.mean(extra_u,axis=0).tolist() if extra_u else [0.0]*4
    I7=np.mean(extra_i,axis=0).tolist() if extra_i else [0.0]*7
    C11=list(candidate['features_G'])+[float(candidate['direction']==1),len(available)/2]
    frame_only=[any(p['short'][j]['observable'] for p in evidence) for j in range(k)]
    frame_only_peers=[sum(x['iou']>=.5 for x in p['free_core'])>=2 and p['short_observable'] for p in evidence]
    return dict(candidate_id=candidate['candidate_id'],parent_id=candidate['parent_id'],k=k,
                short_dates=[int(r[0]) for r in short],core_dates=[int(r[0]) for r in core],
                peers=evidence,gate=gate_same,gate_same_peer=gate_same,gate_any_peer=gate_any,
                eligible=gate_same,qualifies=gate_same,ablation_eligible=gate_same,
                usable_peer_count=len(available),total_short_frames=k,frame_only_observable=frame_only,
                frame_only_observable_count=sum(frame_only),frame_only_peers=frame_only_peers,
                frame_only_gate=any(frame_only_peers),C11=C11,U4=U4,I7=I7,
                features_N11=C11,features_B15=C11+U4,features_I22=C11+U4+I7)


def self_test():
    row=lambda f,p,x:[f,p,x,4,4,4,.8]
    own=np.array([row(f,1,x) for f,x in enumerate((0,4,5,6),1)],float)
    candidate=dict(candidate_id='synthetic',parent_id=1,k=1,window_indices=[0,1,2,3],core_inside=False,short_inside=True,
                   features_G=[0.0]*9,direction=1)
    bundle=dict(canonical_rows=own)
    poly=[[-1,-1],[3,-1],[3,20],[-1,20]]
    # Identical boxes; short same ID vs short changed ID. ID-free evidence equal.
    consistent=own.copy();consistent[:,1]=9
    switched=consistent.copy();switched[0,1]=11
    a=peer_evidence(candidate,bundle,[consistent,consistent],poly)
    b=peer_evidence(candidate,bundle,[switched,switched],poly)
    assert a['gate'] and b['gate'] and a['peers'][0]['identified_id']==9
    assert a['peers'][0]['short'][0]['core_iou']==1 and b['peers'][0]['short'][0]['core_iou'] is None
    assert a['peers'][0]['short'][0]['any_iou']==b['peers'][0]['short'][0]['any_iou']==1
    assert a['peers'][0]['free_core']==b['peers'][0]['free_core']
    assert a['U4']==b['U4'] and a['I7']!=b['I7']
    assert len(a['features_N11'])==11 and len(a['features_B15'])==15 and len(a['features_I22'])==22
    # Missing short frame is missing, not opposite identity evidence.
    c=peer_evidence(candidate,bundle,[consistent[1:],consistent[1:]],poly)
    assert not c['gate'] and c['peers'][0]['short'][0]['missing']
    assert c['U4']==[0.0]*4 and c['I7']==[0.0]*7
    # Exact duplicate eligible core tracks are ambiguous (zero margin).
    duplicate=consistent.copy();duplicate[:,1]=10
    ambiguous=np.r_[consistent,duplicate]
    d=peer_evidence(candidate,bundle,[ambiguous,ambiguous],poly)
    assert not d['gate'] and d['peers'][0]['best_id']==9 and d['peers'][0]['margin']==0
    cross=peer_evidence(candidate,bundle,[consistent[1:],ambiguous],poly)
    assert cross['gate_any_peer'] and not cross['eligible'] and cross['I7']==[0.0]*7
    # IDs changing independently across core dates cannot identify a stable core.
    transient=consistent.copy();transient[:,1]=[11,12,13,14]
    e=peer_evidence(candidate,bundle,[transient,transient],poly)
    assert not e['gate'] and all(x['iou']==1 for x in e['peers'][0]['free_core'])
    assert e['frame_only_gate']
    own2=np.array([row(f,1,x) for f,x in enumerate((0,0,4,5,6),1)],float)
    cp=dict(candidate,k=2,window_indices=[0,1,2,3,4]);bp=dict(canonical_rows=own2)
    partly=own2.copy();partly[:,1]=9;partly[1,1]=11
    pair=peer_evidence(cp,bp,[partly,partly],poly)
    np.testing.assert_allclose(pair['U4'],[1,1,1,0],rtol=0,atol=1e-15)
    np.testing.assert_allclose(pair['I7'],[1,1,.5,.5,.5,0,.5],rtol=0,atol=1e-15)
    print('Peer evidence synthetic stable/switched IDs, ID-free equal boxes, missing and ambiguous core passed')


if __name__=='__main__':self_test()
