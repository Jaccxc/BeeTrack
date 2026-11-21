"""Independent 206 ownership arithmetic, using the accepted 203 GT cache."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import numpy as np
import cv2
import verify206 as core

_SCENES={}


def scene(raw,sources,replay):
    key=id(raw)
    if key not in _SCENES:
        C,origin=replay.canonical(raw);dates={}
        order=np.argsort(C[:,0],kind='stable');values,starts=np.unique(C[order,0],return_index=True)
        for date,lo,hi in zip(values,starts,[*starts[1:],len(order)]):dates[int(date)]=order[lo:hi]
        _SCENES[key]=dict(raw=raw,C=C,origin=origin,dates=dates,owners={})
    result=_SCENES[key];assert result['raw'] is raw;return result


def original_evidence(raw,sources,needed,gt,polygon,threshold,replay,previous,v):
    """Frame factorization: every original box on the requested date is present."""
    data=scene(raw,sources,replay);cache=data['owners'].setdefault(str(threshold),{})
    dates={int(raw[i,0]) for i in needed if i not in cache}
    for date in sorted(dates):
        ii=data['dates'][date];indices=data['origin'][ii];keys=[sources[j] for j in indices]
        proof,_=replay.evidence_rows(data['C'][ii],keys,set(range(len(ii))),gt,polygon,threshold,previous,v)
        for j,native in enumerate(indices):cache[int(native)]=proof[j]
    return {i:cache[i] for i in needed}


def side(row,polygon):
    return cv2.pointPolygonTest(np.asarray(polygon,np.float32),tuple((row[2:4]+row[4:6]/2).tolist()),False)>=0


def geometry(raws,sources,G,gi,index,threshold,gt,polygon,replay,previous,v):
    row=G[index];date=int(row[0]);valid=bool(np.isfinite(row).all() and (row[4:6]>0).all());members=[]
    for slot,raw in enumerate(raws):
        raw=np.asarray(raw);prepared=scene(raw,sources[slot],replay);C,order=prepared['C'],prepared['origin'];jj=prepared['dates'].get(date,np.empty(0,int));P=C[jj]
        known=valid and bool(np.isfinite(P).all() and (P[:,4:6]>0).all())
        values=v.official_iou_independent(row[None,2:6],P[:,2:6])[0] if len(P) and known else np.zeros(len(P))
        hits=[int(order[jj[j]]) for j in np.flatnonzero(values>=threshold)]
        evidence=original_evidence(raw,sources[slot],set(hits),gt,polygon,threshold,replay,previous,v)
        members.append(dict(member_index=slot,scene_frame_rows=len(P),all_geometry_known=known,max_iou=float(values.max()) if len(values) else None,
            has_threshold_box=bool(hits),confirmed_same_run=any(x['qualified'] and x['official_owner']==int(row[1]) for x in evidence.values()),
            frame_rows=[dict(canonical_row=int(i),source_key=list(sources[slot][order[i]]),row=C[i].tolist(),iou=float(score)) for i,score in zip(jj,values)],
            threshold_neighbor_evidence=[evidence[i] for i in hits]))
    return dict(frame=date,GT_canonical_index=int(index),GT_raw_index=int(gi[index]),GT_row=row.tolist(),members=members,
        all_geometry_known=all(x['all_geometry_known'] for x in members),geometric_members=sum(x['has_threshold_box'] for x in members),
        confirmed_members=sum(x['confirmed_same_run'] for x in members),all_scene_frames_empty=all(not x['scene_frame_rows'] for x in members),
        all_no_threshold_box=all(x['all_geometry_known'] and not x['has_threshold_box'] for x in members))


def insertion_frames(raw,sources,query,newrow,newsource,gt,threshold,replay,v):
    prepared=scene(raw,sources,replay);C,origin=prepared['C'],prepared['origin'];indices=prepared['dates'].get(int(newrow[0]),np.empty(0,int));original=C[indices]
    added=np.asarray(newrow,float).copy();added[1]=query['canonical_id'];block=np.vstack((original,added));order=np.argsort(block[:,1],kind='stable');P=block[order]
    keys=[sources[origin[i]] for i in indices]+[newsource];keys=[keys[i] for i in order]
    before=replay.cached_owners(gt,original,threshold,set(range(len(original))),v)
    after=replay.cached_owners(gt,P,threshold,set(range(len(P))),v)
    owners=lambda record:[-1 if record['rows'][i]['owner'] is None else record['rows'][i]['owner'] for i in range(len(record['P']))]
    b,a=owners(before),owners(after);mapped={int(j):a[i] for i,j in enumerate(order) if j<len(original)}
    return dict(other_original_owners_unchanged=all(mapped[i]==owner for i,owner in enumerate(b)),original_frame_rows=original.tolist(),inserted_frame_rows=P.tolist(),
        original_owners=b,inserted_owners=a,order_from_original_plus_added=order.tolist(),source_keys=[list(k) for k in keys])


def analyze(query,raws,sources,gt,polygon,nf,video,gt_sha,threshold,replay,previous,v):
    """Returns an independent compact classification and first fixed witness."""
    slot=query['member_index'];raw=np.asarray(raws[slot]);anchorids=query['anchor_raw_indices']
    proofs=original_evidence(raw,sources[slot],set(anchorids),gt,polygon,threshold,replay,previous,v)
    cache=replay._GT_CACHE[id(gt)];anchors=[proofs[i] for i in anchorids];G,gi=cache['G'],cache['gi'];segments=cache['segments'][tuple(map(tuple,polygon))]
    out=dict(confirmed=False,candidate=None,enumerated=[],classification='unknown',reason=None,anchor_rows=anchors)
    if not query['eligible']:out.update(classification='ineligible',reason=';'.join(query['ineligible_reasons']));return out
    if not anchors or not all(p['qualified'] for p in anchors):out['reason']='anchors_incomplete_or_competing_owner';return out
    if len({p['official_owner'] for p in anchors})!=1:
        out['reason']='anchors_'+('complete_distinct_original_GT_IDs' if len({p['original_gt_id'] for p in anchors})>1 else 'same_original_ID_distinct_gap_runs');return out
    pid=anchors[0]['official_owner'];seg=segments[pid];ii=np.flatnonzero(G[:,1]==pid);R=G[ii]
    events=replay.track_events(R,polygon)
    out.update(GT_run_key=[video,seg['original_gt_id'],seg.get('run')],canonical_gt_id=pid,gt_run_first=seg['first_frame'],gt_run_last=seg['last_frame'],
        GT_events=events,missing_internal_frames=int(R[-1,0]-R[0,0]+1-len(R)),at_clip_start=seg['first_frame']==1,at_clip_end=seg['last_frame']==nf)
    if any(p['pred_inside']!=p['gt_inside'] for p in anchors):out['reason']='anchor_side_disagreement';return out
    if out['at_clip_start'] or out['at_clip_end']:out['reason']='GT_run_clip_truncated';return out
    if not (R[:,4:6]>0).all():out['reason']='GT_nonpositive_geometry';return out
    if events:out.update(classification='interpretable_nonzero_GT_event',reason='complete_observed_GT_run_has_net_event');return out
    returns=[int(i) for i in ii if G[i,0]>query['endpoint_frame'] and side(G[i],polygon)==query['starting_side']]
    returns.sort(key=lambda i:(G[i,0],int(gi[i])))
    if not returns:out.update(classification='interpretable_no_return',reason='no_later_same_run_start_side_observation');return out
    out.update(classification='return_evidence_no_legal_candidate',reason='all_return_rows_rejected')
    for index in returns:
        row=G[index];record=dict(GT_canonical_index=index,GT_raw_index=int(gi[index]),frame=int(row[0]),reason=None);out['enumerated'].append(record)
        if not core.in_fixed_search(query,row):record['reason']='outside_fixed_time_or_disk';continue
        geo=geometry(raws,sources,G,gi,index,threshold,gt,polygon,replay,previous,v);record['geometry']=geo
        if not geo['all_geometry_known']:record['reason']='unknown_scene_geometry';continue
        if not geo['all_no_threshold_box']:record['reason']='one_or_more_members_has_threshold_box';continue
        augmented,why=core.insert_native(raw,query,np.r_[row[0],gt[gi[index],1],row[2:6]],polygon,nf)
        if augmented is None:record['reason']=why;continue
        newsource=['GT:'+gt_sha,int(gi[index])]
        augmented_keys=list(sources[slot])+[newsource]
        evidence,_=replay.evidence_rows(augmented,augmented_keys,set(anchorids+[len(raw)]),gt,polygon,threshold,previous,v)
        required=[evidence[j] for j in anchorids+[len(raw)]]
        if not all(p['qualified'] and p['official_owner']==pid and p['pred_inside']==p['gt_inside'] for p in required):record['reason']='inserted_row_not_isolated_same_run';continue
        record['inserted_frames']=insertion_frames(raw,sources[slot],query,augmented[-1],newsource,gt,threshold,replay,v)
        record['reason']='first_legal_confirmed_return'
        out.update(confirmed=True,classification='confirmed_legal_return',reason=record['reason'],candidate=dict(query_id=query['query_id'],frame=int(row[0]),GT_raw_index=int(gi[index]),GT_canonical_index=index,
            GT_row_key=[video,seg['original_gt_id'],seg.get('run'),int(row[0]),int(gi[index])],added_row=augmented[-1].tolist(),source_key=newsource,
            deleted_event=query['event'],delta_windows=(-core.bins([(query['event']['frame'],query['event']['direction'])],nf)).tolist(),strict_no_GT_internal_gaps=bool(np.all(np.diff(R[:,0])==1)),required_rows=required))
        break
    return out


def joint(raw,sources,items,gt,polygon,nf,threshold,replay,previous,v):
    """Registered qualification only; unrelated Hungarian owners are diagnostic."""
    raw=np.asarray(raw,float)
    keys=[tuple(item['candidate']['source_key']) for item in items]
    if len(keys)!=len(set(keys)):return dict(safe=False,reason='same_member_GT_row_reused')
    additions=np.asarray([item['candidate']['added_row'] for item in items],float).reshape(-1,7)
    combined=np.vstack((raw,additions));combined_keys=list(sources)+[list(k) for k in keys]
    try:old=core.native_membership(raw);new=core.native_membership(combined)
    except AssertionError:return dict(safe=False,reason='combined_native_conflict')
    if any(new[i]!=alias for i,alias in old.items()):return dict(safe=False,reason='combined_changes_original_run_alias')
    wanted=set()
    for i,item in enumerate(items):
        q=item['query']
        if new[len(raw)+i]!=q['canonical_id']:return dict(safe=False,reason='combined_added_row_wrong_run')
        wanted.update(q['anchor_raw_indices']);wanted.add(len(raw)+i)
    evidence,_=replay.evidence_rows(combined,combined_keys,wanted,gt,polygon,threshold,previous,v)
    for i,item in enumerate(items):
        q=item['query'];pid=item['canonical_gt_id']
        required=[evidence[j] for j in q['anchor_raw_indices']+[len(raw)+i]]
        if not all(x['qualified'] and x['official_owner']==pid and x['pred_inside']==x['gt_inside'] for x in required):return dict(safe=False,reason='combined_anchor_or_added_owner_not_unique_same_run')
    removed={(item['query']['native_id'],item['query']['run_index']) for item in items}
    before={(p,r):(f,s) for p,r,_,_,f,s in core.signed_events(raw,polygon)}
    after={(p,r):(f,s) for p,r,_,_,f,s in core.signed_events(combined,polygon)}
    if {k:value for k,value in before.items() if k not in removed}!=after:return dict(safe=False,reason='combined_not_exact_selected_event_deletions')
    return dict(safe=True,reason='registered_joint_qualification',rows=combined,source_keys=combined_keys,
        numerator_event_delta=(-sum((core.bins([(item['query']['event']['frame'],item['query']['event']['direction'])],nf) for item in items),np.zeros_like(core.bins([],nf)))).tolist())


def coverage(records,raws,sources,gt,polygon,nf,video,replay,previous,v):
    """Every canonical GT run is in the denominator, including no-return runs."""
    first_scene,_=replay.canonical(raws[0]);owners=replay.cached_owners(gt,first_scene,.5,set(),v)
    G,gi=owners['G'],owners['gt_origin'];segments=v.GT_segments(owners,polygon);out=[]
    for pid,seg in segments.items():
        ii=np.flatnonzero(G[:,1]==pid);rows=G[ii];dates=list(map(int,rows[:,0]));sides=[side(row,polygon) for row in rows]
        probe=replay.first_return(dates,sides)
        item=dict(canonical_gt_id=pid,original_gt_id=seg['original_gt_id'],gt_run=seg.get('run'),probe=probe,thresholds={})
        if probe is not None:
            index=int(ii[probe['return_index']]);row=G[index]
            for threshold in (.5,.3):
                names=('all_query_time_space_ids','eligible_query_time_space_ids','same_run_anchor_query_ids','same_run_confirmed_query_ids','exact_probe_first_candidate_query_ids')
                proof={name:[] for name in names};proof['geometry']=geometry(raws,sources,G,gi,index,threshold,gt,polygon,replay,previous,v)
                proof['time_space_known']=bool(np.isfinite(row).all() and (row[4:6]>0).all())
                if not proof['time_space_known']:
                    proof.update({name:None for name in names});item['thresholds'][str(threshold)]=proof;continue
                for q in records:
                    if not core.in_fixed_search(q,row):continue
                    proof[names[0]].append(q['query_id'])
                    if q['eligible']:proof[names[1]].append(q['query_id'])
                    label=q['labels'][str(threshold)]
                    if label.get('canonical_gt_id')==pid:proof[names[2]].append(q['query_id'])
                    if label['confirmed'] and label.get('canonical_gt_id')==pid:
                        proof[names[3]].append(q['query_id'])
                        if label['candidate']['GT_canonical_index']==index:proof[names[4]].append(q['query_id'])
                item['thresholds'][str(threshold)]=proof
        out.append(item)
    return out
