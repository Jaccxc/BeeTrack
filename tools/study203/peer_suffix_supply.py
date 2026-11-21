"""Unregistered synthetic/source-only forward peer suffix supply.

No GT, freezer, optimizer or field loader. Mutual unique observations are not
proof that the peer retains physical identity beyond the two anchor frames.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
from peer_suffix_prototype import iou,mutual,validate


def _member(member,nframes):
    from beetrack.evaluate import by_track
    rows=np.asarray(member['rows'],float).reshape(-1,7);validate(rows.tolist())
    sources=[tuple(s) for s in member['sources']]
    if len(sources)!=len(rows) or len(set(sources))!=len(sources):raise ValueError('source cardinality/duplicate')
    for s in sources:
        if len(s)!=2 or not isinstance(s[0],str) or not s[0] or int(s[1])!=s[1] or s[1]<0:raise ValueError('source SHA/index required')
    if ((rows[:,0]<1)|(rows[:,0]>nframes)).any():raise ValueError('frame outside clip')
    aliases=set()
    for pid in np.unique(rows[:,1]):
        dates=np.sort(rows[rows[:,1]==pid,0])
        for run in range(1+int(np.sum(np.diff(dates)>25))):
            alias=int(pid)+100000*run
            if alias in aliases:raise ValueError('canonical run alias collision')
            aliases.add(alias)
    tagged=by_track(np.column_stack((rows,np.arange(len(rows)))))
    index=tagged[:,-1].astype(int);canonical=tagged[:,:7]
    # Aliased raw IDs must not collapse different rows at a single date.
    if len(set(map(tuple,canonical[:,:2])))!=len(rows):raise ValueError('canonical run alias collision')
    tracks={int(pid):np.flatnonzero(canonical[:,1]==pid) for pid in np.unique(canonical[:,1])}
    dates={int(f):[tuple(r) for r in canonical[canonical[:,0]==f]] for f in np.unique(canonical[:,0])}
    return dict(rows=rows,canonical=canonical,index=index,tracks=tracks,dates=dates,sources=sources)


def catalog(members,polygon,nframes):
    """Exactly three native members with parallel (detector SHA, D index).

    Return every qualifying host/peer alternative separately, longest observed
    forward-only consecutive suffix up to25 frames. No prefix or interpolation.
    """
    from tools.study197.candidates import event_times,window_counts
    if len(members)!=3 or len({str(m['member']) for m in members})!=3:raise ValueError('three distinct members required')
    data=[_member(m,nframes) for m in members];result=[]
    for hi,host in enumerate(data):
        for hid,ix in host['tracks'].items():
            track=host['canonical'][ix]
            if len(track)<3 or track[-1,0]-track[-2,0]!=1:continue
            f=int(track[-1,0]);dates=[f-1,f]
            anchor=[tuple(r) for r in track[-2:]]
            for pi,peer in enumerate(data):
                if pi==hi:continue
                for pid,pix in peer['tracks'].items():
                    donor=peer['canonical'][pix]
                    if len(donor)<3:continue
                    bydate={int(peer['canonical'][j,0]):j for j in pix}
                    if any(d not in bydate for d in dates):continue
                    pa=[tuple(peer['canonical'][bydate[d]]) for d in dates]
                    if not all(mutual(a,b,host['dates'][d],peer['dates'][d]) for d,a,b in zip(dates,anchor,pa)):continue
                    suffix=[];stop='limit25'
                    for d in range(f+1,min(f+25,nframes)+1):
                        if d not in bydate:stop='peer_date_missing';break
                        j=bydate[d];row=tuple(peer['canonical'][j])
                        if any(iou(row,r)>=.5 for r in host['dates'].get(d,[])):stop='host_overlap';break
                        native=int(peer['index'][j]);suffix.append(dict(peer_row_index=native,
                            source=list(peer['sources'][native]),frame=d,peer_track_id=pid,
                            original_peer_id=int(peer['rows'][native,1]),row=list(row),box=list(row[2:6])))
                    if not suffix:continue
                    extension=np.array([s['row'] for s in suffix]);extension[:,1]=hid
                    before=event_times(track,polygon);after=event_times(np.vstack((track,extension)),polygon)
                    delta=window_counts(after,nframes)-window_counts(before,nframes)
                    result.append(dict(candidate_id=f'{members[hi]["member"]}|{hid}|{members[pi]["member"]}|{pid}|{f}',
                        host_member=str(members[hi]['member']),peer_member=str(members[pi]['member']),
                        host_track_id=hid,peer_track_id=pid,anchor_frames=dates,
                        host_anchor_row_indices=[int(host['index'][j]) for j in ix[-2:]],
                        host_anchor_rows=[list(a) for a in anchor],peer_anchor_rows=[list(a) for a in pa],
                        host_anchor_sources=[list(host['sources'][int(host['index'][j])]) for j in ix[-2:]],
                        peer_anchor_sources=[list(peer['sources'][int(peer['index'][bydate[d]])]) for d in dates],
                        suffix=suffix,stop=stop,events_before=before,events_after=after,delta_windows=delta.tolist(),
                        net_event_delta=len(after)-len(before),direction='forward_after_only'))
    return sorted(result,key=lambda c:c['candidate_id'])


def materialize(members,candidates,selected_ids):
    """Validate global source/host capacities and append native measurements.

    Also reject any same-date overlap>=.5 between selected additions to the same
    output member. Each original K row and ID remains bit-for-bit unchanged.
    """
    if len(set(selected_ids))!=len(selected_ids):raise ValueError('duplicate selection')
    table={c['candidate_id']:c for c in candidates}
    output={str(m['member']):np.asarray(m['rows'],float).reshape(-1,7).copy() for m in members}
    used=set();hosts=set();additions={key:[] for key in output};provenance={key:[] for key in output}
    for cid in sorted(selected_ids):
        c=table[cid];key=c['host_member'];host=(key,c['host_track_id'])
        if host in hosts:raise ValueError('host track has more than one continuation')
        hosts.add(host)
        # Canonical run alias maps back to the existing native ID at anchor.
        original=output[key];anchor_index=c['host_anchor_row_indices'][-1]
        anchor=original[anchor_index];expected=np.asarray(c['host_anchor_rows'][-1])
        if not np.array_equal(anchor[[0,2,3,4,5,6]],expected[[0,2,3,4,5,6]]):raise ValueError('native host endpoint index drift')
        native_id=anchor[1]
        for s in c['suffix']:
            source=tuple(s['source'])
            if source in used:raise ValueError('same donor source added twice')
            used.add(source);r=np.array(s['row'],float);r[1]=native_id
            if any(old[0]==r[0] and iou(old,r)>=.5 for old in additions[key]):raise ValueError('selected additions overlap')
            if any(old[0]==r[0] and old[1]==r[1] for old in original):raise ValueError('track/date collision')
            additions[key].append(r);provenance[key].append(dict(source=list(source),candidate_id=cid))
    for key in output:
        base=output[key]
        output[key]=np.vstack((base,np.asarray(additions[key],float).reshape(-1,7)))
        assert np.array_equal(output[key][:len(base)],base)
        validate(output[key].tolist())
    return output,provenance


def self_test():
    def row(f,p,x):return [f,p,x,0,4,4,.9]
    host=[row(f,1,x) for f,x in [(747,-6),(748,-5),(749,-4)]]
    peer=host+[row(f,1,x) for f,x in [(750,-3),(751,-2),(752,-1),(753,1)]]
    def member(k,r):return dict(member=k,rows=r,sources=[('synthetic-'+k,i) for i in range(len(r))])
    members=[member('H',host),member('P',peer),member('Q',peer)]
    poly=[(0,-10),(100,-10),(100,100),(0,100)]
    cs=catalog(members,poly,800);hc=[c for c in cs if c['host_member']=='H']
    assert len(hc)==2 and [s['frame'] for s in hc[0]['suffix']]==[750,751,752,753]
    assert hc[0]['net_event_delta']==1 and np.sum(np.abs(hc[0]['delta_windows']))==30
    out,proof=materialize(members,cs,[hc[0]['candidate_id']]);assert len(out['H'])==7
    try:materialize(members,cs,[c['candidate_id'] for c in hc])
    except ValueError:pass
    else:raise AssertionError('two extensions same host')
    missing=[member('H',host),member('P',[r for r in peer if r[0]!=751]),member('Q',peer)]
    gap=[c for c in catalog(missing,poly,800) if c['host_member']=='H' and c['peer_member']=='P'][0]
    assert [s['frame'] for s in gap['suffix']]==[750] and gap['stop']=='peer_date_missing'
    blocked=[member('H',host+[row(751,8,-2)]),member('P',peer),member('Q',peer)]
    block=[c for c in catalog(blocked,poly,800) if c['host_member']=='H'][0]
    assert [s['frame'] for s in block['suffix']]==[750] and block['stop']=='host_overlap'
    longpeer=peer+[row(f,1,1) for f in range(754,781)]
    limited=[c for c in catalog([member('H',host),member('P',longpeer),member('Q',peer)],poly,800)
             if c['host_member']=='H' and c['peer_member']=='P'][0]
    assert len(limited['suffix'])==25 and limited['suffix'][-1]['frame']==774
    # Same raw detector row cannot be newly copied twice, even to different hosts.
    import copy
    duplicate=copy.deepcopy(hc[0]);duplicate['candidate_id']='copy';duplicate['host_member']='Q'
    duplicate['host_track_id']=1;duplicate['host_anchor_rows']=copy.deepcopy(hc[0]['host_anchor_rows'])
    try:materialize(members,[hc[0],duplicate],[hc[0]['candidate_id'],'copy'])
    except ValueError as exc:assert 'same donor source' in str(exc)
    else:raise AssertionError('donor source duplicated')
    from tools.study197.candidates import window_counts
    # Equal single events at749 versus753 occupy different edge-window bins.
    a=window_counts([dict(frame=749,direction=1)],800)
    b=window_counts([dict(frame=753,direction=1)],800)
    assert int(np.abs(a-b).sum())==2
    return dict(status='PASS synthetic only',separate_peer_alternatives=len(hc),
                chosen_dates=[750,751,752,753],date_gap_stops_at=750,boundary_749_753_L1=2,
                scope='forward only; no GT, model, solver, freeze or field read')


if __name__=='__main__':
    import argparse,json,sys
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--self-test',action='store_true',required=True)
    a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()));print(json.dumps(self_test(),indent=2))
