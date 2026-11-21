"""Unregistered GT-free203 freeze; source/synthetic until published GO.

Only202sealed native baseline K and its original detector row lineage are used.
No natural GT, detector, model, reserve seed, GPU or external source is opened.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from peer_suffix_supply import catalog as reference_catalog,materialize
from peer_suffix_prototype import iou,mutual,validate

MARKER='[peer-suffix-observed-v1]'
SOURCE202='7261c2573773a23342286a39ce712950cb986aa6f34248cd12f5705ab420f469'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def save(path,obj):Path(path).write_text(json.dumps(plain(obj),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')


def verify(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('manifest seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError('artifact drift '+name)
    return manifest


def prepare(member,nf):
    """One pass of frame/ID indexing instead of scanning full tables per date."""
    from tools.study197.physical import canonicalize
    rows=np.asarray(member['rows'],float).reshape(-1,7);validate(rows.tolist())
    sources=[tuple(s) for s in member['sources']]
    if len(sources)!=len(rows) or len(set(sources))!=len(sources):raise ValueError('source cardinality/uniqueness')
    if any(len(s)!=2 or not isinstance(s[0],str) or not s[0] or int(s[1])!=s[1] or s[1]<0 for s in sources):raise ValueError('invalid source key')
    if ((rows[:,0]<1)|(rows[:,0]>nf)).any():raise ValueError('outside clip')
    b=canonicalize(rows);canonical=b['canonical_rows'];index=b['source_raw_index']
    tracks={};dates=defaultdict(list);lookup={}
    for block in b['parent_blocks']:
        tracks[block['parent_id']]=np.arange(block['start'],block['end'])
    for j,r in enumerate(canonical):
        dates[int(r[0])].append(tuple(r));lookup[(int(r[0]),int(r[1]))]=j
    return dict(rows=rows,canonical=canonical,index=index,tracks=tracks,dates=dates,lookup=lookup,sources=sources)


def indexed_catalog(members,polygon,nf,prepared=None,host_filter=None):
    """Numerically identical to supply.catalog, only frame-overlap IDs inspected."""
    from tools.study197.candidates import event_times,window_counts
    if len(members)!=3 or len({str(m['member']) for m in members})!=3:raise ValueError('three distinct members')
    data=[prepare(m,nf) for m in members] if prepared is None else prepared;result=[]
    for hi,host in enumerate(data):
        if host_filter is not None and hi!=host_filter[0]:continue
        track_items=host['tracks'].items() if host_filter is None else [(host_filter[1],host['tracks'][host_filter[1]])]
        for hid,ix in track_items:
            track=host['canonical'][ix]
            if len(track)<3 or track[-1,0]-track[-2,0]!=1:continue
            f=int(track[-1,0]);frames=[f-1,f];ha=[tuple(r) for r in track[-2:]]
            for pi,peer in enumerate(data):
                if pi==hi:continue
                # Every valid donor must overlap the last anchor. Uniqueness is
                # still checked against ALL rows in each full anchor frame.
                possibilities=[b for b in peer['dates'].get(f,[]) if iou(ha[-1],b)>=.5]
                if len(possibilities)!=1:continue
                pid=int(possibilities[0][1]);pix=peer['tracks'][pid]
                if len(pix)<3 or (f-1,pid) not in peer['lookup']:continue
                pa=[tuple(peer['canonical'][peer['lookup'][(d,pid)]]) for d in frames]
                if not all(mutual(a,b,host['dates'][d],peer['dates'][d]) for d,a,b in zip(frames,ha,pa)):continue
                suffix=[];stop='limit25'
                for d in range(f+1,min(f+25,nf)+1):
                    if (d,pid) not in peer['lookup']:stop='peer_date_missing';break
                    j=peer['lookup'][(d,pid)];row=tuple(peer['canonical'][j])
                    if any(iou(row,r)>=.5 for r in host['dates'].get(d,[])):stop='host_overlap';break
                    native=int(peer['index'][j]);suffix.append(dict(peer_row_index=native,source=list(peer['sources'][native]),frame=d,
                        peer_track_id=pid,original_peer_id=int(peer['rows'][native,1]),row=list(row),box=list(row[2:6])))
                if not suffix:continue
                ext=np.array([s['row'] for s in suffix]);ext[:,1]=hid
                before=event_times(track,polygon);after=event_times(np.vstack((track,ext)),polygon)
                result.append(dict(candidate_id=f'{members[hi]["member"]}|{hid}|{members[pi]["member"]}|{pid}|{f}',
                    host_member=str(members[hi]['member']),peer_member=str(members[pi]['member']),host_track_id=hid,peer_track_id=pid,
                    anchor_frames=frames,host_anchor_row_indices=[int(host['index'][j]) for j in ix[-2:]],
                    host_anchor_rows=[list(a) for a in ha],peer_anchor_rows=[list(a) for a in pa],
                    host_anchor_sources=[list(host['sources'][int(host['index'][j])]) for j in ix[-2:]],
                    peer_anchor_sources=[list(peer['sources'][int(peer['index'][peer['lookup'][(d,pid)]])]) for d in frames],
                    suffix=suffix,stop=stop,events_before=before,events_after=after,
                    delta_windows=(window_counts(after,nf)-window_counts(before,nf)).tolist(),
                    net_event_delta=len(after)-len(before),direction='forward_after_only'))
    return sorted(result,key=lambda c:c['candidate_id'])


def enrich(candidates,members,context,source_indices=None):
    """Keep supply resource schema; add explicit slot/index aliases and audits."""
    slots={str(m['member']):i for i,m in enumerate(members)}
    if source_indices is None:source_indices=[{tuple(s):j for j,s in enumerate(m['sources'])} for m in members]
    for c in candidates:
        hi,pi=slots[c['host_member']],slots[c['peer_member']]
        donor=np.asarray(members[pi]['rows'],float)
        dates=c['anchor_frames'];pid=c['peer_track_id']
        donor_indices=[]
        for d,anchor in zip(dates,c['peer_anchor_rows']):
            # Stable197allocated IDs are canonical here; source index is exact.
            donor_indices.append(source_indices[pi][tuple(c['peer_anchor_sources'][len(donor_indices)])])
        c.update(context,action_id=c['candidate_id'],host_member_index=hi,donor_member_index=pi,
                 host_parent_id=c['host_track_id'],donor_parent_id=pid,
                 host_anchor_K_indices=c['host_anchor_row_indices'],donor_anchor_K_indices=donor_indices,
                 suffix_rows=[s['row'] for s in c['suffix']],suffix_source_keys=[s['source'] for s in c['suffix']],
                 donor_K_indices=[s['peer_row_index'] for s in c['suffix']],
                 donor_suffix_K_indices=[s['peer_row_index'] for s in c['suffix']],
                 insert_host_anchor_indices=c['host_anchor_row_indices'],
                 insert_suffix_indices=list(range(len(members[hi]['rows']),len(members[hi]['rows'])+len(c['suffix']))),
                 anchor_ious=[iou(a,b) for a,b in zip(c['host_anchor_rows'],c['peer_anchor_rows'])])
    unique={}
    for c in candidates:
        path=(c['host_member'],c['host_parent_id'],tuple(tuple(s['source']) for s in c['suffix']),
              tuple((s['frame'],*s['box'],s['row'][6]) for s in c['suffix']),
              tuple(map(tuple,c['peer_anchor_sources'])),
              tuple((r[0],*r[2:]) for r in c['peer_anchor_rows']))
        if path in unique:
            unique[path].setdefault('equivalent_action_ids',[]).append(c['action_id'])
            unique[path].setdefault('equivalent_actions',[]).append(dict(c))
        else:unique[path]=c
    return list(unique.values())


FEATURE_NAMES=('min_anchor_iou','mean_anchor_iou','first_extrapolation_error',
               'first_endpoint_distance','host_endpoint_speed','suffix_length_over25',
               'host_observations_capped_over25','host_score_change','host_polygon_distance','host_neighbor_count')


def features(c,members,polygon,host_count=None,scene_rows=None):
    import cv2
    a,b=np.asarray(c['host_anchor_rows'],float)
    first=np.asarray(c['suffix'][0]['row'],float)
    center=lambda r:r[2:4]+r[4:6]/2
    ca,cb,cf=center(a),center(b),center(first);diag=float(np.hypot(b[4],b[5]))
    rows=np.asarray(members[c['host_member_index']]['rows'] if scene_rows is None else scene_rows,float)
    if host_count is None:host_count=int(np.sum(rows[:,1]==c['host_parent_id']))
    at=rows[rows[:,0]==b[0]]
    neighbors=sum(np.linalg.norm(center(r)-cb)<=2*diag for r in at if int(r[1])!=int(b[1]))
    f=[min(c['anchor_ious']),float(np.mean(c['anchor_ious'])),float(np.linalg.norm(cf-(2*cb-ca))/diag),
       float(np.linalg.norm(cf-cb)/diag),float(np.linalg.norm(cb-ca)/diag),len(c['suffix'])/25,
       min(25,host_count)/25,float(b[6]-a[6]),
       abs(float(cv2.pointPolygonTest(np.asarray(polygon,np.float32),tuple(map(float,cb)),True)))/diag,float(neighbors)]
    if not np.isfinite(f).all():raise ValueError('nonfinite203features')
    from beetrack.evaluate import inside
    bits=[bool(inside(b[None],polygon)[0]),f[8]<=1,f[9]>0,f[7]<0,f[4]<=.5]
    c.update(features10=f,G_logit=2*f[0]-f[2],stratum_bits=bits,
             stratum=''.join(str(int(x)) for x in bits));return c


def host_contexts(members,polygon,nf,prepared,candidates,unit):
    from tools.study197.candidates import event_times,window_counts
    grouped=defaultdict(list);strata={}
    for c in candidates:
        key=(c['host_member_index'],c['host_parent_id']);grouped[key].append(c['action_id'])
        strata[key]=c['stratum']
    result=[]
    for slot,data in enumerate(prepared):
        for pid,indices in data['tracks'].items():
            rows=data['canonical'][indices];events=event_times(rows,polygon)
            result.append(dict(host_key=f'{unit}|{slot}|{pid}',phase='natural',host_member_index=slot,
                host_parent_id=pid,host_observations=len(rows),endpoint_frame=int(rows[-1,0]),
                endpoint_consecutive=len(rows)>=2 and rows[-1,0]-rows[-2,0]==1,
                anchor_eligible=len(rows)>=3 and rows[-1,0]-rows[-2,0]==1,
                baseline_events=events,baseline_windows=window_counts(events,nf).tolist(),
                option_ids=grouped[(slot,pid)],stratum=strata.get((slot,pid))))
    return result


def artificial_catalog(members,polygon,nf,context,prepared=None):
    """One SHA-chosen cutoff per host; peers NEVER affect date selection.

    Virtual scene is original three K tables minus ONLY this host's future rows.
    Frame indices reused for matching; no O(hosts*full_rows) scene rebuilding.
    """
    from tools.study197.candidates import event_times,window_counts
    data=[prepare(m,nf) for m in members] if prepared is None else prepared
    source_indices=[{tuple(s):j for j,s in enumerate(m['sources'])} for m in members]
    cases=[];unit=context['unit'];denominator=0
    for slot,host in enumerate(data):
        for pid,ix in host['tracks'].items():
            track=host['canonical'][ix];dates=track[:,0].astype(int);eligible=[]
            for k in range(2,len(ix)-1):
                if dates[k]-dates[k-1]!=1 or not 1<=dates[k+1]-dates[k]<=25:continue
                text=f'suffix-mask-v1|{unit}|{slot}|{pid}|{dates[k]}'
                eligible.append((hashlib.sha256(text.encode()).hexdigest(),k,text))
            if not eligible:continue
            denominator+=1;digest,k,hash_input=min(eligible);f=int(dates[k])
            prefixix=ix[:k+1];hidden=sorted(map(int,host['index'][ix[k+1:]]))
            phase_key=f'{unit}|{slot}|{pid}|{f}|masked'
            # Exact virtual host frame lists. Only the removed host track rows
            # change, and only target host is considered for candidate supply.
            view=dict(host);view['tracks']={pid:prefixix};view['dates']=dict(host['dates'])
            for d in dates[k+1:]:
                view['dates'][int(d)]=[r for r in host['dates'][int(d)] if int(r[1])!=pid]
            virtual=list(data);virtual[slot]=view
            cs=indexed_catalog(members,polygon,nf,virtual,(slot,pid))
            cs=enrich(cs,members,dict(context,phase='artificial',phase_key=phase_key),source_indices)
            original_options=[]
            for c in cs:
                original_id=c['candidate_id'];c['candidate_id']=c['action_id']=phase_key+'|'+original_id
                c['original_candidate_id']=original_id
                features(c,members,polygon,host_count=k+1,scene_rows=np.asarray(host['dates'][f]))
                # Native scene indices after removing hidden rows; sources stay
                # original detector keys and peer indices are unchanged.
                original_indices=list(c['host_anchor_K_indices'])
                remap=[int(j-np.searchsorted(hidden,j,side='left')) for j in original_indices]
                c.update(original_host_anchor_K_indices=original_indices,host_anchor_K_indices=remap,
                         host_anchor_row_indices=remap,insert_host_anchor_indices=remap,
                         insert_suffix_indices=list(range(len(host['rows'])-len(hidden),len(host['rows'])-len(hidden)+len(c['suffix']))))
                c['outcome_windows']=window_counts(c['events_after'],nf).tolist()
                original_options.append(original_id)
            prefix=track[:k+1];baseline=event_times(prefix,polygon);target=event_times(track,polygon)
            cases.append(dict(phase_key=phase_key,host_key=f'{unit}|{slot}|{pid}',phase='artificial',
                host_member_index=slot,host_parent_id=pid,cutoff_frame=f,cutoff_sha256=digest,cutoff_hash_input=hash_input,
                eligible_cutoff_count=len(eligible),hidden_K_indices=hidden,visible_host_observations=k+1,
                eligible_cutoffs=[dict(frame=int(dates[j]),sha256=d) for d,j,text in sorted(eligible)],
                full_host_observations=len(track),baseline_events=baseline,target_events=target,
                baseline_windows=window_counts(baseline,nf).tolist(),target_windows=window_counts(target,nf).tolist(),
                original_option_ids=original_options,candidates=cs,resources=resources(cs)))
    return dict(schema='peer-suffix203-artificial-v1',cutoff_eligible_hosts=denominator,
                selected_cutoffs=len(cases),zero_candidate_cutoffs=sum(not c['candidates'] for c in cases),cases=cases)


def virtual_members(members,case):
    result=[];slot=case['host_member_index']
    for i,m in enumerate(members):
        if i!=slot:result.append(m);continue
        keep=np.ones(len(m['rows']),bool);keep[case['hidden_K_indices']]=False
        result.append(dict(m,rows=np.asarray(m['rows'])[keep],sources=[s for j,s in enumerate(m['sources']) if keep[j]]))
    return result


def resources(candidates):
    """Complete conflicts, indexed by host, source, and inserted frame."""
    groups=defaultdict(set)
    for c in candidates:
        cid=c['candidate_id'];groups[('host',c['host_member'],c['host_track_id'])].add(cid)
        for s in c['suffix']:groups[('row',*s['source'])].add(cid)
    byframe=defaultdict(list)
    for c in candidates:
        for s in c['suffix']:byframe[(c['host_member'],s['frame'])].append((c['candidate_id'],s))
    overlap=set()
    for records in byframe.values():
        for i,(a,s) in enumerate(records):
            for b,t in records[i+1:]:
                if a!=b and iou(s['row'],t['row'])>=.5:overlap.add(tuple(sorted((a,b))))
    for a,b in overlap:groups[('overlap',a,b)]={a,b}
    return [dict(resource=list(key),candidate_ids=sorted(ids),capacity=1) for key,ids in sorted(groups.items(),key=lambda x:str(x[0]))]


def materialize_single(members,candidate):
    output,proof=materialize(members,[candidate],[candidate['candidate_id']])
    hi=candidate['host_member_index'];key=candidate['host_member']
    sources=list(map(tuple,members[hi]['sources']))+[tuple(s['source']) for s in candidate['suffix']]
    return output[key],dict(sources=sources,additions=proof[key],host_member_index=hi,
                            original_K_prefix_exact=True,insertion_order='append selected candidate/date; native IDs retained')


def validate_allocation(group,selected_ids):
    materialize(group['members'],group['candidates'],selected_ids)
    return dict(valid=True,selected_ids=sorted(selected_ids),scope='whole three-member unit')


def materialize_group(group,host_member,selected_ids):
    """Validate entire unit first; return one whole member plus all source keys."""
    output,added=materialize(group['members'],group['candidates'],selected_ids)
    if isinstance(host_member,int):member=group['members'][host_member]
    else:member=next(m for m in group['members'] if str(m['member'])==str(host_member))
    key=str(member['member']);sources=list(map(tuple,member['sources']))+[tuple(r['source']) for r in added[key]]
    return output[key],dict(sources=sources,additions=added[key],selected_ids=sorted(selected_ids),original_K_prefix_exact=True)


def load_group(folder):
    folder=Path(folder);verify(folder);data=json.loads((folder/'catalog.json').read_text())
    members=[]
    with np.load(folder/'K.npz',allow_pickle=False) as z:
        for i,m in enumerate(data['members']):members.append(dict(m,rows=z[f'k{i}']))
    return dict(data,members=members)


def freeze_unit(source,records,dest,prereg,source_manifest):
    from tools.study197.candidates import event_times,window_counts
    members=[]
    for r in records:
        folder=source/r['path'];receipt=json.loads((folder/'receipt.json').read_text())
        if not all(receipt[k] for k in ('exact_K','exact_historical_s1_IDs','exact_native_full_stage1')):raise ValueError('202exactchecks nottrue')
        with np.load(folder/'rows.npz',allow_pickle=False) as z:rows=z['baseline']
        old=json.loads((folder/'catalog.json').read_text())
        if not np.array_equal(rows,np.asarray(old['baseline'],float)):raise ValueError('202baseline numeric mismatch')
        if any(s[0]!=receipt['identity']['detector_sha256'] for s in old['baseline_sources']):raise ValueError('202detector provenance drift')
        expected_threshold=.5 if r['domain'] in ('35','33') else .4
        if receipt['identity']['threshold']!=expected_threshold:raise ValueError('202threshold mismatch')
        members.append(dict(member=f'{r["unit"]}|{r["member_index"]}',member_index=r['member_index'],rows=rows,
                            sources=old['baseline_sources'],source_record=r))
    context={k:records[0][k] for k in ('unit','domain','group','video','nf')}
    identity=dict(prereg=prereg,source202=SOURCE202,member_seals=[r['seal'] for r in records],
                  code={p.name:sha(p) for p in (Path(__file__),Path(__file__).parent.parent/'peer_suffix_supply.py',Path(__file__).parent.parent/'peer_suffix_prototype.py')})
    if dest.exists():
        verify(dest)
        if json.loads((dest/'receipt.json').read_text())['identity']!=identity:raise ValueError('resume identity drift')
        return
    polygon=records[0]['polygon'];nf=records[0]['nf']
    prepared=[prepare(m,nf) for m in members]
    cs=enrich(indexed_catalog(members,polygon,nf,prepared),members,context)
    for c in cs:
        n=len(prepared[c['host_member_index']]['tracks'][c['host_parent_id']])
        features(c,members,polygon,host_count=n,scene_rows=np.asarray(prepared[c['host_member_index']]['dates'][c['anchor_frames'][-1]]))
        c['outcome_windows']=window_counts(c['events_after'],nf).tolist()
    conflict=resources(cs)
    contexts=host_contexts(members,polygon,nf,prepared,cs,records[0]['unit'])
    artificial=artificial_catalog(members,polygon,nf,context,prepared)
    dest.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        np.savez(temp/'K.npz',**{f'k{i}':m['rows'] for i,m in enumerate(members)})
        member_meta=[]
        for m in members:
            events=event_times(m['rows'],polygon)
            member_meta.append(dict(member=m['member'],member_index=m['member_index'],sources=m['sources'],
                                    source_record=m['source_record'],events=events,windows=window_counts(events,nf)))
        save(temp/'catalog.json',dict(schema='peer-suffix203-catalog-v1',**context,polygon=polygon,
            members=member_meta,candidates=cs,resources=conflict,host_contexts=contexts,
            feature_names=list(FEATURE_NAMES),forward_only=True,max_suffix_frames=25,
            host_denominator=len(contexts),zero_candidate_hosts=sum(not c['option_ids'] for c in contexts)))
        save(temp/'artificial.json',artificial)
        save(temp/'receipt.json',dict(identity=identity,candidate_count=len(cs),resource_count=len(conflict),
                                      artificial_cutoffs=artificial['selected_cutoffs'],
                                      exact202_K=True,exact202_sourcekeys=True,baseline_ids='202sealed197stablegapallocated',
                                      source='202sealed baseline K; no GT or model access'))
        save(temp/'manifest.json',{p.name:sha(p) for p in temp.iterdir()});(temp/'SEALED.sha256').write_text(sha(temp/'manifest.json')+'\n')
        temp.rename(dest)
    finally:
        if temp.exists():shutil.rmtree(temp)


def run(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('published full prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('unique203marker missing '+MARKER)
    if args.out.resolve().drive.upper()!='D:':raise ValueError('D: output required')
    if sha(args.source202/'manifest.json')!=SOURCE202:raise ValueError('wrong202source')
    manifest=verify(args.source202);data=json.loads((args.source202/'deployment.json').read_text())
    if len(data['members'])!=60:raise ValueError('60members required')
    grouped=defaultdict(list)
    for r in data['members']:
        if not 0<=r['seed']<=14 or r['split']=='test':raise ValueError('forbidden source')
        grouped[r['unit']].append(r)
    if len(grouped)!=20:raise ValueError('20units required')
    records=[]
    for i,(unit,rr) in enumerate(grouped.items()):
        rr=sorted(rr,key=lambda r:r['member_index'])
        if [r['member_index'] for r in rr]!=[0,1,2]:raise ValueError('slot order')
        token=f'g{i:02d}';dest=args.out/token
        freeze_unit(args.source202,rr,dest,args.prereg,manifest)
        records.append(dict(unit=unit,path=token,domain=rr[0]['domain'],group=rr[0]['group'],video=rr[0]['video'],
                            seal=(dest/'SEALED.sha256').read_text().strip()))
        print('SEALED',token,unit,flush=True)
    top=dict(schema='peer-suffix203-freeze-v1',prereg=args.prereg,marker=MARKER,source202=SOURCE202,groups=records)
    if (args.out/'deployment.json').exists():
        if json.loads((args.out/'deployment.json').read_text())!=top:raise ValueError('top resume drift')
    else:save(args.out/'deployment.json',top)
    allfiles={p.relative_to(args.out).as_posix():sha(p) for p in args.out.rglob('*') if p.is_file() and p not in (args.out/'manifest.json',args.out/'SEALED.sha256')}
    if (args.out/'manifest.json').exists() and json.loads((args.out/'manifest.json').read_text())!=allfiles:raise ValueError('top manifest drift')
    save(args.out/'manifest.json',allfiles);(args.out/'SEALED.sha256').write_text(sha(args.out/'manifest.json')+'\n')
    print('FROZEN20',sha(args.out/'manifest.json'))


def self_test():
    def row(f,p,x):return [f,p,x,0,4,4,.9]
    def m(k,rows):return dict(member=k,rows=rows,sources=[('synthetic-'+k,i) for i in range(len(rows))])
    host=[row(f,1,x) for f,x in [(747,-6),(748,-5),(749,-4)]]
    peer=host+[row(f,1,x) for f,x in [(750,-3),(751,-2),(752,-1),(753,1)]]
    poly=[(0,-10),(100,-10),(100,100),(0,100)]
    fixtures=[]
    fixtures.append([m('H',host),m('P',peer),m('Q',peer)])
    fixtures.append([m('H',host),m('P',[r for r in peer if r[0]!=751]),m('Q',peer)])
    fixtures.append([m('H',host+[row(751,8,-2)]),m('P',peer),m('Q',peer)])
    fixtures.append([m('H',host),m('P',peer+[row(749,8,-4)]),m('Q',peer)])
    for mm in fixtures:
        a=reference_catalog(mm,poly,800);b=indexed_catalog(mm,poly,800)
        if plain(a)!=plain(b):raise AssertionError('indexed catalog differs')
        cs=enrich(b,mm,dict(unit='toy',domain='W',group=1,video='synthetic',nf=800))
        resources(cs)
        if cs:
            out,proof=materialize_single(mm,cs[0]);assert len(proof['sources'])==len(out)
        prepared=[prepare(m,800) for m in mm]
        for c in cs:
            features(c,mm,poly,host_count=len(prepared[c['host_member_index']]['tracks'][c['host_parent_id']]))
        art=artificial_catalog(mm,poly,800,dict(unit='toy',domain='W',group=1,video='synthetic',nf=800),prepared)
        for case in art['cases']:
            virtual=virtual_members(mm,case)
            expected=[c for c in reference_catalog(virtual,poly,800)
                      if c['host_member']==mm[case['host_member_index']]['member'] and c['host_track_id']==case['host_parent_id']]
            wanted={c['candidate_id']:c for c in expected}
            for c in case['candidates']:
                ref=wanted[c['original_candidate_id']]
                for key in ref:
                    if plain(ref[key])!=plain(c[key]):
                        if key=='candidate_id':continue
                        raise AssertionError('artificial virtual mismatch '+key)
            assert len(case['candidates'])==len(expected)
    first=fixtures[0];fc=enrich(indexed_catalog(first,poly,800),first,dict(unit='toy',domain='W',group=1,video='synthetic',nf=800))
    c=next(c for c in fc if c['host_member']=='H');features(c,first,poly,host_count=3)
    assert c['features10'][:3]==[1.,1.,0.] and c['G_logit']==2.
    assert c['features10'][5:8]==[4/25,3/25,0.]
    return 'PASS: indexed/reference all fields exact for clean, date gap, host overlap, ambiguous anchor; resources/materialize'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--self-test',action='store_true')
    p.add_argument('--source202',type=Path);p.add_argument('--out',type=Path);p.add_argument('--prereg')
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    if args.self_test:print(self_test())
    else:
        if args.source202 is None or args.out is None:p.error('--source202 --out --prereg required')
        run(args)
