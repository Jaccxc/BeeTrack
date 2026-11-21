"""Independent203 replay. Source/synthetic only until final sole-CPU GO.

Supply, artificial censoring and source ownership are separate collections.
Only official low-level geometry/track matching may be shared with producers.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import gc
from pathlib import Path
import math
import re
from fractions import Fraction
import numpy as np
import cv2
cv2.setNumThreads(1)


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def source_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def source_key(value):
    assert len(value)==2 and isinstance(value[0],str) and value[0] and int(value[1])==value[1] and value[1]>=0
    return value[0],int(value[1])


def observed_iou(a,b):
    # The preregistered observation gate has no official GT-IoU union epsilon.
    w=max(0.,min(a[2]+a[4],b[2]+b[4])-max(a[2],b[2]));h=max(0.,min(a[3]+a[5],b[3]+b[5])-max(a[3],b[3]))
    intersection=w*h
    return intersection/(a[4]*a[5]+b[4]*b[5]-intersection) if intersection else 0.


def canonical(rows):
    raw=np.asarray(rows,float).reshape(-1,7);assert np.isfinite(raw).all() and (raw[:,4:6]>0).all()
    assert len(np.unique(raw[:,:2],axis=0))==len(raw)
    indices=[];ids=[];seen=set()
    for pid in sorted(set(raw[:,1])):
        ii=np.flatnonzero(raw[:,1]==pid);ii=ii[np.argsort(raw[ii,0],kind='stable')]
        for run,part in enumerate(np.split(ii,np.flatnonzero(np.diff(raw[ii,0])>25)+1)):
            alias=int(pid+run*100000);assert alias not in seen;seen.add(alias);indices.extend(part);ids.extend([alias]*len(part))
    indices=np.asarray(indices,int);ids=np.asarray(ids,int);order=np.lexsort((indices,raw[indices,0],ids));indices=indices[order];ids=ids[order]
    C=raw[indices].copy();C[:,1]=ids
    return C,indices


def prepare(member,nf):
    raw=np.asarray(member['rows'],float);C,index=canonical(raw);keys=[source_key(k) for k in member['sources']]
    assert len(keys)==len(raw)==len(set(keys)) and np.all((raw[:,0]>=1)&(raw[:,0]<=nf))
    dates=defaultdict(list);tracks=defaultdict(list)
    for i,row in enumerate(C):dates[int(row[0])].append(i);tracks[int(row[1])].append(i)
    track_dates={pid:{int(C[j,0]):j for j in ii} for pid,ii in tracks.items()}
    return dict(raw=raw,C=C,index=index,keys=keys,dates=dates,tracks=tracks,track_dates=track_dates)


def track_events(rows,polygon):
    raw=np.asarray(rows,float);assert raw.ndim==2 and raw.shape[1]>=6
    poly=np.asarray(polygon,np.float32);result=[]
    for pid in sorted(set(raw[:,1])):
        tr=raw[raw[:,1]==pid];tr=tr[np.argsort(tr[:,0],kind='stable')]
        for r in np.split(tr,np.flatnonzero(np.diff(tr[:,0])>25)+1):
            side=[cv2.pointPolygonTest(poly,(float(x[2]+x[4]/2),float(x[3]+x[5]/2)),False)>=0 for x in r]
            if side[0]==side[-1]:continue
            direction=1 if side[-1] else -1
            date=next(int(r[j,0]) for j in range(len(r)-1,0,-1) if side[j]!=side[j-1] and side[j]==side[-1])
            result.append((date,direction))
    return result


def bins(events,nf):
    answer=[]
    for offset in range(30):
        a=np.zeros(((nf+offset*25+749)//750,2),np.int64)
        for date,direction in events:a[(date-1+offset*25)//750,int(direction<0)]+=1
        answer.extend(a.tolist())
    return np.asarray(answer,np.int64)


def supply_independent(members,polygon,nf,host_only=None,prepared=None):
    assert len(members)==3 and len({str(m['member']) for m in members})==3
    data=[prepare(m,nf) for m in members] if prepared is None else prepared;result=[]
    for slot,host in enumerate(data):
        for hid,ii in sorted(host['tracks'].items()):
            if host_only is not None and (slot,hid)!=host_only:continue
            own=host['C'][ii]
            if len(own)<3 or own[-1,0]-own[-2,0]!=1:continue
            f=int(own[-1,0]);frames=[f-1,f]
            for donor_slot,donor in enumerate(data):
                if donor_slot==slot:continue
                # Mutual uniqueness implies exactly one qualifying last-date
                # peer box, so this full-frame join loses no legal donor.
                possible=[j for j in donor['dates'].get(f,[]) if observed_iou(own[-1],donor['C'][j])>=.5]
                if len(possible)!=1:continue
                donor_id=int(donor['C'][possible[0],1])
                for did in [donor_id]:
                    jj=donor['tracks'][did]
                    if len(jj)<3:continue
                    bydate=donor['track_dates'][did]
                    if not all(d in bydate for d in frames):continue
                    anchors=[donor['C'][bydate[d]] for d in frames];valid=True
                    for a,b,d in zip(own[-2:],anchors,frames):
                        right=[j for j in donor['dates'].get(d,[]) if observed_iou(a,donor['C'][j])>=.5]
                        left=[i for i in host['dates'].get(d,[]) if observed_iou(b,host['C'][i])>=.5]
                        if right!=[bydate[d]] or left!=[ii[-2] if d==f-1 else ii[-1]]:valid=False;break
                    if not valid:continue
                    suffix=[];stop='limit25'
                    for date in range(f+1,min(f+25,nf)+1):
                        j=bydate.get(date)
                        if j is None:stop='peer_date_missing';break
                        row=donor['C'][j]
                        if any(observed_iou(row,host['C'][i])>=.5 for i in host['dates'].get(date,[])):stop='host_overlap';break
                        native=int(donor['index'][j]);suffix.append(dict(frame=date,peer_row_index=native,source=list(donor['keys'][native]),row=row.tolist()))
                    if not suffix:continue
                    ext=np.asarray([s['row'] for s in suffix]);ext[:,1]=hid
                    before=track_events(own,polygon);after=track_events(np.r_[own,ext],polygon)
                    result.append(dict(candidate_id=f"{members[slot]['member']}|{hid}|{members[donor_slot]['member']}|{did}|{f}",host_member=str(members[slot]['member']),peer_member=str(members[donor_slot]['member']),host_track_id=hid,peer_track_id=did,host_member_index=slot,donor_member_index=donor_slot,
                        anchor_frames=frames,host_anchor_row_indices=[int(host['index'][j]) for j in ii[-2:]],donor_anchor_indices=[int(donor['index'][bydate[d]]) for d in frames],
                        host_anchor_rows=own[-2:].tolist(),peer_anchor_rows=np.asarray(anchors).tolist(),host_anchor_sources=[list(host['keys'][host['index'][j]]) for j in ii[-2:]],peer_anchor_sources=[list(donor['keys'][donor['index'][bydate[d]]]) for d in frames],
                        suffix=suffix,stop=stop,events_before=before,events_after=after,delta_windows=bins(after,nf)-bins(before,nf)))
    return sorted(result,key=lambda c:c['candidate_id'])


def physical_alias(c):
    rowkey=lambda r:(int(r[0]),*map(float,r[2:]))
    # Original donor-anchor observations are part of the fixed alias contract.
    return (c['host_member'],c['host_track_id'],tuple((source_key(s['source']),rowkey(s['row'])) for s in c['suffix']),
            tuple((source_key(key),rowkey(row)) for key,row in zip(c['peer_anchor_sources'],c['peer_anchor_rows'])))


def deduplicate_supply(candidates):
    groups=defaultdict(list)
    for c in candidates:groups[physical_alias(c)].append(c)
    return [(min(cs,key=lambda c:c['candidate_id']),sorted(c['candidate_id'] for c in cs)) for cs in groups.values()]


def artificial_choice(member,unit,slot,nf):
    p=prepare(member,nf);result=[]
    for pid,ii in sorted(p['tracks'].items()):
        dates=p['C'][ii,0].astype(int);eligible=[]
        for j in range(2,len(ii)-1):
            if dates[j]-dates[j-1]!=1 or not 1<=dates[j+1]-dates[j]<=25:continue
            token=f'suffix-mask-v1|{unit}|{slot}|{pid}|{dates[j]}'
            eligible.append((hashlib.sha256(token.encode('utf-8')).hexdigest(),int(dates[j]),j,token))
        if not eligible:continue
        digest,date,j,token=min(eligible);hidden=sorted(int(p['index'][i]) for i in ii[j+1:])
        result.append(dict(host_parent_id=pid,cutoff_frame=date,cutoff_sha256=digest,cutoff_hash_input=token,hidden_K_indices=hidden,visible_host_observations=j+1,full_host_observations=len(ii),eligible_cutoffs=[dict(frame=d,sha256=h) for h,d,k,t in sorted(eligible)]))
    return result


def observed_features(c,data,polygon,visible_count=None):
    slot=c['host_member_index'];host=data[slot];a,b=np.asarray(c['host_anchor_rows'],float);first=np.asarray(c['suffix'][0]['row'],float)
    center=lambda row:row[2:4]+row[4:6]/2
    u,v,w=center(a),center(b),center(first);diag=float(np.sqrt(b[4]**2+b[5]**2));dist=np.linalg.norm
    anchors=[observed_iou(x,y) for x,y in zip(c['host_anchor_rows'],c['peer_anchor_rows'])]
    count=len(host['tracks'][c['host_track_id']]) if visible_count is None else visible_count
    neighbors=sum(host['C'][i,1]!=c['host_track_id'] and dist(center(host['C'][i])-v)<=2*diag for i in host['dates'][int(b[0])])
    boundary=abs(cv2.pointPolygonTest(np.asarray(polygon,np.float32),tuple(map(float,v)),True))
    features=[min(anchors),sum(anchors)/2,float(dist(w-(2*v-u))/diag),float(dist(w-v)/diag),float(dist(v-u)/diag),len(c['suffix'])/25,min(25,count)/25,float(b[6]-a[6]),float(boundary/diag),float(neighbors)]
    inside=cv2.pointPolygonTest(np.asarray(polygon,np.float32),tuple(map(float,v)),False)>=0
    stratum=[int(inside),int(boundary<=diag),int(neighbors>0),int(b[6]<a[6]),int(dist(v-u)<=.5*diag)]
    return features,stratum


def artificial_independent(members,unit,polygon,nf):
    prepared=[prepare(m,nf) for m in members];result=[]
    for slot,m in enumerate(members):
        for case in artificial_choice(m,unit,slot,nf):
            original=prepared[slot];pid=case['host_parent_id'];ii=original['tracks'][pid];keep=case['visible_host_observations'];removed=set(ii[keep:])
            dates=dict(original['dates'])
            for date in np.unique(original['C'][ii[keep:],0]).astype(int):dates[int(date)]=[j for j in dates[int(date)] if j not in removed]
            virtual=dict(original,tracks={pid:ii[:keep]},dates=dates)
            data=list(prepared);data[slot]=virtual
            cs=supply_independent(members,polygon,nf,host_only=(slot,pid),prepared=data)
            target=track_events(original['C'][ii],polygon);base=track_events(original['C'][ii[:keep]],polygon)
            result.append(dict(case,candidates=cs,target_events=target,baseline_events=base,target_windows=bins(target,nf),baseline_windows=bins(base,nf)))
    return result


def outcome_decoder(contexts,scores,candidates):
    """Independent registered outcome mass then greedy feasible action rule."""
    pending=[];table={c['candidate_id']:c for c in candidates};action_unit={o['action_id']:c['unit'] for c in contexts for o in c['options']}
    for ctx in contexts:
        byoutcome=defaultdict(list);base=tuple(np.asarray(ctx['baseline_outcome'],np.int64).ravel());byoutcome[base].append((None,0.))
        for option in ctx['options']:
            value=scores[option['action_id']]
            if value is not None:byoutcome[tuple(np.asarray(option['outcome'],np.int64).ravel())].append((option['action_id'],value))
        keep=lse([v for key,v in byoutcome[base]])
        for outcome,values in byoutcome.items():
            if outcome==base:continue
            advantage=lse([value for key,value in values])-keep
            if advantage>0:
                pending.extend((advantage,value,key) for key,value in values if key is not None)
    chosen=[];host_used=defaultdict(set);source_used=defaultdict(set);new=defaultdict(list)
    for advantage,value,aid in sorted(pending,key=lambda t:(-t[0],-t[1],t[2])):
        c=table[aid];unit=action_unit[aid];host=(c['host_member'],c['host_track_id']);ss={source_key(s['source']) for s in c['suffix']}
        if host in host_used[unit] or ss&source_used[unit]:continue
        if any(s['frame']==f and observed_iou(s['row'],row)>=.5 for s in c['suffix'] for f,row in new[c['host_member']]):continue
        chosen.append(aid);host_used[unit].add(host);source_used[unit].update(ss);new[c['host_member']].extend((s['frame'],s['row']) for s in c['suffix'])
    return chosen


def permute_T(contexts,scores):
    result={}
    for c in contexts:
        ids=sorted(o['action_id'] for o in c['options']);dest=sorted(ids,key=lambda aid:(hashlib.sha256(('peer-suffix-rank-null-v1|'+aid).encode()).hexdigest(),aid))
        result.update({dst:scores[src] for src,dst in zip(ids,dest)})
    return result


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def self_memory():
    """Only this verifier process; never inspect another training process."""
    if os.name!='nt':return {}
    import ctypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in ('peak_working','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
    c=Counters();c.cb=ctypes.sizeof(c);kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=ctypes.c_void_p
    fn=ctypes.WinDLL('psapi').GetProcessMemoryInfo;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    if not fn(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):return {}
    return dict(working_MiB=c.working/2**20,peak_working_MiB=c.peak_working/2**20,private_MiB=c.private/2**20)


def stream_array(path,key=None):
    """One object at a time, finding boundaries BEFORE JSON object decoding.

    This avoids repeatedly decoding a partly read hundred-megabyte owner unit.
    The registered keyed arrays are units/cases/groups; each item is an object.
    """
    pattern=None if key is None else re.compile(r'"'+re.escape(key)+r'"\s*:\s*\[')
    with Path(path).open(encoding='utf-8') as stream:
        buffer=''
        while True:
            part=stream.read(1<<20);assert part,'missing array';buffer+=part
            if pattern is None:
                stripped=buffer.lstrip()
                if stripped.startswith('['):buffer=stripped[1:];break
            else:
                match=pattern.search(buffer)
                if match:buffer=buffer[match.end():];break
        pieces=[];depth=0;quoted=False;escaped=False;collecting=False
        while True:
            start=0 if collecting else None
            for index,char in enumerate(buffer):
                if not collecting:
                    if char.isspace() or char==',':continue
                    if char==']':return
                    assert char=='{','registered array items must be objects';collecting=True;start=index;depth=1;continue
                if quoted:
                    if escaped:escaped=False
                    elif char=='\\':escaped=True
                    elif char=='"':quoted=False
                elif char=='"':quoted=True
                elif char in '[{':depth+=1
                elif char in ']}':
                    depth-=1
                    if depth==0:
                        pieces.append(buffer[start:index+1]);value=json.loads(''.join(pieces));pieces=[]
                        collecting=False;start=None
                        yield value
                        del value
            if collecting:pieces.append(buffer[start:])
            buffer=stream.read(1<<20);assert buffer,'unterminated array'


def save_receipt(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');seal=path.with_suffix('.sha256');st=seal.with_name(seal.name+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');st.write_text(source_sha(temporary)+'\n')
    os.replace(temporary,path);os.replace(st,seal)


def verify_seal(root,expected=None):
    root=Path(root);value=source_sha(root/'manifest.json');assert value==(root/'SEALED.sha256').read_text().strip()
    if expected is not None:assert value==expected
    for name,digest in read(root/'manifest.json').items():
        path=(root/name).resolve();assert path.is_relative_to(root.resolve()) and source_sha(path)==digest
    return value


def helpers(repo):
    path=repo/'tools/study202/verify_orphans.py'
    if not path.exists():path=Path.cwd()/'tools/study202/verify_orphans.py'
    previous=load_module('independent202_for203',path);previous.sha=source_sha;previous.receipt=save_receipt;u,v=previous.own_helpers(repo)
    cert=load_module('independent203_certificate',Path(__file__).with_name('verify203_certificate.py'))
    return previous,u,v,cert


def frozen_groups(args):
    top=read(args.root/'frozen/deployment.json');assert top['prereg']==args.prereg and len(top['groups'])==20
    for ref in top['groups']:
        folder=args.root/'frozen'/ref['path'];assert source_sha(folder/'manifest.json')==ref['seal'];cat=read(folder/'catalog.json')
        with np.load(folder/'K.npz',allow_pickle=False) as z:members=[dict(m,rows=z['k'+str(i)]) for i,m in enumerate(cat['members'])]
        yield ref,cat,members
        del cat,members


def assert_supply(saved,rebuilt,members,polygon,nf,artificial=False,hidden=None):
    groups=deduplicate_supply(rebuilt);table={c['candidate_id']:c for c in saved}
    assert set(table)=={c['candidate_id'] for c,aliases in groups}
    data=[prepare(m,nf) for m in members]
    for own,aliases in groups:
        x=table[own['candidate_id']]
        assert sorted(x.get('equivalent_action_ids',[]))==[aid for aid in aliases if aid!=own['candidate_id']]
        for field in ('host_member','peer_member','host_track_id','peer_track_id','anchor_frames','host_anchor_rows','peer_anchor_rows','host_anchor_sources','peer_anchor_sources','stop'):
            assert x[field]==own[field]
        assert np.array_equal(x['delta_windows'],own['delta_windows'])
        assert x['donor_anchor_K_indices']==own['donor_anchor_indices']
        assert x['donor_suffix_K_indices']==[r['peer_row_index'] for r in own['suffix']]
        for a,b in zip(x['suffix'],own['suffix']):
            for field in ('source','row','frame','peer_row_index'):assert a[field]==b[field]
        assert len(x['suffix'])==len(own['suffix'])
        if not artificial:assert x['host_anchor_K_indices']==own['host_anchor_row_indices']
        f,bits=observed_features(own,data,polygon)
        if not artificial:
            np.testing.assert_allclose(x['features10'],f,atol=2e-12,rtol=2e-12);assert x['stratum']==''.join(map(str,bits))


def audit_supply(args):
    digest=verify_seal(args.root/'frozen',args.frozen_sha);previous,u,v,cert=helpers(args.repo)
    assert source_sha(args.source202/'manifest.json')=='7261c2573773a23342286a39ce712950cb986aa6f34248cd12f5705ab420f469'
    old=read(args.source202/'deployment.json');old_records={(r['unit'],r['member_index']):r for r in old['members']};reports=[]
    for ref,cat,members in frozen_groups(args):
        dest=args.root/'verification/supply'/(ref['path']+'.json');identity=dict(seal=ref['seal'],verifier_sha256=source_sha(__file__))
        if dest.exists() and dest.with_suffix('.sha256').exists():
            report=read(dest);assert report['identity']==identity and source_sha(dest)==dest.with_suffix('.sha256').read_text().strip();reports.append(report)
            del cat,members
            gc.collect()
            continue
        assert len(members)==3 and [m['member_index'] for m in members]==[0,1,2]
        for slot,m in enumerate(members):
            original=old_records[cat['unit'],slot];assert m['source_record']==original and 0<=original['seed']<15 and original['split']!='test'
            assert cat['nf']==original['nf'] and cat['polygon']==original['polygon'] and cat['domain']==original['domain'] and cat['video']==original['video']
            folder=args.source202/original['path'];assert source_sha(folder/'manifest.json')==original['seal']
            with np.load(folder/'rows.npz',allow_pickle=False) as z:assert np.array_equal(m['rows'],z['baseline'])
            assert m['sources']==read(folder/'catalog.json')['baseline_sources']
            assert np.array_equal(m['windows'],bins(track_events(m['rows'],cat['polygon']),cat['nf']))
            assert [(e['frame'],e['direction']) for e in m['events']]==track_events(m['rows'],cat['polygon'])
        cs=supply_independent(members,cat['polygon'],cat['nf']);assert_supply(cat['candidates'],cs,members,cat['polygon'],cat['nf'])
        byid={c['candidate_id']:c for c in cat['candidates']};resources,incidence=cert.compile_resources([dict(c,action_id=c['candidate_id']) for c in cat['candidates']])
        resource_sets={tuple(r['resource']):set(r['candidate_ids']) for r in cat['resources']}
        reconstructed={key:{cat['candidates'][j]['candidate_id'] for j,rows in enumerate(incidence) if i in rows} for i,key in enumerate(resources)}
        assert resource_sets==reconstructed
        prep=[prepare(m,cat['nf']) for m in members];hosts={(h['host_member_index'],h['host_parent_id']):h for h in cat['host_contexts']}
        assert len(hosts)==sum(len(p['tracks']) for p in prep)
        for slot,p in enumerate(prep):
            for pid,ii in p['tracks'].items():
                h=hosts[slot,pid];track=p['C'][ii]
                assert h['host_observations']==len(ii) and h['endpoint_frame']==int(track[-1,0])
                assert h['anchor_eligible']==bool(len(ii)>=3 and track[-1,0]-track[-2,0]==1)
                assert np.array_equal(h['baseline_windows'],bins(track_events(track,cat['polygon']),cat['nf']))
                assert set(h['option_ids'])=={c['candidate_id'] for c in cat['candidates'] if c['host_member_index']==slot and c['host_parent_id']==pid}
                if h['option_ids']:
                    assert all(byid[cid]['stratum']==h['stratum'] for cid in h['option_ids'])
                else:assert h['stratum'] is None
        art=read(args.root/'frozen'/ref['path']/'artificial.json');expected=artificial_independent(members,cat['unit'],cat['polygon'],cat['nf'])
        assert len(expected)==art['selected_cutoffs']==art['cutoff_eligible_hosts']
        arts={(c['host_member_index'],c['host_parent_id']):c for c in art['cases']};zero=0
        for own in expected:
            slot=int(own['cutoff_hash_input'].split('|')[-3]);case=arts[slot,own['host_parent_id']]
            for field in ('cutoff_frame','cutoff_sha256','cutoff_hash_input','hidden_K_indices','visible_host_observations','full_host_observations','eligible_cutoffs'):assert own[field]==case[field]
            assert np.array_equal(own['baseline_windows'],case['baseline_windows']) and np.array_equal(own['target_windows'],case['target_windows'])
            assert own['target_events']==[(e['frame'],e['direction']) for e in case['target_events']]
            # Reconstruct native index removal independently; donor observations
            # stay in the original member, while only the host future is hidden.
            ids={c['original_candidate_id']:c for c in case['candidates']};groups=deduplicate_supply(own['candidates']);assert set(ids)=={c['candidate_id'] for c,aliases in groups}
            for c,aliases in groups:
                x=ids[c['candidate_id']];hidden=case['hidden_K_indices'];orig=c['host_anchor_row_indices'];remap=[int(j-np.searchsorted(hidden,j,side='left')) for j in orig]
                assert x['original_host_anchor_K_indices']==orig and x['host_anchor_K_indices']==remap
                assert np.array_equal(x['delta_windows'],c['delta_windows']) and [s['source'] for s in x['suffix']]==[s['source'] for s in c['suffix']]
                f,bits=observed_features(c,prep,cat['polygon'],case['visible_host_observations']);np.testing.assert_allclose(f,x['features10'],atol=2e-12,rtol=2e-12)
                assert x['stratum']==''.join(map(str,bits))
            zero+=int(not groups)
        assert zero==art['zero_candidate_cutoffs']
        report=dict(passed=True,identity=identity,natural_candidates=len(cat['candidates']),hosts=len(hosts),no_supply_hosts=sum(not h['option_ids'] for h in hosts.values()),artificial_cutoffs=len(expected),artificial_no_supply=zero,memory=self_memory(),scope='All original202 K/source rows, natural supply/aliases/features/deltas/resources, all first-hash artificial dates/indices and original-host event targets; no GT read')
        save_receipt(dest,report);reports.append(report);print('INDEPENDENT203_SUPPLY',cat['unit'],flush=True)
        del art,expected,cs,prep,byid,hosts,resource_sets,reconstructed,cat,members
        gc.collect()
    result=dict(passed=True,frozen_manifest=digest,units=len(reports),natural_candidates=sum(r['natural_candidates'] for r in reports),artificial_cutoffs=sum(r['artificial_cutoffs'] for r in reports),scope='Full20/60 GT-free supply and artificial date/source collection independently reconstructed')
    save_receipt(args.root/'verification/supply.json',result);return result


def first_return(frames,sides):
    changes=[i for i in range(1,len(frames)) if sides[i]!=sides[i-1]]
    for pos,j in enumerate(changes[:-1]):
        r=changes[pos+1]
        if sides[j]!=sides[0] and sides[r]==sides[0] and frames[r]-frames[j]<=25:
            return dict(departure_index=j,return_index=r,departure_frame=frames[j],return_frame=frames[r],initial_inside=sides[0])
    return None


def GT_universe(raw,polygon,v,file_sha,relative):
    G,origin=v.canonical_with_origin(raw);originals=defaultdict(set)
    for row,ix in zip(G,origin):originals[int(row[1])].add(int(raw[ix,1]))
    segments=v.GT_segments(dict(G=G,original_ids=originals),polygon);runs=[];poly=np.asarray(polygon,np.float32)
    for pid,seg in segments.items():
        ii=list(range(seg['start'],seg['end']));frames=[int(G[i,0]) for i in ii]
        sides=[cv2.pointPolygonTest(poly,(float(G[i,2]+G[i,4]/2),float(G[i,3]+G[i,5]/2)),False)>=0 for i in ii]
        runs.append(dict(canonical_gt_id=pid,original_gt_id=seg['original_gt_id'],original_ids=sorted(originals[pid]),run=seg.get('run'),observations=[dict(frame=f,inside=s,positive_extent=bool(G[i,4]>0 and G[i,5]>0),source_raw_index=int(origin[i]),row_sha256=v.array_digest(G[i])) for i,f,s in zip(ii,frames,sides)],eligible_probe_or_null=first_return(frames,sides)))
    return dict(GT_file_sha256=file_sha,GT_relative_path=relative,polygon=polygon,canonical_rows_sha256=v.array_digest(G),runs=runs),G,segments


def classify_evidence(rows):
    if not rows or not all(r['qualified'] for r in rows):return 'U'
    if len({r['official_owner'] for r in rows})==1:return 'S'
    if len({r['original_gt_id'] for r in rows})>1:return 'D'
    return 'U'


_GT_CACHE={}


def cached_owners(gt,P,threshold,needed,v):
    """Cache only immutable GT preparation; each hypothetical frame is rematched."""
    from scipy.optimize import linear_sum_assignment
    def frames(a):
        order=np.argsort(a[:,0],kind='stable');dates,first=np.unique(a[order,0],return_index=True)
        return {int(date):order[lo:hi] for date,lo,hi in zip(dates,first,[*first[1:],len(order)])}
    key=id(gt)
    if key not in _GT_CACHE:
        G,gi=v.canonical_with_origin(gt);originals={}
        for i,row in enumerate(G):originals.setdefault(int(row[1]),set()).add(int(gt[gi[i],1]))
        _GT_CACHE[key]=dict(raw=gt,G=G,gi=gi,originals=originals,frames=frames(G),segments={})
    cache=_GT_CACHE[key];assert cache['raw'] is gt
    G,gi,originals=cache['G'],cache['gi'],cache['originals'];P,pi=v.canonical_with_origin(P)
    rows={i:dict(prediction_index=i,raw_prediction_index=int(pi[i]),frame=int(P[i,0]),pred_id=int(P[i,1]),owner=None,owner_row=None,original_gt_id=None,pred_degree=0,gt_degree=0,iou=None,isolated=False,graph_gt=[]) for i in needed}
    gp,pp=cache['frames'],frames(P);reverse={}
    for date in sorted({int(P[i,0]) for i in needed}):
        gg=gp.get(date,np.empty(0,int));pr=pp[date]
        if not len(gg):continue
        matrix=v.official_iou_independent(G[gg,2:6],P[pr,2:6]);E=matrix>=threshold;degreeP=E.sum(0);degreeG=E.sum(1)
        for j,i in enumerate(pr):
            if i not in rows:continue
            rows[i]['pred_degree']=int(degreeP[j]);rows[i]['graph_gt']=[dict(owner=int(G[gg[k],1]),original_gt_id=next(iter(originals[int(G[gg[k],1])])) if len(originals[int(G[gg[k],1])])==1 else None,iou=float(matrix[k,j]),gt_degree=int(degreeG[k])) for k in np.flatnonzero(E[:,j])]
        aa,bb=linear_sum_assignment(-matrix)
        for a,b in zip(aa,bb):
            i=int(pr[b])
            if i not in rows or not E[a,b]:continue
            pid=int(G[gg[a],1]);oid=next(iter(originals[pid])) if len(originals[pid])==1 else None
            rows[i].update(owner=pid,owner_row=int(gg[a]),original_gt_id=oid,gt_degree=int(degreeG[a]),iou=float(matrix[a,b]),isolated=bool(degreeP[b]==1 and degreeG[a]==1 and oid is not None))
            reverse[i]=[(int(pr[j]),float(matrix[a,j]),int(degreeP[j])) for j in np.flatnonzero(E[a])]
    return dict(G=G,P=P,gt_origin=gi,pred_origin=pi,rows=rows,threshold=threshold,original_ids=originals,reverse=reverse)


def evidence_rows(raw,keys,needed,gt,polygon,threshold,previous,v):
    C,index=v.canonical_with_origin(raw);inverse=np.empty(len(index),int);inverse[index]=np.arange(len(index));ordered_keys=[source_key(keys[i]) for i in index]
    owners=cached_owners(gt,C,threshold,{int(inverse[i]) for i in needed},v)
    cache=_GT_CACHE[id(gt)]['segments'];pkey=tuple(map(tuple,polygon))
    if pkey not in cache:cache[pkey]=v.GT_segments(owners,polygon)
    segments=cache[pkey];output={}
    for native in needed:
        i=int(inverse[native]);z=v.independent_row_record(owners,i,C,np.arange(len(C)),polygon,segments)
        neighbors=[dict(source_key=list(ordered_keys[j]),pred_key=[int(C[j,0]),int(C[j,1])],iou=score,pred_degree=degree) for j,score,degree in owners['reverse'].get(i,[])]
        output[native]=dict(source_key=list(ordered_keys[i]),pred_key=z['pred_key'],pred_row=C[i].tolist(),row_sha256=z['row_sha256'],official_owner=z['official_owner'],original_gt_id=z['owner_gt_original_id'],gt_run=z['owner_gt_run'],gt_key=z['owner_gt_key'],gt_row_sha256=z['owner_gt_row_sha256'],gt_raw_index=z['owner_gt_raw_index'],iou=z['iou'],pred_degree=z['pred_degree'],gt_degree=z['gt_degree'],isolated=z['isolated'],qualified=z['qualified'],pred_inside=z['pred_inside'],gt_inside=z['gt_inside'],graph_gt=z['graph_gt'],owner_pred_neighbors=neighbors)
    return output,owners


def compare_evidence(ours,saved,v):
    for key,value in ours.items():
        if key=='owner_pred_neighbors':
            projected=[{k:val for k,val in r.items() if k!='union_row'} for r in saved[key]];v.compare(value,projected)
        else:v.compare(value,saved[key])


def inserted_evidence(candidate,members,gt,polygon,threshold,previous,v):
    slot=candidate['host_member_index'];host=members[slot];raw=np.asarray(host['rows']);keys=list(host['sources']);rows=[];sources=[];needed=[]
    # Frame matching factorizes by date. Include every host box at each new
    # date, with exactly this candidate's single new observation; no other
    # hypothetical candidate enters the same scene.
    native_id=raw[candidate['host_anchor_K_indices'][-1],1]
    for s in candidate['suffix']:
        date=s['frame'];ii=np.flatnonzero(raw[:,0]==date)
        for i in ii:rows.append(raw[i]);sources.append(keys[i])
        new=np.asarray(s['row'],float).copy();new[1]=native_id;needed.append(len(rows));rows.append(new);sources.append(s['source'])
    value,owners=evidence_rows(np.asarray(rows),sources,needed,gt,polygon,threshold,previous,v)
    return [value[i] for i in needed]


def audit_owners(args):
    from beetrack.evaluate import load_tracks
    previous,u,v,cert=helpers(args.repo);digest=verify_seal(args.root/'owners',args.owners_sha)
    owner_streams={h:stream_array(args.root/'owners'/(h+'.json'),'units') for h in ('35','33','W','G')}
    probes=stream_array(args.root/'owners/probes.json','groups');GT_pins=read(args.root/'owners/summary.json')['GT_sha256']
    clips={};rawGT={};reports=[]
    for ref,cat,members in frozen_groups(args):
        dest=args.root/'verification/owners'/(ref['path']+'.json');identity=dict(seal=digest,group_seal=ref['seal'],verifier_sha256=source_sha(__file__))
        if dest.exists() and dest.with_suffix('.sha256').exists():
            # GT universe is reconstructed below even on resume, to keep the
            # complete denominator sidecar independent of selected probes.
            assert source_sha(dest)==dest.with_suffix('.sha256').read_text().strip()
        video=cat['video'];path=args.repo/'runs/gt'/('dev' if video.endswith('-head') else 'val')/video/'gt/gt.txt'
        if video not in rawGT:
            assert source_sha(path)==GT_pins[path.relative_to(args.repo).as_posix()]
            rawGT[video]=load_tracks(path);universe,G,segs=GT_universe(rawGT[video],cat['polygon'],v,source_sha(path),path.relative_to(args.repo).as_posix());clips[video]=universe
        unit=next(owner_streams[cat['domain']]);probe_group=next(probes)
        assert unit['unit']==probe_group['unit']==cat['unit']
        gt=rawGT[video];universe=clips[video];actions={a['candidate_id']:a for m in unit['members'] for a in m['actions']};targets={h['host_key']:h for h in unit['hosts']}
        assert len(actions)==sum(len(m['actions']) for m in unit['members']) and set(actions)=={c['candidate_id'] for c in cat['candidates']}
        for c in cat['candidates']:
            for field in ('host_member','donor_member_index','host_member_index','host_track_id','peer_track_id','suffix','delta_windows','host_anchor_K_indices','donor_anchor_K_indices','donor_suffix_K_indices','features10'):
                assert actions[c['candidate_id']][field]==c[field]
        assert np.array_equal(unit['gt_bins'],bins(track_events(gt,cat['polygon']),cat['nf']))
        assert np.array_equal(unit['pred_sum'],sum(bins(track_events(m['rows'],cat['polygon']),cat['nf']) for m in members))
        if dest.exists() and dest.with_suffix('.sha256').exists():
            saved=read(dest);old_identity=dict(identity,verifier_sha256='3496259029970cade1d2562bb0a87adb66fe5aca238935545b41e26bfb6e35cf')
            assert saved['identity'] in (identity,old_identity);reports.append(saved)
            del unit,probe_group,actions,targets,cat,members
            gc.collect()
            continue
        prepared=[prepare(m,cat['nf']) for m in members];needed=[set() for _ in members]
        for c in cat['candidates']:
            needed[c['host_member_index']].update(c['host_anchor_K_indices']);needed[c['donor_member_index']].update(c['donor_anchor_K_indices']+c['donor_suffix_K_indices'])
        for h in cat['host_contexts']:
            if h['option_ids']:
                p=prepared[h['host_member_index']];ii=p['tracks'][h['host_parent_id']];needed[h['host_member_index']].update(int(p['index'][i]) for i in [ii[0],ii[-2],ii[-1]])
        probe_by_pid={r['canonical_gt_id']:r for r in probe_group['probes']};universe_probes={r['canonical_gt_id']:r for r in universe['runs'] if r['eligible_probe_or_null'] is not None};assert set(probe_by_pid)==set(universe_probes)
        assert len(probe_by_pid)==len(probe_group['probes'])
        for r in universe_probes.values():
            date=r['eligible_probe_or_null']['return_frame']
            for slot,m in enumerate(members):needed[slot].update(map(int,np.flatnonzero(m['rows'][:,0]==date)))
        cache={};counts={}
        for threshold in (.5,.3):
            key=str(threshold);cache[key]=[];counts[key]=defaultdict(int)
            for slot,m in enumerate(members):value,owner=evidence_rows(m['rows'],m['sources'],needed[slot],gt,cat['polygon'],threshold,previous,v);cache[key].append((value,owner))
            for c in cat['candidates']:
                h=cache[key][c['host_member_index']][0];d=cache[key][c['donor_member_index']][0]
                hr=[h[i] for i in c['host_anchor_K_indices']];dr=[d[i] for i in c['donor_anchor_K_indices']+c['donor_suffix_K_indices']]
                inserted=inserted_evidence(c,members,gt,cat['polygon'],threshold,previous,v);lab=actions[c['candidate_id']]['labels'][key]
                assert classify_evidence(hr+dr+hr+inserted)==lab['class'] and classify_evidence(hr+dr)==lab['native_class'] and classify_evidence(hr+inserted)==lab['insert_class']
                for own,saved in zip(hr,lab['original_host_rows']):compare_evidence(own,saved,v)
                for own,saved in zip(dr,lab['original_donor_rows']):compare_evidence(own,saved,v)
                for own,saved in zip(hr,lab['inserted_host_rows']):compare_evidence(own,saved,v)
                for own,saved in zip(inserted,lab['inserted_suffix_rows']):compare_evidence(own,saved,v)
                counts[key][lab['class']]+=1
            for host in cat['host_contexts']:
                if not host['option_ids']:continue
                slot=host['host_member_index'];p=prepared[slot];ii=p['tracks'][host['host_parent_id']];rawindices=[int(p['index'][i]) for i in [ii[0],ii[-2],ii[-1]]];evidence=[cache[key][slot][0][i] for i in rawindices];target=targets[host['host_key']]['supervision'][key]
                valid=classify_evidence(evidence)=='S';assert valid==target['valid']
                for a,b in zip(evidence,target['anchors']):compare_evidence(a,b,v)
                if valid:
                    owner=cache[key][slot][1];pid=evidence[0]['official_owner'];rows=owner['G'][owner['G'][:,1]==pid]
                    expected=bins(track_events(rows,cat['polygon']),cat['nf']);actual=bins([(e['frame'],e['direction']) for e in target['events']],cat['nf'])
                    assert track_events(rows,cat['polygon'])==[(e['frame'],e['direction']) for e in target['events']]
                    assert np.array_equal(expected,actual) and target['gt_run_rows_sha256']==v.array_digest(rows)
            for pid,r in universe_probes.items():
                saved=probe_by_pid[pid];probe=r['eligible_probe_or_null'];assert saved['return_frame']==probe['return_frame'] and saved['departure_frame']==probe['departure_frame']
                assert saved['initial_inside']==probe['initial_inside'] and saved['original_gt_id']==r['original_gt_id'] and saved['gt_run']==r['run']
                G=cache[key][0][1]['G'];runrows=G[G[:,1]==pid];j=probe['departure_index'];rr=probe['return_index']
                assert np.array_equal(saved['full_GT_run_rows'],runrows) and saved['full_GT_run_sha256']==v.array_digest(runrows)
                for field,i in (('departure_previous_row',j-1),('departure_row',j),('return_previous_row',rr-1),('return_row',rr)):assert np.array_equal(saved[field],runrows[i])
                observed=0;confirmed=0
                for slot,m in enumerate(members):
                    ev,owner=cache[key][slot];date=probe['return_frame'];G=owner['G'];gindices=np.flatnonzero((G[:,0]==date)&(G[:,1]==pid));assert len(gindices)==1
                    pp=np.flatnonzero(m['rows'][:,0]==date);M=v.official_iou_independent(G[gindices,2:6],m['rows'][pp,2:6]);near=[int(pp[j]) for j in np.flatnonzero(M[0]>=threshold)];unique=any(ev[i]['qualified'] and ev[i]['official_owner']==pid for i in near)
                    report=saved['thresholds'][key]['members'][slot];assert report['has_geometric_box']==bool(near) and report['confirmed_same_GT']==unique and report['GT_degree']==len(near)
                    observed+=bool(near);confirmed+=unique
                assert saved['thresholds'][key]['geometric_members']==observed and saved['thresholds'][key]['confirmed_members']==confirmed
        report=dict(passed=True,identity=identity,actions=len(actions),host_targets=len(targets),GT_return_probes=len(universe_probes),classes={t:dict(c) for t,c in counts.items()},memory=self_memory(),scope='All original host/donor and one-suffix inserted full-frame ownership; forward anchor reuse justified by unchanged dates. All source natural target events and all-GT-run return denominator/support independently checked.')
        save_receipt(dest,report);reports.append(report);print('INDEPENDENT203_OWNERS',cat['unit'],flush=True)
        del unit,probe_group,actions,targets,cache,probe_by_pid,cat,members,prepared
        h=d=hr=dr=inserted=owner=None
        gc.collect()
    for stream in [*owner_streams.values(),probes]:
        try:next(stream)
        except StopIteration:pass
        else:raise AssertionError('unexpected extra unit')
    save_receipt(args.root/'verification/gt_run_universe.json',dict(schema='peer-suffix203-independent-GT-universe-v1',clips=clips))
    result=dict(passed=True,owners_manifest=digest,units=len(reports),actions=sum(r['actions'] for r in reports),GT_universe_sha256=source_sha(args.root/'verification/gt_run_universe.json'),scope='All natural identity/target/probe ownership, both IoUs; all6clips allcanonicalGT-run dates/sides emitted once including no-probe runs. No artificial target GT used.')
    save_receipt(args.root/'verification/owners.json',result);return result


def audit_capacity(args):
    previous,u,v,cert=helpers(args.repo);digest=verify_seal(args.root/'capacity',args.capacity_sha);answer=read(args.root/'capacity/capacity.json');receipt=answer['receipt']
    assert receipt['prereg']==args.prereg and receipt['prediction_seal']==source_sha(args.root/'predictions/manifest.json')
    assert receipt['compact_input_sha256']==source_sha(args.root/'capacity/compact_input.json')
    proofs=[];gains=defaultdict(lambda:defaultdict(list))
    for hive in ('35','33'):
        for unit in stream_array(args.root/'owners'/(hive+'.json'),'units'):
            actions=[dict(a,host_member=str(m['member']),host_member_index=m['member_index']) for m in unit['members'] for a in m['actions']]
            problem=dict(unit,actions=actions,member_indices=[m['member_index'] for m in unit['members']])
            for family in ('S','P','A'):
                saved=answer['units'][unit['unit']][family];result=cert.verify_answer(problem,family,saved)
                assert saved['lower_bound']==result['lower_bound'] and saved['upper_bound']==result['upper_bound']
                gains[hive][family].append((Fraction(100*(saved['baseline_numerator']-saved['upper_bound']),saved['denominator']),Fraction(100*(saved['baseline_numerator']-saved['lower_bound']),saved['denominator'])))
                proofs.append(dict(unit=unit['unit'],family=family,lower=result['lower_bound'],upper=result['upper_bound'],tight=result['optimal']))
            del unit,actions,problem
    assert len(proofs)==24
    heads={h:{family:dict(witness_gain=float(sum((p[0] for p in pairs),Fraction())/len(pairs)),gain_ceiling=float(sum((p[1] for p in pairs),Fraction())/len(pairs))) for family,pairs in ff.items()} for h,ff in gains.items()}
    for hive,ff in heads.items():
        for family,values in ff.items():
            for key,value in values.items():assert abs(value-answer['heads'][hive][family][key])<1e-12
    report=dict(passed=True,capacity_manifest=digest,proofs=proofs,heads=heads,scope='All24 general Ax<=1 host/source/overlap resource incidence and exact integer rational L1 certificates/witnesses. No bipartite TU assumption or F1 optimum claim.')
    save_receipt(args.root/'verification/capacity.json',report);return report


def audit_models(args):
    previous,u,v,cert=helpers(args.repo);module=load_module('independent203_models',Path(__file__).with_name('verify203_models.py'))
    return module.audit_models(args,dict(previous=previous,replay=module_from_current(),certificate=cert))


def module_from_current():
    import sys
    if __name__ in sys.modules:return sys.modules[__name__]
    from types import SimpleNamespace
    return SimpleNamespace(**{k:v for k,v in globals().items() if not k.startswith('__')})


def audit_scores(args):
    previous,u,v,cert=helpers(args.repo);module=load_module('independent203_scores',Path(__file__).with_name('verify203_scores.py'))
    chain=read(args.root/'score/RUNNING.json');summary=read(args.root/'score/summary.json');assert chain==summary['input_receipt'] and chain['prereg']==args.prereg
    assert chain['source_seals']=={name:source_sha(args.root/name/'manifest.json') for name in ('frozen','owners','predictions')}
    assert chain['capacity_seal']==source_sha(args.root/'capacity/manifest.json') and chain['source202_score_seal']==source_sha(args.source202_score/'manifest.json')
    return module.audit_scores(args,dict(previous=previous,replay=module_from_current(),native=u))


def materialize_independent(members,candidates,selected):
    byid={c['candidate_id']:c for c in candidates};assert len(selected)==len(set(selected)) and set(selected)<=set(byid)
    output={str(m['member']):np.asarray(m['rows'],float).copy() for m in members};added=defaultdict(list);used=set();hosts=set();sources=defaultdict(list)
    for cid in sorted(selected):
        c=byid[cid];host=c['host_member'];key=(host,c['host_track_id']);assert key not in hosts;hosts.add(key)
        native_id=output[host][c['host_anchor_row_indices'][-1],1]
        for s in c['suffix']:
            source=source_key(s['source']);assert source not in used;used.add(source);r=np.asarray(s['row'],float).copy();r[1]=native_id
            assert all(old[0]!=r[0] or observed_iou(old,r)<.5 for old in added[host])
            assert not np.any((output[host][:,0]==r[0])&(output[host][:,1]==r[1]));added[host].append(r);sources[host].append(source)
    for key in output:
        if added[key]:output[key]=np.r_[output[key],np.asarray(added[key])]
    return output,sources


def lse(values):
    top=max(values);return top+math.log(sum(math.exp(x-top) for x in values))


def marginal_loss(theta,contexts,ridge=.2):
    theta=np.asarray(theta,float);value=ridge/2*float(theta@theta);gradient=ridge*theta
    for c in contexts:
        A=np.asarray(c['matrix'],float);y=np.asarray(c['compatible'],bool);assert y.any();z=A@theta;allz=lse(z);yes=lse(z[y])
        value+=c['weight']*(allz-yes);gradient+=c['weight']*(np.exp(z-allz)@A-np.exp(z[y]-yes)@A[y])
    return value,gradient


def synthetic(repo=None):
    row=lambda f,p,x:[f,p,x,0,4,4,.9]
    h=[row(f,1,x) for f,x in [(747,-6),(748,-5),(749,-4)]];p=h+[row(f,1,x) for f,x in [(750,-3),(751,-2),(752,-1),(753,1)]]
    member=lambda name,r:dict(member=name,rows=r,sources=[('synthetic-'+name,i) for i in range(len(r))])
    mm=[member('H',h),member('P',p),member('Q',p)];poly=[[0,-10],[100,-10],[100,100],[0,100]]
    cs=supply_independent(mm,poly,800);hc=[c for c in cs if c['host_member']=='H'];assert len(hc)==2
    assert [s['frame'] for s in hc[0]['suffix']]==[750,751,752,753] and abs(hc[0]['delta_windows']).sum()==30
    out,_=materialize_independent(mm,cs,[hc[0]['candidate_id']]);assert len(out['H'])==7
    try:materialize_independent(mm,cs,[c['candidate_id'] for c in hc])
    except AssertionError:pass
    else:raise AssertionError('host reused')
    art=artificial_choice(mm[1],'unit',1,800);assert len(art)==1
    target=np.asarray(mm[1]['rows']);keep=np.ones(len(target),bool);keep[art[0]['hidden_K_indices']]=False
    assert len(target[keep])==art[0]['visible_host_observations'] and max(target[keep,0])==art[0]['cutoff_frame']
    ctx=[dict(matrix=[[0.,0.],[1.,1.],[-1.,1.]],compatible=[False,True,True],weight=1.)];theta=np.array([.3,-.1]);loss,g=marginal_loss(theta,ctx)
    for i in range(2):
        e=np.eye(2)[i]*1e-6;numeric=(marginal_loss(theta+e,ctx)[0]-marginal_loss(theta-e,ctx)[0])/2e-6;assert abs(numeric-g[i])<1e-8
    return dict(passed=True,field_opened=False,supply_anchor_and_source=True,global_materialization=True,artificial_hash_before_peer=True,marginal_gradient_finite_difference=True,scope='synthetic only; marginal likelihood is generally nonconvex')


def supply_contract(repo):
    import sys
    sys.path.insert(0,str(repo));producer=load_module('suffix203_contract_producer',Path(__file__).with_name('freeze203.py'))
    def row(f,p,x):return [f,p,x,0,4,4,.9]
    def member(k,r):return dict(member=k,rows=np.asarray(r,float),sources=[('synthetic-'+k,i) for i in range(len(r))])
    h=[row(f,1,x) for f,x in [(747,-6),(748,-5),(749,-4)]];p=h+[row(f,1,x) for f,x in [(750,-3),(751,-2),(752,-1),(753,1)]]
    poly=[[0,-10],[100,-10],[100,100],[0,100]];fixtures=[
        [member('H',h),member('P',p),member('Q',p)],
        [member('H',h),member('P',[r for r in p if r[0]!=751]),member('Q',p)],
        [member('H',h+[row(751,8,-2)]),member('P',p),member('Q',p)],
        [member('H',h),member('P',p+[row(749,8,-4)]),member('Q',p)]]
    comparisons=0;artcases=0
    context=dict(unit='toy',domain='W',group=1,video='synthetic',nf=800)
    for mm in fixtures:
        actual=producer.indexed_catalog(mm,poly,800);expected=supply_independent(mm,poly,800);table={c['candidate_id']:c for c in actual}
        assert set(table)=={c['candidate_id'] for c in expected}
        prep=[prepare(m,800) for m in mm]
        for c in expected:
            saved=table[c['candidate_id']]
            for field in ('host_member','peer_member','host_track_id','peer_track_id','anchor_frames','host_anchor_row_indices','host_anchor_rows','peer_anchor_rows','host_anchor_sources','peer_anchor_sources','stop'):
                assert c[field]==saved[field]
            assert [s['source'] for s in c['suffix']]==[s['source'] for s in saved['suffix']]
            assert [s['peer_row_index'] for s in c['suffix']]==[s['peer_row_index'] for s in saved['suffix']]
            assert np.array_equal(c['delta_windows'],saved['delta_windows']);comparisons+=1
        enriched=producer.enrich(actual,mm,context)
        for c in expected:
            saved=next(x for x in enriched if x['candidate_id']==c['candidate_id']);producer.features(saved,mm,poly)
            own,stratum=observed_features(c,prep,poly);np.testing.assert_allclose(own,saved['features10'],rtol=1e-13,atol=1e-13)
        art=producer.artificial_catalog(mm,poly,800,context);ind=artificial_independent(mm,'toy',poly,800)
        assert len(art['cases'])==len(ind)
        for own in ind:
            saved=next(c for c in art['cases'] if c['host_parent_id']==own['host_parent_id'] and c['host_member_index']==int(own['cutoff_hash_input'].split('|')[2]))
            for field in ('host_parent_id','cutoff_frame','cutoff_sha256','cutoff_hash_input','hidden_K_indices','visible_host_observations','full_host_observations','eligible_cutoffs'):assert own[field]==saved[field]
            assert np.array_equal(own['baseline_windows'],saved['baseline_windows']) and np.array_equal(own['target_windows'],saved['target_windows'])
            assert len(own['candidates'])==len(saved['candidates']);artcases+=1
    return dict(passed=True,natural_candidates_compared=comparisons,artificial_cases_compared=artcases,field_opened=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--supply-contract-tests',action='store_true');parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source202',type=Path,required=True);parser.add_argument('--source202-score',type=Path,required=True)
    parser.add_argument('--prereg',default='69d52924d596d74c56edd25f1bc9bb7c8e66edb4');parser.add_argument('--field-go',action='store_true');parser.add_argument('--score-worker-finished',action='store_true')
    parser.add_argument('--phase',choices=('supply','owners','models','capacity','scores','all'));parser.add_argument('--resume-after',choices=('supply','owners','models','capacity','scores'))
    for stage in ('frozen','owners','predictions','capacity','score'):parser.add_argument('--'+stage+'-sha')
    args=parser.parse_args()
    if args.phase:
        assert args.field_go and args.score_worker_finished,'Final sole-CPU GO plus score exit required'
        import sys
        sys.path.insert(0,str(args.repo))
        for stage in ('frozen','owners','predictions','capacity','score'):verify_seal(args.root/stage,getattr(args,stage+'_sha'))
        stages=dict(supply=audit_supply,owners=audit_owners,models=audit_models,capacity=audit_capacity,scores=audit_scores)
        if args.phase=='all':
            output={};skip=list(stages).index(args.resume_after) if args.resume_after else -1
            for index,(name,fn) in enumerate(stages.items()):
                if index<=skip:
                    path=args.root/'verification'/(name+'.json');assert source_sha(path)==path.with_suffix('.sha256').read_text().strip();output[name]=read(path);assert output[name]['passed']
                else:output[name]=fn(args)
            result=dict(schema='peer-suffix203-independent-v1',passed=True,prereg=args.prereg,verifier_sha256=source_sha(__file__),helper_sha256={name:source_sha(Path(__file__).with_name(name)) for name in ('verify203_certificate.py','verify203_models.py','verify203_scores.py')},
                input_seals={name:source_sha(args.root/name/'manifest.json') for name in ('frozen','owners','predictions','capacity','score')},
                phases={name:dict(path=name+'.json',sha256=source_sha(args.root/'verification'/(name+'.json'))) for name in output},scope={name:value['scope'] for name,value in output.items()},GT_run_universe_sha256=source_sha(args.root/'verification/gt_run_universe.json'),memory=self_memory())
            save_receipt(args.root/'verification/independent.json',result)
        else:result=stages[args.phase](args)
    else:result=supply_contract(args.repo) if args.supply_contract_tests else synthetic()
    print(json.dumps(result,indent=2))
