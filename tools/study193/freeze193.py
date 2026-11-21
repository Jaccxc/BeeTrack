"""Portable GT-free §193 native reconstruction from the original detector caches.

All60 member/videos are rebuilt via native .40 stage1 and the pinned per-domain
TrackletLink model. No private cache, score189 table, GT or old wrapper is loaded.
Prepare registered CS/pl head caches seeds3..14 and loho_noW/noG val caches0..8.
"""
import os
import sys
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import cv2
import numpy as np
if '--repo' in sys.argv:
    sys.path.insert(0,str(Path(sys.argv[sys.argv.index('--repo')+1]).resolve()))
from beetrack.evaluate import by_track, zone_tracks

PREREG='f161c64a6310c663d6cb079e83abee335d90e324'

FEATURES=('in','boxes','frames','missing','tracklets','start_d','end_d','straight','speed','near')
MODEL_SHA={'m_noW':'af0bfea1e3b2a8c90e3e0b1022c4e36161a88e20be3b122fb3a4e6323bb8a0a1',
           'm_noG':'58d0a446fb578e88b8372176d7e81966a4d27838eafce333cb0496a14099a889',
           'm_dev_strict':'fdd788e0dba0e6fa1c2d06feec3127bf84d5e7b2d214cc0f793c452270403dcf'}
CLIPS=(('35','dev','BEE24-35-head',3950),('33','dev','BEE24-33-head',3950),
       ('W','val','BEE24-10',250),('W','val','BEE24-15',240),
       ('G','val','BEE24-24',75),('G','val','BEE24-28',75))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def members():
    result=[]
    for h in ('35','33'):
        for k in range(3,15):
            src='pl' if h=='35' and k in (3,6,7) else 'cs'
            result.append(dict(member=len(result),hive=h,group=1+k//3,seed=k,tag=f'{src}{h}_s{k}'))
    for h in ('W','G'):
        for k in range(9):
            result.append(dict(member=len(result),hive=h,group=1+k//3,seed=k,tag=f'loho_no{h}_s{k}'))
    assert len(result)==42
    return result


def extract(rows,s1,polygon,nframes):
    rows=np.asarray(rows,dtype=np.float64); s1=np.asarray(s1,dtype=np.float64)
    if rows.ndim!=2 or rows.shape[1]!=7 or s1.shape!=(len(rows),):
        raise ValueError('expected full-precision Nx7 rows and aligned N stage1 IDs')
    if not np.isfinite(rows).all() or not np.isfinite(s1).all(): raise ValueError('nonfinite rows')
    if np.any(rows[:,0]<1) or np.any(rows[:,0]>nframes) or np.any(rows[:,:2]!=np.rint(rows[:,:2])):
        raise ValueError('invalid native clock or IDs')
    T=by_track(np.c_[rows,s1]); poly=np.asarray(polygon,np.float32)
    if not np.array_equal(T[:,:7],by_track(rows)): raise ValueError('canonical row alignment changed')
    out=[]
    for tid,(direction,start,end,cross) in zone_tracks(T,polygon).items():
        if not direction: continue
        same=[j for j,d in cross if d==direction]; tr=T[start:end]
        centers=tr[:,2:4]+tr[:,4:6]/2
        body=float(np.median(np.maximum(tr[:,4],tr[:,5])))
        step=np.hypot(*np.diff(centers,axis=0).T); gaps=np.diff(tr[:,0])
        if not (body>0 and step.sum()>0 and np.all(gaps>0)): raise ValueError('invalid event track')
        distance=lambda p:cv2.pointPolygonTest(poly,(float(p[0]),float(p[1])),True)/body
        n=len(tr); duration=tr[-1,0]-tr[0,0]+1
        f=[float(direction=='in'),float(n),float(duration),float(1-n/duration),float(len(np.unique(tr[:,7]))),distance(centers[0]),distance(centers[-1]),float(np.hypot(*(centers[-1]-centers[0]))/step.sum()),float(np.median(step/gaps)/body)]
        index=same[-1] if same else end-1
        b=T[index,2:6]
        out.append(dict(track_id=int(tid),frame=int(T[index,0]),sign=1 if direction=='in' else -1,features=f,
                        canonical_row=int(index),box=[float(b[0]),float(b[1]),float(b[0]+b[2]),float(b[1]+b[3])]))
    frames=np.array([x['frame'] for x in out])
    near=(np.abs(frames[:,None]-frames[None,:])<=25).sum(1)-1
    for x,n in zip(out,near): x['features'].append(float(n))
    return out


def counts(events,nframes):
    windows=[]
    for offset in range(30):
        a=np.zeros(((nframes+25*offset+749)//750,2),int)
        for e in events: a[(e['frame']+25*offset-1)//750,int(e['sign']<0)]+=1
        windows.extend(a.tolist())
    seconds=np.zeros(((nframes+24)//25,2),int)
    for e in events: seconds[(e['frame']-1)//25,int(e['sign']<0)]+=1
    return windows,seconds.tolist()


def stage1_ids(stage1,rows):
    """Exact §189 s1_ids mapping; no rounded coordinates or GT identities."""
    cols=[0,2,3,4,5,6]; n1=len(stage1)
    if not len(rows): return np.empty(0,float)
    A=np.r_[stage1[:,cols],rows[:,cols]]
    flag=np.r_[np.zeros(n1),np.ones(len(rows))]
    order=np.lexsort((flag,*A[:,::-1].T))
    ordered=A[order]; isrow=order>=n1
    eq=(ordered[1:]==ordered[:-1]).all(1)
    assert not isrow[0] and not (eq & ~isrow[1:]).any(), 'duplicate stage1 boxes'
    r=np.flatnonzero(isrow[1:])+1
    assert eq[r-1].all() and not isrow[r-1].any()
    out=np.full(len(rows),np.nan)
    out[order[r]-n1]=stage1[order[r-1],1]
    assert not np.isnan(out).any()
    return out


def seal_receipt(root):
    """Small prediction receipt binding the complete deployment freeze."""
    root=Path(root)
    assert (root/'SEALED.sha256').read_text().strip()==sha(root/'manifest.json')
    return dict(schema='budget193-freeze-receipt-v1',manifest_sha256=sha(root/'manifest.json'),
                deployment_sha256=sha(root/'deployment.json'),seal_file_sha256=sha(root/'SEALED.sha256'))


def load_groups(root):
    """Verified adapter for training/scoring; returns no GT or outcome fields.

    Each return item is one hive/group/actual-video, with exactly 3 ordered
    members, raw full-precision rows and official canonical prediction IDs.
    """
    root=Path(root)
    assert (root/'SEALED.sha256').read_text().strip()==sha(root/'manifest.json')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()): raise ValueError('unsafe manifest path')
        assert sha(p)==digest
    data=json.loads((root/'deployment.json').read_text())
    assert data['schema']=='budget193-deployment-v1'
    result=[]
    for group in data['groups']:
        clip=data['clips'][group['clip_index']]
        rows=[]
        for uid in group['units']:
            unit=data['units'][uid]
            with np.load(root/unit['rows_file'],allow_pickle=False) as z:
                native=z['rows']
            records=[dict(pid=e['track_id'],frame=e['frame'],direction=e['sign'],features10=e['features'],box=e['box'],event_index=e['event']) for e in data['events'] if e['unit']==uid]
            rows.append(dict(member=str(unit['member']),tag=data['members'][unit['member']]['tag'],rows=native,event_records=records,windows=unit['windows'],seconds=unit['seconds'],unit_index=uid,
                             raw_receipt=dict(file=unit['rows_file'],sha256=unit['frozen_rows_sha256'],source=unit['source_receipt']),linker_name=unit['linker_name'],linker_receipt=unit['linker_receipt']))
        linker=rows[0]['linker_receipt']
        assert all(m['linker_receipt']==linker for m in rows)
        result.append(dict(unit=f"{group['hive']}_g{group['group']}_{clip['clip']}",domain=group['hive'],video=clip['clip'],group=group['group'],nf=clip['nframes'],polygon=clip['polygon'],split=clip['split'],members=rows,freeze_receipt=seal_receipt(root),linker_model=linker['name'],linker_sha256=linker['sha256'],linker_excluded_domains=linker['excluded_domains']))
    return result


def freeze(args):
    if args.prereg!=PREREG: raise ValueError('registered full §193 preregistration commit required')
    proc=subprocess.run(['git','show',f'{args.prereg}:docs/experiment_log.md'],cwd=args.repo,capture_output=True,check=True)
    if not re.search(r'^## 193\.',proc.stdout.decode('utf-8'),re.M):
        raise ValueError('commit does not register section193')
    dest=args.out/'frozen'
    if dest.exists(): raise FileExistsError(dest)
    # Fixed known inputs; no directory scan can accidentally include new seeds.
    zonepath=args.repo/'beetrack/zones.json'; zones=json.loads(zonepath.read_text(encoding='utf-8'))
    mm=members(); source_hashes={'zones':sha(zonepath),'extractor':sha(__file__)}
    for name in ('evaluate.py','common.py','p0.py','track.py','link.py'):
        source_hashes['beetrack/'+name]=sha(args.repo/'beetrack'/name)
    for name,pinned in MODEL_SHA.items():
        actual=sha(args.repo/'beetrack/models'/f'{name}.json')
        if actual!=pinned: raise ValueError('linker model digest changed')
        source_hashes[f'beetrack/models/{name}.json']=actual
    # Import the native builder only after preregistration and model checks.
    from beetrack import p0 as P
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    linkers={name:P.link_model(name) for name in MODEL_SHA}
    args.out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.freeze193-',dir=args.out) as temp:
        target=Path(temp)/'payload'; target.mkdir()
        (target/'rows').mkdir()
        units=[]; events=[]; groups=[]
        for m in mm:
            for ci,(h,split,clip,nf) in enumerate(CLIPS):
                if h!=m['hive']: continue
                dp=args.repo/'runs/dets'/m['tag']/split/f'{clip}.npy'
                detector_sha=sha(dp)
                linker_name='m_no'+h if h in ('W','G') else 'm_dev_strict'
                linker_receipt=dict(name=linker_name,sha256=MODEL_SHA[linker_name],excluded_domains=[h] if h in ('W','G') else ['33','35'])
                D=np.load(dp,allow_pickle=False)
                if D.ndim!=2 or D.shape[1]!=6 or not np.isfinite(D).all():
                    raise ValueError('expected finite native Nx6 detection cache')
                selected=D[D[:,5]>=0.40]
                if (not np.equal(selected[:,0],np.rint(selected[:,0])).all() or
                        np.any(selected[:,0]<1) or np.any(selected[:,0]>nf) or
                        np.any(selected[:,3:5]<=selected[:,1:3])):
                    raise ValueError('invalid retained native detections')
                S1=P.stage1(selected,nf)
                raw=np.asarray(P.p0(S1,linkers[linker_name]),dtype=np.float64).reshape(-1,7)
                s1=stage1_ids(S1,raw)
                source_receipt=dict(kind='native_clean_rebuild',detector_sha256=detector_sha,
                                    detector_path=dp.relative_to(args.repo).as_posix(),
                                    stage1='beetrack.p0.stage1(D[D[:,5]>=0.40],nf)',
                                    p0='beetrack.p0.p0(stage1,'+linker_name+')',
                                    trim='native P0 link .15 then drop tracks shorter than3; no MOT serialization',
                                    mapping='exact float row join preserving raw native order',linker=linker_receipt)
                if raw.dtype!=np.float64: raise ValueError('original rows no longer float64')
                assert sha(dp)==detector_sha, 'detector changed during freeze'
                ev=extract(raw,s1,zones[clip],nf)
                uid=len(units)
                for local,e in enumerate(ev):
                    events.append(dict(event=len(events),unit=uid,member=m['member'],hive=h,group=m['group'],clip=clip,clip_index=ci,local_event=local,**e))
                wc,sc=counts(ev,nf)
                file=f"rows/u{uid:02d}.npz"
                np.savez(target/file,rows=raw,s1=s1)
                with np.load(target/file,allow_pickle=False) as reread:
                    assert np.array_equal(reread['rows'],raw) and np.array_equal(reread['s1'],s1)
                units.append(dict(unit=uid,member=m['member'],hive=h,group=m['group'],clip_index=ci,clip=clip,split=split,nframes=nf,rows_file=file,row_count=len(raw),source_receipt=source_receipt,linker_name=linker_name,linker_receipt=linker_receipt,detector_sha256=detector_sha,frozen_rows_sha256=sha(target/file),windows=wc,seconds=sc))
        assert len(units)==60
        for h in ('35','33','W','G'):
            for g in sorted(set(m['group'] for m in mm if m['hive']==h)):
                mids=[m['member'] for m in mm if m['hive']==h and m['group']==g]; assert len(mids)==3
                for ci,(_,split,clip,nf) in enumerate(CLIPS):
                    us=[next(u for u in units if u['member']==m and u['clip_index']==ci) for m in mids] if CLIPS[ci][0]==h else []
                    if not us: continue
                    groups.append(dict(hive=h,group=g,clip_index=ci,members=mids,units=[u['unit'] for u in us],member_windows=[u['windows'] for u in us],member_seconds=[u['seconds'] for u in us]))
        payload=dict(schema='budget193-deployment-v1',prereg=args.prereg,threshold=0.40,feature_names=FEATURES,members=mm,clips=[dict(clip_index=i,hive=h,split=sp,clip=clip,nframes=nf,polygon=zones[clip]) for i,(h,sp,clip,nf) in enumerate(CLIPS)],units=units,events=events,groups=groups,source_hashes=source_hashes,clock=dict(fps=25,window_frames=750,offset_seconds=list(range(30)),frame_origin=1,second='(frame-1)//25',windows='stack offsets, bin=(frame+25*offset-1)//750; partial windows included'),prediction_fields=['features','member_windows','member_seconds'],excluded=['harm','p_loho','matched','GT','score'],precision='rows float64 preserved exactly; canonical track IDs from official by_track')
        (target/'deployment.json').write_text(json.dumps(payload,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
        manifest={p.relative_to(target).as_posix():sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        (target/'SEALED.sha256').write_text(sha(target/'manifest.json')+'\n')
        target.rename(dest)
    print('Frozen deployment-only inputs:',dest)


def selfcheck():
    poly=[[0,0],[10,0],[10,10],[0,10]]
    raw=np.array([[1,1,11,4,2,2,.8],[2,1,9,4,2,2,.8],[3,1,7,4,2,2,.8],[30,2,3,4,2,2,.8],[31,2,11,4,2,2,.8]],float)
    s1=np.array([1,1,2,3,3],float)
    e=extract(raw,s1,poly,40)
    assert [(x['track_id'],x['frame'],x['sign']) for x in e]==[(1,2,1),(2,31,-1)]
    assert len(e[0]['features'])==10 and e[0]['features'][4]==2
    changed=raw.copy();changed[:,6]=.123
    assert extract(changed,s1,poly,40)==e
    w,s=counts(e,40)
    assert np.array(w).sum()==60 and np.array(s).sum()==2
    assert len(members())==42
    stage=raw.copy();stage[:,1]=[4,4,5,6,6]
    np.testing.assert_array_equal(stage1_ids(stage,raw),[4,4,5,6,6])
    assert stage1_ids(stage,raw[:0]).shape==(0,)
    print('freeze193 synthetic events, features, score independence, exact stage1 mapping, clocks passed')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--out',type=Path);ap.add_argument('--prereg')
    ap.add_argument('--self-check',action='store_true')
    args=ap.parse_args()
    if args.self_check: selfcheck()
    else:
        if not args.out or not args.prereg: ap.error('--out and --prereg required for real freeze')
        args.repo=args.repo.resolve();args.out=args.out.resolve()
        freeze(args)
