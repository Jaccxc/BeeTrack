"""Source-only214 endpoint roster and one-decode-per-frame producer helpers.

Original18 K only. No loader for212 receiver states, GT, labels or assessments.
Pixel/G6 specification is injected by the separately owned pure feature API.
No field CLI before the final registered geometry/pixel contract is fixed.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from collections import defaultdict
import hashlib
import gzip
import io
import json
from pathlib import Path
import argparse
import sys
import re
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
cv2.setNumThreads(1)

SOURCE212='61774c5f725073b1d887fdc06a8dd3a390a7566406eef12fe31c3fc859fd3b2b'
SOURCE208='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94'
MARKER='[endpoint-image-recenter-v1]'
FEATURE_SHA='63ff00043717c889d6ff674cf44851d72ca44f7b1ee6aa144b6aa6a8609d2165'
SPECS=(('35','BEE24-35-head','BEE24-35',3950),('33','BEE24-33-head','BEE24-33',3950),
       ('W','BEE24-10','BEE24-10',250),('W','BEE24-15','BEE24-15',240),
       ('G','BEE24-24','BEE24-24',75),('G','BEE24-28','BEE24-28',75))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_plan(repo):
    """Paths from208's explicit train-only reader, no directory/file probing."""
    repo=Path(repo)
    return [dict(domain=d,clip=clip,capture=capture,nf=nf,capture_first_frame=1,
                 image_root=(repo/'data/BEE24/train'/capture/'img1').as_posix(),slots=[0,1,2],
                 seeds=[3,4,5] if d in ('35','33') else [0,1,2]) for d,clip,capture,nf in SPECS]


def save(path,obj):
    with Path(path).open('w',encoding='utf8',newline='\n') as f:json.dump(obj,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')


def save_gz(path,obj):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
            with io.TextIOWrapper(gz,encoding='utf8',newline='\n') as f:json.dump(obj,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')


def verify(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('center214seal mismatch')
    files=json.loads((root/'manifest.json').read_text())
    for name,digest in files.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError('center214file drift '+name)
    return files


def endpoints(rows,sources,nf,context):
    """Head/tail of every ORIGINAL >25gap run, singleton exactly once.

    Geometry-invalid endpoints stay unknown in the ledger. Timestamp/source
    conflicts fail the scene instead of silently removing denominator rows.
    """
    from beetrack.evaluate import by_track
    R=np.asarray(rows,float)
    if R.ndim!=2 or R.shape[1]!=7:raise ValueError('native[M,7] required')
    if not np.isfinite(R[:,:2]).all() or (R[:,:2]!=np.floor(R[:,:2])).any() or (R[:,0]<1).any() or (R[:,0]>nf).any() or (R[:,1]<0).any():
        raise ValueError('invalid native clock/ID')
    if len(set(map(tuple,R[:,:2])))!=len(R):raise ValueError('duplicate native date/ID scene')
    keys=[tuple(s) for s in sources]
    if len(keys)!=len(R) or len(set(keys))!=len(keys):raise ValueError('source lineage cardinality/collision')
    aliases=set()
    for pid in np.unique(R[:,1]):
        dates=np.sort(R[R[:,1]==pid,0])
        for run in range(1+int(np.sum(np.diff(dates)>25))):
            alias=int(pid)+run*100000
            if alias in aliases:raise ValueError('canonical alias collision')
            aliases.add(alias)
    tagged=by_track(np.column_stack((R,np.arange(len(R)))))
    records=[];canon=np.empty(len(R),np.int64)
    canon[tagged[:,-1].astype(int)]=tagged[:,1].astype(np.int64)
    ids,starts=np.unique(tagged[:,1],return_index=True)
    for pid,a,b in zip(ids,starts,[*starts[1:],len(tagged)]):
        ix=tagged[a:b,-1].astype(int);n=len(ix)
        choices=[('single',0,None)] if n==1 else [('start',0,1),('end',n-1,n-2)]
        for role,position,neighbor_position in choices:
            j=int(ix[position]);neighbor=None if neighbor_position is None else int(ix[neighbor_position]);row=R[j]
            key=f'{context["domain"]}|{context["clip"]}|{context["native_slot"]}|{int(pid)}|{role}|{j}'
            records.append(dict(endpoint_id=key,**context,canonical_id=int(pid),native_id=int(row[1]),raw_index=j,
                frame=int(row[0]),role=role,source_key=list(keys[j]),box=row[2:6].tolist(),score=float(row[6]),nobs=n,
                nearest_other_observation_raw_index=neighbor,
                nearest_other_observation_frame=None if neighbor is None else int(R[neighbor,0]),
                nearest_other_observation_center=None if neighbor is None else (R[neighbor,2:4]+R[neighbor,4:6]/2).tolist(),
                frame_gap=None if neighbor is None else abs(int(row[0]-R[neighbor,0]))))
    return dict(schema='center214-endpoint-roster-prototype-v1',records=records,canon=canon,
                raw_rows=len(R),canonical_runs=len(ids),endpoints=len(records),
                singletons=sum(e['role']=='single' for e in records),all_original_runs_retained=True)


def geometry_gate(endpoint,polygon,image_shape):
    box=np.asarray(endpoint['box'],float);h,w=map(int,image_shape[:2])
    if box.shape!=(4,) or not np.isfinite(box).all() or (box[2:]<=0).any():
        return dict(status='unknown',eligible=False,reason='nonpositive_or_unknown_box',image_shape=[h,w])
    x,y,bw,bh=box;center=np.array([x+bw/2,y+bh/2]);diag=float(np.hypot(bw,bh))
    crop=[float(center[0]-bw),float(center[1]-bh),float(center[0]+bw),float(center[1]+bh)]
    distance=float(cv2.pointPolygonTest(np.asarray(polygon,np.float32),tuple(map(float,center)),True))
    strict=crop[0]>=0 and crop[1]>=0 and crop[2]<=w-1 and crop[3]<=h-1
    near=abs(distance)<=.25*diag
    reasons=[]
    if not strict:reasons.append('2w2h_crop_outside_closed_pixel_domain')
    if not near:reasons.append('outside_quarter_diagonal_polygon_band')
    return dict(status='eligible' if not reasons else 'ineligible',eligible=not reasons,reason='OK' if not reasons else '+'.join(reasons),
                image_shape=[h,w],crop_xyxy=crop,center=center.tolist(),diagonal=diag,signed_polygon_distance=distance,
                crop_inside=bool(strict),near_polygon=bool(near),crop_bounds_rule='closed [0,width-1]x[0,height-1]')


def process_frames(records,polygon,nf,read_bgr,extract):
    """Pure callback integration; each represented frame decoded once.

    A missing/unreadable source frame creates unknown endpoint records, never
    a fabricated feature. No feature-validity condition changes the roster.
    """
    bydate=defaultdict(list)
    for e in records:
        if not 1<=e['frame']<=nf:raise ValueError('endpoint outside original clip')
        bydate[e['frame']].append(e)
    result=[];receipts=[]
    for frame,items in sorted(bydate.items()):
        image=read_bgr(frame)
        if image is None:
            result.extend(dict(e,gate=dict(status='unknown',eligible=False,reason='source_frame_unavailable'),features=None) for e in items)
            receipts.append(dict(frame=frame,available=False));continue
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('native BGR uint8 required')
        receipts.append(dict(frame=frame,available=True,width=image.shape[1],height=image.shape[0]))
        for e in items:
            gate=geometry_gate(e,polygon,image.shape)
            result.append(dict(e,gate=gate,features=extract(image,e) if gate['eligible'] else None))
    return result,receipts


def load_clip(root,record):
    folder=Path(root)/record['path']
    with gzip.open(folder/'roster.json.gz','rt',encoding='utf8') as f:roster=json.load(f)
    return dict(roster,folder=folder)


def load_scene(clip,slot):
    with np.load(clip['folder']/'native.npz',allow_pickle=False) as z:rows=z['receiver' if slot==0 else 'peer'+str(slot)]
    with gzip.open(clip['folder']/'sources.json.gz','rt',encoding='utf8') as f:sources=json.load(f)
    return dict(rows=rows,sources=sources['receiver' if slot==0 else 'peers'][slot-1] if slot else sources['receiver'],
                endpoints=[e for e in clip['records'] if e['native_slot']==slot],metadata=clip['members'][slot])


def run(args):
    import features214 as F
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('full public prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    text=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if text.count(MARKER)!=1:raise ValueError('unique214marker missing')
    if args.supplement:
        if not re.fullmatch('[0-9a-f]{40}',args.supplement):raise ValueError('full supplement SHA required')
        subprocess.run(['git','merge-base','--is-ancestor',args.supplement,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    if args.out.resolve().drive.upper()!='D:':raise ValueError('D: output required')
    if sha(F.__file__)!=FEATURE_SHA:raise ValueError('feature214 source changed')
    if args.out.exists():verify(args.out);raise FileExistsError('immutable complete214root already exists')
    for root,digest in ((args.source212,SOURCE212),(args.inputs208,SOURCE208)):
        if sha(root/'manifest.json')!=digest or (root/'SEALED.sha256').read_text().strip()!=digest:raise ValueError('old source seal mismatch')
    m212=json.loads((args.source212/'manifest.json').read_text());m208=json.loads((args.inputs208/'manifest.json').read_text())
    if sha(args.source212/'deployment.json')!=m212['deployment.json']:raise ValueError('212deployment drift')
    top=json.loads((args.source212/'deployment.json').read_text())
    # Only original K native/sources and original-K metadata below. No clip.json
    # (which includes86receiverstates), state map, featurebank or assessment read.
    refs={(r['domain'],r['clip']):r for r in top['clips']}
    if set(refs)!={(d,c) for d,c,capture,nf in SPECS}:raise ValueError('six old clips only')
    args.out.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.'+args.out.name+'-',dir=args.out.parent))
    outputs=[]
    try:
        for domain,clip,capture,nf in SPECS:
            path=refs[domain,clip]['path'];source=args.source212/path;old208=args.inputs208/domain/clip
            for filename in ('native.npz','sources.json.gz',*[f'K/m{i}/events.json.gz' for i in range(3)]):
                if sha(source/filename)!=m212[path+'/'+filename]:raise ValueError('named originalK carrier drift')
            for filename in ('metadata.json','image_receipts.json'):
                if sha(old208/filename)!=m208[f'{domain}/{clip}/{filename}']:raise ValueError('208nominal metadata drift')
            meta=json.loads((old208/'metadata.json').read_text())
            if meta['capture']!=capture or meta['nf']!=nf or meta['capture_first_frame']!=1:raise ValueError('native capture/date mapping mismatch')
            expected={r['frame']:r for r in json.loads((old208/'image_receipts.json').read_text())}
            sizes={(r['height'],r['width']) for r in expected.values()}
            if len(sizes)!=1:raise ValueError('ambiguous nominal native image dimensions')
            nominal=next(iter(sizes));polygon=meta['polygon']
            with gzip.open(source/'sources.json.gz','rt',encoding='utf8') as f:sources=json.load(f)
            with np.load(source/'native.npz',allow_pickle=False) as z:native={name:z[name] for name in ('receiver','receiver_canon','peer1','peer2')}
            records=[];members=[];canons={}
            for slot in (0,1,2):
                with gzip.open(source/f'K/m{slot}/events.json.gz','rt',encoding='utf8') as f:scope=json.load(f)
                if scope['native_slot']!=slot or scope['seed']!=(3 if domain in ('35','33') else 0)+slot:raise ValueError('original18 member roster mismatch')
                context={k:scope[k] for k in ('domain','clip','native_slot','member','member_index','seed','tag')}
                context['member_role']='receiver' if slot==0 else 'reserve'
                rows=native['receiver' if slot==0 else 'peer'+str(slot)]
                keys=sources['receiver'] if slot==0 else sources['peers'][slot-1]
                roster=endpoints(rows,keys,nf,context)
                if slot==0 and not np.array_equal(roster['canon'],native['receiver_canon']):raise ValueError('original212 receiver canonical mapping differs')
                canons[f'canon{slot}']=roster.pop('canon')
                records.extend(roster.pop('records'));members.append(dict(context,**roster))
            image_receipts={}
            def reader(frame):
                rel=Path('data/BEE24/train')/capture/'img1'/f'{frame:06d}.jpg';content=(args.repo/rel).read_bytes()
                digest=hashlib.sha256(content).hexdigest()
                if frame in expected and digest!=expected[frame]['sha256']:raise ValueError('original image changed since208')
                image=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
                if image is None:raise ValueError('missing/corrupt source image '+rel.as_posix())
                if image.shape[:2]!=nominal:raise ValueError('nominal/native image coordinate mismatch')
                image_receipts[frame]=dict(frame=frame,path=rel.as_posix(),sha256=digest,width=image.shape[1],height=image.shape[0])
                if len(image_receipts)%500==0:print('CENTER214_FRAMES',domain,clip,len(image_receipts),flush=True)
                return image
            G=[];chi=[];feature_indices=[]
            def extract(image,e):
                g=F.endpoint_geometry(e['box'],e['score'],e['role'],e['nobs'],neighbor_center=e['nearest_other_observation_center'],frame_gap=e['frame_gap'])
                patch=F.sample_patch_rgb(image,e['box']);x=F.chi_from_patch(patch)
                row=len(G);G.append(g);chi.append(x);feature_indices.append(e['endpoint_id'])
                return dict(feature_row=row,patch_rgb_sha256=hashlib.sha256(patch.tobytes()).hexdigest())
            produced,_=process_frames(records,polygon,nf,reader,extract)
            for e in produced:e.update(run_id=e['canonical_id'],box_xywh=e['box'],eligible=e['gate']['eligible'])
            destination=temp/domain/clip;destination.mkdir(parents=True)
            np.savez(destination/'native.npz',**native,**canons);save_gz(destination/'sources.json.gz',sources)
            np.savez(destination/'data.npz',G=np.asarray(G,float).reshape(-1,6),chi=np.asarray(chi,float).reshape(-1,24))
            save(destination/'feature_rows.json',feature_indices)
            counts={status:sum(e['gate']['status']==status for e in produced) for status in ('eligible','ineligible','unknown')}
            save_gz(destination/'roster.json.gz',dict(schema='center214-clip-v1',domain=domain,clip=clip,capture=capture,nf=nf,polygon=polygon,
                nominal_image_shape=list(nominal),capture_first_frame=1,members=members,records=produced,counts=counts,
                G_names=list(F.G_NAMES),chi_names=list(F.CHI_NAMES),source212_path=path,source212_native_sha256=m212[path+'/native.npz']))
            save(destination/'image_receipts.json',[image_receipts[f] for f in sorted(image_receipts)])
            outputs.append(dict(domain=domain,clip=clip,path=domain+'/'+clip,members=3,endpoints=len(produced),**counts))
            print('CENTER214_PREPARED',domain,clip,flush=True)
        identity=dict(prereg=args.prereg,supplement=args.supplement,marker=MARKER,source212=SOURCE212,source208_nominal=SOURCE208,
                      code_sha256={p.name:sha(p) for p in (Path(__file__),Path(F.__file__))})
        save(temp/'deployment.json',dict(schema='center214-prepare-v1',identity=identity,clips=outputs,originalK=18))
        save(temp/'receipt.json',dict(identity=identity,GT_access=False,receiver86state_access=False,original18_K_only=True,
                                    all_endpoint_denominators_retained=True,one_decode_per_capture_date=True))
        files={p.relative_to(temp).as_posix():sha(p) for p in temp.rglob('*') if p.is_file()}
        save(temp/'manifest.json',files);(temp/'SEALED.sha256').write_text(sha(temp/'manifest.json')+'\n');temp.rename(args.out)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print('CENTER214_FROZEN18',sha(args.out/'manifest.json'))


def self_test():
    R=np.array([[1,7,48,45,4,4,.9],[2,7,49,45,4,4,.8],[30,7,48,45,4,4,.9],
                [1,8,-1,45,4,4,.9]],float)
    context=dict(domain='synthetic',clip='synthetic',native_slot=0,member='synthetic',seed=0)
    roster=endpoints(R,[('synthetic',i) for i in range(len(R))],30,context)
    assert roster['canonical_runs']==3 and roster['endpoints']==4 and roster['singletons']==2
    assert {e['raw_index'] for e in roster['records']}==set(range(4))
    polygon=[(50,0),(90,0),(90,90),(50,90)];seen=[]
    output,receipts=process_frames(roster['records'],polygon,30,lambda f:(seen.append(f) or np.zeros((100,100,3),np.uint8)),lambda image,e:[e['raw_index']])
    assert len(seen)==len(set(seen))==3 and len(output)==4
    near=geometry_gate(dict(box=[48,45,4,4]),polygon,(100,100));assert near['eligible']
    border=geometry_gate(dict(box=[2,45,4,4]),polygon,(100,100));assert border['crop_inside']  # crop left=0 now INCLUDED
    outside=geometry_gate(dict(box=[1.99,45,4,4]),polygon,(100,100));assert not outside['crop_inside']
    unknown=geometry_gate(dict(box=[48,45,0,4]),polygon,(100,100));assert unknown['status']=='unknown'
    return dict(passed=True,scope='synthetic only',tests=['raw-row/canonical mapping','>25 gap','singleton once',
        'closed2w2h pixel border','inclusive quarterdiag band','unknown geometry','one decode per date'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--self-test',action='store_true')
    p.add_argument('--source212',type=Path);p.add_argument('--inputs208',type=Path);p.add_argument('--out',type=Path);p.add_argument('--prereg');p.add_argument('--supplement')
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    if args.self_test:print(json.dumps(self_test()))
    else:
        if any(getattr(args,k) is None for k in ('source212','inputs208','out')):p.error('explicit original18 sources/output required')
        run(args)
