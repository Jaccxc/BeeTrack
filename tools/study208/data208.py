"""GT-free fixed-six208 mmap inputs. SOURCE/SYNTHETIC until root CPU GO.

No GT/model/score loader. Crop callback is injected; field reader uses only
the exact selected old train frames, never common.seq_dir's test fallback.
"""
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
import cv2
import numpy as np
cv2.setNumThreads(1)

MARKER='[natural-switch-hazard-v1]'
SOURCE206='066aaa245d78ee5d926cafeb709d200746e9ea9f736f820606270c35a2188d5d'
SPECS=(('35','BEE24-35-head',3950,3),('33','BEE24-33-head',3950,3),
       ('W','BEE24-10',250,0),('W','BEE24-15',240,0),('G','BEE24-24',75,0),('G','BEE24-28',75,0))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def save(path,x):Path(path).write_text(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')


def verify(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('seal mismatch')
    m=json.loads((root/'manifest.json').read_text())
    for name,digest in m.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError('artifact drift '+name)
    return m


def enumerate_windows(rows,sources,nf):
    from tools.study197.physical import canonicalize
    R=np.asarray(rows,float).reshape(-1,7);keys=[tuple(s) for s in sources]
    if len(keys)!=len(R) or len(set(keys))!=len(keys):raise ValueError('source index cardinality/collision')
    if (R[:,0]>nf).any():raise ValueError('clock outside clip')
    b=canonicalize(R);canon=np.empty(len(R),np.int64)
    canon[b['source_raw_index']]=b['canonical_rows'][:,1].astype(np.int64)
    allwindows=[]
    for block in b['parent_blocks']:
        native=b['source_raw_index'][block['start']:block['end']]
        for i in range(max(0,len(native)-3)):allwindows.append(native[i:i+4])
    allwindows=np.asarray(allwindows,np.int64).reshape(-1,4)
    dt=np.diff(R[allwindows,0],axis=1)
    gap_ok=((dt>=1)&(dt<=3)).all(1)
    candidate_to_all=np.flatnonzero(gap_ok)
    return dict(rows=R,canon=canon,windows_all=allwindows,gap_ok=gap_ok,
                candidate_to_all=candidate_to_all,indices=allwindows[gap_ok],source_keys=keys)


def common_geometry(rows4):
    """NumPy float32 equivalent of switch_head.motion_features, no model call."""
    R=np.asarray(rows4,float);centers=R[:,:,2:4]+R[:,:,4:6]/2
    scale=np.median(np.hypot(R[:,:,4],R[:,:,5]),axis=1)
    xy=((centers-centers.mean(1,keepdims=True))/scale[:,None,None]).astype(np.float32)
    frames=R[:,:,0].astype(np.float32);dt=np.diff(frames,axis=1)
    steps=np.diff(xy,axis=1)/np.abs(dt)[:,:,None]
    speed=np.sqrt(np.sum(steps*steps,axis=-1))
    normalized=steps/np.maximum(speed[:,:,None],np.float32(1e-8))
    cosine=np.sum(normalized[:,:-1]*normalized[:,1:],axis=-1)
    geometry=np.stack((speed[:,1],(speed[:,0]+speed[:,2])/2,np.abs(speed[:,0]-speed[:,2]),
                       cosine.mean(1),np.sqrt(np.sum((xy[:,-1]-xy[:,0])**2,axis=1)),np.abs(dt[:,1])),axis=1)
    return geometry.astype(np.float32),scale,centers


def build_arrays(rows,sources,nf,read_rgb,dest,progress_label=None):
    """Decode one needed frame once; zero invalid windows, preserve denominator.

    N rows are all gap1..3 windows. valid[N] MUST mask downstream train/infer.
    reject_codes 0=valid,1=shared sampling grid exceeds native image boundary.
    Missing/unreadable frame raises, never becomes a convenient rejected sample.
    """
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    e=enumerate_windows(rows,sources,nf);R=e['rows'];indices=e['indices'];n=len(indices)
    for key in ('rows','canon','indices','windows_all','gap_ok','candidate_to_all'):
        np.save(dest/(key+'.npy'),e[key],allow_pickle=False)
    geometry,scale,centers=common_geometry(R[indices])
    if progress_label:print('PREPARED',progress_label,'Krows',len(R),'gapwindows',n,flush=True)
    np.save(dest/'geometry.npy',geometry,allow_pickle=False)
    crops=np.lib.format.open_memmap(dest/'crops.npy',mode='w+',dtype=np.uint8,shape=(n,4,32,32));crops[:]=0
    valid=np.ones(n,bool);reject=np.zeros(n,np.uint8)
    tasks=defaultdict(list)
    for j in range(n):
        for slot,rawidx in enumerate(indices[j]):tasks[int(R[rawidx,0])].append((j,slot))
    images=[];base=(np.arange(32)+.5)/32-.5;weights=np.array([.299,.587,.114],np.float32)
    for processed,(frame,items) in enumerate(sorted(tasks.items()),1):
        supplied=read_rgb(frame)
        if isinstance(supplied,tuple):image,receipt=supplied
        else:image=supplied;receipt=dict(frame=frame,synthetic=True)
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('unreadable/non-RGB native frame '+str(frame))
        images.append(dict(receipt,frame=frame,width=image.shape[1],height=image.shape[0]))
        native=image.astype(np.float32)/255
        for j,slot in items:
            if not valid[j]:continue
            center=centers[j,slot];offset=base*(1.5*scale[j])
            xx,yy=np.meshgrid(center[0]+offset-.5,center[1]+offset-.5)
            if xx.min()<0 or yy.min()<0 or xx.max()>image.shape[1]-1 or yy.max()>image.shape[0]-1:
                valid[j]=False;reject[j]=1;continue
            rgb=cv2.remap(native,xx.astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR)
            gray=rgb@weights
            crops[j,slot]=np.rint(255*gray).clip(0,255).astype(np.uint8)
        if progress_label and processed%500==0:
            print('FRAMES',progress_label,processed,'/',len(tasks),'native-frame',frame,flush=True)
    crops[~valid]=0;crops.flush();del crops
    np.save(dest/'valid.npy',valid,allow_pickle=False);np.save(dest/'reject_codes.npy',reject,allow_pickle=False)
    save(dest/'source_keys.json',[list(s) for s in e['source_keys']]);save(dest/'image_receipts.json',images)
    return dict(raw_rows=len(R),all_four_observation_windows=len(e['windows_all']),gap_eligible_windows=n,
                gap_rejected_windows=int((~e['gap_ok']).sum()),valid_windows=int(valid.sum()),
                boundary_rejected_windows=int((~valid).sum()),decoded_frames=len(images),
                cut_semantics='indices[n,1] | indices[n,2], four adjacent observations in one canonical run',
                crop_order=['before2','before1','after1','after2'],
                crop_spec=dict(size=32,padding=1.5,scale='median four box diagonals',angle=0,RGB_gray=[.299,.587,.114],
                               remap='cv2.INTER_LINEAR physical centres minus .5',quantization='round(255*x), nearest even'),
                reject_codes={'0':'valid','1':'shared crop crosses native boundary'},
                no_silent_boundary_drop=True,geometry='motion_features equivalent float32[N,6]')


def freeze_clip(rows,sources,metadata,identity,reader,dest):
    dest=Path(dest)
    if dest.exists():
        verify(dest)
        if json.loads((dest/'receipt.json').read_text())['identity']!=identity:raise ValueError('resume input/code drift')
        return
    dest.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        stats=build_arrays(rows,sources,metadata['nf'],reader,temp,metadata['domain']+'/'+metadata['clip'])
        save(temp/'metadata.json',dict(metadata,schema='switch208-input-clip-v1',**stats))
        save(temp/'receipt.json',dict(identity=identity,GT_access=False,model_access=False,**stats))
        save(temp/'manifest.json',{p.name:sha(p) for p in temp.iterdir()});(temp/'SEALED.sha256').write_text(sha(temp/'manifest.json')+'\n')
        temp.rename(dest)
    finally:
        if temp.exists():shutil.rmtree(temp)


def run(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('published full prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('unique208marker missing '+MARKER)
    if args.out.resolve().drive.upper()!='D:':raise ValueError('D: artifact root required')
    if sha(args.source206/'manifest.json')!=SOURCE206 or (args.source206/'SEALED.sha256').read_text().strip()!=SOURCE206:
        raise ValueError('source206seal mismatch')
    manifest=json.loads((args.source206/'manifest.json').read_text())
    if sha(args.source206/'deployment.json')!=manifest['deployment.json']:raise ValueError('source206metadata drift')
    top=json.loads((args.source206/'deployment.json').read_text());selected=[]
    for domain,clip,nf,seed in SPECS:
        groups=[g for g in top['groups'] if g['domain']==domain and g['clip']==clip and g['group']==(2 if domain in ('35','33') else 1)]
        if len(groups)!=1:raise ValueError('one fixed unit perclip required')
        g=groups[0];folder=args.source206/g['path']
        for name in ('queries.json','K.npz','receipt.json','manifest.json','SEALED.sha256'):
            rel=f'{g["path"]}/{name}'
            if sha(args.source206/rel)!=manifest[rel]:raise ValueError('206named source changed')
        q=json.loads((folder/'queries.json').read_text());mm=[m for m in q['members'] if m['seed']==seed]
        if len(mm)!=1 or q['nf']!=nf:raise ValueError('fixed seed/clock mismatch')
        m=mm[0];slot=m['member_index'];tag=m['tag']
        if slot!=0 or seed not in (0,3) or q['split'] not in ('dev','val'):raise ValueError('fixed member guard')
        with np.load(folder/'K.npz',allow_pickle=False) as z:rows=z[f'k{slot}']
        capture=clip[:-5] if clip.endswith('-head') else clip
        metadata=dict(domain=domain,clip=clip,unit=q['unit'],member=m['member'],member_index=slot,seed=seed,tag=tag,
                      nf=nf,polygon=q['polygon'],split=q['split'],region='head' if clip.endswith('-head') else 'whole',
                      capture=capture,capture_first_frame=1,source206_group=g,
                      source206_K_sha256=manifest[f'{g["path"]}/K.npz'],source_provenance=m['source203_receipt'])
        metadata.update(source_gt_relative=f'data/BEE24/train/{capture}/gt/gt.txt',
                        prepared_gt_relative=f'runs/gt/{q["split"]}/{clip}/gt/gt.txt')
        identity=dict(prereg=args.prereg,source206=SOURCE206,source206_group_seal=g['seal'],seed=seed,member=m['member'],
                      code_sha256=sha(__file__))
        def reader(frame,capture=capture):
            rel=Path('data/BEE24/train')/capture/'img1'/f'{frame:06d}.jpg'
            content=(args.repo/rel).read_bytes();image=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError('unreadable exact oldsource frame '+rel.as_posix())
            return cv2.cvtColor(image,cv2.COLOR_BGR2RGB),dict(path=rel.as_posix(),sha256=hashlib.sha256(content).hexdigest())
        dest=args.out/domain/clip;freeze_clip(rows,m['source_keys'],metadata,identity,reader,dest)
        stats=json.loads((dest/'metadata.json').read_text())
        selected.append(dict(domain=domain,clip=clip,member=m['member'],member_index=slot,seed=seed,
                             n_cuts=stats['gap_eligible_windows'],n_valid=stats['valid_windows'],
                             path=f'{domain}/{clip}',seal=(dest/'SEALED.sha256').read_text().strip()))
        print('SEALED',domain,clip,flush=True)
    deployment=dict(schema='switch208-inputs-v1',prereg=args.prereg,marker=MARKER,source206=SOURCE206,clips=selected)
    if (args.out/'deployment.json').exists() and json.loads((args.out/'deployment.json').read_text())!=deployment:
        raise ValueError('top resume drift')
    save(args.out/'deployment.json',deployment)
    files={p.relative_to(args.out).as_posix():sha(p) for p in args.out.rglob('*') if p.is_file() and p not in (args.out/'manifest.json',args.out/'SEALED.sha256')}
    if (args.out/'manifest.json').exists() and json.loads((args.out/'manifest.json').read_text())!=files:raise ValueError('top files drift')
    save(args.out/'manifest.json',files);(args.out/'SEALED.sha256').write_text(sha(args.out/'manifest.json')+'\n')
    print('INPUTS6',sha(args.out/'manifest.json'))


def self_test(repo):
    import importlib.util
    path=Path(__file__).with_name('switch_inputs.py')
    sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location('synthetic_switch_inputs',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=np.array([[f,1,30+f,30,8,6,.9] for f in range(1,7)]+[[f,2,0,0,8,6,.9] for f in range(1,5)],float)
    sources=[('synthetic',i) for i in range(len(rows))];seen=[]
    image=np.full((80,100,3),200,np.uint8);image[30:40,30:50]=10
    with tempfile.TemporaryDirectory() as tmp:
        stats=build_arrays(rows,sources,6,lambda f:(seen.append(f) or image.copy()),tmp)
        idx=np.load(Path(tmp)/'indices.npy');valid=np.load(Path(tmp)/'valid.npy');crops=np.load(Path(tmp)/'crops.npy',mmap_mode='r')
        assert len(idx)==4 and stats['valid_windows']==3 and stats['boundary_rejected_windows']==1 and len(seen)==len(set(seen))
        for j in np.flatnonzero(valid):
            pack=module.aligned_four(rows,sources,idx[j],lambda f:image.copy(),6)
            assert np.array_equal(crops[j],np.rint(255*pack['crops'][:,0]).clip(0,255).astype(np.uint8))
        import torch
        from switch_head import motion_features
        torch.set_num_threads(1)
        pack=module.aligned_four(rows,sources,idx[0],lambda f:image.copy(),6)
        expected=motion_features(torch.tensor(pack['centers'][None]),torch.tensor(pack['frames'][None])).numpy()[0]
        assert np.allclose(np.load(Path(tmp)/'geometry.npy')[0],expected,rtol=1e-6,atol=1e-7)
        assert np.load(Path(tmp)/'canon.npy').shape==(len(rows),)
        assert not np.asarray(crops[-1]).any()
        del crops
        gaprows=rows[:4].copy();gaprows[:,0]=[1,2,6,7]
        gap=build_arrays(gaprows,sources[:4],7,lambda f:(_ for _ in ()).throw(AssertionError('no eligible frame should decode')),Path(tmp)/'gap')
        assert gap['all_four_observation_windows']==1 and gap['gap_rejected_windows']==1 and gap['gap_eligible_windows']==0
        gaprows[:,0]=[1,4,7,10]
        assert len(enumerate_windows(gaprows,sources[:4],10)['indices'])==1
    return 'PASS synthetic: mmap/zero-invalid denominator; shared-switch crop uint8 exact; numpy geometry equivalent; original indices/canon; one decode per needed frame. NO FIELD.'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--self-test',action='store_true')
    p.add_argument('--source206',type=Path);p.add_argument('--out',type=Path);p.add_argument('--prereg')
    a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
    if a.self_test:print(self_test(a.repo))
    else:
        if a.source206 is None or a.out is None:p.error('--source206 --out --prereg required')
        run(a)
