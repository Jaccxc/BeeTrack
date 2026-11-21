"""Fixed source-only pair texture descriptors; synthetic unless preregistered.

Roster is sealed before JPEG access. No model fitting, GT read, candidate
reselection, crop resizing, image export, or neural/GPU feature extraction.
"""
import os
import sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import cv2
import numpy as np

MARKER='[pair-texture-v1]'
SELECTION_SHA='027e127674826303042df7074b3b855bb282b7caaf2e7205da3030cb6a4943a4'
SOURCES={'W':tuple(f'BEE24-{k:02d}' for k in (*range(1,10),11,14)),
         'G':tuple(f'BEE24-{k:02d}' for k in range(17,33) if k not in (24,28))}
FEATURE_NAMES=['mean_R','mean_G','mean_B','std_R','std_G','std_B',
               'r1_mean_HV','r1_abs_HV','r2_mean_HV','r2_abs_HV','r4_mean_HV','r4_abs_HV']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')


def describe(grid):
    x=np.asarray(grid,dtype=np.float64)
    if x.shape!=(16,16,3) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
        raise ValueError('16x16 RGB grid in[0,1] required')
    flat=x.reshape(256,3)
    color=np.r_[flat.mean(0),flat.std(0)]
    y=(x[:,:,0]+2*x[:,:,1]+x[:,:,2])/4
    sd=y.std(); z=np.zeros_like(y) if sd<=1e-12 else (y-y.mean())/sd
    texture=[]
    for r in (1,2,4):
        h=float(np.mean(z[:,:-r]*z[:,r:]));v=float(np.mean(z[:-r,:]*z[r:,:]))
        texture.extend(((h+v)/2,abs(h-v)))
    return np.r_[color,texture]


def crop_grid(image,row):
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('native uint8 RGB image required')
    a=np.asarray(row,dtype=float)
    if a.shape!=(6,) or not np.isfinite(a).all() or np.any(a[4:6]<=0):
        raise ValueError('finite positive Nx6 box row required')
    x,y,w,h=a[2:6]
    x0,x1=np.ceil(np.array([x+.25*w,x+.75*w])-.5).astype(int)
    y0,y1=np.ceil(np.array([y+.25*h,y+.75*h])-.5).astype(int)
    bounds=[int(x0),int(y0),int(x1),int(y1)]
    if x0<0 or y0<0 or x1>image.shape[1] or y1>image.shape[0] or x1<=x0 or y1<=y0:
        return None,bounds
    crop=image[y0:y1,x0:x1]
    ys=np.floor((np.arange(16)+.5)*crop.shape[0]/16).astype(int)
    xs=np.floor((np.arange(16)+.5)*crop.shape[1]/16).astype(int)
    return crop[ys[:,None],xs[None,:],:].astype(np.float64)/255,bounds


def shuffle_grid(grid,clip,actual_frame,bounds,rep):
    if rep not in range(4) or len(bounds)!=4: raise ValueError('invalid shuffle site')
    prefix='texture-null-v1|'+str(clip)+'|'+str(int(actual_frame))+'|'+'|'.join(str(int(x)) for x in bounds)+'|'+str(rep)+'|'
    order=sorted(range(256),key=lambda i:(hashlib.sha256((prefix+str(i)).encode('utf-8')).digest(),i))
    return np.asarray(grid).reshape(256,3)[order].reshape(16,16,3)


def order(pair):
    return tuple(pair[k] for k in ('digest','id1','run1','id2','run2','frame'))


def make_roster(selection,prereg):
    sources=[];pairs=[]
    if selection['selected_count']!=93: raise ValueError('old cohort must remain93')
    found=set()
    for source in selection['sources']:
        h=source['hive'];clip=source['clip']
        if h not in SOURCES or clip not in SOURCES[h] or clip in found: raise ValueError('source whitelist mismatch')
        found.add(clip)
        sources.append({k:source[k] for k in ('clip','hive','nframes','polygon','gt_path','gt_sha256','clip_failure')})
        sources[-1]['eligible_count']=source.get('eligible_count',0)
        if source['clip_failure']:
            if source['selected']: raise ValueError('failed source cannot have selections')
            continue
        eligible=sorted((p for p in source['pairs'] if p['state']=='eligible'),key=order)
        if len(eligible)!=source['eligible_count']: raise ValueError('eligible count mismatch')
        if eligible[:4]!=source['selected']: raise ValueError('old first4 roster changed')
        old={p['digest'] for p in source['selected']}
        if len({p['digest'] for p in eligible})!=len(eligible): raise ValueError('duplicate pair digest')
        for cohort,chosen in [('old',source['selected']),('new',[p for p in eligible if p['digest'] not in old][:12])]:
            for pick in chosen:
                text=f"paired-rgb-v1|{clip}|{pick['id1']}|{pick['run1']}|{pick['id2']}|{pick['run2']}|{pick['frame']}"
                if pick['hash_input']!=text or hashlib.sha256(text.encode()).hexdigest()!=pick['digest']:
                    raise ValueError('historical candidate digest mismatch')
                if len(pick['endpoint_rows'])!=4: raise ValueError('four crop boxes required')
                pairs.append(dict(pick,pair_index=len(pairs),cohort=cohort,subset=cohort,pair_key=clip+'|'+pick['digest']))
    if found!=set(c for clips in SOURCES.values() for c in clips): raise ValueError('25-source roster incomplete')
    if sum(p['cohort']=='old' for p in pairs)!=93: raise ValueError('old93 roster changed')
    return dict(schema='pair-texture-roster-v1',prereg=prereg,marker=MARKER,selection_sha256=SELECTION_SHA,
                sources=sources,pairs=pairs,code_sha256=sha(__file__),
                rule='perclip old historicalselectedfirst4;new eligibleminusold SHAorder first12; failedclips neverrefilled')


def measure(pair,read_rgb):
    rows=pair['endpoint_rows'];before,after=pair['endpoint_frames']
    if [int(r[0]) for r in rows]!=[before,before,after,after]: raise ValueError('four-box clock/order changed')
    def arm(shift):
        dates=[before+shift,before+shift,after+shift,after+shift] if shift is not None else [None]*4
        output=dict(valid=True,crops=[],bounds=[],actual_frames=dates,unavailable=[])
        grids=[]
        for row,date in zip(rows,dates):
            image=read_rgb(date) if date is not None else None
            if image is None:
                grid,bounds,reason=None,None,'image_unavailable' if date is not None else 'control_date_unavailable'
            else:
                grid,bounds=crop_grid(image,row);reason=None if grid is not None else 'central_crop_outside_or_empty'
            output['crops'].append(None if grid is None else describe(grid).tolist())
            output['bounds'].append(bounds);output['unavailable'].append(reason)
            output['valid']=output['valid'] and grid is not None
            grids.append(grid)
        return output,grids
    primary,grids=arm(0);shifted,_=arm(pair['frame_shift'])
    shuffles=[]
    for rep in range(4):
        data=dict(valid=primary['valid'],rep=rep,crops=[],bounds=primary['bounds'],actual_frames=primary['actual_frames'],unavailable=primary['unavailable'])
        for i,grid in enumerate(grids):
            if grid is None: data['crops'].append(None);continue
            shuffled=shuffle_grid(grid,pair['clip'],primary['actual_frames'][i],primary['bounds'][i],rep)
            f=describe(shuffled)
            np.testing.assert_allclose(f[:6],primary['crops'][i][:6],rtol=0,atol=1e-12)
            data['crops'].append(f.tolist())
        shuffles.append(data)
    return dict(pair,primary=primary,shifted=shifted,shuffles=shuffles)


def run(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg): raise ValueError('full prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,capture_output=True,check=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,capture_output=True,check=True).stdout.decode('utf-8')
    if log.count(MARKER)!=1: raise ValueError('unique prereg marker required')
    if args.out.exists(): raise ValueError('fresh output required')
    if sha(args.selection)!=SELECTION_SHA: raise ValueError('historical selection SHA mismatch')
    if (args.selection.parent/'SELECTION.sha256').read_text().strip()!=SELECTION_SHA:
        raise ValueError('historical selection sidecar mismatch')
    selection=json.loads(args.selection.read_text(encoding='utf-8'))
    roster=make_roster(selection,args.prereg)
    args.out.mkdir(parents=True)
    save(args.out/'roster.json',roster)
    (args.out/'ROSTER.sha256').write_text(sha(args.out/'roster.json')+'\n')
    print('ROSTER_SEALED',sha(args.out/'roster.json'),flush=True)
    # Only after roster seal may a source image be opened. Cache two dates per
    # pair, never a whole film; shifted dates likewise keep original boxes.
    cv2.setNumThreads(1);images={};records=[]
    for pair in roster['pairs']:
        allowed=set(pair['endpoint_frames'])|set(pair['control_frames'] or [])
        cache={}
        def read_rgb(date):
            if date not in allowed: raise ValueError('unregistered image date')
            if date in cache:return cache[date]
            rel=Path('data/BEE24/train')/pair['clip']/'img1'/f'{date:06d}.jpg'
            path=args.repo/rel
            if not path.is_file():
                images[rel.as_posix()]=None;cache[date]=None;return None
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if rel.as_posix() in images and images[rel.as_posix()]!=digest: raise ValueError('source JPEG changed')
            images[rel.as_posix()]=digest
            try:image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            except cv2.error:image=None
            cache[date]=None if image is None else image[:,:,::-1].copy()
            return cache[date]
        records.append(measure(pair,read_rgb))
    if sha(args.selection)!=SELECTION_SHA or sha(__file__)!=roster['code_sha256']: raise ValueError('input/code changed')
    save(args.out/'features.json',dict(schema='pair-texture-features-v1',prereg=args.prereg,marker=MARKER,
         roster_sha256=sha(args.out/'roster.json'),selection_sha256=SELECTION_SHA,code_sha256=roster['code_sha256'],
         feature_names=FEATURE_NAMES,image_sha256=images,pairs=records))
    manifest={p.name:sha(p) for p in sorted(args.out.iterdir()) if p.is_file()}
    save(args.out/'manifest.json',manifest)
    (args.out/'SEALED.sha256').write_text(sha(args.out/'manifest.json')+'\n')
    print('FEATURES_SEALED',sha(args.out/'manifest.json'),flush=True)


def self_test():
    flat=np.ones((16,16,3))*.4
    np.testing.assert_array_equal(describe(flat)[6:],np.zeros(6))
    grid=np.zeros((16,16,3));grid[:8]=1
    assert describe(grid).shape==(12,) and describe(grid)[6]>0
    shuffled=shuffle_grid(grid,'synthetic',10,[1,2,17,18],0)
    np.testing.assert_allclose(describe(grid)[:6],describe(shuffled)[:6],rtol=0,atol=1e-12)
    assert not np.array_equal(shuffled,grid)
    im=np.arange(30*40*3,dtype=np.uint8).reshape(30,40,3)
    sampled,bounds=crop_grid(im,[10,0,2,4,8,12])
    assert sampled.shape==(16,16,3) and bounds==[4,7,8,13]
    assert crop_grid(im,[10,0,-20,0,4,4])[0] is None
    p=dict(clip='synthetic',endpoint_rows=[[10,0,2,4,8,12],[10,1,12,4,8,12],[18,0,3,5,8,12],[18,1,13,5,8,12]],endpoint_frames=[10,18],frame_shift=25)
    calls=[]
    def read(f):calls.append(f);return im
    measured=measure(p,read)
    assert measured['primary']['actual_frames']==[10,10,18,18]
    assert measured['shifted']['actual_frames']==[35,35,43,43]
    assert all(len(s['crops'])==4 for s in measured['shuffles'])
    print('Pairtexture synthetic centralcrop/grid/color/texture/RGBshuffle/fixedtimecontrol passed')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path);ap.add_argument('--selection',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--prereg')
    ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:self_test()
    else:
        if any(getattr(args,k) is None for k in ('repo','selection','out','prereg')):ap.error('--repo --selection --out --prereg required')
        args.repo=args.repo.resolve();args.selection=args.selection.resolve();args.out=args.out.resolve()
        run(args)
