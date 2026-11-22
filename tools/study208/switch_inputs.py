"""Private source/synthetic bridge; no field loader, fitting or pretrained model.

Unregistered crop choice: all four use one detector-only physical side length,
axis-aligned orientation, fixed output grid, individually centered translation.
No per-box anisotropic resize or GT-driven rotation/scale is performed.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np


def aligned_four(rows,sources,indices,read_rgb,nframes,size=32,padding=1.5):
    """Chronological [before2,before1,after1,after2], one native K run.

    read_rgb(frame) is injected by a future authorized producer. Bounds fail
    closed. Returns grayscale[4,1,H,W], body-scale centers,frames and provenance.
    """
    import cv2
    from queries import prepare
    data=prepare(rows,sources,nframes);ix=np.asarray(indices,int)
    if ix.shape!=(4,) or len(set(ix))!=4 or (ix<0).any() or (ix>=len(data['raw'])).any():raise ValueError('four distinct raw K indices required')
    chosen=data['raw'][ix];frames=chosen[:,0]
    if (np.diff(frames)<=0).any():raise ValueError('native chronological order required')
    if len({data['raw_to_canonical'][int(j)] for j in ix})!=1:raise ValueError('cut crosses original >25-gap run')
    run=next(r for r in data['runs'] if int(ix[0]) in r['raw_indices'])
    positions=[list(run['raw_indices']).index(int(j)) for j in ix]
    if positions!=list(range(positions[0],positions[0]+4)):raise ValueError('four rows must be adjacent observations around one cut')
    if size<4 or padding<=0:raise ValueError('positive crop geometry required')
    scale=float(np.median(np.hypot(chosen[:,4],chosen[:,5])))
    side=padding*scale;centers=chosen[:,2:4]+chosen[:,4:6]/2
    offset=((np.arange(size)+.5)/size-.5)*side
    crops=[];grids=[]
    for row,center in zip(chosen,centers):
        image=read_rgb(int(row[0]))
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('native uint8 RGB required')
        # Coordinates are pixel-centre positions; OpenCV integer coordinates
        # index pixel centres, so physical MOT centres are shifted by halfpixel.
        xx,yy=np.meshgrid(center[0]+offset-.5,center[1]+offset-.5)
        if xx.min()<0 or yy.min()<0 or xx.max()>image.shape[1]-1 or yy.max()>image.shape[0]-1:
            raise ValueError('shared crop exceeds image; no silent padding/clipping')
        rgb=cv2.remap(image.astype(np.float32)/255,xx.astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR)
        crops.append((rgb@np.array([.299,.587,.114],np.float32))[None])
        grids.append(dict(frame=int(row[0]),center=center.tolist(),physical_side=side,angle=0))
    return dict(crops=np.stack(crops).astype(np.float32),centers=((centers-centers.mean(0))/scale).astype(np.float32),
                frames=frames.astype(np.float32),raw_indices=ix.tolist(),source_keys=[list(data['sources'][j]) for j in ix],
                transform=dict(body_scale=scale,common_side=side,angle=0,size=size,grid=grids),
                field_status='unregistered; only synthetic tested')


def source_interval(labels,reliable,left,right,source_hive,heldout):
    """Input source-only (original GT ID,official GT run) evidence, never model features.

    Different reliable original IDs establish one positive bracket, not an exact
    first gap. Same endpoints alone do NOT create survival negatives.
    """
    if source_hive==heldout:raise ValueError('heldout labels forbidden')
    n=len(labels);reliable=np.asarray(reliable,bool)
    if reliable.shape!=(n,) or not 0<=left<right<n:raise ValueError('bracket shape/range')
    survival=np.zeros(n-1,bool);event=np.zeros(n-1,bool);kind='unknown'
    if reliable[left] and reliable[right]:
        if labels[left][0]!=labels[right][0]:event[left:right]=True;kind='positive_interval'
        elif all(reliable[left:right+1]) and all(tuple(x)==tuple(labels[left]) for x in labels[left:right+1]):
            survival[left:right]=True;kind='certified_survival'
    return dict(survival_mask=survival,event_mask=event,label_kind=kind,source_hive=source_hive)


def self_test():
    import torch
    from switch_head import SwitchHead,motion_features
    from switch_interval_loss import switch_interval_loss
    torch.set_num_threads(1)
    rows=np.array([[f,1,35+f,30,8,6,.9] for f in range(1,5)],float)
    image=np.full((80,100,3),230,np.uint8);image[30:36,36:47]=20
    pack=aligned_four(rows,[('synthetic',j) for j in range(4)],range(4),lambda f:image.copy(),4)
    assert pack['crops'].shape==(4,1,32,32)
    assert len({g['physical_side'] for g in pack['transform']['grid']})==1
    crops=torch.tensor(pack['crops'][None]);centers=torch.tensor(pack['centers'][None]);frames=torch.tensor(pack['frames'][None])
    geometry=motion_features(centers,frames)
    with torch.no_grad():assert SwitchHead(channels=4)(crops,geometry)['logit'].shape==(1,)
    bracket=source_interval([(1,0),(1,0),(2,0),(2,0)],[True,False,False,True],0,3,'W','35')
    assert bracket['event_mask'].all() and not bracket['survival_mask'].any()
    loss=switch_interval_loss(torch.zeros(3),event_mask=torch.tensor(bracket['event_mask']))
    assert torch.isfinite(loss)
    unknown=source_interval([(1,0)]*4,[True,False,False,True],0,3,'W','35')
    assert not unknown['survival_mask'].any()
    try:source_interval([(1,0)]*4,[True]*4,0,3,'35','35')
    except ValueError:pass
    else:raise AssertionError('heldout supervision accepted')
    return 'PASS synthetic: common-scale/no-rotation four crops -> real SwitchHead CPU API; source interval loss; unknown/heldout guards. No training/field.'


if __name__=='__main__':
    import argparse,sys
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--self-test',action='store_true',required=True)
    a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()));print(self_test())
