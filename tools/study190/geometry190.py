"""Post-score prediction-geometry sidecar; never edits registered inputs/results.

The one-pixel margin is conservative: reject only a disk separated from every
polygon edge by MORE than radius+1 and on the original endpoint's side.
Boundary/tangent/within-one-pixel cases stay in the broader all-side envelope.
"""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import bounds190 as B


def boundary_distance(center,polygon):
    p=np.asarray(center,float);v=np.asarray(polygon,float)
    if p.shape!=(2,) or v.ndim!=2 or v.shape[1]!=2 or len(v)<3 or not np.isfinite(p).all() or not np.isfinite(v).all():
        raise ValueError('finite2D center/simple polygon required')
    a=v;b=np.roll(v,-1,axis=0);d=b-a;length=(d*d).sum(axis=1)
    t=np.divide(((p-a)*d).sum(axis=1),length,out=np.zeros_like(length),where=length>0)
    nearest=a+np.clip(t,0,1)[:,None]*d
    return float(np.linalg.norm(p-nearest,axis=1).min())


def side(center,polygon):
    return int(cv2.pointPolygonTest(np.asarray(polygon,np.float32),tuple(map(float,center)),False)>=0)


def prune(canonical,candidates,polygon):
    rows=np.asarray(canonical,float);kept=[];reasons=[]
    for c in candidates:
        track=rows[rows[:,1]==c['B']];track=track[np.argsort(track[:,0])]
        endpoint=track[0 if c['kind']=='start' else -1]
        center=np.asarray(c['center'],float);radius=float(c['radius'])
        if not np.isfinite(radius) or radius<0:
            raise ValueError('radius must be finite and nonnegative')
        endpoint_center=endpoint[2:4]+endpoint[4:6]/2
        distance=boundary_distance(center,polygon)
        same=side(center,polygon)==side(endpoint_center,polygon)
        removed=bool(distance>radius+1 and same)
        if not removed:
            kept.append(c)
        reasons.append(dict(candidate_id=c['candidate_id'],B=c['B'],kind=c['kind'],
                            center=center.tolist(),radius=radius,endpoint_frame=int(endpoint[0]),
                            endpoint_center=endpoint_center.tolist(),endpoint_side=side(endpoint_center,polygon),
                            center_side=side(center,polygon),boundary_distance=distance,
                            disk_touches_boundary=distance<=radius,within_guard=distance<=radius+1,
                            removed=removed,reason='same-side disk strictly beyond radius+1' if removed else
                            'disk boundary/guard overlap or endpoint-side disagreement'))
    return kept,reasons


def _normalize(menu):
    return {None if e is None else tuple(map(int,e)) for e in menu}


def selfcheck():
    square=[[0,0],[100,0],[100,100],[0,100]]
    concave=[[0,0],[100,0],[100,100],[60,100],[60,40],[40,40],[40,100],[0,100]]
    assert boundary_distance([50,50],concave)==10
    assert side([50,50],concave)==0  # bounding rectangle is insufficient
    assert boundary_distance([50,50],square)==50
    row=lambda f,x:[f,1,x-1,49,2,2,.9]
    base=np.array([row(13,50),row(14,50)])
    candidates=[dict(candidate_id=1,B=1,kind='start',center=[50,50],radius=20,permitted_frames=[12]),
                dict(candidate_id=2,B=1,kind='end',center=[50,50],radius=49,permitted_frames=[15])]
    kept,reasons=prune(base,candidates,square)
    assert [c['candidate_id'] for c in kept]==[2]  # exactr+1 equality retained
    for radius,keep in ((50,True),(49,True),(48.999,False)):
        c=dict(candidates[0],radius=radius)
        assert bool(prune(base,[c],square)[0])==keep
    # Convex andconcave disks proven confined have no alternative side anywhere
    # in this synthetic dense polar sampling; both-end compositions remain valid.
    for polygon,center,radius in ((square,[50,50],20),(concave,[20,20],5),(concave,[50,70],5)):
        diskbase=np.array([[13,1,center[0]-1,center[1]-1,2,2,.9],
                           [14,1,center[0]-1,center[1]-1,2,2,.9]])
        for angle in np.linspace(0,2*np.pi,65):
            for rr in (0,radius/2,radius):
                assert side(np.asarray(center)+rr*np.array([np.cos(angle),np.sin(angle)]),polygon)==side(center,polygon)
        cc=[dict(candidates[0],center=center,radius=radius),dict(candidates[1],center=center,radius=radius)]
        kk,_=prune(diskbase,cc,polygon)
        assert kk==[]
        assert B.segment_menus(diskbase,kk,polygon,30)[1]==[None]
    # Removingone endpoint never blocks edits to the other; preserve joint menu.
    menu=B.segment_menus(base,kept,square,30)
    assert set(menu[1])=={None,(15,-1)}
    print('Geometry syntheticconvex/concave/tangent/1px/both-end checks passed')


def replay(path):
    """Replay compact sidecar without canonical rows, GT files, or rescoring."""
    report=json.loads(Path(path).read_text(encoding='utf-8'));ranges={}
    for name,cell in report['cells'].items():
        nf=cell['nf'];base=np.asarray(cell['baseline_phase_counts'],np.int64)
        for family in ('universal','geometry','labelled'):
            lower=base.copy();upper=base.copy()
            for change in cell['menus']:
                original=[e for e in change['original'] if e is not None]
                counts=B.phase_counts(original,nf)
                lo,hi=B.menu_extrema({change['B']:change[family]},nf)
                lower+=lo-counts;upper+=hi-counts
            bound=B.interval_bound(lower,upper,cell['gt_events'],nf)
            expected=cell['geometry_bound']['W_lower_percent'] if family=='geometry' else cell['allside_lower' if family=='universal' else 'labelled_lower']
            assert abs(bound['W_lower_percent']-expected)<1e-10
            if family=='geometry':
                ranges[cell['hive'],cell['seed']]=(lower,upper,cell['gt_events'])
    for h in (33,35):
        for group,seeds in (('345',(3,4,5)),('678',(6,7,8))):
            lo=sum(ranges[h,s][0] for s in seeds);hi=sum(ranges[h,s][1] for s in seeds)
            bound=B.interval_bound(lo,hi,ranges[h,seeds[0]][2],3950,streams=3)
            assert bound==report['heads'][str(h)]['groups'][group]['bound']
    print('Compact geometry sidecar replay passed')


def run(root,out,repo):
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    score=root/'score';hashes=load(score/'result_hashes.json')
    assert (score/'SEALED.sha256').read_text().strip()==sha(score/'result_hashes.json')
    for name,digest in hashes.items():
        assert sha(score/name)==digest
    manifest=load(root/'manifest.json')
    assert (root/'ALL_FROZEN.sha256').read_text().strip()==sha(root/'manifest.json')
    assert len(manifest['streams'])==12
    summary=load(score/'summary.json');zones=load(repo/'beetrack/zones.json')
    report=dict(schema='geometry190-sidecar-v1',diagnostic_only=True,margin_pixels=1,
                score_seal=sha(score/'result_hashes.json'),freeze_sha256=sha(root/'manifest.json'),
                source_sha256=sha(__file__),bounds_source_sha256=sha(Path(B.__file__)),cells={},heads={})
    ranges={}
    for h in (33,35):
        for seed in range(3,9):
            name=f'h{h}_s{seed}';folder=root/name;receipt=load(folder/'receipt.json')
            assert sha(folder/'receipt.json')==next(r['receipt_sha256'] for r in manifest['streams'] if (r['hive'],r['seed'])==(h,seed))
            for file in ('canonical.npy','candidates.json'):
                assert sha(folder/file)==receipt['files'][file]
            rows=np.load(folder/'canonical.npy',allow_pickle=False);candidates=load(folder/'candidates.json')['candidates']
            polygon=zones[receipt['clip']];nf=receipt['nframes']
            kept,reasons=prune(rows,candidates,polygon)
            menus=B.segment_menus(rows,kept,polygon,nf);lo,hi=B.menu_extrema(menus,nf)
            old=load(score/name/'bounds.json');scores=load(score/name/'scores.json')
            for identity,menu in menus.items():
                assert _normalize(old['gt_evidenced']['menus'][str(identity)])<=_normalize(menu)<=_normalize(old['universal_all_side']['menus'][str(identity)])
            gt=[(abs(int(date)),1 if date>0 else -1) for _,date in scores['base']['evidence']['gt']]
            bound=B.interval_bound(lo,hi,gt,nf)
            a=old['universal_all_side']['bound']['W_lower_percent'];z=old['gt_evidenced']['bound']['W_lower_percent']
            assert a-1e-10<=bound['W_lower_percent']<=z+1e-10
            original=B.segment_menus(rows,[],polygon,nf)
            differences=[]
            for identity in sorted({int(c['B']) for c in candidates}):
                differences.append(dict(B=identity,original=original[identity],
                    universal=old['universal_all_side']['menus'][str(identity)],geometry=menus[identity],
                    labelled=old['gt_evidenced']['menus'][str(identity)]))
            report['cells'][name]=dict(hive=h,seed=seed,nf=nf,polygon=polygon,reasons=reasons,
                endpoint_count=len(candidates),kept_count=len(kept),removed_count=len(candidates)-len(kept),
                disk_touch_count=sum(r['disk_touches_boundary'] for r in reasons),
                menus=differences,baseline_phase_counts=scores['base']['integer_bins']['pred'],gt_events=gt,
                allside_lower=a,geometry_bound=bound,labelled_lower=z,
                maximum_single_gain=scores['base']['w30_percent']-bound['W_lower_percent'])
            ranges[h,seed]=(lo,hi,gt)
    for h in (33,35):
        groups={}
        for group,seeds in (('345',(3,4,5)),('678',(6,7,8))):
            lo=sum(ranges[h,s][0] for s in seeds);hi=sum(ranges[h,s][1] for s in seeds)
            bound=B.interval_bound(lo,hi,ranges[h,seeds[0]][2],3950,streams=3)
            old=summary['bound_maximum_gains'][str(h)]
            assert old['universal_all_side']['E3'][group]['W_lower_percent']-1e-10<=bound['W_lower_percent']<=old['gt_evidenced']['E3'][group]['W_lower_percent']+1e-10
            base=summary['aggregation'][str(h)]['base']['E3'][group]['W_percent']
            groups[group]=dict(bound=bound,maximum_E3_gain=base-bound['W_lower_percent'])
        cells=[report['cells'][f'h{h}_s{s}'] for s in range(3,9)]
        report['heads'][str(h)]=dict(endpoints=sum(c['endpoint_count'] for c in cells),
             disk_touch_endpoints=sum(c['disk_touch_count'] for c in cells),
             retained_endpoints=sum(c['kept_count'] for c in cells),
             mean_single_ceiling=float(np.mean([c['maximum_single_gain'] for c in cells])),
             groups=groups,mean_E3_ceiling=float(np.mean([g['maximum_E3_gain'] for g in groups.values()])))
    out.mkdir(parents=True,exist_ok=False)
    path=out/'geometry.json';path.write_text(json.dumps(report,separators=(',',':'),sort_keys=True)+'\n',encoding='utf-8')
    (out/'SEALED.sha256').write_text(sha(path)+'\n')
    replay(path)
    print(json.dumps(report['heads'],indent=2));print(str(path))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--replay',type=Path);a=p.parse_args()
    selfcheck()
    if a.replay is not None:
        replay(a.replay)
    if a.root is not None:
        if a.out is None:
            p.error('--out required with--root')
        run(a.root,a.out,a.repo)
