"""Durable independent214 stages; field access requires explicit --field-go.

No producer214 module imports. Native event-edge semantics shared from public197;
coordinates, crop/odd features, ridge, Hungarian cache, DFS matching and clock
are independent. Each finished clip receipt is reusable after hash validation.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,hashlib,sys,shutil
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import audit214 as own
sys.path.insert(0,str(Path(__file__).parents[1]/'study212'))
import field212_features as upstream
p=upstream.prior;old=upstream.previous
PREREG='79dbed053bcddd4290300494ceff29c8130a5807';SUPP='21c6c803290fad2816ca472ec673b9f2eda76759'
PINS={'frozen':'ba6d026f3026f78d6809779d12b171588460abac5ce6a1065343566d6b0794fa','labels':'732ee647f22ecf77d6ca227b6a760523a24dc155cc5f7191dc4f39b5d441804f','predictions':'c71ddcc0188fcb87cb37ffec8a05990d78a1ffbb8c005d1eeab8caff9fac5569'}
ARMS=('K','G','I','Z','O')

def records(a):
    top=p.read(a.root/'frozen/deployment.json');assert top['identity']['prereg']==PREREG and top['identity']['supplement']==SUPP
    for ref in top['clips']:
        folder=a.root/'frozen'/ref['path'];c=p.read(folder/'roster.json.gz')
        with np.load(folder/'native.npz') as z:arrays={k:z[k] for k in z.files}
        yield ref,folder,c,arrays

def scene(R):return list(upstream.groups(R))
def slotrows(arrays,i):return arrays['receiver' if i==0 else 'peer'+str(i)]
def ends(c,i):return [e for e in c['records'] if e['native_slot']==i]
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def frozen_features(folder,c):
    with np.load(folder/'data.npz') as z:G=z['G'];chi=z['chi']
    out=[]
    for e in c['records']:
        f=e['features'];row=dict(e)
        if f is not None:row.update(G6=G[f['feature_row']].tolist(),chi24=chi[f['feature_row']].tolist())
        out.append(row)
    return out

def pixels(a):
    reports=[]
    for ref,folder,c,arrays in records(a):
        dest=a.out/'pixels'/ref['domain']/(ref['clip']+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert p.read(dest)['frozen']==PINS['frozen'];reports.append(p.read(dest));continue
        sources=p.read(folder/'sources.json.gz');original=a.source212/c['source212_path'];assert p.sha(original/'native.npz')==c['source212_native_sha256']
        with np.load(original/'native.npz') as z:
            for i in range(3):assert np.array_equal(slotrows(arrays,i),z['receiver' if i==0 else 'peer'+str(i)])
        expected=set();runs={};all_features=frozen_features(folder,c)
        for i in range(3):
            R=slotrows(arrays,i);roster=ends(c,i);mapping={e['raw_index']:e for e in roster}
            assert len(mapping)==len(roster)
            for pid,ix in scene(R):
                endpoint_indices=sorted({int(ix[0]),int(ix[-1])});expected.update((i,j) for j in endpoint_indices)
                for j in endpoint_indices:
                    e=mapping[j];assert e['canonical_id']==pid and e['nobs']==len(ix) and e['frame']==R[j,0] and e['box']==R[j,2:6].tolist()
                    role='single' if len(ix)==1 else 'start' if j==ix[0] else 'end';assert e['role']==role
                    runs[i,j]=(R[ix],int(np.flatnonzero(ix==j)[0]));key=sources['receiver'][j] if i==0 else sources['peers'][i-1][j];assert e['source_key']==key
        assert expected=={(e['native_slot'],e['raw_index']) for e in c['records']}
        images={int(r['frame']):r for r in p.read(folder/'image_receipts.json')};bydate=defaultdict(list)
        for e in all_features:bydate[e['frame']].append(e)
        assert set(images)==set(bydate);checked=0;eligible=0
        for frame,ee in sorted(bydate.items()):
            receipt=images[frame];path=a.repo/receipt['path'];assert p.sha(path)==receipt['sha256'];image=cv2.imread(str(path));assert image is not None
            h,w=image.shape[:2];assert [h,w]==c['nominal_image_shape'] and (h,w)==(receipt['height'],receipt['width'])
            for e in ee:
                x,y,bw,bh=e['box'];center=(x+bw/2,y+bh/2);sd=cv2.pointPolygonTest(np.asarray(c['polygon'],np.float32),center,True)
                good=bool(bw>0 and bh>0 and center[0]-bw>=0 and center[0]+bw<=w-1 and center[1]-bh>=0 and center[1]+bh<=h-1 and abs(sd)<=.25*np.hypot(bw,bh))
                assert good==e['gate']['eligible']==e['eligible'];checked+=1
                if good:
                    patch=own.patch(image,e['box']);assert digest(patch)==e['features']['patch_rgb_sha256'];p.close(own.odd24(patch),e['chi24'],2e-14)
                    run,j=runs[e['native_slot'],e['raw_index']];p.close(own.geometry(run,j),e['G6'],1e-14);eligible+=1
            del image
        report=dict(passed=True,frozen=PINS['frozen'],domain=c['domain'],clip=c['clip'],all_endpoints=checked,eligible_features=eligible,images=len(images),all18_source_rows_exact=True,scope='Every represented original endpoint frame hash/decode and every eligible RGB patch/G6/odd24 recomputed; full endpoint roster including ineligible retained.')
        p.save(dest,report);reports.append(report);print('AUDIT214_PIXELS',c['domain'],c['clip'],flush=True)
    p.save(a.out/'pixels.json',dict(passed=True,clips=reports))

class Geometry:
    """Full-frame Hungarian on changed boxes; cache includes ordered box bytes."""
    def __init__(self,GT,native,v,polygon):
        self.G=native.track_structure(GT,polygon);self.v=v;self.bydate=defaultdict(list);self.cache={};self.calls=0
        for j,r in enumerate(self.G['rows']):self.bydate[int(r[0])].append(j)
    def match(self,P):
        result={t:np.full(len(P['rows']),-1,np.int64) for t in ('0.5','0.3')};raw_gt=np.full(len(P['rows']),-1,np.int64);frames=defaultdict(list)
        for j,r in enumerate(P['rows']):frames[int(r[0])].append(j)
        for date,jj in frames.items():
            boxes=P['rows'][jj,2:6];key=(date,digest(boxes))
            if key not in self.cache:
                gg=self.bydate.get(date,[]);pairs=[]
                if gg:
                    M=self.v.official_iou_independent(self.G['rows'][gg,2:6],boxes);aa,bb=linear_sum_assignment(-M);pairs=[(int(b),gg[int(g)],float(M[g,b])) for g,b in zip(aa,bb)]
                self.cache[key]=pairs;self.calls+=1
            for b,g,iou in self.cache[key]:
                for t in result:
                    if iou>=float(t):result[t][P['index'][jj[b]]]=int(self.G['rows'][g,1])
                if iou>=.5:raw_gt[P['index'][jj[b]]]=g
        return result,raw_gt

def helpers(a):
    r=p.module('audit214_public203',a.repo/'tools/study203/verify203.py');_,native,v,_=r.helpers(a.repo);return native,v

def labels(a):
    native,v=helpers(a);top=p.read(a.root/'labels/deployment.json');refs={(r['domain'],r['clip'],r['member_index']):r for r in top['original_K']};reports=[]
    from beetrack.evaluate import load_tracks
    for ref,folder,c,arrays in records(a):
        output=a.out/'labels'/ref['domain']/(ref['clip']+'.json');output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():reports.append(p.read(output));continue
        GTpath=a.repo/p.read(a.root/'labels'/refs[c['domain'],c['clip'],0]['path'])['GT_relative_path'];assert p.sha(GTpath)==top['GT_sha256'][GTpath.relative_to(a.repo).as_posix()]
        geo=Geometry(load_tracks(GTpath),native,v,c['polygon']);members=[]
        for i in range(3):
            R=slotrows(arrays,i);P=native.track_structure(R,c['polygon']);owners,gtrows=geo.match(P);saved=p.read(a.root/'labels'/refs[c['domain'],c['clip'],i]['path'])['records'];assert len(saved)==len(ends(c,i))
            ownrows=[]
            for e,s in zip(ends(c,i),saved):
                j=e['raw_index'];g=gtrows[j];assert s['endpoint_id']==e['endpoint_id'] and s['has_owner']==bool(g>=0)==s['target_known']
                row=dict(endpoint_id=e['endpoint_id'],raw_index=j,canonical_run_id=e['canonical_id'],eligible=e['eligible'],known=bool(g>=0),GT_raw_index=None)
                if g>=0:
                    T=geo.G['rows'][g];assert s['GT_raw_index']==int(geo.G['index'][g]) and np.array_equal(T,s['GT_row']) and s['canonical_GT_owner']==int(T[1])
                    delta=((T[2:4]+T[4:6]/2)-(R[j,2:4]+R[j,4:6]/2))/R[j,4:6];p.close(delta,[s['dx_raw']/R[j,4],s['dy_raw']/R[j,5]],1e-14)
                    row.update(GT_raw_index=int(geo.G['index'][g]),target=delta.tolist(),GT_center=(T[2:4]+T[4:6]/2).tolist(),GT_row=T.tolist())
                ownrows.append(row)
            members.append(dict(member_index=i,records=ownrows))
        report=dict(passed=True,domain=c['domain'],clip=c['clip'],members=members,GT_sha256=p.sha(GTpath),fresh_fullframe_hungarian=True,scope='Complete original scene; actual Hungarian GT row recovered even when same date/canonical ID repeats. No isolation filter.')
        p.save(output,report);reports.append(dict(domain=c['domain'],clip=c['clip'],path=str(output),sha256=p.sha(output)));print('AUDIT214_LABELS',c['domain'],c['clip'],flush=True)
    p.save(a.out/'labels.json',dict(passed=True,clips=reports))

def models(a):
    assert p.read(a.out/'pixels.json')['passed'] and p.read(a.out/'labels.json')['passed'];allrows=[];truth={}
    for ref,folder,c,arrays in records(a):
        allrows+=frozen_features(folder,c)
        L=p.read(a.out/'labels'/ref['domain']/(ref['clip']+'.json'))
        for member in L['members']:
            for e in member['records']:truth[e['endpoint_id']]=e
    P=p.read(a.root/'predictions/predictions.json');reports=[]
    for h in ('35','33','W','G'):
        model=p.read(a.root/'predictions/models'/f'{h}.json');source=[r for r in allrows if r['domain']!=h and r['native_slot']==0 and r['eligible'] and truth[r['endpoint_id']]['known']]
        source=sorted(source,key=lambda r:(r['domain'],r['clip'],str(r['run_id']),r['raw_index'],r['endpoint_id']));assert len(source)==model['n'];w=own.endpoint_weights(source);w=w/w.sum()
        X=np.asarray([e['G6']+e['chi24'] for e in source]);Y=np.asarray([np.clip(truth[e['endpoint_id']]['target'],-.25,.25) for e in source]);assert set(model['source_domains'])==set(('35','33','W','G'))-{h}
        for i,(r,s) in enumerate(zip(source,model['training_records'])):
            assert r['endpoint_id']==s['endpoint_id'] and s['native_slot']==0 and s['domain']!=h;p.close(w[i],s['weight']);p.close(Y[i],s['target']);p.close(X[i],s['features30'])
        for head,d in (('G',6),('I',30)):
            ownmodel=own.ridge(X[:,:d],Y,w);saved=model['heads'][head]
            for f in ('mean','scale','beta','intercept'):p.close(ownmodel[f],saved[f],2e-11)
        for member in P['members']:
            if member['domain']!=h:continue
            rr=[r for r in allrows if (r['domain'],r['clip'],r['native_slot'])==(h,member['clip'],member['member_index'])];assert member['endpoint_ids']==[e['endpoint_id'] for e in rr]
            for arm in ('K','G','I','Z'):
                expected=np.zeros((len(rr),2));head=model['heads']['G' if arm=='G' else 'I']
                if arm!='K':
                    for j,r in enumerate(rr):
                        if r['eligible']:
                            x=np.asarray(r['G6'] if arm=='G' else r['G6']+((-np.asarray(r['chi24'])).tolist() if arm=='Z' else r['chi24']));expected[j]=np.clip(((x-head['mean'])/head['scale'])@np.asarray(head['beta'])+head['intercept'],-.25,.25)
                p.close(expected,np.load(a.root/'predictions'/member['shifts'][arm]),2e-13)
        reports.append(dict(heldout=h,source_endpoints=len(source),independent_refits=2))
    p.save(a.out/'models.json',dict(passed=True,folds=reports,all18_fourblind_shifts=True,source_only_receiverK=True));print('AUDIT214_MODELS_PASS',flush=True)

def run(a):
    if not a.field_go:raise ValueError('field GO required')
    sys.path.insert(0,str(a.repo.resolve()));cv2.setNumThreads(1);a.out.mkdir(parents=True,exist_ok=True)
    for stage,sha in PINS.items():p.seal(a.root/stage,sha)
    if a.score_sha:p.seal(a.root/'score',a.score_sha)
    if a.stage=='pixels':pixels(a)
    elif a.stage=='labels':labels(a)
    elif a.stage=='models':models(a)
    else:raise ValueError('unknown stage')

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--repo',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--source212',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--stage',choices=['pixels','labels','models'],required=True);q.add_argument('--score-sha');q.add_argument('--field-go',action='store_true');run(q.parse_args())
