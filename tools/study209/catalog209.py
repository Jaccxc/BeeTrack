"""GT-free209 pair catalog; SOURCE/SYNTHETIC until published prereg+root GO.

First find positive-area same-frame intersections, then include ALL shared
native dates in each pair's mean (zero IoU dates included). No image/GT/model.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import gzip
import hashlib
import heapq
import itertools
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import numpy as np

MARKER='[cross-part-motion-v1]'
SOURCE208='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94'
SPECS=(('35','BEE24-35-head',3),('33','BEE24-33-head',3),('W','BEE24-10',0),('W','BEE24-15',0),('G','BEE24-24',0),('G','BEE24-28',0))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,obj):Path(path).write_text(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')


def save_gz(path,obj):
    with Path(path).open('wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g:
            with io.TextIOWrapper(g,encoding='utf8',newline='\n') as text:
                json.dump(obj,text,sort_keys=True,separators=(',',':'),allow_nan=False);text.write('\n')


def verify(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError('artifact drift '+name)
    return manifest


def build(rows,canon,nf,coobserved_pair_limit=1000000):
    """Pure arrays+nf API; returns complete positive-intersection pair records.

    Zero-shared span-overlap pairs are counted with a capped co-observed set;
    if that optional count exceeds its memory budget it is explicitly unknown.
    The cap NEVER truncates intersection candidates, actions, or their rows.
    """
    R=np.asarray(rows,float);C=np.asarray(canon)
    if isinstance(nf,bool) or int(nf)!=nf or nf<1:raise ValueError('positive integer clip length required')
    if R.ndim!=2 or R.shape[1]!=7 or C.shape!=(len(R),):raise ValueError('rows[M,7],canon[M] required')
    if not np.isfinite(R).all() or (R[:,4:6]<=0).any() or not np.isfinite(C).all():raise ValueError('finite positive boxes required')
    if (C!=np.floor(C)).any() or (R[:,:2]!=np.floor(R[:,:2])).any() or (R[:,0]<1).any() or (R[:,0]>nf).any():raise ValueError('integral canonical IDs/native clock required')
    if len(set(zip(R[:,0],C)))!=len(R):raise ValueError('canonical same-frame duplicates')
    from beetrack.evaluate import by_track
    tagged=by_track(np.column_stack((R,np.arange(len(R)))))
    mapped=np.empty(len(R),np.int64);mapped[tagged[:,-1].astype(int)]=tagged[:,1].astype(np.int64)
    if not np.array_equal(mapped,C):raise ValueError('canon differs from official original K mapping')
    ids=sorted(map(int,np.unique(C)));rank={pid:i for i,pid in enumerate(ids)};T=len(ids)
    ordered=np.argsort(C,kind='stable');unique,starts=np.unique(C[ordered],return_index=True)
    tracks={int(pid):ordered[a:b] for pid,a,b in zip(unique,starts,[*starts[1:],len(R)])}
    lookups={pid:{int(R[j,0]):int(j) for j in ix} for pid,ix in tracks.items()}
    spans=sorted((min(lookups[pid]),max(lookups[pid]),pid) for pid in ids)
    active=[];span_pairs=0
    for first,last,pid in spans:
        while active and active[0]<first:heapq.heappop(active)
        span_pairs+=len(active);heapq.heappush(active,last)
    frameorder=np.argsort(R[:,0],kind='stable');frames,firsts=np.unique(R[frameorder,0],return_index=True)
    intersect=set();coobserved=set();count_exact=True
    for first,last in zip(firsts,[*firsts[1:],len(R)]):
        ix=frameorder[first:last];pids=C[ix].astype(np.int64);B=R[ix,2:6]
        if len(ix)<2:continue
        lo=np.maximum(B[:,None,:2],B[None,:,:2]);hi=np.minimum(B[:,None,:2]+B[:,None,2:],B[None,:,:2]+B[None,:,2:])
        hit=np.triu(np.all(hi>lo,axis=-1),1)
        a,b=np.nonzero(hit)
        for i,j in zip(a,b):intersect.add(tuple(sorted((int(pids[i]),int(pids[j])))))
        if count_exact:
            for a,b in itertools.combinations(map(int,pids),2):
                x,y=sorted((rank[a],rank[b]));coobserved.add(x*T+y)
                if len(coobserved)>coobserved_pair_limit:
                    coobserved.clear();count_exact=False;break
    pairs=[];rejected_pairs=[];mean_ok=0;no_date=0
    for pa,pb in sorted(intersect):
        common=sorted(lookups[pa].keys() & lookups[pb].keys())
        indices=np.array([[lookups[pa][f],lookups[pb][f]] for f in common],np.int64)
        A=R[indices[:,0],2:6];B=R[indices[:,1],2:6]
        wh=np.maximum(0,np.minimum(A[:,:2]+A[:,2:],B[:,:2]+B[:,2:])-np.maximum(A[:,:2],B[:,:2]))
        intersection=np.prod(wh,axis=1);aa=np.prod(A[:,2:],axis=1);ba=np.prod(B[:,2:],axis=1)
        ious=intersection/(aa+ba-intersection)
        distance=np.linalg.norm((A[:,:2]+A[:,2:]/2)-(B[:,:2]+B[:,2:]/2),axis=1)
        ratio=np.minimum(aa,ba)/np.maximum(aa,ba)
        mean=float(ious.mean());maximum=float(ious.max());eligible=mean>=.20
        if not eligible:
            rejected_pairs.append(dict(canonical_ids=[pa,pb],shared_frames=len(common),iou_sum=float(ious.sum()),
                                       mean_iou=mean,max_iou=maximum,reason='mean_iou_below_0.20',
                                       shared_rows_reconstructible_from_source208=True))
            continue
        options=np.flatnonzero((np.asarray(common)>=1)&(np.asarray(common)<nf)&(ious>0)&(ious<.80))
        representative=None
        if len(options):
            r=int(options[(len(options)-1)//2])
            representative=dict(frame=int(common[r]),shared_index=r,row_indices=indices[r].tolist(),
                                boxes=[A[r].tolist(),B[r].tolist()],iou=float(ious[r]),area_ratio=float(ratio[r]))
        mean_ok+=int(eligible);no_date+=int(eligible and representative is None)
        g4=[mean,maximum,float(np.log1p(len(common))),0. if representative is None else representative['area_ratio']]
        pairs.append(dict(pair_id=f'{pa}|{pb}',canonical_ids=[pa,pb],shared_dates=common,
            row_indices=indices.tolist(),ious=ious.tolist(),center_distance_px=distance.tolist(),
            area_ratio_minmax=ratio.tolist(),shared_frames=len(common),mean_iou=mean,max_iou=maximum,
            geometric_eligible=eligible,representative=representative,date_status='OK' if representative is not None else 'NO_DATE',
            geometry4=g4,g4=g4))
    return dict(schema='common209-catalog-v1',nframes=int(nf),raw_rows=len(R),canonical_runs=T,pairs=pairs,rejected_pairs=rejected_pairs,
        geometry_names=['mean_iou','max_iou','log1p_shared_frames','representative_area_ratio'],
        denominator=dict(all_distinct_run_pairs=T*(T-1)//2,span_overlapping_pairs=span_pairs,
            positive_intersection_pairs=len(intersect),mean_iou_eligible_pairs=mean_ok,eligible_NO_DATE_pairs=no_date,
            mean_iou_rejected_pairs=len(intersect)-mean_ok,
            coobserved_pairs=len(coobserved) if count_exact else None,
            span_overlap_zero_shared_pairs=span_pairs-len(coobserved) if count_exact else None,
            coobserved_count_exact=count_exact,coobserved_pair_count_memory_cap=coobserved_pair_limit),
        rules=dict(mean_iou_threshold=.20,positive_intersection_strict=True,include_zero_iou_shared_dates=True,
                   representative='lower median date where1<=t<nf and0<IoU<.80',
                   no_date_action='keep for representative-dependent models; geometry-only J remains eligible',
                   rejected_detail='numerator/count ledger; exact shared rows reconstruct from source208'))


def load_clip(folder,inputs208):
    """Explicit field-stage consumer entrance; no GT/image/metric file accesses."""
    folder=Path(folder);verify(folder)
    with gzip.open(folder/'catalog.json.gz','rt',encoding='utf8') as f:catalog=json.load(f)
    metadata=catalog['metadata'];root=Path(inputs208)/metadata['input208_root_relative']
    for name,key in (('rows.npy','rows_sha256'),('canon.npy','canon_sha256'),('source_keys.json','sources_sha256')):
        if sha(root/name)!=metadata[key]:raise ValueError('referenced208source drift '+name)
    return dict(catalog,rows=np.load(root/'rows.npy',mmap_mode='r'),canon=np.load(root/'canon.npy',mmap_mode='r'),
                source_keys=json.loads((root/'source_keys.json').read_text()))


def pair_rows(catalog,pair):
    ix=np.asarray(pair['row_indices'],np.int64)
    return np.asarray(catalog['rows'])[ix],[[catalog['source_keys'][j] for j in row] for row in ix]


def freeze_clip(rows,canon,nf,metadata,identity,dest):
    dest=Path(dest)
    if dest.exists():
        verify(dest)
        if json.loads((dest/'receipt.json').read_text())['identity']!=identity:raise ValueError('resume inputs/code differ')
        return
    result=build(rows,canon,nf)
    dest.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        save_gz(temp/'catalog.json.gz',dict(result,metadata=metadata))
        save(temp/'receipt.json',dict(identity=identity,denominator=result['denominator'],GT_access=False,image_access=False))
        save(temp/'manifest.json',{p.name:sha(p) for p in temp.iterdir()});(temp/'SEALED.sha256').write_text(sha(temp/'manifest.json')+'\n')
        temp.rename(dest)
    finally:
        if temp.exists():shutil.rmtree(temp)


def run(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('published full prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('unique209marker required '+MARKER)
    if args.out.resolve().drive.upper()!='D:':raise ValueError('D: output required')
    if sha(args.inputs/'manifest.json')!=SOURCE208 or (args.inputs/'SEALED.sha256').read_text().strip()!=SOURCE208:raise ValueError('208input seal mismatch')
    manifest=json.loads((args.inputs/'manifest.json').read_text())
    if sha(args.inputs/'deployment.json')!=manifest['deployment.json']:raise ValueError('208deployment drift')
    top=json.loads((args.inputs/'deployment.json').read_text())
    if {(r['domain'],r['clip'],r['seed']) for r in top['clips']}!=set(SPECS) or len(top['clips'])!=6:raise ValueError('sixfixed roster mismatch')
    clips=[]
    for record in top['clips']:
        folder=args.inputs/record['path']
        for name in ('rows.npy','canon.npy','metadata.json','source_keys.json'):
            rel=f'{record["path"]}/{name}'
            if sha(args.inputs/rel)!=manifest[rel]:raise ValueError('named208source drift '+rel)
        metadata=json.loads((folder/'metadata.json').read_text())
        metadata.update(input208_root_relative=record['path'],rows_sha256=manifest[f'{record["path"]}/rows.npy'],
                        canon_sha256=manifest[f'{record["path"]}/canon.npy'],sources_sha256=manifest[f'{record["path"]}/source_keys.json'])
        identity=dict(prereg=args.prereg,source208=SOURCE208,clip_seal=record['seal'],code_sha256=sha(__file__))
        dest=args.out/record['path']
        freeze_clip(np.load(folder/'rows.npy',mmap_mode='r'),np.load(folder/'canon.npy',mmap_mode='r'),metadata['nf'],metadata,identity,dest)
        stats=json.loads((dest/'receipt.json').read_text())['denominator']
        clips.append(dict(domain=record['domain'],clip=record['clip'],seed=record['seed'],member=record['member'],
                          path=record['path'],source208_clip_seal=record['seal'],
                          n_pairs=stats['positive_intersection_pairs'],n_geometric=stats['mean_iou_eligible_pairs'],
                          n_NO_DATE=stats['eligible_NO_DATE_pairs'],seal=(dest/'SEALED.sha256').read_text().strip()))
        print('SEALED',record['domain'],record['clip'],flush=True)
    deployment=dict(schema='common209-freeze-v1',prereg=args.prereg,marker=MARKER,source208=SOURCE208,clips=clips)
    if (args.out/'deployment.json').exists() and json.loads((args.out/'deployment.json').read_text())!=deployment:raise ValueError('top resume mismatch')
    save(args.out/'deployment.json',deployment)
    files={p.relative_to(args.out).as_posix():sha(p) for p in args.out.rglob('*') if p.is_file() and p not in (args.out/'manifest.json',args.out/'SEALED.sha256')}
    if (args.out/'manifest.json').exists() and json.loads((args.out/'manifest.json').read_text())!=files:raise ValueError('top manifest drift')
    save(args.out/'manifest.json',files);(args.out/'SEALED.sha256').write_text(sha(args.out/'manifest.json')+'\n')
    print('CATALOG6',sha(args.out/'manifest.json'))


def self_test():
    def r(f,p,x,w=4):return [f,p,x,0,w,4,.9]
    rows=np.array([r(1,1,0),r(2,1,0),r(3,1,0),r(1,2,1),r(2,2,20),r(3,2,1),
                   r(1,3,100),r(3,3,100)],float)
    canon=rows[:,1].astype(np.int64)
    out=build(rows,canon,3);p=out['pairs'][0]
    assert p['canonical_ids']==[1,2] and np.allclose(p['ious'],[.6,0,.6])
    assert np.isclose(p['mean_iou'],.4) and p['representative']['frame']==1
    assert p['geometry4'][3]==1 and p['shared_frames']==3
    duplicate=np.array([r(1,1,0),r(2,1,0),r(1,2,0),r(2,2,0)],float)
    q=build(duplicate,duplicate[:,1],3)['pairs'][0]
    assert q['geometric_eligible'] and q['date_status']=='NO_DATE' and q['geometry4'][3]==0
    no_shared=np.array([r(1,1,0),r(3,1,0),r(2,2,0)],float)
    d=build(no_shared,no_shared[:,1],4)['denominator']
    assert d['span_overlap_zero_shared_pairs']==1
    capped=build(rows,canon,3,coobserved_pair_limit=0)
    assert capped['pairs']==out['pairs'] and not capped['denominator']['coobserved_count_exact']
    perm=np.array([4,1,6,3,0,7,5,2]);shuffled=build(rows[perm],canon[perm],3)['pairs'][0]
    assert shuffled['shared_dates']==p['shared_dates'] and shuffled['ious']==p['ious']
    assert np.array_equal(rows[perm][np.asarray(shuffled['row_indices'])],rows[np.asarray(p['row_indices'])])
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'catalog.json.gz';save_gz(path,out)
        with gzip.open(path,'rt',encoding='utf8') as f:assert json.load(f)==out
    return 'PASS synthetic: all-shared mean includes zero; lower median/native indices; NO_DATE; zero-shared span count; cap only optional denominator; no field.'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--self-test',action='store_true')
    p.add_argument('--inputs',type=Path);p.add_argument('--out',type=Path);p.add_argument('--prereg')
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    if args.self_test:print(self_test())
    else:
        if args.inputs is None or args.out is None:p.error('--inputs --out --prereg required')
        run(args)
