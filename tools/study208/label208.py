"""Source-only208 isolated GT anchors and conservative short switch bags.

Imports and pure functions perform no field I/O. A real runner/CLI remains
disabled until the fixed input schema, published protocol and field GO exist.
"""
import os
import sys
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import hashlib
import argparse
import subprocess
import re
import tempfile
import gzip
from collections import Counter
import numpy as np


def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def reliable_anchors(raw_GT,rows,source_keys,polygon,return_prepared=False):
    """Full same-frame official .5 match, with both threshold degrees one."""
    prior=load_module('switch208_owner_core',Path(__file__).parents[1]/'study203'/'owners203.py')
    gt=prior.prepare_gt(raw_GT,polygon);scene=prior.scene(rows,source_keys)
    records=prior.owners(gt,scene,list(range(len(rows))),.5)
    owners=np.full((len(rows),2),-1,dtype=np.int64)
    for i,r in enumerate(records):
        if r['qualified'] and r['original_gt_id'] is not None and r['gt_run'] is not None:
            owners[i]=[int(r['original_gt_id']),int(r['gt_run'])]
    return (owners,records,gt) if return_prepared else (owners,records)


def duplicate_bee_census(rows,canon,owners,prepared_GT,indices,valid):
    """All adjacent reliable identity changes, independent of crop validity."""
    rows=np.asarray(rows);canon=np.asarray(canon);owners=np.asarray(owners)
    represented={tuple(map(int,w[1:3])):n for n,w in enumerate(indices)}
    gtsegments=prepared_GT['segments'];G=prepared_GT['rows'];result=[]
    for pid in sorted(set(map(int,canon))):
        ix=np.flatnonzero(canon==pid);ix=ix[np.argsort(rows[ix,0],kind='stable')]
        for a,b in zip(ix[:-1],ix[1:]):
            if (owners[[a,b]]<0).any() or owners[a,0]==owners[b,0]:continue
            leftdate=int(rows[a,0]);right=owners[b];other=np.flatnonzero((rows[:,0]==leftdate)&(canon!=pid)&(owners[:,0]==right[0])&(owners[:,1]>=0))
            segments=[s for s in gtsegments.values() if s['original_gt_id']==int(right[0])]
            possible=[]
            for s in segments:possible.extend(i for i in range(s['start'],s['end']) if G[i,0]==leftdate)
            known=len(possible)==1 and bool(np.isfinite(G[possible]).all() and (G[possible,4:6]>0).all())
            state='yes' if len(other) else ('not_confirmed_other_track' if known else 'unknown')
            reason='other_canonical_run_qualified_same_new_bee' if len(other) else ('annotated_new_bee_but_no_confirmed_other_run' if known else
                ('new_bee_has_no_GT_observation_at_left_date' if not possible else 'ambiguous_or_nonpositive_GT_presence'))
            ordinal=represented.get((int(a),int(b)))
            result.append(dict(canonical_parent_id=pid,left_raw_index=int(a),right_raw_index=int(b),
                left_frame=leftdate,right_frame=int(rows[b,0]),delta_frame=int(rows[b,0]-leftdate),
                left_owner=owners[a].tolist(),right_owner=right.tolist(),state=state,reason=reason,
                GT_presence_known=known,GT_presence_canonical_indices=possible,
                GT_presence_raw_indices=[int(prepared_GT['source_index'][i]) for i in possible],
                GT_presence_canonical_GT_IDs=[int(G[i,1]) for i in possible],
                GT_presence_run_aliases=[gtsegments[int(G[i,1])].get('run') for i in possible],
                other_confirmed_raw_indices=other.tolist(),other_canonical_parent_ids=list(map(int,canon[other])),
                other_confirmed_GT_run_aliases=owners[other,1].tolist(),
                represented_cut_ordinal=ordinal,input_valid=bool(valid[ordinal]) if ordinal is not None else False))
    return dict(scope='all adjacent reliable original-ID changes, including unrepresented/invalid crop cuts',
                records=result,counts=dict(Counter(r['state'] for r in result)))


def build_bags(rows,canon,indices,valid,owners):
    """X cut ordinals, y=1: sum in [1,length]; y=0: exact adjacent survival."""
    rows=np.asarray(rows,float);canon=np.asarray(canon);indices=np.asarray(indices);valid=np.asarray(valid)
    owners=np.asarray(owners)
    if rows.ndim!=2 or rows.shape[1]!=7 or canon.shape!=(len(rows),) or owners.shape!=(len(rows),2):
        raise ValueError('rows[M,7], canon[M], owners[M,2] required')
    if indices.ndim!=2 or indices.shape[1]!=4 or valid.shape!=(len(indices),):raise ValueError('indices[N,4],valid[N] required')
    if not np.issubdtype(canon.dtype,np.integer) or valid.dtype!=np.bool_:raise ValueError('canon integer and valid boolean required')
    if not np.issubdtype(indices.dtype,np.integer) or not np.issubdtype(owners.dtype,np.signedinteger):raise ValueError('integer indices and signed owners required')
    if np.any(indices<0) or np.any(indices>=len(rows)):raise ValueError('window index outside native rows')
    if np.any((owners[:,0]<0)!=(owners[:,1]<0)):raise ValueError('unknown owner must have both ID/run unknown')
    ordinal={};run_positions={};counts=Counter(total_native_rows=len(rows),reliable_rows=int(np.sum(owners[:,0]>=0)),input_cuts=len(indices),valid_input_cuts=int(valid.sum()))
    for pid in sorted(set(map(int,canon))):
        ix=np.flatnonzero(canon==pid);ix=ix[np.argsort(rows[ix,0],kind='stable')]
        if len(ix)>1 and np.any(np.diff(rows[ix,0])<=0):raise ValueError('canonical run dates not unique increasing')
        run_positions[pid]=ix
        for a,b in zip(ix[:-1],ix[1:]):ordinal[(int(a),int(b))]=None
    for n,window in enumerate(indices):
        a,b=map(int,window[1:3])
        if len(set(map(int,canon[window])))!=1 or (a,b) not in ordinal:raise ValueError('input cut is not adjacent in one canonical native run')
        if ordinal[(a,b)] is not None:raise ValueError('duplicate represented native cut')
        ordinal[(a,b)]=n
    X=[];Y=[];anchors=[];meta=[]
    for pid,ix in run_positions.items():
        reliable=[p for p,i in enumerate(ix) if owners[i,0]>=0]
        for lp,rp in zip(reliable[:-1],reliable[1:]):
            a,b=int(ix[lp]),int(ix[rp]);length=rp-lp;left,right=owners[a],owners[b]
            counts['adjacent_reliable_anchor_brackets']+=1
            if length>3:counts['bracket_more_than_three_observation_gaps']+=1;continue
            if left[0]==right[0]:
                if left[1]!=right[1]:counts['same_original_ID_different_GT_run_unknown']+=1;continue
                if length!=1:counts['same_ID_across_unknown_not_negative']+=1;continue
                y=0
            else:y=1
            cuts=[ordinal[(int(ix[p]),int(ix[p+1]))] for p in range(lp,rp)]
            if any(n is None for n in cuts):counts['bracket_missing_input_cut']+=1;continue
            if not all(bool(valid[n]) for n in cuts):counts['bracket_has_invalid_input_cut']+=1;continue
            X.append(list(map(int,cuts))+[-1]*(3-length));Y.append(y);anchors.append([a,b])
            meta.append(dict(canonical_parent_id=int(pid),left_raw_index=a,right_raw_index=b,
                left_frame=int(rows[a,0]),right_frame=int(rows[b,0]),left_owner=left.tolist(),right_owner=right.tolist(),
                delta_frame=int(rows[b,0]-rows[a,0]),represented_cut_right_frames=[int(rows[indices[n,2],0]) for n in cuts],
                exact_timing_positive=bool(y and length==1 and rows[b,0]-rows[a,0]==1),
                kind='exact_switch' if y and length==1 else ('interval_switch' if y else 'exact_survival'),
                cut_ordinals=list(map(int,cuts)),length=length,lower=y,upper=length if y else 0))
            counts[meta[-1]['kind']]+=1
    return dict(X=np.asarray(X,dtype=np.int64).reshape(-1,3),y=np.asarray(Y,dtype=np.int8),
                anchors=np.asarray(anchors,dtype=np.int64).reshape(-1,2),metadata=meta,denominators=dict(counts))


DOMAINS=('35','33','W','G')


def source_domains(heldout):
    if heldout not in DOMAINS:raise ValueError('invalid held-out domain')
    return tuple(h for h in DOMAINS if h!=heldout)


def load_source_bags(root,heldout,requested_domains=None,expected_prereg=None,expected_input_sha256=None):
    """Whitelist all requested domains before opening any label payload."""
    allowed=source_domains(heldout);requested=allowed if requested_domains is None else tuple(requested_domains)
    if len(requested)!=3 or set(requested)!=set(allowed):raise ValueError('training may load exactly the other three hives')
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('labels top seal mismatch')
    result=[];identity=None
    for domain in requested:
        name=domain+'.json'
        if digest(root/name)!=manifest[name]:raise ValueError('domain index drift')
        index=json.loads((root/name).read_text())
        current=(index['prereg'],index['input_manifest_sha256'])
        if identity is None:identity=current
        if index['domain']!=domain or current!=identity:raise ValueError('domain index identity')
        if expected_prereg is not None and current[0]!=expected_prereg:raise ValueError('wrong label prereg')
        if expected_input_sha256 is not None and current[1]!=expected_input_sha256:raise ValueError('wrong label parent input')
        refs=index['clips']
        for ref in refs:
            if ref['domain']!=domain:raise ValueError('cross-domain bag reference')
            path=(root/ref['bags']).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path)!=manifest[ref['bags']]:raise ValueError('source bag drift')
            with np.load(path,allow_pickle=False) as z:result.append(dict(domain=domain,clip=ref['clip'],X=z['X'],y=z['y']))
    return result


def selftest():
    rows=np.array([[f,1,f,0,2,2,.9] for f in range(1,8)],float)
    canon=np.ones(7,dtype=np.int64);indices=np.array([[i,i+1,i+2,i+3] for i in range(4)],np.int64)
    valid=np.ones(4,bool);owners=np.full((7,2),-1,np.int64)
    owners[1]=[10,0];owners[3]=[20,0];owners[4]=[20,0];owners[5]=[20,1]
    b=build_bags(rows,canon,indices,valid,owners)
    assert b['X'].tolist()==[[0,1,-1],[2,-1,-1]] and b['y'].tolist()==[1,0]
    assert b['denominators']['same_original_ID_different_GT_run_unknown']==1
    owners[3]=[10,0];b=build_bags(rows,canon,indices,valid,owners)
    assert b['denominators']['same_ID_across_unknown_not_negative']==1
    owners[3]=[20,0];valid[1]=False;b=build_bags(rows,canon,indices,valid,owners)
    assert b['y'].tolist()==[0] and b['denominators']['bracket_has_invalid_input_cut']==1
    try:load_source_bags(Path('must-not-open'),'35',['35','W','G'])
    except ValueError:pass
    else:raise AssertionError('target label request not rejected before I/O')
    return 'LABEL208_TINY_SYNTHETIC_PASS'


MARKER='[natural-switch-hazard-v1]'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')


def verify(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('label top seal mismatch')
    for name,digest in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=digest:raise ValueError('label file drift '+name)
    return dict(manifest_sha256=sha(root/'manifest.json'))


def represented_exact_labels(indices,valid,owners):
    """Posthoc only: interval bags never become exact AP/timing targets."""
    indices=np.asarray(indices);valid=np.asarray(valid,bool);owners=np.asarray(owners)
    labels=np.full(len(indices),-1,np.int8)
    for n in np.flatnonzero(valid):
        a,b=owners[indices[n,1]],owners[indices[n,2]]
        if (a<0).any() or (b<0).any():continue
        if a[0]!=b[0]:labels[n]=1
        elif a[1]==b[1]:labels[n]=0
    return labels


def load_posthoc_owners(root,domain,clip):
    """Explicit target-analysis API, separate from training's bag-only loader."""
    if domain not in DOMAINS:raise ValueError('invalid domain')
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('labels seal mismatch')
    name=f'{domain}/{clip}/owners.npy';path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path)!=manifest[name]:raise ValueError('posthoc owner file drift')
    return np.load(path,allow_pickle=False)


def run(args):
    """Real I/O entry; execution still requires the root agent's field GO."""
    sys.path.insert(0,str(args.repo.resolve()))
    from beetrack.evaluate import load_tracks
    data=load_module('switch208data',Path(__file__).with_name('data208.py'))
    prior=load_module('switch208source_owner',Path(__file__).parents[1]/'study203'/'owners203.py')
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('full published preregistration required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('unique208 marker required')
    if args.out.exists() or args.out.resolve().drive.upper()!='D:':raise ValueError('fresh D: labels root required')
    manifest=data.verify(args.inputs);top=json.loads((args.inputs/'deployment.json').read_text())
    if top['schema']!='switch208-inputs-v1' or top['prereg']!=args.prereg or top['source206']!=data.SOURCE206:
        raise ValueError('input schema/prereg/source mismatch')
    expected={(h,c,nf,seed) for h,c,nf,seed in data.SPECS};actual=set()
    for ref in top['clips']:
        folder=args.inputs/ref['path'];meta=json.loads((folder/'metadata.json').read_text())
        if sha(folder/'manifest.json')!=ref['seal'] or (folder/'SEALED.sha256').read_text().strip()!=ref['seal']:raise ValueError('input child seal')
        key=(ref['domain'],ref['clip'],meta['nf'],ref['seed'])
        split='dev' if ref['clip'].endswith('-head') else 'val'
        if key not in expected or key in actual or meta['member_index']!=0 or meta['split']!=split or meta['clip']!=ref['clip'] or meta['domain']!=ref['domain']:
            raise ValueError('not fixed six authorized members')
        actual.add(key)
    if actual!=expected:raise ValueError('incomplete six input clips')
    parent=sha(args.inputs/'manifest.json');code_sha=sha(__file__);refs=[];gt_hashes={};summaries={}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.switch208-labels-',dir=args.out.parent) as temporary:
        target=Path(temporary)/'payload';target.mkdir()
        for ref in top['clips']:
            folder=args.inputs/ref['path'];meta=json.loads((folder/'metadata.json').read_text());domain=ref['domain'];clip=ref['clip']
            rows=np.load(folder/'rows.npy',mmap_mode='r',allow_pickle=False)
            canon=np.load(folder/'canon.npy',mmap_mode='r',allow_pickle=False)
            indices=np.load(folder/'indices.npy',mmap_mode='r',allow_pickle=False)
            valid=np.load(folder/'valid.npy',mmap_mode='r',allow_pickle=False)
            sourcekeys=json.loads((folder/'source_keys.json').read_text())
            gtpath=args.repo/'runs/gt'/meta['split']/clip/'gt/gt.txt';digest=sha(gtpath)
            if digest!=prior.GT_SHA256.get(clip):raise ValueError('GT differs authorized historical six')
            relative=gtpath.relative_to(args.repo).as_posix();gt_hashes[relative]=digest
            owners,evidence,prepared_GT=reliable_anchors(load_tracks(gtpath),rows,sourcekeys,meta['polygon'],return_prepared=True)
            if not np.array_equal(canon,np.array([r['pred_key'][1] for r in evidence],np.int64)):
                raise ValueError('input canon not identical to official full-scene canonicalization')
            bags=build_bags(rows,canon,indices,valid,owners)
            duplicate=duplicate_bee_census(rows,canon,owners,prepared_GT,indices,valid)
            dest=target/domain/clip;dest.mkdir(parents=True)
            np.save(dest/'owners.npy',owners,allow_pickle=False)
            np.savez(dest/'bags.npz',X=bags['X'],y=bags['y'],anchors=bags['anchors'])
            save(dest/'bags.json',dict(schema='switch208-bag-metadata-v1',bags=bags['metadata'],denominators=bags['denominators'],
                semantics=dict(positive='sum selected cut hazards in [1,length]',negative='exactly one adjacent cut, target0',unknown='not a loss target')))
            save(dest/'duplicate_bee.json',dict(schema='switch208-duplicate-bee-census-v1',**duplicate))
            with gzip.open(dest/'evidence.json.gz','wt',encoding='utf8') as stream:
                json.dump(dict(schema='switch208-row-owners-v1',threshold=.5,rows=evidence),stream,separators=(',',':'),allow_nan=False)
            receipt=dict(schema='switch208-label-clip-v1',prereg=args.prereg,input_manifest_sha256=parent,
                input_clip_seal=ref['seal'],input_named_files={name:manifest[f'{ref["path"]}/{name}'] for name in ('rows.npy','canon.npy','indices.npy','valid.npy','source_keys.json')},
                domain=domain,clip=clip,member=ref['member'],seed=ref['seed'],GT_relative_path=relative,GT_sha256=digest,
                code_sha256=code_sha,owner_unknown=-1,owner_columns=['original_GT_ID','GT_gap_run'],
                cut_semantics='indices[n,1] | indices[n,2]',denominators=bags['denominators'],duplicate_bee_counts=duplicate['counts'])
            save(dest/'receipt.json',receipt);summaries[clip]=bags['denominators']
            refs.append(dict(domain=domain,clip=clip,member=ref['member'],seed=ref['seed'],path=f'{domain}/{clip}',
                bags=f'{domain}/{clip}/bags.npz',owners=f'{domain}/{clip}/owners.npy',n_bags=len(bags['y'])))
            print('LABEL208_CLIP',domain,clip,bags['denominators'],flush=True)
            del owners,evidence,bags,rows,canon,indices,valid,prepared_GT,duplicate
        if any(sha(args.repo/path)!=digest for path,digest in gt_hashes.items()):raise ValueError('GT changed during labeling')
        deployment=dict(schema='switch208-labels-v1',prereg=args.prereg,marker=MARKER,input_manifest_sha256=parent,
            input_deployment_sha256=sha(args.inputs/'deployment.json'),code_sha256=code_sha,GT_sha256=gt_hashes,clips=refs)
        save(target/'deployment.json',deployment)
        for domain in DOMAINS:save(target/(domain+'.json'),dict(domain=domain,prereg=args.prereg,input_manifest_sha256=parent,clips=[r for r in refs if r['domain']==domain]))
        save(target/'summary.json',dict(schema='switch208-label-summary-v1',prereg=args.prereg,input_manifest_sha256=parent,clips=summaries))
        (target/'label208.py').write_bytes(Path(__file__).read_bytes())
        save(target/'manifest.json',{p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()})
        (target/'SEALED.sha256').write_text(sha(target/'manifest.json')+'\n');target.rename(args.out)
    print('LABELS208_SEALED',sha(args.out/'manifest.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--self-test',action='store_true')
    p.add_argument('--inputs',type=Path);p.add_argument('--out',type=Path);p.add_argument('--prereg')
    args=p.parse_args()
    if args.self_test:print(selftest())
    else:
        if any(getattr(args,k) is None for k in ('inputs','out','prereg')):p.error('--inputs --out --prereg required')
        run(args)
