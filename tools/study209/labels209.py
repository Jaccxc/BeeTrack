"""Source-only209 rectangle-compatible weak labels, never pixel identity truth.

Each box needs exactly one compatible GT rectangle at .30. GT may support both
predictions; no reverse-prediction-degree restriction is applied. Only the
GT-free frozen representative date supplies the training label.
"""
import os
import sys
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.dont_write_bytecode=True
from pathlib import Path
from collections import Counter
import hashlib
import json
import importlib.util
import argparse
import subprocess
import re
import tempfile
import gzip
import numpy as np

MARKER='[cross-part-motion-v1]'
DOMAINS=('35','33','W','G')
THRESHOLD=.30


def date_labels(frame,boxes,GT_rows,GT_raw_indices):
    """Two xywh boxes against ALL original GT rectangles on one actual date."""
    from beetrack.evaluate import iou_matrix
    boxes=np.asarray(boxes,float);G=np.asarray(GT_rows,float).reshape(-1,6);rawindices=np.asarray(GT_raw_indices,np.int64)
    if boxes.shape!=(2,4) or rawindices.shape!=(len(G),):raise ValueError('two xywh boxes and parallel original GT indices required')
    if len(G) and not np.all(G[:,0]==frame):raise ValueError('GT frame table contains another date')
    result=dict(frame=int(frame),**{'class':'U'},reason=None,boxes=boxes.tolist(),box_evidence=[])
    if not np.isfinite(boxes).all() or not (boxes[:,2:]>0).all():result['reason']='invalid_prediction_geometry';return result
    if len(G) and (not np.isfinite(G).all() or not (G[:,4:6]>0).all()):result['reason']='uninterpretable_GT_frame_geometry';return result
    if not len(G):result['reason']='no_GT_annotation_at_shared_date';return result
    if (G[:,:2]!=np.rint(G[:,:2])).any():result['reason']='nonintegral_GT_identity_or_date';return result
    xy=lambda b:np.c_[b[:,:2],b[:,:2]+b[:,2:]]
    matrix=iou_matrix(xy(G[:,2:6]),xy(boxes));multiplicity=Counter(map(int,G[:,1]));owners=[];reasons=[]
    for slot in range(2):
        hits=np.flatnonzero(matrix[:,slot]>=THRESHOLD)
        record=dict(slot=slot,compatible_count=len(hits),
            compatible=[dict(GT_raw_index=int(rawindices[i]),original_GT_ID=int(G[i,1]),iou=float(matrix[i,slot])) for i in hits],
            original_GT_ID=None,GT_raw_index=None,reason=None)
        if len(hits)!=1:record['reason']='zero_compatible_GT' if not len(hits) else 'multiple_compatible_GT'
        else:
            i=int(hits[0]);oid=int(G[i,1])
            if oid<0:record['reason']='invalid_original_GT_ID'
            elif multiplicity[oid]!=1:record['reason']='duplicate_original_GT_ID_at_shared_date'
            else:record.update(original_GT_ID=oid,GT_raw_index=int(rawindices[i]))
        if record['reason']:reasons.append(record['reason'])
        owners.append(record['original_GT_ID']);result['box_evidence'].append(record)
    if any(owner is None for owner in owners):result['reason']=';'.join(sorted(set(reasons)));return result
    result.update(**{'class':'S' if owners[0]==owners[1] else 'D'},reason='same_unique_original_GT_ID' if owners[0]==owners[1] else 'different_unique_original_GT_IDs')
    return result


def summarize_pair(pair_key,common_evidence,representative_frame):
    """A collision anywhere is diagnostic; it never replaces the frozen rep y."""
    dates=[e['frame'] for e in common_evidence]
    if dates!=sorted(set(dates)):raise ValueError('actual common dates must be unique and increasing')
    counts=Counter(e['class'] for e in common_evidence)
    representative=next((e for e in common_evidence if e['frame']==representative_frame),None)
    if representative_frame is not None and representative is None:raise ValueError('frozen representative date is not an actual common date')
    label='U' if representative is None else representative['class']
    return dict(pair_key=pair_key,representative_frame=representative_frame,representative_class=label,
        loss_eligible=label in ('S','D'),y=1 if label=='S' else (0 if label=='D' else None),
        representative_evidence=representative,common_counts={c:counts[c] for c in ('S','D','U')},
        common_dates=len(common_evidence),any_verified_collision=bool(counts['D']),
        collision_dates=[e['frame'] for e in common_evidence if e['class']=='D'],common_evidence=common_evidence,
        weak_label_scope='GT rectangle compatibility at observed dates, not same-object pixel truth')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source_labels(root,heldout,requested_domains=None,expected_prereg=None,expected_candidate_sha256=None,expected_feature_sha256=None):
    """Validate the full source whitelist before opening any label payload."""
    if heldout not in DOMAINS:raise ValueError('invalid held-out domain')
    allowed=tuple(h for h in DOMAINS if h!=heldout);requested=allowed if requested_domains is None else tuple(requested_domains)
    if len(requested)!=3 or set(requested)!=set(allowed):raise ValueError('only the other three source domains may be loaded')
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('labels209 top seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text());result=[];identity=None
    for domain in requested:
        name=domain+'.json'
        if sha(root/name)!=manifest[name]:raise ValueError('source shard drift')
        data=json.loads((root/name).read_text())
        current=(data['prereg'],data['candidate_manifest_sha256'],data['feature_manifest_sha256'])
        if identity is None:identity=current
        if data['domain']!=domain or current!=identity:raise ValueError('source shard identity mismatch')
        for expected,actual in zip((expected_prereg,expected_candidate_sha256,expected_feature_sha256),current):
            if expected is not None and expected!=actual:raise ValueError('label source parent/prereg mismatch')
        # Training never receives the all-common-date diagnostics.
        result.extend(dict(pair_key=r['pair_key'],domain=domain,clip=r['clip'],pair_id=r['pair_id'],
            representative_frame=r['representative_frame'],representative_class=r['representative_class'],
            loss_eligible=r['loss_eligible'],y=r['y']) for r in data['records'])
    return result


def selftest():
    # No field, images, GT loader, detector or fitting is used.
    G=np.array([[7,11,0,0,10,10],[7,22,30,0,10,10]],float)
    a=date_labels(7,[[0,0,10,10],[0,0,10,10]],G,[4,5]);assert a['class']=='S' # two predictions may share one GT
    b=date_labels(7,[[0,0,10,10],[30,0,10,10]],G,[4,5]);assert b['class']=='D'
    duplicate=np.r_[G,[[7,11,80,0,10,10]]]
    assert date_labels(7,[[0,0,10,10],[30,0,10,10]],duplicate,[4,5,6])['class']=='U'
    ambiguous=np.r_[G,[[7,33,0,0,10,10]]]
    assert date_labels(7,[[0,0,10,10],[30,0,10,10]],ambiguous,[4,5,6])['class']=='U'
    assert summarize_pair('synthetic',[a],None)['y'] is None
    assert summarize_pair('synthetic',[b],7)['y']==0
    try:load_source_labels(Path('must-not-open'),'35',['35','W','G'])
    except ValueError:pass
    else:raise AssertionError('target whitelist was not checked before I/O')
    return 'LABELS209_TINY_SYNTHETIC_PASS'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def verify_tree(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('input seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=digest:raise ValueError('artifact drift '+name)
    return manifest


def label_clip(catalog,GT_rows):
    """One clip at a time, one shared GT frame table, all frozen pair dates."""
    G=np.asarray(GT_rows,float)
    if G.ndim!=2 or G.shape[1]!=6:raise ValueError('original GT Nx6 required')
    metadata=catalog['metadata'];domain=metadata['domain'];clip=metadata['clip'];rows=catalog['rows']
    date_indices={int(date):np.flatnonzero(G[:,0]==date) for date in np.unique(G[:,0])}
    frame_tables={};records=[]
    for pair in catalog['pairs']:
        indices=np.asarray(pair['row_indices'],np.int64);dates=pair['shared_dates']
        if indices.shape!=(len(dates),2):raise ValueError('pair common date/index alignment')
        evidence=[]
        for date,ix in zip(dates,indices):
            P=rows[ix]
            if not np.all(P[:,0]==date):raise ValueError('frozen pair row/date mismatch')
            gi=date_indices.get(int(date),np.empty(0,np.int64))
            if str(date) not in frame_tables:frame_tables[str(date)]=dict(GT_raw_indices=gi.tolist(),GT_rows=G[gi].tolist())
            proof=date_labels(date,P[:,2:6],G[gi],gi)
            proof.update(prediction_raw_indices=ix.tolist(),prediction_canonical_ids=[int(x) for x in pair['canonical_ids']],
                prediction_source_keys=[catalog['source_keys'][i] for i in ix],GT_frame_table_key=str(date))
            evidence.append(proof)
        representative=pair['representative'];date=None if representative is None else int(representative['frame'])
        key=f'{domain}/{clip}/{pair["pair_id"]}'
        record=summarize_pair(key,evidence,date)
        record.update(domain=domain,clip=clip,pair_id=pair['pair_id'],geometric_eligible=pair['geometric_eligible'],
                      date_status=pair['date_status'],canonical_ids=pair['canonical_ids'])
        records.append(record)
    return dict(schema='common209-label-clip-evidence-v1',domain=domain,clip=clip,GT_frames=frame_tables,records=records)


def run(args):
    """Explicit field-stage I/O; invoking it requires root's separate CPU GO."""
    sys.path.insert(0,str(args.repo.resolve()))
    from beetrack.evaluate import load_tracks
    base=Path(__file__).parent;catalog_module=module('common209catalog',base/'catalog209.py')
    prior=module('common209_GTpins',base.parent/'study203'/'owners203.py')
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('published full prereg required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('unique209 marker required')
    if args.out.exists() or args.out.resolve().drive.upper()!='D:':raise ValueError('fresh D: labels root required')
    verify_tree(args.candidates);verify_tree(args.features)
    top=json.loads((args.candidates/'deployment.json').read_text());featuretop=json.loads((args.features/'deployment.json').read_text())
    if top['schema']!='common209-freeze-v1' or top['prereg']!=args.prereg or featuretop['prereg']!=args.prereg or top['source208']!=catalog_module.SOURCE208:
        raise ValueError('candidate/feature prereg or source mismatch')
    if {(r['domain'],r['clip'],r['seed']) for r in top['clips']}!=set(catalog_module.SPECS) or len(top['clips'])!=6:raise ValueError('fixed-six source roster required')
    candidate_sha=sha(args.candidates/'manifest.json');feature_sha=sha(args.features/'manifest.json')
    if sha(args.inputs208/'manifest.json')!=catalog_module.SOURCE208 or (args.inputs208/'SEALED.sha256').read_text().strip()!=catalog_module.SOURCE208:
        raise ValueError('wrong historical208 K source')
    shards={h:dict(schema='common209-label-shard-v1',domain=h,prereg=args.prereg,
        candidate_manifest_sha256=candidate_sha,feature_manifest_sha256=feature_sha,records=[]) for h in DOMAINS}
    hashes={};clip_refs=[];args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.labels209-',dir=args.out.parent) as temporary:
        target=Path(temporary)/'payload';target.mkdir()
        for ref in top['clips']:
            folder=args.candidates/ref['path']
            if sha(folder/'manifest.json')!=ref['seal']:raise ValueError('catalog child seal differs')
            catalog=catalog_module.load_clip(folder,args.inputs208);meta=catalog['metadata'];domain=ref['domain'];clip=ref['clip']
            split='dev' if clip.endswith('-head') else 'val'
            if meta['domain']!=domain or meta['clip']!=clip or meta['seed']!=ref['seed'] or meta['split']!=split:raise ValueError('unauthorized fixed clip metadata')
            gtpath=args.repo/'runs/gt'/split/clip/'gt/gt.txt';digest=sha(gtpath)
            if digest!=prior.GT_SHA256.get(clip):raise ValueError('original GT differs historical six pins')
            relative=gtpath.relative_to(args.repo).as_posix();hashes[relative]=digest
            data=label_clip(catalog,load_tracks(gtpath));data.update(prereg=args.prereg,GT_relative_path=relative,GT_sha256=digest,
                candidate_manifest_sha256=candidate_sha,feature_manifest_sha256=feature_sha,catalog_clip_seal=ref['seal'])
            dest=target/domain/clip;dest.mkdir(parents=True)
            catalog_module.save_gz(dest/'evidence.json.gz',data)
            for r in data['records']:
                shards[domain]['records'].append({k:r[k] for k in ('pair_key','domain','clip','pair_id','representative_frame','representative_class','loss_eligible','y','geometric_eligible','date_status')})
            counts=dict(Counter(r['representative_class'] for r in data['records']))
            catalog_module.save(dest/'receipt.json',dict(prereg=args.prereg,candidate_manifest_sha256=candidate_sha,feature_manifest_sha256=feature_sha,
                GT_sha256=digest,catalog_clip_seal=ref['seal'],code_sha256=sha(__file__),pairs=len(data['records']),representative_classes=counts,
                weak_label_scope='rectangle compatibility, not same-object pixel identity'))
            clip_refs.append(dict(domain=domain,clip=clip,path=f'{domain}/{clip}/evidence.json.gz',pairs=len(data['records'])))
            print('LABEL209_CLIP',domain,clip,len(data['records']),flush=True)
            del data,catalog
        if any(sha(args.repo/path)!=digest for path,digest in hashes.items()):raise ValueError('GT changed during labels')
        for domain,data in shards.items():catalog_module.save(target/(domain+'.json'),data)
        catalog_module.save(target/'deployment.json',dict(schema='common209-labels-v1',prereg=args.prereg,marker=MARKER,
            candidate_manifest_sha256=candidate_sha,feature_manifest_sha256=feature_sha,input208_manifest_sha256=catalog_module.SOURCE208,
            GT_sha256=hashes,clips=clip_refs,code_sha256=sha(__file__)))
        (target/'labels209.py').write_bytes(Path(__file__).read_bytes())
        catalog_module.save(target/'manifest.json',{p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()})
        (target/'SEALED.sha256').write_text(sha(target/'manifest.json')+'\n');target.rename(args.out)
    print('LABELS209_SEALED',sha(args.out/'manifest.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--self-test',action='store_true')
    p.add_argument('--candidates',type=Path);p.add_argument('--features',type=Path);p.add_argument('--inputs208',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--prereg')
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    if args.self_test:print(selftest())
    else:
        if any(getattr(args,k) is None for k in ('candidates','features','inputs208','out','prereg')):p.error('--candidates --features --inputs208 --out --prereg required')
        run(args)
