"""Source-only214 original endpoint center targets from full-scene matching.

No isolation gate: any official .5 assignment supplies the original GT target.
Raw GT-minus-K displacement is never clipped here. GT-side/boundary diagnostics
and peer targets are separate from receiver-only source-fold payloads.
"""
import os
import sys
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.dont_write_bytecode=True
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np
import importlib.util
import argparse
import subprocess
import re
import tempfile

DOMAINS=('35','33','W','G')


def canonical(raw):
    from beetrack.evaluate import by_track
    raw=np.asarray(raw,float)
    if raw.ndim!=2 or raw.shape[1]<6 or not np.isfinite(raw).all():raise ValueError('finite native rows required')
    tagged=by_track(np.c_[raw,np.arange(len(raw))]);return tagged[:,:-1],tagged[:,-1].astype(np.int64)


def label_scene(raw_GT,rows,endpoints,polygon,image_wh=None):
    """Match EVERY original row first; then project frozen endpoint indices."""
    import cv2
    from beetrack.evaluate import frame_match,inside,iou_matrix
    G,gi=canonical(raw_GT);P,pi=canonical(rows);rows=np.asarray(rows,float)
    assigned=frame_match(G,P,thr=.5);inverse=np.empty(len(pi),np.int64);inverse[pi]=np.arange(len(pi))
    GTlookup=defaultdict(list)
    for i,row in enumerate(G):GTlookup[(int(row[0]),int(row[1]))].append(i)
    poly=np.asarray(polygon,np.float32);seen=set();ids=set();records=[];duplicate_frame_assignments={}
    center=lambda r:r[2:4]+r[4:6]/2
    signed_distance=lambda c:float(cv2.pointPolygonTest(poly,tuple(map(float,c)),True))
    def edge_distance(c):
        if image_wh is None:return None
        width,height=image_wh;return float(min(c[0],c[1],width-c[0],height-c[1]))
    for endpoint in endpoints:
        raw=int(endpoint['raw_index']);key=endpoint['endpoint_id']
        if not 0<=raw<len(rows) or raw in seen or key in ids:raise ValueError('endpoint native indices/IDs must be unique, including singleton dedup')
        seen.add(raw);ids.add(key);position=int(inverse[raw]);R=rows[raw];pc=center(R);owner=int(assigned[position])
        record=dict(endpoint,raw_index=raw,frame=int(R[0]),original_prediction_id=int(R[1]),
            canonical_run_id=int(P[position,1]),original_row=R.tolist(),pred_center=pc.tolist(),
            pred_width=float(R[4]),pred_height=float(R[5]),pred_diagonal=float(np.hypot(R[4],R[5])),
            pred_inside=bool(inside(R[None],poly)[0]),pred_polygon_signed_distance=signed_distance(pc),
            pred_image_edge_distance=edge_distance(pc),has_owner=owner>=0,target_known=False,
            canonical_GT_owner=owner if owner>=0 else None,GT_raw_index=None,GT_original_input_id=None,
            GT_owner_ID_multiplicity=0,used_full_frame_row_assignment_recovery=False,
            GT_row=None,GT_center=None,GT_inside=None,GT_diagonal=None,GT_polygon_signed_distance=None,GT_image_edge_distance=None,
            dx_raw=None,dy_raw=None,matched_iou=None,reason=None,
            matching_scope='official .5 frame_match on complete original scene, no reverse-uniqueness condition')
        if owner<0:record['reason']='no_official_GT_assignment'
        else:
            indices=GTlookup[(int(R[0]),owner)]
            record['GT_owner_ID_multiplicity']=len(indices)
            if len(indices)!=1:
                # frame_match exposes IDs, not Hungarian row indices. Recover
                # the exact full-frame assignment if canonical GT IDs collide.
                from scipy.optimize import linear_sum_assignment
                date=int(R[0])
                if date not in duplicate_frame_assignments:
                    gg=np.flatnonzero(G[:,0]==date);pp=np.flatnonzero(P[:,0]==date)
                    xy=lambda a:np.c_[a[:,2:4],a[:,2:4]+a[:,4:6]]
                    matrix=iou_matrix(xy(G[gg]),xy(P[pp]));a,b=linear_sum_assignment(-matrix)
                    mapping={int(pp[j]):int(gg[i]) for i,j in zip(a,b) if matrix[i,j]>=.5}
                    recovered=np.array([int(G[mapping[int(p)],1]) if int(p) in mapping else -1 for p in pp],np.int64)
                    if not np.array_equal(recovered,assigned[pp]):raise AssertionError('full-frame row assignment differs official complete ID array')
                    duplicate_frame_assignments[date]=mapping
                indices=[duplicate_frame_assignments[date][position]]
                record['used_full_frame_row_assignment_recovery']=True
            if indices:
                g=int(indices[0]);T=G[g];gc=center(T)
                xy=lambda r:np.r_[r[2:4],r[2:4]+r[4:6]][None]
                value=float(iou_matrix(xy(T),xy(R))[0,0])
                if value<.5:raise AssertionError('official .5 assignment below threshold')
                record.update(target_known=True,GT_raw_index=int(gi[g]),GT_original_input_id=int(np.asarray(raw_GT)[gi[g],1]),
                    GT_row=T.tolist(),GT_center=gc.tolist(),GT_inside=bool(inside(T[None],poly)[0]),
                    GT_polygon_signed_distance=signed_distance(gc),GT_image_edge_distance=edge_distance(gc),
                    GT_diagonal=float(np.hypot(T[4],T[5])),dx_raw=float(gc[0]-pc[0]),dy_raw=float(gc[1]-pc[1]),
                    matched_iou=value,reason='original_official_assignment_raw_uncensored_target')
        records.append(record)
    return records


def minimal_training_record(record,domain,clip,member_index=0,member=None):
    """GT inside/distances are deliberately absent; fitting owns w/h + clipping."""
    return dict(domain=domain,clip=clip,role='receiverK',member_index=int(member_index),member=str(member),
        endpoint_id=record['endpoint_id'],raw_index=record['raw_index'],canonical_run_id=record['canonical_run_id'],
        eligible=bool(record.get('gate',{}).get('eligible',True)),
        has_owner=record['has_owner'],target_known=record['target_known'],dx_raw=record['dx_raw'],dy_raw=record['dy_raw'],
        pred_width=record['pred_width'],pred_height=record['pred_height'])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source_labels(root,heldout,requested_domains=None,expected_prereg=None,expected_frozen_sha256=None):
    if heldout not in DOMAINS:raise ValueError('invalid heldout hive')
    allowed=tuple(h for h in DOMAINS if h!=heldout);requested=allowed if requested_domains is None else tuple(requested_domains)
    if len(requested)!=3 or set(requested)!=set(allowed):raise ValueError('only other3 source domains permitted')
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('labels214 top seal')
    manifest=json.loads((root/'manifest.json').read_text());result=[];identity=None
    for domain in requested:
        name=domain+'.json'
        if sha(root/name)!=manifest[name]:raise ValueError('source-only shard hash')
        data=json.loads((root/name).read_text());current=(data['prereg'],data['frozen_manifest_sha256'])
        if identity is None:identity=current
        if data['domain']!=domain or current!=identity:raise ValueError('source identity differs')
        if expected_prereg is not None and current[0]!=expected_prereg:raise ValueError('wrong source prereg')
        if expected_frozen_sha256 is not None and current[1]!=expected_frozen_sha256:raise ValueError('wrong endpoint feature parent')
        if any(r['role']!='receiverK' or r['member_index']!=0 or r['domain']!=domain for r in data['records']):raise ValueError('peer or target rows in receiver source shard')
        result.extend(data['records'])
    return result


def selftest():
    R=np.array([[1,1,0,0,10,10,.9],[2,1,1,0,10,10,.9],[1,2,0,0,10,10,.8]],float)
    GT=np.array([[1,10,2,0,10,10],[2,10,3,0,10,10]],float);poly=[(-10,-10),(50,-10),(50,50),(-10,50)]
    endpoints=[dict(endpoint_id='e0',raw_index=0,positions=['first']),dict(endpoint_id='e1',raw_index=1,positions=['last'])]
    records=label_scene(GT,R,endpoints,poly)
    # Full scene includes a competing prediction, yet an official assignment is
    # usable without reciprocal-IoU uniqueness. No regression target is filled.
    assert records[0]['has_owner'] and records[0]['dx_raw']==2 and records[1]['dx_raw']==2
    assert all('GT_inside' not in minimal_training_record(r,'W','synthetic') for r in records)
    duplicate=np.array([[1,10,2,0,10,10],[1,10,32,0,10,10]],float)
    two=np.array([[1,1,0,0,10,10,.9],[1,2,30,0,10,10,.9]],float)
    recovered=label_scene(duplicate,two,[dict(endpoint_id=str(i),raw_index=i) for i in range(2)],poly)
    assert [r['GT_raw_index'] for r in recovered]==[0,1]
    assert all(r['has_owner'] and r['dx_raw']==2 and r['GT_owner_ID_multiplicity']==2 and r['used_full_frame_row_assignment_recovery'] for r in recovered)
    try:load_source_labels(Path('must-not-open'),'35',['35','W','G'])
    except ValueError:pass
    else:raise AssertionError('heldout source request not rejected before I/O')
    return 'LABELS214_TINY_SYNTHETIC_PASS'


MARKER='[endpoint-image-recenter-v1]'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def run(args):
    """Real I/O stage; root must grant sole CPU after complete GT-free freeze."""
    sys.path.insert(0,str(args.repo.resolve()));base=Path(__file__).parent;sys.path.insert(0,str(base))
    from beetrack.evaluate import load_tracks
    F=module('center214prepare',base/'prepare214.py');prior=module('center214GTpins',base.parent/'peer_suffix203'/'owners203.py')
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('published full prereg required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    text=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if text.count(MARKER)!=1:raise ValueError('unique214 marker required')
    if args.out.exists() or args.out.resolve().drive.upper()!='D:':raise ValueError('fresh D: labels root required')
    F.verify(args.frozen);top=json.loads((args.frozen/'deployment.json').read_text());parent=sha(args.frozen/'manifest.json')
    if top['schema']!='center214-prepare-v1' or top['identity']['prereg']!=args.prereg or top['originalK']!=18:
        raise ValueError('complete original18 K feature freeze required')
    if top['identity']['source212']!=F.SOURCE212 or top['identity']['source208_nominal']!=F.SOURCE208:raise ValueError('native source lineage mismatch')
    supplement=top['identity'].get('supplement')
    if not re.fullmatch('[0-9a-f]{40}',supplement or ''):raise ValueError('full registered supplementary envelope required')
    subprocess.run(['git','merge-base','--is-ancestor',supplement,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,supplement],cwd=args.repo,check=True,capture_output=True)
    expected={(h,c) for h,c,capture,nf in F.SPECS}
    if {(r['domain'],r['clip']) for r in top['clips']}!=expected or len(top['clips'])!=6:raise ValueError('only fixed six clips permitted')
    shards={h:dict(schema='center214-source-labels-v1',domain=h,prereg=args.prereg,supplement=supplement,frozen_manifest_sha256=parent,records=[]) for h in DOMAINS}
    args.out.parent.mkdir(parents=True,exist_ok=True);refs=[];hashes={};nK=0
    with tempfile.TemporaryDirectory(prefix='.labels214-',dir=args.out.parent) as temporary:
        target=Path(temporary)/'payload';target.mkdir()
        for ref in top['clips']:
            clip=F.load_clip(args.frozen,ref);domain=clip['domain'];name=clip['clip'];split='dev' if name.endswith('-head') else 'val'
            gtpath=args.repo/'runs/gt'/split/name/'gt/gt.txt';digest=sha(gtpath)
            if digest!=prior.GT_SHA256.get(name):raise ValueError('GT differs original prepared/repair semantics pin')
            relative=gtpath.relative_to(args.repo).as_posix();hashes[relative]=digest;GT=load_tracks(gtpath)
            shape=clip['nominal_image_shape'];image_wh=(shape[1],shape[0])
            for slot in (0,1,2):
                scene=F.load_scene(clip,slot);meta=scene['metadata']
                if meta['member_index']!=slot or meta['native_slot']!=slot or meta['seed']!=(3 if domain in ('35','33') else 0)+slot:
                    raise ValueError('not original fixed18 members')
                endpoints=scene['endpoints'];records=label_scene(GT,scene['rows'],endpoints,clip['polygon'],image_wh)
                if any(r['canonical_run_id']!=r['canonical_id'] for r in records):raise ValueError('endpoint original canonical run changed')
                if any(r['has_owner']!=r['target_known'] for r in records):raise AssertionError('assigned endpoint raw GT target not resolved')
                destination=target/'assessment'/domain/name/('m'+str(slot)+'.json.gz');destination.parent.mkdir(parents=True,exist_ok=True)
                F.save_gz(destination,dict(schema='center214-endpoint-targets-v1',domain=domain,clip=name,member=meta['member'],member_index=slot,
                    role='receiverK' if slot==0 else 'peerK',records=records,GT_relative_path=relative,GT_sha256=digest,
                    frozen_manifest_sha256=parent,raw_target_sign='GT center minus original K center',raw_uncensored=True,
                    official_assignment_no_isolation_filter=True))
                if slot==0:
                    shards[domain]['records'].extend(minimal_training_record(r,domain,name,slot,meta['member']) for r in records if r['gate']['eligible'])
                refs.append(dict(domain=domain,clip=name,member=str(meta['member']),member_index=slot,seed=meta['seed'],
                    role='receiverK' if slot==0 else 'peerK',path=destination.relative_to(target).as_posix()))
                nK+=1;print('LABEL214_SCENE_COMPLETE',domain,name,'m'+str(slot),flush=True)
                del scene,records
            del clip,GT
        if nK!=18:raise ValueError('full18 original scenes required')
        if any(sha(args.repo/path)!=digest for path,digest in hashes.items()):raise ValueError('GT changed during label stage')
        for domain,data in shards.items():F.save(target/(domain+'.json'),data)
        F.save(target/'deployment.json',dict(schema='center214-labels-v1',prereg=args.prereg,supplement=supplement,frozen_manifest_sha256=parent,
            original_K=refs,GT_sha256=hashes,source212_native_sha256=F.SOURCE212,
            receiver_only_source_shards=True,source_eligibility_is_frozen_GTfree_gate=True,
            peer_and_GT_side_diagnostics_not_fit_inputs=True,raw_targets_uncensored=True,code_sha256=sha(__file__)))
        (target/'labels214.py').write_bytes(Path(__file__).read_bytes())
        F.save(target/'manifest.json',{p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()})
        (target/'SEALED.sha256').write_text(sha(target/'manifest.json')+'\n');target.rename(args.out)
    print('LABELS214_SEALED',sha(args.out/'manifest.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--self-test',action='store_true')
    p.add_argument('--frozen',type=Path);p.add_argument('--out',type=Path);p.add_argument('--prereg')
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    if args.self_test:print(selftest())
    else:
        if any(getattr(args,k) is None for k in ('frozen','out','prereg')):p.error('--frozen --out --prereg required')
        run(args)
