"""Source-only210 three-anchor ownership and conservative pair labels.

All original GT IDs remain private. Source-fold loading exposes only pair
S/D/U labels; co-annotation and bracket ownership are separate diagnostics.
"""
import os
import sys
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.dont_write_bytecode=True
from pathlib import Path
from collections import Counter
import numpy as np
import hashlib
import json
import importlib.util
import argparse
import subprocess
import re
import tempfile
import gzip

DOMAINS=('35','33','W','G')
SOURCE208_LABELS='e73ba698916d5e60fda6289b04f7a9d53d75ef1767621250370bf9bb299a473a'


def GT_presence_index(raw_GT):
    G=np.asarray(raw_GT,float)
    if G.ndim!=2 or G.shape[1]!=6 or not np.isfinite(G).all() or (G[:,:2]!=np.rint(G[:,:2])).any():
        raise ValueError('finite original GT Nx6 integer clock/IDs required')
    result={}
    for i,row in enumerate(G):result.setdefault((int(row[0]),int(row[1])),[]).append(i)
    return result


def label_bracket(bracket,rows,owners,canonical_GT_ids,presence):
    rows=np.asarray(rows);owners=np.asarray(owners);canonical_GT_ids=np.asarray(canonical_GT_ids)
    indices=list(map(int,bracket['raw_indices']))
    if len(indices)!=3 or len(set(indices))!=3 or min(indices)<0 or max(indices)>=len(rows):raise ValueError('three distinct native source rows required')
    own=owners[indices];gtids=canonical_GT_ids[indices];frames=list(map(int,rows[indices,0]))
    result=dict(raw_indices=indices,frames=frames,owners=own.tolist(),canonical_GT_ids=gtids.tolist(),
                **{'class':'U'},reason=None,both_visible=None,GT_presence=None)
    if any(not 1<=b-a<=3 for a,b in zip(frames[:-1],frames[1:])):
        result['reason']='bracket_not_time_legal';return result
    if (own<0).any() or (gtids<0).any():result['reason']='one_or_more_unreliable_original_row_owners';return result
    if own[0,0]!=own[2,0] or gtids[0]!=gtids[2] or own[0,1]!=own[2,1]:
        result['reason']='outer_anchors_not_same_original_and_canonical_GT_run';return result
    outer=int(own[0,0]);middle=int(own[1,0]);result.update(outer_original_GT_ID=outer,middle_original_GT_ID=middle)
    if middle==outer and gtids[1]==gtids[0] and own[1,1]==own[0,1]:
        result.update(**{'class':'S'},reason='three_reliable_rows_same_original_and_GT_run')
    elif middle!=outer:
        observed=[dict(original_GT_ID=oid,GT_raw_indices=presence.get((frames[1],oid),[])) for oid in (outer,middle)]
        result.update(**{'class':'D'},reason='reliable_middle_different_original_GT_ID',
            both_visible=all(bool(x['GT_raw_indices']) for x in observed),GT_presence=observed)
    else:result['reason']='middle_same_original_different_canonical_GT_run'
    return result


def label_pair(pair_key,brackets,rows,owners,canonical_GT_ids,presence):
    evidence=[dict(bracket_index=i,**label_bracket(b,rows,owners,canonical_GT_ids,presence)) for i,b in enumerate(brackets)]
    count=Counter(b['class'] for b in evidence);Sids={b['outer_original_GT_ID'] for b in evidence if b['class']=='S'}
    mixed=bool(count['S'] and count['D'])
    if count['S'] and not count['D'] and len(Sids)==1:label='S';reason='one_consistent_S_original_no_D'
    elif count['D'] and not count['S']:label='D';reason='D_evidence_without_S'
    else:label='U';reason='mixed_S_and_D' if mixed else ('S_evidence_multiple_original_GT_IDs' if len(Sids)>1 else 'no_decisive_bracket')
    return dict(pair_key=pair_key,**{'class':label},reason=reason,y=1 if label=='S' else (0 if label=='D' else None),
        loss_eligible=label in ('S','D'),mixed=mixed,multiple_S_originals=len(Sids)>1,
        bracket_counts={c:count[c] for c in ('S','D','U')},brackets=evidence,
        co_visible=bool(label=='D' and any(b['class']=='D' and b['both_visible'] for b in evidence)),
        visibility_scope='both original GT IDs annotated at middle date; not image-level visibility truth')


def canonical_owner_ids(owners,raw_GT,polygon):
    """Recover canonical GT run IDs from the pinned208 original-ID/run table."""
    from tools.study199.owners199 import prepare_gt
    owners=np.asarray(owners)
    if owners.ndim!=2 or owners.shape[1]!=2 or not np.issubdtype(owners.dtype,np.signedinteger):raise ValueError('signed original-ID/run owner table required')
    prepared=prepare_gt(raw_GT,polygon);lookup={}
    for pid,seg in prepared['segments'].items():
        if seg['original_gt_id'] is None or seg.get('run') is None:continue
        key=(int(seg['original_gt_id']),int(seg['run']))
        if key in lookup:raise ValueError('ambiguous original-ID/run canonical mapping')
        lookup[key]=int(pid)
    result=np.full(len(owners),-1,np.int64)
    for key in np.unique(owners,axis=0):
        if (key<0).any():
            if not (key==-1).all():raise ValueError('unknown owner must be [-1,-1]')
            continue
        value=lookup.get(tuple(map(int,key)))
        if value is None:raise ValueError('reliable208 owner has no matching original GT run')
        result[np.all(owners==key,axis=1)]=value
    return result


def label_clip(catalog,owners,canonical_GT_ids,raw_GT):
    metadata=catalog['metadata'];domain=metadata['domain'];clip=metadata['clip'];rows=np.asarray(catalog['rows'])
    if np.shape(owners)!=(len(rows),2) or np.shape(canonical_GT_ids)!=(len(rows),):raise ValueError('owner/source native row alignment')
    presence=GT_presence_index(raw_GT);records=[]
    for pair in catalog['pairs']:
        record=label_pair(f'{domain}/{clip}/{pair["pair_id"]}',pair['brackets'],rows,owners,canonical_GT_ids,presence)
        record.update(domain=domain,clip=clip,pair_id=pair['pair_id'],canonical_ids=pair['canonical_ids'])
        records.append(record)
    return dict(schema='interleave210-label-evidence-v1',domain=domain,clip=clip,records=records,
        GT_presence=[dict(frame=f,original_GT_ID=oid,GT_raw_indices=indices) for (f,oid),indices in sorted(presence.items())])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source_labels(root,heldout,requested_domains=None,expected_prereg=None,expected_candidate_sha256=None):
    if heldout not in DOMAINS:raise ValueError('invalid heldout hive')
    allowed=tuple(h for h in DOMAINS if h!=heldout);requested=allowed if requested_domains is None else tuple(requested_domains)
    if len(requested)!=3 or set(requested)!=set(allowed):raise ValueError('only other three source domains may be loaded')
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('labels210 seal')
    manifest=json.loads((root/'manifest.json').read_text());result=[];identity=None
    for domain in requested:
        name=domain+'.json'
        if sha(root/name)!=manifest[name]:raise ValueError('source shard hash')
        data=json.loads((root/name).read_text());current=(data['prereg'],data['candidate_manifest_sha256'],data['source208_labels_sha256'])
        if identity is None:identity=current
        if data['domain']!=domain or current!=identity or current[2]!=SOURCE208_LABELS:raise ValueError('source shard identity')
        if expected_prereg is not None and current[0]!=expected_prereg:raise ValueError('source prereg mismatch')
        if expected_candidate_sha256 is not None and current[1]!=expected_candidate_sha256:raise ValueError('source candidate parent mismatch')
        result.extend(data['records'])
    return result


def selftest():
    rows=np.array([[f,1,0,0,2,2,.9] for f in range(1,8)],float)
    owners=np.array([[1,0],[1,0],[1,0],[2,0],[1,0],[1,1],[-1,-1]],np.int64)
    canon=np.array([1,1,1,2,1,100001,-1],np.int64);presence={(4,1):[2],(4,2):[3]}
    s=dict(raw_indices=[0,1,2]);d=dict(raw_indices=[2,3,4]);u=dict(raw_indices=[0,1,5])
    assert label_pair('toy',[s],rows,owners,canon,presence)['class']=='S'
    assert label_pair('toy',[d],rows,owners,canon,presence)['co_visible']
    assert label_pair('toy',[s,d],rows,owners,canon,presence)['mixed']
    assert label_pair('toy',[u],rows,owners,canon,presence)['class']=='U'
    assert not label_pair('toy',[d],rows,owners,canon,{(4,2):[3]})['co_visible']
    try:load_source_labels(Path('must-not-open'),'35',['35','W','G'])
    except ValueError:pass
    else:raise AssertionError('target requested before I/O gate')
    return 'LABELS210_TINY_SYNTHETIC_PASS'


MARKER='[alternating-offset-v1]'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def verify_tree(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('tree seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=digest:raise ValueError('artifact drift '+name)
    return manifest


def run(args):
    """Executing real I/O also requires the root agent's separate field/CPU GO."""
    sys.path.insert(0,str(args.repo.resolve()))
    from beetrack.evaluate import load_tracks
    base=Path(__file__).parent;sys.path.insert(0,str(base))
    C=module('interleave210catalog',base/'catalog210.py')
    prior=module('interleave210GTpins',base.parent/'study203'/'owners203.py')
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('published full prereg required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('unique210 marker required')
    if args.out.exists() or args.out.resolve().drive.upper()!='D:':raise ValueError('fresh D: output required')
    verify_tree(args.candidates);top=json.loads((args.candidates/'deployment.json').read_text())
    if top['schema']!='interleave210-freeze-v1' or top['prereg']!=args.prereg or top['source208']!=C.SOURCE208:raise ValueError('candidate prereg/source mismatch')
    if {(r['domain'],r['clip'],r['seed']) for r in top['clips']}!=C.SPECS or len(top['clips'])!=6:raise ValueError('fixed six members required')
    if sha(args.inputs208/'manifest.json')!=C.SOURCE208 or (args.inputs208/'SEALED.sha256').read_text().strip()!=C.SOURCE208:raise ValueError('wrong source208 native input')
    if sha(args.labels208/'manifest.json')!=SOURCE208_LABELS or (args.labels208/'SEALED.sha256').read_text().strip()!=SOURCE208_LABELS:raise ValueError('wrong reliable208 labels')
    oldmanifest=json.loads((args.labels208/'manifest.json').read_text());parent=sha(args.candidates/'manifest.json')
    shards={h:dict(schema='interleave210-label-shard-v1',domain=h,prereg=args.prereg,
        candidate_manifest_sha256=parent,source208_labels_sha256=SOURCE208_LABELS,records=[]) for h in DOMAINS}
    args.out.parent.mkdir(parents=True,exist_ok=True);hashes={};refs=[]
    with tempfile.TemporaryDirectory(prefix='.labels210-',dir=args.out.parent) as temporary:
        target=Path(temporary)/'payload';target.mkdir()
        for ref in top['clips']:
            folder=args.candidates/ref['path']
            if sha(folder/'manifest.json')!=ref['seal']:raise ValueError('candidate clip seal')
            catalog=C.load_clip(folder,args.inputs208);meta=catalog['metadata'];domain=ref['domain'];clip=ref['clip']
            split='dev' if clip.endswith('-head') else 'val'
            if meta['domain']!=domain or meta['clip']!=clip or meta['seed']!=ref['seed'] or meta['split']!=split:raise ValueError('fixed clip metadata mismatch')
            oldpath=f'{domain}/{clip}';ownername=oldpath+'/owners.npy';receiptname=oldpath+'/receipt.json'
            for name in (ownername,receiptname):
                if sha(args.labels208/name)!=oldmanifest[name]:raise ValueError('reliable208 named source drift')
            receipt=json.loads((args.labels208/receiptname).read_text())
            if receipt['input_named_files']['rows.npy']!=meta['rows_sha256'] or receipt['input_manifest_sha256']!=C.SOURCE208:
                raise ValueError('owner row order/source differs candidate native K')
            gtpath=args.repo/'runs/gt'/split/clip/'gt/gt.txt';digest=sha(gtpath)
            if digest!=prior.GT_SHA256.get(clip) or digest!=receipt['GT_sha256']:raise ValueError('GT differs pinned208 owner source')
            relative=gtpath.relative_to(args.repo).as_posix();hashes[relative]=digest
            owners=np.load(args.labels208/ownername,mmap_mode='r',allow_pickle=False);G=load_tracks(gtpath)
            gtcanon=canonical_owner_ids(owners,G,meta['polygon']);data=label_clip(catalog,owners,gtcanon,G)
            data.update(prereg=args.prereg,candidate_manifest_sha256=parent,source208_labels_sha256=SOURCE208_LABELS,
                source208_owner_sha256=oldmanifest[ownername],GT_relative_path=relative,GT_sha256=digest)
            dest=target/domain/clip;dest.mkdir(parents=True);C.save_gz(dest/'evidence.json.gz',data)
            for r in data['records']:
                shards[domain]['records'].append({k:r[k] for k in ('pair_key','domain','clip','pair_id','class','y','loss_eligible')})
            C.save(dest/'receipt.json',dict(prereg=args.prereg,candidate_manifest_sha256=parent,source208_labels_sha256=SOURCE208_LABELS,
                source208_owner_sha256=oldmanifest[ownername],GT_sha256=digest,code_sha256=sha(__file__),engineering_complete=True))
            refs.append(dict(domain=domain,clip=clip,path=f'{domain}/{clip}/evidence.json.gz'))
            print('LABEL210_CLIP_COMPLETE',domain,clip,flush=True)
            del catalog,owners,gtcanon,data,G
        if any(sha(args.repo/path)!=digest for path,digest in hashes.items()):raise ValueError('GT changed during label extraction')
        for domain,data in shards.items():C.save(target/(domain+'.json'),data)
        C.save(target/'deployment.json',dict(schema='interleave210-labels-v1',prereg=args.prereg,marker=MARKER,
            candidate_manifest_sha256=parent,source208_labels_sha256=SOURCE208_LABELS,input208_manifest_sha256=C.SOURCE208,
            GT_sha256=hashes,clips=refs,code_sha256=sha(__file__)))
        (target/'labels210.py').write_bytes(Path(__file__).read_bytes())
        C.save(target/'manifest.json',{p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()})
        (target/'SEALED.sha256').write_text(sha(target/'manifest.json')+'\n');target.rename(args.out)
    print('LABELS210_SEALED',sha(args.out/'manifest.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--self-test',action='store_true')
    p.add_argument('--candidates',type=Path);p.add_argument('--inputs208',type=Path);p.add_argument('--labels208',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--prereg')
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    if args.self_test:print(selftest())
    else:
        if any(getattr(args,k) is None for k in ('candidates','inputs208','labels208','out','prereg')):p.error('--candidates --inputs208 --labels208 --out --prereg required')
        run(args)
