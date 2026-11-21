"""Whole-scene endpoint-peer physical verification; explicit GT entrance only.

No field I/O on import. Caller verifies prereg/input/mask seals before passing GT.
Each immutable member directory is reusable across orchestrator restarts.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')


def verify(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():
        raise ValueError('physical manifest seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        if Path(name).name!=name or sha(root/name)!=digest:raise ValueError('physical file hash mismatch: '+name)
    return json.loads((root/'score.json').read_text()),json.loads((root/'receipt.json').read_text())


def compare_graph(actual,expected):
    """Graph cannot derive tp_id; all other supplied common evidence is checked."""
    from copy import deepcopy
    actual=deepcopy(actual)
    for threshold,value in actual['iou'].items():
        for key,derived in (('fp',int(value['pred'])-int(value['tp'])),
                            ('fn',int(value['gt'])-int(value['tp']))):
            if derived<0 or (key in value and value[key]!=derived):
                raise ValueError('inconsistent native '+threshold+'/'+key)
            value[key]=derived
    for threshold,value in expected.get('iou',{}).items():
        for key,total in (('fp','pred'),('fn','gt')):
            if key in value and value[key]!=value[total]-value['tp']:
                raise ValueError('inconsistent graph '+threshold+'/'+key)
    def walk(a,b,path):
        if isinstance(b,dict):
            for k,v in b.items():
                if k=='tp_id':continue
                if k not in a:raise ValueError('missing graph field '+path+'/'+k)
                walk(a[k],v,path+'/'+k)
        elif isinstance(b,(list,tuple,np.ndarray)):
            if not np.array_equal(np.asarray(a),np.asarray(b)):raise ValueError('graph array mismatch '+path)
        elif isinstance(b,(float,np.floating)):
            if not np.isclose(a,b,rtol=0,atol=1e-12):raise ValueError('graph number mismatch '+path)
        elif a!=b:raise ValueError('graph value mismatch '+path)
    for key in ('nG','nP','pred_bins','gt_bins','iou','window','W','evidence'):
        if key in expected:walk(actual[key],expected[key],key)


def score_member(gt,member,selected_ids,polygon,nf,dest,expected_graph=None):
    """Return metrics/receipt, publishing full .17g MOT once, never overwriting.

    member has rows (or raw), bundle and arbitrary globally keyed candidate IDs.
    expected_graph is graph198.evaluate output; optional only for historical checks.
    """
    from tools.study197.physical import realize,array_sha,write_physical
    from tools.study195.run import score_member_native
    raw=np.asarray(member.get('raw',member.get('rows')),dtype=float)
    selected=sorted(set(map(str,selected_ids)))
    rows,realization=realize(raw,member['bundle'],selected)
    columns=[0]+list(range(2,raw.shape[1]))
    if not np.array_equal(raw[:,columns],rows[:,columns]):raise ValueError('physical geometry/order changed')
    identity=dict(raw_sha256=array_sha(raw),gt_sha256=array_sha(gt),polygon_sha256=array_sha(polygon),
                  nframes=int(nf),selected_ids=selected,rows_sha256=array_sha(rows),
                  code_sha256=sha(__file__))
    dest=Path(dest)
    if dest.exists():
        metrics,receipt=verify(dest)
        if receipt['identity']!=identity:raise ValueError('existing physical member has different inputs')
        if expected_graph is not None:compare_graph(metrics,expected_graph)
        return dict(metrics=metrics,receipt=receipt,path=str(dest),reused=True)
    dest.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        reread,mot_receipt=write_physical(rows,temporary/'mot.txt')
        if not np.array_equal(reread,rows):raise ValueError('physical readback mismatch')
        metrics=score_member_native(np.asarray(gt,float),reread,polygon,int(nf))
        if expected_graph is not None:compare_graph(metrics,expected_graph)
        receipt=dict(schema='endpoint-peer-physical-member-v1',identity=identity,
                     mot=mot_receipt,realization=realization,
                     geometry_unchanged=True,raw_row_order_preserved=True,
                     scorer='tools.study195.run.score_member_native',
                     tie_order='study197 fixed canonical rank odd/even IDs')
        dump(temporary/'score.json',metrics);dump(temporary/'receipt.json',receipt)
        dump(temporary/'manifest.json',{p.name:sha(p) for p in temporary.iterdir()})
        (temporary/'SEALED.sha256').write_text(sha(temporary/'manifest.json')+'\n')
        if dest.exists():raise FileExistsError(dest)
        temporary.rename(dest)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return dict(metrics=metrics,receipt=receipt,path=str(dest),reused=False)


def score_batch(groups,masks,gt_provider,out,expected_provider=None,member_filter=None):
    """masks[unit][str(member_index)] -> global IDs; GT provider is explicit.

    For capacity.json use {u:r['oracle']['member_masks'] for u,r in units.items()}.
    expected_provider(group,member) returns mandatory graph oracle witnesses for O.
    A stage2 caller supplies the sealed I masks; cached O need not be rescored.
    """
    out=Path(out);videos=[];proofs=[]
    for group in groups:
        selected_members=[m for m in group['members'] if member_filter is None or member_filter(group,m)]
        if not selected_members:continue
        gt=gt_provider(group);scored=[]
        for member in selected_members:
            index=str(member['member_index']);unit=group['unit']
            ids=masks[unit][index]
            token=hashlib.sha256((unit+'|'+index).encode()).hexdigest()[:20]
            expected=None if expected_provider is None else expected_provider(group,member)
            result=score_member(gt,member,ids,group['polygon'],group['nf'],out/token,expected)
            scored.append(dict(member=member['member'],metrics=result['metrics']))
            proofs.append(dict(unit=unit,member_index=int(index),path=token,
                               seal=(out/token/'SEALED.sha256').read_text().strip(),
                               native_verified=True,graph_verified=expected is not None,
                               pred_bins=result['metrics']['pred_bins'],gt_bins=result['metrics']['gt_bins'],
                               nG=result['metrics']['nG'],nP=result['metrics']['nP'],
                               window=result['metrics']['window'],iou=result['metrics']['iou']))
        videos.append(dict(domain=group['domain'],group=group['group'],video=group['video'],members=scored))
    return dict(schema='endpoint-peer-physical-batch-v1',videos=videos,members=proofs,
                complete_members=len(proofs),requested_masks_sha256=hashlib.sha256(
                    json.dumps(masks,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def self_test():
    from tools.study197.physical import canonicalize,frame_invariance
    rows=np.array([[f,9,x,2,2,2,.8] for f,x in [(1,-3),(2,1),(3,2),(4,3),(5,4)]],float)
    bundle=canonicalize(rows)
    bundle['candidates']=[dict(candidate_id='arbitrary|global',parent_id=9,end='start',k=1,cut_after_canonical_row=0)]
    poly=[(0,0),(20,0),(20,20),(0,20)]
    gt=rows[:,:6].copy();member=dict(rows=rows,bundle=bundle)
    with tempfile.TemporaryDirectory() as tmp:
        a=score_member(gt,member,[],poly,75,Path(tmp)/'K')
        b=score_member(gt,member,['arbitrary|global'],poly,75,Path(tmp)/'O')
        score_member(gt,member,['arbitrary|global'],poly,75,Path(tmp)/'O',b['metrics'])
        from copy import deepcopy
        expected=deepcopy(b['metrics'])
        for value in expected['iou'].values():
            value.update(fp=value['pred']-value['tp'],fn=value['gt']-value['tp'])
        compare_graph(b['metrics'],expected)
        expected['iou']['0.5']['fp']+=1
        try:compare_graph(b['metrics'],expected)
        except ValueError:pass
        else:raise AssertionError('inconsistent derived FP not rejected')
        assert score_member(gt,member,[],poly,75,Path(tmp)/'K')['reused']
        reread=np.loadtxt(Path(tmp)/'O/mot.txt',delimiter=',')[:,:7]
        frame_invariance(gt,rows,reread)
        try:score_member(gt,member,[],poly,75,Path(tmp)/'O')
        except ValueError:pass
        else:raise AssertionError('overwrite input mismatch not rejected')
        assert a['metrics']['window']['numerator']==0
    return 'PASS: official .5/.3, geometry/order, physical readback, graph equality, restart and overwrite guards'


if __name__=='__main__':
    import argparse,sys
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--self-test',action='store_true',required=True)
    a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
    print(self_test())
