"""Authorized §198 phase1:sealed restricted count→420graph→60native O→gate.

No training or stage2 entry. One CPU; completed physical members are immutable
and reusable. No images,detector/tracker runs,new seeds,or test clips.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from capacity import finalize_capacity

PEER_SEAL='d53e70b757984aa42a7d2e8b6c6ea0f287bedfe564d3a6a1fde685dc2d85765f'
PEER_DATA='5465da2caf8c9cbc4d12ff853bde9458fa94788346dba694fbeeb0009a83342b'
SCORE197='3b21533355dd1ba4ff441c2fd4d6bdc518c22b6b806b456e7c699fdb9a56e7d8'
ALLOW={'BEE24-35-head','BEE24-33-head','BEE24-10','BEE24-15','BEE24-24','BEE24-28'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def write(path,value):
    text=json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    if path.exists():
        if path.read_text(encoding='utf8')!=text:raise ValueError('existing phase1 artifact mismatch:'+str(path))
    else:
        with path.open('x',encoding='utf8') as stream:stream.write(text)


def verify_small(root,names,expected=None):
    digest=sha(root/'manifest.json')
    assert digest==(root/'SEALED.sha256').read_text().strip()
    if expected is not None:assert digest==expected
    manifest=load(root/'manifest.json')
    for name in names:assert sha(root/name)==manifest[name],name
    return digest


def run(args):
    sys.path.insert(0,str(args.repo.resolve()))
    import cv2
    cv2.setNumThreads(1)
    import graph198 as graph
    import physical198 as physical
    from tools.study197.freeze import load_groups
    assert verify_small(args.root/'frozen',['deployment.json'],PEER_SEAL)==PEER_SEAL
    assert sha(args.root/'frozen/deployment.json')==PEER_DATA
    count=load(args.count/'capacity.json');assert sha(args.count/'capacity.json')==(args.count/'SEALED.sha256').read_text().strip()
    assert count['input_sha256']==PEER_DATA
    score_seal=verify_small(args.source197/'score',['summary.json','native_scores.json'],SCORE197)
    pred_seal=verify_small(args.source197/'predictions',[h+'.masks.json' for h in ('35','33','W','G')])
    summary=load(args.source197/'score/summary.json');native=load(args.source197/'score/native_scores.json')
    decisions={h:load(args.source197/'predictions'/f'{h}.masks.json') for h in ('35','33','W','G')}
    original=load_groups(args.source197/'frozen')
    assert len(original)==20 and sum(len(g['members']) for g in original)==60
    assert {g['unit'] for g in original}==set(count['units'])
    references={(arm,g['unit'],str(m['member'])):m['metrics'] for arm,rows in native.items() for g in rows for m in g['members']}
    records=[]
    for g in original:
        assert g['video'] in ALLOW
        sourceO=next(o for o in summary['oracle_certificates'] if o['unit']==g['unit'])
        for member in g['members']:
            decision=next(m for m in decisions[g['domain']]['member_clips'] if m['unit']==g['unit'] and m['member_index']==member['member_index'])
            for arm in graph.ARMS197:
                selected=decision['selected'][arm] if arm!='O' else [sourceO['global_to_local'][cid][1]
                    for cid in sourceO['oracle']['selected_ids'] if int(sourceO['global_to_local'][cid][0])==member['member_index']]
                records.append(dict(unit=g['unit'],member=member['member'],arm=arm,mask=selected,
                                     proof=dict(metrics=references[arm,g['unit'],str(member['member'])])))
    args.out.mkdir(parents=True,exist_ok=True)
    checked=graph.models_from_197(original,records)
    assert checked['checked_streams']==60 and checked['checked_arms']==420
    write(args.out/'graph_data.json',checked)
    models=checked['models'];expected={}
    masks={unit:value['member_masks'] for unit,value in count['units'].items()}
    for g in original:
        for m in g['members']:
            expected[g['unit'],str(m['member_index'])]=graph.evaluate(models[g['unit']][str(m['member'])],masks[g['unit']][str(m['member_index'])])
    write(args.out/'expected_O.json',{u:{str(m['member_index']):expected[u,str(m['member_index'])] for m in g['members']} for g in original for u in [g['unit']]})
    print('GRAPH420 VERIFIED; beginning all60 native physical O',flush=True)

    @lru_cache(maxsize=6)
    def truth(split,clip):
        assert clip in ALLOW
        path=args.repo/'runs/gt'/split/clip/'gt/gt.txt'
        entry=summary['GT_entry'][clip]
        assert sha(path)==entry['sha256']
        return np.loadtxt(path,delimiter=',',ndmin=2)[:,:6]

    audit=physical.score_batch(original,masks,lambda g:truth(g['split'],g['video']),args.out/'O_physical',
        expected_provider=lambda g,m:expected[g['unit'],str(m['member_index'])])
    assert audit['complete_members']==60
    write(args.out/'physical_batch.json',audit)
    receipt=dict(physical_batch_sha256=sha(args.out/'physical_batch.json'),graph_data_sha256=sha(args.out/'graph_data.json'),
                 original197_score_seal=score_seal,original197_prediction_seal=pred_seal,
                 peer_seal=PEER_SEAL,peer_data_sha256=PEER_DATA,count_sha256=sha(args.count/'capacity.json'))
    final=finalize_capacity(count,audit,receipt)
    final['code_receipts']={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('capacity.py'),Path(graph.__file__),Path(physical.__file__))}
    write(args.out/'capacity.json',final)
    write(args.out/'receipt.json',dict(schema='endpoint198-stage1-receipt-v1',resource_gate=final['resource_gate'],
            physical_verified_members=60,graph_checked_arms=420,capacity_sha256=sha(args.out/'capacity.json'),**receipt))
    files={p.name:sha(p) for p in args.out.iterdir() if p.is_file() and p.name not in ('manifest.json','SEALED.sha256')}
    write(args.out/'manifest.json',files)
    seal=sha(args.out/'manifest.json')
    path=args.out/'SEALED.sha256'
    if path.exists():assert path.read_text().strip()==seal
    else:path.write_text(seal+'\n')
    print(json.dumps(dict(resource_gate=final['resource_gate'],physical_verified=60,graph_verified=420,
                          hives=final['hives'],seal=seal),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source197',type=Path,required=True);p.add_argument('--count',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a)
