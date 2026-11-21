"""Post-geometry diagnostic: one menu choice/segment shared by all30 phases.

Only reads the verified compact geometry.json. Complete integer enumeration
gives an attainable event-menu witness and independently replayable optimum.
This is an ideal geometry-menu solution, not a deployable repair or physical
box witness. Triggered after seeing the geometry ceiling; not preregistered.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import bounds190 as B

EXPECTED='3c931b9ecb5187fd8b92facc476a47a9333b5a51d31678959a54dd9dc446ac90'


def solve(cell):
    nf=cell['nf'];fixed=np.asarray(cell['baseline_phase_counts'],np.int64).copy()
    truth=B.phase_counts(cell['gt_events'],nf)
    groups=[];singletons=[];space=[]
    for menu in cell['menus']:
        original=[e for e in menu['original'] if e is not None]
        fixed-=B.phase_counts(original,nf)
        choices=[];seen=set()
        for event in menu['geometry']:
            vector=B.phase_counts([] if event is None else [event],nf)
            key=vector.tobytes()
            if key not in seen:
                seen.add(key);choices.append((event,vector))
        assert choices
        space.append(dict(B=menu['B'],raw_options=len(menu['geometry']),distinct_vectors=len(choices)))
        if len(choices)==1:
            event,vector=choices[0];fixed+=vector
            singletons.append(dict(B=menu['B'],event=event))
        else:
            groups.append((menu['B'],choices))
    assert (fixed>=0).all()
    best=None;best_indices=None;histogram=Counter();leaves=0
    chosen=[0]*len(groups)

    def visit(depth,counts):
        nonlocal best,best_indices,leaves
        if depth==len(groups):
            objective=int(np.abs(counts-truth).sum())
            histogram[objective]+=1;leaves+=1
            if best is None or objective<best:
                best=objective;best_indices=chosen.copy()
            return
        for index,(_,vector) in enumerate(groups[depth][1]):
            chosen[depth]=index
            visit(depth+1,counts+vector)

    visit(0,fixed)
    expected=math.prod(len(choices) for _,choices in groups)
    assert leaves==expected==sum(histogram.values()) and min(histogram)==best
    witness=singletons.copy();counts=fixed.copy()
    for (identity,choices),index in zip(groups,best_indices):
        event,vector=choices[index];counts+=vector
        witness.append(dict(B=identity,event=event))
    assert int(np.abs(counts-truth).sum())==best
    # Reconstruct chosen complete counts a second way: remove originals then
    # add every affected segment's single selected event, including fixed menus.
    replay=np.asarray(cell['baseline_phase_counts'],np.int64).copy()
    for menu in cell['menus']:
        replay-=B.phase_counts([e for e in menu['original'] if e is not None],nf)
    replay+=B.phase_counts([w['event'] for w in witness if w['event'] is not None],nf)
    assert np.array_equal(counts,replay)
    baseline=int(np.abs(np.asarray(cell['baseline_phase_counts'],np.int64)-truth).sum())
    denom=30*len(cell['gt_events']);geometry_num=cell['geometry_bound']['numerator']
    assert geometry_num<=best<=baseline
    return dict(hive=cell['hive'],seed=cell['seed'],space=space,variable_segments=len(groups),
                combinations=expected,enumerated=leaves,optimal=True,minimum_numerator=best,
                denominator=denom,minimum_W_percent=100*best/denom,
                baseline_numerator=baseline,maximum_gain_percent=100*(baseline-best)/denom,
                geometry_relaxed_numerator=geometry_num,
                relaxed_optimism_percent=100*(best-geometry_num)/denom,
                histogram={str(k):v for k,v in sorted(histogram.items())},
                witness=sorted(witness,key=lambda w:w['B']))


def main(source,destination,replay_path):
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert digest(source)==EXPECTED,'Use the independently verified geometry sidecar'
    data=json.loads(source.read_text());cells={}
    for name,cell in data['cells'].items():
        cells[name]=solve(cell)
        print(name,cells[name]['combinations'],cells[name]['minimum_numerator'],flush=True)
    report=dict(schema='coherent-geometry190-posthoc-v1',posthoc=True,ideal_not_deployable=True,
                source_geometry_sha256=digest(source),code_sha256=digest(__file__),cells=cells,heads={})
    for hive in (33,35):
        members=[c for c in cells.values() if c['hive']==hive]
        report['heads'][str(hive)]=dict(mean_minimum_W_percent=float(np.mean([c['minimum_W_percent'] for c in members])),
            mean_maximum_gain_percent=float(np.mean([c['maximum_gain_percent'] for c in members])),
            mean_relaxed_optimism_percent=float(np.mean([c['relaxed_optimism_percent'] for c in members])))
    if replay_path is not None:
        old=json.loads(replay_path.read_text())
        assert report==old,'Exhaustive proof/witness replay mismatch'
        print('Complete integer exhaustive optimum and witness replay passed')
    elif destination is not None:
        with destination.open('x',encoding='utf-8') as stream:
            json.dump(report,stream,sort_keys=True,separators=(',',':'));stream.write('\n')
        checksum=destination.with_name(destination.stem+'.sha256')
        with checksum.open('x',encoding='utf-8') as stream:
            stream.write(digest(destination)+'\n')
    else:
        raise ValueError('Choose --out or --replay')
    print(json.dumps(report['heads'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('geometry',type=Path)
    parser.add_argument('--out',type=Path);parser.add_argument('--replay',type=Path);args=parser.parse_args()
    main(args.geometry,args.out,args.replay)
