"""Endpoint-cut full-scene physical MOT; no row/box/score/date deletion or mixing.

Canonical parent rank r has IDs2r+1(left/uncut),2r+2(right), keeping every
frame-Hungarian parent tie order fixed across all intervention configurations.
Only already official-gap-split native parents may be cut once.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.dont_write_bytecode=True
import numpy as np


def array_sha(a):
    a=np.ascontiguousarray(a,dtype='<f8')
    return hashlib.sha256(json.dumps(list(a.shape),separators=(',',':')).encode()+b'\n'+a.tobytes()).hexdigest()


def canonicalize(raw):
    """Official sorted parents plus a bijective canonical→raw native-row map."""
    from beetrack.evaluate import by_track
    R=np.asarray(raw,dtype=np.float64)
    if (R.ndim!=2 or R.shape[1]<6 or not np.isfinite(R).all() or
            not np.equal(R[:,:2],np.floor(R[:,:2])).all() or (R[:,0]<1).any() or
            (R[:,1]<0).any() or (R[:,4:6]<=0).any()):
        raise ValueError('finite positive native MOT rows and integral clock/IDs required')
    if len(np.unique(R[:,:2],axis=0))!=len(R):raise ValueError('duplicate native(frame,ID)')
    aliases={}
    for identity in np.unique(R[:,1]):
        dates=np.sort(R[R[:,1]==identity,0]);splits=1+int(np.sum(np.diff(dates)>25))
        for run in range(splits):
            alias=int(identity)+100000*run
            if alias>=2**53 or alias in aliases:raise ValueError('ambiguous or unsafe official run alias')
            aliases[alias]=(int(identity),run)
    tagged=by_track(np.column_stack((R,np.arange(len(R)))))
    C=tagged[:,:R.shape[1]];index=tagged[:,-1].astype(np.int64)
    if not np.array_equal(np.sort(index),np.arange(len(R))) or not np.array_equal(by_track(C),C):
        raise ValueError('official canonicalization is not bijective/idempotent')
    blocks=[]
    for rank,parent in enumerate(np.unique(C[:,1])):
        ix=np.flatnonzero(C[:,1]==parent)
        if len(np.unique(C[ix,0]))!=len(ix):raise ValueError('canonical simultaneous parent rows')
        blocks.append(dict(rank=rank,parent_id=int(parent),start=int(ix[0]),end=int(ix[-1])+1))
    return dict(raw_rows=R,canonical_rows=C,source_raw_index=index,parent_blocks=blocks)


def _validated(raw,bundle):
    reference=canonicalize(raw)
    for key in ('canonical_rows','source_raw_index'):
        if not np.array_equal(np.asarray(bundle[key]),reference[key]):raise ValueError('producer canonical provenance differs: '+key)
    keys=('rank','parent_id','start','end')
    if [{k:b[k] for k in keys} for b in bundle.get('parent_blocks',[])]!=reference['parent_blocks']:
        raise ValueError('producer canonical parent blocks differ')
    return reference


def realize(raw,bundle,selected_ids):
    """Return complete native-row-order MOT and exact split/provenance metadata."""
    reference=_validated(raw,bundle)
    R,C,index=reference['raw_rows'],reference['canonical_rows'],reference['source_raw_index']
    candidates=bundle['candidates']
    known={c['candidate_id']:c for c in candidates}
    selected=list(selected_ids)
    if len(known)!=len(candidates) or len(set(selected))!=len(selected) or not set(selected)<=set(known):
        raise ValueError('unique known candidate IDs required')
    cuts={}
    for candidate_id in selected:
        candidate=known[candidate_id];parent=candidate['parent_id']
        if parent in cuts:raise ValueError('at most one registered endpoint cut per canonical parent')
        cuts[parent]=candidate
    out=C.copy();segments=[]
    for block in reference['parent_blocks']:
        lo,hi,rank,parent=(block[k] for k in ('start','end','rank','parent_id'))
        left_id,right_id=2*rank+1,2*rank+2
        out[lo:hi,1]=left_id
        chosen=cuts.get(parent)
        cut=None
        if chosen:
            last=chosen['cut_after_canonical_row']
            if type(last) is not int or not lo<=last<hi-1:raise ValueError('cut must be an interior registered canonical boundary')
            cut=last+1
            k=chosen['k']
            if k not in (1,2) or chosen['end'] not in ('start','end'):raise ValueError('registered short endpoint length1/2 only')
            if (cut-lo if chosen['end']=='start' else hi-cut)!=k:raise ValueError('registered endpoint count differs from cut boundary')
            out[cut:hi,1]=right_id
        segments.append(dict(parent_id=parent,rank=rank,canonical_start=lo,canonical_end=hi,
                             left_id=left_id,right_id=right_id if cut is not None else None,
                             cut_before_canonical_row=cut,candidate_id=chosen['candidate_id'] if chosen else None))
    from beetrack.evaluate import by_track
    if not np.array_equal(by_track(out),out):raise ValueError('fresh IDs unexpectedly require further gap splitting')
    physical=np.empty_like(R);physical[index]=out
    columns=[0,*range(2,R.shape[1])]
    if not np.array_equal(physical[:,columns],R[:,columns]):raise ValueError('native row order/date/box/score altered')
    return physical,dict(segments=segments,selected_candidates=selected,source_raw_index=index.tolist(),
                         source_rows_sha256=array_sha(R),realized_rows_sha256=array_sha(physical),
                         allocator='canonical sorted parent rank r:2r+1 left/uncut,2r+2 right',
                         no_box_or_date_or_score_or_row_order_change=True)


def frame_invariance(gt,baseline,intervention):
    """Official full-scene frame matching, including identical-box competitors."""
    from beetrack.evaluate import by_track,frame_match
    truth=by_track(gt);left,right=by_track(baseline),by_track(intervention)
    orders=[np.argsort(R[:,0],kind='stable') for R in (left,right)]
    columns=[0,*range(2,left.shape[1])]
    if not np.array_equal(left[orders[0]][:,columns],right[orders[1]][:,columns]):
        raise ValueError('framewise geometry/score tie order changed')
    evidence={}
    for threshold in (.5,.3):
        owners=[frame_match(truth,R,thr=threshold)[o] for R,o in zip((left,right),orders)]
        if not np.array_equal(owners[0],owners[1]):raise ValueError('official frame GT owners changed')
        evidence[str(threshold)]=dict(equal=True,owner_count=len(owners[0]),owner_sha256=array_sha(owners[0]),
                                      assigned=int(np.sum(owners[0]>=0)))
    return dict(frame_geometry_order_sha256=array_sha(left[orders[0]][:,columns]),thresholds=evidence)


def write_physical(rows,path):
    from tools.study195.freeze import write_mot_exact
    if np.asarray(rows).shape[1]!=7:raise ValueError('native physical MOT requires exact7columns including score')
    receipt=write_mot_exact(rows,path)
    data=np.loadtxt(path,delimiter=',',usecols=range(7),ndmin=2) if Path(path).stat().st_size else np.empty((0,7))
    return data,receipt


def score_configuration(gt,raw,bundle,selected_ids,polygon,nframes,physical_path=None):
    """Full-scene official event metrics; no component-only Hungarian matching."""
    baseline,_=realize(raw,bundle,[])
    rows,metadata=realize(raw,bundle,selected_ids)
    invariance=frame_invariance(gt,baseline,rows)
    original=frame_invariance(gt,raw,baseline)
    receipt=None
    if physical_path is not None:rows,receipt=write_physical(rows,physical_path)
    from tools.study195.run import score_member_native
    metrics=score_member_native(gt,rows,polygon,nframes)
    return dict(metrics=metrics,realization=metadata,tie_invariance=invariance,
                native_baseline_tie_invariance=original,physical_receipt=receipt)


def self_test(repo):
    sys.path.insert(0,str(repo.resolve()))
    from beetrack.evaluate import by_track
    poly=[[0,0],[10,0],[10,30],[0,30]]
    def track(tid,frames,xs,y):return [[f,tid,x,y,2,2,.812345678901] for f,x in zip(frames,xs)]
    raw=np.asarray(track(7,range(1,6),[14,7,6,5,4],5)+track(11,range(1,6),[14,7,6,5,4],5)+
                   track(15,range(1,6),[4,5,6,7,14],18)+track(7,(40,41,42),[14,5,4],5),float)
    raw=raw[np.random.default_rng(197).permutation(len(raw))]
    gt=np.asarray(track(101,range(1,6),[14,7,6,5,4],5)+track(102,range(1,6),[4,5,6,7,14],18)+
                  track(103,(40,41,42),[14,5,4],5),float)
    bundle=canonicalize(raw);bundle['candidates']=[]
    for block in bundle['parent_blocks']:
        if block['end']-block['start']==5:
            prefix=block['parent_id']!=15
            bundle['candidates'].append(dict(candidate_id=len(bundle['candidates']),parent_id=block['parent_id'],
                                             end='start' if prefix else 'end',k=1,
                                             cut_after_canonical_row=block['start'] if prefix else block['end']-2))
    from itertools import combinations
    ids=[c['candidate_id'] for c in bundle['candidates']]
    baseline,_=realize(raw,bundle,[])
    assert frame_invariance(gt,raw,baseline)['thresholds']['0.5']['equal']
    columns=[0,2,3,4,5,6]
    with TemporaryDirectory(prefix='endpoint-physical-synthetic-') as tmp:
        for length in range(len(ids)+1):
            for chosen in combinations(ids,length):
                rows,meta=realize(raw,bundle,chosen)
                assert np.array_equal(rows[:,columns],raw[:,columns])
                assert np.array_equal(by_track(by_track(rows)),by_track(rows))
                proof=score_configuration(gt,raw,bundle,chosen,poly,75,Path(tmp)/(str(length)+'_'+''.join(map(str,chosen))+'.txt'))
                assert proof['tie_invariance']['thresholds']['0.3']['equal']
    empty=canonicalize(np.empty((0,7)));empty['candidates']=[]
    assert realize(np.empty((0,7)),empty,[])[0].shape==(0,7)
    print('Synthetic all endpoint subsets, prefix/suffix/gap runs, identical-box owner ties, native order/precision/physical score passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test(args.repo)
    else:parser.error('field orchestration remains disabled until preregistration GO')
