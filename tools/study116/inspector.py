"""Freeze native detector rows for a public, annotation-free list of requested frames."""
import argparse
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def wire(obj):
    return (json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()


def sha(obj):
    return hashlib.sha256(wire(obj)).hexdigest()


def integer(value, minimum):
    if isinstance(value,bool) or not isinstance(value,Integral) or value < minimum:
        raise ValueError(f'Expected an integer >= {minimum}')
    return int(value)


def number(value):
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value):
        raise ValueError('A detector row must contain finite numbers')
    return int(value) if isinstance(value,Integral) else float(value)


def freeze_predictions(requests, run, frames):
    """Return a JSON-ready manifest. Input frame keys may be integers or canonical decimal strings.

    requests has exactly {clip,nframes,frames}; frames is an ascending unique list of native 1-based indices.
    run has fold,seed,dose, optional JSON provenance, and status (default available). Explicitly unavailable
    runs require a nonempty unavailable_reason and frames=None or {}. Available runs require each requested
    frame, even when its detection list is empty. No score cutoff, sorting, or deduplication is performed.
    """
    if not isinstance(requests,dict) or set(requests)!={'clip','nframes','frames'}:
        raise ValueError('Expected only the public request fields clip, nframes and frames')
    clip=requests['clip']
    if not isinstance(clip,str) or not clip.strip() or not isinstance(requests['frames'],list):
        raise ValueError('Expected a clip name and a frame list')
    n=integer(requests['nframes'],1)
    wanted=[integer(f,1) for f in requests['frames']]
    if wanted!=sorted(set(wanted)) or (wanted and wanted[-1]>n):
        raise ValueError('Requested frames must be unique, ascending and inside this clip')
    if not isinstance(run,dict) or not {'fold','seed','dose'} <= run.keys():
        raise ValueError('A run needs fold, seed and dose')
    run=json.loads(wire(run))  # detach provenance so caller mutations cannot change a frozen record
    integer(run['fold'],1); integer(run['seed'],0)
    dose=run['dose']
    if isinstance(dose,str):
        if not dose.strip():
            raise ValueError('A dose must be named')
    else:
        number(dose)
    status=run.setdefault('status','available')
    reason=run.get('unavailable_reason')
    predictions=None
    if status=='unavailable':
        if not isinstance(reason,str) or not reason.strip() or (frames is not None and frames != {}):
            raise ValueError('An unavailable run requires a reason and no prediction data')
    elif status=='available':
        if reason is not None or not isinstance(frames,dict):
            raise ValueError('An available run requires a prediction mapping and no unavailable reason')
        by_frame={}
        for raw_key, rows in frames.items():
            if isinstance(raw_key,str):
                if not raw_key.isascii() or not raw_key.isdecimal() or raw_key!=str(int(raw_key)):
                    raise ValueError('JSON frame keys must be canonical decimal frame numbers')
                f=integer(int(raw_key),1)
            else:
                f=integer(raw_key,1)
            if f in by_frame:
                raise ValueError(f'Duplicate native frame {f}')
            by_frame[f]=rows
        if set(by_frame)!=set(wanted):
            raise ValueError(f'Missing frames {sorted(set(wanted)-set(by_frame))}; extra frames {sorted(set(by_frame)-set(wanted))}')
        predictions={}
        for f in wanted:
            rows=by_frame[f]
            if hasattr(rows,'tolist'):
                rows=rows.tolist()
            if not isinstance(rows,(list,tuple)):
                raise ValueError(f'Expected a list of detector rows at frame {f}')
            values=[]
            for row in rows:
                if not isinstance(row,(list,tuple)) or len(row) not in (5,6):
                    raise ValueError('Expected xyxy,score and optional class index')
                obj=[number(x) for x in row]
                x1,y1,x2,y2,score=obj[:5]
                if x2<x1 or y2<y1 or not 0<=score<=1:
                    raise ValueError('Invalid detection box area or confidence')
                if len(obj)==6 and (obj[5]<0 or int(obj[5])!=obj[5]):
                    raise ValueError('A detector class must be a nonnegative integer')
                values.append(obj)
            predictions[str(f)]=values
    else:
        raise ValueError('Run status must be available or unavailable')
    public=dict(clip=clip,nframes=n,frames=wanted)
    return dict(format='warp-pilot-predictions-v1',clip=clip,nframes=n,requested_frames=wanted,
                run=run,status=status,predictions=predictions,requests_sha256=sha(public),
                predictions_sha256=sha(predictions) if predictions is not None else None)


def read_json(path):
    def unique(pairs):
        obj=dict(pairs)
        if len(obj)!=len(pairs):
            raise ValueError(f'Duplicate JSON object key in {path}')
        return obj
    return json.loads(path.read_text(encoding='utf-8-sig'),object_pairs_hook=unique)


def write_once(path, record):
    raw=wire(record)
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        partial=Path(temp)/'manifest'
        partial.write_bytes(raw)
        try:
            os.link(partial,path)
        except FileExistsError:
            if path.read_bytes()!=raw:
                raise ValueError(f'The frozen manifest already differs: {path}')
    return hashlib.sha256(raw).hexdigest()


def self_check():
    public=dict(clip='mock-tail',nframes=1000,frames=[1,19,1000])
    run=dict(fold=33,seed=0,dose='medium')
    a=[1,2,3,4,.05]; b=[2,2,4,4,.05,0.0]
    frames={1:[],19:[b,a,b],1000:[]}
    ok=freeze_predictions(public,run,frames)
    assert ok['predictions']=={'1':[],'19':[b,a,b],'1000':[]}
    clipped=[[1,2,1,4,.9],[1,2,3,2,.8],[1,2,1,2,.7]]
    assert freeze_predictions(public,run,{**frames,19:clipped})['predictions']['19']==clipped
    assert freeze_predictions(public,run,{str(k):v for k,v in frames.items()})==ok
    swap=freeze_predictions(public,run,{**frames,19:[a,b,b]})
    assert swap['predictions_sha256'] != ok['predictions_sha256']
    unavailable=dict(run,status='unavailable',unavailable_reason='synthetic warp outside the frame')
    assert freeze_predictions(public,unavailable,None)['predictions_sha256'] is None
    assert freeze_predictions(public,unavailable,{})['predictions'] is None
    for req,settings,data in ((public,run,{1:[],19:[]}), (public,run,{**frames,12:[]}),
                              (dict(public,gt_id=8),run,frames),(public,run,{**frames,'19':[]}),
                              (dict(public,frames=[19,1]),run,frames),(dict(public,frames=[1,1]),run,frames),
                              (public,dict(run,status='unavailable'),None),(public,unavailable,frames),
                              (public,run,{**frames,19:[[1,2,3,4,float('nan')]]}),
                              (public,run,{**frames,19:[[3,2,1,4,.9]]}),
                              (public,run,{**frames,19:[[1,4,3,2,.9]]}),
                              (public,run,{**frames,19:[[1,2,3,4,.99,0,1]]})):
        try:
            freeze_predictions(req,settings,data)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed test case accepted')
    with TemporaryDirectory() as tmp:
        p=Path(tmp)/'freeze.json'
        assert write_once(p,ok)==write_once(p,ok) and read_json(p)==ok
        try:
            write_once(p,swap)
        except ValueError:
            pass
        else:
            raise AssertionError('Overwritten freeze')
    print('Synthetic request schema, unavailable dose, missing/empty frames, detector order, and immutable output passed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='mode',required=True)
    modes.add_parser('self-check')
    op=modes.add_parser('freeze')
    for name in ('requests','run','frames','output'):
        op.add_argument(name,type=Path)
    args=parser.parse_args()
    if args.mode=='self-check':
        self_check()
    else:
        result=freeze_predictions(read_json(args.requests),read_json(args.run),read_json(args.frames))
        print(f'{write_once(args.output,result)} {args.output}')
