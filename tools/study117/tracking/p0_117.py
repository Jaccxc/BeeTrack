"""Run a fresh native forward P0 pass on one clip's per-frame post-detector-NMS outputs."""
import argparse
import hashlib
import io
import json
from numbers import Integral
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np

from beetrack.common import write_mot
from beetrack.link import model
from tools.study104.track_reverse import STRICT_SHA256, track_arms


_TRACK_RUN = Lock()  # Ultralytics' identity counter and shared Kalman state are process-global.


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class StrictLinker:
    """Read weights once and pin precisely the bytes used to construct the linker probability."""
    def __init__(self, weights_path):
        raw=Path(weights_path).read_bytes()
        self.sha256=sha(raw)
        if self.sha256!=STRICT_SHA256:
            raise ValueError('Expected the original m_dev_strict.json byte fingerprint')
        self.probability=model(json.loads(raw))


def tracker_input(dets, nframes):
    if isinstance(nframes,bool) or not isinstance(nframes,Integral) or nframes<1:
        raise ValueError('nframes must be a positive integer on the native clip clock')
    file_sha=None
    if isinstance(dets,(str,os.PathLike)):
        raw=Path(dets).read_bytes()
        file_sha=sha(raw)
        dets=np.load(io.BytesIO(raw),allow_pickle=False)
    arr=np.asarray(dets)
    if arr.ndim!=2 or arr.shape[1] not in (6,7) or arr.dtype.kind not in 'fiu' or not np.isfinite(arr).all():
        raise ValueError('Expected a finite detections matrix [1-based frame,xyxy,score[,bee class]]')
    f=arr[:,0]
    if (np.any(f<1) or np.any(f>nframes) or not np.array_equal(f,np.floor(f))
            or np.any(arr[:,3:5]<arr[:,1:3]) or np.any((arr[:,5]<0)|(arr[:,5]>1))):
        raise ValueError('Invalid detection frame, geometry or confidence')
    if arr.shape[1]==7 and np.any(arr[:,6]!=0):
        raise ValueError('The registered detector has one class (bee=0)')
    # The historical six columns and row order are passed intact. Tracker.update casts the five
    # box/confidence fields to float32 at the native entry point before applying its own 0.5 NMS.
    native=arr[:,:6]
    effective=np.ascontiguousarray(native,dtype='<f4')
    return native,dict(nframes=int(nframes),input_shape=list(arr.shape),input_dtype=arr.dtype.str,
                       input_rows_sha256=sha(np.ascontiguousarray(arr).tobytes()),input_file_sha256=file_sha,
                       tracker_rows_f32_sha256=sha(effective.tobytes()))


def track_detections(dets, nframes, weights_path, output_path):
    """Write canonical P0 MOT and return input/output fingerprints. No detection or tracker results are cached.

    `dets` is one original-order numeric Nx6/Nx7 matrix or an `.npy` file. Empty frames are implicit;
    use an empty (0,6)/(0,7) matrix for an all-empty clip. `weights_path` may be a verified StrictLinker.
    """
    if isinstance(dets,(str,os.PathLike)) and Path(dets).resolve()==Path(output_path).resolve():
        raise ValueError('Detection input and MOT output must have different paths')
    if not isinstance(weights_path,StrictLinker) and Path(weights_path).resolve()==Path(output_path).resolve():
        raise ValueError('Strict weights and MOT output must have different paths')
    native,receipt=tracker_input(dets,nframes)
    linker=weights_path if isinstance(weights_path,StrictLinker) else StrictLinker(weights_path)
    with _TRACK_RUN:
        arm, rows=next(track_arms(native,int(nframes),linker.probability))
    if arm!='F':
        raise ValueError('Native tracker did not return the forward P0 arm')
    path=Path(output_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        candidate=Path(tmp)/'native.txt'
        write_mot(candidate,rows)
        raw=candidate.read_bytes()
        try:
            os.link(candidate,path)
        except FileExistsError:
            if path.read_bytes()!=raw:
                raise ValueError(f'MOT target already has different bytes: {path}')
    return dict(**receipt,linker_sha256=linker.sha256,arm=arm,mot_rows=len(rows),
                mot_sha256=sha(raw),mot_lf_sha256=sha(raw.replace(b'\r\n',b'\n')))


def self_check():
    from unittest.mock import patch

    from beetrack.link import link
    from beetrack.track import Tracker,drop_short,run_sequence

    # Unsorted frame order, equal-score overlapping boxes, a very low score and leading/trailing empty frames.
    cache=np.asarray([[f,x,3,x+9,13,s] for f in (3,2,4,6,7,8,9)
                     for x,s in ((4,.8),(5,.8),(46,.8),(75,.051))],dtype=np.float32)
    new=np.column_stack((cache,np.zeros(len(cache),np.float32)))
    n=12
    with TemporaryDirectory() as tmp:
        tmp=Path(tmp); weight=tmp/'fixture.json'
        # Mock only the identity of the explicitly generated toy model. Never read the real linker.
        from beetrack.link import features
        from beetrack.link import summarize
        toy_rows=np.asarray([[1,1,1,1,2,2,.8],[2,1,1,1,2,2,.8],[4,2,2,2,2,2,.8],[5,2,2,2,2,2,.8]])
        _,summary=summarize(toy_rows)
        width=features(summary,np.array([0]),np.array([1])).shape[1]
        toy=dict(mu=[0]*width,sd=[1]*width,layers=[([[0.0]]*width,[2.0])])
        raw=json.dumps(toy).encode();weight.write_bytes(raw)
        with patch.dict(globals(),STRICT_SHA256=sha(raw)):
            handle=StrictLinker(weight)
            # Independent composition of native §93/104 first-pass, link and MOT serialization.
            baseline,_=run_sequence(cache,n,Tracker('link',link=0,min_len=0))
            ref=drop_short(sorted(link(baseline,.15,handle.probability).tolist()),3,0.)
            refpath=tmp/'expected.txt';write_mot(refpath,ref)
            expected=refpath.read_bytes()
            assert len(ref)>0 and max(r[0] for r in ref)<n
            previous=None
            for i,source in enumerate((cache,cache.astype(np.float64),new)):
                p=tmp/f'result_{i}.txt';got=track_detections(source,n,handle,p)
                assert p.read_bytes()==expected and got['mot_sha256']==sha(expected)
                if previous:
                    assert got['tracker_rows_f32_sha256']==previous['tracker_rows_f32_sha256']
                previous=got
            # IoU at 0.5 and a sub-FP32 perturbation must be resolved by native Tracker.update.
            edge=np.array([[f,x,4,x+6,10,.8] for f in (2,3,4,5)
                           for x in (120.,122.+1e-7,160.,162.-1e-7)],dtype=np.float64)
            supplied,_=tracker_input(edge,n)
            assert supplied.dtype==edge.dtype and np.array_equal(supplied,edge)
            _,edge_ref=next(track_arms(edge,n,handle.probability))
            edge_path=tmp/'edge-ref.txt';write_mot(edge_path,edge_ref)
            edge_receipt=track_detections(edge,n,handle,tmp/'edge-bridge.txt')
            assert edge_receipt['mot_sha256']==sha(edge_path.read_bytes())
            fullpath=tmp/'detections.npy';np.save(fullpath,new,allow_pickle=False)
            fromfile=track_detections(fullpath,n,weight,tmp/'result_2.txt')
            assert fromfile['input_file_sha256']==sha(fullpath.read_bytes()) and fromfile['mot_sha256']==sha(expected)
            empty=track_detections(np.empty((0,7),np.float32),n,handle,tmp/'empty.txt')
            assert empty['mot_rows']==0 and (tmp/'empty.txt').read_bytes()==b''
            try:
                track_detections(new[:0],n,handle,tmp/'result_2.txt')
            except ValueError as e:
                assert 'different bytes' in str(e)
            else:
                raise AssertionError('Conflicting MOT overwrite was accepted')
            weight.write_bytes(raw+b' ')
            try:
                track_detections(new,n,weight,tmp/'mismatch.txt')
            except ValueError as e:
                assert 'fingerprint' in str(e)
            else:
                raise AssertionError('A previously verified path accepted different weights')
    print('Synthetic 6/7-column P0, explicit native composition, fresh handles, checksum conflicts, empty frames and exact MOT bytes passed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='mode',required=True)
    modes.add_parser('self-check')
    cmd=modes.add_parser('track')
    cmd.add_argument('detections',type=Path);cmd.add_argument('output',type=Path)
    cmd.add_argument('--weights',required=True,type=Path);cmd.add_argument('--nframes',required=True,type=int)
    args=parser.parse_args()
    if args.mode=='self-check':
        self_check()
    else:
        print(json.dumps(track_detections(args.detections,args.nframes,args.weights,args.output),sort_keys=True))
