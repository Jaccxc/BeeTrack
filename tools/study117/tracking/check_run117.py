"""Synthetic end-to-end test of §117. This command creates its own weights, inputs and GT."""
import argparse
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import p0_117
import run117
import score117
from beetrack.common import write_mot
from beetrack.link import summarize, features


def arms(fold, seed, arm):
    if arm=='A':
        return ('in','out') if fold==33 or seed<2 else ('in',)
    return {'U':('in',),'Z':('in','out') if fold==35 else ('in',),
            'Q':('out',),'Zm':(),'D':('in','out')}[arm]


def fixture(root):
    prereg='a'*40
    runs=root/'runs'; zones=root/'zones.json'; weights=root/'synthetic-link.json'; archive=root/'history.json.gz'
    folds=(33,35); dirs=('in','out'); files={}; gt={}; polygons={}
    for fold in folds:
        clip=run117.clip_name(fold)
        edge=20 if fold==33 else 28
        polygons[clip]=[[edge,0],[edge+35,0],[edge+35,40],[edge,40]]
        def bee(direction):
            start,tid,y=(100,14,4) if direction=='in' else (770,27,22)
            offsets=list(range(-8,1)) if direction=='in' else list(range(0,-9,-1))
            return [[start+i,tid,edge+x,y,8,6,.875] for i,x in enumerate(offsets)]
        gt[clip]=np.asarray(bee('in')+bee('out'),float)
        write_mot(runs/'gt'/'val'/clip/'gt'/'gt.txt',gt[clip])
        for seed in range(3):
            case_files={}
            for arm in run117.ARMS:
                det=[]
                for direction in arms(fold,seed,arm):
                    for f,tid,x,y,w,h,score in bee(direction):
                        det.append([f,x,y,x+w,y+h,score])
                # Non-event observations remain competitors in every arm, including the arm with zero crossings.
                det.extend([[f,70,4,78,10,.625] for f in range(500,504)])
                name=run117.expected_npy(fold,seed,arm);path=root/name
                path.parent.mkdir(exist_ok=True,parents=True)
                mat=np.asarray(sorted(det),dtype=np.float32).astype(np.float64)
                np.save(path,mat,allow_pickle=False)
                case_files[name]=run117.file_sha(path)
            receipt=f'cases/{fold}_{seed}.synthetic';(root/receipt).parent.mkdir(exist_ok=True)
            (root/receipt).write_text('Synthetic fixture only',encoding='ascii')
            case_files[receipt]=run117.file_sha(root/receipt)
            name=f'cases/{fold}_{seed}.json'
            meta=run117.write_once(root/name,dict(study=117,prereg=prereg,fold=fold,seed=seed,frames=1000,files=case_files))
            files.update(case_files);files[name]=meta
    zone_sha=run117.write_once(zones,polygons)
    assert len(files)==48
    detector_sha=run117.write_once(root/'detector_done.json',dict(study=117,prereg=prereg,files=files,backbone_forwards=6000))
    # Valid small learned-linker schema and a constant probability, without any trained weights.
    _,tracklets=summarize(np.asarray([[1,1,1,1,4,4,.8],[2,1,1,1,4,4,.8],
                                       [7,2,4,2,4,4,.8],[8,2,4,2,4,4,.8]],float))
    d=features(tracklets,np.array([0]),np.array([1])).shape[1]
    strict_sha=run117.write_once(weights,dict(mu=[0]*d,sd=[1]*d,layers=[([[0]]*d,[2])]))
    history=dict(masks=dict(role='source',predictions={},prediction_sha256={}),labels={})
    for fold in folds:
        clip=run117.clip_name(fold);gt_sha=run117.file_sha(runs/'gt'/'val'/clip/'gt'/'gt.txt')
        different=np.array([[f,1,76,4,8,6] for f in range(500,504)],float)
        old=score117.score_case(gt[clip][:,:6],different,polygons[clip])['evidence']
        assert old['format']==score117.FORMAT and old['predictions']==[] and len(old['gt'])==len(dirs)
        for seed in range(3):
            key=f'{fold}_{seed}_tail';mock_sha='f'*64
            history['masks']['prediction_sha256'][key]=mock_sha
            history['masks']['predictions'][key]=dict(clip=clip,sources={'beetrack/zones.json':zone_sha},
                                                       mot_sha256='e'*64,predictions=old['predictions'])
            history['labels'][key]=dict(gt_mot_sha256=gt_sha,predictions_sha256=mock_sha,
                                        **{k:old[k] for k in ('gt','edges05','edges03')})
    archive.write_bytes(gzip.compress(json.dumps(history,sort_keys=True).encode(),mtime=0))
    return prereg,runs,zones,weights,strict_sha,archive,detector_sha


def expect_failure(call,error,fragment):
    try:
        call()
    except error as problem:
        assert fragment in str(problem), str(problem)
    else:
        raise AssertionError(f'Expected {error.__name__} containing {fragment!r}')


def self_check():
    with TemporaryDirectory() as temp:
        root=Path(temp);reg,runs,zones,weights,strict_sha,archive,det_sha=fixture(root)
        fresh=lambda folds,seeds,final:run117.track(root,reg,det_sha,zones,weights,folds,seeds,final)
        with patch.object(p0_117,'STRICT_SHA256',strict_sha), redirect_stdout(io.StringIO()):
            assert fresh([33],[1],False) is None
            assert not (root/'tracking/index.json').exists() and len(list((root/'tracking/mot').rglob('*.txt')))==6
            track_sha=fresh((33,35),range(3),True)
            with patch('p0_117.track_detections', side_effect=AssertionError('A frozen P0 arm unexpectedly retracked')):
                assert fresh((33,35),range(3),True)==track_sha
        index=run117.load_checked(root/'tracking/index.json',track_sha)
        assert len(index['cases'])==36 and index['stage']=='predictions_only' and index['detector_index_sha256']==det_sha
        assert [(x['fold'],x['seed'],x['arm']) for x in index['cases']]==run117.schedule()
        for c in index['cases']:
            row=run117.load_checked(root/'tracking'/c['file'],c['sha256'])
            receipt=row['tracking_receipt']
            assert receipt['linker_sha256']==strict_sha and receipt['input_file_sha256']==row['det_sha256']
            assert receipt['mot_sha256']==c['mot_sha256'] and receipt['input_dtype']=='<f8'
            assert row['totals']=={d:int(d in arms(c['fold'],c['seed'],c['arm'])) for d in ('in','out')}
        hist_sha=run117.file_sha(archive)
        def score(hash_, hist=archive, raw_runs=runs):
            return run117.score(root,reg,hash_,raw_runs,zones,hist,history_sha=hist_sha)
        absent=root/'unread-and-absent'
        expect_failure(lambda:score('0'*64,absent,absent),ValueError,'index.json')
        prior=index['cases'][-1];mot=root/'tracking'/prior['mot_file']; saved=mot.read_bytes(); mot.write_bytes(saved+b'\n')
        try:
            expect_failure(lambda:score(track_sha,absent,absent),ValueError,'provenance changed')
        finally:
            mot.write_bytes(saved)
        assert not (root/'scores').exists()
        bad=archive.with_suffix('.wrong');bad.write_bytes(archive.read_bytes()+b'x')
        expect_failure(lambda:score(track_sha,bad,absent),ValueError,str(bad))
        final=score(track_sha); assert final==score(track_sha)
        report=run117.load_checked(root/'scores/index.json',final)
        summary=run117.load_checked(root/'scores'/report['summary_file'],report['summary_sha256'])
        assert report['tracking_index_sha256']==summary['tracking_index_sha256']==track_sha and len(report['cases'])==36
        assert [(c['fold'],c['seed'],c['arm']) for c in report['cases']]==run117.schedule()
        for c in report['cases']:
            result=run117.load_checked(root/'scores'/c['file'],c['sha256'])
            assert result['gt_counts']=={'in':1,'out':1}
            for d in ('in','out'):
                expected=int(d in arms(c['fold'],c['seed'],c['arm']))
                assert result['pred_counts'][d]==expected
                for threshold in ('0.5','0.3'):
                    line=result['iou'][threshold]['by_direction'][d]
                    assert (line['tp'],line['fp'],line['fn'])==(expected,0,1-expected)
        assert summary['historical_gpu_fp16']['archive_sha256']==hist_sha
        historical=summary['historical_gpu_fp16']['cases']
        assert len(historical)==6 and all(h['pred_counts']=={'in':0,'out':0} and h['w30_percent']==100 for h in historical.values())
        for f in (33,35):
            res=summary['folds'][str(f)]
            assert res['means']['U']['w']==50 and len(res['against_new_cpu_u'])==18
            assert res['utility']['passed'] and res['utility']['positive_seeds']==(3 if f==33 else 2)
            assert res['complexity']['passed']==(f==33)
            assert all(item['w_reduction_pp']==0 for item in res['against_new_cpu_u'] if item['arm']=='U')
    print('PASS: generated 36 synthetic 1000-frame float64 inputs; actual P0, partial resume, indexed MOTs, '
          'two-threshold directions, both summary gates, distinct historical benchmark and pre-GT integrity checks.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['self-check']);parser.parse_args()
    self_check()
