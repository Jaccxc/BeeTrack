"""Controlled two-true-track identity switch, not detector performance.

MOT/GT arrays are float64 Nx6: frame,id,x,y,width,height (native absolute
frames). Only the frozen pair's complete original runs are retained, remapped
to1/2; swapping IDs on rows strictly AFTER contact preserves every date/box.
Field input wiring awaits the separately published pair-texture protocol.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.dont_write_bytecode=True
import numpy as np


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def array_sha(value):
    a=np.ascontiguousarray(value,dtype='<f8')
    return hashlib.sha256(json.dumps(list(a.shape),separators=(',',':')).encode()+b'\n'+a.tobytes()).hexdigest()


def two_runs(original_gt,pair,nframes):
    from tools.paired_rgb_source.track_choice_source_probe import split_original,SOURCES,valid_box
    clip=pair['clip']
    if not any(clip in clips for clips in SOURCES.values()):raise ValueError('only original25train source clips allowed')
    fields=('id1','run1','id2','run2','frame')
    if any(type(pair[k]) is not int for k in fields):raise ValueError('frozen native IDs/runs/date must be integers')
    keys=((pair['id1'],pair['run1']),(pair['id2'],pair['run2']))
    if keys[0]>=keys[1] or not 1<=pair['frame']<=nframes:raise ValueError('canonical unordered pair order/native date required')
    gt=np.asarray(original_gt,dtype=np.float64)
    runs,failure=split_original(gt,nframes)
    if failure:raise ValueError('original clip identity ambiguity: '+json.dumps(failure,sort_keys=True))
    if any(key not in runs for key in keys):raise ValueError('frozen original run not present')
    positions={}
    for target,key in enumerate(keys,1):
        rows=runs[key]['rows']
        at=rows[rows[:,0]==pair['frame']]
        if len(at)!=1 or not valid_box(at[0]):raise ValueError('frozen contact no longer has both positive source boxes')
        for index in runs[key]['original_row_indices']:
            if index in positions:raise ValueError('pair runs overlap original row membership')
            positions[int(index)]=target
    indices=sorted(positions)
    identity=gt[indices,:6].copy()
    identity[:,1]=[positions[i] for i in indices]
    cross=identity.copy()
    later=cross[:,0]>pair['frame']
    cross[later,1]=3-cross[later,1]
    assert np.array_equal(identity[:,[0,2,3,4,5]],cross[:,[0,2,3,4,5]])
    return identity,cross,dict(original_row_indices=indices,
                               mappings=[dict(native_id=k[0],run=k[1],remapped_id=i) for i,k in enumerate(keys,1)],
                               native_format='float64 Nx6 frame,id,x,y,width,height',
                               original_gt_sha256=array_sha(gt[:,:6]),identity_rows_sha256=array_sha(identity),
                               cross_rows_sha256=array_sha(cross),switch='ID1<->ID2 iff frame>contact; no box/date alteration')


def score_arm(truth,prediction,polygon,nframes):
    from tools.study193.source193 import score_native
    from beetrack.evaluate import match_events
    value=score_native(truth,prediction,polygon,nframes)
    official=match_events(truth,prediction,polygon)
    # N is ONE stream's sum over all30 offsets; no3-member scaling.
    N=int(value['window']['numerator'])
    gt_count=int(value['nG']);pred_count=int(value['nP'])
    iou={}
    for key in ('0.5','0.3'):
        record=value['iou'][key]
        iou[key]=dict(tp=int(record['tp']),fp=int(record['fp']),fn=int(record['fn']),f1=float(record['f1']),
                      matching=record['matching'],by_direction=record['by_direction'])
    assert official['tp']==iou['0.5']['tp']
    return dict(nGT=gt_count,nPred=pred_count,TP=iou['0.5']['tp'],TP_ID=int(official['tp_id']),
                N=N,window_denominator=30*gt_count,W_percent=100*N/(30*gt_count) if gt_count else None,
                iou=iou,window=value['window'],integer_bins=value['integer_bins'],
                signed_GT_events=value['evidence']['gt'],signed_prediction_events=value['evidence']['predictions'],
                legal_graph=value['evidence'],prediction_rows_sha256=array_sha(prediction),
                zero_GT=gt_count==0)


def score_pair(original_gt,pair,polygon,nframes):
    truth,cross,provenance=two_runs(original_gt,pair,nframes)
    original=score_arm(truth,truth,polygon,nframes)
    switched=score_arm(truth,cross,polygon,nframes)
    identity_ok=original['N']==0 and all(original['iou'][key]['tp']==original['nGT'] for key in ('0.5','0.3'))
    return dict(pair={k:pair[k] for k in ('clip','id1','run1','id2','run2','frame')},
                key=pair['clip']+'|'+str(pair.get('digest','')),hive=pair.get('hive'),subset=pair.get('subset'),
                digest=pair.get('digest'),nframes=nframes,polygon=polygon,provenance=provenance,
                identity=original,cross=switched,
                identity_official_check=dict(passed=identity_ok,expected='TP==GT at.5/.3 and N==0, including zeroGT',
                                             unexpected_official_mismatch=not identity_ok),
                changed=switched['N']!=original['N'] or any(switched['iou'][key]['tp']!=original['iou'][key]['tp'] or
                         switched['nPred']!=original['nPred'] for key in ('0.5','0.3')),
                interpretation='two original truth runs controlled identity-switch only; never a detector result')


def score_batch(pairs,gt_by_clip,polygons,nframes_by_clip):
    """Explicit in-memory source boundary; each pair is scored independently."""
    return [score_pair(gt_by_clip[p['clip']],p,polygons[p['clip']],nframes_by_clip[p['clip']]) for p in pairs]


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bundle(root):
    root=Path(root).resolve()
    digest=file_sha(root/'manifest.json')
    if (root/'SEALED.sha256').read_text().strip()!=digest:raise ValueError('bundle top seal mismatch')
    for name,expected in json.loads((root/'manifest.json').read_text()).items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or file_sha(path)!=expected:raise ValueError('bundle file hash mismatch')
    return digest


def field_batch(args):
    if not args.prereg or not re.fullmatch('[0-9a-f]{40}',args.prereg):raise ValueError('full preregistration SHA required')
    repo=args.repo.resolve();sys.path.insert(0,str(repo))
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'refs/remotes/origin/main'],cwd=repo,capture_output=True,check=True)
    text=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=repo,capture_output=True,check=True).stdout.decode('utf-8')
    if '[pair-texture-v1]' not in text:raise ValueError('pair-texture-v1 registration missing')
    rosterpath=args.roster.resolve();features_root=rosterpath.parent
    features_seal=verify_bundle(features_root)
    if (features_root/'ROSTER.sha256').read_text().strip()!=file_sha(rosterpath):raise ValueError('prepixel roster seal mismatch')
    prediction_root=args.predictions_dir.resolve();predictions_seal=verify_bundle(prediction_root)
    import model
    model.verify_prediction_seal(prediction_root)
    prediction=json.loads((prediction_root/'predictions.json').read_text())
    roster=json.loads(rosterpath.read_text())
    if prediction['prereg']!=args.prereg or roster['prereg']!=args.prereg:raise ValueError('preregistration mismatch')
    if prediction['features_sha256']!=file_sha(features_root/'features.json'):raise ValueError('predictions do not bind this feature freeze')
    new_pairs=[r for r in roster['pairs'] if r['subset']=='new']
    if len(new_pairs)!=191:raise ValueError('registered new191 frozen pair roster required')
    chosen_keys={r['clip']+'|'+r['digest'] for r in new_pairs}
    predicted_keys={r.get('key',r['clip']+'|'+r['digest']) for r in prediction['pairs']}
    if chosen_keys!=predicted_keys or len(prediction['pairs'])!=len(new_pairs):raise ValueError('all and only new191 choices must be frozen before physical GT scoring')
    out=args.out.resolve()
    if out.exists():raise ValueError('fresh physical output required')
    from tools.paired_rgb_source.track_choice_source_probe import SOURCES
    source_info={r['clip']:r for r in roster['sources']}
    # GT opens only after both complete feature and prediction seals above.
    gt_arrays={};source_proofs={}
    for clip in sorted({p['clip'] for p in new_pairs}):
        source=source_info[clip]
        if source['hive'] not in SOURCES or clip not in SOURCES[source['hive']]:raise ValueError('source clip outside25train allowlist')
        expected=f'data/BEE24/train/{clip}/gt/gt.txt'
        if source['gt_path']!=expected:raise ValueError('source GT path must be original train')
        path=repo/expected
        if file_sha(path)!=source['gt_sha256']:raise ValueError('original source GT changed')
        gt_arrays[clip]=np.loadtxt(path,delimiter=',',usecols=range(6),ndmin=2)
        source_proofs[clip]=dict(gt_path=expected,gt_sha256=file_sha(path),nframes=source['nframes'],polygon=source['polygon'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out.parent,prefix='.pair-physical-') as tmp:
        work=Path(tmp)/'payload';work.mkdir()
        results={}
        for pair in new_pairs:
            source=source_info[pair['clip']]
            key=pair['clip']+'|'+pair['digest']
            if key in results:raise ValueError('duplicate fixed pair key')
            results[key]=score_pair(gt_arrays[pair['clip']],pair,source['polygon'],source['nframes'])
        for clip,source in source_proofs.items():
            if file_sha(repo/source['gt_path'])!=source['gt_sha256']:raise ValueError('source GT changed during scoring')
        payload=dict(schema='pair-texture-physical-v1',prereg=args.prereg,results=results,
                     input_receipts=dict(roster_sha256=file_sha(rosterpath),features_manifest_sha256=features_seal,
                                         features_sha256=file_sha(features_root/'features.json'),predictions_manifest_sha256=predictions_seal,
                                         predictions_sha256=file_sha(prediction_root/'predictions.json')),
                     sources=source_proofs,code_sha256={Path(__file__).name:file_sha(__file__)},
                     interpretation='separate controlled two-GT-run truth/swap; not complete-scene or detector scoring')
        (work/'physical.json').write_text(json.dumps(plain(payload),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
        manifest={'physical.json':file_sha(work/'physical.json')}
        (work/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        (work/'SEALED.sha256').write_text(file_sha(work/'manifest.json')+'\n')
        if out.exists():raise FileExistsError(out)
        work.rename(out)
    print('PHYSICAL_SEALED '+str(out)+' '+file_sha(out/'manifest.json'),flush=True)


def self_test(repo):
    sys.path.insert(0,str(repo.resolve()))
    poly=[[0,0],[10,0],[10,10],[0,10]]
    def rows(a,b):
        return np.asarray([[f,identity,x-1,4,2,2] for f,(x,y) in enumerate(zip(a,b),1)
                           for identity,x in ((7,x),(9,y))],float)
    pair=dict(clip='BEE24-01',id1=7,run1=0,id2=9,run2=0,frame=2)
    double=rows([12,11,9,8,7],[8,9,11,12,13])
    result=score_pair(double,pair,poly,75)
    assert result['identity_official_check']['passed'] and result['identity']['nGT']==2
    assert result['cross']['nPred']==0 and result['cross']['N']==60 and result['changed']
    # Whole-source tracks are both nonevents; zeroGT keeps N and false positives.
    quiet=rows([12,12,13,12,12],[16,15,16,15,16])
    result=score_pair(quiet,pair,poly,75)
    assert result['identity_official_check']['passed'] and result['identity']['nGT']==0
    assert result['cross']['N']==0 and result['cross']['W_percent'] is None
    zero_with_fp=rows([12,11,12,11,12],[8,9,8,9,8])
    result=score_pair(zero_with_fp,pair,poly,75)
    assert result['identity']['nGT']==0 and result['cross']['nPred']==2
    assert result['cross']['N']==60 and all(result['cross']['iou'][t]['fp']==2 for t in ('0.5','0.3'))
    # An identity exchange of two equal-direction paths can be count/F1 tied.
    same=rows([12,11,9,8,7],[15,14,9,6,3])
    result=score_pair(same,pair,poly,75)
    assert result['identity_official_check']['passed']
    assert result['cross']['N']==0
    # Invalid geometry remains in its native date, never silently repairs GT.
    _,cross,proof=two_runs(double,pair,75)
    assert proof['original_row_indices']==list(range(len(double)))
    assert np.array_equal(cross[double[:,0]<=2,1],np.where(double[double[:,0]<=2,1]==7,1,2))
    print('Synthetic doublecrossing/zeroGT/identityTP==GT/N0/tiedcounts/native-row preservation passed')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--roster',type=Path);ap.add_argument('--predictions-dir',type=Path)
    ap.add_argument('--out',type=Path);ap.add_argument('--prereg')
    args=ap.parse_args()
    if args.self_test:self_test(args.repo)
    else:
        if any(getattr(args,k) is None for k in ('roster','predictions_dir','out','prereg')):
            ap.error('batch requires --roster --predictions-dir --out --prereg')
        field_batch(args)
