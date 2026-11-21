"""Source-only natural RGB pair probe; selection seals before any pixel decode.

Only the25 original W/G train clips, official§157/159 polygons, one CPU.
Real execution requires an origin/main ancestor with [paired-rgb-v1] in LOG.
"""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.dont_write_bytecode = True
import cv2
import numpy as np

MARKER = '[paired-rgb-v1]'
SOURCES = {'W': tuple(f'BEE24-{k:02d}' for k in (*range(1, 10), 11, 14)),
           'G': tuple(f'BEE24-{k:02d}' for k in range(17, 33) if k not in (24, 28))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def valid_box(row):
    return bool(np.isfinite(row[2:6]).all() and (row[4:6] > 0).all())


def split_original(gt, nf):
    """Preserve invalid geometry dates; fail whole clip on native/official aliases."""
    G = np.asarray(gt, dtype=float)
    if G.ndim != 2 or G.shape[1] < 6 or not np.isfinite(G[:, :2]).all() or not np.equal(G[:, :2], np.floor(G[:, :2])).all():
        return None, dict(reason='invalid_native_frame_or_ID')
    if (G[:, 0] < 1).any() or (G[:, 0] > nf).any() or (G[:, 1] < 0).any() or (G[:, 1] > 2**53 - 1).any():
        return None, dict(reason='invalid_native_clock_or_ID_range')
    unique, count = np.unique(G[:, :2], axis=0, return_counts=True)
    if (count > 1).any():
        return None, dict(reason='duplicate_native_ID_date', duplicates=unique[count > 1].astype(int).tolist())
    runs, aliases = {}, {}
    for identity in sorted(set(G[:, 1].astype(int))):
        indices = np.flatnonzero(G[:, 1] == identity)
        indices = indices[np.argsort(G[indices, 0], kind='stable')]
        cuts = np.flatnonzero(np.diff(G[indices, 0]) > 25) + 1
        for run, chosen in enumerate(np.split(indices, cuts)):
            alias = int(identity) + run * 100000
            key = (int(identity), run)
            if alias > 2**53 - 1 or alias in aliases:
                return None, dict(reason='official_run_alias_ambiguity', alias=alias,
                                  first=list(aliases[alias]) if alias in aliases else None, second=list(key))
            aliases[alias] = key
            runs[key] = dict(rows=G[chosen, :6], original_row_indices=chosen.tolist(), official_alias=alias)
    return runs, None


def transitions(rows, polygon, before, after):
    """Adjacent ORIGINAL observations, both inside window; invalid rows block."""
    poly = np.asarray(polygon, dtype=np.float32)
    changes = []
    for left, right in zip(rows[:-1], rows[1:]):
        if not (before <= left[0] <= after and before <= right[0] <= after and valid_box(left) and valid_box(right)):
            continue
        centre = lambda r: tuple(float(x) for x in r[2:4] + r[4:6] / 2)
        a = cv2.pointPolygonTest(poly, centre(left), False) >= 0
        b = cv2.pointPolygonTest(poly, centre(right), False) >= 0
        if a != b:
            changes.append(dict(left_frame=int(left[0]), frame=int(right[0]), direction='in' if b else 'out'))
    return changes


def select_clip(clip, hive, gt, polygon, nf):
    if hive not in SOURCES or clip not in SOURCES[hive]:
        raise ValueError('only original25 source train clips accepted')
    runs, failure = split_original(gt, nf)
    result = dict(clip=clip, hive=hive, nframes=nf, polygon=polygon, clip_failure=failure,
                  raw_rows=len(gt), selected=[], pairs=[], attrition={})
    if failure:
        return result
    result['runs'] = [dict(original_id=k[0], run=k[1], first=int(v['rows'][0, 0]), last=int(v['rows'][-1, 0]),
                           observations=len(v['rows']), bad_geometry_dates=[int(r[0]) for r in v['rows'] if not valid_box(r)],
                           original_row_indices=v['original_row_indices'], official_alias=v['official_alias']) for k, v in runs.items()]
    qualified = []
    for key1, key2 in combinations(sorted(runs), 2):
        R1, R2 = runs[key1]['rows'], runs[key2]['rows']
        lookup1, lookup2 = ({int(r[0]): r for r in R} for R in (R1, R2))
        item = dict(id1=key1[0], run1=key1[1], id2=key2[0], run2=key2[1])
        closest = None
        for f in sorted(set(lookup1) & set(lookup2)):
            a, b = lookup1[f], lookup2[f]
            if not valid_box(a) or not valid_box(b):
                continue
            scale = (np.sqrt(a[4] * a[5]) + np.sqrt(b[4] * b[5])) / 2
            distance = float(np.linalg.norm(a[2:4] + a[4:6]/2 - b[2:4] - b[4:6]/2) / scale)
            if closest is None or distance < closest[0]:
                closest = (distance, f)
        if closest is None:
            item['state'] = 'no_valid_covisible_date'
        elif closest[0] > 2:
            item.update(state='closest_distance_over2', frame=closest[1], normalized_distance=closest[0])
        else:
            distance, f = closest; before, after = f - 4, f + 4
            item.update(frame=f, normalized_distance=distance, endpoint_frames=[before, after])
            if not all(t in lookup and valid_box(lookup[t]) for lookup in (lookup1, lookup2) for t in (before, after)):
                item['state'] = 'fixed_nearest_date_endpoint_attrition'
            else:
                shift = 25 if 1 <= before + 25 and after + 25 <= nf else -25 if 1 <= before - 25 and after - 25 <= nf else None
                changed = [dict(original_id=k[0], run=k[1], **event)
                           for k, rows in ((key1, R1), (key2, R2)) for event in transitions(rows, polygon, before, after)]
                text = f'paired-rgb-v1|{clip}|{key1[0]}|{key1[1]}|{key2[0]}|{key2[1]}|{f}'
                endpoint_rows = []
                for t in (before, after):
                    for index, lookup in enumerate((lookup1, lookup2)):
                        row = lookup[t].copy(); row[1] = index
                        endpoint_rows.append(row.tolist())
                item.update(state='eligible', clip=clip, hive=hive, hash_input=text,
                            digest=hashlib.sha256(text.encode('ascii')).hexdigest(), endpoint_rows=endpoint_rows,
                            frame_shift=shift, control_frames=[before+shift, after+shift] if shift is not None else None,
                            near_boundary=bool(changed), boundary_transitions=changed)
                qualified.append(item)
        result['pairs'].append(item)
    result['attrition'] = dict(Counter(p['state'] for p in result['pairs']))
    result['eligible_count'] = len(qualified)
    result['selected'] = sorted(qualified, key=lambda p: (p['digest'], p['id1'], p['run1'], p['id2'], p['run2'], p['frame']))[:4]
    return result


def summary(records):
    def counts(items, arm):
        votes = [r[arm]['vote'] for r in items]
        return dict(selected=len(items), usable=sum(r[arm].get('valid',False) for r in items),
                    informative=votes.count(1)+votes.count(-1), videos=len({r['clip'] for r in items}),
                    correct=votes.count(1), wrong=votes.count(-1), abstain=votes.count(0),
                    S=(votes.count(1)-votes.count(-1))/len(items) if items else None)
    def section(items):
        control = [r for r in items if r['frame_shift'] is not None]
        both = [r for r in control if r['primary'].get('valid', False) and r['control'].get('valid', False)]
        p, q = counts(control, 'primary'), counts(control, 'control')
        return dict(primary=counts(items, 'primary'), time_control_intersection=dict(primary=p, control=q,
                     S_difference=p['S']-q['S'] if control else None),
                    both_valid_intersection=dict(primary=counts(both, 'primary'), control=counts(both, 'control')),
                    control_date_unavailable=len(items)-len(control))
    output=dict(overall=section(records),
                hives={h: section([r for r in records if r['hive']==h]) for h in SOURCES},
                clips={clip: section([r for r in records if r['clip']==clip]) for clips in SOURCES.values() for clip in clips},
                boundary_strata={key: section([r for r in records if r['near_boundary']==flag]) for key, flag in (('near', True), ('other', False))},
                hive_boundary_strata={h+'_'+key: section([r for r in records if r['hive']==h and r['near_boundary']==flag])
                                     for h in SOURCES for key,flag in (('near',True),('other',False))})
    output['equal_clip_weight']={}
    main_pass=True;coverage=True
    for hive,clips in SOURCES.items():
        primary=[output['clips'][clip]['primary']['S'] for clip in clips if output['clips'][clip]['primary']['S'] is not None]
        comparable=[output['clips'][clip]['time_control_intersection'] for clip in clips
                    if output['clips'][clip]['time_control_intersection']['primary']['S'] is not None]
        output['equal_clip_weight'][hive]=dict(nonempty_clips=len(primary),primary_S=float(np.mean(primary)) if primary else None,
                                             comparable_clips=len(comparable),
                                             original_S=float(np.mean([c['primary']['S'] for c in comparable])) if comparable else None,
                                             shifted_S=float(np.mean([c['control']['S'] for c in comparable])) if comparable else None)
        p=output['hives'][hive]['primary']; paired=output['hives'][hive]['time_control_intersection']
        covered=p['selected']>=8 and p['videos']>=3
        coverage=coverage and covered
        main_pass=main_pass and covered and p['informative']>=.25*p['selected'] and p['S']>=.20 and paired['S_difference'] is not None and paired['S_difference']>0
    near=output['boundary_strata']['near']['primary']
    output['registered_gates']=dict(primary_coverage=coverage,primary_pass=bool(main_pass),
                                     primary_status='pass' if main_pass else 'not_verified_coverage' if not coverage else 'fail',
                                     boundary_coverage=near['selected']>=8,
                                     boundary_signal_positive=near['S'] is not None and near['S']>0,
                                     boundary_explanation_pass=bool(main_pass and near['selected']>=8 and near['S']>0))
    return output


def registration(repo, prereg):
    if not prereg or not re.fullmatch('[0-9a-f]{40}', prereg):
        raise ValueError('explicit full public preregistration SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',prereg,'refs/remotes/origin/main'],cwd=repo,capture_output=True,check=True)
    text = subprocess.run(['git','show',prereg+':docs/experiment_log.md'],cwd=repo,capture_output=True,check=True).stdout.decode('utf-8')
    if MARKER not in text:
        raise ValueError('missing '+MARKER+' registration')


def run(args):
    repo, out = args.repo.resolve(), args.out.resolve()
    registration(repo,args.prereg)
    if out.exists() or out.drive.upper()!='D:':
        raise ValueError('fresh D: output required')
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'tools'))
    from whiteboard_fold import POLY
    from gray_fold import POLY_G
    import track_choice_features as features
    if not hasattr(features,'evaluate_fixed_pair'):
        raise RuntimeError('fixed-pair evaluator must be available before source selection')
    cv2.setNumThreads(1)
    polygons=POLY|POLY_G
    selected, sources=[],[]
    for hive,clips in SOURCES.items():
        for clip in clips:
            root=repo/'data/BEE24/train'/clip
            paths=sorted((root/'img1').glob('*.jpg'))
            if [p.name for p in paths]!=[f'{i:06d}.jpg' for i in range(1,len(paths)+1)]:
                raise ValueError('noncontiguous train frame roster: '+clip)
            gp=root/'gt/gt.txt'
            G=np.loadtxt(gp,delimiter=',',usecols=range(6),ndmin=2)
            result=select_clip(clip,hive,G,polygons[clip],len(paths))
            result.update(gt_path=gp.relative_to(repo).as_posix(),gt_sha256=sha(gp),polygon_sha256=object_sha(polygons[clip]))
            sources.append(result);selected.extend(result['selected'])
    code_paths=[Path(__file__),Path(features.__file__),repo/'beetrack/zones.json',repo/'tools/whiteboard_fold.py',repo/'tools/gray_fold.py']
    code_hashes={p.name:sha(p) for p in code_paths}
    out.mkdir(parents=True,exist_ok=False)
    selection=dict(schema='paired-rgb-source-selection-v1',prereg=args.prereg,marker=MARKER,sources=sources,
                   selected_count=len(selected),code_hashes=code_hashes,
                   scope='originalW11/G14 train only; no detector/model fitting; one CPU',
                   rule='one fixed minimum valid covisible date per unordered nativeID/run pair; no endpoint fallback; SHA first4 perclip')
    save(out/'selection.json',selection)
    (out/'SELECTION.sha256').write_text(sha(out/'selection.json')+'\n')
    print('SELECTION_SEALED '+str(out/'selection.json')+' '+sha(out/'selection.json'),flush=True)
    image_hashes={}; measurements=[]
    for pick in selected:
        clip=pick['clip']
        def read_rgb(frame,clip=clip):
            path=repo/'data/BEE24/train'/clip/'img1'/f'{int(frame):06d}.jpg'
            relative=path.relative_to(repo).as_posix()
            if not path.is_file():
                image_hashes[relative]=None
                return None
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if relative in image_hashes and image_hashes[relative]!=digest:
                raise ValueError('source RGB changed between fixed pairs')
            image_hashes[relative]=digest
            try:bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            except cv2.error:bgr=None
            return None if bgr is None else bgr[:,:,::-1].copy()
        mot=np.asarray(pick['endpoint_rows'],float)
        before,after=pick['endpoint_frames']
        primary=features.evaluate_fixed_pair(mot,(0,1),before,after,read_rgb)
        shift=pick['frame_shift']
        control=(features.evaluate_fixed_pair(mot,(0,1),before,after,lambda f:read_rgb(f+shift)) if shift is not None
                 else dict(state='control_dates_outside_source',valid=False,informative=False,vote=0))
        measurements.append(dict(**pick,primary=primary,control=control))
    for source in sources:
        assert sha(repo/source['gt_path'])==source['gt_sha256']
    assert all(digest is None or sha(repo/name)==digest for name,digest in image_hashes.items())
    assert all(sha(path)==code_hashes[path.name] for path in code_paths)
    save(out/'measurements.json',dict(schema='paired-rgb-source-measurements-v1',selection_sha256=sha(out/'selection.json'),
                                    pairs=measurements,source_image_sha256=image_hashes))
    save(out/'summary.json',dict(schema='paired-rgb-source-summary-v1',prereg=args.prereg,
                               selection_sha256=sha(out/'selection.json'),measurements_sha256=sha(out/'measurements.json'),
                               results=summary(measurements),pixels_public=False))
    files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    save(out/'manifest.json',files)
    (out/'SEALED.sha256').write_text(sha(out/'manifest.json')+'\n')
    print('ALL_SEALED '+str(out)+' '+sha(out/'manifest.json'),flush=True)


def self_test():
    poly=[[0,0],[30,0],[30,30],[0,30]]
    rows=[]
    for f in range(1,76):
        rows.extend([[f,1,10,5,4,4],[f,2,15+abs(f-35),5,4,4]])
    G=np.asarray(rows,float)
    a=select_clip('BEE24-01','W',G,poly,75)
    assert len(a['selected'])==1 and a['selected'][0]['frame']==35
    p=a['selected'][0]
    assert p['endpoint_frames']==[31,39] and p['frame_shift']==25 and p['hash_input']=='paired-rgb-v1|BEE24-01|1|0|2|0|35'
    reversed_input=select_clip('BEE24-01','W',G[::-1],poly,75)
    assert reversed_input['selected']==a['selected'] and reversed_input['attrition']==a['attrition']
    absent=G[~((G[:,0]==31)&(G[:,1]==1))]
    b=select_clip('BEE24-01','W',absent,poly,75)
    assert not b['selected'] and b['attrition']['fixed_nearest_date_endpoint_attrition']==1
    bad=G.copy();bad[(bad[:,0]==35)&(bad[:,1]==2),4]=-1
    c=select_clip('BEE24-01','W',bad,poly,75)
    assert c['selected'][0]['frame']==34  # earliest exact-distance tie, invalid date retained
    duplicated=np.vstack((G,G[0]))
    assert select_clip('BEE24-01','W',duplicated,poly,75)['clip_failure']['reason']=='duplicate_native_ID_date'
    aliases=np.asarray([[1,1,0,0,2,2],[40,1,0,0,2,2],[2,100001,0,0,2,2]],float)
    assert split_original(aliases,75)[1]['reason']=='official_run_alias_ambiguity'
    # Invalid intervening observation blocks crossing; outside-window edges excluded.
    crossing=np.asarray([[30,1,35,5,2,2],[31,1,20,5,2,2],[32,1,35,5,-1,2],[33,1,20,5,2,2]],float)
    assert transitions(crossing,poly,31,33)==[]
    assert transitions(crossing,poly,30,33)==[dict(left_frame=30,frame=31,direction='in')]
    with TemporaryDirectory(dir='D:/beetrack_runs',prefix='rgb-source-synthetic-') as temp:
        path=Path(temp)/'selection.json';save(path,a);assert json.loads(path.read_text())==a
    print('Synthetic native-run/alias/bad-date/nearest-no-fallback/endpoint/SHA/time-shift/boundary/source25 checks passed')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--prereg')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test()
    else:
        if not args.repo or not args.out or not args.prereg:ap.error('require --repo --out --prereg')
        run(args)
