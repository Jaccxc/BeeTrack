"""Independent natural train two-bee RGB source diagnostic. Synthetic-only by default.

Never imports production sampling or RGB measurement functions.
"""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name]='1'
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from collections import Counter,OrderedDict
import numpy as np
import cv2
cv2.setNumThreads(1)

MARKER='paired-rgb-v1'


def split_original(raw,nf=None):
    """Invalid rectangles retain their observation/date and may not bridge gaps."""
    a=np.asarray(raw,float)
    if a.ndim!=2 or a.shape[1]<6:raise ValueError('expected original Nx6+ GT')
    if not np.isfinite(a[:,:2]).all() or np.any(a[:,:2]!=np.floor(a[:,:2])) or np.any(a[:,0]<1) or np.any(a[:,1]<0) or np.any(a[:,1]>2**53-1):raise ValueError('invalid date/ID')
    if nf is not None and np.any(a[:,0]>nf):raise ValueError('invalid clock')
    if len(set(map(tuple,a[:,:2])))!=len(a):raise ValueError('duplicate(frame,id)')
    result=[];aliases=set()
    for identity in sorted(set(int(i) for i in a[:,1])):
        indices=np.flatnonzero(a[:,1]==identity);indices=indices[np.argsort(a[indices,0],kind='stable')];rows=a[indices]
        cuts=[0,*list(np.flatnonzero(np.diff(rows[:,0])>25)+1),len(rows)]
        for run,(lo,hi) in enumerate(zip(cuts,cuts[1:])):
            alias=identity+100000*run
            if alias in aliases or alias>2**53-1:raise ValueError('official alias collision')
            aliases.add(alias);rr=rows[lo:hi,:6]
            result.append(dict(original_id=identity,run=run,official_alias=alias,original_row_indices=indices[lo:hi].tolist(),rows={int(r[0]):r for r in rr},
                valid={int(r[0]):bool(np.isfinite(r[2:6]).all() and (r[4:6]>0).all()) for r in rr}))
    return result


def selected_pairs(raw,clip,nf=None):
    runs=split_original(raw,nf);close=[];eligible=[];allpairs=[]
    for i,left in enumerate(runs):
        for j in range(i+1,len(runs)):
            right=runs[j];best=None
            for t in sorted(set(left['rows'])&set(right['rows'])):
                if not left['valid'][t] or not right['valid'][t]:continue
                a,b=left['rows'][t],right['rows'][t]
                dx=float(a[2]+a[4]/2-b[2]-b[4]/2);dy=float(a[3]+a[5]/2-b[3]-b[5]/2)
                scale=(math.sqrt(float(a[4]*a[5]))+math.sqrt(float(b[4]*b[5])))/2
                distance=math.sqrt(dx*dx+dy*dy)/scale
                if best is None or distance<best[0]:best=(distance,t)
            base=dict(left=i,right=j,original_ids=[left['original_id'],right['original_id']],runs=[left['run'],right['run']])
            if best is None:
                allpairs.append(dict(base,state='no_valid_covisible_date'));continue
            if best[0]>2:
                allpairs.append(dict(base,state='closest_distance_over2',frame=best[1],distance=best[0]));continue
            distance,t=best;pair=dict(left=i,right=j,frame=t,distance=distance,
                original_ids=[left['original_id'],right['original_id']],runs=[left['run'],right['run']])
            close.append(pair)
            if not all(date in r['rows'] and r['valid'][date] for r in (left,right) for date in (t-4,t+4)):
                allpairs.append(dict(pair,state='fixed_nearest_date_endpoint_attrition'));continue
            string=f"{MARKER}|{clip}|{left['original_id']}|{left['run']}|{right['original_id']}|{right['run']}|{t}"
            pair=dict(pair,sha256=hashlib.sha256(string.encode('ascii')).hexdigest(),hash_input=string)
            eligible.append(pair)
            allpairs.append(dict(pair,state='eligible'))
    eligible.sort(key=lambda p:p['sha256'])
    return dict(runs=runs,pairs=allpairs,close=close,eligible=eligible,selected=eligible[:4])


def near_boundary(pair,runs,polygon):
    if polygon is None:return None
    middle=pair['frame'];poly=np.asarray(polygon,np.float32)
    for index in (pair['left'],pair['right']):
        run=runs[index];dates=sorted(run['rows'])
        for a,b in zip(dates,dates[1:]):
            if not middle-4<=a<b<=middle+4 or not run['valid'][a] or not run['valid'][b]:continue
            sides=[]
            for t in (a,b):
                row=run['rows'][t];sides.append(cv2.pointPolygonTest(poly,(float(row[2]+row[4]/2),float(row[3]+row[5]/2)),False)>=0)
            if sides[0]!=sides[1]:return True
    return False


def pixel_crop(rgb,row,inset):
    """Select integer pixel centres, independently of production crop helper."""
    x,y,w,h=map(float,row[2:6]);height,width=rgb.shape[:2]
    xs=[p for p in range(width) if x+inset*w<=p+.5<x+(1-inset)*w]
    ys=[p for p in range(height) if y+inset*h<=p+.5<y+(1-inset)*h]
    if not xs or not ys:return None
    # A crop extending past the image must abstain, not silently clip.
    lo_x=math.ceil(x+inset*w-.5);hi_x=math.ceil(x+(1-inset)*w-.5)
    lo_y=math.ceil(y+inset*h-.5);hi_y=math.ceil(y+(1-inset)*h-.5)
    if lo_x<0 or lo_y<0 or hi_x>width or hi_y>height:return None
    return rgb[np.ix_(ys,xs)].astype(np.float64)/255


def describe(crop):
    pixels=crop.reshape(-1,3)
    mean=np.sum(pixels,axis=0)/len(pixels)
    std=np.sqrt(np.sum((pixels-mean)**2,axis=0)/len(pixels))
    return np.r_[mean,std]


def measure(rows,images):
    """rows=[left_before,right_before,left_after,right_after], fixed boxes/dates."""
    absent=lambda state:dict(state=state,valid=False,informative=False,vote=0)
    if any(i is None for i in images):return absent('missing_RGB')
    if images[0].shape!=images[1].shape:return absent('RGB_shape_changed')
    crops=[pixel_crop(images[i//2],row,.25) for i,row in enumerate(rows)]
    if any(c is None for c in crops):return absent('centre_crop_outside_or_empty')
    descriptors=[describe(c) for c in crops]
    distance=lambda a,b:sum(abs(float(x-y)) for x,y in zip(a,b))/6
    p,q,r,s=descriptors
    aligned=distance(p,r)+distance(q,s);exchanged=distance(p,s)+distance(q,r);margin=aligned-exchanged
    contrast=[distance(p,q),distance(r,s)];changes=[]
    for row in rows:
        a,b=pixel_crop(images[0],row,0),pixel_crop(images[1],row,0)
        if a is None or b is None:return absent('temporal_support_outside_or_empty')
        changes.append(float(np.abs(a-b).sum()/a.size))
    change=sum(changes)/4
    state=('same_appearance_no_information' if min(contrast)<.02 else
           'static_image_no_information' if change<1/255 else
           'equal_assignment_cost_no_information' if abs(margin)<=1/255 else
           'informative_exchange_preferred' if margin>0 else 'informative_original_preferred')
    informative=state.startswith('informative_')
    return dict(state=state,valid=True,informative=informative,vote=(1 if margin<0 else -1) if informative else 0,
        descriptors=[d.tolist() for d in descriptors],aligned_cost=aligned,
        exchanged_cost=exchanged,raw_margin=margin,pair_contrast=contrast,temporal_change=change,temporal_change_by_fixed_box=changes)


def synthetic():
    rows=[]
    for t in range(1,18):
        rows.extend([[t,7,5+t,10,8,8],[t,9,35-t,10,8,8]])
    selection=selected_pairs(np.array(rows,float),'synthetic')
    assert len(selection['close'])==1 and selection['close'][0]['frame']==15
    # Closest date lacks both endpoints: no fallback to the second closest.
    assert not selection['selected']
    for t in range(18,26):rows.extend([[t,7,5+t,10,8,8],[t,9,35-t,10,8,8]])
    selection=selected_pairs(np.array(rows,float),'synthetic')
    assert len(selection['selected'])==1 and selection['selected'][0]['frame']==15
    bad=np.array([[1,1,1,1,4,4],[25,1,1,1,-2,4],[50,1,1,1,4,4],[76,1,1,1,4,4]],float)
    runs=split_original(bad)
    assert len(runs)==2 and not runs[0]['valid'][25] and sorted(runs[0]['rows'])==[1,25,50]
    for a in (np.r_[bad,bad[:1]],np.array([[1,1,1,1,4,4],[27,1,1,1,4,4],[1,100001,2,2,4,4]],float)):
        try:split_original(a)
        except ValueError:pass
        else:raise AssertionError('duplicate or official-alias collision accepted')
    before=np.full((48,96,3),20,np.uint8);after=before.copy()
    before[10:26,10:26]=(220,30,20);before[10:26,60:76]=(20,40,220)
    after[10:26,60:76]=(220,30,20);after[10:26,10:26]=(20,40,220)
    positions=np.array([[1,0,10,10,16,16],[1,1,60,10,16,16],[9,0,60,10,16,16],[9,1,10,10,16,16]],float)
    own=measure(positions,[before,after]);assert own['state']=='informative_original_preferred' and own['aligned_cost']==0
    swapped=positions.copy();swapped[[2,3]]=swapped[[3,2]]
    assert measure(swapped,[before,after])['state']=='informative_exchange_preferred'
    assert measure(positions,[before,before])['state']=='static_image_no_information'
    assert measure(positions,[np.full_like(before,20),np.full_like(after,20)])['state']=='same_appearance_no_information'
    assert pixel_crop(before,[1,0,-2,2,8,8],0) is None
    crop=pixel_crop(before,[1,0,.5,.5,4,4],.25)
    assert crop.shape==(2,2,3)
    return dict(selection_no_fallback=True,gap_invalid_row_retained=True,duplicate_alias_rejected=True,
        crop_mean_std=True,RGB_assignment_and_static_gates=True,passed=True)


def summarize(records,roster):
    def counts(items,arm):
        positive=sum(r[arm]['vote']==1 for r in items);negative=sum(r[arm]['vote']==-1 for r in items)
        return dict(selected=len(items),usable=sum(r[arm]['valid'] for r in items),informative=positive+negative,
            videos=len(set(r['clip'] for r in items)),correct=positive,wrong=negative,abstain=len(items)-positive-negative,
            S=(positive-negative)/len(items) if items else None)
    def section(items):
        control=[r for r in items if r['frame_shift'] is not None]
        both=[r for r in control if r['primary']['valid'] and r['control']['valid']]
        main=counts(control,'primary');shift=counts(control,'control')
        return dict(primary=counts(items,'primary'),time_control_intersection=dict(primary=main,control=shift,
            S_difference=main['S']-shift['S'] if control else None),both_valid_intersection=dict(primary=counts(both,'primary'),control=counts(both,'control')),
            control_date_unavailable=len(items)-len(control))
    result=dict(overall=section(records),hives={h:section([r for r in records if r['hive']==h]) for h in roster},
        clips={clip:section([r for r in records if r['clip']==clip]) for clips in roster.values() for clip in clips},
        boundary_strata={name:section([r for r in records if r['near_boundary']==flag]) for name,flag in [('near',True),('other',False)]},
        hive_boundary_strata={h+'_'+name:section([r for r in records if r['hive']==h and r['near_boundary']==flag]) for h in roster for name,flag in [('near',True),('other',False)]})
    result['equal_clip_weight']={};coverage=True;passed=True
    for h,clips in roster.items():
        ss=[result['clips'][c]['primary']['S'] for c in clips if result['clips'][c]['primary']['S'] is not None]
        cc=[result['clips'][c]['time_control_intersection'] for c in clips if result['clips'][c]['time_control_intersection']['primary']['S'] is not None]
        result['equal_clip_weight'][h]=dict(nonempty_clips=len(ss),primary_S=sum(ss)/len(ss) if ss else None,comparable_clips=len(cc),
            original_S=sum(c['primary']['S'] for c in cc)/len(cc) if cc else None,shifted_S=sum(c['control']['S'] for c in cc)/len(cc) if cc else None)
        p=result['hives'][h]['primary'];pair=result['hives'][h]['time_control_intersection']
        covered=p['selected']>=8 and p['videos']>=3;coverage&=covered
        passed&=covered and p['informative']>=.25*p['selected'] and p['S']>=.2 and pair['S_difference'] is not None and pair['S_difference']>0
    near=result['boundary_strata']['near']['primary']
    result['registered_gates']=dict(primary_coverage=coverage,primary_pass=bool(passed),primary_status='pass' if passed else 'not_verified_coverage' if not coverage else 'fail',
        boundary_coverage=near['selected']>=8,boundary_signal_positive=near['S'] is not None and near['S']>0,
        boundary_explanation_pass=bool(passed and near['selected']>=8 and near['S']>0))
    return result


def field(repo,out,prereg,require_producer_seal=None,audit_out=None):
    repo=Path(repo).resolve();out=Path(out).resolve()
    if prereg!='adab39fbcef0d7cb3e4507d7934598f30a4db388':raise ValueError('Require root-authorized published SHA')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(out/'manifest.json');top=digest(out/'manifest.json')
    assert top==(out/'SEALED.sha256').read_text().strip()
    if require_producer_seal is not None:assert top==require_producer_seal
    for name,sha in manifest.items():assert Path(name).name==name and digest(out/name)==sha
    selection=read(out/'selection.json');measurements=read(out/'measurements.json');published=read(out/'summary.json')
    assert selection['prereg']==prereg and digest(out/'selection.json')==(out/'SELECTION.sha256').read_text().strip()
    assert measurements['selection_sha256']==published['selection_sha256']==digest(out/'selection.json')
    assert published['measurements_sha256']==digest(out/'measurements.json')
    codefiles=[Path(__file__).with_name('track_choice_source_probe.py'),Path(__file__).with_name('track_choice_features.py'),
        repo/'beetrack/zones.json',repo/'tools/whiteboard_fold.py',repo/'tools/gray_fold.py']
    assert {p.name:digest(p) for p in codefiles}==selection['code_hashes']
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'tools'))
    from whiteboard_fold import POLY
    from gray_fold import POLY_G
    polygons=POLY|POLY_G
    roster={'W':[f'BEE24-{i:02d}' for i in [*range(1,10),11,14]],'G':[f'BEE24-{i:02d}' for i in range(17,33) if i not in (24,28)]}
    assert len(selection['sources'])==25 and {s['clip'] for s in selection['sources']}=={c for clips in roster.values() for c in clips}
    own_picks=[];clip_checks=[];gt_hashes={}
    for source in selection['sources']:
        clip=source['clip'];hive=source['hive'];assert clip in roster[hive]
        root=repo/'data/BEE24/train'/clip;gp=root/'gt/gt.txt';gt_hashes[clip]=digest(gp)
        assert gt_hashes[clip]==source['gt_sha256'] and gp.relative_to(repo).as_posix()==source['gt_path']
        files=sorted((root/'img1').glob('*.jpg'));nf=len(files)
        assert [p.name for p in files]==[f'{i:06d}.jpg' for i in range(1,nf+1)] and nf==source['nframes']
        assert polygons[clip]==source['polygon']
        G=np.loadtxt(gp,delimiter=',',usecols=range(6),ndmin=2)
        try:own=selected_pairs(G,clip,nf)
        except ValueError as err:
            assert source['clip_failure'] is not None
            clip_checks.append(dict(clip=clip,failed=True,reason=str(err)));continue
        assert source['clip_failure'] is None
        assert len(own['runs'])==len(source['runs'])
        for run,actual in zip(own['runs'],source['runs']):
            expected=dict(original_id=run['original_id'],run=run['run'],first=min(run['rows']),last=max(run['rows']),observations=len(run['rows']),
                bad_geometry_dates=[t for t in sorted(run['rows']) if not run['valid'][t]],original_row_indices=run['original_row_indices'],official_alias=run['official_alias'])
            assert expected==actual
        assert len(own['pairs'])==len(source['pairs'])
        for pair,actual in zip(own['pairs'],source['pairs']):
            assert [*pair['original_ids'],*pair['runs']]==[actual['id1'],actual['id2'],actual['run1'],actual['run2']]
            assert pair['state']==actual['state']
            if 'frame' in pair:assert pair['frame']==actual['frame'] and abs(pair['distance']-actual['normalized_distance'])<1e-12
        assert len(own['eligible'])==source['eligible_count'] and dict(Counter(p['state'] for p in own['pairs']))==source['attrition']
        assert [p['sha256'] for p in own['selected']]==[p['digest'] for p in source['selected']]
        for pick,actual in zip(own['selected'],source['selected']):
            f=pick['frame'];a,b=f-4,f+4;shift=25 if b+25<=nf else -25 if a-25>=1 else None
            assert actual['hash_input']==pick['hash_input'] and actual['endpoint_frames']==[a,b] and actual['frame_shift']==shift
            assert actual['near_boundary']==near_boundary(pick,own['runs'],polygons[clip])
            rows=[]
            for t in (a,b):
                for identity,rindex in enumerate((pick['left'],pick['right'])):
                    row=own['runs'][rindex]['rows'][t].copy();row[1]=identity;rows.append(row.tolist())
            assert rows==actual['endpoint_rows']
            own_picks.append(dict(clip=clip,hive=hive,digest=pick['sha256'],frame=f,frame_shift=shift,near_boundary=actual['near_boundary'],rows=rows))
        clip_checks.append(dict(clip=clip,failed=False,runs=len(own['runs']),all_pairs=len(own['pairs']),eligible=len(own['eligible']),selected=len(own['selected'])))
        print('INDEPENDENT_SELECTION_PASS',clip,flush=True)
    assert len(own_picks)==selection['selected_count']==len(measurements['pairs'])
    print('INDEPENDENT_ALL25_SELECTED_BEFORE_RGB',len(own_picks),flush=True)
    cache=OrderedDict();image_hashes={}
    def image(clip,t):
        key=(clip,t)
        if key in cache:cache.move_to_end(key);return cache[key]
        path=repo/'data/BEE24/train'/clip/'img1'/f'{t:06d}.jpg';name=path.relative_to(repo).as_posix()
        if path.is_file():
            raw=path.read_bytes();image_hashes[name]=hashlib.sha256(raw).hexdigest()
            bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);value=None if bgr is None else bgr[:,:,::-1].copy()
        else:image_hashes[name]=None;value=None
        assert image_hashes[name]==measurements['source_image_sha256'][name]
        cache[key]=value
        if len(cache)>4:cache.popitem(last=False)
        return value
    independent=[];actual_by_hash={p['digest']:p for p in measurements['pairs']}
    for pick in own_picks:
        actual=actual_by_hash[pick['digest']];a,b=pick['frame']-4,pick['frame']+4;row=np.asarray(pick['rows'],float)
        primary=measure(row,[image(pick['clip'],a),image(pick['clip'],b)])
        shift=pick['frame_shift']
        control=measure(row,[image(pick['clip'],a+shift),image(pick['clip'],b+shift)]) if shift is not None else dict(state='control_dates_outside_source',valid=False,informative=False,vote=0)
        for arm,own in [('primary',primary),('control',control)]:
            saved=actual[arm]
            for key in ('state','valid','informative','vote'):assert own[key]==saved[key],(pick['digest'],arm,key,own[key],saved[key])
            for key in ('aligned_cost','exchanged_cost','raw_margin','pair_contrast','temporal_change','temporal_change_by_fixed_box'):
                if key in own:assert np.allclose(own[key],saved[key],rtol=0,atol=1e-12),(pick['digest'],arm,key)
        independent.append({k:pick[k] for k in ('clip','hive','digest','frame','frame_shift','near_boundary')}|dict(primary=primary,control=control))
    assert image_hashes==measurements['source_image_sha256']
    result=summarize(independent,roster)
    # Every number and gate has been rebuilt without the production summarizer.
    assert result==published['results']
    for source in selection['sources']:assert digest(repo/source['gt_path'])==source['gt_sha256']
    assert {p.name:digest(p) for p in codefiles}==selection['code_hashes']
    target=Path(audit_out).resolve() if audit_out is not None else out/'independent'
    target.mkdir(exist_ok=False)
    receipt=dict(schema='independent-natural-rgb-v1',prereg=prereg,producer_manifest_sha256=top,
        selection_sha256=digest(out/'selection.json'),measurements_sha256=digest(out/'measurements.json'),
        all_passed=True,clip_checks=clip_checks,gt_sha256=gt_hashes,results=result,
        verifier_sha256=digest(Path(__file__)),
        limitations=['Original GT boxes/identity supplied; engineering feature-source diagnostic only, not detector/tracker performance.',
                     'OpenCV JPEG decoding is shared; crop, RGB moments, costs, gates, selection and aggregation are independently implemented.',
                     'Repeated bees/pairs/frames are not independent hive samples; shifted images may contain bees.'])
    (target/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    (target/'numeric_pairs.json').write_text(json.dumps(dict(schema='independent-natural-rgb-numerics-v1',pairs=independent),indent=2,sort_keys=True)+'\n',encoding='utf-8')
    hashes={p.name:digest(p) for p in sorted(target.iterdir())}
    (target/'manifest.json').write_text(json.dumps(hashes,sort_keys=True)+'\n',encoding='utf-8')
    (target/'SEALED.sha256').write_text(digest(target/'manifest.json')+'\n')
    print(json.dumps(dict(all_passed=True,selected=len(independent),seal=digest(target/'manifest.json'),results=result['hives'],near=result['boundary_strata']['near']),indent=2))
    return receipt


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-sealed',action='store_true');parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--out',type=Path);parser.add_argument('--prereg')
    parser.add_argument('--require-producer-seal',help='Optional exact historical producer manifest SHA256')
    parser.add_argument('--audit-out',type=Path,help='Fresh audit directory; default is producer/independent')
    args=parser.parse_args()
    if args.verify_sealed:
        if args.out is None:parser.error('--verify-sealed requires explicit --out')
        field(args.repo,args.out,args.prereg,args.require_producer_seal,args.audit_out)
    else:print(json.dumps(synthetic(),indent=2))
