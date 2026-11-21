"""Independent source-material verifier. Default execution is synthetic only.

No prepare_assets selection or QA algorithm is imported. Field entry remains
disabled until an explicit completed-run path is supplied by the coordinator.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

PREREG = 'e90ae7df2d8d07eb0924f030c6f46834be8a5a4f'
SOURCE_NUMBERS = {'W': (*range(1,10),11,14), 'G': (*range(17,24),25,26,27,*range(29,33))}
ALLOWED = {f'BEE24-{i:02d}':h for h,ids in SOURCE_NUMBERS.items() for i in ids}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def background_dates(nframes, donor_frames):
    """Independent exact integer form of floor(linspace(0,n-1,k))."""
    available = [f for f in range(1,nframes+1) if f not in set(donor_frames)]
    k = min(15,len(available))
    return [available[(i*(len(available)-1))//(k-1)] for i in range(k)] if k > 1 else available[:k]


def _event(rows, polygon):
    from beetrack.evaluate import by_track, zone_tracks
    rows = by_track(rows)
    entries = zone_tracks(rows,polygon)
    if len(entries) != 1:
        raise ValueError('expected one complete original ID')
    direction,_,_,crosses = next(iter(entries.values()))
    if direction is None:
        return None
    j = next(j for j,d in reversed(crosses) if d == direction)
    return direction,int(rows[j,0])


def eligibility(clips):
    """clips are explicit in-memory {seq,hive,gt,nframes,wh,polygon} records."""
    records, qualified = [], []
    for clip in clips:
        seq,hive = clip['seq'],clip['hive']
        if ALLOWED.get(seq) != hive and not clip.get('synthetic',False):
            raise ValueError('clip is not preregistered original train source')
        gt = np.asarray(clip['gt'],float).reshape(-1,6)
        if not np.isfinite(gt).all() or np.any(gt[:,:2] != np.rint(gt[:,:2])):
            raise ValueError('nonfinite or nonintegral original GT identity/frame')
        width,height = clip['wh']
        for identity in np.unique(gt[:,1]):
            rows = gt[gt[:,1] == identity]
            rows = rows[np.argsort(rows[:,0],kind='stable')]
            out = dict(hive=hive,seq=seq,track_id=int(identity),reasons=[])
            records.append(out)
            if len(np.unique(rows[:,0])) != len(rows):
                out['reasons'].append('duplicate_frame_in_original_id')
            if np.any(np.diff(rows[:,0]) > 25):
                out['reasons'].append('original_id_gap_gt25')
            if out['reasons']:
                continue
            event = _event(rows,clip['polygon'])
            if event is None:
                out['reasons'].append('no_complete_path_event')
                continue
            direction,frame = event
            out.update(direction=direction,f=frame)
            dates = list(range(frame-8,frame+9))
            window = rows[np.isin(rows[:,0],dates)]
            if dates[0] < 1 or dates[-1] > clip['nframes'] or list(window[:,0]) != dates:
                out['reasons'].append('incomplete_contiguous_17_frames')
                continue
            sizes = np.array([clip['size_at'](int(f)) if 'size_at' in clip else (width,height) for f in window[:,0]])
            if (np.any(window[:,4:6] <= 0) or np.any(window[:,2:4] < 0)
                    or np.any(window[:,2]+window[:,4] > sizes[:,0])
                    or np.any(window[:,3]+window[:,5] > sizes[:,1])):
                out['reasons'].append('nonpositive_or_out_of_image_donor_box')
                continue
            if _event(window,clip['polygon']) != event:
                out['reasons'].append('window_event_or_date_differs')
                continue
            exact = f'{hive}|{direction}|{seq}|{int(identity)}|{frame}'
            out.update(hash_input=exact,digest=sha_bytes(exact.encode('ascii')),frames17=dates,
                       background_frames=background_dates(clip['nframes'],dates))
            qualified.append(out)
    cells = {}
    for hive in ('W','G'):
        for direction in ('in','out'):
            pool = [r for r in qualified if (r['hive'],r['direction']) == (hive,direction)]
            pool.sort(key=lambda r:(r['digest'],r['seq'],r['track_id'],r['f']))
            selected,seen = [],set()
            for r in pool:
                if r['seq'] not in seen:
                    selected.append(r.copy());seen.add(r['seq'])
                if len(selected) == 2:
                    break
            cells[f'{hive}_{direction}'] = dict(eligible_count=len(pool),selected=selected)
    return dict(tracks=records,cells=cells,exclusion_counts=dict(Counter(reason for r in records for reason in r['reasons'])))


def polygon_for(seq,zones):
    """Independent literal reconstruction of public §157/159 region formulas."""
    if seq not in ALLOWED:
        raise ValueError('unapproved source polygon')
    if ALLOWED[seq] == 'G':
        return (np.asarray(zones['BEE24-28']) + ([-1,1] if seq in ('BEE24-19','BEE24-25') else [0,0])).tolist()
    if seq in ('BEE24-07','BEE24-09'):
        return (np.asarray(zones['BEE24-15'])+[-1,79]).tolist()
    return zones['BEE24-15'] if seq == 'BEE24-14' else zones['BEE24-10']


def rectangle_pixels(box, wh):
    """Conservative native bbox mask: floor left/top, ceil right/bottom."""
    x,y,w,h = map(float,box)
    W,H = wh
    if w <= 0 or h <= 0:
        return (0,0,0,0)
    return (max(0,int(np.floor(x))),max(0,int(np.floor(y))),
            min(W,int(np.ceil(x+w))),min(H,int(np.ceil(y+h))))


def mask_qa(alpha,origin,prompt,others,wh):
    """Independent full-image alpha checks, raw prompt IoU and other-bee hits."""
    alpha = np.asarray(alpha,bool)
    x,y = map(int,origin)
    W,H = wh
    if alpha.ndim != 2 or x < 0 or y < 0 or x+alpha.shape[1] > W or y+alpha.shape[0] > H:
        raise ValueError('alpha/origin leaves original image')
    ay,ax = np.nonzero(alpha)
    if not len(ax):
        return dict(nonempty=False,bbox_iou=0.,other_gt_box_intersections=0,geometry_ok=False)
    gx,gy = ax+x,ay+y
    bbox = np.array([gx.min(),gy.min(),gx.max()+1,gy.max()+1],float)
    px,py,pw,ph = map(float,prompt)
    target = np.array([px,py,px+pw,py+ph])
    intersection = float(np.maximum(0,np.minimum(bbox[2:],target[2:])-np.maximum(bbox[:2],target[:2])).prod())
    union = float(np.prod(bbox[2:]-bbox[:2])+pw*ph-intersection)
    iou = intersection/union if union > 0 else 0.
    contact = np.zeros(len(gx),bool)
    for box in others:
        a,b,c,d = rectangle_pixels(box,wh)
        contact |= (gx >= a)&(gx < c)&(gy >= b)&(gy < d)
    return dict(nonempty=True,bbox_iou=iou,other_gt_box_intersections=int(contact.sum()),
                geometry_ok=iou >= .5 and not contact.any())


def background_at_pixels(gt,nframes,dates,xy,image_loader,wh):
    """Read only selected background dates; recompute support and RGB medians.

    xy is Nx2 original-image pixel coordinates. image_loader is injected;
    synthetic tests pass in-memory RGB arrays and never access real data.
    """
    xy = np.asarray(xy,int).reshape(-1,2)
    x,y = xy.T
    samples = []
    valid_samples = []
    for frame in dates:
        if not 1 <= frame <= nframes:
            raise ValueError('background date outside original source')
        image = np.asarray(image_loader(frame))
        assert image.shape == (wh[1],wh[0],3)
        valid = np.ones(len(xy),bool)
        for row in gt[gt[:,0] == frame]:
            a,b,c,d = rectangle_pixels(row[2:6],wh)
            valid &= ~((x >= a)&(x < c)&(y >= b)&(y < d))
        values = image[y,x].astype(np.float32)
        values[~valid] = np.nan
        samples.append(values);valid_samples.append(valid)
    if not samples:
        return np.zeros(len(xy),int),np.full((len(xy),3),np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median = np.nanmedian(np.stack(samples),axis=0)
    return np.stack(valid_samples).sum(axis=0),median


def verified_bytes(root,relative,digest):
    path = (Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('artifact path escapes verified root')
    data = path.read_bytes()
    assert sha_bytes(data) == digest, ('artifact hash mismatch',relative)
    return data


def verify_donor(folder,manifest,gt,image_at,nframes):
    """Decode every actual PNG and recompute all crop background pixels."""
    import cv2
    from io import BytesIO
    first,last = manifest['f']-8,manifest['f']+8
    assert [r['frame'] for r in manifest['frames']] == list(range(first,last+1))
    dates = background_dates(nframes,range(first,last+1))
    assert manifest['background_frames'] == dates
    invalid_background = []
    for date in dates:
        day = gt[gt[:,0] == date]
        bad = day[np.any(day[:,4:6] <= 0,axis=1)]
        if len(bad):
            invalid_background.append(dict(frame=date,rows=len(bad),
                width_nonpositive=int((bad[:,4] <= 0).sum()),height_nonpositive=int((bad[:,5] <= 0).sum()),
                width_zero=int((bad[:,4] == 0).sum()),width_negative=int((bad[:,4] < 0).sum()),
                height_zero=int((bad[:,5] == 0).sum()),height_negative=int((bad[:,5] < 0).sum())))
    records = []
    for r in manifest['frames']:
        f = r['frame']
        evidence = dict(frame=f,mask=False,sham=False,geometry_ok=False,
                        producer_mask_error=r.get('error'),producer_sham_error=r.get('sham_error'))
        records.append(evidence)
        own = gt[(gt[:,0] == f)&(gt[:,1] == manifest['track_id'])]
        assert len(own) == 1
        if not r.get('png'):
            assert r['mask_status'] != 'ok'
            evidence['failure'] = 'producer reports no mask; no replacement'
            continue
        data = verified_bytes(folder,r['png'],r['png_sha256'])
        rgba = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
        assert rgba is not None and rgba.dtype == np.uint8 and rgba.ndim == 3 and rgba.shape[2] == 4
        assert np.isin(rgba[:,:,3],(0,255)).all() and (rgba[:,:,3]>0).any()
        alpha = rgba[:,:,3]>0
        h,w = alpha.shape
        x,y = int(r['x']),int(r['y'])
        assert [h,w] == r['crop_shape']
        assert r['original_box_xywh'] == own[0,2:6].tolist()
        original = image_at(f)
        H,W = original.shape[:2]
        assert np.array_equal(rgba[:,:,:3],original[y:y+h,x:x+w]), 'true PNG is not original source RGB'
        ay,ax = np.nonzero(alpha)
        assert (ax.min(),ay.min(),ax.max()+1,ay.max()+1) == (0,0,w,h), 'crop is not tight alpha bbox'
        others = gt[(gt[:,0] == f)&(gt[:,1] != manifest['track_id']),2:6]
        qa = mask_qa(alpha,(x,y),own[0,2:6],others,(W,H))
        assert qa['nonempty'] and abs(qa['bbox_iou']-r['mask_bbox_iou']) < 1e-12
        assert (qa['other_gt_box_intersections']>0) == r['intersects_other_bee']
        assert qa['geometry_ok'] == r['mechanical_frame_pass']
        gx,gy = ax+x,ay+y
        px,py,pw,ph = own[0,2:6]
        target_hits = int(((gx+1>px)&(gx<px+pw)&(gy+1>py)&(gy<py+ph)).sum())
        assert target_hits == r['qa']['mask_pixels_inside_target_box']
        assert len(ax)-target_hits == r['qa']['mask_pixels_outside_target_box']
        evidence.update(mask=True,geometry_ok=qa['geometry_ok'],bbox_iou=qa['bbox_iou'],
                        other_gt_intersection_pixels=qa['other_gt_box_intersections'],
                        target_pixels=target_hits,mask_pixels=len(ax),origin=[x,y],crop_shape=[h,w])
        yy,xx = np.indices((h,w))
        xy = np.column_stack(((xx+x).ravel(),(yy+y).ravel()))
        bg_rows = gt[np.isin(gt[:,0],dates)]
        if invalid_background or len(dates) < 3:
            assert r['sham_status'] != 'ok'
            evidence['background_failure'] = 'background_annotation_validation_rejected' if invalid_background else 'fewer_than_three_background_dates'
            continue
        counts,median = background_at_pixels(gt,nframes,dates,xy,image_at,(W,H))
        counts = counts.reshape(h,w)
        valid = counts >= 3
        background = np.zeros((h,w,3),np.uint8)
        background[valid] = np.rint(median.reshape(h,w,3)[valid]).astype(np.uint8)
        evidence.update(background_support_min=int(counts[alpha].min()),
                        unsupported_alpha_pixels=int((~valid[alpha]).sum()))
        if r.get('background_file'):
            bgdata = verified_bytes(folder,r['background_file'],r['background_sha256'])
            with np.load(BytesIO(bgdata),allow_pickle=False) as z:
                assert np.array_equal(z['counts'],counts)
                assert np.array_equal(z['valid'],valid)
                assert np.array_equal(z['background'],background)
            assert r['alpha_support_min_samples'] == int(counts[alpha].min())
        if r.get('sham_png'):
            shamdata = verified_bytes(folder,r['sham_png'],r['sham_sha256'])
            sham = cv2.imdecode(np.frombuffer(shamdata,np.uint8),cv2.IMREAD_UNCHANGED)
            assert sham is not None and sham.shape == rgba.shape and np.array_equal(sham[:,:,3],rgba[:,:,3])
            assert valid[alpha].all() and np.array_equal(sham[:,:,:3],background)
            assert r['sham_status'] == 'ok'
            evidence['sham'] = True
        else:
            assert r['sham_status'] != 'ok'
            assert not valid[alpha].all(), 'supported mask unexpectedly lacks sham'
    masks = sum(r['mask'] for r in records)
    shams = sum(r['sham'] for r in records)
    good = sum(r['geometry_ok'] for r in records)
    gate = masks == 17 and shams == 17 and good >= 15
    assert (masks,shams,good,gate) == (manifest['mask_successes'],manifest['sham_successes'],manifest['qa_good_frames'],manifest['engineering_pass'])
    return dict(seq=manifest['seq'],track_id=manifest['track_id'],f=manifest['f'],
                mask_successes=masks,sham_successes=shams,qa_good_frames=good,engineering_pass=gate,frames=records,
                invalid_background_dates=invalid_background,background_dates=dates,
                background_annotation_rejection_frames=sum(r.get('background_failure') == 'background_annotation_validation_rejected' for r in records),
                pixel_support_failure_frames=sum(r.get('unsupported_alpha_pixels',0) > 0 for r in records),
                visual_semantics='Original pixels verified; bbox/overlap QA does not establish a pure bee silhouette. Human mixed-bee/edge/flicker review remains separate.')


def field_entry(repo,assets,out,prereg,selection_sha=None):
    """Call ONLY after coordinator supplies completed-run paths and GO."""
    import cv2
    import subprocess
    from functools import lru_cache
    from PIL import Image
    if prereg != PREREG:
        raise ValueError('exact published material preregistration required')
    repo,assets,out = Path(repo).resolve(),Path(assets).resolve(),Path(out).resolve()
    if out.drive.upper() != 'D:' or out.exists():
        raise ValueError('fresh D: verifier output required')
    subprocess.run(['git','merge-base','--is-ancestor',prereg,'refs/remotes/origin/main'],cwd=repo,check=True,capture_output=True)
    text = subprocess.run(['git','show',prereg+':docs/experiment_log.md'],cwd=repo,check=True,capture_output=True).stdout.decode('utf-8')
    assert '[paired-donor-mask]' in text
    top = json.loads((assets/'manifest.json').read_text())
    assert sha_bytes((assets/'manifest.json').read_bytes()) == (assets/'SEALED.sha256').read_text().strip()
    for relative,digest in top.items():
        verified_bytes(assets,relative,digest)
    selection_bytes = (assets/'selection.json').read_bytes()
    digest = sha_bytes(selection_bytes)
    assert digest == (assets/'SELECTION.sha256').read_text().strip()
    if selection_sha is not None:
        assert digest == selection_sha
    selected = json.loads(selection_bytes)
    assert selected['registration']['prereg'] == prereg
    assert {s['seq'] for s in selected['sources']} == set(ALLOWED) and len(selected['sources']) == len(ALLOWED)
    for relative,wanted in selected['source_hashes'].items():
        assert sha_bytes((repo/relative).read_bytes()) == wanted
    zones = json.loads((repo/'beetrack/zones.json').read_text())
    clips,lookup = [],{}
    for source in selected['sources']:
        seq = source['seq'];root = repo/'data/BEE24/train'/seq
        gtpath = root/'gt/gt.txt'
        assert source['hive'] == ALLOWED[seq] and gtpath.relative_to(repo).as_posix() == source['gt_path']
        assert sha_bytes(gtpath.read_bytes()) == source['gt_sha256']
        names = sorted((root/'img1').glob('*.jpg'))
        assert [p.name for p in names] == [f'{f:06d}.jpg' for f in range(1,len(names)+1)]
        assert len(names) == source['nframes']
        polygon = polygon_for(seq,zones)
        assert polygon == source['polygon']
        gt = np.loadtxt(gtpath,delimiter=',',usecols=range(6),ndmin=2)
        def size_at(f,root=root):
            with Image.open(root/'img1'/f'{f:06d}.jpg') as im:
                return im.size
        item = dict(seq=seq,hive=ALLOWED[seq],gt=gt,nframes=len(names),wh=size_at(1),polygon=polygon,size_at=size_at)
        clips.append(item);lookup[seq]=item
    expected = eligibility(clips)
    key = lambda r:(r['hive'],r['seq'],int(r['track_id']),r['direction'],int(r['f']))
    assert {key(r) for r in expected['tracks'] if not r['reasons']} == {key(r) for r in selected['tracks'] if r['eligible']}
    assert len(expected['tracks']) == selected['original_id_count']
    picks = []
    for cell,record in expected['cells'].items():
        actual = selected['cells'][cell]
        assert record['eligible_count'] == actual['eligible_count']
        assert [key(r) for r in record['selected']] == [key(r) for r in actual['selected']]
        for mine,theirs in zip(record['selected'],actual['selected']):
            assert mine['digest'] == theirs['digest'] and mine['hash_input'] == theirs['hash_input']
            assert mine['frames17'] == theirs['frames'] and mine['background_frames'] == theirs['background_frames']
            picks.append(theirs)
    donors = []
    expected_image_names = set()
    for pick in picks:
        for f in pick['frames']+pick['background_frames']:
            expected_image_names.add(f"data/BEE24/train/{pick['seq']}/img1/{f:06d}.jpg")
    assert set(selected['selected_image_sha256']) == expected_image_names
    for relative,wanted in selected['selected_image_sha256'].items():
        verified_bytes(repo,relative,wanted)
    for pick in picks:
        seq = pick['seq'];clip = lookup[seq]
        name = f"{pick['hive']}_{pick['direction']}_{seq}_id{pick['track_id']}_f{pick['f']}"
        folder = assets/'donors'/name
        data = (folder/'manifest.json').read_bytes()
        assert sha_bytes(data) == (folder/'SEALED.sha256').read_text().strip()
        manifest = json.loads(data)
        assert key(manifest) == key(pick)
        @lru_cache(maxsize=20)
        def image_at(f,seq=seq):
            path = repo/'data/BEE24/train'/seq/'img1'/f'{f:06d}.jpg'
            blob = path.read_bytes()
            assert sha_bytes(blob) == selected['selected_image_sha256'][path.relative_to(repo).as_posix()]
            im = cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)
            assert im is not None
            return im
        verified = verify_donor(folder,manifest,clip['gt'],image_at,clip['nframes'])
        verified.update(hive=pick['hive'],direction=pick['direction'])
        donors.append(verified)
        print('VERIFIED_DONOR '+name,flush=True)
    results = json.loads((assets/'results.json').read_text())
    cells = {}
    for cell in expected['cells']:
        own = [r for r in donors if r['hive']+'_'+r['direction'] == cell]
        ready = len(own) == 2 and all(r['engineering_pass'] for r in own)
        assert ready == results['cells'][cell]['pair_ready']
        cells[cell] = dict(selected=len(own),pair_ready=ready,engineering_passes=sum(r['engineering_pass'] for r in own))
    receipt = dict(prereg=prereg,selection_sha256=digest,source_seal_sha256=sha_bytes((assets/'manifest.json').read_bytes()),
                   selection_rebuilt_independently=True,all_png_and_background_crops_checked=True,
                   donors=len(donors),frames=sum(len(r['frames']) for r in donors),cells=cells,
                   background_annotation_rejection_frames=sum(r['background_annotation_rejection_frames'] for r in donors),
                   pixel_support_failure_frames=sum(r['pixel_support_failure_frames'] for r in donors),
                   interpretation='Material engineering verification only; no detector/tracker/event-benefit evaluation.')
    out.mkdir(parents=True)
    (out/'evidence.json').write_text(json.dumps(dict(receipt=receipt,eligibility=expected,donors=donors),separators=(',',':'),allow_nan=False),encoding='utf-8')
    (out/'acceptance.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    files = {p.name:sha_bytes(p.read_bytes()) for p in out.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(files,sort_keys=True),encoding='utf-8')
    (out/'SEALED.sha256').write_text(sha_bytes((out/'manifest.json').read_bytes())+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


def selfcheck():
    poly = [[0,0],[10,0],[10,30],[0,30]]
    def path(identity,frame=20,direction='in'):
        return [[f,identity,14 if (f < frame) == (direction == 'in') else 4,4,2,2] for f in range(1,45)]
    clips = []
    for hive in ('W','G'):
        for seq in ('synthetic-1','synthetic-2','synthetic-3'):
            rows = path(1)+path(2,22,'out')
            clips.append(dict(seq=seq,hive=hive,gt=np.array(rows,float),nframes=44,wh=(40,40),polygon=poly,synthetic=True))
    result = eligibility(clips)
    assert all(c['eligible_count'] == 3 and len(c['selected']) == 2 for c in result['cells'].values())
    assert all(len({r['seq'] for r in c['selected']}) == 2 for c in result['cells'].values())
    assert result['cells'] == eligibility(list(reversed(clips)))['cells']
    assert background_dates(20,range(3,20)) == [1,2,20]
    dates = background_dates(44,range(12,29))
    assert len(dates) == 15 and dates[0] == 1 and dates[-1] == 44 and not set(dates) & set(range(12,29))
    alpha = np.ones((2,2),bool)
    assert mask_qa(alpha,(4,4),[4,4,2,2],[],(40,40))['geometry_ok']
    assert not mask_qa(alpha,(4,4),[4,4,2,2],[[5,5,2,2]],(40,40))['geometry_ok']
    gt = np.array([[1,1,0,0,1,1],[2,1,0,0,1,1]],float)
    loader = lambda f:np.full((4,4,3),10*f,np.uint8)
    support,median = background_at_pixels(gt,4,[1,2,3,4],[[0,0],[2,2]],loader,(4,4))
    assert support.tolist() == [2,4] and median[:,0].tolist() == [35.,25.]
    bad = clips[0].copy();bad['gt'] = np.concatenate((bad['gt'],bad['gt'][:1]))
    assert eligibility([bad])['exclusion_counts']['duplicate_frame_in_original_id'] == 1
    gap = clips[0].copy();g=gap['gt'].copy();g[(g[:,1] == 1)&(g[:,0] > 20),0] += 30;gap['gt']=g
    assert eligibility([gap])['exclusion_counts']['original_id_gap_gt25'] == 1
    from tempfile import TemporaryDirectory
    import cv2
    with TemporaryDirectory(prefix='verify_assets_synthetic_') as temp:
        folder = Path(temp)
        dates17 = list(range(12,29))
        gt = np.array([[f,1,4,4,2,2] for f in dates17],float)
        def source_image(f):
            image = np.full((10,10,3),80,np.uint8)
            if f in dates17:
                image[4:6,4:6] = 200
            return image
        manifest = dict(seq='synthetic',track_id=1,f=20,background_frames=background_dates(44,dates17),
                        frames=[],mask_successes=17,sham_successes=17,qa_good_frames=17,engineering_pass=True)
        for f in dates17:
            rgba = np.full((2,2,4),200,np.uint8);rgba[:,:,3] = 255
            sham = rgba.copy();sham[:,:,:3] = 80
            png,empty,bg = f'{f}.png',f'{f}.sham.png',f'{f}.npz'
            cv2.imwrite(str(folder/png),rgba);cv2.imwrite(str(folder/empty),sham)
            np.savez(folder/bg,background=np.full((2,2,3),80,np.uint8),valid=np.ones((2,2),bool),counts=np.full((2,2),15))
            manifest['frames'].append(dict(frame=f,png=png,png_sha256=sha_bytes((folder/png).read_bytes()),
                mask_status='ok',sham_status='ok',sham_png=empty,sham_sha256=sha_bytes((folder/empty).read_bytes()),
                x=4,y=4,crop_shape=[2,2],original_box_xywh=[4.,4.,2.,2.],mask_bbox_iou=1.,
                intersects_other_bee=False,mechanical_frame_pass=True,
                qa=dict(mask_pixels_inside_target_box=4,mask_pixels_outside_target_box=0),
                background_file=bg,background_sha256=sha_bytes((folder/bg).read_bytes()),alpha_support_min_samples=15))
        assert verify_donor(folder,manifest,gt,source_image,44)['engineering_pass']
        (folder/manifest['frames'][0]['png']).write_bytes(b'tampered')
        try:
            verify_donor(folder,manifest,gt,source_image,44)
        except AssertionError:
            pass
        else:
            raise AssertionError('tampered PNG accepted')
    print('verify_assets independent eligibility/hash distinct-clips, mask QA and background support synthetic checks passed')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field',action='store_true')
    parser.add_argument('--repo',type=Path);parser.add_argument('--assets',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--prereg');parser.add_argument('--selection-sha256')
    args = parser.parse_args()
    if args.field:
        if any(getattr(args,k) is None for k in ('repo','assets','out','prereg')):
            parser.error('--field requires --repo --assets --out --prereg')
        field_entry(args.repo,args.assets,args.out,args.prereg,args.selection_sha256)
    else:
        selfcheck()
