"""Independent verifier for the separately registered 17-frame BG recheck.

Default is synthetic. The original assets and original acceptance are never
modified. No model/segmentation/selection is run here.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from pathlib import Path
import json
from verify_assets import rectangle_pixels, verified_bytes, sha_bytes, background_dates

PREREG = '08bdcb78f8ce482f9c028d6c8e5122c2c4d60384'


def conservative_background(gt, dates, xy, image_at, wh):
    """Same dates; any invalid GT extent removes that entire day's support."""
    gt = np.asarray(gt,float)
    xy = np.asarray(xy,int).reshape(-1,2)
    x,y = xy.T
    values,available,bad_dates = [],[],[]
    for frame in dates:
        rows = gt[gt[:,0] == frame]
        valid = np.ones(len(xy),bool)
        sample = np.full((len(xy),3),np.nan,np.float32)
        bad = rows[np.any(rows[:,4:6] <= 0,axis=1)]
        if len(bad):
            valid[:] = False
            bad_dates.append(dict(frame=int(frame),rows=len(bad),
                width_nonpositive=int((bad[:,4] <= 0).sum()),height_nonpositive=int((bad[:,5] <= 0).sum())))
        else:
            image = np.asarray(image_at(frame))
            assert image.shape == (wh[1],wh[0],3) and image.dtype == np.uint8
            for row in rows:
                left,top,right,bottom = rectangle_pixels(row[2:6],wh)
                valid &= ~((x >= left)&(x < right)&(y >= top)&(y < bottom))
            sample[valid] = image[y[valid],x[valid]]
        values.append(sample);available.append(valid)
    count = np.stack(available).sum(axis=0) if available else np.zeros(len(xy),int)
    supported = count >= 3
    background = np.zeros((len(xy),3),np.uint8)
    if supported.any():
        background[supported] = np.rint(np.nanmedian(np.stack(values)[:,supported,:],axis=0)).astype(np.uint8)
    return dict(background=background,valid=supported,counts=count,bad_dates=bad_dates,
                sampled_dates=list(dates),replacement_dates=[])


def selfcheck():
    images = {f:np.full((4,4,3),10*f,np.uint8) for f in range(1,6)}
    gt = np.array([[2,1,0,0,1,-1],[4,2,0,0,1,1]],float)
    result = conservative_background(gt,[1,2,3,4,5],[[0,0],[2,2]],images.__getitem__,(4,4))
    assert result['sampled_dates'] == [1,2,3,4,5] and result['replacement_dates'] == []
    assert result['counts'].tolist() == [3,4]
    assert result['valid'].tolist() == [True,True]
    assert result['background'][:,0].tolist() == [30,35]
    assert result['bad_dates'] == [dict(frame=2,rows=1,width_nonpositive=0,height_nonpositive=1)]
    fewer = conservative_background(gt,[1,2,3],[[0,0]],images.__getitem__,(4,4))
    assert fewer['counts'].tolist() == [2] and not fewer['valid'][0] and not fewer['background'].any()
    print('supplement conservative whole-day invalidation/no replacement/support/median synthetic checks passed')


def field_entry(repo,original,supplement,out,prereg,supplement_sha=None):
    import cv2
    import subprocess
    from io import BytesIO
    from functools import lru_cache
    if prereg != PREREG:
        raise ValueError('exact supplemental preregistration required')
    repo,original,supplement,out = map(lambda p:Path(p).resolve(),(repo,original,supplement,out))
    if out.drive.upper() != 'D:' or out.exists():
        raise ValueError('fresh D: independent supplement output required')
    subprocess.run(['git','merge-base','--is-ancestor',prereg,'refs/remotes/origin/main'],cwd=repo,check=True,capture_output=True)
    registration = subprocess.run(['git','show',prereg+':docs/experiment_log.md'],cwd=repo,check=True,capture_output=True).stdout.decode('utf-8')
    assert '[paired-background-recheck]' in registration
    roots = {}
    for key,root in (('original',original),('supplement',supplement)):
        digest = sha_bytes((root/'manifest.json').read_bytes())
        assert digest == (root/'SEALED.sha256').read_text().strip()
        if key == 'original':
            assert digest == '96561379ba7bc1a3d09dd08bf52771f2b7b94e43335528b18bbdeb2d99015119'
        elif supplement_sha is not None:
            assert digest == supplement_sha
        files = json.loads((root/'manifest.json').read_text())
        for relative,wanted in files.items():
            verified_bytes(root,relative,wanted)
        roots[key] = digest
    summary = json.loads((supplement/'summary.json').read_text())
    assert summary['prereg'] == PREREG and summary['original_manifest_sha256'] == roots['original']
    selection = json.loads((original/'selection.json').read_text())
    assert sha_bytes((original/'selection.json').read_bytes()) == summary['original_selection_sha256']
    assert (summary['seq'],summary['track_id'],summary['f']) == ('BEE24-18',10,36)
    donor_folder = original/'donors/G_in_BEE24-18_id10_f36'
    donor = json.loads((donor_folder/'manifest.json').read_text())
    assert sha_bytes((donor_folder/'manifest.json').read_bytes()) == summary['original_donor_manifest_sha256']
    source = next(s for s in selection['sources'] if s['seq'] == 'BEE24-18')
    gtpath = repo/'data/BEE24/train/BEE24-18/gt/gt.txt'
    assert source['gt_path'] == summary['source_gt_path'] == gtpath.relative_to(repo).as_posix()
    assert sha_bytes(gtpath.read_bytes()) == summary['source_gt_sha256'] == source['gt_sha256']
    gt = np.loadtxt(gtpath,delimiter=',',ndmin=2)
    dates = background_dates(source['nframes'],range(28,45))
    assert dates == donor['background_frames'] == summary['background_dates']
    assert [r['frame'] for r in summary['background_day_records']] == dates
    for day in summary['background_day_records']:
        index = np.flatnonzero(gt[:,0] == day['frame'])
        rows = gt[index]
        bad = np.any(rows[:,4:6] <= 0,axis=1)
        assert index.tolist() == day['original_row_indices']
        assert rows.tolist() == day['all_original_gt_rows']
        assert index[bad].tolist() == day['nonpositive_original_row_indices']
        assert bool(not bad.any()) == day['trusted_date']
    expected_images = {f'data/BEE24/train/BEE24-18/img1/{f:06d}.jpg' for f in dates}
    assert set(summary['source_image_sha256']) == expected_images
    for relative,wanted in summary['source_image_sha256'].items():
        assert selection['selected_image_sha256'][relative] == wanted
        verified_bytes(repo,relative,wanted)
    @lru_cache(maxsize=20)
    def image_at(frame):
        relative = f'data/BEE24/train/BEE24-18/img1/{frame:06d}.jpg'
        data = verified_bytes(repo,relative,selection['selected_image_sha256'][relative])
        return cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    assert [r['frame'] for r in summary['frames']] == list(range(28,45))
    frames = []
    for record,old in zip(summary['frames'],donor['frames']):
        assert record['frame'] == old['frame'] and (record['x'],record['y'],record['crop_shape']) == (old['x'],old['y'],old['crop_shape'])
        rgba_data = verified_bytes(supplement,record['true_png'],record['true_png_sha256'])
        assert sha_bytes(rgba_data) == old['png_sha256'] == record['original_png_sha256']
        assert rgba_data == verified_bytes(donor_folder,old['png'],old['png_sha256'])
        rgba = cv2.imdecode(np.frombuffer(rgba_data,np.uint8),cv2.IMREAD_UNCHANGED)
        assert rgba is not None and rgba.shape[2] == 4 and np.isin(rgba[:,:,3],(0,255)).all()
        alpha = rgba[:,:,3]>0
        h,w = alpha.shape;x,y = record['x'],record['y']
        image = image_at(record['frame']);H,W = image.shape[:2]
        assert np.array_equal(rgba[:,:,:3],image[y:y+h,x:x+w])
        yy,xx = np.indices((h,w))
        grid = np.column_stack(((xx+x).ravel(),(yy+y).ravel()))
        expected = conservative_background(gt,dates,grid,image_at,(W,H))
        assert [r['frame'] for r in expected['bad_dates']] == record['invalid_background_dates']
        counts,valid,bg = expected['counts'].reshape(h,w),expected['valid'].reshape(h,w),expected['background'].reshape(h,w,3)
        npz = verified_bytes(supplement,record['background_file'],record['background_sha256'])
        with np.load(BytesIO(npz),allow_pickle=False) as arrays:
            assert np.array_equal(arrays['counts'],counts)
            assert np.array_equal(arrays['valid'],valid)
            assert np.array_equal(arrays['background'],bg)
        lacking = int((~valid[alpha]).sum())
        minimum = int(counts[alpha].min())
        assert lacking == record['unsupported_alpha_pixels'] and minimum == record['alpha_support_min_samples']
        if not lacking:
            assert record['sham_status'] == 'ok'
            data = verified_bytes(supplement,record['sham_png'],record['sham_sha256'])
            sham = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
            assert np.array_equal(sham[:,:,3],rgba[:,:,3]) and np.array_equal(sham[:,:,:3],bg)
        else:
            assert record['sham_status'] != 'ok' and not record.get('sham_png')
        frames.append(dict(frame=record['frame'],origin=[x,y],crop_shape=[h,w],
                           unsupported_alpha_pixels=lacking,alpha_support_min_samples=minimum,
                           sham_ready=not lacking,invalid_background_dates=expected['bad_dates']))
    successes = sum(r['sham_ready'] for r in frames)
    assert successes == summary['sham_successes']
    assert summary['original_qa_good_frames'] == donor['qa_good_frames'] == 0
    assert summary['original_engineering_pass'] == donor['engineering_pass'] == False
    receipt = dict(prereg=PREREG,original_manifest_sha256=roots['original'],supplement_manifest_sha256=roots['supplement'],
                   all_17_alpha_and_origin_identical=True,all_15_dates_unchanged=True,
                   whole_invalid_days_provide_zero_support=True,no_replacement_dates=True,
                   background_counts_medians_and_shams_match=True,sham_successes=successes,frames=17,
                   original_geometry_gate_unchanged=True,original_geometry_pass=False,
                   interpretation='Separate engineering recheck; original sealed outcome unchanged, no segmentation or downstream model evaluation.')
    out.mkdir(parents=True)
    (out/'evidence.json').write_text(json.dumps(dict(receipt=receipt,frames=frames),separators=(',',':')),encoding='utf-8')
    (out/'acceptance.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    files = {p.name:sha_bytes(p.read_bytes()) for p in out.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(files,sort_keys=True))
    (out/'SEALED.sha256').write_text(sha_bytes((out/'manifest.json').read_bytes())+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field',action='store_true');parser.add_argument('--repo',type=Path)
    parser.add_argument('--original',type=Path);parser.add_argument('--supplement',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--prereg');parser.add_argument('--supplement-sha256')
    args = parser.parse_args()
    if args.field:
        if any(getattr(args,k) is None for k in ('repo','original','supplement','out','prereg')):
            parser.error('--field requires --repo --original --supplement --out --prereg')
        field_entry(args.repo,args.original,args.supplement,args.out,args.prereg,args.supplement_sha256)
    else:
        selfcheck()
