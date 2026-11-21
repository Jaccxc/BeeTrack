"""Registered G18-only background recheck; original selection/masks stay sealed.

Any reference date with nonpositive GT width/height supplies zero trusted pixels.
Dates and true alpha/origin remain exactly those of the original selected donor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.dont_write_bytecode = True
import cv2
import numpy as np
import donor

PREREG = '08bdcb78f8ce482f9c028d6c8e5122c2c4d60384'
MARKER = '[paired-background-recheck]'
ORIGINAL_MANIFEST = '96561379ba7bc1a3d09dd08bf52771f2b7b94e43335528b18bbdeb2d99015119'
ORIGINAL_SELECTION = '9c5d63d5868c4d561ee8a2a8d7bb395516b9e335fe62183c29c2767f1106c5e8'
NAME = 'donors/G_in_BEE24-18_id10_f36'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def background_crop(images, all_rows, xy, shape, min_samples=3):
    """Pure local masked median. Return real counts and explicit inert unknowns."""
    x, y = xy; h, w = shape
    if len(images) != len(all_rows) or not images:
        raise ValueError('aligned fixed background dates required')
    crops, masks, invalid = [], [], []
    for i, (image, rows) in enumerate(zip(images, all_rows)):
        R = np.asarray(rows, float)
        if R.ndim != 2 or R.shape[1] < 6 or not np.isfinite(R).all():
            raise ValueError('invalid full original GT table')
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or y < 0 or x < 0 or y+h > image.shape[0] or x+w > image.shape[1]:
            raise ValueError('invalid image/crop')
        trusted = np.ones((h, w), dtype=bool)
        bad = np.any(R[:, 4:6] <= 0, axis=1)
        if bad.any():
            trusted[:] = False
            invalid.append(i)
        else:
            for bx, by, bw, bh in R[:, 2:6]:
                l, t = max(0, int(np.floor(bx))-x), max(0, int(np.floor(by))-y)
                r, b = min(w, int(np.ceil(bx+bw))-x), min(h, int(np.ceil(by+bh))-y)
                if r > l and b > t:
                    trusted[t:b, l:r] = False
        crop = image[y:y+h, x:x+w].astype(float)
        crop[~trusted] = np.nan
        crops.append(crop); masks.append(trusted)
    counts = np.stack(masks).sum(0)
    valid = counts >= min_samples
    background = np.zeros((h, w, 3), dtype=np.uint8)
    if valid.any():
        background[valid] = np.rint(np.nanmedian(np.stack(crops)[:, valid], axis=0)).astype(np.uint8)
    return background, valid, counts, invalid


def execute(args):
    if args.prereg != PREREG:
        raise ValueError('exact supplemental preregistration SHA required')
    proof = donor.registration(args.repo, args.prereg)
    published = subprocess.run(['git', 'show', args.prereg + ':docs/experiment_log.md'], cwd=args.repo,
                               capture_output=True, check=True).stdout.decode('utf-8')
    if MARKER not in published:
        raise ValueError('supplemental marker missing')
    repo, original, out = args.repo.resolve(), args.original.resolve(), args.out.resolve()
    if out.drive.upper() != 'D:' or out.exists():
        raise ValueError('fresh D: supplemental output required')
    if sha(original/'manifest.json') != ORIGINAL_MANIFEST or (original/'SEALED.sha256').read_text().strip() != ORIGINAL_MANIFEST:
        raise ValueError('original top seal mismatch')
    top = json.loads((original/'manifest.json').read_text())
    if sha(original/'selection.json') != ORIGINAL_SELECTION or (original/'SELECTION.sha256').read_text().strip() != ORIGINAL_SELECTION:
        raise ValueError('original selection seal mismatch')
    selection = json.loads((original/'selection.json').read_text())
    folder = original/NAME
    if sha(folder/'manifest.json') != top[NAME+'/manifest.json'] or (folder/'SEALED.sha256').read_text().strip() != sha(folder/'manifest.json'):
        raise ValueError('G18 donor seal mismatch')
    asset = json.loads((folder/'manifest.json').read_text())
    pick = next(p for p in selection['cells']['G_in']['selected'] if p['seq']=='BEE24-18')
    assert (pick['track_id'],pick['f']) == (10,36)
    assert (asset['seq'],asset['track_id'],asset['f']) == ('BEE24-18',10,36)
    dates = pick['background_frames']
    assert asset['background_frames'] == dates and len(dates) == 15
    source = next(s for s in selection['sources'] if s['seq']=='BEE24-18')
    gp = repo/source['gt_path']
    if sha(gp) != source['gt_sha256']:
        raise ValueError('G18 original GT source changed')
    gt = np.loadtxt(gp, delimiter=',', ndmin=2)
    images, tables, day_records = [], [], []
    image_hashes = {}
    for f in dates:
        path = repo/'data/BEE24/train/BEE24-18/img1'/f'{f:06d}.jpg'
        rel = path.relative_to(repo).as_posix()
        if sha(path) != selection['selected_image_sha256'][rel]:
            raise ValueError('fixed background image changed')
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('background source image unavailable')
        indices = np.flatnonzero(gt[:,0]==f)
        rows = gt[indices]
        bad = np.any(rows[:,4:6] <= 0, axis=1)
        day_records.append(dict(frame=f, trusted_date=not bool(bad.any()),
                                original_row_indices=indices.tolist(), all_original_gt_rows=rows.tolist(),
                                nonpositive_original_row_indices=indices[bad].tolist()))
        images.append(image); tables.append(rows); image_hashes[rel] = sha(path)
    out.parent.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    with TemporaryDirectory(dir=out.parent,prefix='.background-recheck-') as tmp:
        work = Path(tmp)/'payload';work.mkdir()
        frames = []
        for old in asset['frames']:
            assert old['mask_status']=='ok'
            truepath=folder/old['png']
            if sha(truepath)!=old['png_sha256'] or sha(truepath)!=top[NAME+'/'+old['png']]:
                raise ValueError('sealed original true BGRA changed')
            rgba=cv2.imread(str(truepath),cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2]!=4 or not np.isin(rgba[:,:,3],(0,255)).all():
                raise ValueError('invalid sealed true BGRA')
            f=old['frame'];xy=(old['x'],old['y'])
            background,valid,counts,bad=background_crop(images,tables,xy,rgba.shape[:2])
            support=rgba[:,:,3]>0
            npz=f'{f:06d}.background.npz'
            np.savez(work/npz,background=background,valid=valid,counts=counts)
            truefile=f'{f:06d}.true.png';shutil.copyfile(truepath,work/truefile)
            row=dict(frame=f,x=xy[0],y=xy[1],crop_shape=list(rgba.shape[:2]),
                     true_png=truefile,true_png_sha256=sha(work/truefile),original_png_sha256=sha(truepath),
                     background_file=npz,background_sha256=sha(work/npz),
                     unsupported_alpha_pixels=int((~valid[support]).sum()),
                     alpha_support_min_samples=int(counts[support].min()),
                     invalid_background_dates=[dates[i] for i in bad],sham_status='failed',sham_png=None)
            try:
                sham,origin=donor.make_sham_rgba(rgba,xy,background,valid,background_xy=xy)
                assert origin==xy and np.array_equal(sham[:,:,3],rgba[:,:,3])
                png=f'{f:06d}.sham.png'
                if not cv2.imwrite(str(work/png),sham) or not np.array_equal(cv2.imread(str(work/png),cv2.IMREAD_UNCHANGED),sham):
                    raise ValueError('physical sham PNG readback failed')
                row.update(sham_status='ok',sham_png=png,sham_sha256=sha(work/png))
            except ValueError as exc:
                row['failure_reason']=str(exc)
            frames.append(row)
        assert [f['frame'] for f in frames]==pick['frames'] and len(frames)==17
        assert sha(gp)==source['gt_sha256'] and all(sha(repo/p)==s for p,s in image_hashes.items())
        summary=dict(schema='paired-background-recheck-v1',prereg=PREREG,registration_marker=MARKER,
                     registration=proof,original_manifest_sha256=ORIGINAL_MANIFEST,original_selection_sha256=ORIGINAL_SELECTION,
                     original_donor_manifest_sha256=sha(folder/'manifest.json'),seq='BEE24-18',track_id=10,f=36,
                     source_gt_path=source['gt_path'],source_gt_sha256=sha(gp),background_dates=dates,
                     background_day_records=day_records,source_image_sha256=image_hashes,frames=frames,
                     sham_successes=sum(f['sham_status']=='ok' for f in frames),total_frames=17,
                     original_qa_good_frames=asset['qa_good_frames'],original_engineering_pass=asset['engineering_pass'],
                     rule='any nonpositive GT extent excludes all pixels of that unchanged reference date; support>=3 under unchanged alpha',
                     code_sha256={p.name:sha(p) for p in (Path(__file__),Path(donor.__file__))},
                     no_new_segmentation=True,no_replacement_dates=True,original_artifacts_unchanged=True)
        save(work/'summary.json',summary)
        files={p.name:sha(p) for p in sorted(work.iterdir()) if p.is_file()}
        save(work/'manifest.json',files)
        (work/'SEALED.sha256').write_text(sha(work/'manifest.json')+'\n')
        if out.exists():raise FileExistsError(out)
        work.rename(out)
    print(json.dumps(dict(output=str(out),manifest_sha256=sha(out/'manifest.json'),
                          sham_successes=summary['sham_successes'],total_frames=17,
                          invalid_reference_dates=[r['frame'] for r in day_records if not r['trusted_date']])))


def self_test():
    images=[np.full((6,8,3),10+i,np.uint8) for i in range(4)]
    rows=[np.asarray([[1,1,0,0,1,-1]],float)] + [np.empty((0,6)) for _ in range(3)]
    bg,valid,counts,invalid=background_crop(images,rows,(1,1),(3,4),3)
    assert invalid==[0] and (counts==3).all() and valid.all() and (bg==12).all()
    # The invalid reference is not merely ignored at its bad rectangle: its
    # pixels everywhere in this off-box crop must contribute zero support.
    rows=[np.asarray([[1,1,0,0,1,-1]],float)]*4
    bg,valid,counts,invalid=background_crop(images,rows,(1,1),(3,4),3)
    assert not counts.any() and not valid.any() and not bg.any()
    rgba=np.zeros((3,4,4),np.uint8);rgba[1,1,3]=255
    try:donor.make_sham_rgba(rgba,(1,1),bg,valid,background_xy=(1,1))
    except ValueError:pass
    else:raise AssertionError('unsupported alpha accepted')
    print('Synthetic invalid-date full-frame exclusion, fixed-date median/counts and alpha-support rejection passed')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path);ap.add_argument('--original',type=Path)
    ap.add_argument('--out',type=Path);ap.add_argument('--prereg')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test()
    else:
        if not args.repo or not args.original or not args.out or not args.prereg:ap.error('require --repo --original --out --prereg')
        execute(args)
