"""One native-ID donor window, source-only and MobileSAM CPU, no mask correction.

Real export requires a published ancestor of origin/main containing the explicit
[paired-donor-mask] registration marker. Source W train clips are
01..09,11,14; source G train clips17..32 excluding val24/28. Windows2..75 frames.
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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
sys.dont_write_bytecode = True

import cv2
import numpy as np

REGISTRATION_MARKER = '[paired-donor-mask]'
ALLOW = {'W': {f'BEE24-{k:02d}': 360 if k == 14 else 75 for k in (*range(1, 10), 11, 14)},
         'G': {f'BEE24-{k:02d}': 75 for k in range(17, 33) if k not in (24, 28)}}


def source_spec(seq, heldout, first, last, track_id, available_length=None):
    if heldout not in ('W', 'G', '33', '35'):
        raise ValueError('heldout must be W/G/33/35')
    hives = [h for h, clips in ALLOW.items() if seq in clips]
    if len(hives) != 1 or hives[0] == heldout:
        raise ValueError('sequence is not an allowed other-hive old train source')
    h = hives[0]
    if any(type(x) is not int for x in (first, last, track_id)) or track_id < 0:
        raise ValueError('first/last/native track ID must be integers; ID nonnegative')
    length = ALLOW[h][seq] if available_length is None else available_length
    if type(length) is not int or length < 1:
        raise ValueError('source frame count must be a positive integer')
    if not 1 <= first <= last <= length or not 2 <= last - first + 1 <= 75:
        raise ValueError('require a complete2..75-frame window within the known source train range')
    return dict(seq=seq, source_hive=h, heldout_hive=heldout, track_id=track_id,
                first=first, last=last, source_frame_count=length)


def registration(repo, prereg, run=subprocess.run):
    """Validate a named published registration before source or model access."""
    if not isinstance(prereg, str) or not re.fullmatch('[0-9a-f]{40}', prereg):
        raise ValueError('explicit full --prereg commit required')
    repo = Path(repo).resolve()
    def git(*args):
        result = run(['git', *args], cwd=repo, capture_output=True, check=True)
        return result.stdout.decode('utf-8').strip()
    git('merge-base', '--is-ancestor', prereg, 'refs/remotes/origin/main')
    text = git('show', prereg + ':docs/experiment_log.md')
    if REGISTRATION_MARKER not in text:
        raise ValueError('registration commit lacks ' + REGISTRATION_MARKER)
    published = git('rev-parse', 'refs/remotes/origin/main')
    if not re.fullmatch('[0-9a-f]{40}', published):
        raise ValueError('invalid origin/main commit')
    return dict(prereg=prereg, registration_marker=REGISTRATION_MARKER,
                origin_main_sha=published, ancestor_of_origin_main=True)


def _tracks(gt, spec):
    g = np.asarray(gt, dtype=float)
    if g.ndim != 2 or g.shape[1] < 6 or not np.isfinite(g[:, :6]).all():
        raise ValueError('expected finite original MOT GT rows')
    if not np.equal(g[:, :2], np.floor(g[:, :2])).all():
        raise ValueError('native GT frame and IDs must be integral')
    selected = g[(g[:, 1] == spec['track_id']) & (g[:, 0] >= spec['first']) & (g[:, 0] <= spec['last'])]
    selected = selected[np.argsort(selected[:, 0], kind='stable')]
    expected = np.arange(spec['first'], spec['last'] + 1)
    if not np.array_equal(selected[:, 0], expected) or (selected[:, 4:6] <= 0).any():
        raise ValueError('source ID must have exactly one positive original box in every frame; no filling')
    return selected, g


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _crop_geometry(xy, shape):
    if (len(xy) != 2 or len(shape) != 2 or any(type(v) is not int for v in (*xy, *shape))
            or min(xy) < 0 or min(shape) < 1):
        raise ValueError('integer nonnegative x,y and positive crop height,width required')
    return xy[0], xy[1], shape[0], shape[1]


def local_clean_background(images, all_gt_xywh_per_frame, xy, shape, min_samples):
    """Same-view local temporal median after masking EVERY GT rectangle.

    Caller supplies same-camera BGR frames and all original boxes, not only the
    donor ID. Only the requested crop is stacked. Pixel/rectangle intersection
    is conservative for fractional boxes. Unsupported pixels are inert zero and
    valid=False; the sham consumer checks its alpha support, not transparent
    corners. No original-bee fallback, inpainting or extrapolation is performed.
    """
    x, y, h, w = _crop_geometry(xy, shape)
    if type(min_samples) is not int or min_samples < 1 or len(images) != len(all_gt_xywh_per_frame) or len(images) < min_samples:
        raise ValueError('aligned frames/all-GT lists and positive feasible min_samples required')
    crops, seen, image_shape = [], [], None
    for image, boxes in zip(images, all_gt_xywh_per_frame):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('expected same-view uint8 BGR images')
        if image_shape is None:
            image_shape = image.shape
        if image.shape != image_shape or y + h > image.shape[0] or x + w > image.shape[1]:
            raise ValueError('inconsistent view dimensions or crop outside original image')
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        if not np.isfinite(boxes).all() or (boxes[:, 2:4] <= 0).any():
            raise ValueError('all-GT boxes must be finite positive xywh rectangles')
        valid = np.ones((h, w), dtype=bool)
        for bx, by, bw, bh in boxes:
            left, top = max(0, int(np.floor(bx)) - x), max(0, int(np.floor(by)) - y)
            right, bottom = min(w, int(np.ceil(bx + bw)) - x), min(h, int(np.ceil(by + bh)) - y)
            if right > left and bottom > top:
                valid[top:bottom, left:right] = False
        crop = image[y:y + h, x:x + w].astype(float)
        crop[~valid] = np.nan
        crops.append(crop); seen.append(valid)
    counts = np.stack(seen).sum(axis=0)
    valid = counts >= min_samples
    background = np.zeros((h, w, 3), dtype=np.uint8)
    if valid.any():
        # Reduce supported pixels only: no all-NaN median warnings or fallback.
        background[valid] = np.rint(np.nanmedian(np.stack(crops)[:, valid, :], axis=0)).astype(np.uint8)
    return background, valid, counts


def make_sham_rgba(donor_rgba, xy, background, valid, background_xy=(0, 0)):
    """Keep exact donor alpha/origin; replace bee RGB with verified background.

    Arrays use OpenCV BGR(A). background_xy locates a background crop in the
    same source-view coordinate system; default accepts a full background image.
    A true transparent control remains a distinct operation (zero alpha).
    """
    donor = np.asarray(donor_rgba)
    background, valid = np.asarray(background), np.asarray(valid)
    if donor.dtype != np.uint8 or donor.ndim != 3 or donor.shape[2] != 4 or not np.isin(donor[:, :, 3], (0, 255)).all():
        raise ValueError('expected binary-alpha uint8 BGRA donor')
    h, w = donor.shape[:2]
    x, y, _, _ = _crop_geometry(xy, (h, w))
    bx, by, _, _ = _crop_geometry(background_xy, (h, w))
    if background.dtype != np.uint8 or background.ndim != 3 or background.shape[2] != 3 or valid.shape != background.shape[:2] or valid.dtype != bool:
        raise ValueError('background must be uint8 BGR with explicit bool valid-pixel map')
    rx, ry = x - bx, y - by
    if rx < 0 or ry < 0 or ry + h > len(background) or rx + w > background.shape[1]:
        raise ValueError('background does not cover donor source crop')
    known = valid[ry:ry + h, rx:rx + w]
    if not (donor[:, :, 3] > 0).any():
        raise ValueError('same-contour sham requires nonempty donor alpha support')
    if not known[donor[:, :, 3] > 0].all():
        raise ValueError('unknown background under donor alpha; no bee-pixel fallback allowed')
    sham = donor.copy()
    sham[:, :, :3] = background[ry:ry + h, rx:rx + w]
    assert np.array_equal(sham[:, :, 3], donor[:, :, 3])
    return sham, (x, y)


def _qa(mask, target, others):
    ys, xs = np.nonzero(mask)
    cx, cy = xs + .5, ys + .5
    def hits(row):
        x, y, w, h = row[2:6]
        # A positive-area pixel-cell/rectangle intersection counts, including
        # partial edge pixels (equivalent to floor-left/ceil-right rasterization).
        return int(((xs + 1 > x) & (xs < x + w) & (ys + 1 > y) & (ys < y + h)).sum())
    overlap = []
    for row in others:
        count = hits(row)
        if count:
            overlap.append(dict(track_id=int(row[1]), box_xywh=row[2:6].tolist(), mask_pixels=count))
    return dict(mask_pixels=len(xs), mask_pixels_inside_target_box=hits(target),
                mask_pixels_outside_target_box=len(xs) - hits(target),
                mask_area_over_target_box=float(len(xs) / (target[4] * target[5])),
                other_gt_box_intersections=overlap,
                review_status='not_manually_reviewed', manual_mask_correction=False,
                background_or_second_bee='unknown; inspect original RGB pixels with saved alpha and overlap QA')


def export_inputs(dest, spec, original_gt, image_at, segment, provenance):
    """Atomic exporter with injected inputs; used by synthetic fixtures and guarded CLI.

    segment(BGRimage,original_xyxy)->binary full-resolution mask. No track-ID
    reassignment, interpolation, prompt jitter, mask cleaning, feather or filling.
    """
    verified = source_spec(spec['seq'], spec['heldout_hive'], spec['first'], spec['last'], spec['track_id'],
                           available_length=spec['source_frame_count'])
    if verified != spec:
        raise ValueError('source specification drift')
    selected, gt = _tracks(original_gt, spec)
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix='.donor-') as tmp:
        work = Path(tmp) / 'payload'; work.mkdir()
        records = []
        for row in selected:
            f = int(row[0])
            image = image_at(f)
            if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f'missing/invalid source BGR frame{f}')
            H, W = image.shape[:2]
            x, y, w, h = row[2:6]
            prompt = [max(float(x), 0.), max(float(y), 0.), min(float(x + w), W), min(float(y + h), H)]
            if prompt[2] <= prompt[0] or prompt[3] <= prompt[1]:
                raise ValueError(f'source box outside image at frame{f}')
            mask = segment(image, prompt)
            if mask is None:
                raise ValueError(f'missing mask at frame{f}')
            mask = np.asarray(mask)
            if mask.shape != (H, W) or not np.isin(mask, (0, 1, 255)).all():
                raise ValueError(f'expected binary original-resolution mask at frame{f}')
            mask = mask.astype(bool)
            yy, xx = np.nonzero(mask)
            if not len(xx):
                raise ValueError(f'empty mask at frame{f}')
            left, top, right, bottom = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
            rgba = np.dstack((image[top:bottom, left:right], mask[top:bottom, left:right].astype(np.uint8) * 255))
            filename = f'{f:06d}.png'
            if not cv2.imwrite(str(work / filename), rgba):
                raise OSError('PNG writer failed')
            reread = cv2.imread(str(work / filename), cv2.IMREAD_UNCHANGED)
            if not np.array_equal(reread, rgba) or not np.isin(reread[:, :, 3], (0, 255)).all():
                raise ValueError('physical BGRA readback differs')
            others = gt[(gt[:, 0] == f) & (gt[:, 1] != spec['track_id'])]
            records.append(dict(frame=f, x=left, y=top, png=filename,
                                png_sha256=_sha(work / filename), original_box_xywh=row[2:6].tolist(),
                                prompt_xyxy=prompt, qa=_qa(mask, row, others)))
        manifest = dict(schema='native-donor-v1', **spec, frames=records, provenance=provenance,
                        alpha='binary0/255; pixels are unmodified source BGR; original integer top-left x,y',
                        manual_mask_correction=False, human_review='pending; QA is evidence, not proof of clean bee pixels')
        (work / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
        (work / 'SEALED.sha256').write_text(_sha(work / 'manifest.json') + '\n')
        if dest.exists():
            raise FileExistsError(dest)
        work.rename(dest)
    return manifest


def mobile_cpu(weights):
    """No download: caller must supply the existing MobileSAM file."""
    weights = Path(weights)
    if not weights.is_file() or weights.name != 'mobile_sam.pt':
        raise ValueError('require an existing local mobile_sam.pt; downloads forbidden')
    import torch
    from ultralytics import SAM
    torch.set_num_threads(1); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    model = SAM(str(weights))
    def segment(image, box):
        result = model.predict(image, bboxes=[box], device='cpu', conf=0., verbose=False)[0]
        if result.masks is None or len(result.masks.data) != 1:
            return None
        return result.masks.data[0].cpu().numpy()
    return segment


def real_export(args):
    proof = registration(args.repo, args.prereg)
    # Check the sequence/fold before any source enumeration; actual length follows.
    source_spec(args.seq, args.heldout, 1, 2, args.track_id)
    if args.out.resolve().drive.upper() != 'D:':
        raise ValueError('all donor artifacts must be on D:')
    weights = args.weights.resolve()
    if not weights.is_file() or weights.name != 'mobile_sam.pt':
        raise ValueError('only existing MobileSAM weights accepted')
    repo = args.repo.resolve(); sys.path.insert(0, str(repo))
    from beetrack.common import frame_path, load_gt, seq_dir, n_frames
    source_dir = seq_dir(args.seq).resolve()
    if not source_dir.is_relative_to((repo / 'data/BEE24/train').resolve()):
        raise ValueError('source must resolve to original train split')
    actual_length = n_frames(args.seq)
    spec = source_spec(args.seq, args.heldout, args.first, args.last, args.track_id, actual_length)
    # Metadata-only check: no gap can hide behind a simple file count.
    if any(not (source_dir / 'img1' / f'{f:06d}.jpg').is_file() for f in range(spec['first'], spec['last'] + 1)):
        raise ValueError('source train window has missing physical frame files')
    gt = load_gt(spec['seq'])
    _tracks(gt, spec)
    source_gt = seq_dir(spec['seq']) / 'gt/gt.txt'
    provenance = dict(model='MobileSAM', model_sha256=_sha(weights), device='cpu',
                      source_gt_sha256=_sha(source_gt), mask_export='no correction, no identity interpolation')
    provenance['registration'] = proof
    provenance['expected_historical_frame_count'] = ALLOW[spec['source_hive']][spec['seq']]
    provenance['actual_train_frame_count'] = actual_length
    segment = mobile_cpu(weights)
    return export_inputs(args.out, spec, gt, lambda f: cv2.imread(str(frame_path(spec['seq'], f))), segment, provenance)


def self_test(tmp_root):
    image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    gt = np.asarray([[f, 8, 3, 2, 5, 5] for f in (1, 2, 3)] + [[2, 9, 6, 2, 4, 5]], dtype=float)
    spec = source_spec('BEE24-01', 'G', 1, 3, 8)
    def fake(im, box):
        assert np.array_equal(im, image) and box == [3., 2., 8., 7.]
        mask = np.zeros(im.shape[:2], dtype=bool); mask[2:7, 3:8] = True
        return mask
    with TemporaryDirectory(dir=tmp_root, prefix='donor-synthetic-') as tmp:
        root = Path(tmp)
        got = export_inputs(root / 'good', spec, gt, lambda f: image, fake, {'model': 'synthetic fake'})
        assert [(e['frame'], e['x'], e['y']) for e in got['frames']] == [(1, 3, 2), (2, 3, 2), (3, 3, 2)]
        assert got['frames'][1]['qa']['other_gt_box_intersections'][0]['track_id'] == 9
        assert all(not e['qa']['manual_mask_correction'] for e in got['frames'])
        for label, source, segment in (('missing_frame', gt[gt[:, 0] != 2], fake),
                                       ('missing_ID', gt[gt[:, 1] != 8], fake),
                                       ('missing_mask', gt, lambda im, box: None),
                                       ('empty_mask', gt, lambda im, box: np.zeros(im.shape[:2], bool))):
            try:
                export_inputs(root / label, spec, source, lambda f: image, segment, {})
            except ValueError:
                assert not (root / label).exists()
            else:
                raise AssertionError(label + ' accepted')
        try:
            export_inputs(root / 'good', spec, gt, lambda f: image, fake, {})
        except FileExistsError:
            pass
        else:
            raise AssertionError('overwrite accepted')
    for seq, heldout, first, last in (('BEE24-01', 'W', 1, 3), ('BEE24-24', 'W', 1, 3),
                                      ('BEE24-12', 'G', 1, 3), ('BEE24-33', 'G', 1, 3),
                                      ('BEE24-17', 'W', 74, 76)):
        try:
            source_spec(seq, heldout, first, last, 8)
        except ValueError:
            pass
        else:
            raise AssertionError('forbidden source/range accepted')
    assert ALLOW['W']['BEE24-14'] == 360
    assert all(n == 75 for s, n in ALLOW['W'].items() if s != 'BEE24-14')
    assert all(n == 75 for n in ALLOW['G'].values())
    assert source_spec('BEE24-01', 'G', 75, 76, 8, 76)['source_frame_count'] == 76
    calls = []
    def fake_git(command, **kwargs):
        calls.append(command)
        value = ('document ' + REGISTRATION_MARKER if command[1] == 'show' else 'b' * 40 if command[1] == 'rev-parse' else '')
        return type('Result', (), {'stdout': value.encode()})()
    proof = registration('synthetic-repo', 'a' * 40, fake_git)
    assert proof['ancestor_of_origin_main'] and len(calls) == 3
    assert calls[0] == ['git', 'merge-base', '--is-ancestor', 'a' * 40, 'refs/remotes/origin/main']
    def no_marker(command, **kwargs):
        return type('Result', (), {'stdout': b'not a donor registration'})()
    try:
        registration('synthetic-repo', 'a' * 40, no_marker)
    except ValueError:
        pass
    else:
        raise AssertionError('unregistered commit accepted')
    # The same-contour empty-bee patch is visible on a contrasting recipient;
    # only the distinct fully transparent control must equal the recipient.
    background_color = np.asarray([190, 210, 230], np.uint8)
    views = [np.broadcast_to(background_color, (12, 20, 3)).copy() for _ in range(3)]
    views[0][5:7, 6:8] = [0, 0, 250]
    views[1][:2, :2] = [0, 0, 250]; views[2][:2, :2] = [0, 0, 250]
    bg, known, count = local_clean_background(views, [[[6, 5, 2, 2]], [[0, 0, 2, 2]], [[0, 0, 2, 2]]],
                                              (4, 3), (6, 8), 2)
    assert count.min() == 2 and np.all(bg == background_color)
    donor = np.zeros((6, 8, 4), np.uint8)
    donor[:, :, :3] = [0, 0, 250]; donor[2:4, 2:4, 3] = 255
    sham, origin = make_sham_rgba(donor, (4, 3), bg, known, background_xy=(4, 3))
    assert origin == (4, 3) and np.array_equal(sham[:, :, 3], donor[:, :, 3])
    assert np.all(sham[:, :, :3] == background_color)  # no donor-bee color retained
    recipient = np.full((6, 8, 3), 30, np.uint8)
    render = lambda rgba: np.where(rgba[:, :, 3:] > 0, rgba[:, :, :3], recipient)
    pasted = render(sham)
    assert np.array_equal(pasted[0, 0], recipient[0, 0])
    assert np.array_equal(pasted[2, 2], background_color) and not np.array_equal(pasted, recipient)
    transparent = sham.copy(); transparent[:, :, 3] = 0
    assert np.array_equal(render(transparent), recipient)
    bad_valid = known.copy(); bad_valid[2, 2] = False
    corners = [[[6, 5, 2, 2], [4, 3, 1, 1]], [[0, 0, 2, 2], [4, 3, 1, 1]], [[0, 0, 2, 2], [4, 3, 1, 1]]]
    partial_bg, partial_valid, partial_counts = local_clean_background(views, corners, (4, 3), (6, 8), 2)
    assert partial_counts[0, 0] == 0 and not partial_valid[0, 0] and not partial_bg[0, 0].any()
    corner_sham, _ = make_sham_rgba(donor, (4, 3), partial_bg, partial_valid, background_xy=(4, 3))
    assert np.array_equal(render(corner_sham), pasted)
    absent_bg, absent_valid, absent_counts = local_clean_background(views, [[[4, 3, 8, 6]]] * 3, (4, 3), (6, 8), 1)
    assert not absent_valid.any() and not absent_counts.any() and not absent_bg.any()
    for action in (lambda: make_sham_rgba(donor, (4, 3), bg, bad_valid, background_xy=(4, 3)),
                   lambda: make_sham_rgba(donor, (4, 3), absent_bg, absent_valid, background_xy=(4, 3))):
        try:
            action()
        except ValueError:
            pass
        else:
            raise AssertionError('unobserved background accepted')
    print('Synthetic donor BGRA/origins/QA/continuity/identity/fold/range/missing-mask/atomic/no-overwrite passed')
    print('Synthetic sham exact alpha/xy, clean local median, visible contour, no bee color, transparent identity and unknown rejection passed')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path)
    ap.add_argument('--weights', type=Path)
    ap.add_argument('--seq'); ap.add_argument('--heldout', choices=('W', 'G', '33', '35'))
    ap.add_argument('--first', type=int); ap.add_argument('--last', type=int); ap.add_argument('--track-id', type=int)
    ap.add_argument('--out', type=Path); ap.add_argument('--prereg')
    ap.add_argument('--self-test', action='store_true'); ap.add_argument('--check-assets', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test(Path('D:/beetrack_runs'))
    elif args.check_assets:
        if not args.weights: ap.error('--check-assets requires --weights')
        print(json.dumps(dict(exists=args.weights.is_file(), filename=args.weights.name)))
    else:
        if any(getattr(args, k) is None for k in ('repo', 'weights', 'seq', 'heldout', 'first', 'last', 'track_id', 'out', 'prereg')):
            ap.error('export requires --repo --weights --seq --heldout --first --last --track-id --out --prereg')
        print(json.dumps(real_export(args), indent=2))


if __name__ == '__main__':
    main()
