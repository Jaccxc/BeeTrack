"""Registered source-only17-frame donor assets; one MobileSAM CPU model instance.

Selection is sealed before segmentation. Failed donors remain failed; no refill.
Only old W/G train sources are allowed. Pixels and sheets stay in a fresh D: root.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
sys.dont_write_bytecode = True
import cv2
import numpy as np
import donor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def dated_events(rows, polygon, by_track, zone_tracks):
    T = by_track(np.asarray(rows, dtype=float))
    result = []
    for direction, start, end, cross in zone_tracks(T, polygon).values():
        if direction:
            at = next(j for j, d in reversed(cross) if d == direction)
            result.append((direction, int(T[at, 0])))
    return result


def background_frames(nframes, first, last):
    available = [f for f in range(1, nframes + 1) if not first <= f <= last]
    n = min(15, len(available))
    indices = [0] if n == 1 else [i * (len(available) - 1) // (n - 1) for i in range(n)]
    return [available[i] for i in indices]


def eligibility(seq, hive, gt, polygon, nframes, size_at, by_track, zone_tracks):
    """Pure candidate logic. Original IDs, no repaired GT or identity joining."""
    G = np.asarray(gt, dtype=float)
    if G.ndim != 2 or G.shape[1] < 6 or not np.isfinite(G[:, :6]).all() or not np.equal(G[:, :2], np.floor(G[:, :2])).all():
        raise ValueError('invalid original source GT')
    records = []
    for identity in sorted(set(G[:, 1].astype(int))):
        R = G[G[:, 1] == identity, :6]
        R = R[np.argsort(R[:, 0], kind='stable')]
        result = dict(seq=seq, hive=hive, track_id=int(identity), original_rows=len(R), eligible=False)
        reason = None
        if len(np.unique(R[:, 0])) != len(R):
            reason = 'duplicate_native_id_frame'
        elif np.any(np.diff(R[:, 0]) > 25):
            reason = 'native_identity_gap_over25'
        elif (R[:, 0] < 1).any() or (R[:, 0] > nframes).any():
            reason = 'native_clock_outside_source'
        else:
            event = dated_events(R, polygon, by_track, zone_tracks)
            if len(event) != 1:
                reason = 'no_single_full_track_event'
            else:
                direction, f = event[0]
                result.update(direction=direction, f=f)
                first, last = f - 8, f + 8
                W = R[(R[:, 0] >= first) & (R[:, 0] <= last)]
                if first < 1 or last > nframes:
                    reason = 'seventeen_frame_window_outside_source'
                elif not np.array_equal(W[:, 0], np.arange(first, last + 1)):
                    reason = 'missing_or_nonconsecutive_window_frame'
                elif (W[:, 4:6] <= 0).any():
                    reason = 'nonpositive_window_box'
                else:
                    for row in W:
                        width, height = size_at(int(row[0]))
                        x, y, w, h = row[2:6]
                        if x < 0 or y < 0 or x + w > width or y + h > height:
                            reason = 'window_box_outside_image'
                            break
                    if reason is None and dated_events(W, polygon, by_track, zone_tracks) != [(direction, f)]:
                        reason = 'window_does_not_reproduce_full_event_date'
                if reason is None:
                    text = f'{hive}|{direction}|{seq}|{identity}|{f}'
                    result.update(eligible=True, first=first, last=last, frames=list(range(first, last + 1)),
                                  hash_input=text, digest=hashlib.sha256(text.encode('ascii')).hexdigest(),
                                  background_frames=background_frames(nframes, first, last))
        result['exclusion_reason'] = reason
        records.append(result)
    return records


def choose(records):
    cells = {}
    for hive in ('W', 'G'):
        for direction in ('in', 'out'):
            qualified = [r for r in records if r['eligible'] and r['hive'] == hive and r['direction'] == direction]
            ordered = sorted(qualified, key=lambda r: (r['digest'], r['seq'], r['track_id'], r['f']))
            picked, seen = [], set()
            for r in ordered:
                if r['seq'] not in seen:
                    picked.append(r); seen.add(r['seq'])
                if len(picked) == 2:
                    break
            cells[f'{hive}_{direction}'] = dict(eligible_count=len(qualified), eligible_videos=len({r['seq'] for r in qualified}),
                                               selected=picked, selected_count=len(picked), no_replacement=True)
    return cells


def mask_iou(frame):
    x, y = frame['x'], frame['y']
    h, w = frame['crop_shape']
    gx, gy, gw, gh = frame['original_box_xywh']
    intersection = max(0., min(x + w, gx + gw) - max(x, gx)) * max(0., min(y + h, gy + gh) - max(y, gy))
    return intersection / (w * h + gw * gh - intersection)


def write_png(path, image):
    if not cv2.imwrite(str(path), image) or not np.array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), image):
        raise OSError('physical PNG roundtrip failed: ' + path.name)


def contact_sheet(folder, frames):
    tilew, tileh, cols = 360, 150, 4
    sheet = np.full((((len(frames) + cols - 1) // cols) * tileh, cols * tilew, 3), 235, np.uint8)
    for j, record in enumerate(frames):
        y, x = (j // cols) * tileh, (j % cols) * tilew
        title = f"f{record['frame']} mask:{record['mask_status']} sham:{record['sham_status']}"
        cv2.putText(sheet, title, (x + 4, y + 15), cv2.FONT_HERSHEY_SIMPLEX, .35, (20, 20, 20), 1)
        for k, name in enumerate(('png', 'sham_png')):
            if record.get(name):
                rgba = cv2.imread(str(folder / record[name]), cv2.IMREAD_UNCHANGED)
                checker = np.indices(rgba.shape[:2]).sum(axis=0) // 6 % 2 * 60 + 130
                bgr = np.where(rgba[:, :, 3:] > 0, rgba[:, :, :3], checker[:, :, None]).astype(np.uint8)
                scale = min(170 / bgr.shape[1], 120 / bgr.shape[0])
                preview = cv2.resize(bgr, (max(1, int(bgr.shape[1] * scale)), max(1, int(bgr.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
                sheet[y + 24:y + 24 + len(preview), x + 4 + 178 * k:x + 4 + 178 * k + preview.shape[1]] = preview
    write_png(folder / 'contact_sheet.png', sheet)


def prepare_one(dest, pick, gt, image_at, segment, provenance):
    """Attempt all17 frames, retaining explicit failure records; never refill."""
    seq, hive = pick['seq'], pick['hive']
    gt = np.asarray(gt, dtype=float)
    bg_dates = pick['background_frames']
    backgrounds = [image_at(f) for f in bg_dates]
    bg_boxes = [gt[gt[:, 0] == f, 2:6] for f in bg_dates]
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    with TemporaryDirectory(dir=dest.parent, prefix='.asset-') as tmp:
        folder = Path(tmp) / 'payload'; folder.mkdir()
        records = []
        for f in pick['frames']:
            r = dict(frame=f, mask_status='failed', sham_status='not_attempted', png=None, sham_png=None,
                     mechanical_frame_pass=False, error=None)
            try:
                rows = gt[(gt[:, 0] == f) & (gt[:, 1] == pick['track_id'])]
                if len(rows) != 1:
                    raise ValueError('selected original ID no longer unique')
                row = rows[0]; image = image_at(f)
                H, W = image.shape[:2]
                x, y, w, h = row[2:6]
                prompt = [float(x), float(y), float(x + w), float(y + h)]
                mask = segment(image, prompt)
                if mask is None:
                    raise ValueError('missing_single_mask')
                mask = np.asarray(mask)
                if mask.shape != (H, W) or not np.isin(mask, (0, 1)).all() or not mask.any():
                    raise ValueError('invalid_or_empty_full_resolution_binary_mask')
                mask = mask.astype(bool)
                yy, xx = np.nonzero(mask)
                left, top, right, bottom = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
                rgba = np.dstack((image[top:bottom, left:right], mask[top:bottom, left:right].astype(np.uint8) * 255))
                png = f'{f:06d}.png'; write_png(folder / png, rgba)
                others = gt[(gt[:, 0] == f) & (gt[:, 1] != pick['track_id'])]
                r.update(mask_status='ok', png=png, png_sha256=sha(folder / png), x=left, y=top,
                         crop_shape=[bottom - top, right - left], original_box_xywh=row[2:6].tolist(),
                         qa=donor._qa(mask, row, others))
                r['mask_bbox_iou'] = float(mask_iou(r))
                r['intersects_other_bee'] = any(q['mask_pixels'] > 0 for q in r['qa']['other_gt_box_intersections'])
                r['mechanical_frame_pass'] = r['mask_bbox_iou'] >= .5 and not r['intersects_other_bee']
                try:
                    bg, valid, counts = donor.local_clean_background(backgrounds, bg_boxes, (left, top), rgba.shape[:2], 3)
                    witness = f'{f:06d}.background.npz'
                    np.savez(folder / witness, background=bg, valid=valid, counts=counts)
                    r.update(background_file=witness, background_sha256=sha(folder / witness),
                             alpha_support_min_samples=int(counts[rgba[:, :, 3] > 0].min()),
                             background_valid_alpha_pixels=int(valid[rgba[:, :, 3] > 0].sum()))
                    sham, xy = donor.make_sham_rgba(rgba, (left, top), bg, valid, background_xy=(left, top))
                    assert xy == (left, top) and np.array_equal(sham[:, :, 3], rgba[:, :, 3])
                    shamfile = f'{f:06d}.sham.png'; write_png(folder / shamfile, sham)
                    transparent = rgba.copy(); transparent[:, :, 3] = 0
                    base = np.full(rgba.shape[:2] + (3,), 97, np.uint8)
                    assert np.array_equal(np.where(transparent[:, :, 3:] > 0, transparent[:, :, :3], base), base)
                    r.update(sham_status='ok', sham_png=shamfile, sham_sha256=sha(folder / shamfile))
                except Exception as exc:
                    r.update(sham_status='failed', sham_error=type(exc).__name__ + ': ' + str(exc))
            except Exception as exc:
                r['error'] = type(exc).__name__ + ': ' + str(exc)
            records.append(r)
        masks = sum(r['mask_status'] == 'ok' for r in records)
        shams = sum(r['sham_status'] == 'ok' for r in records)
        good = sum(r['mechanical_frame_pass'] for r in records)
        manifest = dict({k: v for k, v in pick.items() if k not in ('frames', 'background_frames')},
                        schema='paired-donor-asset-v1', frames=records, background_frames=bg_dates,
                        provenance=provenance, mask_successes=masks, sham_successes=shams,
                        qa_good_frames=good, other_bee_intersection_frames=sum(r.get('intersects_other_bee', False) for r in records),
                        engineering_pass=masks == 17 and shams == 17 and good >= 15,
                        manual_mask_correction=False, human_review='pending; inspect mixed bees/edges/temporal flicker')
        contact_sheet(folder, records)
        manifest['contact_sheet'] = 'contact_sheet.png'
        manifest['contact_sheet_sha256'] = sha(folder / 'contact_sheet.png')
        save(folder / 'manifest.json', manifest)
        (folder / 'SEALED.sha256').write_text(sha(folder / 'manifest.json') + '\n')
        if dest.exists():
            raise FileExistsError(dest)
        folder.rename(dest)
    return manifest


def run(args):
    proof = donor.registration(args.repo, args.prereg)
    repo, out = args.repo.resolve(), args.out.resolve()
    if out.drive.upper() != 'D:' or out.exists():
        raise ValueError('fresh D: output directory required')
    if not args.weights.is_file() or args.weights.name != 'mobile_sam.pt':
        raise ValueError('existing local mobile_sam.pt required')
    sys.path.insert(0, str(repo)); sys.path.insert(0, str(repo / 'tools'))
    from PIL import Image
    from beetrack.evaluate import by_track, zone_tracks
    from whiteboard_fold import POLY
    from gray_fold import POLY_G
    polygons = POLY | POLY_G
    sources, datasets, records = [], {}, []
    source_hashes = {str(p.relative_to(repo)): sha(p) for p in (repo / 'beetrack/zones.json', repo / 'tools/whiteboard_fold.py', repo / 'tools/gray_fold.py', repo / 'beetrack/evaluate.py')}
    code_hashes = {p.name: sha(p) for p in (Path(__file__), Path(donor.__file__))}
    for hive, clips in donor.ALLOW.items():
        for seq in sorted(clips):
            root = repo / 'data/BEE24/train' / seq
            gtpath = root / 'gt/gt.txt'
            names = sorted((root / 'img1').glob('*.jpg'))
            if [p.name for p in names] != [f'{f:06d}.jpg' for f in range(1, len(names) + 1)]:
                raise ValueError('source frame roster must be contiguous1..N: ' + seq)
            gt = np.loadtxt(gtpath, delimiter=',', usecols=range(6), ndmin=2)
            def size_at(f, root=root):
                with Image.open(root / 'img1' / f'{f:06d}.jpg') as image:
                    return image.size
            rows = eligibility(seq, hive, gt, polygons[seq], len(names), size_at, by_track, zone_tracks)
            sources.append(dict(seq=seq, hive=hive, gt_path=gtpath.relative_to(repo).as_posix(), gt_sha256=sha(gtpath),
                                nframes=len(names), polygon=polygons[seq], polygon_sha256=object_sha(polygons[seq])))
            datasets[seq] = (gt, root)
            records.extend(rows)
    cells = choose(records)
    picks = [p for cell in cells.values() for p in cell['selected']]
    image_hashes = {}
    for p in picks:
        for f in p['frames'] + p['background_frames']:
            path = datasets[p['seq']][1] / 'img1' / f'{f:06d}.jpg'
            image_hashes[path.relative_to(repo).as_posix()] = sha(path)
    out.mkdir(parents=True, exist_ok=False); (out / 'donors').mkdir()
    selection = dict(schema='paired-donor-selection-v1', registration=proof, source_hashes=source_hashes,
                     code_hashes=code_hashes, weights_sha256=sha(args.weights), sources=sources, tracks=records,
                     original_id_count=len(records), eligible_count=sum(r['eligible'] for r in records),
                     exclusions=dict(Counter(r['exclusion_reason'] for r in records if not r['eligible'])),
                     cells=cells, selected_image_sha256=image_hashes,
                     hash_rule='ASCII hive|direction|seq|id|f SHA256 asc; ties seq,id,f; first2 distinctseq; no refill')
    save(out / 'selection.json', selection)
    (out / 'SELECTION.sha256').write_text(sha(out / 'selection.json') + '\n')
    print('SELECTION_SEALED ' + str(out / 'selection.json') + ' ' + sha(out / 'selection.json'), flush=True)
    start = time.perf_counter()
    segment = donor.mobile_cpu(args.weights)  # exactly one CPU model load, after selection seal
    outcomes = []
    for pick in picks:
        gt, root = datasets[pick['seq']]
        name = f"{pick['hive']}_{pick['direction']}_{pick['seq']}_id{pick['track_id']}_f{pick['f']}"
        provenance = dict(selection_sha256=sha(out / 'selection.json'), weights_sha256=selection['weights_sha256'],
                          model='MobileSAM', device='cpu', manual_correction=False)
        result = prepare_one(out / 'donors' / name, pick, gt,
                             lambda f, root=root: cv2.imread(str(root / 'img1' / f'{f:06d}.jpg')), segment, provenance)
        outcomes.append(dict(directory='donors/' + name, hive=pick['hive'], direction=pick['direction'], seq=pick['seq'],
                             track_id=pick['track_id'], f=pick['f'], engineering_pass=result['engineering_pass'],
                             mask_successes=result['mask_successes'], sham_successes=result['sham_successes'],
                             qa_good_frames=result['qa_good_frames'], other_bee_intersection_frames=result['other_bee_intersection_frames'],
                             manifest_sha256=sha(out / 'donors' / name / 'manifest.json')))
        print('DONOR ' + name + ' ' + json.dumps(outcomes[-1], separators=(',', ':')), flush=True)
    for source in sources:
        assert sha(repo / source['gt_path']) == source['gt_sha256']
    assert all(sha(repo / name) == digest for name, digest in image_hashes.items())
    assert all(sha(repo / name) == digest for name, digest in source_hashes.items())
    assert sha(args.weights) == selection['weights_sha256']
    cell_results = {}
    for name in cells:
        results = [r for r in outcomes if name == r['hive'] + '_' + r['direction']]
        cell_results[name] = dict(selected=len(results), engineering_passes=sum(r['engineering_pass'] for r in results),
                                  pair_ready=len(results) == 2 and all(r['engineering_pass'] for r in results),
                                  no_replacement=True, human_review='pending')
    save(out / 'results.json', dict(schema='paired-donor-results-v1', selection_sha256=sha(out / 'selection.json'),
                                  donors=outcomes, cells=cell_results, segmentation_seconds=time.perf_counter()-start,
                                  interpretation='source asset engineering only; no downstream detector/tracker/model evaluation'))
    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    save(out / 'manifest.json', files)
    (out / 'SEALED.sha256').write_text(sha(out / 'manifest.json') + '\n')
    print('ALL_ASSETS_SEALED ' + str(out) + ' ' + sha(out / 'manifest.json'), flush=True)


def self_test(repo):
    sys.path.insert(0, str(repo.resolve()))
    from beetrack.evaluate import by_track, zone_tracks
    poly = [[0, 0], [50, 0], [50, 60], [0, 60]]
    # Event at25, all17 frames change from outside to inside once.
    def track(identity=1):
        return np.asarray([[f, identity, 60 - f / 2, 20, 4, 4] for f in range(1, 60)], float)
    gt = track()
    rows = eligibility('BEE24-01', 'W', gt, poly, 75, lambda f: (100, 70), by_track, zone_tracks)
    assert rows[0]['eligible'] and len(rows[0]['frames']) == 17
    candidates = []
    for hive in ('W', 'G'):
        for direction in ('in', 'out'):
            for seq in ('BEE24-01', 'BEE24-02', 'BEE24-03'):
                r = dict(rows[0], hive=hive, direction=direction, seq=seq)
                r['hash_input'] = f"{hive}|{direction}|{seq}|1|{r['f']}"
                r['digest'] = hashlib.sha256(r['hash_input'].encode('ascii')).hexdigest()
                candidates.extend([r, dict(r, track_id=2)])
    chosen = choose(candidates)
    assert all(len(c['selected']) == 2 and len({r['seq'] for r in c['selected']}) == 2 for c in chosen.values())
    assert choose(list(reversed(candidates))) == chosen
    duplicate = np.vstack((gt, gt[0]))
    assert eligibility('BEE24-01', 'W', duplicate, poly, 75, lambda f: (100, 70), by_track, zone_tracks)[0]['exclusion_reason'] == 'duplicate_native_id_frame'
    hole = gt[gt[:, 0] != rows[0]['f']]
    assert not eligibility('BEE24-01', 'W', hole, poly, 75, lambda f: (100, 70), by_track, zone_tracks)[0]['eligible']
    assert background_frames(75, 17, 33) == [f for i, f in enumerate([f for f in range(1, 76) if not 17 <= f <= 33]) if i in [j * 57 // 14 for j in range(15)]]
    pick = rows[0]
    def image_at(f):
        image = np.full((70, 100, 3), 180, np.uint8)
        return image
    def fake(image, box):
        mask = np.zeros(image.shape[:2], bool)
        x, y, xx, yy = box; mask[int(y):int(yy), int(x):int(xx)] = True
        return mask
    with TemporaryDirectory(dir='D:/beetrack_runs', prefix='assets-synthetic-') as tmp:
        root = Path(tmp)
        result = prepare_one(root / 'good', pick, gt, image_at, fake, {'model': 'fake'})
        assert len(result['frames']) == 17 and result['mask_successes'] == 17
        assert result['qa_good_frames'] == 17
        failed = prepare_one(root / 'failed', pick, gt, image_at, lambda im, box: None, {'model': 'fake'})
        assert failed['mask_successes'] == 0 and not failed['engineering_pass'] and len(failed['frames']) == 17
    print('Synthetic eligibility/fullID/17frames/hashdistinctclip/backgrounddates/PNGsham/QA/failure-no-refill passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--prereg')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(args.repo)
    else:
        if not args.out or not args.weights or not args.prereg:
            parser.error('real preparation requires --out --weights --prereg')
        run(args)
