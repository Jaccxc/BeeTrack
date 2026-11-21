"""Portable native §191 reconstruction from existing detector caches, without GT.

Run in a BeeTrack checkout with --repo --out --hive --seed --recipe. Requires
the registered old CS/pl or VT 640 cache and strict linker; no §190 files needed.
No detector inference, checkpoint loading, scoring or training is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

PREREG = '8ceeeae5f749e348f4a9b760b65639068bbfe436'
STRICT_SHA = 'fdd788e0dba0e6fa1c2d06feec3127bf84d5e7b2d214cc0f793c452270403dcf'
NF = 3950


def source(hive, seed, recipe):
    if type(hive) is not int or hive not in (33, 35) or type(seed) is not int or seed not in range(3, 9) or recipe not in ('CS', 'VT'):
        raise ValueError('Only registered old heads33/35 seeds3..8 CS/VT allowed')
    family = 'vt' if recipe == 'VT' else 'cs' if hive == 33 or seed in (4, 5, 8) else 'pl'
    tag = f'{family}{hive}_s{seed}'
    return dict(hive=hive, seed=seed, recipe=recipe, tag=tag, clip=f'BEE24-{hive}-head',
                group='345' if seed < 6 else '678', nframes=NF,
                source_det_path=f'runs/dets/{tag}/dev/BEE24-{hive}-head.npy')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def native_events(raw, polygon, by_track, zone_tracks):
    tracked = by_track(np.column_stack((raw, np.arange(len(raw)))))
    records = []
    for tid, (direction, start, end, crossings) in zone_tracks(tracked, polygon).items():
        if direction:
            row = next(i for i, d in reversed(crossings) if d == direction)
            original = int(tracked[row, 7])
            records.append(dict(frame=int(tracked[row, 0]), direction=1 if direction == 'in' else -1,
                                track_id=int(tid), native_id=int(raw[original, 1]), source_row=original))
    return records


def window_bins(events, nf=NF):
    counts, boundaries = [], [0]
    for phase in range(30):
        block = np.zeros(((nf + 25 * phase + 749) // 750, 2), dtype=np.int64)
        for f, d in events:
            if not 1 <= f <= nf or d not in (-1, 1):
                raise ValueError('Invalid event clock/direction')
            block[(int(f) - 1 + 25 * phase) // 750, int(d == -1)] += 1
        counts.append(block); boundaries.append(boundaries[-1] + len(block))
    return np.concatenate(counts), boundaries


def reconstruct(repo, out, hive, seed, recipe):
    spec = source(hive, seed, recipe)
    repo, out = Path(repo).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if out.drive.upper() != 'D:':
        raise ValueError('Output must be on D:')
    sys.path.insert(0, str(repo))
    import cv2
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    from beetrack import p0
    from beetrack.common import write_mot
    from beetrack.evaluate import by_track, zone_tracks
    strict = repo / 'beetrack/models/m_dev_strict.json'
    if sha(strict) != STRICT_SHA:
        raise ValueError('Pinned strict linker mismatch')
    dp = repo / spec['source_det_path']
    digest = sha(dp)
    D = np.load(dp, allow_pickle=False)
    if D.ndim != 2 or D.shape[1] != 6 or not np.isfinite(D).all():
        raise ValueError('Expected finite native detector Nx6 rows')
    kept = D[D[:, 5] >= .4]
    if ((kept[:, 0] < 1).any() or (kept[:, 0] > NF).any()
            or not np.equal(kept[:, 0], np.floor(kept[:, 0])).all()
            or (kept[:, 3:5] <= kept[:, 1:3]).any()):
        raise ValueError('Invalid retained native detection')
    raw = np.asarray(p0.p0(p0.stage1(kept, NF), p0.link_model('m_dev_strict')), dtype=float).reshape(-1, 7)
    records = native_events(raw, p0.ZONE[spec['clip']], by_track, zone_tracks)
    E = np.asarray([[e['frame'], e['direction']] for e in records], dtype=np.int64).reshape(-1, 2)
    assert np.array_equal(E, p0.events(raw, p0.ZONE[spec['clip']]).astype(np.int64))
    bins, phases = window_bins(E)
    if sha(dp) != digest:
        raise ValueError('Source detection changed during reconstruction')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix='.rebuild191-') as temp:
        payload = Path(temp) / 'payload'; payload.mkdir()
        np.save(payload / 'mot.npy', raw, allow_pickle=False)
        write_mot(payload / 'mot.txt', raw)
        np.save(payload / 'events.npy', E, allow_pickle=False)
        np.save(payload / 'bins.npy', bins, allow_pickle=False)
        save(payload / 'events.json', records)
        codes = [Path(__file__)] + [repo / 'beetrack' / (name + '.py') for name in ('p0', 'common', 'evaluate', 'track', 'link')]
        metadata = dict(schema='mix191-stream-v1', prereg=PREREG, **spec,
                        source_det_sha256=digest, strict_sha256=STRICT_SHA,
                        source190=None, reconstruction='native source cache, no §190 reuse',
                        native_mot_rows=len(raw), event_count=len(records), polygon=p0.ZONE[spec['clip']],
                        code_sha256={p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in codes},
                        event_semantics='unrounded native P0; last crossing in final direction',
                        bin_phase_offsets=phases,
                        files={p.name: sha(p) for p in sorted(payload.iterdir())})
        save(payload / 'receipt.json', metadata)
        (payload / 'SEALED.sha256').write_text(sha(payload / 'receipt.json') + '\n')
        if out.exists():
            raise FileExistsError(out)
        payload.rename(out)
    return metadata


def seal_manifest(repo, out):
    """Seal an already reconstructed 24-stream directory, without GT or reruns."""
    repo, out = Path(repo).resolve(), Path(out).resolve()
    if any((out / name).exists() for name in ('manifest.json', 'ALL_FROZEN.sha256')):
        raise FileExistsError('Top seal already exists')
    streams = []
    for hive in (33, 35):
        for seed in range(3, 9):
            for recipe in ('CS', 'VT'):
                directory = f'h{hive}_s{seed}_{recipe}'
                folder = out / directory
                r = json.loads((folder / 'receipt.json').read_text())
                digest = sha(folder / 'receipt.json')
                if ((folder / 'SEALED.sha256').read_text().strip() != digest or
                        r['prereg'] != PREREG or not all(r[k] == v for k, v in source(hive, seed, recipe).items()) or
                        not all(sha(folder / name) == value for name, value in r['files'].items()) or
                        sha(repo / r['source_det_path']) != r['source_det_sha256']):
                    raise ValueError('Reconstructed stream/source seal mismatch')
                streams.append(dict(hive=hive, seed=seed, recipe=recipe, directory=directory,
                                    receipt_sha256=digest, events=r['event_count'], source_det_sha256=r['source_det_sha256']))
    metadata = dict(schema='mix191-all-frozen-v1', prereg=PREREG, complete=True, streams=streams,
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    save(out / 'manifest.tmp', metadata)
    (out / 'manifest.tmp').rename(out / 'manifest.json')
    (out / 'ALL_FROZEN.sha256').write_text(sha(out / 'manifest.json') + '\n')
    return dict(manifest_sha256=sha(out / 'manifest.json'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hive', type=int, choices=(33, 35))
    ap.add_argument('--seed', type=int, choices=range(3, 9))
    ap.add_argument('--recipe', choices=('CS', 'VT'))
    ap.add_argument('--seal-manifest', action='store_true', help='seal an existing24-stream output root without rerunning')
    args = ap.parse_args()
    if args.seal_manifest:
        print(json.dumps(seal_manifest(args.repo, args.out), indent=2))
        return
    if args.hive is None or args.seed is None or args.recipe is None:
        ap.error('Require --hive --seed --recipe unless --seal-manifest')
    r = reconstruct(args.repo, args.out, args.hive, args.seed, args.recipe)
    print(json.dumps(dict(tag=r['tag'], files=r['files']), indent=2))


if __name__ == '__main__':
    main()
