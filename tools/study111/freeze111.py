"""Freeze one registered §111 eligibility manifest from an audited native P0 pass."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np


ARCHIVE_SHA = '6c6c455b54023b3f87cc957c205122db552133c21fb389c1d6423d345e314541'
ZONES_SHA = 'fa89f3f7fe2d5f9b9825407641e5d3f10e692c0e8f08d83630aa4cebc6c13212'
NFRAMES = 3950


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pinned(path, expected):
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f'Input fingerprint differs: {path}')
    return data


def prepared_arrays(raw, dets, expected):
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        meta = json.loads(data['metadata'].item())
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f'Clean P0 preparation differs on {key}')
        survivors, rows, origins = (data[k] for k in ('survivors','pass1','origins'))
    for name, arr, dtype in [('survivors',survivors,'<i8'), ('pass1',rows,'<f8'), ('origins',origins,'<i8')]:
        if arr.dtype != np.dtype(dtype) or sha(arr.tobytes(order='C')) != meta[name+'_sha256']:
            raise ValueError(f'Clean P0 array differs: {name}')
    if (survivors.ndim != 1 or np.unique(survivors).size != len(survivors)
            or np.any(survivors < 0) or np.any(survivors >= len(dets))
            or rows.ndim != 2 or rows.shape[1] != 7 or not np.isfinite(rows).all()
            or origins.shape != (len(rows),) or not np.isin(origins, survivors).all()
            or not np.array_equal(rows[:,0], dets[origins,0])):
        raise ValueError('Invalid raw-row lineage for the clean P0 preparation')
    return survivors, rows, origins, meta


def write_once(path, record):
    raw = (json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        partial = Path(temp) / 'manifest'
        partial.write_bytes(raw)
        try:
            os.link(partial, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError(f'Frozen output differs: {path}')
    return sha(raw)


def freeze(hive, seed, outdir, prereg):
    from beetrack.common import ROOT, RUNS
    from beetrack.evaluate import ZONES, load_tracks
    from episodes111 import build_case
    from tools.study104.track_reverse import STRICT_SHA256

    if hive not in (33,35) or seed not in range(9) or not re.fullmatch('[0-9a-f]{40}',prereg):
        raise ValueError('Expected a registered head, detector seed, and full preregistration SHA')
    clip = f'BEE24-{hive}-head'
    tag = f'loho_no{hive}_s{seed}'
    archive = json.loads(pinned(ROOT/'docs/study105_results.json', ARCHIVE_SHA))
    source, = [row['sources'] for row in archive['cases'] if (row['hive'],row['seed']) == (hive,seed)]
    paths = {'det':f'dets/{tag}/dev/{clip}.npy', 'gt':f'gt/dev/{clip}/gt/gt.txt',
             'p0':f'trackers/dev/q_lk_{tag}/data/{clip}.txt'}
    hashes = {k:source[path] for k,path in paths.items()}
    with (RUNS/paths['p0']).open('rb') as native:
        if hashlib.file_digest(native, 'sha256').hexdigest() != hashes['p0']:
            raise ValueError('P0 MOT differs from the archived §105 fingerprint')
    det = np.load(io.BytesIO(pinned(RUNS/paths['det'], hashes['det'])), allow_pickle=False)
    gt = load_tracks(io.BytesIO(pinned(RUNS/paths['gt'], hashes['gt'])))
    poly = json.loads(pinned(ZONES, ZONES_SHA))[clip]
    prepared = (outdir/'prepared'/f'{hive}_{seed}.npz').read_bytes()
    expected = dict(study=111, prereg=prereg, fold=hive, seed=seed, clip=clip, nframes=NFRAMES,
                    det_sha256=hashes['det'], baseline_sha256=hashes['p0'], native_sha256=hashes['p0'],
                    strict_sha256=STRICT_SHA256)
    survivors, first, origins, meta = prepared_arrays(prepared, det, expected)
    record = build_case(gt, poly, det, survivors, first, origins, hive, NFRAMES)
    hashes.update(zones=ZONES_SHA, prepared=sha(prepared), strict=STRICT_SHA256, archive105=ARCHIVE_SHA,
                  **{k:meta[k+'_sha256'] for k in ('survivors','pass1','origins')})
    record.update(study=111, prereg=prereg, role='target', fold=hive, hive=hive, seed=seed, clip=clip,
                  nframes=NFRAMES, hashes=hashes)
    path = outdir/'target'/f'{hive}_{seed}.json'
    print(f'{write_once(path, record)} {path}', flush=True)


def inventory(outdir, prereg):
    if not re.fullmatch('[0-9a-f]{40}', prereg):
        raise ValueError('Expected a full preregistration SHA')
    cases, folds = {}, {}
    for hive in (33,35):
        nearall, paired = [], []
        for seed in range(9):
            key = f'{hive}_{seed}'
            raw = (outdir/'target'/f'{key}.json').read_bytes()
            row = json.loads(raw)
            expected = dict(study=111, prereg=prereg, role='target', fold=hive, hive=hive, seed=seed,
                            clip=f'BEE24-{hive}-head', nframes=NFRAMES)
            if (any(row[k] != value for k,value in expected.items())
                    or row['hashes']['archive105'] != ARCHIVE_SHA or row['hashes']['zones'] != ZONES_SHA):
                raise ValueError(f'Frozen case metadata differs: {key}')
            cases[key] = sha(raw)
            nearall.append(len(row['rows_all']))
            paired.append(len(row['rows']))
        folds[str(hive)] = dict(paired_run=sum(n >= 10 for n in paired) >= 7,
                                paired_counts=paired, nearall_counts=nearall)
    path = outdir/'freeze111.json'
    print(f'{write_once(path, dict(study=111, prereg=prereg, cases=cases, folds=folds))} {path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cli = commands.add_parser('freeze')
    cli.add_argument('hive', type=int, choices=(33,35))
    cli.add_argument('seed', type=int, choices=range(9))
    aggregate = commands.add_parser('inventory')
    for sub in (cli, aggregate):
        sub.add_argument('outdir', type=Path)
        sub.add_argument('--prereg', required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.hive, args.seed, args.outdir, args.prereg)
    else:
        inventory(args.outdir, args.prereg)
