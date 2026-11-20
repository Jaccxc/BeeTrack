"""CPU-only preregistered source fit, sealed inference, then physical-MOT scoring."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

STRICT_SHA = 'fdd788e0dba0e6fa1c2d06feec3127bf84d5e7b2d214cc0f793c452270403dcf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


@contextmanager
def publication(dest):
    """Publish a complete new directory with a same-volume atomic rename."""
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bridge185-', dir=dest.parent) as name:
        temp = Path(name) / 'payload'
        temp.mkdir()
        yield temp
        if dest.exists():
            raise FileExistsError(dest)
        temp.rename(dest)


def physical(path):
    import numpy as np
    return (np.loadtxt(path, delimiter=',', usecols=range(6), ndmin=2)
            if path.read_bytes().strip() else np.empty((0, 6)))


def provenance(repo):
    files = [Path(__file__), Path(__file__).with_name('selector.py'), Path(__file__).with_name('labels.py')]
    files += [repo / 'beetrack' / name for name in ('p0.py', 'common.py', 'track.py', 'link.py', 'evaluate.py')]
    return {p.name if p.parent == Path(__file__).parent else 'beetrack/' + p.name: code_sha(p) for p in files}


def stable_mask(original, selected):
    """Recover exact original indices, preserving duplicate multiplicity and row order."""
    import numpy as np
    original, selected = np.asarray(original), np.asarray(selected)
    mask = np.zeros(len(original), dtype=bool)
    cursor = 0
    for row in selected:
        while cursor < len(original) and not np.array_equal(original[cursor], row):
            cursor += 1
        if cursor == len(original):
            raise ValueError('Arm rows are not a faithful stable native-row subsequence')
        mask[cursor] = True
        cursor += 1
    return mask


def train(args):
    import numpy as np
    import selector
    import labels
    xs, ys, gs, sources = [], [], [], []
    for seed in range(3):
        tag = f'loho_no{args.hive}_s{seed}'
        for clip, group in ((10, 'W'), (15, 'W'), (24, 'G'), (28, 'G')):
            ev = f'BEE24-{clip}'
            dp = args.repo / 'runs/dets' / tag / 'val' / f'{ev}.npy'
            gp = args.repo / 'runs/gt/val' / ev / 'gt/gt.txt'
            D, GT = np.load(dp, allow_pickle=False), physical(gp)
            ix, X = selector.features(D)
            iy, y, counts = labels.bridge_labels(D, GT)
            if not np.array_equal(ix, iy) or len(X) != len(y):
                raise ValueError('Feature/label original indices differ')
            y = np.asarray(y)
            if not np.isin(y, (0, 1)).all():
                raise ValueError('Expected binary bridge labels')
            xs.append(X); ys.append(y); gs.extend([group] * len(y))
            sources.append(dict(tag=tag, clip=ev, group=group, det_sha256=sha(dp), gt_sha256=sha(gp),
                                counts=counts, class_hist=np.bincount(y.astype(int), minlength=2).tolist()))
    X, y, groups = np.concatenate(xs), np.concatenate(ys), np.asarray(gs)
    hist = {g: np.bincount(y[groups == g].astype(int), minlength=2).tolist() for g in ('W', 'G')}
    if any(min(h) == 0 for h in hist.values()):
        raise ValueError('Both classes required in each W/G group')
    model = selector.fit(X, y, groups)
    meta = dict(prereg=args.prereg, hive=args.hive, sources=sources, group_class_hist=hist,
                per_sample_weights={g: [1 / (4 * n) for n in hist[g]] for g in hist},
                class_hist=np.bincount(y.astype(int), minlength=2).tolist(), code=provenance(args.repo),
                protocol='12 source sets pooled; no tuning; fixed CPU')
    with publication(args.out / f'model{args.hive}') as dest:
        save(dest / 'model.json', model)
        meta['model_sha256'] = sha(dest / 'model.json')
        save(dest / 'fit.json', meta)


def verify_model(args, hive):
    root = args.out / f'model{hive}'
    meta = json.loads((root / 'fit.json').read_text())
    if (meta['prereg'] != args.prereg or meta['hive'] != hive or
            meta['model_sha256'] != sha(root / 'model.json') or meta['code'] != provenance(args.repo)):
        raise ValueError('Source-fit seal mismatch')
    return root, meta


def freeze(args):
    import numpy as np
    import selector
    from beetrack import p0
    from beetrack.common import write_mot
    modelroot, meta = verify_model(args, args.hive)
    ev = f'BEE24-{args.hive}-head'
    dp = args.repo / 'runs/dets' / f'loho_no{args.hive}_s{args.seed}' / 'dev' / f'{ev}.npy'
    D = np.load(dp, allow_pickle=False)
    arms = selector.arms(D, json.loads((modelroot / 'model.json').read_text()))
    if set(arms) != set('HLMC'):
        raise ValueError('Expected exactly H/L/M/C arms')
    strict = args.repo / 'beetrack/models/m_dev_strict.json'
    if sha(strict) != STRICT_SHA:
        raise ValueError('Strict linker digest mismatch')
    receipt = dict(prereg=args.prereg, hive=args.hive, seed=args.seed, clip=ev, det_sha256=sha(dp),
                   model_sha256=meta['model_sha256'], fit_sha256=sha(modelroot / 'fit.json'),
                   code=provenance(args.repo), strict_sha256=sha(strict), arms={})
    with publication(args.out / f'freeze{args.hive}_s{args.seed}') as dest:
        for arm in 'HLMC':
            rows = np.asarray(arms[arm]).reshape(-1, D.shape[1])
            mask = stable_mask(D, rows)
            if arm in 'HL' and not np.array_equal(rows, D[D[:, 5] >= (.4 if arm == 'H' else .25)]):
                raise ValueError('H/L threshold arm mismatch')
            np.save(dest / f'{arm}.mask.npy', mask, allow_pickle=False)
            tracks = p0.p0(p0.stage1(rows, 3950), p0.link_model('m_dev_strict'))
            mot = dest / f'{arm}.txt'
            write_mot(mot, tracks)
            reread = physical(mot)
            if len(reread) != len(tracks):
                raise ValueError('Physical MOT row count mismatch')
            expected = np.asarray(tracks, dtype=float).reshape(-1, 7)[:, :6].copy()
            expected[:, :2] = expected[:, :2].astype(int)
            expected[:, 2:6] = [[float(f'{x:.2f}') for x in row] for row in expected[:, 2:6]] if len(expected) else expected[:, 2:6]
            if not np.array_equal(reread, expected):
                raise ValueError('Physical MOT coordinates differ from native writer precision')
            receipt['arms'][arm] = dict(input_rows=len(rows), track_rows=len(reread),
                                        mask_sha256=sha(dest / f'{arm}.mask.npy'), mot_sha256=sha(mot))
        save(dest / 'receipt.json', receipt)
        (dest / 'SEALED.sha256').write_text(sha(dest / 'receipt.json') + '\n')


def score(args):
    from beetrack import p0
    from tools.study117.tracking.score117 import score_case
    receipts = []
    # Verify all six seals and every artifact before any target GT is opened.
    for hive in (33, 35):
        modelroot, meta = verify_model(args, hive)
        for seed in range(3):
            root = args.out / f'freeze{hive}_s{seed}'
            r = json.loads((root / 'receipt.json').read_text())
            if ((root / 'SEALED.sha256').read_text().strip() != sha(root / 'receipt.json') or
                    r['prereg'] != args.prereg or r['hive'] != hive or r['seed'] != seed or
                    r['model_sha256'] != meta['model_sha256'] or r['code'] != provenance(args.repo) or
                    r['fit_sha256'] != sha(modelroot / 'fit.json') or r['strict_sha256'] != STRICT_SHA):
                raise ValueError('Frozen receipt seal mismatch')
            dp = args.repo / 'runs/dets' / f'loho_no{hive}_s{seed}' / 'dev' / f'BEE24-{hive}-head.npy'
            if r['det_sha256'] != sha(dp) or set(r['arms']) != set('HLMC'):
                raise ValueError('Frozen detector/arms mismatch')
            for arm, a in r['arms'].items():
                if sha(root / f'{arm}.txt') != a['mot_sha256'] or sha(root / f'{arm}.mask.npy') != a['mask_sha256']:
                    raise ValueError('Frozen artifact mismatch')
            receipts.append((root, r))
    cases = []
    for root, r in receipts:
        ev = r['clip']
        if ev != f"BEE24-{r['hive']}-head":
            raise ValueError('Unexpected target clip')
        gtpath = args.repo / 'runs/gt/dev' / ev / 'gt/gt.txt'
        GT = physical(gtpath)
        for arm in 'HLMC':
            mot = physical(root / f'{arm}.txt')
            official = p0.score(mot, 'dev', ev)
            independent = score_case(GT, mot, p0.ZONE[ev], 3950)
            if (official['gt'], official['pred'], official['tp']) != (sum(independent['gt_counts'].values()), sum(independent['pred_counts'].values()), independent['iou']['0.5']['tp']):
                raise ValueError('Official and independent event counts disagree')
            W = 100 * official['err'] / official['gt']
            if abs(W - independent['w30_percent']) > 1e-9:
                raise ValueError('Official and independent window score disagree')
            cases.append(dict(hive=r['hive'], seed=r['seed'], arm=arm, W_percent=W, err=official['err'],
                              nG=official['gt'], nP=official['pred'], tp=official['tp'],
                              f1=2 * official['tp'] / max(official['gt'] + official['pred'], 1),
                              gt_counts=independent['gt_counts'], pred_counts=independent['pred_counts'],
                              iou=independent['iou'], P=official['P'].astype(int).tolist(), G=official['G'].astype(int).tolist(),
                              evidence=independent['evidence'], gt_sha256=sha(gtpath),
                              receipt_sha256=sha(root / 'receipt.json')))
    with publication(args.out / 'score') as dest:
        save(dest / 'summary.json', dict(prereg=args.prereg, cases=cases,
             fits={str(h): json.loads((args.out / f'model{h}' / 'fit.json').read_text()) for h in (33, 35)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prereg', required=True)
    subs = ap.add_subparsers(dest='command', required=True)
    for cmd in ('train', 'freeze'):
        sub = subs.add_parser(cmd)
        sub.add_argument('--hive', type=int, choices=(33, 35), required=True)
        if cmd == 'freeze':
            sub.add_argument('--seed', type=int, choices=range(3), required=True)
    subs.add_parser('score')
    args = ap.parse_args()
    args.repo, args.out = args.repo.resolve(), args.out.resolve()
    if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', args.prereg):
        ap.error('--prereg must be a full SHA')
    if args.out.drive.upper() != 'D:':
        ap.error('--out must be on D:')
    if not (args.repo / 'beetrack/p0.py').is_file():
        ap.error('--repo must be a BeeTrack checkout')
    sys.path.insert(0, str(args.repo))
    import torch
    import cv2
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
