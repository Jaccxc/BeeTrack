"""GT-free §190 design-stream freeze. Await root preregistration before invoking."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from candidates190 import canonicalize, candidates, self_test

STRICT_SHA = 'fdd788e0dba0e6fa1c2d06feec3127bf84d5e7b2d214cc0f793c452270403dcf'
PREREG = 'aeaf95b76e99baef940dddbdbdb91d33611bf3dd'
NF = 3950
P1 = {35: (448, 352, 352, 352, 1280, 448, 352, 352, 960),
      33: (1600, 1280, 960, 1600, 960, 1600, 1280, 1600, 1280)}
CS = {35: {2: 352, 3: 352}, 33: {2: 1280, 3: 960}}


def source(hive, seed):
    if type(hive) is not int or hive not in CS or type(seed) is not int or seed not in range(3, 9):
        raise ValueError('Only design streams heads33/35, seeds3..8 allowed')
    group = 2 if seed < 6 else 3
    family = 'cs' if P1[hive][seed] != CS[hive][group] else 'pl'
    return dict(hive=hive, seed=seed, group=group, tag=f'{family}{hive}_s{seed}',
                clip=f'BEE24-{hive}-head', cs_scale=CS[hive][group], nframes=NF)


def sha(path, code=False):
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b'\r\n', b'\n') if code else raw).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


@contextmanager
def publication(dest):
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix='.freeze190-') as tmp:
        payload = Path(tmp) / 'payload'
        payload.mkdir()
        yield payload
        if dest.exists():
            raise FileExistsError(dest)
        payload.rename(dest)


def freeze_stream(repo, out, prereg, hive, seed):
    if PREREG is None or prereg != PREREG:
        raise RuntimeError('Field execution blocked pending public preregistration SHA from root')
    spec = source(hive, seed)
    repo, out = Path(repo).resolve(), Path(out).resolve()
    if out.drive.upper() != 'D:':
        raise ValueError('All artifacts must be on D:')
    sys.path.insert(0, str(repo))
    import torch
    import cv2
    torch.set_num_threads(1); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    from beetrack import p0
    from beetrack.common import write_mot
    from beetrack.evaluate import by_track
    dp = repo / 'runs/dets' / spec['tag'] / 'dev' / (spec['clip'] + '.npy')
    strict = repo / 'beetrack/models/m_dev_strict.json'
    if sha(strict) != STRICT_SHA:
        raise ValueError('Strict linker SHA mismatch')
    source_hash = sha(dp)
    D = np.load(dp, allow_pickle=False)
    if (D.ndim != 2 or D.shape[1] != 6 or not np.isfinite(D).all()
            or not np.equal(D[:, 0], np.floor(D[:, 0])).all()
            or (D[:, 0] < 1).any() or (D[:, 0] > NF).any()):
        raise ValueError('Invalid native detection clock/shape')
    kept = np.flatnonzero(D[:, 5] >= .4)
    selected = D[kept]
    if (selected[:, 3:5] <= selected[:, 1:3]).any():
        raise ValueError('Nonpositive retained detection')
    raw = np.asarray(p0.p0(p0.stage1(selected, NF), p0.link_model('m_dev_strict')), float).reshape(-1, 7)
    canonical, membership, segments = canonicalize(raw, by_track)
    frozen = candidates(canonical, NF)
    if sha(dp) != source_hash:
        raise ValueError('Source detector changed during freeze')
    codefiles = [Path(__file__), Path(__file__).with_name('candidates190.py')]
    codefiles += [repo / 'beetrack' / f'{name}.py' for name in ('p0', 'track', 'link', 'common', 'evaluate')]
    meta = dict(schema='emerge190-freeze-v1', prereg=prereg, **spec,
                source_det_path=dp.relative_to(repo).as_posix(),
                source_det_sha256=source_hash, strict_sha256=sha(strict),
                code_sha256={p.name: sha(p, code=True) for p in codefiles},
                segment_version='official by_track per native ID; compact canonical IDs; raw row order',
                segments=segments, candidates=len(frozen['candidates']),
                native_mot_rows=len(raw), selected_detection_rows=len(selected))
    dest = out / f'h{hive}_s{seed}'
    with publication(dest) as temp:
        np.save(temp / 'dets.npy', selected, allow_pickle=False)
        np.save(temp / 'det_indices.npy', kept, allow_pickle=False)
        np.save(temp / 'mot.npy', raw, allow_pickle=False)
        np.save(temp / 'canonical.npy', canonical, allow_pickle=False)
        np.save(temp / 'membership.npy', membership, allow_pickle=False)
        np.save(temp / 'row_keys.npy', canonical[:, :2].astype(np.int64)[:, ::-1], allow_pickle=False)
        write_mot(temp / 'mot.txt', raw)
        save(temp / 'candidates.json', frozen)
        meta['files'] = {p.name: sha(p) for p in sorted(temp.iterdir())}
        save(temp / 'receipt.json', meta)
        (temp / 'SEALED.sha256').write_text(sha(temp / 'receipt.json') + '\n')
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--prereg')
    parser.add_argument('--hive', type=int, choices=(33, 35))
    parser.add_argument('--seed', type=int, choices=range(3, 9))
    args = parser.parse_args()
    if args.self_test:
        sys.path.insert(0, str(args.repo.resolve()))
        from beetrack.evaluate import by_track
        self_test(by_track)
        expected = {33: ['cs', 'cs', 'cs', 'cs', 'cs', 'cs'], 35: ['pl', 'cs', 'cs', 'pl', 'pl', 'cs']}
        assert all(source(h, k)['tag'] == f'{expected[h][k-3]}{h}_s{k}' for h in (33, 35) for k in range(3, 9))
        try:
            freeze_stream('nonexistent', 'nonexistent', 'unregistered', 33, 3)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unregistered field execution accepted')
        print('Synthetic source mapping and preregistration gating passed')
        return
    if not (args.out and args.prereg and args.hive and args.seed is not None):
        parser.error('Require --out --prereg --hive --seed')
    if not re.fullmatch('[0-9a-f]{40}', args.prereg):
        parser.error('Require full preregistration SHA')
    print(freeze_stream(args.repo, args.out, args.prereg, args.hive, args.seed))


if __name__ == '__main__':
    main()
