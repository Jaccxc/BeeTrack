"""Freeze anchor-silent, two-buddy crossing proposals from the two GT-free §108 masks."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'study108'))
from support import _observations


PINNED = {
    'source': '36fe1cc5470b9155a61bb8bed2f0f06242cccbbbda1b90842607976ce08e831e',
    'target': '011b06f9fb98325d0407beac86819fda5ea8058b6627934e34c06c403ae63375',
}


def adjacency(left, right):
    _, af, a = left
    _, bf, b = right
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)[:, None]
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)[None, :]
    return ((np.sign(af[:, None]) == np.sign(bf[None, :]))
            & (np.abs(af[:, None] - bf[None, :]) <= 5)
            & (3 * intersection >= area_a + area_b))


def pairs(matrix, left_ids, right_ids):
    rows, cols = linear_sum_assignment(matrix, maximize=True)
    valid = matrix[rows, cols]
    return np.column_stack((left_ids[rows[valid]], right_ids[cols[valid]])).tolist()


def direction_counts(events):
    return {'in': sum(row[1] > 0 for row in events), 'out': sum(row[1] < 0 for row in events)}


def propose(a_rows, b_rows, c_rows, nframes, previous):
    a, b, c = (_observations(data, nframes) for data in (a_rows, b_rows, c_rows))
    ab, ac = adjacency(a, b), adjacency(a, c)
    anchor_pairs = [pairs(ab, a[0], b[0]), pairs(ac, a[0], c[0])]
    kept_ids = sorted({i for group in anchor_pairs for i, _ in group})
    if kept_ids != previous['retained'] or anchor_pairs != previous['pairs']:
        raise ValueError('Aligned anchor support differs from the frozen §108 IDs or pairs')

    # Every anchor edge disqualifies a buddy, including edges lost to an assignment tie.
    eligible_b, eligible_c = ~ab.any(axis=0), ~ac.any(axis=0)
    b_sub = tuple(x[eligible_b] for x in b)
    c_sub = tuple(x[eligible_c] for x in c)
    # Keep all remaining rows and columns in resolved-ID order, even those with no B:C edges.
    matches = pairs(adjacency(b_sub, c_sub), b_sub[0], c_sub[0])
    a_map, b_map, c_map = ({row[0]: row for row in data} for data in (a_rows, b_rows, c_rows))
    kept = [a_map[i] for i in kept_ids]
    donors = [dict(b=b_map[i], c=c_map[j], direction='in' if b_map[i][1] > 0 else 'out')
              for i, j in matches]
    counts = {key: direction_counts(rows) for key, rows in (
        ('A', a_rows), ('B', b_rows), ('C', c_rows), ('K', kept), ('D', [d['b'] for d in donors]))}
    return dict(counts=counts, kept_ids=kept_ids, kept=kept, donors=donors)


def make_cases(role, masks):
    if masks['role'] != role:
        raise ValueError('Unexpected §108 role')
    result = []
    for fold in (33, 35):
        hive = 68 - fold if role == 'source' else fold
        for part in (('head', 'tail') if role == 'source' else ('head',)):
            clip = f'BEE24-{hive}-{part}'
            nframes = 3950 if part == 'head' else 1000
            for anchor in (0, 3, 6):
                seeds = range(anchor, anchor + 3)
                keys = [f'{fold}_{s}_{part}' for s in seeds]
                pred = [masks['predictions'][k] for k in keys]
                for s, row in zip(seeds, pred):
                    if (row['fold'], row['seed'], row['clip'], row['nframes']) != (fold,s,clip,nframes):
                        raise ValueError(f'Inconsistent input case: {role}, {fold}, {s}, {part}')
                core = propose(*(row['predictions'] for row in pred), nframes, masks['controls'][keys[0]])
                result.append(dict(role=role, fold=fold, part=part, clip=clip, nframes=nframes,
                                   anchor_seed=anchor, buddy_seeds=[anchor + 1, anchor + 2],
                                   prediction_sha256={str(s): masks['prediction_sha256'][k] for s,k in zip(seeds,keys)},
                                   mot_sha256={str(s): row['mot_sha256'] for s,row in zip(seeds,pred)}, **core))
    return result


def freeze(input_dir, prereg):
    if not re.fullmatch(r'[0-9a-f]{40}', prereg):
        raise ValueError('Expected the complete preregistration commit hash')
    cases = []
    for role, checksum in PINNED.items():
        raw = (input_dir / f'{role}_masks.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError(f'{role} input is not the original GT-free §108 mask')
        cases.extend(make_cases(role, json.loads(raw)))
    return dict(study=110, prereg=prereg, fps=25, inputs_sha256=PINNED, cases=cases,
                event_fields=['resolved_id','signed_clip_frame','x1','y1','x2','y2'],
                xyxy_unit='pixel', donor_representative='b',
                versions={'numpy': np.__version__, 'scipy': scipy.__version__})


def wire(data):
    return (json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def write_once(output, data):
    raw = wire(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        part = Path(temp) / 'freeze.json'
        part.write_bytes(raw)
        try:
            os.link(part, output)
        except FileExistsError:
            if output.read_bytes() != raw:
                raise ValueError(f'{output} already contains a different freeze')
    return hashlib.sha256(raw).hexdigest()


def from_public(archive_dir, output_dir):
    """Restore only the original §108 prediction-mask subdocuments for geometry replay."""
    for role, checksum in PINNED.items():
        with gzip.open(archive_dir / f'study108_{role}.json.gz', 'rt', encoding='utf-8') as handle:
            archive = json.load(handle)
        masks = archive['masks']
        if archive['masks_sha256'] != checksum or hashlib.sha256(wire(masks)).hexdigest() != checksum:
            raise ValueError(f'Archived {role} prediction-only manifest differs from the original freeze')
        path = output_dir / f'{role}_masks.json'
        print(f'{write_once(path, masks)} {path}')


def synthetic():
    # B11 loses an A:B tie. C21 has an edge to B11 but none to A, so cannot rescue B11.
    a = [[1,10,0,0,2,1], [2,-10,0,0,2,1]]
    b = [[10,10,0,0,2,1], [11,15,0,0,2,1], [12,-15,0,0,1,1],
         [13,36,0,0,2,1], [14,-46,0,0,2,1], [15,60,0,0,2,1], [16,60,0,0,2,1]]
    c = [[21,20,0,0,2,1], [22,41,0,0,1,1], [23,-51,0,0,.99,1],
         [24,-60,0,0,2,1], [25,60,0,0,2,1], [26,60,0,0,2,1]]
    old = {'retained':[1,2], 'pairs':[[[1,10],[2,12]],[]]}
    result = propose(a, b, c, 1000, old)
    assert result['kept'] == a
    assert [(x['b'][0], x['c'][0]) for x in result['donors']] == [(13,22),(15,25),(16,26)]
    assert result['counts']['D'] == {'in':3, 'out':0}
    assert propose(a[::-1], b[::-1], c[::-1], 1000, old) == result
    empty = {'retained':[], 'pairs':[[],[]]}
    assert propose([], [], [], 1000, empty)['donors'] == []
    assert propose([], b, [], 1000, empty)['counts']['D'] == {'in':0, 'out':0}
    assert propose([], [], c, 1000, empty)['counts']['K'] == {'in':0, 'out':0}
    negative = propose([], [[2,-6,.01,0,2.01,1]], [[9,-1,.01,0,1.01,1]], 1000, empty)
    assert negative['donors'][0]['direction'] == 'out' and negative['donors'][0]['b'][1] == -6
    assert propose([], [[2,1,0,0,1,1]], [[9,1000,0,0,1,1]], 1000, empty)['donors'] == []
    # A greedy B:C edge choice would take 1:3 and lose 2 entirely.
    cardinal = propose([], [[1,15,0,0,1,1],[2,10,0,0,1,1]],
                       [[3,10,0,0,1,1],[4,20,0,0,1,1]], 1000, empty)
    assert [(d['b'][0],d['c'][0]) for d in cardinal['donors']] == [(1,4),(2,3)]
    print('Synthetic aligned support, any-edge veto, centipixel cutoffs, direction, ties, cardinality and empties passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode', required=True)
    subs.add_parser('synthetic')
    op = subs.add_parser('from-public', help='restore pinned §108 prediction-only manifests for retrospective replay')
    op.add_argument('archive_dir', type=Path)
    op.add_argument('output_dir', type=Path)
    op = subs.add_parser('freeze')
    op.add_argument('input_dir', type=Path, help='directory with pinned source_masks.json and target_masks.json')
    op.add_argument('output', type=Path)
    op.add_argument('--prereg', required=True)
    args = parser.parse_args()
    if args.mode == 'synthetic':
        synthetic()
    elif args.mode == 'from-public':
        from_public(args.archive_dir, args.output_dir)
    else:
        record = freeze(args.input_dir, args.prereg)
        print(f'{len(record["cases"])} cases {write_once(args.output, record)} {args.output}')
