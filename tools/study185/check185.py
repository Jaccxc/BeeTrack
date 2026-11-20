"""Independent frozen-MOT verifier and compact integer evidence replay for §185.

Uses canonical track splitting, zone geometry and frame assignment only; event
graphs and the 30 shifted windows are reconstructed independently of the runner.
"""
import argparse
import gzip
import configparser
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from beetrack.evaluate import by_track, frame_match, zone_tracks, ZONES


def cardinality(edges, n_pred, n_gt):
    adjacency = [[] for _ in range(n_pred)]
    for p, g in edges:
        assert 0 <= p < n_pred and 0 <= g < n_gt
        adjacency[p].append(g)
    owner = {}

    def augment(p, seen):
        for g in adjacency[p]:
            if g in seen:
                continue
            seen.add(g)
            if g not in owner or augment(owner[g], seen):
                owner[g] = p
                return True
        return False

    return sum(augment(p, set()) for p in range(n_pred))


def phase_counts(events, nframes):
    result = []
    for shift in range(30):
        counts = [[0, 0] for _ in range((nframes + shift * 25 + 749) // 750)]
        for frame, direction in events:
            assert 1 <= frame <= nframes and direction in ('in', 'out')
            counts[(frame - 1 + shift * 25) // 750][direction == 'out'] += 1
        result.append(counts)
    return result


def replay(case):
    g, p = case['gt_counts'], case['pred_counts']
    if 'gt_events' in case:
        assert phase_counts(case['gt_events'], case['nframes']) == g
        assert phase_counts(case['pred_events'], case['nframes']) == p
    assert len(g) == len(p) == 30
    denominator = sum(sum(row) for phase in g for row in phase)
    assert denominator == 30 * case['nG'] and denominator > 0
    assert sum(sum(row) for phase in p for row in phase) == 30 * case['nP']
    numerator = 0
    for a, b in zip(g, p):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert len(x) == len(y) == 2
            assert all(isinstance(v, int) and v >= 0 for v in x + y)
            numerator += sum(abs(u - v) for u, v in zip(x, y))
    scores = {}
    for threshold, edges in case['edges'].items():
        tp = cardinality(edges, case['nP'], case['nG'])
        scores[threshold] = dict(tp=tp, fp=case['nP'] - tp,
                                 fn=case['nG'] - tp,
                                 f1=2 * tp / max(case['nP'] + case['nG'], 1))
    return dict(W=100 * numerator / denominator, absolute_error=numerator,
                denominator=denominator, nG=case['nG'], nP=case['nP'], scores=scores)


def verify_case(gt_path, mot_path, polygon, nframes):
    def read(path):
        path = Path(path)
        return (np.loadtxt(path, delimiter=',', ndmin=2)[:, :6]
                if path.read_bytes().strip() else np.empty((0, 6)))
    gt, pred = (by_track(read(p)) for p in (gt_path, mot_path))
    ge = {k: v for k, v in zone_tracks(gt, polygon).items() if v[0]}
    pe = {k: v for k, v in zone_tracks(pred, polygon).items() if v[0]}

    def times(rows, events):
        return [[int(rows[next(j for j, d in reversed(v[3]) if d == v[0]), 0]), v[0]]
                for v in events.values()]

    gt_events, pred_events = times(gt, ge), times(pred, pe)
    case = dict(nG=len(ge), nP=len(pe), nframes=int(nframes),
                gt_ids=list(ge), pred_ids=list(pe), gt_events=gt_events, pred_events=pred_events,
                gt_counts=phase_counts(gt_events, nframes),
                pred_counts=phase_counts(pred_events, nframes), edges={})
    columns = {tid: c for c, tid in enumerate(ge)}
    for threshold in (.5, .3):
        owners = frame_match(gt, pred, thr=threshold)
        edges = set()
        for r, event in enumerate(pe.values()):
            direction = event[0]
            for row, d in event[3]:
                if d != direction:
                    continue
                for tid in (owners[row - 1], owners[row]):
                    if tid not in ge or ge[tid][0] != direction:
                        continue
                    if any(gd == direction and abs(gt[j, 0] - pred[row, 0]) <= 5
                           for j, gd in ge[tid][3]):
                        edges.add((r, columns[tid]))
        case['edges'][str(threshold)] = [list(e) for e in sorted(edges)]
    case['sha256'] = {kind: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                      for kind, path in [('gt', gt_path), ('mot', mot_path)]}
    case['result'] = replay(case)
    return case


def assert_counts_equal(detections, mask_m, mask_c):
    """Check retained detector counts, before tracking can discard observations."""
    frames = np.asarray(detections)[:, 0].astype(int)
    m, c = np.asarray(mask_m, bool), np.asarray(mask_c, bool)
    assert m.shape == c.shape == frames.shape
    assert np.array_equal(np.bincount(frames[m]), np.bincount(frames[c]))


def verify_run(repo, out):
    """Read physical files only after all six freeze receipts exist and match."""
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    summary = json.loads((out / 'score/summary.json').read_text(encoding='utf-8'))
    expected = {(c['hive'], c['seed'], c['arm']): c for c in summary['cases']}
    assert set(expected) == {(h, s, a) for h in (33, 35) for s in range(3) for a in 'HLMC'}
    receipts = {}
    # Do not open any target GT before checking every physical freeze seal.
    for hive in (33, 35):
        for seed in range(3):
            root = out / f'freeze{hive}_s{seed}'
            receipt = json.loads((root / 'receipt.json').read_text(encoding='utf-8'))
            assert (root / 'SEALED.sha256').read_text().strip() == sha(root / 'receipt.json')
            assert receipt['prereg'] == summary['prereg']
            assert (receipt['hive'], receipt['seed'], receipt['clip']) == (hive, seed, f'BEE24-{hive}-head')
            assert set(receipt['arms']) == set('HLMC')
            for arm in 'HLMC':
                assert sha(root / f'{arm}.txt') == receipt['arms'][arm]['mot_sha256']
                assert sha(root / f'{arm}.mask.npy') == receipt['arms'][arm]['mask_sha256']
            receipts[hive, seed] = receipt
    zones = json.loads(ZONES.read_text(encoding='utf-8'))
    evidence = dict(schema='bridge185-independent-v1', prereg=summary['prereg'], cases={}, budgets={})
    for (hive, seed), receipt in receipts.items():
        root = out / f'freeze{hive}_s{seed}'
        clip = receipt['clip']
        detpath = repo / 'runs/dets' / f'loho_no{hive}_s{seed}' / 'dev' / f'{clip}.npy'
        assert sha(detpath) == receipt['det_sha256']
        detections = np.load(detpath, allow_pickle=False)
        masks = {a: np.load(root / f'{a}.mask.npy', allow_pickle=False) for a in 'HLMC'}
        assert all(m.dtype == bool and m.shape == (len(detections),) for m in masks.values())
        assert np.array_equal(masks['H'], detections[:, 5] >= .4)
        assert np.array_equal(masks['L'], detections[:, 5] >= .25)
        assert all(np.all(masks['H'] <= masks[a]) and np.all(masks[a] <= masks['L']) for a in 'MC')
        assert_counts_equal(detections, masks['M'], masks['C'])
        frames = detections[:, 0].astype(int)
        evidence['budgets'][f'{hive}_s{seed}'] = np.bincount(frames[masks['M']], minlength=3951).tolist()
        gtroot = repo / 'runs/gt/dev' / clip
        ini = configparser.ConfigParser()
        ini.read(gtroot / 'seqinfo.ini', encoding='utf-8')
        assert ini.getint('Sequence', 'frameRate') == 25
        nframes = ini.getint('Sequence', 'seqLength')
        assert nframes == 3950
        for arm in 'HLMC':
            key = f'{hive}_s{seed}_{arm}'
            case = verify_case(gtroot / 'gt/gt.txt', root / f'{arm}.txt', zones[clip], nframes)
            actual, ref = case['result'], expected[hive, seed, arm]
            assert actual['nG'] == ref['nG'] and actual['nP'] == ref['nP'], key
            assert abs(actual['W'] - ref['W_percent']) < 1e-9, key
            assert abs(actual['absolute_error'] / 30 - ref['err']) < 1e-9, key
            for threshold in ('0.5', '0.3'):
                for metric in ('tp', 'fp', 'fn', 'f1'):
                    assert abs(actual['scores'][threshold][metric] - ref['iou'][threshold][metric]) < 1e-9, (key, threshold, metric)
            assert actual['scores']['0.5']['tp'] == ref['tp'], key
            assert abs(actual['scores']['0.5']['f1'] - ref['f1']) < 1e-9, key
            assert case['sha256']['gt'] == ref['gt_sha256'], key
            assert sha(root / 'receipt.json') == ref['receipt_sha256'], key
            evidence['cases'][key] = case
    return evidence


def self_check():
    assert cardinality([[0, 0], [0, 1], [1, 0]], 2, 2) == 2
    assert cardinality([], 0, 3) == 0
    g = phase_counts([[750, 'in']], 790)
    p = phase_counts([[751, 'in']], 790)
    c = dict(nG=1, nP=1, gt_counts=g, pred_counts=p, edges={'0.5': [[0, 0]]})
    assert replay(c)['absolute_error'] == 2
    assert replay(c)['scores']['0.5']['f1'] == 1
    assert_counts_equal(np.array([[1], [1], [2]]), [1, 0, 1], [0, 1, 1])
    with tempfile.TemporaryDirectory(prefix='check185-synthetic-') as directory:
        root = Path(directory)
        gt = np.array([[747, 1, 14, 5, 2, 7], [748, 1, 6, 5, 2, 7],
                       [749, 1, 14, 5, 2, 7], [750, 1, 6, 5, 2, 7]], float)
        pred = gt.copy()
        pred[:, 3] += 3
        np.savetxt(root / 'gt.txt', gt, delimiter=',')
        np.savetxt(root / 'pred.txt', pred, delimiter=',')
        case = verify_case(root / 'gt.txt', root / 'pred.txt', [[0, 0], [10, 0], [10, 30], [0, 30]], 790)
        assert case['result']['W'] == 0
        assert case['result']['scores']['0.5']['tp'] == 0
        assert case['result']['scores']['0.3']['tp'] == 1
        assert replay(json.loads(json.dumps(case))) == case['result']
        (root / 'pred.txt').write_text('')
        assert verify_case(root / 'gt.txt', root / 'pred.txt', [[0, 0], [10, 0], [10, 30], [0, 30]], 790)['result']['W'] == 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    self_check()
    if args.repo or args.out or args.evidence:
        assert args.repo and args.out and args.evidence
        assert not args.evidence.exists(), 'Evidence destination already exists'
        evidence = verify_run(args.repo, args.out)
        args.evidence.write_text(json.dumps(evidence, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
        print(f"Verified {len(evidence['cases'])} frozen MOT cases and six M/C budgets")
    elif args.replay:
        raw = args.replay.read_bytes()
        evidence = json.loads(gzip.decompress(raw) if args.replay.suffix == '.gz' else raw)
        for key, case in evidence['cases'].items():
            assert replay(case) == case['result'], key
        print(f"Replayed {len(evidence['cases'])} cases")
    else:
        print('check185 synthetic checks passed')
