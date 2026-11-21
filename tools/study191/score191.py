"""§191 fixed-exposure CS/VT factorial count-team scoring; CPU only."""
from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import hashlib
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import contextmanager

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from mix_check import compare

PREREG = '8ceeeae5f749e348f4a9b760b65639068bbfe436'
EVIDENCE190_SHA256 = 'cdf5a77fef156b947129823838a0ca8515dbfca535358350b63a1fbfa86829ab'
REFERENCE = {'33': {'345': {'CS': 19.92, 'VT': 20.21}, '678': {'CS': 20.56, 'VT': 20.54},
                    'mean': {'CS': 20.24, 'VT': 20.37}},
             '35': {'345': {'CS': 6.25, 'VT': 6.72}, '678': {'CS': 6.49, 'VT': 7.18},
                    'mean': {'CS': 6.37, 'VT': 6.95}}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wire(value):
    if isinstance(value, np.ndarray):
        return wire(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(v) for v in value]
    return value


def ratio(value):
    return dict(numerator=value.numerator, denominator=value.denominator, value=float(value))


def save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(wire(value), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')


@contextmanager
def publication(dest):
    if dest.exists():
        raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(prefix='.score191-', dir=dest.parent) as name:
        work = Path(name) / 'payload'; work.mkdir()
        yield work
        if dest.exists():
            raise FileExistsError(dest)
        work.rename(dest)


def f1_fraction(member, threshold):
    metric = member['iou'][threshold]
    return Fraction(2 * metric['tp'], max(2 * metric['tp'] + metric['fp'] + metric['fn'], 1))


def group_report(members, seeds):
    """members[(seed,CS/VT)] are independently scored whole native MOT streams."""
    truth = np.asarray(members[seeds[0], 'CS']['integer_bins']['gt'], np.int64)
    assert all(np.array_equal(m['integer_bins']['gt'], truth) for m in members.values())
    cs = np.stack([members[s, 'CS']['integer_bins']['pred'] for s in seeds]).astype(np.int64)
    vt = np.stack([members[s, 'VT']['integer_bins']['pred'] for s in seeds]).astype(np.int64)
    calculation = compare(cs, vt, truth)
    denominator = 18 * int(truth.sum())
    assert denominator > 0
    teams = {}
    for team in calculation['teams']:
        bits = team['arms']
        name = ''.join('V' if bit else 'C' for bit in bits)
        chosen = [members[s, 'VT' if bit else 'CS'] for s, bit in zip(seeds, bits)]
        pred_sum = sum(np.asarray(m['integer_bins']['pred'], np.int64) for m in chosen)
        numerator = int(np.abs(pred_sum - 3 * truth).sum())
        assert 6 * numerator == int(team['error18'].sum())
        assert denominator == 18 * 30 * chosen[0]['nG']
        teams[name] = dict(arms=['VT' if b else 'CS' for b in bits],
                           E3=dict(numerator=numerator, denominator=3 * int(truth.sum()), W_percent=100 * numerator / (3 * int(truth.sum()))),
                           mean_member_W=ratio(sum((Fraction(int(m['window']['numerator']) * 100, int(m['window']['denominator'])) for m in chosen), Fraction()) / 3),
                           mean_member_F1={t: ratio(sum((f1_fraction(m, t) for m in chosen), Fraction()) / 3) for t in ('0.5', '0.3')})
    pure = [name for name in teams if len(set(name)) == 1]
    mixed = [name for name in teams if len(set(name)) == 2]
    assert len(pure) == 2 and len(mixed) == 6
    pure_W = Fraction(calculation['pure_error18_total'] * 100, denominator)
    mixed_W = Fraction(calculation['mixed_error18_total'] * 100, denominator)
    gain = pure_W - mixed_W
    exposure = {}
    for grouping, names in [('pure', pure), ('mixed', mixed)]:
        slots = [(s, 'VT' if n[i] == 'V' else 'CS') for n in names for i, s in enumerate(seeds)]
        exposure[grouping] = {f'{s}_{arm}': slots.count((s, arm)) for s in seeds for arm in ('CS', 'VT')}
    assert all(v == 1 for v in exposure['pure'].values()) and all(v == 3 for v in exposure['mixed'].values())
    exact_means = {}
    for metric in ('W', 'F1_05', 'F1_03'):
        def member_value(key):
            m = members[key]
            return (Fraction(100 * int(m['window']['numerator']), int(m['window']['denominator']))
                    if metric == 'W' else f1_fraction(m, '0.5' if metric == 'F1_05' else '0.3'))
        means = {g: sum((member_value((s, arm)) * exposure[g][f'{s}_{arm}'] for s in seeds for arm in ('CS', 'VT')), Fraction()) /
                 sum(exposure[g].values()) for g in ('pure', 'mixed')}
        assert means['pure'] == means['mixed']
        exact_means[metric] = ratio(means['pure'])
    delta = vt.astype(object) - cs.astype(object)
    S = delta.sum(axis=0)
    a = np.abs(cs.astype(object).sum(axis=0) + vt.astype(object).sum(axis=0) - 6 * truth)
    b0_6 = np.abs(S)
    bi_6 = np.stack([np.abs(S - 2 * delta[i]) for i in range(3)])
    gain18 = calculation['pure_error18'] - calculation['mixed_error18']
    ceiling18 = np.maximum(0, 3 * b0_6 - bi_6.sum(axis=0))
    assert np.all(gain18 <= ceiling18)
    same = np.all(delta >= 0, axis=0) | np.all(delta <= 0, axis=0)
    diagnostics = {}
    for label, mask in [('same_direction', same), ('other', ~same)]:
        total_gain = int(gain18[mask].sum())
        total_cap = int(ceiling18[mask].sum())
        diagnostics[label] = dict(cells=int(mask.sum()), gain18=total_gain, score_free_ceiling18=total_cap,
                                  gain_pp=100 * total_gain / denominator,
                                  score_free_ceiling_pp=100 * total_cap / denominator)
    assert sum(d['gain18'] for d in diagnostics.values()) == calculation['pure_error18_total'] - calculation['mixed_error18_total']
    assert sum(d['score_free_ceiling18'] for d in diagnostics.values()) == int(ceiling18.sum())
    return wire(dict(seeds=list(seeds), teams=teams,
                     pure=dict(error18=calculation['pure_error18_total'], W_percent=float(pure_W)),
                     mixed=dict(error18=calculation['mixed_error18_total'], W_percent=float(mixed_W)),
                     improvement=dict(error18=int(gain18.sum()), denominator=denominator, pp=float(gain)),
                     exposure=exposure, exact_equal_member_means=exact_means,
                     score_free_ceiling=dict(error18=int(ceiling18.sum()), denominator=denominator, pp=100 * int(ceiling18.sum()) / denominator),
                     categories=diagnostics, all_zero_delta_cells=int(np.all(delta == 0, axis=0).sum()),
                     closed_form=dict(delta=delta, a6=a, b0_6=b0_6, bi_6=bi_6,
                                      pure_error18=calculation['pure_error18'], mixed_error18=calculation['mixed_error18'],
                                      gain18=gain18, score_free_ceiling18=ceiling18, same_direction=same)))


def average_groups(groups):
    assert set(groups) == {'345', '678'}
    output = {label: float(np.mean([g[label]['W_percent'] for g in groups.values()])) for label in ('pure', 'mixed')}
    output['improvement_pp'] = float(np.mean([g['improvement']['pp'] for g in groups.values()]))
    output['score_free_ceiling_pp'] = float(np.mean([g['score_free_ceiling']['pp'] for g in groups.values()]))
    output['CS'] = float(np.mean([g['teams']['CCC']['E3']['W_percent'] for g in groups.values()]))
    output['VT'] = float(np.mean([g['teams']['VVV']['E3']['W_percent'] for g in groups.values()]))
    output['categories'] = {name: {metric: float(np.mean([g['categories'][name][metric] for g in groups.values()]))
                                    for metric in ('gain_pp', 'score_free_ceiling_pp')} for name in ('same_direction', 'other')}
    assert abs(output['improvement_pp'] - sum(v['gain_pp'] for v in output['categories'].values())) < 1e-10
    return output


def freeze_reference(out, prereg):
    if PREREG is None or prereg != PREREG:
        raise RuntimeError('Scoring blocked pending published preregistration')
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    assert (out / 'ALL_FROZEN.sha256').read_text().strip() == sha(out / 'manifest.json')
    assert manifest['schema'] == 'mix191-all-frozen-v1' and manifest['prereg'] == prereg and manifest['complete'] is True
    expected = {(h, s, r) for h in (33, 35) for s in range(3, 9) for r in ('CS', 'VT')}
    assert {(x['hive'], x['seed'], x['recipe']) for x in manifest['streams']} == expected
    assert len(manifest['streams']) == len(expected)
    receipts, reference = {}, {}
    for stream in manifest['streams']:
        h, s, recipe = stream['hive'], stream['seed'], stream['recipe']
        key = f'h{h}_s{s}_{recipe}'
        assert stream['directory'] == key
        folder = out / key
        receipt = json.loads((folder / 'receipt.json').read_text(encoding='utf-8'))
        assert sha(folder / 'receipt.json') == stream['receipt_sha256'] == (folder / 'SEALED.sha256').read_text().strip()
        assert receipt['schema'] == 'mix191-stream-v1' and receipt['prereg'] == prereg
        assert (receipt['hive'], receipt['seed'], receipt['recipe'], receipt['clip'], receipt['nframes']) == (h, s, recipe, f'BEE24-{h}-head', 3950)
        assert set(receipt['files']) == {'mot.npy', 'mot.txt', 'events.npy', 'bins.npy', 'events.json'}
        for filename, digest in receipt['files'].items():
            assert Path(filename).name == filename and sha(folder / filename) == digest
        receipts[h, s, recipe] = receipt
        reference[key] = dict(receipt_sha256=sha(folder / 'receipt.json'), files=receipt['files'])
    return receipts, dict(prereg=prereg, all_frozen_sha256=sha(out / 'manifest.json'), streams=reference)


def score_member(gt, rows, polygon, nf):
    from tools.study117.tracking.score117 import score_case
    from tools.study103.swap_score import vector_of_counts
    from beetrack import p0
    from beetrack.evaluate import match_events
    score = score_case(gt, rows, polygon, nf)
    pred_events = p0.events(rows, polygon).astype(int)
    gt_events = p0.events(gt, polygon).astype(int)
    gt_bins, pred_bins = vector_of_counts(gt, polygon, nf), vector_of_counts(rows, polygon, nf)
    official = match_events(gt, rows, polygon)
    assert (official['gt'], official['pred'], official['tp']) == (len(gt_events), len(pred_events), score['iou']['0.5']['tp'])
    assert abs(score['w30_percent'] - 100 * p0.window_err(pred_events, gt_events, nf) / len(gt_events)) < 1e-9
    score.update(integer_bins=dict(gt=gt_bins.astype(int).tolist(), pred=pred_bins.astype(int).tolist()),
                 nG=len(gt_events), nP=len(pred_events), gt_events=gt_events.tolist(), pred_events=pred_events.tolist())
    return score


def previous_cs(source190, evidence190):
    """Read public numeric §190 evidence; private reconstruction is optional."""
    if source190 is not None:
        oldscore = source190 / 'score'
        oldhashes = json.loads((oldscore / 'result_hashes.json').read_text(encoding='utf-8'))
        assert (oldscore / 'SEALED.sha256').read_text().strip() == sha(oldscore / 'result_hashes.json')
        assert sha(oldscore / 'summary.json') == oldhashes['summary.json']
        previous = json.loads((oldscore / 'summary.json').read_text(encoding='utf-8'))
        return {key: value['base'] for key, value in previous['cells'].items()}, dict(
            source190_score_seal=sha(oldscore / 'result_hashes.json'),
            source190_summary_sha256=sha(oldscore / 'summary.json'), mode='optional_native_history')
    assert evidence190 is not None and sha(evidence190) == EVIDENCE190_SHA256
    evidence = json.loads(gzip.decompress(evidence190.read_bytes()))
    expected = {f'h{h}_s{s}' for h in (33, 35) for s in range(3, 9)}
    assert set(evidence['cells']) == expected
    return {key: value['scores']['base'] for key, value in evidence['cells'].items()}, dict(
        source190_evidence_sha256=sha(evidence190), source190_score_seal=evidence['score_hashes_sha256'],
        mode='published_compact_numeric_history')


def run_field(repo, out, prereg, source190=None, evidence190=None):
    receipts, reference = freeze_reference(out, prereg)
    # Every prediction seal is verified before any target GT or prior score opens.
    for receipt in receipts.values():
        assert sha(repo / receipt['source_det_path']) == receipt['source_det_sha256']
    previous, previous_reference = previous_cs(source190, evidence190 or repo / 'tools/study190/evidence.json.gz')
    reference['source190'] = previous_reference
    from beetrack import p0
    members = {}
    with publication(out / 'score') as dest:
        save(dest / 'freeze_reference.json', reference)
        for (h, s, recipe), receipt in sorted(receipts.items()):
            key = f'h{h}_s{s}_{recipe}'
            folder = out / key
            gtpath = repo / 'runs/gt/dev' / receipt['clip'] / 'gt/gt.txt'
            gt = np.loadtxt(gtpath, delimiter=',', usecols=range(6), ndmin=2)
            rows = np.load(folder / 'mot.npy', allow_pickle=False)
            assert rows.ndim == 2 and rows.shape[1] == 7
            value = score_member(gt, rows, receipt['polygon'], 3950)
            assert np.array_equal(np.asarray(value['pred_events'], int), np.load(folder / 'events.npy', allow_pickle=False))
            assert np.array_equal(np.asarray(value['integer_bins']['pred'], int), np.load(folder / 'bins.npy', allow_pickle=False))
            records = json.loads((folder / 'events.json').read_text(encoding='utf-8'))
            native_signed = {r['track_id']: int(r['frame']) * int(r['direction']) for r in records}
            assert native_signed == {r[0]: r[1] for r in value['evidence']['predictions']}
            assert receipt['polygon'] == p0.ZONE[receipt['clip']]
            if recipe == 'CS':
                old = previous[f'h{h}_s{s}']
                assert value['integer_bins'] == old['integer_bins']
                assert abs(value['w30_percent'] - old['w30_percent']) < 1e-10
                for t in ('0.5', '0.3'):
                    assert all(value['iou'][t][metric] == old['iou'][t][metric] for metric in ('tp', 'fp', 'fn', 'f1'))
                if source190 is not None:
                    assert sha(folder / 'mot.npy') == sha(source190 / f'h{h}_s{s}' / 'mot.npy')
            value['integrity'] = dict(gt_sha256=sha(gtpath), native_mot_npy_sha256=sha(folder / 'mot.npy'),
                                      receipt_sha256=sha(folder / 'receipt.json'), polygon=receipt['polygon'])
            save(dest / f'{key}.json', value)
            members[h, s, recipe] = value
            print(f'SCORED {key}', flush=True)
        heads = {}
        for h in (33, 35):
            groups = {}
            for name, seeds in [('345', (3, 4, 5)), ('678', (6, 7, 8))]:
                group_members = {(s, recipe): members[h, s, recipe] for s in seeds for recipe in ('CS', 'VT')}
                groups[name] = group_report(group_members, seeds)
                for recipe, team in [('CS', 'CCC'), ('VT', 'VVV')]:
                    assert abs(groups[name]['teams'][team]['E3']['W_percent'] - REFERENCE[str(h)][name][recipe]) <= .0051
                save(dest / f'h{h}_g{name}.json', groups[name])
            head = dict(groups=groups, mean=average_groups(groups))
            for recipe in ('CS', 'VT'):
                assert abs(head['mean'][recipe] - REFERENCE[str(h)]['mean'][recipe]) <= .0051
            heads[str(h)] = head
        summary = dict(schema='mix191-score-v1', prereg=prereg, heads=heads,
                       members={f'h{h}_s{s}_{r}': v for (h, s, r), v in members.items()},
                       published_reference=REFERENCE,
                       interpretation='Known-development fixed-exposure recipe team-composition comparison; E3 is a count average without identity F1; no causal identification of shared-label errors.')
        save(dest / 'summary.json', summary)
        hashes = {p.relative_to(dest).as_posix(): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
        save(dest / 'result_hashes.json', hashes)
        (dest / 'SEALED.sha256').write_text(sha(dest / 'result_hashes.json') + '\n')
    return out / 'score/summary.json'


def self_test():
    from tools.study117.tracking.score117 import score_case
    from tools.study103.swap_score import vector_of_counts
    poly = [[0, 0], [10, 0], [10, 30], [0, 30]]
    gt = np.array([[749, 1, 14, 5, 2, 7], [750, 1, 6, 5, 2, 7]], float)
    def member(rows):
        m = score_case(gt, rows, poly, 790)
        m['integer_bins'] = dict(gt=vector_of_counts(gt, poly, 790).tolist(), pred=vector_of_counts(rows, poly, 790).tolist())
        m['nG'] = 1
        return m
    a = member(gt)
    delayed = gt.copy(); delayed[:, 0] += 1
    b = member(delayed)
    members = {(s, arm): a if arm == 'CS' else b for s in (3, 4, 5) for arm in ('CS', 'VT')}
    report = group_report(members, (3, 4, 5))
    assert report['exact_equal_member_means']['F1_05']['value'] == float((f1_fraction(a, '0.5') + f1_fraction(b, '0.5')) / 2)
    assert len(report['teams']) == 8
    json.dumps(report, allow_nan=False)
    print('score191 exact exposure, team enumeration and closed-form diagnostics synthetic checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--source190', type=Path, help='Optional sealed native §190 reconstruction for extra byte checks')
    parser.add_argument('--evidence190', type=Path, help='Published tools/study190/evidence.json.gz (default under --repo)')
    parser.add_argument('--prereg')
    args = parser.parse_args()
    if args.repo:
        sys.path.insert(0, str(args.repo.resolve()))
    if args.self_test or not args.out:
        self_test()
    else:
        if PREREG is None or args.prereg != PREREG:
            parser.error('Require published registered --prereg')
        if not args.repo or args.out.resolve().drive.upper() != 'D:':
            parser.error('Require --repo and D: --out')
        print(run_field(args.repo.resolve(), args.out.resolve(), args.prereg,
                        args.source190.resolve() if args.source190 else None,
                        args.evidence190.resolve() if args.evidence190 else None))
