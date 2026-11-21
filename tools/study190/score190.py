"""CPU-only §190 scoring, compact evidence, and same-three-seed aggregation.

Field coordinator is gated by a published preregistration and twelve complete
prediction freezes. Pure helpers below accept arrays and perform no field reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from contextlib import contextmanager
import tempfile

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np

PREREG = 'aeaf95b76e99baef940dddbdbdb91d33611bf3dd'
REFERENCE = {'35': {'345': 6.25, '678': 6.49, 'mean_E3': 6.37},
             '33': {'345': 19.92, '678': 20.56, 'mean_E3': 20.24}}


@contextmanager
def publication(dest):
    if dest.exists():
        raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(prefix='.score190-', dir=dest.parent) as tmp:
        payload = Path(tmp) / 'payload'
        payload.mkdir()
        yield payload
        if dest.exists():
            raise FileExistsError(dest)
        payload.rename(dest)


def save_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_mot(path):
    path = Path(path)
    return np.loadtxt(path, delimiter=',', ndmin=2) if path.read_bytes().strip() else np.empty((0, 7))


def write_exact_mot(path, rows):
    """17 significant digits round-trip native float64; score no rounded boxes."""
    rows = np.asarray(rows, float).reshape(-1, 7)
    np.savetxt(path, rows, delimiter=',', fmt='%.17g')
    assert np.array_equal(physical_mot(path), rows)


def native_identity_map(raw, canonical):
    from beetrack.evaluate import by_track
    indexed = np.column_stack((raw, np.arange(len(raw))))
    resolved = by_track(indexed)
    ids = np.empty(len(raw))
    ids[resolved[:, 7].astype(int)] = resolved[:, 1]
    mapping = {}
    for segment in np.unique(canonical[:, 1]).astype(int):
        members = ids[canonical[:, 1] == segment]
        assert len(np.unique(members)) == 1
        mapping[int(segment)] = int(members[0])
    assert len(set(mapping.values())) == len(mapping), 'Official resolved native ID collision'
    return mapping


def remap_ids(rows, mapping):
    result = np.array(rows, copy=True)
    result[:, 1] = [mapping[int(i)] for i in rows[:, 1]]
    return result


def official_anchor_owners(gt, raw, canonical, mapping):
    from beetrack.evaluate import by_track, frame_match
    G = by_track(gt)
    indexed = np.column_stack((raw, np.arange(len(raw))))
    official = by_track(indexed)
    mapped = by_track(np.column_stack((remap_ids(canonical, mapping), np.arange(len(raw)))))
    assert np.array_equal(official[:, 7], mapped[:, 7]), 'Native frame-matching row permutation changed'
    assert np.array_equal(official[:, :7], mapped[:, :7])
    owners_sorted = frame_match(G, official)
    owners = np.empty(len(raw), dtype=np.int64)
    owners[official[:, 7].astype(int)] = owners_sorted
    assert np.array_equal(frame_match(G, mapped), owners_sorted)
    plain = frame_match(G, raw)
    return owners, int(np.count_nonzero(plain != owners))


def whole_event_map(rows, polygon):
    from beetrack.evaluate import by_track, zone_tracks
    rows = by_track(rows)
    return {int(identity): (int(rows[next(j for j, d in reversed(info[3]) if d == info[0]), 0]), info[0])
            if info[0] else None for identity, info in zone_tracks(rows, polygon).items()}


def score_arrays(gt, mot, polygon, nframes):
    """Recompute native events/graphs; cross-check separately implemented scores."""
    from tools.study117.tracking.score117 import score_case
    from tools.study103.swap_score import event_scores, vector_of_counts, window_error
    result = score_case(gt, mot, polygon, nframes)
    native = event_scores(gt, mot, polygon)
    g, p = vector_of_counts(gt, polygon, nframes), vector_of_counts(mot, polygon, nframes)
    assert abs(result['w30_percent'] - window_error(g, p)) < 1e-10
    for threshold in (.5, .3):
        assert all(abs(result['iou'][str(threshold)][k] - v) < 1e-10
                   for k, v in native[threshold].items())
    result['integer_bins'] = dict(gt=g.astype(int).tolist(), pred=p.astype(int).tolist())
    result['nG'] = sum(result['gt_counts'].values())
    result['nP'] = sum(result['pred_counts'].values())
    return result


def e3(cases):
    """Window error of the average of exactly these three seed count streams."""
    assert len(cases) == 3
    gt = np.asarray(cases[0]['integer_bins']['gt'], dtype=np.int64)
    assert all(np.array_equal(np.asarray(c['integer_bins']['gt']), gt) for c in cases)
    pred_sum = sum(np.asarray(c['integer_bins']['pred'], dtype=np.int64) for c in cases)
    numerator = int(np.abs(pred_sum - 3 * gt).sum())
    denominator = int(3 * gt.sum())
    assert denominator > 0
    return dict(W_percent=100 * numerator / denominator, numerator=numerator,
                denominator=denominator, pred_sum=pred_sum.tolist(), gt_tripled=(3 * gt).tolist())


def aggregate(cases):
    """Input cells keyed (hive,seed), each containing base/witness score results."""
    result = {}
    for hive in (33, 35):
        heads = {}
        for arm in ('base', 'witness'):
            values = [cases[hive, seed][arm] for seed in range(3, 9)]
            groups = {name: e3([cases[hive, seed][arm] for seed in seeds])
                      for name, seeds in [('345', (3, 4, 5)), ('678', (6, 7, 8))]}
            heads[arm] = dict(mean_single_W=float(np.mean([v['w30_percent'] for v in values])),
                              mean_F1_05=float(np.mean([v['iou']['0.5']['f1'] for v in values])),
                              mean_F1_03=float(np.mean([v['iou']['0.3']['f1'] for v in values])),
                              E3=groups, mean_E3=float(np.mean([v['W_percent'] for v in groups.values()])))
        heads['witness_gain'] = {metric: heads['base'][metric] - heads['witness'][metric]
                                 for metric in ('mean_single_W', 'mean_E3')}
        result[str(hive)] = heads
    return result


def validate_witness(base, witness, participants):
    """Original serialized observations remain an exact prefix; only participants gain rows."""
    base, witness = np.asarray(base), np.asarray(witness)
    assert base.shape[1] == witness.shape[1] == 7
    assert len(witness) >= len(base) and np.array_equal(base, witness[:len(base)])
    added = witness[len(base):]
    assert set(added[:, 1].astype(int)) <= set(map(int, participants))
    assert len({(int(r[1]), int(r[0])) for r in witness}) == len(witness)
    counts = {}
    for segment in np.unique(base[:, 1]).astype(int):
        old, new = base[base[:, 1] == segment], witness[witness[:, 1] == segment]
        if segment not in participants:
            assert np.array_equal(old, new)
        counts[str(segment)] = dict(original=len(old), added=len(new) - len(old))
    return counts


def assert_reference(aggregation, reference):
    """Published rounded CS-C numbers supplied in preregistered immutable metadata."""
    for hive, fields in reference.items():
        base = aggregation[str(hive)]['base']
        for metric, value in fields.items():
            actual = base['E3'][metric]['W_percent'] if metric in ('345', '678') else base[metric]
            assert abs(actual - value) <= .0051, (hive, metric, actual, value)


def feasibility_gate(cases, changed_gt_ids):
    """Registered gate uses single-seed means; bounds and E3 cannot pass it."""
    result = {}
    for hive in (33, 35):
        group_gains = {}
        for name, seeds in [('345', (3, 4, 5)), ('678', (6, 7, 8))]:
            group_gains[name] = float(np.mean([cases[hive, s]['base']['w30_percent'] -
                                               cases[hive, s]['witness']['w30_percent'] for s in seeds]))
        gain = float(np.mean(list(group_gains.values())))
        f1_delta = float(np.mean([cases[hive, s]['witness']['iou']['0.5']['f1'] -
                                  cases[hive, s]['base']['iou']['0.5']['f1'] for s in range(3, 9)]))
        # The same prepared GT segment seen across detector seeds counts once.
        identities = sorted(set(map(int, changed_gt_ids[hive])))
        checks = dict(mean_single_gain_2pp=gain >= 2, official_f1_nondecrease=f1_delta >= 0,
                      both_groups_positive=all(v > 0 for v in group_gains.values()),
                      at_least_5_changed_gt_identities=len(identities) >= 5)
        result[str(hive)] = dict(pass_gate=all(checks.values()), checks=checks,
                                 mean_single_gain=gain, official_f1_delta=f1_delta,
                                 group_single_gains=group_gains, changed_gt_segment_ids=identities)
    result['scope'] = ('potential_shared_feasibility' if all(result[str(h)]['pass_gate'] for h in (33, 35))
                       else 'dense_feasibility_only' if result['33']['pass_gate']
                       else 'registered_feasibility_not_met')
    return result


def freeze_manifest(out, prereg):
    """Verify every prediction artifact before any GT-loading function is called."""
    if prereg != PREREG:
        raise RuntimeError('Unregistered scoring invocation')
    global_manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    assert (out / 'ALL_FROZEN.sha256').read_text().strip() == sha(out / 'manifest.json')
    assert global_manifest['schema'] == 'emerge190-all-frozen-v1'
    assert global_manifest['prereg'] == prereg and global_manifest['complete'] is True
    assert len(global_manifest['streams']) == 12
    global_streams = {(r['hive'], r['seed']): r for r in global_manifest['streams']}
    receipts, manifest = {}, {}
    for hive in (33, 35):
        for seed in range(3, 9):
            folder = out / f'h{hive}_s{seed}'
            receipt = json.loads((folder / 'receipt.json').read_text(encoding='utf-8'))
            assert global_streams[hive, seed]['directory'] == folder.name
            assert global_streams[hive, seed]['receipt_sha256'] == sha(folder / 'receipt.json')
            assert (folder / 'SEALED.sha256').read_text().strip() == sha(folder / 'receipt.json')
            assert receipt['schema'] == 'emerge190-freeze-v1' and receipt['prereg'] == prereg
            assert (receipt['hive'], receipt['seed'], receipt['clip'], receipt['nframes']) == (hive, seed, f'BEE24-{hive}-head', 3950)
            for filename, digest in receipt['files'].items():
                assert Path(filename).name == filename and sha(folder / filename) == digest
            assert set(receipt['files']) == {'dets.npy', 'det_indices.npy', 'mot.npy', 'mot.txt', 'canonical.npy', 'membership.npy', 'row_keys.npy', 'candidates.json'}
            receipts[hive, seed] = receipt
            manifest[f'h{hive}_s{seed}'] = dict(receipt_sha256=sha(folder / 'receipt.json'), files=receipt['files'])
    return receipts, manifest


def bounds_for_cell(canonical, candidates, labels, polygon, nframes, gt_events):
    import bounds190 as B
    endpoint_rows = {}
    for option in labels['constrained_menu']:
        endpoint_rows.setdefault((int(option['B']), option['kind']), []).extend(option['rows'])
    output = {}
    for family, supplied in [('universal_all_side', None), ('gt_evidenced', endpoint_rows)]:
        menus = B.segment_menus(canonical, candidates, polygon, nframes, supplied)
        lower, upper = B.menu_extrema(menus, nframes)
        bound = B.interval_bound(lower, upper, gt_events, nframes)
        output[family] = dict(menus={str(k): v for k, v in menus.items()}, lower=lower.tolist(), upper=upper.tolist(), bound=bound)
    return output


def aggregate_bounds(cases, cell_bounds):
    import bounds190 as B
    report = {}
    for hive in (33, 35):
        head = {}
        for family in ('universal_all_side', 'gt_evidenced'):
            singles = [cell_bounds[hive, s][family]['bound']['W_lower_percent'] for s in range(3, 9)]
            groups = {}
            for name, seeds in [('345', (3, 4, 5)), ('678', (6, 7, 8))]:
                lower = sum(np.asarray(cell_bounds[hive, s][family]['lower'], np.int64) for s in seeds)
                upper = sum(np.asarray(cell_bounds[hive, s][family]['upper'], np.int64) for s in seeds)
                evidence = cases[hive, seeds[0]]['base']['evidence']
                events = [(abs(int(date)), 1 if date > 0 else -1) for _, date in evidence['gt']]
                bound = B.interval_bound(lower, upper, events, 3950, streams=3)
                bound['maximum_E3_gain'] = e3([cases[hive, s]['base'] for s in seeds])['W_percent'] - bound['W_lower_percent']
                groups[name] = bound
            mean_base = float(np.mean([cases[hive, s]['base']['w30_percent'] for s in range(3, 9)]))
            head[family] = dict(mean_single_W_lower=float(np.mean(singles)),
                                maximum_mean_single_gain=mean_base - float(np.mean(singles)),
                                E3=groups, mean_E3_W_lower=float(np.mean([g['W_lower_percent'] for g in groups.values()])),
                                maximum_mean_E3_gain=float(np.mean([g['maximum_E3_gain'] for g in groups.values()])))
        report[str(hive)] = head
    return report


def run_field(repo, out, prereg):
    from beetrack import p0
    from beetrack.common import gt_slice, split_specs
    from labels190 import gate_and_witness
    receipts, manifest = freeze_manifest(out, prereg)
    # Only prediction-side files and their immutable source checks precede this point.
    for receipt in receipts.values():
        source = repo / receipt['source_det_path']
        assert sha(source) == receipt['source_det_sha256']
    cases, cell_bounds, changed_gt_ids = {}, {}, {33: set(), 35: set()}
    with publication(out / 'score') as dest:
        save_json(dest / 'freeze_reference.json', dict(prereg=prereg, streams=manifest))
        for (hive, seed), receipt in receipts.items():
            folder = out / f'h{hive}_s{seed}'
            target = dest / f'h{hive}_s{seed}'
            target.mkdir()
            clip, nf = receipt['clip'], receipt['nframes']
            gt_path = repo / 'runs/gt/dev' / clip / 'gt/gt.txt'
            gt = physical_mot(gt_path)[:, :6]
            source, first, last = next((s, a, b) for ev, s, a, b in split_specs('dev') if ev == clip)
            source_gt = gt_slice(source, first, last)[:, :6]
            canonical = np.load(folder / 'canonical.npy', allow_pickle=False)
            raw = np.load(folder / 'mot.npy', allow_pickle=False)
            identity_map = native_identity_map(raw, canonical)
            members = np.load(folder / 'membership.npy', allow_pickle=False)
            assert np.array_equal(canonical[:, 1], members)
            assert np.array_equal(canonical[:, [0, 2, 3, 4, 5, 6]], raw[:, [0, 2, 3, 4, 5, 6]])
            candidates = json.loads((folder / 'candidates.json').read_text())['candidates']
            dets = np.load(folder / 'dets.npy', allow_pickle=False)
            owners, plain_difference = official_anchor_owners(gt, raw, canonical, identity_map)
            np.save(target / 'anchor_owners.npy', owners, allow_pickle=False)
            labels = gate_and_witness(canonical, dets, gt, source_gt, candidates, p0.ZONE[clip], owners=owners)
            participants = {int(r['B']) for r in labels['candidates'] if r.get('witness_status') == 'added'}
            unchanged = validate_witness(canonical, labels['rows'], participants)
            np.save(target / 'witness.npy', labels['rows'], allow_pickle=False)
            write_exact_mot(target / 'canonical_base.txt', canonical)
            write_exact_mot(target / 'canonical_witness.txt', labels['rows'])
            write_exact_mot(target / 'base.txt', remap_ids(canonical, identity_map))
            write_exact_mot(target / 'witness.txt', remap_ids(labels['rows'], identity_map))
            base, witness = physical_mot(target / 'base.txt')[:, :7], physical_mot(target / 'witness.txt')[:, :7]
            validate_witness(base, witness, {identity_map[i] for i in participants})
            scores = {name: score_arrays(gt, rows, p0.ZONE[clip], nf) for name, rows in [('base', base), ('witness', witness)]}
            raw_score = score_arrays(gt, raw, p0.ZONE[clip], nf)
            for metric in ('w30_percent', 'nG', 'nP'):
                assert scores['base'][metric] == raw_score[metric], (hive, seed, metric)
            assert scores['base']['iou'] == raw_score['iou'] or all(
                scores['base']['iou'][t][k] == raw_score['iou'][t][k] for t in ('0.5', '0.3') for k in ('tp', 'fp', 'fn', 'f1'))
            for arm, rows in [('base', base), ('witness', witness)]:
                official = p0.score(rows, 'dev', clip)
                value = scores[arm]
                assert abs(value['w30_percent'] - 100 * official['err'] / official['gt']) < 1e-9
                assert (value['nG'], value['nP'], value['iou']['0.5']['tp']) == (official['gt'], official['pred'], official['tp'])
            before, after = (whole_event_map(rows, p0.ZONE[clip]) for rows in (canonical, labels['rows']))
            changed = {int(i) for i in before if before[i] != after[i]}
            assert changed == set(labels['event_changed_segments']), 'Whole-event designation/date changed across implementations'
            ids = {int(r['query_gt_id']) for r in labels['candidates'] if r.get('witness_status') == 'added' and int(r['B']) in changed}
            assert ids == set(labels['event_changed_query_gt_ids'])
            changed_gt_ids[hive].update(ids)
            witness_rows = labels.pop('rows')
            save_json(target / 'labels.json', labels)
            events = [(abs(int(date)), 1 if date > 0 else -1) for _, date in scores['base']['evidence']['gt']]
            cell_bounds[hive, seed] = bounds_for_cell(canonical, candidates, labels, p0.ZONE[clip], nf, events)
            for family, family_bounds in cell_bounds[hive, seed].items():
                lower, upper = np.asarray(family_bounds['lower']), np.asarray(family_bounds['upper'])
                for arm in ('base', 'witness'):
                    counts = np.asarray(scores[arm]['integer_bins']['pred'])
                    assert np.all(lower <= counts) and np.all(counts <= upper), (hive, seed, family, arm)
                family_bounds['maximum_single_gain'] = scores['base']['w30_percent'] - family_bounds['bound']['W_lower_percent']
            save_json(target / 'bounds.json', cell_bounds[hive, seed])
            scores['integrity'] = dict(gt_sha256=sha(gt_path), base_sha256=sha(target / 'base.txt'),
                                       witness_sha256=sha(target / 'witness.txt'), original_rows=len(base),
                                       added_rows=len(witness) - len(base), participants=sorted(participants),
                                       event_changed_segments=sorted(changed), unchanged_segment_rows=unchanged,
                                       plain_native_vs_official_anchor_assignment_differences=plain_difference,
                                       compact_to_official_native_segment_id=identity_map)
            save_json(target / 'scores.json', scores)
            cases[hive, seed] = scores
            print(f'SCORED h{hive}_s{seed}', flush=True)
        aggregation = aggregate(cases)
        assert_reference(aggregation, REFERENCE)
        report = dict(prereg=prereg, schema='emerge190-score-v1', aggregation=aggregation,
                      bound_maximum_gains=aggregate_bounds(cases, cell_bounds),
                      gate=feasibility_gate(cases, changed_gt_ids), published_reference=REFERENCE,
                      cells={f'h{h}_s{s}': v for (h, s), v in cases.items()},
                      caveat='Known development scenes; bounded optional-endpoint edit family only; unknown GT support is not physical visibility.')
        save_json(dest / 'summary.json', report)
        digests = {p.relative_to(dest).as_posix(): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
        save_json(dest / 'result_hashes.json', digests)
        (dest / 'SEALED.sha256').write_text(sha(dest / 'result_hashes.json') + '\n')
    return out / 'score/summary.json'


def self_check():
    import replay190
    poly = [[0, 0], [10, 0], [10, 30], [0, 30]]
    gt = np.array([[749, 1, 14, 5, 2, 7, 1], [750, 1, 6, 5, 2, 7, 1]], float)
    delayed = gt.copy(); delayed[:, 0] += 1
    a, b = score_arrays(gt, gt, poly, 790), score_arrays(gt, delayed, poly, 790)
    assert a['w30_percent'] == 0 and b['window']['numerator'] == 2
    replay190.score(a); replay190.score(b)
    combined = e3([a, a, b])
    assert combined['numerator'] == 2 and combined['denominator'] == 90
    close = replay190.e3([a, a, b])
    assert abs(close - combined['W_percent']) < 1e-10
    cells = {(h, s): dict(base=b, witness=a) for h in (33, 35) for s in range(3, 9)}
    agg = aggregate(cells)
    assert agg['33']['witness']['mean_E3'] == 0
    assert agg['33']['witness_gain']['mean_E3'] > 0
    added = np.r_[gt, [[751, 1, 5, 5, 2, 7, 1]]]
    assert validate_witness(gt, added, {1})['1']['added'] == 1
    canonical = gt.copy(); canonical[:, 1] = 8
    mapping = native_identity_map(gt, canonical)
    json.dumps(mapping, sort_keys=True)
    owners, differences = official_anchor_owners(gt, gt, canonical, mapping)
    assert owners.tolist() == [1, 1] and differences == 0
    with tempfile.TemporaryDirectory(prefix='score190-synthetic-') as name:
        path = Path(name) / 'precise.txt'
        native = gt.copy(); native[:, 2] += np.pi
        write_exact_mot(path, native)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--prereg')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.repo:
        sys.path.insert(0, str(args.repo.resolve()))
    if args.self_test or not args.out:
        self_check()
        print('score190 synthetic checks passed')
    else:
        if not args.repo or args.prereg != PREREG or args.out.resolve().drive.upper() != 'D:':
            parser.error('Require registered --prereg, --repo and D: --out')
        print(run_field(args.repo.resolve(), args.out.resolve(), args.prereg))
