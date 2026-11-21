"""§193 post-all-fold-seal physical scorer. Synthetic selfcheck only by default.

run_sealed accepts an injected GT loader and refuses to call it until all four
fold receipts and every prediction/raw-MOT hash have been verified. No target
labels participate in deployable I/Q/R reconstruction.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from beetrack.evaluate import by_track, match_events
from tools.study103.swap_score import event_scores, vector_of_counts, window_error
from tools.study104.event_overlap import archive_arm, score_arm
from identity_flow import identity_flow
from physical import realize_subset, verify_subset

ARMS = ('B', 'I', 'Q', 'R')
FOLDS = {'33', '35', 'W', 'G'}
PREREG = 'f161c64a6310c663d6cb079e83abee335d90e324'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def roundtrip(mot):
    """Actually materialize one temporary MOT, preserving float64 exactly."""
    with TemporaryDirectory(prefix='synthetic_or_sealed193_') as tmp:
        path = Path(tmp) / 'predictions.txt'
        np.savetxt(path, mot, fmt='%.17g', delimiter=',')
        loaded = np.loadtxt(path, delimiter=',', ndmin=2) if len(mot) else np.empty_like(mot)
    assert np.array_equal(loaded, mot), 'physical MOT roundtrip changed precision'
    return loaded


def score_cell(gt, raw, polygon, nframes, kept, *, flexible_pool=None, dense=False):
    """All arrays are supplied after the all-four-fold deployment freeze.

    kept={I,Q,R:set canonical event IDs}; B is the complete original MOT.
    flexible_pool contains events allowed to change at fixed Q quotas; all
    other original events are mandatory in the retrospective identity oracle.
    Returns scalar scores plus private count arrays for exact aggregation.
    """
    if set(kept) != {'I', 'Q', 'R'}:
        raise ValueError('require all sealed deployable I/Q/R kept-ID sets')
    base = by_track(np.asarray(raw, float))
    gt = by_track(np.asarray(gt, float))
    graph = score_arm(gt, base, polygon)
    event_ids = set(graph.events)
    kept = {arm: set(ids) for arm, ids in kept.items()}
    if any(not ids <= event_ids for ids in kept.values()):
        raise ValueError('sealed keep references unknown canonical event')
    pool = event_ids if flexible_pool is None else set(flexible_pool)
    if not pool <= event_ids:
        raise ValueError('unknown flexible-pool ID')
    mandatory = event_ids - pool
    if any(not mandatory <= kept[arm] for arm in ('Q', 'R')):
        raise ValueError('Q/R removed event outside sealed flexible pool')
    buckets = {}
    for pid, (direction, frame, _) in graph.events.items():
        b = ('cell', (frame-1)//25, 1 if direction == 'in' else -1)
        buckets.setdefault(b, []).append(pid)
    quotas = {b: len(set(ids) & kept['Q']) for b, ids in buckets.items()}
    assert all(len(set(ids) & kept['R']) == quotas[b] for b, ids in buckets.items()), 'Q/R per-member bucket quota differs'
    edges = {pid: [] for pid in graph.events}
    for pid, gid in graph.edges:
        edges[pid].append(gid)
    oracle = identity_flow(buckets, quotas, edges, {'cell': list(graph.gt)}, mandatory=mandatory)
    oracle_kept = {pid for ids in oracle['retained'].values() for pid in ids}
    choices = dict(B=event_ids, **kept, O=oracle_kept)
    gc = vector_of_counts(gt, polygon, nframes)
    out = {}
    for arm, ids in choices.items():
        # B also traverses the exact same physical writer/reader path.
        physical, segmentation = realize_subset(base, polygon, ids)
        physical = roundtrip(physical)
        official = match_events(gt, physical, polygon)
        metrics = event_scores(gt, physical, polygon)
        assert official['tp'] == metrics[.5]['tp']
        pc = vector_of_counts(physical, polygon, nframes)
        numerator = int(np.abs(pc-gc).sum())
        denominator = int(gc.sum())
        assert denominator == 30*official['gt']
        scores = {'0.5': metrics[.5]}
        if dense:
            scores['0.3'] = metrics[.3]
        retained_map = {s['output_id']: s['source_id'] for s in segmentation['segments'] if s['event_retained']}
        for threshold in (.5, .3) if dense else (.5,):
            native_graph = graph if threshold == .5 else score_arm(gt, base, polygon, threshold)
            actual_graph = score_arm(gt, physical, polygon, threshold)
            assert {(retained_map[p], g) for p, g in actual_graph.edges} == {
                (p, g) for p, g in native_graph.edges if p in ids}, 'physical rewrite changed official legal edge'
        out[arm] = dict(gt=official['gt'], pred=official['pred'], tp=official['tp'],
                       tp_id=official['tp_id'], W=window_error(gc, pc) if denominator else None,
                       W_numerator=numerator, W_denominator=denominator,
                       over_numerator=int(np.maximum(pc-gc, 0).sum()),
                       under_numerator=int(np.maximum(gc-pc, 0).sum()),
                       F1_denominator=official['pred']+official['gt'], metrics=scores,
                       _counts=pc, _gt_counts=gc)
    assert np.array_equal(out['Q']['_counts'], out['R']['_counts']), 'Q/R window counts differ'
    assert np.array_equal(out['Q']['_counts'], out['O']['_counts']), 'oracle changed fixed-quota window counts'
    for arm in ('R', 'O'):
        for field in ('pred', 'gt', 'W_numerator', 'W_denominator', 'F1_denominator'):
            assert out['Q'][field] == out[arm][field], ('fixed quota mismatch', arm, field)
    assert out['O']['tp'] == oracle['maximum_tp'] == oracle['minimum_cut_capacity']
    assert out['O']['tp'] >= max(out['Q']['tp'], out['R']['tp'])
    assert out['B']['tp'] == len(graph.matches) and out['B']['pred'] == len(graph.events)
    # Identity re-expression must match the official native baseline exactly.
    assert match_events(gt, base, polygon) == {k: out['B'][k] for k in ('gt', 'pred', 'tp', 'tp_id')}
    archived = archive_arm(graph)
    evidence = dict(gt=archived['gt'], predictions=archived['predictions'], edges05=archived['edges'],
                    raw_array_sha256=hashlib.sha256(np.ascontiguousarray(raw, dtype=np.float64).tobytes()).hexdigest(),
                    raw_array_shape=list(np.asarray(raw).shape), raw_array_dtype='float64',
                    canonical_array_sha256=hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest())
    if dense:
        evidence['edges03'] = archive_arm(score_arm(gt, base, polygon, .3))['edges']
    return dict(arms=out, native_evidence=evidence, oracle=dict(maximum_tp=oracle['maximum_tp'],
                minimum_cut_capacity=oracle['minimum_cut_capacity'], minimum_cut=oracle['minimum_cut'],
                retained_ids=sorted(oracle_kept), matching=oracle['matching'],
                mandatory_ids=sorted(mandatory), attainable=True))


def aggregate(cells):
    """Cells carry fold, group, member, clip, and score_cell's result.

    Pool clip numerators before member W; group E3 compares the sum of the
    three member count arrays to 3*GT on the original native clip clocks.
    """
    members, groups = {}, {}
    for c in cells:
        key = f"{c['fold']}/{c['group']}/{c['member']}"
        member = members.setdefault(key, {arm: dict(gt=0, pred=0, tp=0, W_numerator=0, W_denominator=0, over_numerator=0, under_numerator=0, metrics={}) for arm in (*ARMS, 'O')})
        groups.setdefault((str(c['fold']), str(c['group'])), []).append(c)
        for arm, scores in c['result']['arms'].items():
            m = member[arm]
            for field in ('gt', 'pred', 'tp', 'W_numerator', 'W_denominator', 'over_numerator', 'under_numerator'):
                m[field] += scores[field]
            for threshold, metric in scores['metrics'].items():
                mm = m['metrics'].setdefault(threshold, dict(tp=0, fp=0, fn=0))
                for field in mm:
                    mm[field] += metric[field]
    for value in members.values():
        for m in value.values():
            m['W'] = 100*m['W_numerator']/m['W_denominator'] if m['W_denominator'] else None
            for part in ('over', 'under'):
                m[part] = 100*m[f'{part}_numerator']/m['W_denominator'] if m['W_denominator'] else None
            m['f1'] = 2*m['tp']/max(1, m['pred']+m['gt'])
            for metric in m['metrics'].values():
                metric['f1'] = 2*metric['tp']/max(1, 2*metric['tp']+metric['fp']+metric['fn'])
    grouped = {}
    for (fold, group), cc in groups.items():
        byclip = {}
        for c in cc:
            byclip.setdefault(c['clip'], []).append(c)
        pooled = {arm: dict(E3_numerator=0, over_numerator=0, under_numerator=0, denominator=0, gt=0, pred=0, tp=0) for arm in (*ARMS, 'O')}
        for clip, stream in byclip.items():
            assert len(stream) == 3 and len({str(c['member']) for c in stream}) == 3
            for arm in pooled:
                scores = [c['result']['arms'][arm] for c in stream]
                truth = scores[0]['_gt_counts']
                assert all(np.array_equal(s['_gt_counts'], truth) for s in scores)
                residual = sum(s['_counts'] for s in scores)-3*truth
                pooled[arm]['E3_numerator'] += int(np.abs(residual).sum())
                pooled[arm]['over_numerator'] += int(np.maximum(residual, 0).sum())
                pooled[arm]['under_numerator'] += int(np.maximum(-residual, 0).sum())
                pooled[arm]['denominator'] += 3*int(truth.sum())
                for field in ('gt', 'pred', 'tp'):
                    pooled[arm][field] += sum(s[field] for s in scores)
        for arm, v in pooled.items():
            v['E3'] = 100*v['E3_numerator']/v['denominator'] if v['denominator'] else None
            for part in ('over', 'under'):
                v[part] = 100*v[f'{part}_numerator']/v['denominator'] if v['denominator'] else None
            mm = [value[arm] for key, value in members.items() if key.startswith(f'{fold}/{group}/')]
            assert len(mm) == 3
            v['mean_single_W'] = float(np.mean([m['W'] for m in mm]))
            v['mean_single_f1'] = float(np.mean([m['f1'] for m in mm]))
            v['f1'] = 2*v['tp']/max(1, v['pred']+v['gt'])
            v['metrics'] = {}
            for threshold in mm[0]['metrics']:
                metric = {field: sum(m['metrics'][threshold][field] for m in mm) for field in ('tp', 'fp', 'fn')}
                metric['f1'] = 2*metric['tp']/max(1, 2*metric['tp']+metric['fp']+metric['fn'])
                v['metrics'][threshold] = metric
        assert pooled['Q']['E3_numerator'] == pooled['R']['E3_numerator'] == pooled['O']['E3_numerator']
        grouped[f'{fold}/{group}'] = pooled
    heads = {}
    for fold in sorted({c['fold'] for c in cells}):
        values = [v for k,v in grouped.items() if k.startswith(f'{fold}/')]
        heads[fold] = {arm: {field:float(np.mean([v[arm][field] for v in values]))
                             for field in ('E3', 'over', 'under', 'mean_single_W', 'mean_single_f1')}
                       for arm in (*ARMS, 'O')}
    return dict(members=members, groups=grouped, heads=heads)


def verify_all_sealed(prediction_dir, frozen_root, expected_sha=None):
    """Verify all eight source models/deployments and GT-free raw inputs first."""
    from freeze193 import load_groups, seal_receipt, MODEL_SHA
    prediction_dir, frozen_root = Path(prediction_dir), Path(frozen_root)
    path = prediction_dir/'ALL_FOUR_SEAL.json'
    digest = sha(path)
    assert digest == (prediction_dir/'ALL_FOUR_SEAL.sha256').read_text().strip()
    if expected_sha is not None:
        assert digest == expected_sha
    seal = json.loads(path.read_text(encoding='utf-8'))
    assert seal['schema'] == 'field193-all-four-seal-v1'
    assert set(seal['domains']) == FOLDS and len(seal['domains']) == 4
    assert set(seal['files']) == {f'{domain}.{kind}.json' for domain in FOLDS for kind in ('model', 'predictions')}
    for name, wanted in seal['files'].items():
        assert Path(name).name == name and sha(prediction_dir/name) == wanted
    for name, wanted in seal.get('code_receipts', {}).items():
        assert Path(name).name == name and sha(Path(__file__).with_name(name)) == wanted, 'training code changed after deployment freeze'
    receipt = seal['input_receipt']
    assert receipt == seal_receipt(frozen_root) if isinstance(receipt, dict) else receipt == (frozen_root/'SEALED.sha256').read_text().strip(), 'input freeze receipt mismatch'
    units = load_groups(frozen_root)  # checks every raw npz and metadata hash
    for u in units:
        if u['split'] == 'synthetic':
            continue
        expected = 'm_no'+u['domain'] if u['domain'] in ('W', 'G') else 'm_dev_strict'
        assert u['linker_model'] == expected and u['linker_sha256'] == MODEL_SHA[expected]
        assert u['domain'] in u['linker_excluded_domains']
        for m in u['members']:
            assert m['linker_receipt']['name'] == expected and m['linker_receipt']['sha256'] == MODEL_SHA[expected]
            if u['domain'] in ('W', 'G'):
                assert m['raw_receipt']['source']['kind'] == 'native_clean_rebuild'
            assert sha(frozen_root/m['raw_receipt']['file']) == m['raw_receipt']['sha256']
    predictions = []
    for domain in seal['domains']:
        value = json.loads((prediction_dir/f'{domain}.predictions.json').read_text(encoding='utf-8'))
        assert all(str(u['domain']) == domain for u in value)
        predictions.extend(value)
    native = {u['unit']: u for u in units}
    assert len(native) == len(units) == len(predictions)
    assert {p['unit'] for p in predictions} == set(native)
    for p in predictions:
        u = native[p['unit']]
        assert (p['domain'], p['video'], p['group'], p['nf']) == (u['domain'], u['video'], u['group'], u['nf'])
        for key in ('freeze_receipt','linker_model','linker_sha256','linker_excluded_domains'):
            assert p.get(key) == u.get(key), ('prediction/input metadata mismatch', p['unit'], key)
        assert {m['member'] for m in p['members']} == {m['member'] for m in u['members']}
        source_members = {m['member']:m for m in u['members']}
        for m in p['members']:
            source = source_members[m['member']]
            for key in ('raw_receipt','linker_name','linker_receipt'):
                assert m.get(key) == source.get(key), ('prediction/raw receipt mismatch', p['unit'], m['member'], key)
            assert set(m['arms']) == set(ARMS)
            assert m['arms']['Q']['quotas'] == m['arms']['R']['quotas']
    return seal, native, predictions


def validate189(report, published):
    """33/35 reproduce §189; clean-linker W/G only compare to its old baseline.

    §189 W/G used m_dev_strict, whose training saw that target. Their new
    m_noW/m_noG native baseline is intentionally not an equality reference.
    """
    ref = published['results']['0.40']
    matched = {(str(r['member']), r['ev']): r for r in ref['match']}
    changed_cells, changed_groups = [], []
    for c in report['cells']:
        wanted = matched[(str(c['member']), c['clip'])]
        got = c['result']['arms']['B']
        if str(c['fold']) in ('33', '35'):
            for key in ('gt', 'pred', 'tp', 'tp_id'):
                assert got[key] == wanted[key], ('baseline189', c['member'], c['clip'], key)
        else:
            changed_cells.append(dict(fold=c['fold'], member=c['member'], clip=c['clip'],
                                      old_linker='m_dev_strict',
                                      delta={key:got[key]-wanted[key] for key in ('gt', 'pred', 'tp', 'tp_id')},
                                      old={key:wanted[key] for key in ('gt', 'pred', 'tp', 'tp_id')},
                                      clean={key:got[key] for key in ('gt', 'pred', 'tp', 'tp_id')}))
    for key, arms in report['groups'].items():
        domain, group = key.split('/')
        wanted = ref['loho'][domain]['before']['groups'][group]
        if domain in ('33', '35'):
            assert abs(arms['B']['E3']-wanted['e3']) < 1e-9
            assert abs(arms['B']['mean_single_W']-wanted['single']) < 1e-9
        else:
            changed_groups.append(dict(fold=domain, group=group,
                                       old_E3=wanted['e3'], clean_E3=arms['B']['E3'],
                                       delta_E3=arms['B']['E3']-wanted['e3'],
                                       old_mean_single_W=wanted['single'],
                                       clean_mean_single_W=arms['B']['mean_single_W'],
                                       delta_mean_single_W=arms['B']['mean_single_W']-wanted['single']))
    return dict(threshold='0.40', strict_domains=['33', '35'],
                strict_official_counts_and_tp=True, strict_member_W_and_group_E3=True,
                clean_linker_domains=['W', 'G'], W_G_old_baseline_cell_differences=changed_cells,
                W_G_old_baseline_group_differences=changed_groups,
                clean192_reference='pending published .40 baseline with matching recipe')


def run_sealed(prediction_dir, frozen_root, load_gt, *, expected_seal_sha=None, published189=None):
    """All target labels are injected only after the complete deployment seal."""
    seal, native, predictions = verify_all_sealed(prediction_dir, frozen_root, expected_seal_sha)
    scored = []
    for p in predictions:
        u = native[p['unit']]
        print(f"Target scoring after ALL_FOUR verification: {p['unit']}", flush=True)
        gt = load_gt(str(p['domain']), p['video'])
        bymember = {m['member']: m for m in u['members']}
        for m in p['members']:
            source = bymember[m['member']]
            ids = {int(e['pid']) for e in source['event_records']}
            assert set(m['arms']['B']['kept_ids']) == ids
            for arm in ARMS:
                assert set(m['arms'][arm]['flexible_pool_ids']) == ids
            qids, rids = set(m['arms']['Q']['kept_ids']), set(m['arms']['R']['kept_ids'])
            room = dict(bucket_count=0, rankable_buckets=0, changed_buckets=0, replaced_events=len(qids-rids))
            for q in m['arms']['Q']['quotas']:
                pool = {int(e['pid']) for e in source['event_records'] if (int(e['frame'])-1)//25 == q['second'] and e['direction'] == q['direction']}
                assert len(pool) == q['capacity'] and len(qids & pool) == len(rids & pool) == q['keep']
                room['bucket_count'] += 1
                room['rankable_buckets'] += int(0 < q['keep'] < q['capacity'])
                room['changed_buckets'] += int(qids & pool != rids & pool)
            result = score_cell(gt, source['rows'], u['polygon'], u['nf'],
                                {a:m['arms'][a]['kept_ids'] for a in ('I', 'Q', 'R')},
                                flexible_pool=ids, dense=str(p['domain']) == '33')
            result['identity_room'] = room
            result['input_metadata'] = {k:source.get(k) for k in ('raw_receipt', 'linker_name', 'linker_receipt')}
            result['kept_ids'] = {arm:list(m['arms'][arm]['kept_ids']) for arm in ARMS}
            result['kept_ids']['O'] = result['oracle']['retained_ids']
            result['nframes'] = int(u['nf'])
            print(f"  member {m['member']}: B/I/Q/R/O official physical scores checked", flush=True)
            scored.append(dict(fold=str(p['domain']), group=p['group'], member=m['member'], clip=p['video'], result=result))
    report = aggregate(scored)
    report['all_fold_manifest_sha256'] = sha(Path(prediction_dir)/'ALL_FOUR_SEAL.json')
    report['input_receipt'] = seal['input_receipt']
    report['oracle_scope'] = 'Exact fixed-Q-quota maximum F1 at IoU 0.5; 33 IoU0.3 scores this same physical witness, not a separate optimum.'
    report['cells'] = [{**{k: c[k] for k in ('fold', 'group', 'member', 'clip')},
                        'result': {**c['result'], 'arms': {a: {k:v for k,v in s.items() if not k.startswith('_')} for a,s in c['result']['arms'].items()}}}
                       for c in scored]
    if published189 is not None:
        report['baseline189_reproduction'] = validate189(report, published189)
    elif all(u['split'] != 'synthetic' for u in native.values()):
        report['baseline189_reproduction'] = validate_published_head_means(report)
    report['identity_room'] = {domain:{field:sum(c['result']['identity_room'][field] for c in scored if c['fold'] == domain)
                                     for field in ('bucket_count','rankable_buckets','changed_buckets','replaced_events')}
                               for domain in FOLDS}
    return report


def validate_published_head_means(report):
    """Portable check of four-decimal published head means, without §189 files.

    Native per-cell B still undergoes the exact official-MOT checks above.
    W/G are independently verified clean-linker P0; old WG is not an equality
    reference. Supply --baseline189 for the extra historic cell/group checks.
    """
    expected = {'35': 6.8303, '33': 19.3629}
    for domain, value in expected.items():
        assert abs(report['heads'][domain]['B']['E3']-value) <= .00005, ('published head B mismatch', domain)
    return dict(threshold='0.40', published_head_B_means=expected,
                tolerance=.00005, native_per_cell_official_checks=True,
                clean_W_G_linker_checks=True, historic_cell_group_checks='not requested')

def selfcheck():
    from e2e_selfcheck import fixture, POLY, NF
    target = fixture(7193, 'synthetic-score193')
    cells = []
    for member, data in target['members'].items():
        all_ids = {p[1] for b, ids in target['buckets'].items() if b[0] == member for p in ids}
        good = {p for p in all_ids if p % 2 == 0}
        bad = all_ids-good
        result = score_cell(data['gt'], data['rows'], POLY, NF,
                            dict(I=bad, Q=bad, R=good), dense=True)
        cells.append(dict(fold='33', group='synthetic', member=member, clip='synthetic', result=result))
    report = aggregate(cells)
    group = report['groups']['33/synthetic']
    assert group['Q']['f1'] == 0 and group['R']['f1'] == group['O']['f1'] == 1
    assert group['Q']['E3'] == group['R']['E3'] == 0
    assert group['B']['E3'] == 100
    sealing = sealed_synthetic_check()
    print(json.dumps(dict(synthetic_only=True, groups=report['groups'], sealing=sealing), indent=2))


def sealed_synthetic_check():
    """Real four-fold train/seal/materialize/evaluate plus failed-seal guard."""
    from e2e_selfcheck import fixture, POLY, NF
    from freeze193 import extract, counts, load_groups
    from field_train193 import train_and_freeze, _write
    from source193 import training_clip
    with TemporaryDirectory(prefix='allfour_synthetic193_') as tmp:
        root = Path(tmp)/'frozen'
        (root/'rows').mkdir(parents=True)
        deployment = dict(schema='budget193-deployment-v1', clips=[], units=[], events=[], members=[], groups=[])
        truths = {}
        for ci, domain in enumerate(sorted(FOLDS)):
            f = fixture(8000+ci, f'synthetic-{domain}')
            video = f['video']
            truths[domain, video] = next(iter(f['members'].values()))['gt']
            deployment['clips'].append(dict(clip=video, nframes=NF, polygon=POLY, split='synthetic'))
            uid_list = []
            for data in f['members'].values():
                uid = len(deployment['units'])
                uid_list.append(uid)
                file = f'rows/u{uid:02d}.npz'
                np.savez(root/file, rows=data['rows'], s1=data['rows'][:, 1])
                ev = extract(data['rows'], data['rows'][:, 1], POLY, NF)
                wc, sc = counts(ev, NF)
                deployment['members'].append(dict(tag=f'synthetic-{uid}'))
                deployment['units'].append(dict(rows_file=file, member=uid, windows=wc, seconds=sc,
                    frozen_rows_sha256=sha(root/file), source_receipt=dict(kind='synthetic'),
                    linker_name='synthetic',linker_receipt=dict(name='synthetic',sha256='synthetic',excluded_domains=[domain])))
                for e in ev:
                    deployment['events'].append(dict(**e, unit=uid, event=len(deployment['events'])))
            deployment['groups'].append(dict(clip_index=ci, hive=domain, group=0, units=uid_list))
        _write(root/'deployment.json', deployment)
        manifest = {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
        _write(root/'manifest.json', manifest)
        receipt = sha(root/'manifest.json')
        (root/'SEALED.sha256').write_text(receipt)
        units = load_groups(root)
        def sources(heldout):
            labels = {}
            for u in units:
                if u['domain'] != heldout:
                    label = training_clip(truths[u['domain'], u['video']],
                                          {m['member']:m['rows'] for m in u['members']}, POLY, NF)
                    label['domain'] = u['domain']
                    labels[u['video']] = label
            return labels
        predictions = Path(tmp)/'predictions'
        train_and_freeze(units, sources, predictions, receipt)
        calls = []
        def target_gt(domain, video):
            calls.append((domain, video))
            return truths[domain, video]
        result = run_sealed(predictions, root, target_gt)
        assert len(calls) == 4 and len(result['groups']) == 4
        tamper = predictions/'33.predictions.json'
        tamper.write_text(tamper.read_text()+' ')
        calls.clear()
        try:
            run_sealed(predictions, root, target_gt)
        except AssertionError:
            pass
        else:
            raise AssertionError('tampered seal accepted')
        assert calls == [], 'GT loader reached before all-four verification'
        return dict(folds=4, groups=4, scored_cells=len(result['cells']),
                    all_four_before_target_gt=True, tampered_prediction_blocks_gt=True,
                    physical_roundtrips=True, Q_R_O_counts_and_denominators_equal=True)


def field_entry(args):
    """Standalone official run: no target GT opened before all-four gate."""
    import configparser
    import gzip
    import re
    import subprocess
    from source193 import SOURCES, EXPECTED_DOMAIN_GT
    if args.prereg != PREREG:
        raise ValueError('exact published §193 preregistration commit required')
    registered = subprocess.run(['git','show',f'{PREREG}:docs/experiment_log.md'], cwd=args.repo,
                                check=True,capture_output=True).stdout.decode('utf-8')
    assert re.search(r'^## 193\.', registered, re.M)
    if args.out.drive.upper() != 'D:':
        raise ValueError('new field outputs must be on D:')
    if args.out.exists():
        raise FileExistsError(args.out)
    seal, units, _ = verify_all_sealed(args.predictions, args.frozen, args.seal_sha256)
    deployment = json.loads((args.frozen/'deployment.json').read_text(encoding='utf-8'))
    assert deployment['prereg'] == PREREG and deployment['threshold'] == .40
    assert len(units) == 20 and sum(len(u['members']) for u in units.values()) == 60
    loaded_gt, gt_receipts = {}, {}
    def load_gt(domain, clip):
        allowed = {c:s for s,c in SOURCES[domain]}
        if clip not in allowed:
            raise ValueError('target clip outside original §193 allowlist')
        key = domain,clip
        if key not in loaded_gt:
            directory = args.repo/'runs/gt'/allowed[clip]/clip
            path = directory/'gt/gt.txt'
            config = configparser.ConfigParser()
            config.read(directory/'seqinfo.ini',encoding='utf-8')
            expected = next(u['nf'] for u in units.values() if u['video'] == clip)
            assert config.getint('Sequence','frameRate') == 25 and config.getint('Sequence','seqLength') == expected
            loaded_gt[key] = np.loadtxt(path,delimiter=',',usecols=range(6),ndmin=2)
            gt_receipts[f'{domain}/{clip}'] = dict(path=str(path),sha256=sha(path))
        return loaded_gt[key]
    # Reference labels/results are opened only after the full target seal.
    published = json.loads(args.baseline189.read_text(encoding='utf-8')) if args.baseline189 is not None else None
    report = run_sealed(args.predictions,args.frozen,load_gt,
                        expected_seal_sha=args.seal_sha256,published189=published)
    unique = {}
    for c in report['cells']:
        unique[c['fold'],c['clip']] = c['result']['arms']['B']['gt']
    assert {d:sum(v for (domain,_),v in unique.items() if domain == d) for d in FOLDS} == EXPECTED_DOMAIN_GT
    report.update(prereg=PREREG,gt_receipts=gt_receipts,baseline189_sha256=sha(args.baseline189) if args.baseline189 is not None else None,
                  score_code_sha256=sha(__file__),prediction_seal=seal,
                  physical_precision='Actual one-arm temporary MOT %.17g write/read; exact float64 equality; native geometry and Hungarian tie order preserved.')
    h = report['heads']
    gain35 = h['35']['B']['E3']-h['35']['R']['E3']
    gates = dict(gain35_strict_gt1_4=gain35 > 1.4,
                 degradation33_le1_3=h['33']['R']['E3']-h['33']['B']['E3'] <= 1.3,
                 degradationW_le1_4=h['W']['R']['E3']-h['W']['B']['E3'] <= 1.4,
                 degradationG_le1_4=h['G']['R']['E3']-h['G']['B']['E3'] <= 1.4,
                 f1_35_nondecrease=h['35']['R']['mean_single_f1'] >= h['35']['B']['mean_single_f1'],
                 f1_33_nondecrease=h['33']['R']['mean_single_f1'] >= h['33']['B']['mean_single_f1'])
    report['investment_gate'] = dict(conditions=gates,all_pass=all(gates.values()),gain35=gain35)
    args.out.mkdir(parents=True)
    compact = json.dumps(report,separators=(',',':'),sort_keys=True,allow_nan=False).encode()
    (args.out/'score193.json.gz').write_bytes(gzip.compress(compact,mtime=0))
    summary = {k:report[k] for k in ('prereg','heads','groups','identity_room','investment_gate','baseline189_reproduction')}
    (args.out/'summary193.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')
    receipts = {p.name:sha(p) for p in args.out.iterdir() if p.is_file()}
    (args.out/'manifest.json').write_text(json.dumps(receipts,sort_keys=True),encoding='utf-8')
    (args.out/'SEALED.sha256').write_text(sha(args.out/'manifest.json')+'\n')
    print(json.dumps(dict(output=str(args.out),seal=sha(args.out/'manifest.json'),heads=h,
                          investment_gate=report['investment_gate']),indent=2),flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--repo',type=Path)
    parser.add_argument('--frozen',type=Path)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--prereg')
    parser.add_argument('--seal-sha256')
    parser.add_argument('--baseline189',type=Path)
    args = parser.parse_args()
    if args.self_check or args.repo is None:
        selfcheck()
    else:
        if any(getattr(args,k) is None for k in ('frozen','predictions','out','prereg')):
            parser.error('--repo --frozen --predictions --out --prereg required; --baseline189 optional')
        field_entry(args)
