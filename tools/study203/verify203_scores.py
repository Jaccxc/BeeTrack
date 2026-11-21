"""Independent 203 score accounting. Importing performs no field IO.

Only audit_scores(args, helpers=None) opens field inputs. args needs root,
repo, source202_score and optionally score_sha. helpers may supply already
loaded independent modules as {'previous': verify_orphans, 'replay': verify203,
'native': study197_verify}. No producer scorer or selection code is imported.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

from collections import defaultdict
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ARMS = ('K', 'G', 'I', 'L', 'T', 'M')
FAMILIES = ('S', 'P', 'A')
HIVES = ('35', '33', 'W', 'G')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _inside(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError('path escapes sealed stage')
    return target


def _equal_compact(metric, full):
    for field in ('nG', 'nP', 'pred_bins', 'gt_bins'):
        assert metric[field] == full[field], field
    for threshold in ('0.5', '0.3'):
        for field in ('tp', 'pred', 'gt', 'tp_id', 'f1'):
            if field in metric['iou'][threshold]:
                assert metric['iou'][threshold][field] == full['iou'][threshold][field], field


def _native_graph(full, rows, gt, polygon, nf, native, phase_bins):
    """Verify every supplied NEW physical graph's maximum event cardinality."""
    truth, prediction = native.track_structure(gt, polygon), native.track_structure(rows, polygon)
    gb = phase_bins(truth['events'].values(), nf)
    pb = phase_bins(prediction['events'].values(), nf)
    assert np.array_equal(gb, full['gt_bins']) and np.array_equal(pb, full['pred_bins'])
    assert full['nG'] == len(truth['events']) and full['nP'] == len(prediction['events'])
    assert full['evidence']['gt'] == [[pid, date*direction] for pid, (date, direction) in truth['events'].items()]
    assert [[x[0], x[1]] for x in full['evidence']['predictions']] == [[pid, date*direction] for pid, (date, direction) in prediction['events'].items()]
    for threshold, field in (('0.5', 'edges05'), ('0.3', 'edges03')):
        adjacency = defaultdict(set)
        edges = full['evidence'][field]
        assert len({tuple(e) for e in edges}) == len(edges)
        for pid, gid in edges:
            assert pid in prediction['events'] and gid in truth['events']
            assert prediction['events'][pid][1] == truth['events'][gid][1]
            adjacency[pid].add(gid)
        tp = native.augmenting_tp(prediction['events'], adjacency)
        metric = full['iou'][threshold]
        assert metric['tp'] == tp and metric['pred'] == full['nP'] and metric['gt'] == full['nG']
        assert abs(metric['f1'] - 2*tp/max(1, full['nP']+full['nG'])) < 1e-14
    if 'window' in full:
        numerator, denominator = int(abs(pb-gb).sum()), int(gb.sum())
        assert full['window']['numerator'] == numerator and full['window']['denominator'] == denominator
        assert full['W'] is None if not denominator else abs(full['W']-100*numerator/denominator) < 1e-11
    return truth, prediction


def _aggregate_checks(scored, summary, previous):
    exact = previous.aggregate_independent(scored)
    for hive, methods in exact.items():
        for arm, value in methods.items():
            saved = summary['domains'][hive]['methods'][arm]
            assert Fraction(*saved['E3_fraction']) == value['E3']
            assert abs(saved['E3'] - float(value['E3'])) < 1e-11
            for threshold in ('0.5', '0.3'):
                assert abs(saved['F1'][threshold] - float(value['F1'][threshold])) < 1e-13
    gates = {}
    for arm in ('L', 'T'):
        gates[arm] = dict(
            head35_gain=exact['35']['K']['E3']-exact['35'][arm]['E3'] > Fraction(14, 10),
            head33_gain=exact['33']['K']['E3']-exact['33'][arm]['E3'] > Fraction(13, 10),
            W_no_harm=exact['W'][arm]['E3']-exact['W']['K']['E3'] <= Fraction(14, 10),
            F1_no_harm=all(exact[h][arm]['F1']['0.5'] >= exact[h]['K']['F1']['0.5']-Fraction(1, 200) for h in ('35', '33', 'W')))
        assert summary['investment_checks'][arm] == gates[arm]
        assert summary['investment_pass'][arm] == all(gates[arm].values())
    attribution = dict(
        T_strictly_beats_L_both_heads=all(exact[h]['T']['E3'] < exact[h]['L']['E3'] for h in ('35', '33')),
        T_strictly_beats_I_G_M_both_heads=all(exact[h]['T']['E3'] < exact[h][arm]['E3'] for h in ('35', '33') for arm in ('I', 'G', 'M')))
    for key, value in attribution.items():
        assert summary['attribution'][key] == value
    return exact, gates, attribution


def audit_scores(args, helpers=None):
    """Explicit field entry; caller must have obtained root's field GO first."""
    from beetrack.evaluate import load_tracks, frame_match, match_events
    helpers = {} if helpers is None else helpers
    here = Path(__file__).parent
    previous = helpers.get('previous') or _load('independent202_score203', Path.cwd()/'tools/study202/verify_orphans.py')
    replay = helpers.get('replay') or _load('independent203_score_replay', here/'verify203.py')
    native = helpers.get('native')
    if native is None:
        native, _ = previous.own_helpers(args.repo)
    root = Path(args.root)
    score_seal = previous.sealed(root/'score', getattr(args, 'score_sha', None))
    scored, calls, summary = [_read(root/'score'/name) for name in ('native_scores.json', 'physical_receipts.json', 'summary.json')]
    assert set(scored) == set(ARMS+FAMILIES) and len(calls) == 432
    roles = {(g['unit'], str(m['host_member']), arm): (g, m) for arm, groups in scored.items() for g in groups for m in g['members']}
    assert len(roles) == 432
    previous.sealed(args.source202_score)
    old = {(g['unit'], int(m['member_index'])): m['metrics'] for g in _read(Path(args.source202_score)/'native_scores.json')['K'] for m in g['members']}
    descriptors = _read(root/'frozen/deployment.json')['groups']
    assert len(descriptors) == 20 and len({d['unit'] for d in descriptors}) == 20
    capacity = _read(root/'capacity/capacity.json')['units']
    labels = {u['unit']: {a['action_id']: a['labels']['0.5']['class'] for a in u['actions']} for u in _read(root/'owners/classes.json')['units']}
    masks = {}
    for hive in HIVES:
        for member in _read(root/'predictions'/(hive+'.masks.json'))['member_clips']:
            host = str(member.get('member', member.get('host_member')))
            for arm, state in member['selected'].items():
                masks[member['unit'], host, arm] = sorted(state['candidate_ids'])
    call_index = {(c['unit'], str(c['member']), c['arm']): c for c in calls}
    assert len(call_index) == len(calls) and set(call_index) == set(roles)
    cache, fresh, gt_cache, stats = {}, {}, {}, {}
    requested = 0
    for descriptor in descriptors:
        folder = _inside(root/'frozen', descriptor['path'])
        previous.sealed(folder, descriptor['seal'])
        group = _read(folder/'catalog.json')
        with np.load(folder/'K.npz', allow_pickle=False) as archive:
            members = [dict(m, rows=archive['k'+str(i)].copy()) for i, m in enumerate(group['members'])]
        assert sorted(m['member_index'] for m in members) == [0, 1, 2]
        unit, hive = group['unit'], group['domain']
        arms = ARMS+(FAMILIES if hive in ('35', '33') else ())
        # Parent verifies complete source supply equality. Here independently
        # replay ALL selected native rows jointly, rather than trust producers.
        candidates = group['candidates']
        byid = {c['candidate_id']: c for c in candidates}
        assert len(byid) == len(candidates)
        record = members[0]['source_record']
        gt_key = record['split'], group['video']
        if gt_key not in gt_cache:
            gt_cache[gt_key] = load_tracks(Path(args.repo)/'runs/gt'/gt_key[0]/gt_key[1]/'gt/gt.txt')
        gt = gt_cache[gt_key]
        for arm in arms:
            ids = []
            for member in members:
                host, slot = str(member['member']), int(member['member_index'])
                chosen = masks[unit, host, arm] if arm in ARMS else sorted(capacity[unit][arm]['member_masks'][str(slot)]['candidate_ids'])
                assert len(chosen) == len(set(chosen)) and all(cid in byid and byid[cid]['host_member'] == host for cid in chosen)
                assert arm != 'K' or not chosen
                if arm in ('S', 'P'):
                    assert all(labels[unit][cid] in ({'S'} if arm == 'S' else {'S', 'U'}) for cid in chosen)
                ids.extend(chosen)
            physical, new_sources = replay.materialize_independent(members, candidates, ids)
            for member in members:
                requested += 1
                host, slot = str(member['member']), int(member['member_index'])
                role_group, role = roles[unit, host, arm]
                call = call_index[unit, host, arm]
                chosen = sorted(cid for cid in ids if byid[cid]['host_member'] == host)
                assert role['candidate_ids'] == call['candidate_ids'] == chosen
                assert call['member_index'] == slot and role['member_index'] == slot
                assert role['host_member'] == host and role['member'] == member['source_record']['member']
                assert role['physical'] == call['path'] and role['seal'] == call['seal']
                assert role_group['domain'] == hive and role_group['group'] == group['group'] and role_group['video'] == group['video']
                rows, raw = physical[host], np.asarray(member['rows'], np.float64)
                assert rows[:len(raw)].tobytes() == raw.tobytes()
                assert np.isfinite(rows).all() and (rows[:, 4:6] > 0).all()
                assert np.array_equal(rows[:, :2], np.rint(rows[:, :2])) and (rows[:, :2] >= 1).all() and (rows[:, 0] <= group['nf']).all()
                assert len(np.unique(rows[:, :2], axis=0)) == len(rows)
                sources = [list(s) for s in member['sources']] + [list(s) for s in new_sources[host]]
                assert len(sources) == len(rows) == len({tuple(s) for s in sources})
                pb = replay.bins(replay.track_events(rows, group['polygon']), group['nf'])
                baseline = replay.bins(replay.track_events(raw, group['polygon']), group['nf'])
                delta = sum((np.asarray(byid[cid]['delta_windows'], np.int64) for cid in chosen), np.zeros_like(baseline))
                assert np.array_equal(pb, baseline+delta) and np.array_equal(pb, role['metrics']['pred_bins'])
                if arm == 'K':
                    _equal_compact(role['metrics'], old[unit, slot])
                stat = stats.setdefault((hive, arm), dict(selected=defaultdict(int), available=defaultdict(int), effects=defaultdict(int), no_window_effect=0, selected_rows=0))
                for cid, c in byid.items():
                    if c['host_member'] != host:
                        continue
                    label = labels[unit][cid]
                    stat['available'][label] += 1
                    if cid in chosen:
                        stat['selected'][label] += 1
                        before, after = c['events_before'], c['events_after']
                        effect = 'added' if len(after)>len(before) else 'removed' if len(after)<len(before) else 'unchanged' if before==after else 'date_only'
                        stat['effects'][effect] += 1
                        stat['no_window_effect'] += int(not np.asarray(c['delta_windows']).any())
                        stat['selected_rows'] += len(c['suffix'])
                token = call['path']
                if token not in cache:
                    path = _inside(root/'score/physical', token)
                    previous.sealed(path, call['seal'])
                    meta, full, compact = [_read(path/name) for name in ('receipt.json', 'score.json', 'compact.json')]
                    ident = meta['identity']
                    assert ident['unit'] == unit and ident['member'] == host and ident['candidate_ids'] == chosen and ident['nf'] == group['nf']
                    assert token == _object_sha([unit, host, chosen])
                    for key, data in (('rows_sha256', rows), ('raw_sha256', raw), ('GT_sha256', gt), ('polygon_sha256', group['polygon'])):
                        assert ident[key] == previous.native202_array_sha(data), key
                    assert ident['sources_sha256'] == _object_sha(sources)
                    assert meta['new_fullscene_official_matching'] is True and meta['original_K_rows_unchanged'] is True
                    assert meta['original_rows'] == len(raw) and meta['added_rows'] == len(rows)-len(raw)
                    disk = np.loadtxt(path/'mot.txt', delimiter=',', usecols=range(7), ndmin=2) if len(rows) else np.empty((0, 7))
                    assert disk.tobytes() == rows.tobytes()
                    with gzip.open(path/'row_sources.jsonl.gz', 'rt', encoding='utf-8') as stream:
                        assert [json.loads(line) for line in stream] == sources
                    truth, prediction = _native_graph(full, rows, gt, group['polygon'], group['nf'], native, previous.phase_bins)
                    _equal_compact(compact, full)
                    if chosen and hive not in fresh:
                        owners = {}
                        for threshold in (.5, .3):
                            matching = frame_match(truth['rows'], prediction['rows'], thr=threshold)
                            scattered = np.empty(len(rows), np.int64)
                            scattered[prediction['index']] = matching
                            owners[str(threshold)] = scattered
                        own = native.score_from_owners(prediction, truth, owners, group['polygon'], group['nf'])
                        for threshold, edgefield in (('0.5', 'edges05'), ('0.3', 'edges03')):
                            for field in ('tp', 'pred', 'gt', 'tp_id'):
                                assert own['iou'][threshold][field] == full['iou'][threshold][field]
                            assert own['edges'][threshold] == full['evidence'][edgefield]
                        assert match_events(gt, rows, group['polygon'])['tp'] == full['iou']['0.5']['tp']
                        fresh[hive] = dict(unit=unit, member=host, physical=token, added_rows=len(rows)-len(raw), thresholds=[.5, .3])
                    cache[token] = (compact, call['seal'], previous.native202_array_sha(rows), _object_sha(sources))
                compact, seal, prior_rows, prior_sources = cache[token]
                assert seal == call['seal'] and prior_rows == previous.native202_array_sha(rows) and prior_sources == _object_sha(sources)
                assert role['metrics'] == compact
        print('INDEPENDENT203_SCORES '+unit, flush=True)
    assert requested == 432
    capacity_count = 0
    for arm in FAMILIES:
        for group in scored[arm]:
            cap = capacity[group['unit']][arm]
            pred = sum(np.asarray(m['metrics']['pred_bins'], np.int64) for m in group['members'])
            gt = np.asarray(group['members'][0]['metrics']['gt_bins'], np.int64)
            assert np.array_equal(pred, cap['witness_pred_sum'])
            assert int(abs(pred-3*gt).sum()) == cap['upper_bound'] and 3*int(gt.sum()) == cap['denominator']
            capacity_count += 1
    assert capacity_count == 24
    exact, gates, attribution = _aggregate_checks(scored, summary, previous)
    for (hive, arm), actual in stats.items():
        saved = summary['domains'][hive]['methods'][arm]['action_stats']
        for key, value in actual.items():
            assert (dict(value) if isinstance(value, defaultdict) else value) == saved[key]
        total = sum(actual['selected'].values())
        for label in ('S', 'D', 'U'):
            assert saved['selected_proportions'][label] == (actual['selected'].get(label, 0)/total if total else None)
            denominator = actual['available'].get(label, 0)
            assert saved['class_coverage'][label] == (actual['selected'].get(label, 0)/denominator if denominator else None)
    for hive in ('35', '33'):
        for family in FAMILIES:
            units = [capacity[g['unit']][family] for g in scored[family] if g['domain'] == hive]
            lower = sum((Fraction(100*u['lower_bound'], u['denominator']) for u in units), Fraction())/len(units)
            upper = sum((Fraction(100*u['upper_bound'], u['denominator']) for u in units), Fraction())/len(units)
            baseline = exact[hive]['K']['E3']
            saved = summary['domains'][hive]['capacity_bounds'][family]
            for key, value in dict(mean_lower_E3=lower, mean_witness_E3=upper,
                                   guaranteed_gain=baseline-upper, maximum_possible_gain=baseline-lower).items():
                assert abs(saved[key]-float(value)) < 1e-11
            line = Fraction(14 if hive == '35' else 13, 10)
            assert saved['crosses_count_line'] == (baseline-upper > line)
            assert saved['excluded_by_lower_bound'] == (baseline-lower <= line)
            assert saved['all_optimal'] == all(u['lower_bound'] == u['upper_bound'] for u in units)
    scope = summary['physical_scope']
    assert scope['requested_arm_members'] == 432 and scope['unique_streams'] == len(cache) and scope['historical_reused'] == 0
    assert scope['fresh_unique_physical'] + scope['resumed_unique_physical'] == len(cache)
    assert scope['same_job_shared'] == len(calls)-len(cache)
    unavailable = [h for h in HIVES if h not in fresh]
    # A domain with no changed output cannot supply a fresh changed rematch;
    # expose that limitation rather than call an unchanged K scene a sample.
    assert all(not any(c['candidate_ids'] and roles[c['unit'], str(c['member']), c['arm']][0]['domain'] == h for c in calls) for h in unavailable)
    result = dict(passed=True, score_manifest=score_seal, requested=432, unique=len(cache),
        capacity_physical_witness_groups=24, fresh_native_changed_geometry=fresh,
        fresh_native_unavailable_domains=unavailable, investment_checks=gates, attribution=attribution,
        means={h: {a: dict(E3_fraction=[v['E3'].numerator, v['E3'].denominator], F1={t: float(f) for t, f in v['F1'].items()}) for a, v in aa.items()} for h, aa in exact.items()},
        scope='All432 physical matrices/provenance/counts; all unique NEW supplied graph maximum cardinalities. Fresh official full-frame .5/.3 spatial rematching on listed changed outputs only; TP_ID independently checked only on those samples. Parent verifies full source supply, masks/models/lineage and native owner taxonomy.')
    previous.receipt(root/'verification/scores.json', result)
    return result


def synthetic():
    # Pure graph-accounting fixture: no field files, native scorer or selection.
    class Native:
        @staticmethod
        def track_structure(rows, polygon):
            return dict(events={int(r[0]): (int(r[1]), int(r[2])) for r in rows})
        @staticmethod
        def augmenting_tp(events, adjacency):
            matched = {}
            def visit(pid, seen):
                for gid in adjacency[pid]:
                    if gid in seen:
                        continue
                    seen.add(gid)
                    if gid not in matched or visit(matched[gid], seen):
                        matched[gid] = pid
                        return True
                return False
            return sum(visit(pid, set()) for pid in events)
    phase = lambda ev, nf: [[sum(d == 1 for _, d in ev), 0]]
    rows, gt = [[10, 2, 1], [20, 3, 1]], [[1, 2, 1], [2, 3, 1]]
    metric = dict(tp=2, pred=2, gt=2, f1=1.)
    full = dict(nP=2, nG=2, pred_bins=[[2, 0]], gt_bins=[[2, 0]],
        iou={'0.5': dict(metric), '0.3': dict(metric)},
        evidence=dict(gt=[[1, 2], [2, 3]], predictions=[[10, 2], [20, 3]],
        edges05=[[10, 1], [10, 2], [20, 1]], edges03=[[10, 1], [10, 2], [20, 1]]))
    _native_graph(full, rows, gt, [], 3, Native, phase)
    full['iou']['0.5']['tp'] = 1
    try:
        _native_graph(full, rows, gt, [], 3, Native, phase)
    except AssertionError:
        pass
    else:
        raise AssertionError('nonmaximum supplied-graph TP accepted')
    # Execute only the allowed independent aggregation helper from SOURCE so
    # this arithmetic fixture does not require native OpenCV/tracker imports.
    import ast
    from types import SimpleNamespace
    path = Path.cwd()/'tools/study202/verify_orphans.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'aggregate_independent')
    namespace = dict(np=np, Fraction=Fraction)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    previous = SimpleNamespace(aggregate_independent=namespace['aggregate_independent'])
    scored = {arm: [] for arm in ARMS+FAMILIES}
    summary = dict(domains={}, investment_checks={}, investment_pass={}, attribution={
        'T_strictly_beats_L_both_heads': False, 'T_strictly_beats_I_G_M_both_heads': False})
    for hive in HIVES:
        heads = hive in ('35', '33')
        arms = ARMS+(FAMILIES if heads else ())
        summary['domains'][hive] = dict(methods={})
        for arm in arms:
            for group in range(4 if heads else 3):
                for clip in range(1 if heads else 2):
                    p, tp = (1, 1) if clip == 0 else (3, 0)
                    mm = [dict(member_index=slot, metrics=dict(pred_bins=[[p, 0]], gt_bins=[[1, 0]],
                        iou={t: dict(tp=tp, pred=p, gt=1) for t in ('0.5', '0.3')})) for slot in range(3)]
                    scored[arm].append(dict(domain=hive, group=group, members=mm))
            # Two W/G clips must pool event counts BEFORE computing F1:
            # pooled F1=2/(4+2)=1/3, not mean(1,0)=1/2.
            summary['domains'][hive]['methods'][arm] = dict(E3=0. if heads else 100.,
                E3_fraction=[0 if heads else 100, 1],
                F1={t: 1. if heads else 1/3 for t in ('0.5', '0.3')})
    for arm in ('L', 'T'):
        summary['investment_checks'][arm] = dict(head35_gain=False, head33_gain=False, W_no_harm=True, F1_no_harm=True)
        summary['investment_pass'][arm] = False
    _aggregate_checks(scored, summary, previous)
    return dict(passed=True, field_opened=False, maximum_cardinality_and_tampering=True,
                exact_E3_and_clip_pooling=True)


if __name__ == '__main__':
    print(json.dumps(synthetic(), sort_keys=True))
