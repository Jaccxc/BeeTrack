"""Numeric replay of the finalized208 release; no image or raw GT input.

Fixed scores are replayed, not inferred again from unpublished natural crops.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import math
import json
import sys
from pathlib import Path
from numeric208_paths import state_events

ARMS = ('geometry_interval', 'relative_cross_exact', 'relative_cross_interval')
STATES = ('K', *(a + suffix for a in ARMS for suffix in ('_CUT', '_CUT_RELINK')),
          'relative_cross_interval_DISPLACED_CUT',
          'relative_cross_interval_DISPLACED_CUT_RELINK')
CLIPS = (('35', 'BEE24-35-head', 3), ('33', 'BEE24-33-head', 3),
         ('W', 'BEE24-10', 0), ('W', 'BEE24-15', 0),
         ('G', 'BEE24-24', 0), ('G', 'BEE24-28', 0))


def source_threshold(values, *, active=True):
    """Exact higher empirical .999 quantile, retaining source multiplicities."""
    assert all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
               for v in values)
    if not active or not values:
        return 1.0
    position = (999 * (len(values) - 1) + 999) // 1000
    return max(.5, sorted(values)[position])


def select_cuts(cuts, scores, threshold, *, active=True):
    assert len(cuts) == len(scores) and .5 <= threshold <= 1
    assert [c['candidate'] for c in cuts] == list(range(len(cuts)))
    if not active:
        return []
    for cut, score in zip(cuts, scores):
        if cut['valid']:
            assert score is not None and math.isfinite(score) and 0 <= score <= 1
        else:
            assert score is None or (math.isfinite(score) and 0 <= score <= 1)
    proposed = [c for c, p in zip(cuts, scores) if c['valid'] and p > threshold]
    proposed.sort(key=lambda c: (-scores[c['candidate']], c['right_frame'], c['candidate']))
    accepted = defaultdict(list)
    selected = []
    for cut in proposed:
        if any(abs(old['right_frame'] - cut['right_frame']) <= 3
               for old in accepted[cut['run_key']]):
            continue
        accepted[cut['run_key']].append(cut)
        selected.append(cut['candidate'])
    return sorted(selected)


def displaced_scores(cuts, scores):
    """Rotate right only inside uninterrupted valid observation-cut stretches."""
    assert len(cuts) == len(scores)
    by_run = defaultdict(list)
    for cut in cuts:
        if cut['valid']:
            by_run[cut['run_key']].append(cut)
    output = list(scores)
    for records in by_run.values():
        stretches = []
        for cut in sorted(records, key=lambda c: c['right_ordinal']):
            if not stretches or cut['right_ordinal'] != stretches[-1][-1]['right_ordinal'] + 1:
                stretches.append([])
            stretches[-1].append(cut)
        for stretch in stretches:
            ids = [c['candidate'] for c in stretch]
            for destination, source in zip(ids, ids[-1:] + ids[:-1]):
                output[destination] = scores[source]
    return output


def event_bins(signed_dates, nframes):
    assert type(nframes) is int and nframes > 0
    result = []
    for offset in range(30):
        windows = [[0, 0] for _ in range((nframes + 25 * offset + 749) // 750)]
        for date in signed_dates:
            assert type(date) is int and date and 1 <= abs(date) <= nframes
            windows[(abs(date) - 1 + 25 * offset) // 750][int(date < 0)] += 1
        result.extend(windows)
    return result


def maximum_matching(edges, npred, ngt):
    assert len(edges) == len(set(map(tuple, edges)))
    graph = defaultdict(list)
    for p, g in edges:
        assert type(p) is int and type(g) is int and 0 <= p < npred and 0 <= g < ngt
        graph[p].append(g)
    match = {}

    def augment(p, seen):
        for g in graph[p]:
            if g in seen:
                continue
            seen.add(g)
            if g not in match or augment(match[g], seen):
                match[g] = p
                return True
        return False

    return sum(augment(p, set()) for p in graph)


def score_statistics(state, nframes):
    """Aggregated truth windows and opaque graph nodes, never raw GT rows."""
    predicted = event_bins(state['predicted_signed_dates'], nframes)
    assert predicted == state['pred_bins']
    truth = state['truth_window_counts']
    assert len(truth) == len(predicted)
    assert all(len(row) == 2 and all(type(x) is int and x >= 0 for x in row) for row in truth)
    denominator = sum(map(sum, truth))
    assert denominator == 30 * state['nGT']
    numerator = sum(abs(p - g) for a, b in zip(predicted, truth) for p, g in zip(a, b))
    counts = {}
    for threshold in ('0.5', '0.3'):
        value = state['official'][threshold]
        tp = maximum_matching(state['opaque_edges'][threshold], len(state['predicted_signed_dates']), state['nGT'])
        assert (tp, len(state['predicted_signed_dates']), state['nGT']) == (value['tp'], value['pred'], value['gt'])
        counts[threshold] = dict(TP=tp, FP=value['pred'] - tp, FN=value['gt'] - tp,
                                 F1=Fraction(2 * tp, max(value['pred'] + value['gt'], 1)))
        assert math.isclose(value['f1'], float(counts[threshold]['F1']), rel_tol=1e-12, abs_tol=1e-12)
    return dict(numerator=numerator, denominator=denominator, counts=counts)


def replay_frame_costs(frames):
    """Bootstrap the official assignment formula, only if costs were exported."""
    from scipy.optimize import linear_sum_assignment
    count = 0
    for frame in frames:
        ng, npred = frame['gt_row_count'], frame['pred_row_count']
        matrix = [[0.0] * npred for _ in range(ng)]
        seen = set()
        for g, p, value in frame['iou_nonzero']:
            assert 0 <= g < ng and 0 <= p < npred and (g, p) not in seen
            assert math.isfinite(value) and 0 < value <= 1
            seen.add((g, p))
            matrix[g][p] = value
        pairs = list(zip(*linear_sum_assignment([[-x for x in row] for row in matrix]))) if ng and npred else []
        for threshold in ('0.5', '0.3'):
            owners = [-1] * npred
            for g, p in pairs:
                if matrix[g][p] >= float(threshold):
                    owners[int(p)] = int(g)
            assert owners == frame['opaque_owner_positions'][threshold]
        count += 1
    return count


def _equal(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _equal(a, b)
    elif isinstance(actual, float):
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    else:
        assert actual == expected


def run_contract(contract, component, diagnostics):
    """Pure normalized-contract adapter. `diagnostics` is the sealed verifier.

    Component storage is injected so final packaging can be sharded/streamed.
    No image, native MOT, original GT file or tensor model is loaded here.
    """
    from prepare208_release import validate_contract
    validate_contract(contract)
    thresholds = {}
    source_negative_indices = defaultdict(set)
    for fold in contract['folds']:
        heldout = fold['heldout']
        assert set(fold['models']) == set(ARMS)
        for arm, model in fold['models'].items():
            calibration = component(model['calibration_component'])
            assert all(r['hive'] != heldout and r['kind'] == 'exact_survival'
                       and r['length'] == 1 and r['valid'] for r in calibration['negative_records'])
            probabilities = [r['score'] for r in calibration['negative_records']]
            for r in calibration['negative_records']:
                source_negative_indices[r['clip']].add(r['cut_index'])
            active = model['status'] == 'trained'
            threshold = source_threshold(probabilities, active=active)
            assert model['threshold'] == threshold
            if active:
                assert diagnostics.calibration_threshold(probabilities) == threshold
            thresholds[heldout, arm] = threshold, active
    pooled = defaultdict(list)
    clip_reports = []
    count_states = 0
    verified_frame_costs = 0
    for clip in contract['clips']:
        catalog = component(clip['catalog_component'])
        forecasts = component(clip['prediction_component'])
        states = component(clip['state_component'])
        truth_rows = component(clip['diagnostic_component'])
        if clip.get('frame_matching_component') is not None:
            verified_frame_costs += replay_frame_costs(component(clip['frame_matching_component']))
        assert set(states) == set(STATES) and set(forecasts) == set(ARMS)
        n = len(catalog['cuts'])
        assert catalog['native_rows'] == sum(r['nrows'] for r in catalog['runs'])

        def mother_rows():
            if 'adjacency_mother' in truth_rows:
                yield from truth_rows['adjacency_mother']
            else:
                for identifier in truth_rows['adjacency_components']:
                    yield from component(identifier)

        census = {r['run_key']: bytearray(r['nrows']-1) for r in catalog['runs']}
        unverified_negatives = set(source_negative_indices[clip['domain']+'/'+clip['clip']])
        for row in mother_rows():
            run, ordinal = row['run_key'], row['right_ordinal']
            assert run in census and 1 <= ordinal <= len(census[run])
            assert not census[run][ordinal-1]
            census[run][ordinal-1] = 1
            if row['candidate'] is not None:
                cut = catalog['cuts'][row['candidate']]
                assert row['represented'] and row['valid'] == cut['valid']
                assert (run, ordinal) == (cut['run_key'], cut['right_ordinal'])
                assert row['right_frame'] == cut['right_frame']
                assert row['delta_frame'] == cut['right_frame']-cut['left_frame']
                if row['candidate'] in unverified_negatives:
                    assert row['valid'] and row['length'] == 1 and row['kind'] == 'exact_survival'
                    unverified_negatives.remove(row['candidate'])
            else:
                assert not row['represented'] and not row['valid']
        assert all(all(flags) for flags in census.values())
        assert not unverified_negatives
        selections, displaced = {}, None
        adjacency, selected_adjacency, timing = {}, {}, {}
        for arm, forecast in forecasts.items():
            scores = forecast['probabilities']
            assert len(scores) == n
            threshold, active = thresholds[clip['domain'], arm]
            assert forecast['threshold'] == threshold
            masks = [bool(c['valid'] and p is not None and p > threshold)
                     for c, p in zip(catalog['cuts'], scores)]
            assert masks == forecast['selected']
            assert [bool(c['valid'] and p is not None and p > .5)
                    for c, p in zip(catalog['cuts'], scores)] == forecast['raw05']
            chosen = select_cuts(catalog['cuts'], scores, threshold, active=active)
            for suffix in ('_CUT', '_CUT_RELINK'):
                selections[arm + suffix] = chosen
            # Keep all known mother positives, including invalid/no-input rows.
            def diagnostic_rows():
                for source in mother_rows():
                    row = dict(source)
                    index = row['candidate']
                    if index is None:
                        assert not row['represented']
                        row['score'] = float('nan')
                    else:
                        assert 0 <= index < n and row['represented']
                        assert row['valid'] == catalog['cuts'][index]['valid']
                        row['score'] = float('nan') if scores[index] is None else scores[index]
                    yield row
            adjacency[arm] = diagnostics.binary_diagnostics(diagnostic_rows(), threshold)
            chosen_set = set(chosen)
            tp = fp = positives = unknown = 0
            for row in diagnostic_rows():
                exact = row['length'] == 1 and row['kind'] in ('exact_switch', 'exact_survival')
                positive = exact and row['kind'] == 'exact_switch'
                chosen_here = row['candidate'] in chosen_set
                positives += positive
                tp += positive and chosen_here
                fp += exact and not positive and chosen_here
                unknown += not exact and chosen_here
            selected_adjacency[arm] = dict(TP=tp, FP=fp, pipeline_FN=positives-tp,
                all_known_positive=positives,
                selected_unknown=unknown)
            timing_rows = (dict(r, score=r['score'] if r['candidate'] in chosen_set
                                or not math.isfinite(r['score']) else min(r['score'], threshold)) for r in diagnostic_rows())
            timing[arm] = diagnostics.timing_diagnostics(timing_rows, threshold)
            if arm == 'relative_cross_interval':
                displaced = displaced_scores(catalog['cuts'], scores)
                selected = select_cuts(catalog['cuts'], displaced, threshold, active=active)
                for suffix in ('_DISPLACED_CUT', '_DISPLACED_CUT_RELINK'):
                    selections[arm + suffix] = selected
        selections['K'] = []
        for name, state in states.items():
            reconnect = name.endswith('_CUT_RELINK')
            selected = selections[name]
            dates, pieces, links = state_events(catalog, selected, reconnect)
            assert sorted(state['predicted_signed_dates']) == dates
            if name != 'K':
                assert state['deployment']['nms']['selected'] == selected
                assert state['deployment']['cut']['selected'] == selected
                projected = state['deployment']['cut']['segments']
                assert len(projected) == len(pieces)
                for raw, piece in zip(projected, pieces):
                    for key in ('segment_id', 'original_run', 'first_raw', 'last_raw',
                                'first_frame', 'last_frame', 'new_start', 'new_end', 'nrows'):
                        assert raw[key] == piece[key]
                if links is not None:
                    for key in ('eligible_edges', 'mutual_proposals', 'accepted_edges', 'rejected_edges'):
                        _equal(links[key], state['deployment']['relink'][key])
            rebuilt = score_statistics(state, clip['nframes'])
            assert rebuilt['numerator'] == state['numerator']
            assert rebuilt['denominator'] == state['denominator']
            assert state['physical_rows'] == catalog['native_rows'] and state['raw_row_bijection']
            for threshold in ('0.5', '0.3'):
                value = state['official'][threshold]
                pooled[clip['domain'], name, threshold].append(dict(
                    numerator=rebuilt['numerator'], denominator=rebuilt['denominator'],
                    TP=value['tp'], Pred=value['pred'], GT=value['gt']))
            count_states += 1
        clip_reports.append(dict(domain=clip['domain'], clip=clip['clip'],
                                 conditional_adjacency=adjacency,
                                 selected_adjacency=selected_adjacency, physical_pm1=timing))
        print('Verified nine numeric states:', clip['domain'], clip['clip'], file=sys.stderr, flush=True)
    domains = defaultdict(dict)
    for (hive, state, threshold), clips in pooled.items():
        assert len(clips) == (1 if hive in ('35', '33') else 2)
        domains[hive].setdefault(state, {})[threshold] = diagnostics.pooled_physical(clips)
    assert count_states == 54

    def W(hive, state):
        return Fraction(*domains[hive][state]['0.5']['W_fraction'])

    def F(hive, state):
        return Fraction(*domains[hive][state]['0.5']['F1_fraction'])

    main = 'relative_cross_interval_CUT_RELINK'
    checks = dict(head35_gain_at_least1=W('35', 'K') - W('35', main) >= 1,
                  head35_strictly_beats_geometry_exact_displaced=all(
                      W('35', main) < W('35', name) for name in (
                          'geometry_interval_CUT_RELINK', 'relative_cross_exact_CUT_RELINK',
                          'relative_cross_interval_DISPLACED_CUT_RELINK')),
                  head33_no_harm_halfpoint=W('33', main) - W('33', 'K') <= Fraction(1, 2),
                  head_F1_no_harm_onehundredth=all(
                      F(hive, main) >= F(hive, 'K') - Fraction(1, 100) for hive in ('35', '33')))
    expected = component(contract['summary_component'])
    for hive, methods in domains.items():
        assert set(methods) == set(expected['domains'][hive]['methods'])
        for state, state_thresholds in methods.items():
            original = expected['domains'][hive]['methods'][state]
            assert Fraction(*state_thresholds['0.5']['W_fraction']) == Fraction(100*original['numerator'], original['denominator'])
            for t, values in state_thresholds.items():
                counts = original['iou'][t]
                assert (values['TP'], values['Pred'], values['GT']) == (counts['tp'], counts['pred'], counts['gt'])
    assert expected['registered_gate']['checks'] == checks
    assert expected['registered_gate']['all_checks'] == all(checks.values())
    target_diagnostics = pooled_targets(contract, component, diagnostics, thresholds)
    return dict(clips=clip_reports, domains=dict(domains), state_count=count_states,
                targets=target_diagnostics,
                registered_gate=dict(checks=checks, all_checks=all(checks.values())),
                CNN_forward_or_refit_verified=False,
                supplied_frame_assignment_witnesses_verified=verified_frame_costs,
                full_native_frame_IoU_recomputed=False)


def pooled_targets(contract, component, diagnostics, thresholds):
    """Pool target clips before AP, preserving unscored mother positives."""
    result = {}
    for hive in ('35', '33', 'W', 'G'):
        result[hive] = {}
        for arm in ARMS + ('relative_cross_interval_DISPLACED',):
            base = arm.replace('_DISPLACED', '')
            threshold, active = thresholds[hive, base]
            def rows(nms=False):
                for clip in contract['clips']:
                    if clip['domain'] != hive:
                        continue
                    catalog = component(clip['catalog_component'])
                    forecast = component(clip['prediction_component'])[base]
                    scores = forecast['probabilities']
                    if arm.endswith('_DISPLACED'):
                        scores = displaced_scores(catalog['cuts'], scores)
                    chosen = set(select_cuts(catalog['cuts'], scores, threshold, active=active))
                    header = component(clip['diagnostic_component'])
                    batches = (component(k) for k in header['adjacency_components'])
                    for batch in batches:
                        for old in batch:
                            row = dict(old)
                            index = row['candidate']
                            row['run_key'] = clip['domain']+'/'+clip['clip']+'|p'+str(row['run_key'])
                            row['score'] = (float(index in chosen) if nms else
                                float('nan') if index is None or scores[index] is None else scores[index])
                            yield row
            raw = diagnostics.binary_diagnostics(rows(), .5)
            calibrated = diagnostics.binary_diagnostics(rows(), threshold)
            nms = diagnostics.binary_diagnostics(rows(True), .5)
            nms.pop('AP_fraction')
            result[hive][arm] = dict(raw05=raw, calibrated=calibrated, NMS=nms,
                timing_conservative=diagnostics.timing_diagnostics(rows(True), .5))
        print('Verified pooled target diagnostics:', hive, file=sys.stderr, flush=True)
    return result


def self_test():
    # Defined for a later sole-CPU synthetic dispatch; not run during preparation.
    assert source_threshold([.1, .2, .9]) == .9
    assert source_threshold([]) == 1
    cuts = [dict(candidate=i, run_key='r', right_ordinal=i, right_frame=10 + i,
                 valid=(i != 2)) for i in range(5)]
    scores = [.6, .9, None, .8, .7]
    assert displaced_scores(cuts, scores) == [.9, .6, None, .7, .8]
    assert select_cuts(cuts, scores, .7) == [1]
    assert maximum_matching([(0, 0), (0, 1), (1, 0)], 2, 2) == 2
    assert sum(map(sum, event_bins([2, -20], 30))) == 60
    catalog = dict(runs=[dict(run_key=1, nrows=4, first_raw=0, last_raw=3,
                             side_changes=[dict(ordinal=2, frame=3, inside=True)]),
                         dict(run_key=2, nrows=2, first_raw=4, last_raw=5,
                              side_changes=[dict(ordinal=1, frame=6, inside=False)])],
                   endpoints={str(i): dict(frame=i+1, inside=i in (2, 3, 4),
                                           center=[1 if i in (2, 3, 4) else -1, 0],
                                           diagonal=4.) for i in range(6)},
                   cuts=[dict(candidate=0, run_key=1, right_ordinal=2,
                              left_raw=1, right_raw=2, valid=True)])
    assert state_events(catalog, [], False)[0] == [-6, 3]
    assert state_events(catalog, [0], False)[0] == [-6]
    assert state_events(catalog, [0], True)[0] == []
    return 'PASS numeric-only synthetic; no crop forward/refit/native claim'


def replay_archive(evidence, work_dir=None):
    """Finalized normalized archives only; actual field projection is separate."""
    import gzip
    import hashlib
    import importlib.util
    import sys
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, str(Path.cwd()/'tools/study203'))
    from archive203 import open_archive
    from tools.study202.stream202 import verified_path
    from prepare208_release import public_numeric
    with open_archive(evidence, work_dir) as root:
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['schema'] == 'switch208-public-numeric-shards-v1'
        entries = {e['role']+'|'+e['key']: e for e in manifest['shards']}
        assert len(entries) == len(manifest['shards'])

        def component(identifier):
            entry = entries[identifier]
            path = verified_path(root, entry)
            digest, size = hashlib.sha256(), 0
            with gzip.open(path, 'rb') as stream:
                for chunk in iter(lambda: stream.read(65536), b''):
                    digest.update(chunk)
                    size += len(chunk)
            assert digest.hexdigest() == entry['inflated_sha256'] and size == entry['inflated_bytes']
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                value = json.load(stream, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
            public_numeric(value)
            return value

        contract = component('contract|shared')
        source = Path(__file__).with_name('verify208.py')
        assert hashlib.sha256(source.read_bytes()).hexdigest() == contract['diagnostic_source_sha256']
        spec = importlib.util.spec_from_file_location('accepted208_diagnostic_math', source)
        diagnostics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnostics)
        return run_contract(contract, component, diagnostics)


def json_report(value):
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    raise TypeError(type(value).__name__)


def presentation(value):
    """Keep exact AP arithmetic internal; present its numeric value compactly."""
    if isinstance(value, dict):
        return {('AP' if k == 'AP_fraction' else k):
                (None if v is None else float(Fraction(*v))) if k == 'AP_fraction'
                else presentation(v) for k, v in value.items()}
    if isinstance(value, list):
        return [presentation(v) for v in value]
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--duplicate-census', type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(self_test())
    elif args.evidence:
        result = replay_archive(args.evidence, args.work_dir)
        if args.duplicate_census:
            from duplicate208 import replay as duplicate_replay
            result['duplicate_census'] = duplicate_replay(args.duplicate_census, result)
        print(json.dumps(presentation(result), default=json_report, indent=2))
    else:
        parser.error('Supply a finalized numeric archive or an explicitly authorized synthetic self-test')
