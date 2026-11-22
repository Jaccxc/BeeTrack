"""Portable numeric replay for the registered §211 22-state assessment.

Uses only the anonymous compact ledger. Refits source-only constrained heads,
recreates the seven policy choices, and recomputes official event matching,
window error, diagnostics, Pareto sets and strict redundancy witnesses.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import itertools
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import fit211


SCHEMA = 'match211.numeric.v1'
DOMAINS = ('35', '33', 'W', 'G')
ARMS = ('K', 'Q', 'L', 'X', 'V', 'C', 'Z')
THRESHOLDS = ('0.5', '0.3')


def _core(repo_root):
    repo_root = Path(repo_root).resolve()
    if not (repo_root / 'beetrack' / 'p0.py').is_file():
        raise ValueError('--repo must point to final_project')
    sys.path.insert(0, str(repo_root / 'tools' / 'study209'))
    from replay209 import score_arm, _maximum_matching
    from replay_release209 import bins
    return score_arm, _maximum_matching, bins


def _deep_equal(left, right, *, atol=2e-11):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_deep_equal(left[k], right[k], atol=atol) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_deep_equal(a, b, atol=atol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=atol)
    return left == right


def _match_metrics(graph, maximum_matching):
    left = [str(row[0]) for row in graph['left']]
    right = [str(row[0]) for row in graph['right']]
    edges = [[str(p), str(g)] for p, g in graph['edges']]
    if len(left) != len(set(left)) or len(right) != len(set(right)):
        raise ValueError('duplicate legal graph vertex')
    tp = maximum_matching(left, right, edges)
    return left, right, edges, int(tp)


def _state_by_id(clip):
    result = {state['state_id']: state for state in clip['states']}
    if len(result) != len(clip['states']):
        raise ValueError('duplicate state ID')
    return result


def _k_event_rows(clip, maximum_matching):
    states = _state_by_id(clip)
    k = next(s for s in clip['states'] if not s['selected_pair_ids'])
    true = {'left': k['pred_events'], 'right': k['gt_events'],
            'edges': k['true_graphs']['0.5']['edges']}
    left, right, edges, whole_tp = _match_metrics(true, maximum_matching)
    features = {e['event_id']: e for e in clip['k_events']}
    if set(features) != set(left):
        raise ValueError('K peer-feature and true-prediction event rosters differ')
    peer = k['peer_graphs']
    if len(peer) != 2:
        raise ValueError('exactly two peer reference graphs required')
    peer_rows = []
    for graph in peer:
        pids = [str(r[0]) for r in graph['pred']]
        rids = [str(r[0]) for r in graph['reference_events']]
        pedges = [[str(p), str(r)] for p, r in graph['edges']]
        full = maximum_matching(pids, rids, pedges)
        neigh = {p for p, _ in pedges}
        peer_rows.append((pids, rids, pedges, int(full), neigh))
    records = []
    labels = {}
    event_by_id = {}
    for event_id in sorted(left):
        left_edges = [e for e in edges if e[0] != event_id]
        y = whole_tp - maximum_matching([p for p in left if p != event_id], right, left_edges)
        if y not in (0, 1):
            raise ValueError('official K left-deletion criticality must be binary')
        xls, xxs = [], []
        for pids, rids, pedges, full, neigh in peer_rows:
            if event_id not in pids:
                raise ValueError('peer K left-event roster differs from true K')
            xls.append(float(event_id in neigh))
            deleted = maximum_matching([p for p in pids if p != event_id], rids,
                                       [e for e in pedges if e[0] != event_id])
            drop = full - deleted
            if drop not in (0, 1):
                raise ValueError('peer left deletion loss must be 0 or 1')
            xxs.append(float(drop))
        xL = sum(xls) / 2
        xX = sum(xxs) / 2
        frozen = features[event_id]
        if not math.isclose(xL, float(frozen['xL_reference']), abs_tol=1e-12, rel_tol=0):
            raise ValueError('recomputed peer-neighbor feature differs from frozen K')
        if not math.isclose(xX, float(frozen['xX_reference']), abs_tol=1e-12, rel_tol=0):
            raise ValueError('recomputed peer-deletion feature differs from frozen K')
        if xX > xL:
            raise ValueError('peer deletion feature exceeds any-neighbor feature')
        row = {'domain': clip['domain'], 'clip': clip['clip'], 'pid': int(frozen['run_id']),
               'x_L': xL, 'x_X': xX}
        key = fit211.event_key(row)
        labels[key] = int(y)
        event_by_id[event_id] = {'row': row, 'y': int(y)}
        records.append({'event_id': event_id, 'run_id': int(frozen['run_id']),
                        'xL': xL, 'xX': xX, 'y': int(y)})
    return records, labels, event_by_id


def _refit(bundle, maximum_matching):
    all_rows, all_labels, event_index, k_labels = [], {}, {}, {}
    for clip in bundle['clips']:
        records, labels, by_event = _k_event_rows(clip, maximum_matching)
        all_rows.extend(row['row'] for row in by_event.values())
        all_labels.update(labels)
        event_index.update({(row['row']['domain'], row['row']['clip'], row['row']['pid']): event_id
                            for event_id, row in by_event.items()})
        k_labels[clip['domain'], clip['clip']] = records

    models = {}
    for heldout in DOMAINS:
        source = [row for row in all_rows if row['domain'] != heldout]
        source_labels = {fit211.event_key(row): all_labels[fit211.event_key(row)] for row in source}
        model = fit211.fit(source, source_labels, heldout=heldout)
        expected_heads = bundle['models_reference'][heldout]
        for head in ('Q', 'L', 'X'):
            expected = expected_heads[head]
            observed = {key: model[head].get(key) for key in expected}
            if not _deep_equal(observed, expected):
                raise ValueError('refitted source head differs from sealed fit: ' + heldout + '/' + head)

        expected_training = bundle['training_reference'][heldout]
        observed_training = []
        for order, row in enumerate(model['training_records']):
            event_id = event_index[row['domain'], row['clip'], int(row['pid'])]
            observed_training.append({'event_id': event_id, 'run_id': int(row['pid']),
                                      'xL': float(row['x_L']), 'xX': float(row['x_X']),
                                      'y': int(row['y']), 'weight': float(row['weight']),
                                      'source_order': order})
        if not _deep_equal(observed_training, expected_training):
            raise ValueError('refit source event order/labels/weights differ: ' + heldout)
        models[heldout] = model
    references = Counter(r['event_id'] for rows in bundle['training_reference'].values() for r in rows)
    event_ids = set(event_index.values())
    if set(references) != event_ids or any(references[event_id] != 3 for event_id in event_ids):
        raise ValueError('each K event must appear in exactly the three non-heldout source folds')
    return models, k_labels


def _peer_summary(state, nframes, bins, maximum_matching):
    out = []
    for graph in state['peer_graphs']:
        pred, right, edges, cardinality = _match_metrics(
            {'left': graph['pred'], 'right': graph['reference_events'], 'edges': graph['edges']},
            maximum_matching)
        linked = {p for p, _ in edges}
        pred_bins = bins(graph['pred'], nframes)
        right_bins = bins(graph['reference_events'], nframes)
        if pred_bins != graph['pred_bins'] or right_bins != graph['reference_bins']:
            raise ValueError('peer signed-event clock does not reproduce frozen bins')
        out.append({'J': len(linked), 'R': cardinality, 'nP': len(pred), 'nR': len(right),
                    'pred_bins': pred_bins, 'reference_bins': right_bins,
                    'countL1': sum(abs(a - b) for x, y in zip(pred_bins, right_bins)
                                   for a, b in zip(x, y))})
    mean_j = sum(x['J'] for x in out) / 2
    mean_m = sum(x['R'] for x in out) / 2
    mean_l1 = sum(x['countL1'] for x in out) / 2
    if any(not math.isclose(actual, float(state['peer_summary'][name]), rel_tol=0, abs_tol=1e-12)
           for actual, name in ((mean_j, 'meanJ'), (mean_m, 'meanM'), (mean_l1, 'meanCountL1'))):
        raise ValueError('peer graph summary differs from frozen whole-state inputs')
    return out, mean_j, mean_m, mean_l1


def _plan(bundle, models, bins, maximum_matching):
    plans = {}
    frozen_choice = {(c['domain'], c['clip']): c['frozen_choices'] for c in bundle['clips']}
    for clip in bundle['clips']:
        active_ids = {p['pair_id'] for p in clip['active_pairs']}
        if active_ids != set(clip['registered210_slate_ids']):
            raise ValueError('all-active action roster differs from sealed §210 slate')
        endpoint_ids = [int(x) for p in clip['active_pairs'] for x in p['canonical_ids']]
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError('registered §210 active actions are not pairwise disjoint')
        if clip.get('active_roster_is_210_slate') is not True:
            raise ValueError('§210 closure assertion missing')
        fit_clip = {'domain': clip['domain'], 'clip': clip['clip'],
                    'active_pairs': clip['active_pairs'], 'states': []}
        peer_expanded = {}
        for state in clip['states']:
            peer, mean_j, mean_m, mean_l1 = _peer_summary(state, clip['nframes'], bins, maximum_matching)
            n_p = len(state['pred_events'])
            if any(g['nP'] != n_p for g in peer):
                raise ValueError('peer right-scene receiver prediction count differs')
            if state['edits'] != len(state['selected_pair_ids']):
                raise ValueError('declared edit count differs')
            fit_state = {'state_id': state['state_id'], 'selected_pair_ids': state['selected_pair_ids'],
                         'edits': state['edits'], 'nP': n_p, 'meanJ': mean_j,
                         'meanM': mean_m, 'meanCountL1': mean_l1}
            fit_clip['states'].append(fit_state)
            peer_expanded[state['state_id']] = peer
        chosen = fit211.choose_clip(fit_clip, models[clip['domain']])
        if chosen['selected'] != frozen_choice[clip['domain'], clip['clip']]:
            raise ValueError('replayed seven policy choices differ from sealed choices: '
                             + clip['domain'] + '/' + clip['clip'] + ' observed=' + repr(chosen['selected'])
                             + ' frozen=' + repr(frozen_choice[clip['domain'], clip['clip']]))
        by_state = {s['state_id']: s for s in chosen['states']}
        for state in clip['states']:
            expected = state['frozen_utility']
            actual = by_state[state['state_id']]['utilities']
            rounded = by_state[state['state_id']]['rounded_utilities']
            if not _deep_equal(actual, expected):
                raise ValueError('per-state raw utility differs from sealed choices')
            if not _deep_equal(rounded, state['frozen_rounded_utility']):
                raise ValueError('per-state rounded utility differs from sealed choices')
        plans[clip['domain'], clip['clip']] = {
            'state_summaries': chosen['states'], 'selected': chosen['selected'],
            'peer_graph_summaries': peer_expanded,
        }
    return plans


def _score_all(bundle, score_arm, bins, plans):
    clip_results = {}
    for clip in bundle['clips']:
        by_state = {}
        for state in clip['states']:
            if bins(state['gt_events'], clip['nframes']) != state['bins']['gt']:
                raise ValueError('GT signed event times do not reproduce 30-phase bins')
            if bins(state['pred_events'], clip['nframes']) != state['bins']['pred']:
                raise ValueError('prediction signed event times do not reproduce 30-phase bins')
            arm_input = {'bins': state['bins'], 'events': {
                'gt': state['gt_events'], 'pred': state['pred_events'],
                'edges05': state['true_graphs']['0.5']['edges'],
                'edges03': state['true_graphs']['0.3']['edges'],
            }, 'declared': state['declared']}
            score = score_arm(arm_input)
            for threshold, metric in (('0.5', 'tp05'), ('0.3', 'tp03')):
                if score[metric] != int(state['true_graphs'][threshold]['declared_tp']):
                    raise ValueError('true graph cardinality differs from sealed independent TP')
            if (score['gt'] != int(state['declared']['nG']) or score['pred'] != int(state['declared']['nP'])
                    or score['window_error_sum'] != int(state['declared']['numerator'])
                    or score['window_gt_repeated'] != int(state['declared']['denominator'])):
                raise ValueError('true-state official score differs from sealed native metrics')
            by_state[state['state_id']] = score

        selected_scores = {}
        selection = plans[clip['domain'], clip['clip']]['selected']
        for arm in ARMS:
            sid = selection[arm]['state_id']
            selected_scores[arm] = by_state[sid]
        clip_results[clip['domain'], clip['clip']] = {
            'states': by_state, 'selected': selected_scores, 'selection': selection,
        }
    return clip_results


def _aggregate(clip_results, arms=ARMS):
    domains = {}
    for domain in DOMAINS:
        clips = [(key, val) for key, val in clip_results.items() if key[0] == domain]
        if not clips:
            raise ValueError('fixed domain absent')
        policies = {}
        for arm in arms:
            vals = [result['selected'][arm] for _, result in clips]
            den = sum(v['window_gt_repeated'] for v in vals)
            num = sum(v['window_error_sum'] for v in vals)
            gt = sum(v['gt'] for v in vals); pred = sum(v['pred'] for v in vals)
            tp05 = sum(v['tp05'] for v in vals); tp03 = sum(v['tp03'] for v in vals)
            fp05 = pred - tp05; fn05 = gt - tp05
            fp03 = pred - tp03; fn03 = gt - tp03
            policies[arm] = {
                'W_percent': 100.0 * num / den if den else None,
                'F1_05': 2 * tp05 / (gt + pred) if gt + pred else 0.0,
                'F1_03': 2 * tp03 / (gt + pred) if gt + pred else 0.0,
                'nG': gt, 'nP': pred, 'tp05': tp05, 'fp05': fp05, 'fn05': fn05,
                'E05': fp05 + fn05, 'tp03': tp03, 'fp03': fp03, 'fn03': fn03,
                'E03': fp03 + fn03, 'window_numerator': num, 'window_denominator': den,
            }
        domains[domain] = policies
    return domains


def _event_multiset(state):
    counts = Counter((e['family_id'], int(e['signed_frame'])) for e in state['family_events'])
    return counts


def _delta_multiset(before, after):
    a, b = _event_multiset(before), _event_multiset(after)
    def rows(counter):
        return [{'family_id': family, 'signed_frame': frame, 'multiplicity': count}
                for (family, frame), count in sorted(counter.items()) if count]
    return {'unchanged': rows(a & b), 'removed': rows(a - b), 'added': rows(b - a)}


def _singletons(bundle, clip_results, maximum_matching):
    out = {}
    for clip in bundle['clips']:
        states = _state_by_id(clip)
        k_state = next(s for s in clip['states'] if not s['selected_pair_ids'])
        k_score = clip_results[clip['domain'], clip['clip']]['states'][k_state['state_id']]
        action_rows = []
        for pair in clip['active_pairs']:
            pair_ids = set(pair['canonical_ids'])
            singleton = [s for s in clip['states'] if set(s['selected_pair_ids']) == {pair['pair_id']}]
            if len(singleton) != 1:
                raise ValueError('each active pair needs one singleton state')
            state = singleton[0]
            score = clip_results[clip['domain'], clip['clip']]['states'][state['state_id']]
            deltas = {}
            for suffix, tp, fp, fn in (('0.5', 'tp05', 'fp05', 'fn05'), ('0.3', 'tp03', 'fp03', 'fn03')):
                d = {key: int(score[key]) - int(k_score[key]) for key in (tp, fp, fn, 'pred', 'gt')}
                dE = (int(score['pred']) - int(score[tp]) + int(score['gt']) - int(score[tp])) - (
                    int(k_score['pred']) - int(k_score[tp]) + int(k_score['gt']) - int(k_score[tp]))
                d['E'] = dE
                d['FP_only_improvement'] = d[fp] < 0 and d[fn] == 0
                d['FN_only_improvement'] = d[fn] < 0 and d[fp] == 0
                d['both_FP_FN_improvement'] = d[fp] < 0 and d[fn] < 0
                deltas[suffix] = d
            y_by_run = {}
            k_true = {'left': k_state['pred_events'], 'right': k_state['gt_events'],
                      'edges': k_state['true_graphs']['0.5']['edges']}
            left, right, edges, whole_tp = _match_metrics(k_true, maximum_matching)
            neighbors = {p: [] for p in left}
            for p, g in edges:
                neighbors[p].append(g)
            for event in k_state['family_events']:
                for run_id in event['parent_run_ids']:
                    without_event = [e for e in edges if e[0] != event['event_id']]
                    y_by_run.setdefault(int(run_id), set()).add(int(whole_tp - maximum_matching(
                        [p for p in left if p != event['event_id']], right, without_event)))
            weak = {str(run): (next(iter(y_by_run[run])) if len(y_by_run.get(run, set())) == 1 else None)
                    for run in sorted(pair_ids)}

            strict = {}
            for threshold, tpkey, predkey in (('0.5', 'tp05', 'pred'), ('0.3', 'tp03', 'pred')):
                graph0, graph1 = k_state['true_graphs'][threshold], state['true_graphs'][threshold]
                before_by_run = {}
                for ev in k_state['family_events']:
                    if len(ev['parent_run_ids']) == 1 and ev['parent_run_ids'][0] in pair_ids:
                        before_by_run.setdefault(int(ev['parent_run_ids'][0]), []).append(ev)
                old_ok = all(len(before_by_run.get(pid, [])) == 1 for pid in pair_ids)
                old_events = [before_by_run[pid][0] for pid in sorted(pair_ids)] if old_ok else []
                same_direction = old_ok and old_events[0]['direction'] == old_events[1]['direction']
                old_neighbors = []
                for ev in old_events:
                    old_neighbors.append(sorted({g for p, g in graph0['edges'] if p == ev['event_id']}))
                same_singleton = (same_direction and len(old_neighbors[0]) == 1
                                  and old_neighbors[0] == old_neighbors[1])
                target_gt = old_neighbors[0][0] if same_singleton else None
                new_events = [ev for ev in state['family_events']
                              if set(ev['parent_run_ids']) == pair_ids]
                new_neighbors = {ev['event_id']: sorted({g for p, g in graph1['edges'] if p == ev['event_id']})
                                 for ev in new_events}
                survivor_edge = (len(new_events) == 1 and target_gt is not None
                                 and target_gt in new_neighbors.get(new_events[0]['event_id'], []))
                tp_non_decrease = score[tpkey] >= k_score[tpkey]
                pred_minus_one = score[predkey] == k_score[predkey] - 1
                conditions = {
                    'two_original_parent_events': old_ok,
                    'same_direction': bool(same_direction),
                    'same_singleton_K_GT_neighbor': bool(same_singleton),
                    'one_merged_survivor_with_edge': bool(survivor_edge),
                    'whole_scene_TP_not_lower': bool(tp_non_decrease),
                    'whole_scene_predictions_minus_one': bool(pred_minus_one),
                }
                strict[threshold] = {'conditions': conditions,
                                     'strict_event_redundancy_witness': all(conditions.values()),
                                     'anonymous_singleton_GT_event_id': target_gt,
                                     'before_parent_events': [
                                         {'event_id': e['event_id'], 'parent_run_ids': e['parent_run_ids'],
                                          'signed_frame': e['signed_frame'], 'direction': e['direction'],
                                          'legal_GT_neighbors': n}
                                         for e, n in zip(old_events, old_neighbors)],
                                     'new_family_events': [
                                         {'event_id': e['event_id'], 'parent_run_ids': e['parent_run_ids'],
                                          'signed_frame': e['signed_frame'], 'direction': e['direction'],
                                          'legal_GT_neighbors': new_neighbors[e['event_id']]}
                                         for e in new_events],
                                     'exact_same_TP': score[tpkey] == k_score[tpkey],
                                     'true_bee_identity_claim': False}
            baseline_state = k_state
            delta_w = score['W_percent'] - k_score['W_percent']
            action_rows.append({
                'pair_id': pair['pair_id'], 'canonical_ids': pair['canonical_ids'],
                'selected_state_id': state['state_id'], 'weak_K_deletion_labels': weak,
                'delta_window_numerator': score['window_error_sum'] - k_score['window_error_sum'],
                'delta_window_denominator': score['window_gt_repeated'],
                'delta_W_percentage_points': delta_w,
                'event_multiset_delta': _delta_multiset(baseline_state, state),
                'event_metrics': deltas, 'strict_duplicate': strict,
            })
        out[clip['domain'] + '/' + clip['clip']] = action_rows
    return out


def _pareto(bundle, clip_results):
    result = {}
    for clip in bundle['clips']:
        key = (clip['domain'], clip['clip'])
        all_states = clip_results[key]['states']
        rows = []
        for state in clip['states']:
            score = all_states[state['state_id']]
            rows.append({'state_id': state['state_id'], 'selected_pair_ids': state['selected_pair_ids'],
                         'edits': state['edits'], 'score': score})

        def point(row, threshold):
            score = row['score']
            tp = score['tp05' if threshold == '0.5' else 'tp03']
            return (Fraction(score['window_error_sum'], score['window_gt_repeated']),
                    Fraction(2 * tp, score['pred'] + score['gt']))

        frontiers = {}
        for threshold in ('0.5', '0.3'):
            pts = {r['state_id']: point(r, threshold) for r in rows}
            frontier = []
            for row in rows:
                w, f = pts[row['state_id']]
                dominated = any((other_w <= w and other_f >= f and (other_w < w or other_f > f))
                    for other_id, (other_w, other_f) in pts.items() if other_id != row['state_id'])
                if not dominated:
                    score = row['score']
                    frontier.append({'state_id': row['state_id'], 'edits': row['edits'],
                        'W_percent': score['W_percent'], 'F1': score['f1_05' if threshold == '0.5' else 'f1_03'],
                        'numerator': score['window_error_sum'], 'denominator': score['window_gt_repeated'],
                        'tp': score['tp05' if threshold == '0.5' else 'tp03'],
                        'pred': score['pred'], 'gt': score['gt']})
            frontiers[threshold] = frontier
        best_w = min(r['score']['window_error_sum'] for r in rows)
        min_rows = [r for r in rows if r['score']['window_error_sum'] == best_w]
        witness = min(min_rows, key=lambda r: (r['edits'],
            tuple(tuple(map(int, p.split('|'))) for p in r['selected_pair_ids'])))
        sw = witness['score']
        result['/'.join(key)] = {
            'strict_frontier_05': frontiers['0.5'], 'sensitivity_frontier_03': frontiers['0.3'],
            'W_minimum_witness': {
                'state_id': witness['state_id'], 'selected_pair_ids': witness['selected_pair_ids'],
                'edits': witness['edits'], 'window_numerator': best_w,
                'window_denominator': sw['window_gt_repeated'], 'W_percent': sw['W_percent'],
                'F1_05': sw['f1_05'], 'F1_03': sw['f1_03'],
                'same_W_configurations': len(min_rows),
            },
        }
    return result


def _peer_contrasts(bundle, plans, models):
    result = {}
    for clip in bundle['clips']:
        k = next(s for s in clip['states'] if not s['selected_pair_ids'])
        k_jm = k['peer_summary']['meanJ'] - k['peer_summary']['meanM']
        audit = {s['state_id']: s for s in plans[clip['domain'], clip['clip']]['state_summaries']}
        rows = []
        for state in clip['states']:
            peer_delta = state['peer_summary']['meanJ'] - state['peer_summary']['meanM'] - k_jm
            rows.append({'state_id': state['state_id'], 'delta_meanJ': state['peer_summary']['meanJ'] - k['peer_summary']['meanJ'],
                         'delta_meanM': state['peer_summary']['meanM'] - k['peer_summary']['meanM'],
                         'delta_meanJ_minus_M': peer_delta,
                         'delta_nP': audit[state['state_id']]['delta_nP'],
                         'X_utility': audit[state['state_id']]['utilities']['X'],
                         'X_rounded_utility': audit[state['state_id']]['rounded_utilities']['X']})
        result[clip['domain'] + '/' + clip['clip']] = {
            'source_coefficients': {
                'L': {k: models[clip['domain']]['L'][k] for k in ('a', 'b')},
                'X': {k: models[clip['domain']]['X'][k] for k in ('a', 'b')},
            },
            'contrast_all_states': rows,
            'peer_contrast_zero_all_states': all(x['delta_meanJ_minus_M'] == 0 for x in rows),
        }
    return result


def _registered_algebra(bundle, plans, models):
    """Evaluate the preregistered post-result same-coefficient identity only."""
    w_clip = next(c for c in bundle['clips'] if c['domain'] == 'W' and c['source_clip_order'] == 2)
    g_clip = next(c for c in bundle['clips'] if c['domain'] == 'G' and c['source_clip_order'] == 1)
    w_plan = plans['W', w_clip['clip']]['state_summaries']
    g_plan = plans['G', g_clip['clip']]['state_summaries']
    w_state = [s for s in w_plan if s['delta_nP'] == -1 and s['delta_meanM'] == -.5]
    g_state = [s for s in g_plan if s['delta_nP'] == 1 and s['delta_meanM'] == 1.]
    if len(w_state) != 1 or len(g_state) != 1:
        raise ValueError('registered W15/G24 algebra configuration not uniquely present')
    W = models['W']['X']; G = models['G']['X']
    same_coeffs = math.isclose(W['a'], G['a'], rel_tol=0, abs_tol=1e-12) and math.isclose(
        W['b'], G['b'], rel_tol=0, abs_tol=1e-12)
    return {
        'scope': 'post-result algebra under a shared hypothetical (a,b) only; not an inference from distinct heldout folds',
        'W15_configuration': {'state_id': w_state[0]['state_id'], 'delta_nP': -1,
                              'delta_meanM': -.5, 'X_utility_actual_fold': w_state[0]['utilities']['X']},
        'G24_configuration': {'state_id': g_state[0]['state_id'], 'delta_nP': 1,
                              'delta_meanM': 1., 'X_utility_actual_fold': g_state[0]['utilities']['X']},
        'shared_coeff_formula': {
            'W15_utility': '1 - 2a - b', 'G24_utility': '2a - 1 + 2b',
            'sum': 'b', 'if_b_gt_1e-10_then_at_least_one_unrounded_utility_positive': True,
            'if_b_zero_both_zero_only_at_a_half': True,
        },
        'actual_W_and_G_fold_coefficients_equal': same_coeffs,
        'actual_W_fold_X_coefficients': {'a': W['a'], 'b': W['b']},
        'actual_G_fold_X_coefficients': {'a': G['a'], 'b': G['b']},
        'do_not_apply_shared_coefficient_identity_to_actual_loho_choices': True,
    }


def replay(bundle, repo_root):
    if bundle.get('schema') != SCHEMA or bundle.get('marker') != '[peer-event-capacity-v1]':
        raise ValueError('unexpected §211 main numeric evidence schema/marker')
    if bundle.get('receipt', {}).get('primary22_only') is not True:
        raise ValueError('primary replay must be isolated from joint addendum')
    score_arm, maximum_matching, bins = _core(repo_root)
    models, k_labels = _refit(bundle, maximum_matching)

    # Deployment choices use only peer graphs and the source-fitted heads.
    plans = _plan(bundle, models, bins, maximum_matching)
    clip_scores = _score_all(bundle, score_arm, bins, plans)
    pooled = _aggregate(clip_scores)
    singleton = _singletons(bundle, clip_scores, maximum_matching)
    pareto = _pareto(bundle, clip_scores)
    contrasts = _peer_contrasts(bundle, plans, models)
    algebra = _registered_algebra(bundle, plans, models)

    gates = {
        '35': {
            'X_improves_K_by_at_least_1pp': pooled['35']['K']['W_percent'] - pooled['35']['X']['W_percent'] >= 1.,
            'X_strictly_beats_L_and_Z': all(pooled['35']['X']['W_percent'] < pooled['35'][arm]['W_percent'] for arm in ('L', 'Z')),
            'F1_05_nonworse_than_K_by_more_than_.01': pooled['35']['X']['F1_05'] >= pooled['35']['K']['F1_05'] - .01,
        },
        '33': {
            'X_loss_vs_K_at_most_.5pp': pooled['33']['X']['W_percent'] <= pooled['33']['K']['W_percent'] + .5,
            'F1_05_nonworse_than_K_by_more_than_.01': pooled['33']['X']['F1_05'] >= pooled['33']['K']['F1_05'] - .01,
        },
    }
    gates['W'] = {'W_not_worse_than_K': pooled['W']['X']['W_percent'] <= pooled['W']['K']['W_percent']}
    gates['G'] = {'W_not_worse_than_K': pooled['G']['X']['W_percent'] <= pooled['G']['K']['W_percent']}
    gates['investment_pass'] = all(v for domain in ('35', '33') for v in gates[domain].values()) and gates['W']['W_not_worse_than_K'] and gates['G']['W_not_worse_than_K']
    attribution = {}
    for domain in ('35', '33'):
        domain_clips = [c for c in bundle['clips'] if c['domain'] == domain]
        if len(domain_clips) != 1:
            raise ValueError('one fixed target clip expected per heldout domain')
        clip = domain_clips[0]
        chosen = plans[domain, clip['clip']]['selected']
        sx, sl = chosen['X']['state_id'], chosen['L']['state_id']
        xscore = clip_scores[domain, clip['clip']]['states'][sx]
        lscore = clip_scores[domain, clip['clip']]['states'][sl]
        attribution[domain] = {
            'X_and_L_choose_different_configurations': sx != sl,
            'X_state': sx, 'L_state': sl,
            'E05_X': xscore['fp05'] + xscore['fn05'], 'E05_L': lscore['fp05'] + lscore['fn05'],
            'E05_X_strictly_below_L': xscore['fp05'] + xscore['fn05'] < lscore['fp05'] + lscore['fn05'],
        }
    all_clip_x_l_same = all(
        plan['selected']['X']['state_id'] == plan['selected']['L']['state_id']
        for plan in plans.values())
    gates['capacity_attribution_pass'] = all(
        x['X_and_L_choose_different_configurations'] and x['E05_X_strictly_below_L']
        for x in attribution.values())

    return {
        'schema': 'match211.replay-result.v1',
        'primary': {
            'marker': bundle['marker'], 'prereg': bundle['prereg'],
            'receipt': bundle['receipt'], 'refitted_models': {
                h: {head: {key: model[head].get(key) for key in ('a', 'b', 'status', 'objective',
                       'gradient', 'KKT', 'candidates', 'ridge_slope', 'intercept_penalized', 'solver')}
                    for head in ('Q', 'L', 'X')} for h, model in models.items()},
            'clips': {
                '/'.join(key): {'selection': val['selection'],
                                'state_scores': val['states'],
                                'selected_policy_scores': val['selected'],
                                'peer_graph_summaries': plans[key]['peer_graph_summaries']}
                for key, val in clip_scores.items()},
            'pooled': pooled, 'singleton_attribution': singleton,
            'pareto': pareto, 'peer_contrasts': contrasts,
            'registered_algebra': algebra,
            'capacity_attribution': attribution, 'gates': gates,
            'X_L_same_configuration_all_six_clips': all_clip_x_l_same,
        },
        'scope': bundle['scope'],
    }


def replay_joint(bundle, repo_root, primary_models=None):
    """Replay joint211 frozen utilities, choices, matching and windows; no refit."""
    if bundle.get('schema') != 'joint211.numeric.v1' or bundle.get('marker') != '[cross-family-peer-v1]':
        raise ValueError('unexpected joint211 schema/marker')
    score_arm, maximum_matching, bins = _core(repo_root)
    from joint_deploy211 import choose_joint, frozen_utilities
    clip_results, clip_output = {}, {}
    arm_results = {'joint':{}, 'C_only':{}, 'I_only':{}}
    for clip in bundle['clips']:
        actions = {a['action_id']: {'family': a['family'], 'canonical_ids': a['endpoint_ids'], **a}
                   for a in clip['actions']}
        if len(actions) != len(clip['actions']):
            raise ValueError('duplicate joint action ID')
        # Reconstruct the complete legal roster from anonymous endpoint pairs.
        action_ids = sorted(actions)
        feasible = set()
        for mask in range(1 << len(action_ids)):
            used, chosen_ids, valid = set(), [], True
            for i, aid in enumerate(action_ids):
                if not (mask & (1 << i)):
                    continue
                pair = actions[aid]['canonical_ids']
                if (len(pair) != 2 or any(type(x) is not int for x in pair)
                        or pair[0] >= pair[1] or used.intersection(pair)):
                    valid = False
                    break
                used.update(pair); chosen_ids.append(aid)
            if valid:
                feasible.add(frozenset(chosen_ids))
        actual_states = [s['state_id'] for s in clip['states']]
        actual_roster = [frozenset(s['selected_action_ids']) for s in clip['states']]
        if len(set(actual_states)) != len(actual_states) or len(set(actual_roster)) != len(actual_roster):
            raise ValueError('duplicate joint state or selected action set')
        if set(actual_roster) != feasible:
            raise ValueError('joint state inventory is not the complete legal endpoint-disjoint subset roster')
        if primary_models is not None:
            if clip['domain'] not in primary_models:
                raise ValueError('joint domain has no primary fitted model')
            for head in ('Q','L','X'):
                for coefficient in ('a','b'):
                    left = float(clip['model'][head][coefficient])
                    right = float(primary_models[clip['domain']][head][coefficient])
                    if left != right:
                        raise ValueError('joint inherited coefficient differs from primary replay: '
                                         + clip['domain'] + '/' + head + '/' + coefficient)
        states, source_states, peer_by_state, scores = [], {}, {}, {}
        for source_state in clip['states']:
            state = dict(state_id=source_state['state_id'],
                         selected_action_ids=source_state['selected_action_ids'],
                         edits=source_state['edits'], nPred=len(source_state['pred_events']),
                         meanJ=source_state['peer_summary']['meanJ'],
                         meanM=source_state['peer_summary']['meanM'],
                         meanCountL1=source_state['peer_summary']['meanCountL1'])
            states.append(state); source_states[state['state_id']] = source_state
            pred, gt = source_state['pred_events'], source_state['gt_events']
            if bins(pred, clip['nframes']) != source_state['bins']['pred'] or bins(gt, clip['nframes']) != source_state['bins']['gt']:
                raise ValueError('joint signed events do not reproduce 30-phase bins')
            peer_rows = []
            for graph in source_state['peer_graphs']:
                left = [str(row[0]) for row in graph['pred']]
                right = [str(row[0]) for row in graph['reference_events']]
                edges = [[str(a), str(b)] for a,b in graph['edges']]
                if len(left) != len(set(left)) or len(right) != len(set(right)):
                    raise ValueError('duplicate peer graph vertex')
                if dict((str(i),int(f)) for i,f in graph['pred']) != dict((str(i),int(f)) for i,f in pred):
                    raise ValueError('peer graph left receiver does not match full-state prediction events')
                if len(edges) != len({tuple(e) for e in edges}) or any(a not in set(left) or b not in set(right) for a,b in edges):
                    raise ValueError('peer legal edge has duplicate or unknown endpoint')
                pred_bins = bins(graph['pred'], clip['nframes'])
                reference_bins = bins(graph['reference_events'], clip['nframes'])
                if (not isinstance(graph['pred_bins'], list) or not isinstance(graph['reference_bins'], list)
                        or len(graph['pred_bins']) != len(graph['reference_bins'])
                        or len(graph['pred_bins']) != len(pred_bins)
                        or len(graph['reference_bins']) != len(reference_bins)):
                    raise ValueError('peer windows must match the complete offset-window clock shape')
                if any(not isinstance(row,list) or len(row)!=2 or any(type(v) is not int for v in row)
                       for matrix in (graph['pred_bins'],graph['reference_bins']) for row in matrix):
                    raise ValueError('peer windows must contain two-integer rows')
                cardinality = int(maximum_matching(left, right, edges))
                linked = {a for a,_ in edges}
                if pred_bins != graph['pred_bins'] or reference_bins != graph['reference_bins']:
                    raise ValueError('peer prediction or reference event clock differs from frozen windows')
                count_l1 = sum(abs(a-b) for x,y in zip(pred_bins,reference_bins) for a,b in zip(x,y))
                measured = {'J':len(linked),'R':cardinality,'nP':len(left),'nR':len(right),'countL1':count_l1}
                if any(measured[k] != int(graph['declared'][k]) for k in ('J','R','nP','nR')):
                    raise ValueError('joint peer graph statistics differ from declared values')
                if bins(graph['pred'], clip['nframes']) != graph['pred_bins']:
                    raise ValueError('joint peer event clock differs')
                peer_rows.append(measured)
            if len(peer_rows) != 2:
                raise ValueError('joint state must have exactly two peer graphs')
            for key, value in (('meanJ',sum(x['J'] for x in peer_rows)/2),
                               ('meanM',sum(x['R'] for x in peer_rows)/2),
                               ('meanCountL1',sum(x['countL1'] for x in peer_rows)/2)):
                if not math.isclose(value, float(state[key]), rel_tol=0, abs_tol=1e-12):
                    raise ValueError('joint peer summary does not recompute')
            peer_by_state[state['state_id']] = peer_rows
            metrics_input = {'bins':source_state['bins'], 'events':{'gt':gt,'pred':pred,
                'edges05':source_state['true_graphs']['0.5']['edges'],
                'edges03':source_state['true_graphs']['0.3']['edges']},
                'declared':source_state['declared']}
            score = score_arm(metrics_input)
            for threshold, metric in (('0.5','tp05'),('0.3','tp03')):
                graph = source_state['true_graphs'][threshold]
                tp = maximum_matching([str(x[0]) for x in pred], [str(x[0]) for x in gt],
                                      [[str(a),str(b)] for a,b in graph['edges']])
                if tp != int(graph['declared_tp']) or tp != int(score[metric]):
                    raise ValueError('joint official event matching differs')
            if score['window_error_sum'] != int(source_state['declared']['numerator']) or score['window_gt_repeated'] != int(source_state['declared']['denominator']):
                raise ValueError('joint signed-window score differs')
            scores[state['state_id']] = score
        baseline = next(s for s in states if not s['selected_action_ids'])
        for state in states:
            actual = frozen_utilities(clip['model'], state, baseline)
            if not _deep_equal(actual, source_states[state['state_id']]['frozen_utility']):
                raise ValueError('joint utility differs from inherited frozen model')
        selected = choose_joint(states, actions, lambda s: frozen_utilities(clip['model'], s, baseline))
        frozen = {arm: row['state_id'] for arm,row in clip['frozen_choices'].items()}
        if selected['choices'] != frozen:
            raise ValueError('joint seven policy choices differ from frozen choice table')
        for family in ('C','I'):
            if selected['family_only'][family]['choices'] != {a:r['state_id'] for a,r in clip['family_only'][family]['choices'].items()}:
                raise ValueError('joint pure-family policy choices differ from frozen table')
            if set(selected['family_only'][family]['legal_state_ids']) != set(clip['family_only'][family]['legal_state_ids']):
                raise ValueError('joint pure-family legal state roster differs')
        chosen_scores = {f'joint_{arm}':scores[sid] for arm,sid in frozen.items()}
        for family in ('C','I'):
            chosen_scores.update({f'{family}_{arm}':scores[row['state_id']]
                                  for arm,row in clip['family_only'][family]['choices'].items()})
        key = (clip['domain'],clip['clip'])
        clip_results[key] = {'states':scores,'selected':chosen_scores,
            'selection':{arm:{'state_id':sid} for arm,sid in frozen.items()}}
        arm_results['joint'][key] = {'selected':{arm:scores[sid] for arm,sid in frozen.items()}}
        for family in ('C','I'):
            arm_results[family+'_only'][key] = {'selected':{
                arm:scores[row['state_id']] for arm,row in clip['family_only'][family]['choices'].items()}}
        clip_output['/'.join(key)] = {'state_count':len(states),
            'selection':{arm:sid for arm,sid in frozen.items()},
            'family_only_selection':{family:{arm:row['state_id'] for arm,row in clip['family_only'][family]['choices'].items()} for family in ('C','I')},
            'selected_policy_scores':chosen_scores,
            'state_scores':scores,
            'peer_graph_summaries':peer_by_state,
            'state_actions':{s['state_id']:s['selected_action_ids'] for s in states},
            'action_families':{aid:a['family'] for aid,a in actions.items()}}
    pooled = {name:_aggregate(data, tuple(data[next(iter(data))]['selected'].keys()))
              for name,data in arm_results.items()}
    joint = pooled['joint']
    gain = lambda h, arm='X': Fraction(
        100 * (joint[h]['K']['window_numerator'] - joint[h][arm]['window_numerator']),
        joint[h]['K']['window_denominator'])
    f1 = lambda h, arm: Fraction(2 * joint[h][arm]['tp05'],
                                joint[h][arm]['nP'] + joint[h][arm]['nG'])
    gates = {
        'head35_gain_at_least_1pp': gain('35') >= 1,
        'head33_gain_at_least_1pp': gain('33') >= 1,
        'head35_F1_loss_at_most_point01': f1('35','K') - f1('35','X') <= Fraction(1,100),
        'head33_F1_loss_at_most_point01': f1('33','K') - f1('33','X') <= Fraction(1,100),
        'W_no_worse': gain('W') >= 0,
        'G_no_worse': gain('G') >= 0,
    }
    attribution, complementarity, count_minimum = {}, {}, {}
    for h in ('35','33'):
        clip_key, clip_result = next((key,val) for key,val in clip_output.items() if key.startswith(h+'/'))
        sel = clip_result['selection']; scores_h = clip_result['selected_policy_scores']
        x_sid, l_sid = sel['X'], sel['L']
        x_score, l_score = scores_h['joint_X'], scores_h['joint_L']
        attribution[h] = {'X_state':x_sid,'L_state':l_sid,
            'X_differs_from_L':x_sid != l_sid,
            'X_E05':x_score['fp05']+x_score['fn05'],'L_E05':l_score['fp05']+l_score['fn05'],
            'X_has_lower_E05':x_score['fp05']+x_score['fn05'] < l_score['fp05']+l_score['fn05']}
        x_actions = clip_result['state_actions'][x_sid]
        used_families = {clip_result['action_families'][aid] for aid in x_actions}
        x_num = x_score['window_error_sum']
        pure_c = clip_result['family_only_selection']['C']['X']
        pure_i = clip_result['family_only_selection']['I']['X']
        complementarity[h] = {'X_uses_both_action_families':used_families == {'C','I'},
            'X_strictly_below_pure_C_X_window_numerator':x_num < clip_result['state_scores'][pure_c]['window_error_sum'],
            'X_strictly_below_pure_I_X_window_numerator':x_num < clip_result['state_scores'][pure_i]['window_error_sum'],
            'pure_C_X_state':pure_c,'pure_I_X_state':pure_i}
    for h in ('35','33','W','G'):
        clips = [(key,val) for key,val in clip_output.items() if key.startswith(h+'/')]
        per_clip_options = []
        for key,val in clips:
            per_clip_options.append([(sid,score['window_error_sum']) for sid,score in val['state_scores'].items()])
        combos = []
        for combo in itertools.product(*per_clip_options):
            combos.append((sum(x[1] for x in combo), tuple(x[0] for x in combo), combo))
        min_num = min(x[0] for x in combos)
        witnesses = [x for x in combos if x[0] == min_num]
        denom = sum(next(iter(val['state_scores'].values()))['window_gt_repeated'] for _,val in clips)
        # Count minimum is a window-only oracle over scored legal states; the witness F1 is always recomputed.
        count_minimum[h] = {'numerator':min_num,'denominator':denom,'W_percent':100*min_num/denom,
            'witness_count':len(witnesses),'witnesses':[{'state_ids':list(x[1]),
                'F1_05':2*sum(clips[i][1]['state_scores'][x[2][i][0]]['tp05'] for i in range(len(clips))) /
                    (sum(clips[i][1]['state_scores'][x[2][i][0]]['pred'] for i in range(len(clips)))+
                     sum(clips[i][1]['state_scores'][x[2][i][0]]['gt'] for i in range(len(clips)))),
                'F1_03':2*sum(clips[i][1]['state_scores'][x[2][i][0]]['tp03'] for i in range(len(clips))) /
                    (sum(clips[i][1]['state_scores'][x[2][i][0]]['pred'] for i in range(len(clips)))+
                     sum(clips[i][1]['state_scores'][x[2][i][0]]['gt'] for i in range(len(clips))))}
                for x in witnesses]}
    return {'schema':'joint211.replay-result.v1','marker':bundle['marker'],'prereg':bundle['prereg'],
            'receipts':bundle['receipts'],'clips':clip_output,'pooled':pooled,
            'gates':gates,'all_gates_pass':all(gates.values()),'capacity_attribution':attribution,
            'complementarity':complementarity,'count_minimum':count_minimum,
            'scope':'Replays inherited utilities and choices, official event matching, peer matching and windows from anonymous numeric states. Raw geometry and owner reassignment are outside the replay.'}


def _load_bundle(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--joint', type=Path, help='optional separate joint211 numeric bundle')
    args = parser.parse_args()
    if args.bundle.suffix == '.gz':
        with gzip.open(args.bundle, 'rt', encoding='utf8') as stream:
            bundle = json.load(stream)
    else:
        bundle = json.loads(args.bundle.read_text(encoding='utf8'))
    result = replay(bundle, args.repo)
    if args.joint:
        joint_bundle = _load_bundle(args.joint)
        result['joint'] = replay_joint(joint_bundle, args.repo, result['primary']['refitted_models'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
