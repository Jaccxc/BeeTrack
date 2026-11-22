"""Project the sealed §211 main assessment into a compact anonymous ledger.

This private packer reads only sealed numeric stage files. It does not publish,
mutate producer stages, or copy native matrices/GT/coordinate data.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

DOMAINS = ('35', '33', 'W', 'G')
ARMS = ('K', 'Q', 'L', 'X', 'V', 'C', 'Z')
PREREG = '0e005ffeee692b104a4a6430843c0af721b6bfa1'
MARKER = '[peer-event-capacity-v1]'
SEAL210 = '3810c1afabe45160642b6187c8fed89dac4e5e05dabb9f5be2cb696d666a7005'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf8'))


def read_gzip(path: Path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return json.load(stream)


def verify_tree(root: Path) -> tuple[str, dict]:
    root = root.resolve()
    manifest_path = root / 'manifest.json'
    digest = sha(manifest_path)
    if digest != (root / 'SEALED.sha256').read_text(encoding='ascii').strip():
        raise ValueError('stage seal differs: ' + root.name)
    manifest = read_json(manifest_path)
    for rel, expected in manifest.items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise ValueError('sealed stage member differs: ' + root.name)
    return digest, manifest


def verify_sidecar(path: Path) -> str:
    digest = sha(path)
    sidecar = path.with_suffix('.sha256')
    if not sidecar.is_file() or sidecar.read_text(encoding='ascii').strip() != digest:
        raise ValueError('verification sidecar differs: ' + path.name)
    return digest


def _pair(value: str) -> tuple[int, int]:
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]+\|[0-9]+', value):
        raise ValueError('ascending numeric canonical pair required')
    a, b = map(int, value.split('|'))
    if a >= b:
        raise ValueError('canonical pair endpoints must increase')
    return a, b


def _sort_pairs(values):
    return sorted(values, key=_pair)


def _int_vector(rows, name):
    out = []
    if not isinstance(rows, list):
        raise ValueError(name + ' must be an array')
    for row in rows:
        if not isinstance(row, list) or len(row) != 2 or any(type(x) is not int for x in row):
            raise ValueError(name + ' must be integer pairs')
        out.append([int(row[0]), int(row[1])])
    return out


def _event_frame(row):
    return int(row[1])


def _event_id(parent_ids, signed_frame, direction, event_map):
    key = (tuple(sorted(map(int, parent_ids))), int(signed_frame), int(direction))
    return event_map[key]


def _project_clip(root, domain, source_clip, clip_id, frozen_ref, prediction_clip,
                  assessment_ref, canonical_map, event_map, gt_map, ref_maps, pair_map,
                  original210_ids):
    frozen_dir = root / 'frozen' / frozen_ref['path']
    frozen = read_gzip(frozen_dir / 'states.json.gz')
    assessment_paths = {s['state_id']: s['path'] for s in assessment_ref['states']}
    pred_state_by_id = {s['state_id']: s for s in prediction_clip['states']}
    frozen_state_by_id = {s['state_id']: s for s in frozen['states']}
    if set(assessment_paths) != set(frozen_state_by_id) or set(pred_state_by_id) != set(frozen_state_by_id):
        raise ValueError('frozen, fit and true assessment state inventories differ')

    active_pairs = []
    source_ids = set()
    for pair in frozen['active_pairs']:
        a, b = map(int, pair['canonical_ids'])
        if not a < b or pair['pair_id'] != f'{a}|{b}':
            raise ValueError('active-pair canonical lineage differs')
        source_ids.update((a, b))
        new_pair = pair_map[pair['pair_id']]
        active_pairs.append({'pair_id': new_pair,
                             'canonical_ids': [canonical_map[a], canonical_map[b]],
                             'min_offset_norm': float(pair['min_offset_norm'])})
    active_pairs.sort(key=lambda p: tuple(p['canonical_ids']))

    k_state = next(s for s in frozen['states'] if s['is_K'])
    k_features = {int(r['canonical_id']): r for r in frozen['K_event_features']}
    k_score = read_json(root / 'assessment' / assessment_paths[k_state['state_id']] / 'score.json')
    k_families = {int(r['pid']): r for r in k_score['family_events']}
    if set(k_features) != set(k_families):
        raise ValueError('frozen peer K event feature inventory differs from K true event inventory')
    k_events = []
    for pid in sorted(k_features):
        feature = k_features[pid]
        event = k_families[pid]
        eid = _event_id(event['original_parent_ids'], event['signed_frame'], event['direction'], event_map)
        k_events.append({'event_id': eid, 'run_id': canonical_map[pid], 'xL_reference': float(feature['x_L']),
                         'xX_reference': float(feature['x_X'])})

    state_rows = []
    for source_state in sorted(frozen['states'], key=lambda s: (s['state_index'], s['state_id'])):
        sid = source_state['state_id']
        prediction_state = pred_state_by_id[sid]
        selected = [pair_map[p] for p in source_state['selected_pair_ids']]
        if set(selected) != set(pair_map[p] for p in prediction_state['selected_pair_ids']):
            raise ValueError('frozen state and calibrated choice roster differ')
        score = read_json(root / 'assessment' / assessment_paths[sid] / 'score.json')
        metrics, family = score['metrics'], score['family_events']
        fam_by_pid = {int(v['pid']): v for v in family}
        if len(fam_by_pid) != len(family):
            raise ValueError('duplicate prediction event family PID')
        pred_map = {}
        pred_vertices = []
        for raw in metrics['evidence']['predictions']:
            pid, signed_frame = int(raw[0]), int(raw[1])
            if pid not in fam_by_pid or int(fam_by_pid[pid]['signed_frame']) != signed_frame:
                raise ValueError('native event and family provenance differ')
            event = fam_by_pid[pid]
            eid = _event_id(event['original_parent_ids'], signed_frame, event['direction'], event_map)
            pred_map[pid] = eid
            pred_vertices.append([eid, signed_frame])
        if set(pred_map) != set(fam_by_pid):
            raise ValueError('prediction evidence and family-event inventory differ')
        pred_vertices.sort()

        gt_ids = {int(r[0]) for r in metrics['evidence']['gt']}
        if not gt_ids <= set(gt_map):
            raise ValueError('GT event missing from cross-state anonymous map')
        gt_vertices = sorted([[gt_map[int(r[0])], int(r[1])] for r in metrics['evidence']['gt']])

        true_graphs = {}
        for threshold, edgekey in (('0.5', 'edges05'), ('0.3', 'edges03')):
            graph = score['true_graphs'][threshold]
            graph_left = {int(v['pid']): int(v['signed_frame']) for v in graph['left_vertices']}
            graph_right = {int(v['gt_event_id']): int(v['signed_frame']) for v in graph['right_vertices']}
            if graph_left != {int(r[0]): int(r[1]) for r in metrics['evidence']['predictions']}:
                raise ValueError('true graph left vertices differ from official event evidence')
            if graph_right != {int(r[0]): int(r[1]) for r in metrics['evidence']['gt']}:
                raise ValueError('true graph right vertices differ from official event evidence')
            raw_edges = graph['edges']
            evidence_edges = metrics['evidence'][edgekey]
            if {tuple(map(int, e)) for e in raw_edges} != {tuple(map(int, e)) for e in evidence_edges}:
                raise ValueError('true graph legal edges differ from official score evidence')
            true_graphs[threshold] = {
                'edges': sorted([[pred_map[int(p)], gt_map[int(g)]] for p, g in raw_edges]),
                'declared_tp': int(graph['R']),
            }

        family_rows = []
        for event in family:
            parents = sorted(canonical_map[int(x)] for x in event['original_parent_ids'])
            famkey = tuple(sorted(map(int, event['family'])))
            family_rows.append({
                'event_id': _event_id(event['original_parent_ids'], event['signed_frame'], event['direction'], event_map),
                'family_id': 'f' + '_'.join(str(canonical_map[x]) for x in famkey),
                'parent_run_ids': parents,
                'frame': int(event['frame']), 'direction': int(event['direction']),
                'signed_frame': int(event['signed_frame']),
            })
        family_rows.sort(key=lambda e: (e['event_id'], e['family_id']))

        peers = []
        for peer_index, graph in enumerate(source_state['peer_graphs']):
            ref_map = ref_maps[peer_index]
            p_vertices = []
            for v in graph['prediction_vertices']:
                pid, signed = int(v['canonical_id']), int(v['signed_frame'])
                event = fam_by_pid.get(pid)
                if event is None or int(event['signed_frame']) != signed:
                    raise ValueError('peer graph left event differs from official true prediction')
                eid = _event_id(event['original_parent_ids'], signed, event['direction'], event_map)
                p_vertices.append([eid, signed])
            r_vertices = sorted([[ref_map[int(v['canonical_id'])], int(v['signed_frame'])]
                                 for v in graph['reference_vertices']])
            peer_edges = sorted([[pred_map[int(p)], ref_map[int(r)]] for p, r in graph['edges']])
            if {x[0] for x in p_vertices} != set(pred_map.values()):
                raise ValueError('peer and true left event sets differ')
            peers.append({
                'reference': 'peer' + str(peer_index),
                'pred': sorted(p_vertices), 'reference_events': r_vertices,
                'edges': peer_edges,
                'pred_bins': _int_vector(graph['pred_windows'], 'peer prediction bins'),
                'reference_bins': _int_vector(graph['reference_windows'], 'peer reference bins'),
                'declared': {'J': int(graph['J']), 'R': int(graph['R']),
                             'nP': int(graph['nPred']), 'nR': int(graph['nReference'])},
            })

        bins = {'gt': _int_vector(metrics['gt_bins'], 'gt bins'),
                'pred': _int_vector(metrics['pred_bins'], 'pred bins')}
        iou = metrics['iou']
        declared = {
            'nG': int(metrics['nG']), 'nP': int(metrics['nP']),
            'tp05': int(iou['0.5']['tp']), 'tp03': int(iou['0.3']['tp']),
            'numerator': int(score['numerator']), 'denominator': int(score['denominator']),
            'W': float(score['W']), 'f105': float(iou['0.5']['f1']),
            'f103': float(iou['0.3']['f1']),
        }
        state_rows.append({
            'state_id': sid, 'selected_pair_ids': _sort_pairs(selected), 'edits': int(source_state['edits']),
            'peer_graphs': peers, 'true_graphs': true_graphs,
            'gt_events': gt_vertices, 'pred_events': pred_vertices,
            'bins': bins, 'family_events': family_rows, 'declared': declared,
            'peer_summary': {'meanJ': float(source_state['meanJ']),
                             'meanM': float(source_state['meanM']),
                             'meanCountL1': float(source_state['meanCountL1'])},
            'frozen_utility': prediction_state['utilities'],
            'frozen_rounded_utility': prediction_state['rounded_utilities'],
        })

    expected_choice = {}
    for arm in ARMS:
        record = prediction_clip['selected'][arm]
        expected_choice[arm] = {
            'state_id': record['state_id'],
            'selected_pair_ids': _sort_pairs(pair_map[p] for p in record['selected_pair_ids']),
            'edits': int(record['edits']),
        }
    return {
        'domain': domain, 'clip': clip_id, 'source_clip_order': int(clip_id[1:]),
        'nframes': int(frozen['metadata']['nf']), 'active_pairs': active_pairs,
        'registered210_slate_ids': sorted(pair_map[p] for p in original210_ids),
        'k_events': k_events, 'states': state_rows, 'frozen_choices': expected_choice,
        'active_roster_is_210_slate': True,
        '_raw_run_map': {int(raw): int(anon) for raw, anon in canonical_map.items()},
    }


def make_bundle(root: Path, study210_root: Path) -> dict:
    root = root.resolve(); study210_root = study210_root.resolve()
    seals, manifests = {}, {}
    for stage in ('frozen', 'labels', 'predictions', 'assessment'):
        seals[stage], manifests[stage] = verify_tree(root / stage)
    study210_seal, study210_manifest = verify_tree(study210_root / 'catalog')
    if study210_seal != SEAL210:
        raise ValueError('registered study210 catalog seal differs')

    independent_path = root / 'verification' / 'independent.json'
    independent_sha = verify_sidecar(independent_path)
    independent = read_json(independent_path)
    if independent.get('passed') is not True or independent.get('prereg') != PREREG:
        raise ValueError('passing independent primary211 audit required')
    if independent.get('seals') != seals:
        raise ValueError('independent audit does not bind frozen, labels, predictions and assessment')
    if independent.get('scope', '').find('Original22 only') < 0:
        raise ValueError('independent audit scope differs from primary22')
    for name, expected in independent.get('reports', {}).items():
        path = root / 'verification' / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError('independent report changed: ' + name)
    for name, expected in independent.get('source_sha256', {}).items():
        path = root / 'verification' / 'source_snapshot' / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError('independent executed-source snapshot changed: ' + name)

    frozen_top = read_json(root / 'frozen' / 'deployment.json')
    predictions = read_json(root / 'predictions' / 'predictions.json')
    assessment_index = read_json(root / 'assessment' / 'index.json')
    study210_top = read_json(study210_root / 'catalog' / 'deployment.json')
    if (frozen_top.get('schema') != 'match211-freeze-v1' or frozen_top.get('GT_access') is not False
            or frozen_top.get('prereg') != PREREG or frozen_top.get('marker') != MARKER
            or frozen_top.get('source210') != SEAL210):
        raise ValueError('frozen main211 header differs')
    if predictions.get('schema') != 'match211-constrained-calibration-v1' or predictions.get('arms') != list(ARMS):
        raise ValueError('four-fold predictions schema differs')
    if assessment_index.get('schema') != 'match211-private-assessment-v1' or assessment_index.get('private') is not True:
        raise ValueError('full private original22 assessment required')

    frozen_refs = {(c['domain'], c['clip']): c for c in frozen_top['clips']}
    assessment_refs = {(c['domain'], c['clip']): c for c in assessment_index['clips']}
    prediction_refs = {(c['domain'], c['clip']): c for c in predictions['clips'].values()}
    study210_refs = {(c['domain'], c['clip']): c for c in study210_top['clips']}
    if not (len(frozen_refs) == len(assessment_refs) == len(prediction_refs) == len(study210_refs) == 6):
        raise ValueError('six fixed clip roster required')

    # Clip aliases preserve each original lexical order without disclosing names.
    aliases = {}
    for domain in DOMAINS:
        originals = sorted(c for h, c in frozen_refs if h == domain)
        for index, original in enumerate(originals, 1):
            aliases[domain, original] = 'c' + format(index, '02d')

    clips = []
    source_run_maps = {}
    for domain, original_clip in sorted(frozen_refs, key=lambda x: (DOMAINS.index(x[0]), x[1])):
        frozen_ref = frozen_refs[domain, original_clip]
        prediction_clip = prediction_refs[domain, original_clip]
        assessment_ref = assessment_refs[domain, original_clip]
        study_ref = study210_refs[domain, original_clip]
        folder = root / 'frozen' / frozen_ref['path']
        frozen = read_gzip(folder / 'states.json.gz')
        study_clip = read_gzip(study210_root / 'catalog' / study_ref['path'] / 'catalog.json.gz')
        frozen_pairs = {p['pair_id']: p for p in frozen['active_pairs']}
        original210_ids = list(study_clip['active_slate_ids'])
        if set(frozen_pairs) != set(original210_ids):
            raise ValueError('lineage failure: §211 all-active IDs differ from original §210 slate')
        if len({x for p in frozen_pairs.values() for x in _pair(p['pair_id'])}) != 2 * len(frozen_pairs):
            raise ValueError('registered §210 action set is no longer mutually disjoint')
        all_run_ids = {x for p in frozen_pairs for x in _pair(p)}
        for state in frozen['states']:
            score_path = root / 'assessment' / next(s['path'] for s in assessment_ref['states'] if s['state_id'] == state['state_id']) / 'score.json'
            score = read_json(score_path)
            for event in score['family_events']:
                all_run_ids.update(map(int, event['original_parent_ids']))
                all_run_ids.update(map(int, event['family']))
        canonical_map = {value: i + 1 for i, value in enumerate(sorted(all_run_ids))}
        pair_map = {pid: f'{canonical_map[a]}|{canonical_map[b]}' for pid, (a, b) in
                    ((p, _pair(p)) for p in frozen_pairs)}
        all_event_keys = set()
        gt_ids = set()
        ref_ids = [set(), set()]
        for state in frozen['states']:
            score = read_json(root / 'assessment' / next(s['path'] for s in assessment_ref['states'] if s['state_id'] == state['state_id']) / 'score.json')
            for event in score['family_events']:
                all_event_keys.add((tuple(sorted(map(int, event['original_parent_ids']))), int(event['signed_frame']), int(event['direction'])))
            for threshold in ('0.5', '0.3'):
                gt_ids.update(int(v['gt_event_id']) for v in score['true_graphs'][threshold]['right_vertices'])
            for ref_index, graph in enumerate(state['peer_graphs']):
                ref_ids[ref_index].update(int(v['canonical_id']) for v in graph['reference_vertices'])
        event_map = {key: domain + '_' + aliases[domain, original_clip] + '_e' + format(i + 1, '05d')
                     for i, key in enumerate(sorted(all_event_keys))}
        gt_map = {raw: 'g' + format(i + 1, '05d') for i, raw in enumerate(sorted(gt_ids))}
        ref_maps = [{raw: 'r' + str(index) + '_' + format(i + 1, '05d')
                     for i, raw in enumerate(sorted(values))} for index, values in enumerate(ref_ids)]
        projected = _project_clip(root, domain, original_clip, aliases[domain, original_clip],
                                   frozen_ref, prediction_clip, assessment_ref,
                                   canonical_map, event_map, gt_map, ref_maps, pair_map,
                                   original210_ids)
        source_run_maps[domain, original_clip] = projected.pop('_raw_run_map')
        clips.append(projected)

    if [(c['domain'], len(c['states'])) for c in clips] != [
            ('35', 8), ('33', 8), ('W', 1), ('W', 2), ('G', 2), ('G', 1)]:
        raise ValueError('registered 22 primary state roster differs')

    frozen_models = {}
    source_training = {}
    for domain in DOMAINS:
        model = read_json(root / 'predictions' / 'models' / (domain + '.json'))
        if model.get('heldout') != domain:
            raise ValueError('frozen fold key differs')
        frozen_models[domain] = {
            head: {key: model[head].get(key) for key in ('a', 'b', 'status', 'objective',
                    'gradient', 'KKT', 'candidates', 'ridge_slope', 'intercept_penalized', 'solver')}
            for head in ('Q', 'L', 'X')
        }
        records = []
        for order, row in enumerate(model['training_records']):
            if row['domain'] == domain or (row['domain'], row['clip']) not in aliases:
                raise ValueError('heldout or unregistered source row leaked into fold')
            raw_pid = int(row['pid'])
            if row.get('event_key') != f"{row['domain']}/{row['clip']}/{raw_pid}":
                raise ValueError('frozen training event key differs from source inventory')
            source_clip = aliases[row['domain'], row['clip']]
            clipdata = next(c for c in clips if c['domain'] == row['domain'] and c['clip'] == source_clip)
            run_id = source_run_maps[row['domain'], row['clip']][raw_pid]
            event = next((e for e in clipdata['k_events'] if e['run_id'] == run_id), None)
            if event is None:
                raise ValueError('frozen training event does not join anonymous K graph')
            records.append({'event_id': event['event_id'], 'y': int(row['y']),
                            'run_id': run_id, 'xL': float(row['x_L']), 'xX': float(row['x_X']),
                            'weight': float(row['weight']), 'source_order': order})
        source_training[domain] = records

    return {
        'schema': 'match211.numeric.v1', 'marker': MARKER, 'prereg': PREREG,
        'receipt': {
            'producer_seals': seals, 'study210_catalog_seal': study210_seal,
            'independent_sha256': independent_sha,
            'independent_source_report_sha256': independent['reports'],
            'source_code_sha256': independent['source_sha256'],
            'primary22_only': True, 'all_active_equals_210_slate': True,
            'all_active_pairs_mutually_disjoint': True,
        },
        'models_reference': frozen_models, 'training_reference': source_training,
        'clips': clips, 'scope': 'Anonymous numeric replay: refits event-criticality heads, reproduces choices, official event graphs and windows. It does not rebuild raw frame ownership, raw GT geometry, or images. No raw track/GT IDs, boxes, coordinates, source paths or native MOT rows are included.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--study210-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--go', action='store_true', help='requires independent main22 pass and CPU handoff')
    args = parser.parse_args()
    if not args.go:
        parser.error('projection requires --go after the independent primary22 audit')
    bundle = make_bundle(args.root, args.study210_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=9) as gz:
            data = json.dumps(bundle, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8') + b'\n'
            gz.write(data)
    print('bundle_sha256=' + sha(args.out))
    print('bundle_bytes=' + str(args.out.stat().st_size))


if __name__ == '__main__':
    main()
