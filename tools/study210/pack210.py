"""Seal-check and privacy-project the final interleave210 numeric ledger.

Reads only sealed catalog/features/labels/prediction/native score summaries
and independent audit receipts. It does not load original 208 arrays, GT,
MOT rows, frames, or images. Run only after bound_sources hands off the one-CPU
token and the fixed-k plus all-disjoint audit receipts are complete.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

from replay210 import ALL_DISJOINT_REGISTRATION, ARMS, DOMAINS, SCHEMA


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf8'))


def verify_sidecar(path):
    path = Path(path)
    digest = sha(path)
    sidecar = path.with_suffix('.sha256')
    if sidecar.exists() and sidecar.read_text(encoding='ascii').strip() != digest:
        raise ValueError('verification JSON sidecar mismatch: ' + path.name)
    return digest


def verify_tree(root):
    root = Path(root).resolve()
    manifest_path = root / 'manifest.json'
    digest = sha(manifest_path)
    if digest != (root / 'SEALED.sha256').read_text(encoding='ascii').strip():
        raise ValueError('stage seal mismatch: ' + root.name)
    manifest = read(manifest_path)
    for rel, expected in manifest.items():
        target = (root / rel).resolve()
        if not target.is_relative_to(root) or not target.is_file() or sha(target) != expected:
            raise ValueError('manifest member changed: ' + root.name)
    return digest, manifest


def _key(pair_id):
    if not isinstance(pair_id, str) or not re.fullmatch(r'[0-9]+\|[0-9]+', pair_id):
        raise ValueError('canonical numeric A|B pair key required')
    a, b = map(int, pair_id.split('|'))
    if not a < b:
        raise ValueError('canonical endpoint order must be increasing')
    return a, b


def _bins(values, name):
    if not isinstance(values, list):
        raise ValueError(name + ' must be list')
    out = []
    for row in values:
        if not isinstance(row, list) or len(row) != 2 or any(type(x) is not int for x in row):
            raise ValueError(name + ' must be integer [in,out] rows')
        out.append([int(row[0]), int(row[1])])
    return out


def _same_numbers(left, right, *, rel=1e-12, abs_=1e-14):
    return (isinstance(left, list) and isinstance(right, list) and len(left) == len(right)
            and all(math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
                    for a, b in zip(left, right)))


def _vertices(rows, is_prediction):
    raw_ids = sorted({int(row[0]) for row in rows})
    prefix = 'p' if is_prediction else 'g'
    mapping = {x: prefix + format(i, '05d') for i, x in enumerate(raw_ids)}
    vertices = []
    for row in rows:
        if len(row) < 2 or type(row[1]) is not int:
            raise ValueError('signed event frame missing')
        vertices.append([mapping[int(row[0])], int(row[1])])
    return mapping, vertices


def _score_projection(path, source_selected, pair_remap):
    record = read(path)
    metrics = record['metrics']
    evidence = metrics['evidence']
    gt_map, gt_vertices = _vertices(evidence['gt'], False)
    pred_map, pred_vertices = _vertices(evidence['predictions'], True)
    if set(record['selected_pair_ids']) != set(source_selected):
        raise ValueError('physical score and frozen selection differ')
    selected = [pair_remap[x] for x in source_selected]
    iou = metrics['iou']
    return {
        'selected': selected,
        'bins': {'gt': _bins(metrics['gt_bins'], 'gt_bins'),
                 'pred': _bins(metrics['pred_bins'], 'pred_bins')},
        'events': {
            'gt': gt_vertices, 'pred': pred_vertices,
            'edges05': [[pred_map[int(p)], gt_map[int(g)]] for p, g in evidence['edges05']],
            'edges03': [[pred_map[int(p)], gt_map[int(g)]] for p, g in evidence['edges03']],
        },
        'declared': {
            'nG': int(metrics['nG']), 'nP': int(metrics['nP']),
            'tp05': int(iou['0.5']['tp']), 'f105': float(iou['0.5']['f1']),
            'tp03': int(iou['0.3']['tp']), 'f103': float(iou['0.3']['f1']),
            'numerator': int(record['numerator']), 'denominator': int(record['denominator']),
            'W': float(record['W']),
        },
    }


def make_bundle(root):
    root = Path(root).resolve()
    seals, manifests = {}, {}
    for stage in ('catalog', 'labels', 'predictions', 'score', 'capacity', 'global_capacity'):
        seals[stage], manifests[stage] = verify_tree(root / stage)
    independent_sha = verify_sidecar(root / 'verification/independent.json')
    independent = read(root / 'verification/independent.json')
    candidates_sha = verify_sidecar(root / 'verification/numeric_candidates.json')
    candidates = read(root / 'verification/numeric_candidates.json')
    fixed_capacity = read(root / 'capacity/capacity.json')
    global_capacity = read(root / 'global_capacity/support.json')
    global_receipt = read(root / 'verification/global_support.json')
    verify_sidecar(root / 'verification/global_support.json')
    if not independent.get('passed') or candidates.get('schema') != 'interleave210-independent-numeric-v1':
        raise ValueError('passing independent audit and numeric candidate schema required')
    if candidates.get('seals') != independent.get('seals') or any(
            independent['seals'].get(stage) != seals[stage] for stage in ('catalog', 'labels', 'predictions', 'score')):
        raise ValueError('independent audit does not bind all sealed source stages')
    if global_receipt.get('passed') is not True:
        raise ValueError('global all-disjoint receipt must pass')
    if (global_receipt.get('capacity_seal') != seals['capacity']
            or global_receipt.get('global_capacity_seal') != seals['global_capacity']
            or global_receipt.get('main_independent_sha256') != independent_sha):
        raise ValueError('global support audit receipt does not bind final sealed inputs')
    if fixed_capacity.get('receipt', {}).get('catalog_seal') != seals['catalog'] or fixed_capacity.get('receipt', {}).get('score_seal') != seals['score']:
        raise ValueError('fixed capacity receipt does not bind current catalog and score')
    if global_capacity.get('receipt', {}).get('catalog_seal') != seals['catalog'] or global_capacity.get('receipt', {}).get('score_seal') != seals['score']:
        raise ValueError('all-disjoint receipt does not bind current catalog and score')
    if global_capacity.get('receipt', {}).get('supplement_sha') != ALL_DISJOINT_REGISTRATION:
        raise ValueError('all-disjoint addendum registration receipt differs')
    if global_receipt.get('supplement_sha') != ALL_DISJOINT_REGISTRATION:
        raise ValueError('global receipt has wrong preregistration hash')
    if len(fixed_capacity.get('clips', [])) != 6 or len(global_capacity.get('clips', [])) != 6:
        raise ValueError('both capacity sidecars must cover six clips')
    fixed_by_clip = {(c['domain'], c['clip']): c['capacity'] for c in fixed_capacity['clips']}
    global_by_clip = {(c['domain'], c['clip']): c['support'] for c in global_capacity['clips']}

    deployment = read(root / 'catalog/deployment.json')
    prediction = read(root / 'predictions/predictions.json')
    clip_refs = {(r['domain'], r['clip']): r for r in deployment['clips']}
    if len(clip_refs) != 6 or len(candidates['clips']) != 6:
        raise ValueError('fixed six-clip roster required')
    prediction_clips = prediction['clips']

    frozen_models = {}
    training = {}
    for domain in DOMAINS:
        model = read(root / 'predictions/models' / (domain + '.json'))
        if model.get('heldout') != domain:
            raise ValueError('model heldout roster mismatch')
        frozen_models[domain] = {
            'G': {k: model['G'].get(k) for k in ('status', 'feature_names', 'mean', 'scale', 'beta', 'intercept')},
            'correction': {k: model['correction'].get(k) for k in ('status', 'feature_names', 'mean', 'scale', 'beta', 'intercept')},
        }
        lookup = {}
        correction_order = 0
        for order, row in enumerate(model['training_records']):
            key = row['key']
            if key in lookup:
                raise ValueError('duplicate source training key in frozen model')
            corr = correction_order if row['full_support'] else None
            if row['full_support']:
                correction_order += 1
            lookup[key] = {
                'source_order_G': order,
                'source_weight_G': row.get('weight_G'),
                'source_order_correction': corr,
                'source_weight_correction': row.get('weight_correction'),
                'G6': row['G6'], 'X2': row['X2'], 'Xminus2': row['Xminus2'],
                'full_support': bool(row['full_support']), 'y': int(row['y']),
            }
        training[domain] = lookup

    per_domain_index = {h: 0 for h in DOMAINS}
    clips = []
    for candidate_clip in sorted(candidates['clips'], key=lambda c: c['clip']):
        source_key = candidate_clip['clip']
        if '/' not in source_key:
            raise ValueError('candidate clip needs domain/name key')
        domain, original_clip = source_key.split('/', 1)
        if domain not in DOMAINS or (domain, original_clip) not in clip_refs:
            raise ValueError('candidate clip is outside frozen roster')
        per_domain_index[domain] += 1
        public_clip = 'c' + format(per_domain_index[domain], '02d')
        ref = clip_refs[domain, original_clip]
        folder = root / 'catalog' / ref['path']
        with gzip.open(folder / 'catalog.json.gz', 'rt', encoding='utf8') as stream:
            catalog = json.load(stream)
        cat_pairs = {p['pair_id']: p for p in catalog['pairs']}
        rows = candidate_clip['rows']
        if {r['pair_id'] for r in rows} != set(cat_pairs):
            raise ValueError('independent candidates do not cover the full frozen catalog')
        endpoint_ids = sorted({i for r in rows for i in _key(r['pair_id'])})
        endpoint_remap = {original: i + 1 for i, original in enumerate(endpoint_ids)}
        pair_remap = {}
        projected_candidates = []
        for row in rows:
            a, b = _key(row['pair_id']); pair = cat_pairs[row['pair_id']]
            if list(pair['canonical_ids']) != [a, b]:
                raise ValueError('catalog canonical endpoints differ')
            if not (_same_numbers(row['G6'], pair['G6']) and _same_numbers(row['X2'], pair['X2'])
                    and _same_numbers(row['Xminus2'], pair['Xminus2'])):
                raise ValueError('candidate feature projection differs from frozen catalog')
            if (bool(row['active']) != bool(pair['active'])
                    or bool(row['full_support']) != bool(pair['full_support'])
                    or not math.isclose(float(row['min_offset_norm']), float(pair['min_offset_norm']), rel_tol=1e-12, abs_tol=1e-14)):
                raise ValueError('candidate support/slate feature differs from frozen catalog')
            na, nb = endpoint_remap[a], endpoint_remap[b]
            new_pair = f'{na}|{nb}'; pair_remap[row['pair_id']] = new_pair
            source_key_row = domain + '/' + original_clip + '/' + row['pair_id']
            order = {h: training[h].get(source_key_row) for h in DOMAINS}
            for heldout in DOMAINS:
                expected_training = row['weak'] in ('S', 'D') and domain != heldout
                if expected_training != (order[heldout] is not None):
                    raise ValueError('sealed source row inventory differs from weak labels')
                if expected_training:
                    fitrow = order[heldout]
                    if (not _same_numbers(fitrow['G6'], row['G6'])
                            or not _same_numbers(fitrow['X2'], row['X2'])
                            or not _same_numbers(fitrow['Xminus2'], row['Xminus2'])
                            or fitrow['full_support'] != bool(row['full_support'])
                            or fitrow['y'] != int(row['weak'] == 'S')):
                        raise ValueError('source training row differs from anonymous labels/features')
            source_fields = {
                field: {h: (None if order[h] is None else order[h][field]) for h in DOMAINS}
                for field in ('source_order_G', 'source_weight_G',
                              'source_order_correction', 'source_weight_correction')
            }
            projected_candidates.append({
                'domain': domain, 'pair_id': new_pair, 'canonical_ids': [na, nb],
                'eligible': True, 'active': bool(row['active']),
                'min_offset_norm': float(row['min_offset_norm']),
                'full_support': bool(row['full_support']),
                'G6': row['G6'], 'X2': row['X2'], 'Xminus2': row['Xminus2'],
                'weak': row['weak'], 'co_visible': bool(row['co_visible']),
                'mixed': bool(row.get('mixed', False)),
                'expected_logits': row['logits'],
                **source_fields,
            })
        selected = {arm: [pair_remap[x] for x in candidate_clip['selected'][arm]] for arm in ARMS}
        if set(candidate_clip['selected']) != set(ARMS):
            raise ValueError('frozen model choices must include all six arms')
        prediction_clip = prediction_clips[source_key]
        if any(set(selected[arm]) != {pair_remap[x] for x in prediction_clip['selected'][arm]}
               for arm in ARMS):
            raise ValueError('independent and producer selected-pair sets differ')

        states = {}
        for arm in ARMS:
            rel = f"clips/{domain}/{original_clip}/{arm}/score.json"
            if rel not in manifests['score']:
                raise ValueError('six-state score missing from sealed manifest')
            states[arm] = _score_projection(root / 'score' / rel, candidate_clip['selected'][arm], pair_remap)

        pair_by_id = {r['pair_id']: r for r in rows}
        def action_list(source_ids):
            actions = []
            for pid in source_ids:
                pair = cat_pairs[pid]
                delta = _bins(pair['delta_windows'], 'pair delta')
                if not any(x != [0, 0] for x in delta):
                    raise ValueError('active action has an all-zero delta')
                a, b = pair['canonical_ids']
                actions.append({'pair_id': pair_remap[pid],
                    'canonical_ids': [endpoint_remap[a], endpoint_remap[b]],
                    'delta_windows': delta})
            return actions
        active_slate = candidate_clip['active_slate']
        all_active_ids = [r['pair_id'] for r in rows if r['active']]
        if not set(active_slate) <= set(all_active_ids):
            raise ValueError('active slate contains an inactive edge')
        population = {
            'trigger_candidates': len(rows),
            'active_candidates': len(all_active_ids),
            'slate_pairs': len(active_slate),
        }
        if population['active_candidates'] != population['slate_pairs']:
            raise ValueError('all-disjoint addendum requires every active edge in the active slate')
        if len(all_active_ids) > 3:
            raise ValueError('all-disjoint preregistered 2^a scope requires a<=3')
        for row in rows:
            delta = _bins(cat_pairs[row['pair_id']]['delta_windows'], 'pair delta')
            if bool(row['active']) != any(x != [0, 0] for x in delta):
                raise ValueError('active flag differs from full-graph count delta')
        action_map = {a['pair_id']: a for a in action_list(all_active_ids)}
        all_actions = [action_map[pair_remap[x]] for x in all_active_ids]
        fixed_actions = [action_map[pair_remap[x]] for x in active_slate]
        fixed_cap = fixed_by_clip[domain, original_clip]
        global_cap = global_by_clip[domain, original_clip]
        declared_fixed = {k: fixed_cap[k] for k in
            ('mode', 'a', 'k', 'combinations', 'enumerated', 'denominator',
             'baseline_numerator', 'min_loss_interval', 'max_loss_interval')}
        declared_fixed['actual_numerators'] = {
            arm: int(fixed_cap['actual'][arm]['numerator']) for arm in ('N', 'G', 'X', 'M')}
        vector_receipts = [{
            'vector_sha256': v['vector_sha256'],
            'active_subset_count': int(v['active_subset_count']),
            'numerator': int(v['numerator']),
            'by_active_budget': v['by_active_budget'],
        } for v in global_cap['vectors']]
        declared_global = {
            'global_scope_proved': bool(global_cap['global_scope_proved']),
            'active_support_pair_count': len(global_cap['active_support_pair_ids']),
            'denominator': int(global_cap['denominator']),
            'baseline_numerator': int(global_cap['baseline_numerator']),
            'enumerated_active_subsets': int(global_cap['enumerated_active_subsets']),
            'by_active_budget': [{k: int(v[k]) for k in
                ('active_budget', 'active_subset_count', 'unique_window_vectors', 'min_numerator', 'max_numerator')}
                for v in global_cap['by_active_budget']],
            'vectors': vector_receipts,
        }
        clips.append({
            'domain': domain, 'clip': public_clip, 'nframes': int(catalog['nframes']),
            'candidates': projected_candidates, 'states': states,
            'capacity': {'actions': fixed_actions, 'population': population,
                         'declared': declared_fixed},
            'all_disjoint_registration': {'sha1': ALL_DISJOINT_REGISTRATION},
            'all_disjoint_actions': all_actions,
            'declared_all_disjoint': declared_global,
        })
    if per_domain_index != {'35': 1, '33': 1, 'W': 2, 'G': 2}:
        raise ValueError('frozen six-target roster differs')
    return {
        'schema': SCHEMA,
        'receipt': {
            'producer_seals': seals,
            'independent_sha256': independent_sha,
            'numeric_candidates_sha256': candidates_sha,
            'fixed_capacity_seal_sha256': seals['capacity'],
            'all_disjoint_seal_sha256': seals['global_capacity'],
            'global_support_audit_sha256': verify_sidecar(root / 'verification/global_support.json'),
            'all_disjoint_registration_sha1': ALL_DISJOINT_REGISTRATION,
            'source_only_training_weights': True,
        },
        'models': frozen_models,
        'clips': clips,
        'scope': 'Anonymous numeric replay. No native images, boxes, MOT rows, original GT identities, or absolute source paths. Event matching replays complete anonymous legal graphs; raw-box bracket/offset feature extraction and raw GT ownership assignment remain outside the numeric replay.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('D:/beetrack_runs/interleave210'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--go', action='store_true', help='requires passing independent and bound receipts plus CPU handoff')
    args = parser.parse_args()
    if not args.go:
        parser.error('projection is blocked until --go after independent/capacity receipts and CPU handoff')
    bundle = make_bundle(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, 'wt', encoding='utf8', compresslevel=9) as stream:
        json.dump(bundle, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    print('bundle_sha256=' + sha(args.out))
    print('bundle_bytes=' + str(args.out.stat().st_size))


if __name__ == '__main__':
    main()
