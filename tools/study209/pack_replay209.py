"""Seal-check and privacy-project the completed common209 numeric evidence.

This stage reads existing sealed numeric summaries only. It never loads native
images, raw GT, MOT rows, box coordinates, or the original 208 source archive.
Run only after the independent 36-state audit is sealed and the one-CPU token
is assigned to this projection/replay step.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

from replay209 import ARMS, HEADS, SCHEMA


EXPECTED_INDEPENDENT_SHA = '98e8471ab4e5339f86d4d3aa20d307927687147d45f199d5a075b6b5c056be33'
DOMAINS = ('35', '33', 'W', 'G')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verify_sealed_tree(root):
    root = Path(root).resolve()
    manifest_path = root / 'manifest.json'
    seal_path = root / 'SEALED.sha256'
    manifest_sha = sha(manifest_path)
    if manifest_sha != seal_path.read_text(encoding='ascii').strip():
        raise ValueError('top manifest seal mismatch: ' + root.name)
    manifest = read(manifest_path)
    for rel, digest in manifest.items():
        target = (root / rel).resolve()
        if not target.is_relative_to(root) or not target.is_file() or sha(target) != digest:
            raise ValueError('sealed member mismatch: ' + root.name)
    return manifest_sha, manifest


def verify_sidecar(path, expected=None):
    path = Path(path)
    digest = sha(path)
    sidecar = path.with_suffix(path.suffix + '.sha256')
    if sidecar.is_file() and sidecar.read_text(encoding='ascii').strip() != digest:
        raise ValueError('verification receipt sidecar mismatch: ' + path.name)
    if expected is not None and digest != expected:
        raise ValueError('unexpected independent audit seal: ' + path.name)
    return digest


def parse_pair_id(pair_id):
    match = re.fullmatch(r'([0-9]+)\|([0-9]+)', pair_id)
    if not match:
        raise ValueError('opaque pair key is not a canonical endpoint pair')
    left, right = map(int, match.groups())
    if left >= right:
        raise ValueError('canonical endpoint order is not increasing')
    return left, right


def _ints2(values, name):
    if not isinstance(values, list):
        raise ValueError(name + ' must be a JSON list')
    out = []
    for row in values:
        if not isinstance(row, list) or len(row) != 2 or any(type(x) is not int or x < 0 for x in row):
            raise ValueError(name + ' must contain nonnegative integer pairs')
        out.append([int(row[0]), int(row[1])])
    return out


def _event_map(items, is_prediction):
    # Per-state remapping makes IDs unlinkable across arms and removes all GT IDs.
    ids = sorted({int(row[0]) for row in items})
    prefix = 'p' if is_prediction else 'g'
    mapping = {old: prefix + format(i, '05d') for i, old in enumerate(ids)}
    records = []
    for row in items:
        if len(row) < 2 or type(row[1]) is not int:
            raise ValueError('event evidence needs [id,signed frame,...]')
        records.append([mapping[int(row[0])], int(row[1])])
    return mapping, records


def project_score(score_file, source_selected, pair_map):
    score = read(score_file)
    metrics = score['metrics']
    evidence = metrics['evidence']
    gt_map, gt = _event_map(evidence['gt'], False)
    pred_map, pred = _event_map(evidence['predictions'], True)
    def map_edges(field):
        return [[pred_map[int(p)], gt_map[int(g)]] for p, g in evidence[field]]
    selected = score['selected_pair_ids']
    if set(selected) != set(source_selected) or len(selected) != len(source_selected):
        raise ValueError('physical state and sealed prediction choices differ')
    candidate_selected = [pair_map[p] for p in source_selected]
    iou = metrics['iou']
    return {
        'selected': candidate_selected,
        'bins': {'gt': _ints2(metrics['gt_bins'], 'gt_bins'),
                 'pred': _ints2(metrics['pred_bins'], 'pred_bins')},
        'events': {'gt': gt, 'pred': pred,
                   'edges05': map_edges('edges05'), 'edges03': map_edges('edges03')},
        'declared': {
            'nG': int(metrics['nG']), 'nP': int(metrics['nP']),
            'tp05': int(iou['0.5']['tp']), 'f105': float(iou['0.5']['f1']),
            'tp03': int(iou['0.3']['tp']), 'f103': float(iou['0.3']['f1']),
            'numerator': int(score['numerator']), 'denominator': int(score['denominator']),
            'W': float(score['W']),
        },
    }


def make_bundle(root):
    root = Path(root).resolve()
    pred_seal, pred_manifest = verify_sealed_tree(root / 'predictions')
    score_seal, score_manifest = verify_sealed_tree(root / 'score')
    independent_sha = verify_sidecar(root / 'verification/independent.json', EXPECTED_INDEPENDENT_SHA)
    independent = read(root / 'verification/independent.json')
    audit = read(root / 'verification/physical.json')
    candidates = read(root / 'verification/numeric_candidates.json')
    verify_sidecar(root / 'verification/numeric_candidates.json')
    if not independent.get('passed') or not audit.get('passed'):
        raise ValueError('independent candidate/model/native-state audit must pass')
    if audit.get('seals') != independent.get('seals') or candidates.get('seals') != independent.get('seals'):
        raise ValueError('independent evidence does not bind to common stage seals')
    seals = independent['seals']
    if seals.get('predictions') != pred_seal or seals.get('score') != score_seal:
        raise ValueError('independent audit and sealed producer roots differ')
    if candidates.get('schema') != 'common209-independent-numeric-v1':
        raise ValueError('unexpected independent numeric candidate schema')

    model_bundle = {}
    source_training = {}
    for domain in DOMAINS:
        model = read(root / 'predictions/models' / (domain + '.json'))
        if model.get('heldout') != domain:
            raise ValueError('wrong heldout model file')
        model_bundle[domain] = {}
        for head in HEADS:
            h = model['heads'][head]
            model_bundle[domain][head] = {k: h.get(k) for k in
                ('status', 'feature_names', 'mean', 'scale', 'beta', 'intercept')}
        source_training[domain] = {}
        for order, record in enumerate(model.get('training_records', [])):
            key = record['key']
            if key in source_training[domain]:
                raise ValueError('duplicate sealed source training row')
            source_training[domain][key] = {'order': order, 'weight': float(record['weight'])}

    candidate_clips = candidates['clips']
    if len(candidate_clips) != 6:
        raise ValueError('expected exactly six sealed target clips')
    per_domain_index = {h: 0 for h in DOMAINS}
    bundle_clips = []
    for clip_record in sorted(candidate_clips, key=lambda c: c['clip']):
        source_clip = clip_record['clip']
        if '/' not in source_clip:
            raise ValueError('candidate clip key must start with a registered domain')
        domain, original_name = source_clip.split('/', 1)
        if domain not in DOMAINS or not original_name:
            raise ValueError('unregistered target clip')
        per_domain_index[domain] += 1
        public_clip = 'c' + format(per_domain_index[domain], '02d')
        rows = clip_record['rows']
        endpoint_pairs = {r['pair_id']: parse_pair_id(r['pair_id']) for r in rows}
        endpoint_ids = sorted({v for pair in endpoint_pairs.values() for v in pair})
        endpoint_map = {old: i + 1 for i, old in enumerate(endpoint_ids)}
        pair_map = {}
        projected_rows = []
        for row in rows:
            left, right = endpoint_pairs[row['pair_id']]
            new_pair = f"{endpoint_map[left]}|{endpoint_map[right]}"
            pair_map[row['pair_id']] = new_pair
            source_order, source_weight = {}, {}
            source_key = domain + '/' + original_name + '/' + row['pair_id']
            for heldout in DOMAINS:
                item = source_training[heldout].get(source_key)
                source_order[heldout] = None if item is None else item['order']
                source_weight[heldout] = None if item is None else item['weight']
            projected_rows.append({
                'pair_id': new_pair,
                'canonical_ids': [endpoint_map[left], endpoint_map[right]],
                'geometry4': row['geometry4'],
                'true_valid': bool(row['true_valid']),
                'true_features': row['true4'],
                'sham_valid': bool(row['sham_valid']),
                'sham_features': row['sham4'],
                'weak': row['weak'],
                'anyD': bool(row['anyD']),
                'expected_logits': row['logits'],
                'source_order': source_order,
                'source_weight': source_weight,
            })
        selected_source = clip_record['selected']
        if set(clip_record['selected']) != set(ARMS):
            raise ValueError('candidate record must contain all six arm selections')
        arms = {}
        for arm in ARMS:
            rel = f"clips/{domain}/{original_name}/{arm}/score.json"
            if rel not in score_manifest:
                raise ValueError('state score is absent from the sealed score manifest')
            arms[arm] = project_score(root / 'score' / rel, selected_source[arm], pair_map)
        bundle_clips.append({
            'domain': domain, 'clip': public_clip, 'candidates': projected_rows,
            'arms': arms,
        })
    if per_domain_index != {'35': 1, '33': 1, 'W': 2, 'G': 2}:
        raise ValueError('frozen six-clip roster changed')
    return {
        'schema': SCHEMA,
        'receipt': {
            'prereg_sha1': read(root / 'predictions/receipt.json').get('prereg'),
            'independent_sha256': independent_sha,
            'prediction_seal_sha256': pred_seal,
            'score_seal_sha256': score_seal,
            'verified_stage_seals': seals,
            'source_models_are_frozen_weights_not_public_refits': True,
        },
        'models': model_bundle,
        'clips': bundle_clips,
        'scope': 'Anonymous numeric projection. Models use sealed weights; LK traces and GT rectangle pairing are not reconstructed here. Event matching is recomputed from supplied anonymous legal edges.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('D:/beetrack_runs/common209'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--go', action='store_true', help='required explicit gate after independent PASS and CPU handoff')
    args = parser.parse_args()
    if not args.go:
        parser.error('projection is blocked until --go after independent PASS and CPU handoff')
    bundle = make_bundle(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, 'wt', encoding='utf8', compresslevel=9) as stream:
        json.dump(bundle, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    print('bundle_sha256=' + sha(args.out))
    print('bundle_bytes=' + str(args.out.stat().st_size))


if __name__ == '__main__':
    main()
