"""Project sealed joint211 numeric outputs; no images, matrices or GT boxes.

Explicit --go is required. Output event IDs are stable within each anonymous
clip across every state. They are event linkage, not physical bee identities.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

DOMAINS = ('35', '33', 'W', 'G')
MARKER = '[cross-family-peer-v1]'
PREREG = '3323107ea3560e9ee507e65891097cf2ca605b00'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def tree(root):
    root = Path(root).resolve()
    digest = sha(root / 'manifest.json')
    if (root / 'SEALED.sha256').read_text().strip() != digest:
        raise ValueError('sealed manifest mismatch')
    manifest = read(root / 'manifest.json')
    for name, expected in manifest.items():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or sha(file) != expected:
            raise ValueError('sealed member mismatch')
    return digest


def vector(rows):
    if not isinstance(rows, list) or any(not isinstance(r, list) or len(r) != 2 or
            any(type(v) is not int for v in r) for r in rows):
        raise ValueError('integer directional window vector required')
    return rows


def vertices(rows, key):
    return {int(v[key]): int(v['signed_frame']) for v in rows}


def project_clip(prediction, assessment, alias, nframes):
    actions = prediction['actions']
    endpoints = sorted({int(v) for a in actions.values() for v in a['canonical_ids']})
    endpoint_map = {value: i + 1 for i, value in enumerate(endpoints)}
    action_order = sorted(actions, key=lambda k: (actions[k]['family'], tuple(actions[k]['canonical_ids'])))
    amap = {value: 'a%02d' % (i + 1) for i, value in enumerate(action_order)}
    state_order = sorted(prediction['states'], key=lambda s: (len(s['selected_action_ids']),
        tuple(sorted(action_order.index(a) for a in s['selected_action_ids']))))
    smap = {s['state_id']: 's%03d' % i for i, s in enumerate(state_order)}
    if set(smap) != set(assessment['states']):
        raise ValueError('prediction/assessment state inventories differ')
    pids, gids, refs = set(), set(), [set(), set()]
    for state in state_order:
        sid = state['state_id']
        metrics = assessment['states'][sid]['metrics']
        pids.update(int(r[0]) for r in metrics['evidence']['predictions'])
        gids.update(int(r[0]) for r in metrics['evidence']['gt'])
        for i, graph in enumerate(prediction['peer_graphs'][sid]):
            refs[i].update(int(v['canonical_id']) for v in graph['reference_vertices'])
    pmap = {v: 'p%05d' % (i + 1) for i, v in enumerate(sorted(pids))}
    gmap = {v: 'g%05d' % (i + 1) for i, v in enumerate(sorted(gids))}
    rmaps = [{v: 'r%d_%05d' % (peer, i + 1) for i, v in enumerate(sorted(values))}
             for peer, values in enumerate(refs)]
    output_actions = []
    for aid in action_order:
        a = actions[aid]
        projected = dict(action_id=amap[aid], family=a['family'],
            endpoint_ids=[endpoint_map[int(v)] for v in a['canonical_ids']],
            delta_windows=vector(a['delta_windows']))
        field = 'mean_shared_iou' if a['family'] == 'C' else 'min_local_norm'
        projected[field] = float(a[field])
        output_actions.append(projected)
    states = []
    for state in state_order:
        sid = state['state_id']; scored = assessment['states'][sid]
        m = scored['metrics']; evidence = m['evidence']
        pred = {int(r[0]): int(r[1]) for r in evidence['predictions']}
        gt = {int(r[0]): int(r[1]) for r in evidence['gt']}
        if len(pred) != len(evidence['predictions']) or len(gt) != len(evidence['gt']):
            raise ValueError('event vertex IDs are not unique within state')
        truth = {}
        for t, edge_key in (('0.5', 'edges05'), ('0.3', 'edges03')):
            g = scored['true_graphs'][t]
            if vertices(g['left_vertices'], 'pid') != pred or vertices(g['right_vertices'], 'gt_event_id') != gt:
                raise ValueError('true graph event inventory differs')
            if {tuple(e) for e in g['edges']} != {tuple(e) for e in evidence[edge_key]}:
                raise ValueError('true graph/evidence edges differ')
            truth[t] = dict(edges=sorted([pmap[int(p)], gmap[int(r)]] for p, r in g['edges']),
                            declared_tp=int(g['R']))
        peers = []
        for i, g in enumerate(prediction['peer_graphs'][sid]):
            if vertices(g['prediction_vertices'], 'canonical_id') != pred:
                raise ValueError('peer/true left vertex inventories differ')
            peers.append(dict(reference='peer%d' % i,
                pred=sorted([pmap[p], frame] for p, frame in pred.items()),
                reference_events=sorted([rmaps[i][int(v['canonical_id'])], int(v['signed_frame'])]
                                        for v in g['reference_vertices']),
                edges=sorted([pmap[int(p)], rmaps[i][int(r)]] for p, r in g['edges']),
                pred_bins=vector(g['pred_windows']), reference_bins=vector(g['reference_windows']),
                declared={k: int(g[v]) for k, v in [('J', 'J'), ('R', 'R'), ('nP', 'nPred'), ('nR', 'nReference')]}))
        gb, pb = vector(m['gt_bins']), vector(m['pred_bins'])
        if pb != vector(state['windows']):
            raise ValueError('native/predicted windows differ')
        numerator = sum(abs(p - g) for pr, gr in zip(pb, gb) for p, g in zip(pr, gr))
        denominator = 30 * int(m['nG'])
        states.append(dict(state_id=smap[sid], selected_action_ids=sorted(amap[a] for a in state['selected_action_ids']),
            edits=len(state['selected_action_ids']), pred_events=sorted([pmap[p], f] for p, f in pred.items()),
            gt_events=sorted([gmap[g], f] for g, f in gt.items()), true_graphs=truth, peer_graphs=peers,
            bins=dict(gt=gb, pred=pb), peer_summary=prediction['state_summaries'][sid],
            frozen_utility=prediction['utilities'][sid],
            declared=dict(nG=int(m['nG']), nP=int(m['nP']), numerator=numerator, denominator=denominator,
                W=float(m['W']), tp05=int(m['iou']['0.5']['tp']), tp03=int(m['iou']['0.3']['tp']),
                f105=float(m['iou']['0.5']['f1']), f103=float(m['iou']['0.3']['f1']))))
    def choices(table):
        raw_states = {s['state_id']: s for s in state_order}
        return {arm: dict(state_id=smap[sid],
                selected_action_ids=sorted(amap[a] for a in raw_states[sid]['selected_action_ids']))
                for arm, sid in table.items()}
    return dict(domain=prediction['domain'], clip=alias,
        nframes=int(nframes), actions=output_actions, states=states,
        model=prediction['model'], frozen_choices=choices(prediction['choices']),
        family_only={family: dict(choices=choices(table['choices']),
            legal_state_ids=[smap[sid] for sid in table['legal_state_ids']])
            for family, table in prediction['family_only'].items()})


def build(root, out, primary):
    root = Path(root)
    assessment_seal = tree(root / 'joint_score')
    assessment = read(root / 'joint_score/assessment.json')
    seal_path = root / 'joint_predictions/ALL_FOUR_SEAL.json'
    seal = read(seal_path)
    if sha(seal_path) != (root / 'joint_predictions/ALL_FOUR_SEAL.sha256').read_text().strip():
        raise ValueError('four-fold seal sidecar mismatch')
    for name, expected in seal['files'].items():
        if Path(name).name != name or sha(root / 'joint_predictions' / name) != expected:
            raise ValueError('joint prediction seal mismatch')
    by_clip = {(x['domain'], x['clip']): x for x in assessment['clips']}
    with gzip.open(primary, 'rt', encoding='utf8') as stream:
        primary_bundle = json.load(stream)
    frames = {(c['domain'], c['clip']): c['nframes'] for c in primary_bundle['clips']}
    clips = []
    for domain in DOMAINS:
        doc = read(root / 'joint_predictions' / (domain + '.joint_predictions.json'))
        for index, prediction in enumerate(doc['clips']):
            alias = 'c%02d' % (index + 1)
            clips.append(project_clip(prediction, by_clip[domain, prediction['clip']], alias, frames[domain, alias]))
    if len(clips) != 6 or sum(len(c['states']) for c in clips) != 86:
        raise ValueError('joint six-clip/86-state scope differs')
    bundle = dict(schema='joint211.numeric.v1', marker=MARKER, prereg=PREREG,
        receipts=dict(assessment_manifest_sha256=assessment_seal,
            primary_bundle_sha256=sha(primary),
            prediction_seal_sha256=sha(seal_path),
            independent_manifest_sha256=tree(root / 'joint_verification'),
            independent_sha256=sha(root / 'joint_verification/independent.json')),
        clips=clips, scope='Fixed sealed graphs/windows and inherited models; no raw geometry or owner reassignment.')
    out = Path(out)
    if out.exists():
        raise FileExistsError('new output required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
        zipped.write((json.dumps(bundle, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
    print(json.dumps(dict(clips=len(clips), states=86, sha256=sha(out))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--go', action='store_true', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.out, args.primary)
