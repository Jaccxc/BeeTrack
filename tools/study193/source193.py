"""Explicit-source §193 label boundary and pure native-MOT score proofs.

The fold loader may open prepared labels only for the three allowlisted source
domains, after every GT-free prediction artifact is sealed. Pure helpers below
do not read files; the target evaluator can use them after its own freeze gate.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

PREREG = 'f161c64a6310c663d6cb079e83abee335d90e324'
SOURCES = {
    '33': (('dev', 'BEE24-33-head'),),
    '35': (('dev', 'BEE24-35-head'),),
    'W': (('val', 'BEE24-10'), ('val', 'BEE24-15')),
    'G': (('val', 'BEE24-24'), ('val', 'BEE24-28')),
}
EXPECTED_DOMAIN_GT = {'33': 184, '35': 129, 'W': 69, 'G': 17}


def source_allowlist(heldout):
    if heldout not in SOURCES:
        raise ValueError('heldout must be one of 33,35,W,G')
    return {domain: clips for domain, clips in SOURCES.items() if domain != heldout}


def event_descriptor(rows, polygon, nframes):
    from beetrack.evaluate import by_track, zone_tracks
    from tools.study103.swap_score import counts_from_events
    rows = by_track(rows)
    signed = []
    for tid, info in zone_tracks(rows, polygon).items():
        direction = info[0]
        if direction:
            index = next(j for j, d in reversed(info[3]) if d == direction)
            frame = int(rows[index, 0])
            assert 1 <= frame <= nframes
            signed.append([int(tid), frame if direction == 'in' else -frame])
    events = np.array([[abs(frame), 1 if frame > 0 else -1] for _, frame in signed], dtype=int).reshape(-1, 2)
    bins = counts_from_events(events, nframes).astype(np.int64)
    return dict(signed=signed, events=events, bins=bins,
                nG=len(signed), direction_counts={1: int(np.sum(events[:, 1] > 0)), -1: int(np.sum(events[:, 1] < 0))})


def score_native(gt, rows, polygon, nframes):
    """Native graph at both IoUs, checked independently against official TP."""
    from beetrack.evaluate import by_track, frame_match, match_events, zone_tracks
    from beetrack.p0 import events
    from tools.study117.tracking.score117 import FORMAT, array_hash, replay_case
    from tools.study103.swap_score import event_scores
    gt, rows = np.asarray(gt, float), np.asarray(rows, float)
    # Native prepared GT contains negative extents in BEE24-24. Official
    # matching keeps them verbatim; score117's extra extent validator cannot.
    assert all(a.ndim == 2 and a.shape[1] >= 6 and np.isfinite(a).all() for a in (gt, rows))
    assert all(np.array_equal(a[:, :2], np.rint(a[:, :2])) and np.all((a[:, 0] >= 1) & (a[:, 0] <= nframes)) for a in (gt, rows))
    G, P = by_track(gt), by_track(rows)
    gz, pz = zone_tracks(G, polygon), zone_tracks(P, polygon)
    truth_events = {i: v for i, v in gz.items() if v[0]}
    prediction_events = {i: v for i, v in pz.items() if v[0]}
    evidence = dict(format=FORMAT, nframes=int(nframes),
                    inputs_sha256=dict(gt=array_hash(gt), mot=array_hash(rows), polygon=array_hash(np.asarray(polygon, float))))
    archived = []
    for identity, info in prediction_events.items():
        j = next(j for j, d in reversed(info[3]) if d == info[0])
        x, y, w, h = P[j, 2:6]
        archived.append([int(identity), int(P[j, 0]) * (1 if info[0] == 'in' else -1), float(x), float(y), float(x + w), float(y + h)])
    evidence['predictions'] = archived
    evidence['gt'] = event_descriptor(gt, polygon, nframes)['signed']
    for threshold, key in ((.5, 'edges05'), (.3, 'edges03')):
        owners = frame_match(G, P, thr=threshold)
        edges = set()
        for identity, info in prediction_events.items():
            direction = info[0]
            for row, crossing in info[3]:
                if crossing != direction:
                    continue
                for owner in (owners[row - 1], owners[row]):
                    target = truth_events.get(owner)
                    if target and target[0] == direction and any(
                        d == direction and abs(G[j, 0] - P[row, 0]) <= 5 for j, d in target[3]
                    ):
                        edges.add((int(identity), int(owner)))
        evidence[key] = [list(edge) for edge in sorted(edges)]
    value = replay_case(evidence)
    truth = event_descriptor(gt, polygon, nframes)
    prediction = event_descriptor(rows, polygon, nframes)
    assert np.array_equal(events(gt, polygon).astype(int), truth['events'])
    assert np.array_equal(events(rows, polygon).astype(int), prediction['events'])
    assert value['evidence']['gt'] == truth['signed']
    assert [[row[0], row[1]] for row in value['evidence']['predictions']] == prediction['signed']
    official = match_events(gt, rows, polygon)
    assert (official['gt'], official['pred'], official['tp']) == (truth['nG'], prediction['nG'], value['iou']['0.5']['tp'])
    independently = event_scores(gt, rows, polygon)
    for threshold in (.5, .3):
        assert all(value['iou'][str(threshold)][key] == expected for key, expected in independently[threshold].items())
    assert value['window']['numerator'] == int(np.abs(truth['bins'] - prediction['bins']).sum())
    assert value['window']['denominator'] == int(truth['bins'].sum()) == 30 * truth['nG']
    value.update(integer_bins=dict(gt=truth['bins'].tolist(), pred=prediction['bins'].tolist()),
                 nG=truth['nG'], nP=prediction['nG'], official_tp_checked=True,
                 native_gt_contract=dict(rows=len(gt), negative_extent_rows=int(np.any(gt[:, 4:6] < 0, axis=1).sum()),
                                         all_rows_retained=True, precision='native float64; original row order and extents preserved'))
    return value


def training_clip(gt, member_rows, polygon, nframes):
    """Return fit193's exact source-only budget and whole-member rank labels.

    Direction keys are +1(in) and -1(out). Rank prediction IDs are official
    gap-resolved native IDs; GT identities remain member-local. H.intervals(T)
    uses zero-based seconds and T=ceil(nframes/25), including the partial tail.
    """
    from harm_budget import intervals
    truth = event_descriptor(gt, polygon, nframes)
    seconds = (nframes + 24) // 25
    windows = intervals(seconds)
    assert len(windows) == len(truth['bins'])
    second_counts = {d: np.zeros(seconds, np.int64) for d in (1, -1)}
    for frame, direction in truth['events']:
        second_counts[int(direction)][(int(frame) - 1) // 25] += 1
    gt_window_counts = {}
    for direction, column in ((1, 0), (-1, 1)):
        via_seconds = np.array([second_counts[direction][lo:hi].sum() for lo, hi in windows], np.int64)
        assert np.array_equal(via_seconds, truth['bins'][:, column])
        gt_window_counts[direction] = via_seconds
    rank_edges, gt_events, members = {}, {}, {}
    for member, rows in member_rows.items():
        value = score_native(gt, rows, polygon, nframes)
        gt_events[member] = [int(row[0]) for row in value['evidence']['gt']]
        for row in value['evidence']['predictions']:
            rank_edges[member, int(row[0])] = []
        for pid, gid in value['evidence']['edges05']:
            rank_edges[member, int(pid)].append(int(gid))
        members[member] = value
    return dict(nframes=int(nframes), seconds=seconds, windows=windows,
                gt_window_counts=gt_window_counts, gt_events=gt_events, rank_edges=rank_edges,
                gt_signed_events=truth['signed'], direction_counts=truth['direction_counts'],
                members=members)


def load_fold_sources(repo, frozen_root, heldout, prereg, *, source_domains):
    """Open exactly the explicit three source domains; never heldout labels.

    Return ``groups`` in freeze193 order, each original GT-free group plus a
    ``labels`` training_clip result. ``clips`` contains source-only GT proofs.
    The caller may train any number of models from this returned source data.
    """
    allowed = source_allowlist(heldout)
    source_domains = tuple(source_domains)
    if set(source_domains) != set(allowed) or len(source_domains) != 3:
        raise ValueError('Explicit source_domains must equal the three non-heldout domains')
    if PREREG is None or prereg != PREREG:
        raise RuntimeError('Source labels blocked pending published registration')
    from freeze193 import load_groups, sha, MODEL_SHA
    from beetrack.evaluate import ZONES
    repo, frozen_root = Path(repo), Path(frozen_root)
    # This routine verifies every GT-free artifact before any label file opens.
    frozen_groups = load_groups(frozen_root)
    freeze_hash = sha(frozen_root / 'manifest.json')
    deployment = json.loads((frozen_root / 'deployment.json').read_text(encoding='utf-8'))
    assert deployment['prereg'] == prereg
    all_clips = {(domain, split, clip) for domain, clips in SOURCES.items() for split, clip in clips}
    zones = json.loads(ZONES.read_text(encoding='utf-8'))
    for group in frozen_groups:
        assert (group['domain'], group['split'], group['video']) in all_clips
        assert np.array_equal(group['polygon'], zones[group['video']])
        assert len(group['members']) == 3 and len({m['member'] for m in group['members']}) == 3
        expected_linker = 'm_no' + group['domain'] if group['domain'] in ('W', 'G') else 'm_dev_strict'
        assert group['linker_model'] == expected_linker
        assert group['linker_sha256'] == MODEL_SHA[expected_linker]
        assert group['domain'] in group['linker_excluded_domains']
        for member in group['members']:
            assert member['linker_name'] == expected_linker
            assert member['linker_receipt']['sha256'] == MODEL_SHA[expected_linker]
            if group['domain'] in ('W', 'G'):
                assert member['raw_receipt']['source']['kind'] == 'native_clean_rebuild'
    source_groups = [g for g in frozen_groups if g['domain'] in allowed]
    assert all(g['domain'] != heldout for g in source_groups)
    gt_arrays, clips = {}, {}
    for domain, domain_clips in allowed.items():
        domain_total = 0
        for split, clip in domain_clips:
            group = next(g for g in source_groups if g['video'] == clip)
            gtroot = repo / 'runs/gt' / split / clip
            ini = configparser.ConfigParser()
            ini.read(gtroot / 'seqinfo.ini', encoding='utf-8')
            assert ini.getint('Sequence', 'frameRate') == 25
            assert ini.getint('Sequence', 'seqLength') == group['nf']
            path = gtroot / 'gt/gt.txt'
            gt = (np.loadtxt(path, delimiter=',', usecols=range(6), ndmin=2)
                  if path.read_bytes().strip() else np.empty((0, 6)))
            descriptor = event_descriptor(gt, zones[clip], group['nf'])
            gt_arrays[clip] = gt
            clips[clip] = dict(domain=domain, split=split, nframes=group['nf'], gt_sha256=sha(path),
                               seqinfo_sha256=sha(gtroot / 'seqinfo.ini'),
                               gt_signed_events=descriptor['signed'], gt_bins=descriptor['bins'].tolist(),
                               nG=descriptor['nG'], direction_counts=descriptor['direction_counts'])
            domain_total += descriptor['nG']
        assert domain_total == EXPECTED_DOMAIN_GT[domain], (domain, domain_total)
    labelled, video_labels = [], {}
    for group in source_groups:
        clip = group['video']
        members = {member['member']: member['rows'] for member in group['members']}
        labels = training_clip(gt_arrays[clip], members, group['polygon'], group['nf'])
        labels['domain'] = group['domain']
        for member in group['members']:
            proof = labels['members'][member['member']]
            assert proof['integer_bins']['pred'] == member['windows']
            signed = {int(e['pid']): int(e['frame']) * int(e['direction']) for e in member['event_records']}
            assert signed == {r[0]: r[1] for r in proof['evidence']['predictions']}
            proof['integrity'] = dict(gt_sha256=clips[clip]['gt_sha256'], frozen_unit_index=member['unit_index'],
                                      frozen_member=member['member'], domain=group['domain'], clip=clip,
                                      linker_receipt=member['linker_receipt'], raw_receipt=member['raw_receipt'])
        if clip not in video_labels:
            video_labels[clip] = {k: labels[k] for k in ('domain', 'nframes', 'seconds', 'windows', 'gt_window_counts', 'gt_signed_events', 'direction_counts')}
            video_labels[clip].update(rank_edges={}, gt_events={}, members={})
        combined = video_labels[clip]
        assert all(np.array_equal(combined['gt_window_counts'][d], labels['gt_window_counts'][d]) for d in (1, -1))
        assert not set(combined['rank_edges']) & set(labels['rank_edges'])
        for field in ('rank_edges', 'gt_events', 'members'):
            combined[field].update(labels[field])
        labelled.append(dict(**group, labels=labels))
    for clip, combined in video_labels.items():
        proof = dict(gt=clips[clip], members=combined['members'])
        labels_hash = hashlib.sha256((json.dumps(proof, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()).hexdigest()
        combined['source_receipt'] = dict(domain=combined['domain'], clip=clip,
                                          gt_sha256=clips[clip]['gt_sha256'],
                                          seqinfo_sha256=clips[clip]['seqinfo_sha256'],
                                          freeze_sha256=freeze_hash, label_proof_sha256=labels_hash)
        domain_group = next(g for g in source_groups if g['video'] == clip)
        combined['source_receipt'].update(linker_model=domain_group['linker_model'],
                                          linker_sha256=domain_group['linker_sha256'],
                                          linker_excluded_domains=domain_group['linker_excluded_domains'])
    assert sha(frozen_root / 'manifest.json') == freeze_hash
    return dict(schema='budget193-fold-source-labels-v1', prereg=prereg, heldout=heldout,
                source_domains=list(allowed), groups=labelled, clips=clips, labels=video_labels,
                freeze_sha256=freeze_hash)


def make_label_provider(repo, frozen_root, prereg, *, evidence_dir=None):
    """Fit193 callback: only source labels for requested heldout domain."""
    def provider(heldout):
        fold = load_fold_sources(repo, frozen_root, heldout, prereg,
                                 source_domains=tuple(source_allowlist(heldout)))
        if evidence_dir is not None:
            root = Path(evidence_dir)
            if root.resolve().drive.upper() != 'D:':
                raise ValueError('Source label evidence must be on D:')
            root.mkdir(parents=True, exist_ok=True)
            path = root / f'fold_{heldout}.json'
            with path.open('x', encoding='utf-8') as handle:
                json.dump(fold_evidence(fold), handle, sort_keys=True, separators=(',', ':'), allow_nan=False)
                handle.write('\n')
        return fold['labels']
    return provider


def fold_evidence(fold):
    """Path-free serializable proof, excluding array-valued deployment features."""
    return dict(schema=fold['schema'], prereg=fold['prereg'], heldout=fold['heldout'],
                source_domains=fold['source_domains'], freeze_sha256=fold['freeze_sha256'], clips=fold['clips'],
                source_receipts={clip: labels['source_receipt'] for clip, labels in fold['labels'].items()},
                groups=[dict(unit=g['unit'], domain=g['domain'], clip=g['video'], group=g['group'],
                             members=g['labels']['members']) for g in fold['groups']])


def loader_self_test():
    """Heldout synthetic label files do not exist: any accidental read fails."""
    import freeze193
    import beetrack.evaluate as evaluation
    global PREREG, EXPECTED_DOMAIN_GT
    old = PREREG, EXPECTED_DOMAIN_GT, freeze193.load_groups, evaluation.ZONES
    polygon = [[0, 0], [10, 0], [10, 30], [0, 30]]
    gt = np.array([[1, 1, 14, 5, 2, 7], [2, 1, 6, 5, 2, 7]], float)
    try:
        PREREG = 'synthetic-source-only'
        EXPECTED_DOMAIN_GT = {'33': 1, '35': 1, 'W': 2, 'G': 2}
        groups = []
        for domain, clips in SOURCES.items():
            for split, clip in clips:
                members = []
                linker_name = 'm_no' + domain if domain in ('W', 'G') else 'm_dev_strict'
                linker = dict(name=linker_name, sha256=freeze193.MODEL_SHA[linker_name], excluded_domains=[domain])
                for i in range(3):
                    rows = np.c_[gt.copy(), np.ones(2)]; rows[:, 1] = i + 10
                    windows = event_descriptor(rows, polygon, 4)['bins'].tolist()
                    members.append(dict(member=f'{domain}_{i}', rows=rows, unit_index=len(groups) * 3 + i,
                                        linker_name=linker_name, linker_receipt=linker,
                                        raw_receipt=dict(source=dict(kind='native_clean_rebuild' if domain in ('W', 'G') else 'section189_head_cache')),
                                        windows=windows, event_records=[dict(pid=i + 10, frame=2, direction=1)]))
                groups.append(dict(unit=clip, domain=domain, video=clip, split=split, nf=4, polygon=polygon, group=1, members=members,
                                   linker_model=linker_name, linker_sha256=linker['sha256'], linker_excluded_domains=[domain]))
        freeze193.load_groups = lambda _: groups
        for heldout in SOURCES:
            with tempfile.TemporaryDirectory(prefix='source193-synthetic-') as name:
                root = Path(name); frozen = root / 'frozen'; frozen.mkdir()
                (frozen / 'manifest.json').write_text('{}')
                (frozen / 'deployment.json').write_text(json.dumps(dict(prereg=PREREG)))
                evaluation.ZONES = root / 'zones.json'
                evaluation.ZONES.write_text(json.dumps({g['video']: polygon for g in groups}))
                for domain, clips in source_allowlist(heldout).items():
                    for split, clip in clips:
                        target = root / 'runs/gt' / split / clip
                        (target / 'gt').mkdir(parents=True)
                        (target / 'seqinfo.ini').write_text('[Sequence]\nframeRate=25\nseqLength=4\n')
                        np.savetxt(target / 'gt/gt.txt', gt, delimiter=',')
                fold = load_fold_sources(root, frozen, heldout, PREREG, source_domains=tuple(source_allowlist(heldout)))
                assert heldout not in {label['domain'] for label in fold['labels'].values()}
                assert all(label['source_receipt']['label_proof_sha256'] for label in fold['labels'].values())
                json.dumps(fold_evidence(fold), allow_nan=False)
    finally:
        PREREG, EXPECTED_DOMAIN_GT, freeze193.load_groups, evaluation.ZONES = old


def self_test():
    from tools.study117.tracking.score117 import score_case
    poly = [[0, 0], [10, 0], [10, 30], [0, 30]]
    for nf in (24, 25, 26, 749, 750, 751, 790, 3950):
        gt = np.array([[max(1, nf - 1), 1, 14, 5, 2, 7], [nf, 1, 6, 5, 2, 7]], float)
        if nf == 1:
            continue
        rows = gt.copy(); rows[:, 1] = 9
        result = training_clip(gt, {'member0': rows}, poly, nf)
        assert result['members']['member0']['iou']['0.5']['tp'] == 1
        assert result['rank_edges']['member0', 9] == [1]
        assert result['gt_events']['member0'] == [1]
        assert result['gt_window_counts'][1].sum() == 30
        assert result['gt_window_counts'][-1].sum() == 0
        assert result['members']['member0']['iou'] == score_case(gt, rows, poly, nf)['iou']
    # Negative GT extents stay in the official count and matching competition.
    gt = np.array([[1, 1, 14, 5, 2, 7], [2, 1, 6, 5, 2, 7],
                   [1, 2, 8, 5, -3, 2], [2, 2, 14, 5, -3, 2]], float)
    original = gt.copy()
    value = score_native(gt, gt[:2], poly, 4)
    assert np.array_equal(gt, original)
    assert value['nG'] == 2 and value['nP'] == value['iou']['0.5']['tp'] == 1
    assert value['native_gt_contract']['negative_extent_rows'] == 2
    assert value['native_gt_contract']['all_rows_retained']
    for heldout in SOURCES:
        allowed = source_allowlist(heldout)
        assert heldout not in allowed and len(allowed) == 3
    try:
        source_allowlist('test')
    except ValueError:
        pass
    else:
        raise AssertionError('Unregistered heldout accepted')
    try:
        load_fold_sources('missing', 'missing', '33', 'not-published', source_domains=['33', 'W', 'G'])
    except ValueError:
        pass
    else:
        raise AssertionError('Heldout domain entered source allowlist')
    loader_self_test()
    print('source193 synthetic official graphs, source allowlists and partial-tail window labels passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.parse_args()
    self_test()
