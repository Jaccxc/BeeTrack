"""Pure in-memory §190 GT gate; caller must verify the preregistration/freeze."""
from collections import Counter, defaultdict

import numpy as np

from beetrack.evaluate import by_track, frame_match, inside, zone_tracks


def _physical_key(row):
    # Same native-frame/two-decimal geometry key as study187.gtcheck187.gt.
    return (int(row[0]), *(f'{x:.2f}' for x in row[2:6]))


def _owners(gt, rows):
    return frame_match(gt, rows, .5) if len(gt) and len(rows) else np.full(len(rows), -1)


def gate_and_witness(rows, D, prepared_GT, source_GT, candidates, poly, *, owners=None):
    """Return full witness MOT, per-candidate answers and optional support menus.

    Inputs: canonical Nx7 MOT baseline (same raw row indices), raw Nx6 xyxy
    detections, prepared Nx6 MOT GT, original pre-repair Nx6 GT in clip frames,
    frozen candidates, and official entrance polygon. Optional owners are official
    native-baseline sorted assignments scattered back to original row order.
    No files are read.
    """
    rows = np.asarray(rows, float).reshape(-1, 7)
    D = np.asarray(D, float).reshape(-1, 6)
    source = np.asarray(source_GT, float).reshape(-1, 6)
    G = by_track(np.asarray(prepared_GT, float).reshape(-1, 6))
    assert len(np.unique(rows[:, :2], axis=0)) == len(rows), 'baseline frame/id tie'
    source_groups = defaultdict(list)
    pair_counts = Counter((int(r[0]), int(r[1])) for r in source)
    for i, row in enumerate(source):
        source_groups[_physical_key(row)].append(i)
    physical, uncertain = {}, set()
    for i, row in enumerate(G):
        hits = source_groups[_physical_key(row)]
        pair = (int(row[0]), int(row[1]))
        if len(hits) != 1 or pair_counts[(int(source[hits[0], 0]), int(source[hits[0], 1]))] != 1:
            uncertain.add(pair)
        else:
            physical[i] = hits[0]
    gt_pairs = defaultdict(list)
    for i, row in enumerate(G):
        gt_pairs[(int(row[0]), int(row[1]))].append(i)
    uncertain.update(pair for pair, hits in gt_pairs.items() if len(hits) != 1)
    if owners is None:
        under = _owners(G, rows)
    else:
        under = np.asarray(owners)
        assert under.shape == (len(rows),), 'owner shape differs from baseline'
        assert np.issubdtype(under.dtype, np.integer), 'owners must be integer GT IDs'
        assert np.all((under == -1) | np.isin(under, G[:, 1])), 'owner absent from canonical GT'
    retained = D[D[:, 5] >= .4]
    det_mot = np.column_stack((retained[:, 0], np.arange(len(retained)),
                               retained[:, 1:3], retained[:, 3:5] - retained[:, 1:3]))
    det_under = _owners(G, det_mot)
    observed = {(int(r[0]), int(g)) for r, g in zip(retained, det_under) if g >= 0}
    reports, menus, proposals = [], [], []
    for c in candidates:
        rep = dict(candidate_id=c['candidate_id'], B=int(c['B']), kind=c['kind'],
                   status='unknown', reason='', support=[], chosen=None)
        reports.append(rep)
        anchors = c['anchor_frames']
        ids, why = {}, ''
        for who in ('A', 'B'):
            ix = [int(i) for i in c['row_indices'][who] if rows[int(i), 0] in anchors]
            if len(ix) != 2 or set(rows[ix, 0]) != set(anchors):
                why = 'missing anchor row'
                break
            owners = [int(under[i]) for i in ix]
            if -1 in owners:
                why = 'anchor unmatched at IoU .5'
                break
            if any((int(rows[i, 0]), g) in uncertain for i, g in zip(ix, owners)):
                why = 'anchor source identity ambiguous'
                break
            if len(set(owners)) != 1:
                rep['status'], why = 'excluded', 'anchor GT identity changes'
                break
            ids[who] = owners[0]
        if why:
            rep['reason'] = why
            continue
        rep.update(resident_gt_id=ids['A'], query_gt_id=ids['B'])
        if ids['A'] == ids['B']:
            rep.update(status='same_bee', reason='anchors belong to same GT identity')
            continue
        rep.update(status='excluded', reason='no qualified missing observation at IoU .5')
        segment = rows[rows[:, 1] == c['B']]
        endpoint = segment[np.argmin(segment[:, 0]) if c['kind'] == 'start' else np.argmax(segment[:, 0])]
        assert all(0 < abs(f - endpoint[0]) <= 12 and
                   (f < endpoint[0] if c['kind'] == 'start' else f > endpoint[0])
                   for f in c['permitted_frames']), 'invalid frozen support date'
        endpoint_side = bool(inside(endpoint[None], poly)[0])
        options, qualified = [], []
        for frame in c['permitted_frames']:
            pair = (int(frame), ids['B'])
            hit = gt_pairs[pair]
            s = dict(frame=int(frame), status='unknown')
            rep['support'].append(s)
            if len(hit) != 1 or pair in uncertain or hit[0] not in physical:
                s['reason'] = 'missing or ambiguous physical GT row'
                continue
            gi = hit[0]
            ri = physical[gi]
            raw = source[ri]
            centre = raw[2:4] + raw[4:6] / 2
            if np.linalg.norm(centre - np.asarray(c['center'])) > c['radius']:
                s.update(status='excluded', reason='outside frozen crop disk')
                continue
            if pair in observed:
                s.update(status='excluded', reason='retained detector assigned at IoU .5')
                continue
            add = [int(frame), int(c['B']), *raw[2:6].tolist(), 1.0]
            side = bool(inside(np.asarray(add)[None], poly)[0])
            s.update(status='confirmed', reason='no retained detector assignment at IoU .5',
                     gt_id=ids['B'], native_gt_id=int(raw[1]), source_row=ri,
                     row=add, inside=side, opposite_endpoint=side != endpoint_side)
            options.append(add)
            qualified.append(s)
        menus.append(dict(candidate_id=c['candidate_id'], B=int(c['B']), kind=c['kind'], rows=options))
        if qualified:
            rep.update(status='confirmed', reason='qualified missing observations at IoU .5')
        elif any(s['status'] == 'unknown' for s in rep['support']):
            rep.update(status='unknown', reason='missing/ambiguous GT support, no confirmed observation')
        opposite = [s for s in qualified if s['opposite_endpoint']]
        if opposite:
            chosen = min(opposite, key=lambda s: (s['frame'] if c['kind'] == 'start' else -s['frame'],
                                                  s['native_gt_id'], s['source_row']))
            rep['chosen'] = chosen
            proposals.append((rep, chosen))
    usage = Counter((s['frame'], s['gt_id']) for _, s in proposals)
    destination = Counter((s['frame'], r['B']) for r, s in proposals)
    additions, conflicts = [], []
    occupied = {tuple(map(int, r[:2])) for r in rows}
    for rep, s in proposals:
        if usage[(s['frame'], s['gt_id'])] > 1 or destination[(s['frame'], rep['B'])] > 1 or (s['frame'], rep['B']) in occupied:
            rep['witness_status'] = 'conflict_dropped'
            conflicts.append(rep['candidate_id'])
        else:
            rep['witness_status'] = 'added'
            additions.append((rep['candidate_id'], s['row']))
    additions.sort(key=lambda p: (p[1][0], p[1][1], str(p[0])))
    witness = np.concatenate((rows, np.asarray([r for _, r in additions], float).reshape(-1, 7)))
    assert np.array_equal(witness[:len(rows)], rows)
    assert len(np.unique(witness[:, :2], axis=0)) == len(witness)
    def whole_events(mot):
        mot = by_track(mot)
        result = {}
        for tid, (event, _, end, crosses) in zone_tracks(mot, poly).items():
            dates = [j for j, direction in crosses if direction == event]
            result[tid] = (int(mot[dates[-1] if dates else end - 1, 0]), event) if event else None
        return result

    before = whole_events(rows) if len(rows) else {}
    after = whole_events(witness) if len(witness) else {}
    changed = {i for i in before if before[i] != after[i]}
    changed_queries = sorted({r['query_gt_id'] for r in reports
                              if r.get('witness_status') == 'added' and r['B'] in changed})
    both_ends = [int(b) for b in sorted({r['B'] for r in reports})
                 if {r['kind'] for r in reports if r['B'] == b and r.get('witness_status') == 'added'} == {'start', 'end'}]
    return dict(rows=witness, candidates=reports, constrained_menu=menus,
                anchor_owner_order='supplied official baseline scattered to original order' if owners is not None else 'input baseline order',
                conflicts=conflicts, both_ends_added=both_ends,
                event_changed_segments=sorted(changed), event_changed_query_gt_ids=changed_queries,
                interpretation='Missing observation means no retained raw detector assignment at IoU .5; not visibility or true absence.')


def _self_check():
    poly = [[0, 0], [20, 0], [20, 20], [0, 20]]
    rows = np.array([[f, who, x, 4, 2, 2, .8]
                     for f in (3, 4) for who, x in ((10, 30), (20, 5))], float)
    G = np.array([[f, who, x, 4, 2, 2]
                  for f in (1, 2, 3, 4) for who, x in ((1, 30), (2, 25 if f < 3 else 5))], float)
    D = np.column_stack((rows[:, 0], rows[:, 2:4], rows[:, 2:4] + rows[:, 4:6], rows[:, 6]))
    candidate = dict(candidate_id='s', B=20, A=10, kind='start', boundary_frame=3,
                     anchor_frames=[3, 4], row_indices={'A': [0, 2], 'B': [1, 3]},
                     permitted_frames=[2, 1], center=[16, 5], radius=40)
    run = lambda cs, src=G, ds=D: gate_and_witness(rows, ds, G, src, cs, poly)
    r = run([candidate])
    official_owners = _owners(by_track(G), rows)
    overridden = gate_and_witness(rows, D, G, G, [candidate], poly, owners=official_owners)
    assert np.array_equal(r['rows'], overridden['rows'])
    for invalid in (official_owners[:-1], official_owners.astype(float)):
        try:
            gate_and_witness(rows, D, G, G, [candidate], poly, owners=invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid owners accepted')
    assert len(r['rows']) == 5 and r['rows'][-1, 0] == 1
    assert r['event_changed_query_gt_ids'] == [2]
    assert len(r['constrained_menu'][0]['rows']) == 2
    assert np.array_equal(r['rows'][:4], rows)
    # Detector coverage at .5 blocks additions; below .4 is excluded first.
    extra = np.array([[1, 25, 4, 27, 6, .4], [2, 25, 4, 27, 6, .4]])
    assert len(run([candidate], ds=np.concatenate((D, extra)))['rows']) == 4
    extra[:, 5] = .399
    assert len(run([candidate], ds=np.concatenate((D, extra)))['rows']) == 5
    # Duplicate raw native identity in a frame invalidates both physical rows.
    src = G.copy()
    src[(src[:, 0] == 3) & (src[:, 1] == 1), 1] = 2
    assert run([candidate], src=src)['candidates'][0]['status'] == 'unknown'
    # Shared proposal claims are all dropped with no second-choice retry.
    c2 = dict(candidate, candidate_id='s2')
    rr = run([candidate, c2])
    assert len(rr['rows']) == 4 and len(rr['conflicts']) == 2
    # Raw support geometry is copied, not prepared rounded geometry.
    src = G.copy()
    src[0, 2] += .001
    src[1, 2] += .001
    rr = run([candidate], src=src)
    assert rr['rows'][-1, 2] == 25.001
    # Missing GT support is unknown rather than an absence/visibility claim.
    gg = G[G[:, 0] >= 3]
    rr = gate_and_witness(rows, D, gg, gg, [candidate], poly)
    assert rr['candidates'][0]['status'] == 'unknown'
    # Mirrored end queries select the latest qualifying opposite-zone row.
    end = dict(candidate, candidate_id='e', kind='end', boundary_frame=4,
               permitted_frames=[5, 6])
    eg = np.concatenate((G, [[f, who, x, 4, 2, 2]
                              for f in (5, 6) for who, x in ((1, 30), (2, 25))]))
    rr = gate_and_witness(rows, D, eg, eg, [end], poly)
    assert rr['rows'][-1, 0] == 6
    rr = gate_and_witness(rows, D, eg, eg, [candidate, end], poly)
    assert rr['both_ends_added'] == [20] and len(rr['rows']) == 6
    # Menus retain same-zone physical options even when witness cannot use them.
    same = G.copy()
    same[(same[:, 0] < 3) & (same[:, 1] == 2), 2] = 6
    rr = gate_and_witness(rows, D, same, same, [candidate], poly)
    assert len(rr['rows']) == 4 and len(rr['constrained_menu'][0]['rows']) == 2


if __name__ == '__main__':
    _self_check()
    print('labels190 synthetic checks passed')
