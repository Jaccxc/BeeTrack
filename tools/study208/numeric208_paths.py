"""GT-free compact K-path arithmetic. Source only until numeric dispatch."""
from collections import defaultdict


def segments(catalog, selected):
    cuts = {c['candidate']: c for c in catalog['cuts']}
    assert len(selected) == len(set(selected)) and set(selected) <= cuts.keys()
    chosen = defaultdict(list)
    for i in selected:
        c = cuts[i]
        assert c['valid']
        chosen[c['run_key']].append(c)
    endpoints = catalog['endpoints']
    result = []
    for run in catalog['runs']:
        key, length = run['run_key'], run['nrows']
        positions = sorted(chosen[key], key=lambda c: c['right_ordinal'])
        assert len({c['right_ordinal'] for c in positions}) == len(positions)
        boundaries = [0, *(c['right_ordinal'] for c in positions), length]
        assert all(a < b for a, b in zip(boundaries, boundaries[1:]))
        changes = run['side_changes']
        initial = endpoints[str(run['first_raw'])]['inside']
        current = initial
        last_position, last_frame = 0, endpoints[str(run['first_raw'])]['frame']
        for change in changes:
            assert last_position < change['ordinal'] < length and change['frame'] > last_frame
            assert type(change['inside']) is bool and change['inside'] != current
            current = change['inside']
            last_position, last_frame = change['ordinal'], change['frame']
        assert current == endpoints[str(run['last_raw'])]['inside']
        for j, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
            first = run['first_raw'] if j == 0 else positions[j - 1]['right_raw']
            last = run['last_raw'] if j == len(positions) else positions[j]['left_raw']
            start, end = endpoints[str(first)], endpoints[str(last)]
            inside = initial
            for change in changes:
                if change['ordinal'] <= a:
                    inside = change['inside']
            assert inside == start['inside'] and start['frame'] <= end['frame']
            local = [dict(frame=c['frame'], inside=c['inside']) for c in changes
                     if a < c['ordinal'] < b]
            assert all(start['frame'] < c['frame'] <= end['frame'] for c in local)
            assert (local[-1]['inside'] if local else inside) == end['inside']
            result.append(dict(segment_id=len(result) + 1, original_run=key,
                               first_raw=first, last_raw=last, first_frame=start['frame'],
                               last_frame=end['frame'], new_start=j > 0,
                               new_end=j < len(positions), start_inside=inside,
                               end_inside=end['inside'], changes=local, nrows=b - a))
    assert sum(s['nrows'] for s in result) == sum(r['nrows'] for r in catalog['runs'])
    return result


def relink(catalog, pieces):
    import numpy as np
    endpoints = catalog['endpoints']
    starts = defaultdict(list)
    for b in pieces:
        starts[b['first_frame']].append(b)
    eligible = []
    for a in pieces:
        for date in range(a['last_frame'] + 1, a['last_frame'] + 4):
            for b in starts[date]:
                if a['original_run'] == b['original_run'] or not (a['new_end'] or b['new_start']):
                    continue
                left, right = endpoints[str(a['last_raw'])], endpoints[str(b['first_raw'])]
                scale = (left['diagonal'] + right['diagonal']) / 2
                assert scale > 0
                distance = float(np.linalg.norm(np.asarray(left['center'], dtype=np.float64)
                                                - np.asarray(right['center'], dtype=np.float64)) / scale)
                if distance <= .75:
                    eligible.append(dict(left=a['segment_id'], right=b['segment_id'],
                                         gap=date - a['last_frame'], distance=distance,
                                         left_frame=a['last_frame'], right_frame=date))
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for edge in eligible:
        outgoing[edge['left']].append(edge)
        incoming[edge['right']].append(edge)

    def unique(edges):
        minimum = min(e['distance'] for e in edges)
        best = [e for e in edges if e['distance'] == minimum]
        return best[0] if len(best) == 1 else None

    forward = {k: unique(v) for k, v in outgoing.items()}
    backward = {k: unique(v) for k, v in incoming.items()}
    proposals = [e for e in eligible if forward[e['left']] is e and backward[e['right']] is e]
    proposals.sort(key=lambda e: (e['left_frame'], e['right_frame'], e['left'], e['right']))
    parent = {s['segment_id']: s['segment_id'] for s in pieces}

    def find(sid):
        while parent[sid] != sid:
            sid = parent[sid]
        return sid

    accepted, rejected = [], []
    for edge in proposals:
        a, b = find(edge['left']), find(edge['right'])
        if a == b:
            rejected.append(dict(edge, reason='cycle'))
        else:
            parent[b] = a
            accepted.append(edge)
    return dict(eligible_edges=eligible, mutual_proposals=proposals,
                accepted_edges=accepted, rejected_edges=rejected)


def signed_events(pieces, accepted_edges):
    by_id = {s['segment_id']: s for s in pieces}
    after, before = {}, {}
    for edge in accepted_edges:
        a, b = edge['left'], edge['right']
        assert a not in after and b not in before
        assert by_id[a]['last_frame'] < by_id[b]['first_frame']
        after[a], before[b] = b, a
    visited, result = set(), []
    for head in sorted(set(by_id) - set(before)):
        path, sid = [], head
        while True:
            assert sid not in visited
            visited.add(sid)
            path.append(by_id[sid])
            if sid not in after:
                break
            sid = after[sid]
        first, last = path[0]['start_inside'], path[-1]['end_inside']
        changes, previous = [], None
        for segment in path:
            if previous is not None and previous != segment['start_inside']:
                changes.append(dict(frame=segment['first_frame'], inside=segment['start_inside']))
            changes.extend(segment['changes'])
            previous = segment['end_inside']
        if first != last:
            dates = [c['frame'] for c in changes if c['inside'] == last]
            assert dates
            result.append(dates[-1] * (1 if last else -1))
    assert visited == set(by_id)
    return sorted(result)


def state_events(catalog, selected, reconnect):
    pieces = segments(catalog, selected)
    links = relink(catalog, pieces) if reconnect else None
    return signed_events(pieces, [] if links is None else links['accepted_edges']), pieces, links
