"""Synthetic only: observed peer suffix copying, not an identity/visibility model.

Rows are native MOT (frame,id,x,y,w,h,score); inputs are never modified.
Two consecutive anchors must be mutual unique IoU>=.5 matches. This is
observation uniqueness, not proof of physical identity after the anchors.
Run with final_project/.venv/Scripts/python.exe (official scorer dependencies).
"""
import json


def iou(a, b):
    x = max(0, min(a[2]+a[4], b[2]+b[4])-max(a[2], b[2]))
    y = max(0, min(a[3]+a[5], b[3]+b[5])-max(a[3], b[3]))
    inter = x*y
    return inter/(a[4]*a[5]+b[4]*b[5]-inter) if inter else 0.


def validate(rows):
    import math
    keys = set()
    for r in rows:
        if len(r) != 7 or not all(math.isfinite(v) for v in r):
            raise ValueError('finite Nx7 MOT required')
        if r[0] != int(r[0]) or r[1] != int(r[1]) or min(r[4:6]) <= 0:
            raise ValueError('integer frame/id and positive dimensions required')
        key = (r[0], r[1])
        if key in keys:
            raise ValueError('duplicate frame/id')
        keys.add(key)


def mutual(a, b, left, right):
    return ([r for r in right if iou(a, r) >= .5] == [b]
            and [r for r in left if iou(b, r) >= .5] == [a])


def borrow(host, peer, requests):
    """requests: (host_id, peer_id, [two anchor frames]).

    Conservatively stop at missing/nonconsecutive donor dates or any host
    overlap (unique duplicate or ambiguous). Shared donor tracks invalidate ALL
    conflicting proposals, with no second choices. Native host order is kept;
    copied suffix rows append in request/date order, preserving peer geometry.
    """
    host, peer = [tuple(r) for r in host], [tuple(r) for r in peer]
    validate(host)
    validate(peer)
    if len({q[0] for q in requests}) != len(requests):
        raise ValueError('one request per host track')
    reports = []
    for hid, pid, dates in requests:
        report = dict(host_id=hid, peer_id=pid, anchored=False, sources=[], stop=None)
        reports.append(report)
        own = sorted((r for r in host if r[1] == hid), key=lambda r:r[0])
        if len(dates) != 2 or dates[1] != dates[0]+1 or not own or dates[-1] != own[-1][0]:
            report['stop'] = 'anchors_not_consecutive_at_host_end'
            continue
        for f in dates:
            left = [r for r in host if r[0] == f]
            right = [r for r in peer if r[0] == f]
            a = next((r for r in left if r[1] == hid), None)
            b = next((r for r in right if r[1] == pid), None)
            if a is None or b is None or not mutual(a, b, left, right):
                report['stop'] = 'anchor_not_mutual_unique'
                break
        if report['stop']:
            continue
        report['anchored'] = True
        expected = dates[-1]+1
        suffix = sorted(((i,r) for i,r in enumerate(peer)
                         if r[1] == pid and r[0] > dates[-1]), key=lambda ir:ir[1][0])
        for index, row in suffix:
            if row[0] != expected:
                report['stop'] = 'donor_date_gap'
                break
            left = [r for r in host if r[0] == row[0]]
            right = [r for r in peer if r[0] == row[0]]
            overlaps = [r for r in left if iou(r, row) >= .5]
            if overlaps:
                report['stop'] = ('host_already_observes_donor' if len(overlaps)==1
                                  and mutual(overlaps[0], row, left, right)
                                  else 'ambiguous_host_overlap')
                break
            report['sources'].append(dict(peer_row=index, frame=row[0], peer_id=pid))
            expected += 1
        report['stop'] = report['stop'] or 'donor_observations_exhausted'
    claims = {}
    for j, report in enumerate(reports):
        if report['anchored']:
            claims.setdefault(report['peer_id'], set()).add(j)
    conflicts = set().union(*(v for v in claims.values() if len(v)>1)) if claims else set()
    result = list(host)
    for j, report in enumerate(reports):
        if j in conflicts:
            report['sources'] = []
            report['stop'] = 'donor_competition_drop_all'
        for source in report['sources']:
            r = peer[source['peer_row']]
            result.append((r[0], report['host_id'], *r[2:]))
    validate(result)
    assert result[:len(host)] == host
    return result, reports


def selftest():
    def row(f, pid, x): return (f,pid,x,0,4,4,.9)
    host = [row(1,1,0), row(2,1,4)]
    prefix = [row(1,9,0), row(2,9,4)]
    # Identical host prefix, genuine departure vs occluded turnaround.
    exit_peer = prefix+[row(3,9,8),row(4,9,12)]
    return_peer = prefix+[row(3,9,0),row(4,9,-4)]
    exit_rows, ex = borrow(host,exit_peer,[(1,9,[1,2])])
    return_rows, ret = borrow(host,return_peer,[(1,9,[1,2])])
    assert exit_rows[:2] == return_rows[:2] == host
    assert [r[2] for r in exit_rows[2:]] == [8,12]
    assert [r[2] for r in return_rows[2:]] == [0,-4]
    # Unseen peer ID switch: identical framewise unlabeled geometry/counts.
    clean = exit_peer+[row(3,10,20),row(4,10,24)]
    switched = [((r[0],19-r[1],*r[2:]) if r[0]>2 else r) for r in clean]
    geometry = lambda rows: sorted((r[0],*r[2:]) for r in rows)
    assert geometry(clean) == geometry(switched)
    good, g = borrow(host,clean,[(1,9,[1,2])])
    wrong, w = borrow(host,switched,[(1,9,[1,2])])
    assert [r[2] for r in good[2:]] == [8,12]
    assert [r[2] for r in wrong[2:]] == [20,24]
    # Already observed by another host track; retain prefix and stop.
    out, duplicate = borrow(host+[row(3,2,8)],exit_peer,[(1,9,[1,2])])
    assert len(out)==3 and duplicate[0]['stop']=='host_already_observes_donor'
    # Two different host IDs can have disjoint anchor dates for one donor.
    h2 = [row(0,2,-4),row(1,2,0),row(2,1,4),row(3,1,8)]
    p2 = [row(f,9,4*(f-1)) for f in range(5)]
    out, competition = borrow(h2,p2,[(2,9,[0,1]),(1,9,[2,3])])
    assert out==h2 and all(r['stop']=='donor_competition_drop_all' for r in competition)
    _, gap = borrow(host,prefix+[row(4,9,12)],[(1,9,[1,2])])
    assert gap[0]['stop']=='donor_date_gap'
    _, ambiguous = borrow(host+[row(2,3,4)],exit_peer,[(1,9,[1,2])])
    assert ambiguous[0]['stop']=='anchor_not_mutual_unique'
    return dict(status='PASS',departure=ex,occluded_turnaround=ret,
                identity_switch=dict(frame_geometry_identical=True,clean=g,switched=w,
                                     undetectable_from_two_anchors=True),
                duplicate=duplicate,competition=competition,gap=gap,ambiguous=ambiguous,
                scope='Synthetic observations only; no trained model or causal claim.')


def identical_input_worlds():
    """Only hidden physical correspondence changes; not even deployable IDs do.

    Every box has the same appearance token. The synthetic renderer contract
    is fixed solid squares, so identical box bytes also specify identical RGB.
    Official zone_tracks determines net events; date is last same-way crossing.
    """
    import hashlib
    import math
    import sys
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'final_project'))
    from beetrack.evaluate import by_track, zone_tracks
    from beetrack.p0 import per_window
    from tools.study197.candidates import window_counts

    def box(f, pid, x, y): return (f+748,pid,x-.1,y-.1,.2,.2,.9)
    poly = np.array([[0,-10],[10,-10],[10,10],[0,10]], np.float32)

    def event(points):
        rows = by_track(np.array([box(f,1,x,y) for f,x,y in points],float))
        ev, _, _, crossings = zone_tracks(rows,poly)[1]
        dates = [int(rows[i,0]) for i,d in crossings if d==ev]
        return dict(net_in=int(ev=='in'),event=ev,date=dates[-1] if dates else None,
                    crossings=[(int(rows[i,0]),d) for i,d in crossings])

    prefix = [(0,-1,1),(1,2,1),(2,2,1)]
    host = [box(f,1,x,y) for f,x,y in prefix]
    host += [box(f,2,2,-1) for f in range(3)]
    peer_prefix = [(r[0],9 if r[1]==1 else 10,*r[2:]) for r in host]

    def pair(branches):
        peer = peer_prefix+[box(3,9,*branches[0]),box(3,10,*branches[1])]
        visible = dict(host=host,peer=peer,anchors=[[1,9,[749,750]],[2,10,[749,750]]],
                       appearance='same solid RGB for every bee',zone='x>=0',
                       motion_bound=4.,renderer='solid .2 by .2 squares')
        # Independent serializations of the two worlds include all deployed IDs.
        raw = json.dumps(visible,sort_keys=True,separators=(',',':')).encode()
        world_bytes = [raw,bytes(bytearray(raw))]
        assert world_bytes[0] == world_bytes[1]
        def render(payload):
            frames = np.zeros((4,100,160,3),dtype=np.uint8)
            rows = json.loads(payload)['peer']
            first = min(r[0] for r in rows)
            for r in rows:
                # Integer pixel cells from a fixed shared .1-unit/pixel grid.
                x,y = round((r[2]+4)*10),round((r[3]+4)*10)
                frames[r[0]-first,y:y+2,x:x+2,:] = (170,90,30)
            return frames.tobytes()
        rgb_bytes = [render(b) for b in world_bytes]
        assert rgb_bytes[0] == rgb_bytes[1]
        out, reports = borrow(host,peer,[(1,9,[749,750]),(2,10,[749,750])])
        assert all(r['anchored'] for r in reports) and len(out)==len(host)+2
        costs = [[math.dist(p,b) for b in branches] for p in [(2,1),(2,-1)]]
        assert max(c for row in costs for c in row)<=visible['motion_bound']
        assert costs[0][0]+costs[1][1] == costs[0][1]+costs[1][0]
        outcomes = [event(prefix+[(3,*b)]) for b in branches]
        other_prefix = [(f,2,-1) for f in range(3)]
        other_events = [event(other_prefix+[(3,*branches[1-i])]) for i in range(2)]
        complete_gt = []
        totals = []
        for i in range(2):
            rows = [box(f,1,x,y) for f,x,y in prefix+[(3,*branches[i])]]
            rows += [box(f,2,x,y) for f,x,y in other_prefix+[(3,*branches[1-i])]]
            complete_gt.append(rows)
            events = zone_tracks(by_track(np.asarray(rows,float)),poly)
            totals.append([sum(v[0]==d for v in events.values()) for d in ('in','out')])
        return dict(input_sha256=[hashlib.sha256(b).hexdigest() for b in world_bytes],
                    rgb_sha256=[hashlib.sha256(b).hexdigest() for b in rgb_bytes],
                    hidden_focus_peer_id_at_frame3=[9,10],
                    all_input_bytes_equal=True,assignment_costs=costs,
                    worlds=outcomes,other_bee_events=other_events,
                    complete_two_bee_gt=complete_gt,group_in_out=totals,
                    group_signed_net=[a-b for a,b in totals],borrow_reports=reports)

    opposite = pair([(-1,0),(5,0)])
    assert opposite['worlds'][0]['net_in']==0 and opposite['worlds'][0]['date'] is None
    assert opposite['worlds'][1]['net_in']==1 and opposite['worlds'][1]['date']==749
    assert opposite['group_in_out']==[[0,0],[1,1]]
    assert opposite['group_signed_net']==[0,0]
    assert all(abs(c-math.sqrt(10))<1e-12 for r in opposite['assignment_costs'] for c in r)
    same_side = pair([(1,0),(3,0)])
    assert [w['net_in'] for w in same_side['worlds']]==[1,1]
    assert [w['date'] for w in same_side['worlds']]==[749,749]
    # Same final side identifies net count, but does NOT generally identify date.
    # Both continuations remain motion-feasible under the identical bound=4.
    paths = [prefix+[(3,1,0),(4,-1,0),(5,1,0)],
             prefix+[(3,3,0),(4,3,0),(5,3,0)]]
    dated = [event(p) for p in paths]
    assert [w['net_in'] for w in dated]==[1,1]
    assert [w['date'] for w in dated]==[753,749]
    assert all(math.dist(a[1:],b[1:])<=4 for p in paths for a,b in zip(p,p[1:]))
    date_arrays = [np.array([[w['date'],1]],dtype=int) for w in dated]
    windows = [per_window(e,1000,750) for e in date_arrays]
    assert windows[0].tolist()==[[0,0],[1,0]]
    assert windows[1].tolist()==[[1,0],[0,0]]
    phase_windows = [window_counts([dict(frame=w['date'],direction=1)],1000) for w in dated]
    assert not np.array_equal(*phase_windows)
    assert np.array_equal(phase_windows[0][:2],windows[0])
    assert np.array_equal(phase_windows[1][:2],windows[1])
    return dict(opposite_sides=opposite,same_side=same_side,
                same_final_side_date_counterexample=dated,
                date_window_check=dict(per_window=[w.tolist() for w in windows],
                                       all30offset_L1=int(np.abs(phase_windows[0]-phase_windows[1]).sum())),
                conclusion='Identical deployable inputs cannot identify hidden identity; '
                'net count and last same-direction event date require separate checks.')


if __name__ == '__main__':
    print(json.dumps(dict(original=selftest(),identical_inputs=identical_input_worlds()), indent=2))
