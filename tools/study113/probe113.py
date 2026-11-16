"""Cross-fit static detector landmarks between clip halves and measure native event touches."""

import argparse
from fractions import Fraction

import numpy as np

from beetrack.evaluate import by_track, zone_tracks


def quarter_frames(nframes):
    frames = []
    for j in range(4):
        a, b = 1+j*nframes//4, (j+1)*nframes//4
        if b-a < 30:
            raise ValueError('Each quarter requires at least 31 distinct frames')
        frames.append([a+k*(b-a)//30 for k in range(31)])
    return frames


def iou(a, b):
    dx, dy = min(a[2],b[2])-max(a[0],b[0]), min(a[3],b[3])-max(a[1],b[1])
    if dx <= 0 or dy <= 0:
        return Fraction(0)
    inter = dx*dy
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def quarter_boxes(raw):
    native = []
    for raw_index, row in enumerate(raw):
        if len(row) != 5 or not np.isfinite(row).all():
            raise ValueError('Expected finite xyxy and score from the static detector')
        x1,y1,x2,y2,score = map(float,row)
        if not (x2 > x1 and y2 > y1 and 0 <= score <= 1):
            raise ValueError('Invalid native detector box or confidence')
        if score >= .25:
            native.append(dict(raw_index=raw_index,score=score,xyxy=[x1,y1,x2,y2]))
    native.sort(key=lambda r: (-r['score'],*r['xyxy'],r['raw_index']))
    # Preserve actual FP32 coordinates exactly for inclusive comparisons and equal-IoU tie ranks.
    exact = [tuple(Fraction.from_float(x) for x in r['xyxy']) for r in native]
    return native, exact


def stable_marks(left, right):
    candidates = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            value = iou(a,b)
            if value >= Fraction(1,2):
                candidates.append((-value,i,j))
    seen_l, seen_r, marks = set(), set(), []
    for negative_iou,i,j in sorted(candidates):
        if i not in seen_l and j not in seen_r:
            seen_l.add(i)
            seen_r.add(j)
            marks.append(dict(ranks=[i,j],iou=float(-negative_iou),selected_events=0,touching_events=0))
    return sorted(marks,key=lambda m:m['ranks'])


def mot_box(row):
    # Native MOT x,y,w,h are decimal pixels. Adding in decimal-rational form avoids rounding the right edge.
    x,y,w,h = (Fraction(str(float(v))) for v in row[2:6])
    return (x,y,x+w,y+h)


def measured_case(mot, polygon, quarters, nframes):
    """Return an outcome-free manifest from native decimal xywh MOT and 4 FP32 probe streams.

    Ranks, quarter indices, and half indices start at 0. Frame numbers start at 1.
    The canonical event time is f+ even when the preceding observed frame is separated by a gap.
    """
    if len(quarters) != 4:
        raise ValueError('Expected four time-ordered quarterly detector outputs')
    ranked, boxes = zip(*(quarter_boxes(raw) for raw in quarters))
    landmarks = [stable_marks(boxes[q],boxes[q+1]) for q in (0,2)]
    rows = np.asarray(mot,dtype=float)
    if rows.ndim != 2 or rows.shape[1] < 6 or not np.isfinite(rows[:,:6]).all():
        raise ValueError('Expected a finite raw MOT matrix')
    rows = by_track(rows[:,:6])
    if (np.any(rows[:,4:6] <= 0) or np.any(rows[:,0] < 1) or np.any(rows[:,0] > nframes)
            or not np.array_equal(rows[:,:2],np.rint(rows[:,:2]))
            or len(np.unique(rows[:,:2],axis=0)) != len(rows)):
        raise ValueError('Native MOT requires unique frame/ID and valid integer frames and boxes')
    geometry = zone_tracks(rows,polygon)
    events = []
    for pid, (direction,_,_,crossings) in geometry.items():
        if not direction:
            continue
        at = next(i for i,d in reversed(crossings) if d == direction)
        previous, after = int(rows[at-1,0]), int(rows[at,0])
        prebox, postbox = mot_box(rows[at-1]), mot_box(rows[at])
        event = dict(id=pid, direction=direction, f_minus=previous, f_plus=after,
                     pre_xyxy=list(map(float,prebox)), post_xyxy=list(map(float,postbox)),
                     reference_half=None, landmarks_touched=[], selected_landmark=None,
                     best_landmark=None, best_ious=None, best_score=None)
        if (previous <= nframes//2) == (after <= nframes//2):
            ref = int(after <= nframes//2)
            event['reference_half'] = ref
            scored = []
            for mark in landmarks[ref]:
                i,j = mark['ranks']
                left,right = boxes[2*ref][i],boxes[2*ref+1][j]
                four = [iou(prebox,left),iou(prebox,right),iou(postbox,left),iou(postbox,right)]
                scored.append((-min(four),i,j,four,mark))
            if scored:
                negative,i,j,four,best = min(scored)
                event.update(best_landmark=[i,j],best_score=float(-negative),best_ious=list(map(float,four)))
                if -negative >= Fraction(3,10):
                    event['selected_landmark'] = [i,j]
                    best['selected_events'] += 1
            for negative,i,j,four,mark in sorted(scored):
                if -negative >= Fraction(3,10):
                    mark['touching_events'] += 1
                    event['landmarks_touched'].append(dict(ranks=[i,j],ious=list(map(float,four))))
        events.append(event)
    events.sort(key=lambda e:(e['f_plus'],e['direction']=='out',e['id']))
    touched = [e for e in events if e['selected_landmark'] is not None]
    return dict(quarters=list(ranked),landmarks=landmarks,events=events,
                touched_ids=[e['id'] for e in touched],
                counts_by_half=[{direction:dict(assessable=sum(e['direction']==direction and e['reference_half']==1-h
                                                               for e in events),
                                               touched=sum(e['direction']==direction and e['reference_half']==1-h
                                                           for e in touched)) for direction in ('in','out')}
                                for h in (0,1)],
                counts={direction:dict(total=sum(e['direction']==direction for e in events),
                                       split_excluded=sum(e['direction']==direction and e['reference_half'] is None
                                                          for e in events),
                                       assessable=sum(e['direction']==direction and e['reference_half'] is not None
                                                      for e in events),
                                       touched=sum(e['direction']==direction for e in touched))
                        for direction in ('in','out')})


def self_check():
    poly = [[0,0],[10,0],[10,20],[0,20]]
    samples = quarter_frames(3950)
    assert [(s[0],s[-1]) for s in samples] == [(1,987),(988,1975),(1976,2962),(2963,3950)]
    assert len(set(f for s in samples for f in s)) == 4*31 and all(len(s)==31 for s in samples)
    original = [[-2,3,2,7,.5],[-2,3,2,7,.5],[-2,3,2,7,.2499]]
    ranked,_ = quarter_boxes(original)
    assert [r['raw_index'] for r in ranked] == [0,1]
    # Identical pairs create ties at both pairing and event selection; stable ranks resolve both.
    q = [[],[],original,original]
    def seq(tid, frames, xs):
        return [[f,tid,x,3,4,4] for f,x in zip(frames,xs)]
    data = np.array(seq(12,[3,5],[-3,-1]) + seq(13,[3,5],[-1,-3])
                    + seq(14,[100,101],[-3,-1]) + seq(15,[105,108],[-3,-1])
                    + seq(16,[7,8,9,10],[-3,-1,-10,-1]) + seq(17,[3,5],[-3,-1]),float)
    out = measured_case(data[::-1],poly,q,200)
    assert out['landmarks'][0] == [] and [l['ranks'] for l in out['landmarks'][1]] == [[0,0],[1,1]]
    assert out['touched_ids'] == [12,17,13]  # No contention between different native IDs.
    assert out['counts'] == {'in':{'total':5,'split_excluded':1,'assessable':4,'touched':2},
                             'out':{'total':1,'split_excluded':0,'assessable':1,'touched':1}}
    assert [l['selected_events'] for l in out['landmarks'][1]] == [3,0]
    assert [l['touching_events'] for l in out['landmarks'][1]] == [3,3]
    assert out['counts_by_half'][0]['in']['touched'] == 2
    byid = {e['id']:e for e in out['events']}
    assert byid[12]['selected_landmark'] == [0,0] and len(byid[12]['landmarks_touched']) == 2
    assert byid[12]['f_minus'] == 3 and byid[12]['f_plus'] == 5 and byid[12]['reference_half'] == 1
    assert byid[14]['reference_half'] is None and byid[15]['reference_half'] == 0
    # Earlier crossings and postbox alone cannot turn the final non-touch into a hit.
    assert (byid[16]['f_minus'],byid[16]['f_plus'],byid[16]['selected_landmark']) == (9,10,None)
    assert measured_case(data,poly,q,200) == out
    assert measured_case(np.empty((0,6)),poly,q,200)['events'] == []
    # A valid quarter pair can agree on the prebox while only its left box admits the postbox.
    shifted = [[],[],[[-2,3,2,7,.5]],[[-3.25,3,.75,7,.5]]]
    one_sided = measured_case(np.array(seq(31,[3,4],[-3,-1]),float),poly,shifted,200)
    assert len(one_sided['landmarks'][1]) == 1 and not one_sided['touched_ids']
    assert one_sided['events'][0]['best_score'] < .3 and one_sided['events'][0]['best_landmark'] == [0,0]
    # The same raw ID far apart denotes two native events, with distinct resolved IDs and opposite reference halves.
    reused = np.array(seq(31,[3,4],[-3,-1])+seq(31,[160,161],[-1,-3]),float)
    two = measured_case(reused,poly,[original]*4,200)
    assert two['touched_ids'] == [31,100031] and [r['reference_half'] for r in two['events']] == [1,0]
    # Inclusive exact boundaries, both landmark and four-way contact comparisons.
    assert iou(tuple(map(Fraction,[0,0,2,1])),tuple(map(Fraction,[0,0,1,1]))) == Fraction(1,2)
    assert iou(tuple(map(Fraction,[0,0,10,1])),tuple(map(Fraction,[0,0,3,1]))) == Fraction(3,10)
    try:
        measured_case(np.vstack((data,data[0])),poly,q,200)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate MOT ID and frame accepted')
    print('Synthetic: quarter sampling, cross-fit halves, inclusive IoU, native last crossing, ties and deduplication passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['self-check'])
    parser.parse_args()
    self_check()
