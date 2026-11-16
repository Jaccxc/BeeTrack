"""Synthetic three-stream count rounding and geometry-aware representatives. No file inputs.

Within each direction, order native observations by (positive frame, seed, track ID).
The rank-2 output after m arrivals counts floor((m+1)/3), so 3*output-m lies in
{-1,0,1}. Taking a difference of two prefixes bounds any interval discrepancy
from the exact three-stream mean by 2/3. Choosing any one item in each complete
triple (or final pair) keeps 3*output-m in [-2,2], yielding 4/3. Intervals with
equal endpoint times include all tied events at once. Moving a representative
within its original 25-frame second preserves every official 750-frame count at
all 30 offsets in multiples of 25. None of these bounds compares against GT.
"""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
from operator import index
import random


def centipixels(value):
    exact = Decimal(str(value))*100
    if not exact.is_finite() or exact != exact.to_integral_value():
        raise ValueError('Native xyxy coordinates must have hundredth-pixel precision')
    return int(exact)


def overlaps_half(a, b):
    inter = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])
    return 3*inter >= area


def integer_events(streams):
    """Input {seed: [[resolved_track_id,signed_clip_frame,x1,y1,x2,y2], ...]}.

    Each block's members index the original stream rows by their stable (seed,ID).
    Evidence for each eligible representative includes every agreeing track on
    *either other stream*, including tracks outside its own triple. Each other
    seed gets at most one vote. Preserving all input rows also exposes any singleton.
    """
    if len(streams) != 3:
        raise ValueError('Three native detector streams are required')
    rows, boxes, times, by_direction = {}, {}, defaultdict(list), {1:[], -1:[]}
    for seed, records in streams.items():
        seed = index(seed)
        for record in records:
            tid, signed, x1, y1, x2, y2 = record
            tid, signed = index(tid), index(signed)
            key = (seed,tid)
            box = tuple(centipixels(x) for x in (x1,y1,x2,y2))
            if key in rows or signed == 0 or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('Invalid native ID, signed frame or box')
            rows[key], boxes[key] = list(record), box
            sign, frame = (1 if signed > 0 else -1), abs(signed)
            times[(sign,frame)].append(key)
            by_direction[sign].append((frame,seed,tid))

    result = {}
    for sign, triples in by_direction.items():
        order = [(s,i) for _,s,i in sorted(triples)]
        blocks = []
        for j in range(0,len(order),3):
            members = order[j:j+3]
            block = dict(members=[list(k) for k in members],canonical=None,geometry=None,evidence=[])
            blocks.append(block)
            if len(members) < 2:
                continue
            center = members[1]
            block['canonical'] = list(center)
            second = (abs(rows[center][1])-1)//25
            ranked = []
            for ordinal, key in enumerate(members):
                frame = abs(rows[key][1])
                if (frame-1)//25 != second:
                    continue
                agree = sorted(other for f in range(frame-5,frame+6) for other in times.get((sign,f),())
                               if other[0] != key[0] and overlaps_half(boxes[key],boxes[other]))
                n_seeds = len({s for s,_ in agree})
                ranked.append((-n_seeds,abs(ordinal-1),key))
                block['evidence'].append(dict(member=list(key), other_seeds=n_seeds, supporters=[list(k) for k in agree]))
            block['geometry'] = list(min(ranked)[2])
        result['in' if sign > 0 else 'out'] = blocks
    return dict(seeds=sorted(map(index,streams)), observations=[dict(seed=k[0],row=rows[k]) for k in sorted(rows)], blocks=result)


def verify(streams):
    record = integer_events(streams)
    lookup = {(s,r[0]):abs(r[1]) for s,rs in streams.items() for r in rs}
    for direction, blocks in record['blocks'].items():
        inputs = [abs(r[1]) for rs in streams.values() for r in rs if (r[1]>0) == (direction=='in')]
        arm = {name:[lookup[tuple(b[name])] for b in blocks if b[name] is not None] for name in ('canonical','geometry')}
        # Each integer-frame interval is a difference of two prefixes. It suffices to
        # check the complete prefix range; equal times must enter the prefix together.
        counts = Counter(inputs)
        for name, events in arm.items():
            assert len(events) == (len(inputs)+1)//3
            emitted = Counter(events)
            prefix = [0]
            for f in sorted(counts):
                prefix.append(prefix[-1] + 3*emitted[f] - counts[f])
            assert max(prefix)-min(prefix) <= (2 if name == 'canonical' else 4)
        for offset in range(0,750,25):
            bins = lambda events: Counter((f-1+offset)//750 for f in events)
            assert bins(arm['canonical']) == bins(arm['geometry'])
    return record


def self_check():
    # Every remainder, empty streams, simultaneous opposite directions and ties by seed+ID.
    for n in range(13):
        same = {s:[] for s in range(3)}
        for i in reversed(range(n)):
            same[i%3].append([i,1 if i%5 else -1,0,0,1,1])
        a = verify(same)
        shuffled = {s:list(reversed(rs)) for s,rs in reversed(list(same.items()))}
        assert a == verify(shuffled)
        for blocks in a['blocks'].values():
            for block in blocks:
                assert block['geometry'] == block['canonical']  # Equal support prefers rank 2.

    # Only records 7 and 12 share geometry at dt=5 and IoU=1/2. They are in
    # different triples: geometry selects their endpoints, leaving 8..11 empty.
    demo = {0:[[70,7,0,0,2,1],[90,9,30,0,32,1]],
            1:[[80,8,10,0,12,1],[110,11,50,0,52,1]],
            2:[[100,10,40,0,42,1],[120,12,0,0,1,1]]}
    blocks = verify(demo)['blocks']['in']
    assert [b['canonical'] for b in blocks] == [[1,80],[1,110]]
    assert [b['geometry'] for b in blocks] == [[0,70],[2,120]]
    assert (3*2-4,3*0-4) == (2,-4)  # signed interval errors scaled by three.
    assert not overlaps_half((0,0,200,100),(0,0,99,100))

    # Tied outer ranks resolve by seed/ID; two supporting tracks of the same seed cast one vote.
    tied = {7:[[1,10,0,0,2,1]], 1:[[8,12,0,0,2,1],[9,13,0,0,2,1]], 3:[[2,11,9,0,11,1]]}
    tie_block = verify(tied)['blocks']['in'][0]
    assert tie_block['geometry'] == [1,8] and tie_block['canonical'] == [3,2]
    assert tie_block['evidence'][0] == dict(member=[7,1],other_seeds=1,supporters=[[1,8],[1,9]])

    # The same per-detector counts and frames can support a different identity decision.
    moved = {s:[r.copy() for r in rows] for s,rows in demo.items()}
    moved[2][1][2:6] = [80,0,82,1]
    changed = verify(moved)['blocks']['in']
    assert [b['canonical'] for b in changed] == [b['geometry'] for b in changed]
    assert [b['canonical'] for b in changed] == [b['canonical'] for b in blocks]
    # A single fixed detector stream cannot recover the other two counts.
    assert len(verify({0:demo[0],1:[],2:[]})['blocks']['in']) == 1
    assert len(blocks) == 2

    # A nearby candidate with better support but in a different 25-frame second cannot move rank 2.
    restricted = {0:[[1,25,0,0,1,1],[2,27,5,0,6,1]],1:[[3,26,7,0,8,1]],2:[[4,28,0,0,1,1]]}
    check = verify(restricted)['blocks']['in'][0]
    assert check['canonical'] == check['geometry'] == [1,3] and [0,1] not in [x['member'] for x in check['evidence']]
    try:
        integer_events({0:[[1,3,0,0,1,1],[1,-4,0,0,1,1]],1:[],2:[]})
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate resolved track identity within one stream accepted')
    rng = random.Random(813)
    for _ in range(120):
        banks = {s:[] for s in (1,4,9)}
        for s in banks:
            for i in range(rng.randrange(26)):
                f = rng.choice([1,25,26,749,750,751,rng.randint(1,3950)])*rng.choice([-1,1])
                x = rng.randrange(5)
                banks[s].append([i,f,x,0,x+2,1])
        verify(banks)
    print('Synthetic intervals and 30-offset windows passed; interval [8,11] attains 2/3 and 4/3.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['self-check'])
    parser.parse_args()
    self_check()
