"""Synthetic audit for the seven-frame study. All inputs are generated here."""
from collections import Counter, defaultdict
from itertools import product
from random import Random

import numpy as np

from capacity import (analyze, append_witnesses, categories, minimum_error,
                      observations, options, physical_metrics)
from beetrack.evaluate import match_events
from tools.study104.event_overlap import score_arm


POLYGON = [(0,0),(20,0),(20,20),(0,20)]


def box(t, x, y=4, tid=1):
    return [t,tid,x-1,y-1,2,2]


def occurrences(times, sides, n):
    """Independently enumerate seen real frames and all two-observation candidates."""
    flips = [i for i in range(1,len(times)) if sides[i-1]!=sides[i]]
    if sides[0]==sides[-1]:
        return None
    canonical=flips[-1]
    target=times[canonical]
    pairs=set()
    for i,j in product(range(len(times)),repeat=2):
        if (4 <= times[j] <= n-3 and 1 <= times[j]-times[i] <= 3
                and abs(times[j]-target) <= 5 and sides[i] == sides[0] and sides[j] == sides[-1]):
            pairs.add((times[i],times[j], j==canonical and i==j-1))
    return target,pairs


def check_rosters():
    n=22
    counters=Counter()
    for times in ((1,3,4,5,6,10,14,18,19,20,22),(1,2,4,16,17,21,22),(4,5,8,9,10,12)):
        for sides in product((False,True),repeat=len(times)):
            rows=np.array([box(t,0 if side else -3) for t,side in zip(times,sides)],float)
            roster,flips=observations(rows, POLYGON,n)
            expectation=occurrences(times,sides,n)
            assert [r["t"] for r in flips]==[t for i,t in enumerate(times) if i and sides[i-1]!=sides[i]]
            if expectation is None:
                assert not roster
            else:
                expected_t,expected_pairs=expectation
                assert set(roster)=={1} and roster[1]["t"]==expected_t
                witnesses=roster[1]["witnesses"]
                actual={(z["u"],z["t"],z["canonical"]) for z in witnesses}
                assert actual==expected_pairs
                for family in ("canonical","tolerant"):
                    expected=defaultdict(list)
                    for u,t,is_last in expected_pairs:
                        if family=="tolerant" or is_last:
                            expected[t].append((int(is_last),u))
                    answer=options(roster,[1],family)
                    assert [(w["t"],w["u"]) for w in answer]==[(t,max(v)[1]) for t,v in sorted(expected.items())]
            counters["cases"]+=1
    # >25 frames splits; equality at 25 does not. No fabricated intervening box.
    for gap,expected_ids in [(25,set()),(26,{1,100001})]:
        clocks=[5,6,6+gap,7+gap]
        rows=np.array([box(t,x) for t,x in zip(clocks,[-3,3,3,-3])])
        roster,_=observations(rows,POLYGON,50)
        assert set(roster)==expected_ids
    return counters["cases"]


def cells(times):
    by_cell=Counter()
    for f in times:
        for origin in range(0,750,25):
            by_cell[(origin,(abs(f)-1+origin)//750, f<0)]+=1
    return by_cell


def cost(times,truth):
    state=cells(times)
    state.subtract(cells(truth))
    return sum(map(abs,state.values()))


def check_best():
    cases=[([], [750,751], [[750,751]]),([], [750,751], [[750],[751]]),
           ([1,-1000],[1,1,-1000],[[1,2],[1,2],[-999,-1000]]),
           ([],[1],[]),([],[],[]),([-25],[25],[[25]]),([24,26],[25,26],[[25,26]])]
    rng=Random(119)
    boundary=[1,2,24,25,26,27,49,50,51,724,725,726,749,750,751,752,975,976,1000]
    timestamp=lambda s:s*rng.choice(boundary)
    for i in range(40):
        base=[timestamp(rng.choice([-1,1])) for _ in range(rng.randrange(5))]
        truth=[timestamp(rng.choice([-1,1])) for _ in range(rng.randrange(1,5))]
        groups=[]
        for _ in range(rng.randrange(1,5)):
            s=rng.choice([-1,1])
            groups.append(sorted({timestamp(s) for _ in range(rng.randrange(1,4))}))
        cases.append((base,truth,groups))
    for _ in range(25):
        truth=[timestamp(rng.choice([-1,1])) for _ in range(rng.randrange(1,6))]
        base=[timestamp(rng.choice([-1,1])) for _ in range(rng.randrange(0,6))]
        candidate=[f for f in truth if rng.random()<.8]
        cases.append((base,truth,[[v] for v in candidate]))
        jittered=[]
        for v in candidate:
            sign=1 if v>0 else -1
            alternates={v, sign*max(1,min(1000,abs(v)+rng.randint(-5,5)))}
            jittered.append(sorted(alternates))
        cases.append((base,truth,jittered))
    for base,truth,groups in cases:
        opts=[dict(id=i,signed_t=z) for i,l in enumerate(groups) for z in l]
        try:
            solution=minimum_error(base,truth,opts,1000)
        except RuntimeError as exc:
            raise AssertionError((base,truth,opts)) from exc
        candidates=product(*[[None,*g] for g in groups])
        minimum=min(cost(base+[v for v in choice if v is not None],truth) for choice in candidates)
        assert (solution["min_numerator"],solution["base_numerator"],solution["denominator"])==(minimum,cost(base,truth),30*len(truth))
        selected=[opts[i] for i in solution["selected"]]
        assert len(selected)==len({o["id"] for o in selected})
        assert cost(base+[o["signed_t"] for o in selected],truth)==minimum
    assert minimum_error([], [750,751], [dict(id=0,signed_t=z) for z in (750,751)],1000)["min_numerator"]==30
    return len(cases)


def check_competition():
    # Native singleton reports keep the two perfect detections without creating an event.
    # The raw-frame tie selects the earlier native ID over an exact same-bee appended trace.
    gt=np.array([box(10,-2),box(11,2)],float)
    p0=np.array([box(10,-2,tid=4),box(11,2,tid=5)],float)
    result=analyze(gt,p0,POLYGON,20)
    bound=result["bounds"]["canonical_no_edge"]
    assert (bound["base_numerator"],bound["min_numerator"])==(30,0)
    physical=bound["official_f1_of_this_witness"]["0.5"]
    assert physical["tp"]==0 and physical["added_with_own_edge"]==0
    # A single native crossing can have an edge to both synchronous GT bees.
    gt=np.array([box(10,-2,4,1),box(11,2,4,1),box(10,-2,14,2),box(11,2,14,2)],float)
    p0=np.array([box(10,-2,4,10),box(11,2,14,10)],float)
    native=score_arm(gt,p0,POLYGON,.5)
    assert set(map(tuple,native.edges))=={(10,1),(10,2)}
    assert set(categories(native).values())=={"matched","edge_unmatched"}
    assert native.variable_gt=={1,2}
    missed=[g for g,type_ in categories(native).items() if type_=="edge_unmatched"]
    roster,_=observations(gt,POLYGON,20)
    chosen=options(roster,missed,"canonical")
    union,mapping=append_witnesses(p0,chosen)
    truth=match_events(gt,union,POLYGON)
    measured=physical_metrics(gt=gt,native=native,loose_native=score_arm(gt,p0,POLYGON,.3),
                              extra=union,mapping=mapping,poly=POLYGON)
    assert measured["0.5"]["tp"]==truth["tp"]
    # Both GT arrivals occupy the same cell at all 30 window origins. Their optimal
    # W is identical, but a witness for the second bee can win both physical rows.
    gt=np.array([box(10,-2,4,1),box(11,2,4,1),box(13,-2,14,2),box(14,2,14,2)],float)
    p0=np.array([box(10,-2,4,4),box(11,2,4,5),box(7,-2,18,7),box(8,2,18,7)],float)
    native=score_arm(gt,p0,POLYGON,.5)
    roster,_=observations(gt,POLYGON,20)
    assert categories(native)=={1:"no_edge",2:"no_edge"}
    profiles=[]
    for g in (1,2):
        choice=options(roster,[g],"canonical")
        union,mapping=append_witnesses(p0,choice)
        window_cost=cost([8,choice[0]["signed_t"]],[11,14])
        official=physical_metrics(gt=gt,native=native,loose_native=score_arm(gt,p0,POLYGON,.3),
                                  extra=union,mapping=mapping,poly=POLYGON)
        profiles.append([window_cost,official["0.5"]["f1"]])
    assert profiles==[[0,0.],[0,.5]]
    return physical,truth,profiles


if __name__=="__main__":
    print("Boundary/return rosters",check_rosters())
    print("Exhaustive finite-window oracles",check_best())
    print("Competing physical assignments",check_competition())
