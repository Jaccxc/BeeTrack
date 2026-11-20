"""Portable, standard-library replay of section 186 integer evidence."""
import argparse
import json


def counts(events, nf=3950):
    out = []
    for offset in range(30):
        n = (nf + offset * 25 + 749) // 750
        bins = [[0, 0] for _ in range(n)]
        for frame, sign in events:
            bins[min((frame + offset * 25 - 1) // 750, n - 1)][0 if sign > 0 else 1] += 1
        out.extend(bins)
    return out


def floor(K, a, b, gi, go):
    lo, hi = max(0, a + b - K), min(a, b)
    cost = lambda s: abs(b - s - gi) + abs(a - s - go)
    best = min(range(lo, hi + 1), key=cost)
    return dict(lo=lo, hi=hi, best_s=best, numerator=cost(best), signed=abs(b - a - gi + go))


def summarize(cells):
    result = {}
    for h in (33, 35):
        group = [c for c in cells if c['h'] == h]
        assert [c['seed'] for c in group] == [0, 1, 2]
        G = counts(group[0]['gt_events'])
        gi = sum(s > 0 for _, s in group[0]['gt_events'])
        go = len(group[0]['gt_events']) - gi
        N = gi + go
        for c in group:
            assert c['gt_events'] == group[0]['gt_events']
            p = counts(c['events'])
            assert p == c['phase_counts']
            c['W'] = 100 * sum(abs(x-y) for pp, gg in zip(p, G) for x,y in zip(pp,gg)) / (30*N)
            c['floor'] = floor(**c['endpoints'], gi=gi, go=go)
            c['floor_percent'] = 100*c['floor']['numerator']/N
            c['max_gain'] = c['W']-c['floor_percent']
        numerator = sum(abs(sum(c['phase_counts'][j][d] for c in group)-3*G[j][d]) for j in range(len(G)) for d in range(2))
        # S is the integer sum of three independently relaxed endpoint overlaps.
        lo = sum(c['floor']['lo'] for c in group)
        hi = sum(c['floor']['hi'] for c in group)
        A = sum(c['endpoints']['a'] for c in group)
        B = sum(c['endpoints']['b'] for c in group)
        best = min(range(lo, hi+1), key=lambda s: abs(B-s-3*gi)+abs(A-s-3*go))
        ef = 100*(abs(B-best-3*gi)+abs(A-best-3*go))/(3*N)
        mean = sum(c['W'] for c in group)/3
        mf = sum(c['floor_percent'] for c in group)/3
        result[str(h)] = dict(single=mean, single_floor=mf, maximum_mean_gain=mean-mf,
            mean_2pp_precluded=mf > mean-2+1e-9,
            e3=100*numerator/(90*N), e3_floor=ef, e3_s_range=[lo,hi], e3_best_s=best,
            e3_strict_target_precluded=ef >= ({35:5.05,33:20.95}[h])-1e-9,
            bias_in=sum(sum(s>0 for _,s in c['events'])-gi for c in group)/3,
            bias_out=sum(sum(s<0 for _,s in c['events'])-go for c in group)/3)
    return result


def selfcheck():
    assert floor(2,1,1,0,0)['numerator'] == 0
    assert floor(2,0,2,0,0)['numerator'] == 2
    assert floor(2,0,2,1,1)['signed'] == 2
    assert counts([[750,1],[751,-1]])[0:2] == [[1,0],[0,1]]


if __name__ == '__main__':
    selfcheck()
    p = argparse.ArgumentParser()
    p.add_argument('evidence')
    a = p.parse_args()
    data = json.load(open(a.evidence, encoding='utf-8'))
    print(json.dumps(summarize(data['cells']), indent=2))
