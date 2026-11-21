"""§187 steps 3-5: base check against §186, diagnostic repairs (甲 / 乙), full rescoring, optimistic per-window bound.
From final_project (after gtcheck187.py):
  PYTHONIOENCODING=utf-8 PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 uv run --offline --no-sync python tools/study187/repair187.py [--gt gt_check_iou0.3.json]
Writes D:/beetrack_runs/t187/results_<gt file stem>.json.
甲 (jia): real boxes kept, only the after-scene pairing becomes the GT one, on scenes whose (i) is confirmed.
乙 (yi): the same scenes, the one middle box replaced by the two bees' GT boxes (a's bee on a's chain), GT pairing.
*_samepair: the same two rewrites on every scene whose two bees are known and the same before and after (no
double-centre witness needed). Every other scene and row stays the C arm. Bound: groups of scenes that share a track
id, every legal joint option enumerated, per-cell [min, max] change (lib187.group_range / lower_bound); 'all' uses
every frozen candidate, 'open' only those where neither (i) nor (ii) is excluded.
"""
from multiprocessing import Pool
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import lib187 as X
import selfcheck187
from gtcheck187 import verify_freeze
from swap_score import event_scores

REF = {33: ([19.96, 23.99, 23.91], [0.6116, 0.5352, 0.4703]), 35: ([6.98, 5.74, 2.48], [0.8846, 0.9147, 0.9421])}
VARIANTS = ('jia', 'yi', 'jia_samepair', 'yi_samepair')


def cell(job):
    h, k, info = job
    ev = f'BEE24-{h}-head'
    poly, raw = X.P.ZONE[ev], X.P.gt('dev', ev)
    rows = np.load(X.OUT / f'base_{h}_s{k}.npy')
    S = X.labelled(rows)
    GE = X.P.events(raw, poly).astype(int)
    gc, n = X.counts(GE), len(GE)
    ref = next(c for c in X.read_json(X.SEALED / 'evidence.json')['cells'] if (c['h'], c['seed']) == (h, k))
    assert GE.tolist() == ref['gt_events']

    def score(S2):
        f, E = X.final(S2), X.events(S2, poly)
        pc = X.counts(E)
        o = event_scores(raw, f, poly)[0.5]
        return dict(W=X.W(pc, gc, n), f1=o['f1'], tp=o['tp'], fp=o['fp'], fn=o['fn'],
                    n_in=int((E[:, 1] > 0).sum()), n_out=int((E[:, 1] < 0).sum())), pc

    base, pcb = score(S)
    assert X.events(S, poly).tolist() == ref['events'], 'base events differ from §186'
    assert round(base['W'], 2) == REF[h][0][k] and round(base['f1'], 4) == REF[h][1][k], (h, k, base)
    assert X.lower_bound(pcb, gc, n, []) == base['W']                     # self-check (a) on real data
    F0 = int(rows[:, 1].max()) + 1
    scenes, by = [], {}
    for m, (c, g) in enumerate(info):
        sc = X.scene(c, (F0 + 2 * m, F0 + 2 * m + 1))
        sc['gt_mid'] = g.get('gt_mid')
        scenes.append(sc)
        by[sc['cid']] = g
    pick = dict(jia=lambda g: g['i'] == 'confirmed', samepair=lambda g: g['same_pair'])
    choices = {}
    for v in VARIANTS:
        sel = pick['samepair' if v.endswith('samepair') else 'jia']
        choices[v] = {cid: (g['pair_gt'], 'gt' if v.startswith('yi') else 'keep') for cid, g in by.items() if sel(g)}
    out = dict(h=h, seed=k, n_gt=n, n_candidates=len(scenes), base=base, variants={}, bound={}, checks={})
    pcs = dict(base=pcb)
    for v, ch in choices.items():
        r, pcs[v] = score(X.apply_all(S, scenes, ch))
        out['variants'][v] = dict(r, rewritten=sorted(ch))
    # self-check (c) on real scenes: the joint rewrite = base + sum of group-local diffs
    gs = X.groups(S, scenes)
    first = {sc['cid']: next(o for o in X.options(sc)[1:] if X.apply(S, sc, o) is not None) for sc in scenes}
    for name, ch in dict(choices, first_rewrite=first).items():
        joint = X.counts(X.events(X.apply_all(S, scenes, ch), poly))
        summed = pcb.copy()
        for g, ids in gs:
            Sl = X.local(S, ids)
            summed += X.counts(X.events(X.apply_all(Sl, g, ch), poly)) - X.counts(X.events(Sl, poly))
        assert np.array_equal(joint, summed), name
    out['checks']['joint_equals_sum'] = sorted(dict(choices, first_rewrite=first))
    out['checks']['group_sizes'] = sorted(len(g) for g, _ in gs)
    lohi = {}
    for subset in ('all', 'open'):
        keep = [sc for sc in scenes if subset == 'all' or (by[sc['cid']]['i'] != 'excluded' and by[sc['cid']]['ii'] != 'excluded')]
        ranges, states = [], []
        for g, ids in X.groups(S, keep):
            lo, hi, ns = X.group_range(S, g, ids, poly)
            ranges.append((lo, hi))
            states.append(ns)
        lb = X.lower_bound(pcb, gc, n, ranges)
        lohi[subset] = (pcb + sum((r[0] for r in ranges), 0 * pcb), pcb + sum((r[1] for r in ranges), 0 * pcb))
        out['bound'][subset] = dict(scenes=len(keep), groups=len(ranges), joint_states=states, W_lower=lb,
                                    max_gain=base['W'] - lb)
        assert base['W'] >= lb - 1e-9
        for v in VARIANTS:                     # every diagnostic is one of the enumerated legal options
            if subset == 'all':
                assert out['variants'][v]['W'] >= lb - 1e-9, (v, out['variants'][v]['W'], lb)
    return out, pcs, gc, lohi


def main():
    gt_name = sys.argv[sys.argv.index('--gt') + 1] if '--gt' in sys.argv else 'gt_check.json'
    freeze_sha = verify_freeze()
    gtc = X.read_json(X.OUT / gt_name)
    assert gtc['freeze_sha256'] == freeze_sha, 'gt check was made on another freeze'
    synthetic = selfcheck187.run()                                  # self-checks (a), (b), (c) on synthetic scenes
    cands = {c['cid']: c for c in X.read_json(X.OUT / 'candidates.json') if c['status'] == 'ok'}
    jobs = [(h, k, [(cands[g['cid']], g) for g in gtc['candidates'] if (g['h'], g['seed']) == (h, k)]) for h, k in X.CELLS]
    with Pool(6) as pool:
        res = pool.map(cell, jobs)
    report = dict(gt_file=gt_name, gt_sha256=X.sha(X.OUT / gt_name), freeze_sha256=freeze_sha, iou=gtc['iou'],
                  synthetic_selfcheck=synthetic, cells=[r[0] for r in res], heads={})
    for h in (33, 35):
        R = [r for r in res if r[0]['h'] == h]
        n, gc = R[0][0]['n_gt'], R[0][2]
        e3 = lambda pcs: float(100 * np.abs(sum(pcs) - 3 * gc).sum() / (90 * n))
        hd = dict(Y=gtc['heads'][str(h)]['Y'], U=gtc['heads'][str(h)]['U'], scenes=gtc['heads'][str(h)]['scenes'],
                  base=dict(W=[r[0]['base']['W'] for r in R], f1=[r[0]['base']['f1'] for r in R],
                            E3=e3([r[1]['base'] for r in R])))
        hd['base']['mean_W'] = float(np.mean(hd['base']['W']))
        for v in VARIANTS:
            Ws, f1 = [r[0]['variants'][v]['W'] for r in R], [r[0]['variants'][v]['f1'] for r in R]
            hd[v] = dict(W=Ws, f1=f1, mean_W=float(np.mean(Ws)), E3=e3([r[1][v] for r in R]),
                         mean_gain=hd['base']['mean_W'] - float(np.mean(Ws)),
                         seeds_W_better=sum(a < b - 1e-9 for a, b in zip(Ws, hd['base']['W'])),
                         f1_never_lower=all(a >= b - 1e-12 for a, b in zip(f1, hd['base']['f1'])),
                         in_out=[(r[0]['variants'][v]['n_in'], r[0]['variants'][v]['n_out']) for r in R])
        for subset in ('all', 'open'):
            lbs = [r[0]['bound'][subset]['W_lower'] for r in R]
            lo, hi = sum(r[3][subset][0] for r in R), sum(r[3][subset][1] for r in R)
            e3lb = float(100 * np.maximum(0, np.maximum(lo - 3 * gc, 3 * gc - hi)).sum() / (90 * n))
            hd['bound_' + subset] = dict(W_lower=lbs, mean_W_lower=float(np.mean(lbs)),
                                         max_mean_gain=hd['base']['mean_W'] - float(np.mean(lbs)),
                                         E3_lower=e3lb, E3_max_gain=hd['base']['E3'] - e3lb,
                                         allows_2pp=hd['base']['mean_W'] - float(np.mean(lbs)) >= 2 - 1e-9)
        report['heads'][str(h)] = hd
    dest = X.OUT / f'results_{Path(gt_name).stem}.json'
    X.write_json(dest, report)
    for h, hd in report['heads'].items():
        print(f'--- {h}-head  scenes {hd["scenes"]}  Y {hd["Y"]}  U {hd["U"]}')
        print('base', [round(w, 2) for w in hd['base']['W']], round(hd['base']['mean_W'], 2), 'E3', round(hd['base']['E3'], 2),
              'F1', [round(f, 4) for f in hd['base']['f1']])
        for v in VARIANTS:
            d = hd[v]
            print(v, [round(w, 2) for w in d['W']], round(d['mean_W'], 2), 'E3', round(d['E3'], 2), 'F1',
                  [round(f, 4) for f in d['f1']], 'gain', round(d['mean_gain'], 3), 'F1 ok', d['f1_never_lower'], d['in_out'])
        for s in ('all', 'open'):
            b = hd['bound_' + s]
            print('bound', s, [round(w, 2) for w in b['W_lower']], 'mean', round(b['mean_W_lower'], 2), 'max gain',
                  round(b['max_mean_gain'], 3), 'E3 lb', round(b['E3_lower'], 2), '2pp possible', b['allows_2pp'])
    for c in report['cells']:
        print(c['h'], c['seed'], 'groups', c['checks']['group_sizes'], {s: c['bound'][s]['joint_states'] for s in c['bound']})
    print(dest)


if __name__ == '__main__':
    main()
