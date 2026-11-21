"""§187 post-hoc sensitivity: what if the candidates that abstained for piece reuse could be rewritten too?
From final_project (after repair187.py):
  PYTHONIOENCODING=utf-8 PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 uv run --offline --no-sync python tools/study187/reuse187.py
Writes D:/beetrack_runs/t187/results_reuse_sensitivity.json.
Reuse candidates share stage-1 pieces, so they chain into groups of up to 23 scenes: the exact joint enumeration of
repair187 does not fit. Instead an oracle coordinate descent (it sees the GT window counts) picks, scene by scene,
the legal option that lowers the cell's W most, until a sweep changes nothing. The result is a W some legal rewrite
reaches (an achieved value, not a bound): it shows how much gain exists, it cannot prove that more does not.
Run on the frozen 'ok' scenes alone as well, where it must stay at or above repair187's exact bound.
"""
from multiprocessing import Pool
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import lib187 as X
from gtcheck187 import verify_freeze
from swap_score import event_scores


def legal(S, sc, opt):
    try:
        return X.apply(S, sc, opt)
    except IndexError:              # an earlier scene of the group dropped a box this one needs
        return None


def joint(S, scenes, choice):
    for sc in sorted(scenes, key=lambda s: s['t0']):
        S = legal(S, sc, choice.get(sc['cid']))
        if S is None:
            return None
    return S


def descend(S, scenes, poly, pcb, gc, n):
    gs = [(g, X.local(S, ids)) for g, ids in X.groups(S, scenes)]
    base = [X.counts(X.events(Sl, poly)) for _, Sl in gs]
    delta = [0 * b for b in base]
    choice, best = {}, X.W(pcb, gc, n)
    for sweep in range(20):
        moved = False
        for gi, (g, Sl) in enumerate(gs):
            for sc in sorted(g, key=lambda s: s['t0']):
                rest = pcb + sum(delta) - delta[gi]
                for o in X.options(sc):
                    trial = dict(choice, **{sc['cid']: o})
                    U = joint(Sl, g, trial)
                    if U is None:
                        continue
                    d = X.counts(X.events(U, poly)) - base[gi]
                    w = X.W(rest + d, gc, n)
                    if w < best - 1e-9:
                        best, delta[gi], choice, moved = w, d, trial, True
        if not moved:
            break
    return best, {c: o for c, o in choice.items() if o is not None}, sweep + 1


def cell(job):
    h, k, info = job
    ev = f'BEE24-{h}-head'
    poly, raw = X.P.ZONE[ev], X.P.gt('dev', ev)
    rows = np.load(X.OUT / f'base_{h}_s{k}.npy')
    S = X.labelled(rows)
    GE = X.P.events(raw, poly).astype(int)
    gc, n = X.counts(GE), len(GE)
    pcb = X.counts(X.events(S, poly))
    F0 = int(rows[:, 1].max()) + 1
    scenes = []
    for m, (c, gt_mid) in enumerate(info):
        sc = X.scene(c, (F0 + 2 * m, F0 + 2 * m + 1))
        sc['gt_mid'], sc['status'] = gt_mid, c['status']
        scenes.append(sc)
    out = dict(h=h, seed=k, base_W=X.W(pcb, gc, n), base_f1=event_scores(raw, X.final(S), poly)[0.5]['f1'])
    for name in ('ok', 'ok+reuse'):
        use = [sc for sc in scenes if name == 'ok+reuse' or sc['status'] == 'ok']
        w, ch, sweeps = descend(S, use, poly, pcb, gc, n)
        S2 = joint(S, use, ch)                          # the found joint rewrite on the full rows gives the same W
        assert S2 is not None and abs(X.W(X.counts(X.events(S2, poly)), gc, n) - w) < 1e-9, (h, k, name)
        o = event_scores(raw, X.final(S2), poly)[0.5]
        E = X.events(S2, poly)
        out[name] = dict(scenes=len(use), group_sizes=sorted((len(g) for g, _ in X.groups(S, use)), reverse=True),
                         W=w, f1=o['f1'], n_in=int((E[:, 1] > 0).sum()), n_out=int((E[:, 1] < 0).sum()),
                         rewritten={c: list(v) for c, v in sorted(ch.items())}, sweeps=sweeps)
    return out


def main():
    freeze_sha = verify_freeze()
    gtc = X.read_json(X.OUT / 'gt_check.json')
    assert gtc['freeze_sha256'] == freeze_sha
    gt_mid = {g['cid']: g.get('gt_mid') for g in gtc['candidates']}
    exact = X.read_json(X.OUT / 'results_gt_check.json')
    C = X.read_json(X.OUT / 'candidates.json')
    jobs = [(h, k, [(c, gt_mid.get(c['cid'])) for c in C if (c['h'], c['seed']) == (h, k)]) for h, k in X.CELLS]
    with Pool(6) as pool:
        res = pool.map(cell, jobs)
    report = dict(freeze_sha256=freeze_sha, cells=res, heads={})
    for h in (33, 35):
        R = [r for r in res if r['h'] == h]
        lb = exact['heads'][str(h)]['bound_all']['W_lower']
        for r, b in zip(R, lb):
            assert r['ok']['W'] >= b - 1e-9, (h, r['seed'], r['ok']['W'], b)   # the exact bound holds for the search
        base = float(np.mean([r['base_W'] for r in R]))
        hd = dict(base_mean_W=base, exact_bound_ok_mean=float(np.mean(lb)))
        for name in ('ok', 'ok+reuse'):
            Ws = [r[name]['W'] for r in R]
            hd[name] = dict(W=Ws, mean_W=float(np.mean(Ws)), found_gain=base - float(np.mean(Ws)),
                            f1=[r[name]['f1'] for r in R], f1_never_lower=all(r[name]['f1'] >= r['base_f1'] - 1e-12 for r in R))
        report['heads'][str(h)] = hd
    dest = X.OUT / 'results_reuse_sensitivity.json'
    X.write_json(dest, report)
    for r in res:
        print(r['h'], r['seed'], 'base', round(r['base_W'], 2),
              {s: (r[s]['scenes'], r[s]['group_sizes'][:3], round(r[s]['W'], 2), round(r[s]['f1'], 4), len(r[s]['rewritten'])) for s in ('ok', 'ok+reuse')})
    for h, hd in report['heads'].items():
        print(h, 'base mean', round(hd['base_mean_W'], 2), 'exact bound (ok)', round(hd['exact_bound_ok_mean'], 2),
              {s: (round(hd[s]['mean_W'], 2), 'gain', round(hd[s]['found_gain'], 3), 'F1 ok', hd[s]['f1_never_lower']) for s in ('ok', 'ok+reuse')})
    print(dest)


if __name__ == '__main__':
    main()
