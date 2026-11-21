"""§187 step 2: GT answers for the frozen 2->1->2 candidates. Refuses to run unless every freeze.json hash matches.
From final_project:
  PYTHONIOENCODING=utf-8 PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 uv run --offline --no-sync python tools/study187/gtcheck187.py [--iou 0.3]
Writes D:/beetrack_runs/t187/gt_check.json (--iou 0.5, the official frame assignment: primary) or gt_check_iou0.3.json
(secondary, chosen after seeing that about half of all detector boxes miss GT at 0.5; 0.3 is §186's dense-scene F1).
A side box's bee = the GT id the official one-to-one frame assignment (beetrack.evaluate.frame_match) puts under it.
(i)  same two bees before / after and the middle box covered both annotated centres at least once;
(ii) some legal rewrite of this scene alone changes the whole-track event stream.
Each is confirmed / excluded / undeterminable. Scenes = candidates of one head merged across seeds by frame range
(+-2 frames) and place (mean middle centres within 2 body lengths).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import lib187 as X
from beetrack.common import gt_slice, split_specs
from beetrack.evaluate import by_track, frame_match


def verify_freeze():
    fr = X.read_json(X.OUT / 'freeze.json')
    for name, h in fr['files'].items():
        assert X.sha(X.OUT / name) == h, f'{name} changed after the freeze'
    for name, h in fr['sealed_mot'].items():
        assert X.sha(X.SEALED / name) == h, f'{name} changed after the freeze'
    return X.sha(X.OUT / 'freeze.json')


def gt(h):
    """Eval GT in by_track ids, and which of its rows are ambiguous: prep gave a fresh id where the source GT used one
    id twice in a frame, so every row of such a (frame, id) is flagged (both bees' identity is uncertain)."""
    ev = f'BEE24-{h}-head'
    raw = X.P.gt('dev', ev)
    G = by_track(raw)
    src, a, b = next((s, a, b) for e, s, a, b in split_specs('dev') if e == ev)
    g = gt_slice(src, a, b)
    assert len(g) == len(G)
    _, inv, cnt = np.unique(g[:, 0] * 1e6 + g[:, 1], return_inverse=True, return_counts=True)
    key = lambda r: (int(r[0]),) + tuple(f'{v:.2f}' for v in r[2:6])
    amb = {key(r) for r in g[cnt[inv] > 1]}
    flag = np.array([key(r) in amb for r in G])
    assert flag.sum() == (cnt[inv] > 1).sum()
    return raw, G, flag


def side_id(xs, rows, under, flagged):
    ids = {int(under[x]) for x in xs}
    if -1 in ids:
        return None, 'a side box has no GT box at the IoU threshold'
    if len(ids) > 1:
        return None, 'the GT id under a side box changes between its two frames'
    g = ids.pop()
    if any(flagged.get((g, int(rows[x, 0])), False) for x in xs):
        return None, 'the GT id is one the source used twice in a frame'
    return g, ''


def answer_i(sc, rows, under, G, flagged, span, box):
    (ga, ra), (gb, rb), (gc, rc), (gd, rd) = (side_id(sc[s], rows, under, flagged) for s in 'abcd')
    known = None not in (ga, gb) and ga != gb
    out = dict(ga=ga, gb=gb, gc=gc, gd=gd, same_pair=False, pair_gt=None, known_before=known)
    if None in (ga, gb, gc, gd):
        return dict(out, i='undeterminable', why=ra or rb or rc or rd)
    if ga == gb or gc == gd:
        return dict(out, i='excluded', why='one GT bee under both boxes of one side')
    if {ga, gb} != {gc, gd}:
        handover = any(span[g][1] <= sc['t1'] + 2 for g in {ga, gb} - {gc, gd}) or \
                   any(span[g][0] >= sc['t0'] - 2 for g in {gc, gd} - {ga, gb})
        return dict(out, i='undeterminable' if handover else 'excluded',
                    why='GT ids change inside the scene' if handover else 'different bees before and after')
    out.update(same_pair=True, pair_gt=0 if gc == ga else 1)
    witness, missing = [], False
    for t, m in zip(range(sc['t0'], sc['t1'] + 1), sc['mid']):
        x, y, w, h = rows[m, 2:6]
        hits = []
        for g in (ga, gb):
            r = box.get((g, t))
            if r is None or flagged.get((g, t), False):
                missing = True
                hits.append(False)
                continue
            cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
            hits.append(x <= cx <= x + w and y <= cy <= y + h)
        if all(hits):
            witness.append(t)
    if witness:
        return dict(out, i='confirmed', why=f'middle box covers both centres in frames {witness}')
    if not missing:
        return dict(out, i='excluded', why='both bees annotated in every middle frame, never both under the box')
    return dict(out, i='undeterminable', why='a second centre is missing in a middle frame, no double cover seen')


def main():
    iou = float(sys.argv[sys.argv.index('--iou') + 1]) if '--iou' in sys.argv else 0.5
    freeze_sha = verify_freeze()
    cands = [c for c in X.read_json(X.OUT / 'candidates.json') if c['status'] == 'ok']
    report = dict(freeze_sha256=freeze_sha, iou=iou, candidates=[], heads={})
    for h in (33, 35):
        raw, G, flag = gt(h)
        flagged = {(int(r[1]), int(r[0])): bool(f) for r, f in zip(G, flag)}
        box = {(int(r[1]), int(r[0])): r[2:6] for r in G}
        ids, first = np.unique(G[:, 1], return_index=True)
        last = np.r_[first[1:], len(G)] - 1
        span = {int(i): (int(G[a, 0]), int(G[b, 0])) for i, a, b in zip(ids, first, last)}
        poly = X.P.ZONE[f'BEE24-{h}-head']
        mine = []
        for k in (0, 1, 2):
            rows = np.load(X.OUT / f'base_{h}_s{k}.npy')
            S = X.labelled(rows)
            under = frame_match(G, rows[:, :7], iou)
            F0 = int(rows[:, 1].max()) + 1
            cell = [c for c in cands if c['h'] == h and c['seed'] == k]
            for n, c in enumerate(cell):
                sc = X.scene(c, (F0 + 2 * n, F0 + 2 * n + 1))
                r = answer_i(sc, rows, under, G, flagged, span, box)
                if r['known_before']:
                    r['gt_mid'] = [[t, who, *map(float, box[(g, t)])] for t in range(sc['t0'], sc['t1'] + 1)
                                   for who, g in (('a', r['ga']), ('b', r['gb']))
                                   if (g, t) in box and not flagged[(g, t)]]
                sc['gt_mid'] = r.get('gt_mid')
                Sl = X.local(S, X.touched(S, sc))
                E0 = sorted(map(tuple, X.events(Sl, poly).tolist()))
                changing = []
                for o in X.options(sc)[1:]:
                    U = X.apply(Sl, sc, o)
                    if U is not None and sorted(map(tuple, X.events(U, poly).tolist())) != E0:
                        changing.append(o)
                r['ii'] = 'confirmed' if changing else 'excluded' if sc['gt_mid'] else 'undeterminable'
                r['ii_options_changing'] = changing
                centre = np.mean([rows[m, 2:4] + rows[m, 4:6] / 2 for m in sc['mid']], 0).tolist()
                mine.append(dict(cid=c['cid'], h=h, seed=k, t0=c['t0'], t1=c['t1'], L=c['L'], centre=centre, **r))
        # one physical scene seen by several detector seeds counts once
        parent = list(range(len(mine)))

        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i
        for i, p in enumerate(mine):
            for j, q in enumerate(mine[:i]):
                close = np.hypot(*np.subtract(p['centre'], q['centre'])) <= X.NEAR * (p['L'] + q['L']) / 2
                if p['seed'] != q['seed'] and max(p['t0'], q['t0']) - 2 <= min(p['t1'], q['t1']) + 2 and close:
                    parent[root(i)] = root(j)
        scenes = {}
        for i, p in enumerate(mine):
            p['scene'] = f'{h}_{root(i):03d}'
            scenes.setdefault(p['scene'], []).append(p)
        both = lambda p: p['i'] == 'confirmed' and p['ii'] == 'confirmed'
        open_ = lambda p: p['i'] != 'excluded' and p['ii'] != 'excluded'
        Y = sorted(s for s, ps in scenes.items() if any(both(p) for p in ps))
        U = sorted(s for s, ps in scenes.items() if s not in Y and any(open_(p) for p in ps))
        report['heads'][str(h)] = dict(candidates_per_seed=[sum(p['seed'] == k for p in mine) for k in (0, 1, 2)],
                                       scenes=len(scenes), Y=len(Y), U=len(U), Y_scenes=Y, U_scenes=U,
                                       excluded=len(scenes) - len(Y) - len(U))
        report['candidates'] += mine
        print(h, report['heads'][str(h)], flush=True)
    X.write_json(X.OUT / ('gt_check.json' if iou == 0.5 else f'gt_check_iou{iou}.json'), report)
    for p in report['candidates']:
        print(p['cid'], p['scene'], p['t0'], p['t1'], 'i', p['i'], '|', p['why'], '| ii', p['ii'], p['ii_options_changing'])


if __name__ == '__main__':
    main()
