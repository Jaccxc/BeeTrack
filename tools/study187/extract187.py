"""§187 step 1, no GT: rebuild the §186 C arm (must equal its sealed MOT), extract 2->1->2 candidates, freeze them.
From final_project:
  PYTHONIOENCODING=utf-8 PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 uv run --offline --no-sync python tools/study187/extract187.py
Writes D:/beetrack_runs/t187/: base_<h>_s<k>.npy (linked rows before drop_short), piece_<h>_s<k>.npy (stage-1 ids),
candidates.json, components.json, tally.json, then freeze.json with their SHA-256 and the six sealed MOT hashes.
"""
from multiprocessing import Pool
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import lib187 as X


def cell(hk):
    h, k = hk
    rows, piece = X.build(h, k)
    assert np.array_equal(X.final(rows), np.load(X.SEALED / f'c_{h}_s{k}.npy')), (h, k, 'rebuild differs from §186 seal')
    np.save(X.OUT / f'base_{h}_s{k}.npy', rows)
    np.save(X.OUT / f'piece_{h}_s{k}.npy', piece)
    found, tally, L0 = X.extract(rows, piece)
    print(f'EXTRACTED {h} seed {k}: {tally}', flush=True)
    return h, k, found, tally, L0


def main():
    X.OUT.mkdir(parents=True, exist_ok=True)
    assert not (X.OUT / 'freeze.json').exists(), 'freeze.json exists; never overwrite a freeze'
    seal = X.read_json(X.SEALED / 'freeze.json')
    for c in seal['cells']:
        assert X.sha(X.SEALED / c['mot']) == c['mot_sha256'], c['mot']
        assert X.sha(X.P.RUNS / 'dets' / f"pl{c['h']}_s{c['seed']}" / 'dev' / f"BEE24-{c['h']}-head.npy") == c['input_sha256']
    with Pool(6) as pool:
        results = pool.map(cell, X.CELLS)
    cands, comps, tallies = [], {}, {}
    for h, k, found, tally, L0 in results:
        rows, piece = np.load(X.OUT / f'base_{h}_s{k}.npy'), np.load(X.OUT / f'piece_{h}_s{k}.npy')
        for n, c in enumerate(found):
            cid = f'{h}_s{k}_{n:03d}'
            cands.append(dict(cid=cid, h=h, seed=k, **c))
            comps[cid] = [[t, int(rows[r, 1]), int(piece[r]), *map(float, rows[r, 2:6])] for t, v in c['comp'].items() for r in v]
        tallies[f'{h}_s{k}'] = dict(tally, L0=L0)
    X.write_json(X.OUT / 'candidates.json', cands)
    X.write_json(X.OUT / 'components.json', comps)
    X.write_json(X.OUT / 'tally.json', tallies)
    names = ['candidates.json', 'components.json', 'tally.json'] + \
            [f'{w}_{h}_s{k}.npy' for h, k in X.CELLS for w in ('base', 'piece')]
    X.write_json(X.OUT / 'freeze.json', dict(
        rule=dict(max_mid=X.MAX_MID, near_body_lengths=X.NEAR, separated_overlap=X.SEP, side_frames=X.SIDE),
        files={n: X.sha(X.OUT / n) for n in names},
        sealed_mot={c['mot']: X.sha(X.SEALED / c['mot']) for c in seal['cells']},
        source={n: X.sha(Path(__file__).parent / n) for n in ('lib187.py', 'extract187.py')}))
    print('FROZEN', len(cands), 'candidates;', sum(c['status'] == 'ok' for c in cands), 'ok', flush=True)


if __name__ == '__main__':
    main()
