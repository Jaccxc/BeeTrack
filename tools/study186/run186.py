"""Section 186. Freeze all six native MOTs before GT; CPU, dev 33/35 seed 0..2 only.
Run from final_project: uv run python tools/study186/run186.py
  --prereg-sha SHA --out D:/beetrack_runs/endpoints186 [--reference historical-eval.json]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from beetrack import p0 as P
from beetrack import evaluate as V
from replay186 import counts, summarize, selfcheck
sys.path.insert(0, str(Path.cwd()/'tools'/'study103'))
from swap_score import event_scores


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prereg-sha', required=True)
    p.add_argument('--out', type=Path, default=Path('D:/beetrack_runs/endpoints186'))
    p.add_argument('--reference', type=Path)
    p.add_argument('--rounding', action='store_true')
    a = p.parse_args()
    assert len(a.prereg_sha) >= 7 and all(c in '0123456789abcdef' for c in a.prereg_sha)
    a.out.mkdir(parents=True, exist_ok=True)
    assert not (a.out/'freeze.json').exists(), 'Use a fresh output directory; never overwrite a freeze.'
    freeze = dict(prereg_sha=a.prereg_sha, model_sha256=digest(P.MODELS/'m_dev_strict.json'), cells=[])
    for h in (33,35):
        for k in (0,1,2):
            path = P.RUNS/'dets'/f'pl{h}_s{k}'/'dev'/f'BEE24-{h}-head.npy'
            D = np.load(path)
            rows = np.asarray(P.p0(P.stage1(D[D[:,5]>=0.4],3950),P.link_model('m_dev_strict')),float).reshape(-1,7)
            dest = a.out/f'c_{h}_s{k}.npy'
            np.save(dest, rows)
            freeze['cells'].append(dict(h=h, seed=k, input=str(path), input_sha256=digest(path), mot=dest.name, mot_sha256=digest(dest)))
            print(f'FROZEN {h} seed {k}', flush=True)
    write(a.out/'freeze.json', freeze)
    print('ALL SIX FROZEN; beginning GT scoring',flush=True)
    evidence = dict(prereg_sha=a.prereg_sha, freeze_sha256=digest(a.out/'freeze.json'), cells=[])
    for f in freeze['cells']:
        h,k=f['h'],f['seed']
        rows=np.load(a.out/f['mot'])
        gt=P.gt('dev',f'BEE24-{h}-head')
        poly=P.ZONE[f'BEE24-{h}-head']
        T=V.by_track(rows)
        sides=V.inside(T,poly)
        starts=np.flatnonzero(np.r_[True,T[1:,1]!=T[:-1,1]])
        ends=np.r_[starts[1:]-1,len(T)-1]
        E=P.events(rows,poly).astype(int).tolist()
        G=P.events(gt,poly).astype(int).tolist()
        c=dict(h=h,seed=k,events=E,gt_events=G,phase_counts=counts(E),
               endpoints=dict(K=len(starts),a=int(sides[starts].sum()),b=int(sides[ends].sum())))
        scores=event_scores(gt,rows,poly)
        c['official']={str(t):scores[t] for t in ((0.5,0.3) if h==33 else (0.5,))}
        m=V.match_events(gt,rows,poly)
        assert c['official']['0.5']['tp']==m['tp']
        assert abs(100*P.window_err(np.array(E).reshape(-1,2),np.array(G).reshape(-1,2),3950)/len(G)-100*sum(abs(x-y) for pp,gg in zip(counts(E),counts(G)) for x,y in zip(pp,gg))/(30*len(G)))<1e-9
        if a.rounding:
            from beetrack.common import write_mot
            dest=a.out/f'c_{h}_s{k}.txt'
            write_mot(dest,rows)
            rounded=V.load_tracks(dest)
            mm=V.match_events(gt,rounded,poly)
            c['canonical_text']=dict(sha256=digest(dest),W=100*P.window_err(P.events(rounded,poly),np.array(G),3950)/len(G),official=mm)
        evidence['cells'].append(c)
        print(f'SCORED {h} seed {k}',flush=True)
    summary=summarize(evidence['cells'])
    if a.reference:
        ref=json.loads(a.reference.read_text())['seeds 0-2']['C']
        for h in (33,35):
            r=ref[f'{h}-head']; s=summary[str(h)]
            for field in ('single','e3','bias_in','bias_out'):
                assert abs(s[field]-r[field])<1e-7,(h,field,s[field],r[field])
            for c,old in zip([c for c in evidence['cells'] if c['h']==h],r['seeds']):
                assert abs(c['W']-old)<1e-7,(h,c['seed'])
        evidence['reference']=dict(sha256=digest(a.reference),subset='seeds 0-2/C',passed=True)
    evidence['summary']=summary
    write(a.out/'evidence.json', evidence)
    write(a.out/'scored_seal.json',dict(freeze_sha256=digest(a.out/'freeze.json'),evidence_sha256=digest(a.out/'evidence.json')))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    selfcheck()
    main()
