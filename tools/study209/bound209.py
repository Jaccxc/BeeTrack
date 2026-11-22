"""Posthoc count-capacity sidecar verification; no GT arrays or new pairing.

Run only after main independent209 exits, under the same sole CPU grant.
Uses original local-run rows and proven public event/window primitives, not
the producer attainable helper. Does not change a selection or a gate.
"""
from fractions import Fraction
from pathlib import Path
import argparse
import numpy as np
import audit209 as audit
import verify209 as independent

def run(args):
    import sys
    sys.path.insert(0,str(args.repo.resolve()))
    root=args.root;main=audit.read(root/'verification/independent.json');assert main['passed']
    path=root/'attainable_diagnostic.json';assert audit.sha(path)=='3c3a27465e49cf1291f4c9446bdc98a7babb96d080988a6c84ee96842f31c9da'
    side=audit.read(path);top=audit.read(root/'catalog/deployment.json');pred=audit.read(root/'predictions/predictions.json')
    replay=audit.module('bound209_events',args.repo/'tools/study203/verify203.py');previous,native,v,cert=replay.helpers(args.repo)
    reports=[]
    for item in side['clips']:
        domain,clip=item['domain'],item['clip'];key=domain+'/'+clip
        ref=next(r for r in top['clips'] if (r['domain'],r['clip'])==(domain,clip))
        cat=audit.read(root/'catalog'/ref['path']/'catalog.json.gz');meta=cat['metadata'];folder=args.source208/meta['input208_root_relative']
        rows=np.load(folder/'rows.npy',mmap_mode='r');canon=np.load(folder/'canon.npy',mmap_mode='r')
        own_deltas={};norms={};details={p['pair_id']:p for p in item['pairs']}
        assert item['common_slate']==pred['clips'][key]['slate'] and item['declared_k']==pred['clips'][key]['k']
        for pid in item['common_slate']:
            a,b=map(int,pid.split('|'));ix=np.flatnonzero(np.isin(canon,[a,b]));original=np.asarray(rows[ix]);cc=canon[ix]
            merged,retained=independent.merge_rows(original,cc,[(a,b)])
            before=native.track_structure(original,meta['polygon']);after=native.track_structure(merged,meta['polygon'])
            delta=replay.bins(after['events'].values(),meta['nf'])-replay.bins(before['events'].values(),meta['nf'])
            saved=details[pid];assert np.array_equal(delta,saved['delta_windows'])
            assert saved['retained_raw_indices']==ix[retained].tolist()
            deleted=sorted(set(ix.tolist())-set(ix[retained].tolist()));assert deleted==saved['deleted_raw_indices']
            norm=int(abs(delta).sum());assert norm==saved['delta_l1'] and bool(np.any(delta))==saved['changes_any_window']
            own_deltas[pid]=delta;norms[pid]=norm
        ordered=sorted(norms,key=lambda pid:(-norms[pid],tuple(map(int,pid.split('|')))))
        # Tie ordering in the declared bound does not affect any prefix sum.
        prefixes=[sum(sorted(norms.values(),reverse=True)[:k]) for k in range(len(norms)+1)]
        k=item['declared_k'];bound=prefixes[k];assert bound==item['topk_l1_upper']
        assert max(norms.values(),default=0)==item['max_single_pair_l1']
        denominator=item['normalization_denominator'];assert denominator==30*item['GTevents']
        fraction=Fraction(100*bound,denominator) if denominator else None
        if fraction is not None:assert fraction==Fraction(*item['W_gain_upper_fraction'])
        baseline=audit.read(root/'score/clips'/domain/clip/'K/score.json')['metrics']['pred_bins']
        for arm in ('U','G','F','X'):
            chosen=pred['clips'][key]['selected'][arm]
            total=np.zeros_like(np.asarray(baseline))
            for pid in chosen:total+=own_deltas[pid]
            actual=audit.read(root/'score/clips'/domain/clip/arm/'score.json')['metrics']['pred_bins']
            assert np.array_equal(np.asarray(baseline)+total,actual)
        reports.append(dict(clip=key,slate=len(norms),changed=sum(n>0 for n in norms.values()),k=k,topk_L1=bound,denominator=denominator,W_gain_upper_fraction=None if fraction is None else [fraction.numerator,fraction.denominator],all_four_selected_delta_closure=True))
    audit.save(root/'verification/attainable.json',dict(passed=True,posthoc=True,main_independent_sha256=audit.sha(root/'verification/independent.json'),producer_sidecar_sha256=audit.sha(path),clips=reports,scope='All common-slate local physical merges independently rebuild event/window deltas, original retained/deleted rows, sortedL1 necessary count bound and U/G/F/X actual subset closure. Not J, not F1, no sufficiency/attainability claim, no GT arrays or Hungarian pairing.'))
    print('VERIFY209_ATTAINABLE_PASS',audit.sha(root/'verification/attainable.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('D:/beetrack_runs/common209'));p.add_argument('--source208',type=Path,default=Path('D:/beetrack_runs/switch208/inputs'));p.add_argument('--field-go',action='store_true');a=p.parse_args()
    if not a.field_go:raise ValueError('sole CPU field permission required')
    run(a)
