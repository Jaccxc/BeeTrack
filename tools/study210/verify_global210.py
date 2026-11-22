"""Supplemental independent all-graph support/witness check, count only."""
import argparse
from itertools import combinations
from pathlib import Path
import sys
import numpy as np
import field210 as field
import audit210 as independent
prior=field.prior

SUPPLEMENT='4685f0b9702978c3b083c7a1e9fa698d54bd28ae'
GLOBAL='871389181734ff68e1b7fa7e6499681b683b9f04660f08da802ccb2fe88afbde'
FIXED='9d17f12a592d0fd022d33c25848dce4402bf47f335ecf2c16b39281926672af2'

def run(args):
    sys.path.insert(0,str(args.repo.resolve()));root=args.root
    main=prior.read(root/'verification/independent.json');assert main['passed']
    prior.seal(root/'global_capacity',GLOBAL);prior.seal(root/'capacity',FIXED)
    side=prior.read(root/'global_capacity/support.json');assert side['receipt']['supplement_sha']==SUPPLEMENT
    catalog=prior.read(root/'catalog/deployment.json');replay=prior.module('global210_independent_native',args.repo/'tools/study203/verify203.py');_,native,_,_=replay.helpers(args.repo)
    reports=[]
    for clip in side['clips']:
        domain,name=clip['domain'],clip['clip'];answer=clip['support']
        ref=next(r for r in catalog['clips'] if (r['domain'],r['clip'])==(domain,name));cat=prior.read(root/'catalog'/ref['path']/'catalog.json.gz')
        table={p['pair_id']:p for p in cat['pairs']};active=sorted([pid for pid,p in table.items() if np.any(p['delta_windows'])],key=lambda p:tuple(map(int,p.split('|'))))
        assert set(active)==set(cat['active_slate_ids'])==set(answer['premise']['global_nonzero_pair_ids'])
        assert len({c for pid in active for c in table[pid]['canonical_ids']})==2*len(active)
        assert answer['global_scope_proved'] and len(active)==answer['a']
        K=prior.read(root/'score/clips'/domain/name/'K/score.json')['metrics'];P=np.asarray(K['pred_bins'],np.int64);G=np.asarray(K['gt_bins'],np.int64)
        assert np.array_equal(P,answer['count_input']['pred_bins']) and np.array_equal(G,answer['count_input']['gt_bins'])
        assert {p['pair_id'] for p in answer['count_input']['all_pairs']}==set(table)
        for p in answer['count_input']['all_pairs']:assert np.array_equal(p['delta_windows'],table[p['pair_id']]['delta_windows'])
        states=[]
        for n in range(len(active)+1):
            for subset in combinations(active,n):
                bins=P.copy()
                for pid in subset:bins+=np.asarray(table[pid]['delta_windows'],np.int64)
                assert (bins>=0).all();states.append((int(abs(bins-G).sum()),subset,bins))
        assert len(states)==2**len(active)==answer['enumerated_active_subsets']
        best=min(states,key=lambda s:(s[0],tuple(tuple(map(int,p.split('|'))) for p in s[1])))
        worst=max(s[0] for s in states)
        assert list(best[1])==answer['best']['selected'] and best[0]==answer['best']['numerator'] and worst==answer['worst']['numerator']
        assert np.array_equal(best[2],answer['best']['pred_bins'])
        vectors={tuple(b.ravel()) for _,_,b in states};assert len(vectors)==answer['distinct_window_vectors']
        assert {tuple(np.asarray(s['pred_bins']).ravel()) for s in answer['vectors']}==vectors
        for budget in answer['by_active_budget']:
            group=[s for s in states if len(s[1])==budget['active_budget']]
            assert len(group)==budget['active_subset_count'] and min(s[0] for s in group)==budget['min_numerator'] and max(s[0] for s in group)==budget['max_numerator']
        physical=None
        if domain in ('35','33'):
            meta=cat['metadata'];source=args.source208/meta['source208_path'];rows=np.load(source/'rows.npy',mmap_mode='r');canon=np.load(source/'canon.npy',mmap_mode='r')
            assert prior.sha(source/'rows.npy')==meta['rows_sha256'] and prior.sha(source/'canon.npy')==meta['canon_sha256']
            merged,indices=independent.merge(rows,canon,[tuple(map(int,p.split('|'))) for p in best[1]],return_indices=True)
            assert np.array_equal(merged[:,[0,2,3,4,5,6]],rows[indices][:,[0,2,3,4,5,6]]) and sorted(indices)==list(range(len(rows)))
            events=list(native.track_structure(merged,meta['polygon'])['events'].values());bins=independent.clock(events,meta['nf'])
            proof=clip['physical_count_checks']['best']
            assert proof['output_raw_indices']==indices and sorted(map(tuple,proof['events']))==sorted(map(tuple,events))
            assert np.array_equal(bins,best[2]) and int(abs(bins-G).sum())==proof['numerator']==best[0]
            physical=dict(all_original_rows=len(rows),rows_preserved=True,selected=list(best[1]),native_event_clock_exact=True,canonical_matrix_sha256=prior.hashlib.sha256(np.ascontiguousarray(merged,dtype='<f8').tobytes()).hexdigest())
        k=(len(active)+3)//4 if active else 0;fixed=[s[0] for s in states if len(s[1])==k]
        reports.append(dict(domain=domain,clip=name,candidates=len(table),nonzero_actions=len(active),inert_actions=len(table)-len(active),all_nonzero_equal_active_slate=True,nonzero_endpoint_disjoint=True,allgraph_count_coverage=True,subsets=len(states),window_vectors=len(vectors),minimum_numerator=best[0],maximum_numerator=worst,baseline_numerator=int(abs(P-G).sum()),denominator=int(G.sum()),best_selected=list(best[1]),original_k=k,fixed_k_minimum=min(fixed),fixed_k_maximum=max(fixed),physical_best=physical))
        print('GLOBAL210_INDEPENDENT_CLIP_PASS',domain,name,flush=True)
    prior.save(root/'verification/global_support.json',dict(passed=True,supplement_sha=SUPPLEMENT,main_independent_sha256=prior.sha(root/'verification/independent.json'),capacity_seal=FIXED,global_capacity_seal=GLOBAL,clips=reports,scope='All six frozen candidate nonzero-column sets and endpoint-disjointness, complete powersets/by-budget/window-vector support independently checked. Full-graph count coverage: remove zero-delta actions from any legal one-shot matching without changing windows; each remaining nonzero subset is itself legal. All local delta columns were independently physically rebuilt in main210. Two head best witnesses additionally materialized from original raw rows, all-row bijection and complete native event/independent30clock matched count proof. No new GT rectangle matching/F1; no iterative/shared-run/multiway edits or out-of-catalog inference.'))
    source=Path(__file__);snapshot=root/'verification/source_snapshot_global';snapshot.mkdir(exist_ok=True);(snapshot/source.name).write_bytes(source.read_bytes())
    prior.save(root/'verification/global_support_receipt.json',dict(passed=True,supplement_sha=SUPPLEMENT,summary_sha256=prior.sha(root/'verification/global_support.json'),source_sha256={source.name:prior.sha(source),'audit210.py':prior.sha(independent.__file__),'field210.py':prior.sha(field.__file__)},scope='Separate count-only supplemental audit; original main seal unchanged.'))
    print('GLOBAL210_INDEPENDENT_PASS',prior.sha(root/'verification/global_support_receipt.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('D:/beetrack_runs/interleave210'));p.add_argument('--source208',type=Path,default=Path('D:/beetrack_runs/switch208/inputs'));p.add_argument('--field-go',action='store_true');a=p.parse_args()
    if not a.field_go:raise ValueError('soleCPU permission required')
    run(a)
