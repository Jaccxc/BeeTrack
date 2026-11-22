"""Independent minimum-window physical witness check, registration ec0c0e.

No GT file, image, frame matching or F1 access. Run only with sole CPU token.
Main §209 verification and its seal are never modified.
"""
import argparse
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import sys
import numpy as np
import audit209 as audit
import verify209 as independent

REGISTRATION='ec0c0ef4d761b950ad551e00941bd2974a13b716'
EXACT_SEAL='2d2ecf6ca51b8b82a90142913bc1bbe039b760908ff8c61cc31df123b47b8dd1'

def clock(events,nf):
    out=[]
    for phase in range(30):
        shift=25*phase;count=(nf+shift+749)//750
        for index in range(count):
            left=750*index-shift+1;right=750*(index+1)-shift
            out.append([sum(left<=f<=right and d==1 for f,d in events),sum(left<=f<=right and d==-1 for f,d in events)])
    return np.asarray(out,np.int64)

def run(args):
    sys.path.insert(0,str(args.repo.resolve()))
    root=args.root;folder=root/'exact_window_support';audit.seal(folder,EXACT_SEAL)
    exact=audit.read(folder/'exact.json');triangle=audit.read(root/'attainable_diagnostic.json')
    assert audit.sha(root/'attainable_diagnostic.json')=='3c3a27465e49cf1291f4c9446bdc98a7babb96d080988a6c84ee96842f31c9da'
    main=audit.read(root/'verification/independent.json');assert main['passed']
    replay=audit.module('support209_native',args.repo/'tools/study203/verify203.py');previous,native,v,cert=replay.helpers(args.repo)
    top=audit.read(root/'catalog/deployment.json');reports=[]
    for domain in ('35','33'):
        item=next(c for c in exact['clips'] if c['domain']==domain);clip=item['clip'];key=domain+'/'+clip;e=item['exact'];state=e['minimum']
        t=next(c for c in triangle['clips'] if c['domain']==domain and c['clip']==clip)
        ref=next(c for c in top['clips'] if c['domain']==domain and c['clip']==clip)
        cat=audit.read(root/'catalog'/ref['path']/'catalog.json.gz');meta=cat['metadata'];src=args.source208/meta['input208_root_relative']
        assert audit.sha(src/'rows.npy')==meta['rows_sha256'] and audit.sha(src/'canon.npy')==meta['canon_sha256']
        rows=np.load(src/'rows.npy',mmap_mode='r');canonical=np.load(src/'canon.npy',mmap_mode='r')
        selected=state['selected'];slate=t['common_slate'];assert len(selected)==len(set(selected))==e['k']==t['declared_k'] and set(selected)<=set(slate)
        pairs=[tuple(map(int,p.split('|'))) for p in selected];assert len(set(x for p in pairs for x in p))==2*len(pairs)
        materialized,retained=independent.merge_rows(rows,canonical,pairs,[tuple(p['canonical_ids']) for p in cat['pairs']])
        disk=np.loadtxt(folder/domain/clip/'min/mot.txt',delimiter=',',usecols=range(7),ndmin=2)
        assert np.array_equal(materialized,disk)
        saved=audit.read(folder/domain/clip/'min/counts.json')
        assert saved['selected']==selected and saved['retained_raw_indices']==retained
        assert saved['deleted_raw_indices']==sorted(set(range(len(rows)))-set(retained))
        events=list(native.track_structure(materialized,meta['polygon'])['events'].values())
        own=clock(events,meta['nf']);baseline=audit.read(root/'score/clips'/domain/clip/'K/score.json')['metrics']
        truth=np.asarray(baseline['gt_bins'],np.int64);K=np.asarray(baseline['pred_bins'],np.int64)
        numerator=int(abs(own-truth).sum());baseline_numerator=int(abs(K-truth).sum());denominator=int(truth.sum())
        assert np.array_equal(own,state['pred_bins']) and numerator==saved['numerator']==state['numerator']
        assert numerator==item['physical_count_checks']['min']['numerator']
        assert baseline_numerator==e['baseline_numerator'] and denominator==e['denominator']
        columns={p['pair_id']:np.asarray(p['delta_windows'],np.int64) for p in t['pairs']};additive=K.copy()
        for pid in selected:additive+=columns[pid]
        assert np.array_equal(additive,own)
        active=[pid for pid in slate if np.any(columns[pid])];zeros=len(slate)-len(active);values=[]
        for count in range(max(0,e['k']-zeros),min(e['k'],len(active))+1):
            for subset in combinations(active,count):
                bins=K.copy()
                for pid in subset:bins+=columns[pid]
                assert (bins>=0).all();values.append(int(abs(bins-truth).sum()))
        assert min(values)==numerator and max(values)==e['maximum']['numerator']
        above=sum(100*(baseline_numerator-n)>=denominator for n in values)
        reports.append(dict(domain=domain,clip=clip,n=e['n'],k=e['k'],selected=selected,raw_rows=len(rows),physical_rows=len(materialized),deleted_rows=len(rows)-len(materialized),minimum_numerator=numerator,baseline_numerator=baseline_numerator,denominator=denominator,best_gain_fraction=[100*(baseline_numerator-numerator),denominator],best_gain_percent=float(Fraction(100*(baseline_numerator-numerator),denominator)),active_subsets=len(values),onepp_active_subsets=above,maximum_numerator=max(values),raw_minimum_physical_exact=True,selected_additive_delta_exact=True))
        print('SUPPORT209_MINIMUM_WITNESS_PASS',domain,numerator,denominator,flush=True)
    audit.save(root/'verification/window_support_summary.json',dict(passed=True,registration=REGISTRATION,exact_directory_seal=EXACT_SEAL,exact_json_sha256=audit.sha(folder/'exact.json'),triangle_sha256=audit.sha(root/'attainable_diagnostic.json'),main_independent_sha256=audit.sha(root/'verification/independent.json'),heads=reports,scope='Both head minimum witnesses rebuilt from original208 raw rows by independent merge implementation; all retained/deleted rows/date/box/score/ID equality against actual minimum MOT, quota/disjointness and native event+independent30phase integer numerators. Numeric active-support enumeration uses sealed candidate deltas; only the selected minimum sum is independently physically cross-checked here. No raw GT, pixels, Hungarian, F1, training, changed deployment or new seed.'))
    source=Path(__file__);snapshot=root/'verification/source_snapshot_support';snapshot.mkdir(exist_ok=True);(snapshot/source.name).write_bytes(source.read_bytes())
    audit.save(root/'verification/window_support_receipt.json',dict(passed=True,registration=REGISTRATION,summary_sha256=audit.sha(root/'verification/window_support_summary.json'),source_sha256={source.name:audit.sha(source),'audit209.py':audit.sha(audit.__file__),'verify209.py':audit.sha(independent.__file__)},CPU_exit_expected=0,scope='Separate supplemental receipt; original independent209 seal unchanged.'))
    print('SUPPORT209_PASS',audit.sha(root/'verification/window_support_receipt.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('D:/beetrack_runs/common209'));p.add_argument('--source208',type=Path,default=Path('D:/beetrack_runs/switch208/inputs'));p.add_argument('--field-go',action='store_true');a=p.parse_args()
    if not a.field_go:raise ValueError('explicit soleCPU permission required')
    run(a)
