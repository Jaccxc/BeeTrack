"""Exact count support of the fixed209 common slate; no F1 inference."""
import argparse
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path
from exact209 import enumerate_clip, pooled


def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as stream:return json.load(stream)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def replay(bundle_path, support_path):
    support=read(support_path)
    assert support['schema']=='common209-public-window-support-v1'
    assert hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest()==support['main_bundle_sha256']
    main=read(bundle_path);lookup={(c['domain'],c['clip']):c for c in main['clips']}
    results=[]
    for clip in support['clips']:
        source=lookup[clip['domain'],clip['clip']];P=source['arms']['K']['bins']['pred'];G=source['arms']['K']['bins']['gt']
        assert len(clip['slate'])==len(set(clip['slate'])) and set(clip['slate'])<=set(r['pair_id'] for r in source['candidates'])
        sparse={r['pair_id']:r['delta'] for r in clip['active_columns']}
        assert len(sparse)==len(clip['active_columns']) and set(sparse)<=set(clip['slate'])
        columns=[]
        for key in clip['slate']:
            D=[[0,0] for _ in P];seen=set()
            for cell,delta in sparse.get(key,[]):
                assert type(cell)is int and type(delta)is int and delta!=0 and cell not in seen and 0<=cell<2*len(P)
                seen.add(cell);D[cell//2][cell%2]=delta
            columns.append(dict(pair_id=key,delta_windows=D))
        actual={a:dict(selected=source['arms'][a]['selected'],pred_bins=source['arms'][a]['bins']['pred'],
            numerator=source['arms'][a]['declared']['numerator']) for a in ('U','G','F','X')}
        exact=enumerate_clip(P,G,columns,clip['slate'],clip['k'],actual)
        original=clip['declared']
        for key in ('n','k','active_count','baseline_numerator','denominator','feasible_active_subsets','distinct_window_vectors'):
            assert exact[key]==original[key],(clip['domain'],key)
        assert exact['minimum']['numerator']==original['minimum_numerator'] and exact['maximum']['numerator']==original['maximum_numerator']
        assert exact['minimum']['vector_sha256']==original['minimum_vector_sha256']
        assert exact['maximum']['vector_sha256']==original['maximum_vector_sha256']
        labels={r['pair_id']:r for r in source['candidates']}
        actions=[dict(pair_id=key,weak=labels[key]['weak'],anyD=labels[key]['anyD'],
            selected_by_X=key in actual['X']['selected']) for key in exact['active_pair_ids']]
        results.append(dict(domain=clip['domain'],clip=clip['clip'],exact=exact,active_weak_labels=actions))
    means=pooled(results)
    for hive, mean in means.items():
        variants=[c['exact']['vectors'] for c in results if c['domain']==hive]
        total=passed=0
        for state in product(*variants):
            multiplicity=1
            for vector in state:multiplicity*=vector['active_subset_count']
            total+=multiplicity
            if 100*(mean['baseline_numerator']-sum(v['numerator'] for v in state))>=mean['denominator']:passed+=multiplicity
        mean.update(onepp_active_types=passed,total_active_types=total)
    return dict(schema=support['schema'],pooled=means,clips=results,
        scope='Fixed slate/k count support only, not J; active types are not probabilities over physical k-subsets. No F1 reconstruction for extrema.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path);parser.add_argument('support',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args();value=replay(args.bundle,args.support)
    text=json.dumps(value,sort_keys=True,separators=(',',':'))+'\n'
    if args.out:args.out.write_text(text,encoding='utf-8',newline='\n')
    else:print(text,end='')
