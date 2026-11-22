"""Post-audit opaque full-adjacency table and timing-definition comparison."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import numpy as np
import verify208 as core
import audit208 as io


def run(args):
    root=args.root;accepted=root/'verification/independent.json';assert io.sha(accepted)==accepted.with_suffix('.sha256').read_text().strip() and io.read(accepted)['passed']
    top=io.read(root/'inputs/deployment.json');pred=io.read(root/'predictions/metadata.json');summ=io.read(root/'score/summary.json');outputs={r['domain']+'/'+r['clip']:r for r in pred['clips']};clips=[];diagnostics=[]
    for ref in top['clips']:
        key=ref['domain']+'/'+ref['clip'];folder=root/'inputs'/ref['path'];rows=np.load(folder/'rows.npy',mmap_mode='r').tolist();canon=np.load(folder/'canon.npy',mmap_mode='r').tolist();indices=np.load(folder/'indices.npy',mmap_mode='r').tolist();valid=np.load(folder/'valid.npy',mmap_mode='r').tolist();owners=np.load(root/'labels'/ref['path']/'owners.npy')
        runs,_=core.candidate_table(rows,canon,indices);position={raw:j for parent,ii in runs.items() for j,raw in enumerate(ii)};census=core.adjacent_census(rows,canon,indices,valid,owners,ref['clip'])
        assert len(census)==sum(len(ii)-1 for ii in runs.values())
        target=root/'verification/truth'/ref['path'];target.mkdir(parents=True,exist_ok=True);truthfile=target/'adjacency.jsonl.gz'
        with gzip.open(truthfile,'wt',encoding='utf-8',compresslevel=6) as stream:
            for r in census:
                public=dict(key=r['key'],run_key=f'{key}|p{r["parent"]}',right_ordinal=position[r['right_raw']],candidate=r['input_ordinal'],right_frame=r['right_frame'],delta_frame=r['delta_frame'],kind=r['kind'],length=1,represented=r['represented'],valid=r['valid'])
                stream.write(json.dumps(public,separators=(',',':'),allow_nan=False)+'\n')
        bags,_=core.independent_bags(rows,canon,indices,valid,owners);bagfile=target/'bags.jsonl.gz'
        with gzip.open(bagfile,'wt',encoding='utf-8',compresslevel=6) as stream:
            for number,b in enumerate(bags):stream.write(json.dumps(dict(ordinal=number,X=b['X'],y=b['y'],length=b['length'],lower=b['y'],upper=b['length'] if b['y'] else 0,kind=b['kind']),separators=(',',':'))+'\n')
        duplicate=io.read(root/'labels'/ref['path']/'duplicate_bee.json');dupfile=target/'duplicate.jsonl.gz'
        with gzip.open(dupfile,'wt',encoding='utf-8',compresslevel=6) as stream:
            for r in duplicate['records']:
                public=dict(key=f'{ref["clip"]}:{r["canonical_parent_id"]}:{r["left_raw_index"]}:{r["right_raw_index"]}',state=r['state'],represented=r['represented_cut_ordinal'] is not None,candidate=r['represented_cut_ordinal'],valid=r['input_valid'],delta_frame=r['delta_frame'])
                stream.write(json.dumps(public,separators=(',',':'))+'\n')
        clip_summary=next(c for c in summ['clips'] if c['domain']==ref['domain'] and c['clip']==ref['clip'])
        for arm in io.ARMS:
            p=np.load(root/'predictions'/outputs[key]['outputs'][arm]['probabilities'],mmap_mode='r');theta=outputs[key]['outputs'][arm]['threshold'];active=pred['folds'][ref['domain']]['models'][arm]['status']=='trained'
            variants=[(arm+'_CUT',p)]
            if arm=='relative_cross_interval':variants.append((arm+'_DISPLACED_CUT',core.displaced_scores(rows,canon,indices,p,valid)[0]))
            for state,scores in variants:
                chosen,_=core.selected_cuts(rows,canon,indices,scores,theta,valid,active);chosen=set(chosen)
                normalized=[dict(r,score=float(scores[r['input_ordinal']]) if r['represented'] else float('nan')) for r in census]
                exact=core.binary_diagnostics(normalized,theta);selected=[dict(r,score=1. if r['input_ordinal'] in chosen else 0.) for r in normalized]
                conservative=core.timing_diagnostics(selected,.5);original=clip_summary['states'][state]['cut_diagnostics']
                ap=exact['AP_fraction'];assert (ap is None and original['adjacency']['AP'] is None) or abs(float(ap[0]/ap[1])-original['adjacency']['AP'])<1e-12
                assert exact['counts'].get('exact_positive',0)==original['adjacency']['known_positive'] and exact['counts'].get('exact_same',0)==original['adjacency']['known_negative']
                true=original['timing_pm1'];known_fp=len(conservative['known_date_unmatched_predictions']);unknown=len(conservative['unconfirmed_unmatched_predictions'])
                diagnostics.append(dict(clip=key,state=state,conditional_AP=ap,full_known_pipeline_FN=core.binary_diagnostics(selected,.5)['pipeline_FN'],original_valid_only_adjacency=original['adjacency'],
                    original_timing=true,conservative_timing=conservative,difference=dict(truth=conservative['exact_date_truth']-true['truth'],tp=conservative['matched']-true['tp'],FP=known_fp-true['fp'],FN=conservative['FN']-true['fn'],unconfirmed_burden=unknown-true['unmatched_unknown_burden'])))
        clips.append(dict(clip=key,native_rows=len(rows),canonical_runs=len(runs),adjacent_population=len(census),run_lengths=[dict(run_key=f'{key}|p{parent}',rows=len(ii)) for parent,ii in sorted(runs.items())],
            kinds=dict(Counter(r['kind'] for r in census)),represented=sum(r['represented'] for r in census),valid=sum(r['valid'] for r in census),
            files={p.name:dict(path=p.relative_to(root/'verification').as_posix(),sha256=io.sha(p)) for p in (truthfile,bagfile,dupfile)}))
        print('INDEPENDENT208_OPAQUE',key,flush=True)
    result=dict(schema='switch208-independent-opaque-diagnostics-v1',passed=True,source_sha256=io.sha(__file__),main_sha256=io.sha(accepted),clips=clips,diagnostics=diagnostics,
        scope='Full original adjacency including no-input/invalid-crop known positives; no original GT IDs or rawGT/crops exported. AP conditional on finite valid represented scores; full-known abstentions remain pipeline FN. Original tool and conservative frame-resolution timing shown separately; interval/unknown/gap2or3 unmatched predictions are not exact-date false positives. No rematching, fitting, threshold, or physical-state changes.')
    io.save(root/'verification/opaque_index.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('D:/beetrack_runs/switch208'));p.add_argument('--field-go',action='store_true');args=p.parse_args();assert args.field_go
    print(json.dumps(run(args),indent=2))
