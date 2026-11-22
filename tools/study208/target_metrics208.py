"""Pool opaque208 target gaps before AP/timing; never average clip AP."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import verify208 as core
import audit208 as io


def run(root):
    index=io.read(root/'verification/opaque_index.json');assert index['passed']
    meta=io.read(root/'predictions/metadata.json');pred={r['domain']+'/'+r['clip']:r for r in meta['clips']}
    verified=io.read(root/'verification/predictions.json');decisions={(r['clip'],r['arm']):r for r in verified['diagnostics']};bydomain={}
    for hive in core.HIVES:
        bydomain[hive]={}
        for arm in io.ARMS+('relative_cross_interval_DISPLACED',):
            modelarm=arm.replace('_DISPLACED','');rows=[];nmsrows=[];thresholds=set();originals=[]
            for clip in [c for c in index['clips'] if c['clip'].split('/')[0]==hive]:
                key=clip['clip'];record=decisions[key,modelarm];chosen=set(record['displaced_selected'] if arm.endswith('_DISPLACED') else record['selected']);thresholds.add(record['source_threshold'])
                probs=np.load(root/'predictions'/pred[key]['outputs'][modelarm]['probabilities'],mmap_mode='r')
                # Opaque table cut ordinals define the same valid eligible stretches.
                if arm.endswith('_DISPLACED'):
                    ordinal_to_candidate={};full=[]
                    with gzip.open(root/'verification'/clip['files']['adjacency.jsonl.gz']['path'],'rt') as stream:
                        for line in stream:full.append(json.loads(line))
                    for r in full:
                        if r['represented'] and r['valid']:ordinal_to_candidate[r['run_key'],r['right_ordinal']]=r['candidate']
                    shifted=np.array(probs,copy=True)
                    for run in {k[0] for k in ordinal_to_candidate}:
                        positions=sorted(k[1] for k in ordinal_to_candidate if k[0]==run);stretches=[]
                        for p in positions:
                            if not stretches or p!=stretches[-1][-1]+1:stretches.append([])
                            stretches[-1].append(p)
                        for stretch in stretches:
                            ids=[ordinal_to_candidate[run,p] for p in stretch];shifted[ids]=np.roll(probs[ids],1)
                    probs=shifted
                else:full=None
                def consume(records):
                    for r in records:
                        c=r['candidate'];normalized=dict(r,score=float(probs[c]) if c is not None else float('nan'));rows.append(normalized);nmsrows.append(dict(r,score=1. if c in chosen else 0.))
                if full is not None:consume(full)
                else:
                    with gzip.open(root/'verification'/clip['files']['adjacency.jsonl.gz']['path'],'rt') as stream:consume(json.loads(line) for line in stream)
                state=modelarm+('_DISPLACED_CUT' if arm.endswith('_DISPLACED') else '_CUT');originals.append(next(d['original_timing'] for d in index['diagnostics'] if d['clip']==key and d['state']==state))
            assert len(thresholds)==1;theta=next(iter(thresholds));conservative=core.timing_diagnostics(nmsrows,.5);old={k:sum(x[k] for x in originals) for k in ('truth','tp','fp','fn','unmatched_unknown_burden','all_selected_burden')};nms=core.binary_diagnostics(nmsrows,.5);nms.pop('AP_fraction')
            bydomain[hive][arm]=dict(raw05=core.binary_diagnostics(rows,.5),calibrated=core.binary_diagnostics(rows,theta),NMS=nms,timing_original=old,timing_conservative=conservative,
                timing_difference=dict(truth=conservative['exact_date_truth']-old['truth'],tp=conservative['matched']-old['tp'],FP=len(conservative['known_date_unmatched_predictions'])-old['fp'],FN=conservative['FN']-old['fn'],unconfirmed_burden=len(conservative['unconfirmed_unmatched_predictions'])-old['unmatched_unknown_burden']))
            print('INDEPENDENT208_TARGET',hive,arm,flush=True)
    result=dict(schema='switch208-independent-target-diagnostics-v1',passed=True,source_sha256=io.sha(__file__),opaque_index_sha256=io.sha(root/'verification/opaque_index.json'),targets=bydomain,
        scope='Scores pool clips within heldout target before AP; never mean clip AP. All original known adjacency positives remain pipeline FN when unrepresented/invalid; score-conditional AP is explicitly narrower. Timing compares producer valid-only convention with represented gap1 truth including invalid-crop positives and conservative unconfirmed unmatched burden. No new fitting, pairing or physical decisions.')
    io.save(root/'verification/target_diagnostics.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('D:/beetrack_runs/switch208'));p.add_argument('--field-go',action='store_true');args=p.parse_args();assert args.field_go;print(json.dumps(run(args.root),indent=2))
