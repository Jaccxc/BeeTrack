"""Tiny source-contract fixture only: no experimental files or field loaders."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np
import verify206 as core
import verify206_field as field
import verify206_owners as own


def check(repo):
    replay,previous,_,v=field.helpers(repo)
    path=Path(__file__).with_name('owners206.py');spec=importlib.util.spec_from_file_location('producer206_synthetic_contract',path);producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
    poly=[[0,-20],[50,-20],[50,20],[0,20]];nf=10
    gt=np.asarray([[f,17,x,0,2,2] for f,x in ((2,-3),(3,-3),(4,1),(5,1),(6,-3),(7,-3),(8,-3))],float)
    raw=np.asarray([[f,7,x,0,2,2,.9] for f,x in ((2,-3),(3,-3),(4,1))],float)
    raws=[raw.copy() for _ in range(3)];sources=[[('synthetic'+str(i),j) for j in range(3)] for i in range(3)]
    prepared=producer.prepare_gt(gt,poly);scenes=[producer.prior.scene(a,s) for a,s in zip(raws,sources)];count=0
    for slot in range(3):
        query=core.query_roster(raws[slot],sources[slot],poly,nf,'toy',slot)[0]
        for threshold in (.5,.3):
            result=own.analyze(query,raws,sources,gt,poly,nf,'toy','syntheticGT',threshold,replay,previous,v)
            reference=producer.analyze_query(prepared,scenes,raws,sources,query,threshold,nf,'toy','syntheticGT')
            for key in ('confirmed','classification','reason','canonical_gt_id','GT_run_key','missing_internal_frames'):assert result[key]==reference[key],key
            assert result['candidate']['frame']==reference['candidate']['frame']==6
            assert result['candidate']['added_row']==reference['candidate']['added_row']
            for a,b in zip(result['anchor_rows'],reference['anchor_rows']):replay.compare_evidence(a,b,v)
            frame=result['enumerated'][-1]['inserted_frames']
            for key,value in frame.items():assert value==json.loads(json.dumps(reference['enumerated'][-1]['inserted'][key])),key
            count+=1
    # Three same-run anchors do not certify every intervening observation.
    gt2=np.asarray([[f,17,x,0,2,2] for f,x in ((2,-3),(3,20),(4,-3),(5,-3),(6,1),(7,1),(8,-3))]+
        [[f,23,x,0,2,2] for f,x in ((2,-20),(3,1),(4,20))],float)
    raw2=np.asarray([[f,7,x,0,2,2,.9] for f,x in ((2,-3),(3,1),(4,-3),(5,-3),(6,1),(7,1))],float)
    sources2=[[('different-synthetic'+str(i),j) for j in range(len(raw2))] for i in range(3)]
    q=core.query_roster(raw2,sources2[0],poly,nf,'another',0)[0]
    label=own.analyze(q,[raw2.copy() for _ in range(3)],sources2,gt2,poly,nf,'toy2','syntheticGT2',.5,replay,previous,v)
    assert label['confirmed'] and label['candidate']['frame']==8
    from beetrack.evaluate import match_events
    after=np.vstack((raw2,label['candidate']['added_row']))
    assert match_events(gt2,raw2,poly)['tp']==1 and match_events(gt2,after,poly)['tp']==0
    return dict(passed=True,field_opened=False,source_contract_scenes=count,earliest_return_frame=6,both_thresholds=True,all_three_receivers=True,zero_net_same_run_does_not_imply_original_FP=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=Path.cwd());args=parser.parse_args();print(check(args.repo))
