"""Small diagnostics from independently accepted 203 numeric outputs only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import argparse
from collections import Counter
from pathlib import Path
import json
import verify203 as replay


def diagnose(root):
    receipt=root/'verification/independent.json'
    assert replay.source_sha(receipt)==receipt.with_suffix('.sha256').read_text().strip()
    assert replay.read(receipt)['passed']
    native=replay.read(root/'score/native_scores.json')
    byarm={a:{(g['unit'],m['host_member']):(g,m) for g in groups for m in g['members']} for a,groups in native.items()}
    domains={}
    for hive in ('35','33','W','G'):
        path=root/'predictions'/(hive+'.masks.json')
        decisions={}
        for member in replay.stream_array(path,'member_clips'):
            decisions[member['unit'],member['host_member']]={a:tuple(v['candidate_ids']) for a,v in member['selected'].items()}
        comparisons={}
        for left,right in (('T','M'),('L','T')):
            counts=Counter();changes=[]
            for key,choices in decisions.items():
                g,l=byarm[left][key];_,r=byarm[right][key]
                counts['members']+=1
                samechoice=choices[left]==choices[right]
                samebins=l['metrics']['pred_bins']==r['metrics']['pred_bins']
                counts['same_choice']+=samechoice;counts['same_bins']+=samebins
                counts['different_choice_same_bins']+=not samechoice and samebins
                delta={t:r['metrics']['iou'][t]['tp']-l['metrics']['iou'][t]['tp'] for t in ('0.5','0.3')}
                for t,d in delta.items():counts['TP_changed_'+t]+=d!=0
                if not samechoice or any(delta.values()):
                    changes.append(dict(unit=key[0],host_member=key[1],member=l['member'],left_ids=list(choices[left]),right_ids=list(choices[right]),same_bins=samebins,TP_right_minus_left=delta))
            comparisons[left+'_vs_'+right]=dict(counts=dict(counts),changes=changes)
        counters=Counter();twohosts=[]
        for c in replay.stream_array(path,'contexts'):
            options=c['options'];n=len(options);counters['hosts']+=1
            if n!=2:continue
            counters['two_nonempty_destinations']+=1
            different=options[0]['outcome']!=options[1]['outcome']
            equalkeep=[o['outcome']==c['baseline_outcome'] for o in options]
            counters['two_destinations_different_full_windows']+=different
            counters['different_windows_with_one_keep_equivalent']+=different and any(equalkeep)
            twohosts.append(dict(host_key=c['host_key'],action_ids=[o['action_id'] for o in options],different_full_windows=different,keep_equivalent=equalkeep))
        moved_hosts=set()
        for p in replay.stream_array(path,'M_permutation'):
            counters['hash_permutation_entries']+=1
            counters['hash_moved_entries']+=p['source']!=p['destination']
            if p['source']!=p['destination']:moved_hosts.add(p['host_key'])
        counters['hash_reassigned_hosts']=len(moved_hosts)
        domains[hive]=dict(comparisons=comparisons,supply=dict(counts=dict(counters),two_destination_hosts=twohosts))
    result=dict(schema='peer-suffix203-independent-posthoc-v1',passed=True,independent_sha256=replay.source_sha(receipt),domains=domains,
        scope='Posthoc diagnostic of the frozen realised permutation and choices only. Equal realised T/M windows does not imply all possible reassignments have equal windows; cross-host resource conflicts remain relevant. No fitting or threshold changes.')
    replay.save_receipt(root/'verification/posthoc.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--field-go',action='store_true');args=p.parse_args()
    assert args.field_go
    print(json.dumps(diagnose(args.root),indent=2))
