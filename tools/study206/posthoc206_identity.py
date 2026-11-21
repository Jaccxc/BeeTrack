"""Posthoc descriptive anchor-ID table; consumes accepted annotations only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import argparse
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
import json
import verify206_field as field


def describe(root):
    accepted=root/'verification/independent.json';assert field.sha(accepted)==accepted.with_suffix('.sha256').read_text().strip() and field.read(accepted)['passed']
    references=field.read(root/'frozen/deployment.json')['groups'];rows=[];denominators=Counter();counts=defaultdict(Counter);bygroup=defaultdict(Counter);runs=defaultdict(set);pairs=defaultdict(set);ledger_hashes={}
    for reference in references:
        path=root/'verification/owner_compact'/(reference['path']+'.json');assert field.sha(path)==path.with_suffix('.sha256').read_text().strip();ledger_hashes[reference['unit']]=field.sha(path);unit=field.read(path)
        for member in unit['members']:
            for query in member['queries']:
                h=unit['domain'];denominators[h+':all_native_events']+=1
                if not query['eligible']:continue
                denominators[h+':eligible_queries']+=1;label=query['labels']['0.5'];anchors=label['anchors'];assert len(anchors)==3
                qualified=[(i,a) for i,a in enumerate(anchors) if a['qualified']];ids={a['original_gt_id'] for _,a in qualified};owners={a['official_owner'] for _,a in qualified}
                assert None not in ids
                if len(ids)>=2:category='observed_distinct_original_GT_IDs'
                elif len(ids)==1 and len(owners)>=2:category='same_original_ID_distinct_GT_gap_runs'
                elif len(qualified)==3 and len(owners)==1:
                    badside=any(a['pred_inside']!=a['gt_inside'] for a in anchors)
                    category='three_same_run_clip_or_side_problem' if badside or label.get('at_clip_start',False) or label.get('at_clip_end',False) else 'three_same_run_no_listed_problem'
                else:category='insufficient_anchor_ownership'
                counts[h][category]+=1;bygroup[h+':'+str(unit['group'])][category]+=1
                compact=[dict(position=i,frame=a['pred_key'][0],canonical_GT_run=a['official_owner'],original_GT_ID=a['original_gt_id'],GT_run=a['gt_run']) for i,a in qualified]
                rows.append(dict(query_id=query['query_id'],unit=unit['unit'],hive=h,group=unit['group'],video=unit['video'],member=member['member'],member_index=member['member_index'],category=category,
                    qualified_anchor_count=len(qualified),third_anchor_unresolved=len(qualified)==2,qualified_anchors=compact,label_reason=label['reason']))
                if category=='observed_distinct_original_GT_IDs':
                    runs[h].update((unit['video'],a['official_owner']) for _,a in qualified)
                    pairs[h].update((unit['video'],a,b) for a,b in combinations(sorted(ids),2))
    result=dict(schema='return-query206-posthoc-observed-anchor-identity-v1',passed=True,threshold=.5,source_sha256=field.sha(__file__),independent_sha256=field.sha(accepted),compact_ledger_sha256=ledger_hashes,
        denominators=dict(denominators),counts={h:dict(c) for h,c in counts.items()},by_group={h:dict(c) for h,c in bygroup.items()},
        distinct_annotated_runs_in_positive_queries={h:[list(x) for x in sorted(values)] for h,values in runs.items()},distinct_original_ID_pairs={h:[list(x) for x in sorted(values)] for h,values in pairs.items()},queries=rows,
        scope='Posthoc, descriptive only. Numerator: eligible native-event queries with at least two individually mutual-unique anchors assigned to different original GT IDs, even if the third is unresolved. Counts are query opportunities, not independent bees; deduplication is within (clip,annotated GT run). Same-ID gap runs are separate. Three same-run anchors do not exclude identity changes between them. No inference of reciprocal A/B swaps, change time, visibility, or repairability. No rematching, fitting, threshold, or main gate changes.')
    field.save(root/'verification/posthoc_identity.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--field-go',action='store_true');args=p.parse_args();assert args.field_go and args.root
    print(json.dumps(describe(args.root),indent=2))
