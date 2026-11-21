"""Independent links between greedy prefixes, physical states and 203 history."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import argparse
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import verify206_field as field
from verify206_scores import object_sha


def check(args):
    import sys
    sys.path.insert(0,str(args.repo.resolve()))
    from beetrack.evaluate import load_tracks
    root=args.root;accepted=root/'verification/independent.json'
    assert field.sha(accepted)==accepted.with_suffix('.sha256').read_text().strip() and field.read(accepted)['passed']
    refs=field.read(root/'frozen/deployment.json')['groups'];headers={r['unit']:field.read(root/'verification/owner_compact'/(r['path']+'.json')) for r in refs}
    qmap={q['query_id']:(u['unit'],str(m['member']),q) for u in headers.values() for m in u['members'] for q in m['queries']}
    greedy=field.read(root/'score/greedy.json');native=field.read(root/'score/native_scores.json');calls=field.read(root/'score/physical_receipts.json')
    callmap={(c['unit'],str(c['member']),c['arm']):c for c in calls};final={(v['unit'],str(m['member']),arm):m for arm,values in native.items() for v in values for m in v['members']}
    prefix_requests=steps=0
    for group in greedy:
        roster={(u['unit'],str(m['member'])) for u in headers.values() if u['domain']==group['domain'] and u['group']==group['group'] for m in u['members']}
        for t,report in group['thresholds'].items():
            cumulative=defaultdict(set)
            for step in report['steps']:
                unit,member,q=qmap[step['query_id']];assert unit==step['unit'];cumulative[unit,member].add(step['query_id']);steps+=1
                candidate=q['labels'][t]['candidate'];assert step['fixed_return_frame']==candidate['frame'] and step['gt_row_key']==candidate['GT_row_key']
                requests=step['physical_prefix']['requests'];assert {(r['unit'],str(r['member'])) for r in requests}==roster and len(requests)==len(roster)
                for r in requests:
                    key=r['unit'],str(r['member']);expected=sorted(cumulative[key]);assert r['query_ids']==expected
                    call=r['physical'];assert call==callmap[key[0],key[1],f'prefix:{t}:{step["step"]}'];prefix_requests+=1
            arm='R05' if t=='0.5' else 'R03'
            for unit,member in roster:assert final[unit,member,arm]['selected_query_ids']==sorted(cumulative[unit,member])
    assert prefix_requests==sum(c['arm'].startswith('prefix:') for c in calls)
    replay,previous,_,_=field.helpers(args.repo);GT={};history=[]
    for ref,group,members in field.groups(args):
        video=group['video']
        if video not in GT:GT[video]=load_tracks(args.repo/'runs/gt'/group['split']/video/'gt/gt.txt')
        for m in members:
            call=callmap[group['unit'],str(m['member']),'K'];meta=field.read(root/'score/physical'/call['path']/'receipt.json')
            if meta['origin']!='historical':continue
            origin=meta['historical'];folder=args.source203_score/'physical'/origin['source_leaf'];field.seal(folder,origin['source_leaf_seal']);old=field.read(folder/'receipt.json')['identity']
            assert old['unit']==group['unit'] and old['member']==f'{group["unit"]}|{m["member_index"]}' and old['candidate_ids']==[] and old['nf']==group['nf']
            for key,array in (('rows_sha256',m['rows']),('raw_sha256',m['rows']),('GT_sha256',GT[video]),('polygon_sha256',group['polygon'])):assert old[key]==previous.native202_array_sha(array)
            assert old['sources_sha256']==object_sha(m['sources'])
            history.append(dict(unit=group['unit'],member=m['member'],source_leaf=origin['source_leaf'],source_leaf_seal=origin['source_leaf_seal']))
    result=dict(schema='return-query206-independent-state-links-v1',passed=True,source_sha256=field.sha(__file__),independent_sha256=field.sha(accepted),prefix_steps=steps,prefix_requests=prefix_requests,historical_K_members=history,
        scope='Every accepted prefix materializes exactly its cumulative registered greedy choices, for all group members/clips; final masks agree. Historical K leaf identities independently rebound to actual206 K/raw GT/polygon/source arrays using203 serialization. No additional pairing or native scoring.')
    field.save(root/'verification/state_links.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--source203-score',type=Path);p.add_argument('--field-go',action='store_true');args=p.parse_args();assert args.field_go and args.root
    print(json.dumps(check(args),indent=2))
