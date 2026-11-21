"""Incremental203 numeric replay entry; no pixels/caches/GT/native scorer access."""
import argparse
from collections import defaultdict,Counter
from fractions import Fraction
import importlib
import json
from pathlib import Path
import sys
from itertools import groupby
sys.path.insert(0,str(Path.cwd()))
from archive203 import Archive203,open_archive
from cap_replay203 import verify as verify_capacity
from refit203_records import refit
from replay203_core import equal,physical_metric
from probes203 import replay as replay_probes
from tools.study202.replay202 import signed_bins,matching
from tools.study202.stream202 import verified_path
from tools.study198.run198 import combine
from owners_replay203 import replay as replay_owners


def run(root,reference202,code_dir,work_dir=None):
    sys.path.insert(0,str(code_dir));cap=importlib.import_module('capacity203');fit=importlib.import_module('fit203')
    archive=Archive203(root,reference202)
    for entry in archive.entries.values():verified_path(archive.root,entry)
    owner_report=replay_owners(archive)
    certificates=0
    for key in archive.keys('capacity_input'):
        unit=archive.get('capacity_input',key);answers=archive.get('capacity_result',key)
        for family,answer in answers.items():verify_capacity(unit,family,answer,cap);certificates+=1
        del unit,answers
    fitted=0
    for key in archive.keys('model'):
        model=archive.resolve(archive.get('model',key))
        def records():
            for row in archive.array('training_records',key):yield archive.resolve(row)
        refit(model,records,fit,work_dir);fitted+=1;del model
    # Fourfold decoder checks one complete three-member unit at a time.
    masks={};supply={h:Counter() for h in ('35','33','W','G')};reassigned={h:set() for h in supply}
    for hive in ('35','33','W','G'):
        models={mode:archive.resolve(archive.get('model',hive+'|'+mode)) for mode in ('I','L','T')}
        all_units={m['unit'] for m in archive.array('mask_member_clips',hive)}
        seen_units=set()
        def replay_unit(unit,contexts):
            expected={m['unit']+'|'+str(m['host_member']):m for m in archive.array('mask_member_clips',hive) if m['unit']==unit}
            skeleton=[{k:m[k] for k in ('unit','host_member','hive','clip','source_member') if k in m}
                      for m in expected.values() if m['unit']==unit]
            output=fit.deploy(contexts,models,skeleton)
            for context in contexts:
                supply[hive]['hosts']+=1
                if len(context['options'])==2:
                    options=context['options'];different=options[0]['outcome']!=options[1]['outcome']
                    supply[hive]['two_nonempty_destinations']+=1
                    supply[hive]['two_destinations_different_full_windows']+=different
                    supply[hive]['different_windows_with_one_keep_equivalent']+=different and any(o['outcome']==context['baseline_outcome'] for o in options)
            for permutation in output['M_permutation']:
                supply[hive]['hash_permutation_entries']+=1
                moved=permutation['source']!=permutation['destination'];supply[hive]['hash_moved_entries']+=moved
                if moved:reassigned[hive].add(permutation['host_key'])
            for member in output['member_clips']:
                target=archive.resolve(expected[unit+'|'+str(member['host_member'])])
                equal(member['selected'],target['selected']);equal(member['scores'],target['scores'])
                masks[unit,str(member['host_member'])]=member['selected']
            del output
        for unit,raw_contexts in groupby(archive.array('mask_contexts',hive),lambda c:c['unit']):
            assert unit not in seen_units;seen_units.add(unit)
            replay_unit(unit,[archive.resolve(c) for c in raw_contexts])
        for unit in all_units-seen_units:replay_unit(unit,[])
        del models
    physical_keys=set(archive.keys('physical_metric'))|set(archive.keys('physical_metric_ref202'))
    for key in physical_keys:
        metric=archive.metric(key);receipt=archive.get('physical_receipt',key)
        physical_metric(metric,receipt['identity']['nf'],signed_bins,matching);del metric,receipt
    groups=defaultdict(list);requests={}
    for key in archive.keys('native_clip'):
        arm=key.split('|',1)[0];clip=archive.get('native_clip',key)
        groups[arm,clip['domain'],str(clip['group'])].append(key)
        for m in clip['members']:
            host=str(m.get('host_member',m['member']));requests[clip['unit'],host,arm]=dict(physical_key=m['physical_key'],candidate_ids=m['candidate_ids'],seal=m['seal'])
            if arm in ('K','G','I','L','T','M'):equal(m['candidate_ids'],masks[clip['unit'],host][arm]['candidate_ids'])
    means=defaultdict(dict)
    totals=defaultdict(list)
    for (arm,hive,group),keys in groups.items():
        clips=[]
        for key in keys:
            clip=archive.get('native_clip',key)
            for m in clip['members']:
                metric=archive.metric(m['physical_key'])
                compact={k:metric[k] for k in ('nG','nP','pred_bins','gt_bins')}
                compact['iou']={t:{k:metric['iou'][t][k] for k in ('tp','pred','gt','f1')} for t in ('0.5','0.3')}
                equal(compact,m['metrics']);del metric
            clips.append(clip)
        value=combine(clips);value.pop('clips');totals[hive,arm].append(value);del clips
    expected=archive.get('score_summary','domains')
    for (hive,arm),rows in totals.items():
        e3=sum((Fraction(100*r['numerator'],r['denominator']) for r in rows),Fraction())/len(rows)
        f1={t:sum((sum((Fraction(2*m['tp'],max(m['pred']+m['gt'],1)) for m in r['iou'][t]['members']),Fraction())/3 for r in rows),Fraction())/len(rows) for t in ('0.5','0.3')}
        equal([e3.numerator,e3.denominator],expected[hive]['methods'][arm]['E3_fraction'])
        for t,value in f1.items():equal([value.numerator,value.denominator],expected[hive]['methods'][arm]['F1_fraction'][t])
        means[hive][arm]=(e3,f1)
    checks={}
    for arm in ('L','T'):
        gain=lambda h:means[h]['K'][0]-means[h][arm][0]
        checks[arm]=dict(head35_gain=gain('35')>Fraction(14,10),head33_gain=gain('33')>Fraction(13,10),W_no_harm=gain('W')>=-Fraction(14,10),
                         F1_no_harm=all(means[h][arm][1]['0.5']>=means[h]['K'][1]['0.5']-Fraction(5,1000) for h in ('35','33','W')))
    equal(checks,archive.get('score_summary','investment_checks'));equal({a:all(v.values()) for a,v in checks.items()},archive.get('score_summary','investment_pass'))
    attribution=archive.get('score_summary','attribution')
    equal(all(means[h]['T'][0]<means[h]['L'][0] for h in ('35','33')),attribution['T_strictly_beats_L_both_heads'])
    equal(all(means[h]['T'][0]<means[h][a][0] for h in ('35','33') for a in ('I','G','M')),attribution['T_strictly_beats_I_G_M_both_heads'])
    seen=set();fresh=resumed=shared=0;request_count=0
    for request in archive.array('physical_requests','shared'):
        key=request['unit'],str(request['member']),request['arm'];role=requests[key]
        equal(request['candidate_ids'],role['candidate_ids']);equal(request['path'],role['physical_key']);equal(request['seal'],role['seal'])
        token=request['path'];equal(request['within_job_shared'],token in seen)
        if token in seen:shared+=1
        elif request['resumed_from_disk']:resumed+=1
        else:fresh+=1
        seen.add(token);request_count+=1
    physical_scope=archive.get('score_summary','physical_scope')
    for key,value in dict(requested_arm_members=request_count,unique_streams=len(seen),fresh_unique_physical=fresh,
                          resumed_unique_physical=resumed,same_job_shared=shared,historical_reused=0,study_fresh_unique_physical=fresh+resumed).items():
        equal(value,physical_scope[key])
    assert len(seen)==len(physical_keys) and request_count==432
    assert physical_scope.get('prior_attempt_completed_unique_physical',resumed)>=resumed
    # Descriptive registered-control check: different choices need not change any count bin.
    control={h:dict(members=0,different_selections=0,edge_symmetric_difference=0,equal_full_windows=0,different_TP_05=0) for h in ('35','33','W','G')}
    for key in archive.keys('native_clip'):
        if not key.startswith('T|'):continue
        t=archive.get('native_clip',key);m=archive.get('native_clip','M|'+t['unit']);stat=control[t['domain']]
        assert [r['member'] for r in t['members']]==[r['member'] for r in m['members']]
        for left,right in zip(t['members'],m['members']):
            stat['members']+=1;stat['different_selections']+=set(left['candidate_ids'])!=set(right['candidate_ids'])
            stat['edge_symmetric_difference']+=len(set(left['candidate_ids'])^set(right['candidate_ids']))
            stat['equal_full_windows']+=left['metrics']['pred_bins']==right['metrics']['pred_bins']
            stat['different_TP_05']+=left['metrics']['iou']['0.5']['tp']!=right['metrics']['iou']['0.5']['tp']
    universe=archive.get('GT_universe_meta','shared');universe['clips']={}
    for key in archive.keys('GT_universe_clip_meta'):
        clip=archive.get('GT_universe_clip_meta',key);clip['runs']=archive.array('GT_universe_runs',key)
        universe['clips'][key]=clip
    def probe_groups():
        for key in archive.keys('return_probes'):
            rows=archive.get('probe_owner_rows',key)
            yield archive.resolve(archive.get('return_probes',key),rows)
    probes=replay_probes(universe,probe_groups())
    if ('supplement','verification/posthoc.json')in archive.entries:
        diagnostic=archive.get('supplement','verification/posthoc.json');assert diagnostic['passed']
        for hive,counters in supply.items():
            counters['hash_reassigned_hosts']=len(reassigned[hive])
            equal(dict(counters),diagnostic['domains'][hive]['supply']['counts'])
    assert certificates==24 and fitted==12
    return dict(capacity_certificates=certificates,source_refits=fitted,unique_physical_graphs=len(physical_keys),physical_scope=physical_scope,
                registered_control_descriptive=control,control_supply={h:dict(v) for h,v in supply.items()},owner_counts=owner_report['counts'],probe=probes,
                scope='numeric contexts/masks/windows/annotated event graphs; original full-evidence SHA and native geometry audit are distinct')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path)
    p.add_argument('--reference202',type=Path,default=Path('tools/study202/evidence.zip'))
    p.add_argument('--code-dir',type=Path,default=Path(__file__).parent);p.add_argument('--work-dir',type=Path)
    args=p.parse_args()
    with open_archive(args.evidence,args.work_dir) as root:print(json.dumps(run(root,args.reference202,args.code_dir,args.work_dir),indent=2,ensure_ascii=False))
