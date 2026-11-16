"""Summarize frozen paired visibility at U,L,C,H, retaining missing data and physical GT identity."""
import argparse
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path


DOSES = ('U','L','C','H')
THRESHOLDS = {'0.5':'visible05', '0.3':'visible03'}
PATTERNS = [''.join(x) for x in product('01',repeat=4)]
PAIRS = ['00','01','10','11']


def case_key(value):
    if isinstance(value,dict):
        value = [value[k] for k in ('clip','fold','seed')]
    clip, fold, seed = value
    if not isinstance(clip,str) or not clip.strip() or type(fold) is not int or fold<0 or type(seed) is not int or seed<0:
        raise ValueError('Invalid physical clip, detector fold or seed')
    return clip,fold,seed


def validate(record):
    pop, status = record['population'], record['status']
    n, planned = pop['selected'], pop['planned']
    if type(n) is not int or type(planned) is not int or not 0<=n<=planned:
        raise ValueError('Expected a consistent selected and planned population')
    if record['dose'] not in DOSES or status not in ('available','unavailable'):
        raise ValueError('Unknown dose or availability status')
    if status=='unavailable' and not str(record.get('reason') or '').strip():
        raise ValueError('Unavailable dose needs a reason')
    events = record['events']
    if not isinstance(events,list) or (len(events)!=n and (status!='unavailable' or events)):
        raise ValueError('Dose event table length differs from its frozen selected population')
    by_id = {}
    for event in events:
        tid,direction,before,arrival = (event[k] for k in ('gt_tid','dir','before','arrival'))
        if type(tid) is not int or tid<0 or tid in by_id or direction not in ('in','out'):
            raise ValueError('Expected one row per resolved GT event ID, with a direction')
        if type(before) is not int or type(arrival) is not int or not 1<=before<arrival:
            raise ValueError('Invalid GT observation frames')
        for field in THRESHOLDS.values():
            v = event[field]
            if v is not None and (type(v) is not bool or status!='available'):
                raise ValueError('Expected true/false visibility on available doses, null when unavailable')
        if event['visible05'] is True and event['visible03'] is False:
            raise ValueError('Visibility at IoU 0.5 contradicts the same assignment at 0.3')
        by_id[tid] = event
    if 'shortfall' in pop and pop['shortfall']!=planned-n:
        raise ValueError('Selected population has inconsistent planned shortfall')
    if len(events)==n and 'selected_by_direction' in pop:
        totals = {d:sum(e['dir']==d for e in events) for d in ('in','out')}
        if pop['selected_by_direction']!=totals:
            raise ValueError('Selected direction counts differ from event rows')
    return by_id


def cell_tables(rows):
    # Rows retain the full population, even when only one contrast is observable.
    patterns = Counter(x['bits'] for x in rows)
    pairs = {}
    for j,dose in enumerate(DOSES[1:],1):
        grid = Counter(x['bits'][0]+x['bits'][j] for x in rows)
        pairs['U_to_'+dose] = dict(table={k:grid[k] for k in PAIRS},
                                   known=sum(grid[k] for k in PAIRS),
                                   unpaired=sum(v for k,v in grid.items() if '?' in k))
    recovered = [x for x in rows if x['bits'][0]=='0' and x['bits'][3]=='1']
    lc = Counter(x['bits'][1:3] for x in recovered)
    doses = {d:dict(visible=sum(r['bits'][i]=='1' for r in rows),known=sum(r['bits'][i]!='?' for r in rows))
             for i,d in enumerate(DOSES)}
    complete = sum(patterns[k] for k in PATTERNS)
    return dict(n=len(rows),known_four=complete, four_bits={k:patterns[k] for k in PATTERNS},
                incomplete_patterns={k:v for k,v in sorted(patterns.items()) if '?' in k}, doses=doses,pairs=pairs,
                h_gained_from_u=dict(n=len(recovered),lc_table={k:lc[k] for k in PAIRS},
                                     incomplete_lc={k:v for k,v in sorted(lc.items()) if '?' in k},
                                     l_success_ids=[r['gt_tid'] for r in recovered if r['bits'][1]=='1']))


def endpoint_bits(event,thr):
    # score_dose saves the .3 owner in match, with the official .5 assignment in owner05.
    ends = event.get('ends')
    if not ends:
        return (None,None)
    if not isinstance(ends,list) or len(ends)!=2:
        raise ValueError('Expected before/arrival endpoint records in that order')
    flags = []
    for end,key in zip(ends,('before','arrival')):
        if not isinstance(end,dict) or ('frame' in end and end['frame']!=event[key]):
            raise ValueError('Endpoint record is inconsistent with its frozen GT frame')
        if 'match' not in end:
            flags.append(None)
            continue
        match = end['match']
        if match is None:
            flags.append(False)
            continue
        side, strict = match.get('correct_side'),match.get('owner05')
        if (side is not None and type(side) is not bool) or (strict is not None and type(strict) is not bool):
            raise ValueError('Expected boolean correct-side/official-ownership flags')
        if thr=='0.3':
            flags.append(side)
        elif side is False or strict is False:
            flags.append(False)
        else:
            flags.append(True if side is True and strict is True else None)
    reported = event[THRESHOLDS[thr]]
    if reported is not None and None not in flags and reported!=all(flags):
        raise ValueError('Reported event visibility and its two endpoint assignments conflict')
    return flags


def endpoint_counts(actual,rows,thr):
    result = {}
    for dose in DOSES:
        if dose not in actual or actual[dose][0]['status']!='available':
            result[dose] = None
            continue
        by_dir = {'in':[], 'out':[]}
        for row in rows:
            event = actual[dose][1][row['gt_tid']]
            by_dir[row['dir']].append(endpoint_bits(event,thr))
        result[dose] = {}
        for direction, flags in by_dir.items():
            stats = {}
            for i,key in enumerate(('u','t')):
                successes = sum(f[i] is True for f in flags)
                unknown = sum(f[i] is None for f in flags)
                known = len(flags)-unknown
                stats[key] = dict(success=successes if known or not flags else None,known=known,
                                  unknown=unknown,selected=len(flags))
            result[dose][direction] = stats
    return result


def summarize_case(rows,planned,actual,thr):
    cells = cell_tables(rows)
    doses = cells['doses']
    seed_contrast = None
    if planned==16 and len(rows)==16 and cells['known_four']==16:
        seed_contrast = (doses['L']['visible']>doses['U']['visible'] and doses['L']['visible']>doses['C']['visible']
                         and bool(cells['h_gained_from_u']['l_success_ids']))
    return dict(**cells, events=rows,seed_contrast_if_complete_16=seed_contrast,
                by_direction={d:cell_tables([r for r in rows if r['dir']==d]) for d in ('in','out')},
                endpoints=endpoint_counts(actual,rows,thr))


def id_list(ids):
    return [list(key) for key in sorted(ids)]


def support(tallies):
    twice = {key for key,seeds in tallies.items() if len(seeds)>=2}
    all_three = {key for key,seeds in tallies.items() if {0,1,2}.issubset(seeds)}
    return dict(n_any=len(tallies),ids_any=id_list(tallies),n_at_least_two_seeds=len(twice),
                ids_at_least_two_seeds=id_list(twice),n_all_seeds_012=len(all_three),
                ids_all_seeds_012=id_list(all_three))


def distinct(cases,threshold):
    # Keep both (fold, seed) raw contexts. The same numeric seed in another fold does not make two seed votes.
    identities = defaultdict(list)
    for case in cases:
        for r in case['thresholds'][threshold]['events']:
            identities[case['clip'],r['gt_tid']].append((case['fold'],case['seed'],r['bits']))
    directions = ('neither_visible','gain','loss','both_visible')
    comparisons = {}
    for i,d in enumerate(DOSES[1:],1):
        votes = {name:defaultdict(set) for name in directions}
        unknown = set()
        for identity,contexts in identities.items():
            for _,seed,bits in contexts:
                pair = bits[0]+bits[i]
                if '?' in pair:
                    unknown.add(identity)
                else:
                    votes[directions[int(pair,2)]][identity].add(seed)
        gains,losses = votes['gain'],votes['loss']
        mixed = gains.keys() & losses.keys()
        mixed_across_seeds = {k for k in mixed if len(gains[k]|losses[k])>=2}
        repeated_both = {k for k in mixed if len(gains[k])>=2 and len(losses[k])>=2}
        never_observed = identities.keys()-set.union(*(set(v) for v in votes.values()))
        comparisons['U_to_'+d] = dict(**{k:support(v) for k,v in votes.items()},
                                       gain_and_loss=dict(n_any=len(mixed),ids_any=id_list(mixed),
                                                          ids_across_distinct_seeds=id_list(mixed_across_seeds),
                                                          ids_each_direction_in_two_seeds=id_list(repeated_both)),
                                       unknown_in_any_context=id_list(unknown),never_paired=id_list(never_observed))
    groups = {k:defaultdict(set) for k in PAIRS}
    all_high_gains, l_regained, c_regained, unresolved = (defaultdict(set) for _ in range(4))
    for identity,contexts in identities.items():
        for _,seed,bits in contexts:
            if bits[0]=='0' and bits[3]=='1':
                all_high_gains[identity].add(seed)
                lc = bits[1:3]
                (groups[lc] if lc in groups else unresolved)[identity].add(seed)
                if bits[1]=='1':
                    l_regained[identity].add(seed)
                if bits[2]=='1':
                    c_regained[identity].add(seed)
    return dict(n_population_with_ids=len(identities),comparisons=comparisons,
                h_gained_from_u=dict(all=support(all_high_gains),lc={k:support(v) for k,v in groups.items()},
                                     regained_by_L=support(l_regained),regained_by_C=support(c_regained),
                                     incomplete_lc=support(unresolved)),
                by_bee=[dict(clip=c,gt_tid=t,fold_seed_bits=[list(v) for v in visits])
                        for (c,t),visits in sorted(identities.items())])


def summarize(records,expected_cases=None):
    """Return paired tables for actual or explicitly scheduled (physical clip, detector fold, seed) cases.

    Each dose record must carry the same GT-only population for its physical clip. `events=[]` on an
    unavailable dose is permitted. A whole missing case can be listed in expected_cases, where its IDs
    are recovered from another case for that clip if available. Null visibility never counts as false.
    """
    contexts = defaultdict(dict)
    clip_pop,clip_anchors = {},{}
    for record in records:
        context = case_key(record)
        clip,_,_ = context
        dose = record['dose']
        if dose in contexts[context]:
            raise ValueError(f'Duplicate dose record {context}/{dose}')
        by_id = validate(record)
        population = record['population']
        if clip in clip_pop and clip_pop[clip]!=population:
            raise ValueError(f'Mismatched frozen population metadata for {clip}')
        clip_pop[clip] = population
        if by_id or population['selected']==0:
            anchors = {i:(r['dir'],r['before'],r['arrival']) for i,r in by_id.items()}
            if clip in clip_anchors and clip_anchors[clip]!=anchors:
                raise ValueError(f'Mismatched GT identities, directions or frames for {clip}')
            clip_anchors[clip] = anchors
        contexts[context][dose] = (record,by_id)
    keys = set(contexts)
    if expected_cases is not None:
        registered = [case_key(k) for k in expected_cases]
        if len(set(registered))!=len(registered) or not keys.issubset(registered):
            raise ValueError('Duplicate planned case or unexpected observed case')
        keys = set(registered)
    cases = []
    for clip,fold,seed in sorted(keys):
        actual = contexts.get((clip,fold,seed),{})
        runs = {d:dict(status=actual[d][0]['status'],reason=actual[d][0].get('reason')) if d in actual
                else dict(status='missing',reason='no dose record supplied') for d in DOSES}
        base = clip_anchors.get(clip)
        populations = clip_pop.get(clip)
        by_thr = {}
        for thr,field in THRESHOLDS.items():
            table = []
            for tid,(direction,before,arrival) in sorted((base or {}).items()):
                bits = []
                for d in DOSES:
                    r = actual.get(d)
                    value = r[1][tid][field] if r and tid in r[1] else None
                    bits.append('?' if value is None else '1' if value else '0')
                table.append(dict(gt_tid=tid,dir=direction,before=before,arrival=arrival,bits=''.join(bits)))
            by_thr[thr] = summarize_case(table,populations['planned'],actual,thr) if base is not None else None
        cases.append(dict(clip=clip,fold=fold,seed=seed,population=populations,population_ids_known=base is not None,
                          sample_complete=(populations['selected']==populations['planned']) if populations else None,
                          all_doses_available=all(r['status']=='available' for r in runs.values()),
                          runs=runs,thresholds=by_thr))
    def unique(rows):
        known = [r for r in rows if r['population_ids_known']]
        return {thr:distinct(known,thr) for thr in THRESHOLDS}
    groups = sorted({(r['clip'],r['fold']) for r in cases})
    return dict(format='paired-warp-visibility-v1',dose_order=list(DOSES),endpoint_labels={'u':'before','t':'arrival'},
                schedule='explicit' if expected_cases is not None else 'observed_case_keys_only',cases=cases,
                population_unknown_cases=[[r['clip'],r['fold'],r['seed']] for r in cases if not r['population_ids_known']],
                unique_by_clip_fold=[dict(clip=c,fold=f,thresholds=unique([r for r in cases if r['clip']==c and r['fold']==f]))
                                     for c,f in groups],unique_all_clips=unique(cases))


def self_check():
    from copy import deepcopy

    def scores(clip,fold,seed,patterns):
        result = []
        for i,d in enumerate(DOSES):
            events = [dict(gt_tid=j+1,dir='in' if j%2==0 else 'out',before=2*j+1,arrival=2*j+2,
                           visible05=bit[i]=='1',visible03=bit[i]=='1',ends=[]) for j,bit in enumerate(patterns)]
            result.append(dict(clip=clip,fold=fold,seed=seed,dose=d,status='available',reason=None,
                               population=dict(planned=16,selected=len(patterns)),events=events))
        return result

    # ID 1 gains in seeds 0/1 and loses in seed 2. ID 2 gains in seed 0, whose fold repetition is only one seed.
    records = (scores('a',33,0,['0101','0111']) + scores('a',35,0,['0101','0111'])
               + scores('a',33,1,['0111','1011']) + scores('a',33,2,['1010','0000'])
               + scores('b',33,0,['0101']))
    y = summarize(records)
    u = y['unique_all_clips']['0.5']
    gain = u['comparisons']['U_to_L']['gain']
    assert u['n_population_with_ids']==3 and gain['n_any']==3 and gain['ids_at_least_two_seeds']==[['a',1]]
    assert u['comparisons']['U_to_L']['gain_and_loss']['ids_any']==[['a',1],['a',2]]
    assert u['h_gained_from_u']['lc']['10']['n_any']==2 and u['h_gained_from_u']['lc']['11']['n_any']==2
    assert u['h_gained_from_u']['all']['ids_at_least_two_seeds']==[['a',1]]
    t = y['cases'][0]['thresholds']['0.5']
    assert len(t['four_bits'])==16 and t['four_bits']['0101']==t['four_bits']['0111']==1
    assert t['pairs']['U_to_H']['table']=={'00':0,'01':2,'10':0,'11':0} and t['known_four']==2
    assert t['by_direction']['in']['four_bits']['0101']==1 and t['by_direction']['out']['four_bits']['0111']==1
    assert t['by_direction']['in']['pairs']['U_to_L']['table']['01']==1
    assert t['endpoints']['U']['in']['u']==dict(success=None,known=0,unknown=1,selected=1)
    assert t['seed_contrast_if_complete_16'] is None

    # An L rescue at all three specified seeds may have different C bits at those seeds.
    # ID 3 gains in three numeric seeds (0,1,3) plus a fold duplicate of 0, but never seed 2.
    mixed = (scores('a',33,0,['0101','1000','0101']) + scores('a',35,0,['0101','1000','0101'])
             + scores('a',33,1,['0111','1000','0101']) + scores('a',33,2,['0111','1000','0000'])
             + scores('a',35,3,['0000','0000','0101']))
    all_seeds = summarize(mixed)['unique_all_clips']['0.5']
    assert all_seeds['comparisons']['U_to_L']['gain']['ids_all_seeds_012']==[['a',1]]
    assert all_seeds['comparisons']['U_to_L']['loss']['ids_all_seeds_012']==[['a',2]]
    regained = all_seeds['h_gained_from_u']
    assert regained['regained_by_L']['n_all_seeds_012']==1 and regained['regained_by_L']['ids_all_seeds_012']==[['a',1]]
    assert regained['lc']['10']['n_all_seeds_012']==regained['lc']['11']['n_all_seeds_012']==0

    # Official .3 assignment can be correct at one endpoint where its official .5 ownership fails.
    ee = scores('x',33,0,['0101','0101'])
    events = ee[0]['events']
    events[0].update(visible03=True,ends=[dict(frame=1,match=dict(correct_side=True,owner05=False)),
                                         dict(frame=2,match=dict(correct_side=True,owner05=True))])
    events[1]['ends'] = [dict(frame=3,match=dict(correct_side=False,owner05=True)),dict(frame=4,match=None)]
    ep=summarize(ee)['cases'][0]['thresholds']
    assert ep['0.3']['endpoints']['U']['in']['u']['success']==1 and ep['0.5']['endpoints']['U']['in']['u']['success']==0
    assert ep['0.5']['endpoints']['U']['in']['t']==dict(success=1,known=1,unknown=0,selected=1)
    assert ep['0.3']['endpoints']['U']['out']['u']['success']==ep['0.3']['endpoints']['U']['out']['t']['success']==0
    assert ep['0.5']['by_direction']['in']['pairs']['U_to_L']['table']['01']==1
    assert ep['0.3']['by_direction']['in']['pairs']['U_to_L']['table']['11']==1

    # A missing L must not hide an observable U-H gain; an unavailable H must not erase the U-L pair.
    partial = scores('x',33,0,['0101'])
    partial.pop(1)
    partial[1]['events'][0]['visible03'] = None
    partial.extend(scores('x',33,1,['0101']))
    partial[-1].update(status='unavailable',reason='synthetic failure',events=[])
    explicit = [('x',33,0),('x',33,1),('x',33,2),('no-data',35,0)]
    y = summarize(partial,expected_cases=explicit)
    unknown,zero,one,two = y['cases']
    assert unknown['population_ids_known'] is False and unknown['thresholds']['0.5'] is None
    assert zero['runs']['L']['status']=='missing' and one['runs']['H']['status']=='unavailable'
    assert zero['thresholds']['0.5']['endpoints']['L'] is None and one['thresholds']['0.3']['endpoints']['H'] is None
    assert zero['thresholds']['0.5']['incomplete_patterns']=={'0?01':1}
    assert zero['thresholds']['0.3']['incomplete_patterns']=={'0??1':1}
    assert zero['thresholds']['0.5']['pairs']['U_to_H']['table']['01']==1
    assert zero['thresholds']['0.5']['h_gained_from_u']['incomplete_lc']=={'?0':1}
    assert one['thresholds']['0.5']['pairs']['U_to_L']['table']['01']==1
    assert two['thresholds']['0.5']['events'][0]['bits']=='????'
    assert y['unique_all_clips']['0.5']['comparisons']['U_to_L']['gain']['n_at_least_two_seeds']==0

    complete = scores('c',33,0,['0101']+['0000']*15)
    assert summarize(complete)['cases'][0]['thresholds']['0.5']['seed_contrast_if_complete_16'] is True
    assert summarize([])['cases']==[]
    assert summarize(scores('empty',35,2,[]))['cases'][0]['thresholds']['0.5']['known_four']==0
    for key in ('before','gt_tid','dir','selected'):
        bad = deepcopy(records)
        if key=='selected':
            bad[0]['population']['selected'] = 1
        else:
            bad[0]['events'][0][key] = {'before':2,'gt_tid':42,'dir':'out'}[key]
        try:
            summarize(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched population or anchor accepted')
    try:
        summarize(records+[records[0]])
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate dose accepted')
    print('Synthetic four-dose tables verified: threshold-specific missingness, 16 cells, gains/losses, '
          'same-bee opposite changes, duplicate fold, different physical clips and rejection of mismatched populations.')


def main():
    from inspector import read_json, write_once
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('self-check')
    run=sub.add_parser('summarize',help='read explicit scored dose JSON files or lists of dose records')
    run.add_argument('input',nargs='+',type=Path)
    run.add_argument('--expected',type=Path,help='JSON list of (clip, fold, seed) keys including wholly missing cases')
    run.add_argument('--out',type=Path,help='write an immutable summary; otherwise print JSON')
    a=p.parse_args()
    if a.command=='self-check':
        return self_check()
    records=[]
    for path in a.input:
        value=read_json(path)
        records.extend(value if isinstance(value,list) else [value])
    keys=read_json(a.expected) if a.expected else None
    result=summarize(records,keys)
    if a.out:
        print(f'{write_once(a.out,result)} {a.out}')
    else:
        print(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
