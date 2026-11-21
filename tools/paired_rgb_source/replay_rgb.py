"""Standard-library replay of source-pair selection, votes, strata and registered gates."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

ROSTER={'W':[f'BEE24-{i:02d}' for i in (*range(1,10),11,14)],
        'G':[f'BEE24-{i:02d}' for i in range(17,33) if i not in (24,28)]}
def close(a,b):assert abs(a-b)<1e-10,(a,b)
def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k,v in a.items():compare(v,b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for v,w in zip(a,b):compare(v,w)
    elif isinstance(a,float):close(a,b)
    else:assert a==b,(a,b)
def vote(r):
    if not r.get('valid',False):assert r['vote']==0 and not r.get('informative',False);return 0
    margin=r['aligned_cost']-r['exchanged_cost'];close(margin,r['raw_margin'])
    close(sum(r['temporal_change_by_fixed_box'])/4,r['temporal_change'])
    if min(r['pair_contrast'])<.02:state='same_appearance_no_information';v=0
    elif r['temporal_change']<1/255:state='static_image_no_information';v=0
    elif abs(margin)<=1/255:state='equal_assignment_cost_no_information';v=0
    else:state='informative_exchange_preferred' if margin>0 else 'informative_original_preferred';v=-1 if margin>0 else 1
    assert r['state']==state and r['vote']==v and r['informative']==(v!=0)
    if 'descriptors' in r:
        p,q,s,t=r['descriptors'];dist=lambda a,b:sum(abs(x-y) for x,y in zip(a,b))/6
        close(dist(p,s)+dist(q,t),r['aligned_cost']);close(dist(p,t)+dist(q,s),r['exchanged_cost'])
        compare([dist(p,q),dist(s,t)],r['pair_contrast'])
    return v
def counts(items,arm):
    vv=[r[arm]['vote'] for r in items];n=len(vv)
    return dict(selected=n,usable=sum(r[arm].get('valid',False) for r in items),
        informative=vv.count(1)+vv.count(-1),videos=len({r['clip'] for r in items}),
        correct=vv.count(1),wrong=vv.count(-1),abstain=vv.count(0),S=(vv.count(1)-vv.count(-1))/n if n else None)
def section(items):
    control=[r for r in items if r['frame_shift'] is not None]
    both=[r for r in control if r['primary'].get('valid',False) and r['control'].get('valid',False)]
    p,q=counts(control,'primary'),counts(control,'control')
    return dict(primary=counts(items,'primary'),time_control_intersection=dict(primary=p,control=q,S_difference=p['S']-q['S'] if control else None),
        both_valid_intersection=dict(primary=counts(both,'primary'),control=counts(both,'control')),control_date_unavailable=len(items)-len(control))
def summary(rows):
    out=dict(overall=section(rows),hives={h:section([r for r in rows if r['hive']==h]) for h in ROSTER},
        clips={c:section([r for r in rows if r['clip']==c]) for cc in ROSTER.values() for c in cc},
        boundary_strata={k:section([r for r in rows if r['near_boundary']==v]) for k,v in (('near',True),('other',False))},
        hive_boundary_strata={h+'_'+k:section([r for r in rows if r['hive']==h and r['near_boundary']==v]) for h in ROSTER for k,v in (('near',True),('other',False))})
    out['equal_clip_weight']={};covered=main=True
    for h,clips in ROSTER.items():
        pp=[out['clips'][c]['primary']['S'] for c in clips if out['clips'][c]['primary']['S'] is not None]
        cc=[out['clips'][c]['time_control_intersection'] for c in clips if out['clips'][c]['time_control_intersection']['primary']['S'] is not None]
        mean=lambda v:sum(v)/len(v) if v else None
        out['equal_clip_weight'][h]=dict(nonempty_clips=len(pp),primary_S=mean(pp),comparable_clips=len(cc),
            original_S=mean([v['primary']['S'] for v in cc]),shifted_S=mean([v['control']['S'] for v in cc]))
        p=out['hives'][h]['primary'];q=out['hives'][h]['time_control_intersection']
        enough=p['selected']>=8 and p['videos']>=3;covered=covered and enough
        main=main and enough and p['informative']>=.25*p['selected'] and p['S']>=.20 and q['S_difference'] is not None and q['S_difference']>0
    near=out['boundary_strata']['near']['primary']
    out['registered_gates']=dict(primary_coverage=covered,primary_pass=bool(main),primary_status='pass' if main else 'not_verified_coverage' if not covered else 'fail',
        boundary_coverage=near['selected']>=8,boundary_signal_positive=near['S'] is not None and near['S']>0,
        boundary_explanation_pass=bool(main and near['selected']>=8 and near['S']>0))
    return out
def replay(e):
    assert e['prereg']=='adab39fbcef0d7cb3e4507d7934598f30a4db388' and e['marker']=='[paired-rgb-v1]'
    sources=e['selection']['sources'];rows=e['measurements']['pairs']
    assert len(sources)==25 and len({s['clip'] for s in sources})==25
    selected=[]
    for s in sources:
        assert s['clip'] in ROSTER[s['hive']]
        if s['clip_failure']:
            assert not s['selected_digests'] and not s['eligible_pool'];continue
        pool=s['eligible_pool'];assert len(pool)==s['eligible_count']==s['attrition'].get('eligible',0)
        for p in pool:
            assert (p['id1'],p['run1'])<(p['id2'],p['run2']) and p['normalized_distance']<=2
            text=f"paired-rgb-v1|{s['clip']}|{p['id1']}|{p['run1']}|{p['id2']}|{p['run2']}|{p['frame']}"
            assert p['hash_input']==text and p['digest']==hashlib.sha256(text.encode('utf-8')).hexdigest()
            assert p['endpoint_frames']==[p['frame']-4,p['frame']+4]
            a,b=p['endpoint_frames'];nf=s['nframes'];shift=25 if 1<=a+25 and b+25<=nf else -25 if 1<=a-25 and b-25<=nf else None
            assert shift==p['frame_shift'] and p['control_frames']==([a+shift,b+shift] if shift is not None else None)
        ordered=sorted(pool,key=lambda p:(p['digest'],p['id1'],p['run1'],p['id2'],p['run2'],p['frame']))[:4]
        assert [p['digest'] for p in ordered]==s['selected_digests'];selected+=s['selected_digests']
    assert len(rows)==e['selection']['selected_count']==len(selected)
    assert len({r['digest'] for r in rows})==len(rows) and set(selected)=={r['digest'] for r in rows}
    for r in rows:
        vote(r['primary']);vote(r['control'])
        assert r['endpoint_rows']==r['primary']['fixed_box_rows']
        if r['frame_shift'] is not None:assert r['primary']['fixed_box_rows']==r['control']['fixed_box_rows']
        assert r['near_boundary']==bool(r['boundary_transitions'])
    got=summary(rows);compare(got,e['summary']['results'])
    descriptor_arms=invalid_arms=0
    if 'independent' in e:
        audit=e['independent'];receipt=audit['receipt'];own=audit['numerics']['pairs']
        assert receipt['all_passed'] and len(own)==len(rows)
        actual={r['digest']:r for r in rows};assert set(actual)=={r['digest'] for r in own}
        for r in own:
            saved=actual[r['digest']]
            for field in ('clip','hive','frame','frame_shift','near_boundary'):assert r[field]==saved[field]
            for arm in ('primary','control'):
                record=r[arm];vote(record)
                for field in ('state','valid','informative','vote'):assert record[field]==saved[arm][field]
                for field in ('aligned_cost','exchanged_cost','raw_margin','pair_contrast','temporal_change','temporal_change_by_fixed_box'):
                    if field in record:compare(record[field],saved[arm][field])
                if record['valid']:
                    assert len(record['descriptors'])==4 and all(len(d)==6 for d in record['descriptors'])
                    assert all(0<=v<=1 for d in record['descriptors'] for v in d)
                    descriptor_arms+=1
                else:invalid_arms+=1
        compare(summary(own),receipt['results']);compare(receipt['results'],got)
        byclip={s['clip']:s for s in sources}
        for c in receipt['clip_checks']:
            s=byclip[c['clip']]
            assert c['failed']==bool(s['clip_failure'])
            if not c['failed']:
                assert c['runs']==s['run_count'] and c['all_pairs']==sum(s['attrition'].values())
                assert c['eligible']==s['eligible_count'] and c['selected']==len(s['selected_digests'])
    return dict(pairs=len(rows),clips_with_pairs=got['overall']['primary']['videos'],
        states=dict(Counter(r['primary']['state'] for r in rows)),gates=got['registered_gates'],
        informative_coverage={h:got['hives'][h]['primary']['informative']/got['hives'][h]['primary']['selected'] for h in ROSTER},
        independent_descriptor_arms=descriptor_arms,independent_unusable_arms=invalid_arms)
def selfcheck():
    r=dict(valid=True,aligned_cost=.01,exchanged_cost=.08,raw_margin=-.07,pair_contrast=[.1,.1],
           temporal_change=.2,temporal_change_by_fixed_box=[.2]*4,state='informative_original_preferred',informative=True,vote=1)
    assert vote(r)==1
    r=dict(r,raw_margin=.07,aligned_cost=.08,exchanged_cost=.01,state='informative_exchange_preferred',vote=-1)
    assert vote(r)==-1
    r=dict(r,pair_contrast=[.01,.1],state='same_appearance_no_information',vote=0,informative=False)
    assert vote(r)==0 and counts([],'primary')['S'] is None
if __name__=='__main__':
    selfcheck();p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);a=p.parse_args()
    raw=a.evidence.read_bytes();raw=gzip.decompress(raw) if a.evidence.suffix=='.gz' else raw
    print(json.dumps(replay(json.loads(raw))))
