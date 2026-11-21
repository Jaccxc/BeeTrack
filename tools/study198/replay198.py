"""Private numeric peer-gate replay core; no field inputs on default run."""
from fractions import Fraction
import numpy as np

def peer_features(record):
    """Validate compact per-date IoU evidence and recompute same-peer gate."""
    available=[];free=[]
    for p in record['peers']:
        qualified=sorted(p['qualified'],key=lambda r:(-r['mean'],r['track_id']))
        for q in qualified:
            assert len(q['iou'])==3 and sum(x>=.5 for x in q['iou'])>=2
            np.testing.assert_allclose(np.mean(q['iou']),q['mean'],rtol=0,atol=1e-15)
        best=qualified[0] if qualified else None;second=qualified[1] if len(qualified)>1 else None
        margin=best['mean']-(second['mean'] if second else 0) if best else 0
        identified=bool(best and margin>=.1)
        assert p['best_id']==(best['track_id'] if best else None) and p['second_id']==(second['track_id'] if second else None)
        np.testing.assert_allclose(p['best_mean'],best['mean'] if best else 0,rtol=0,atol=1e-15)
        np.testing.assert_allclose(p['second_mean'],second['mean'] if second else 0,rtol=0,atol=1e-15)
        assert p['identified']==identified and p['identified_id']==(best['track_id'] if identified else None)
        np.testing.assert_allclose(p['margin'],margin,rtol=0,atol=1e-15)
        observable=[s for s in p['short'] if s['any_iou']>=.5]
        for s in p['short']:assert s['observable']==(s['any_iou']>=.5)
        assert p['short_observable']==bool(observable)
        free.append(sum(x['iou']>=.5 for x in p['free_core'])>=2 and bool(observable))
        if identified and observable:available.append((p,observable))
    gate=bool(available)
    for key in ('gate','gate_same_peer','eligible','qualifies','ablation_eligible'):assert record[key]==gate
    assert record['frame_only_gate']==any(free) and record['usable_peer_count']==len(available)
    u=[];identity=[];k=record['k'];core=record['core_inside'];short=record['short_inside']
    for p,obs in available:
        u.append([np.mean([r['iou'] for r in p['free_core']]),len(obs)/k,np.mean([r['any_iou'] for r in obs]),np.mean([r['any_side']==core for r in obs])])
        identity.append([p['best_mean'],p['margin'],np.mean([r['core_row_present'] for r in obs]),np.mean([r['core_iou'] or 0 for r in obs]),np.mean([(r['core_iou'] or 0)>=.5 and r['core_side']==short for r in obs]),np.mean([(r['core_iou'] or 0)>=.5 and r['core_side']==core for r in obs]),np.mean([r['any_best_id']==p['identified_id'] for r in obs])])
    U=np.mean(u,axis=0).tolist() if u else [0.]*4;I=np.mean(identity,axis=0).tolist() if identity else [0.]*7
    n=list(record['features_G'])+[float(record['direction']==1),len(available)/2]
    np.testing.assert_allclose(record['features_N11'],n,rtol=0,atol=1e-15)
    np.testing.assert_allclose(record['features_B15'],n+U,rtol=0,atol=1e-15)
    np.testing.assert_allclose(record['features_I22'],n+U+I,rtol=0,atol=1e-15)
    return gate

def count_gate(group_records,physical_validated=False):
    statuses={};hives={}
    for hive,line in (('35',Fraction(14,10)),('33',Fraction(13,10))):
        rows=[r for r in group_records if r['domain']==hive]
        if not rows:statuses[hive]='unknown';continue
        ceiling=sum(Fraction(100*(r['K_numerator']-r['O_lower']),r['denominator']) for r in rows)/len(rows)
        attained=sum(Fraction(100*(r['K_numerator']-r['O_upper']),r['denominator']) for r in rows)/len(rows)
        status='excluded' if ceiling<=line else 'pass' if attained>line and physical_validated else 'unknown'
        statuses[hive]=status;hives[hive]=dict(gain_ceiling=float(ceiling),witness_gain=float(attained),status=status)
    proceed=all(statuses.get(h)=='pass' for h in ('35','33'))
    return dict(proceed_to_training=proceed,hives=hives,statuses=statuses)

def refit_head(records,name):
    dim={'N':11,'B':15,'I':22}[name]
    if not records:return dict(mean=[0.]*dim,scale=[1.]*dim,beta=np.zeros((dim,2)).tolist(),intercept=[0.,.5])
    keys=[(r['hive'],r['group_id'],r['clip'],r['delta']) for r in records]
    hives={k[0] for k in keys};groups={h:{k[1] for k in keys if k[0]==h} for h in hives}
    clips={(h,g):{k[2] for k in keys if k[:2]==(h,g)} for h in hives for g in groups[h]}
    classes={k[:3]:{q[3] for q in keys if q[:3]==k[:3]} for k in keys}
    w=np.asarray([1/(len(hives)*len(groups[h])*len(clips[h,g])*len(classes[h,g,c])*keys.count((h,g,c,s))) for h,g,c,s in keys])
    np.testing.assert_allclose(w,[r['weight'] for r in records],rtol=0,atol=1e-15)
    x=np.asarray([r[f'features_{name}{dim}'] for r in records]);y=np.asarray([[r['y'],r['z']] for r in records]);assert (abs(y[:,0])<=1).all() and np.isin(y[:,1],(0,1)).all()
    mean=w@x;constant=np.ptp(x,axis=0)==0;mean[constant]=x[0,constant];scale=np.sqrt(w@((x-mean)**2));scale[(scale==0)|constant]=1;z=(x-mean)/scale
    a=np.column_stack((np.ones(len(x)),z));penalty=np.diag([0.]+[.2]*dim);normal=a.T@(w[:,None]*a)+penalty
    beta=np.zeros((dim,2));intercept=np.zeros(2)
    for col in range(2):
        if np.ptp(y[:,col])==0:intercept[col]=y[0,col]
        else:
            coef=np.linalg.solve(normal,a.T@(w*y[:,col]));intercept[col]=coef[0];beta[:,col]=coef[1:]
    return dict(mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),intercept=intercept.tolist())

def predict_head(head,features,delta):
    raw=np.asarray(head['intercept'])+((np.asarray(features)-head['mean'])/head['scale'])@head['beta']
    y=np.clip(raw[:,0],-1,1);z=np.clip(raw[:,1],0,1);identity=np.asarray(delta)*(2*z-1);utility=(y+identity)/2
    return dict(raw=raw.tolist(),yhat=y.tolist(),zhat=z.tolist(),identity_utility=identity.tolist(),utility=utility.tolist(),cut=(utility>.1).tolist())

def posthoc_drop_necessity35(add_only_gain_ceiling=Fraction(575,1161),group_gt=129,groups=4,line=Fraction(14,10)):
    """Necessary bound only: arbitrary adds plus at most n event removals.

    One single-event deletion changes at most one bin per phase:30 numerator
    units. Three-member denominator90*GT and four-group mean give25/387pp.
    This is limited to the registered35 action family, not a training cutoff.
    """
    ceiling=Fraction(add_only_gain_ceiling)
    per_drop=Fraction(100*30,90*group_gt*groups)
    assert ceiling==Fraction(575,1161) and per_drop==Fraction(25,387)
    bounds={n:ceiling+n*per_drop for n in range(16)}
    minimum=next(n for n,bound in bounds.items() if bound>line)
    assert bounds[12]==Fraction(1475,1161) and bounds[14]==Fraction(1625,1161)
    assert bounds[14]<line and minimum==15
    encode=lambda f:[f.numerator,f.denominator]
    return dict(add_only_gain_ceiling=encode(ceiling),per_drop_gain_ceiling=encode(per_drop),
        at_most12_drops=encode(bounds[12]),at_most14_drops=encode(bounds[14]),
        minimum_drops_necessary=minimum,sufficient=False,posthoc=True,
        curve=[dict(drops=n,gain_ceiling=encode(bound),can_exceed_line=bound>line) for n,bound in bounds.items()])

def close(a,b):
    if isinstance(a,dict):
        for k,v in a.items():assert k in b,k;close(v,b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,float):np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-10)
    else:assert a==b

def replay(data,reference197):
    import hashlib
    import sys
    from pathlib import Path
    repo=Path.cwd()
    if not (repo/'beetrack/p0.py').is_file():repo=Path(__file__).resolve().parents[2]
    assert (repo/'beetrack/p0.py').is_file(),'run from BeeTrack repository root'
    sys.path.insert(0,str(repo))
    import graph198,run198
    from tools.study197.replay197 import verify_oracle,posthoc as posthoc197
    peer=data['peer'];records={r['candidate_id']:r for g in peer['groups'] for m in g['members'] for r in m['records']}
    original={c['candidate_id']:c for g in reference197['frozen']['groups'] for m in g['members'] for c in m['bundle']['candidates']}
    assert records.keys()==original.keys()
    for key,r in records.items():
        for field in ('frame','direction','delta','features_G','delta_windows','k','short_inside','core_inside'):close(r[field],original[key][field])
    for r in records.values():peer_features(r)
    graphs=data['graph_data']['models'];oldscores=reference197['native_scores']
    oldmasks={(m['unit'],m['member']):m['selected'] for f in reference197['folds'].values() for m in f['masks']['member_clips']}
    for p in reference197['summary']['oracle_certificates']:
        g=next(g for g in reference197['frozen']['groups'] if g['unit']==p['unit'])
        for m in g['members']:oldmasks[p['unit'],m['member']]['O']=[]
        for cid in p['oracle']['selected_ids']:
            slot,local=p['global_to_local'][cid];oldmasks[p['unit'],g['members'][slot]['member']]['O'].append(local)
    checked=0
    for arm,videos in oldscores.items():
        for v in videos:
            for m in v['members']:
                graph198.assert_reference(graphs[v['unit']][m['member']],oldmasks[v['unit'],m['member']][arm],m['metrics']);checked+=1
    assert checked==420
    capacity=data['capacity'];byunit={g['unit']:g for g in peer['groups']}
    group_totals={}
    for unit,row in capacity['units'].items():
        g=byunit[unit];eligible=[r for m in g['members'] for r in m['records'] if r['eligible']]
        verify_oracle(row['baseline_pred_sum'],row['gt_bins'],eligible,g['nf'],row['oracle'])
        total=group_totals.setdefault((g['domain'],g['group']),[0,0,0,0]);o=row['oracle']
        for i,field in enumerate(('baseline_numerator','lower_bound','upper_bound','denominator')):total[i]+=o[field]
    assert len(group_totals)==len(capacity['group_records'])
    for row in capacity['group_records']:assert group_totals[row['domain'],row['group']]==[row['K_numerator'],row['O_lower'],row['O_upper'],row['denominator']]
    assert capacity['all60_native_graph_verified'] and capacity['physical_verified_members']==60
    assert count_gate(capacity['group_records'],True)['proceed_to_training']==capacity['stage2_authorized_by_count_gate']
    labels={}
    for g in peer['groups']:
        labels.update(graph198.labels_for_group([graphs[g['unit']][m['member']] for m in g['members']]))
    masks={};refit_count=0
    for h,fold in data['folds'].items():
        model=fold['model'];training=model['training_records']
        assert {r['key'] for r in training}=={k for k,r in records.items() if r['hive']!=h and r['eligible']}
        for r in training:
            candidate=records[r['key']];label=labels[r['key']]
            for k in ('y','z','improvement_numerator','event_on_TP','event_off_TP'):close(label[k],r[k])
            for name,dim in (('N',11),('B',15),('I',22)):close(candidate[f'features_{name}{dim}'],r[f'features_{name}{dim}'])
        for name in ('N','B','I'):close(refit_head(training,name),model['heads'][name]);refit_count+=1
        for m in fold['masks']['member_clips']:
            cc=[records[c] for c in m['candidate_ids']];expected={a:[] for a in ('K','T197','Tg','N','B','I','F','Y','M')}
            for c,eligible,index in zip(cc,m['eligible'],range(len(cc))):
                assert eligible==c['eligible'];cid=c['candidate_id'];old=cid in oldmasks[m['unit'],m['member']]['T']
                if old:expected['T197'].append(cid)
                if old and eligible:expected['Tg'].append(cid)
                for name,dim in (('N',11),('B',15),('I',22)):
                    if not eligible:assert m['scores'][name][index] is None;continue
                    p=predict_head(model['heads'][name],[c[f'features_{name}{dim}']],[c['delta']]);saved={k:v[0] for k,v in p.items() if k!='cut'};close(saved,m['scores'][name][index])
                    if p['cut'][0]:expected[name].append(cid)
                    if name=='I':
                        if p['identity_utility'][0]>.1:expected['F'].append(cid)
                        if p['yhat'][0]>.1:expected['Y'].append(cid)
            hashes={c['candidate_id']:hashlib.sha256(('endpoint-peer-null-v1|'+c['candidate_id']).encode()).hexdigest() for c in cc}
            assert m['M_hashes']==[hashes[c['candidate_id']] for c in cc]
            for delta in (-1,1):
                for direction in (-1,1):
                    pool=[c for c in cc if c['eligible'] and c['delta']==delta and c['direction']==direction];n=sum(c['candidate_id'] in expected['I'] for c in pool)
                    expected['M'].extend(c['candidate_id'] for c in sorted(pool,key=lambda c:(hashes[c['candidate_id']],c['candidate_id']))[:n])
            for arm,ids in expected.items():assert set(ids)==set(m['selected'][arm]) and m['masks'][arm]==[c['candidate_id'] in ids for c in cc]
            masks[m['unit'],m['member']]=m['selected']
    scores=data['native_scores']
    for arm,videos in scores.items():
        for v in videos:
            for slot,m in enumerate(v['members']):
                selected=m['selected_ids']
                if selected is None:
                    assert arm=='O'
                    selected=capacity['units'][v['unit']]['member_masks'][str(slot)]
                if arm!='O':assert set(selected)==set(masks[v['unit'],m['member']][arm])
                graph198.assert_reference(graphs[v['unit']][m['member']],selected,m['metrics'])
    actual=run198.summarize(scores);close(actual,data['summary'])
    for h,domain in data['summary']['domains'].items():
        for arm,method in domain['methods'].items():
            cut_count=sum(len(capacity['units'][v['unit']]['member_masks'][str(i)] if arm=='O' else masks[v['unit'],m['member']][arm]) for v in scores[arm] if v['domain']==h for i,m in enumerate(v['members']))
            assert method['cut_count']==cut_count
    # The first-stage actual O metrics must agree with the final reused O arm.
    for physical in data['physical_O']['videos']:
        final=next(v for v in scores['O'] if (v['domain'],v['group'],v['video'])==(physical['domain'],physical['group'],physical['video']))
        for a,b in zip(physical['members'],final['members']):run198.assert_metrics(a['metrics'],b['metrics'])
    addcap=Fraction(str(posthoc197(reference197)['35']['add_only_E3_lower']))
    # Derive exact registered197 add ceiling from original integer group sums.
    from tools.study197.replay197 import columns
    original_groups=reference197['frozen']['groups'];baseline={v['unit']:v for v in oldscores['K']};bounds={};enumerated=[]
    for g in original_groups:
        if g['domain']!='35':continue
        v=baseline[g['unit']];gt=np.asarray(v['members'][0]['metrics']['gt_bins']);r=(sum(np.asarray(m['metrics']['pred_bins']) for m in v['members'])-3*gt).ravel();cc=[c for m in g['members'] for c in m['bundle']['candidates']];V=columns(cc,g['nf']);A=V[:,[i for i,c in enumerate(cc) if c['delta']==1]];add=A.sum(axis=1);low=int(np.maximum(r,0).sum()+np.maximum(-(r+add),0).sum());bounds[g['group']]=(int(abs(r).sum())-low,90*v['members'][0]['metrics']['nG'])
        import itertools
        minimum=min(int(abs(r+A@np.asarray(bits,dtype=np.int64)).sum()) for bits in itertools.product((0,1),repeat=A.shape[1]))
        enumerated.append(dict(group=g['group'],adds=A.shape[1],configurations=2**A.shape[1],K=int(abs(r).sum()),minimum=minimum,denominator=90*v['members'][0]['metrics']['nG']))
    ceiling=sum(Fraction(100*n,d) for n,d in bounds.values())/len(bounds);necessary=posthoc_drop_necessity35(ceiling)
    enumerated.sort(key=lambda r:r['group']);assert [r['minimum'] for r in enumerated]==[518,432,522,458] and sum(r['configurations'] for r in enumerated)==3072
    k=sum(r['K'] for r in enumerated);minimum=sum(r['minimum'] for r in enumerated);den=sum(r['denominator'] for r in enumerated);assert (k,minimum,den)==(2014,1930,46440)
    curve=[dict(drops=n,gain_ceiling=Fraction(100*(k-minimum+30*n),den)) for n in range(20)]
    assert curve[12]['gain_ceiling']==Fraction(370,387) and curve[18]['gain_ceiling']<Fraction(7,5)<curve[19]['gain_ceiling']
    exact=dict(enumerated=enumerated,at_most12=[370,387],minimum_drops_necessary=19,sufficient=False,curve=[dict(drops=r['drops'],gain_ceiling=[r['gain_ceiling'].numerator,r['gain_ceiling'].denominator]) for r in curve])
    switches={}
    for h in ('35','33','W','G'):
        count={'B_only':0,'I_only':0,'I_add':0,'I_drop':0,'I_only_y_minus1':0,'I_only_z1':0}
        for fold in data['folds'].values():
            for m in fold['masks']['member_clips']:
                if m['hive']!=h:continue
                bs=set(m['selected']['B']);is_=set(m['selected']['I']);count['B_only']+=len(bs-is_);count['I_only']+=len(is_-bs)
                count['I_add']+=sum(records[k]['delta']==1 for k in is_);count['I_drop']+=sum(records[k]['delta']==-1 for k in is_)
                count['I_only_y_minus1']+=sum(labels[k]['y']==-1 for k in is_-bs);count['I_only_z1']+=sum(labels[k]['z']==1 for k in is_-bs)
        switches[h]=count
    assert switches['35']['I_drop']==12 and switches['W']['I_drop']==30 and switches['W']['I_add']==0
    assert switches['W']['I_only']==switches['W']['I_only_y_minus1']==6 and switches['W']['I_only_z1']==5
    return dict(candidates=len(records),old_arms_checked=checked,capacity_certificates=len(capacity['units']),refitted_heads=refit_count,new_arms=len(scores),gates=True,posthoc_35_drop_necessity=necessary,posthoc_35_exact_add_enum=exact,posthoc_I_B=switches)

def self_test():
    # Strict lines: equal to1.4 does not pass; a finite witness needs physical proof.
    rows=[dict(domain='35',K_numerator=100,O_lower=58,O_upper=58,denominator=3000),dict(domain='33',K_numerator=100,O_lower=0,O_upper=0,denominator=3000)]
    a=count_gate(rows,True);assert a['statuses']['35']=='excluded' and not a['proceed_to_training']
    rows[0]['O_lower']=rows[0]['O_upper']=57
    assert not count_gate(rows,False)['proceed_to_training'] and count_gate(rows,True)['proceed_to_training']
    assert posthoc_drop_necessity35()['minimum_drops_necessary']==15
    return dict(synthetic=True,exact_strict_gate=True,physical_witness_required=True)

if __name__=='__main__':
    import argparse,gzip,hashlib,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path,nargs='?');p.add_argument('--reference197',type=Path);a=p.parse_args()
    if a.evidence:
        def read(path):
            op=gzip.open if path.suffix=='.gz' else open
            with op(path,'rt',encoding='utf-8') as f:return json.load(f)
        data=read(a.evidence);reference=a.reference197 or Path(__file__).resolve().parent.parent/'study197/evidence.json.gz'
        assert hashlib.sha256(reference.read_bytes()).hexdigest()==data['reference197_gzip_sha256']
        result=replay(data,read(reference))
        print(json.dumps({k:v for k,v in result.items() if not k.startswith('posthoc_')}|dict(posthoc35_minimum_drops=result['posthoc_35_exact_add_enum']['minimum_drops_necessary'],posthoc_I_B=result['posthoc_I_B'])))
    else:print(json.dumps(self_test()))
