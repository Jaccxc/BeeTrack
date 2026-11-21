"""Source-only §199 replay primitives; no field files opened by default."""
from fractions import Fraction
import numpy as np

def owner_class(left,right,threshold=.5):
    """Validate annotated threshold degrees, not regenerate full-frame geometry.

    OriginalGT input identity is explicit, never canonicalID modulo100000.
    Reverse GT-neighbor completeness needs native/independent graph evidence.
    """
    accepted=[]
    for row in (left,right):
        neighbors=row['graph_gt'];assert len(neighbors)==row['pred_degree']
        assert all(n['iou']>=threshold for n in neighbors)
        owner=row['official_owner'];hits=[n for n in neighbors if n['canonical_gt_id']==owner]
        matched=owner is not None
        assert bool(row['matched'])==matched
        if matched:
            assert len(hits)==1 and row['iou']>=threshold and hits[0]['gt_degree']==row['gt_degree']
            assert hits[0]['original_gt_id']==row['owner_gt_original_id']
        qualified=matched and row['pred_degree']==1 and row['gt_degree']==1 and row['owner_gt_original_id'] is not None
        assert row['qualified']==qualified and row['isolated']==qualified
        accepted.append(qualified)
    if not all(accepted):return 'U'
    if left['official_owner']==right['official_owner']:return 'S'
    if left['owner_gt_original_id']!=right['owner_gt_original_id']:return 'D'
    return 'U'

def protected_certificate(residual,V,cost,budget,proof):
    r=[int(x) for x in residual];matrix=np.asarray(V);p=proof['p'];g=proof['g'];Q=proof['Q']
    assert isinstance(g,int) and g>=0 and isinstance(Q,int) and Q>0
    assert len(p)==len(r) and all(isinstance(x,int) and abs(x)<=g for x in p)
    assert matrix.shape==(len(r),len(cost)) and np.issubdtype(matrix.dtype,np.integer) and all(c in (0,1) for c in cost)
    linear=sum(a*b for a,b in zip(p,r));dots=[Q*cost[i]+sum(p[j]*int(matrix[j,i]) for j in range(len(r))) for i in range(len(cost))]
    negative=sum(min(0,x) for x in dots);scaled=linear-g*budget+negative;lower=max(0,-((-scaled)//Q))
    expected=dict(p=p,g=g,Q=Q,budget=budget,linear=linear,adjusted_column_dots=dots,negative_sum=negative,scaled_bound=scaled,lower_bound=lower)
    assert expected==proof
    return lower

def auroc(labels,scores):
    positive=[s for y,s in zip(labels,scores) if y==1];negative=[s for y,s in zip(labels,scores) if y==0]
    if not positive or not negative:return None
    wins=sum(Fraction(1) if p>n else Fraction(1,2) if p==n else Fraction(0) for p in positive for n in negative)
    return float(wins/(len(positive)*len(negative)))

def classification(classes,cut,scores=None):
    assert len(classes)==len(cut) and set(classes)<={'D','S','U'}
    d=sum(x=='D' for x in classes);s=sum(x=='S' for x in classes)
    tp=sum(c and y=='D' for c,y in zip(cut,classes));fp=sum(c and y=='S' for c,y in zip(cut,classes))
    recall=tp/d if d else None;wrong=fp/s if s else None
    result=dict(D=d,S=s,U=sum(x=='U' for x in classes),D_cut=tp,S_cut=fp,D_recall=recall,S_false_split=wrong,balanced_accuracy=(recall+1-wrong)/2 if d and s else None)
    if scores is not None:
        known=[(int(y=='D'),v) for y,v in zip(classes,scores) if y in ('D','S') and v is not None]
        result['AUROC']=auroc([p[0] for p in known],[p[1] for p in known]) if len(known)==d+s else None
    return result

def protected_same_cut_budget35(protected_minimum=1466,K=2014,total_denominator=46440,witness_count=1352,witness_S=4):
    """Count proof only; link inputs to sealed integer cert/witness in adapter."""
    line=Fraction(7,5);threshold=Fraction(K)-line*total_denominator/100
    budget=-((-threshold.numerator)//threshold.denominator)-1
    assert budget==1363 and protected_minimum-3*30==1376>budget
    lower=next(n for n in range(100) if protected_minimum-30*n<=budget)
    assert lower==4 and witness_S==4 and witness_count<=budget
    gain=Fraction(100*(K-witness_count),total_denominator);assert gain>line
    error=Fraction(100*witness_count,total_denominator)
    return dict(strict_budget=budget,protected_minimum=protected_minimum,three_S_best_possible=protected_minimum-90,
        minimum_S_lower=lower,witness_S_upper=witness_S,minimum_S_exact=lower,
        witness_numerator=witness_count,witness_E3=[error.numerator,error.denominator],gain=[gain.numerator,gain.denominator],
        scope='registered multi-member candidate cuts, not independent bees or a deployment prescription',physical_validation_pending=True)

def guarded_minimum_same(lower,witness_same,witness_mean_F1,K_mean_F1):
    """Posthoc feasible witness closes an unchanged valid lower bound.

    Caller must derive the two mean F1 values from the SAME physical masks,
    pooling clips per member before equal member/group averaging, at IoU.5.
    This imposes a hive-mean guard only, never each-seed or IoU.3 guards.
    """
    assert isinstance(lower,int) and lower>=0 and witness_same==lower
    assert Fraction(witness_mean_F1)>=Fraction(K_mean_F1)
    return dict(minimum_same_cuts=lower,posthoc=True,F1_condition='hive mean official IoU.5 >= originalK',
                F1_maximized=False,each_seed_guard=False,IoU03_guard=False)

def distinct_same_runs(minimum_S_lower,witness_same_rows,*,count_line_passed,F1_mean_guard_passed):
    """Logical corollary only; exact originalGT run identity, not modulo IDs."""
    assert isinstance(minimum_S_lower,int) and minimum_S_lower>=0
    assert count_line_passed and F1_mean_guard_passed
    keys={(r['clip'],r['original_gt_id'],r['run']) for r in witness_same_rows}
    assert len(witness_same_rows)>=minimum_S_lower
    lower=int(minimum_S_lower>0);upper=len(keys)
    assert lower<=upper
    return dict(distinct_lower=lower,witness_distinct_upper=upper,
        exact_minimum=lower if lower==upper else None,posthoc=True,
        necessity_of_this_specific_GT_run='undetermined',independent_bees=False)

def censor_run_counts(rows):
    """Count only supported cases; deduplicate across seeds/members/cuts.

    Type is deliberately NOT part of the GT-run key: a run with two types
    remains one observed annotated run. Threshold/scopes must be fixed first.
    """
    result={}
    for hive in ('35','33','W','G'):
        cases=[r for r in rows if r['hive']==hive and r['supported']]
        keys={(r['clip'],r['original_gt_id'],r['run']) for r in cases}
        result[hive]=dict(observations=len(cases),distinct_annotated_runs=len(keys))
    return result

def close(a,b):
    if isinstance(a,dict):
        for k,v in a.items():assert k in b,k;close(v,b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,float):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
    else:assert a==b

def refit_head(rows,name):
    dim={'N':11,'B':15,'I':22}[name]
    if not rows:return dict(mean=[0.]*dim,scale=[1.]*dim,beta=[0.]*dim,intercept=.5,status='empty')
    keys=[(r['hive'],r['group_id'],r['clip'],r['delta']) for r in rows]
    hs={k[0] for k in keys};gg={h:{k[1] for k in keys if k[0]==h} for h in hs};cc={(h,g):{k[2] for k in keys if k[:2]==(h,g)} for h in hs for g in gg[h]};dd={k[:3]:{q[3] for q in keys if q[:3]==k[:3]} for k in keys}
    w=np.asarray([1/(len(hs)*len(gg[h])*len(cc[h,g])*len(dd[h,g,c])*keys.count((h,g,c,d))) for h,g,c,d in keys]);np.testing.assert_allclose(w,[r['weight'] for r in rows],rtol=0,atol=1e-15)
    x=np.asarray([r[f'features_{name}{dim}'] for r in rows]);y=np.asarray([r['identity_class']=='D' for r in rows],float);assert np.array_equal(y,[r['y'] for r in rows])
    mean=w@x;constant=np.ptp(x,axis=0)==0;mean[constant]=x[0,constant];scale=np.sqrt(w@((x-mean)**2));scale[(scale==0)|constant]=1;z=(x-mean)/scale
    if np.ptp(y)==0:intercept=float(y[0]);beta=np.zeros(dim);status='constant'
    else:
        a=np.column_stack((np.ones(len(x)),z));coef=np.linalg.solve(a.T@(w[:,None]*a)+np.diag([0.]+[.2]*dim),a.T@(w*y));intercept=float(coef[0]);beta=coef[1:];status='ridge'
    return dict(mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),intercept=intercept,status=status)

def classification_table(records,selected,scores=None):
    selected=set(selected);counts={c:sum(r['class']==c for r in records) for c in ('D','S','U')};cuts={c:sum(r['class']==c and r['candidate_id'] in selected for r in records) for c in counts}
    recall=cuts['D']/counts['D'] if counts['D'] else None;false=cuts['S']/counts['S'] if counts['S'] else None;labels=[];values=[];missing=0
    for r in records:
        if r['class']=='U':continue
        value=dict(pD=float(r['candidate_id'] in selected),abstain=False) if scores is None else scores.get(r['candidate_id'])
        if value is None or value.get('abstain') or value.get('pD') is None:missing+=1;continue
        labels.append(int(r['class']=='D'));values.append(value['pD'])
    directional={}
    for d in (-1,1):
        for q in (-1,1):
            rr=[r for r in records if r['delta']==d and r['direction']==q]
            directional[f'{d}/{q}']=dict(total=len(rr),cuts=sum(r['candidate_id'] in selected for r in rr),classes={c:sum(r['class']==c for r in rr) for c in counts},cut_classes={c:sum(r['class']==c and r['candidate_id'] in selected for r in rr) for c in counts})
    return dict(total=len(records),class_denominators=counts,cut_classes=cuts,cut_count=len(selected),D_recall=recall,S_false_split_rate=false,U_cut_count=cuts['U'],U_cut_rate=cuts['U']/counts['U'] if counts['U'] else None,balanced_accuracy=(recall+1-false)/2 if recall is not None and false is not None else None,AUROC=auroc(labels,values) if not missing else None,AUROC_score_type='continuous_pD' if scores is not None else 'binary_mask',known_score_count=len(values),known_abstentions=missing,by_delta_direction=directional)

def replay(data,reference198):
    import hashlib,sys
    from pathlib import Path
    repo=Path.cwd()
    if not (repo/'beetrack/p0.py').is_file():repo=Path(__file__).resolve().parents[2]
    assert (repo/'beetrack/p0.py').is_file();sys.path.insert(0,str(repo))
    from tools.study197.replay197 import columns,verify_oracle
    from tools.study198 import graph198,run198
    import score199
    peer={r['candidate_id']:r for g in reference198['peer']['groups'] for m in g['members'] for r in m['records']};graphs=reference198['graph_data']['models'];units={u['unit']:u for shard in data['owners'].values() for u in shard['units']};owners={c['candidate_id']:c for u in units.values() for m in u['members'] for c in m['records']}
    assert owners.keys()==peer.keys() and len(owners)==1196
    censored=[]
    for key,c in owners.items():
        for t in ('0.5','0.3'):assert owner_class(c['labels'][t]['left'],c['labels'][t]['right'],float(t))==c['labels'][t]['class']
        for field in ('frame','direction','delta','delta_windows'):close(c[field],peer[key][field])
        z=c['labels']['0.5']['censor']
        if z['supported']:
            assert c['delta']==-1 and c['labels']['0.5']['class']=='S';rr=z['rows'];assert len(rr)==4 and all(r['qualified'] for r in rr) and len({r['official_owner'] for r in rr})==1
            gt=z['gt_run'];assert gt['start_inside']==gt['end_inside'];assert rr[0]['pred_inside']==rr[0]['gt_inside'] and rr[1]['pred_inside']==rr[1]['gt_inside']
            if z['type']=='later_return':assert z['return_observation']['frame']>z['p_last_frame'] and z['return_observation']['inside']==gt['start_inside']==z['p_start_inside']
            elif z['type']=='earlier_part_missed':assert z['earlier_observation']['frame']<z['p_first_frame'] and z['earlier_observation']['inside']==gt['start_inside']==z['p_end_inside']
            else:raise AssertionError('unknown supported censor type')
            censored.append(dict(hive=c['hive'],clip=c['clip'],original_gt_id=gt['original_gt_id'],run=gt['run'],supported=True))
    census=censor_run_counts(censored);assert [census[h]['observations'] for h in ('35','33','W','G')]==[0,40,11,9] and [census[h]['distinct_annotated_runs'] for h in ('35','33','W','G')]==[0,15,4,4]
    capacity=data['capacity'];restricted_masks={}
    for key,u in units.items():
        cc=[c for m in u['members'] for c in m['records']];restricted_masks[key]={}
        for family in ('O_D','O_DU'):
            pool=[c for c in cc if c['labels']['0.5']['class']=='D' or family=='O_DU' and c['labels']['0.5']['class']=='U'];answer=capacity['units'][key]['families'][family]
            verify_oracle(u['pred_sum'],u['gt_bins'],pool,u['nf'],answer['oracle']);assert answer['eligible_count']==len(pool)
            restricted_masks[key][family]=answer['member_masks']
    masks={};probabilities={a:{} for a in ('H_N','H_B','H_I')};refitted=0
    oldI={(m['unit'],m['member']):m['selected']['I'] for f in reference198['folds'].values() for m in f['masks']['member_clips']}
    for h,fold in data['folds'].items():
        model=fold['model'];training=model['training_records'];expected={k for k,c in owners.items() if c['hive']!=h and peer[k]['eligible'] and c['labels']['0.5']['class'] in ('D','S')};assert {r['key'] for r in training}==expected
        for r in training:
            c=owners[r['key']];assert r['identity_class']==c['labels']['0.5']['class']
            for name,dim in (('N',11),('B',15),('I',22)):close(r[f'features_{name}{dim}'],peer[r['key']][f'features_{name}{dim}'])
        for name in ('N','B','I'):close(refit_head(training,name),model['heads'][name]);refitted+=1
        for m in fold['masks']['member_clips']:
            ids=m['candidate_ids'];expected_masks={a:[] for a in ('K','I198','H_N','H_B','H_I','M')};hashes={cid:hashlib.sha256(('endpoint-identity-null-v1|'+cid).encode()).hexdigest() for cid in ids}
            assert m['M_hashes']==[hashes[cid] for cid in ids]
            for i,cid in enumerate(ids):
                assert m['eligible'][i]==peer[cid]['eligible']
                if cid in oldI[m['unit'],m['member']]:expected_masks['I198'].append(cid)
                for name,dim in (('N',11),('B',15),('I',22)):
                    arm='H_'+name;saved=m['scores'][arm][i]
                    if not peer[cid]['eligible']:assert saved is None;continue
                    head=model['heads'][name];raw=float(head['intercept']+((np.asarray(peer[cid][f'features_{name}{dim}'])-head['mean'])/head['scale'])@head['beta']);p=float(np.clip(raw,0,1));abstain=head['status']=='empty';close(dict(raw=raw,pD=p,abstain=abstain),saved);probabilities[arm][cid]=saved
                    if p>.5 and not abstain:expected_masks[arm].append(cid)
            for d in (-1,1):
                for q in (-1,1):
                    pool=[cid for cid in ids if peer[cid]['eligible'] and peer[cid]['delta']==d and peer[cid]['direction']==q];n=sum(cid in expected_masks['H_I'] for cid in pool);expected_masks['M'].extend(sorted(pool,key=lambda cid:(hashes[cid],cid))[:n])
            for a,chosen in expected_masks.items():assert set(chosen)==set(m['selected'][a]) and m['masks'][a]==[cid in chosen for cid in ids]
            masks[m['unit'],m['member']]=expected_masks
    for arm,videos in data['native_scores'].items():
        for v in videos:
            u=units[v['unit']]
            for slot,m in enumerate(v['members']):
                chosen=restricted_masks[v['unit']][arm][str(slot)] if arm in ('O_D','O_DU') else masks[v['unit'],m['member']][arm]
                assert set(chosen)==set(m['selected_ids']);graph198.assert_reference(graphs[v['unit']][m['member']],chosen,m['metrics'])
    actual=score199.summarize(data['native_scores'],run198.combine);close(actual,data['summary'])
    byid={r['candidate_id']:r for r in data['classification_inputs']}
    assert byid.keys()==owners.keys()
    for key,r in byid.items():
        c=owners[key];assert (r['class'],r['class03'])==(c['labels']['0.5']['class'],c['labels']['0.3']['class']);assert r['eligible']==peer[key]['eligible'] and r['censor_supported']==c['labels']['0.5']['censor']['supported']
        for field in ('delta','direction','frame','unit','member','member_index','domain'):assert r[field]==c[field]
        for arm in data['native_scores']:
            selected=restricted_masks[c['unit']][arm][str(c['member_index'])] if arm in ('O_D','O_DU') else masks[c['unit'],c['member']][arm]
            assert r['selected'][arm]==(key in selected)
        for a in probabilities:close(probabilities[a].get(key),r['probabilities'][a])
    for h in ('35','33','W','G'):
        for arm,videos in data['native_scores'].items():
            selected=[cid for v in videos if v['domain']==h for m in v['members'] for cid in m['selected_ids']]
            assert data['summary']['domains'][h]['methods'][arm]['cut_count']==len(selected)
            for scope in ('all_candidates','deployment_eligible'):
                rr=[r for r in byid.values() if r['domain']==h and (scope=='all_candidates' or r['eligible'])];choose=[cid for cid in selected if cid in {r['candidate_id'] for r in rr}]
                close(classification_table(rr,choose,probabilities.get(arm)),data['summary']['classification'][h][arm][scope])
                migration={}
                for r in rr:
                    key=r['class']+'->'+r['class03'];migration[key]=migration.get(key,0)+1
                close(migration,data['summary']['taxonomy_migration'][h][scope])
                for direction in (-1,1):
                    ss=[r for r in rr if r['class']=='S' and r['delta']==-1 and r['direction']==direction];types={}
                    for r in ss:
                        if r['censor_supported']:types[r['censor_type']]=types.get(r['censor_type'],0)+1
                    close(dict(S_deletion_denominator=len(ss),supported_types=types,unconfirmed=sum(not r['censor_supported'] for r in ss)),data['summary']['observed_truncation'][h][scope][str(direction)])
        comparison=data['summary']['restricted_capacity'][h]
        baseline=sum(Fraction(100*g['numerator'],g['denominator']) for g in data['summary']['domains'][h]['methods']['K']['groups'])/len(data['summary']['domains'][h]['methods']['K']['groups'])
        for family in ('O_D','O_DU'):
            bounds={}
            for key,u in units.items():
                if u['domain']!=h:continue
                a=capacity['units'][key]['families'][family]['oracle'];r=bounds.setdefault(str(u['group']),dict(lower=0,upper=0,denominator=0))
                for k,f in (('lower','lower_bound'),('upper','upper_bound'),('denominator','denominator')):r[k]+=a[f]
            low=sum(Fraction(100*r['lower'],r['denominator']) for r in bounds.values())/len(bounds);high=sum(Fraction(100*r['upper'],r['denominator']) for r in bounds.values())/len(bounds);line={'35':Fraction(7,5),'33':Fraction(13,10)}.get(h)
            close(dict(groups=bounds,mean_lower_E3=float(low),mean_witness_E3=float(high),guaranteed_gain=float(baseline-high),maximum_possible_gain=float(baseline-low),all_optimal=low==high,witness_crosses_line=baseline-high>line if line is not None else None,excluded_by_lower_bound=baseline-low<=line if line is not None else None),comparison[family])
    head_results={};same_rows={}
    for h,line in (('35',Fraction(7,5)),('33',Fraction(13,10))):
        uu=data['owners'][h]['units'];residual=[];vv=[];cc=[]
        for u in uu:
            residual.extend((np.asarray(u['pred_sum'])-3*np.asarray(u['gt_bins'])).ravel().tolist());c=[c for m in u['members'] for c in m['records']];vv.append(columns(c,u['nf']));cc.extend(c)
        V=np.zeros((sum(v.shape[0] for v in vv),sum(v.shape[1] for v in vv)),dtype=np.int64);i=j=0
        for v in vv:V[i:i+v.shape[0],j:j+v.shape[1]]=v;i+=v.shape[0];j+=v.shape[1]
        cost=[int(c['labels']['0.5']['class']=='S') for c in cc];answer=capacity['minimum_S'][h];den=sum(3*int(np.asarray(u['gt_bins']).sum()) for u in uu);K=sum(abs(int(v)) for v in residual);threshold=Fraction(K)-line*den/100;budget=-((-threshold.numerator)//threshold.denominator)-1
        assert answer['budget']==budget and answer['K_numerator']==K and answer['total_denominator']==den
        lower=protected_certificate(residual,V,cost,budget,answer['certificate']);x=answer['x'];count=int(abs(np.asarray(residual)+V@np.asarray(x,dtype=np.int64)).sum());upper=sum(c*b for c,b in zip(cost,x));assert all(b in (0,1) for b in x) and count<=budget
        assert (lower,upper,count)==(answer['protected_lower'],answer['protected_upper'],answer['count_numerator']) and answer['gap']==upper-lower and answer['optimal']==(upper==lower)
        chosen={c['candidate_id'] for c,b in zip(cc,x) if b};same_rows[h]=[];head_videos=[]
        for u in uu:
            members=[]
            for slot,m in enumerate(u['members']):
                selected=answer['unit_member_masks'][u['unit']][str(slot)];assert set(selected)=={c['candidate_id'] for c in m['records'] if c['candidate_id'] in chosen}
                entry=next(e for e in data['head_native'] if e['request']['arm']=='min_S_'+h and e['request']['unit']==u['unit'] and e['request']['member']==m['member']);assert set(entry['request']['selected_ids'])==set(selected);graph198.assert_reference(graphs[u['unit']][m['member']],selected,entry['metrics']);members.append(dict(member=m['member'],metrics=entry['metrics']))
                for c in m['records']:
                    if c['candidate_id'] in selected and c['labels']['0.5']['class']=='S':
                        l=c['labels']['0.5']['left'];r=c['labels']['0.5']['right'];assert l['official_owner']==r['official_owner'];same_rows[h].append(dict(clip=c['clip'],original_gt_id=l['owner_gt_original_id'],run=l['owner_gt_run'],group=c['group'],event_frame=c['frame'],delta=c['delta'],direction=c['direction'],left_date=l['pred_key'][0],right_date=r['pred_key'][0],end=peer[c['candidate_id']]['end'],k=c['k'],censor_reason=c['labels']['0.5']['censor']['reason']))
            head_videos.append(dict(unit=u['unit'],domain=h,group=u['group'],video=u['video'],members=members))
        groups=[run198.combine([v]) for v in head_videos];saved=data['summary']['head_min_S'][h];close(groups,saved['groups']);assert sum(g['numerator'] for g in groups)==count and saved['S_selected']==upper
        mean=lambda gg:sum((sum(Fraction(2*m['tp'],max(m['pred']+m['gt'],1)) for m in g['iou']['0.5']['members'])/3 for g in gg),Fraction())/len(gg)
        witness_F1=mean(groups);K_F1=mean(data['summary']['domains'][h]['methods']['K']['groups']);head_results[h]=guarded_minimum_same(lower,upper,witness_F1,K_F1);head_results[h]['distinct']=distinct_same_runs(lower,same_rows[h],count_line_passed=count<=budget,F1_mean_guard_passed=witness_F1>=K_F1)
    assert head_results['35']['distinct']['exact_minimum']==1 and head_results['33']['distinct']['exact_minimum']==0
    four=same_rows['35'];assert len(four)==4 and sorted(r['group'] for r in four)==[2,3,4,4]
    assert all((r['original_gt_id'],r['run'],r['left_date'],r['right_date'],r['event_frame'],r['delta'],r['direction'],r['end'],r['k'])==(239,0,3411,3412,3412,-1,-1,'start',1) for r in four)
    assert all(r['censor_reason']=='parent_endpoints_and_cut_not_one_isolated_GT_run' for r in four)
    noS=sum(row['families']['O_DU']['oracle']['lower_bound'] for key,row in capacity['units'].items() if units[key]['domain']=='35');assert noS==1466;protected_same_cut_budget35(noS)
    requests=data['physical_requests'];assert len(requests)==216 and len({r['path'] for r in requests})==115 and sum(r['within_job_cache'] for r in requests)==101
    return dict(owner_candidates=len(owners),restricted_certificates=40,head_S_proofs=2,refitted_heads=refitted,new_arms=len(data['native_scores']),head_physical_scores=len(data['head_native']),gates=True,minimum_same=head_results,censor=census,four_same_cuts=four)

def self_test():
    def row(owner,original,run=0):return dict(graph_gt=[dict(canonical_gt_id=owner,original_gt_id=original,iou=.8,gt_degree=1)],pred_degree=1,gt_degree=1,official_owner=owner,owner_gt_original_id=original,owner_gt_run=run,iou=.8,matched=True,qualified=True,isolated=True)
    assert owner_class(row(100005,7),row(100005,7))=='S'
    assert owner_class(row(100005,7),row(200005,7,1))=='U'
    assert owner_class(row(100005,7),row(100006,8))=='D'
    assert auroc([1,0],[.5,.5])==.5 and auroc([1,1],[.2,.7]) is None
    x=classification(['D','S','U'],[True,False,True],[.9,.1,.5]);assert x['balanced_accuracy']==1 and x['AUROC']==1
    assert classification(['D','S'],[True,False],[.9,None])['AUROC'] is None
    assert protected_same_cut_budget35()['minimum_S_exact']==4
    proof=dict(p=[1],g=1,Q=1,budget=0,linear=1,adjusted_column_dots=[0],negative_sum=0,scaled_bound=1,lower_bound=1)
    assert protected_certificate([1],np.asarray([[-1]],dtype=int),[1],0,proof)==1
    assert guarded_minimum_same(4,4,Fraction('0.845872503'),Fraction('0.844455376'))['minimum_same_cuts']==4
    assert guarded_minimum_same(0,0,Fraction('0.506891176'),Fraction('0.503949494'))['minimum_same_cuts']==0
    witness=[dict(clip='BEE24-35-head',original_gt_id=239,run=0,group=g,delta=-1,direction=-1) for g in (2,3,4,4)]
    assert distinct_same_runs(4,witness,count_line_passed=True,F1_mean_guard_passed=True)['exact_minimum']==1
    assert distinct_same_runs(0,[],count_line_passed=True,F1_mean_guard_passed=True)['exact_minimum']==0
    return dict(synthetic=True,explicit_original_identity=True,ambiguous_gap_U=True,AUROC_ties=True,undefined_single_class=True)

if __name__=='__main__':
    import argparse,gzip,hashlib,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path,nargs='?');p.add_argument('--reference197',type=Path);p.add_argument('--reference198',type=Path);a=p.parse_args()
    if a.evidence:
        def read(path):
            op=gzip.open if path.suffix=='.gz' else open
            with op(path,'rt',encoding='utf-8') as f:return json.load(f)
        data=read(a.evidence);parent=Path(__file__).resolve().parent.parent
        r197=a.reference197 or parent/'study197/evidence.json.gz';r198=a.reference198 or parent/'study198/evidence.json.gz'
        for year,path in ((197,r197),(198,r198)):assert hashlib.sha256(path.read_bytes()).hexdigest()==data[f'reference{year}_gzip_sha256']
        reference=read(r198);assert reference['reference197_gzip_sha256']==data['reference197_gzip_sha256']
        for key in list(reference):
            if key not in ('peer','graph_data','folds'):reference.pop(key)
        for fold in reference['folds'].values():fold.pop('model',None)
        result=replay(data,reference)
        print(json.dumps({k:v for k,v in result.items() if k!='four_same_cuts'}))
    else:print(json.dumps(self_test()))
