"""Numeric-only endpoint cut replay core; field schema adapter pending seals."""
import hashlib
import itertools
from fractions import Fraction
import numpy as np

def phase(events,nf):
    result=[]
    for offset in range(30):
        rows=[[0,0] for _ in range((nf+25*offset+749)//750)]
        for frame,direction in events:
            assert isinstance(frame,int) and 1<=frame<=nf and direction in (-1,1)
            rows[(frame-1+25*offset)//750][int(direction<0)]+=1
        result.extend(rows)
    return np.asarray(result,dtype=np.int64)

def graph_tp(evidence,key):
    pred={r[0] for r in evidence['predictions']};truth={r[0] for r in evidence['gt']};adj={p:[] for p in pred};owner={}
    for p,g in evidence[key]:assert p in pred and g in truth;adj[p].append(g)
    def visit(p,seen):
        for g in adj[p]:
            if g in seen:continue
            seen.add(g)
            if g not in owner or visit(owner[g],seen):owner[g]=p;return True
        return False
    for p in sorted(pred):visit(p,set())
    return len(owner)

def columns(candidates,nf):
    shape=phase([],nf).shape;out=[]
    assert len({c['candidate_id'] for c in candidates})==len(candidates)
    for c in candidates:
        assert c['delta'] in (-1,1)
        a=c['delta']*phase([(c['frame'],c['direction'])],nf)
        if 'delta_windows' in c:assert np.array_equal(a,c['delta_windows'])
        assert int(abs(a).sum())==30
        out.append(a.ravel())
    return np.column_stack(out) if out else np.zeros((shape[0]*2,0),dtype=np.int64)

def labels(pred_sum,gt,candidates,nf):
    r=(np.asarray(pred_sum)-3*np.asarray(gt)).ravel();v=columns(candidates,nf)
    base=int(abs(r).sum())
    values=[(base-int(abs(r+v[:,i]).sum()))/30 for i in range(len(candidates))]
    assert all(-1<=x<=1 for x in values)
    return values

def verify_oracle(pred_sum,gt,candidates,nf,answer):
    r=(np.asarray(pred_sum)-3*np.asarray(gt)).ravel();v=columns(candidates,nf)
    proof=answer['certificate'];p=proof['p'];q=proof['q']
    assert isinstance(q,int) and q>0 and len(p)==len(r)
    assert all(isinstance(x,int) and abs(x)<=q for x in p)
    linear=sum(int(a)*int(b) for a,b in zip(p,r))
    dots=[sum(int(a)*int(b) for a,b in zip(p,v[:,i])) for i in range(v.shape[1])]
    neg=sum(min(0,x) for x in dots);scaled=linear+neg;lower=-((-scaled)//q)
    expected=dict(linear_term=linear,column_dot_products=dots,negative_part_sum=neg,scaled_numerator=scaled,lower_bound=lower)
    for k,x in expected.items():assert proof[k]==x
    x=answer['x'];assert len(x)==len(candidates) and all(z in (0,1) for z in x)
    witness=np.asarray(pred_sum).ravel()+v@np.asarray(x,dtype=np.int64)
    assert (witness>=0).all() and np.array_equal(witness.reshape(np.asarray(gt).shape),answer['witness_pred_sum'])
    upper=int(abs(r+v@np.asarray(x,dtype=np.int64)).sum())
    assert upper==answer['upper_bound'] and answer['baseline_numerator']==int(abs(r).sum())
    assert answer['selected_ids']==[c['candidate_id'] for c,b in zip(candidates,x) if b]
    certified=max(0,lower)
    if answer.get('enumeration'):
        assert len(candidates)<=22 # explicit bounded exact enumeration only
        hist={}
        for bits in itertools.product((0,1),repeat=len(candidates)):
            cost=int(abs(r+v@np.asarray(bits,dtype=np.int64)).sum());hist[cost]=hist.get(cost,0)+1
        assert min(hist)==upper
        assert {str(k):value for k,value in hist.items()}=={str(k):value for k,value in answer['enumeration']['objective_histogram'].items()}
        certified=upper
    assert certified==answer['lower_bound']<=upper
    assert answer['gap']==upper-certified and answer['optimal']==(certified==upper)
    return dict(lower=certified,upper=upper,gap=upper-certified)

def matched_control(candidates,selected,unit,member_index):
    ids=set(selected);result=[]
    for delta in (-1,1):
        for direction in (-1,1):
            pool=[c for c in candidates if c['delta']==delta and c['direction']==direction]
            n=sum(c['candidate_id'] in ids for c in pool)
            def key(c):
                parent=c.get('parent_canonical_id',c.get('parent_id'))
                text=f'endpoint-null-v1|{unit}|{member_index}|{parent}'
                return hashlib.sha256(text.encode()).hexdigest(),parent
            result.extend(c['candidate_id'] for c in sorted(pool,key=key)[:n])
    return result

def fit_head(records,name):
    dim=9 if name=='G' else 12
    if not records:return dict(mean=[0.]*dim,scale=[1.]*dim,beta=[0.]*dim,intercept=0.,status='empty')
    hives={r['hive'] for r in records};groups={h:{r['group_id'] for r in records if r['hive']==h} for h in hives}
    clips={(h,g):{r['clip'] for r in records if r['hive']==h and r['group_id']==g} for h in hives for g in groups[h]}
    weight=np.asarray([1/(len(hives)*len(groups[r['hive']])*len(clips[r['hive'],r['group_id']])*sum((q['hive'],q['group_id'],q['clip'])==(r['hive'],r['group_id'],r['clip']) for q in records)) for r in records])
    np.testing.assert_allclose(weight,[r['weight'] for r in records],rtol=0,atol=1e-15)
    x=np.asarray([r['features_G9' if name=='G' else 'features_T12'] for r in records]);y=np.asarray([r['y'] for r in records]);mean=weight@x
    constant=np.ptp(x,axis=0)==0;mean[constant]=x[0,constant];scale=np.sqrt(weight@((x-mean)**2));scale[(scale==0)|constant]=1;z=(x-mean)/scale
    if np.ptp(y)==0:beta=np.zeros(dim);intercept=float(y[0]);status='constant'
    else:
        a=np.column_stack((np.ones(len(x)),z));penalty=np.diag([0.]+[.2]*dim)
        coef=np.linalg.solve(a.T@(weight[:,None]*a)+penalty,a.T@(weight*y));intercept=float(coef[0]);beta=coef[1:];status='ridge'
    return dict(mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),intercept=intercept,status=status)

def replay_predictions(frozen,folds):
    allc={c['candidate_id']:c for g in frozen['groups'] for m in g['members'] for c in m['bundle']['candidates']}
    for c in allc.values():
        k=c['k'];d=np.asarray(c['d_normalized']);tau=np.asarray(c['tau'])
        assert k in (1,2) and len(d)==k+3 and len(tau)==k+3
        t=tau[k:];values=d[k:];slope=float(((t-t.mean())*(values-values.mean())).sum()/((t-t.mean())**2).sum());intercept=float(values.mean()-slope*t.mean())
        pred=intercept+slope*tau[:k];fitted=intercept+slope*t
        np.testing.assert_allclose([intercept,slope],c['core_fit_ab'],rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(pred,c['pred_short'],rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(fitted,c['core_fitted'],rtol=1e-8,atol=1e-9)
        assert c['H']==bool(np.all((pred>=0)==c['core_inside']))
        g=c['features_G'];tt=c['features_T'];np.testing.assert_array_equal(tt[:9],g)
        np.testing.assert_allclose(g[:5],[k-1,c['span']/12,float(c['end']=='end'),float(c['short_inside']),float(c['base_has_event'])],rtol=0,atol=1e-15)
        np.testing.assert_allclose(g[-2:],np.clip([abs(d[:k]).mean(),abs(d[k:]).mean()],-4,4),rtol=1e-10,atol=1e-10)
        extra=[abs(d[:k]-pred).mean(),(1 if c['core_inside'] else -1)*pred.mean(),abs(values-fitted).mean()]
        np.testing.assert_allclose(tt[9:],np.clip(extra,-4,4),rtol=1e-8,atol=1e-9)
    for g in frozen['groups']:
        for m in g['members']:
            b=m['bundle'];nf=g['nf']
            event=[(e['frame'],e['direction']) for e in b['baseline_events']]
            assert np.array_equal(phase(event,nf),b['baseline_windows'])
            columns(b['candidates'],nf)
    label_audit={}
    decisions=0
    for h,fold in folds.items():
        saved=fold['model'];records=saved['training_records'];assert saved['heldout']==h and saved['threshold']==.1
        assert {r['key'] for r in records}=={k for k,c in allc.items() if c['hive']!=h}
        for r in records:
            c=allc[r['key']];assert r['hive']!=h and r['hive']==c['hive']
            np.testing.assert_array_equal(r['features_G9'],c['features_G']);np.testing.assert_array_equal(r['features_T12'],c['features_T'])
            assert r['improvement_numerator']/30==r['y'] and -1<=r['y']<=1
            if r['key'] in label_audit:assert label_audit[r['key']]==r['improvement_numerator']
            label_audit[r['key']]=r['improvement_numerator']
        for name in ('G','T'):
            fitted=fit_head(records,name)
            for k,v in fitted.items():
                if k=='status':assert v==saved['heads'][name][k]
                else:np.testing.assert_allclose(v,saved['heads'][name][k],rtol=1e-9,atol=1e-10)
        masks=fold['masks'];assert masks['heldout']==h
        for member in masks['member_clips']:
            candidates=[allc[c['key']] for c in member['candidates']];decisions+=len(candidates)
            expected=dict(K=[],A=[c['candidate_id'] for c in candidates],H=[c['candidate_id'] for c in candidates if c['H']])
            for name in ('G','T'):
                head=saved['heads'][name];x=np.asarray([c['features_'+name] for c in candidates]).reshape(-1,len(head['mean']))
                scores=head['intercept']+((x-head['mean'])/head['scale'])@head['beta']
                np.testing.assert_allclose(scores,member['scores'][name],rtol=1e-9,atol=1e-10)
                expected[name]=[c['candidate_id'] for c,s in zip(candidates,scores) if s>.1]
            expected['M']=matched_control(candidates,expected['T'],member['unit'],member['member_index'])
            for arm,chosen in expected.items():
                assert set(member['selected'][arm])==set(chosen)
                assert member['masks'][arm]==[c['candidate_id'] in chosen for c in candidates]
    assert decisions==len(allc)
    return dict(candidates=len(allc),folds=len(folds),source_labels_consistent=len(label_audit),refit=True,masks=True)

def verify_scores(data):
    native=data['native_scores'];summary=data['summary'];frozen=data['frozen'];folds=data['folds']
    groups={g['unit']:g for g in frozen['groups']};base={v['unit']:v for v in native['K']}
    labelmap={}
    oracle_masks={}
    for proof in summary['oracle_certificates']:
        unit=proof['unit'];g=groups[unit];video=base[unit]
        gt=video['members'][0]['metrics']['gt_bins'];pred=sum(np.asarray(m['metrics']['pred_bins']) for m in video['members'])
        cc=[]
        for mi,m in enumerate(g['members']):
            for c in m['bundle']['candidates']:cc.append(dict(c,candidate_id=f'{mi}|'+c['candidate_id']))
        verify_oracle(pred,gt,cc,g['nf'],proof['oracle'])
        values=labels(pred,gt,cc,g['nf'])
        for c,y in zip(cc,values):labelmap[c['key']]=y
        oracle_masks[unit]={i:[] for i in range(3)}
        for key in proof['oracle']['selected_ids']:
            mi,local=proof['global_to_local'][key];oracle_masks[unit][mi].append(local)
    for fold in folds.values():
        for r in fold['model']['training_records']:assert r['y']==labelmap[r['key']]
    mask_records={(m['unit'],m['member_index']):m for fold in folds.values() for m in fold['masks']['member_clips']}
    for arm,videos in native.items():
        assert len(videos)==len(groups)
        for v in videos:
            g=groups[v['unit']]
            for mi,(m,b) in enumerate(zip(v['members'],base[v['unit']]['members'])):
                metric=m['metrics'];bm=b['metrics'];candidates=g['members'][mi]['bundle']['candidates']
                e=metric['evidence']
                for field,target in (('gt','gt_bins'),('predictions','pred_bins')):
                    events=[(abs(row[1]),1 if row[1]>0 else -1) for row in e[field]]
                    np.testing.assert_array_equal(phase(events,g['nf']),metric[target])
                assert len(e['gt'])==metric['nG'] and len(e['predictions'])==metric['nP']
                for t,key in (('0.5','edges05'),('0.3','edges03')):
                    met=metric['iou'][t];assert graph_tp(e,key)==met['tp'] and (met['pred'],met['gt'])==(metric['nP'],metric['nG'])
                selected=oracle_masks[v['unit']][mi] if arm=='O' else mask_records[v['unit'],mi]['selected'][arm]
                cols=columns(candidates,g['nf']);x=np.asarray([c['candidate_id'] in selected for c in candidates],dtype=np.int64)
                predicted=np.asarray(bm['pred_bins']).ravel()+cols@x
                np.testing.assert_array_equal(predicted.reshape(np.asarray(bm['pred_bins']).shape),metric['pred_bins'])
                np.testing.assert_array_equal(metric['gt_bins'],bm['gt_bins'])
                assert metric['nG']==bm['nG']
    for h,domain in summary['domains'].items():
        cert_groups={}
        for proof in summary['oracle_certificates']:
            g=groups[proof['unit']]
            if g['domain']!=h:continue
            row=cert_groups.setdefault(g['group'],[0,0,0]);a=proof['oracle'];row[0]+=a['lower_bound'];row[1]+=a['upper_bound'];row[2]+=a['denominator']
        bounds=domain['oracle_bounds'];assert len(bounds['groups'])==len(cert_groups)
        for row in bounds['groups']:assert [row['lower'],row['upper'],row['denominator']]==cert_groups[row['group']]
        for side,index in (('lower',0),('upper',1)):
            value=float(sum(Fraction(100*r[index],r[2]) for r in cert_groups.values())/len(cert_groups))
            np.testing.assert_allclose(value,bounds['mean_'+side+'_E3'],rtol=0,atol=1e-10)
        assert bounds['all_optimal']==all(r[0]==r[1] for r in cert_groups.values())
        for arm,method in domain['methods'].items():
            count=sum(len(m['bundle']['candidates']) for g in groups.values() if g['domain']==h for m in g['members'])
            cuts=sum(len(oracle_masks[g['unit']][i] if arm=='O' else mask_records[g['unit'],i]['selected'][arm]) for g in groups.values() if g['domain']==h for i in range(3))
            assert method['candidate_count']==count and method['cut_count']==cuts
            videos=[v for v in native[arm] if v['domain']==h];gids=sorted({v['group'] for v in videos});e3=[];f1={t:[] for t in ('0.5','0.3')}
            for gid,saved in zip(gids,method['groups']):
                vv=[v for v in videos if v['group']==gid];numerator=0;denominator=0
                for v in vv:
                    gt=np.asarray(v['members'][0]['metrics']['gt_bins']);pred=sum(np.asarray(m['metrics']['pred_bins']) for m in v['members'])
                    numerator+=int(abs(pred-3*gt).sum());denominator+=90*v['members'][0]['metrics']['nG']
                assert saved['numerator']==numerator and saved['denominator']==denominator
                e3.append(Fraction(100*numerator,denominator))
                for t in f1:
                    values=[]
                    for i in range(3):
                        metrics=[v['members'][i]['metrics']['iou'][t] for v in vv];tp=sum(m['tp'] for m in metrics);p=sum(m['pred'] for m in metrics);g=sum(m['gt'] for m in metrics)
                        values.append(Fraction(2*tp,max(p+g,1)))
                        pooled=saved['iou'][t]['members'][i];assert (tp,p,g)==(pooled['tp'],pooled['pred'],pooled['gt'])
                    f1[t].append(sum(values)/3)
            np.testing.assert_allclose(float(sum(e3)/len(e3)),method['E3'],rtol=0,atol=1e-10)
            for t in f1:np.testing.assert_allclose(float(sum(f1[t])/len(f1[t])),method['F1'][t],rtol=0,atol=1e-12)
    domains=summary['domains'];change=lambda h:domains[h]['methods']['T']['E3']-domains[h]['methods']['K']['E3']
    checks=dict(head35_gain=change('35') < -1.4,head33_gain=change('33') < -1.3,W_no_harm=change('W')<=1.4,F1_no_harm=all(domains[h]['methods']['T']['F1']['0.5']>=domains[h]['methods']['K']['F1']['0.5']-.005 for h in ('35','33','W')))
    assert checks==summary['investment_checks'] and all(checks.values())==summary['investment_pass']
    attribution=all(domains[h]['methods']['T']['E3']<min(domains[h]['methods'][a]['E3'] for a in ('G','M')) for h in ('35','33'))
    assert attribution==summary['motion_attribution_pass']
    for h,line in (('35',1.4),('33',1.3)):
        excluded=domains[h]['oracle_bounds']['mean_lower_E3']>=domains[h]['methods']['K']['E3']-line
        assert excluded==summary['oracle_family_excluded'][h]
    return dict(source_labels_rebuilt=len(labelmap),oracle_certificates=len(oracle_masks),arms=len(native),physical_count_vectors=True,group_E3_and_member_F1=True,gates=True)

def verify_local_observations(data):
    import cv2
    observations=data['local_observations'];count=0
    for g in data['frozen']['groups']:
        polygon=np.asarray(g['polygon'],dtype=np.float32)
        for m in g['members']:
            for c in m['bundle']['candidates']:
                record=observations[c['candidate_id']];rows=np.asarray(record['window_rows']);k=c['k']
                assert len(rows)==k+3 and (rows[:,4:6]>0).all()
                scale=np.sqrt(rows[:,4]*rows[:,5]).mean();center=rows[:,2:4]+rows[:,4:6]/2
                d=np.asarray([cv2.pointPolygonTest(polygon,(float(x),float(y)),True)/scale for x,y in center]);tau=abs(rows[:,0]-rows[0,0])
                np.testing.assert_array_equal(d,c['d_normalized']);np.testing.assert_array_equal(tau,c['tau'])
                np.testing.assert_allclose(scale,c['distance_scale'],rtol=0,atol=1e-15)
                np.testing.assert_allclose(c['features_G'][5],rows[:k,6].mean()-rows[k:,6].mean(),rtol=0,atol=1e-15)
                np.testing.assert_allclose(c['features_G'][6],np.log1p(record['parent_observations'])/np.log(51),rtol=0,atol=1e-15)
                count+=1
    assert count==len(observations)
    return count

def posthoc(data):
    native={v['unit']:v for v in data['native_scores']['K']};folds=data['folds'];by_member={(m['unit'],m['member_index']):m for fold in folds.values() for m in fold['masks']['member_clips']};totals={}
    for g in data['frozen']['groups']:
        video=native[g['unit']];gt=np.asarray(video['members'][0]['metrics']['gt_bins']);pred=sum(np.asarray(m['metrics']['pred_bins']) for m in video['members']);r=(pred-3*gt).ravel()
        cc=[c for m in g['members'] for c in m['bundle']['candidates']];v=columns(cc,g['nf']);add=v[:,[i for i,c in enumerate(cc) if c['delta']==1]].sum(axis=1)
        # Every window relaxed independently. This is a necessary lower bound,
        # not a coherent attainable choice or an identity attribution.
        upper=r+add;lower=int(np.maximum(r,0).sum()+np.maximum(-upper,0).sum());key=(g['domain'],g['group'])
        row=totals.setdefault(key,dict(add_only_lower=0,denominator=0,T_single_gain=0,M_single_gain=0));row['add_only_lower']+=lower;row['denominator']+=90*video['members'][0]['metrics']['nG']
        yy=labels(pred,gt,cc,g['nf']);byid={c['candidate_id']:y for c,y in zip(cc,yy)}
        for arm in ('T','M'):
            row[arm+'_single_gain']+=sum(round(30*byid[c]) for mi in range(3) for c in by_member[g['unit'],mi]['selected'][arm])
    result={}
    for h in ('35','33','W','G'):
        rr=[r for (domain,_),r in totals.items() if domain==h]
        result[h]=dict(add_only_E3_lower=float(sum(Fraction(100*r['add_only_lower'],r['denominator']) for r in rr)/len(rr)),T_single_gain=float(sum(Fraction(100*r['T_single_gain'],r['denominator']) for r in rr)/len(rr)),M_single_gain=float(sum(Fraction(100*r['M_single_gain'],r['denominator']) for r in rr)/len(rr)))
    return result

def replay(data):
    r=replay_predictions(data['frozen'],data['folds']);r.update(verify_scores(data));r['local_features']=verify_local_observations(data);r['posthoc']=posthoc(data)
    assert abs(r['posthoc']['35']['add_only_E3_lower']-3.84151593453919)<1e-8
    return r

def self_test():
    candidates=[dict(candidate_id='a',frame=25,direction=1,delta=-1,parent_id=1),dict(candidate_id='b',frame=25,direction=1,delta=1,parent_id=2)]
    gt=phase([(25,1)],75);pred=3*gt
    assert labels(pred,gt,candidates,75)==[-1.,-1.]
    v=columns(candidates,75);assert np.array_equal(v[:,0],-v[:,1])
    assert len(matched_control(candidates,['a'],'synthetic',0))==1
    return dict(synthetic=True,clock=True,singleton_labels=True,signed_columns=True,matched_quantity=True)

if __name__=='__main__':
    import argparse,gzip,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path,nargs='?');a=p.parse_args()
    if a.evidence:
        op=gzip.open if a.evidence.suffix=='.gz' else open
        with op(a.evidence,'rt',encoding='utf-8') as f:data=json.load(f)
        print(json.dumps(replay(data)))
    else:print(json.dumps(self_test()))
