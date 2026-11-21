"""Private §199 D/DU capacity and joint four-group minimumprotected-S cuts.

NoF1 constraint in either countoptimization. F1 is evaluated only on the same
concrete originalbinary-mask witness using previously validated fullgraphs.
Exact integerdual bounds,not floating solverflags,decide optimality.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import warnings
from numbers import Integral
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,linprog,milp
from scipy.sparse import block_diag,csr_matrix,eye,hstack,vstack
sys.path.insert(0,str(Path.cwd()))
from tools.study197.oracle import solve_clip,make_columns,phase_counts
from tools.study198.graph198 import evaluate

PREREG='530ee05ab8401e4313a4fc53e50318d2a37fe74e'


def ceil_fraction(x):return -((-x.numerator)//x.denominator)


def protected_certificate(r,V,cost,budget,p,g,Q):
    """ceil((p.r-g*budget+sum min(0,Q*cost_i+p.v_i))/Q).

    p/Q is a residualscore andg/Q≥||p/Q||∞. L1≤budget implies
    p.(r+Vx)≤g*budget; addingcost andminimizing each binary term provesLB.
    """
    def integer(v):
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral):raise ValueError('exactcertificate requires integerinput')
        return int(v)
    r=list(map(integer,r));p=list(map(integer,p));cost=list(map(integer,cost));g=integer(g);Q=integer(Q);budget=integer(budget)
    if Q<=0 or g<0 or len(p)!=len(r) or any(abs(a)>g for a in p):raise ValueError('invalid exactdual multipliers')
    matrix=np.asarray(V);n=matrix.shape[1]
    if not np.issubdtype(matrix.dtype,np.integer):raise ValueError('exactintegerV required')
    if matrix.shape!=(len(r),len(cost)) or any(c not in(0,1) for c in cost):raise ValueError('binaryS costs required')
    linear=sum(a*b for a,b in zip(p,r));dots=[]
    for i in range(n):dots.append(Q*cost[i]+sum(p[j]*int(matrix[j,i]) for j in range(len(r))))
    negative=sum(min(0,v) for v in dots);numerator=linear-g*int(budget)+negative
    lower=-((-numerator)//Q)
    return dict(p=p,g=g,Q=Q,budget=int(budget),linear=linear,adjusted_column_dots=dots,
                negative_sum=negative,scaled_bound=numerator,lower_bound=max(0,lower))


def verify_protected_certificate(r,V,cost,budget,proof):
    assert protected_certificate(r,V,cost,budget,proof['p'],proof['g'],proof['Q'])==proof
    return True


def classify(c):
    label=c['labels']['0.5']['class']
    if label not in('S','D','U'):raise ValueError('formal0.5S/D/U class required')
    return label


def member_masks(unit,selected):
    chosen=set(selected)
    return {str(m):[c['candidate_id'] for c in unit['candidates'] if c['member_index']==m and c['candidate_id'] in chosen]
            for m in unit['member_indices']}


def graph_witness(unit,selected,models):
    if len(selected)!=len(set(selected)) or not set(selected)<=set(c['candidate_id'] for c in unit['candidates']):
        raise ValueError('unique known originalcandidate IDs required')
    masks=member_masks(unit,selected);scored={}
    for member in unit['member_indices']:
        name=str(member);graph_key=unit.get('member_graph_keys',{}).get(name,name);model=models[unit['unit']][graph_key]
        scored[name]=evaluate(model,masks[name])
    bins=sum(np.asarray(v['pred_bins'],np.int64) for v in scored.values())
    gt=np.asarray(unit['gt_bins'],np.int64)
    assert all(np.array_equal(np.asarray(v['gt_bins']),gt) for v in scored.values())
    chosen=[c for c in unit['candidates'] if c['candidate_id'] in set(selected)]
    expected=np.asarray(unit['pred_sum'],np.int64).ravel()+make_columns(chosen,unit['nf']).sum(axis=1)
    assert np.array_equal(bins.ravel(),expected)
    return dict(member_masks=masks,members=scored,pred_sum=bins.tolist(),numerator=int(np.abs(bins-3*gt).sum()))


def restricted(unit,family,models=None):
    allowed={'D'} if family=='O_D' else {'D','U'} if family=='O_DU' else None
    if allowed is None:raise ValueError('familyO_D/O_DU required')
    candidates=[c for c in unit['candidates'] if classify(c) in allowed]
    oracle=solve_clip(unit['pred_sum'],unit['gt_bins'],candidates,unit['nf'])
    report=dict(oracle=oracle,member_masks=member_masks(unit,oracle['selected_ids']),eligible_count=len(candidates))
    if models is not None:
        witness=graph_witness(unit,oracle['selected_ids'],models)
        assert witness['pred_sum']==oracle['witness_pred_sum'] and witness['numerator']==oracle['upper_bound']
        report['graph_witness']=witness
    return report


def _construct_head(units):
    if len(units)!=4 or len({u['group'] for u in units})!=4:raise ValueError('head requires exactly4oneclipgroups')
    r=[];matrices=[];candidates=[];location=[];denominators=[];base=0
    for unit in units:
        pred=np.asarray(unit['pred_sum'],np.int64);gt=np.asarray(unit['gt_bins'],np.int64)
        residual=(pred-3*gt).ravel();r.extend(residual.tolist());base+=int(np.abs(residual).sum())
        matrices.append(csr_matrix(make_columns(unit['candidates'],unit['nf'])))
        candidates.extend(unit['candidates']);location.extend([unit['unit']]*len(unit['candidates']))
        denominators.append(3*int(gt.sum()))
    if len(set(denominators))!=1 or not denominators[0]:raise ValueError('fourheadgroup denominators must be equal andpositive')
    V=block_diag(matrices,format='csr').toarray().astype(np.int64)
    cost=np.array([classify(c)=='S' for c in candidates],np.int64)
    return np.asarray(r,np.int64),V,cost,candidates,location,denominators,base


def minimum_S(units,line,models=None,time_limit=60):
    """Joint whole-head optimization,not sum of four individualminima."""
    r,V,cost,candidates,locations,denominators,base=_construct_head(units)
    line=F(str(line));threshold=F(base)-line*sum(denominators)/100;budget=ceil_fraction(threshold)-1
    proof=protected_certificate(r,V,cost,budget,[0]*len(r),0,1)
    upper=None;best=None;messages=[];count_proofs=[]
    # The unprotectedDU witness may establish minS=0 immediately.
    for family in ('DU','full'):
        x=[];lower_count=0;upper_count=0
        for unit in units:
            pool=[c for c in unit['candidates'] if family=='full' or classify(c)!='S']
            answer=solve_clip(unit['pred_sum'],unit['gt_bins'],pool,unit['nf'])
            selected=set(answer['selected_ids']);x.extend(int(c['candidate_id'] in selected) for c in unit['candidates'])
            lower_count+=answer['lower_bound'];upper_count+=answer['upper_bound']
            if family=='full':count_proofs.append(dict(unit=unit['unit'],oracle=answer))
        if upper_count<=budget:
            value=int(cost@np.asarray(x));upper=value;best=x
            if value==0:break
        if family=='full' and lower_count>budget:
            return dict(feasible=False,infeasibility_proved=True,reason='unrestricted countcertificate exceeds strictbudget',
                        budget=budget,full_count_lower=lower_count,full_count_certificates=count_proofs,
                        protected_lower=None,protected_upper=None,optimal=False)
    if upper!=0:
        # Aggregate identical countcolumn,S cost andunit. No other memberquota
        # is imposed; originalcandidatefamily alreadyhas one switch/parent.
        grouped={}
        for i,(candidate,unit) in enumerate(zip(candidates,locations)):
            key=(unit,(candidate['frame']-1)//25,candidate['direction'],candidate['delta'],int(cost[i]))
            grouped.setdefault(key,[]).append(i)
        indices=list(grouped.values());cap=np.array(list(map(len,indices)),np.int64)
        U=V[:,[ids[0] for ids in indices]];groupcost=cost[[ids[0] for ids in indices]];m,n=U.shape
        A=vstack([hstack([csr_matrix(U),-eye(m)]),hstack([-csr_matrix(U),-eye(m)]),
                  csr_matrix(np.r_[np.zeros(n),np.ones(m)][None,:])],format='csr')
        rhs=np.r_[-r,r,budget];objective=np.r_[groupcost,np.zeros(m)]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
            lp=linprog(objective,A_ub=A,b_ub=rhs,bounds=[(0,int(v)) for v in cap]+[(0,None)]*m,
                       method='highs-ds',options=dict(time_limit=time_limit,parallel=False))
        messages.append('LP:'+lp.message)
        if lp.success:
            Q=1_000_000;y=lp.ineqlin.marginals[m:2*m]-lp.ineqlin.marginals[:m]
            if np.isfinite(y).all() and np.isfinite(lp.ineqlin.marginals[-1]):
                p=[int(round(float(v)*Q)) for v in y]
                g=max(max(map(abs,p),default=0),int(np.ceil(-float(lp.ineqlin.marginals[-1])*Q)))
                proof=protected_certificate(r,V,cost,budget,p,g,Q)
        incumbent=lp.x[:n] if lp.x is not None else None
        if incumbent is None or np.max(np.abs(incumbent-np.rint(incumbent)),initial=0)>1e-7:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
                mip=milp(objective,integrality=np.r_[np.ones(n),np.zeros(m)],
                    bounds=Bounds(np.zeros(n+m),np.r_[cap,np.full(m,np.inf)]),
                    constraints=LinearConstraint(A,-np.inf,rhs),options=dict(time_limit=time_limit,mip_rel_gap=0.,parallel=False))
            messages.append('MILP:'+mip.message);incumbent=mip.x[:n] if mip.x is not None else None
        if incumbent is not None and np.max(np.abs(incumbent-np.rint(incumbent)),initial=0)<=1e-7:
            q=np.rint(incumbent).astype(int)
            if (q>=0).all() and (q<=cap).all():
                x=np.zeros(len(candidates),np.int64)
                for ids,k in zip(indices,q):x[ids[:int(k)]]=1
                count=int(np.abs(r+V@x).sum());value=int(cost@x)
                if count<=budget and (upper is None or value<upper):upper=value;best=x.tolist()
    verify_protected_certificate(r,V,cost,budget,proof)
    lower=proof['lower_bound'];result=dict(budget=budget,strict_threshold=[threshold.numerator,threshold.denominator],
        K_numerator=base,total_denominator=sum(denominators),protected_lower=lower,protected_upper=upper,
        gap=upper-lower if upper is not None else None,optimal=upper is not None and lower==upper,
        feasible=True if upper is not None else None,infeasibility_proved=False,certificate=proof,solver_messages=messages,
        interpretation='minimumS for strictcountgain only;noF1 constraint,only this concretewitness evaluated')
    if best is not None:
        x=np.asarray(best,np.int64);count=int(np.abs(r+V@x).sum())
        assert count<=budget and int(cost@x)==upper and lower<=upper
        gain=F(100*(base-count),sum(denominators));assert gain>line
        masks={u['unit']:[] for u in units}
        for c,u,b in zip(candidates,locations,x):
            if b:masks[u].append(c['candidate_id'])
        result.update(x=best,count_numerator=count,attained_gain_fraction=[gain.numerator,gain.denominator],
                      unit_selected_ids=masks,unit_member_masks={u['unit']:member_masks(u,masks[u['unit']]) for u in units})
        if models is not None:
            result['graph_witness']={u['unit']:graph_witness(u,masks[u['unit']],models) for u in units}
            assert sum(g['numerator'] for g in result['graph_witness'].values())==count
    return result


def _pooled_member_F1(graphs):
    slots={}
    for witness in graphs:
        for member,metric in witness['members'].items():slots.setdefault(member,[]).append(metric)
    result={}
    for threshold in ('0.5','0.3'):
        values=[]
        for member,clips in slots.items():
            tp=sum(c['iou'][threshold]['tp'] for c in clips);den=sum(c['nP']+c['nG'] for c in clips)
            values.append(F(2*tp,den) if den else None)
        result[threshold]=float(sum(values)/len(values)) if values and all(v is not None for v in values) else None
    return result


def summarize(units,reports):
    groups={};hives={}
    for unit in units:groups.setdefault((unit['hive'],unit['group']),[]).append(unit)
    group_report={}
    for(hive,gid),rows in groups.items():
        denominator=sum(3*int(np.asarray(u['gt_bins']).sum()) for u in rows)
        data=dict(hive=hive,group=gid,denominator=denominator,families={})
        for family in ('K','O_D','O_DU'):
            if family=='K':
                low=high=sum(reports[u['unit']]['baseline_graph']['numerator'] for u in rows)
                graphs=[reports[u['unit']]['baseline_graph'] for u in rows]
            else:
                low=sum(reports[u['unit']]['families'][family]['oracle']['lower_bound'] for u in rows)
                high=sum(reports[u['unit']]['families'][family]['oracle']['upper_bound'] for u in rows)
                graphs=[reports[u['unit']]['families'][family]['graph_witness'] for u in rows]
            data['families'][family]=dict(lower=low,upper=high,gap=high-low,optimal=low==high,
                W_lower=float(F(100*low,denominator)) if denominator else None,
                W_witness=float(F(100*high,denominator)) if denominator else None,F1=_pooled_member_F1(graphs))
        group_report[f'{hive}/{gid}']=data
    for hive in sorted({key[0] for key in groups}):
        rows=[g for g in group_report.values() if g['hive']==hive];result={}
        for family in ('K','O_D','O_DU'):
            low=sum(F(100*g['families'][family]['lower'],g['denominator']) for g in rows)/len(rows)
            high=sum(F(100*g['families'][family]['upper'],g['denominator']) for g in rows)/len(rows)
            gain_lower=sum(F(100*(g['families']['K']['upper']-g['families'][family]['upper']),g['denominator']) for g in rows)/len(rows)
            gain_upper=sum(F(100*(g['families']['K']['upper']-g['families'][family]['lower']),g['denominator']) for g in rows)/len(rows)
            line={'35':F('1.4'),'33':F('1.3')}.get(hive)
            result[family]=dict(mean_W_lower=float(low),mean_W_witness=float(high),
                gain_lower_fraction=[gain_lower.numerator,gain_lower.denominator],gain_upper_fraction=[gain_upper.numerator,gain_upper.denominator],
                witness_exceeds_line=gain_lower>line if line is not None else None,
                excluded_by_lower=gain_upper<=line if line is not None else None,
                mean_F1={threshold:float(np.mean([g['families'][family]['F1'][threshold] for g in rows]))
                         if all(g['families'][family]['F1'][threshold] is not None for g in rows) else None for threshold in ('0.5','0.3')})
        hives[hive]=result
    return dict(groups=group_report,hives=hives)


def normalize_owners(shards):
    """Ownercatalogue normalizedcontract; no label inference done here."""
    units=[]
    for hive,shard in shards.items():
        for raw in shard.get('units',shard.get('groups',[])):
            candidates=[];members=[];graph_keys={}
            for member in raw['members']:
                index=member['member_index'];members.append(index);graph_keys[str(index)]=str(member['member'])
                candidates.extend(dict(c,member_index=index) for c in member['records'])
            if len(members)!=3 or len(set(members))!=3:raise ValueError('exactly3original members required')
            pred=raw.get('pred_sum',raw.get('Ksum3bins'));gt=raw.get('gt_bins',raw.get('gt_windows'))
            if pred is None or gt is None:raise ValueError('sealed baseline/GT numericbins required')
            for c in candidates:classify(c)
            if any(c['labels'].get('0.3',{}).get('class') not in('S','D','U') for c in candidates):
                raise ValueError('0.3 sensitivitycensus labels required;not used to changeformal capacities')
            units.append(dict(unit=raw['unit'],hive=str(hive),group=raw['group'],video=raw['video'],nf=raw['nf'],
                              pred_sum=pred,gt_bins=gt,candidates=candidates,member_indices=members,member_graph_keys=graph_keys))
    if len({u['unit'] for u in units})!=len(units):raise ValueError('duplicateunit')
    return units


def run_cli(args):
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest();load=lambda path:json.loads(Path(path).read_text())
    assert args.prereg==PREREG and sha(args.owners/'manifest.json')==args.owners_seal==(args.owners/'SEALED.sha256').read_text().strip()
    assert sha(args.graphs)==args.graphs_sha
    manifest=load(args.owners/'manifest.json');shards={}
    for hive in ('35','33','W','G'):
        name=hive+'.json';assert sha(args.owners/name)==manifest[name];shards[hive]=load(args.owners/name)
    units=normalize_owners(shards)
    if len(units)!=20 or sum(len(u['candidates']) for u in units)!=1196:raise ValueError('original20units/1196 support required')
    models=load(args.graphs)['models'];args.out.mkdir(parents=True,exist_ok=True)
    identity=dict(prereg=PREREG,owners_seal=args.owners_seal,graphs_sha=args.graphs_sha,code_sha256=sha(__file__),time_limit=args.time_limit)
    def checkpoint(path,value):
        text=json.dumps(value,separators=(',',':'),sort_keys=True,allow_nan=False)+'\n'
        if path.exists():
            if path.read_text()!=text:raise ValueError('resume checkpoint mismatch:'+str(path))
        else:
            with path.open('x') as stream:stream.write(text)
        if path.name!='manifest.json':
            seal=path.with_name(path.name+'.sha256')
            if seal.exists():assert seal.read_text().strip()==sha(path)
            else:seal.write_text(sha(path)+'\n')
    def checked_load(path):
        assert path.with_name(path.name+'.sha256').read_text().strip()==sha(path)
        return load(path)
    checkpoint(args.out/'inputs.json',identity);reports={}
    for unit in units:
        path=args.out/('unit_'+hashlib.sha256(unit['unit'].encode()).hexdigest()[:16]+'.json')
        if path.exists():
            saved=checked_load(path);assert saved['input_receipt']==identity and saved['unit']==unit['unit'];reports[unit['unit']]=saved
        else:
            report=dict(unit=unit['unit'],input_receipt=identity,
                classes={label:sum(classify(c)==label for c in unit['candidates']) for label in ('S','D','U')},
                migration_05_to_03={a:{b:sum(classify(c)==a and c['labels']['0.3']['class']==b for c in unit['candidates']) for b in ('S','D','U')} for a in ('S','D','U')},
                baseline_graph=graph_witness(unit,[],models),
                families={family:restricted(unit,family,models) for family in ('O_D','O_DU')})
            checkpoint(path,report);reports[unit['unit']]=report
        print('CAPACITY',unit['unit'],flush=True)
    protected={}
    for hive,line in (('35',1.4),('33',1.3)):
        path=args.out/f'minS_{hive}.json'
        if path.exists():
            record=checked_load(path);assert record['input_receipt']==identity;protected[hive]=record['result']
        else:
            result=minimum_S([u for u in units if u['hive']==hive],line,models,time_limit=args.time_limit)
            checkpoint(path,dict(input_receipt=identity,result=result));protected[hive]=result
        print('PROTECTED_MINIMUM',hive,protected[hive].get('protected_lower'),protected[hive].get('protected_upper'),flush=True)
    result=dict(input_receipt=identity,units=reports,minimum_S=protected,**summarize(units,reports),
        caveat='Count-only optimization;F1 is this actual graphwitness,not anF1-constrained bound. Physical validation follows separately.')
    checkpoint(args.out/'capacity199.json',result)
    files={p.name:sha(p) for p in args.out.iterdir() if p.is_file() and p.name not in ('manifest.json','SEALED.sha256')}
    checkpoint(args.out/'manifest.json',files);seal=sha(args.out/'manifest.json')
    if (args.out/'SEALED.sha256').exists():assert (args.out/'SEALED.sha256').read_text().strip()==seal
    else:(args.out/'SEALED.sha256').write_text(seal+'\n')
    print(json.dumps(dict(hives=result['hives'],protected={h:{k:v.get(k) for k in ('feasible','optimal','protected_lower','protected_upper','gap')} for h,v in protected.items()},seal=seal),indent=2))


def selfcheck():
    import itertools
    # Fourindependent oneclipgroups,oneS addition andonefreeD addition each.
    units=[]
    for i in range(4):
        c=[dict(candidate_id=f'{i}S',frame=2,direction=1,delta=1,member_index=0,labels={'0.5':{'class':'S'}}),
           dict(candidate_id=f'{i}D',frame=2,direction=1,delta=1,member_index=1,labels={'0.5':{'class':'D'}})]
        units.append(dict(unit=f'u{i}',hive='35',group=i,video=f'v{i}',nf=3,pred_sum=phase_counts([(2,1)],3).tolist(),
                          gt_bins=phase_counts([(2,1)],3).tolist(),candidates=c,member_indices=[0,1,2]))
    for line in (0,35,50):
        result=minimum_S(units,line)
        r,V,cost,*_=_construct_head(units)
        feasible=[(int(cost@x),int(np.abs(r+V@x).sum())) for bits in itertools.product((0,1),repeat=8)
                  for x in [np.asarray(bits)] if int(np.abs(r+V@x).sum())<=result['budget']]
        assert feasible and result['optimal'] and result['protected_upper']==min(v[0] for v in feasible)
        verify_protected_certificate(r,V,cost,result['budget'],result['certificate'])
    for unit in units:
        assert restricted(unit,'O_D')['oracle']['optimal']
        assert restricted(unit,'O_DU')['oracle']['optimal']
    impossible=minimum_S(units,70)
    assert impossible['infeasibility_proved'] and not impossible['feasible']
    rng=np.random.default_rng(199);random_checks=0
    for _ in range(5):
        mixed=[]
        for unit in units:
            candidates=[dict(c,delta=sign,labels={'0.5':{'class':str(rng.choice(['S','D','U']))}})
                        for c,sign in zip(unit['candidates'],(-1,1))]
            mixed.append(dict(unit,candidates=candidates,pred_sum=phase_counts([(2,1)]*int(rng.integers(1,6)),3).tolist()))
        result=minimum_S(mixed,10)
        r,V,cost,*_=_construct_head(mixed)
        feasible=[int(cost@x) for bits in itertools.product((0,1),repeat=8) for x in [np.asarray(bits)]
                  if int(np.abs(r+V@x).sum())<=result['budget']]
        if feasible:assert result['optimal'] and result['protected_upper']==min(feasible)
        else:assert result['infeasibility_proved']
        random_checks+=1
    print('Synthetic D/DU,jointstrictbudget,minS0/1/3,exactdual,exhaustive,infeasible and',random_checks,'mixed-sign cases passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--owners',type=Path);parser.add_argument('--owners-seal')
    parser.add_argument('--graphs',type=Path);parser.add_argument('--graphs-sha');parser.add_argument('--out',type=Path)
    parser.add_argument('--prereg');parser.add_argument('--time-limit',type=float,default=60)
    args=parser.parse_args();selfcheck()
    if args.owners:
        assert args.owners_seal and args.graphs and args.graphs_sha and args.out and args.prereg
        run_cli(args)
