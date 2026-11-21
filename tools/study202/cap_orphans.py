"""Source/synthetic conflict-aware orphan-continuation countcapacity.

K selects noedge. Cuts andextensions share parentcapacity1; extensions share
orphancapacity1. Eachcut uses its unique dummyorphan. Sparse integercolumns
are whole-action changes shared by all30phases. No fixed199ownership F1 graph
is used:new full-MOT witnesses must be physically scored separately.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import hashlib
import json
from numbers import Integral
from pathlib import Path
import warnings
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,linprog,milp
from scipy.sparse import coo_matrix,csr_matrix,eye,hstack,vstack
from matching_bound import min_cost_matching,vertex_dual

FAMILIES=('E','DE','PE','ALL')


def integer(v):
    if isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral):raise ValueError('integerrequired')
    return int(v)


def phase_shape(nf):
    nf=integer(nf)
    if nf<1:raise ValueError('positive fullclip nf required')
    return (sum((nf+25*o+749)//750 for o in range(30)),2)


def sparse_delta(action,nf):
    size=2*phase_shape(nf)[0];values={}
    if 'delta_events' in action:
        for event in action['delta_events']:
            f,d,k=(integer(event[name]) for name in ('frame','direction','delta'))
            if not 1<=f<=nf or d not in(-1,1):raise ValueError('invalid datedeventchange')
            offset_index=0
            for o in range(30):
                row=offset_index+(f-1+25*o)//750;index=2*row+(d<0)
                values[index]=values.get(index,0)+k
                offset_index+=(nf+25*o+749)//750
    elif 'v_indices' in action:
        if len(action['v_indices'])!=len(action['v_values']):raise ValueError('sparseindices/values mismatch')
        for index,value in zip(action['v_indices'],action['v_values']):
            index,value=integer(index),integer(value)
            if not 0<=index<size:raise ValueError('sparsewindowindex outsideclip')
            values[index]=values.get(index,0)+value
    else:
        raw=np.asarray(action['delta_windows'])
        if raw.shape!=phase_shape(nf) or not np.issubdtype(raw.dtype,np.integer):raise ValueError('integer30phase delta_windows required')
        values={int(i):int(raw.ravel()[i]) for i in np.flatnonzero(raw.ravel())}
    return {i:v for i,v in sorted(values.items()) if v}


def allowed(action,family):
    if family not in FAMILIES:raise ValueError('unknownfamily')
    kind=action['kind'];label=action.get('labels',{}).get('0.5',{}).get('class')
    if kind=='cut':return family=='ALL' or family=='DE' and label=='D' or family=='PE' and label in('D','U')
    if kind not in('extension','continuation','extend'):raise ValueError('cut/extension kindrequired')
    return family=='ALL' or label=='S'


def compile_problem(unit,family):
    shape=phase_shape(unit['nf']);pred=np.asarray(unit['pred_sum']);gt=np.asarray(unit['gt_bins'])
    for a in (pred,gt):
        if a.shape!=shape or not np.issubdtype(a.dtype,np.integer) or (a<0).any():raise ValueError('nonnegative integer30phase binsrequired')
    if int(gt.sum())%30:raise ValueError('GTfullclip census invalid')
    pool=[a for a in unit['actions'] if allowed(a,family)]
    ids=[a['action_id'] for a in pool]
    if len(set(ids))!=len(ids):raise ValueError('uniqueaction IDsrequired')
    cut_orphans=[a['orphan_key'] for a in pool if a['kind']=='cut']
    real_orphans={a['orphan_key'] for a in pool if a['kind']!='cut'}
    if len(set(cut_orphans))!=len(cut_orphans) or set(cut_orphans)&real_orphans:
        raise ValueError('cuts require distinctdummy orphans disjointfrom realorphans')
    ownership={};parent_alias={}
    for action in pool:
        for role,key in (('parent',action['parent_key']),('orphan',action['orphan_key'])):
            previous=ownership.setdefault((role,key),action['member_index'])
            if previous!=action['member_index']:raise ValueError('row sources fromdifferent members cannot share resources')
        if 'parent_canonical_id' in action:
            key=(action['member_index'],action['parent_canonical_id'])
            prior=parent_alias.setdefault(key,action['parent_key'])
            if prior!=action['parent_key']:raise ValueError('cutandextension forsameparent mustshare the same capacitykey')
    # Removing zero-delta actions is safe for the count optimum:leaving thematK
    # frees capacities. This does NOT assert they cannot changeidentity/F1.
    columns=[sparse_delta(a,unit['nf']) for a in pool]
    inactive=[a['action_id'] for a,col in zip(pool,columns) if not col]
    pairs=[(a,col) for a,col in zip(pool,columns) if col];actions=[a for a,_ in pairs];columns=[col for _,col in pairs]
    parents=list(dict.fromkeys(a['parent_key'] for a in actions));orphans=list(dict.fromkeys(a['orphan_key'] for a in actions))
    parent_index={v:i for i,v in enumerate(parents)};orphan_index={v:i for i,v in enumerate(orphans)}
    edges=[dict(edge_id=a['action_id'],recipient=parent_index[a['parent_key']],orphan=orphan_index[a['orphan_key']]) for a in actions]
    rr=[];cc=[];vv=[]
    for j,col in enumerate(columns):
        for i,value in col.items():rr.append(i);cc.append(j);vv.append(value)
    V=coo_matrix((np.array(vv,np.int64),(rr,cc)),shape=(pred.size,len(actions))).tocsc()
    rows=[];cols=[]
    for j,e in enumerate(edges):rows.extend([e['recipient'],len(parents)+e['orphan']]);cols.extend([j,j])
    A=coo_matrix((np.ones(len(rows),np.int64),(rows,cols)),shape=(len(parents)+len(orphans),len(actions))).tocsr()
    return dict(r=(pred-3*gt).ravel().astype(np.int64),pred=pred,gt=gt,actions=actions,columns=columns,
                V=V,capacity=A,parents=parents,orphans=orphans,edges=edges,inactive=inactive,pool_count=len(pool))


def witness(problem,ids):
    if len(ids)!=len(set(ids)):raise ValueError('matchingID repeats')
    lookup={a['action_id']:i for i,a in enumerate(problem['actions'])}
    if not set(ids)<=set(lookup):raise ValueError('unknownorzero action inwitness')
    x=np.zeros(len(lookup),np.int64)
    for key in ids:x[lookup[key]]=1
    if np.any(problem['capacity']@x>1):raise ValueError('parent/orphan reused')
    changed=np.asarray(problem['V']@x,np.int64).ravel()
    pred=problem['pred'].ravel()+changed
    if np.any(pred<0):raise ValueError('countvector givesnegative prediction;actionfamily physicallyinvalid')
    return int(np.abs(problem['r']+changed).sum()),pred.reshape(problem['pred'].shape).tolist(),x


def certificate(problem,p,q,a,b,linear_limit=200):
    p=list(map(integer,p));q=integer(q);a=list(map(integer,a));b=list(map(integer,b))
    if q<=0 or len(p)!=len(problem['r']) or any(abs(v)>q for v in p) or min(a+b,default=0)<0:
        raise ValueError('invalidintegerduals')
    if len(a)!=len(problem['parents']) or len(b)!=len(problem['orphans']):raise ValueError('vertexdualshape mismatch')
    costs=[sum(p[i]*v for i,v in col.items()) for col in problem['columns']]
    # Repair upward,never assume floatingpoint LPduals are a certificate.
    for e,c in zip(problem['edges'],costs):a[e['recipient']]=max(a[e['recipient']],-c-b[e['orphan']])
    selected=None
    if len(costs)<=linear_limit:
        chosen,_=min_cost_matching(problem['edges'],costs)
        exact_a,exact_b=vertex_dual(problem['edges'],costs,chosen)
        aa=[exact_a[i] for i in range(len(a))];bb=[exact_b[i] for i in range(len(b))]
        if sum(aa)+sum(bb)<=sum(a)+sum(b):a,b=aa,bb;selected=[problem['actions'][i]['action_id'] for i in chosen]
    assert all(c+a[e['recipient']]+b[e['orphan']]>=0 for e,c in zip(problem['edges'],costs))
    linear=sum(v*int(r) for v,r in zip(p,problem['r']));scaled=linear-sum(a)-sum(b)
    return dict(p=p,q=q,a=a,b=b,edge_dot_products=costs,linear_term=linear,scaled_bound=scaled,
                lower_bound=max(0,-((-scaled)//q)),minimum_linear_matching_ids=selected)


def verify_certificate(problem,proof):
    p=list(map(integer,proof['p']));q=integer(proof['q']);a=list(map(integer,proof['a']));b=list(map(integer,proof['b']))
    assert q>0 and len(p)==len(problem['r']) and all(abs(v)<=q for v in p)
    assert len(a)==len(problem['parents']) and len(b)==len(problem['orphans']) and min(a+b,default=0)>=0
    costs=[sum(p[i]*v for i,v in col.items()) for col in problem['columns']]
    assert costs==proof['edge_dot_products']
    assert all(c+a[e['recipient']]+b[e['orphan']]>=0 for c,e in zip(costs,problem['edges']))
    linear=sum(v*int(r) for v,r in zip(p,problem['r']));scaled=linear-sum(a)-sum(b)
    assert linear==proof['linear_term'] and scaled==proof['scaled_bound']
    assert max(0,-((-scaled)//q))==proof['lower_bound']
    if proof['minimum_linear_matching_ids'] is not None:
        selected=proof['minimum_linear_matching_ids'];lookup={a['action_id']:i for i,a in enumerate(problem['actions'])}
        assert len(selected)==len(set(selected)) and set(selected)<=set(lookup)
        x=np.zeros(len(lookup),np.int64)
        for key in selected:x[lookup[key]]=1
        assert np.all(problem['capacity']@x<=1)
        assert sum(costs[lookup[key]] for key in selected)==-sum(a)-sum(b)
    return True


def solve_unit(unit,family,warm_starts=(),lp_limit=10,mip_limit=30,linear_limit=200):
    problem=compile_problem(unit,family);r=problem['r'];V=problem['V'];C=problem['capacity'];m,n=V.shape
    baseline,pred,_=witness(problem,[]);best=baseline;best_ids=[];messages=[]
    known={a['action_id'] for a in problem['actions']};aliases={}
    for a in problem['actions']:
        aliases[a['action_id']]=a['action_id']
        if a['kind']=='cut':aliases[a.get('cut_id',a.get('candidate_id',a['action_id'].removeprefix('cut:')))]=a['action_id']
        else:aliases[a.get('edge_id',a['action_id'])]=a['action_id']
    for proposed in warm_starts:
        ids=list(dict.fromkeys(aliases[key] for key in proposed if key in aliases))
        value,_,_=witness(problem,ids)
        if value<best:best=value;best_ids=ids
    proof=certificate(problem,[int(v>0)-int(v<0) for v in r],1,[0]*len(problem['parents']),[0]*len(problem['orphans']),linear_limit)
    if n:
        A=vstack([hstack([V,-eye(m)]),hstack([-V,-eye(m)]),hstack([C,csr_matrix((C.shape[0],m))])],format='csr')
        rhs=np.r_[-r,r,np.ones(C.shape[0])];objective=np.r_[np.zeros(n),np.ones(m)]
        bounds=[(0,1)]*n+[(0,None)]*m
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
            lp=linprog(objective,A_ub=A,b_ub=rhs,bounds=bounds,method='highs-ds',options=dict(time_limit=lp_limit,parallel=False))
        messages.append('LP:'+lp.message)
        if lp.success:
            q=1_000_000;y=lp.ineqlin.marginals[m:2*m]-lp.ineqlin.marginals[:m]
            pp=[max(-q,min(q,int(round(float(v)*q)))) for v in y]
            cover=-lp.ineqlin.marginals[2*m:]
            aa=[max(0,int(np.ceil(float(v)*q))) for v in cover[:len(problem['parents'])]]
            bb=[max(0,int(np.ceil(float(v)*q))) for v in cover[len(problem['parents']):]]
            proposal=certificate(problem,pp,q,aa,bb,linear_limit)
            if proposal['lower_bound']>proof['lower_bound']:proof=proposal
        solutions=[]
        if lp.x is not None:solutions.append(lp.x[:n])
        for x in solutions:
            if np.max(np.abs(x-np.rint(x)),initial=0)<=1e-7:
                ids=[a['action_id'] for a,b in zip(problem['actions'],np.rint(x)) if b]
                try:value,_,_=witness(problem,ids)
                except ValueError:continue
                if value<best:best=value;best_ids=ids
        if proof['lower_bound']<best and mip_limit>0:
            # A knownfeasible warmobjective bounds z without discarding optimum.
            AA=vstack([A,csr_matrix(np.r_[np.zeros(n),np.ones(m)][None,:])],format='csr')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
                mip=milp(objective,integrality=np.r_[np.ones(n),np.zeros(m)],
                    bounds=Bounds(np.zeros(n+m),np.r_[np.ones(n),np.full(m,np.inf)]),
                    constraints=LinearConstraint(AA,-np.inf,np.r_[rhs,best]),
                    options=dict(time_limit=mip_limit,mip_rel_gap=0.,parallel=False))
            messages.append('MILP:'+mip.message)
            if mip.x is not None and np.max(np.abs(mip.x[:n]-np.rint(mip.x[:n])),initial=0)<=1e-7:
                ids=[a['action_id'] for a,b in zip(problem['actions'],np.rint(mip.x[:n])) if b]
                value,_,_=witness(problem,ids)
                if value<best:best=value;best_ids=ids
    verify_certificate(problem,proof);value,pred,x=witness(problem,best_ids)
    assert proof['lower_bound']<=value==best
    masks={str(member):dict(cuts=[],edges=[]) for member in unit['member_indices']}
    for action,bit in zip(problem['actions'],x):
        if bit:
            member=str(action['member_index']);is_cut=action['kind']=='cut'
            masks[member]['cuts' if is_cut else 'edges'].append(action.get('cut_id',action.get('candidate_id',action['action_id'].removeprefix('cut:'))) if is_cut else action.get('edge_id',action['action_id']))
    denominator=3*int(problem['gt'].sum())
    return dict(family=family,baseline_numerator=baseline,lower_bound=proof['lower_bound'],upper_bound=best,
        gap=best-proof['lower_bound'],optimal=best==proof['lower_bound'],denominator=denominator,
        W_lower=100*proof['lower_bound']/denominator if denominator else None,W_witness=100*best/denominator if denominator else None,
        selected_action_ids=best_ids,member_masks=masks,witness_pred_sum=pred,certificate=proof,
        parent_keys=problem['parents'],orphan_keys=problem['orphans'],action_order=[a['action_id'] for a in problem['actions']],
        original_family_actions=problem['pool_count'],solver_active_actions=n,zero_delta_count=len(problem['inactive']),
        sparse_nonzeros=int(V.nnz),solver_messages=messages,
        F1='Must score this new completeMOT;199 fixedownership graphs are not valid for extensions')


def summarize(units,results):
    from fractions import Fraction as F
    grouped={}
    for unit in units:grouped.setdefault((str(unit['hive']),unit['group']),[]).append(unit)
    groups={};heads={}
    for(hive,gid),rows in grouped.items():
        data={}
        for family in FAMILIES:
            cases=[results[u['unit']][family] for u in rows];den=sum(c['denominator'] for c in cases)
            lower=sum(c['lower_bound'] for c in cases);upper=sum(c['upper_bound'] for c in cases);base=sum(c['baseline_numerator'] for c in cases)
            data[family]=dict(lower=lower,upper=upper,baseline=base,denominator=den,gap=upper-lower,optimal=lower==upper)
        groups[f'{hive}/{gid}']=data
    for hive in ('35','33'):
        rows=[data for key,data in groups.items() if key.split('/')[0]==hive];head={}
        if not rows:continue
        for family in FAMILIES:
            if not all(data[family]['denominator']>0 for data in rows):raise ValueError('wholegroup zeroGT normalizedcapacity undefined')
            gainlo=sum(F(100*(data[family]['baseline']-data[family]['upper']),data[family]['denominator']) for data in rows)/len(rows)
            gainhi=sum(F(100*(data[family]['baseline']-data[family]['lower']),data[family]['denominator']) for data in rows)/len(rows)
            head[family]=dict(witness_gain_fraction=[gainlo.numerator,gainlo.denominator],gain_ceiling_fraction=[gainhi.numerator,gainhi.denominator],
                witness_gain=float(gainlo),gain_ceiling=float(gainhi),all_optimal=all(data[family]['optimal'] for data in rows))
        heads[hive]=head
    return dict(groups=groups,heads=heads)


def normalize_catalogue(data):
    """Normalizedunits can be supplied directly;ownerunits may nestmembers."""
    units=[]
    for row in data.get('units',data.get('groups',[])):
        hive=str(row.get('hive',row.get('domain')))
        if hive not in('35','33'):continue
        if 'actions' in row:
            actions=row['actions'];members=row['member_indices']
        else:
            actions=[];members=[]
            for member in row['members']:
                slot=member['member_index'];members.append(slot)
                actions.extend(dict(a,member_index=slot) for a in member['cuts'])
                actions.extend(dict(a,member_index=slot) for a in member['edges'])
        if len(members)!=3 or len(set(members))!=3:raise ValueError('threeoriginalmember sources required')
        units.append(dict(unit=row['unit'],hive=hive,group=row['group'],video=row['video'],nf=row['nf'],
            pred_sum=row['pred_sum'],gt_bins=row['gt_bins'],actions=actions,member_indices=members))
    if len(units)!=8 or len({u['unit'] for u in units})!=8:raise ValueError('exactlythe eightfixedhead units required')
    if any(sum(u['hive']==h for u in units)!=4 or len({u['group'] for u in units if u['hive']==h})!=4 for h in ('35','33')):
        raise ValueError('exactlyfour originalgroups perhead required')
    return units


def run_cli(args):
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest();load=lambda path:json.loads(Path(path).read_text())
    assert sha(args.input)==args.sha256
    data=load(args.input);assert data['prereg']==args.prereg
    units=normalize_catalogue(data);args.out.mkdir(parents=True,exist_ok=True)
    receipt=dict(prereg=args.prereg,input_sha256=args.sha256,code_sha256=sha(__file__),matching_code_sha256=sha(Path(__file__).with_name('matching_bound.py')),
                 lp_limit=args.lp_limit,mip_limit=args.mip_limit,all_limit=args.all_limit,linear_limit=args.linear_limit)
    def save(path,value):
        text=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
        if path.exists():
            if path.read_text()!=text:raise ValueError('immutablecheckpoint mismatch:'+str(path))
        else:
            with path.open('x') as stream:stream.write(text)
        if path.name!='manifest.json':
            seal=path.with_name(path.name+'.sha256')
            if seal.exists():assert seal.read_text().strip()==sha(path)
            else:seal.write_text(sha(path)+'\n')
    save(args.out/'input_receipt.json',receipt);results={}
    for unit in units:
        results[unit['unit']]={}
        for family in FAMILIES:
            token=hashlib.sha256((unit['unit']+'|'+family).encode()).hexdigest()[:20];path=args.out/(token+'.json')
            if path.exists():
                assert path.with_name(path.name+'.sha256').read_text().strip()==sha(path)
                record=load(path);assert record['input_receipt']==receipt
                answer=record['answer'];problem=compile_problem(unit,family);verify_certificate(problem,answer['certificate'])
                actual,pred,_=witness(problem,answer['selected_action_ids']);assert actual==answer['upper_bound'] and pred==answer['witness_pred_sum']
            else:
                answer=solve_unit(unit,family,data.get('warm_starts',{}).get(unit['unit'],[]),
                    args.lp_limit,args.all_limit if family=='ALL' else args.mip_limit,args.linear_limit)
                save(path,dict(unit=unit['unit'],family=family,input_receipt=receipt,answer=answer))
            results[unit['unit']][family]=answer
            print('CAPACITY',unit['unit'],family,answer['lower_bound'],answer['upper_bound'],answer['gap'],flush=True)
    comparisons={};reference_units=data.get('references',{}).get('units',{})
    for unit in units:
        key=unit['unit']
        for family,old in (('DE','O_D'),('PE','O_DU'),('ALL','allcut197')):
            ref=reference_units.get(key,{}).get(old)
            if ref is None:continue
            current=results[key][family]
            if ref['denominator']!=current['denominator']:raise ValueError('reference comparison denominator changed')
            comparisons.setdefault(key,{})[family]=dict(reference=old,reference_lower=ref['lower'],reference_upper=ref['upper'],
                extra_gain_guaranteed_numerator=ref['lower']-current['upper_bound'],
                extra_gain_possible_numerator=ref['upper']-current['lower_bound'],denominator=current['denominator'])
    result=dict(input_receipt=receipt,units=results,**summarize(units,results),
        references=data.get('references'),reference_comparisons=comparisons,
        scope='Countonly matching constraints;native fullMOT F1 stillrequired. Timeoutgap isnotinfeasibility.')
    save(args.out/'capacity.json',result)
    files={p.name:sha(p) for p in args.out.iterdir() if p.is_file() and p.name not in('manifest.json','SEALED.sha256')}
    save(args.out/'manifest.json',files);seal=sha(args.out/'manifest.json')
    if(args.out/'SEALED.sha256').exists():assert(args.out/'SEALED.sha256').read_text().strip()==seal
    else:(args.out/'SEALED.sha256').write_text(seal+'\n')
    print(json.dumps(dict(heads=result['heads'],seal=seal),indent=2))


def selfcheck():
    import itertools
    def action(name,parent,orphan,v,kind='extension',label='S',member=0):
        return dict(action_id=name,parent_key=f'{member}|{parent}',orphan_key=f'{member}|{orphan}',kind=kind,
                    member_index=member,labels={'0.5':{'class':label}},v_indices=list(range(0,60,2)),v_values=[v]*30)
    # A cut andextension tosameparent cannot bothfixseparatecounts.
    nf=3;shape=phase_shape(nf);base=np.zeros(shape,np.int64);base[:,0]=2;gt=np.zeros(shape,np.int64)
    cut=action('cut',1,'dummycut',-1,'cut','D');ext=action('edge',1,7,-1)
    unit=dict(unit='synthetic',hive='35',group=1,nf=nf,pred_sum=base.tolist(),gt_bins=gt.tolist(),actions=[cut,ext],member_indices=[0,1,2])
    result=solve_unit(unit,'DE');assert result['upper_bound']==result['lower_bound']==30
    # Competingrecipients share atrueorphan;cross-member row sources doNOTshare.
    shared=dict(unit,actions=[action('a',1,7,-1),action('b',2,7,-1)])
    assert solve_unit(shared,'E')['upper_bound']==30
    independent=dict(unit,actions=[action('a',1,7,-1,member=0),action('b',1,7,-1,member=1)])
    assert solve_unit(independent,'E')['upper_bound']==0
    rng=np.random.default_rng(200);checked=0
    for repeat in range(20):
        acts=[]
        for i in range(6):
            a=action(str(i),int(rng.integers(3)),int(rng.integers(3)),int(rng.integers(-1,2)))
            acts.append(a)
        trial=dict(unit,actions=acts,pred_sum=np.full(shape,6,np.int64).tolist(),gt_bins=np.ones(shape,np.int64).tolist())
        problem=compile_problem(trial,'ALL');oracle=10**9
        for bits in itertools.product((0,1),repeat=len(problem['actions'])):
            ids=[a['action_id'] for a,b in zip(problem['actions'],bits) if b]
            try:value,_,_=witness(problem,ids)
            except ValueError:continue
            oracle=min(oracle,value)
        answer=solve_unit(trial,'ALL')
        assert answer['lower_bound']<=oracle<=answer['upper_bound']
        if answer['optimal']:assert answer['upper_bound']==oracle
        checked+=1
    moved=dict(action('move',1,7,0),delta_events=[dict(frame=25,direction=1,delta=-1),dict(frame=26,direction=1,delta=1)])
    sparse=sparse_delta(moved,26)
    assert len(sparse)==2 and sum(sparse.values())==0 and sum(map(abs,sparse.values()))==2
    empty=solve_unit(dict(unit,actions=[]),'E');assert empty['optimal'] and not empty['selected_action_ids']
    print(json.dumps(dict(exhaustive_graphs=checked,parent_cut_extension_conflict=True,orphan_competition=True,
                          member_sources_independent=True,date_move_two_entries=True,empty=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path);parser.add_argument('--sha256');parser.add_argument('--prereg')
    parser.add_argument('--out',type=Path);parser.add_argument('--lp-limit',type=float,default=10);parser.add_argument('--mip-limit',type=float,default=30)
    parser.add_argument('--all-limit',type=float,default=10);parser.add_argument('--linear-limit',type=int,default=200)
    args=parser.parse_args();selfcheck()
    if args.input:
        assert args.sha256 and args.prereg and args.out
        run_cli(args)
