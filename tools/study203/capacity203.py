"""Suffix-only set-packing count capacity; private synthetic until explicit GO.

S=formal S, P=formal S/U, A=all. Formal classes are supplied by owners; this
module does not infer physical identity from boxes or from count improvement.
Every selected action is shared by all30 phases. New F1 needs native scoring.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,hashlib,json,warnings
from fractions import Fraction
from itertools import combinations,product
from numbers import Integral
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix,csr_matrix,eye,hstack,vstack

FAMILIES=('S','P','A')
def integer(x):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral):raise ValueError('integer required')
    return int(x)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def phase_shape(nf):
    nf=integer(nf)
    if nf<1:raise ValueError('positive clip length required')
    return (sum((nf+25*o+749)//750 for o in range(30)),2)
def iou(a,b):
    x,y,w,h=a;u,v,s,t=b
    if min(w,h,s,t)<=0:raise ValueError('positive box sizes required')
    intersection=max(0,min(x+w,u+s)-max(x,u))*max(0,min(y+h,v+t)-max(y,v))
    return intersection/(w*h+s*t-intersection)

def compile_resources(actions):
    """Sparse source and per-member/date indices, never dense n-by-n matrix.

    One clip of exactly three source members. source=(detector SHA,D row index)
    is global across ALL hosts. One conflict row per overlapping action pair.
    """
    rows=[];date_index={};ids=[a['action_id'] for a in actions]
    if len(set(ids))!=len(ids):raise ValueError('unique action IDs required')
    for i,a in enumerate(actions):
        member=str(a['host_member']);parent=integer(a['host_track_id'])
        resources=[('host',member,parent)];sources=[];frames=[]
        for s in a['suffix']:
            source=tuple(s['source'])
            if len(source)!=2 or not isinstance(source[0],str) or integer(source[1])<0:raise ValueError('SHA/index source required')
            sources.append(source);resources.append(('row',*source));frames.append(integer(s['frame']))
            if len(s['box'])!=4 or not np.isfinite(np.asarray(s['box'],float)).all() or min(s['box'][2:])<=0:raise ValueError('finite positive box required')
            date_index.setdefault((member,frames[-1]),[]).append((i,s['box']))
        if not sources or len(sources)!=len(set(sources)) or len(frames)!=len(set(frames)):raise ValueError('nonempty unique source rows/dates required')
        rows.append(resources)
    conflicts=set()
    for entries in date_index.values():
        for (i,a),(j,b) in combinations(entries,2):
            if i!=j and iou(a,b)>=.5:conflicts.add(tuple(sorted((i,j))))
    for i,j in sorted(conflicts):
        resource=('overlap',*sorted((ids[i],ids[j])));rows[i].append(resource);rows[j].append(resource)
    return rows

def permitted(a,family):
    if family not in FAMILIES:raise ValueError('unknown family')
    label=a['labels']['0.5']['class']
    if label not in ('S','D','U'):raise ValueError('formal owner class required')
    return family=='A' or label=='S' or family=='P' and label=='U'

def compile_problem(unit,family):
    shape=phase_shape(unit['nf']);pred=np.asarray(unit['pred_sum']);gt=np.asarray(unit['gt_bins'])
    for x in (pred,gt):
        if x.shape!=shape or not np.issubdtype(x.dtype,np.integer) or (x<0).any():raise ValueError('integer full30 bins required')
    if int(gt.sum())%30:raise ValueError('GT census differs across phases')
    pool=[a for a in unit['actions'] if permitted(a,family)];actions=[];columns=[];inactive=[]
    for a in pool:
        if any(not 1<=integer(s['frame'])<=unit['nf'] for s in a['suffix']):raise ValueError('suffix frame outside clip')
        v=np.asarray(a['delta_windows'])
        if v.shape!=shape or not np.issubdtype(v.dtype,np.integer):raise ValueError('integer delta shape required')
        if not np.any(v):inactive.append(a['action_id']);continue
        actions.append(a);columns.append({int(i):int(v.ravel()[i]) for i in np.flatnonzero(v.ravel())})
    # Omitting a zero-count action is safe for count optimum, not an F1 claim.
    resources=compile_resources(actions);names=list(dict.fromkeys(k for rs in resources for k in rs));lookup={k:i for i,k in enumerate(names)}
    indices=[sorted({lookup[k] for k in rs}) for rs in resources];vr=[];vc=[];vv=[];ar=[];ac=[]
    for j,(column,rs) in enumerate(zip(columns,indices)):
        for i,v in column.items():vr.append(i);vc.append(j);vv.append(v)
        for i in rs:ar.append(i);ac.append(j)
    n=len(actions)
    V=coo_matrix((np.array(vv,np.int64),(vr,vc)),shape=(pred.size,n)).tocsc()
    A=coo_matrix((np.ones(len(ar),np.int64),(ar,ac)),shape=(len(names),n)).tocsr()
    return dict(r=(pred-3*gt).ravel(),pred=pred,gt=gt,actions=actions,columns=columns,indices=indices,
                resource_names=names,V=V,A=A,pool_size=len(pool),inactive=inactive)

def witness(problem,selected):
    table={a['action_id']:i for i,a in enumerate(problem['actions'])}
    if len(selected)!=len(set(selected)) or not set(selected)<=table.keys():raise ValueError('unknown/duplicate selected ID')
    x=np.zeros(len(table),np.int64)
    for key in selected:x[table[key]]=1
    if (problem['A']@x>1).any():raise ValueError('Ax exceeds1')
    delta=problem['V']@x;pred=problem['pred'].ravel()+delta
    if (pred<0).any():raise ValueError('negative physical count')
    value=sum(abs(int(r)+int(d)) for r,d in zip(problem['r'],delta))
    return value,pred.reshape(problem['pred'].shape).tolist(),x.tolist()

def certificate(problem,p,q,prices,repair=False):
    q=integer(q);p=list(map(integer,p));prices=list(map(integer,prices))
    if q<=0 or len(p)!=len(problem['r']) or any(abs(v)>q for v in p):raise ValueError('invalid p/q')
    if len(prices)!=len(problem['resource_names']) or any(v<0 for v in prices):raise ValueError('nonnegative resource prices required')
    dots=[sum(p[i]*v for i,v in col.items()) for col in problem['columns']]
    for dot,rs in zip(dots,problem['indices']):
        deficit=-dot-sum(prices[i] for i in rs)
        if deficit>0:
            if not repair:raise ValueError('negative reduced action cost')
            prices[rs[0]]+=deficit
    scaled=sum(int(r)*v for r,v in zip(problem['r'],p))-sum(prices)
    return dict(p=p,q=q,prices=prices,edge_dot_products=dots,scaled_bound=scaled,lower_bound=max(0,-((-scaled)//q)))

def verify_certificate(problem,proof):
    actual=certificate(problem,proof['p'],proof['q'],proof['prices'])
    if actual!=proof:raise ValueError('certificate arithmetic mismatch')
    return actual['lower_bound']

def verify_answer(unit,family,answer):
    problem=compile_problem(unit,family);lower=verify_certificate(problem,answer['certificate'])
    upper,pred,x=witness(problem,answer['selected_ids'])
    masks={str(i):dict(candidate_ids=[]) for i in unit['member_indices']}
    for a,bit in zip(problem['actions'],x):
        if bit:masks[str(a['host_member_index'])]['candidate_ids'].append(a['action_id'])
    if not (answer['family']==family and answer['lower_bound']==lower<=upper==answer['upper_bound']
        and answer['gap']==upper-lower and answer['optimal']==(upper==lower)
        and answer['witness_pred_sum']==pred and answer['member_masks']==masks
        and answer['denominator']==3*int(problem['gt'].sum())
        and answer['action_order']==[a['action_id'] for a in problem['actions']]
        and list(map(tuple,answer['resource_names']))==problem['resource_names']
        and answer['action_resources']==problem['indices']):raise ValueError('answer witness/proof/masks differ')
    return True

def solve_unit(unit,family,warm_starts=(),lp_limit=30,mip_limit=120):
    problem=compile_problem(unit,family);r=problem['r'];V=problem['V'];C=problem['A'];m,n=V.shape
    baseline,_,_=witness(problem,[]);best=baseline;selected=[];messages=[]
    known={a['action_id'] for a in problem['actions']}
    for warm in warm_starts:
        ids=[key for key in warm if key in known];value,_,_=witness(problem,ids)
        if value<best:best=value;selected=ids
    proof=certificate(problem,[int(x>0)-int(x<0) for x in r],1,[0]*C.shape[0],repair=True)
    if n:
        A=vstack([hstack([V,-eye(m)]),hstack([-V,-eye(m)]),hstack([C,csr_matrix((C.shape[0],m))])],format='csr')
        rhs=np.r_[-r,r,np.ones(C.shape[0])];objective=np.r_[np.zeros(n),np.ones(m)]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
            lp=linprog(objective,A_ub=A,b_ub=rhs,bounds=[(0,1)]*n+[(0,None)]*m,
                method='highs-ds',options=dict(time_limit=lp_limit,parallel=False))
        messages.append('LP:'+lp.message)
        if lp.success:
            q=1_000_000;y=lp.ineqlin.marginals[m:2*m]-lp.ineqlin.marginals[:m]
            p=[max(-q,min(q,int(round(float(v)*q)))) for v in y]
            prices=[max(0,int(np.ceil(-float(v)*q))) for v in lp.ineqlin.marginals[2*m:]]
            candidate=certificate(problem,p,q,prices,repair=True)
            if candidate['lower_bound']>proof['lower_bound']:proof=candidate
        def accept(solution):
            nonlocal best,selected
            if solution is None or np.max(abs(solution[:n]-np.rint(solution[:n])),initial=0)>1e-7:return
            ids=[a['action_id'] for a,b in zip(problem['actions'],np.rint(solution[:n])) if b]
            try:value,_,_=witness(problem,ids)
            except ValueError:return
            if value<best:best=value;selected=ids
        accept(lp.x)
        if proof['lower_bound']<best and mip_limit>0:
            AA=vstack([A,csr_matrix(np.r_[np.zeros(n),np.ones(m)][None,:])],format='csr')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
                mip=milp(objective,integrality=np.r_[np.ones(n),np.zeros(m)],
                    bounds=Bounds(np.zeros(n+m),np.r_[np.ones(n),np.full(m,np.inf)]),
                    constraints=LinearConstraint(AA,-np.inf,np.r_[rhs,best]),
                    options=dict(time_limit=mip_limit,mip_rel_gap=0.,parallel=False))
            messages.append('MILP:'+mip.message);accept(mip.x)
    lower=verify_certificate(problem,proof);upper,pred,x=witness(problem,selected)
    assert lower<=upper==best
    masks={str(i):dict(candidate_ids=[]) for i in unit['member_indices']}
    for a,bit in zip(problem['actions'],x):
        if bit:masks[str(a['host_member_index'])]['candidate_ids'].append(a['action_id'])
    denominator=3*int(problem['gt'].sum())
    chosen=[a for a,bit in zip(problem['actions'],x) if bit]
    return dict(family=family,baseline_numerator=baseline,lower_bound=lower,upper_bound=upper,gap=upper-lower,
        optimal=lower==upper,denominator=denominator,selected_ids=selected,member_masks=masks,witness_pred_sum=pred,
        certificate=proof,resource_names=problem['resource_names'],action_resources=problem['indices'],
        action_order=[a['action_id'] for a in problem['actions']],family_actions=problem['pool_size'],
        selected_action_count=len(chosen),inserted_row_count=sum(len(a['suffix']) for a in chosen),
        selected_class_counts={label:sum(a['labels']['0.5']['class']==label for a in chosen) for label in ('S','U','D')},
        active_actions=n,zero_delta_count=len(problem['inactive']),solver_messages=messages,
        scope='Count-only;witness is not a minimum-cardinality claim;F1 needs native fullscene suffix scoring;timeout is not infeasibility.')

def normalize(data):
    units=[]
    for row in data['units']:
        h=str(row.get('domain',row.get('hive')))
        if h not in ('35','33'):continue
        actions=[];members=[]
        for m in row['members']:
            slot=integer(m['member_index']);members.append(slot)
            for a in m['actions']:
                if str(a.get('host_member',m['member']))!=str(m['member']) or a.get('host_member_index',slot)!=slot:raise ValueError('host member mapping drift')
                actions.append(dict(a,host_member=str(m['member']),host_member_index=slot))
        if set(members)!={0,1,2} or len(members)!=3:raise ValueError('three original source slots required')
        units.append(dict(row,hive=h,actions=actions,member_indices=members))
    if len(units)!=8 or len({u['unit'] for u in units})!=8:raise ValueError('eight registered head units required')
    if any(len({u['group'] for u in units if u['hive']==h})!=4 for h in ('35','33')):raise ValueError('four groups each head required')
    return units

def summarize(units,results):
    groups=[];heads={}
    for h in ('35','33'):
        gains={f:[] for f in FAMILIES}
        for g in sorted({u['group'] for u in units if u['hive']==h}):
            rows=[results[u['unit']] for u in units if u['hive']==h and u['group']==g];den=sum(x['S']['denominator'] for x in rows)
            if not den:raise ValueError('undefined zero-GT whole group')
            base=sum(x['S']['baseline_numerator'] for x in rows);record=dict(hive=h,group=g,denominator=den,K=base,families={})
            for f in FAMILIES:
                lower=sum(x[f]['lower_bound'] for x in rows);upper=sum(x[f]['upper_bound'] for x in rows)
                lo=Fraction(100*(base-upper),den);hi=Fraction(100*(base-lower),den);gains[f].append((lo,hi))
                record['families'][f]=dict(lower=lower,upper=upper,witness_gain=float(lo),gain_ceiling=float(hi))
            groups.append(record)
        heads[h]={}
        for f,values in gains.items():
            lo=sum((v[0] for v in values),Fraction())/len(values);hi=sum((v[1] for v in values),Fraction())/len(values)
            heads[h][f]=dict(witness_gain=float(lo),gain_ceiling=float(hi),witness_gain_fraction=[lo.numerator,lo.denominator],
                gain_ceiling_fraction=[hi.numerator,hi.denominator],all_optimal=all(results[u['unit']][f]['optimal'] for u in units if u['hive']==h))
    return dict(groups=groups,heads=heads)

def save(path,data):
    text=json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=text:raise ValueError('immutable checkpoint differs '+str(path))
    else:
        with path.open('x') as stream:stream.write(text)

def run(args):
    # Root explicit field GO plus a complete common prediction seal are required.
    if sha(args.predictions/'manifest.json')!=args.prediction_seal or (args.predictions/'SEALED.sha256').read_text().strip()!=args.prediction_seal:
        raise ValueError('common prediction seal differs')
    for name,value in load(args.predictions/'manifest.json').items():
        path=(args.predictions/name).resolve()
        if not path.is_relative_to(args.predictions.resolve()) or sha(path)!=value:raise ValueError('prediction file differs')
    if sha(args.input)!=args.sha256:raise ValueError('compact input differs')
    data=load(args.input)
    if data['prereg']!=args.prereg or data['prediction_seal']!=args.prediction_seal:raise ValueError('prereg/prediction lineage differs')
    units=normalize(data);args.out.mkdir(parents=True,exist_ok=True);save(args.out/'compact_input.json',data)
    receipt=dict(input_sha256=args.sha256,prereg=args.prereg,prediction_seal=args.prediction_seal,code_sha256=sha(__file__),
        lp_limit=args.lp_limit,mip_limit=args.mip_limit,compact_input_sha256=sha(args.out/'compact_input.json'),
        source_receipts=data.get('source_receipts'));save(args.out/'input_receipt.json',receipt);results={}
    for unit in units:
        key=unit['unit'];results[key]={}
        for family in FAMILIES:
            path=args.out/(hashlib.sha256((key+'|'+family).encode()).hexdigest()[:20]+'.json')
            if path.exists():
                record=load(path)
                if record['receipt']!=receipt or record['unit']!=key:raise ValueError('checkpoint lineage differs')
                answer=record['answer'];verify_answer(unit,family,answer)
            else:
                answer=solve_unit(unit,family,data.get('warm_starts',{}).get(key,[]),args.lp_limit,args.mip_limit)
                verify_answer(unit,family,answer)
                save(path,dict(unit=key,receipt=receipt,answer=answer))
            results[key][family]=answer;print('CAP203',key,family,answer['lower_bound'],answer['upper_bound'],flush=True)
    result=dict(receipt=receipt,units=results,**summarize(units,results));save(args.out/'capacity.json',result)
    save(args.out/'manifest.json',{p.name:sha(p) for p in args.out.iterdir() if p.is_file() and p.name not in ('manifest.json','SEALED.sha256')})
    seal=sha(args.out/'manifest.json')
    path=args.out/'SEALED.sha256'
    if path.exists():
        if path.read_text().strip()!=seal:raise ValueError('output seal differs')
    else:path.write_text(seal+'\n')
    print(json.dumps(dict(heads=result['heads'],seal=seal)))

def selfcheck():
    import random
    rng=random.Random(203);shape=phase_shape(3)
    def action(key,slot,parent,source,v,label='S',box=(0,0,2,2),frame=3):
        return dict(action_id=key,host_member=str(slot+10),host_member_index=slot,host_track_id=parent,
            labels={'0.5':{'class':label}},suffix=[dict(source=['detector',source],frame=frame,box=list(box))],delta_windows=v)
    base=np.zeros(shape,np.int64);base[:,0]=5;gt=np.zeros(shape,np.int64);gt[:,0]=1
    minus=np.zeros(shape,np.int64);minus[:,0]=-1
    a=action('A',0,1,7,minus.tolist());b=action('B',1,2,7,minus.tolist())
    unit=dict(nf=3,pred_sum=base.tolist(),gt_bins=gt.tolist(),actions=[a,b],member_indices=[0,1,2])
    assert solve_unit(unit,'S')['upper_bound']==30 # shared donor cannot fix60
    c=action('C',0,2,8,minus.tolist())
    assert solve_unit(dict(unit,actions=[a,c]),'S')['upper_bound']==30 # distinct source but overlapping new boxes
    d=action('D',0,1,8,minus.tolist(),box=(20,0,2,2))
    assert solve_unit(dict(unit,actions=[a,d]),'S')['upper_bound']==30 # same parent even without overlap
    for repeat in range(32):
        aa=[]
        for i in range(6):
            v=np.zeros(shape,np.int64);v[:,int(rng.randrange(2))]=rng.randrange(-1,2)
            aa.append(action(str(i),i%3,rng.randrange(2),rng.randrange(4),v.tolist(),rng.choice(['S','U','D']),box=(rng.randrange(3)*4,0,2,2)))
        trial=dict(unit,actions=aa,pred_sum=np.full(shape,6,np.int64).tolist(),gt_bins=np.ones(shape,np.int64).tolist())
        for family in FAMILIES:
            problem=compile_problem(trial,family);best=witness(problem,[])[0]
            for bits in product((0,1),repeat=len(problem['actions'])):
                try:value,_,_=witness(problem,[a['action_id'] for a,x in zip(problem['actions'],bits) if x])
                except ValueError:continue
                best=min(best,value)
            answer=solve_unit(trial,family,lp_limit=2,mip_limit=2)
            verify_answer(trial,family,json.loads(json.dumps(answer)))
            assert answer['lower_bound']<=best<=answer['upper_bound']
            if answer['optimal']:assert answer['upper_bound']==best
    empty=solve_unit(dict(unit,actions=[]),'A');assert empty['optimal']
    # K5 pair conflicts have an LP integrality gap: even MILP optimal success
    # must not become an exact claim when this resource-price proof is loose.
    clique=[action(str(i),0,i,100+i,minus.tolist()) for i in range(5)]
    cli=solve_unit(dict(unit,actions=clique),'A')
    assert cli['upper_bound']==30 and cli['lower_bound']<30 and not cli['optimal']
    print('PASS 96 exhaustive family graphs; shared donor across hosts; host; IoU pair; signed deltas; empty; MILP success with uncertified gap retained')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path);p.add_argument('--sha256');p.add_argument('--prereg');p.add_argument('--out',type=Path)
    p.add_argument('--predictions',type=Path);p.add_argument('--prediction-seal');p.add_argument('--field-go',action='store_true')
    p.add_argument('--lp-limit',type=float,default=30);p.add_argument('--mip-limit',type=float,default=120);args=p.parse_args()
    if args.input:
        if not (args.sha256 and args.prereg and args.out and args.predictions and args.prediction_seal and args.field_go):p.error('pinned inputs, sealed predictions, and explicit field GO required')
        run(args)
    else:selfcheck()
