"""Private group-true-GT count oracle for independent single-event ID cuts.

One binary toggle per original track; each column adds/drops exactly one dated
directional event in all30 phases. This numerical tool does not verify track
legality orF1. Only exact witness/certificate equality is called optimal.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,linprog,milp
from scipy.sparse import csr_matrix,eye,hstack,vstack
sys.path.insert(0,str(Path(__file__).resolve().parent))
from local_choice_certificate import certificate,verify_certificate,feasible_numerator


def _ints(a,name):
    a=np.asarray(a)
    if not np.issubdtype(a.dtype,np.number) or np.iscomplexobj(a) or not np.isfinite(a).all() or np.any(a!=np.rint(a)):
        raise ValueError(name+' must be finite integer values')
    return a.astype(np.int64)


def phase_counts(events,nf):
    if isinstance(nf,bool) or int(nf)!=nf or nf<1:
        raise ValueError('nf must be a positive integer')
    nf=int(nf)
    result=[]
    for offset in range(30):
        counts=np.zeros(((nf+25*offset+749)//750,2),np.int64)
        for frame,direction in events:
            if int(frame)!=frame or not 1<=frame<=nf or direction not in(-1,1):
                raise ValueError('invalid full-clip event')
            counts[(int(frame)-1+25*offset)//750,0 if direction==1 else 1]+=1
        result.extend(counts.tolist())
    return np.asarray(result,np.int64)


def make_columns(candidates,nf):
    shape=phase_counts([],nf).shape;columns=[];ids=[]
    for c in candidates:
        if c['delta'] not in(-1,1):
            raise ValueError('one toggle must add/drop exactly one event')
        column=int(c['delta'])*phase_counts([(c['frame'],c['direction'])],nf)
        if 'delta_windows' in c and not np.array_equal(column,_ints(c['delta_windows'],'delta_windows')):
            raise ValueError('supplied windows disagree with dated single event')
        assert int(np.abs(column).sum())==30
        ids.append(c['candidate_id']);columns.append(column.ravel())
    if len(set(ids))!=len(ids):
        raise ValueError('candidate IDs must be globally unique within clip')
    return np.column_stack(columns) if columns else np.zeros((shape[0]*2,0),np.int64)


def _inputs(pred_sum,gt_bins,candidates,nf):
    pred=_ints(pred_sum,'pred_sum');gt=_ints(gt_bins,'gt_bins')
    shape=phase_counts([],nf).shape
    if pred.shape!=shape or gt.shape!=shape or (pred<0).any() or (gt<0).any():
        raise ValueError('nonnegative stacked30phase in/out bins required')
    if int(gt.sum())%30:
        raise ValueError('GT total is not a complete30phase event inventory')
    V=make_columns(candidates,nf)
    if (pred.ravel()+np.minimum(V,0).sum(axis=1)<0).any():
        raise ValueError('simultaneous deletion candidates exceed original counts')
    return pred,gt,(pred-3*gt).ravel(),V


def singleton_labels(pred_sum,gt_bins,candidates,nf):
    _,_,r,V=_inputs(pred_sum,gt_bins,candidates,nf);baseline=int(np.abs(r).sum())
    result=[]
    for i,c in enumerate(candidates):
        improvement=baseline-int(np.abs(r+V[:,i]).sum())
        assert -30<=improvement<=30
        result.append(dict(candidate_id=c['candidate_id'],improvement_numerator=improvement,y=improvement/30))
    return result


def solve_clip(pred_sum,gt_bins,candidates,nf,time_limit=60,exhaustive_limit=18):
    pred,gt,r,V=_inputs(pred_sum,gt_bins,candidates,nf);n=len(candidates);m=len(r)
    groups={}
    for i,c in enumerate(candidates):
        groups.setdefault(((int(c['frame'])-1)//25,int(c['direction']),int(c['delta'])),[]).append(i)
    keys=list(groups);members=[groups[key] for key in keys];cap=np.array(list(map(len,members)),np.int64)
    U=np.column_stack([V[:,indices[0]] for indices in members]) if members else np.zeros((m,0),np.int64)
    for j,indices in enumerate(members):
        assert all(np.array_equal(V[:,i],U[:,j]) for i in indices)
    baseline=int(np.abs(r).sum());best=baseline;best_x=[0]*n;messages=[];p=[0]*m;q=1
    if not n:
        p=[int(v>0)-int(v<0) for v in r]
    proof=certificate(r,V,p,q)
    if n:
        A=vstack([hstack([csr_matrix(U),-eye(m)]),hstack([-csr_matrix(U),-eye(m)])],format='csr')
        objective=np.r_[np.zeros(len(keys)),np.ones(m)];rhs=np.r_[-r,r]
        bounds=[(0,int(c)) for c in cap]+[(0,None)]*m
        lp=linprog(objective,A_ub=A,b_ub=rhs,bounds=bounds,method='highs-ds',options={'time_limit':time_limit})
        messages.append('LP: '+lp.message)
        if lp.success:
            y=lp.ineqlin.marginals[m:]-lp.ineqlin.marginals[:m];q=1_000_000
            p=[max(-q,min(q,int(round(float(value)*q)))) for value in y]
            proof=certificate(r,V,p,q);verify_certificate(r,V,proof)
        incumbent=lp.x[:len(keys)] if lp.x is not None else None
        if incumbent is None or np.max(np.abs(incumbent-np.rint(incumbent)),initial=0)>1e-7:
            mip=milp(objective,integrality=np.r_[np.ones(len(keys)),np.zeros(m)],
                bounds=Bounds(np.zeros(len(objective)),np.r_[cap,np.full(m,np.inf)]),
                constraints=LinearConstraint(A,-np.inf,rhs),options=dict(time_limit=time_limit,mip_rel_gap=0.))
            messages.append('MILP: '+mip.message)
            incumbent=mip.x[:len(keys)] if mip.x is not None else None
        if incumbent is not None and np.max(np.abs(incumbent-np.rint(incumbent)),initial=0)<=1e-7:
            count=np.rint(incumbent).astype(int)
            if (count>=0).all() and (count<=cap).all():
                x=[0]*n
                for indices,k in zip(members,count):
                    for i in indices[:int(k)]:
                        x[i]=1
                value=feasible_numerator(r,V,x)
                if value<=best:
                    best=value;best_x=x
    lower=max(0,proof['lower_bound']);enumeration=None
    if lower<best and n<=exhaustive_limit:
        histogram={};choice=[0]*n
        def visit(i,current):
            nonlocal best,best_x
            if i==n:
                value=int(np.abs(current).sum());histogram[value]=histogram.get(value,0)+1
                if value<best:
                    best=value;best_x=choice.copy()
                return
            choice[i]=0;visit(i+1,current)
            choice[i]=1;visit(i+1,current+V[:,i])
        visit(0,r)
        enumeration=dict(configurations=2**n,objective_histogram=histogram)
        assert sum(histogram.values())==2**n and min(histogram)==best
        lower=best
    verify_certificate(r,V,proof)
    assert lower<=best==feasible_numerator(r,V,best_x)
    selected=[c['candidate_id'] for c,bit in zip(candidates,best_x) if bit]
    changed=pred.ravel()+V@np.asarray(best_x,np.int64)
    assert (changed>=0).all()
    denominator=3*int(gt.sum())
    return dict(baseline_numerator=baseline,lower_bound=lower,upper_bound=best,gap=best-lower,
        optimal=lower==best,denominator=denominator,W_lower_percent=100*lower/denominator if denominator else None,
        W_witness_percent=100*best/denominator if denominator else None,
        selected_ids=selected,x=best_x,witness_pred_sum=changed.reshape(pred.shape).tolist(),
        certificate=proof,enumeration=enumeration,solver_messages=messages,
        grouped_capacity=[dict(second=k[0],direction=k[1],delta=k[2],capacity=len(indices),
            candidate_ids=[candidates[i]['candidate_id'] for i in indices]) for k,indices in zip(keys,members)],
        interpretation='count-only oracle;F1 must be scored on selected completephysical output;no TP splicing')


def solve_group(clips,**options):
    """clips:[{clip,pred_sum,gt_bins,candidates,nf}];videos pool before division."""
    results={}
    for clip in clips:
        if clip['clip'] in results:
            raise ValueError('duplicate clip in group')
        results[clip['clip']]=solve_clip(clip['pred_sum'],clip['gt_bins'],clip['candidates'],clip['nf'],**options)
    lower=sum(v['lower_bound'] for v in results.values());upper=sum(v['upper_bound'] for v in results.values())
    denominator=sum(v['denominator'] for v in results.values())
    return dict(clips=results,lower_bound=lower,upper_bound=upper,gap=upper-lower,optimal=lower==upper,
                denominator=denominator,W_lower_percent=100*lower/denominator if denominator else None,
                W_witness_percent=100*upper/denominator if denominator else None)


def selfcheck():
    rng=np.random.default_rng(197);checked=0
    for nf in (3,26,751,1600):
        for n in range(7):
            candidates=[dict(candidate_id=i,frame=int(rng.integers(1,nf+1)),direction=(-1,1)[i%2],delta=(-1,1)[i%2]) for i in range(n)]
            events=[(c['frame'],c['direction']) for c in candidates if c['delta']==-1]+[(1,1),(nf,-1)]
            pred=phase_counts(events,nf);gt=phase_counts([(1,1),(nf,-1)],nf)
            r=(pred-3*gt).ravel();V=make_columns(candidates,nf)
            oracle=min(feasible_numerator(r,V,x) for x in itertools.product((0,1),repeat=n))
            answer=solve_clip(pred,gt,candidates,nf)
            assert answer['optimal'] and answer['lower_bound']==answer['upper_bound']==oracle
            assert answer['selected_ids']==[c['candidate_id'] for c,b in zip(candidates,answer['x']) if b]
            for row in singleton_labels(pred,gt,candidates,nf):
                assert -1<=row['y']<=1
            checked+=1
    gt0=phase_counts([],3);empty=solve_clip(gt0,gt0,[],3)
    assert empty['optimal'] and empty['upper_bound']==0 and empty['W_witness_percent'] is None
    group=solve_group([dict(clip='emptyGT',pred_sum=phase_counts([(1,1)],3),gt_bins=gt0,candidates=[],nf=3),
                      dict(clip='hasGT',pred_sum=phase_counts([(1,1)],3),gt_bins=phase_counts([(1,1)],3),candidates=[],nf=3)])
    assert group['upper_bound']==90 and group['denominator']==90 and group['W_witness_percent']==100
    print(json.dumps(dict(exhaustive_synthetic_cases=checked,emptyGT_raw_preserved=True,all_passed=True)))


if __name__=='__main__':
    selfcheck()
