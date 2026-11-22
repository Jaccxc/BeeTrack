"""Independent §211 source/synthetic primitives; no field IO on import.

Deletion criticality is a full-graph counterfactual, not neighbor presence or
an additive decomposition of maximum TP. This distinction is tested below.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import itertools
import json
import numpy as np

def maximum_matching(left,right,edges,removed=None):
    left=set(left);right=set(right)
    adjacency={p:set() for p in left if p!=removed}
    for p,g in edges:
        if p not in left or g not in right:raise ValueError('foreign graph endpoint')
        if p in adjacency:adjacency[p].add(g)
    assigned={}
    def augment(p,seen):
        for g in sorted(adjacency[p],key=repr):
            if g in seen:continue
            seen.add(g)
            if g not in assigned or augment(assigned[g],seen):assigned[g]=p;return True
        return False
    for p in sorted(adjacency,key=repr):augment(p,set())
    return len(assigned)

def graph_diagnostics(left,right,edges):
    base=maximum_matching(left,right,edges);neighbors={p:False for p in left}
    for p,g in edges:neighbors[p]=True
    marginal={p:base-maximum_matching(left,right,edges,p) for p in left}
    assert all(value in (0,1) for value in marginal.values())
    assert all(marginal[p]<=int(neighbors[p]) for p in left)
    return dict(maxTP=base,neighbor=neighbors,marginal=marginal)

def paired_predictors(left,peer_graphs):
    if len(peer_graphs)!=2:raise ValueError('two full reference event graphs required')
    reports=[graph_diagnostics(left,g['right'],g['edges']) for g in peer_graphs]
    return {p:dict(xL=sum(int(r['neighbor'][p]) for r in reports)/2,
                   xX=sum(r['marginal'][p] for r in reports)/2,
                   neighbor_peers=sum(int(r['neighbor'][p]) for r in reports),
                   critical_peers=sum(r['marginal'][p] for r in reports)) for p in left}

def source_weights(records):
    """Nonempty source hive equal, nonempty-K-event clip equal, event equal."""
    hives={r['domain'] for r in records};out=np.empty(len(records))
    for hive in hives:
        clips={r['clip'] for r in records if r['domain']==hive}
        for clip in clips:
            ii=[i for i,r in enumerate(records) if r['domain']==hive and r['clip']==clip]
            out[ii]=1/(len(hives)*len(clips)*len(ii))
    return out

def bounded_ls(x,y,w):
    """Closed form candidates for min weighted SSE+.01b² on a,b>=0,a+b<=1."""
    x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float)
    if not len(x):return dict(a=.5,b=0.,status='empty',objective=None)
    if x.shape!=y.shape or x.shape!=w.shape or np.any(w<0) or not np.isfinite(np.r_[x,y,w]).all() or w.sum()<=0:raise ValueError('parallel finite weighted observations')
    s=float(w.sum());sx=float(w@x);sy=float(w@y);xx=float(w@(x*x))+.01;xy=float(w@(x*y))
    matrix=np.array([[s,sx],[sx,xx]]);rhs=np.array([sy,xy]);free=np.linalg.solve(matrix,rhs)
    proposals=[(0.,0.),(1.,0.),(0.,1.)]
    if free[0]>=0 and free[1]>=0 and free.sum()<=1:proposals.append(tuple(free))
    proposals.append((float(np.clip(sy/s,0,1)),0.))
    proposals.append((0.,float(np.clip(xy/xx,0,1))))
    b=float(np.clip(np.sum(w*(1-y)*(1-x))/(np.sum(w*(1-x)**2)+.01),0,1));proposals.append((1-b,b))
    objective=lambda ab:float(np.sum(w*(ab[0]+ab[1]*x-y)**2)+.01*ab[1]**2)
    a,b=min(proposals,key=lambda ab:(objective(ab),ab))
    gradient=np.array([2*np.sum(w*(a+b*x-y)),2*np.sum(w*x*(a+b*x-y))+.02*b])
    return dict(a=float(a),b=float(b),status='fitted',objective=objective((a,b)),gradient=gradient.tolist())

def utility(state,baseline,model):
    """Complete-state graph summaries, never the sum of deletion marginals."""
    dP=state['nP']-baseline['nP'];dJ=state['meanJ']-baseline['meanJ'];dM=state['meanM']-baseline['meanM']
    return dict(Q=(2*model['Q']-1)*dP,
                L=(2*model['L']['a']-1)*dP+2*model['L']['b']*dJ,
                X=(2*model['X']['a']-1)*dP+2*model['X']['b']*dM,
                V=2*dM-dP,C=baseline['meanCountL1']-state['meanCountL1'])

def choose_states(states,model,local_norm):
    baseline=next(s for s in states if not s['selected_pair_ids'])
    def key(state):return tuple(sorted(tuple(map(int,p.split('|'))) for p in state['selected_pair_ids']))
    values={s['state_id']:utility(s,baseline,model) for s in states};selected={}
    for head in ('Q','L','X','V','C'):
        selected[head]=min(states,key=lambda s:(-round(values[s['state_id']][head],10),len(s['selected_pair_ids']),key(s)))['state_id']
    x=next(s for s in states if s['state_id']==selected['X']);count=len(x['selected_pair_ids'])
    chosen=sorted(local_norm,key=lambda pid:(local_norm[pid],tuple(map(int,pid.split('|')))))[:count]
    selected['Z']=next(s for s in states if set(s['selected_pair_ids'])==set(chosen))['state_id']
    return selected,values

def pareto(states,N='numerator',E='event_error'):
    """Count-window loss and actual FP+FN are separate observed objectives."""
    return [s for s in states if not any(t[N]<=s[N] and t[E]<=s[E] and (t[N]<s[N] or t[E]<s[E]) for t in states)]

def strict_duplicate(before,after,old_family,new_family):
    """Conservative official event-graph witness, not a same-bee assertion.

    Graph objects have left vertices and edges. Family events include direction.
    Both old events must have the same singleton GT neighbor; new sole event
    retains that edge, maxTP does not drop and whole-state nPred drops exactly1.
    """
    if len(old_family)!=2 or len(new_family)!=1:return False
    oldids=[e['id'] for e in old_family];newid=new_family[0]['id']
    if len(set(e['direction'] for e in old_family+new_family))!=1:return False
    neighbors=lambda graph,p:{g for pp,g in graph['edges'] if pp==p}
    first,second=[neighbors(before,p) for p in oldids]
    if len(first)!=1 or first!=second or not first<=neighbors(after,newid):return False
    return maximum_matching(after['left'],after['right'],after['edges'])>=maximum_matching(before['left'],before['right'],before['edges']) and len(after['left'])==len(before['left'])-1

def audit_native_state(pred_rows,reference_structure,polygon,nframes,output_raw_indices,geometry_cache,native,clock,prediction_structure=None,thresholds=('0.5','0.3')):
    """Reusable joint entrance: fresh cached geometry -> full event legal graphs.

    Caller supplies independently materialized rows and their ORIGINAL raw-index
    mapping. Supports retained-row subsets and permutations; no assumption that
    prediction row position equals the source raw index. Reference may be true
    GT or a peer prediction, and MUST be identified correctly by the caller.
    Low-level native event semantics are shared; graph max matching is our DFS.
    """
    prediction=native.track_structure(pred_rows,polygon) if prediction_structure is None else prediction_structure
    owners=geometry_cache.owners(pred_rows,output_raw_indices)
    result=native.score_from_owners(prediction,reference_structure,{t:owners[t] for t in thresholds},polygon,nframes)
    left=list(prediction['events']);right=list(reference_structure['events']);graphs={}
    for threshold in thresholds:
        edges=result['edges'][threshold]
        cardinality=maximum_matching(left,right,edges)
        if cardinality!=result['iou'][threshold]['tp']:raise AssertionError('independent maximum matching disagrees event helper')
        graphs[threshold]=dict(left=left,right=right,edges=edges,maxTP=cardinality)
    return dict(prediction_structure=prediction,reference_structure=reference_structure,
                pred_windows=clock(list(prediction['events'].values()),nframes),
                reference_windows=clock(list(reference_structure['events'].values()),nframes),
                graphs=graphs,iou=result['iou'])

def brute_matching(left,right,edges):
    allowed=set(map(tuple,edges));best=0
    for n in range(min(len(left),len(right))+1):
        for ll in itertools.combinations(left,n):
            for rr in itertools.permutations(right,n):
                if all((a,b) in allowed for a,b in zip(ll,rr)):best=max(best,n)
    return best

def self_test():
    ambiguous=graph_diagnostics([1,2],[9],[(1,9),(2,9)])
    assert ambiguous['maxTP']==1 and sum(ambiguous['marginal'].values())==0 and all(ambiguous['neighbor'].values())
    protected=graph_diagnostics([1,2],[8,9],[(1,8),(2,9)])
    assert sum(protected['marginal'].values())==2
    paired=paired_predictors([1,2],[dict(right=[9],edges=[(1,9),(2,9)]),dict(right=[8],edges=[(1,8)])])
    assert paired[1]['xL']==1 and paired[1]['xX']==.5 and paired[2]['xL']==.5 and paired[2]['xX']==0
    rng=np.random.default_rng(211)
    for case in range(50):
        left=list(range(3));right=list(range(5,8));edges=[(p,g) for p in left for g in right if rng.random()<.4]
        assert maximum_matching(left,right,edges)==brute_matching(left,right,edges)
    from scipy.optimize import minimize
    for case in range(25):
        x=rng.choice([0.,.5,1.],12);y=rng.integers(0,2,12);w=rng.uniform(size=12);w/=w.sum();fit=bounded_ls(x,y,w)
        f=lambda ab:np.sum(w*(ab[0]+ab[1]*x-y)**2)+.01*ab[1]**2
        numerical=minimize(f,[.3,.3],bounds=[(0,1),(0,1)],constraints={'type':'ineq','fun':lambda ab:1-ab.sum()},method='SLSQP',options={'ftol':1e-12,'maxiter':100})
        assert numerical.success and abs(f([fit['a'],fit['b']])-numerical.fun)<1e-9
    assert bounded_ls([],[],[])['a']==.5
    base=dict(state_id='K',selected_pair_ids=[],nP=2,meanJ=2,meanM=1,meanCountL1=3)
    edit=dict(state_id='E',selected_pair_ids=['1|2'],nP=1,meanJ=1,meanM=1,meanCountL1=2)
    choices,_=choose_states([base,edit],dict(Q=.5,L=dict(a=.5,b=0),X=dict(a=.5,b=0)),{'1|2':.2})
    assert choices['Q']==choices['L']==choices['X']==choices['Z']=='K' and choices['V']==choices['C']=='E'
    assert not strict_duplicate(dict(left=[1,2],right=[9],edges=[(1,9),(2,9)]),dict(left=[3],right=[9],edges=[(3,9)]),[dict(id=1,direction=1),dict(id=2,direction=-1)],[dict(id=3,direction=1)])
    assert strict_duplicate(dict(left=[1,2],right=[9],edges=[(1,9),(2,9)]),dict(left=[3],right=[9],edges=[(3,9)]),[dict(id=1,direction=1),dict(id=2,direction=1)],[dict(id=3,direction=1)])
    return dict(passed=True,scope='synthetic only',checks=['ambiguous neighbor-versus-criticality graph','50 brute matching comparisons','25 independent bounded-LS solver comparisons','empty source fallback','relativeK utilities/abstain/ties','strict event duplicate direction guard'])

if __name__=='__main__':
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--audit',action='store_true')
    parser.add_argument('--repo',type=Path);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source206',type=Path,required=True);parser.add_argument('--source208',type=Path,required=True)
    parser.add_argument('--labels208',type=Path,required=True);parser.add_argument('--catalog210',type=Path,required=True);parser.add_argument('--reference210',type=Path,required=True)
    parser.add_argument('--seal-pins',type=Path);parser.add_argument('--field-go',action='store_true');parser.add_argument('--fit-finished',action='store_true');args=parser.parse_args()
    if args.audit:
        if args.repo is None or args.seal_pins is None:parser.error('--repo --seal-pins required')
        import field211
        field211.run(args)
    else:print(json.dumps(self_test(),sort_keys=True))
