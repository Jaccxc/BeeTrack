"""Synthetic recipient–orphan matching bounds for whole-event-change vectors.

Eachrecipient/orphan can be used at mostonce. Edge v mayadd/drop/move events.
Exactinteger min-cost matching and nonnegativevertex duals certify anL1 lower
bound. This cannot establish physical edge legality,F1,or integerL1 optimality
unless a legal full-vector witness reaches the bound. No data files are read.
"""
from numbers import Integral
import itertools
import json


def integer(value):
    if isinstance(value,bool) or not isinstance(value,Integral):raise ValueError('exactinteger required')
    return int(value)


def inputs(r,edges):
    r=list(map(integer,r));edges=list(edges)
    ids=[e['edge_id'] for e in edges]
    if len(set(ids))!=len(ids):raise ValueError('edge IDs must be unique')
    for e in edges:
        if len(e['v'])!=len(r):raise ValueError('one completewindow vector peredge required')
        list(map(integer,e['v']))
        hash(e['recipient']);hash(e['orphan'])
    return r,edges


def min_cost_matching(edges,costs):
    """Exact optional-cardinality mincost flow,negativeaugmenting paths only.

    Bellman–Ford uses Pythonintegers and handles residualreverse edges,so two
    earlier choices can be reassigned. Zero/positive paths need not be added.
    A separately verified vertex dual below certifies global linearoptimality.
    """
    recipients=list(dict.fromkeys(e['recipient'] for e in edges));orphans=list(dict.fromkeys(e['orphan'] for e in edges))
    source=('source',);sink=('sink',);graph={source:[],sink:[]};references=[]
    def add(a,b,cost):
        graph.setdefault(a,[]);graph.setdefault(b,[])
        forward=[b,len(graph[b]),1,integer(cost)];reverse=[a,len(graph[a]),0,-integer(cost)]
        graph[a].append(forward);graph[b].append(reverse)
        return forward
    for u in recipients:add(source,('R',u),0)
    for v in orphans:add(('O',v),sink,0)
    for e,c in zip(edges,costs):references.append(add(('R',e['recipient']),('O',e['orphan']),c))
    while True:
        distance={source:0};parent={}
        for _ in range(len(graph)-1):
            changed=False
            for a,arcs in graph.items():
                if a not in distance:continue
                for j,(b,_,cap,cost) in enumerate(arcs):
                    candidate=distance[a]+cost
                    if cap and (b not in distance or candidate<distance[b]):
                        distance[b]=candidate;parent[b]=(a,j);changed=True
            if not changed:break
        if sink not in distance or distance[sink]>=0:break
        b=sink
        while b!=source:
            a,j=parent[b];edge=graph[a][j];edge[2]-=1;graph[b][edge[1]][2]+=1;b=a
    chosen=[i for i,edge in enumerate(references) if edge[2]==0]
    return chosen,sum(integer(costs[i]) for i in chosen)


def vertex_dual(edges,costs,selected):
    """Exact nonnegativevertex potentials,derived from difference constraints.

    Let pi_R=a,pi_O=-b relative toanchor. Everyedge imposes pi_O<=pi_R+c;
    matchededge additionally enforces equality. Unmatchedvertices have0dual.
    Bellman–Ford finds integers or exposes a nonoptimalmatching via negcycle.
    """
    anchor=('anchor',);nodes={anchor};constraints=[]
    usedR={edges[i]['recipient'] for i in selected};usedO={edges[i]['orphan'] for i in selected}
    recipients=list(dict.fromkeys(e['recipient'] for e in edges));orphans=list(dict.fromkeys(e['orphan'] for e in edges))
    for e,c in zip(edges,costs):
        R,O=('R',e['recipient']),('O',e['orphan']);nodes.update((R,O));constraints.append((R,O,integer(c)))
    for i in selected:
        e=edges[i];constraints.append((('O',e['orphan']),('R',e['recipient']),-integer(costs[i])))
    for u in recipients:
        R=('R',u);constraints.append((R,anchor,0))
        if u not in usedR:constraints.append((anchor,R,0))
    for v in orphans:
        O=('O',v);constraints.append((anchor,O,0))
        if v not in usedO:constraints.append((O,anchor,0))
    pi={n:0 for n in nodes}
    for iteration in range(len(nodes)):
        changed=False
        for a,b,c in constraints:
            if pi[b]>pi[a]+c:pi[b]=pi[a]+c;changed=True
        if not changed:break
    else:raise ValueError('negativecycle: suppliedmatching is not linearoptimal')
    a={u:pi['R',u]-pi[anchor] for u in recipients};b={v:pi[anchor]-pi['O',v] for v in orphans}
    assert all(x>=0 for x in list(a.values())+list(b.values()))
    assert all(integer(c)+a[e['recipient']]+b[e['orphan']]>=0 for e,c in zip(edges,costs))
    assert sum(integer(costs[i]) for i in selected)==-sum(a.values())-sum(b.values())
    return a,b


def witness_value(r,edges,selected_ids):
    chosen=set(selected_ids)
    if len(chosen)!=len(selected_ids) or not chosen<=set(e['edge_id'] for e in edges):raise ValueError('unique knownedge IDs required')
    selected=[e for e in edges if e['edge_id'] in chosen]
    if len({e['recipient'] for e in selected})!=len(selected) or len({e['orphan'] for e in selected})!=len(selected):
        raise ValueError('not a legalmatching')
    return sum(abs(integer(base)+sum(integer(e['v'][j]) for e in selected)) for j,base in enumerate(r))


def bound(r,edges,p,q):
    r,edges=inputs(r,edges);p=list(map(integer,p));q=integer(q)
    if q<=0 or len(p)!=len(r) or any(abs(v)>q for v in p):raise ValueError('|p_j|<=q andq>0 required')
    costs=[sum(a*integer(v) for a,v in zip(p,e['v'])) for e in edges]
    selected,minimum=min_cost_matching(edges,costs);a,b=vertex_dual(edges,costs,selected)
    linear=sum(x*y for x,y in zip(p,r));scaled=linear+minimum
    lower=max(0,-((-scaled)//q));ids=[edges[i]['edge_id'] for i in selected]
    value=witness_value(r,edges,ids);empty=witness_value(r,edges,[])
    witness=ids if value<=empty else [];upper=min(value,empty)
    proof=dict(p=p,q=q,edge_costs=costs,linear_term=linear,matching_ids=ids,minimum_matching_cost=minimum,
        recipient_dual=[dict(recipient=u,a=v) for u,v in a.items()],orphan_dual=[dict(orphan=u,b=v) for u,v in b.items()],
        scaled_bound=scaled,lower_bound=lower,witness_ids=witness,witness_upper=upper,optimal_if_tight=lower==upper)
    assert verify(r,edges,proof)
    return proof


def verify(r,edges,proof):
    r,edges=inputs(r,edges);p=list(map(integer,proof['p']));q=integer(proof['q'])
    assert q>0 and len(p)==len(r) and all(abs(v)<=q for v in p)
    costs=[sum(x*integer(v) for x,v in zip(p,e['v'])) for e in edges];assert costs==proof['edge_costs']
    matching=proof['matching_ids'];witness_value(r,edges,matching)
    chosen=set(matching);minimum=sum(c for e,c in zip(edges,costs) if e['edge_id'] in chosen)
    a={r['recipient']:integer(r['a']) for r in proof['recipient_dual']};b={r['orphan']:integer(r['b']) for r in proof['orphan_dual']}
    assert set(a)=={e['recipient'] for e in edges} and set(b)=={e['orphan'] for e in edges}
    assert all(v>=0 for v in list(a.values())+list(b.values()))
    assert all(c+a[e['recipient']]+b[e['orphan']]>=0 for e,c in zip(edges,costs))
    assert minimum==proof['minimum_matching_cost']==-sum(a.values())-sum(b.values())
    linear=sum(x*y for x,y in zip(p,r));scaled=linear-sum(a.values())-sum(b.values())
    assert linear==proof['linear_term'] and scaled==proof['scaled_bound']
    assert proof['lower_bound']==max(0,-((-scaled)//q))
    upper=witness_value(r,edges,proof['witness_ids']);assert upper==proof['witness_upper']
    assert proof['optimal_if_tight']==(proof['lower_bound']==upper)
    return True


def lp_bound(r,edges,q=1_000_000):
    """SciPy proposes p only;integer matching/dual verification proves thebound.

    This LP permits fractionaledge choices. It is not an integer L1 witness.
    """
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix,eye,hstack,vstack
    r,edges=inputs(r,edges);q=integer(q)
    if not edges:return bound(r,edges,[q*((v>0)-(v<0)) for v in r],q)
    m,n=len(r),len(edges);V=csr_matrix(np.asarray([e['v'] for e in edges],float).T)
    recipients=list(dict.fromkeys(e['recipient'] for e in edges));orphans=list(dict.fromkeys(e['orphan'] for e in edges))
    capacity=np.array([[int(e['recipient']==u) for e in edges] for u in recipients]+
                      [[int(e['orphan']==v) for e in edges] for v in orphans],float)
    A=vstack([hstack([V,-eye(m)]),hstack([-V,-eye(m)]),hstack([csr_matrix(capacity),csr_matrix((len(capacity),m))])],format='csr')
    solution=linprog(np.r_[np.zeros(n),np.ones(m)],A_ub=A,b_ub=np.r_[-np.asarray(r),r,np.ones(len(capacity))],
                     bounds=[(0,1)]*n+[(0,None)]*m,method='highs-ds')
    if not solution.success:raise RuntimeError('LP proposal failed:'+solution.message)
    score=solution.ineqlin.marginals[m:2*m]-solution.ineqlin.marginals[:m]
    p=[max(-q,min(q,int(round(float(v)*q)))) for v in score]
    proof=bound(r,edges,p,q);proof['lp_relaxation_diagnostic']=float(solution.fun)
    return proof


def enumerate_optimum(r,edges):
    """Tiny exhaustive reference only;do not call on empirical large pools."""
    if len(edges)>20:raise ValueError('tinyexhaustive helper limited to20edges')
    valid=[]
    for bits in itertools.product((0,1),repeat=len(edges)):
        ids=[e['edge_id'] for e,bit in zip(edges,bits) if bit]
        try:value=witness_value(r,edges,ids)
        except ValueError:continue
        valid.append((value,ids))
    return min(valid,key=lambda row:row[0]),valid


def selfcheck():
    edge=lambda name,u,o,v:dict(edge_id=name,recipient=u,orphan=o,v=v)
    shared=[edge('a','R1','O',[-1]),edge('b','R2','O',[-1])]
    proof=bound([2],shared,[1],1)
    assert proof['lower_bound']==proof['witness_upper']==1
    assert enumerate_optimum([2],shared)[0][0]==1  # ignoringorphanreuse wouldfake0
    assert bound([-2,3],[],[-1,1],1)['lower_bound']==5
    moved=[edge('move',1,1,[-1,1])]
    assert bound([1,-1],moved,[1,-1],1)['optimal_if_tight']
    adddrop=[edge('add',1,1,[1,0]),edge('drop',2,2,[0,-1])]
    assert bound([-1,1],adddrop,[-1,1],1)['lower_bound']==0
    # Must reroute anearlierselectededge to reachminimumcost-8,notgreedy-5.
    reroute=[edge('a',0,0,[-5]),edge('b',0,1,[-4]),edge('c',1,0,[-4])]
    assert bound([10],reroute,[1],1)['minimum_matching_cost']==-8
    checked=0
    import random
    rng=random.Random(199)
    for _ in range(80):
        edges=[edge(str(i),rng.randrange(3),rng.randrange(3),[rng.randrange(-2,3) for _ in range(3)]) for i in range(6)]
        r=[rng.randrange(-3,4) for _ in range(3)];q=rng.randrange(1,6);p=[rng.randrange(-q,q+1) for _ in r]
        proof=bound(r,edges,p,q);(optimum,_),valid=enumerate_optimum(r,edges)
        assert proof['lower_bound']<=optimum<=proof['witness_upper']
        minimum=min(sum(sum(a*v for a,v in zip(p,e['v'])) for e in edges if e['edge_id'] in ids) for _,ids in valid)
        assert minimum==proof['minimum_matching_cost']
        checked+=1
    proposal=lp_bound([2],shared);assert proposal['lower_bound']==1 and verify([2],shared,proposal)
    print(json.dumps(dict(random_graphs_exhaustively_checked=checked,shared_orphan_fake_gain_prevented=True,
                          rerouting=True,empty=True,moving_event=True,add_drop=True,LP_integer_proof=True)))


if __name__=='__main__':selfcheck()
