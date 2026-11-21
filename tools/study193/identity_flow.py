"""Synthetic reusable fixed-quota identity/F1 ceiling, no dataset imports.

Bucket keys are(member,second,direction); prediction IDs are globally unique.
GT IDs are local to member. Edges must be a FROZEN official event graph:
deletion/order-dependent box matching invalidates this conditional ceiling.
"""
from collections import deque
from itertools import combinations,product
import json


class Flow:
    def __init__(self):
        self.graph={};self.original=[]

    def add(self,a,b,capacity):
        self.graph.setdefault(a,[]);self.graph.setdefault(b,[])
        forward=[b,len(self.graph[b]),capacity]
        backward=[a,len(self.graph[a]),0]
        self.graph[a].append(forward);self.graph[b].append(backward)
        self.original.append((a,b,capacity,forward))

    def solve(self,source,sink):
        self.graph.setdefault(source,[]);self.graph.setdefault(sink,[])
        total=0
        while True:
            levels={source:0};queue=deque([source])
            while queue:
                a=queue.popleft()
                for b,_,cap in self.graph[a]:
                    if cap and b not in levels:
                        levels[b]=levels[a]+1;queue.append(b)
            if sink not in levels:
                break
            cursor={v:0 for v in self.graph}

            def push(a,amount):
                if a==sink:
                    return amount
                while cursor[a]<len(self.graph[a]):
                    edge=self.graph[a][cursor[a]];b,reverse,cap=edge
                    if cap and levels.get(b)==levels[a]+1:
                        sent=push(b,min(amount,cap))
                        if sent:
                            edge[2]-=sent;self.graph[b][reverse][2]+=sent
                            return sent
                    cursor[a]+=1
                return 0

            while True:
                sent=push(source,10**30)
                if not sent:
                    break
                total+=sent
        reachable={source};queue=deque([source])
        while queue:
            a=queue.popleft()
            for b,_,cap in self.graph[a]:
                if cap and b not in reachable:
                    reachable.add(b);queue.append(b)
        cut=[dict(source=a,target=b,capacity=cap) for a,b,cap,_ in self.original
             if a in reachable and b not in reachable and cap]
        assert sum(e['capacity'] for e in cut)==total
        return total,reachable,cut


def identity_flow(buckets,quotas,edges,gt_events,mandatory=()):
    """Exact maxTP, retained candidates and min-cut certificate at fixed quotas.

    buckets:{(member,second,direction):[predIDs]},quotas:same keys->K.
    edges:{predID:[localGTIDs]},gt_events:{member:[all localGTIDs]}.
    mandatory predictions are forced retained via singleton quota1 buckets,
    subtracting their counts from original bucket quotas (not extra events).
    IDs may be any hashable values; input insertion order breaks witness ties.
    """
    if set(buckets)!=set(quotas):
        raise ValueError('bucket/quota keys differ')
    owner={};gt={};gt_order={}
    for member,ids in gt_events.items():
        ids=list(ids)
        if len(set(ids))!=len(ids):
            raise ValueError('GT IDs must be unique within each member')
        gt[member]=set(ids)
        gt_order[member]=ids
    for bucket,preds in buckets.items():
        if not isinstance(bucket,tuple) or len(bucket)!=3 or bucket[2] not in (-1,1):
            raise ValueError('bucket must be(member,second,+1/-1)')
        K=quotas[bucket]
        if isinstance(K,bool) or not isinstance(K,int) or not 0<=K<=len(preds):
            raise ValueError('quota must be an integer between0 and pool size')
        if bucket[0] not in gt:
            raise ValueError('missing complete GT universe for member')
        for pred in preds:
            if pred in owner:
                raise ValueError('prediction must belong to exactly one bucket')
            owner[pred]=bucket
    if set(edges)-set(owner):
        raise ValueError('edges reference unknown predictions')
    mandatory=set(mandatory)
    if mandatory-set(owner):
        raise ValueError('unknown mandatory prediction')
    groups=[]
    for bucket,preds in buckets.items():
        forced=[p for p in preds if p in mandatory]
        if len(forced)>quotas[bucket]:
            raise ValueError('mandatory retain exceeds quota')
        groups.append((('bucket',bucket),[p for p in preds if p not in mandatory],quotas[bucket]-len(forced)))
        groups.extend((('mandatory',p),[p],1) for p in forced)
    flow=Flow();source=('source',);sink=('sink',)
    for group,preds,K in groups:
        flow.add(source,group,K)
        for pred in preds:
            pnode=('pred',pred);flow.add(group,pnode,1)
            member=owner[pred][0]
            for event in dict.fromkeys(edges.get(pred,())):
                if event not in gt[member]:
                    raise ValueError('edge references GT outside member universe')
                flow.add(pnode,('GT',member,event),1)
    for member,ids in gt_order.items():
        for event in ids:
            flow.add(('GT',member,event),sink,1)
    maximum,reachable,cut=flow.solve(source,sink)
    matched=[]
    for a,b,capacity,edge in flow.original:
        if a[0]=='pred' and b[0]=='GT' and capacity-edge[2]==1:
            matched.append(dict(pred=a[1],member=b[1],gt=b[2]))
    assert len(matched)==maximum
    selected={e['pred'] for e in matched}|mandatory
    retained={}
    for bucket,preds in buckets.items():
        picked=[p for p in preds if p in selected]
        assert len(picked)<=quotas[bucket]
        picked.extend(p for p in preds if p not in selected and len(picked)<quotas[bucket])
        assert len(picked)==quotas[bucket]
        retained[bucket]=picked
    assert all(e['pred'] in retained[owner[e['pred']]] for e in matched)
    nP=sum(quotas.values());nG=sum(len(s) for s in gt.values())
    per_member={}
    for member in gt:
        tp=sum(e['member']==member for e in matched)
        npred=sum(K for b,K in quotas.items() if b[0]==member)
        ng=len(gt[member])
        per_member[member]=dict(tp=tp,pred=npred,gt=ng,f1=2*tp/max(1,npred+ng))
    return dict(maximum_tp=maximum,retained=retained,matching=matched,
                minimum_cut_capacity=sum(e['capacity'] for e in cut),minimum_cut=cut,
                residual_source_side=list(reachable),pred=nP,gt=nG,
                pooled_f1_ceiling=2*maximum/max(1,nP+nG),per_member=per_member)


def selfcheck():
    def matching_size(preds,edges):
        best=0
        def visit(i,used,n):
            nonlocal best
            if i==len(preds):
                best=max(best,n);return
            visit(i+1,used,n)
            for g in edges.get(preds[i],[]):
                if g not in used:
                    visit(i+1,used|{g},n+1)
        visit(0,set(),0);return best
    checked=0
    # Every edge graph2pred×2GT, all legal quotas across different seconds.
    # Shared GT capacity must span seconds, not restart independently per cell.
    b0=('m',0,1);b1=('m',1,1)
    for bits in product((0,1),repeat=4):
        edges={p:[g for g in range(2) if bits[2*p+g]] for p in range(2)}
        for k0,k1 in product((0,1),repeat=2):
            result=identity_flow({b0:[0],b1:[1]},{b0:k0,b1:k1},edges,{'m':[0,1]})
            oracle=matching_size(([0] if k0 else [])+([1] if k1 else []),edges)
            assert result['maximum_tp']==oracle==result['minimum_cut_capacity']
            checked+=1
    # Quota2 of3, all2^6 edge graphs, optional mandatory has to fit the quota.
    for bits in product((0,1),repeat=6):
        edges={p:[g for g in range(2) if bits[2*p+g]] for p in range(3)}
        for K in range(4):
            for mandatory in ([],[0]) if K else ([],):
                subsets=[s for s in combinations(range(3),K) if set(mandatory)<=set(s)]
                oracle=max(matching_size(s,edges) for s in subsets)
                result=identity_flow({b0:list(range(3))},{b0:K},edges,{'m':[0,1]},mandatory)
                assert result['maximum_tp']==oracle==result['minimum_cut_capacity']
                assert set(mandatory)<=set(result['retained'][b0])
                checked+=1
    # The same physicalGT label is reusable across distinct member scores.
    b2=('other',0,-1)
    r=identity_flow({b0:['a'],b2:['b']},{b0:1,b2:1},{'a':[7],'b':[7]}, {'m':[7],'other':[7]})
    assert r['maximum_tp']==2
    empty=identity_flow({}, {}, {}, {})
    assert empty['maximum_tp']==empty['minimum_cut_capacity']==0
    print(json.dumps(dict(exhaustive_graph_quota_checks=checked,member_namespace=True,all_passed=True)))


if __name__=='__main__':
    selfcheck()
