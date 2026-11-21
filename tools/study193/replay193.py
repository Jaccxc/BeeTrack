"""Standard-library replay of §193 signed events, quotas, graphs, flow and metrics."""
import argparse
from collections import defaultdict,deque
import gzip
import json
from pathlib import Path

ARMS=('B','I','Q','R','O')
def close(a,b):assert abs(a-b)<1e-8,(a,b)
def f1(tp,p,g):return 2*tp/max(1,p+g)
def bins(events,nf):
    result=[]
    for off in range(30):
        part=[0]*(2*((nf+25*off+749)//750))
        for date in events:
            assert 1<=abs(date)<=nf
            part[2*((abs(date)+25*off-1)//750)+(date<0)]+=1
        result+=part
    return result
def matching(edges,kept):
    graph=defaultdict(list)
    for p,g in edges:
        if p in kept:graph[p].append(g)
    owner={}
    def visit(p,seen):
        for g in graph[p]:
            if g in seen:continue
            seen.add(g)
            if g not in owner or visit(owner[g],seen):owner[g]=p;return True
        return False
    return sum(visit(p,set()) for p in graph)
def flow(cap,source,sink):
    residual=defaultdict(dict)
    for (a,b),c in cap.items():residual[a][b]=c;residual[b].setdefault(a,0)
    total=0
    while True:
        parent={source:None};q=deque([source])
        while q and sink not in parent:
            a=q.popleft()
            for b,c in residual[a].items():
                if c and b not in parent:parent[b]=a;q.append(b)
        if sink not in parent:return total
        v=sink;amount=10**9
        while parent[v] is not None:amount=min(amount,residual[parent[v]][v]);v=parent[v]
        v=sink
        while parent[v] is not None:
            u=parent[v];residual[u][v]-=amount;residual[v][u]+=amount;v=u
        total+=amount
def tup(x):return tuple(tup(v) for v in x) if isinstance(x,list) else x
def oracle_check(pred,truth,edges,kept,oracle):
    groups=defaultdict(list)
    for p,date in pred.items():groups[('cell',(abs(date)-1)//25,1 if date>0 else -1)].append(p)
    mandatory=set(oracle['mandatory_ids']);assert mandatory<=kept
    cap={};source=('source',);sink=('sink',)
    adj=defaultdict(set)
    for p,g in edges:adj[p].add(g)
    for bucket,preds in groups.items():
        k=sum(p in kept for p in preds);forced=set(preds)&mandatory
        assert len(forced)<=k
        choices=[(('bucket',bucket),[p for p in preds if p not in forced],k-len(forced))]
        choices += [(('mandatory',p),[p],1) for p in forced]
        for node,pool,K in choices:
            cap[source,node]=K
            for p in pool:
                cap[node,('pred',p)]=1
                for g in adj[p]:cap[('pred',p),('GT','cell',g)]=1
    for g in truth:cap[('GT','cell',g),sink]=1
    best=flow(cap,source,sink)
    assert best==oracle['maximum_tp']==oracle['minimum_cut_capacity']
    cut={(tup(e['source']),tup(e['target'])):e['capacity'] for e in oracle['minimum_cut']}
    assert all(cap.get(edge)==value for edge,value in cut.items()) and sum(cut.values())==best
    remaining=defaultdict(list)
    for (a,b),c in cap.items():
        if c and (a,b) not in cut:remaining[a].append(b)
    seen={source};q=deque([source])
    while q:
        for b in remaining[q.popleft()]:
            if b not in seen:seen.add(b);q.append(b)
    assert sink not in seen
    selected=set(oracle['retained_ids']);assert selected==kept
    assert matching(edges,selected)==best
def replay(e):
    assert e['prereg']=='f161c64a6310c663d6cb079e83abee335d90e324'
    assert len(e['cells'])==60
    cellrows=[];members={};groups=defaultdict(list)
    for cell in e['cells']:
        r=cell['result'];native=r['native_evidence'];nf=r['nframes']
        pred={p:d for p,d,*_ in native['predictions']};truth=dict(native['gt'])
        assert len(pred)==len(native['predictions']) and len(truth)==len(native['gt'])
        G=bins(truth.values(),nf);den=30*len(truth);scores={};vectors={}
        kept={a:set(r['kept_ids'][a]) for a in ARMS}
        assert kept['B']==set(pred)
        buckets=lambda ids:sorted((sec,dr,sum(p in ids for p,d in pred.items() if (abs(d)-1)//25==sec and (1 if d>0 else -1)==dr))
                                  for sec,dr in {( (abs(d)-1)//25,1 if d>0 else -1) for d in pred.values()})
        assert buckets(kept['Q'])==buckets(kept['R'])==buckets(kept['O'])
        oracle_check(pred,truth,native['edges05'],kept['O'],r['oracle'])
        for arm in ARMS:
            ids=kept[arm];assert ids<=set(pred) and len(ids)==len(r['kept_ids'][arm])
            P=bins([pred[p] for p in ids],nf);vectors[arm]=P
            over=sum(max(p-g,0) for p,g in zip(P,G));under=sum(max(g-p,0) for p,g in zip(P,G))
            metrics={}
            for threshold,key in (('0.5','edges05'),('0.3','edges03')):
                if key not in native:continue
                tp=matching(native[key],ids)
                metrics[threshold]=dict(tp=tp,fp=len(ids)-tp,fn=len(truth)-tp,f1=f1(tp,len(ids),len(truth)))
                for key0,value in metrics[threshold].items():close(value,r['arms'][arm]['metrics'][threshold][key0])
            s=dict(gt=len(truth),pred=len(ids),tp=metrics['0.5']['tp'],W_numerator=over+under,W_denominator=den,
                   over_numerator=over,under_numerator=under,metrics=metrics)
            for k,v in s.items():
                if k!='metrics':assert v==r['arms'][arm][k],(cell['fold'],arm,k)
            close(100*(over+under)/den,r['arms'][arm]['W'])
            scores[arm]=s
        key=f"{cell['fold']}/{cell['group']}/{cell['member']}"
        pooled=members.setdefault(key,{a:dict(gt=0,pred=0,tp=0,W_numerator=0,W_denominator=0,over_numerator=0,under_numerator=0) for a in ARMS})
        for a in ARMS:
            for k in pooled[a]:pooled[a][k]+=scores[a][k]
        groups[f"{cell['fold']}/{cell['group']}"].append((cell,G,vectors,scores))
    for key,arms in members.items():
        for a,m in arms.items():
            for k,v in m.items():assert v==e['members'][key][a][k]
            m['W']=100*m['W_numerator']/m['W_denominator'];m['f1']=f1(m['tp'],m['pred'],m['gt'])
            close(m['W'],e['members'][key][a]['W']);close(m['f1'],e['members'][key][a]['f1'])
    rebuilt={}
    for key,cells in groups.items():
        byclip=defaultdict(list)
        for c,G,P,s in cells:byclip[c['clip']].append((G,P,s))
        rebuilt[key]={}
        for a in ARMS:
            over=under=den=0
            for streams in byclip.values():
                assert len(streams)==3;G=streams[0][0];assert all(g==G for g,p,s in streams)
                residual=[sum(p[a][i] for g,p,s in streams)-3*G[i] for i in range(len(G))]
                over+=sum(max(v,0) for v in residual);under+=sum(max(-v,0) for v in residual);den+=3*sum(G)
            mm=[arms[a] for k,arms in members.items() if k.startswith(key+'/')];assert len(mm)==3
            v=dict(E3=100*(over+under)/den,over=100*over/den,under=100*under/den,
                   mean_single_W=sum(m['W'] for m in mm)/3,mean_single_f1=sum(m['f1'] for m in mm)/3)
            for k,value in v.items():close(value,e['groups'][key][a][k])
            assert over+under==e['groups'][key][a]['E3_numerator'] and den==e['groups'][key][a]['denominator']
            rebuilt[key][a]=v
    heads={}
    for h in ('35','33','W','G'):
        gg=[v for k,v in rebuilt.items() if k.startswith(h+'/')]
        heads[h]={a:{k:sum(g[a][k] for g in gg)/len(gg) for k in gg[0][a]} for a in ARMS}
        for a in ARMS:
            for k,v in heads[h][a].items():close(v,e['heads'][h][a][k])
    gain=heads['35']['B']['E3']-heads['35']['R']['E3']
    gates=dict(gain35_strict_gt1_4=gain>1.4,degradation33_le1_3=heads['33']['R']['E3']-heads['33']['B']['E3']<=1.3,
        degradationW_le1_4=heads['W']['R']['E3']-heads['W']['B']['E3']<=1.4,
        degradationG_le1_4=heads['G']['R']['E3']-heads['G']['B']['E3']<=1.4,
        f1_35_nondecrease=heads['35']['R']['mean_single_f1']>=heads['35']['B']['mean_single_f1'],
        f1_33_nondecrease=heads['33']['R']['mean_single_f1']>=heads['33']['B']['mean_single_f1'])
    assert gates==e['investment_gate']['conditions'] and all(gates.values())==e['investment_gate']['all_pass']
    close(gain,e['investment_gate']['gain35'])
    return len(e['cells'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);a=p.parse_args()
    raw=a.evidence.read_bytes();raw=gzip.decompress(raw) if a.evidence.suffix=='.gz' else raw
    print('Replayed',replay(json.loads(raw)),'member-clips, both-IoU matching, quotas, oracle cuts, groups, heads and gates')
