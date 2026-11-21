"""Independent §193 math/graph checks. Synthetic-only until explicit sealed field release."""
from collections import deque
from itertools import product,combinations
import json
import numpy as np
import cv2
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import hashlib
import gzip


def count_events(events,nf):
    result=[]
    for o in range(30):
        phase=[[0,0] for _ in range((nf+25*o+749)//750)]
        for f,d in events:phase[(int(f)-1+25*o)//750][int(d<0)]+=1
        result.extend(phase)
    return np.asarray(result,np.int64)


def native_descriptors(raw,pieces,polygon,compute_features=True):
    """Independent native gap resolution, signed last-crossing and10 geometry features."""
    result={};poly=np.asarray(polygon,np.float32)
    for identity in sorted(set(raw[:,1])):
        indices=np.flatnonzero(raw[:,1]==identity);indices=indices[np.argsort(raw[indices,0],kind='stable')]
        cuts=[0,*list(np.flatnonzero(np.diff(raw[indices,0])>25)+1),len(indices)]
        for part,(lo,hi) in enumerate(zip(cuts,cuts[1:])):
            ii=indices[lo:hi];rows=raw[ii];centers=rows[:,2:4]+rows[:,4:6]/2
            sides=[cv2.pointPolygonTest(poly,(float(x),float(y)),False)>=0 for x,y in centers]
            if sides[0]==sides[-1]:continue
            direction=1 if sides[-1] else -1
            cross=max(j for j in range(1,len(rows)) if sides[j]!=sides[j-1] and sides[j]==sides[-1])
            pid=int(identity+part*100000);assert pid not in result
            features=[]
            if compute_features:
                steps=np.linalg.norm(np.diff(centers,axis=0),axis=1);gaps=np.diff(rows[:,0]);body=float(np.median(np.maximum(rows[:,4],rows[:,5])))
                signed=lambda xy:cv2.pointPolygonTest(poly,(float(xy[0]),float(xy[1])),True)/body
                duration=rows[-1,0]-rows[0,0]+1
                features=[int(direction==1),len(rows),duration,1-len(rows)/duration,len(set(pieces[ii])),signed(centers[0]),signed(centers[-1]),
                    float(np.linalg.norm(centers[-1]-centers[0])/steps.sum()),float(np.median(steps/gaps)/body)]
            r=rows[cross]
            result[pid]=dict(frame=int(r[0]),direction=direction,box=[r[2],r[3],r[2]+r[4],r[3]+r[5]],features10=features)
    if compute_features:
        for pid,e in result.items():e['features10'].append(sum(abs(e['frame']-other['frame'])<=25 for other in result.values())-1)
    return result


def standardize(X,residual,domains,videos):
    """Equal domain, equal distinct video, equal distinct (X,r) rows."""
    X=np.asarray(X,float);residual=np.asarray(residual);domains=np.asarray(domains);videos=np.asarray(videos)
    records=[];weights=[]
    source_names=sorted(set(domains))
    for domain in source_names:
        clips=sorted(set(videos[domains==domain]))
        for clip in clips:
            values=sorted(set(tuple(x)+ (int(r),) for x,r,d,v in zip(X,residual,domains,videos) if d==domain and v==clip))
            raw=np.array([row[:-1] for row in values]);labels=np.array([row[-1] for row in values])
            records.append((str(domain),str(clip),raw,labels))
            weights.extend([1/(len(source_names)*len(clips)*len(values))]*len(values))
    raw=np.concatenate([r[2] for r in records]);w=np.array(weights)
    mean=np.sum(raw*w[:,None],axis=0);scale=np.sqrt(np.sum((raw-mean)**2*w[:,None],axis=0));scale[scale<1e-12]=1
    return dict(mean=mean,scale=scale,k=5,records=[(d,v,(x-mean)/scale,r) for d,v,x,r in records])


def survival(model,X,maximum):
    result=np.zeros((len(X),maximum));domains=sorted({r[0] for r in model['records']})
    for i,x in enumerate(np.asarray(X,float)):
        z=(x-model['mean'])/model['scale']
        for domain in domains:
            clips=[]
            for d,v,rows,labels in model['records']:
                if d!=domain:continue
                distances=[sum(float(a-b)**2 for a,b in zip(row,z)) for row in rows]
                nearest=min(distances);responses=[int(y) for y,dist in zip(labels,distances) if dist==nearest]
                clips.append((nearest,v,responses))
            selected=sorted(clips,key=lambda t:(t[0],t[1]))[:5]
            for _,_,responses in selected:
                for j in range(maximum):result[i,j]+=sum(y>=j+1 for y in responses)/(len(domains)*len(selected)*len(responses))
    return result


def allocate(q,counts,second):
    total=sum(counts)
    if not total:return [0,0,0]
    answer=[q*n//total for n in counts]
    order=sorted(range(3),key=lambda i:(-(q*counts[i]%total),(i-second%3)%3))
    for i in order[:q-sum(answer)]:answer[i]+=1
    assert sum(answer)==q and all(a<=n for a,n in zip(answer,counts))
    return answer


def verify_budget_optimum(windows,capacity,costs,q):
    """Independent MILP certificate for one sampled direction; original uses linprog."""
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import csr_matrix,hstack
    cap=np.asarray(capacity,int);q=np.asarray(q,int);T=len(cap);W=len(windows)
    rr=[];cc=[]
    for w,(lo,hi) in enumerate(windows):
        rr.extend([w]*(hi-lo));cc.extend(range(lo,hi))
    A=csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(W,T))
    maxcount=np.rint(A@cap).astype(int);segments=np.repeat(np.arange(W),maxcount)
    if not len(segments):assert not q.any();return dict(objective=0.,deletions=0)
    S=csr_matrix((np.ones(len(segments)),(segments,np.arange(len(segments)))),shape=(W,len(segments)))
    B=hstack((A,-S),format='csr');cost=np.r_[np.zeros(T),np.concatenate([costs[w,:n] for w,n in enumerate(maxcount)])]
    bounds=Bounds(np.zeros(len(cost)),np.r_[cap,np.ones(len(segments))]);eq=LinearConstraint(B,np.zeros(W),np.zeros(W))
    integrality=np.r_[np.ones(T),np.zeros(len(segments))]
    optimum=milp(cost,integrality=integrality,bounds=bounds,constraints=[eq],options={'mip_rel_gap':0.})
    assert optimum.success and optimum.mip_gap==0
    actual=sum(float(costs[w,:int(k)].sum()) for w,k in enumerate(np.rint(A@q).astype(int)))
    assert abs(actual-optimum.fun)<1e-7 and abs(actual-optimum.mip_dual_bound)<1e-7
    face=LinearConstraint(csr_matrix(cost[None,:]),actual,actual)
    tie=milp(np.r_[np.ones(T),np.zeros(len(segments))],integrality=integrality,bounds=bounds,constraints=[eq,face],options={'mip_rel_gap':0.})
    assert tie.success and abs(tie.fun-int(q.sum()))<1e-7
    return dict(objective=actual,deletions=int(q.sum()),independent_mip_gap=float(optimum.mip_gap))


def supported(records):
    """Records require member,frame,direction,box(xyxy); no GT consumed."""
    result=[]
    for event in records:
        ok=False
        for other in records:
            if event['member']==other['member'] or event['direction']!=other['direction'] or abs(event['frame']-other['frame'])>5:continue
            a,b=np.array(event['box']),np.array(other['box'])
            inter=float(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod())
            union=float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter)
            if inter/union>=.5:ok=True;break
        result.append(int(ok))
    return result


def unit_features(unit):
    """Rebuild all support conditions,7 quota features and11 rank features independently."""
    nf=unit['nf'];T=(nf+24)//25;windows=[]
    for offset in range(30):
        for j in range((nf+25*offset+749)//750):windows.append((max(0,30*j-offset),min(T,30*(j+1)-offset)))
    records=[];buckets={};ordered_members=[str(m['member']) for m in unit['members']]
    for m in unit['members']:
        for e in sorted(m['event_records'],key=lambda e:int(e['pid'])):
            row=dict(e,member=str(m['member']));records.append(row)
            bucket=(str(m['member']),(int(e['frame'])-1)//25,int(e['direction']))
            buckets.setdefault(bucket,[]).append((str(m['member']),int(e['pid'])))
    flags=supported(records)
    rank={(e['member'],int(e['pid'])):[*e['features10'],flag] for e,flag in zip(records,flags)}
    result={}
    for direction in (1,-1):
        cap=np.zeros((T,3),int);ss=np.zeros(T,int)
        for e,flag in zip(records,flags):
            if e['direction']==direction:
                t=(e['frame']-1)//25;cap[t,ordered_members.index(e['member'])]+=1;ss[t]+=flag
        X=[];pooled=[]
        for lo,hi in windows:
            counts=cap[lo:hi].sum(0);total=int(counts.sum());pooled.append(total)
            X.append([*sorted(counts.tolist()),int(counts.max()-counts.min()),ss[lo:hi].sum()/max(total,1),
                (min(25*hi,nf)-25*lo)/25,int(direction==1)])
        result[direction]=dict(X=np.array(X,float),capacity=cap,pooled_counts=np.array(pooled,int))
    return dict(windows=windows,buckets=buckets,features=rank,directions=result)


def matching(predictions,edges):
    owner={}
    def add(p,seen):
        for g in sorted(edges.get(p,())):
            if g in seen:continue
            seen.add(g)
            if g not in owner or add(owner[g],seen):owner[g]=p;return True
        return False
    for p in sorted(predictions):add(p,set())
    return len(owner)


def oracle(buckets,quotas,edges):
    """Independent Edmonds-Karp network and reachable cut for one member/clip."""
    capacity={};adj={}
    def edge(a,b,c):
        capacity[a,b]=c;capacity.setdefault((b,a),0);adj.setdefault(a,[]).append(b);adj.setdefault(b,[]).append(a)
    source=('s',);sink=('t',)
    for bucket,ids in buckets.items():
        node=('b',bucket);edge(source,node,quotas[bucket])
        for p in ids:
            edge(node,('p',p),1)
            for g in edges.get(p,()):edge(('p',p),('g',g),1)
    for g in {g for ids in edges.values() for g in ids}:edge(('g',g),sink,1)
    residual=dict(capacity);flow=0
    while True:
        parent={source:None};queue=deque([source])
        while queue and sink not in parent:
            a=queue.popleft()
            for b in adj.get(a,()):
                if residual[a,b]>0 and b not in parent:parent[b]=a;queue.append(b)
        if sink not in parent:break
        node=sink
        while parent[node] is not None:
            prev=parent[node];residual[prev,node]-=1;residual[node,prev]+=1;node=prev
        flow+=1
    cut=sum(c for (a,b),c in capacity.items() if a in parent and b not in parent)
    assert cut==flow
    return flow,cut


def rank_training_check(model,pair_cache=None):
    """Rebuild source normalization, whole-member swap labels and weighted score gradient."""
    rows=[];examples=model['rank_training'];X=[];Y=[];weights=[];npairs=0;opportunities=0;active_videos=0
    for video in examples:
        features={(e['member'],int(e['pid'])):np.asarray(e['features'],float) for e in video['events']}
        rows.append(np.array([features[k] for k in sorted(features)]))
    mean=np.mean([r.mean(0) for r in rows],axis=0)
    scale=np.sqrt(np.mean([((r-mean)**2).mean(0) for r in rows],axis=0));scale[scale<1e-12]=1
    assert np.allclose(mean,model['identity']['mean'],rtol=0,atol=1e-9) and np.allclose(scale,model['identity']['scale'],rtol=0,atol=1e-9)
    for video in examples:
        f={(e['member'],int(e['pid'])):np.array(e['features'],float) for e in video['events']}
        edges={(e['member'],int(e['pid'])):[(e['member'],g) for g in e['legal_gt']] for e in video['events']}
        buckets=[([(b['member'],p) for p in b['pred_ids']],b['keep']) for b in video['buckets']]
        opportunities+=sum(k*(len(ids)-k) for ids,k in buckets)
        key=json.dumps(dict(video=video['video'],buckets=video['buckets'],events=[{k:e[k] for k in ('member','pid','legal_gt')} for e in video['events']]),sort_keys=True)
        if pair_cache is not None and key in pair_cache:pairs=pair_cache[key]
        else:
            kept={p for ids,k in buckets for p in ids[:k]};before=matching(kept,edges);pairs=[]
            for ids,k in buckets:
                for a in ids[:k]:
                    for b in ids[k:]:
                        delta=matching((kept-{a})|{b},edges)-before
                        if delta:pairs.append((a,b,delta))
            if pair_cache is not None:pair_cache[key]=pairs
        declared={(tuple(p['a']),tuple(p['b']),p['delta']) for p in video['pairs']}
        assert set(pairs)==declared
        audit=dict(positive=sum(d>0 for _,_,d in pairs),negative=sum(d<0 for _,_,d in pairs),zero=sum(k*(len(ids)-k) for ids,k in buckets)-len(pairs))
        assert audit==video['audit'];active_videos+=int(bool(pairs))
        for p in video['pairs']:
            a,b=tuple(p['a']),tuple(p['b']);delta=p['delta'];w=1/(len(examples)*2*len(pairs))
            assert abs(w-p['weight_each_reversal'])<1e-15
            difference=(f[b]-f[a])/scale
            X.extend([difference,-difference]);Y.extend([delta,-delta]);weights.extend([w,w])
        npairs+=len(pairs)
    coef=np.array(model['identity']['coef'])
    if not X:assert np.array_equal(coef,np.zeros_like(coef));grad=0.
    else:
        X=np.array(X);Y=np.array(Y);w=np.array(weights);margin=Y*(X@coef)
        grad=float(np.abs(X.T@(-w*Y*np.exp(-np.logaddexp(0,margin)))+coef).max())
        assert grad<5e-5
    assert npairs==model['identity']['pairs']
    return dict(rank_pairs=npairs,rank_pair_opportunities=opportunities,rank_active_videos=active_videos,
                rank_gradient_max=grad,source_videos=[v['video'] for v in examples])


def field(root,historical_cache=None,baseline189=None):
    """Call only after BOTH all-fold deployment and target-score seals are released."""
    import freeze193 as F
    from beetrack import p0 as P
    from beetrack.evaluate import by_track,match_events
    from physical import realize_subset
    root=Path(root);frozen=root/'frozen';deploy=root/'deploy';scored=root/'score'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(scored/'manifest.json')==(scored/'SEALED.sha256').read_text().strip()
    for file,digest in read(scored/'manifest.json').items():assert sha(scored/file)==digest
    seal=read(deploy/'ALL_FOUR_SEAL.json')
    assert sha(deploy/'ALL_FOUR_SEAL.json')==(deploy/'ALL_FOUR_SEAL.sha256').read_text().strip()
    for file,digest in seal['files'].items():assert sha(deploy/file)==digest
    assert seal['protocol']['k']==5 and seal['protocol']['l2']==1
    for file,digest in seal['code_receipts'].items():assert sha(Path(__file__).with_name(file))==digest
    report=json.loads(gzip.decompress((scored/'score193.json.gz').read_bytes()))
    groups=F.load_groups(frozen);features={};native={};rawbycell={};byunit={g['unit']:g for g in groups}
    deployment=read(frozen/'deployment.json');repo=Path.cwd()
    for name,digest in deployment['source_hashes'].items():
        path=(repo/'beetrack/zones.json' if name=='zones' else Path(F.__file__) if name=='extractor' else repo/name)
        assert sha(path)==digest
    # Read only these six registered old GT clips, after the complete score seal.
    truths={};truthbins={};truthpaths={}
    for u in groups:
        expected='m_no'+u['domain'] if u['domain'] in ('W','G') else 'm_dev_strict'
        assert u['linker_model']==expected and u['linker_sha256']==sha(P.MODELS/(expected+'.json'))
        assert u['domain'] in u['linker_excluded_domains']
        clip=u['video']
        if clip not in truths:
            gt=P.gt(u['split'],clip);truths[clip]=gt
            proof=native_descriptors(gt,np.zeros(len(gt)),u['polygon'],compute_features=False)
            truthbins[clip]=count_events([(e['frame'],e['direction']) for e in proof.values()],u['nf'])
            truthpaths[clip]=P.RUNS/'gt'/u['split']/clip/'gt/gt.txt'
        for m in u['members']:
            origin=m['raw_receipt'];kind=origin['source']['kind']
            assert kind in ({'native_clean_rebuild'} if u['domain'] in ('W','G') else {'native_clean_rebuild','section189_head_cache'})
            assert origin['source']['linker']==m['linker_receipt']
            dp=P.RUNS/'dets'/m['tag']/u['split']/(clip+'.npy')
            assert sha(dp)==origin['source']['detector_sha256']
            if kind=='section189_head_cache' and historical_cache is not None:
                assert sha(historical_cache/origin['source']['cache_name'])==origin['source']['cache_sha256']
            with np.load(frozen/origin['file'],allow_pickle=False) as z:pieces=z['s1']
            descriptor=native_descriptors(m['rows'],pieces,u['polygon'])
            assert set(descriptor)=={int(e['pid']) for e in m['event_records']}
            for e in m['event_records']:
                own=descriptor[int(e['pid'])]
                assert (own['frame'],own['direction'])==(e['frame'],e['direction'])
                assert np.allclose(own['features10'],e['features10'],rtol=0,atol=1e-9)
                assert np.array_equal(own['box'],e['box'])
            key=(u['domain'],str(u['group']),str(m['member']),clip)
            native[key]=descriptor;rawbycell[key]=m['rows']
        features[u['unit']]=unit_features(u)
    print('INDEPENDENT_NATIVE_FEATURES_PASS',len(groups),flush=True)
    models={};predictions={};foldchecks={};cache={}
    for heldout in ('33','35','W','G'):
        model=read(deploy/f'{heldout}.model.json');pred=read(deploy/f'{heldout}.predictions.json')
        source=read(root/'source_labels'/f'fold_{heldout}.json');models[heldout]=model
        assert set(model['source_domains'])==set(source['source_domains'])=={'33','35','W','G'}-{heldout}
        assert set(source['clips'])==set(model['source_videos']) and all(c['domain']!=heldout for c in source['clips'].values())
        assert source['freeze_sha256']==sha(frozen/'manifest.json')
        assert model['source_label_receipts']==source['source_receipts']
        for video,proof in source['clips'].items():
            assert proof['gt_sha256']==sha(truthpaths[video]) and np.array_equal(proof['gt_bins'],truthbins[video])
        # Reconstruct raw training rows/residuals directly from GT-free frozen
        # predictions and source-only GT counts, before any model normalization.
        XX=[];rr=[];dd=[];vv=[]
        sourceunits=[u for u in groups if u['domain']!=heldout]
        for video in sorted({u['video'] for u in sourceunits}):
            for u in [u for u in sourceunits if u['video']==video]:
                for d in (1,-1):
                    feat=features[u['unit']]['directions'][d];g=truthbins[video][:,int(d<0)]
                    XX.extend(feat['X']);rr.extend(feat['pooled_counts']-3*g)
                    dd.extend([u['domain']]*len(g));vv.extend([video]*len(g))
        training=model['budget_training']
        assert np.allclose(XX,training['raw_X'],rtol=0,atol=1e-12) and np.array_equal(rr,training['residual'])
        assert dd==training['source'] and vv==training['video']
        own=standardize(XX,rr,dd,vv);budget=model['budget']
        assert budget['k']==5 and np.allclose(own['mean'],budget['mean'],rtol=0,atol=1e-12) and np.allclose(own['scale'],budget['scale'],rtol=0,atol=1e-12)
        assert len(own['records'])==len(budget['records'])
        for a,b in zip(own['records'],budget['records']):
            assert a[0:2]==tuple(b[0:2]) and np.allclose(a[2],b[2],rtol=0,atol=1e-11) and np.array_equal(a[3],b[3])
        # Source rank features and graphs must be the same frozen native streams.
        source_graph={}
        for g in source['groups']:
            for member,value in g['members'].items():
                source_graph[g['clip'],str(member)]=value['evidence']
        for video in model['rank_training']:
            for e in video['events']:
                evidence=source_graph[video['video'],str(e['member'])]
                assert sorted(e['legal_gt'])==sorted(g for p,g in evidence['edges05'] if p==e['pid'])
                matched=[u for u in sourceunits if u['video']==video['video'] and any(str(m['member'])==str(e['member']) for m in u['members'])]
                assert len(matched)==1
                f=features[matched[0]['unit']]['features'][str(e['member']),int(e['pid'])]
                assert np.allclose(f,e['features'],rtol=0,atol=1e-9)
        rankcheck=rank_training_check(model,cache)
        targetunits=[u for u in groups if u['domain']==heldout]
        sampledvideo=sorted(u['video'] for u in targetunits)[0]
        sampledunits=[u for u in targetunits if u['video']==sampledvideo]
        coverage={}
        for domain in sorted(set(dd)):
            videos=sorted({v for d,v in zip(dd,vv) if d==domain})
            coverage[domain]=dict(distinct_videos=len(videos),effective_k=min(5,len(videos)),
                maximum_window_seconds={v:max(float(x[5]) for x,video in zip(XX,vv) if video==v) for v in videos})
        foldchecks[heldout]=dict(**rankcheck,budget_source_rows=len(XX),source_coverage=coverage,
            sampled_cost_video=sampledvideo,cost_samples=[],physical_sample=None)
        for p in pred:
            u=byunit[p['unit']];f=features[p['unit']];predictions[p['unit']]=p
            assert u['domain']==heldout
            declared={(str(e['member']),int(e['pid'])):e['features'] for e in p['rank_features']}
            assert set(declared)==set(f['features'])
            assert all(np.allclose(declared[k],v,rtol=0,atol=1e-9) for k,v in f['features'].items())
            expectedquotas={'B':{},'I':{},'Q':{}}
            for direction in (1,-1):
                ds=f['directions'][direction];proof=p['direction_proof'][str(direction)]
                assert np.allclose(ds['X'],proof['window_features'],rtol=0,atol=1e-12)
                cap=ds['capacity'];pooledcap=cap.sum(1)
                qQ=np.asarray(proof['q_Q'],int);qI=np.asarray(proof['q_I'],int)
                assert np.all(qQ>=0) and np.all(qQ<=pooledcap) and np.all((qI==0)|(qI==pooledcap))
                if u['video']==sampledvideo:
                    maximum=max(sum(pooledcap[lo:hi]) for lo,hi in f['windows'])
                    costs=1-2*survival(own,ds['X'],int(maximum))
                    first=np.zeros(len(pooledcap))
                    if maximum:
                        for w,(lo,hi) in enumerate(f['windows']):first[lo:hi]+=costs[w,0]
                    assert np.array_equal(qI,pooledcap*(first<0))
                    objective=sum(float(costs[w,:int(qQ[lo:hi].sum())].sum()) for w,(lo,hi) in enumerate(f['windows']))
                    assert abs(objective-proof['objective'])<1e-7 and abs(objective-proof['dual'])<1e-7
                    if u['unit']==sampledunits[0]['unit']:check=verify_budget_optimum(f['windows'],pooledcap,costs,qQ)
                    else:check=dict(objective=objective,deletions=int(qQ.sum()))
                    foldchecks[heldout]['cost_samples'].append(dict(unit=u['unit'],direction=direction,**check))
                for t,ns in enumerate(cap):
                    allocations={'B':[0,0,0],'I':allocate(int(qI[t]),ns.tolist(),t),'Q':allocate(int(qQ[t]),ns.tolist(),t)}
                    for i,m in enumerate(u['members']):
                        if ns[i]:
                            b=(str(m['member']),t,direction)
                            for arm in allocations:expectedquotas[arm][b]=int(ns[i]-allocations[arm][i])
            coef=np.array(model['identity']['coef']);mu=np.array(model['identity']['mean']);sd=np.array(model['identity']['scale'])
            scores={k:float(((np.array(v)-mu)/sd)@coef) for k,v in f['features'].items()}
            for m in p['members']:
                member=str(m['member'])
                for arm in ('B','I','Q','R'):
                    assert {(q['second'],q['direction']) for q in m['arms'][arm]['quotas']}=={b[1:] for b in f['buckets'] if b[0]==member}
                    retained=[]
                    for bucket,ids in f['buckets'].items():
                        if bucket[0]!=member:continue
                        k=expectedquotas['Q' if arm=='R' else arm][bucket]
                        ordering=sorted(ids,key=lambda p:-scores[p]) if arm=='R' else ids
                        if arm=='R' and 0<k<len(ids) and abs(scores[ordering[k-1]]-scores[ordering[k]])<=1e-12:ordering=ids
                        retained.extend(pid for _,pid in ordering[:k])
                        declaredquota=next(q for q in m['arms'][arm]['quotas'] if (q['second'],q['direction'])==bucket[1:])
                        assert (declaredquota['capacity'],declaredquota['keep'],declaredquota['delete'])==(len(ids),k,len(ids)-k)
                    assert sorted(retained)==m['arms'][arm]['kept_ids']
        print('INDEPENDENT_SOURCE_AND_DEPLOYMENT_PASS',heldout,flush=True)
    # Replay every retained subset on the original official event graph. The
    # independent flow solver has no dependency on identity_flow.py.
    counts={};scalar={};oracle_room=[];opportunity=dict(buckets=0,nondegenerate=0,changed_Q_R=0)
    celllookup={(c['fold'],str(c['group']),str(c['member']),c['clip']):c for c in report['cells']}
    for key,c in celllookup.items():
        domain,group,member,clip=key;value=c['result'];evidence=value['native_evidence'];raw=rawbycell[key];desc=native[key]
        assert evidence['raw_array_sha256']==hashlib.sha256(np.ascontiguousarray(raw,np.float64).tobytes()).hexdigest()
        signed={int(r[0]):int(r[1]) for r in evidence['predictions']}
        assert signed=={pid:e['frame']*e['direction'] for pid,e in desc.items()}
        gt_events=[(abs(int(date)),1 if date>0 else -1) for _,date in evidence['gt']]
        gc=count_events(gt_events,value['nframes']);assert np.array_equal(gc,truthbins[clip])
        edges={pid:[] for pid in desc}
        for p,g in evidence['edges05']:edges[p].append(g)
        buckets={}
        for pid,e in desc.items():buckets.setdefault(((e['frame']-1)//25,e['direction']),[]).append(pid)
        quotas={b:len(set(ids)&set(value['kept_ids']['Q'])) for b,ids in buckets.items()}
        for b,ids in buckets.items():
            opportunity['buckets']+=1;opportunity['nondegenerate']+=int(0<quotas[b]<len(ids))
            q=set(ids)&set(value['kept_ids']['Q']);r=set(ids)&set(value['kept_ids']['R'])
            opportunity['changed_Q_R']+=int(q!=r);assert len(q)==len(r)
        maximum,cut=oracle(buckets,quotas,edges)
        assert maximum==cut==value['oracle']['maximum_tp']==value['oracle']['minimum_cut_capacity']
        assert not value['oracle']['mandatory_ids']
        pairs=value['oracle']['matching']
        assert len(pairs)==maximum and len({e['pred'] for e in pairs})==maximum and len({e['gt'] for e in pairs})==maximum
        assert all(e['pred'] in value['kept_ids']['O'] and e['gt'] in edges[e['pred']] for e in pairs)
        oracle_room.append(maximum-value['arms']['R']['tp'])
        for arm in ('B','I','Q','R','O'):
            ids=set(value['kept_ids'][arm]);assert ids<=set(desc)
            pc=count_events([(desc[pid]['frame'],desc[pid]['direction']) for pid in ids],value['nframes'])
            tp=matching(ids,edges);n=int(np.abs(pc-gc).sum());den=int(gc.sum());saved=value['arms'][arm]
            assert (len(gt_events),len(ids),tp,n,den)==(saved['gt'],saved['pred'],saved['tp'],saved['W_numerator'],saved['W_denominator'])
            for threshold,edgekey in [('0.5','edges05')]+([('0.3','edges03')] if domain=='33' else []):
                graph={pid:[] for pid in desc}
                for p,g in evidence[edgekey]:graph[p].append(g)
                hit=matching(ids,graph);metric=saved['metrics'][threshold]
                assert (hit,len(ids)-hit,len(gt_events)-hit)==(metric['tp'],metric['fp'],metric['fn'])
                assert abs(2*hit/max(1,len(ids)+len(gt_events))-metric['f1'])<1e-12
            counts[key,arm]=(pc,gc)
            scalar[key,arm]=dict(gt=len(gt_events),pred=len(ids),tp=tp,numerator=n,denominator=den)
        assert np.array_equal(counts[key,'Q'][0],counts[key,'R'][0]) and np.array_equal(counts[key,'Q'][0],counts[key,'O'][0])
        if foldchecks[domain]['physical_sample'] is None:
            unit=next(u for u in groups if u['domain']==domain and str(u['group'])==group and u['video']==clip)
            for arm in ('B','I','Q','R','O'):
                realized,_=realize_subset(raw,unit['polygon'],set(value['kept_ids'][arm]))
                official=match_events(truths[clip],realized,unit['polygon'])
                assert (official['gt'],official['pred'],official['tp'])==(value['arms'][arm]['gt'],value['arms'][arm]['pred'],value['arms'][arm]['tp'])
                if domain=='33':
                    from tools.study103.swap_score import event_scores
                    metric=event_scores(truths[clip],realized,unit['polygon'])[.3]
                    assert metric==value['arms'][arm]['metrics']['0.3']
                actual=P.events(realized,unit['polygon']).astype(int)
                assert np.array_equal(count_events(actual,unit['nf']),counts[key,arm][0])
            foldchecks[domain]['physical_sample']=dict(clip=clip,group=group,member=member,all_five_arms=True)
            print('INDEPENDENT_PHYSICAL_SAMPLE_PASS',domain,flush=True)
    # Pool videos before member metrics, then average members/groups as registered.
    for key,value in report['members'].items():
        domain,group,member=key.split('/')
        cc=[k for k in celllookup if k[:3]==(domain,group,member)]
        for arm in ('B','I','Q','R','O'):
            sums={field:sum(scalar[c,arm][field] for c in cc) for field in ('gt','pred','tp','numerator','denominator')}
            saved=value[arm]
            assert (sums['gt'],sums['pred'],sums['tp'])==(saved['gt'],saved['pred'],saved['tp'])
            assert abs(100*sums['numerator']/sums['denominator']-saved['W'])<1e-10
            assert abs(2*sums['tp']/max(1,sums['pred']+sums['gt'])-saved['f1'])<1e-12
    for key,value in report['groups'].items():
        domain,group=key.split('/');cc=[k for k in celllookup if k[:2]==(domain,group)];videos=sorted({k[3] for k in cc})
        for arm in ('B','I','Q','R','O'):
            numerator=denominator=over=under=0
            for video in videos:
                cells=[k for k in cc if k[3]==video];assert len(cells)==3
                gc=counts[cells[0],arm][1];residual=sum(counts[c,arm][0] for c in cells)-3*gc
                numerator+=int(np.abs(residual).sum());denominator+=3*int(gc.sum())
                over+=int(np.maximum(residual,0).sum());under+=int(np.maximum(-residual,0).sum())
            saved=value[arm]
            assert (numerator,denominator)==(saved['E3_numerator'],saved['denominator'])
            for field,val in [('E3',numerator),('over',over),('under',under)]:assert abs(100*val/denominator-saved[field])<1e-10
            membervalues=[v[arm] for k,v in report['members'].items() if k.startswith(domain+'/'+group+'/')]
            assert abs(np.mean([v['W'] for v in membervalues])-saved['mean_single_W'])<1e-10
            assert abs(np.mean([v['f1'] for v in membervalues])-saved['mean_single_f1'])<1e-12
    for domain,value in report['heads'].items():
        gg=[v for k,v in report['groups'].items() if k.startswith(domain+'/')]
        for arm in ('B','I','Q','R','O'):
            for field,val in value[arm].items():assert abs(np.mean([g[arm][field] for g in gg])-val)<1e-10
    history={}
    if baseline189 is not None:
        old=read(baseline189)['results']['0.40']['base']
        for domain in ('33','35','W','G'):
            before=old[domain]['mean']['e3'];after=report['heads'][domain]['B']['E3']
            if domain in ('33','35'):assert abs(before-after)<1e-9
            history[domain]=dict(old_strict_E3=before,current_B_E3=after,delta=after-before,
                                 interpretation='must reproduce' if domain in ('33','35') else 'clean excluded-domain linker; old strict is historical only')
    return dict(folds=foldchecks,all_passed=True,cells=len(celllookup),identity_opportunity=opportunity,
        total_oracle_TP_room_over_R=sum(oracle_room),score_manifest_sha256=sha(scored/'manifest.json'),
        baseline189=history,historical_inputs_supplied=dict(cache=historical_cache is not None,baseline=baseline189 is not None),
        limits=['Official realized-MOT cross-check uses the shared physical.realize_subset writer, one member/clip per fold; subset matching and flow are independent for every cell.',
                'Nearest-video survival/costs recomputed for one physical target video per fold, all its seed groups and both directions; other target videos audited for features/quotas/retained IDs only.',
                'Oracle maximum TP is optimized at IoU0.5; dense IoU0.3 evaluates the same retained witness, not an independently optimized IoU0.3 oracle.',
                'Detector/tracker generation was not rerun; original input files, clean linker model files, frozen matrices and code fingerprints were checked.'])


def synthetic():
    import harm_budget as H
    rng=np.random.default_rng(193061)
    for _ in range(40):
        X=rng.integers(0,5,(36,7)).astype(float);y=rng.integers(-3,5,36)
        domains=np.repeat(['a','b','c'],12);videos=np.repeat(['a0','a1','b0','b1','c0','c1'],6)
        own=standardize(X,y,domains,videos);model=H.fit(X,y,domains,videos)
        assert np.allclose(own['mean'],model['mean'],atol=1e-13) and np.allclose(own['scale'],model['scale'],atol=1e-13)
        query=rng.integers(0,5,(8,7)).astype(float)
        assert np.allclose(survival(own,query,8),H.survival(model,query,8),atol=1e-13)
    for counts in product(range(4),repeat=3):
        for q in range(sum(counts)+1):
            for second in range(3):assert allocate(q,counts,second)==H.allocate_member_quota(q,np.array(counts),second).tolist()
    for _ in range(80):
        buckets={0:[0,1,2],1:[3,4,5]};quota={b:int(rng.integers(0,4)) for b in buckets}
        edges={p:[g for g in range(5) if rng.random()<.3] for p in range(6)}
        value,cut=oracle(buckets,quota,edges)
        brute=max(matching(set(a)|set(b),edges) for a in combinations(buckets[0],quota[0]) for b in combinations(buckets[1],quota[1]))
        assert value==brute
    r=[dict(member=0,frame=10,direction=1,box=[0,0,10,10]),dict(member=1,frame=15,direction=1,box=[0,0,10,10]),dict(member=2,frame=16,direction=-1,box=[0,0,10,10])]
    assert supported(r)==[1,1,0]
    from field_train193 import synthetic_contract,unit_features as original,fit_fold
    units,labels=synthetic_contract()
    for unit in units:
        own=unit_features(unit);other=original(unit)
        assert own['windows']==other['windows'] and own['buckets']==other['buckets'] and own['features']==other['features']
        for d in (1,-1):
            for k in ('X','capacity','pooled_counts'):assert np.array_equal(own['directions'][d][k],other['directions'][d][k])
    fitted=fit_fold('35',units,{v:l for v,l in labels.items() if l['domain']!='35'})
    rank=rank_training_check(fitted)
    windows=H.intervals(4);capacity=np.array([1,2,0,1]);costs=np.tile(np.array([-.6,-.2,.2,.6]),(len(windows),1))
    solved=H.decode(windows,[[f'{i}/{j}' for j in range(n)] for i,n in enumerate(capacity)],costs)
    opt=verify_budget_optimum(windows,capacity,costs,solved['q'])
    from e2e_selfcheck import fixture,POLY,NF
    from freeze193 import extract
    fixture0=fixture(9193,'independent-native-synthetic')
    for member in fixture0['members'].values():
        raw=member['rows'];pieces=raw[:,1]
        own=native_descriptors(raw,pieces,POLY)
        official=extract(raw,pieces,POLY,NF)
        for e in official:
            row=own[e['track_id']]
            assert (row['frame'],row['direction'])==(e['frame'],e['sign'])
            assert np.allclose(row['features10'],e['features'],rtol=0,atol=1e-9) and np.array_equal(row['box'],e['box'])
    from beetrack.p0 import events as official_events
    negative=np.array([[1,1,14,2,-2,2],[2,1,6,2,-2,2]],float)
    polygon=[[0,0],[10,0],[10,10],[0,10]]
    native=native_descriptors(negative,np.zeros(2),polygon,compute_features=False)
    assert [(e['frame'],e['direction']) for e in native.values()]==[tuple(e) for e in official_events(negative,polygon).astype(int)]
    return dict(source_normalization_survival_cases=40,small_oracle_flow_vs_brute=80,quota_exhaustive=True,support_boundary=True,
        independent_unit_features=len(units),rank_training=rank,independent_milp=opt,native_geometry_feature_check=True,passed=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--historical-cache',type=Path)
    parser.add_argument('--baseline189',type=Path)
    parser.add_argument('--verify-sealed',action='store_true')
    args=parser.parse_args()
    if args.verify_sealed:
        if args.root is None:parser.error('--verify-sealed requires explicit --root')
        result=field(args.root,args.historical_cache,args.baseline189);target=args.root/'independent193.json'
        if target.exists():raise FileExistsError(target)
        target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    else:result=synthetic()
    print(json.dumps(result,indent=2))
