"""Independent §209 numeric primitives. No field IO without a future explicit stage.

Source/synthetic CLI only. Never imports producer selection or measurement code.
"""
import argparse
import hashlib
import itertools
import json
import numpy as np


def iou(a, b):
    x=max(0.,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))
    y=max(0.,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    z=x*y
    return z/(a[2]*a[3]+b[2]*b[3]-z)


def pair_catalog(rows, canonical, nf):
    """Deliberately simple independent all-pairs reference, small/unit use."""
    tracks={}
    for j,(row,pid) in enumerate(zip(rows,canonical)):
        d=tracks.setdefault(int(pid),{})
        if int(row[0]) in d: raise ValueError('duplicate canonical date')
        d[int(row[0])]=j
    out=[]
    for a,b in itertools.combinations(sorted(tracks),2):
        dates=sorted(set(tracks[a]) & set(tracks[b]))
        ix=[[tracks[a][d],tracks[b][d]] for d in dates]
        values=[iou(rows[x,2:6],rows[y,2:6]) for x,y in ix]
        if not any(v>0 for v in values): continue
        legal=[j for j,(d,v) in enumerate(zip(dates,values)) if d<nf and 0<v<.8]
        rep=legal[(len(legal)-1)//2] if legal else None
        out.append(dict(pair_id=f'{a}|{b}',shared_dates=dates,row_indices=ix,
                        ious=values,mean_iou=sum(values)/len(values),
                        geometric_eligible=sum(values)/len(values)>=.2,
                        representative_index=rep))
    return out


def intersecting_keys(rows,canonical):
    """Independent x-axis sweep proves candidate completeness without dense IoU."""
    dates={};keys=set()
    for j,row in enumerate(rows): dates.setdefault(int(row[0]),[]).append(j)
    for indices in dates.values():
        active=[]
        for j in sorted(indices,key=lambda j:(rows[j,2],j)):
            x,y,w,h=rows[j,2:6]
            active=[i for i in active if rows[i,2]+rows[i,4]>x]
            for i in active:
                yi,hi=rows[i,3],rows[i,5]
                if min(y+h,yi+hi)>max(y,yi): keys.add(tuple(sorted((int(canonical[i]),int(canonical[j])))))
            active.append(j)
    return keys


def catalog_reference(rows,canonical,nf):
    tracks={}
    for j,(row,pid) in enumerate(zip(rows,canonical)):
        d=tracks.setdefault(int(pid),{})
        if int(row[0]) in d: raise ValueError('duplicate canonical date')
        d[int(row[0])]=j
    result=[]
    for a,b in sorted(intersecting_keys(rows,canonical)):
        dates=sorted(set(tracks[a]) & set(tracks[b]))
        ix=[[tracks[a][d],tracks[b][d]] for d in dates]
        overlaps=[iou(rows[x,2:6],rows[y,2:6]) for x,y in ix]
        options=[i for i,(d,v) in enumerate(zip(dates,overlaps)) if d<nf and 0<v<.8]
        r=options[(len(options)-1)//2] if options else None
        area_ratio=0. if r is None else min(np.prod(rows[ix[r][0],4:6]),np.prod(rows[ix[r][1],4:6]))/max(np.prod(rows[ix[r][0],4:6]),np.prod(rows[ix[r][1],4:6]))
        result.append(dict(pair_id=f'{a}|{b}',canonical_ids=[a,b],shared_dates=dates,row_indices=ix,
                           ious=overlaps,geometry4=[float(np.mean(overlaps)),max(overlaps),float(np.log1p(len(dates))),float(area_ratio)],
                           representative_index=r))
    return result


def private_images(before, after, mask):
    """Independent explicit pixel copy and erosion; no producer/cv2 masking."""
    imgs=[]
    for raw in (before,after):
        image=np.full(raw.shape,127,dtype=np.uint8)
        image[mask]=raw[mask]
        imgs.append(image)
    support=np.zeros(mask.shape,bool)
    for y in range(2,mask.shape[0]-2):
        for x in range(2,mask.shape[1]-2):
            support[y,x]=bool(mask[y-2:y+3,x-2:x+3].all())
    return imgs,support


def rigid_fit(p,q):
    p=np.asarray(p,float);q=np.asarray(q,float)
    pc=p-p.mean(0);qc=q-q.mean(0)
    u,_,vt=np.linalg.svd(qc.T@pc)
    sign=np.diag([1.,np.linalg.det(u@vt)])
    rotation=u@sign@vt
    return rotation,q.mean(0)-rotation@p.mean(0)


def features_from_traces(record):
    """Recompute the four statistics from retained points, independent SVD."""
    fits={}
    for name in ('a','b','background'):
        trace=record['traces'][name]
        keep=np.asarray(trace['keep'],bool)
        if sum(keep)<4: return None
        p=np.asarray(trace['before'],float)[keep];q=np.asarray(trace['after'],float)[keep]
        for xy in (p,q):
            singular=np.linalg.svd(xy-np.mean(xy,axis=0),compute_uv=False)
            if singular[0]<=0 or singular[1]/singular[0]<.05: return None
        rotation,translation=rigid_fit(p,q)
        fits[name]=(p,q,rotation,translation)
    def error(source,target):
        p,q,_,_=fits[target];_,_,r,t=fits[source]
        return np.sqrt(np.sum((p@r.T+t-q)**2,axis=1))
    ba,bb=error('background','a'),error('background','b')
    if min(np.median(ba),np.median(bb))<.5: return None
    diag=float(record['mean_diagonal'])
    def rms(a,b): return float(np.sqrt(np.mean(np.r_[a,b]**2))/diag)
    motions=[np.median(fits[n][1]-fits[n][0],axis=0) for n in ('a','b')]
    return [float(np.linalg.norm(motions[0]-motions[1])/diag),
            rms(error('a','b'),error('b','a')),rms(ba,bb),rms(error('a','a'),error('b','b'))]


def mask_boxes(boxes,roi):
    x0,y0,x1,y1=roi
    mask=np.zeros((y1-y0,x1-x0),bool)
    for yy in range(y0,y1):
        for xx in range(x0,x1):
            mask[yy-y0,xx-x0]=any(x<=xx<x+w and y<=yy<y+h for x,y,w,h in boxes)
    return mask


def independently_track(before,after,mask,offset):
    """Native OpenCV primitives, independent private inputs and endpoint gates."""
    import cv2
    cv2.setNumThreads(1)
    images,support=private_images(before,after,mask)
    corners=cv2.goodFeaturesToTrack(images[0],16,.01,3,mask=support.astype(np.uint8)*255,blockSize=3)
    hashes=[hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    if corners is None: return {'before':[],'after':[],'keep':[],'image_hashes':hashes}
    right,status1,_=cv2.calcOpticalFlowPyrLK(images[0],images[1],corners,None,winSize=(15,15),maxLevel=2)
    if right is None: return {'before':(corners.reshape(-1,2)+offset).tolist(),'after':[],'keep':[],'image_hashes':hashes}
    finite=np.isfinite(right).all(axis=(1,2));safe=right.copy();safe[~finite]=corners[~finite]
    left,status2,_=cv2.calcOpticalFlowPyrLK(images[1],images[0],safe,None,winSize=(15,15),maxLevel=2)
    if left is None: return {'before':(corners.reshape(-1,2)+offset).tolist(),'after':(safe.reshape(-1,2)+offset).tolist(),'keep':[],'image_hashes':hashes}
    p=corners.reshape(-1,2);q=right.reshape(-1,2)
    error=np.linalg.norm(left.reshape(-1,2)-p,axis=1)
    def supported(point):
        if not np.isfinite(point).all(): return False
        x,y=np.rint(point).astype(int)
        return 0<=y<support.shape[0] and 0<=x<support.shape[1] and bool(support[y,x])
    keep=[bool(status1[j,0] and status2[j,0] and finite[j] and np.isfinite(error[j]) and error[j]<=1 and
               0<=q[j,0]<before.shape[1] and 0<=q[j,1]<before.shape[0] and supported(p[j]) and supported(q[j])) for j in range(len(p))]
    return {'before':(p+offset).tolist(),'after':(q+offset).tolist(),'keep':keep,'image_hashes':hashes}


def merge_rows(rows,canonical,pairs,all_pairs=None):
    """Whole-run disjoint merge; overlapping date keeps highest score/raw first."""
    selected=set();remove=set();mapping={}
    ranks={p:i for i,p in enumerate(sorted(all_pairs if all_pairs is not None else pairs))}
    base=max(max(map(int,canonical),default=0),int(rows[:,1].max()) if len(rows) else 0)+1
    for a,b in pairs:
        if a==b or a in selected or b in selected: raise ValueError('non-disjoint slate')
        selected.update((a,b));mapping[a]=mapping[b]=base+ranks[a,b]
        dates={}
        for j in range(len(rows)):
            if canonical[j] in (a,b): dates.setdefault(int(rows[j,0]),[]).append(j)
        for indices in dates.values():
            winner=min(indices,key=lambda j:(-float(rows[j,6]),j))
            remove.update(j for j in indices if j!=winner)
    retained=[j for j in range(len(rows)) if j not in remove]
    out=np.asarray(rows)[retained].copy()
    for i,j in enumerate(retained): out[i,1]=mapping.get(int(canonical[j]),rows[j,1])
    return out,retained


def weak_date(boxes,gt):
    """Rectangle compatibility only; GT reverse degree is intentionally absent."""
    gt=np.asarray(gt,float).reshape(-1,6)
    boxes=np.asarray(boxes,float)
    if not np.isfinite(boxes).all() or np.any(boxes[:,2:]<=0): return 'U'
    if not len(gt) or not np.isfinite(gt).all() or np.any(gt[:,4:6]<=0): return 'U'
    if np.any(gt[:,:2]!=np.rint(gt[:,:2])): return 'U'
    chosen=[]
    for b in boxes:
        hits=[]
        for j,g in enumerate(gt):
            a=g[2:6]
            wh=np.maximum(0,np.minimum(a[:2]+a[2:],b[:2]+b[2:])-np.maximum(a[:2],b[:2]))
            inter=wh[0]*wh[1]
            overlap=inter/(a[2]*a[3]+b[2]*b[3]-inter+1e-9)
            if overlap>=.3: hits.append(j)
        if len(hits)!=1: return 'U'
        oid=int(gt[hits[0],1])
        if oid<0 or sum(gt[:,1]==oid)!=1: return 'U'
        chosen.append(oid)
    return 'S' if chosen[0]==chosen[1] else 'D'


def loss_gradient(theta,z,y,w):
    """Independent convex logistic formula, unpenalized intercept, lambda=.1."""
    eta=np.asarray(z)@theta[:-1]+theta[-1]
    probability=np.exp(-np.logaddexp(0.,-eta))
    loss=np.dot(w,np.logaddexp(0.,eta)-y*eta)+.05*np.dot(theta[:-1],theta[:-1])
    residual=w*(probability-y)
    return float(loss),np.r_[z.T@residual+.1*theta[:-1],sum(residual)]


def refit(training_records,head):
    from scipy.optimize import minimize
    d={'G':4,'F':5,'X':8}[head]
    x=np.asarray([r['geometry4']+r['pixels4'] for r in training_records],float)
    y=np.asarray([r['y'] for r in training_records],float)
    weight=np.empty(len(y))
    for c in (0,1):
        domains={r['domain'] for r in training_records if r['y']==c}
        if not domains: return {'status':'noop_missing_class'}
        for h in domains:
            idx=[j for j,r in enumerate(training_records) if r['y']==c and r['domain']==h]
            weight[idx]=.5/(len(domains)*len(idx))
    mean=np.sum(x*weight[:,None],axis=0)
    scale=np.sqrt(np.sum((x-mean)**2*weight[:,None],axis=0));scale[scale<1e-6]=1
    z=(x[:,:d]-mean[:d])/scale[:d]
    result=minimize(lambda t:loss_gradient(t,z,y,weight),np.zeros(d+1),jac=True,
                    method='L-BFGS-B',options={'maxiter':500,'gtol':1e-8,'ftol':1e-12})
    if not result.success: raise AssertionError(result.message)
    return dict(status='fitted',mean=mean[:d],scale=scale[:d],beta=result.x[:-1],
                intercept=result.x[-1],weight=weight,objective=result.fun,
                gradient=loss_gradient(result.x,z,y,weight)[1])


def select(rows,logits,active=True):
    def ids(r): return tuple(map(int,r['pair_id'].split('|')))
    def greedy(pixel):
        pool=[r for r in rows if r['geometry4'][0]>=.2 and (not pixel or r['true']['valid'])]
        pool.sort(key=lambda r:(-r['geometry4'][0],ids(r)))
        used=set();out=[]
        for row in pool:
            if not used.intersection(ids(row)): out.append(row['pair_id']);used.update(ids(row))
        return out
    common=greedy(True);n=len(common);k=(n+3)//4 if n>=4 else 0
    out={'K':[],'J':greedy(False),'U':common[:k]}
    for head in ('G','F','X'):
        out[head]=sorted(common,key=lambda p:(-logits[head][p],tuple(map(int,p.split('|')))))[:k] if active else []
    return out,common,k


def average_precision(labels,scores):
    """Threshold-group AP, half/tied scores receive one shared precision step."""
    from fractions import Fraction
    if not labels or not sum(labels):return None
    ranked=sorted(zip(scores,labels),reverse=True);tp=fp=0;area=Fraction(0)
    j=0
    while j<len(ranked):
        end=j+1
        while end<len(ranked) and ranked[end][0]==ranked[j][0]:end+=1
        new=sum(y for _,y in ranked[j:end]);tp+=new;fp+=end-j-new
        area+=Fraction(new*tp,(tp+fp)*sum(labels));j=end
    return float(area)


def self_test():
    rng=np.random.default_rng(209)
    a=rng.integers(0,256,(40,50),dtype=np.uint8);b=a.copy()
    mask=np.zeros(a.shape,bool);mask[4:35,6:39]=True
    original,support=private_images(a,b,mask)
    a2=a.copy();b2=b.copy()
    a2[~mask]=rng.integers(0,256,int((~mask).sum()),dtype=np.uint8)
    b2[~mask]=rng.integers(0,256,int((~mask).sum()),dtype=np.uint8)
    changed,s2=private_images(a2,b2,mask)
    assert all(np.array_equal(x,y) for x,y in zip(original,changed))
    assert np.array_equal(support,s2) and support.sum()==27*29
    trace=independently_track(a,b,mask,np.zeros(2))
    altered=independently_track(a2,b2,mask,np.zeros(2))
    assert trace==altered
    p=rng.normal(size=(12,2));angle=.3
    r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    q=p@r.T+[2,3];rr,t=rigid_fit(p,q)
    assert np.max(abs(rr-r))<1e-12 and np.max(abs(t-[2,3]))<1e-12
    rows=np.array([[1,1,0,0,2,2,.8],[2,1,0,0,2,2,.7],
                   [1,2,1,0,2,2,.9],[2,2,8,0,2,2,.6]],float)
    cat=pair_catalog(rows,rows[:,1].astype(int),3)
    assert len(cat)==1 and cat[0]['mean_iou']==1/6
    sweep=catalog_reference(rows,rows[:,1].astype(int),3)
    assert sweep[0]['shared_dates']==cat[0]['shared_dates'] and sweep[0]['ious']==cat[0]['ious']
    merged,kept=merge_rows(rows,rows[:,1].astype(int),[(1,2)])
    assert kept==[1,2] and len(merged)==2
    gt=np.array([[1,4,0,0,2,2],[1,8,8,0,2,2]],float)
    assert weak_date([[0,0,2,2],[0,0,2,2]],gt)=='S'
    assert weak_date([[0,0,2,2],[8,0,2,2]],gt)=='D'
    assert weak_date([[0,0,2,2],[8,0,2,2]],np.r_[gt,[[1,4,20,0,2,2]]])=='U'
    z=rng.normal(size=(9,4));y=rng.integers(0,2,9);w=np.ones(9)/9;theta=rng.normal(size=5)
    value,gradient=loss_gradient(theta,z,y,w)
    numeric=[]
    for j in range(5):
        eps=np.zeros(5);eps[j]=1e-6
        numeric.append((loss_gradient(theta+eps,z,y,w)[0]-loss_gradient(theta-eps,z,y,w)[0])/2e-6)
    assert np.max(abs(gradient-numeric))<1e-8
    return {'pass':True,'scope':'synthetic only; no field data/model/image access',
            'checks':['all-shared-date zero contribution','fixed-mask isolation',
                      'two-pixel erosion','independent masked LK invariance','sweep completeness','rigid transform','whole-run merge tie/order',
                      'weak S/D/U incl duplicate GT identity','logistic finite-difference gradient']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true')
    parser.parse_args()
    print(json.dumps(self_test(),sort_keys=True))
