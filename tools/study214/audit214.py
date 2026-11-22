"""Independent214 numeric reference. No field reads; --synthetic only.

cv2.remap is the registered native uint8 interpolation kernel. Coordinates,
cell aggregation, odd features, geometry and immutable-row checks are own code.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_k]='1'
import argparse
import itertools
import numpy as np
import cv2

def patch(image_bgr,xywh):
    x,y,w,h=map(float,xywh)
    if w<=0 or h<=0 or not np.isfinite([x,y,w,h]).all():raise ValueError('invalid box')
    xx=np.empty((16,16),np.float32);yy=np.empty_like(xx)
    for i in range(16):
        for j in range(16):
            xx[i,j]=x+w/2+(j-7.5)*w/8
            yy[i,j]=y+h/2+(i-7.5)*h/8
    return cv2.remap(image_bgr,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[:,:,::-1].copy()

def odd24(rgb):
    rgb=np.asarray(rgb,float)
    if rgb.shape!=(16,16,3):raise ValueError('16x16 RGB required')
    result=[]
    for by in range(2):
        for bx in range(4):
            for channel in range(3):
                plane=rgb[:,:,channel];mu=float(plane.sum()/256);sigma=float(np.sqrt(((plane-mu)**2).sum()/256));scale=max(1.,sigma)
                here=plane[4*by:4*by+4,4*bx:4*bx+4].sum()/16
                opposite=plane[12-4*by:16-4*by,12-4*bx:16-4*bx].sum()/16
                result.append((here-opposite)/(2*scale))
    return np.asarray(result)

def geometry(run,index):
    run=np.asarray(run,float);order=sorted(range(len(run)),key=lambda i:run[i,0]);r=run[index]
    if len(run)==1:role=0;velocity=np.zeros(2)
    else:
        if index==order[0]:role=1;j=order[1]
        elif index==order[-1]:role=-1;j=order[-2]
        else:raise ValueError('not endpoint')
        velocity=np.clip(((run[j,2:4]+run[j,4:6]/2)-(r[2:4]+r[4:6]/2))/r[4:6]/abs(run[j,0]-r[0]),-1,1)
    return np.r_[np.log(r[4]/r[5]),r[6],role,velocity,np.log1p(len(run))]

def row_integrity(before,after,allowed_indices):
    before=np.asarray(before);after=np.asarray(after);assert before.shape==after.shape
    assert np.array_equal(before[:,[0,1,4,5,6]],after[:,[0,1,4,5,6]])
    changed=np.flatnonzero(np.any(before[:,2:4]!=after[:,2:4],axis=1));assert set(changed)<=set(allowed_indices)
    return changed.tolist()

def signed_distance_error(center,GTcenter,xywh,polygon):
    polygon=np.asarray(polygon,np.float32)
    distances=[cv2.pointPolygonTest(polygon,tuple(map(float,p)),True) for p in (center,GTcenter)]
    return abs(distances[0]-distances[1])/np.hypot(xywh[2],xywh[3])

def endpoint_weights(records):
    """Only supplied known source endpoints: hive/clip/run/endpoint equal."""
    keys=[(str(r['domain']),str(r['clip']),int(r.get('canonical_run_id',r.get('run_id')))) for r in records]
    domains={d for d,c,r in keys};out=[]
    for domain,clip,run in keys:
        clips={c for d,c,r in keys if d==domain};runs={r for d,c,r in keys if d==domain and c==clip}
        n=sum(k==(domain,clip,run) for k in keys)
        out.append(1/(len(domains)*len(clips)*len(runs)*n))
    return np.asarray(out,float)

def ridge(X,Y,w):
    X=np.asarray(X,float);Y=np.asarray(Y,float);w=np.asarray(w,float)
    if not len(X):return dict(empty=True)
    w=w/w.sum();mu=w@X;constant=np.all(X==X[0],axis=0);mu[constant]=X[0,constant]
    scale=np.sqrt(w@((X-mu)**2));scale[constant]=0;scale[scale==0]=1
    Z=(X-mu)/scale;b=w@Y
    beta=np.linalg.solve(Z.T@(w[:,None]*Z)+.1*np.eye(X.shape[1]),Z.T@(w[:,None]*(Y-b)))
    residual=Z@beta+b-Y
    assert np.max(abs(2*Z.T@(w[:,None]*residual)+.2*beta))<1e-10
    assert np.max(abs(w@residual))<1e-10
    return dict(mean=mu,scale=scale,beta=beta,intercept=b)

def materialize(rows,endpoints,offsets):
    R=np.asarray(rows,float);result=R.copy();seen=set()
    for e,offset in zip(endpoints,offsets):
        i=int(e['raw_index']);assert i not in seen;seen.add(i)
        if e.get('eligible',True):result[i,2:4]+=np.clip(offset,-.25,.25)*R[i,4:6]
    assert len(endpoints)==len(offsets);row_integrity(R,result,seen)
    return result

def event(frames,sides):
    if sides[0]==sides[-1]:return None
    last=max(j for j in range(1,len(sides)) if sides[j]!=sides[j-1] and sides[j]==sides[-1])
    return int(frames[last]),1 if sides[-1] else -1

def clock(events,nf):
    rows=[]
    for phase in range(30):
        shift=25*phase
        for j in range((nf+shift+749)//750):
            lo=750*j-shift+1;hi=750*(j+1)-shift
            rows.append([sum(lo<=f<=hi and d==direction for f,d in events) for direction in (1,-1)])
    return np.asarray(rows,dtype=np.int64)

def side_menu(frames,sides,allowed):
    """allowed maps endpoint position to permitted bits; singleton shared once."""
    positions=sorted(allowed);assert set(positions)<={0,len(frames)-1}
    events=set()
    for values in itertools.product(*(allowed[j] for j in positions)):
        s=list(sides)
        for j,v in zip(positions,values):s[j]=bool(v)
        events.add(event(frames,s))
    return events

def envelope(menus,nf):
    zero=clock([],nf);lo=zero.copy();hi=zero.copy()
    for menu in menus:
        vectors=[clock([] if e is None else [e],nf) for e in menu]
        lo+=np.min(vectors,axis=0);hi+=np.max(vectors,axis=0)
    return lo,hi

def interval_distance(lo,hi,target):
    assert np.all(lo<=hi)
    return int(np.maximum(lo-target,0).sum()+np.maximum(target-hi,0).sum())

def synthetic():
    import features214 as production
    cv2.setNumThreads(1);rng=np.random.default_rng(214);cases=0
    for image in (np.full((48,64,3),91,np.uint8),rng.integers(0,256,(48,64,3),dtype=np.uint8)):
        for box in ([20,12,8,10],[20.17,12.33,8.41,10.25],[-2,-1,8,10]):
            own=patch(image,box);assert np.array_equal(own,production.sample_patch_rgb(image,box))
            np.testing.assert_allclose(odd24(own),production.chi_from_patch(own),atol=2e-15)
            np.testing.assert_allclose(odd24(own[::-1,::-1]),-odd24(own),atol=2e-15);cases+=1
    run=np.array([[4,7,8,9,2,3,.8],[1,7,2,3,2,3,.9],[2,7,5,6,2,3,.7]],float)
    for rows,indices in ((run,[0,1]),(run[:1],[0])):
        for i in indices:np.testing.assert_allclose(geometry(rows,i),production.geometry_from_run(rows,i),atol=1e-15)
    moved=run.copy();moved[0,2]+=.1;assert row_integrity(run,moved,[0])==[0]
    # Equal unsigned boundary distance cannot substitute for signed distance.
    poly=[[0,0],[10,0],[10,10],[0,10]]
    assert signed_distance_error((1,5),(-1,5),(0,0,2,2),poly)>0
    rows=[dict(domain='a',clip='x',canonical_run_id=1),dict(domain='a',clip='x',canonical_run_id=1),dict(domain='a',clip='x',canonical_run_id=2),dict(domain='b',clip='z',canonical_run_id=3)]
    np.testing.assert_allclose(endpoint_weights(rows),[.125,.125,.25,.5])
    X=rng.normal(size=(9,6));X[:,0]=3;Y=rng.normal(size=(9,2))*.1;w=np.arange(1,10,dtype=float);model=ridge(X,Y,w)
    assert model['scale'][0]==1 and np.array_equal(model['beta'][0],np.zeros(2))
    one=ridge(X[:1],Y[:1],np.ones(1));np.testing.assert_array_equal(one['intercept'],Y[0])
    # Quarter-diagonal eligibility does not imply any capped-axis flip exists.
    center=(-.3,5);assert abs(cv2.pointPolygonTest(np.asarray(poly,np.float32),center,True))<=.25*np.hypot(1,1)
    assert all(cv2.pointPolygonTest(np.asarray(poly,np.float32),(center[0]+dx,center[1]+dy),False)<0 for dx in (-.25,.25) for dy in (-.25,.25))
    # B={original,O} can omit a model side: original and O stay outside, I flips.
    R=np.array([[1,1,-.6,4.5,1,1,.8]],float);O=materialize(R,[dict(raw_index=0)],[[0,0]]);I=materialize(R,[dict(raw_index=0)],[[.25,0]])
    assert O[0,2]+.5<0<I[0,2]+.5
    import bound214 as bound
    for n in range(1,5):
        frames=np.arange(n,dtype=np.int64)*25+1
        for bits in itertools.product((False,True),repeat=n):
            eligible=sorted({0,n-1});own=side_menu(frames,bits,{i:(False,True) for i in eligible})
            prod=bound.run_menu(frames,np.asarray(bits,bool),eligible)
            assert own=={None if e is None else tuple(e) for e in prod['events']}
            for e in own:assert np.array_equal(clock([] if e is None else [e],800),bound.window_counts([] if e is None else [e],800))
    # The coordinatewise lower bound need not have one realizable assignment.
    menu=side_menu([1,26,51],[False,True,False],{0:(False,True),2:(False,True)})
    lo,hi=envelope([menu],800);target=clock([(26,1),(51,-1)],800)
    assert interval_distance(lo,hi,target)==0 and min(int(abs(clock([] if e is None else [e],800)-target).sum()) for e in menu)==30
    return dict(passed=True,synthetic_patch_cases=cases,features=24,field_reads=0,scope='Independent coordinate/odd/geometry loops; shared cv2 native sampler and polygon distance kernels only.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--synthetic',action='store_true');a=p.parse_args()
    if not a.synthetic:p.error('Only synthetic entry enabled before prereg and CPU GO')
    print(synthetic())
