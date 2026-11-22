"""Independent §212 source/tiny reference; no field or artifact IO on import."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import hashlib
import json
import numpy as np

def net_event(dates,bits):
    if len(dates)<2 or bits[0]==bits[-1]:return None
    final=bool(bits[-1]);crossing=None
    for i in range(1,len(dates)):
        if bits[i]!=bits[i-1] and bool(bits[i])==final:crossing=int(dates[i])
    assert crossing is not None
    return dict(direction=1 if final else -1,date=crossing)

def phases(dates,bits,labels,k,event):
    counts=[0,0,0];proof=[]
    for phase in range(k):
        indices=[i for i,label in enumerate(labels) if int(label)==phase];groups=[]
        for i in indices:
            if not groups or dates[i]-dates[groups[-1][-1]]>25:groups.append([])
            groups[-1].append(i)
        events=[]
        for group in groups:
            e=net_event([dates[i] for i in group],[bits[i] for i in group])
            if e is not None:events.append(e)
        observed=any(len(group)>=2 for group in groups)
        directional=len(events)==1 and events[0]['direction']==event['direction']
        temporal=directional and abs(events[0]['date']-event['date'])<=5
        for j,yes in enumerate((observed,directional,temporal)):counts[j]+=int(yes)
        proof.append(dict(phase=phase,indices=indices,segments=[[int(dates[i]) for i in g] for g in groups],events=events,O=observed,D=directional,T=temporal))
    return [c/k for c in counts],proof

def seed(dates,k,salt):
    message=f'phase212-v1|{k}|{salt}|'+','.join(map(str,map(int,dates)))
    digest=hashlib.sha256(message.encode('ascii')).digest()
    return int.from_bytes(digest[0:8],byteorder='big')

def features(frames,inside):
    frames=np.asarray(frames);inside=np.asarray(inside)
    if frames.ndim!=1 or inside.shape!=frames.shape or inside.dtype!=np.bool_:raise ValueError('explicit dates and known boolean side bits required')
    if not np.isfinite(frames).all() or np.any(frames<1) or np.any(frames>=2**53) or np.any(frames!=np.floor(frames)):raise ValueError('positive integral exact native dates required')
    order=np.argsort(frames,kind='stable');dates=frames[order].astype(np.int64);bits=inside[order]
    if np.any(np.diff(dates)<=0):raise ValueError('duplicate original date')
    if not len(dates):return dict(valid=False,reason='empty')
    if np.any(np.diff(dates)>25):return dict(valid=False,reason='unresolved gap alias')
    event=net_event(dates,bits)
    if event is None:return dict(valid=False,reason='no net event')
    n=len(dates);flips=sum(bits[i]!=bits[i-1] for i in range(1,n));prefix=0;suffix=0
    for b in bits:
        if b!=bits[0]:break
        prefix+=1
    for b in bits[::-1]:
        if b!=bits[-1]:break
        suffix+=1
    minority=min(int(sum(bits)),n-int(sum(bits)))
    G=[float(np.log1p(n)),n/(int(dates[-1]-dates[0])+1),float(np.log1p(flips)),min(prefix,suffix)/n,minority/n,float(max(np.diff(dates))/25),event['direction']]
    P=[];R=[];alternative=[];evidence={};controls={}
    for k in (2,3):
        labels=np.array([(int(date)-1)%k for date in dates]);p,details=phases(dates,bits,labels,k,event)
        values=[];records=[]
        for salt in range(16):
            number=seed(dates,k,salt);perm=np.random.Generator(np.random.PCG64(number)).permutation(n);shuffled=labels[perm]
            assert sorted(labels.tolist())==sorted(shuffled.tolist())
            value,proof=phases(dates,bits,shuffled,k,event);values.append(value)
            records.append(dict(salt=salt,seed=number,permutation=perm.tolist(),labels=shuffled.tolist(),ODT=value,phases=proof))
        P.extend(p);R.extend(np.mean(values[:8],axis=0).tolist());alternative.extend(np.mean(values[8:],axis=0).tolist())
        evidence[str(k)]=details;controls[str(k)]=records
    return dict(valid=True,G=G,P=P,R=R,R_alt=alternative,T=(np.asarray(P)-R).tolist(),S_residual=(np.asarray(alternative)-R).tolist(),original_event=event,dates=dates.tolist(),inside=bits.tolist(),phases=evidence,controls=controls)

def model_vector(feature,head):
    if head=='G':return feature['G']
    if head=='R':return feature['G']+feature['R']
    if head=='T':return feature['G']+feature['R']+feature['T']
    if head=='S':return feature['G']+feature['R']+feature['S_residual']
    raise ValueError(head)

def event_weights(rows):
    result=np.empty(len(rows));hives={r['domain'] for r in rows}
    for hive in hives:
        clips={r['clip'] for r in rows if r['domain']==hive}
        for clip in clips:
            ii=[i for i,r in enumerate(rows) if r['domain']==hive and r['clip']==clip]
            result[ii]=1/(len(hives)*len(clips)*len(ii))
    return result

def logistic_loss(theta,Z,y,w):
    score=Z@theta[:-1]+theta[-1];probability=np.exp(-np.logaddexp(0,-score));error=w*(probability-y)
    value=float(np.sum(w*(np.logaddexp(0,score)-y*score))+.05*np.sum(theta[:-1]**2))
    return value,np.r_[Z.T@error+.1*theta[:-1],sum(error)]

def fit_logistic(X,y,w):
    from scipy.optimize import minimize
    X=np.asarray(X,float);y=np.asarray(y,float);w=np.asarray(w,float)
    n=len(y)
    if not n:return dict(status='empty',probability=.5)
    if len(set(y))==1:return dict(status='single_class',probability=float((int(sum(y))+1)/(n+2)))
    constant=np.all(X==X[0],axis=0);mean=np.sum(w[:,None]*X,axis=0);mean[constant]=X[0,constant]
    scale=np.sqrt(np.sum(w[:,None]*(X-mean)**2,axis=0));scale[constant]=1
    scale[scale==0]=1
    Z=(X-mean)/scale
    fitted=minimize(lambda t:logistic_loss(t,Z,y,w),np.zeros(X.shape[1]+1),jac=True,method='L-BFGS-B',options=dict(maxiter=500,gtol=1e-8,ftol=1e-12))
    if not fitted.success:raise AssertionError(fitted.message)
    return dict(status='fitted',mean=mean,scale=scale,beta=fitted.x[:-1],intercept=float(fitted.x[-1]),objective=float(fitted.fun),gradient=logistic_loss(fitted.x,Z,y,w)[1],constant_columns=constant)

def predict(model,X):
    if model['status']!='fitted':return np.full(len(X),model['probability'])
    score=((np.asarray(X)-model['mean'])/model['scale'])@model['beta']+model['intercept']
    return np.exp(-np.logaddexp(0,-score))

def inside_bits(rows,polygon):
    """Use the registered boundary-inclusive OpenCV polygon primitive only."""
    import cv2
    polygon=np.asarray(polygon,np.float32)
    centres=np.asarray(rows)[:,2:4]+np.asarray(rows)[:,4:6]/2
    return np.array([cv2.pointPolygonTest(polygon,(float(x),float(y)),False)>=0 for x,y in centres],bool)

def choose_guarded(states,probabilities,actions,base_id):
    table={s['state_id']:s for s in states};base=table[base_id]
    H=lambda s:float(s['nPred']-2*sum(probabilities[s['state_id']]))
    base_H=H(base);eligible=[]
    for state in states:
        if H(state)-base_H<=1e-9:
            identity=tuple(sorted((actions[a]['family'],*actions[a]['endpoint_ids']) for a in state['selected_action_ids']))
            C=float(base['meanCountL1']-state['meanCountL1'])
            eligible.append((state,C,identity))
    assert any(s['state_id']==base_id for s,_,_ in eligible)
    state,C,_=min(eligible,key=lambda item:(-round(item[1],10),item[0]['edits'],item[2]))
    return dict(state_id=state['state_id'],selected_action_ids=state['selected_action_ids'],edits=state['edits'],H=H(state),H_K=base_H,delta_H=H(state)-base_H,C_utility=C,eligible_state_count=len(eligible))

def self_test():
    sustained=features(np.arange(1,19),np.arange(1,19)>=9);assert sustained['P']==[1.]*6
    alternating=features(np.arange(1,19),np.arange(1,19)%2==0);assert alternating['P'][:3]==[1.,0.,0.]
    sparse=features([1,14,27,40],np.array([False,False,True,True]));assert sparse['P'][:3]==[0.]*3
    tiny=features([1,2],np.array([False,True]));assert tiny['P']==[0.]*6
    assert not features([1,27],np.array([False,True]))['valid']
    same=features(np.arange(18,0,-1),(np.arange(1,19)>=9)[::-1]);assert same==sustained
    assert [len(model_vector(sustained,h)) for h in ('G','R','T','S')]==[7,13,19,19]
    assert np.array_equal(model_vector(sustained,'S')[-6:],np.asarray(sustained['R_alt'])-sustained['R'])
    d=[1,6];bits=[False,True]
    assert phases(d,bits,[0,0],2,dict(direction=1,date=11))[0]==[.5,.5,.5]
    assert phases(d,bits,[0,0],2,dict(direction=1,date=12))[0]==[.5,.5,0.]
    for k in (2,3):
        controls=sustained['controls'][str(k)];assert len(controls)==16 and controls[0]['salt']==0 and controls[-1]['salt']==15
    rows=[dict(domain='35' if i<4 else 'W',clip='head' if i<4 else 'short') for i in range(8)]
    weights=event_weights(rows);assert abs(weights.sum()-1)<1e-14
    assert fit_logistic(np.ones((3,2)),[1,1,1],[1/3]*3)['probability']==.8
    assert fit_logistic(np.empty((0,2)),[],[])['probability']==.5
    rng=np.random.default_rng(212);X=np.c_[rng.normal(size=(12,2)),np.ones(12)];y=np.arange(12)%2;w=np.ones(12)/12
    model=fit_logistic(X,y,w);assert model['mean'][-1]==model['scale'][-1]==1 and model['constant_columns'][-1]
    theta=rng.normal(size=4);_,gradient=logistic_loss(theta,X,y,w);numeric=[]
    for j in range(4):
        eps=np.zeros(4);eps[j]=1e-6
        numeric.append((logistic_loss(theta+eps,X,y,w)[0]-logistic_loss(theta-eps,X,y,w)[0])/2e-6)
    assert np.max(abs(gradient-numeric))<1e-8
    states=[dict(state_id='K',nPred=1,selected_action_ids=[],edits=0,meanCountL1=2),dict(state_id='D',nPred=0,selected_action_ids=['I12'],edits=1,meanCountL1=0)]
    actions={'I12':dict(family='I',endpoint_ids=[1,2])}
    assert choose_guarded(states,{'K':[.9],'D':[]},actions,'K')['state_id']=='K'
    assert choose_guarded(states,{'K':[.1],'D':[]},actions,'K')['state_id']=='D'
    return dict(passed=True,scope='source/tiny only',checks=['last same-direction native crossing','empty phases count in k','observed>25 split/no min3','PCG64 16 salts/fixed date-label populations','alt8..15 sham residual','exact temporal5 boundary','7/13/19 dimensions'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.parse_args()
    print(json.dumps(self_test(),sort_keys=True))
