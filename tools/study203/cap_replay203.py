"""Reconstruct exact sparse203 resource packing without dense action windows."""
import numpy as np
from scipy.sparse import coo_matrix


def problem(unit,family,cap):
    shape=cap.phase_shape(unit['nf']);pred=np.asarray(unit['pred_sum']);gt=np.asarray(unit['gt_bins'])
    assert pred.shape==gt.shape==shape and np.issubdtype(pred.dtype,np.integer) and np.issubdtype(gt.dtype,np.integer)
    assert (pred>=0).all() and (gt>=0).all() and int(gt.sum())%30==0
    pool=[a for a in unit['actions'] if cap.permitted(a,family)];actions=[];columns=[];inactive=[]
    for action in pool:
        s=action['delta_sparse'];assert s['shape']==list(shape)
        assert len(s['indices'])==len(s['values']) and len(set(s['indices']))==len(s['indices'])
        col={}
        for index,value in zip(s['indices'],s['values']):
            assert type(index)is int and 0<=index<pred.size and type(value)is int
            if value:col[index]=value
        if not col:inactive.append(action['action_id']);continue
        assert all(1<=cap.integer(row['frame'])<=unit['nf'] for row in action['suffix'])
        actions.append(action);columns.append(col)
    resources=cap.compile_resources(actions);names=list(dict.fromkeys(k for rs in resources for k in rs));lookup={k:i for i,k in enumerate(names)}
    indices=[sorted({lookup[k] for k in rs}) for rs in resources];vr=[];vc=[];vv=[];ar=[];ac=[]
    for j,(column,rs) in enumerate(zip(columns,indices)):
        for i,value in column.items():vr.append(i);vc.append(j);vv.append(value)
        for i in rs:ar.append(i);ac.append(j)
    V=coo_matrix((np.asarray(vv,np.int64),(vr,vc)),shape=(pred.size,len(actions))).tocsc()
    A=coo_matrix((np.ones(len(ar),np.int64),(ar,ac)),shape=(len(names),len(actions))).tocsr()
    return dict(r=(pred-3*gt).ravel(),pred=pred,gt=gt,actions=actions,columns=columns,indices=indices,
                resource_names=names,V=V,A=A,pool_size=len(pool),inactive=inactive)


def verify(unit,family,answer,cap):
    p=problem(unit,family,cap);lower=cap.verify_certificate(p,answer['certificate'])
    upper,pred,x=cap.witness(p,answer['selected_ids'])
    assert answer['lower_bound']==lower<=upper==answer['upper_bound']
    assert answer['gap']==upper-lower and answer['optimal']==(lower==upper)
    assert answer['witness_pred_sum']==pred and answer['denominator']==3*int(p['gt'].sum())
    assert answer['action_order']==[a['action_id'] for a in p['actions']]
    assert list(map(tuple,answer['resource_names']))==p['resource_names'] and answer['action_resources']==p['indices']
    masks={str(i):dict(candidate_ids=[]) for i in unit['member_indices']}
    for a,selected in zip(p['actions'],x):
        if selected:masks[str(a['host_member_index'])]['candidate_ids'].append(a['action_id'])
    assert answer['member_masks']==masks
    return dict(lower=lower,upper=upper,gap=upper-lower)
