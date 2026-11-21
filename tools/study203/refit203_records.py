"""Bounded zero-start refit from numeric records, with disk-backed packed groups."""
from collections import Counter
from pathlib import Path
import tempfile
import numpy as np
from scipy.optimize import minimize
from replay203_core import equal


def refit(model, records, fitter, work_dir=None):
    counts=Counter();noptions=0;hives=set();weight_sum=0.
    for record in records():
        assert record['hive']!=model['heldout']
        counts[record['loss'],len(record['options'])+1]+=1
        noptions+=len(record['options']);hives.add(record['hive']);weight_sum+=record['weight']
    equal(sorted(hives),model['source_hives'])
    if not counts:
        assert model['status']=='empty' and all(model[k] is None for k in ('mean','scale','beta','intercept'))
        return dict(refitted=True,status='empty')
    assert abs(weight_sum-1)<1e-12 and noptions>0
    with tempfile.TemporaryDirectory(dir=work_dir,prefix='numeric203_fit_') as temporary:
        root=Path(temporary)
        X=np.memmap(root/'features.bin',dtype='float64',mode='w+',shape=(noptions,10))
        w=np.memmap(root/'feature_weights.bin',dtype='float64',mode='w+',shape=(noptions,))
        index=0
        for record in records():
            for option in record['options']:
                X[index]=option['features10'];w[index]=record['weight']/len(record['options']);index+=1
        w/=w.sum();mean=w@X;scale=np.sqrt(w@((X-mean)**2));scale[scale==0]=1
        equal(mean.tolist(),model['mean']);equal(scale.tolist(),model['scale']);del X,w
        packed=[];mapping={};positions={k:0 for k in counts}
        for group_index,((kind,n),count) in enumerate(sorted(counts.items())):
            a=np.memmap(root/f'a{group_index}.bin',dtype='float64',mode='w+',shape=(count,n,11));a[:]=0
            weights=np.memmap(root/f'w{group_index}.bin',dtype='float64',mode='w+',shape=(count,))
            target=np.memmap(root/f't{group_index}.bin',dtype='bool' if kind=='compatible' else 'float64',mode='w+',shape=(count,n))
            mapping[kind,n]=(a,weights,target);packed.append((kind,a,weights,target))
        for record in records():
            key=record['loss'],len(record['options'])+1;index=positions[key];positions[key]+=1
            a,weights,target=mapping[key]
            for row,option in enumerate(record['options']):
                a[index,row,:10]=(np.asarray(option['features10'])-mean)/scale;a[index,row,10]=1
            weights[index]=record['weight']
            target[index]=record['compatible'] if key[0]=='compatible' else record['target']
        options=dict(gtol=1e-10,ftol=1e-14,maxiter=2000,maxls=50)
        equal(options,model['optimization']['options'])
        result=minimize(lambda theta:fitter._packed_loss(theta,packed),np.zeros(11),jac=True,method='L-BFGS-B',options=options)
        assert result.success
        equal(result.x[:10].tolist(),model['beta']);equal(float(result.x[10]),model['intercept'])
        equal(float(result.fun),model['optimization']['objective'])
        # Drop every memmap view before TemporaryDirectory cleanup on Windows.
        del a,weights,target,mapping,packed
    return dict(refitted=True,status='local_numerical_convergence',global_optimality_claim=False)
