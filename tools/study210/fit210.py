"""Source-only G logistic plus frozen-G-offset full-support correction.

All eligible known S/D rows train, including conflicting/nonactive pairs.
Each fit independently gives each class .5, each class-containing source hive
equal weight, then each hive-class row equal weight (no extra clip balancing).
Unsupported X/M bypass correction, including its intercept, exactly.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

DOMAINS=('35','33','W','G')
ARMS=('K','N','G','X','M','ALL')
G_NAMES=('min_offset_norm','median_offset_norm','log1p_brackets','orientation_balance','full_support','suffix_zero_rms')
X_NAMES=('prefix_offset_norm','suffix_zero_minus_offset_rms')
RIDGE=.1
SCHEMA='interleave210-offset-logistic-v1'
CLIPS={('35','BEE24-35-head',3),('33','BEE24-33-head',3),('W','BEE24-10',0),('W','BEE24-15',0),('G','BEE24-24',0),('G','BEE24-28',0)}
SOURCE208='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94'
SOURCE208_LABELS='e73ba698916d5e60fda6289b04f7a9d53d75ef1767621250370bf9bb299a473a'
CODE_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')


def pair_ids(r):
    try:a,b=map(int,r['pair_id'].split('|'))
    except (ValueError,TypeError):raise ValueError('numeric canonical A|B pair required') from None
    if a>=b or r['pair_id']!=f'{a}|{b}':raise ValueError('canonical pair A<B required')
    if 'canonical_ids' in r and list(r['canonical_ids'])!=[a,b]:raise ValueError('canonical pair alias differs')
    return a,b


def row_key(r):return r['domain']+'/'+r['clip']+'/'+r['pair_id']


def validate(rows):
    seen=set()
    for r in rows:
        if r['domain'] not in DOMAINS or not isinstance(r['clip'],str):raise ValueError('registered source domain/clip required')
        pair_ids(r);key=row_key(r)
        if key in seen:raise ValueError('duplicate candidate row')
        seen.add(key)
        for name in ('eligible','active','full_support'):
            if not isinstance(r[name],(bool,np.bool_)):raise ValueError(name+' must be frozen bool')
        g,x,m=(np.asarray(r[name],float) for name in ('G6','X2','Xminus2'))
        if g.shape!=(6,) or x.shape!=(2,) or m.shape!=(2,) or not all(np.isfinite(v).all() for v in (g,x,m)):raise ValueError('finite G6/X2/Xminus2 required')
        if g[0]<0 or g[1]+1e-12<g[0] or g[2]<0 or not 0<=g[3]<=1 or g[4]!=float(r['full_support']) or g[5]<0:raise ValueError('G feature semantics differ')
        if x[0]!=m[0] or x[0]<0:raise ValueError('wrong-sign control must retain offset norm')
        if not r['full_support'] and (np.any(x!=0) or np.any(m!=0)):raise ValueError('unsupported correction features must be zero placeholders')
        local=float(r['min_offset_norm'])
        if not np.isfinite(local) or local<0 or (r['eligible'] and local>1):raise ValueError('local three-box-mean normalized distance required')
    return rows


def source_weights(rows,y):
    y=np.asarray(y)
    if y.shape!=(len(rows),) or not np.isin(y,[0,1]).all():raise ValueError('binary source targets required')
    counts=Counter((int(t),r['domain']) for r,t in zip(rows,y))
    hives={t:{h for (c,h),n in counts.items() if c==t and n} for t in (0,1)}
    if not all(hives.values()):return None
    return np.asarray([.5/(len(hives[int(t)])*counts[int(t),r['domain']]) for r,t in zip(rows,y)],float)


def objective(theta,Z,y,w,offset):
    beta,bias=theta[:-1],theta[-1];z=offset+Z@beta+bias
    residual=w*(expit(z)-y)
    return float(w@(np.logaddexp(0.,z)-y*z)+RIDGE/2*(beta@beta)),np.r_[Z.T@residual+RIDGE*beta,residual.sum()]


def _fit_head(X,y,w,offset,names):
    if w is None:
        return dict(status='zero_missing_class',feature_names=list(names),mean=None,scale=None,beta=[0.]*len(names),intercept=0.,
                    optimization=dict(success=True,message='missing source class; fixed zero fallback',iterations=0))
    mean=w@X;scale=np.sqrt(w@((X-mean)**2));scale[scale<1e-6]=1.;Z=(X-mean)/scale
    opt=minimize(lambda t:objective(t,Z,y,w,offset),np.zeros(len(names)+1),jac=True,method='L-BFGS-B',options=dict(maxiter=500,gtol=1e-8,ftol=1e-12))
    if not opt.success or not np.isfinite(opt.x).all():raise RuntimeError('source offset logistic failed: '+str(opt.message))
    return dict(status='fitted',feature_names=list(names),mean=mean.tolist(),scale=scale.tolist(),beta=opt.x[:-1].tolist(),intercept=float(opt.x[-1]),
        optimization=dict(success=True,message=str(opt.message),iterations=int(opt.nit),objective=float(opt.fun),gradient=objective(opt.x,Z,y,w,offset)[1].tolist(),
            initialization=[0.]*(len(names)+1),solver='scipy L-BFGS-B',options=dict(maxiter=500,gtol=1e-8,ftol=1e-12),intercept_penalized=False))


def _head_score(head,x):
    if head['status']!='fitted':return 0.
    return float(((np.asarray(x,float)-head['mean'])/head['scale'])@head['beta']+head['intercept'])


def fit(rows,labels,*,heldout=None):
    validate(rows)
    if heldout is not None and any(r['domain']==heldout for r in rows):raise ValueError('heldout source candidate')
    if set(labels)!={row_key(r) for r in rows} or any(v not in ('S','D','U') for v in labels.values()):raise ValueError('complete source S/D/U label inventory required')
    known=sorted([r for r in rows if r['eligible'] and labels[row_key(r)]!='U'],key=row_key)
    y=np.asarray([labels[row_key(r)]=='S' for r in known],float);w=source_weights(known,y)
    gX=np.asarray([r['G6'] for r in known],float).reshape(-1,6)
    G=_fit_head(gX,y,w,np.zeros(len(known)),G_NAMES)
    indices=[i for i,r in enumerate(known) if r['full_support']];full=[known[i] for i in indices]
    cy=y[indices];cw=source_weights(full,cy)
    offset=np.asarray([_head_score(G,r['G6']) for r in full])
    cX=np.asarray([r['X2'] for r in full],float).reshape(-1,2)
    correction=_fit_head(cX,cy,cw,offset,X_NAMES)
    full_weights={i:None if cw is None else float(cw[j]) for j,i in enumerate(indices)}
    return dict(schema=SCHEMA,heldout=heldout,source_domains=[h for h in DOMAINS if h!=heldout],ridge=RIDGE,
        G=G,correction=correction,correction_offset='frozen fitted G raw logit; never refit G on supported subset',
        source_counts={h:{c:sum(r['domain']==h and labels[row_key(r)]==c for r in known) for c in ('S','D')} for h in DOMAINS if h!=heldout},
        full_support_source_counts={h:{c:sum(r['domain']==h and labels[row_key(r)]==c for r in full) for c in ('S','D')} for h in DOMAINS if h!=heldout},
        weight_rule='independently each fit: class .5, nonempty source hive equal within class, then hive-class rows equal; no additional clip weight',
        training_records=[dict(key=row_key(r),domain=r['domain'],clip=r['clip'],pair_id=r['pair_id'],G6=r['G6'],X2=r['X2'],Xminus2=r['Xminus2'],
            full_support=r['full_support'],active=r['active'],y=int(y[i]),weight_G=None if w is None else float(w[i]),
            weight_correction=full_weights.get(i),G_offset=_head_score(G,r['G6'])) for i,r in enumerate(known)])


def scores(model,row):
    G=_head_score(model['G'],row['G6'])
    if not row['full_support'] or model['correction']['status']!='fitted':
        return dict(G=G,X=G,M=G)  # Exact bypass, including correction intercept.
    return dict(G=G,X=G+_head_score(model['correction'],row['X2']),M=G+_head_score(model['correction'],row['Xminus2']))


def reference_slate(rows):
    selected=[];used=set()
    for r in sorted((r for r in rows if r['eligible']),key=lambda r:(not r['active'],r['min_offset_norm'],pair_ids(r))):
        ids=pair_ids(r)
        if not used.intersection(ids):selected.append(r['pair_id']);used.update(ids)
    return selected


def deploy_clip(rows,model,slate_ids=None):
    validate(rows)
    if len({(r['domain'],r['clip']) for r in rows})>1:raise ValueError('single clip required')
    expected=reference_slate(rows)
    if slate_ids is None:slate_ids=expected
    elif list(slate_ids)!=expected:raise ValueError('frozen common slate/ranking differs')
    byid={r['pair_id']:r for r in rows};active=[byid[k] for k in slate_ids if byid[k]['active']]
    a=len(active);k=(a+3)//4 if a else 0;raw={r['pair_id']:scores(model,r) for r in rows}
    selected=dict(K=[],N=[r['pair_id'] for r in sorted(active,key=lambda r:(r['min_offset_norm'],pair_ids(r)))[:k]],ALL=[r['pair_id'] for r in active])
    for arm in ('G','X','M'):selected[arm]=[r['pair_id'] for r in sorted(active,key=lambda r:(-raw[r['pair_id']][arm],pair_ids(r)))[:k]]
    order={key:i for i,key in enumerate(slate_ids)}
    return dict(slate=list(slate_ids),active_slate=[r['pair_id'] for r in active],a=a,k=k,selected=selected,
        records=[dict(key=row_key(r),pair_id=r['pair_id'],canonical_ids=list(pair_ids(r)),eligible=r['eligible'],active=r['active'],full_support=r['full_support'],
            G6=r['G6'],X2=r['X2'],Xminus2=r['Xminus2'],min_offset_norm=r['min_offset_norm'],slate_order=order.get(r['pair_id']),logits=raw[r['pair_id']]) for r in sorted(rows,key=pair_ids)],
        statuses=dict(G=model['G']['status'],correction=model['correction']['status']),
        ranking='G/X/M descending raw logit; N ascending min local offset; exact ties numeric(A,B) ascending',
        fallback='missing G class =>0 logit; missing correction class =>0 correction; fixed budget still selected',
        budget='ceil(active_slate/4) for any a>0; ALL=all active common-slate pairs')


def grouped_average_precision(y,logits,min_each=5):
    """Diagnostic helper; caller must restrict to full-support known S/D only."""
    y=np.asarray(y);z=np.asarray(logits,float)
    if y.shape!=z.shape or y.ndim!=1 or not np.isin(y,[0,1]).all() or not np.isfinite(z).all():raise ValueError('finite logits and binary labels required')
    positive=int((y==1).sum());negative=int((y==0).sum())
    if min(positive,negative)<min_each:return dict(AP=None,positive=positive,negative=negative,reason='requires at least5 S and5 D')
    total=tp=0;ap=0.
    for value in np.unique(z)[::-1]:
        mask=z==value;new=int((y[mask]==1).sum());tp+=new;total+=int(mask.sum());ap+=(new/positive)*(tp/total)
    return dict(AP=ap,positive=positive,negative=negative,tie_rule='entire equal-logit group added before precision at group end')


def train_and_seal(rows,label_provider,outdir,input_receipt,clip_slates):
    validate(rows)
    if sha(__file__)!=CODE_SHA:raise ValueError('loaded source changed')
    if any(any(k in r for k in ('label','labels','y')) for r in rows):raise ValueError('deployment input must be unlabelled')
    dest=Path(outdir)
    if dest.exists():raise FileExistsError('immutable output root exists')
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        predictions={};folds={}
        for target in DOMAINS:
            supplied=label_provider(target)
            if set(supplied['source_domains'])!=set(DOMAINS)-{target}:raise ValueError('other-three source whitelist required')
            source=[r for r in rows if r['domain']!=target];model=fit(source,supplied['labels'],heldout=target)
            model['source_receipts']=supplied.get('receipts',{});name='models/'+target+'.json';save(tmp/name,model)
            folds[target]=dict(heldout=target,source_domains=[h for h in DOMAINS if h!=target],model_file=name,
                G_status=model['G']['status'],correction_status=model['correction']['status'])
            for (domain,clip),slate in clip_slates.items():
                if domain==target:predictions[domain+'/'+clip]=dict(domain=domain,clip=clip,**deploy_clip([r for r in rows if (r['domain'],r['clip'])==(domain,clip)],model,slate))
        if sha(__file__)!=CODE_SHA:raise ValueError('source changed during fitting')
        save(tmp/'predictions.json',dict(schema=SCHEMA,arms=list(ARMS),folds=folds,clips=predictions))
        save(tmp/'receipt.json',dict(input_receipt,schema=SCHEMA,code_sha256=CODE_SHA,numpy_version=np.__version__,scipy_version=scipy.__version__,all_target_labels_prohibited_before_common_seal=True))
        save(tmp/'manifest.json',{p.relative_to(tmp).as_posix():sha(p) for p in tmp.rglob('*') if p.is_file()})
        digest=sha(tmp/'manifest.json');(tmp/'SEALED.sha256').write_text(digest+'\n');tmp.rename(dest)
        return dict(root=str(dest),manifest_sha256=digest)
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def verify_seal(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('common prediction seal differs')
    files=json.loads((root/'manifest.json').read_text())
    if set(files)!={'receipt.json','predictions.json'}|{'models/'+h+'.json' for h in DOMAINS}:raise ValueError('four model heads/common predictions required')
    for name,digest in files.items():
        if sha(root/name)!=digest:raise ValueError('sealed output changed')
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt['code_sha256']!=sha(__file__):raise ValueError('frozen fit source changed')
    predictions=json.loads((root/'predictions.json').read_text())
    if set(predictions['folds'])!=set(DOMAINS):raise ValueError('all four folds required')
    for h,f in predictions['folds'].items():
        if f['heldout']!=h or set(f['source_domains'])!=set(DOMAINS)-{h}:raise ValueError('source boundary differs')
    return receipt


def load_predictions(root):
    receipt=verify_seal(root);predictions=json.loads((Path(root)/'predictions.json').read_text());decisions={}
    for c in predictions['clips'].values():decisions.setdefault(c['domain'],{})[c['clip']]=c['selected']
    return dict(receipt=receipt,predictions=predictions,decisions=decisions)


def manifest(root):
    root=Path(root);digest=sha(root/'manifest.json')
    if (root/'SEALED.sha256').read_text().strip()!=digest:raise ValueError('input seal differs')
    files=json.loads((root/'manifest.json').read_text())
    for name in files:
        if not (root/name).resolve().is_relative_to(root.resolve()):raise ValueError('input manifest path escapes root')
    return dict(manifest_sha256=digest,files=files)


def checked(root,receipt,name):
    if name not in receipt['files'] or sha(Path(root)/name)!=receipt['files'][name]:raise ValueError('sealed input changed: '+name)
    return Path(root)/name


def load_numeric(candidates,receipt,prereg):
    """Project sealed numeric features; never load raw K, images, or labels."""
    top=json.loads(checked(candidates,receipt,'deployment.json').read_text())
    if top['schema']!='interleave210-freeze-v1' or top['prereg']!=prereg or top['source208']!=SOURCE208:raise ValueError('catalog schema/registration/lineage differs')
    if len(top['clips'])!=6 or {(r['domain'],r['clip'],r['seed']) for r in top['clips']}!=CLIPS:raise ValueError('fixed six old clips required')
    rows=[];slates={};support=[]
    for ref in top['clips']:
        domain,clip=ref['domain'],ref['clip'];path=domain+'/'+clip
        if ref['path']!=path or sha(checked(candidates,receipt,path+'/manifest.json'))!=ref['seal']:raise ValueError('catalog clip path/seal differs')
        with gzip.open(checked(candidates,receipt,path+'/catalog.json.gz'),'rt',encoding='utf8') as stream:data=json.load(stream)
        if data['schema']!='interleave210-catalog-v1' or data['metadata']['domain']!=domain or data['metadata']['clip']!=clip:raise ValueError('catalog clip identity differs')
        current=[]
        for p in data['pairs']:
            f=p['features']
            if not f['valid'] or f['G_names']!=list(G_NAMES) or f['X_names']!=list(X_NAMES) or p['G6']!=f['G'] or p['X2']!=f['X'] or p['Xminus2']!=f['Xminus'] or p['full_support']!=f['full_support']:raise ValueError('fixed feature contract differs')
            # Local candidate/slate/N scale is deliberately NOT feature G0's
            # common support-median scale. Recheck both without replacing either.
            if p['min_offset_norm']!=min(b['normalized_offset'] for b in p['brackets']):raise ValueError('local minimum alias differs')
            for b in p['brackets']:
                distance=float(np.linalg.norm(b['raw_middle_minus_outer_interpolation']))
                if b['local_mean_diagonal']<=0 or not np.isclose(b['normalized_offset'],distance/b['local_mean_diagonal'],atol=1e-12,rtol=1e-12):raise ValueError('local three-box-mean scale differs')
            common_min=min(float(np.linalg.norm(b['offset_xy']))/f['common_scale'] for b in p['brackets'])
            if not np.isclose(p['G6'][0],common_min,atol=1e-12,rtol=1e-12):raise ValueError('G0 common support scale differs')
            if p['active']!=bool(np.any(np.asarray(p['delta_windows']))):raise ValueError('active integer-window support differs')
            current.append(dict(domain=domain,clip=clip,pair_id=p['pair_id'],canonical_ids=p['canonical_ids'],eligible=p['eligible'],active=p['active'],
                full_support=p['full_support'],G6=p['G6'],X2=p['X2'],Xminus2=p['Xminus2'],min_offset_norm=p['min_offset_norm']))
        validate(current);frozen=list(data['slate_ids']);expected=reference_slate(current)
        if frozen!=expected:raise ValueError('common frozen slate differs from registered local-distance ordering')
        byid={r['pair_id']:r for r in current}
        if data['active_slate_ids']!=[k for k in frozen if byid[k]['active']]:raise ValueError('active common slate differs')
        positions={k:i for i,k in enumerate(frozen)}
        if any(p['slate_order']!=positions.get(p['pair_id']) for p in data['pairs']):raise ValueError('slate position alias differs')
        rows.extend(current);slates[domain,clip]=frozen
        support.append(dict(domain=domain,clip=clip,eligible_pairs=sum(r['eligible'] for r in current),active_pairs=sum(r['active'] for r in current),full_support=sum(r['full_support'] for r in current),slate_pairs=len(frozen)))
        del data,current
    validate(rows);return rows,slates,support


def load_source_labels(root,heldout,rows,*,requested_domains,receipt,prereg,candidate_sha):
    allowed=tuple(requested_domains)
    if heldout not in DOMAINS or len(allowed)!=3 or set(allowed)!=set(DOMAINS)-{heldout}:raise ValueError('explicit other-three source domains required')
    expected={row_key(r) for r in rows if r['domain'] in allowed};labels={};proof={}
    for h in allowed:
        data=json.loads(checked(root,receipt,h+'.json').read_text())
        if data['domain']!=h or data['prereg']!=prereg or data['candidate_manifest_sha256']!=candidate_sha or data['source208_labels_sha256']!=SOURCE208_LABELS:raise ValueError('source label scope/lineage differs')
        for r in data['records']:
            if r['domain']!=h:raise ValueError('cross-domain source label')
            key=row_key(r);value=r['class']
            if key in labels or key not in expected or value not in ('S','D','U'):raise ValueError('duplicate/invalid source label')
            if r['loss_eligible']!=(value in ('S','D')) or r['y']!={'S':1,'D':0,'U':None}[value]:raise ValueError('weak pair label interpretation differs')
            labels[key]=value
        proof[h]=dict(file=h+'.json',sha256=receipt['files'][h+'.json'],rows=len(data['records']))
        del data
    if set(labels)!=expected:raise ValueError('source label inventory incomplete')
    return dict(source_domains=list(allowed),labels=labels,receipts=dict(label_manifest_sha256=receipt['manifest_sha256'],source_shards=proof,
        target_access='no heldout shard, no global label deployment, no all-bracket/covisibility diagnostics'))


def field_train(args):
    if args.prereg!='9a90eb75b59279d81f637ea490a0e5c58677549a':raise ValueError('published210 prereg required')
    cr,lr=manifest(args.candidates),manifest(args.labels)
    if (cr['manifest_sha256'],lr['manifest_sha256'])!=(args.candidate_sha,args.label_sha):raise ValueError('GO input seals differ')
    rows,slates,support=load_numeric(args.candidates,cr,args.prereg)
    def provider(h):
        supplied=load_source_labels(args.labels,h,rows,requested_domains=[d for d in DOMAINS if d!=h],receipt=lr,prereg=args.prereg,candidate_sha=cr['manifest_sha256'])
        print('SOURCE210',h,'rows',len(supplied['labels']),flush=True);return supplied
    return train_and_seal(rows,provider,args.out,dict(prereg=args.prereg,candidate_manifest_sha256=cr['manifest_sha256'],
        label_manifest_sha256=lr['manifest_sha256'],input208_manifest_sha256=SOURCE208,source208_labels_sha256=SOURCE208_LABELS,numeric_support=support),slates)


def self_test():
    rng=np.random.default_rng(21001);rows=[];labels={}
    for h in DOMAINS:
        for i in range(10):
            support=i%3!=0;g=[.1+.01*i,.3,np.log1p(3),1.,float(support),.4]
            x=[float(rng.uniform(.1,1)),float(rng.normal())] if support else [0.,0.]
            r=dict(domain=h,clip=h+'clip',pair_id=f'{2*i}|{2*i+1}',eligible=True,active=i!=9,full_support=support,G6=g,X2=x,Xminus2=[x[0],-x[1]],min_offset_norm=.1+.01*i)
            rows.append(r);labels[row_key(r)]='S' if i%2 else 'D'
    source=[r for r in rows if r['domain']!='35'];model=fit(source,{row_key(r):labels[row_key(r)] for r in source},heldout='35')
    assert len(model['training_records'])==30
    for weights in ('weight_G','weight_correction'):
        for y in (0,1):assert np.isclose(sum(r[weights] or 0 for r in model['training_records'] if r['y']==y),.5)
    for r in rows:
        p=scores(model,r)
        if not r['full_support']:assert np.float64(p['G']).tobytes()==np.float64(p['X']).tobytes()==np.float64(p['M']).tobytes()
        else:assert p['M']==p['G']+_head_score(model['correction'],r['Xminus2'])
    targets=[r for r in rows if r['domain']=='35'];out=deploy_clip(targets,model)
    assert out['a']==9 and out['k']==3 and len(out['selected']['ALL'])==9
    assert all(len(out['selected'][h])==3 for h in ('N','G','X','M'))
    assert deploy_clip(list(reversed(targets)),model)==out
    one=deploy_clip(targets[:1],model);assert one['a']==one['k']==1
    zero=deploy_clip([dict(targets[0],active=False)],model);assert zero['a']==zero['k']==0
    # Varying box diagonals make local-mean and common-median scales disagree.
    # A raw offset1 with local3frame diagonal mean(2,2,20)=8 and common2 ->.125/.5.
    # B raw offset.4 with localmean2 and common4 ->.2/.1. N must choose A.
    A=dict(targets[0],G6=[.5,*targets[0]['G6'][1:]],min_offset_norm=1/((2+2+20)/3))
    B=dict(targets[1],G6=[.1,*targets[1]['G6'][1:]],min_offset_norm=.4/2)
    # Keep the synthetic median at least the first feature.
    A['G6'][1]=.6
    reversed_scales=deploy_clip([A,B],model)
    assert reversed_scales['slate'][:2]==[A['pair_id'],B['pair_id']] and reversed_scales['selected']['N']==[A['pair_id']]
    noop=fit(source,{row_key(r):'S' for r in source},heldout='35')
    assert noop['G']['status']==noop['correction']['status']=='zero_missing_class'
    assert scores(noop,targets[1])==dict(G=0.,X=0.,M=0.) and len(deploy_clip(targets,noop)['selected']['X'])==3
    # Base can fit while the supported subset lacks a class: correction stays0.
    support_S={row_key(r):('S' if r['full_support'] else 'D') for r in source}
    partial=fit(source,support_S,heldout='35');assert partial['G']['status']=='fitted' and partial['correction']['status']=='zero_missing_class'
    assert all(scores(partial,r)['G']==scores(partial,r)['X']==scores(partial,r)['M'] for r in rows)
    assert grouped_average_precision(np.array([1]*5+[0]*5),np.zeros(10))['AP']==.5
    assert grouped_average_precision([1,0],[1.,0.])['AP'] is None
    Z=rng.normal(size=(20,2));y=rng.integers(0,2,20);w=np.ones(20)/20;offset=rng.normal(size=20);theta=rng.normal(size=3)
    _,grad=objective(theta,Z,y,w,offset);numeric=[]
    for j in range(3):
        d=np.eye(3)[j]*1e-6;numeric.append((objective(theta+d,Z,y,w,offset)[0]-objective(theta-d,Z,y,w,offset)[0])/2e-6)
    assert np.allclose(grad,numeric,atol=1e-8,rtol=1e-6)
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);labelroot=tmp/'labels';labelroot.mkdir()
        for h in DOMAINS:
            records=[dict(domain=r['domain'],clip=r['clip'],pair_id=r['pair_id'],**{'class':labels[row_key(r)]},y=int(labels[row_key(r)]=='S'),loss_eligible=True) for r in rows if r['domain']==h]
            save(labelroot/(h+'.json'),dict(domain=h,prereg='synthetic',candidate_manifest_sha256='c',source208_labels_sha256=SOURCE208_LABELS,records=records))
        save(labelroot/'manifest.json',{p.name:sha(p) for p in labelroot.iterdir()});(labelroot/'SEALED.sha256').write_text(sha(labelroot/'manifest.json'))
        lr=manifest(labelroot)
        for h in DOMAINS:
            path=labelroot/(h+'.json');raw=path.read_bytes();path.unlink()
            supplied=load_source_labels(labelroot,h,rows,requested_domains=[d for d in DOMAINS if d!=h],receipt=lr,prereg='synthetic',candidate_sha='c')
            assert len(supplied['labels'])==30
            path.write_bytes(raw)
        try:load_source_labels(Path('must-not-open'),'35',rows,requested_domains=['35','W','G'],receipt={},prereg='synthetic',candidate_sha='c')
        except ValueError:pass
        else:raise AssertionError('heldout whitelist accepted')
        def provider(h):return dict(source_domains=[d for d in DOMAINS if d!=h],labels={row_key(r):labels[row_key(r)] for r in rows if r['domain']!=h})
        slates={(h,h+'clip'):reference_slate([r for r in rows if r['domain']==h]) for h in DOMAINS}
        sealed=train_and_seal(rows,provider,tmp/'pred',dict(prereg='synthetic'),slates)
        assert len(load_predictions(sealed['root'])['decisions'])==4
    return dict(status='PASS',seed=21001,tests='independent source weights; frozenG correction; bitwise unsupported bypass; wrongsign sharedscaler; a1 budget; local/common-scale reversed ranking; fallback0 still budget; supported missingclass; grouped tieAP; offsetgradient; heldout shards absent; whitelist before IO; common fourfold seal')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--run-field',action='store_true')
    parser.add_argument('--candidates',type=Path);parser.add_argument('--labels',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--candidate-sha');parser.add_argument('--label-sha');parser.add_argument('--prereg');args=parser.parse_args()
    if args.run_field:
        if any(v is None for v in (args.candidates,args.labels,args.out,args.candidate_sha,args.label_sha)):parser.error('all roots and pinned seals required')
        print(json.dumps(field_train(args),indent=2))
    elif args.self_test:print(json.dumps(self_test(),indent=2))
    else:parser.error('--self-test or --run-field required')
