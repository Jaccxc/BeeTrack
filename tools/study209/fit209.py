"""Source-only common-bee logistic heads and fixed-budget blind selections.

No image/crop/GT reader exists here. A normalized row contains domain,clip,
pair_id='A|B' (integer A<B), geometry4, true/sham valid+feature_vector4.
All conflicting main-valid S/D SOURCE rows train all three models. Only the
deployment slate is disjoint. Ranked logits are scores, not calibrated
probability guarantees. Sham is a frozen-X diagnostic, never a main control.
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
HEADS=('G','F','X')
ARMS=('K','J','U','G','F','X')
FEATURE_NAMES=('mean_iou','max_iou','log1p_shared_frames','representative_area_ratio','flow','cross','board','within')
DIMENSIONS=dict(G=4,F=5,X=8)
RIDGE=.1
SCHEMA='common209-logistic-v1'
CLIPS={('35','BEE24-35-head',3),('33','BEE24-33-head',3),
       ('W','BEE24-10',0),('W','BEE24-15',0),('G','BEE24-24',0),('G','BEE24-28',0)}
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


def pair_ids(row):
    fields=row['pair_id'].split('|')
    if len(fields)!=2:raise ValueError('pair_id requires A|B')
    try:ids=tuple(map(int,fields))
    except ValueError:raise ValueError('integer canonical pair required') from None
    if ids[0]>=ids[1] or row['pair_id']!=f'{ids[0]}|{ids[1]}':raise ValueError('canonical A<B IDs required')
    if 'canonical_ids' in row and tuple(row['canonical_ids'])!=ids:raise ValueError('canonical ID alias differs')
    return ids


def row_key(row):return row['domain']+'/'+row['clip']+'/'+row['pair_id']


def geometric(row):return float(row['geometry4'][0])>=.20


def main_valid(row):return geometric(row) and bool(row['true']['valid'])


def validate(rows):
    keys=set()
    for r in rows:
        if r['domain'] not in DOMAINS or not isinstance(r['clip'],str) or not r['clip']:raise ValueError('registered domain and clip required')
        pair_ids(r);key=row_key(r)
        if key in keys:raise ValueError('duplicate pair row')
        keys.add(key)
        g=np.asarray(r['geometry4'],float)
        if g.shape!=(4,) or not np.isfinite(g).all() or not 0<=g[0]<=1 or not 0<=g[1]<=1 or g[0]>g[1]+1e-12 or g[2]<0 or not 0<=g[3]<=1:raise ValueError('invalid geometry4')
        if 'geometric_eligible' in r and bool(r['geometric_eligible'])!=geometric(r):raise ValueError('geometric eligibility alias differs')
        for kind in ('true','sham'):
            if not isinstance(r[kind]['valid'],(bool,np.bool_)):raise ValueError('pixel validity must be bool')
            if r[kind]['valid']:
                f=np.asarray(r[kind]['feature_vector'],float)
                if f.shape!=(4,) or not np.isfinite(f).all():raise ValueError('valid pixel option needs finite four features')
    return rows


def source_weights(rows,y):
    """Class .5 -> nonempty domain equal -> all that-domain class rows equal."""
    y=np.asarray(y)
    if y.shape!=(len(rows),) or not np.isin(y,[0,1]).all():raise ValueError('binary source targets required')
    counts=Counter((int(c),r['domain']) for r,c in zip(rows,y))
    domains={c:{h for (k,h),n in counts.items() if k==c and n} for c in (0,1)}
    if not all(domains.values()):return None
    w=np.asarray([.5/(len(domains[int(c)])*counts[int(c),r['domain']]) for r,c in zip(rows,y)],float)
    if not np.isclose(w.sum(),1.,atol=1e-14):raise AssertionError('weight total differs')
    return w


def logistic_objective(theta,Z,y,w):
    beta,bias=theta[:-1],theta[-1];z=Z@beta+bias
    value=float(w@(np.logaddexp(0.,z)-y*z)+RIDGE/2*(beta@beta))
    residual=w*(expit(z)-y)
    gradient=np.r_[Z.T@residual+RIDGE*beta,residual.sum()]
    return value,gradient


def fit(rows,labels,*,heldout=None):
    validate(rows)
    if heldout is not None and any(r['domain']==heldout for r in rows):raise ValueError('heldout row supplied to fitting')
    if set(labels)!={row_key(r) for r in rows} or any(v not in ('S','D','U') for v in labels.values()):raise ValueError('complete source S/D/U labels required')
    cohort=sorted([r for r in rows if main_valid(r) and labels[row_key(r)]!='U'],key=lambda r:row_key(r))
    y=np.asarray([labels[row_key(r)]=='S' for r in cohort],float)
    w=source_weights(cohort,y)
    counts={h:{c:sum(r['domain']==h and labels[row_key(r)]==c for r in cohort) for c in ('S','D')} for h in DOMAINS if h!=heldout}
    result=dict(schema=SCHEMA,heldout=heldout,source_domains=[h for h in DOMAINS if h!=heldout],
        ridge=RIDGE,status='noop_missing_class' if w is None else 'fitted',source_counts=counts,
        source_rows=len(rows),main_valid_source_rows=sum(main_valid(r) for r in rows),
        weight_rule='class .5; class-containing source domains equal; within domain-class all rows equal, no disjoint training filter',
        training_order='lexicographic domain/clip/pair_id',
        training_records=[dict(key=row_key(r),domain=r['domain'],clip=r['clip'],pair_id=r['pair_id'],geometry4=r['geometry4'],
            pixels4=r['true']['feature_vector'],y=int(t),weight=None if w is None else float(w[i])) for i,(r,t) in enumerate(zip(cohort,y))],heads={})
    if w is None:
        result['heads']={head:dict(status='noop_missing_class',mean=None,scale=None,beta=None,intercept=None) for head in HEADS}
        return result
    # Shared full-vector moments make ablation-prefix normalization identical.
    X=np.asarray([list(r['geometry4'])+list(r['true']['feature_vector']) for r in cohort],float)
    mean=w@X;scale=np.sqrt(w@((X-mean)**2));scale[scale<1e-6]=1.
    for head in HEADS:
        d=DIMENSIONS[head];Z=(X[:,:d]-mean[:d])/scale[:d]
        opt=minimize(lambda t:logistic_objective(t,Z,y,w),np.zeros(d+1),jac=True,method='L-BFGS-B',
            options=dict(maxiter=500,gtol=1e-8,ftol=1e-12))
        if not opt.success or not np.isfinite(opt.x).all():raise RuntimeError(head+' source logistic failed: '+str(opt.message))
        result['heads'][head]=dict(status='fitted',feature_names=list(FEATURE_NAMES[:d]),mean=mean[:d].tolist(),scale=scale[:d].tolist(),
            beta=opt.x[:d].tolist(),intercept=float(opt.x[-1]),optimization=dict(success=True,message=str(opt.message),iterations=int(opt.nit),
                objective=float(opt.fun),gradient=logistic_objective(opt.x,Z,y,w)[1].tolist(),initialization=[0.]*(d+1),solver='scipy L-BFGS-B',
                options=dict(maxiter=500,gtol=1e-8,ftol=1e-12),intercept_penalized=False))
    return result


def score(model,row,head,*,sham=False):
    if head not in HEADS:raise ValueError('unknown head')
    h=model['heads'][head]
    if h['status']!='fitted' or not main_valid(row):return None
    if sham and (head!='X' or not row['sham']['valid']):return None
    pixel=row['sham' if sham else 'true']['feature_vector'];x=np.asarray(list(row['geometry4'])+list(pixel),float)[:DIMENSIONS[head]]
    return float(((x-h['mean'])/h['scale'])@h['beta']+h['intercept'])


def slate(rows,*,require_pixels):
    """One pair consumes its two canonical tracks; no chaining or re-pairing."""
    validate(rows)
    eligible=[r for r in rows if main_valid(r)] if require_pixels else [r for r in rows if geometric(r)]
    eligible.sort(key=lambda r:(-float(r['geometry4'][0]),pair_ids(r)))
    used=set();selected=[]
    for r in eligible:
        ids=pair_ids(r)
        if not used.intersection(ids):selected.append(r);used.update(ids)
    return selected


def deploy_clip(rows,model):
    validate(rows)
    if len({(r['domain'],r['clip']) for r in rows})>1:raise ValueError('one clip per slate required')
    common=slate(rows,require_pixels=True);all_geometry=slate(rows,require_pixels=False)
    n=len(common);k=(n+3)//4 if n>=4 else 0
    logits={head:{r['pair_id']:score(model,r,head) for r in rows} for head in HEADS}
    selected=dict(K=[],J=[r['pair_id'] for r in all_geometry],U=[r['pair_id'] for r in common[:k]])
    for head in HEADS:
        active=model['heads'][head]['status']=='fitted'
        ranking=sorted(common,key=lambda r:(-logits[head][r['pair_id']],pair_ids(r))) if active else []
        selected[head]=[r['pair_id'] for r in ranking[:k]]
    orders={r['pair_id']:i for i,r in enumerate(common)}
    records=[]
    for r in sorted(rows,key=pair_ids):
        z={head:logits[head][r['pair_id']] for head in HEADS};sham=score(model,r,'X',sham=True)
        records.append(dict(key=row_key(r),pair_id=r['pair_id'],canonical_ids=list(pair_ids(r)),geometry4=r['geometry4'],
            true_valid=bool(r['true']['valid']),sham_valid=bool(r['sham']['valid']),main_valid=main_valid(r),
            paired_valid=main_valid(r) and bool(r['sham']['valid']),slate_order=orders.get(r['pair_id']),logits=z,
            sigmoid={h:None if v is None else float(expit(v)) for h,v in z.items()},X_sham_logit=sham,
            X_sham_sigmoid=None if sham is None else float(expit(sham))))
    return dict(n_slate=n,k=k,slate=[r['pair_id'] for r in common],selected=selected,records=records,
        status=model['status'],ranking='raw double logit descending; exact ties numeric canonical(A,B) ascending',
        budget='ceil(n_slate/4) if n_slate>=4 else0; fixed intervention budget, not probability assurance')


def train_and_seal(rows,label_provider,outdir,input_receipt,clips=None):
    if sha(__file__)!=CODE_SHA:raise ValueError('loaded source changed before training')
    validate(rows)
    if any(any(k in r for k in ('label','labels','y')) for r in rows):raise ValueError('deployment rows must be unlabelled')
    clips=clips if clips is not None else sorted({(r['domain'],r['clip']) for r in rows})
    dest=Path(outdir)
    if dest.exists():raise FileExistsError('immutable output root exists')
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        predictions={};folds={}
        for target in DOMAINS:
            supplied=label_provider(target)
            if set(supplied['source_domains'])!=set(DOMAINS)-{target}:raise ValueError('source-only label whitelist differs')
            source=[r for r in rows if r['domain']!=target];model=fit(source,supplied['labels'],heldout=target)
            model['source_receipts']=supplied.get('receipts',{})
            name='models/'+target+'.json';save(tmp/name,model)
            folds[target]=dict(heldout=target,source_domains=[h for h in DOMAINS if h!=target],model_file=name,status=model['status'],source_counts=model['source_counts'])
            for domain,clip in clips:
                if domain==target:predictions[domain+'/'+clip]=dict(domain=domain,clip=clip,**deploy_clip([r for r in rows if (r['domain'],r['clip'])==(domain,clip)],model))
        save(tmp/'predictions.json',dict(schema=SCHEMA,arms=list(ARMS),folds=folds,clips=predictions))
        if sha(__file__)!=CODE_SHA:raise ValueError('fit source changed during training')
        save(tmp/'receipt.json',dict(input_receipt,schema=SCHEMA,code_sha256=CODE_SHA,numpy_version=np.__version__,scipy_version=scipy.__version__,
            raw_logit_ranking=True,sham_controls_main_model=False,all_target_labels_prohibited_before_common_seal=True))
        save(tmp/'manifest.json',{p.relative_to(tmp).as_posix():sha(p) for p in tmp.rglob('*') if p.is_file()})
        digest=sha(tmp/'manifest.json');(tmp/'SEALED.sha256').write_text(digest+'\n');tmp.rename(dest)
        return dict(root=str(dest),manifest_sha256=digest)
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def verify_seal(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('common209 seal differs')
    files=json.loads((root/'manifest.json').read_text())
    if set(files)!={'predictions.json','receipt.json'}|{'models/'+h+'.json' for h in DOMAINS}:raise ValueError('four source models/common predictions required')
    for name,digest in files.items():
        if sha(root/name)!=digest:raise ValueError('sealed artifact changed')
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt['code_sha256']!=sha(__file__):raise ValueError('fit source changed after seal')
    predictions=json.loads((root/'predictions.json').read_text())
    if set(predictions['folds'])!=set(DOMAINS):raise ValueError('four heldout folds required')
    for h,f in predictions['folds'].items():
        if f['heldout']!=h or set(f['source_domains'])!=set(DOMAINS)-{h}:raise ValueError('source boundary differs')
    return receipt


def load_predictions(root):
    receipt=verify_seal(root);predictions=json.loads((Path(root)/'predictions.json').read_text());decisions={}
    for p in predictions['clips'].values():decisions.setdefault(p['domain'],{})[p['clip']]=p['selected']
    return dict(receipt=receipt,predictions=predictions,decisions=decisions)


def manifest(root):
    root=Path(root);digest=sha(root/'manifest.json')
    if (root/'SEALED.sha256').read_text().strip()!=digest:raise ValueError('input top seal differs')
    files=json.loads((root/'manifest.json').read_text())
    for name in files:
        if not (root/name).resolve().is_relative_to(root.resolve()):raise ValueError('input manifest path escapes root')
    return dict(manifest_sha256=digest,files=files)


def checked(root,receipt,name):
    if name not in receipt['files'] or sha(Path(root)/name)!=receipt['files'][name]:raise ValueError('sealed input changed: '+name)
    return Path(root)/name


def load_numeric(candidates,features,cr,fr,prereg):
    """Only frozen candidate/pixel numbers. No image, GT, or label access."""
    ct=json.loads(checked(candidates,cr,'deployment.json').read_text())
    ft=json.loads(checked(features,fr,'deployment.json').read_text())
    source208='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94'
    if ct['schema']!='common209-freeze-v1' or ft['schema']!='common209-features-v1' or ct['prereg']!=prereg or ft['prereg']!=prereg:raise ValueError('input schema/registration differs')
    if ct['source208']!=source208 or ft['source208']!=source208 or ft['candidate_catalog_manifest_sha256']!=cr['manifest_sha256']:raise ValueError('input numerical lineage differs')
    for top in (ct,ft):
        if len(top['clips'])!=6 or {(r['domain'],r['clip'],r['seed']) for r in top['clips']}!=CLIPS:raise ValueError('fixed six old clips/seeds required')
    frefs={(r['domain'],r['clip']):r for r in ft['clips']};rows=[];clips=[];counts=[]
    for ref in ct['clips']:
        domain,clip=ref['domain'],ref['clip'];path=domain+'/'+clip;feature_ref=frefs[domain,clip]
        if ref['path']!=path or feature_ref['path']!=path or feature_ref['features_file']!=path+'/features.json':raise ValueError('clip path/domain differs')
        if sha(checked(candidates,cr,path+'/manifest.json'))!=ref['seal'] or sha(checked(features,fr,path+'/manifest.json'))!=feature_ref['seal']:raise ValueError('child seal differs')
        catalog_path=checked(candidates,cr,path+'/catalog.json.gz')
        with gzip.open(catalog_path,'rt',encoding='utf8') as stream:catalog=json.load(stream)
        pixels=json.loads(checked(features,fr,path+'/features.json').read_text())
        if pixels['schema']!='common209-features-clip-v1' or pixels['prereg']!=prereg or pixels['candidate_catalog_manifest_sha256']!=cr['manifest_sha256'] or pixels['domain']!=domain or pixels['clip']!=clip or pixels['feature_names']!=['flow','cross','board','within']:raise ValueError('pixel feature contract differs')
        catalog_pairs={p['pair_id']:p for p in catalog['pairs']}
        if len(catalog_pairs)!=len(catalog['pairs']) or {p['pair_id'] for p in pixels['pairs']}!=set(catalog_pairs) or len(pixels['pairs'])!=len(catalog_pairs):raise ValueError('feature candidate inventory differs')
        for p in pixels['pairs']:
            original=catalog_pairs[p['pair_id']];rep=original['representative'];frame=None if rep is None else rep['frame']
            if p['canonical_ids']!=original['canonical_ids'] or p['geometry4']!=original['geometry4'] or p['representative_frame']!=frame:raise ValueError('fixed representative/geometry changed')
            if p['main_valid']!=p['true']['valid'] or p['paired_valid']!=(p['true']['valid'] and p['sham']['valid']):raise ValueError('true/sham main support differs')
            if p['true']['valid'] and frame is None:raise ValueError('NO_DATE cannot have valid true pixels')
            rows.append(dict(domain=domain,clip=clip,pair_id=p['pair_id'],canonical_ids=p['canonical_ids'],geometry4=p['geometry4'],
                geometric_eligible=original['geometric_eligible'],representative_frame=frame,
                true={k:p['true'][k] for k in ('valid','feature_vector')},sham={k:p['sham'][k] for k in ('valid','feature_vector')}))
        clips.append((domain,clip));counts.append(dict(domain=domain,clip=clip,catalog_pairs=len(catalog_pairs),true_valid=sum(p['true']['valid'] for p in pixels['pairs']),paired_valid=sum(p['true']['valid'] and p['sham']['valid'] for p in pixels['pairs'])))
        del catalog,pixels,catalog_pairs
    validate(rows);return rows,clips,counts


def load_source_labels(root,heldout,rows,*,requested_domains,receipt,prereg,candidate_sha,feature_sha):
    allowed=tuple(requested_domains)
    if heldout not in DOMAINS or len(allowed)!=3 or set(allowed)!=set(DOMAINS)-{heldout}:raise ValueError('explicit other-three source domains required')
    # Do not parse global deployment or per-clip full-date/evidence diagnostics.
    expected={row_key(r):r for r in rows if r['domain'] in allowed};labels={};proof={}
    for domain in allowed:
        data=json.loads(checked(root,receipt,domain+'.json').read_text())
        if data['domain']!=domain or data['prereg']!=prereg or data['candidate_manifest_sha256']!=candidate_sha or data['feature_manifest_sha256']!=feature_sha:raise ValueError('source label scope/lineage differs')
        count=0
        for r in data['records']:
            if r['domain']!=domain:raise ValueError('cross-domain source label')
            key=row_key(r);value=r['representative_class']
            if key in labels or key not in expected or value not in ('S','D','U'):raise ValueError('duplicate/invalid source label')
            if r['representative_frame']!=expected[key].get('representative_frame'):raise ValueError('source label representative date differs')
            if r['loss_eligible']!=(value in ('S','D')) or r['y']!=({'S':1,'D':0,'U':None}[value]):raise ValueError('representative supervision interpretation differs')
            labels[key]=value;count+=1
        proof[domain]=dict(file=domain+'.json',sha256=receipt['files'][domain+'.json'],source_rows=count)
        del data
    if set(labels)!=set(expected):raise ValueError('complete source label inventory required')
    return dict(source_domains=list(allowed),labels=labels,receipts=dict(label_manifest_sha256=receipt['manifest_sha256'],source_shards=proof,
        target_access='no heldout shard, no global label deployment, no full-date GT diagnostics'))


def field_train(args):
    if args.prereg!='9efbea732621dec804aea9dcc9df87fc1ca6293e':raise ValueError('published209 registration required')
    cr,fr,lr=manifest(args.candidates),manifest(args.features),manifest(args.labels)
    if (cr['manifest_sha256'],fr['manifest_sha256'],lr['manifest_sha256'])!=(args.candidate_sha,args.feature_sha,args.label_sha):raise ValueError('GO input seals differ')
    rows,clips,counts=load_numeric(args.candidates,args.features,cr,fr,args.prereg)
    def provider(target):
        result=load_source_labels(args.labels,target,rows,requested_domains=[h for h in DOMAINS if h!=target],receipt=lr,prereg=args.prereg,candidate_sha=cr['manifest_sha256'],feature_sha=fr['manifest_sha256'])
        print('SOURCE209',target,'rows',len(result['labels']),flush=True);return result
    return train_and_seal(rows,provider,args.out,dict(prereg=args.prereg,candidate_manifest_sha256=cr['manifest_sha256'],
        feature_manifest_sha256=fr['manifest_sha256'],label_manifest_sha256=lr['manifest_sha256'],
        input208_manifest_sha256='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94',numeric_support=counts),clips)


def self_test():
    rng=np.random.default_rng(20901);rows=[];labels={}
    for h in DOMAINS:
        for i in range(12):
            A,B=2*i,2*i+1;g=[.25+.01*(11-i),.8,np.log1p(5+i),.5]
            r=dict(domain=h,clip=h+'clip',pair_id=f'{A}|{B}',geometry4=g,representative_frame=10,true=dict(valid=True,feature_vector=rng.normal(size=4).tolist()),sham=dict(valid=i%3!=0,feature_vector=rng.normal(size=4).tolist()))
            rows.append(r);labels[row_key(r)]='S' if i%2 else 'D'
        # Conflicting pairs train too, but cannot enter the fixed disjoint slate.
        r=dict(rows[-1],pair_id='0|3',geometry4=[.21,.4,np.log1p(2),.5]);rows.append(r);labels[row_key(r)]='S'
    source=[r for r in rows if r['domain']!='35'];model=fit(source,{row_key(r):labels[row_key(r)] for r in source},heldout='35')
    assert len(model['training_records'])==39
    for y in (0,1):
        rs=[r for r in model['training_records'] if r['y']==y]
        assert np.isclose(sum(r['weight'] for r in rs),.5)
        assert all(np.isclose(sum(r['weight'] for r in rs if r['domain']==h),1/6) for h in ('33','W','G'))
    assert all(max(abs(g) for g in x['optimization']['gradient'])<1e-6 for x in model['heads'].values())
    assert model['heads']['G']['mean']==model['heads']['X']['mean'][:4]
    assert model['heads']['X']['scale'][3]==1.
    target=[r for r in rows if r['domain']=='35'];out=deploy_clip(target,model)
    assert out['n_slate']==12 and out['k']==3 and all(len(out['selected'][h])==3 for h in HEADS)
    assert len(out['selected']['J'])==12 and out['selected']['U']==out['slate'][:3]
    assert deploy_clip(list(reversed(target)),model)==out
    other=[dict(r,sham=dict(valid=True,feature_vector=[1000.]*4)) for r in target]
    assert deploy_clip(other,model)['selected']==out['selected']
    for r in target:
        if r['sham']['valid']:
            copy=dict(r,true=r['sham']);assert score(model,r,'X',sham=True)==score(model,copy,'X')
    noop=fit(source,{row_key(r):'S' for r in source},heldout='35');no=deploy_clip(target,noop)
    assert all(no['selected'][h]==[] for h in HEADS) and len(no['selected']['U'])==3
    # J includes true-invalid geometric options, U/model slate excludes them.
    invalid=[dict(r,true=dict(valid=False,feature_vector=None)) for r in target]
    assert len(deploy_clip(invalid,model)['selected']['J'])==12 and deploy_clip(invalid,model)['k']==0
    Z=rng.normal(size=(30,4));y=rng.integers(0,2,30);w=np.ones(30)/30;t=rng.normal(size=5)
    value,gradient=logistic_objective(t,Z,y,w);numeric=[]
    for j in range(5):
        d=np.eye(5)[j]*1e-6;numeric.append((logistic_objective(t+d,Z,y,w)[0]-logistic_objective(t-d,Z,y,w)[0])/2e-6)
    assert np.allclose(gradient,numeric,atol=1e-8,rtol=1e-6)
    assert np.isfinite(logistic_objective(np.array([1e4]*5),Z,y,w)[0])
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);labelroot=tmp/'labels';labelroot.mkdir()
        for h in DOMAINS:
            records=[dict(domain=r['domain'],clip=r['clip'],pair_id=r['pair_id'],representative_frame=10,
                          representative_class=labels[row_key(r)],loss_eligible=True,y=int(labels[row_key(r)]=='S')) for r in rows if r['domain']==h]
            save(labelroot/(h+'.json'),dict(domain=h,prereg='synthetic',candidate_manifest_sha256='c',feature_manifest_sha256='f',records=records))
        save(labelroot/'manifest.json',{p.name:sha(p) for p in labelroot.iterdir()});(labelroot/'SEALED.sha256').write_text(sha(labelroot/'manifest.json'))
        lr=manifest(labelroot)
        for h in DOMAINS:
            path=labelroot/(h+'.json');data=path.read_bytes();path.unlink()
            supplied=load_source_labels(labelroot,h,rows,requested_domains=[d for d in DOMAINS if d!=h],receipt=lr,prereg='synthetic',candidate_sha='c',feature_sha='f')
            assert len(supplied['labels'])==39
            path.write_bytes(data)
        try:load_source_labels(Path('must-not-open'),'35',rows,requested_domains=['35','W','G'],receipt={},prereg='synthetic',candidate_sha='c',feature_sha='f')
        except ValueError:pass
        else:raise AssertionError('heldout label whitelist accepted')
        def provider(h):return dict(source_domains=[d for d in DOMAINS if d!=h],labels={row_key(r):labels[row_key(r)] for r in rows if r['domain']!=h})
        sealed=train_and_seal(rows,provider,tmp/'pred',dict(prereg='synthetic'))
        assert len(load_predictions(sealed['root'])['decisions'])==4
    return dict(status='PASS',seed=20901,tests='class/domain weights; shared moments; conflicts trained; fixed disjoint slate/budget; order invariance; sham no main effect; missingclass noop; J invalid pixels; finite difference; extreme logits; all4 heldout shard absent; whitelist before IO; fourfold common seal')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--run-field',action='store_true')
    parser.add_argument('--candidates',type=Path);parser.add_argument('--features',type=Path);parser.add_argument('--labels',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--candidate-sha');parser.add_argument('--feature-sha');parser.add_argument('--label-sha');parser.add_argument('--prereg')
    args=parser.parse_args()
    if args.run_field:
        if any(v is None for v in (args.candidates,args.features,args.labels,args.out,args.candidate_sha,args.feature_sha,args.label_sha)):parser.error('all roots and pinned seals required')
        print(json.dumps(field_train(args),indent=2))
    elif args.self_test:print(json.dumps(self_test(),indent=2))
    else:parser.error('--self-test or --run-field required')
