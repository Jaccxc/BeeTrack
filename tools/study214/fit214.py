"""Source-only endpoint centering, numeric features only; no image/GT reader.

G uses six frozen geometric features. I adds raw odd chi24. Z uses I's SAME
parameters and source scaler with negative RAW chi, not negated z-scores.
Only eligible, officially matched source receiver-K endpoints fit. All eligible
endpoints deploy, including unmatched ones and both unseen peer members.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

DOMAINS=('35','33','W','G')
HEADS=('G','I')
ARMS=('K','G','I','Z')
G_NAMES=('log_width_over_height','native_score','time_role','vx','vy','log1p_nobs')
CHI_NAMES=tuple(f'odd_cell_y{y}_x{x}_{c}' for y in range(2) for x in range(4) for c in ('R','G','B'))
MAIN_PREREG='79dbed053bcddd4290300494ceff29c8130a5807'
SUPPLEMENT='21c6c803290fad2816ca472ec673b9f2eda76759'
FEATURE_SHA='63ff00043717c889d6ff674cf44851d72ca44f7b1ee6aa144b6aa6a8609d2165'
CLIPS={('35','BEE24-35-head'),('33','BEE24-33-head'),('W','BEE24-10'),('W','BEE24-15'),('G','BEE24-24'),('G','BEE24-28')}
RIDGE=.1
LIMIT=.25
SCHEMA='center214-source-ridge-v1'
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


def member_key(r):return r['domain']+'/'+r['clip']+'/'+str(r['member_index'])


def run_key(r):return member_key(r)+'/'+str(r['run_id'])


def validate(rows):
    seen=set();raw=defaultdict(set)
    for r in rows:
        if r['domain'] not in DOMAINS or not isinstance(r['clip'],str) or r['member_index'] not in (0,1,2):raise ValueError('registered endpoint domain/clip/member slot required')
        if r['role']!=('receiverK' if r['member_index']==0 else 'peerK'):raise ValueError('receiver/peer role mismatch')
        if not isinstance(r['endpoint_id'],str) or r['endpoint_id'] in seen:raise ValueError('unique endpoint ID required')
        seen.add(r['endpoint_id'])
        if int(r['raw_index'])!=r['raw_index'] or r['raw_index']<0 or r['raw_index'] in raw[member_key(r)]:raise ValueError('unique nonnegative native raw index required per member')
        raw[member_key(r)].add(r['raw_index'])
        if not isinstance(r['eligible'],(bool,np.bool_)):raise ValueError('frozen eligibility bool required')
        box=np.asarray(r['box_xywh'],float)
        if box.shape!=(4,) or not np.isfinite(box).all():raise ValueError('finite original native box required')
        if r['eligible']:
            g,c=np.asarray(r['G6'],float),np.asarray(r['chi24'],float)
            if (box[2:]<=0).any() or g.shape!=(6,) or c.shape!=(24,) or not np.isfinite(g).all() or not np.isfinite(c).all():raise ValueError('eligible positive box and finite G6/chi24 required')
    return rows


def source_weights(rows):
    if not rows:return np.empty(0,float)
    hs={r['domain'] for r in rows};clips=defaultdict(set);runs=defaultdict(set);counts=Counter()
    for r in rows:
        h,c=r['domain'],r['clip'];key=run_key(r)
        clips[h].add(c);runs[h,c].add(key);counts[key]+=1
    w=np.asarray([1/(len(hs)*len(clips[r['domain']])*len(runs[r['domain'],r['clip']])*counts[run_key(r)]) for r in rows],float)
    return w/w.sum()


def _label(row,label):
    if not isinstance(label['has_owner'],(bool,np.bool_)) or label.get('target_known',label['has_owner'])!=label['has_owner']:raise ValueError('official matched endpoint must have an exact assigned GT row')
    if label.get('raw_index',row['raw_index'])!=row['raw_index']:raise ValueError('label native row alias differs')
    w,h=row['box_xywh'][2:]
    if label['pred_width']!=w or label['pred_height']!=h:raise ValueError('raw original size/normalizer differs')
    if not label['has_owner']:return None
    raw=np.asarray([label['dx_raw'],label['dy_raw']],float)
    if not np.isfinite(raw).all() or w<=0 or h<=0:raise ValueError('matched finite GTcenter-minus-predcenter and positive original size required')
    return np.clip(raw/np.asarray([w,h]),-LIMIT,LIMIT)


def ridge_objective(beta,intercept,Z,Y,w):
    residual=Z@beta+intercept-Y
    objectives=np.sum(w[:,None]*residual**2,axis=0)+RIDGE*np.sum(beta**2,axis=0)
    gradient_beta=2*Z.T@(w[:,None]*residual)+2*RIDGE*beta
    gradient_intercept=2*np.sum(w[:,None]*residual,axis=0)
    return objectives,gradient_beta,gradient_intercept


def fit(rows,labels,*,heldout=None):
    validate(rows)
    if any(r['role']!='receiverK' or (heldout is not None and r['domain']==heldout) for r in rows):raise ValueError('only other-domain receiver-K source endpoints may fit')
    if set(labels)!={r['endpoint_id'] for r in rows}:raise ValueError('complete source receiver endpoint label inventory required')
    known=[];targets=[];unmatched=0
    for r in sorted(rows,key=lambda r:(r['domain'],r['clip'],str(r['run_id']),r['raw_index'],r['endpoint_id'])):
        if not r['eligible']:continue
        y=_label(r,labels[r['endpoint_id']])
        if y is None:unmatched+=1;continue
        known.append(r);targets.append(y)
    w=source_weights(known);Y=np.asarray(targets,float).reshape(-1,2)
    result=dict(schema=SCHEMA,heldout=heldout,source_domains=[h for h in DOMAINS if h!=heldout],ridge=RIDGE,limit=LIMIT,n=len(known),
        source_receiver_endpoints=len(rows),source_eligible_endpoints=sum(r['eligible'] for r in rows),source_unmatched_eligible=unmatched,
        weight_rule='nonempty source hive equal; matched-endpoint clips equal; labelled runs equal; their labelled endpoints equal; no class/side balancing',
        target_rule='clip((assigned raw GTcenter - original Kcenter)/(original width,height),-.25,.25); no GT-based eligibility or outlier removal',
        heads={},training_records=[dict(r,target=Y[i].tolist(),weight=float(w[i]),features30=list(r['G6'])+list(r['chi24']),
                                        dx_raw=labels[r['endpoint_id']]['dx_raw'],dy_raw=labels[r['endpoint_id']]['dy_raw']) for i,r in enumerate(known)])
    groups={key:i for i,key in enumerate(sorted({(r['domain'],r['clip']) for r in known}))}
    runs={key:i for i,key in enumerate(sorted({run_key(r) for r in known}))}
    result['anonymous_training_rows']=[dict(row_index=i,domain_index=DOMAINS.index(r['domain']),clip_index=groups[r['domain'],r['clip']],
        run_index=runs[run_key(r)],X30=list(r['G6'])+list(r['chi24']),Y2=Y[i].tolist(),weight=float(w[i])) for i,r in enumerate(known)]
    if not known:
        result['heads']={h:dict(status='empty_source',feature_names=list((*G_NAMES,*CHI_NAMES)[:d]),mean=None,scale=None,beta=np.zeros((d,2)).tolist(),intercept=[0.,0.]) for h,d in (('G',6),('I',30))}
        return result
    X=np.asarray([list(r['G6'])+list(r['chi24']) for r in known],float)
    mean=w@X;constant=np.all(X==X[0],axis=0);mean[constant]=X[0,constant]
    scale=np.sqrt(w@((X-mean)**2));scale[constant]=0.;zero=scale==0;scale[zero]=1.
    target_mean=w@Y;constant_y=np.all(Y==Y[0],axis=0);target_mean[constant_y]=Y[0,constant_y]
    for head,d in (('G',6),('I',30)):
        Z=(X[:,:d]-mean[:d])/scale[:d]
        beta=np.linalg.solve(Z.T@(w[:,None]*Z)+RIDGE*np.eye(d),Z.T@(w[:,None]*(Y-target_mean)))
        beta[:,constant_y]=0.
        objective,gradient,bias_gradient=ridge_objective(beta,target_mean,Z,Y,w)
        result['heads'][head]=dict(status='fitted',feature_names=list((*G_NAMES,*CHI_NAMES)[:d]),mean=mean[:d].tolist(),scale=scale[:d].tolist(),zero_scale_columns=np.flatnonzero(zero[:d]).tolist(),
            beta=beta.tolist(),intercept=target_mean.tolist(),source_target_mean=target_mean.tolist(),
            optimization=dict(solver='centered weighted ridge linear solve independently per coordinate',intercept_penalized=False,
                objective_per_axis=objective.tolist(),gradient_beta=gradient.tolist(),gradient_intercept=bias_gradient.tolist(),
                penalty='.1*sum(beta^2), no1/2, separately x/y; intercept=weighted clipped target mean'))
    return result


def predict(model,rows,arm):
    validate(rows)
    if arm not in ARMS:raise ValueError('K/G/I/Z only; O is posthoc truth-owned')
    shifts=np.zeros((len(rows),2),np.float64)
    if arm=='K':return shifts
    head=model['heads']['I' if arm=='Z' else arm]
    if head['status']=='empty_source':return shifts
    for i,r in enumerate(rows):
        if not r['eligible']:continue
        raw=np.asarray(r['G6'] if arm=='G' else list(r['G6'])+((-np.asarray(r['chi24'])) if arm=='Z' else np.asarray(r['chi24'])).tolist(),np.float64)
        value=((raw-head['mean'])/head['scale'])@np.asarray(head['beta'])+head['intercept']
        shifts[i]=np.clip(value,-LIMIT,LIMIT)
    return shifts


def train_and_seal(rows,members,label_provider,outdir,input_receipt):
    validate(rows)
    if sha(__file__)!=CODE_SHA:raise ValueError('loaded source changed')
    if any('has_owner' in r or 'labels' in r or 'target' in r for r in rows):raise ValueError('deployment features must be unlabelled')
    dest=Path(outdir)
    if dest.exists():raise FileExistsError('immutable output root exists')
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        predictions=[];folds={}
        for h in DOMAINS:
            source=[r for r in rows if r['domain']!=h and r['role']=='receiverK' and r['eligible']];supplied=label_provider(h,source)
            if set(supplied['source_domains'])!=set(DOMAINS)-{h}:raise ValueError('explicit other-three source whitelist required')
            model=fit(source,supplied['labels'],heldout=h);model['source_receipts']=supplied.get('receipts',{})
            name='models/'+h+'.json';save(tmp/name,model);folds[h]=dict(heldout=h,source_domains=[d for d in DOMAINS if d!=h],model_file=name,n=model['n'])
            for m in members:
                if m['domain']!=h:continue
                selected=[r for r in rows if member_key(r)==member_key(m)]
                # Preserve the original frozen endpoint order, not model/cohort order.
                ids=[r['endpoint_id'] for r in selected]
                files={}
                for arm in ARMS:
                    file=f'shifts/{h}/{m["clip"]}/m{m["member_index"]}/{arm}.npy';(tmp/file).parent.mkdir(parents=True,exist_ok=True)
                    np.save(tmp/file,predict(model,selected,arm),allow_pickle=False);files[arm]=file
                predictions.append(dict(m,endpoint_ids=ids,raw_indices=[r['raw_index'] for r in selected],eligible=[r['eligible'] for r in selected],
                    original_boxes=[r['box_xywh'] for r in selected],shifts=files))
        if sha(__file__)!=CODE_SHA:raise ValueError('fit source changed during fitting')
        save(tmp/'predictions.json',dict(schema=SCHEMA,arms=list(ARMS),posthoc_only_arm='O',folds=folds,members=predictions,
            application='x+=shift_x*ORIGINAL_width,y+=shift_y*ORIGINAL_height simultaneously; widths/heights/scores/integers unchanged'))
        save(tmp/'receipt.json',dict(input_receipt,schema=SCHEMA,code_sha256=CODE_SHA,numpy_version=np.__version__,
            target_owner_and_peer_labels_prohibited_before_common_seal=True))
        save(tmp/'manifest.json',{p.relative_to(tmp).as_posix():sha(p) for p in tmp.rglob('*') if p.is_file()});digest=sha(tmp/'manifest.json')
        (tmp/'SEALED.sha256').write_text(digest+'\n');tmp.rename(dest)
        return dict(root=str(dest),manifest_sha256=digest)
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def verify_seal(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('prediction seal differs')
    files=json.loads((root/'manifest.json').read_text())
    for name,digest in files.items():
        if not (root/name).resolve().is_relative_to(root.resolve()) or sha(root/name)!=digest:raise ValueError('sealed output changed')
    receipt=json.loads((root/'receipt.json').read_text());predictions=json.loads((root/'predictions.json').read_text())
    if receipt['code_sha256']!=sha(__file__) or set(predictions['folds'])!=set(DOMAINS):raise ValueError('four frozen source folds required')
    for h,f in predictions['folds'].items():
        if f['heldout']!=h or set(f['source_domains'])!=set(DOMAINS)-{h} or f['model_file'] not in files:raise ValueError('source fold boundary differs')
    for m in predictions['members']:
        if len(set(m['raw_indices']))!=len(m['raw_indices']):raise ValueError('endpoint row reused')
        for arm,file in m['shifts'].items():
            values=np.load(root/file,allow_pickle=False)
            if arm not in ARMS or file not in files or values.dtype!=np.float64 or values.shape!=(len(m['endpoint_ids']),2) or not np.isfinite(values).all() or (np.abs(values)>LIMIT).any():raise ValueError('finite capped float64 endpoint shifts required')
            if arm=='K' and np.any(values!=0):raise ValueError('K must retain every original box')
            if np.any(values[~np.asarray(m['eligible'],bool)]!=0):raise ValueError('ineligible endpoint shifted')
            mapping=getattr(values,'_mmap',None)
            if mapping is not None:mapping.close()
    return receipt


def load_predictions(root):
    receipt=verify_seal(root);predictions=json.loads((Path(root)/'predictions.json').read_text())
    return dict(receipt=receipt,predictions=predictions)


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


def load_numeric(frozen,receipt,prereg):
    """Only roster and frozen G/chi numbers, not native MOT/images/labels."""
    top=json.loads(checked(frozen,receipt,'deployment.json').read_text());identity=top['identity']
    if top['schema']!='center214-prepare-v1' or identity['prereg']!=prereg or identity['supplement']!=SUPPLEMENT or top['originalK']!=18:raise ValueError('fixed original18 registrations differ')
    if identity['source212']!='61774c5f725073b1d887fdc06a8dd3a390a7566406eef12fe31c3fc859fd3b2b' or identity['source208_nominal']!='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94' or identity['code_sha256']['features214.py']!=FEATURE_SHA:raise ValueError('old native/feature source lineage differs')
    if len(top['clips'])!=6 or {(c['domain'],c['clip']) for c in top['clips']}!=CLIPS:raise ValueError('fixed six old clips required')
    rows=[];members=[];support=[]
    for ref in top['clips']:
        h,name=ref['domain'],ref['clip'];path=h+'/'+name
        if ref['path']!=path:raise ValueError('frozen clip path differs')
        with gzip.open(checked(frozen,receipt,path+'/roster.json.gz'),'rt',encoding='utf8') as stream:roster=json.load(stream)
        if roster['schema']!='center214-clip-v1' or roster['domain']!=h or roster['clip']!=name or roster['G_names']!=list(G_NAMES) or roster['chi_names']!=list(CHI_NAMES):raise ValueError('frozen endpoint/feature schema differs')
        feature_ids=json.loads(checked(frozen,receipt,path+'/feature_rows.json').read_text())
        with np.load(checked(frozen,receipt,path+'/data.npz'),allow_pickle=False) as data:G=np.array(data['G'],copy=True);chi=np.array(data['chi'],copy=True)
        if G.dtype!=np.float64 or chi.dtype!=np.float64 or G.shape!=(len(feature_ids),6) or chi.shape!=(len(feature_ids),24) or len(set(feature_ids))!=len(feature_ids) or not np.isfinite(G).all() or not np.isfinite(chi).all():raise ValueError('finite eligible-only double feature tables required')
        if [m['native_slot'] for m in roster['members']]!=[0,1,2]:raise ValueError('three original members required')
        for m in roster['members']:
            slot=m['native_slot']
            if m['member_index']!=slot or m['member_role']!=('receiver' if slot==0 else 'reserve') or m['seed']!=(3 if h in ('35','33') else 0)+slot:raise ValueError('original receiver/reserve member identity differs')
            members.append(dict(domain=h,clip=name,member=str(m['member']),member_index=slot,native_slot=slot,role='receiverK' if slot==0 else 'peerK',seed=m['seed'],tag=m['tag']))
        seen_features=[]
        for e in roster['records']:
            slot=e['native_slot'];eligible=e['gate']['eligible']
            if e['domain']!=h or e['clip']!=name or e['member_index']!=slot or e['eligible']!=eligible or e['run_id']!=e['canonical_id']:raise ValueError('endpoint scope/eligibility/run alias differs')
            if eligible:
                j=e['features']['feature_row']
                if not 0<=j<len(feature_ids) or feature_ids[j]!=e['endpoint_id']:raise ValueError('endpoint feature row join differs')
                g,c=G[j].tolist(),chi[j].tolist();seen_features.append(e['endpoint_id'])
                if g[2]!={'start':1.,'end':-1.,'single':0.}[e['role']]:raise ValueError('temporal endpoint role feature differs')
            else:
                if e['features'] is not None:raise ValueError('ineligible endpoint cannot contain a fabricated feature')
                g=c=None
            rows.append(dict(domain=h,clip=name,member=str(e['member']),member_index=slot,native_slot=slot,role='receiverK' if slot==0 else 'peerK',
                run_id=e['run_id'],endpoint_id=e['endpoint_id'],endpoint_kind=e['role'],raw_index=e['raw_index'],frame=e['frame'],
                box_xywh=e['box_xywh'],eligible=eligible,G6=g,chi24=c))
        if set(seen_features)!=set(feature_ids) or len(seen_features)!=len(feature_ids):raise ValueError('all eligible endpoints must have exactly one feature row')
        support.append(dict(domain=h,clip=name,all_endpoints=len(roster['records']),eligible=len(seen_features),counts=roster['counts']))
        del roster,G,chi
    validate(rows)
    if len(members)!=18 or len({member_key(m) for m in members})!=18:raise ValueError('complete original18 members required')
    return rows,members,dict(identity=identity,clip_support=support)


def load_source_labels(root,heldout,source,*,requested_domains,receipt,prereg,frozen_sha):
    allowed=tuple(requested_domains)
    if heldout not in DOMAINS or len(allowed)!=3 or set(allowed)!=set(DOMAINS)-{heldout}:raise ValueError('explicit other-three source domains required')
    if any(r['domain'] not in allowed or r['role']!='receiverK' or not r['eligible'] for r in source):raise ValueError('only eligible source receiver endpoints may be supervised')
    expected={r['endpoint_id']:r for r in source};labels={};proof={}
    for h in allowed:
        data=json.loads(checked(root,receipt,h+'.json').read_text())
        if data['domain']!=h or data['prereg']!=prereg or data['supplement']!=SUPPLEMENT or data['frozen_manifest_sha256']!=frozen_sha:raise ValueError('source label registration/frozen lineage differs')
        for r in data['records']:
            if r['domain']!=h or r['role']!='receiverK' or r['member_index']!=0 or not r['eligible']:raise ValueError('peer/target/ineligible label must not enter source shard')
            key=r['endpoint_id']
            if key not in expected or key in labels:raise ValueError('duplicate/unknown source endpoint label')
            e=expected[key]
            if r['raw_index']!=e['raw_index'] or r['canonical_run_id']!=e['run_id'] or r['clip']!=e['clip']:raise ValueError('source native row/canonical run join differs')
            _label(e,r)  # Validate exact assigned-row target; do not require isolation.
            if not r['has_owner'] and (r['dx_raw'] is not None or r['dy_raw'] is not None):raise ValueError('unmatched endpoints must remain unlabelled, not zero targets')
            labels[key]={k:r[k] for k in ('has_owner','target_known','raw_index','pred_width','pred_height','dx_raw','dy_raw')}
        proof[h]=dict(file=h+'.json',sha256=receipt['files'][h+'.json'],eligible_receiver_endpoints=len(data['records']))
        del data
    if set(labels)!=set(expected):raise ValueError('complete eligible receiver source labels required, including unmatched')
    return dict(source_domains=list(allowed),labels=labels,receipts=dict(label_manifest_sha256=receipt['manifest_sha256'],source_shards=proof,
        target_access='no heldout shard, peer owners, global label deployment, GTside/edge diagnostics or oracleO'))


def field_train(args):
    if args.prereg!=MAIN_PREREG:raise ValueError('registered214 main prereg required')
    fr,lr=manifest(args.frozen),manifest(args.labels)
    if (fr['manifest_sha256'],lr['manifest_sha256'])!=(args.frozen_sha,args.label_sha):raise ValueError('GO input seals differ')
    rows,members,scope=load_numeric(args.frozen,fr,args.prereg)
    def provider(h,source):
        result=load_source_labels(args.labels,h,source,requested_domains=[d for d in DOMAINS if d!=h],receipt=lr,prereg=args.prereg,frozen_sha=fr['manifest_sha256'])
        print('SOURCE214',h,'eligible_receiver_endpoints',len(result['labels']),flush=True);return result
    return train_and_seal(rows,members,provider,args.out,dict(prereg=args.prereg,supplement=SUPPLEMENT,frozen_manifest_sha256=fr['manifest_sha256'],
        label_manifest_sha256=lr['manifest_sha256'],feature_source_sha256=FEATURE_SHA,numeric_scope=scope))


def self_test():
    rng=np.random.default_rng(21401);rows=[];labels={};members=[]
    for h in DOMAINS:
        for clip in (h+'A',h+'B'):
            for member in range(3):
                members.append(dict(domain=h,clip=clip,member=str(member),member_index=member,role='receiverK' if member==0 else 'peerK'))
                for run,n in ((1,2),(2,1)) if clip.endswith('A') else ((1,1),):
                    for j in range(n):
                        eid=f'{h}/{clip}/{member}/{run}/{j}';g=rng.normal(size=6);g[0]=.5
                        r=dict(domain=h,clip=clip,member=str(member),member_index=member,role='receiverK' if member==0 else 'peerK',run_id=run,
                            endpoint_id=eid,raw_index=run*2+j,box_xywh=[10.,20.,4.,8.],eligible=True,G6=g.tolist(),chi24=(rng.normal(size=24)+.7).tolist())
                        rows.append(r);labels[eid]=dict(has_owner=True,target_known=True,raw_index=r['raw_index'],pred_width=4.,pred_height=8.,dx_raw=float(rng.normal()),dy_raw=0.)
    source=[r for r in rows if r['domain']!='35' and r['role']=='receiverK'];model=fit(source,{r['endpoint_id']:labels[r['endpoint_id']] for r in source},heldout='35')
    assert model['n']==12 and model['heads']['G']['scale'][0]==1. and model['heads']['G']['mean']==model['heads']['I']['mean'][:6]
    assert model['heads']['I']['intercept'][1]==0 and np.all(np.asarray(model['heads']['I']['beta'])[:,1]==0)
    for head in HEADS:
        assert model['heads'][head]['intercept']==model['heads'][head]['source_target_mean']
        assert np.max(np.abs(model['heads'][head]['optimization']['gradient_beta']))<1e-10
    rs=model['training_records']
    for h in ('33','W','G'):
        assert np.isclose(sum(r['weight'] for r in rs if r['domain']==h),1/3)
        assert all(np.isclose(sum(r['weight'] for r in rs if r['domain']==h and r['clip']==c),1/6) for c in (h+'A',h+'B'))
    target=[r for r in rows if r['domain']=='35'];z=predict(model,target,'Z');I=model['heads']['I']
    for i,r in enumerate(target):
        raw=np.r_[r['G6'],-np.asarray(r['chi24'])]
        expected=np.clip(((raw-I['mean'])/I['scale'])@np.asarray(I['beta'])+I['intercept'],-LIMIT,LIMIT)
        assert np.array_equal(z[i],expected)
    empty=fit([],{},heldout='35');assert np.all(predict(empty,target,'I')==0)
    one=fit(source[:1],{source[0]['endpoint_id']:labels[source[0]['endpoint_id']]},heldout='35');assert one['n']==1 and one['heads']['I']['status']=='fitted'
    unmatched={r['endpoint_id']:dict(labels[r['endpoint_id']],has_owner=False,target_known=False,dx_raw=None,dy_raw=None) for r in source}
    assert fit(source,unmatched,heldout='35')['n']==0
    capped=dict(labels[source[0]['endpoint_id']],dx_raw=100.,dy_raw=-100.)
    assert np.array_equal(_label(source[0],capped),[.25,-.25])
    # The same raw chi rotated180 is negative; source normalization stays frozen.
    assert np.array_equal(predict(model,[dict(r,chi24=(-np.asarray(r['chi24'])).tolist()) for r in target],'I'),z)
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);labelroot=tmp/'labels';labelroot.mkdir()
        for h in DOMAINS:
            entries=[dict(labels[r['endpoint_id']],domain=h,clip=r['clip'],role='receiverK',member_index=0,eligible=True,endpoint_id=r['endpoint_id'],canonical_run_id=r['run_id']) for r in rows if r['domain']==h and r['role']=='receiverK']
            save(labelroot/(h+'.json'),dict(domain=h,prereg='synthetic',supplement=SUPPLEMENT,frozen_manifest_sha256='frozen',records=entries))
        save(labelroot/'manifest.json',{p.name:sha(p) for p in labelroot.iterdir()});(labelroot/'SEALED.sha256').write_text(sha(labelroot/'manifest.json'))
        lr=manifest(labelroot)
        for h in DOMAINS:
            path=labelroot/(h+'.json');raw=path.read_bytes();path.unlink();source=[r for r in rows if r['domain']!=h and r['role']=='receiverK']
            supplied=load_source_labels(labelroot,h,source,requested_domains=[d for d in DOMAINS if d!=h],receipt=lr,prereg='synthetic',frozen_sha='frozen')
            assert len(supplied['labels'])==12
            path.write_bytes(raw)
        try:load_source_labels(Path('must-not-open'),'35',source,requested_domains=['35','W','G'],receipt={},prereg='synthetic',frozen_sha='frozen')
        except ValueError:pass
        else:raise AssertionError('heldout label whitelist accepted')
        def provider(h,source):return dict(source_domains=[d for d in DOMAINS if d!=h],labels={r['endpoint_id']:labels[r['endpoint_id']] for r in source})
        sealed=train_and_seal(rows,members,provider,tmp/'pred',dict(prereg='synthetic'))
        assert len(load_predictions(sealed['root'])['predictions']['members'])==24
    return dict(status='PASS',seed=21401,tests='hive/clip/run/endpoint weights; shared scaler; exact constant coordinate/std; unpenalized intercept; ridge gradient; single matched row; empty/unmatched; rawchi negation sameIscaler; target/pred clip; heldout shards absent; scope before IO; common fourfold shifts')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--run-field',action='store_true')
    parser.add_argument('--frozen',type=Path);parser.add_argument('--labels',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--frozen-sha');parser.add_argument('--label-sha');parser.add_argument('--prereg');args=parser.parse_args()
    if args.run_field:
        if any(v is None for v in (args.frozen,args.labels,args.out,args.frozen_sha,args.label_sha)):parser.error('all roots and pinned seals required')
        print(json.dumps(field_train(args),indent=2))
    elif args.self_test:print(json.dumps(self_test(),indent=2))
    else:parser.error('--self-test or --run-field required')
