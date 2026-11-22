"""Receiver-K-only source logistic models and frozen count-policy guards.

The weak binary target is existence of ANY legal GT event edge, not membership
in one selected maximum matching. Thus sum(p) is not a one-to-one TP estimate
or a guarantee of F1. Peer-K labels and target outcomes never enter fitting.
R/T share G's source cohort/weights; S reuses T's parameters and scaler, with
independent fixed-salt residual features, never a separately trained model.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter,defaultdict
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
TRAIN_HEADS=('G','R','T')
PRED_HEADS=('G','R','T','S')
ARMS=('K','C',*PRED_HEADS)
DIMENSIONS=dict(G=7,R=13,T=19)
G_NAMES=('log1p_n','n_over_inclusive_span','log1p_side_flips','min_endpoint_same_side_run_over_n',
         'min_inside_outside_over_n','max_original_gap_over_25','direction')
PHASE_NAMES=('k2_O','k2_D','k2_T','k3_O','k3_D','k3_T')
FEATURE_NAMES=(*G_NAMES,*(f'R_{n}' for n in PHASE_NAMES),*(f'P_minus_R_{n}' for n in PHASE_NAMES))
RIDGE=.1
GUARD=1e-9
SCHEMA='phase212-source-logistic-v1'
CLIPS={('35','BEE24-35-head'),('33','BEE24-33-head'),('W','BEE24-10'),('W','BEE24-15'),('G','BEE24-24'),('G','BEE24-28')}
FEATURE_SHA='a6d5a17af04f6e4e6e5211c882166978782c2efe52ca740f9762d8bd8b9e9f70'
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


def validate_bank(bank):
    for key,f in bank.items():
        if not isinstance(key,str) or not key:raise ValueError('nonempty feature-key required')
        for name,n in (('G7',7),('R6',6),('P6',6),('R_alt6',6)):
            x=np.asarray(f[name],float)
            if x.shape!=(n,) or not np.isfinite(x).all():raise ValueError('finite fixed '+name+' required')
        if f['G7'][-1] not in (-1,1):raise ValueError('original predicted net direction +/-1 required, never GT direction')
        if any(np.any((np.asarray(f[n])<0)|(np.asarray(f[n])>1)) for n in ('R6','P6','R_alt6')):raise ValueError('phase O/D/T fractions must be0..1')
    return bank


def feature_vector(f,head):
    G,R,P,alt=(np.asarray(f[n],float) for n in ('G7','R6','P6','R_alt6'))
    if head=='G':return G
    if head=='R':return np.r_[G,R]
    if head=='T':return np.r_[G,R,P-R]
    if head=='S':return np.r_[G,R,alt-R]
    raise ValueError('unknown phase head')


def validate_event_refs(records,bank,*,receiver_only=False):
    seen=set()
    for r in records:
        if r['domain'] not in DOMAINS or not isinstance(r['clip'],str) or not isinstance(r['event_ref'],str):raise ValueError('event scope/ref required')
        if r['event_ref'] in seen or r['feature_key'] not in bank:raise ValueError('duplicate/missing event feature-key')
        seen.add(r['event_ref'])
        if receiver_only and r['role']!='receiverK':raise ValueError('peer/state labels cannot enter training')
    return records


def source_weights(records):
    if not records:return np.empty(0,float)
    hives={r['domain'] for r in records};clips=defaultdict(set);counts=Counter()
    for r in records:clips[r['domain']].add(r['clip']);counts[r['domain'],r['clip']]+=1
    w=np.asarray([1/(len(hives)*len(clips[r['domain']])*counts[r['domain'],r['clip']]) for r in records],float)
    return w/w.sum()  # Only remove floating total error; no label balancing.


def objective(theta,Z,y,w):
    beta,bias=theta[:-1],theta[-1];z=Z@beta+bias;residual=w*(expit(z)-y)
    return float(w@(np.logaddexp(0.,z)-y*z)+RIDGE/2*(beta@beta)),np.r_[Z.T@residual+RIDGE*beta,residual.sum()]


def fit(records,labels,bank,*,heldout=None):
    validate_event_refs(records,bank,receiver_only=True)
    if heldout is not None and any(r['domain']==heldout for r in records):raise ValueError('heldout receiver-K training event')
    if set(labels)!={r['event_ref'] for r in records} or any(v not in (0,1) for v in labels.values()):raise ValueError('complete receiver-K binary edge-existence labels required')
    rows=sorted(records,key=lambda r:(r['domain'],r['clip'],r['event_ref']));n=len(rows)
    y=np.asarray([labels[r['event_ref']] for r in rows],float);w=source_weights(rows);npos=int(y.sum())
    result=dict(schema=SCHEMA,heldout=heldout,source_domains=[h for h in DOMAINS if h!=heldout],ridge=RIDGE,n=n,npos=npos,
        label_scope='any legal event edge in complete official.5 true graph; not fixed matched TP or211 deletion criticality',
        source_counts={h:dict(events=sum(r['domain']==h for r in rows),positive=sum(r['domain']==h and labels[r['event_ref']]==1 for r in rows)) for h in DOMAINS if h!=heldout},
        weight_rule='source receiver-K event hives equal; their nonempty K-event clips equal; events equal; no class balancing; floating-total renormalized',
        training_records=[dict(r,y=int(y[i]),weight=float(w[i]),features19=feature_vector(bank[r['feature_key']],'T').tolist(),
                               G7=bank[r['feature_key']]['G7'],R6=bank[r['feature_key']]['R6'],P6=bank[r['feature_key']]['P6'],R_alt6=bank[r['feature_key']]['R_alt6']) for i,r in enumerate(rows)],heads={})
    clip_indices={key:i for i,key in enumerate(sorted({(r['domain'],r['clip']) for r in rows}))}
    result['anonymous_training_rows']=[dict(row_index=i,domain_index=DOMAINS.index(r['domain']),clip_index=clip_indices[r['domain'],r['clip']],
        X19=feature_vector(bank[r['feature_key']],'T').tolist(),y=int(y[i]),weight=float(w[i])) for i,r in enumerate(rows)]
    if not n or npos in (0,n):
        p=.5 if not n else (npos+1)/(n+2)
        status='empty' if not n else 'single_class_smoothed'
        result['heads']={head:dict(status=status,constant_probability=p,mean=None,scale=None,beta=[0.]*DIMENSIONS[head],
            intercept=float(np.log(p)-np.log1p(-p)),n=n,npos=npos,feature_names=list(FEATURE_NAMES[:DIMENSIONS[head]]),
            optimization=dict(success=True,message='no fitting: constant fallback',iterations=0)) for head in TRAIN_HEADS}
        return result
    X=np.asarray([feature_vector(bank[r['feature_key']],'T') for r in rows],float)
    mean=w@X;constant=np.all(X==X[0],axis=0);mean[constant]=X[0,constant]
    scale=np.sqrt(w@((X-mean)**2));scale[constant]=0.;zero=scale==0;scale[zero]=1.
    for head in TRAIN_HEADS:
        d=DIMENSIONS[head];Z=(X[:,:d]-mean[:d])/scale[:d]
        opt=minimize(lambda t:objective(t,Z,y,w),np.zeros(d+1),jac=True,method='L-BFGS-B',options=dict(maxiter=500,gtol=1e-8,ftol=1e-12))
        if not opt.success or not np.isfinite(opt.x).all():raise RuntimeError('source '+head+' logistic failed: '+str(opt.message))
        result['heads'][head]=dict(status='fitted',feature_names=list(FEATURE_NAMES[:d]),mean=mean[:d].tolist(),scale=scale[:d].tolist(),
            zero_scale_columns=np.flatnonzero(zero[:d]).tolist(),beta=opt.x[:-1].tolist(),intercept=float(opt.x[-1]),
            optimization=dict(success=True,message=str(opt.message),iterations=int(opt.nit),objective=float(opt.fun),gradient=objective(opt.x,Z,y,w)[1].tolist(),
                initialization=[0.]*(d+1),solver='scipy L-BFGS-B',options=dict(maxiter=500,gtol=1e-8,ftol=1e-12),intercept_penalized=False))
    return result


def predict_feature(model,f,head):
    h=model['heads']['T' if head=='S' else head]
    if h['status']!='fitted':return float(h['constant_probability'])
    x=feature_vector(f,head);z=((x-h['mean'])/h['scale'])@h['beta']+h['intercept']
    return float(expit(z))


def canonical_actions(state):
    result=[]
    for a in state['actions']:
        family=a['family'];ids=a['canonical_ids']
        if family not in ('C','I') or len(ids)!=2 or any(int(i)!=i for i in ids) or ids[0]>=ids[1]:raise ValueError('registered family/numeric canonical pair required')
        result.append((family,int(ids[0]),int(ids[1])))
    if len(set(result))!=len(result) or len(result)!=state['edits']:raise ValueError('action/edit count differs')
    return tuple(sorted(result))


def count_choice(states,K):
    return min(states,key=lambda s:(-round(float(K['raw_meanCountL1']-s['raw_meanCountL1']),10),s['edits'],canonical_actions(s)))


def choose_clip(clip,model,bank):
    states=clip['states'];ids=[s['state_id'] for s in states]
    if len(set(ids))!=len(ids):raise ValueError('duplicate state ID')
    baselines=[s for s in states if not canonical_actions(s)]
    if len(baselines)!=1:raise ValueError('exactly one empty K state required')
    K=baselines[0];C=count_choice(states,K)
    if K['state_id']!=clip['K_state_id'] or C['state_id']!=clip['C_state_id']:raise ValueError('original211 K/count-C choice lineage differs')
    probability_cache={}
    def probabilities(refs):
        result=[];seen=set()
        for r in refs:
            if r['event_ref'] in seen:raise ValueError('duplicate event within a scene')
            seen.add(r['event_ref']);key=r['feature_key']
            if key not in bank:raise ValueError('missing event feature')
            if key not in probability_cache:probability_cache[key]={head:predict_feature(model,bank[key],head) for head in PRED_HEADS}
            result.append(dict(r,probabilities=probability_cache[key]))
        return result
    scored=[]
    for s in states:
        canonical_actions(s)
        if s['nPred']!=len(s['events']) or not np.isfinite(s['raw_meanCountL1']) or s['raw_meanCountL1']<0:raise ValueError('state event/count support differs')
        events=probabilities(s['events']);E={h:float(s['nPred']-2*sum(e['probabilities'][h] for e in events)) for h in PRED_HEADS}
        scored.append(dict(s,event_predictions=events,Ehat=E,count_utility=float(K['raw_meanCountL1']-s['raw_meanCountL1'])))
    baseline=next(s for s in scored if s['state_id']==K['state_id']);selected={'K':K,'C':C}
    for s in scored:
        s['delta_Ehat']={h:s['Ehat'][h]-baseline['Ehat'][h] for h in PRED_HEADS}
        s['guard_eligible']={h:s['delta_Ehat'][h]<=GUARD for h in PRED_HEADS}
    for head in PRED_HEADS:
        eligible=[s for s in scored if s['guard_eligible'][head]]
        if not eligible:raise AssertionError('K must always satisfy its own guard')
        selected[head]=count_choice(eligible,K)
    originals=[]
    for m in clip['originals']:
        if m['role'] not in ('receiverK','peerK'):raise ValueError('original K role required')
        originals.append(dict(m,event_predictions=probabilities(m['events'])))
    return dict(domain=clip['domain'],clip=clip['clip'],states=scored,originals=originals,
        selected={arm:dict(state_id=s['state_id'],selected_action_ids=s['selected_action_ids'],edits=s['edits']) for arm,s in selected.items()},
        K_state_id=K['state_id'],C_state_id=C['state_id'],guard_threshold=GUARD,
        choice_rule='within per-head relativeK Ehat<=1e-9: maximize Pythonround(countutility,10), fewer edits, sorted(family,numericA,numericB)lex',
        Ehat_scope='nPred-2*sum(any-legal-edge probabilities); not maximum matching TP/F1 assurance')


def train_and_seal(bank,clips,label_provider,outdir,input_receipt):
    validate_bank(bank)
    if sha(__file__)!=CODE_SHA:raise ValueError('loaded source changed')
    source_records=[]
    for c in clips:
        if c['domain'] not in DOMAINS:raise ValueError('registered domain required')
        for m in c['originals']:
            if m['role']=='receiverK':
                source_records.extend(dict(e,domain=c['domain'],clip=c['clip'],role='receiverK',member=m['member'],member_index=m['member_index']) for e in m['events'])
    validate_event_refs(source_records,bank,receiver_only=True)
    dest=Path(outdir)
    if dest.exists():raise FileExistsError('immutable output root exists')
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        predictions={};folds={}
        for h in DOMAINS:
            source=[r for r in source_records if r['domain']!=h];supplied=label_provider(h,source)
            if set(supplied['source_domains'])!=set(DOMAINS)-{h}:raise ValueError('other-three source scope required')
            model=fit(source,supplied['labels'],bank,heldout=h);model['source_receipts']=supplied.get('receipts',{})
            name='models/'+h+'.json';save(tmp/name,model);folds[h]=dict(heldout=h,source_domains=[d for d in DOMAINS if d!=h],model_file=name,n=model['n'],npos=model['npos'])
            for c in clips:
                if c['domain']==h:predictions[h+'/'+c['clip']]=choose_clip(c,model,bank)
        if sha(__file__)!=CODE_SHA:raise ValueError('fit source changed during fitting')
        save(tmp/'predictions.json',dict(schema=SCHEMA,arms=list(ARMS),folds=folds,clips=predictions))
        save(tmp/'receipt.json',dict(input_receipt,schema=SCHEMA,code_sha256=CODE_SHA,numpy_version=np.__version__,scipy_version=scipy.__version__,
            peer_labels_prohibited_from_fit=True,all_target_true_outcomes_prohibited_before_common_seal=True))
        save(tmp/'manifest.json',{p.relative_to(tmp).as_posix():sha(p) for p in tmp.rglob('*') if p.is_file()});digest=sha(tmp/'manifest.json')
        (tmp/'SEALED.sha256').write_text(digest+'\n');tmp.rename(dest)
        return dict(root=str(dest),manifest_sha256=digest)
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def verify_seal(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('prediction seal differs')
    files=json.loads((root/'manifest.json').read_text())
    if set(files)!={'predictions.json','receipt.json'}|{'models/'+h+'.json' for h in DOMAINS}:raise ValueError('four models/common state+peer predictions required')
    for name,digest in files.items():
        if sha(root/name)!=digest:raise ValueError('sealed output changed')
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt['code_sha256']!=sha(__file__):raise ValueError('frozen fit source changed')
    predictions=json.loads((root/'predictions.json').read_text())
    if set(predictions['folds'])!=set(DOMAINS):raise ValueError('four heldout folds required')
    for h,f in predictions['folds'].items():
        if f['heldout']!=h or set(f['source_domains'])!=set(DOMAINS)-{h}:raise ValueError('source-only fold differs')
    return receipt


def load_predictions(root):
    receipt=verify_seal(root);predictions=json.loads((Path(root)/'predictions.json').read_text());decisions={}
    for c in predictions['clips'].values():decisions.setdefault(c['domain'],{})[c['clip']]=c['selected']
    return dict(receipt=receipt,predictions=predictions,decisions=decisions)


def manifest(root):
    root=Path(root);digest=sha(root/'manifest.json')
    if (root/'SEALED.sha256').read_text().strip()!=digest:raise ValueError('input top seal differs')
    files=json.loads((root/'manifest.json').read_text())
    for name in files:
        if not (root/name).resolve().is_relative_to(root.resolve()):raise ValueError('input path escapes root')
    return dict(manifest_sha256=digest,files=files)


def checked(root,receipt,name):
    if name not in receipt['files'] or sha(Path(root)/name)!=receipt['files'][name]:raise ValueError('sealed numeric input changed: '+name)
    return Path(root)/name


def read_gz(path):
    with gzip.open(path,'rt',encoding='utf8') as stream:return json.load(stream)


def load_numeric(frozen,receipt,prereg):
    """Only phasebank/scene event references/counts. No native, labels or GT."""
    top=json.loads(checked(frozen,receipt,'deployment.json').read_text())
    if top['schema']!='phase212-prepare-v1' or top['prereg']!=prereg or top['states']!=86 or top['originalK']!=18 or len(top['clips'])!=6:raise ValueError('complete original86/18K frozen roster required')
    identity=top['identity']
    if identity['main211_frozen']!='c21b74bbf3accd85a8a48863d16c1d241ce3045032b67ab8e6d92f390b88d6ae' or identity['inputs208']!='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94' or identity['code_sha256']['features212.py']!=FEATURE_SHA:raise ValueError('phase source lineage differs')
    raw=read_gz(checked(frozen,receipt,top['feature_bank_file']))
    if raw['schema']!='phase212-unique-feature-bank-v1' or raw['feature_schema']!='phase212.features.v1' or raw['G_names']!=list(G_NAMES) or raw['phase_names']!=list(PHASE_NAMES):raise ValueError('fixed phase feature contract differs')
    bank={}
    for key,f in raw['features'].items():
        sequence=dict(dates=f['dates'],inside=f['inside'])
        digest=hashlib.sha256(json.dumps(sequence,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if key!=digest or f['feature_key']!=key or not np.array_equal(np.asarray(f['T']),np.asarray(f['P'])-np.asarray(f['R'])):raise ValueError('exact sequence key/residual differs')
        bank[key]=dict(G7=f['G'],R6=f['R'],P6=f['P'],R_alt6=f['R_alt'])
    validate_bank(bank);del raw
    clips=[];roles=Counter();total_states=0
    for ref in top['clips']:
        h,name=ref['domain'],ref['clip'];path=h+'/'+name
        if ref['path']!=path or (h,name) not in CLIPS:raise ValueError('fixed six clip identity required')
        meta=json.loads(checked(frozen,receipt,path+'/clip.json').read_text())
        if meta['schema']!='phase212-clip-v1' or meta['domain']!=h or meta['clip']!=name or meta['original211_joint_prediction_seal']!=identity['joint_GTfree_seal']:raise ValueError('original joint roster lineage differs')
        scenes={};originals=[]
        for scene in meta['scenes']:
            scene_ref=scene['scene_ref']
            if scene_ref in scenes:raise ValueError('duplicate physical scene ref')
            data=read_gz(checked(frozen,receipt,path+'/'+scene['events_file']))
            if data['domain']!=h or data['clip']!=name or data['scene_ref']!=scene_ref or data['nPred']!=len(data['events']) or scene['nPred']!=data['nPred']:raise ValueError('scene event scope/count differs')
            validate_event_refs([dict(e,domain=h,clip=name) for e in data['events']],bank)
            scenes[scene_ref]=data
            if scene['role'] in ('receiverK','peerK'):
                slot=scene['member_index']
                if scene_ref!=f'K/m{slot}' or slot not in (0,1,2) or scene['role']!=('receiverK' if slot==0 else 'peerK'):raise ValueError('original18K member role differs')
                originals.append(dict(role=scene['role'],member=str(scene['member']),member_index=slot,seed=scene['seed'],tag=scene['tag'],scene_ref=scene_ref,
                    nPred=data['nPred'],events=data['events']))
                roles[scene['role']]+=1
        originals.sort(key=lambda r:r['member_index'])
        if [r['member_index'] for r in originals]!=[0,1,2] or originals[0]['seed']!=(3 if h in ('35','33') else 0):raise ValueError('fixed receiver and two original peer K required')
        states=[]
        for s in meta['states']:
            if s['scene_ref'] not in scenes:raise ValueError('state scene missing')
            actions=[]
            for aid in s['selected_action_ids']:
                a=meta['original211_actions'][aid];actions.append(dict(family=a['family'],canonical_ids=a['canonical_ids']))
            projected=dict(state_id=s['state_id'],selected_action_ids=s['selected_action_ids'],actions=actions,edits=s['edits'],nPred=s['nPred'],
                raw_meanCountL1=s['raw_meanCountL1'],events=scenes[s['scene_ref']]['events'])
            if [list(t) for t in canonical_actions(projected)]!=s['tie_tuple'] or s['nPred']!=len(projected['events']):raise ValueError('original edit tuple/event support differs')
            states.append(projected)
        empty=[s for s in states if not s['selected_action_ids']]
        if len(empty)!=1 or empty[0]['events']!=originals[0]['events']:raise ValueError('K event roster must be original receiverK scene')
        clips.append(dict(domain=h,clip=name,states=states,originals=originals,K_state_id=empty[0]['state_id'],C_state_id=meta['original211_C_chosen_state_id']))
        total_states+=len(states);del meta,scenes
    if {(c['domain'],c['clip']) for c in clips}!=CLIPS or total_states!=86 or roles!=dict(receiverK=6,peerK=12):raise ValueError('complete original sixclip86state18K roles required')
    return bank,clips,dict(feature_bank_sequences=len(bank),states=total_states,original_roles=dict(roles),identity=identity)


def receiver_records(clips):
    return [dict(e,domain=c['domain'],clip=c['clip'],role='receiverK',member=m['member'],member_index=0)
            for c in clips for m in c['originals'] if m['role']=='receiverK' for e in m['events']]


def load_source_labels(root,heldout,source,*,requested_domains,receipt,prereg,frozen_sha):
    allowed=tuple(requested_domains)
    if heldout not in DOMAINS or len(allowed)!=3 or set(allowed)!=set(DOMAINS)-{heldout}:raise ValueError('explicit other-three source domains required')
    if any(r['domain'] not in allowed or r['role']!='receiverK' for r in source):raise ValueError('only allowed source receiverK events may be supervised')
    expected={r['event_ref']:r for r in source};labels={};proof={}
    if len(expected)!=len(source):raise ValueError('duplicate source receiver eventref')
    for h in allowed:
        data=json.loads(checked(root,receipt,h+'.json').read_text())
        if data['domain']!=h or data['prereg']!=prereg or data['frozen_manifest_sha256']!=frozen_sha:raise ValueError('source label header differs')
        for r in data['records']:
            if r['domain']!=h or r['role']!='receiverK' or r['member_index']!=0:raise ValueError('peer/target labels must not be parsed in a receiver source shard')
            key=r['event_ref']
            if key not in expected or key in labels or r['y'] not in (0,1) or r['feature_key']!=expected[key]['feature_key'] or r['clip']!=expected[key]['clip']:raise ValueError('source eventref/feature/weak label join differs')
            if 'canonical_id' in expected[key] and r.get('canonical_event_id')!=expected[key]['canonical_id']:raise ValueError('source original K canonical event alias differs')
            labels[key]=int(r['y'])
        proof[h]=dict(file=h+'.json',sha256=receipt['files'][h+'.json'],receiver_events=len(data['records']))
        del data
    if set(labels)!=set(expected):raise ValueError('complete source receiverK labels required')
    return dict(source_domains=list(allowed),labels=labels,receipts=dict(label_manifest_sha256=receipt['manifest_sha256'],source_shards=proof,
        target_access='no heldout shard, peer y, original18K assessment, global label deployment or true-state outcome'))


def field_train(args):
    if not args.prereg or len(args.prereg)!=40 or any(c not in '0123456789abcdef' for c in args.prereg):raise ValueError('full published phase212 prereg required')
    fr,lr=manifest(args.frozen),manifest(args.labels)
    if (fr['manifest_sha256'],lr['manifest_sha256'])!=(args.frozen_sha,args.label_sha):raise ValueError('GO input seals differ')
    bank,clips,proof=load_numeric(args.frozen,fr,args.prereg)
    def provider(h,source):
        supplied=load_source_labels(args.labels,h,source,requested_domains=[d for d in DOMAINS if d!=h],receipt=lr,prereg=args.prereg,frozen_sha=fr['manifest_sha256'])
        print('SOURCE212',h,'receiverK events',len(supplied['labels']),flush=True);return supplied
    return train_and_seal(bank,clips,provider,args.out,dict(prereg=args.prereg,frozen_manifest_sha256=fr['manifest_sha256'],
        label_manifest_sha256=lr['manifest_sha256'],feature_source_sha256=FEATURE_SHA,numeric_scope=proof))


def self_test():
    rng=np.random.default_rng(21201);bank={};records=[];labels={}
    for h in DOMAINS:
        for i in range(8):
            key=h+str(i);g=rng.normal(size=7);g[0]=1.;g[-1]=1 if i%2 else -1
            bank[key]=dict(G7=g.tolist(),R6=rng.uniform(size=6).tolist(),P6=rng.uniform(size=6).tolist(),R_alt6=rng.uniform(size=6).tolist())
            r=dict(domain=h,clip=h+'clip',role='receiverK',event_ref=h+'/'+str(i),feature_key=key);records.append(r);labels[r['event_ref']]=i%2
    validate_bank(bank);source=[r for r in records if r['domain']!='35'];ys={r['event_ref']:labels[r['event_ref']] for r in source}
    model=fit(source,ys,bank,heldout='35');assert model['n']==24 and model['npos']==12
    assert all(model['heads'][h]['scale'][0]==1. for h in TRAIN_HEADS)
    assert model['heads']['G']['mean']==model['heads']['T']['mean'][:7]
    imbalanced=dict.fromkeys(ys,0);imbalanced[next(iter(ys))]=1
    uneven=fit(source,imbalanced,bank,heldout='35')
    assert [r['weight'] for r in uneven['training_records']]==[r['weight'] for r in model['training_records']]
    for r in source:
        f=bank[r['feature_key']];head=model['heads']['T'];x=feature_vector(f,'S')
        assert predict_feature(model,f,'S')==float(expit(((x-head['mean'])/head['scale'])@head['beta']+head['intercept']))
    altered={k:dict(f,R_alt6=[0.]*6) for k,f in bank.items()}
    assert fit(source,ys,altered,heldout='35')['heads']==model['heads']
    assert fit([],{},bank,heldout='35')['heads']['T']['constant_probability']==.5
    constant=fit(source,{r['event_ref']:1 for r in source},bank,heldout='35');assert constant['heads']['T']['constant_probability']==25/26
    try:fit([dict(source[0],role='peerK')],{source[0]['event_ref']:1},bank)
    except ValueError:pass
    else:raise AssertionError('peer label leaked into fitting')
    def refs(k):return [dict(event_ref='state/'+k,feature_key=k)]
    clips=[]
    for h in DOMAINS:
        original=[dict(role=role,member='m'+str(i),member_index=i,events=[dict(event_ref=h+'/'+str(j) if i==0 else h+'/peer'+str(i)+'/'+str(j),feature_key=h+str(j)) for j in range(8)]) for i,role in enumerate(('receiverK','peerK','peerK'))]
        states=[dict(state_id='K',selected_action_ids=[],actions=[],edits=0,nPred=1,raw_meanCountL1=2.,events=refs(h+'0')),
                dict(state_id='A',selected_action_ids=['I|1|2'],actions=[dict(family='I',canonical_ids=[1,2])],edits=1,nPred=1,raw_meanCountL1=1.,events=refs(h+'1'))]
        clips.append(dict(domain=h,clip=h+'clip',states=states,originals=original,K_state_id='K',C_state_id='A'))
    empty=fit([],{},bank);out=choose_clip(clips[0],empty,bank)
    assert all(out['selected'][h]['state_id']=='A' for h in PRED_HEADS)  # Empty prior .5 gives Ehat0 for all.
    assert len(out['originals'])==3 and all(len(m['event_predictions'])==8 for m in out['originals'])
    Z=rng.normal(size=(24,4));y=rng.integers(0,2,24);w=np.ones(24)/24;t=rng.normal(size=5);_,grad=objective(t,Z,y,w);num=[]
    for j in range(5):
        d=np.eye(5)[j]*1e-6;num.append((objective(t+d,Z,y,w)[0]-objective(t-d,Z,y,w)[0])/2e-6)
    assert np.allclose(grad,num,atol=1e-8,rtol=1e-6)
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);labelroot=tmp/'labels';labelroot.mkdir()
        for h in DOMAINS:
            entries=[dict(r,member_index=0,y=labels[r['event_ref']]) for r in records if r['domain']==h]
            save(labelroot/(h+'.json'),dict(domain=h,prereg='synthetic',frozen_manifest_sha256='frozen',records=entries))
        save(labelroot/'manifest.json',{p.name:sha(p) for p in labelroot.iterdir()});(labelroot/'SEALED.sha256').write_text(sha(labelroot/'manifest.json'))
        lr=manifest(labelroot)
        for h in DOMAINS:
            path=labelroot/(h+'.json');raw=path.read_bytes();path.unlink();source=[r for r in records if r['domain']!=h]
            sup=load_source_labels(labelroot,h,source,requested_domains=[d for d in DOMAINS if d!=h],receipt=lr,prereg='synthetic',frozen_sha='frozen')
            assert len(sup['labels'])==24
            path.write_bytes(raw)
        try:load_source_labels(Path('must-not-open'),'35',source,requested_domains=['35','W','G'],receipt={},prereg='synthetic',frozen_sha='frozen')
        except ValueError:pass
        else:raise AssertionError('heldout label whitelist accepted')
        def provider(h,source):return dict(source_domains=[d for d in DOMAINS if d!=h],labels={r['event_ref']:labels[r['event_ref']] for r in source})
        sealed=train_and_seal(bank,clips,provider,tmp/'pred',dict(prereg='synthetic'))
        assert len(load_predictions(sealed['root'])['decisions'])==4
    return dict(status='PASS',seed=21201,tests='receiveronly; no peer y; shared weighted moments/zero std; no sham fit; fixedTscaler; empty/singleclass raw-n; guard fallback; 18K roster pattern; gradient; all4heldout shard absent; scope before IO; common fourfold seal')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--run-field',action='store_true')
    parser.add_argument('--frozen',type=Path);parser.add_argument('--labels',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--frozen-sha');parser.add_argument('--label-sha');parser.add_argument('--prereg');args=parser.parse_args()
    if args.run_field:
        if any(v is None for v in (args.frozen,args.labels,args.out,args.frozen_sha,args.label_sha)):parser.error('all roots and pinned seals required')
        print(json.dumps(field_train(args),indent=2))
    elif args.self_test:print(json.dumps(self_test(),indent=2))
    else:parser.error('--self-test or --run-field required')
