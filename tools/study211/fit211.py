"""Source-only event-criticality calibration and exhaustive blind state choice.

y is the maximum matching loss after deleting ONE fixed K left vertex. A zero
is nonessentiality, not an FP label; such a vertex may occur in a maximum
matching. Criticalities are not additive. Deployment uses each complete state
peer graph's J/M, never a sum of per-vertex deletion labels.

The small constrained least-squares heads are solved by enumerating the
triangle interior and its three edges, without iterative optimization.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

DOMAINS=('35','33','W','G')
ARMS=('K','Q','L','X','V','C','Z')
RIDGE=.01
SCHEMA='match211-constrained-calibration-v1'
SOURCE210='3810c1afabe45160642b6187c8fed89dac4e5e05dabb9f5be2cb696d666a7005'
SOURCE208='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94'
EXPECTED_STATES={('35','BEE24-35-head'):8,('33','BEE24-33-head'):8,('W','BEE24-10'):1,('W','BEE24-15'):2,('G','BEE24-24'):2,('G','BEE24-28'):1}
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


def pair_ids(value):
    try:a,b=map(int,value.split('|'))
    except (TypeError,ValueError):raise ValueError('canonical numeric pair A|B required') from None
    if a>=b or value!=f'{a}|{b}':raise ValueError('canonical A<B pair required')
    return a,b


def canonical_tuple(state):return tuple(sorted((pair_ids(k) for k in state['selected_pair_ids'])))


def event_key(row):return row['domain']+'/'+row['clip']+'/'+str(int(row['pid']))


def validate_events(rows):
    seen=set()
    for r in rows:
        if r['domain'] not in DOMAINS or not isinstance(r['clip'],str) or int(r['pid'])!=r['pid']:raise ValueError('source event domain/clip/integer pid required')
        key=event_key(r)
        if key in seen or r.get('event_key',key)!=key:raise ValueError('duplicate/inconsistent K event key')
        seen.add(key)
        L,X=float(r['x_L']),float(r['x_X'])
        if L not in (0.,.5,1.) or X not in (0.,.5,1.) or X>L:raise ValueError('two-peer K features must have 0<=x_X<=x_L<=1 in half steps')
    return rows


def source_weights(rows):
    if not rows:return np.empty(0,float)
    hives={r['domain'] for r in rows};clips=defaultdict(set);counts=Counter()
    for r in rows:clips[r['domain']].add(r['clip']);counts[r['domain'],r['clip']]+=1
    return np.asarray([1/(len(hives)*len(clips[r['domain']])*counts[r['domain'],r['clip']]) for r in rows],float)


def objective(a,b,x,y,w):
    residual=a+b*x-y
    return float(w@(residual**2)+RIDGE*b*b),np.array([2*(w@residual),2*(w@(residual*x))+2*RIDGE*b])


def _kkt(a,b,gradient):
    normals=[];names=[]
    if a==0:normals.append([-1.,0.]);names.append('a>=0')
    if b==0:normals.append([0.,-1.]);names.append('b>=0')
    if a+b==1:normals.append([1.,1.]);names.append('a+b<=1')
    best=(float(np.linalg.norm(gradient)),np.zeros(len(normals)))
    for bits in itertools.product((False,True),repeat=len(normals)):
        included=np.flatnonzero(bits)
        if not len(included):continue
        N=np.asarray(normals)[included].T;lam=np.linalg.lstsq(N,-gradient,rcond=None)[0]
        if (lam < -1e-10).any():continue
        all_lam=np.zeros(len(normals));all_lam[included]=np.maximum(lam,0.)
        residual=float(np.linalg.norm(gradient+np.asarray(normals).T@all_lam))
        if residual<best[0]:best=residual,all_lam
    return dict(active_constraints=names,multipliers=best[1].tolist(),stationarity_norm=best[0],feasible=bool(0<=a and 0<=b and a+b<=1))


def fit_line(x,y,w):
    x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float)
    if x.shape!=y.shape or x.shape!=w.shape or x.ndim!=1 or not np.isfinite(x).all() or not np.isfinite(w).all() or not np.isin(y,[0,1]).all() or (x<0).any() or (x>1).any() or (w<=0).any():raise ValueError('valid event features/labels/positive weights required')
    if not len(x):return dict(a=.5,b=0.,status='empty_source',objective=None,gradient=None,KKT=None)
    H=np.array([[w.sum(),w@x],[w@x,w@(x*x)+RIDGE]]);rhs=np.array([w@y,w@(x*y)])
    interior=np.linalg.solve(H,rhs);candidates=[]
    def add(tag,a,b):
        if 0<=a and 0<=b and a+b<=1:
            val,grad=objective(float(a),float(b),x,y,w);candidates.append(dict(case=tag,a=float(a),b=float(b),objective=val,gradient=grad.tolist()))
    add('interior',*interior)
    add('b=0',float(np.clip((w@y)/w.sum(),0,1)),0.)
    add('a=0',0.,float(np.clip((w@(x*y))/(w@(x*x)+RIDGE),0,1)))
    u=x-1;v=y-1;b=float(np.clip((w@(u*v))/(w@(u*u)+RIDGE),0,1));add('a+b=1',1-b,b)
    for a,b in ((0.,0.),(1.,0.),(0.,1.)):add('vertex',a,b)
    best=min(candidates,key=lambda r:(r['objective'],r['a'],r['b'],r['case']))
    return dict(best,status='fitted',ridge_slope=RIDGE,intercept_penalized=False,candidates=candidates,
                KKT=_kkt(best['a'],best['b'],np.asarray(best['gradient'])),solver='exact quadratic interior + three bounded edges + vertices')


def fit(rows,labels,*,heldout=None):
    validate_events(rows)
    if heldout is not None and any(r['domain']==heldout for r in rows):raise ValueError('heldout K event supplied to fitting')
    if set(labels)!={event_key(r) for r in rows} or any(v not in (0,1) for v in labels.values()):raise ValueError('complete binary source K deletion marginals required')
    ordered=sorted(rows,key=lambda r:(r['domain'],r['clip'],int(r['pid'])));w=source_weights(ordered)
    y=np.asarray([labels[event_key(r)] for r in ordered],float)
    mean=.5 if not len(y) else float(np.clip((w@y)/w.sum(),0,1))
    # Preserve exact constant source means, including all-zero/all-one labels.
    if len(y) and np.all(y==y[0]):mean=float(y[0])
    return dict(schema=SCHEMA,heldout=heldout,source_domains=[h for h in DOMAINS if h!=heldout],
        Q=dict(a=mean,b=0.,status='empty_source' if not len(y) else 'fitted'),
        L=fit_line([r['x_L'] for r in ordered],y,w),X=fit_line([r['x_X'] for r in ordered],y,w),
        weight_rule='nonempty source K-event hive equal; its K-event clips equal; each clip event equal; no class balancing',
        label_meaning='fixed K official.5 graph maximum-matching loss after deleting this left vertex; not a TP/FP witness label',
        source_event_counts={h:sum(r['domain']==h for r in ordered) for h in DOMAINS if h!=heldout},
        training_records=[dict(r,event_key=event_key(r),y=int(y[i]),weight=float(w[i])) for i,r in enumerate(ordered)])


def utilities(model,state,K):
    dn=float(state['nP']-K['nP']);dJ=float(state['meanJ']-K['meanJ']);dM=float(state['meanM']-K['meanM']);dC=float(state['meanCountL1']-K['meanCountL1'])
    return dict(K=0.,Q=(2*model['Q']['a']-1)*dn,
        L=(2*model['L']['a']-1)*dn+2*model['L']['b']*dJ,
        X=(2*model['X']['a']-1)*dn+2*model['X']['b']*dM,
        V=2*dM-dn,C=-dC)


def choose_state(states,values):
    """values maps state_id -> raw utility. Python round uses 10 decimals."""
    if set(values)!={s['state_id'] for s in states}:raise ValueError('utility state inventory differs')
    if any(not np.isfinite(v) for v in values.values()):raise ValueError('finite state utilities required')
    return min(states,key=lambda s:(-round(float(values[s['state_id']]),10),s['edits'],canonical_tuple(s)))


def validate_clip(clip):
    if clip['domain'] not in DOMAINS:raise ValueError('registered domain required')
    pairs=clip['active_pairs'];ids=[p['pair_id'] for p in pairs];used=set()
    if len(set(ids))!=len(ids):raise ValueError('duplicate active pair')
    for p in pairs:
        tracks=pair_ids(p['pair_id'])
        if used.intersection(tracks) or not np.isfinite(p['min_offset_norm']) or p['min_offset_norm']<0:raise ValueError('active disjoint slate/local distance required')
        used.update(tracks)
    expected={tuple(sorted(combo,key=pair_ids)) for n in range(len(ids)+1) for combo in itertools.combinations(ids,n)}
    got=set();state_ids=set();K=None
    for s in clip['states']:
        key=tuple(sorted(s['selected_pair_ids'],key=pair_ids))
        if key in got or s['state_id'] in state_ids or s['edits']!=len(key) or len(set(key))!=len(key):raise ValueError('state identity/edit count differs')
        got.add(key);state_ids.add(s['state_id'])
        if int(s['nP'])!=s['nP'] or s['nP']<0 or not 0<=s['meanM']<=s['meanJ']<=s['nP'] or s['meanCountL1']<0:raise ValueError('whole-state peer statistic bounds differ')
        if any(not np.isfinite(s[k]) or 2*s[k]!=int(2*s[k]) for k in ('meanJ','meanM','meanCountL1')):raise ValueError('two-peer means must be finite half-integers')
        if not key:K=s
    if got!=expected or K is None:raise ValueError('complete fixed-active-slate subset enumeration required')
    return K


def choose_clip(clip,model):
    K=validate_clip(clip);states=clip['states'];values={s['state_id']:utilities(model,s,K) for s in states}
    selected={'K':K};audits=[]
    for arm in ('Q','L','X','V','C'):selected[arm]=choose_state(states,{s['state_id']:values[s['state_id']][arm] for s in states})
    budget=selected['X']['edits'];local=sorted(clip['active_pairs'],key=lambda p:(p['min_offset_norm'],pair_ids(p['pair_id'])))
    z_ids=tuple(sorted((p['pair_id'] for p in local[:budget]),key=pair_ids))
    selected['Z']=next(s for s in states if tuple(sorted(s['selected_pair_ids'],key=pair_ids))==z_ids)
    for s in states:audits.append(dict(s,delta_nP=s['nP']-K['nP'],delta_meanJ=s['meanJ']-K['meanJ'],delta_meanM=s['meanM']-K['meanM'],
        delta_meanCountL1=s['meanCountL1']-K['meanCountL1'],utilities=values[s['state_id']],rounded_utilities={k:round(float(v),10) for k,v in values[s['state_id']].items()}))
    return dict(domain=clip['domain'],clip=clip['clip'],active_pairs=clip['active_pairs'],states=audits,
        selected={arm:dict(state_id=s['state_id'],selected_pair_ids=sorted(s['selected_pair_ids'],key=pair_ids),edits=s['edits']) for arm,s in selected.items()},
        Z_budget=budget,tie_rule='Python round(raw utility,10) max; fewer edits; sorted numeric canonical pair tuples lex ascending',
        surrogate_limit='fixed-graph single-event criticalities are not additive true TP labels; complete state peer J/M used for utility')


def train_and_seal(clips,events,label_provider,outdir,input_receipt):
    validate_events(events)
    if sha(__file__)!=CODE_SHA:raise ValueError('loaded source changed')
    if any('y' in e or 'labels' in e for e in events):raise ValueError('unlabelled deployment event features required')
    for c in clips:validate_clip(c)
    if len({(c['domain'],c['clip']) for c in clips})!=len(clips):raise ValueError('duplicate deployment clip')
    dest=Path(outdir)
    if dest.exists():raise FileExistsError('immutable output exists')
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        predictions={};folds={}
        for h in DOMAINS:
            supplied=label_provider(h)
            if set(supplied['source_domains'])!=set(DOMAINS)-{h}:raise ValueError('explicit other-three source whitelist required')
            source=[e for e in events if e['domain']!=h];model=fit(source,supplied['labels'],heldout=h);model['source_receipts']=supplied.get('receipts',{})
            name='models/'+h+'.json';save(tmp/name,model);folds[h]=dict(heldout=h,source_domains=[d for d in DOMAINS if d!=h],model_file=name)
            for c in clips:
                if c['domain']==h:predictions[h+'/'+c['clip']]=choose_clip(c,model)
        if sha(__file__)!=CODE_SHA:raise ValueError('source changed during fitting')
        save(tmp/'predictions.json',dict(schema=SCHEMA,arms=list(ARMS),folds=folds,clips=predictions))
        save(tmp/'receipt.json',dict(input_receipt,schema=SCHEMA,code_sha256=CODE_SHA,numpy_version=np.__version__,all_target_true_outcomes_prohibited_before_common_seal=True))
        save(tmp/'manifest.json',{p.relative_to(tmp).as_posix():sha(p) for p in tmp.rglob('*') if p.is_file()});digest=sha(tmp/'manifest.json')
        (tmp/'SEALED.sha256').write_text(digest+'\n');tmp.rename(dest)
        return dict(root=str(dest),manifest_sha256=digest)
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def verify_seal(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('prediction top seal differs')
    files=json.loads((root/'manifest.json').read_text())
    if set(files)!={'predictions.json','receipt.json'}|{'models/'+h+'.json' for h in DOMAINS}:raise ValueError('all four models/common choices required')
    for name,digest in files.items():
        if sha(root/name)!=digest:raise ValueError('sealed output changed')
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt['code_sha256']!=sha(__file__):raise ValueError('frozen fit source changed')
    predictions=json.loads((root/'predictions.json').read_text())
    if set(predictions['folds'])!=set(DOMAINS):raise ValueError('all folds required')
    for h,f in predictions['folds'].items():
        if f['heldout']!=h or set(f['source_domains'])!=set(DOMAINS)-{h}:raise ValueError('source boundary differs')
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
    if name not in receipt['files'] or sha(Path(root)/name)!=receipt['files'][name]:raise ValueError('sealed input changed: '+name)
    return Path(root)/name


def _read_gz(path):
    with gzip.open(path,'rt',encoding='utf8') as stream:return json.load(stream)


def load_numeric(frozen,fr,source210,sr,prereg):
    """Read peer-only numbers; neither native MOT files nor true outcomes."""
    if sr['manifest_sha256']!=SOURCE210:raise ValueError('wrong registered210 parent')
    top=json.loads(checked(frozen,fr,'deployment.json').read_text())
    parent=json.loads(checked(source210,sr,'deployment.json').read_text())
    if top['schema']!='match211-freeze-v1' or top['prereg']!=prereg or top['source210']!=SOURCE210 or top['source208']!=SOURCE208 or top['GT_access'] is not False:raise ValueError('peer-only lineage differs')
    if len(top['clips'])!=6 or {(r['domain'],r['clip']) for r in top['clips']}!=set(EXPECTED_STATES) or top['states']!=22:raise ValueError('fixed sixclip22 state roster required')
    refs={(r['domain'],r['clip']):r for r in parent['clips']};clips=[];events=[]
    for ref in top['clips']:
        h,name=ref['domain'],ref['clip'];path=h+'/'+name;old=refs[h,name]
        if ref['path']!=path or sha(checked(frozen,fr,path+'/manifest.json'))!=ref['seal']:raise ValueError('frozen clip path/seal differs')
        data=_read_gz(checked(frozen,fr,path+'/states.json.gz'))
        original=_read_gz(checked(source210,sr,old['path']+'/catalog.json.gz'))
        if data['schema']!='match211-clip-v1' or data['metadata']['domain']!=h or data['metadata']['clip']!=name:raise ValueError('peer state clip scope differs')
        if data['reference_type']!='other two original prediction members, not actual GT':raise ValueError('true outcomes cannot enter deployment input')
        active=data['active_pairs'];old_byid={p['pair_id']:p for p in original['pairs']}
        # The primary22 lineage is disjoint. Joint cross-family conflicts have
        # their own registration/decoder and must not silently change this one.
        if {p['pair_id'] for p in active}!=set(original['active_slate_ids']):raise ValueError('lineage error: all active family != original210 active slate')
        active_rows=[]
        for p in active:
            if not p['active'] or p['min_offset_norm']!=old_byid[p['pair_id']]['min_offset_norm']:raise ValueError('active/local distance lineage differs')
            active_rows.append(dict(pair_id=p['pair_id'],min_offset_norm=p['min_offset_norm']))
        states=[];rawK=None
        for s in data['states']:
            graphs=s['peer_graphs']
            if len(graphs)!=2 or any(g['reference_is_prediction_not_true_GT'] is not True for g in graphs):raise ValueError('two prediction references required')
            if any(g['nPred']!=s['nP'] for g in graphs) or s['meanJ']!=sum(g['J'] for g in graphs)/2 or s['meanM']!=sum(g['R'] for g in graphs)/2:raise ValueError('whole peer J/M statistics differ')
            distance=[]
            for g in graphs:
                pred,reference=np.asarray(g['pred_windows'],np.int64),np.asarray(g['reference_windows'],np.int64)
                if pred.shape!=reference.shape:raise ValueError('peer count vector clocks differ')
                distance.append(int(np.abs(pred-reference).sum()))
            if graphs[0]['pred_windows']!=graphs[1]['pred_windows'] or s['meanCountL1']!=sum(distance)/2:raise ValueError('raw30phase peer count distance differs')
            states.append({k:s[k] for k in ('state_id','selected_pair_ids','edits','nP','meanJ','meanM','meanCountL1')})
            if s['is_K']:
                if rawK is not None or s['selected_pair_ids']:raise ValueError('unique empty K state required')
                rawK=s
        clip=dict(domain=h,clip=name,active_pairs=active_rows,states=states);K=validate_clip(clip)
        if len(states)!=EXPECTED_STATES[h,name] or rawK is None or K['state_id']!=rawK['state_id']:raise ValueError('registered state enumeration differs')
        tables=[{int(f['canonical_id']):f for f in g['flags']} for g in rawK['peer_graphs']]
        features=data['K_event_features']
        if len(features)!=K['nP'] or set(tables[0])!=set(tables[1]) or {int(f['canonical_id']) for f in features}!=set(tables[0]):raise ValueError('complete K event feature inventory required')
        for f in features:
            pid=int(f['canonical_id']);L=sum(int(t[pid]['has_neighbor']) for t in tables)/2;X=sum(int(t[pid]['drop_left_marginal']) for t in tables)/2
            if f['x_L']!=L or f['x_X']!=X:raise ValueError('peer K event features differ from fixed graph flags')
            events.append(dict(domain=h,clip=name,pid=pid,event_key=h+'/'+name+'/'+str(pid),x_L=L,x_X=X,
                frozen_event_key=f['event_key'],signed_frame=f['signed_frame'],peercoverage=[dict(peer_index=i,nReference=g['nReference'],
                    has_neighbor=bool(tables[i][pid]['has_neighbor']),drop_left_marginal=int(tables[i][pid]['drop_left_marginal'])) for i,g in enumerate(rawK['peer_graphs'])]))
        clips.append(clip);del data,original,rawK
    validate_events(events)
    if sum(len(c['states']) for c in clips)!=22:raise ValueError('primary22 states required')
    return clips,events


def load_source_labels(root,heldout,events,*,requested_domains,receipt,prereg,frozen_sha):
    allowed=tuple(requested_domains)
    if heldout not in DOMAINS or len(allowed)!=3 or set(allowed)!=set(DOMAINS)-{heldout}:raise ValueError('explicit other-three source domains required')
    expected={event_key(r) for r in events if r['domain'] in allowed};labels={};proof={}
    for h in allowed:
        data=json.loads(checked(root,receipt,h+'.json').read_text())
        if data['domain']!=h or data['prereg']!=prereg or data['frozen_manifest_sha256']!=frozen_sha:raise ValueError('source label scope/lineage differs')
        for r in data['records']:
            if r['domain']!=h or r['event_key']!=event_key(r):raise ValueError('source event join key differs')
            key=event_key(r)
            if key not in expected or key in labels or r['y'] not in (0,1):raise ValueError('duplicate/invalid K event necessity label')
            labels[key]=int(r['y'])
        proof[h]=dict(file=h+'.json',sha256=receipt['files'][h+'.json'],events=len(data['records']))
        del data
    if set(labels)!=expected:raise ValueError('complete source K event labels required, including0')
    return dict(source_domains=list(allowed),labels=labels,receipts=dict(label_manifest_sha256=receipt['manifest_sha256'],source_shards=proof,
        target_access='no heldout source shard, global label deployment, target W/F1, assessment or true-GT graph'))


def field_train(args):
    if args.prereg!='0e005ffeee692b104a4a6430843c0af721b6bfa1':raise ValueError('registered primary211 prereg required')
    fr,lr,sr=manifest(args.frozen),manifest(args.labels),manifest(args.source210)
    if fr['manifest_sha256']!=args.frozen_sha or lr['manifest_sha256']!=args.label_sha:raise ValueError('GO input seals differ')
    clips,events=load_numeric(args.frozen,fr,args.source210,sr,args.prereg)
    def provider(h):
        supplied=load_source_labels(args.labels,h,events,requested_domains=[d for d in DOMAINS if d!=h],receipt=lr,prereg=args.prereg,frozen_sha=fr['manifest_sha256'])
        print('SOURCE211',h,'Kevents',len(supplied['labels']),flush=True);return supplied
    return train_and_seal(clips,events,provider,args.out,dict(prereg=args.prereg,frozen_manifest_sha256=fr['manifest_sha256'],
        label_manifest_sha256=lr['manifest_sha256'],source210_manifest_sha256=sr['manifest_sha256'],input208_manifest_sha256=SOURCE208))


def self_test():
    rng=np.random.default_rng(21101)
    rows=[];labels={}
    for h in DOMAINS:
        for clip,n in ((h+'A',3),(h+'B',7)):
            for pid in range(n):
                x=float((pid%3)/2);r=dict(domain=h,clip=clip,pid=pid,x_L=x,x_X=x if pid%2 else 0.,peercoverage='two full K-reference graphs')
                rows.append(r);labels[event_key(r)]=int(pid%2)
    src=[r for r in rows if r['domain']!='35'];model=fit(src,{event_key(r):labels[event_key(r)] for r in src},heldout='35')
    for h in ('33','W','G'):
        records=[r for r in model['training_records'] if r['domain']==h]
        assert np.isclose(sum(r['weight'] for r in records),1/3)
        assert all(np.isclose(sum(r['weight'] for r in records if r['clip']==c),1/6) for c in (h+'A',h+'B'))
    for head in ('L','X'):
        line=model[head];assert line['KKT']['feasible'] and line['KKT']['stationarity_norm']<1e-8
        for a in np.linspace(0,1,31):
            for b in np.linspace(0,1-a,31):
                r=model['training_records'];x=np.asarray([e['x_'+head] for e in r]);y=np.asarray([e['y'] for e in r]);w=np.asarray([e['weight'] for e in r])
                assert line['objective']<=objective(a,b,x,y,w)[0]+1e-12
    empty=fit([],{},heldout='35');assert all(empty[h]['a']==.5 and empty[h]['b']==0 for h in ('Q','L','X'))
    negative=fit_line([0.,1.],[1.,0.],[.5,.5]);assert negative['b']==0 and negative['a']==.5
    allone=fit(src,{event_key(r):1 for r in src},heldout='35');assert allone['Q']['a']==1.
    active=[dict(pair_id='2|3',min_offset_norm=.4),dict(pair_id='10|11',min_offset_norm=.1)]
    states=[dict(state_id='K',selected_pair_ids=[],edits=0,nP=4,meanJ=3.,meanM=2.,meanCountL1=20.),
            dict(state_id='A',selected_pair_ids=['2|3'],edits=1,nP=3,meanJ=3.,meanM=2.,meanCountL1=18.),
            dict(state_id='B',selected_pair_ids=['10|11'],edits=1,nP=3,meanJ=2.,meanM=2.,meanCountL1=21.),
            dict(state_id='AB',selected_pair_ids=['2|3','10|11'],edits=2,nP=2,meanJ=2.,meanM=1.,meanCountL1=19.)]
    clip=dict(domain='35',clip='35clip',active_pairs=active,states=states)
    neutral=choose_clip(clip,empty);assert neutral['selected']['X']['state_id']=='K' and neutral['Z_budget']==0
    coef=dict(Q=dict(a=.2,b=0.),L=dict(a=.2,b=.5),X=dict(a=.2,b=.5))
    chosen=choose_clip(clip,coef);assert chosen['selected']['X']['edits']==chosen['selected']['Z']['edits']
    assert chosen['selected']['Z']['selected_pair_ids']==['10|11']
    assert choose_state(states,dict(K=0.,A=1e-12,B=0.,AB=0.))['state_id']=='K'
    assert choose_state(states,dict(K=-1.,A=1.,B=1.,AB=1.))['state_id']=='A'  # Numeric IDs, not lexical strings.
    reordered=dict(clip,states=list(reversed(states)));assert choose_clip(reordered,coef)['selected']==chosen['selected']
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);labelroot=tmp/'labels';labelroot.mkdir()
        for h in DOMAINS:
            records=[dict(domain=r['domain'],clip=r['clip'],pid=r['pid'],event_key=event_key(r),y=labels[event_key(r)]) for r in rows if r['domain']==h]
            save(labelroot/(h+'.json'),dict(domain=h,prereg='synthetic',frozen_manifest_sha256='frozen',records=records))
        save(labelroot/'manifest.json',{p.name:sha(p) for p in labelroot.iterdir()});(labelroot/'SEALED.sha256').write_text(sha(labelroot/'manifest.json'))
        lr=manifest(labelroot)
        for h in DOMAINS:
            path=labelroot/(h+'.json');raw=path.read_bytes();path.unlink()
            supplied=load_source_labels(labelroot,h,rows,requested_domains=[d for d in DOMAINS if d!=h],receipt=lr,prereg='synthetic',frozen_sha='frozen')
            assert len(supplied['labels'])==30
            path.write_bytes(raw)
        try:load_source_labels(Path('must-not-open'),'35',rows,requested_domains=['35','W','G'],receipt={},prereg='synthetic',frozen_sha='frozen')
        except ValueError:pass
        else:raise AssertionError('heldout whitelist accepted')
        clips=[dict(clip,domain=h,clip=h+'clip') for h in DOMAINS]
        def provider(h):return dict(source_domains=[d for d in DOMAINS if d!=h],labels={event_key(r):labels[event_key(r)] for r in rows if r['domain']!=h})
        sealed=train_and_seal(clips,rows,provider,tmp/'pred',dict(prereg='synthetic'))
        assert len(load_predictions(sealed['root'])['decisions'])==4
    return dict(status='PASS',seed=21101,tests='hierarchical events; exact constrained LS vs dense feasible grid/KKT; empty; negative slope boundary; constantQ; complete states; round10 abstain; numeric tie; whole graph delta utility; Zmatchedbudget; all4heldout shard absent; scope before IO; common fourfold seal')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--run-field',action='store_true')
    parser.add_argument('--frozen',type=Path);parser.add_argument('--labels',type=Path);parser.add_argument('--source210',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--frozen-sha');parser.add_argument('--label-sha');parser.add_argument('--prereg');args=parser.parse_args()
    if args.run_field:
        if any(v is None for v in (args.frozen,args.labels,args.source210,args.out,args.frozen_sha,args.label_sha)):parser.error('all input/output roots and pinned seals required')
        print(json.dumps(field_train(args),indent=2))
    elif args.self_test:print(json.dumps(self_test(),indent=2))
    else:parser.error('--self-test or --run-field required')
