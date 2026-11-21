"""Source-only peer-suffix conditional models; synthetic CLI until GO.

Contexts: host_key,hive,group,clip,unit,host_member,stratum(5bits),
baseline_outcome (all official bin/direction integer counts), options.
Options: action_id,features10,outcome,suffix_source_keys,resources. Null is
always a separate zero-logit option with baseline_outcome. Supervision is
supplied separately; deployment accepts no labels or targets.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import Counter,defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import logsumexp

DOMAINS=('35','33','W','G')
ARMS=('K','G','I','L','T','M')
RIDGE=.20
SCHEMA='peer-suffix203-model-v1'


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
    Path(path).write_text(json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')


def _token(value):return json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False)


def _outcome(value):
    a=np.asarray(value)
    if a.ndim<1 or not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all() or (a!=np.floor(a)).any() or (a<0).any():
        raise ValueError('outcome requires finite nonnegative integer official counts')
    return tuple(int(v) for v in a.ravel())


def _stratum(value):
    if isinstance(value,str) and len(value)==5 and set(value)<={'0','1'}:return value
    if isinstance(value,(list,tuple)) and len(value)==5 and all(v in (0,1) for v in value):return ''.join(str(int(v)) for v in value)
    raise ValueError('fixed five-bit stratum required')


def deduplicate(contexts):
    """One logit per physical suffix path; ID renamings cannot gain mass."""
    hosts=set();action_ids=set();result=[];audit=[]
    for c in sorted(contexts,key=lambda r:r['host_key']):
        if c['host_key'] in hosts:raise ValueError('duplicate host context')
        hosts.add(c['host_key'])
        if c['hive'] not in DOMAINS:raise ValueError('unregistered hive')
        for name in ('group','clip','unit','host_member'):
            if name not in c:raise ValueError('missing context metadata '+name)
        _stratum(c['stratum']); baseline=_outcome(c['baseline_outcome']);paths={};rows=[]
        for r in sorted(c['options'],key=lambda r:r['action_id']):
            if not isinstance(r['action_id'],str) or not r['action_id'] or r['action_id'] in action_ids:raise ValueError('duplicate/empty action id')
            action_ids.add(r['action_id'])
            x=np.asarray(r['features10'],float)
            if x.shape!=(10,) or not np.isfinite(x).all():raise ValueError('fixed finite features10 required')
            if len(_outcome(r['outcome']))!=len(baseline):raise ValueError('outcome clock/dimension mismatch')
            keys=r['suffix_source_keys']
            if not keys or len({_token(k) for k in keys})!=len(keys):raise ValueError('nonempty unique physical source path required')
            path=_token(keys)
            if path in paths:
                if _outcome(r['outcome'])!=_outcome(paths[path]['outcome']):raise ValueError('identical physical path has different outcome')
                audit.append(dict(host_key=c['host_key'],dropped=r['action_id'],canonical=paths[path]['action_id']))
                continue
            paths[path]=r;rows.append(r)
        result.append(dict(c,options=rows))
    return result,audit


def _hierarchy(rows):
    """Hive/group/clip/observed-kind/context equal weights, O(n)."""
    hs=set();gs=defaultdict(set);cs=defaultdict(set);ks=defaultdict(set);n=Counter()
    for r in rows:
        h,g,v,k=r['hive'],str(r['group']),r['clip'],r['kind']
        hs.add(h);gs[h].add(g);cs[h,g].add(v);ks[h,g,v].add(k);n[h,g,v,k]+=1
    return [1/(len(hs)*len(gs[h])*len(cs[h,g])*len(ks[h,g,v])*n[h,g,v,k])
            for h,g,v,k in ((r['hive'],str(r['group']),r['clip'],r['kind']) for r in rows)]


def make_training(contexts,supervision,identity=False):
    """Never silently replace an unattainable target with its closest option."""
    contexts,aliases=deduplicate(contexts); rows=[];coverage=[]
    if set(supervision)!={c['host_key'] for c in contexts}:raise ValueError('source supervision inventory differs')
    for c in contexts:
        s=supervision[c['host_key']];options=c['options']
        if not options:continue
        if identity:
            labels=s['classes']
            if any(r['action_id'] not in labels for r in options):raise ValueError('identity label missing')
            classes=[labels[r['action_id']] for r in options]
            if any(k not in ('S','D','U') for k in classes):raise ValueError('invalid S/D/U class')
            nS=classes.count('S')
            if not nS and 'U' in classes:continue
            chosen=[r for r,k in zip(options,classes) if k!='U']
            if not chosen:continue
            targets=[(1/nS if labels[r['action_id']]=='S' else 0.) for r in chosen]+[float(nS==0)]
            row=dict(c,options=chosen,kind='positive' if nS else 'known-null',loss='crossentropy',target=targets)
        else:
            if not s['valid']:
                coverage.append(dict(host_key=c['host_key'],reason='invalid_owner_supervision'));continue
            truth=_outcome(s['target_outcome']);base=_outcome(c['baseline_outcome'])
            if len(truth)!=len(base):raise ValueError('supervision outcome clock differs')
            compatible=[_outcome(r['outcome'])==truth for r in options]+[base==truth]
            if not any(compatible):
                coverage.append(dict(host_key=c['host_key'],reason='no_compatible_choice',target_outcome=list(truth)));continue
            row=dict(c,kind='K-correct' if base==truth else 'needs-change',loss='compatible',compatible=compatible)
        rows.append(row)
    return rows,dict(coverage_gaps=coverage,physical_aliases=aliases,total_hosts=len(contexts),trainable_hosts=len(rows))


def training_weights(natural,artificial,distribution):
    """Equal hive/group/clip; split types WITHIN each clip, not globally."""
    raw=Counter((c['hive'],str(c['group']),c['clip'],_stratum(c['stratum'])) for c in distribution if c['options'])
    kept=[r for r in artificial if raw[r['hive'],str(r['group']),r['clip'],_stratum(r['stratum'])]]
    natural_by=defaultdict(list);artificial_by=defaultdict(list)
    for r in natural:natural_by[r['hive'],str(r['group']),r['clip']].append(r)
    for r in kept:artificial_by[r['hive'],str(r['group']),r['clip']].append(r)
    clips=set(natural_by)|set(artificial_by);hs={k[0] for k in clips};gs=defaultdict(set);vs=defaultdict(set)
    for h,g,v in clips:gs[h].add(g);vs[h,g].add(v)
    clip_audit=[]
    for key in sorted(clips):
        h,g,v=key;base=1/(len(hs)*len(gs[h])*len(vs[h,g]));nr=natural_by[key];ar=artificial_by[key]
        type_share=.5 if nr and ar else 1.
        ntypes={r['kind'] for r in nr};nn=Counter(r['kind'] for r in nr)
        for r in nr:r.update(weight=base*type_share/len(ntypes)/nn[r['kind']],source_type='natural')
        strata={_stratum(r['stratum']) for r in ar};mass=sum(raw[*key,s] for s in strata)
        kinds=defaultdict(set);n=Counter()
        for r in ar:
            s=_stratum(r['stratum']);kinds[s].add(r['kind']);n[s,r['kind']]+=1
        for r in ar:
            s=_stratum(r['stratum']);r.update(weight=base*type_share*(raw[*key,s]/mass)/len(kinds[s])/n[s,r['kind']],source_type='artificial')
        clip_audit.append(dict(hive=h,group=g,clip=v,clip_weight=base,natural_mass=sum(r['weight'] for r in nr),artificial_mass=sum(r['weight'] for r in ar)))
    bynatural=defaultdict(Counter);uncovered=[]
    for (h,g,v,s),count in raw.items():bynatural[h,g,v][s]=count
    for key,counts in sorted(bynatural.items()):
        represented={_stratum(r['stratum']) for r in artificial_by.get(key,[])}
        missing=sum(n for s,n in counts.items() if s not in represented)
        uncovered.append(dict(hive=key[0],group=key[1],clip=key[2],natural_hosts=sum(counts.values()),
                              uncovered_hosts=missing,uncovered_fraction=missing/sum(counts.values()),common_strata=sorted(represented)))
    rows=natural+kept
    if rows and not np.isclose(sum(r['weight'] for r in rows),1.,atol=1e-14):raise AssertionError('source weight total differs')
    return rows,dict(artificial_strata_coverage=uncovered,clip_type_weights=clip_audit,natural_rows=len(natural),artificial_rows=len(kept),
                            natural_mass=sum(r['weight'] for r in natural),artificial_mass=sum(r['weight'] for r in kept))


def _loss(theta,batches):
    value=RIDGE/2*float(theta@theta);grad=RIDGE*theta
    for a,r in batches:
        logits=a@theta;lz=logsumexp(logits);p=np.exp(logits-lz)
        if r['loss']=='compatible':
            mask=np.asarray(r['compatible'],bool);good=logsumexp(logits[mask]);target=np.zeros(len(logits));target[mask]=np.exp(logits[mask]-good)
            term=lz-good
        else:
            target=np.asarray(r['target']);term=lz-target@logits
        value+=r['weight']*term;grad+=r['weight']*a.T@(p-target)
    return value,grad


def _pack(batches):
    grouped=defaultdict(list)
    for a,r in batches:grouped[r['loss'],len(a)].append((a,r))
    return [(kind,np.stack([a for a,r in rows]),np.asarray([r['weight'] for a,r in rows]),
             np.asarray([r['compatible'] if kind=='compatible' else r['target'] for a,r in rows]))
            for (kind,n),rows in sorted(grouped.items())]


def _packed_loss(theta,packed):
    """At most two nonnull peers: batch the few fixed choice-set shapes."""
    value=RIDGE/2*float(theta@theta);grad=RIDGE*theta
    for kind,a,w,target in packed:
        z=a@theta;normalizer=logsumexp(z,axis=1);p=np.exp(z-normalizer[:,None])
        if kind=='compatible':
            compatible_logits=np.where(target,z,-np.inf)
            good=logsumexp(compatible_logits,axis=1)
            target=np.exp(compatible_logits-good[:,None])
            terms=normalizer-good
        else:terms=normalizer-np.sum(target*z,axis=1)
        value+=float(w@terms);grad+=np.einsum('n,nij,ni->j',w,a,p-target)
    return value,grad


def fit(contexts,supervision,mode='L',*,artificial=(),artificial_supervision=None,heldout=None):
    if mode not in ('I','L','T'):raise ValueError('unknown model')
    if heldout is not None and any(c['hive']==heldout for c in [*contexts,*artificial]):raise ValueError('heldout source context')
    natural,coverage=make_training(contexts,supervision,identity=mode=='I')
    if mode=='T':
        ar,acoverage=make_training(artificial,artificial_supervision or {})
    else:ar=[];acoverage={}
    rows,weight_audit=training_weights(natural,ar,contexts)
    result=dict(schema=SCHEMA,mode=mode,heldout=heldout,ridge=RIDGE,status='empty',mean=None,scale=None,beta=None,intercept=None,
        source_hives=sorted({r['hive'] for r in rows}),training_records=plain(rows),coverage=coverage,artificial_coverage=acoverage,weight_audit=weight_audit)
    if not rows:
        result['optimization']=dict(success=True,message='empty source',gradient=[],iterations=0,objective=0.);return result
    X=[];weights=[]
    for r in rows:
        for o in r['options']:X.append(o['features10']);weights.append(r['weight']/len(r['options']))
    X=np.asarray(X,float);w=np.asarray(weights);w/=w.sum()
    mean=w@X;scale=np.sqrt(w@((X-mean)**2));scale[scale==0]=1.
    batches=[]
    for r in rows:
        a=np.zeros((len(r['options'])+1,11));a[:-1,:10]=(np.asarray([o['features10'] for o in r['options']])-mean)/scale;a[:-1,10]=1.
        batches.append((a,r))
    packed=_pack(batches)
    opt=minimize(lambda t:_packed_loss(t,packed),np.zeros(11),jac=True,method='L-BFGS-B',options=dict(gtol=1e-10,ftol=1e-14,maxiter=2000,maxls=50))
    if not opt.success or not np.isfinite(opt.x).all():raise RuntimeError('fit failed: '+str(opt.message))
    result.update(status='fitted',mean=mean.tolist(),scale=scale.tolist(),beta=opt.x[:10].tolist(),intercept=float(opt.x[10]),
        optimization=dict(success=True,message=str(opt.message),iterations=int(opt.nit),objective=float(opt.fun),gradient=_packed_loss(opt.x,packed)[1].tolist(), initialization=[0.]*11, solver='scipy.optimize.minimize L-BFGS-B', options=dict(gtol=1e-10,ftol=1e-14,maxiter=2000,maxls=50), claim='local numerical convergence; compatible-set loss need not be convex'))
    return result


def logits(model,contexts):
    contexts,_=deduplicate(contexts)
    return {o['action_id']:(None if model['status']=='empty' else float(((np.asarray(o['features10'])-model['mean'])/model['scale'])@model['beta']+model['intercept']))
            for c in contexts for o in c['options']}


def permute(contexts,scores):
    output={};audit=[]
    for c in sorted(contexts,key=lambda c:c['host_key']):
        original=sorted(c['options'],key=lambda o:o['action_id'])
        destination=sorted(original,key=lambda o:(hashlib.sha256(('peer-suffix-rank-null-v1|'+o['action_id']).encode()).hexdigest(),o['action_id']))
        for src,dst in zip(original,destination):
            output[dst['action_id']]=scores[src['action_id']]
            audit.append(dict(host_key=c['host_key'],source=src['action_id'],destination=dst['action_id'],score=scores[src['action_id']]))
    return output,audit


def outcome_choices(contexts,scores):
    """Aggregate all finite path logits by exact outcome; K includes null0."""
    contexts,aliases=deduplicate(contexts);proposals=[];audits=[]
    expected={o['action_id'] for c in contexts for o in c['options']}
    if set(scores)!=expected:raise ValueError('logit inventory differs')
    for c in contexts:
        groups=defaultdict(list);base=_outcome(c['baseline_outcome']);groups[base].append(0.)
        for o in c['options']:
            z=scores[o['action_id']]
            if z is None:continue
            if not np.isfinite(z):raise ValueError('nonfinite logit')
            groups[_outcome(o['outcome'])].append(float(z))
        keep=float(logsumexp(groups[base]));margins={k:float(logsumexp(v)-keep) for k,v in groups.items() if k!=base}
        audits.append(dict(host_key=c['host_key'],keep_logmass=keep,outcomes=[dict(outcome=list(k),logmass=float(logsumexp(v)),margin=0. if k==base else margins[k]) for k,v in sorted(groups.items())]))
        for o in c['options']:
            key=_outcome(o['outcome']);margin=margins.get(key);z=scores[o['action_id']]
            if margin is not None and margin>0 and z is not None:
                resources={_token(v) for v in o.get('resources',[])}
                resources.add(_token(['host',c['host_key']]))
                resources.update(_token(['source',v]) for v in o['suffix_source_keys'])
                proposals.append(dict(action_id=o['action_id'],host_key=c['host_key'],unit=c['unit'],host_member=str(c['host_member']),
                    margin=margin,logit=float(z),resources=sorted(resources)))
    proposals.sort(key=lambda r:(-r['margin'],-r['logit'],r['action_id']))
    used=defaultdict(set);selected=[];decisions=[]
    for p in proposals:
        clashes=sorted(used[p['unit']]&set(p['resources']))
        if not clashes:selected.append(p['action_id']);used[p['unit']].update(p['resources'])
        decisions.append(dict(p,selected=not clashes,conflicts=clashes))
    return dict(selected=sorted(selected),host_audits=audits,greedy_audit=decisions,physical_aliases=aliases)


def deploy(contexts,models,skeleton=()):
    contexts,aliases=deduplicate(contexts)
    scores={arm:logits(models[arm],contexts) for arm in ('I','L','T')}
    scores['G']={o['action_id']:2*o['min_anchor_iou']-o['first_extrapolation_error'] for c in contexts for o in c['options']}
    scores['M'],permutation=permute(contexts,scores['T'])
    outcomes={arm:outcome_choices(contexts,scores[arm]) for arm in ('G','I','L','T','M')}
    members={(m['unit'],str(m['host_member'])):dict(m) for m in skeleton}
    for c in contexts:members.setdefault((c['unit'],str(c['host_member'])),{k:c[k] for k in ('unit','host_member','hive','clip')})
    output=[]
    for (unit,member),meta in sorted(members.items()):
        ids={o['action_id'] for c in contexts if c['unit']==unit and str(c['host_member'])==member for o in c['options']}
        output.append(dict(meta,selected={arm:dict(candidate_ids=[] if arm=='K' else [k for k in outcomes[arm]['selected'] if k in ids]) for arm in ARMS},
            scores={arm:{k:v for k,v in sc.items() if k in ids} for arm,sc in scores.items()}))
    return dict(member_clips=output,contexts=plain(contexts),outcomes=outcomes,M_permutation=permutation,physical_aliases=aliases)


def train_and_seal(contexts,source_provider,outdir,input_receipt,skeleton=()):
    contexts,_=deduplicate(contexts)
    if any(any(k in c for k in ('labels','target_outcome','supervision')) for c in contexts):raise ValueError('deployment inventory must be unlabelled')
    dest=Path(outdir)
    if dest.exists():raise FileExistsError('immutable output already exists')
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        for h in DOMAINS:
            supplied=source_provider(h)
            if set(supplied['source_hives'])!=set(DOMAINS)-{h}:raise ValueError('other-three source whitelist differs')
            src=[c for c in contexts if c['hive']!=h]
            if any(c['hive']==h for c in supplied.get('artificial',[])):raise ValueError('target artificial source')
            models={mode:fit(src,supplied['supervision'],mode,artificial=supplied.get('artificial',[]),artificial_supervision=supplied.get('artificial_supervision'),heldout=h) for mode in ('I','L','T')}
            save(tmp/f'{h}.model.json',dict(heldout=h,source_domains=[d for d in DOMAINS if d!=h],models=models,source_receipts=supplied.get('receipts',{})))
            save(tmp/f'{h}.masks.json',dict(heldout=h,**deploy([c for c in contexts if c['hive']==h],models,[m for m in skeleton if m['hive']==h])))
        save(tmp/'receipt.json',dict(input_receipt,schema=SCHEMA,model_code_sha256=sha(__file__),folds=list(DOMAINS),arms=list(ARMS),numpy_version=np.__version__,scipy_version=scipy.__version__))
        save(tmp/'manifest.json',{p.name:sha(p) for p in tmp.iterdir()});digest=sha(tmp/'manifest.json');(tmp/'SEALED.sha256').write_text(digest+'\n');tmp.rename(dest)
        return dict(root=str(dest),manifest_sha256=digest)
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def verify_seal(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('prediction seal differs')
    manifest=json.loads((root/'manifest.json').read_text())
    if set(manifest)!={'receipt.json'}|{h+'.'+k+'.json' for h in DOMAINS for k in ('model','masks')}:raise ValueError('all four folds required')
    for name,digest in manifest.items():
        if sha(root/name)!=digest:raise ValueError('sealed prediction changed')
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt['model_code_sha256']!=sha(__file__):raise ValueError('code changed after seal')
    return receipt


def load_predictions(root):
    receipt=verify_seal(root);folds={};decisions={}
    for h in DOMAINS:
        model=json.loads((Path(root)/(h+'.model.json')).read_text())
        if model['heldout']!=h or set(model['source_domains'])!=set(DOMAINS)-{h}:raise ValueError('fold boundary differs')
        f=json.loads((Path(root)/(h+'.masks.json')).read_text())
        if f['heldout']!=h:raise ValueError('mask boundary differs')
        folds[h]=f
        for m in f['member_clips']:
            bymember=decisions.setdefault(m['unit'],{});key=str(m['host_member'])
            if key in bymember:raise ValueError('duplicate member')
            bymember[key]=m['selected']
    return dict(receipt=receipt,folds=folds,decisions=decisions)


def manifest(root):
    root=Path(root);digest=sha(root/'manifest.json')
    if (root/'SEALED.sha256').read_text().strip()!=digest:raise ValueError('input top seal differs')
    files=json.loads((root/'manifest.json').read_text())
    for name in files:
        if not (root/name).resolve().is_relative_to(root.resolve()):raise ValueError('input path escape')
    return dict(manifest_sha256=digest,files=files)


def read_sealed(root,receipt,name):
    if name not in receipt['files'] or sha(Path(root)/name)!=receipt['files'][name]:raise ValueError('sealed input changed: '+name)
    return json.loads((Path(root)/name).read_text())


def _project_options(candidates,resources):
    inverse=defaultdict(list)
    for r in resources:
        if r['capacity']!=1:raise ValueError('suffix resources must have capacity1')
        for key in r['candidate_ids']:inverse[key].append(r['resource'])
    projected=[]
    for c in candidates:
        f=c['features10'];given=float(c['G_logit'])
        if given!=2*f[0]-f[2]:raise ValueError('G logit/feature mismatch')
        projected.append(dict(action_id=c['action_id'],features10=f,outcome=c['outcome_windows'],
            suffix_source_keys=c['suffix_source_keys'],resources=inverse[c['action_id']],
            min_anchor_iou=f[0],first_extrapolation_error=f[2],G_logit=given))
    return projected


def natural_catalogue(root,receipt,prereg):
    """Project one natural group at a time; do NOT open artificial files or K."""
    top=read_sealed(root,receipt,'deployment.json')
    if top['schema']!='peer-suffix203-freeze-v1' or top['prereg']!=prereg:raise ValueError('frozen identity differs')
    if top['source202']!='7261c2573773a23342286a39ce712950cb986aa6f34248cd12f5705ab420f469' or len(top['groups'])!=20:raise ValueError('fixed202 twenty units required')
    contexts=[];skeleton=[];denominators=[];refs=[]
    for ref in top['groups']:
        folder=Path(root)/ref['path'];local=manifest(folder)
        if local['manifest_sha256']!=ref['seal']:raise ValueError('group seal differs')
        data=read_sealed(folder,local,'catalog.json')
        if data['unit']!=ref['unit'] or data['domain']!=ref['domain']:raise ValueError('group scope differs')
        if [m['member_index'] for m in data['members']]!=[0,1,2]:raise ValueError('three ordered old members required')
        for m in data['members']:
            original=m['source_record']
            if original['seed']>=15 or original['split']=='test':raise ValueError('forbidden member')
            skeleton.append(dict(unit=data['unit'],host_member=str(m['member']),hive=data['domain'],clip=data['video'],source_member=str(original['member'])))
        options={r['action_id']:r for r in _project_options(data['candidates'],data['resources'])}
        seen=[]
        for host in data['host_contexts']:
            if not host['option_ids']:continue
            slot=host['host_member_index'];rows=[options[k] for k in host['option_ids']];seen.extend(host['option_ids'])
            contexts.append(dict(host_key=host['host_key'],hive=data['domain'],group=data['group'],clip=data['video'],unit=data['unit'],
                host_member=str(data['members'][slot]['member']),host_member_index=slot,stratum=host['stratum'],
                baseline_outcome=host['baseline_windows'],options=rows))
        if set(seen)!=set(options) or len(seen)!=len(set(seen)):raise ValueError('host option partition differs')
        denominators.append(dict(unit=data['unit'],hive=data['domain'],all_hosts=len(data['host_contexts']),zero_candidate_hosts=sum(not h['option_ids'] for h in data['host_contexts'])))
        refs.append(dict(ref,local_receipt=local))
        del data,options
    if len(skeleton)!=60 or Counter(m['hive'] for m in skeleton)!=dict(zip(DOMAINS,(12,12,18,18))):raise ValueError('registered60 member roster differs')
    contexts,aliases=deduplicate(contexts)
    return contexts,skeleton,refs,dict(natural_host_denominators=denominators,physical_aliases=aliases)


def load_source_inputs(root,owners,heldout,contexts,refs,*,source_domains,prereg,freeze_receipt,owners_receipt,domain_loader=None):
    """Whitelist before owner OR artificial payloads; never open probes/summary."""
    allowed=tuple(source_domains)
    if heldout not in DOMAINS or len(allowed)!=3 or set(allowed)!=set(DOMAINS)-{heldout}:raise ValueError('explicit other-three source hives required')
    expected={c['host_key']:c for c in contexts if c['hive'] in allowed};supervision={};receipts={}
    for h in allowed:
        data=read_sealed(owners,owners_receipt,h+'.json') if domain_loader is None else domain_loader(owners,h)
        if data['domain']!=h or data['prereg']!=prereg or data['frozen_manifest_sha256']!=freeze_receipt['manifest_sha256']:raise ValueError('owner scope/freeze receipt differs')
        classes={};targets={}
        for unit in data['units']:
            if unit['domain']!=h:raise ValueError('owner cross-domain unit')
            for m in unit['members']:
                for a in m['actions']:
                    key=a['action_id'];value=a['labels']['0.5']['class']
                    if key in classes or value not in ('S','D','U'):raise ValueError('duplicate/invalid owner class')
                    classes[key]=value
            for host in unit['hosts']:
                if host['host_key'] in targets:raise ValueError('duplicate owner host')
                targets[host['host_key']]=dict(valid=host['event_target_valid'],target_outcome=host['target_windows'])
        e={k:c for k,c in expected.items() if c['hive']==h}
        if set(targets)!=set(e):raise ValueError('source natural supervision host inventory differs')
        if set(classes)!={o['action_id'] for c in e.values() for o in c['options']}:raise ValueError('source natural identity inventory differs')
        for key,c in e.items():supervision[key]=dict(targets[key],classes={o['action_id']:classes[o['action_id']] for o in c['options']})
        receipts[h]=dict(file=h+'.json',sha256=owners_receipt['files'][h+'.json'],hosts=len(targets),actions=len(classes))
        del data,classes,targets
    if set(supervision)!=set(expected):raise ValueError('source supervision incomplete')
    artificial=[];artificial_supervision={};art_receipts=[]
    for ref in refs:
        if ref['domain'] not in allowed:continue
        folder=Path(root)/ref['path'];data=read_sealed(folder,ref['local_receipt'],'artificial.json')
        nzero=0
        for case in data['cases']:
            if not case['candidates']:nzero+=1;continue
            rows=_project_options(case['candidates'],case['resources']);strata={c['stratum'] for c in case['candidates']}
            if len(strata)!=1:raise ValueError('artificial host stratum differs between options')
            key=case['phase_key'];slot=case['host_member_index']
            artificial.append(dict(host_key=key,hive=ref['domain'],group=ref['group'],clip=ref['video'],unit=ref['unit'],
                host_member=f"{ref['unit']}|{slot}",host_member_index=slot,stratum=next(iter(strata)),baseline_outcome=case['baseline_windows'],options=rows,
                cutoff_frame=case['cutoff_frame'],cutoff_sha256=case['cutoff_sha256'],natural_host_key=case['host_key']))
            if key in artificial_supervision:raise ValueError('duplicate artificial case')
            artificial_supervision[key]=dict(valid=True,target_outcome=case['target_windows'])
        art_receipts.append(dict(unit=ref['unit'],hive=ref['domain'],file=ref['path']+'/artificial.json',sha256=ref['local_receipt']['files']['artificial.json'],selected_cutoffs=len(data['cases']),zero_candidate_cutoffs=nzero))
        del data
    return dict(source_hives=list(allowed),supervision=supervision,artificial=artificial,artificial_supervision=artificial_supervision,
        receipts=dict(owners_manifest_sha256=owners_receipt['manifest_sha256'],source_owner_shards=receipts,source_artificial=art_receipts,
                      target_payload_access='no heldout owner/artificial, no summary/probe/capacity/score'))


def field_train(args):
    if args.prereg!='69d52924d596d74c56edd25f1bc9bb7c8e66edb4':raise ValueError('published203 prereg required')
    fr,ow=manifest(args.frozen),manifest(args.owners)
    if fr['manifest_sha256']!=args.frozen_sha or ow['manifest_sha256']!=args.owners_sha:raise ValueError('GO input seal differs')
    contexts,skeleton,refs,denominators=natural_catalogue(args.frozen,fr,args.prereg)
    def provider(h):
        supplied=load_source_inputs(args.frozen,args.owners,h,contexts,refs,source_domains=[d for d in DOMAINS if d!=h],prereg=args.prereg,freeze_receipt=fr,owners_receipt=ow)
        print('SOURCE203',h,'natural_hosts',len(supplied['supervision']),'artificial_hosts',len(supplied['artificial']),flush=True)
        return supplied
    return train_and_seal(contexts,provider,args.out,dict(prereg=args.prereg,freeze_receipt=dict(manifest_sha256=fr['manifest_sha256'],deployment_sha256=fr['files']['deployment.json']),
        owners_manifest_sha256=ow['manifest_sha256'],natural_denominators=denominators),skeleton)


def self_test():
    rng=np.random.default_rng(20301)
    def context(h,i=0):
        key=h+'host'+str(i)
        options=[dict(action_id=key+'/'+str(k),features10=rng.normal(size=10).tolist(),outcome=[int(k!=0),0],suffix_source_keys=[[h+'donor'+str(k),i]],resources=[],min_anchor_iou=.9,first_extrapolation_error=.2) for k in range(2)]
        return dict(host_key=key,hive=h,group=1,clip=h+'clip',unit=h+'unit',host_member='0',stratum='00000',baseline_outcome=[0,0],options=options)
    contexts=[context(h,i) for h in DOMAINS for i in range(3)]
    supervision={c['host_key']:dict(valid=True,target_outcome=[int(i%2),0],classes={o['action_id']:'S' if k==1 else 'D' for k,o in enumerate(c['options'])}) for i,c in enumerate(contexts)}
    models={m:fit(contexts,supervision,m) for m in ('I','L','T')}
    assert all(max(abs(g) for g in m['optimization']['gradient'])<1e-7 for m in models.values())
    assert logits(models['L'],contexts)==logits(json.loads(json.dumps(models['L'])),contexts)
    # Hand calculation: K mass includes null and a zero-effect nonnull path.
    c=contexts[0];scores={c['options'][0]['action_id']:0.,c['options'][1]['action_id']:np.log(2.)}
    out=outcome_choices([c],scores);assert not out['selected']
    scores[c['options'][1]['action_id']]=np.log(2.)+1e-6
    assert out['host_audits'][0]['keep_logmass']==np.log(2.)
    assert outcome_choices([c],scores)['selected']==[c['options'][1]['action_id']]
    # Physical duplicate numeric ID contributes no additional outcome mass.
    duplicate=dict(c,options=c['options']+[dict(c['options'][1],action_id=c['host_key']+'/zrename')])
    ds,audit=deduplicate([duplicate]);assert len(ds[0]['options'])==2 and len(audit)==1
    # Full shared resources span all three members. Better margin wins first.
    d=context('35',9);d['options'][1]['suffix_source_keys']=c['options'][1]['suffix_source_keys'];d['host_member']='1'
    combined={o['action_id']:(3. if cc is c else 4.) for cc in (c,d) for o in cc['options']}
    chosen=outcome_choices([c,d],combined)
    assert len(chosen['selected'])<=1
    # Gradient versus central differences at a nonstationary vector.
    rows,_=make_training(contexts,supervision)
    for r,w in zip(rows,_hierarchy(rows)):r['weight']=w
    batches=[]
    for r in rows:
        a=np.zeros((len(r['options'])+1,11));a[:-1,:10]=[o['features10'] for o in r['options']];a[:-1,10]=1.;batches.append((a,r))
    theta=rng.normal(size=11)*.1;_,gradient=_loss(theta,batches)
    for i in range(10):
        probe=rng.normal(size=11)
        scalar,sg=_loss(probe,batches);vector,vg=_packed_loss(probe,_pack(batches))
        assert np.isclose(scalar,vector,atol=1e-12,rtol=1e-12) and np.allclose(sg,vg,atol=1e-12,rtol=1e-12)
    numerical=[]
    for j in range(11):
        delta=np.eye(11)[j]*1e-6;numerical.append((_loss(theta+delta,batches)[0]-_loss(theta-delta,batches)[0])/2e-6)
    assert np.allclose(gradient,numerical,atol=1e-8,rtol=1e-6)
    empty=fit([],{},'L');assert all(v is None for v in logits(empty,contexts).values())
    # No compatible outcome is an explicit gap, including wrong direction.
    missing={c['host_key']:dict(valid=True,target_outcome=[0,1])}
    rr,coverage=make_training([c],missing);assert not rr and coverage['coverage_gaps'][0]['reason']=='no_compatible_choice'
    # Artificial is matched to GT-free natural strata; unsupported excluded.
    ar=[dict(context('35',20),host_key='artificial1'),dict(context('35',21),host_key='artificial2',stratum='11111')]
    ass={a['host_key']:dict(valid=True,target_outcome=[1,0]) for a in ar}
    mixed=fit([c],{c['host_key']:supervision[c['host_key']]},'T',artificial=ar,artificial_supervision=ass)
    assert mixed['weight_audit']['natural_mass']==.5 and mixed['weight_audit']['artificial_mass']==.5 and mixed['weight_audit']['artificial_rows']==1
    # A clip lacking artificial stays equally weighted, rather than losing half.
    second=dict(context('35',30),clip='another-clip')
    cross=fit([c,second],{c['host_key']:supervision[c['host_key']],second['host_key']:dict(valid=True,target_outcome=[1,0])},'T',artificial=ar,artificial_supervision=ass)
    assert cross['weight_audit']['natural_mass']==.75 and cross['weight_audit']['artificial_mass']==.25
    assert all(sum(r[k] for k in ('natural_mass','artificial_mass'))==.5 for r in cross['weight_audit']['clip_type_weights'])
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);ownerroot=tmp/'owners';ownerroot.mkdir();frozen=tmp/'frozen';frozen.mkdir();refs=[]
        for h in DOMAINS:
            group=[cc for cc in contexts if cc['hive']==h]
            actions=[dict(action_id=o['action_id'],labels={'0.5':{'class':supervision[cc['host_key']]['classes'][o['action_id']]}}) for cc in group for o in cc['options']]
            hosts=[dict(host_key=cc['host_key'],event_target_valid=True,target_windows=supervision[cc['host_key']]['target_outcome']) for cc in group]
            save(ownerroot/(h+'.json'),dict(domain=h,prereg='synthetic',frozen_manifest_sha256='fr',units=[dict(domain=h,members=[dict(actions=actions)],hosts=hosts)]))
            folder=frozen/h;folder.mkdir();first=group[0]
            candidates=[dict(action_id=o['action_id']+'/art',features10=o['features10'],G_logit=2*o['features10'][0]-o['features10'][2],outcome_windows=o['outcome'],suffix_source_keys=o['suffix_source_keys'],stratum='00000') for o in first['options']]
            save(folder/'artificial.json',dict(cases=[dict(candidates=candidates,resources=[],phase_key=h+'/art',host_key=first['host_key'],host_member_index=0,baseline_windows=[0,0],target_windows=[1,0],cutoff_frame=20,cutoff_sha256='synthetic')]))
            refs.append(dict(domain=h,group=1,video=h+'clip',unit=h+'unit',path=h,local_receipt=dict(files={'artificial.json':sha(folder/'artificial.json')})))
        save(ownerroot/'manifest.json',{p.name:sha(p) for p in ownerroot.iterdir()});(ownerroot/'SEALED.sha256').write_text(sha(ownerroot/'manifest.json'))
        receipt=manifest(ownerroot)
        for h in DOMAINS:
            op=ownerroot/(h+'.json');ap=frozen/h/'artificial.json';ob,ab=op.read_bytes(),ap.read_bytes();op.unlink();ap.unlink()
            supplied=load_source_inputs(frozen,ownerroot,h,contexts,refs,source_domains=[v for v in DOMAINS if v!=h],prereg='synthetic',freeze_receipt=dict(manifest_sha256='fr'),owners_receipt=receipt)
            assert len(supplied['supervision'])==9 and len(supplied['artificial'])==3
            op.write_bytes(ob);ap.write_bytes(ab)
        spy=[]
        try:load_source_inputs(frozen,ownerroot,'35',contexts,refs,source_domains=['35','W','G'],prereg='synthetic',freeze_receipt={},owners_receipt=receipt,domain_loader=lambda *a:spy.append(a))
        except ValueError:pass
        else:raise AssertionError('heldout whitelist accepted')
        assert not spy
        def provider(h):return dict(source_hives=[d for d in DOMAINS if d!=h],supervision={c['host_key']:supervision[c['host_key']] for c in contexts if c['hive']!=h})
        result=train_and_seal(contexts,provider,tmp/'pred',dict(prereg='synthetic'))
        assert len(load_predictions(result['root'])['decisions'])==4
    return dict(status='PASS',seed=20301,tests='gradient finite differences; zero/null outcome mass; strictmargin; physical dedup; resource conflict; empty; direction coverage gap; per-clip half/fallback; artificial strata; all4 heldout owner+artificial absent; whitelist spy0; JSON replay; common fourfold seal')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--run-field',action='store_true');parser.add_argument('--prereg')
    parser.add_argument('--frozen',type=Path);parser.add_argument('--owners',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--frozen-sha');parser.add_argument('--owners-sha')
    args=parser.parse_args()
    if args.run_field:
        if any(v is None for v in (args.frozen,args.owners,args.out,args.frozen_sha,args.owners_sha)):parser.error('all input/output roots and pinned seals required')
        print(json.dumps(field_train(args),indent=2))
    else:print(json.dumps(self_test(),indent=2))
