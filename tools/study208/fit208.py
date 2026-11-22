"""Natural source-only LOHO switch training. Field execution requires GO.

No target-label loader is called by train_and_seal. Each fold loads only three
source label domains. Images/geometry/valid are GT-free, indexed by cut ordinal.
Three heads have identical registered parameters, NOT equal effective capacity:
geometry's encoder and appearance-input weights have structurally zero loss
gradient. This is an input/loss ablation, not an equal-capacity appearance test.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

HERE=Path(__file__).resolve().parent


def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


HEAD_PATH=HERE/'switch_head.py'
LOSS_PATH=HERE/'switch_interval_loss.py'
CODE_SNAPSHOT={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),HEAD_PATH,LOSS_PATH)}
SwitchHead=_module('switch208_shared_head',HEAD_PATH).SwitchHead
switch_interval_loss=_module('switch208_shared_interval_loss',LOSS_PATH).switch_interval_loss
DOMAINS=('35','33','W','G')
ARMS={'geometry_interval':('geometry',False),
      'relative_cross_exact':('relative_cross',True),
      'relative_cross_interval':('relative_cross',False)}
CLIPS=(('35','BEE24-35-head',3),('33','BEE24-33-head',3),
       ('W','BEE24-10',0),('W','BEE24-15',0),('G','BEE24-24',0),('G','BEE24-28',0))
SEED,STEPS,BATCH,LR=208,120,32,.003
SCHEMA='switch208-source-training-v1'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')


def manifest(root):
    root=Path(root);digest=sha(root/'manifest.json')
    if (root/'SEALED.sha256').read_text().strip()!=digest:raise ValueError('input seal differs')
    files=json.loads((root/'manifest.json').read_text())
    for name in files:
        if not (root/name).resolve().is_relative_to(root.resolve()):raise ValueError('manifest path escape')
    return dict(manifest_sha256=digest,files=files)


def checked_file(root,receipt,name):
    if name not in receipt['files'] or sha(Path(root)/name)!=receipt['files'][name]:raise ValueError('sealed file differs: '+name)
    return Path(root)/name


def input_index(root,receipt,prereg):
    data=json.loads(checked_file(root,receipt,'deployment.json').read_text())
    if data.get('schema')!='switch208-inputs-v1' or data['prereg']!=prereg:raise ValueError('input registration/schema differs')
    roster=[(r['domain'],r['clip'],int(r['seed'])) for r in data['clips']]
    if len(roster)!=6 or set(roster)!=set(CLIPS):raise ValueError('fixed six old clips/seeds required')
    return sorted(data['clips'],key=lambda r:(DOMAINS.index(r['domain']),r['clip']))


class ClipInputs:
    """Read-only memmaps; only prediction/training minibatches become tensors."""
    def __init__(self,root,receipt,record):
        self.record=dict(record);self.domain=record['domain'];self.clip=record['clip']
        path=record.get('path',self.domain+'/'+self.clip)
        self.key=self.domain+'/'+self.clip
        self.crops=np.load(checked_file(root,receipt,path+'/crops.npy'),mmap_mode='r',allow_pickle=False)
        self.geometry=np.load(checked_file(root,receipt,path+'/geometry.npy'),mmap_mode='r',allow_pickle=False)
        self.valid=np.load(checked_file(root,receipt,path+'/valid.npy'),mmap_mode='r',allow_pickle=False)
        n=len(self.crops)
        if self.crops.dtype!=np.uint8 or self.crops.shape!=(n,4,32,32):raise ValueError('uint8[N,4,32,32] crops required')
        if self.geometry.dtype!=np.float32 or self.geometry.shape!=(n,6):raise ValueError('float32[N,6] geometry required')
        if self.valid.dtype!=np.bool_ or self.valid.shape!=(n,):raise ValueError('bool[N] validity required')
        if not np.isfinite(self.geometry[self.valid]).all():raise ValueError('nonfinite valid geometry')
        if 'n_cuts' in record and record['n_cuts']!=n:raise ValueError('input row count differs')

    def tensors(self,indices):
        indices=np.asarray(indices,dtype=np.int64)
        if indices.ndim!=1 or (indices<0).any() or (indices>=len(self.crops)).any() or not self.valid[indices].all():
            raise ValueError('only valid cut ordinals may enter the head')
        crops=torch.from_numpy(np.array(self.crops[indices],copy=True)).to(dtype=torch.float32).unsqueeze(2)/255.
        geometry=torch.from_numpy(np.array(self.geometry[indices],copy=True))
        return crops,geometry

    def close(self):
        # Windows cannot remove a fixture while any np.memmap handle is open.
        for name in ('crops','geometry','valid'):
            array=getattr(self,name,None)
            mapping=getattr(array,'_mmap',None)
            if mapping is not None:mapping.close()
            setattr(self,name,None)


def validate_bags(X,y,clip):
    if X.dtype!=np.int64 or X.ndim!=2 or X.shape[1]!=3:raise ValueError('X must be int64[B,3]')
    if y.dtype!=np.int8 or y.shape!=(len(X),) or not np.isin(y,[0,1]).all():raise ValueError('y must be int8[B], 0/1')
    mask=X>=0;length=mask.sum(1)
    if (X < -1).any() or (length<1).any() or not np.array_equal(mask,np.arange(3)<length[:,None]):raise ValueError('nonempty left-packed bags, padding -1 required')
    if ((y==0)&(length!=1)).any():raise ValueError('same-ID negatives must be exact single gaps')
    if (X[mask]>=len(clip.valid)).any() or not clip.valid[X[mask]].all():raise ValueError('label references invalid/out-of-range cut; do not shorten bags')
    if any(len(set(row[row>=0]))!=len(row[row>=0]) for row in X):raise ValueError('repeated cut within one bag')
    return dict(X=X,y=y,length=length,clip=clip.key,domain=clip.domain)


def load_source_bags(root,heldout,clips,*,requested_domains,receipt,prereg,input_sha,domain_reader=None):
    """Enforce the other-three whitelist BEFORE opening any label payload."""
    allowed=tuple(requested_domains)
    if heldout not in DOMAINS or len(allowed)!=3 or set(allowed)!=set(DOMAINS)-{heldout}:raise ValueError('explicit other-three source domains required')
    expected={k for k,v in clips.items() if v.domain in allowed}
    source_records={}
    for domain in allowed:
        if domain_reader is not None:domain_reader(root,domain)
        index=json.loads(checked_file(root,receipt,domain+'.json').read_text())
        if index['domain']!=domain or index['prereg']!=prereg or index['input_manifest_sha256']!=input_sha:raise ValueError('source domain index lineage differs')
        for r in index['clips']:
            if r['domain']!=domain:raise ValueError('source index has another domain')
            key=r['domain']+'/'+r['clip']
            if key in source_records:raise ValueError('duplicate source clip')
            source_records[key]=r
    if set(source_records)!=expected:raise ValueError('source label clip inventory differs')
    result={};proof={}
    for key,r in sorted(source_records.items()):
        name=r.get('bags',key+'/bags.npz')
        if name!=key+'/bags.npz':raise ValueError('source bag path must stay in its domain/clip')
        with np.load(checked_file(root,receipt,name),allow_pickle=False) as z:
            X=np.array(z['X'],copy=True);y=np.array(z['y'],copy=True)
        result[key]=validate_bags(X,y,clips[key])
        proof[key]=dict(file=name,sha256=receipt['files'][name],bags=len(y),positive=int((y==1).sum()),negative=int((y==0).sum()),exact=int((result[key]['length']==1).sum()))
    return result,dict(source_domains=list(allowed),source_bags=proof,label_manifest_sha256=receipt['manifest_sha256'],
        forbidden_access='no heldout bags/owners, no target labels/summary')


def cohorts(source_bags,exact):
    pools={c:{h:[] for h in DOMAINS} for c in (0,1)}
    for key,b in sorted(source_bags.items()):
        for i in range(len(b['y'])):
            if not exact or b['length'][i]==1:pools[int(b['y'][i])][b['domain']].append((key,i))
    return pools


def sample_batch(pools,rng):
    sampled=[]
    for c in (1,0):
        domains=[d for d in DOMAINS if pools[c][d]]
        if not domains:raise ValueError('both source classes required')
        for _ in range(BATCH//2):
            h=domains[int(rng.integers(len(domains)))];items=pools[c][h]
            sampled.append(items[int(rng.integers(len(items)))])
    return sampled


def bag_logits(model,sampled,source_bags,clips):
    unique=defaultdict(set)
    for key,i in sampled:unique[key].update(map(int,source_bags[key]['X'][i][source_bags[key]['X'][i]>=0]))
    lookup={};parts=[];offset=0
    for key in sorted(unique):
        ids=sorted(unique[key]);c,g=clips[key].tensors(ids);parts.append(model(c,g)['logit'])
        for j,k in enumerate(ids):lookup[key,k]=offset+j
        offset+=len(ids)
    flat=torch.cat(parts);gather=np.zeros((len(sampled),3),dtype=np.int64);mask=np.zeros_like(gather,dtype=bool);ys=[]
    for row,(key,i) in enumerate(sampled):
        x=source_bags[key]['X'][i];ys.append(int(source_bags[key]['y'][i]))
        for j,k in enumerate(x):
            if k>=0:gather[row,j]=lookup[key,int(k)];mask[row,j]=True
    logits=flat[torch.from_numpy(gather)];mask=torch.from_numpy(mask);y=torch.tensor(ys,dtype=torch.long)
    return logits,mask&(y==0)[:,None],mask&(y==1)[:,None]


def train_one(clips,source_bags,arm,*,steps=STEPS):
    """steps override is for tiny synthetic fixtures; field always passes120."""
    if arm not in ARMS:raise ValueError('unknown arm')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(SEED)
    head_arm,exact=ARMS[arm];model=SwitchHead(channels=8,arm=head_arm).to('cpu')
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    pools=cohorts(source_bags,exact);domains=sorted({b['domain'] for b in source_bags.values()},key=DOMAINS.index)
    counts={h:{str(c):len(pools[c][h]) for c in (0,1)} for h in domains}
    registered=sum(p.numel() for p in model.parameters())
    encoder_count=sum(p.numel() for p in model.encoder.parameters())
    appearance_column=model.classifier[0].weight.shape[0]
    receipt=dict(arm=arm,head_arm=head_arm,exact_only=exact,seed=SEED,steps=steps,batch_size=BATCH,
        positive_per_step=16,negative_per_step=16,learning_rate=LR,channels=8,device='cpu',threads=1,
        sampler='first 16 positive then16 negative; uniform nonempty source domain per class, then uniform bag across its source clips; replacement',
        source_counts=counts,missing_classes_by_source_domain={h:[c for c in (0,1) if not pools[c][h]] for h in domains},
        optimizer=dict(name='Adam',lr=LR,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False,foreach=False),
        sampler_cohort_sha256={h:{str(c):hashlib.sha256(json.dumps(pools[c][h],separators=(',',':')).encode()).hexdigest() for c in (0,1)} for h in domains},
        class_available_domains={str(c):[h for h in DOMAINS if pools[c][h]] for c in (0,1)},
        registered_parameters=registered,encoder_parameters=encoder_count,
        structural_gradient_parameter_limit=registered-encoder_count-appearance_column if head_arm=='geometry' else registered,
        capacity_limit='same registered parameter count; geometry encoder and appearance-input column have zero loss gradient, effective capacity/compute are not matched')
    if any(not any(pools[c][h] for h in domains) for c in (0,1)):
        receipt.update(status='noop_missing_class',completed_steps=0,losses=[],changed_parameter_elements=0,
            sampled_batches_sha256=hashlib.sha256(b'[]').hexdigest())
        return model,receipt
    optimizer=torch.optim.Adam(model.parameters(),lr=LR,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False,foreach=False)
    rng=np.random.default_rng(SEED);trace=[];losses=[];model.train()
    for step in range(steps):
        sampled=sample_batch(pools,rng);trace.append(sampled)
        z,survival,event=bag_logits(model,sampled,source_bags,clips)
        loss=switch_interval_loss(z,survival,event)
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite source training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise RuntimeError('nonfinite source gradient')
        optimizer.step();losses.append(float(loss.detach()))
    receipt.update(status='trained',completed_steps=steps,losses=losses,
        changed_parameter_elements=sum(int((v!=initial[k]).sum()) for k,v in model.state_dict().items()),
        sampled_batches_sha256=hashlib.sha256(json.dumps(trace,separators=(',',':')).encode()).hexdigest(),first_batch=trace[0] if trace else [])
    model.eval();return model,receipt


def predict_clip(model,clip,*,noop=False,batch_size=128):
    if batch_size<1:raise ValueError('positive prediction batch size required')
    result=np.full(len(clip.valid),np.nan,dtype=np.float32);ids=np.flatnonzero(clip.valid)
    if noop:result[ids]=.5;return result
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(ids),batch_size):
            subset=ids[start:start+batch_size];c,g=clip.tensors(subset)
            p=model(c,g)['logit'].sigmoid().cpu().numpy()
            if not np.isfinite(p).all():raise RuntimeError('nonfinite valid prediction')
            result[subset]=p
    return result


def calibrate(model,clips,source_bags,*,noop=False,batch_size=128,return_scores=False):
    pooled=[];counts={};origins=[];bag_indices=[];cut_indices=[];clip_order=sorted(source_bags)
    for key,b in sorted(source_bags.items()):
        exact_negative=(b['y']==0)&(b['length']==1);ids=b['X'][exact_negative,0]
        counts[key]=len(ids)
        if not len(ids):continue
        # Bag multiplicities are preserved; only forward computation is deduped.
        unique,inverse=np.unique(ids,return_inverse=True)
        if noop:probs=np.full(len(unique),.5,dtype=np.float32)
        else:
            parts=[]
            with torch.inference_mode():
                model.eval()
                for start in range(0,len(unique),batch_size):
                    c,g=clips[key].tensors(unique[start:start+batch_size]);parts.append(model(c,g)['logit'].sigmoid().cpu().numpy())
            probs=np.concatenate(parts)
        pooled.append(probs[inverse])
        origins.extend([clip_order.index(key)]*len(ids));bag_indices.extend(np.flatnonzero(exact_negative).tolist());cut_indices.extend(ids.tolist())
    values=np.concatenate(pooled) if pooled else np.empty(0,dtype=np.float32)
    if not np.isfinite(values).all():raise RuntimeError('nonfinite source calibration score')
    q=float(np.quantile(values,.999,method='higher')) if len(values) else None
    threshold=1. if noop or q is None else max(.5,q)
    info=dict(threshold=threshold,quantile=q,quantile_probability=.999,quantile_method='higher',
        exact_negative_bags=len(values),per_source_clip=counts,selection='valid AND probability strictly greater than threshold',
        clip_order=clip_order,reason='noop_missing_class' if noop else 'no_exact_negative' if q is None else 'source_pooled_exact_negative')
    if return_scores:
        return info,dict(probabilities=values,clip_index=np.asarray(origins,dtype=np.int16),bag_index=np.asarray(bag_indices,dtype=np.int64),cut_index=np.asarray(cut_indices,dtype=np.int64))
    return info


def train_and_seal(inputs_root,labels_root,out,*,prereg,input_sha,labels_sha,prediction_batch=128):
    if prediction_batch<1:raise ValueError('positive prediction batch size required')
    if any(sha(path)!=digest for path,digest in CODE_SNAPSHOT.items()):raise ValueError('loaded training/shared source changed before execution')
    ir,lr=manifest(inputs_root),manifest(labels_root)
    if ir['manifest_sha256']!=input_sha or lr['manifest_sha256']!=labels_sha:raise ValueError('GO seal mismatch')
    records=input_index(inputs_root,ir,prereg)
    clips={r['domain']+'/'+r['clip']:ClipInputs(inputs_root,ir,r) for r in records}
    dest=Path(out)
    if dest.exists():raise FileExistsError('immutable prediction root already exists')
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        metadata=dict(schema=SCHEMA,prereg=prereg,arms=list(ARMS),folds={},clips=[],
            input_manifest_sha256=input_sha,label_manifest_sha256=labels_sha,
            raw_diagnostic_threshold=.5,raw_diagnostic_rule='valid AND probability >0.5')
        for target in DOMAINS:
            source,proof=load_source_bags(labels_root,target,clips,requested_domains=[h for h in DOMAINS if h!=target],receipt=lr,prereg=prereg,input_sha=input_sha)
            fold=dict(heldout=target,source_domains=[h for h in DOMAINS if h!=target],models={})
            target_records=[r for r in records if r['domain']==target]
            outputs={r['domain']+'/'+r['clip']:dict(domain=target,clip=r['clip'],seed=r['seed'],n_cuts=len(clips[target+'/'+r['clip']].valid),
                valid_cuts=int(clips[target+'/'+r['clip']].valid.sum()),outputs={}) for r in target_records}
            for arm in ARMS:
                print('FIT208',target,arm,'source_bags',sum(len(b['y']) for b in source.values()),flush=True)
                model,training=train_one(clips,source,arm,steps=STEPS)
                noop=training['status']!='trained';calibration,cal_scores=calibrate(model,clips,source,noop=noop,batch_size=prediction_batch,return_scores=True)
                calibration_file=f'calibration/{target}/{arm}.npz';(tmp/calibration_file).parent.mkdir(parents=True,exist_ok=True)
                np.savez(tmp/calibration_file,**cal_scores);calibration['scores_file']=calibration_file
                training.update(calibration=calibration,source_receipt=proof)
                model_file=f'models/{target}/{arm}.pt';training_file=f'training/{target}/{arm}.json'
                (tmp/model_file).parent.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),tmp/model_file);save(tmp/training_file,training)
                fold['models'][arm]=dict(status=training['status'],threshold=calibration['threshold'],calibration=calibration,model_file=model_file,training_file=training_file,
                    registered_parameters=training['registered_parameters'],structural_gradient_parameter_limit=training['structural_gradient_parameter_limit'])
                for key,record in outputs.items():
                    probs=predict_clip(model,clips[key],noop=noop,batch_size=prediction_batch)
                    stem=f'preds/{target}/{record["clip"]}/{arm}';(tmp/stem).parent.mkdir(parents=True,exist_ok=True)
                    files=dict(probabilities=stem+'.npy',selected=stem+'.selected.npy',raw05=stem+'.raw05.npy')
                    np.save(tmp/files['probabilities'],probs,allow_pickle=False)
                    np.save(tmp/files['selected'],np.asarray(clips[key].valid)&(probs>calibration['threshold']),allow_pickle=False)
                    np.save(tmp/files['raw05'],np.asarray(clips[key].valid)&(probs>.5),allow_pickle=False)
                    record['outputs'][arm]=dict(files,threshold=calibration['threshold'])
            metadata['folds'][target]=fold;metadata['clips'].extend(outputs.values())
            del source
        if any(sha(path)!=digest for path,digest in CODE_SNAPSHOT.items()):raise ValueError('training/shared source changed during execution')
        save(tmp/'metadata.json',metadata)
        save(tmp/'receipt.json',dict(schema=SCHEMA,prereg=prereg,input_manifest_sha256=input_sha,label_manifest_sha256=labels_sha,
            code_receipts=CODE_SNAPSHOT,torch_version=torch.__version__,numpy_version=np.__version__,cpu_threads=1,
            registered_count_is_not_effective_capacity=True,all_target_labels_prohibited_until_common_seal=True))
        save(tmp/'manifest.json',{p.relative_to(tmp).as_posix():sha(p) for p in tmp.rglob('*') if p.is_file()})
        digest=sha(tmp/'manifest.json');(tmp/'SEALED.sha256').write_text(digest+'\n');tmp.rename(dest)
        return dict(root=str(dest),manifest_sha256=digest)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
        for clip in clips.values():clip.close()


def verify_seal(root):
    root=Path(root);receipt=manifest(root)
    for name in receipt['files']:checked_file(root,receipt,name)
    meta=json.loads((root/'metadata.json').read_text());audit=json.loads((root/'receipt.json').read_text())
    if set(meta['folds'])!=set(DOMAINS) or len(meta['clips'])!=6 or set(meta['arms'])!=set(ARMS):raise ValueError('complete12-model six-clip prediction seal required')
    for h,f in meta['folds'].items():
        if f['heldout']!=h or set(f['source_domains'])!=set(DOMAINS)-{h} or set(f['models'])!=set(ARMS):raise ValueError('source-only boundary differs')
        for arm,m in f['models'].items():
            for name in (m['model_file'],m['training_file'],m['calibration']['scores_file']):
                if name not in receipt['files']:raise ValueError('model/calibration artifact missing')
            training=json.loads((root/m['training_file']).read_text())
            if set(training['source_receipt']['source_domains'])!=set(DOMAINS)-{h}:raise ValueError('training receipt source scope differs')
            if m['status'] not in ('trained','noop_missing_class') or not .5<=m['threshold']<=1.:raise ValueError('invalid deployment status/threshold')
    if {(c['domain'],c['clip'],c['seed']) for c in meta['clips']}!=set(CLIPS):raise ValueError('heldout six-clip roster differs')
    for c in meta['clips']:
        if set(c['outputs'])!=set(ARMS):raise ValueError('all target arms required')
        for arm,files in c['outputs'].items():
            model=meta['folds'][c['domain']]['models'][arm]
            if files['threshold']!=model['threshold']:raise ValueError('target/source threshold differs')
            for name in ('probabilities','selected','raw05'):
                if files[name] not in receipt['files']:raise ValueError('target artifact missing')
            p=np.load(root/files['probabilities'],mmap_mode='r',allow_pickle=False)
            selected=np.load(root/files['selected'],mmap_mode='r',allow_pickle=False)
            raw=np.load(root/files['raw05'],mmap_mode='r',allow_pickle=False)
            if p.dtype!=np.float32 or p.shape!=(c['n_cuts'],) or np.isinf(p).any() or np.isfinite(p).sum()!=c['valid_cuts'] or ((p[np.isfinite(p)]<0)|(p[np.isfinite(p)]>1)).any():raise ValueError('raw target probabilities/validity differ')
            if selected.dtype!=np.bool_ or raw.dtype!=np.bool_ or selected.shape!=p.shape or raw.shape!=p.shape:raise ValueError('target selection shape/type differs')
            if not np.array_equal(selected,np.isfinite(p)&(p>files['threshold'])) or not np.array_equal(raw,np.isfinite(p)&(p>.5)):raise ValueError('strict target masks differ')
    for path,digest in audit['code_receipts'].items():
        if sha(path)!=digest:raise ValueError('training/shared code changed after seal')
    return dict(metadata=meta,receipt=audit,manifest_sha256=receipt['manifest_sha256'])


def load_predictions(root):
    return verify_seal(root)


def self_test():
    """Tiny opt-in CPU engineering fixture, not natural training/evidence."""
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);inp=root/'inputs';lab=root/'labels';inp.mkdir();lab.mkdir();records=[]
        for h,clip,seed in CLIPS:
            folder=inp/h/clip;folder.mkdir(parents=True)
            pixels=np.zeros((8,4,32,32),dtype=np.uint8)
            pixels[:,:,::4,:]=220;pixels[1::2,2:]=np.transpose(pixels[1::2,2:],(0,1,3,2))
            np.save(folder/'crops.npy',pixels);np.save(folder/'geometry.npy',np.arange(48,dtype=np.float32).reshape(8,6)/48)
            np.save(folder/'valid.npy',np.array([True]*7+[False]))
            records.append(dict(domain=h,clip=clip,seed=seed,path=h+'/'+clip,n_cuts=8))
            destination=lab/h/clip;destination.mkdir(parents=True)
            np.savez(destination/'bags.npz',X=np.array([[0,-1,-1],[1,-1,-1],[2,3,4],[5,-1,-1]],dtype=np.int64),y=np.array([0,1,1,0],dtype=np.int8))
        save(inp/'deployment.json',dict(schema='switch208-inputs-v1',prereg='synthetic',clips=records))
        save(inp/'manifest.json',{p.relative_to(inp).as_posix():sha(p) for p in inp.rglob('*') if p.is_file()});(inp/'SEALED.sha256').write_text(sha(inp/'manifest.json'))
        ir=manifest(inp);clips={r['domain']+'/'+r['clip']:ClipInputs(inp,ir,r) for r in records}
        label_records=[dict(r,bags=r['path']+'/bags.npz') for r in records]
        save(lab/'deployment.json',dict(schema='switch208-labels-v1',prereg='synthetic',input_manifest_sha256=ir['manifest_sha256'],clips=label_records))
        for h in DOMAINS:save(lab/(h+'.json'),dict(domain=h,prereg='synthetic',input_manifest_sha256=ir['manifest_sha256'],clips=[r for r in label_records if r['domain']==h]))
        save(lab/'manifest.json',{p.relative_to(lab).as_posix():sha(p) for p in lab.rglob('*') if p.is_file()});(lab/'SEALED.sha256').write_text(sha(lab/'manifest.json'))
        lr=manifest(lab)
        for h in DOMAINS:
            index_path=lab/(h+'.json');absent=[(index_path,index_path.read_bytes())];index_path.unlink()
            for r in records:
                if r['domain']==h:
                    path=lab/r['path']/'bags.npz';data=path.read_bytes();path.unlink();absent.append((path,data))
            source,_=load_source_bags(lab,h,clips,requested_domains=[d for d in DOMAINS if d!=h],receipt=lr,prereg='synthetic',input_sha=ir['manifest_sha256'])
            assert set(b['domain'] for b in source.values())==set(DOMAINS)-{h}
            for path,data in absent:path.write_bytes(data)
        spy=[]
        try:load_source_bags(lab,'35',clips,requested_domains=['35','W','G'],receipt=lr,prereg='synthetic',input_sha='',domain_reader=lambda *args:spy.append(args))
        except ValueError:pass
        else:raise AssertionError('heldout whitelist accepted')
        assert not spy
        source,_=load_source_bags(lab,'35',clips,requested_domains=['33','W','G'],receipt=lr,prereg='synthetic',input_sha=ir['manifest_sha256'])
        models={};audits={}
        for arm in ARMS:models[arm],audits[arm]=train_one(clips,source,arm,steps=2)
        assert len({a['registered_parameters'] for a in audits.values()})==1
        assert audits['geometry_interval']['structural_gradient_parameter_limit']<audits['relative_cross_interval']['structural_gradient_parameter_limit']
        assert audits['geometry_interval']['sampled_batches_sha256']==audits['relative_cross_interval']['sampled_batches_sha256']
        clip=next(iter(clips.values()));p1=predict_clip(models['relative_cross_interval'],clip,batch_size=2);p2=predict_clip(models['relative_cross_interval'],clip,batch_size=4)
        assert np.allclose(p1,p2,equal_nan=True,atol=1e-6) and np.isnan(p1[-1])
        model2,audit2=train_one(clips,source,'relative_cross_interval',steps=2)
        assert all(torch.equal(v,model2.state_dict()[k]) for k,v in models['relative_cross_interval'].state_dict().items())
        negatives={k:dict(b,y=np.zeros_like(b['y']),length=np.ones_like(b['length']),X=np.column_stack((b['X'][:,0],np.full((len(b['X']),2),-1,dtype=np.int64)))) for k,b in source.items()}
        no_model,no_audit=train_one(clips,negatives,'relative_cross_interval',steps=2)
        assert no_audit['status']=='noop_missing_class'
        assert calibrate(no_model,clips,negatives,noop=True)['threshold']==1.
        assert np.all(predict_clip(no_model,clip,noop=True)[clip.valid]==.5)
        for item in clips.values():item.close()
    return dict(status='PASS',synthetic_only=True,tests='all4 absent heldout files; whitelist spy0; parameter/effective capacity distinction; paired interval sampler; deterministic2step fit; bounded prediction batches; invalidNaN; missingclass noop')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--run-field',action='store_true')
    parser.add_argument('--inputs',type=Path);parser.add_argument('--labels',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--input-sha');parser.add_argument('--labels-sha');parser.add_argument('--prereg');parser.add_argument('--prediction-batch',type=int,default=128)
    args=parser.parse_args()
    if args.run_field:
        if any(v is None for v in (args.inputs,args.labels,args.out,args.input_sha,args.labels_sha,args.prereg)):parser.error('all roots, seals, prereg required')
        print(json.dumps(train_and_seal(args.inputs,args.labels,args.out,prereg=args.prereg,input_sha=args.input_sha,labels_sha=args.labels_sha,prediction_batch=args.prediction_batch),indent=2))
    elif args.self_test:print(json.dumps(self_test(),indent=2))
    else:parser.error('--self-test or --run-field required')
