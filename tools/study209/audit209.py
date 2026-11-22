"""Explicit §209 field adapter. Import is source-only; run requires final score GO.

Independently enumerates catalog, weak labels, convex heads, selections, raw
materialization. Native low-level Hungarian/event matching helpers are shared
and declared in the receipt. Each phase writes an atomic durable receipt.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import Counter,defaultdict
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import verify209 as independent

PREREG='9efbea732621dec804aea9dcc9df87fc1ca6293e'
SOURCE208='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94'
SPECS={('35','BEE24-35-head',3),('33','BEE24-33-head',3),('W','BEE24-10',0),('W','BEE24-15',0),('G','BEE24-24',0),('G','BEE24-28',0)}
DOMAINS=('35','33','W','G')
EXPECTED_SEALS=dict(catalog='daba4221ee9d297c4d47c786f550d9f62d6aef3318feec3bc0fc7128ffae696f',features='c39fdb1b28b79450e0706c6b52f7f629f40d907d175da199067307657d286bda',labels='5d009e27bfde4cbc257c5942f19596654b237dc14443e1e9f397493f7e299fc5',predictions='238bda44bcbf29fa5cc42db2139d07aff9f386baa36ed49b0a50ffb113bc511e',score='3670fa9c892c8d1a918adfdc253013bf6837b0685897befb1f66af39e5e13381')

def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt',encoding='utf8') as stream:return json.load(stream)
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')
    temp.replace(path);path.with_suffix('.sha256').write_text(sha(path)+'\n')

def seal(root,expected=None):
    root=Path(root);digest=sha(root/'manifest.json')
    assert digest==(root/'SEALED.sha256').read_text().strip()
    if expected is not None:assert digest==expected
    for rel,want in read(root/'manifest.json').items():
        p=(root/rel).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==want,(root,rel)
    return digest

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def close(a,b,tolerance=2e-10):
    assert np.shape(a)==np.shape(b) and np.allclose(a,b,rtol=tolerance,atol=tolerance,equal_nan=False),(a,b)

def canonical(rows):
    result=np.empty(len(rows),np.int64);seen=set()
    for pid in sorted(set(rows[:,1])):
        indices=np.flatnonzero(rows[:,1]==pid);indices=indices[np.argsort(rows[indices,0],kind='stable')]
        assert np.all(np.diff(rows[indices,0])>0)
        for run,part in enumerate(np.split(indices,np.flatnonzero(np.diff(rows[indices,0])>25)+1)):
            cid=int(pid)+100000*run;assert cid not in seen;seen.add(cid);result[part]=cid
    return result

def catalog_labels(args,seals):
    """All six source arrays; complete sweep incl rejected positive pairs."""
    from beetrack.evaluate import load_tracks
    top=read(args.root/'catalog/deployment.json');features=read(args.root/'features/deployment.json');labels=read(args.root/'labels/deployment.json')
    assert all(t['prereg']==PREREG for t in (top,features,labels))
    assert top['source208']==SOURCE208 and sha(args.source208/'manifest.json')==SOURCE208==(args.source208/'SEALED.sha256').read_text().strip()
    source_manifest=read(args.source208/'manifest.json')
    assert {(r['domain'],r['clip'],r['seed']) for r in top['clips']}==SPECS and len(top['clips'])==6
    assert features['candidate_catalog_manifest_sha256']==labels['candidate_manifest_sha256']==seals['catalog']
    assert labels['feature_manifest_sha256']==seals['features']
    refs={(r['domain'],r['clip']):r for r in features['clips']}
    records={};reports=[]
    for ref in top['clips']:
        key=ref['domain']+'/'+ref['clip'];folder=args.root/'catalog'/ref['path'];seal(folder,ref['seal'])
        cat=read(folder/'catalog.json.gz');meta=cat['metadata'];src=args.source208/meta['input208_root_relative']
        assert sha(src/'rows.npy')==meta['rows_sha256'] and sha(src/'canon.npy')==meta['canon_sha256']
        for name,keyname in (('rows.npy','rows_sha256'),('canon.npy','canon_sha256')):assert source_manifest[(src/name).relative_to(args.source208).as_posix()]==meta[keyname]
        rows=np.load(src/'rows.npy',mmap_mode='r');canon=np.load(src/'canon.npy',mmap_mode='r')
        assert rows.dtype==np.float64 and np.array_equal(canonical(rows),canon)
        expected=independent.catalog_reference(rows,canon,cat['nframes'])
        table={p['pair_id']:p for p in cat['pairs']};eligible={p['pair_id']:p for p in expected if p['geometry4'][0]>=.2}
        assert set(eligible)==set(table)
        rejected={tuple(p['canonical_ids']):p for p in cat.get('rejected_pairs',[])}
        for p in expected:
            if p['pair_id'] not in eligible:
                r=rejected[tuple(p['canonical_ids'])];assert r['shared_frames']==len(p['shared_dates'])
                close(r['iou_sum'],sum(p['ious']));continue
            q=table[p['pair_id']]
            for field in ('shared_dates','row_indices','canonical_ids'):assert q[field]==p[field]
            close(q['ious'],p['ious']);close(q['geometry4'],p['geometry4'])
            rep=q['representative'];j=p['representative_index']
            assert (rep is None)==(j is None)
            if j is not None:assert rep['frame']==p['shared_dates'][j] and rep['row_indices']==p['row_indices'][j]
        assert len(expected)==cat['denominator']['positive_intersection_pairs']
        frec=refs[ref['domain'],ref['clip']];ffolder=args.root/'features'/frec['path'];seal(ffolder,frec['seal'])
        fpairs=read(args.root/'features'/frec['features_file'])['pairs'];ftable={p['pair_id']:p for p in fpairs};assert set(ftable)==set(table)
        lab=read(args.root/'labels'/ref['domain']/ref['clip']/'evidence.json.gz');gtpath=args.repo/lab['GT_relative_path']
        assert sha(gtpath)==lab['GT_sha256']==labels['GT_sha256'][lab['GT_relative_path']]
        GT=load_tracks(gtpath);lmap={r['pair_id']:r for r in lab['records']};assert set(lmap)==set(table)
        gtframes={int(d):GT[GT[:,0]==d] for d in np.unique(GT[:,0])}
        for date,t in lab['GT_frames'].items():
            ix=np.flatnonzero(GT[:,0]==int(date));assert t['GT_raw_indices']==ix.tolist();close(np.asarray(t['GT_rows']).reshape(-1,6),GT[ix],0)
        allclasses=Counter();pairrows=[];weak={};collisions={}
        cache={};weak_owners={}
        def owner_date(date):
            if date in weak_owners:return weak_owners[date]
            gg=gtframes.get(date,np.empty((0,6)));jj=np.flatnonzero(rows[:,0]==date);out={int(i):None for i in jj}
            if len(gg) and np.isfinite(gg).all() and (gg[:,4:6]>0).all() and (gg[:,:2]==np.rint(gg[:,:2])).all():
                aa=gg[:,2:6];bb=rows[jj,2:6]
                wh=np.maximum(0,np.minimum(aa[:,None,:2]+aa[:,None,2:],bb[None,:,:2]+bb[None,:,2:])-np.maximum(aa[:,None,:2],bb[None,:,:2]))
                inter=wh[:,:,0]*wh[:,:,1];matrix=inter/(np.prod(aa[:,2:],axis=1)[:,None]+np.prod(bb[:,2:],axis=1)[None,:]-inter+1e-9)
                multiplicity=Counter(map(int,gg[:,1]))
                for col,i in enumerate(jj):
                    hits=np.flatnonzero(matrix[:,col]>=.3)
                    if len(hits)==1:
                        oid=int(gg[hits[0],1])
                        if oid>=0 and multiplicity[oid]==1:out[int(i)]=oid
            weak_owners[date]=out;return out
        for pid,p in table.items():
            proof=lmap[pid];labels_by_date={}
            assert len(proof['common_evidence'])==len(p['shared_dates'])
            for d,ix,ev in zip(p['shared_dates'],p['row_indices'],proof['common_evidence']):
                assert ev['frame']==d and ev['prediction_raw_indices']==ix
                k=(d,tuple(ix))
                if k not in cache:
                    owners=owner_date(d);a,b=[owners[i] for i in ix]
                    cache[k]='U' if a is None or b is None else 'S' if a==b else 'D'
                label=cache[k];assert label==ev['class'],(key,pid,d,label,ev['class'])
                labels_by_date[d]=label;allclasses[label]+=1
            counts=dict(Counter(labels_by_date.values()))
            assert {c:counts.get(c,0) for c in ('S','D','U')}==proof['common_counts']
            representative=p['representative'];rep=None if representative is None else representative['frame']
            y='U' if rep is None else labels_by_date[rep]
            assert proof['representative_class']==y and proof['representative_frame']==rep
            collision=[d for d,c in labels_by_date.items() if c=='D']
            assert proof['collision_dates']==collision and proof['any_verified_collision']==bool(collision)
            weak[pid]=y;collisions[pid]=collision
            f=ftable[pid];close(f['geometry4'],p['geometry4'])
            assert f['main_valid']==f['true']['valid'] and f['paired_valid']==(f['true']['valid'] and f['sham']['valid'])
            pairrows.append(dict(domain=ref['domain'],clip=ref['clip'],pair_id=pid,canonical_ids=p['canonical_ids'],geometry4=p['geometry4'],true=f['true'],sham=f['sham']))
        records[key]=dict(ref=ref,cat=cat,meta=meta,rows=rows,canon=canon,GT=GT,features=ftable,feature_folder=ffolder,pairrows=pairrows,weak=weak,collisions=collisions)
        reports.append(dict(clip=key,raw_rows=len(rows),positive_intersection_pairs=len(expected),eligible_pairs=len(table),rejected_pairs=len(rejected),shared_date_weak_classes=dict(allclasses),representative_classes=dict(Counter(weak.values()))))
        print('VERIFY209_CATALOG_LABEL',key,flush=True)
    save(args.root/'verification/catalog_labels.json',dict(passed=True,seals=seals,clips=reports,scope='All six original208 K/canonical arrays; complete sweep and all-shared mean incl zero; all retained/rejected pair counts; all original shared-frame GT tables and weak S/D/U independently recomputed. Weak labels are not physical same-bee truth.'))
    return records

def fit_predict(args,records,seals):
    pred=read(args.root/'predictions/predictions.json');reports=[];choices={};compact=[]
    allrows=[row for r in records.values() for row in r['pairrows']]
    weak={r['ref']['domain']+'/'+r['ref']['clip']+'/'+p:c for r in records.values() for p,c in r['weak'].items()}
    for heldout in DOMAINS:
        model=read(args.root/'predictions/models'/f'{heldout}.json');assert model['heldout']==heldout and set(model['source_domains'])==set(DOMAINS)-{heldout}
        cohort=sorted([r for r in allrows if r['domain']!=heldout and r['geometry4'][0]>=.2 and r['true']['valid'] and weak[r['domain']+'/'+r['clip']+'/'+r['pair_id']]!='U'],key=lambda r:r['domain']+'/'+r['clip']+'/'+r['pair_id'])
        training=model['training_records'];assert len(training)==len(cohort)
        for a,b in zip(training,cohort):
            key=b['domain']+'/'+b['clip']+'/'+b['pair_id'];assert a['key']==key and a['domain']!=heldout and a['y']==int(weak[key]=='S')
            close(a['geometry4'],b['geometry4']);close(a['pixels4'],b['true']['feature_vector'])
        ownheads={}
        for head in ('G','F','X'):
            own=independent.refit(training,head);actual=model['heads'][head];assert own['status']==actual['status']
            if own['status']=='fitted':
                for field in ('mean','scale','beta','intercept'):close(own[field],actual[field],2e-6)
                close(own['weight'],[r['weight'] for r in training]);close(own['objective'],actual['optimization']['objective'],1e-8)
                assert max(abs(own['gradient']))<2e-6
            ownheads[head]=own
        active=all(h['status']=='fitted' for h in ownheads.values())
        for key,r in records.items():
            if r['ref']['domain']!=heldout:continue
            saved=pred['clips'][key];logits={head:{} for head in ('G','F','X')}
            savedrows={a['pair_id']:a for a in saved['records']}
            for row in r['pairrows']:
                pid=row['pair_id'];main=row['true']['valid'] and row['geometry4'][0]>=.2
                for head,h in ownheads.items():
                    value=None
                    if active and main:
                        d=len(h['mean']);x=np.asarray(row['geometry4']+row['true']['feature_vector'])[:d]
                        # Exact deployment uses frozen coefficients after independent refit validation.
                        frozen=model['heads'][head];value=float(((x-frozen['mean'])/frozen['scale'])@frozen['beta']+frozen['intercept'])
                        close(value,savedrows[pid]['logits'][head],1e-11)
                    else:assert savedrows[pid]['logits'][head] is None
                    logits[head][pid]=value
                if active and main and row['sham']['valid']:
                    h=model['heads']['X'];x=np.asarray(row['geometry4']+row['sham']['feature_vector']);value=float(((x-h['mean'])/h['scale'])@h['beta']+h['intercept'])
                    close(value,savedrows[pid]['X_sham_logit'],1e-11)
                else:assert savedrows[pid]['X_sham_logit'] is None
            selected,slate,k=independent.select(r['pairrows'],logits,active)
            assert selected==saved['selected'] and slate==saved['slate'] and len(slate)==saved['n_slate'] and k==saved['k']
            choices[key]=selected
            compact.append(dict(clip=key,slate=slate,k=k,selected=selected,rows=[dict(pair_id=p['pair_id'],geometry4=p['geometry4'],true_valid=p['true']['valid'],sham_valid=p['sham']['valid'],true4=p['true']['feature_vector'],sham4=p['sham']['feature_vector'],weak=r['weak'][p['pair_id']],anyD=bool(r['collisions'][p['pair_id']]),logits={h:logits[h][p['pair_id']] for h in logits}) for p in r['pairrows']]))
        reports.append(dict(heldout=heldout,source_rows=len(training),source_domains=model['source_domains'],status=model['status'],all_three_independent_refits=True))
        print('VERIFY209_REFIT',heldout,flush=True)
    save(args.root/'verification/models.json',dict(passed=True,seals=seals,folds=reports,scope='All four source-only cohorts/weak rep labels/weights/shared moments/three convex refits, all target logits/sham substitution and six disjoint masks independently rebuilt. No heldout label enters refit.'))
    save(args.root/'verification/numeric_candidates.json',dict(schema='common209-independent-numeric-v1',seals=seals,clips=compact))
    diagnostics={}
    for hive in DOMAINS:
        population=[(key,row) for key,r in records.items() if r['ref']['domain']==hive for row in r['pairrows']]
        classes=Counter(records[key]['weak'][row['pair_id']] for key,row in population)
        main=[(key,row) for key,row in population if row['true']['valid'] and records[key]['weak'][row['pair_id']]!='U']
        paired=[(key,row) for key,row in main if row['sham']['valid']]
        def metric(cohort,head,sham=False):
            yy=[];ss=[]
            for key,row in cohort:
                saved=next(r for r in pred['clips'][key]['records'] if r['pair_id']==row['pair_id'])
                value=saved['X_sham_logit'] if sham else saved['logits'][head]
                if value is None:continue
                yy.append(int(records[key]['weak'][row['pair_id']]=='S'));ss.append(value)
            return dict(rows=len(yy),S=sum(yy),D=len(yy)-sum(yy),AP=independent.average_precision(yy,ss),raw_logit_positive=sum(s>0 for s in ss))
        diagnostics[hive]=dict(weak_all=dict(classes),main_known=len(main),paired_known=len(paired),
                               main={h:metric(main,h) for h in ('G','F','X')},paired={h:metric(paired,h) for h in ('F','X')})
        diagnostics[hive]['paired']['shamX']=metric(paired,'X',True)
    sufficient=all(diagnostics[h]['paired']['X']['S']>=5 and diagnostics[h]['paired']['X']['D']>=5 for h in ('35','33'))
    means={h:sum(diagnostics[d]['paired'][h]['AP'] for d in ('35','33'))/2 for h in ('F','X','shamX')} if sufficient else None
    attribution=dict(sufficient=sufficient,head_equal_AP=means,passed=None if not sufficient else means['X']>means['F'] and means['X']>means['shamX'])
    save(args.root/'verification/weak_diagnostics.json',dict(passed=True,seals=seals,targets=diagnostics,attribution=attribution,scope='Heldout predictions scored only after common seal. Weak rectangle S/D main and paired support; sham uses same frozenX, no refit. AP pools clips within target, head35/33 equal means only for registered attribution. Raw logit>0 is descriptive, deployment uses fixed k.'))
    return choices


def points(args,records,seals):
    """All saved point-statistics; fresh native image/LK sample per natural clip."""
    import cv2
    cv2.setNumThreads(1);reports=[];sample_reports=[]
    for key,r in records.items():
        table={p['pair_id']:p for p in r['cat']['pairs']};seen=set();vectors=0;sample=None
        receipts={int(x['frame']):x for x in read(r['feature_folder']/'image_receipts.json')}
        with gzip.open(r['feature_folder']/'point_evidence.jsonl.gz','rt',encoding='utf8') as stream:
            for line in stream:
                row=json.loads(line);pid=row['pair_id'];assert pid not in seen;seen.add(pid)
                p=table[pid];extract=row['extract'];rep=p['representative'];f=r['features'][pid]
                assert extract['main_valid']==extract['true']['valid']==f['true']['valid']
                for kind in ('true','sham'):
                    record=extract[kind];close(record.get('feature_vector'),f[kind].get('feature_vector')) if record.get('feature_vector') is not None else None
                    if rep is None:
                        assert not record['valid'] and record['frame'] is None;continue
                    frame=rep['frame']
                    expected=frame if kind=='true' else (frame+25 if frame+26<=r['cat']['nframes'] else frame-25 if frame-25>=1 else None)
                    assert record['frame']==expected
                    if expected is None:assert not record['valid'];continue
                    assert record['next_frame']==expected+1
                    if not record['traces']:assert not record['valid'];continue
                    close(record['mean_diagonal'],np.linalg.norm(np.asarray(rep['boxes'])[:,2:],axis=1).mean())
                    for name,trace in record['traces'].items():
                        keep=np.asarray(trace['keep'],bool)
                        assert trace['accepted_count']==int(keep.sum()) and trace['detected_count']==len(trace['before'])
                        if len(keep):
                            pp=np.asarray(trace['before'],float)[keep];qq=np.asarray(trace['after'],float)[keep]
                            assert np.isfinite(pp).all() and np.isfinite(qq).all()
                            assert len(set(map(tuple,np.rint(pp).astype(int))))==len(pp)
                            assert len(set(map(tuple,np.rint(qq).astype(int))))==len(qq)
                            assert all(np.asarray(trace['status_forward'])[keep]) and all(np.asarray(trace['status_backward'])[keep])
                            assert all(np.asarray(trace['source_supported'])[keep]) and all(np.asarray(trace['target_supported'])[keep])
                            assert np.max(np.asarray(trace['fb_error'])[keep],initial=0)<=1
                    for phase in ('before','after'):
                        aa={tuple(np.rint(pt).astype(int)) for pt,k in zip(record['traces']['a'][phase],record['traces']['a']['keep']) if k}
                        bb={tuple(np.rint(pt).astype(int)) for pt,k in zip(record['traces']['b'][phase],record['traces']['b']['keep']) if k}
                        assert not aa.intersection(bb)
                    own=independent.features_from_traces(record)
                    assert (own is not None)==record['valid'],(key,pid,kind)
                    if own is not None:close(own,record['feature_vector'],2e-8);vectors+=1
                if sample is None and rep is not None and extract['true'].get('traces') and all(len(t.get('before',[])) for t in extract['true']['traces'].values()):sample=(pid,extract)
        assert seen==set(table)
        if sample is not None:
            pid,extract=sample;pair=table[pid];boxes=pair['representative']['boxes'];sampled=[]
            for kind in ('true','sham'):
                rr=extract[kind]
                if not rr.get('traces'):continue
                roi=rr['roi'];images=[]
                for date in (rr['frame'],rr['next_frame']):
                    receipt=receipts[date];path=args.repo/receipt['path'];assert sha(path)==receipt['sha256']
                    image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_GRAYSCALE)
                    assert image.shape==(receipt['height'],receipt['width'])
                    x0,y0,x1,y1=roi;images.append(image[y0:y1,x0:x1].copy());del image
                am=independent.mask_boxes(boxes[:1],roi);bm=independent.mask_boxes(boxes[1:],roi)
                native=r['rows'][r['rows'][:,0]==rr['frame'],2:6]
                allmask=independent.mask_boxes(native,roi)
                masks={'a':am & ~bm,'b':bm & ~am,'background':~(am|bm|allmask)}
                traces={name:independent.independently_track(*images,mask,np.asarray(roi[:2])) for name,mask in masks.items()}
                for name,t in traces.items():
                    original=rr['traces'][name]
                    close(t['before'],original['before'],1e-5)
                    close(t['after'],original['after'],1e-5)
                    if 'masked_image_sha256' in original:assert t['image_hashes']==original['masked_image_sha256']
                    _,support=independent.private_images(*images,masks[name])
                    if 'support_pixel_count' in original:assert int(support.sum())==original['support_pixel_count']
                    # Change only another block's private pixels, then rerun this block's LK.
                    for other in masks:
                        if other==name:continue
                        modified=[image.copy() for image in images]
                        for image in modified:image[masks[other]]=255-image[masks[other]]
                        alternate=independent.independently_track(*modified,masks[name],np.asarray(roi[:2]))
                        assert t==alternate,(key,pid,kind,name,other)
                sampled.append(kind)
            sample_reports.append(dict(clip=key,pair_id=pid,true_and_sham=sampled,all_three_blocks=True,other_exclusive_pixel_interventions=True))
        reports.append(dict(clip=key,pairs=len(seen),valid_arm_vectors_rebuilt=vectors))
        print('VERIFY209_POINT_FEATURES',key,flush=True)
    save(args.root/'verification/point_features.json',dict(passed=True,seals=seals,clips=reports,fresh_native_samples=sample_reports,scope='All point-evidence dates, retained status/support flags, rounded collision removals and four numeric statistics rebuilt from saved traces. One deterministic nonempty-point pair per clip, true/sham available dates, native image SHA, independent masked pixels/erosion/GFTT/forward+backward LK and other-block interventions freshly recomputed. This checks exclusive image inputs, not foreground identity. Non-sampled LK point extraction is receipt replay.'))


class NativeCache:
    """Exact ordered original-row geometry key; numeric P IDs are irrelevant."""
    def __init__(self,GT,base,v):
        self.GT=GT;self.base=base;self.v=v;self.alias=canonical(GT);self.frames=defaultdict(list);self.cache={};self.requests=0
        for i in np.lexsort((np.arange(len(GT)),GT[:,0],self.alias)):self.frames[int(GT[i,0])].append(i)
    def owners(self,rows,retained):
        from scipy.optimize import linear_sum_assignment
        pc=canonical(rows);dates=defaultdict(list)
        for i in np.lexsort((np.arange(len(rows)),rows[:,0],pc)):dates[int(rows[i,0])].append(i)
        out={t:np.full(len(rows),-1,np.int64) for t in ('0.5','0.3')}
        for date,indices in dates.items():
            self.requests+=1;raw=tuple(retained[i] for i in indices);key=(date,raw)
            if key not in self.cache:
                gg=self.frames.get(date,[]);answers={t:np.full(len(indices),-1,np.int64) for t in out}
                if gg:
                    matrix=self.v.official_iou_independent(self.GT[gg,2:6],self.base[list(raw),2:6]);a,b=linear_sum_assignment(-matrix)
                    for t in out:
                        ok=matrix[a,b]>=float(t);answers[t][b[ok]]=self.alias[np.asarray(gg)[a[ok]]]
                self.cache[key]=answers
            for t in out:out[t][indices]=self.cache[key][t]
        return out


def physical(args,records,choices,seals):
    replay=module('common209_independent203',args.repo/'tools/study203/verify203.py');previous,native,v,certificate=replay.helpers(args.repo)
    summary=read(args.root/'score/summary.json');reports=[];pooled={h:{a:dict(numerator=0,denominator=0,iou={t:dict(tp=0,pred=0,gt=0) for t in ('0.5','0.3')}) for a in ('K','J','U','G','F','X')} for h in DOMAINS}
    for key,r in records.items():
        checkpoint=args.root/'verification/physical_clips'/r['ref']['domain']/(r['ref']['clip']+'.json')
        closure=dict(seals=seals,source={Path(__file__).name:sha(__file__),Path(independent.__file__).name:sha(independent.__file__)},choices=choices[key])
        if checkpoint.exists():
            cached=read(checkpoint)
            assert sha(checkpoint)==checkpoint.with_suffix('.sha256').read_text().strip()
            if cached.get('closure')==closure:
                report=cached['report'];reports.append(report)
                for arm,m in report['methods'].items():
                    target=pooled[r['ref']['domain']][arm];target['numerator']+=m['numerator'];target['denominator']+=m['denominator']
                    for t in target['iou']:
                        for field in ('tp','pred','gt'):target['iou'][t][field]+=m['iou'][t][field]
                print('VERIFY209_NATIVE_CHECKPOINT',key,flush=True);continue
        rows=r['rows'];GT=r['GT'];polygon=r['meta']['polygon'];nf=r['meta']['nf'];truth=native.track_structure(GT,polygon)
        gb=replay.bins(truth['events'].values(),nf);cache=NativeCache(GT,rows,v);methods={}
        allpairs=[tuple(p['canonical_ids']) for p in r['cat']['pairs']]
        for arm,selected in choices[key].items():
            pairs=[tuple(map(int,p.split('|'))) for p in selected]
            expected,retained=independent.merge_rows(rows,r['canon'],pairs,allpairs)
            folder=args.root/'score/clips'/r['ref']['domain']/r['ref']['clip']/arm
            actual=read(folder/'score.json');disk=np.loadtxt(folder/'mot.txt',delimiter=',',usecols=range(7),ndmin=2)
            assert expected.tobytes()==disk.tobytes(),(key,arm,'materialization')
            assert actual['proof']['retained_raw_indices']==retained and np.load(folder/'retained_raw_indices.npy').tolist()==retained
            deleted=sorted(set(range(len(rows)))-set(retained));assert actual['proof']['deleted_raw_indices']==deleted
            P=native.track_structure(expected,polygon);pb=replay.bins(P['events'].values(),nf)
            own=native.score_from_owners(P,truth,cache.owners(expected,retained),polygon,nf);metric=actual['metrics']
            numerator=int(abs(pb-gb).sum());denominator=int(gb.sum())
            assert numerator==actual['numerator'] and denominator==actual['denominator']
            assert np.array_equal(pb,metric['pred_bins']) and np.array_equal(gb,metric['gt_bins'])
            assert len(P['events'])==metric['nP'] and len(truth['events'])==metric['nG']
            for t,e in (('0.5','edges05'),('0.3','edges03')):
                for field in ('tp','pred','gt','tp_id'):assert own['iou'][t][field]==metric['iou'][t][field],(key,arm,t,field)
                assert own['edges'][t]==metric['evidence'][e]
            expected_collision=sum(bool(r['collisions'][p]) for p in selected)
            assert actual['action_diagnostics']['pairs_with_at_least_one_confirmed_distinct_GT']==expected_collision
            if arm=='K':
                old=read(args.reference208_score/'clips'/r['ref']['domain']/r['ref']['clip']/'K/score.json')['metrics']
                assert old==metric
            target=pooled[r['ref']['domain']][arm];target['numerator']+=numerator;target['denominator']+=denominator
            for t in target['iou']:
                for field in ('tp','pred','gt'):target['iou'][t][field]+=metric['iou'][t][field]
            methods[arm]=dict(numerator=numerator,denominator=denominator,selected=selected,removed=len(deleted),iou={t:{f:metric['iou'][t][f] for f in ('tp','pred','gt')} for t in ('0.5','0.3')},confirmed_D_pairs=expected_collision)
        report=dict(clip=key,methods=methods,scene_frame_requests=cache.requests,unique_frame_orders=len(cache.cache));reports.append(report)
        save(checkpoint,dict(passed=True,closure=closure,report=report))
        print('VERIFY209_ALL_NATIVE',key,flush=True)
    for domain,arms in pooled.items():
        for arm,m in arms.items():
            m['W_fraction']=[100*m['numerator'],m['denominator']]
            m['F1_fraction']={t:[2*c['tp'],c['pred']+c['gt']] for t,c in m['iou'].items()}
            saved=summary['domains'][domain]['methods'][arm]
            assert saved['numerator']==m['numerator'] and saved['denominator']==m['denominator']
            for t in m['iou']:
                for field in ('tp','pred','gt'):assert m['iou'][t][field]==saved['iou'][t][field]
    W=lambda h,a:Fraction(*pooled[h][a]['W_fraction'])
    F=lambda h,a:Fraction(*pooled[h][a]['F1_fraction']['0.5'])
    checks=dict(head35_gain_at_least1=W('35','K')-W('35','X')>=1,
                head35_strictly_beats_G_F_U=all(W('35','X')<W('35',a) for a in ('G','F','U')),
                head33_no_harm_halfpoint=W('33','X')-W('33','K')<=Fraction(1,2),
                head_F1_no_harm_onehundredth=all(F(h,'X')>=F(h,'K')-Fraction(1,100) for h in ('35','33')))
    save(args.root/'verification/physical.json',dict(passed=True,seals=seals,clips=reports,pooled=pooled,registered_gate=dict(checks=checks,all_checks=all(checks.values())),scope='All36 complete MOTs independently materialized, native original-row subset/order/date/geometry/scores and fresh ID policy checked. All36 full native scene-frame Hungarian owners at both IoUs independently rebuilt using exact ordered-original-row cache; full event graphs/maxTP/TP_ID/bins and K208 exact. Only official low-level geometry and proven public197 event-scoring helpers shared. WG clips pool single-member counts, not E3.'))
    return pooled


def execute(args):
    if not args.field_go or not args.physical_finished:raise ValueError('requires root field GO and sole-CPU physical release')
    sys.path.insert(0,str(args.repo.resolve()))
    seals={stage:seal(args.root/stage,EXPECTED_SEALS[stage]) for stage in ('catalog','features','labels','predictions','score')}
    args.root.joinpath('verification').mkdir(exist_ok=True)
    records=catalog_labels(args,seals)
    points(args,records,seals)
    choices=fit_predict(args,records,seals)
    pooled=physical(args,records,choices,seals)
    source={p.name:sha(p) for p in (Path(__file__),Path(independent.__file__))}
    snapshot=args.root/'verification/source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in source:(snapshot/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    reports={name:sha(args.root/'verification'/name) for name in ('catalog_labels.json','point_features.json','models.json','numeric_candidates.json','weak_diagnostics.json','physical.json')}
    save(args.root/'verification/independent.json',dict(passed=True,prereg=PREREG,seals=seals,reports=reports,source_sha256=source,scope='See each stage for full versus sampled/native-shared scope; no producer measurement/selection/model/scorer imported.'))
    print('INDEPENDENT209_PASS',sha(args.root/'verification/independent.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('D:/beetrack_runs/common209'));p.add_argument('--source208',type=Path,default=Path('D:/beetrack_runs/switch208/inputs'));p.add_argument('--reference208-score',type=Path,default=Path('D:/beetrack_runs/switch208/score'));p.add_argument('--field-go',action='store_true');p.add_argument('--physical-finished',action='store_true')
    execute(p.parse_args())
