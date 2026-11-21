"""Independent orphan checks. Default synthetic; field entry intentionally gated.

Durable phases: lineage/candidates; labels/models; integer certificates;
materialized native scores (fresh changed geometry >=one per domain).
No production selection, realization, measurement or bound functions imported.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from fractions import Fraction
import gzip
import numpy as np
import cv2
cv2.setNumThreads(1)
from collections import defaultdict
from scipy.optimize import minimize,linear_sum_assignment


def contexts_independent(options,labels,mode,heldout):
    assert all(r['hive']!=heldout for r in options)
    groups=defaultdict(list)
    for r in sorted(options,key=lambda r:r['action_id']):
        if mode=='L' and r['type']!='extension':continue
        groups[(r['unit'],str(r['member']),str(r['orphan_key'] if mode=='L' else r['recipient']))].append(r)
    result=[]
    for key,rows in sorted(groups.items()):
        known=[r for r in rows if labels[r['action_id']]!='U']
        positive=[r['action_id'] for r in known if labels[r['action_id']]==('S' if r['type']=='extension' else 'D')]
        if not known or (not positive and len(known)!=len(rows)):continue
        first=rows[0];kind='positive' if positive else 'known-null'
        result.append(dict(context=list(key),hive=first['hive'],group=str(first['group']),clip=first['clip'],kind=kind,options=known,
            target=[int(r['action_id'] in positive)/max(1,len(positive)) for r in known]+[int(not positive)],
            omitted_U=[r['action_id'] for r in rows if labels[r['action_id']]=='U']))
    # Hierarchical mass is split only among nonempty child groups.
    leaves=defaultdict(list)
    for c in result:leaves[(c['hive'],c['group'],c['clip'],c['kind'])].append(c)
    hives={key[0] for key in leaves}
    for key,cs in leaves.items():
        h,g,v,k=key
        ng=len({q[1] for q in leaves if q[0]==h})
        nv=len({q[2] for q in leaves if q[:2]==(h,g)})
        nk=len({q[3] for q in leaves if q[:3]==(h,g,v)})
        for c in cs:c['weight']=1/(len(hives)*ng*nv*nk*len(cs))
    return result


def refit_independent(options,labels,mode,heldout):
    """Independent convex softmax fit, including intercept penalty .2."""
    cs=contexts_independent(options,labels,mode,heldout);heads={};slices={};dim=0
    for typ in (('extension',) if mode=='L' else ('extension','cut')):
        observations=[(r,c['weight']/len(c['options'])) for c in cs for r in c['options'] if r['type']==typ]
        if not observations:heads[typ]=dict(status='empty');continue
        X=np.asarray([r['features'] for r,w in observations]);w=np.asarray([w for r,w in observations]);mass=w.sum();w=w/mass
        mean=np.sum(w[:,None]*X,axis=0);constant=np.all(X==X[0],axis=0);mean[constant]=X[0,constant]
        scale=np.sqrt(np.sum(w[:,None]*(X-mean)**2,axis=0));scale[constant | (scale==0)]=1
        heads[typ]=dict(status='fitted',mean=mean,scale=scale,training_mass=float(mass));slices[typ]=slice(dim,dim+X.shape[1]+1);dim+=X.shape[1]+1
    blocks=[]
    for c in cs:
        A=np.zeros((len(c['options'])+1,dim))
        for i,r in enumerate(c['options']):
            h=heads[r['type']];A[i,slices[r['type']]]=np.r_[(r['features']-h['mean'])/h['scale'],1]
        blocks.append((A,np.asarray(c['target']),c['weight']))
    def loss(theta):
        value=.1*float(theta@theta);gradient=.2*theta
        for A,y,w in blocks:
            z=A@theta;largest=z.max();exp=np.exp(z-largest);prob=exp/exp.sum()
            value+=w*(largest+np.log(exp.sum())-y@z);gradient+=w*((prob-y)@A)
        return value,gradient
    if dim:
        answer=minimize(loss,np.zeros(dim),jac=True,method='BFGS',options=dict(gtol=2e-9,maxiter=3000))
        assert np.max(np.abs(loss(answer.x)[1]))<2e-7
        for typ,sl in slices.items():heads[typ].update(beta=answer.x[sl][:-1],intercept=float(answer.x[sl][-1]))
    return dict(heads=heads,contexts=cs,objective=loss(answer.x)[0] if dim else 0)


def choose_independent(options,scores):
    groups=defaultdict(list);selected=[]
    for r in sorted(options,key=lambda r:r['action_id']):
        score=scores[r['action_id']]
        if score is not None and score>0:groups[r['unit'],str(r['member'])].append(r)
    for rows in groups.values():
        parents=sorted({str(r['recipient']) for r in rows})
        right=lambda r:('orphan',str(r['orphan_key'])) if r['type']=='extension' else ('cut',r['action_id'])
        columns=sorted({right(r) for r in rows})+[('null',p) for p in parents]
        pi={p:i for i,p in enumerate(parents)};ci={c:i for i,c in enumerate(columns)}
        M=np.full((len(parents),len(columns)),-np.inf);owners={}
        for p in parents:M[pi[p],ci['null',p]]=0
        for r in rows:
            ij=pi[str(r['recipient'])],ci[right(r)]
            if scores[r['action_id']]>M[ij]:M[ij]=scores[r['action_id']];owners[ij]=r['action_id']
        a,b=linear_sum_assignment(M,maximize=True)
        selected.extend(owners[i,j] for i,j in zip(a,b) if (i,j) in owners)
    return sorted(selected)


def shuffle_independent(options,scores):
    groups=defaultdict(list);out={}
    for r in options:groups[r['unit'],str(r['member']),str(r['orphan_key'])].append(r['action_id'])
    for ids in groups.values():
        sources=sorted(ids);targets=sorted(ids,key=lambda k:(hashlib.sha256(('orphan-recipient-null-v1|'+k).encode()).hexdigest(),k))
        out.update((dst,scores[src]) for src,dst in zip(sources,targets))
    return out


def phase_bins(dated,nframes):
    result=[]
    for offset in range(30):
        cells=np.zeros(((nframes+25*offset+749)//750,2),np.int64)
        for frame,direction in dated:cells[(int(frame)-1+25*offset)//750,int(direction<0)]+=1
        result.extend(cells.tolist())
    return np.asarray(result,np.int64)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sealed(root,expected=None):
    root=Path(root);digest=sha(root/'manifest.json')
    assert digest==(root/'SEALED.sha256').read_text().strip()
    if expected is not None:assert digest==expected
    for name,value in read(root/'manifest.json').items():
        path=(root/name).resolve();assert path.is_relative_to(root.resolve()) and sha(path)==value
    return digest


def own_helpers(repo):
    import importlib.util
    def load(name,path):
        s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    u=load('independent197_orphan',repo/'tools/study197/verify.py')
    v=load('independent199_orphan',repo/'tools/study199/verify199.py')
    return u,v


def features_independent(core,short,side,polygon):
    core=np.asarray(core,float);short=np.asarray(short,float)
    t=core[:,0];dt=t-t.mean();c=core[:,2:4]+core[:,4:6]/2
    slope=(dt[:,None]*(c-c.mean(axis=0))).sum(axis=0)/(dt@dt)
    diag=np.median(np.sqrt((core[:,4:6]**2).sum(axis=1)))
    predicted=c.mean(axis=0)+(short[:,0]-t.mean())[:,None]*slope
    error=np.sqrt(((predicted-short[:,2:4]-short[:,4:6]/2)**2).sum(axis=1))/diag
    a=core[0] if side=='before' else core[-1];b=short[-1] if side=='before' else short[0]
    fit=c.mean(axis=0)+dt[:,None]*slope
    return [abs(a[0]-b[0])/12,len(short)/2,float(side=='before'),float(error.mean()),float(error.max()),
        float(np.linalg.norm(a[2:4]+a[4:6]/2-b[2:4]-b[4:6]/2)/diag),float(abs(np.log(b[4]*b[5]/(a[4]*a[5])))),
        float(abs(np.log((b[4]/b[5])/(a[4]/a[5])))),float(np.sqrt(((fit-c)**2).sum(axis=1).mean())/diag),
        float(inside([a],polygon)[0]!=inside([b],polygon)[0])]


def trace_stage1_independent(D,nf,threshold):
    """Direct native tracker index stream; never infer provenance by geometry."""
    from beetrack.track import Tracker,Dets,nms
    tracker=Tracker('link',link=0,min_len=0);selected=np.flatnonzero(D[:,5]>=threshold);rows=[];indices=[]
    for frame in range(1,nf+1):
        ix=selected[D[selected,0]==frame];boxes=D[ix,1:6].astype(np.float32)
        order=nms(boxes,tracker.args.nms) if tracker.args.nms else np.arange(len(boxes));boxes=boxes[order];ix=ix[order]
        output=tracker.t.update(Dets(boxes))
        for row in output:
            j=int(row[7]);assert row[7]==j and 0<=j<len(boxes)
            x1,y1,x2,y2,confidence=boxes[j]
            # Official coordinates and width subtraction retain float32 arithmetic.
            rows.append([frame,row[4],x1,y1,x2-x1,y2-y1,row[5]])
            indices.append(int(ix[j]))
        assert not getattr(tracker.t,'late',[])
    return np.asarray(rows,float).reshape(-1,7),np.asarray(indices,np.int64)


def audit_candidates(args):
    """Durable all60 lineage, original candidates, features and complete deltas."""
    from beetrack.p0 import stage1,p0,link_model
    from beetrack.link import link
    from beetrack.p0 import THR
    root=args.root;freeze=root/'frozen';digest=sealed(freeze,args.frozen_sha)
    source195_digest=sha(args.source195/'manifest.json')
    assert source195_digest=='4f78f24d7fd13dc2eca9b206501a5326b4e42d476e1a9c4dd1ba98fe4bf61377' and source195_digest==(args.source195/'SEALED.sha256').read_text().strip()
    manifest195=read(args.source195/'manifest.json')
    sealed(args.source197,'e7fd48078f8d0a32bb9018f2bac93dc4b5270d374d5bcd0a631fa97dd4d790e4')
    data=read(freeze/'deployment.json');old195=read(args.source195/'deployment.json')
    old197=read(args.source197/'deployment.json');cuts={(g['unit'],m['member_index']):m['bundle'] for g in old197['groups'] for m in g['members']}
    assert data['prereg']==args.prereg and len(data['members'])==60
    assert {r['source195_unit'] for r in data['members']}==set(range(60))
    assert {h:sum(r['hive']==h for r in data['members']) for h in ('35','33','W','G')}==dict(zip(('35','33','W','G'),(12,12,18,18)))
    ancestry={uid:(g,slot) for g in old195['groups'] for slot,uid in enumerate(g['units'])}
    for ref in data['members']:
        old_unit=old195['units'][ref['source195_unit']];member=old195['members'][old_unit['member']];g,slot=ancestry[ref['source195_unit']];clip=old195['clips'][g['clip_index']]
        assert str(old_unit['member'])==str(ref['member']) and ref['member_index']==slot
        assert ref['seed']==member['seed'] and 0<=ref['seed']<15 and ref['tag']==member['tag']
        assert ref['hive']==g['hive'] and ref['group']==g['group'] and ref['clip']==clip['clip'] and ref['split']==clip['split']!='test'
        assert ref['nf']==clip['nframes'] and ref['polygon']==clip['polygon']
    u,v=own_helpers(args.repo);out=root/'verification/candidates';out.mkdir(parents=True,exist_ok=True);reports=[]
    for ref in data['members']:
        folder=freeze/ref['path'];sealed(folder,ref['seal']);dest=out/(ref['path']+'.json')
        identity=dict(member_seal=ref['seal'],verifier_sha=sha(__file__))
        if dest.exists():
            saved=read(dest);assert saved['identity']==identity and sha(dest)==dest.with_suffix('.sha256').read_text().strip();reports.append(saved);continue
        cat=read(folder/'catalog.json');meta=read(folder/'receipt.json');a=np.load(folder/'rows.npz',allow_pickle=False)
        assert meta['source_det_path']==f"runs/dets/{ref['tag']}/{ref['split']}/{ref['clip']}.npy"
        detpath=args.repo/meta['source_det_path'];assert sha(detpath)==meta['identity']['detector_sha256']
        D=np.load(detpath,allow_pickle=False);nf=int(ref['nf']);threshold=.5 if ref['hive'] in ('35','33') else .4
        native=stage1(D[D[:,5]>=threshold],nf);traced,ix=trace_stage1_independent(D,nf,threshold)
        assert np.array_equal(native,traced) and np.array_equal(native,a['full_stage1']) and np.array_equal(ix,a['stage1_D_indices'])
        previous=old195['units'][ref['source195_unit']];arm=previous['arms']['t50' if threshold==.5 else 't40']
        assert meta['identity']['detector_sha256']==arm['source_receipt']['detector_sha256']
        source=args.source195/arm['rows_file'];assert sha(source)==meta['identity']['source195_rows_sha256']
        assert sha(source)==manifest195[arm['rows_file']]
        h=np.load(source,allow_pickle=False);K=h['rows'];assert np.array_equal(K,a['original_K'])
        model=arm['source_receipt']['linker_name'];prob=link_model(model)
        assert model==('m_no'+ref['hive'] if ref['hive'] in ('W','G') else 'm_dev_strict')
        assert sha(args.repo/'beetrack/models'/(model+'.json'))==meta['identity']['linker_sha256']
        assert np.array_equal(np.asarray(p0(native,prob)).reshape(-1,7),K)
        tagged=np.c_[native,np.arange(len(native))];linked=np.asarray(sorted(link(tagged.tolist(),THR,prob).tolist())).reshape(-1,8)
        assert np.array_equal(linked[:,:7],a['linked']) and np.array_equal(linked[:,7],a['linked_s1_indices'])
        native_linked=np.asarray(sorted(link(native.tolist(),THR,prob).tolist())).reshape(-1,7)
        assert np.array_equal(native_linked,a['linked'])
        proof=lineage(D,native,ix,a['linked'],a['linked_s1_indices'],K,h['s1'])
        C,pos,blocks=u.canonicalize(K);baseline=u.realize_independent(K,C,pos,blocks,[],[])
        assert np.array_equal(baseline,a['baseline']) and np.array_equal(baseline,cat['baseline'])
        linked_ids,linked_n=np.unique(a['linked'][:,1],return_counts=True);keep=np.isin(a['linked'][:,1],linked_ids[linked_n>=3])
        expected_sources=[(meta['identity']['detector_sha256'],int(j)) for j in ix[a['linked_s1_indices'][keep].astype(int)]]
        assert list(map(tuple,cat['baseline_sources']))==expected_sources
        expected_short=[]
        for oid in linked_ids[linked_n<3]:
            mask=a['linked'][:,1]==oid;rr=a['linked'][mask];jj=a['linked_s1_indices'][mask].astype(int)
            if len(rr)>1 and np.any(np.diff(rr[:,0])>25):continue
            expected_short.append((int(oid),rr,jj))
        assert len(expected_short)==len(cat['orphans'])
        for oi,(oid,rr,jj) in enumerate(expected_short):
            old=cat['orphans'][oi];physical=np.asarray(old['rows']);assert old['linked_id']==oid
            assert np.array_equal(physical[:,[0,2,3,4,5,6]],rr[:,[0,2,3,4,5,6]])
            assert np.all(physical[:,1]==baseline[:,1].max()+1+old['original_orphan_index'])
            assert list(map(tuple,old['sources']))==[(meta['identity']['detector_sha256'],int(j)) for j in ix[jj]]
            assert np.array_equal(old['source_s1_index'],jj)
        assert cat['cut_catalog']==cuts[ref['unit'],ref['member_index']]
        rebuilt=u.rebuild_candidates(K,ref['polygon'],nf)
        # Existing independent197 rebuild covers geometry/G9/T12; source bundle
        # comparison makes all original1196 choices traceable, no peer filtering.
        candidates=cat['cut_catalog']['candidates'];assert len(rebuilt['candidates'])==len(candidates)
        old_by_parent={c['parent_id']:c for c in candidates}
        for c in rebuilt['candidates']:
            old=old_by_parent[c['parent_id']]
            for key in ('delta_windows','features_G','features_T','cut_after_canonical_row','end','k','span'):
                if key.startswith('features'):np.testing.assert_allclose(c[key],old[key],atol=2e-11,rtol=2e-11)
                else:assert np.array_equal(c[key],old[key])
        orphan_rows=[np.asarray(o['rows'],float) for o in cat['orphans']]
        expected=catalog(baseline,orphan_rows);lookup={(e['recipient'],e['orphan_index'],e['side']):e for e in cat['edges']}
        assert set(lookup)=={(e['recipient'],e['orphan_index'],e['side']) for e in expected}
        for e in expected:
            saved=lookup[e['recipient'],e['orphan_index'],e['side']];short=orphan_rows[e['orphan_index']]
            rank=(e['recipient']-1)//2;assert e['recipient']==2*rank+1
            assert saved['parent_rank']==rank and saved['parent_canonical_id']==blocks[rank]['parent_id']
            parent_indices=np.flatnonzero(baseline[:,1]==e['recipient']);parent_indices=parent_indices[np.argsort(baseline[parent_indices,0],kind='stable')]
            parent=baseline[parent_indices];core_ix=parent_indices[:3] if e['side']=='before' else parent_indices[-3:];core=baseline[core_ix]
            assert np.array_equal(saved['core_K_row_indices'],core_ix) and saved['endpoint_K_row_index']==int(core_ix[0] if e['side']=='before' else core_ix[-1])
            assert saved['gap']==e['gap'] and abs(saved['normalized_error']-e['normalized_error'])<2e-11
            assert saved['recipient_cut_join_key']==f"{ref['unit']}|{ref['member_index']}|{saved['parent_canonical_id']}"
            np.testing.assert_allclose(features_independent(core,short,e['side'],ref['polygon']),saved['features10'],atol=2e-11,rtol=2e-11)
            added=short.copy();added[:,1]=e['recipient'];after=np.r_[parent,added]
            delta=phase_bins(events(after,ref['polygon']),nf)-phase_bins(events(parent,ref['polygon']),nf)
            assert np.array_equal(delta,saved['delta_windows'])
        result=dict(passed=True,identity=identity,unit=ref['unit'],member=ref['member'],lineage=proof,cuts=len(candidates),extensions=len(expected),native_stage1_fresh=True,native_K_fresh=True,scope='All rows, all candidate geometries and30phase deltas; no GT read')
        receipt(dest,result);reports.append(result);print('INDEPENDENT_ORPHAN_CANDIDATES',ref['path'],flush=True)
    final=dict(passed=True,frozen_manifest=digest,members=len(reports),cuts=sum(r['cuts'] for r in reports),extensions=sum(r['extensions'] for r in reports),scope='GT-free all60 complete native input provenance/candidates/three fresh native reconstruction checks. Existing detector and historical linker learned weights are fingerprinted and used, not independently retrained.',members_receipts=[sha(out/(ref['path']+'.json')) for ref in data['members']])
    receipt(root/'verification/candidates.json',final);return final


def audit_source_allowlist(args):
    """Join already-verified detector digests to the historical F60 receipt.

    This metadata-only supplement never opens another detector/arm matrix.
    """
    old=read(args.source195/'deployment.json');frozen=read(args.root/'frozen/deployment.json');checked=[]
    assert sha(args.source195/'manifest.json')=='4f78f24d7fd13dc2eca9b206501a5326b4e42d476e1a9c4dd1ba98fe4bf61377'
    assert frozen['source195_seal']==sha(args.source195/'manifest.json')
    for ref in frozen['members']:
        meta=read(args.root/'frozen'/ref['path']/'receipt.json')
        expected=f"runs/dets/{ref['tag']}/{ref['split']}/{ref['clip']}.npy"
        assert meta['source_det_path']==expected and ref['split']!='test' and ref['seed']<15
        arm=old['units'][ref['source195_unit']]['arms']['t50' if ref['hive'] in ('35','33') else 't40']
        assert meta['identity']['detector_sha256']==arm['source_receipt']['detector_sha256']
        assert meta['identity']['linker_sha256']==arm['source_receipt']['linker_sha256']
        checked.append(dict(unit=ref['unit'],member=ref['member'],source_det_path=expected,detector_sha256=meta['identity']['detector_sha256']))
    assert len(checked)==60
    result=dict(passed=True,streams=60,verifier_sha256=sha(__file__),records=checked,scope='All60 exact whitelisted detector paths and detector/linker digests joined to pinned historical195 F arm receipts; underlying arrays and native index chains were already independently checked in candidate phase.')
    receipt(args.root/'verification/source_allowlist.json',result);return result


def option_inventory(frozen):
    options=[];members=[]
    for ref in read(frozen/'deployment.json')['members']:
        cat=read(frozen/ref['path']/'catalog.json');members.append(ref)
        meta=dict(unit=ref['unit'],member_index=ref['member_index'],member=str(ref['member']),hive=ref['hive'],group=ref['group'],clip=ref['clip'])
        for e in cat['edges']:
            options.append(dict(meta,action_id=e['edge_id'],type='extension',features=e['features10'],recipient=e['recipient_cut_join_key'],orphan_key=f"{ref['unit']}|{ref['member_index']}|orphan:{e['orphan_linked_id']}",edge_id=e['edge_id'],parent_canonical_id=e['parent_canonical_id'],gap=e['gap'],max_error=e['normalized_error']))
        for c in cat['cut_catalog']['candidates']:
            options.append(dict(meta,action_id='cut:'+c['candidate_id'],type='cut',candidate_id=c['candidate_id'],recipient=c['candidate_id'],features=list(c['features_G'])+[float(c['delta']),float(c['direction']==1)],delta=c['delta'],direction=c['direction'],parent_canonical_id=c.get('parent_canonical_id',c['parent_id'])))
    assert len({r['action_id'] for r in options})==len(options)
    return options,members


def audit_models(args):
    sealed(args.root/'frozen',args.frozen_sha);ps=sealed(args.root/'predictions',args.predictions_sha);os=sealed(args.root/'owners',args.owners_sha)
    options,members=option_inventory(args.root/'frozen');results={}
    for heldout in ('35','33','W','G'):
        source=[r for r in options if r['hive']!=heldout];labels={};opened=[]
        for hive in ('35','33','W','G'):
            if hive==heldout:continue
            shard=read(args.root/'owners'/(hive+'.json'));assert shard['domain']==hive;opened.append(hive)
            for g in shard['units']:
                for m in g['members']:
                    for r in [*m['edges'],*m['cuts']]:labels[r['action_id']]=r['labels']['0.5']['class']
        assert set(labels)=={r['action_id'] for r in source}
        published=read(args.root/'predictions'/(heldout+'.model.json'));mask=read(args.root/'predictions'/(heldout+'.masks.json'))
        assert published['heldout']==heldout and set(published['source_domains'])==set(opened)
        fitted={}
        for mode in ('L','J'):
            ss=[r for r in source if mode=='J' or r['type']=='extension'];ls={r['action_id']:labels[r['action_id']] for r in ss}
            result=refit_independent(ss,ls,mode,heldout);saved=published[mode]
            assert saved['heldout']==heldout and saved['mode']==mode and saved['ridge']==.2
            assert saved['source_label_classes']==ls and saved['contexts']==result['contexts']
            assert abs(saved['optimization']['objective']-result['objective'])<1e-10
            for typ,h in result['heads'].items():
                assert saved['heads'][typ]['status']==h['status']
                if h['status']=='fitted':
                    for key in ('mean','scale','beta','intercept'):np.testing.assert_allclose(h[key],saved['heads'][typ][key],atol=5e-7,rtol=5e-7)
            fitted[mode]=result
        target=[r for r in options if r['hive']==heldout];ext=[r for r in target if r['type']=='extension'];score={}
        # Deploy saved coefficients after independent convex refit agreement;
        # raw binary64 tie policy is then checked exactly, no tolerance scores.
        for mode,rows in (('L',ext),('J',target)):
            score[mode]={}
            for r in rows:
                h=published[mode]['heads'].get(r['type'])
                score[mode][r['action_id']]=None if h is None or h['status']=='empty' else float(((np.asarray(r['features'])-h['mean'])/h['scale'])@h['beta']+h['intercept'])
        score['G']={r['action_id']:1-r['features'][4]/2-r['features'][0] for r in ext};score['M']=shuffle_independent(ext,score['L'])
        choices={arm:choose_independent(target if arm=='J' else ext,score[arm]) for arm in ('G','L','M','J')};choices['K']=[]
        assert len(mask['member_clips'])==sum(r['hive']==heldout for r in members)
        for m in mask['member_clips']:
            rows=[r for r in target if r['unit']==m['unit'] and str(r['member'])==str(m['member'])];ids={r['action_id'] for r in rows}
            assert m['options']==sorted(rows,key=lambda r:r['action_id'])
            for arm in ('K','G','L','M','J'):
                selected=[r for r in rows if r['action_id'] in choices[arm]]
                assert sorted(m['selected'][arm]['edge_ids'])==sorted(r['action_id'] for r in selected if r['type']=='extension')
                assert sorted(m['selected'][arm]['cut_ids'])==sorted(r['candidate_id'] for r in selected if r['type']=='cut')
                if arm!='K':assert m['scores'][arm]=={k:v for k,v in score[arm].items() if k in ids}
        results[heldout]=dict(source_domains_opened=opened,L_contexts=len(fitted['L']['contexts']),J_contexts=len(fitted['J']['contexts']),choices={k:len(v) for k,v in choices.items()})
    final=dict(passed=True,predictions_manifest=ps,owners_manifest=os,folds=results,scope='All4source-only independent refits, hierarchy/normalization and all5raw-score masks/M assignments; archived producer source-access history is code/receipt evidence, not a runtime access monitor')
    receipt(args.root/'verification/models.json',final);return final


def union_independent(cat,u):
    pieces=[np.asarray(cat['baseline'],float)];keys=list(map(tuple,cat['baseline_sources']));orphan_keys={}
    used=sorted({e['orphan_index'] for e in cat['edges']})
    for oi in used:
        o=cat['orphans'][oi];rr=np.asarray(o['rows'],float);order=np.argsort(rr[:,0],kind='stable')
        pieces.append(rr[order]);ss=[tuple(o['sources'][i]) for i in order];keys.extend(ss);orphan_keys[oi]=ss
    raw=np.concatenate(pieces);C,pos,blocks=u.canonicalize(raw);ordered=[keys[i] for i in pos]
    assert len(set(keys))==len(keys);lookup={key:i for i,key in enumerate(ordered)}
    return C,ordered,[lookup[k] for k in keys[:len(cat['baseline'])]],{oi:[lookup[k] for k in ss] for oi,ss in orphan_keys.items()}


def union_owners_independent(gt,P,threshold,needed,v):
    """Same independently verified199 arithmetic, with cached frame indices."""
    G,gi=v.canonical_with_origin(gt);P,pi=v.canonical_with_origin(P);originals={}
    for i,row in enumerate(G):originals.setdefault(int(row[1]),set()).add(int(gt[gi[i],1]))
    rows={i:dict(prediction_index=i,raw_prediction_index=int(pi[i]),frame=int(P[i,0]),pred_id=int(P[i,1]),owner=None,owner_row=None,original_gt_id=None,pred_degree=0,gt_degree=0,iou=None,isolated=False,graph_gt=[]) for i in needed}
    def frame_indices(a):
        order=np.argsort(a[:,0],kind='stable');dates,first=np.unique(a[order,0],return_index=True)
        return {int(date):order[lo:hi] for date,lo,hi in zip(dates,first,[*first[1:],len(order)])}
    gp,pp=frame_indices(G),frame_indices(P);reverse={}
    for date in sorted({int(P[i,0]) for i in needed}):
        gg=gp.get(date,np.empty(0,int));pr=pp[date]
        if not len(gg):continue
        matrix=v.official_iou_independent(G[gg,2:6],P[pr,2:6]);E=matrix>=threshold;degreeP=E.sum(0);degreeG=E.sum(1)
        for j,i in enumerate(pr):
            if i not in rows:continue
            rows[i]['pred_degree']=int(degreeP[j]);rows[i]['graph_gt']=[dict(owner=int(G[gg[k],1]),original_gt_id=next(iter(originals[int(G[gg[k],1])])) if len(originals[int(G[gg[k],1])])==1 else None,iou=float(matrix[k,j]),gt_degree=int(degreeG[k])) for k in np.flatnonzero(E[:,j])]
        aa,bb=linear_sum_assignment(-matrix)
        for a,b in zip(aa,bb):
            i=int(pr[b])
            if i not in rows or not E[a,b]:continue
            pid=int(G[gg[a],1]);oid=next(iter(originals[pid])) if len(originals[pid])==1 else None
            rows[i].update(owner=pid,owner_row=int(gg[a]),original_gt_id=oid,gt_degree=int(degreeG[a]),iou=float(matrix[a,b]),isolated=bool(degreeP[b]==1 and degreeG[a]==1 and oid is not None))
            reverse[i]=[(int(pr[j]),float(matrix[a,j]),int(degreeP[j])) for j in np.flatnonzero(E[a])]
    return dict(G=G,P=P,gt_origin=gi,pred_origin=pi,rows=rows,threshold=threshold,original_ids=originals,reverse=reverse)


def audit_owners(args):
    from beetrack.evaluate import load_tracks
    frozen=args.root/'frozen';os=sealed(args.root/'owners',args.owners_sha);sealed(frozen,args.frozen_sha)
    u,v=own_helpers(args.repo);source199=read(args.source199/'independent_owner_ledger.json')
    previous=read(args.source199/'independent_owners.json');assert previous['owner_ledger_sha256']==sha(args.source199/'independent_owner_ledger.json')
    sealed(args.source199/'owners','0343fa154949939f59c55d7f0a350110b9f302bff9d7b5afdfd62debf13e0c9d')
    oldGT=read(args.source199/'owners/summary.json')['GT_sha256']
    published={};original_cuts={};reports=[];GT={};out=args.root/'verification/owners';out.mkdir(parents=True,exist_ok=True)
    for hive in ('35','33','W','G'):
        shard=read(args.root/'owners'/(hive+'.json'))
        old_shard=read(args.source199/'owners'/(hive+'.json'))
        original_cuts.update({c['candidate_id']:c for g in old_shard['units'] for m in g['members'] for c in m['records']})
        for g in shard['units']:
            for m in g['members']:published[g['unit'],m['member_index']]=(g,m)
    for ref in read(frozen/'deployment.json')['members']:
        saved_group,saved=published[ref['unit'],ref['member_index']];dest=out/(ref['path']+'.json')
        identity=dict(owner_manifest=os,member_seal=ref['seal'],verifier_sha=sha(__file__))
        if dest.exists():
            result=read(dest);assert result['identity']==identity and sha(dest)==dest.with_suffix('.sha256').read_text().strip();reports.append(result);continue
        cat=read(frozen/ref['path']/'catalog.json');P,keys,Kmap,Omap=union_independent(cat,u)
        if ref['clip'] not in GT:
            path=args.repo/'runs/gt'/ref['split']/ref['clip']/'gt/gt.txt';assert sha(path)==oldGT[path.relative_to(args.repo).as_posix()];GT[ref['clip']]=load_tracks(path)
        gt=GT[ref['clip']];needed=set()
        for e in cat['edges']:needed.add(Kmap[e['endpoint_K_row_index']]);needed.update(Omap[e['orphan_index']])
        byid={e['edge_id']:e for e in saved['edges']};assert set(byid)=={e['edge_id'] for e in cat['edges']}
        for e in cat['edges']:
            published_edge=byid[e['edge_id']]
            assert published_edge['action_id']==e['edge_id'] and published_edge['parent_key']==e['recipient_cut_join_key']
            assert published_edge['orphan_key']==f"{ref['unit']}|{ref['member_index']}|orphan:{e['orphan_linked_id']}"
            assert published_edge['unit']==ref['unit'] and published_edge['member_index']==ref['member_index']
            assert published_edge['source_scene']=='K_plus_all_referenced_orphans_union'
            assert np.array_equal(published_edge['delta_windows'],e['delta_windows'])
        classes={};frame_count=0
        for threshold in (.5,.3):
            owners=union_owners_independent(gt,P,threshold,needed,v);assert np.array_equal(owners['P'],P)
            seg=v.GT_segments(owners,ref['polygon']);rows={}
            for i in needed:
                z=v.independent_row_record(owners,i,P,np.arange(len(P)),ref['polygon'],seg)
                neighbors=[];date=int(P[i,0]);owner=z['official_owner'];G=owners['G']
                if owner is not None:
                    neighbors=[dict(union_row=j,source_key=list(keys[j]),pred_key=[date,int(P[j,1])],iou=value,pred_degree=degree) for j,value,degree in owners['reverse'][i]]
                rows[i]=dict(union_row=i,source_key=list(keys[i]),pred_key=z['pred_key'],row_sha256=z['row_sha256'],official_owner=owner,original_gt_id=z['owner_gt_original_id'],gt_run=z['owner_gt_run'],gt_key=z['owner_gt_key'],gt_row_sha256=z['owner_gt_row_sha256'],gt_raw_index=z['owner_gt_raw_index'],iou=z['iou'],pred_degree=z['pred_degree'],gt_degree=z['gt_degree'],isolated=z['isolated'],qualified=z['qualified'],pred_inside=z['pred_inside'],gt_inside=z['gt_inside'],graph_gt=z['graph_gt'],owner_pred_neighbors=neighbors)
            counts=defaultdict(int)
            for e in cat['edges']:
                left=rows[Kmap[e['endpoint_K_row_index']]];rr=[rows[i] for i in Omap[e['orphan_index']]];allrows=[left,*rr]
                label='U'
                if all(r['qualified'] for r in allrows):
                    if len({r['official_owner'] for r in allrows})==1:label='S'
                    elif len({r['original_gt_id'] for r in allrows})>1:label='D'
                published_label=byid[e['edge_id']]['labels'][str(threshold)]
                assert published_label['class']==label
                assert published_label['orphan_original_conflict']==(len({r['original_gt_id'] for r in rr if r['qualified']})>1)
                assert published_label['orphan_run_conflict']==(len({r['official_owner'] for r in rr if r['qualified']})>1)
                v.compare(left,published_label['recipient_row']);v.compare(rr,published_label['orphan_rows']);counts[label]+=1
            classes[str(threshold)]=dict(counts);frame_count+=len({int(P[i,0]) for i in needed})
        assert saved['union_proof']['union_rows_sha256']==v.array_digest(P)
        keysha=lambda values:hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
        assert saved['union_proof']['canonical_source_order_sha256']==keysha(keys)
        assert saved['union_proof']['frame_source_order_sha256']==keysha([keys[i] for i in np.argsort(P[:,0],kind='stable')])
        for cut in saved['cuts']:
            old=source199[cut['candidate_id']]
            assert cut['labels']==original_cuts[cut['candidate_id']]['labels'] and cut['source_scene']=='sealed199_nativeK'
            projected={t:{key:value for key,value in label.items() if key!='reason'} for t,label in cut['labels'].items()}
            assert projected==old['labels']  # only a documented explanatory string is absent in the independent199 ledger
            assert cut['action_id']=='cut:'+cut['candidate_id'] and cut['parent_key']==cut['candidate_id'] and cut['orphan_key']=='cut-dummy|'+cut['candidate_id']
            for key in ('unit','member_index','frame','direction','delta','delta_windows','end','k'):
                assert np.array_equal(cut[key],old[key])
        opportunities={x['candidate_id']:x for x in saved['truncation_opportunities']};specimens=[]
        for cut in saved['cuts']:
            censor=cut['labels']['0.5']['censor']
            if not censor['supported']:assert cut['candidate_id'] not in opportunities;continue
            run=censor['gt_run'];op=opportunities.pop(cut['candidate_id']);possible=[e for e in saved['edges'] if e['parent_key']==cut['candidate_id']]
            same=[];restored=[];side='after' if censor['type']=='later_return' else 'before'
            for e in possible:
                lab=e['labels']['0.5']
                if lab['class']=='S' and lab['recipient_row']['official_owner']==run['canonical_gt_id']:
                    same.append(e['edge_id']);outer=lab['orphan_rows'][-1] if e['side']=='after' else lab['orphan_rows'][0]
                    if e['side']==side and outer['gt_inside']==run['start_inside'] and outer['pred_inside']==run['start_inside']:restored.append(e['edge_id'])
            assert op['natural_edge_count']==len(possible) and op['same_GT_continuation_ids']==same and op['restores_required_endpoint_side_ids']==restored
            assert op['has_same_GT_continuation']==bool(same) and op['restores_required_endpoint_side']==bool(restored)
            assert op['gt_segment_key']==[ref['clip'],run['original_gt_id'],run.get('run')]
            specimens.append(dict(clip=ref['clip'],canonical_GT_run=run['canonical_gt_id'],original_GT_ID=run['original_gt_id'],run=run.get('run'),type=censor['type'],direction=cut['direction'],same=bool(same),restored=bool(restored)))
        assert not opportunities
        result=dict(passed=True,identity=identity,edges=len(cat['edges']),cuts=len(saved['cuts']),classes=classes,full_frame_dates_both_thresholds=frame_count,union_rows=len(P),censor_specimens=specimens,hive=ref['hive'],scope='All union rows/source order and all referenced observations with full scene competitors; original cut labels byte-equivalent verified199 ledger')
        receipt(dest,result);reports.append(result);print('INDEPENDENT_ORPHAN_OWNERS',ref['path'],flush=True)
    dedup={}
    for hive in ('35','33','W','G'):
        ss=[s for r in reports if r['hive']==hive for s in r['censor_specimens']];unique=defaultdict(list)
        for s in ss:unique[s['clip'],s['canonical_GT_run']].append(s)
        dedup[hive]=dict(opportunities=len(ss),unique_runs=len(unique),same_GT_continuation_runs=sum(any(s['same'] for s in group) for group in unique.values()),restores_endpoint_runs=sum(any(s['restored'] for s in group) for group in unique.values()),
            by_type={typ:dict(opportunities=sum(s['type']==typ for s in ss),unique_runs=len({(s['clip'],s['canonical_GT_run']) for s in ss if s['type']==typ})) for typ in ('earlier_part_missed','later_return')})
    final=dict(passed=True,owners_manifest=os,members=len(reports),extensions=sum(x['edges'] for x in reports),cuts=sum(x['cuts'] for x in reports),censor_deduplicated=dedup,scope='All union labels/reverse ownership graphs/both IoUs and199censor natural continuation/run dedup. S is the specified endpoint+orphan local annotation evidence, not full-track/biological identity. U may include valid associations competing in this fixed union. No target labels used in refit phase.')
    receipt(args.root/'verification/owners.json',final);return final


def audit_capacity(args):
    digest=sealed(args.root/'capacity',args.capacity_sha);saved=read(args.root/'capacity/capacity.json');shards=[read(args.root/'owners'/(h+'.json')) for h in ('35','33')];reports=[];knownS={};comparisons=[]
    sealed(args.source199/'capacity_v2','47b704962aac6ae50043d1e277499549eda69ed9731072b3c4033012b968b02c')
    old199=read(args.source199/'capacity_v2/capacity199.json')
    for shard in shards:
        S=[e for unit in shard['units'] for member in unit['members'] for e in member['edges'] if e['labels']['0.5']['class']=='S']
        knownS[shard['domain']]=dict(total=len(S),nonzero_30phase=sum(bool(np.any(np.asarray(e['delta_windows']))) for e in S))
        for unit in shard['units']:
            actions=[]
            for m in unit['members']:
                for x in [*m['cuts'],*m['edges']]:actions.append(x)
            pred=np.asarray(unit['pred_sum'],np.int64);gt=np.asarray(unit['gt_bins'],np.int64);residual=(pred-3*gt).ravel()
            for family in ('E','DE','PE','ALL'):
                answer=saved['units'][unit['unit']][family];active=[]
                for x in actions:
                    lab=x['labels']['0.5']['class'];legal=family=='ALL' or (x['kind']=='cut' and (family=='DE' and lab=='D' or family=='PE' and lab in ('D','U'))) or (x['kind']!='cut' and lab=='S')
                    if not legal:continue
                    delta=np.asarray(x['delta_windows'],np.int64).ravel();sparse={int(j):int(delta[j]) for j in np.flatnonzero(delta)}
                    if sparse:active.append(dict(x,sparse=sparse))
                assert [x['action_id'] for x in active]==answer['action_order']
                assert list(dict.fromkeys(x['parent_key'] for x in active))==answer['parent_keys']
                assert list(dict.fromkeys(x['orphan_key'] for x in active))==answer['orphan_keys']
                lower=sparse_certificate_independent(residual,active,answer['certificate']);lookup={x['action_id']:x for x in active};ids=answer['selected_action_ids'];chosen=[lookup[k] for k in ids]
                assert len(set(ids))==len(ids) and len({x['parent_key'] for x in chosen})==len(chosen)==len({x['orphan_key'] for x in chosen})
                changed=pred.ravel().copy()
                for x in chosen:
                    for j,value in x['sparse'].items():changed[j]+=value
                assert (changed>=0).all() and np.array_equal(changed.reshape(pred.shape),answer['witness_pred_sum'])
                upper=int(abs(changed-3*gt.ravel()).sum());assert lower<=upper==answer['upper_bound'] and answer['optimal']==(lower==upper)
                assert answer['denominator']==3*int(gt.sum()) and answer['baseline_numerator']==int(abs(residual).sum())
                reports.append(dict(unit=unit['unit'],family=family,lower=lower,upper=upper,tight=lower==upper))
                if unit['domain']=='35' and family in ('DE','PE'):
                    old=old199['units'][unit['unit']]['families']['O_D' if family=='DE' else 'O_DU']['oracle']
                    assert old['optimal'] and lower==upper==old['lower_bound']==old['upper_bound']
                    comparisons.append(dict(unit=unit['unit'],family=family,old199_family='O_D' if family=='DE' else 'O_DU',improvement_numerator=old['upper_bound']-upper))
    assert len(reports)==32
    assert knownS=={'35':dict(total=541,nonzero_30phase=4),'33':dict(total=607,nonzero_30phase=39)}
    pe35=[(unit,saved['units'][unit['unit']]['PE']) for shard in shards if shard['domain']=='35' for unit in shard['units']]
    pe_gain_ceiling=sum((Fraction(100*(a['baseline_numerator']-a['lower_bound']),a['denominator']) for unit,a in pe35),Fraction())/4
    assert pe_gain_ceiling<=Fraction(14,10)
    final=dict(passed=True,capacity_manifest=digest,certificates=reports,known_S_extension_nonzero=knownS,head35_family_vs199=comparisons,
        head35_PE_gain_ceiling_fraction=[pe_gain_ceiling.numerator,pe_gain_ceiling.denominator],
        implication='Under this registered action universe, strict35gain>1.4 requires at least one native199S cut or a union extension not confirmedS. U means unknown, not a proven incorrect association.',
        scope='All32 exact integer lower bounds/full-vector feasible witnesses; tightness only when bounds equal; actual native F1 checked separately')
    receipt(args.root/'verification/capacity.json',final);return final


def aggregate_independent(scored):
    exact={}
    for domain in ('35','33','W','G'):
        exact[domain]={}
        for arm,videos in scored.items():
            vv=[g for g in videos if g['domain']==domain]
            if not vv:continue
            groups=[]
            for group in sorted({g['group'] for g in vv}):
                clips=[g for g in vv if g['group']==group];assert len(clips)==(2 if domain in ('W','G') else 1)
                numerator=denominator=0;counts={t:[[0,0,0] for _ in range(3)] for t in ('0.5','0.3')}
                for clip in clips:
                    mm=sorted(clip['members'],key=lambda m:m['member_index']);assert [m['member_index'] for m in mm]==[0,1,2]
                    gt=np.asarray(mm[0]['metrics']['gt_bins'],np.int64);pred=sum(np.asarray(m['metrics']['pred_bins'],np.int64) for m in mm)
                    numerator+=int(abs(pred-3*gt).sum());denominator+=3*int(gt.sum())
                    for t in counts:
                        for slot,m in enumerate(mm):
                            for j,key in enumerate(('tp','pred','gt')):counts[t][slot][j]+=m['metrics']['iou'][t][key]
                ff={t:sum((Fraction(2*tp,max(1,p+g)) for tp,p,g in cc),Fraction())/3 for t,cc in counts.items()}
                groups.append(dict(numerator=numerator,denominator=denominator,F1=ff,E3=Fraction(100*numerator,denominator)))
            assert len(groups)==(4 if domain in ('35','33') else 3)
            exact[domain][arm]=dict(E3=sum((g['E3'] for g in groups),Fraction())/len(groups),F1={t:sum((g['F1'][t] for g in groups),Fraction())/len(groups) for t in ('0.5','0.3')})
    return exact


def audit_scores(args):
    from beetrack.evaluate import load_tracks,frame_match,match_events
    root=args.root;ss=sealed(root/'score',args.score_sha);scored=read(root/'score/native_scores.json');calls=read(root/'score/physical_receipts.json');summary=read(root/'score/summary.json')
    if hasattr(args,'prereg'):
        chain=read(root/'score/RUNNING.json');assert chain==summary['input_receipt'] and chain['prereg']==args.prereg
        assert chain['source_seals']=={name:sha(root/name/'manifest.json') for name in ('frozen','owners','predictions')}
        assert chain['capacity_seal']==sha(root/'capacity/manifest.json')
        compact_input=read(root/'capacity/compact_input.json');capacity_receipt=read(root/'capacity/capacity.json')['input_receipt']
        assert chain['source_seals']==compact_input['source_seals'] and capacity_receipt['input_sha256']==sha(root/'capacity/compact_input.json') and capacity_receipt['prereg']==args.prereg
        assert chain['source197_score_seal']==sha(args.source197.parent/'score/manifest.json')
    assert len(calls)==396
    u,v=own_helpers(args.repo);refs={(r['unit'],str(r['member'])):r for r in read(root/'frozen/deployment.json')['members']}
    roles={(g['unit'],str(m['member']),arm):(g,m) for arm,videos in scored.items() for g in videos for m in g['members']}
    assert len(roles)==396
    historical={}
    if hasattr(args,'source199'):
        sealed(args.source199/'score','ae37297a0080a55d09b4379bf0fde5713ba2e24be255665cfe22c3b3085226bc')
        historical={(g['unit'],str(m['member'])):m['metrics'] for g in read(args.source199/'score/native_scores.json')['K'] for m in g['members']}
    expected_states={};labels={};stats=defaultdict(lambda:dict(cuts=defaultdict(int),extensions=defaultdict(int),effects=defaultdict(int),extension_denominators=defaultdict(int),cut_denominators=defaultdict(int)))
    capacity=read(root/'capacity/capacity.json')
    for hive in ('35','33','W','G'):
        for m in read(root/'predictions'/(hive+'.masks.json'))['member_clips']:
            for arm,state in m['selected'].items():expected_states[m['unit'],str(m['member']),arm]=state
        for group in read(root/'owners'/(hive+'.json'))['units']:
            for m in group['members']:
                labels[group['unit'],str(m['member'])]={r['action_id']:r['labels']['0.5']['class'] for r in [*m['cuts'],*m['edges']]}
                if hive in ('35','33'):
                    for arm,answer in capacity['units'][group['unit']].items():
                        state=answer['member_masks'][str(m['member_index'])];expected_states[group['unit'],str(m['member']),arm]=dict(cut_ids=state['cuts'],edge_ids=state['edges'])
    GT={};cache={};fresh={};checkpoint=root/'verification/native';checkpoint.mkdir(parents=True,exist_ok=True)
    catalog_cache=None;context=None
    for call in calls:
        key=call['unit'],str(call['member']);ref=refs[key];g,role=roles[*key,call['arm']]
        assert call['path']==role['physical'] and call['seal']==role['seal']
        assert role['state']==dict(cut_ids=sorted(call['cut_ids']),edge_ids=sorted(call['edge_ids']))
        if expected_states:assert role['state']=={k:sorted(ids) for k,ids in expected_states[*key,call['arm']].items()}
        if historical and call['arm']=='K':
            old=historical[key]
            for field in ('nG','nP','pred_bins','gt_bins'):assert role['metrics'][field]==old[field]
            for t in ('0.5','0.3'):
                for field in ('tp','pred','gt'):assert role['metrics']['iou'][t][field]==old['iou'][t][field]
        if key!=context:
            folder=root/'frozen'/ref['path'];cat=read(folder/'catalog.json');raw=np.load(folder/'rows.npz',allow_pickle=False)['original_K']
            C,pos,blocks=u.canonicalize(raw);context=key
        cuts=cat['cut_catalog']['candidates'];selected=call['cut_ids'];chosen=call['edge_ids']
        statistic=stats[ref['hive'],call['arm']]
        for c in cuts:
            label=labels[key]['cut:'+c['candidate_id']];statistic['cut_denominators'][label]+=1
            if c['candidate_id'] in selected:statistic['cuts'][label]+=1;statistic['effects']['added' if c['delta']==1 else 'removed']+=1
        for e in cat['edges']:
            label=labels[key][e['edge_id']];statistic['extension_denominators'][label]+=1
            if e['edge_id'] in chosen:statistic['extensions'][label]+=1;statistic['effects'][e['event_effect']]+=1
        rows=u.realize_independent(raw,C,pos,blocks,cuts,selected);sources=list(map(tuple,cat['baseline_sources']));parents={c['parent_id'] for c in cuts if c['candidate_id'] in selected};orphans=set();parts=[rows]
        edges={e['edge_id']:e for e in cat['edges']};delta=np.zeros_like(np.asarray(role['metrics']['pred_bins'],np.int64))
        for c in cuts:
            if c['candidate_id'] in selected:delta+=np.asarray(c['delta_windows'],np.int64)
        for eid in sorted(chosen):
            e=edges[eid];assert e['parent_canonical_id'] not in parents and e['orphan_index'] not in orphans
            parents.add(e['parent_canonical_id']);orphans.add(e['orphan_index']);o=cat['orphans'][e['orphan_index']]
            added=np.asarray(o['rows'],float).copy();added[:,1]=e['recipient'];parts.append(added);sources.extend(map(tuple,o['sources']));delta+=np.asarray(e['delta_windows'],np.int64)
        rows=np.concatenate(parts);assert len(set(sources))==len(rows) and len(np.unique(rows[:,:2],axis=0))==len(rows)
        baseline=phase_bins(events(cat['baseline'],ref['polygon']),ref['nf']);pred=phase_bins(events(rows,ref['polygon']),ref['nf'])
        assert np.array_equal(baseline+delta,pred) and np.array_equal(pred,role['metrics']['pred_bins'])
        token=call['path'];folder=root/'score/physical'/token
        if token not in cache:
            sealed(folder,call['seal']);meta=read(folder/'receipt.json');full=read(folder/'score.json');compact=read(folder/'compact.json')
            assert meta['new_fullscene_official_matching'] and meta['no198_graph_F1'] and meta['provenance_complete'] and meta['source_rows_unique']
            reread=np.loadtxt(folder/'mot.txt',delimiter=',',usecols=range(7),ndmin=2);assert np.array_equal(rows,reread)
            assert meta['identity']['rows_sha256']==native202_array_sha(rows) and meta['identity']['raw_sha256']==native202_array_sha(raw)
            assert meta['identity']['polygon_sha256']==native202_array_sha(ref['polygon'])
            assert meta['identity']['cut_ids']==sorted(selected) and meta['identity']['edge_ids']==sorted(chosen)
            with gzip.open(folder/'row_sources.jsonl.gz','rt',encoding='utf-8') as stream:stored_sources=[tuple(json.loads(line)) for line in stream]
            assert stored_sources==sources
            if ref['clip'] not in GT:GT[ref['clip']]=load_tracks(args.repo/'runs/gt'/ref['split']/ref['clip']/'gt/gt.txt')
            gt=GT[ref['clip']];assert meta['identity']['GT_sha256']==native202_array_sha(gt)
            truth=u.track_structure(gt,ref['polygon']);structure=u.track_structure(rows,ref['polygon'])
            assert full['nG']==len(truth['events']) and full['nP']==len(structure['events'])
            gt_bins=phase_bins(truth['events'].values(),ref['nf']);assert np.array_equal(gt_bins,full['gt_bins']) and np.array_equal(pred,full['pred_bins'])
            numerator=int(abs(pred-gt_bins).sum());denominator=int(gt_bins.sum())
            if 'window' in full:
                assert full['window']['numerator']==numerator and full['window']['denominator']==denominator
                assert full['W'] is None if not denominator else abs(full['W']-100*numerator/denominator)<1e-11
            assert full['evidence']['gt']==[[pid,frame*direction] for pid,(frame,direction) in truth['events'].items()]
            assert [[x[0],x[1]] for x in full['evidence']['predictions']]==[[pid,frame*direction] for pid,(frame,direction) in structure['events'].items()]
            for t,field in (('0.5','edges05'),('0.3','edges03')):
                adjacency=defaultdict(set)
                for pid,gid in full['evidence'][field]:
                    assert pid in structure['events'] and gid in truth['events']
                    assert structure['events'][pid][1]==truth['events'][gid][1]
                    adjacency[pid].add(gid)
                tp=u.augmenting_tp(structure['events'],adjacency)
                assert tp==full['iou'][t]['tp'] and full['iou'][t]['pred']==len(structure['events']) and full['iou'][t]['gt']==len(truth['events'])
                assert abs(full['iou'][t]['f1']-2*tp/max(1,full['nG']+full['nP']))<1e-14
            fresh_here=bool(chosen and ref['hive'] not in fresh)
            if fresh_here:
                rawowners={}
                for threshold in (.5,.3):
                    owners=frame_match(truth['rows'],structure['rows'],thr=threshold);scattered=np.empty(len(rows),np.int64);scattered[structure['index']]=owners;rawowners[str(threshold)]=scattered
                own=u.score_from_owners(structure,truth,rawowners,ref['polygon'],ref['nf'])
                for t in ('0.5','0.3'):
                    for field in ('tp','pred','gt','tp_id'):assert own['iou'][t][field]==full['iou'][t][field]
                    assert own['edges'][t]==full['evidence']['edges05' if t=='0.5' else 'edges03']
                official=match_events(gt,rows,ref['polygon']);assert official['tp']==full['iou']['0.5']['tp']
                fresh[ref['hive']]=dict(unit=ref['unit'],member=ref['member'],physical=token,added_rows=len(rows)-len(raw),thresholds=[.5,.3])
            cache[token]=compact
            receipt(checkpoint/(token+'.json'),dict(passed=True,physical_seal=call['seal'],rows=len(rows),added_rows=len(rows)-len(raw),fresh_native_rematch=fresh_here))
        assert role['metrics']==cache[token]
    physical_witness_groups=0
    for arm in ('E','DE','PE','ALL'):
        for group in scored[arm]:
            expected=capacity['units'][group['unit']][arm]
            if 'witness_pred_sum' not in expected:continue  # tiny synthetic adapter fixture
            predsum=sum(np.asarray(m['metrics']['pred_bins'],np.int64) for m in group['members']);gtbins=np.asarray(group['members'][0]['metrics']['gt_bins'],np.int64)
            assert np.array_equal(predsum,expected['witness_pred_sum'])
            assert int(abs(predsum-3*gtbins).sum())==expected['upper_bound'] and 3*int(gtbins.sum())==expected['denominator']
            physical_witness_groups+=1
    if hasattr(args,'prereg'):assert physical_witness_groups==32
    exact=aggregate_independent(scored);numbers={h:{arm:dict(E3=float(x['E3']),F1={t:float(z) for t,z in x['F1'].items()}) for arm,x in arms.items()} for h,arms in exact.items()}
    for h,arms in numbers.items():
        for arm,x in arms.items():
            published=summary['domains'][h]['methods'][arm];assert abs(published['E3']-x['E3'])<1e-11
            assert all(abs(published['F1'][t]-x['F1'][t])<1e-13 for t in ('0.5','0.3'))
            if hasattr(args,'prereg'):assert 'action_stats' in published
            if 'action_stats' in published:
                for key,value in stats[h,arm].items():assert dict(value)==published['action_stats'][key]
    gates={}
    for arm in ('L','J'):
        gates[arm]=dict(head35_gain=exact['35']['K']['E3']-exact['35'][arm]['E3']>Fraction(14,10),head33_gain=exact['33']['K']['E3']-exact['33'][arm]['E3']>Fraction(13,10),W_no_harm=exact['W'][arm]['E3']-exact['W']['K']['E3']<=Fraction(14,10),F1_no_harm=all(exact[h][arm]['F1']['0.5']>=exact[h]['K']['F1']['0.5']-Fraction(1,200) for h in ('35','33','W')))
        assert summary['investment_checks'][arm]==gates[arm]
    attribution={arm:all(exact[h][arm]['E3']<exact[h][other]['E3'] for h in ('35','33') for other in (('G','M') if arm=='L' else ('L','G'))) for arm in ('L','J')}
    assert summary['L_identity_ranking_pass']==attribution['L'] and summary['J_strictly_beats_L_and_G_both_heads']==attribution['J']
    unavailable=[h for h in ('35','33','W','G') if h not in fresh]
    result=dict(passed=True,score_manifest=ss,requested=len(calls),unique=len(cache),capacity_physical_witness_groups=physical_witness_groups,fresh_native_changed_geometry=fresh,fresh_native_unavailable_domains=unavailable,means=numbers,investment_checks=gates,attribution=attribution,scope='All396 native matrices/provenance/events/integer windows and sealed full-scene event graph cardinalities; fresh native frame rematching at both IoUs on listed changed geometry only. Remaining spatial edge generation is sealed producer evidence, not independently rematched. TP_ID is independently checked only on these fresh samples.')
    receipt(root/'verification/scores.json',result);return result


def sparse_certificate_independent(residual,actions,proof):
    """No need to trust the proposal's solver or a tight linear matching."""
    p=proof['p'];q=proof['q'];a=proof['a'];b=proof['b']
    assert type(q)is int and q>0 and len(p)==len(residual)
    assert all(type(x)is int and abs(x)<=q for x in p)
    assert all(type(x)is int and x>=0 for x in a+b)
    parents=list(dict.fromkeys(e['parent_key'] for e in actions));orphans=list(dict.fromkeys(e['orphan_key'] for e in actions))
    assert len(a)==len(parents) and len(b)==len(orphans)
    costs=[sum(p[j]*int(v) for j,v in e['sparse'].items()) for e in actions]
    assert costs==proof['edge_dot_products']
    assert all(cost+a[parents.index(e['parent_key'])]+b[orphans.index(e['orphan_key'])]>=0 for cost,e in zip(costs,actions))
    linear=sum(int(x)*int(y) for x,y in zip(p,residual));scaled=linear-sum(a)-sum(b)
    assert linear==proof['linear_term'] and scaled==proof['scaled_bound']
    lower=max(0,-((-scaled)//q));assert lower==proof['lower_bound']
    return lower


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native202_array_sha(value):
    a=np.asarray(value,dtype=np.float64)
    metadata=json.dumps([str(a.dtype),list(a.shape)],separators=(',',':'),sort_keys=True,allow_nan=False).encode('utf-8')
    return hashlib.sha256(metadata).hexdigest()+':'+hashlib.sha256(a.tobytes(order='C')).hexdigest()


def receipt(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,sort_keys=True,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    path.with_suffix('.sha256').write_text(sha(path)+'\n',encoding='ascii')


def lineage(D,stage1,indices,linked,linked_indices,K,historical_ids):
    """Check exact indexed measurements; historical s1 is ONLY an ID vector."""
    D=np.asarray(D);s=np.asarray(stage1);ix=np.asarray(indices);l=np.asarray(linked);j=np.asarray(linked_indices)
    assert D.ndim==2 and D.shape[1]==6 and s.ndim==2 and s.shape[1]==7
    assert ix.shape==(len(s),) and np.issubdtype(ix.dtype,np.integer)
    assert len(set(ix.tolist()))==len(ix) and ((0<=ix)&(ix<len(D))).all()
    original=D[ix];xywh=np.column_stack((original[:,1:3],original[:,3:5]-original[:,1:3]))
    assert np.array_equal(s[:,0],original[:,0])
    # Native PASS1 converts detections through float32 before output.
    b=original[:,1:6].astype(np.float32)
    assert np.array_equal(s[:,2:6],np.column_stack((b[:,:2],b[:,2:4]-b[:,:2])))
    assert np.array_equal(s[:,6],b[:,4])
    assert sorted(j.tolist())==list(range(len(s))) and l.shape==s.shape
    cols=[0,2,3,4,5,6]
    assert np.array_equal(l[:,cols],s[j][:,cols])
    ids,count=np.unique(l[:,1],return_counts=True);keep=np.isin(l[:,1],ids[count>=3])
    assert np.array_equal(l[keep],K)
    h=np.asarray(historical_ids)
    assert h.ndim==1 and len(h)==len(K) and np.array_equal(h,s[j[keep],1])
    return dict(stage1_rows=len(s),retained_rows=int(keep.sum()),orphan_rows=int((~keep).sum()),source_bijection=True,historical_s1_kind='per-K-row IDs, not Nx7')


def catalog(base,orphans,max_gap=12,max_error=2):
    """Independent endpoint regression via centered scalar sums (not lstsq)."""
    base=np.asarray(base,float);edges=[]
    for pid in sorted(set(base[:,1])):
        tr=base[base[:,1]==pid];tr=tr[np.argsort(tr[:,0],kind='stable')]
        if len(tr)<3:continue
        for oi,short in enumerate(orphans):
            short=np.asarray(short,float)
            assert len(short) in (1,2) and (short[:,4:6]>0).all() and np.all(np.diff(short[:,0])>0)
            if short[-1,0]<tr[0,0]:side='before';core=tr[:3];gap=tr[0,0]-short[-1,0]
            elif short[0,0]>tr[-1,0]:side='after';core=tr[-3:];gap=short[0,0]-tr[-1,0]
            else:continue
            if gap>max_gap:continue
            t=core[:,0];dt=t-t.mean();c=core[:,2:4]+core[:,4:6]/2
            slope=(dt[:,None]*(c-c.mean(axis=0))).sum(axis=0)/(dt@dt)
            pred=c.mean(axis=0)+(short[:,0]-t.mean())[:,None]*slope
            scale=float(np.median(np.sqrt((core[:,4:6]**2).sum(axis=1))))
            error=float(np.sqrt(((pred-short[:,2:4]-short[:,4:6]/2)**2).sum(axis=1)).max()/scale)
            if error<=max_error:edges.append(dict(recipient=int(pid),orphan_index=oi,side=side,gap=int(gap),normalized_error=error))
    return edges


def inside(rows,polygon):
    p=np.asarray(polygon,np.float32)
    return np.asarray([cv2.pointPolygonTest(p,(float(r[2]+r[4]/2),float(r[3]+r[5]/2)),False)>=0 for r in rows])


def events(rows,polygon):
    """Official whole-track endpoint direction, last matching crossing date."""
    rows=np.asarray(rows,float);out=[]
    for pid in sorted(set(rows[:,1])):
        tr=rows[rows[:,1]==pid];tr=tr[np.argsort(tr[:,0],kind='stable')]
        for run in np.split(tr,np.flatnonzero(np.diff(tr[:,0])>25)+1):
            state=inside(run,polygon)
            if state[0]==state[-1]:continue
            direction=1 if state[-1] else -1
            idx=np.flatnonzero((state[:-1]!=state[1:])&(state[1:]==state[-1]))[-1]+1
            out.append((int(run[idx,0]),direction))
    return out


def matching_certificate(r,edges,proof):
    """Exact rational L1 lower bound via nonnegative vertex potentials."""
    r=list(map(int,r));p=proof['p'];q=proof['q'];assert type(q)is int and q>0 and len(p)==len(r)
    assert all(type(t)is int and abs(t)<=q for t in p)
    a={x['recipient']:x['a'] for x in proof['recipient_dual']};b={x['orphan']:x['b'] for x in proof['orphan_dual']}
    assert set(a)=={e['recipient'] for e in edges} and set(b)=={e['orphan'] for e in edges}
    assert all(type(t)is int and t>=0 for t in list(a.values())+list(b.values()))
    costs=[sum(x*y for x,y in zip(p,e['v'])) for e in edges]
    assert all(c+a[e['recipient']]+b[e['orphan']]>=0 for e,c in zip(edges,costs))
    byid={e['edge_id']:e for e in edges};assert len(byid)==len(edges)
    def chosen(ids):
        assert len(set(ids))==len(ids)
        es=[byid[k] for k in ids]
        assert len({e['recipient'] for e in es})==len(es)==len({e['orphan'] for e in es})
        return es
    optimum=chosen(proof['matching_ids'])
    assert sum(sum(x*y for x,y in zip(p,e['v'])) for e in optimum)==-sum(a.values())-sum(b.values())
    numerator=sum(x*y for x,y in zip(p,r))-sum(a.values())-sum(b.values())
    lower=max(0,-((-numerator)//q));witness=chosen(proof['witness_ids'])
    upper=sum(abs(v+sum(e['v'][j] for e in witness)) for j,v in enumerate(r))
    assert lower==proof['lower_bound'] and upper==proof['witness_upper'] and lower<=upper
    return dict(lower=lower,upper=upper,tight=lower==upper)


def synthetic():
    poly=[(0,0),(10,0),(10,10),(0,10)]
    r=np.array([[f,1,x,2,2,2,.9] for f,x in enumerate((-4,0,2,0,-4),1)],float)
    assert events(r[:4],poly)==[(2,1)] and events(r,poly)==[]
    moved=np.r_[r[:3],r[4:],[[6,1,0,2,2,2,.9]]]
    assert events(moved,poly)==[(6,1)]
    c=catalog(r[:4],[r[4:]])
    assert len(c)==1 and c[0]['side']=='after'
    e=[dict(edge_id='x',recipient=1,orphan=2,v=[-1])]
    proof=dict(p=[1],q=1,recipient_dual=[dict(recipient=1,a=1)],orphan_dual=[dict(orphan=2,b=0)],matching_ids=['x'],witness_ids=['x'],lower_bound=1,witness_upper=1)
    assert matching_certificate([2],e,proof)['tight']
    # Mixed known/unknown context: positive excludes U; no-positive+U omitted.
    rng=np.random.default_rng(202);options=[];labels={}
    for hive in ('33','W','G'):
        for group in (2,3):
            for parent in range(3):
                for typ in ('extension','cut'):
                    aid=f'{hive}-{group}-{parent}-{typ}'
                    options.append(dict(action_id=aid,type=typ,hive=hive,group=group,clip='synthetic',unit=f'{hive}-{group}',member='0',member_index=0,recipient=str(parent),orphan_key=str(parent),features=rng.normal(size=10 if typ=='extension' else 11).tolist()))
                    labels[aid]=('S','D','U')[parent]
    for mode in ('L','J'):
        fit=refit_independent(options,labels,mode,'35')
        assert abs(sum(c['weight'] for c in fit['contexts'])-1)<1e-14
        assert all(abs(sum(c['target'])-1)<1e-14 for c in fit['contexts'])
        assert all(c['hive']!='35' for c in fit['contexts'])
    tests=0
    for trial in range(30):
        rows=[];scores={}
        for j in range(6):
            aid=str(j);row=dict(action_id=aid,type='extension',unit='u',member='0',recipient=str(rng.integers(3)),orphan_key=str(rng.integers(3)))
            rows.append(row);scores[aid]=float(rng.normal())
        ids=choose_independent(rows,scores);actual=sum(scores[k] for k in ids);best=0
        for bits in itertools.product((0,1),repeat=6):
            chosen=[r for r,x in zip(rows,bits) if x]
            if len({r['recipient'] for r in chosen})!=len(chosen) or len({r['orphan_key'] for r in chosen})!=len(chosen):continue
            best=max(best,sum(scores[r['action_id']] for r in chosen))
        assert abs(actual-best)<1e-12;tests+=1
    bins=phase_bins([(750,1),(751,-1)],1500);assert bins.sum()==60
    return dict(passed=True,scope='synthetic only',whole_event_removal_and_date_move=True,independent_regression=True,integer_matching_dual=True,
                conditional_softmax_L_J=True,matching_exhaustive_graphs=tests,phase_integer_windows=True,field_enabled=False)


def contract_synthetic():
    """Production comparison is synthetic only; it supplies no audit result."""
    import importlib.util
    spec=importlib.util.spec_from_file_location('orphans_production_fit_contract',Path(__file__).with_name('fit_orphans.py'))
    prod=importlib.util.module_from_spec(spec);spec.loader.exec_module(prod)
    rng=np.random.default_rng(1202);rows=[];labels={}
    for hive in ('33','W','G'):
        for i in range(6):
            typ='extension' if i%2 else 'cut';aid=f'{hive}|{i}'
            rows.append(dict(action_id=aid,type=typ,hive=hive,group=2,clip='synth',unit=hive,member='0',member_index=0,recipient=str(i//2),orphan_key=str(i//2),features=rng.normal(size=10 if typ=='extension' else 11).tolist()))
            rows[-1]['features'][0]=1.
            labels[aid]=('S','D','U')[i%3]
    for mode in ('L','J'):
        subset=[r for r in rows if mode=='J' or r['type']=='extension'];lab={r['action_id']:labels[r['action_id']] for r in subset}
        actual=prod.fit(subset,lab,mode,'35');ind=refit_independent(subset,lab,mode,'35')
        assert actual['contexts']==ind['contexts']
        assert abs(actual['optimization']['objective']-ind['objective'])<1e-11
        for typ,h in ind['heads'].items():
            for key in ('mean','scale','beta','intercept'):np.testing.assert_allclose(h[key],actual['heads'][typ][key],atol=2e-7,rtol=2e-7)
    scores={r['action_id']:float(rng.normal()) for r in rows}
    assert prod.matching(rows,scores)==choose_independent(rows,scores)
    ext=[r for r in rows if r['type']=='extension'];es={r['action_id']:scores[r['action_id']] for r in ext}
    assert prod.shuffled_scores(ext,es)[0]==shuffle_independent(ext,es)
    return dict(passed=True,source_contract='synthetic production vs independent L/J fits, masks, M permutation',field_opened=False)


def native_synthetic(repo):
    import sys
    sys.path.insert(0,str(repo))
    from beetrack.p0 import stage1,p0,THR
    from beetrack.link import link
    u,v=own_helpers(repo)
    D=np.asarray([[f,x,0,x+4,4,.9] for f in range(1,6) for x in (0,0)]+[[1,100,0,104,4,.9],[2,100,0,104,4,.9],[4,200,0,204,4,.9]],float)
    s,ix=trace_stage1_independent(D,5,.4);assert np.array_equal(s,stage1(D,5))
    prob=lambda X:np.zeros(len(X));tagged=np.c_[s,np.arange(len(s))]
    linked=np.asarray(sorted(link(tagged.tolist(),THR,prob).tolist())).reshape(-1,8);K=np.asarray(p0(s,prob)).reshape(-1,7)
    ids,count=np.unique(linked[:,1],return_counts=True);keep=np.isin(linked[:,1],ids[count>=3]);j=linked[:,7].astype(int)
    proof=lineage(D,s,ix,linked[:,:7],j,K,s[j[keep],1]);assert proof['orphan_rows']==3
    for threshold in (.5,.3):
        GT=np.r_[K[:,:6],K[:2,:6]+np.array([0,900,0,0,0,0])]
        needed=set(range(len(K)));a=union_owners_independent(GT,K,threshold,needed,v);b=v.full_frame_owners(GT,K,threshold,requested_indices=needed)
        assert a['rows']==b['rows'] and np.array_equal(a['P'],b['P']) and np.array_equal(a['G'],b['G'])
    synthetic();return dict(passed=True,native_first_stage_exact=True,index_lineage=True,disjoint_retained_orphans=True,duplicate_geometry_NMS=True,field_opened=False)


def score_contract_synthetic(repo):
    """Exercise the actual396-request adapter with tiny synthetic full MOTs."""
    import tempfile,sys
    from types import SimpleNamespace
    sys.path.insert(0,str(repo))
    from beetrack.evaluate import frame_match
    u,v=own_helpers(repo)
    def dump(path,x):
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(x,sort_keys=True)+'\n',encoding='utf-8')
    def seal_toy(path):
        dump(path/'manifest.json',{p.relative_to(path).as_posix():sha(p) for p in path.rglob('*') if p.is_file() and p.name not in ('manifest.json','SEALED.sha256')})
        (path/'SEALED.sha256').write_text(sha(path/'manifest.json')+'\n')
    poly=[[0,0],[20,0],[20,20],[0,20]]
    tr=lambda pid,xs,y:np.asarray([[f,pid,x,y,2,2,.9] for f,x in enumerate(xs,1)],float)
    returning=tr(1,[-4,1,3,1,-3],2);other=tr(2,[-5,-3,1,3,4],14)
    raw=np.r_[returning[:4],other];truthraw=np.r_[returning,other][:,:6]
    C,pos,blocks=u.canonicalize(raw);baseline=u.realize_independent(raw,C,pos,blocks,[],[]);short=returning[4:].copy();short[:,1]=4
    truth=u.track_structure(truthraw,poly);basebins=phase_bins(events(baseline,poly),75)
    after=np.r_[baseline,short.copy()];after[-1,1]=1;delta=phase_bins(events(after,poly),75)-basebins
    with tempfile.TemporaryDirectory(prefix='independent-orphan-synthetic-') as td:
        root=Path(td);fake_repo=root/'repo';refs=[];scored={k:[] for k in ('K','G','L','M','J','E','DE','PE','ALL')};calls=[]
        for hive in ('35','33','W','G'):
            for group in range(2,6 if hive in ('35','33') else 5):
                for clipindex in range(1 if hive in ('35','33') else 2):
                    clip=f'{hive}-clip{clipindex}';unit=f'{hive}-g{group}-{clip}';roles={a:dict(unit=unit,domain=hive,group=group,video=clip,nf=75,members=[]) for a in (scored if hive in ('35','33') else ('K','G','L','M','J'))}
                    gtfile=fake_repo/'runs/gt/dev'/clip/'gt/gt.txt';gtfile.parent.mkdir(parents=True,exist_ok=True)
                    np.savetxt(gtfile,truthraw,delimiter=',',fmt='%.17g')
                    for member in range(3):
                        path=f'u{len(refs):02d}';folder=root/'frozen'/path;folder.mkdir(parents=True);edge=unit+'|'+str(member)+'|extension'
                        sources=[['toyD',i] for i in range(len(raw))]
                        cat=dict(baseline=baseline.tolist(),baseline_sources=sources,orphans=[dict(rows=short.tolist(),sources=[['toyD',99]])],cut_catalog=dict(candidates=[]),edges=[dict(edge_id=edge,orphan_index=0,recipient=1,parent_canonical_id=1,delta_windows=delta.tolist(),event_effect='removed')])
                        dump(folder/'catalog.json',cat);np.savez(folder/'rows.npz',original_K=raw)
                        refs.append(dict(unit=unit,member=str(member),member_index=member,hive=hive,group=group,clip=clip,split='dev',nf=75,polygon=poly,path=path))
                        for arm,g in roles.items():
                            extended=arm!='K';rows=after if extended else baseline;state=dict(cut_ids=[],edge_ids=[edge] if extended else []);token=hashlib.sha256(json.dumps([unit,member,state],sort_keys=True).encode()).hexdigest();physical=root/'score/physical'/token
                            structure=u.track_structure(rows,poly);owners={}
                            for t in (.5,.3):
                                own=frame_match(truth['rows'],structure['rows'],thr=t);scatter=np.empty(len(rows),int);scatter[structure['index']]=own;owners[str(t)]=scatter
                            result=u.score_from_owners(structure,truth,owners,poly,75)
                            full=dict(nG=result['nG'],nP=result['nP'],gt_bins=result['gt_bins'].tolist(),pred_bins=result['pred_bins'].tolist(),iou={t:dict(result['iou'][t],f1=2*result['iou'][t]['tp']/max(1,result['nG']+result['nP'])) for t in ('0.5','0.3')},evidence=dict(gt=[[pid,f*d] for pid,(f,d) in truth['events'].items()],predictions=[[pid,f*d] for pid,(f,d) in structure['events'].items()],edges05=result['edges']['0.5'],edges03=result['edges']['0.3']))
                            compact={k:full[k] for k in ('nG','nP','gt_bins','pred_bins')};compact['iou']={t:{k:full['iou'][t][k] for k in ('tp','pred','gt','f1')} for t in ('0.5','0.3')}
                            if not physical.exists():
                                physical.mkdir(parents=True);np.savetxt(physical/'mot.txt',rows,delimiter=',',fmt='%.17g');dump(physical/'score.json',full);dump(physical/'compact.json',compact)
                                dump(physical/'receipt.json',dict(identity=dict(rows_sha256=native202_array_sha(rows),raw_sha256=native202_array_sha(raw),polygon_sha256=native202_array_sha(poly),GT_sha256=native202_array_sha(truthraw),**state),new_fullscene_official_matching=True,no198_graph_F1=True,provenance_complete=True,source_rows_unique=True))
                                with gzip.open(physical/'row_sources.jsonl.gz','wt',encoding='utf-8') as stream:
                                    for item in sources+([['toyD',99]] if extended else []):stream.write(json.dumps(item)+'\n')
                                seal_toy(physical)
                            digest=sha(physical/'manifest.json');call=dict(unit=unit,member=str(member),member_index=member,arm=arm,path=token,seal=digest,**state);calls.append(call)
                            g['members'].append(dict(member=str(member),member_index=member,metrics=compact,state=state,physical=token,seal=digest))
                    for arm,g in roles.items():scored[arm].append(g)
        dump(root/'frozen/deployment.json',dict(members=refs));dump(root/'score/native_scores.json',scored);dump(root/'score/physical_receipts.json',calls)
        capacity=dict(units={})
        for hive in ('35','33','W','G'):
            masks=[];units={}
            for ref in [r for r in refs if r['hive']==hive]:
                unit=ref['unit'];member=ref['member'];edge=unit+'|'+member+'|extension'
                selected={a:dict(cut_ids=[],edge_ids=[] if a=='K' else [edge]) for a in ('K','G','L','M','J')}
                masks.append(dict(unit=unit,member=member,selected=selected))
                units.setdefault(unit,dict(unit=unit,members=[]))['members'].append(dict(member=member,member_index=int(member),cuts=[],edges=[dict(action_id=edge,labels={'0.5':{'class':'S'}})]))
                if hive in ('35','33'):
                    for arm in ('E','DE','PE','ALL'):capacity['units'].setdefault(unit,{}).setdefault(arm,dict(member_masks={}))['member_masks'][member]=dict(cuts=[],edges=[edge])
            dump(root/'owners'/(hive+'.json'),dict(units=list(units.values())))
            dump(root/'predictions'/(hive+'.masks.json'),dict(member_clips=masks))
        dump(root/'capacity/capacity.json',capacity)
        exact=aggregate_independent(scored);summary=dict(domains={h:dict(methods={a:dict(E3=float(x['E3']),F1={t:float(z) for t,z in x['F1'].items()}) for a,x in arms.items()}) for h,arms in exact.items()},investment_checks={a:dict(head35_gain=True,head33_gain=True,W_no_harm=True,F1_no_harm=True) for a in ('L','J')},L_identity_ranking_pass=False,J_strictly_beats_L_and_G_both_heads=False)
        dump(root/'score/summary.json',summary);seal_toy(root/'score')
        args=SimpleNamespace(root=root,repo=repo,score_sha=sha(root/'score/manifest.json'))
        # Actual GT is synthetic and lives only in this temporary tree; helper
        # imports remain in the real repo, so redirect only prepared GT paths.
        real_load=__import__('beetrack.evaluate',fromlist=['load_tracks']).load_tracks
        import beetrack.evaluate as evaluate
        evaluate.load_tracks=lambda path:real_load(fake_repo/Path(path).relative_to(repo))
        try:result=audit_scores(args)
        finally:evaluate.load_tracks=real_load
        assert result['requested']==396 and len(result['fresh_native_changed_geometry'])==4
        return dict(passed=True,synthetic_requests=396,synthetic_unique=result['unique'],fresh_native_domains=4,field_opened=False)


def capacity_contract_synthetic():
    import importlib.util
    spec=importlib.util.spec_from_file_location('orphan_capacity_contract',Path(__file__).with_name('cap_orphans.py'));prod=importlib.util.module_from_spec(spec);spec.loader.exec_module(prod)
    rng=np.random.default_rng(2202);checked=0
    for index in range(8):
        gt=phase_bins([(10,1)],75);pred=3*gt+phase_bins([(10,1)]*3+[(40,-1)]*3,75);actions=[]
        for j in range(6):
            frame,direction=(10,1) if j%2 else (40,-1);delta=-phase_bins([(frame,direction)],75);typ='cut' if j%3==0 else 'extension'
            actions.append(dict(action_id=str(j),kind=typ,member_index=0,parent_key='p'+str(int(rng.integers(3))),orphan_key=('private'+str(j)) if typ=='cut' else 'o'+str(int(rng.integers(3))),labels={'0.5':{'class':('S','D','U')[j%3]}},delta_windows=delta.tolist()))
        unit=dict(unit=str(index),nf=75,hive='35',group=index,video='synthetic',pred_sum=pred.tolist(),gt_bins=gt.tolist(),actions=actions,member_indices=[0,1,2])
        for family in ('E','DE','PE','ALL'):
            answer=prod.solve_unit(unit,family,lp_limit=2,mip_limit=2,linear_limit=200);active=[]
            for a in actions:
                label=a['labels']['0.5']['class'];allowed=family=='ALL' or (a['kind']=='extension' and label=='S') or (a['kind']=='cut' and (family=='DE' and label=='D' or family=='PE' and label in ('D','U')))
                if allowed:
                    x=np.asarray(a['delta_windows']).ravel();active.append(dict(a,sparse={int(i):int(x[i]) for i in np.flatnonzero(x)}))
            lower=sparse_certificate_independent((pred-3*gt).ravel(),active,answer['certificate']);best=int(abs(pred-3*gt).sum())
            for bits in itertools.product((0,1),repeat=len(active)):
                chosen=[a for a,x in zip(active,bits) if x]
                if len({a['parent_key'] for a in chosen})!=len(chosen) or len({a['orphan_key'] for a in chosen})!=len(chosen):continue
                q=pred.copy()
                for a in chosen:q+=np.asarray(a['delta_windows'])
                if (q>=0).all():best=min(best,int(abs(q-3*gt).sum()))
            assert lower<=best<=answer['upper_bound']
            if answer['optimal']:assert lower==best==answer['upper_bound']
            checked+=1
    return dict(passed=True,exact_enumerated_family_certificates=checked,field_opened=False)


def union_contract_synthetic(repo):
    import importlib.util,sys
    sys.path.insert(0,str(repo));spec=importlib.util.spec_from_file_location('orphan_union_contract',Path(__file__).with_name('owners_orphans.py'));prod=importlib.util.module_from_spec(spec);spec.loader.exec_module(prod)
    u,v=own_helpers(repo);poly=[[0,0],[20,0],[20,20],[0,20]]
    K=np.asarray([[f,1,f,2,2,2,.9] for f in (1,2,3)],float);short=np.asarray([[4,9,4,2,2,2,.9]],float)
    edge=dict(edge_id='e',orphan_index=0,recipient=1,endpoint_K_row_index=2,side='after',recipient_cut_join_key='parent')
    cat=dict(baseline=K,baseline_sources=[['D',i] for i in range(3)],orphans=[dict(rows=short,sources=[['D',3]],linked_id=9),dict(rows=short+np.array([0,1,0,0,0,0,0]),sources=[['D',99]],linked_id=10)],edges=[edge,dict(edge,edge_id='duplicate-edge')])
    P,keys,Kmap,Omap=union_independent(cat,u);published=prod.build_union(cat)
    assert np.array_equal(P,published['rows']) and keys==list(map(tuple,published['source_keys'])) and len(P)==4 and set(Omap)=={0}
    checked=0
    for kind in ('S','D','U'):
        GT=np.r_[K[:,:6],short[:,:6]];GT[:,1]=7
        if kind=='D':GT[-1,1]=8
        if kind=='U':GT=np.r_[GT,GT[-1:]+np.array([0,90,0,0,0,0])]
        labels=prod.label_member(GT,cat,poly,dict(unit='u',member='0',member_index=0))
        for threshold in (.5,.3):
            needed={Kmap[2],*Omap[0]};owners=union_owners_independent(GT,P,threshold,needed,v)
            left=owners['rows'][Kmap[2]];right=owners['rows'][Omap[0][0]]
            label='U' if not(left['isolated'] and right['isolated']) else 'S' if left['owner']==right['owner'] else 'D' if left['original_gt_id']!=right['original_gt_id'] else 'U'
            assert label==kind and all(e['labels'][str(threshold)]['class']==label for e in labels['edges']);checked+=1
    return dict(passed=True,label_threshold_cases=checked,union_source_dedup=True,unreferenced_orphan_excluded=True,full_frame_competition=True,field_opened=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--contract-tests',action='store_true');parser.add_argument('--native-synthetic',action='store_true');parser.add_argument('--score-contract-tests',action='store_true');parser.add_argument('--capacity-contract-tests',action='store_true');parser.add_argument('--union-contract-tests',action='store_true')
    parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--root',type=Path,default=None)
    parser.add_argument('--source195',type=Path,default=None);parser.add_argument('--source197',type=Path,default=None);parser.add_argument('--source199',type=Path,default=None)
    parser.add_argument('--phase',choices=('candidates','owners','models','capacity','scores','all'));parser.add_argument('--resume-after',choices=('candidates','owners','models','capacity','scores'))
    parser.add_argument('--field-go',action='store_true');parser.add_argument('--score-worker-finished',action='store_true');parser.add_argument('--prereg',default='e2d3871227319b6674c3b6569f07d73e3663bdd8')
    for name in ('frozen','owners','predictions','capacity','score'):parser.add_argument('--'+name+'-sha')
    args=parser.parse_args()
    if args.phase:
        assert args.field_go and args.score_worker_finished,'Explicit GO and released sole CPU required'
        import sys
        sys.path.insert(0,str(args.repo))
        for name in ('frozen','owners','predictions','capacity','score'):sealed(args.root/name,getattr(args,name+'_sha'))
        stages=dict(candidates=audit_candidates,owners=audit_owners,models=audit_models,capacity=audit_capacity,scores=audit_scores)
        if args.phase=='all':
            output={'source_allowlist':audit_source_allowlist(args)}
            skip=list(stages).index(args.resume_after) if args.resume_after else -1
            for index,(name,fn) in enumerate(stages.items()):
                if index<=skip:
                    path=args.root/'verification'/(name+'.json');assert sha(path)==path.with_suffix('.sha256').read_text().strip()
                    output[name]=read(path);assert output[name]['passed']
                else:output[name]=fn(args)
            final=dict(passed=True,prereg=args.prereg,verifier_sha256=sha(__file__),execution_sources={name:sha(Path(__file__).with_name(name)) for name in ('verify_orphans_run.py','verify_orphans_resume.py','verify_orphans_scores.py','verify_orphans_finalize.py') if Path(__file__).with_name(name).exists()},helper_sha256={name:sha(args.repo/name) for name in ('tools/study197/verify.py','tools/study199/verify199.py')},input_seals={name:sha(args.root/name/'manifest.json') for name in ('frozen','owners','predictions','capacity','score')},phases={name:dict(path=name+'.json',sha256=sha(args.root/'verification'/(name+'.json'))) for name in output},scope={name:value['scope'] for name,value in output.items()})
            receipt(args.root/'verification/independent.json',final)
            output=final
        else:output=stages[args.phase](args)
    else:output=union_contract_synthetic(args.repo) if args.union_contract_tests else capacity_contract_synthetic() if args.capacity_contract_tests else score_contract_synthetic(args.repo) if args.score_contract_tests else native_synthetic(args.repo) if args.native_synthetic else contract_synthetic() if args.contract_tests else synthetic()
    print(json.dumps(output,indent=2))
