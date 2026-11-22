"""Explicit sole-CPU208 field audit. No IO occurs until audit(args)."""
import hashlib
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path
from collections import defaultdict,Counter
import verify208 as independent

PREREG='c358042292d912cbf74e21859f6e0fba45ad1cf8'
SPECS={('35','BEE24-35-head',3),('33','BEE24-33-head',3),('W','BEE24-10',0),('W','BEE24-15',0),('G','BEE24-24',0),('G','BEE24-28',0)}
ARMS=('geometry_interval','relative_cross_exact','relative_cross_interval')


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def seal(root,expected=None):
    root=Path(root);digest=sha(root/'manifest.json');assert digest==(root/'SEALED.sha256').read_text().strip()
    if expected:assert digest==expected
    for name,value in read(root/'manifest.json').items():
        path=(root/name).resolve();assert path.is_relative_to(root.resolve()) and sha(path)==value
    return digest


def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');digest=sha(tmp);tmp.replace(p);p.with_suffix('.sha256').write_text(digest+'\n')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def canonical_ids(rows):
    import numpy as np
    ids=np.empty(len(rows),np.int64);runs={};seen=set()
    for pid in sorted(set(rows[:,1])):
        ix=np.flatnonzero(rows[:,1]==pid);ix=ix[np.argsort(rows[ix,0],kind='stable')]
        for run,part in enumerate(np.split(ix,np.flatnonzero(np.diff(rows[ix,0])>25)+1)):
            alias=int(pid)+100000*run;assert alias not in seen and alias<2**53;seen.add(alias);ids[part]=alias;runs[alias]=(int(pid),run)
    return ids,runs


def frame_owners(GT,P,v):
    """All original frames, independent Hungarian/degree arithmetic, all GT rows."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    gc,runs=canonical_ids(GT);pc,_=canonical_ids(P)
    go=np.lexsort((np.arange(len(GT)),GT[:,0],gc));po=np.lexsort((np.arange(len(P)),P[:,0],pc))
    gf=defaultdict(list);pf=defaultdict(list)
    for i in go:gf[int(GT[i,0])].append(i)
    for i in po:pf[int(P[i,0])].append(i)
    result=np.full((len(P),2),-1,np.int64)
    for date,pr in pf.items():
        gg=gf.get(date,[])
        if not gg:continue
        matrix=v.official_iou_independent(GT[gg,2:6],P[pr,2:6]);edges=matrix>=.5;pd=edges.sum(0);gd=edges.sum(1);a,b=linear_sum_assignment(-matrix)
        for g,p in zip(a,b):
            if edges[g,p] and pd[p]==1 and gd[g]==1:result[pr[p]]=runs[int(gc[gg[g]])]
    return result


class FrameCache:
    """Matching depends on a date's ordered geometry, not the numeric P IDs."""
    def __init__(self,GT,base,v):
        import numpy as np
        self.GT,self.base,self.v=GT,base,v;self.alias,self.runs=canonical_ids(GT);self.frames=defaultdict(list);self.cache={};self.requests=0
        for i in np.lexsort((np.arange(len(GT)),GT[:,0],self.alias)):self.frames[int(GT[i,0])].append(i)

    def owners(self,rows):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        alias,_=canonical_ids(rows);frames=defaultdict(list)
        for i in np.lexsort((np.arange(len(rows)),rows[:,0],alias)):frames[int(rows[i,0])].append(i)
        out={t:np.full(len(rows),-1,np.int64) for t in ('0.5','0.3')}
        for date,pr in frames.items():
            self.requests+=1;key=(date,tuple(pr))
            if key not in self.cache:
                gg=self.frames.get(date,[]);answers={t:np.full(len(pr),-1,np.int64) for t in out}
                if gg:
                    matrix=self.v.official_iou_independent(self.GT[gg,2:6],self.base[pr,2:6]);a,b=linear_sum_assignment(-matrix)
                    for t in out:
                        ok=matrix[a,b]>=float(t);answers[t][b[ok]]=self.alias[np.asarray(gg)[a[ok]]]
                self.cache[key]=answers
            for t in out:out[t][pr]=self.cache[key][t]
        return out


def sampled_forward(state,crops,geometry,arm):
    """Independent Torch functional forward; does not instantiate producer model."""
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    x=torch.tensor(crops.copy(),dtype=torch.float32).unsqueeze(2)/255.;g=torch.tensor(geometry.copy(),dtype=torch.float32);batch=len(x)
    with torch.inference_mode():
        e=F.relu(F.conv2d(x.flatten(0,1),state['encoder.0.weight'],state['encoder.0.bias'],padding=1))
        e=F.relu(F.conv2d(e,state['encoder.2.weight'],state['encoder.2.bias'],stride=2,padding=1)).mean((-2,-1)).reshape(batch,4,-1);e=F.normalize(e,dim=-1)
        within=((e[:,0]-e[:,1]).square().sum(-1)+(e[:,2]-e[:,3]).square().sum(-1))/2
        cross=(e[:,:2,None]-e[:,None,2:]).square().sum(-1).mean((1,2));appearance=cross-within
        if arm=='geometry_interval':appearance=appearance*0
        z=F.relu(F.linear(torch.cat((g,appearance[:,None]),1),state['classifier.0.weight'],state['classifier.0.bias']))
        return F.linear(z,state['classifier.2.weight'],state['classifier.2.bias']).squeeze(-1).sigmoid().numpy()


def audit(args):
    import os,sys,gc
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.path.insert(0,str(args.repo.resolve()))
    import numpy as np
    from beetrack.evaluate import load_tracks
    root=args.root;seals={s:seal(root/s,getattr(args,s+'_sha',None)) for s in ('inputs','labels','predictions','score')}
    top=read(root/'inputs/deployment.json');ltop=read(root/'labels/deployment.json');pred=read(root/'predictions/metadata.json');summary=read(root/'score/summary.json')
    assert all(x['prereg']==PREREG for x in (top,ltop,pred)) and top['source206']=='066aaa245d78ee5d926cafeb709d200746e9ea9f736f820606270c35a2188d5d'
    assert {(r['domain'],r['clip'],r['seed']) for r in top['clips']}==SPECS and len(top['clips'])==6
    assert ltop['input_manifest_sha256']==pred['input_manifest_sha256']==seals['inputs'] and pred['label_manifest_sha256']==seals['labels']
    replay=module('independent203_for208',args.repo/'tools/study203/verify203.py');previous,native,v,_=replay.helpers(args.repo)
    graph=module('independent203_graph208',args.repo/'tools/study203/verify203_scores.py')
    source206=read(args.source206/'deployment.json');assert sha(args.source206/'manifest.json')==top['source206']
    records={};input_reports=[];label_reports=[]
    for ref in top['clips']:
        key=ref['domain']+'/'+ref['clip'];folder=root/'inputs'/ref['path'];seal(folder,ref['seal']);meta=read(folder/'metadata.json')
        raw=np.load(folder/'rows.npy',mmap_mode='r');canon=np.load(folder/'canon.npy',mmap_mode='r');indices=np.load(folder/'indices.npy',mmap_mode='r');valid=np.load(folder/'valid.npy',mmap_mode='r')
        assert raw.shape[1]==7 and raw.dtype==np.float64 and canon.dtype==np.int64 and valid.dtype==np.bool_
        oldgroup=meta['source206_group'];assert oldgroup in source206['groups']
        oldmeta=read(args.source206/oldgroup['path']/'queries.json');oldmember=next(m for m in oldmeta['members'] if m['seed']==ref['seed'])
        with np.load(args.source206/oldgroup['path']/'K.npz') as z:assert raw.tobytes()==z['k'+str(oldmember['member_index'])].tobytes()
        derived,_=canonical_ids(raw);assert np.array_equal(derived,canon)
        R=raw.tolist();C=canon.tolist();I=indices.tolist();V=valid.tolist();runs,_=independent.candidate_table(R,C,I)
        allwindows=[ii[j:j+4] for parent,ii in sorted(runs.items()) for j in range(len(ii)-3)];expected=[w for w in allwindows if all(1<=R[b][0]-R[a][0]<=3 for a,b in zip(w,w[1:]))]
        assert I==expected and np.array_equal(np.asarray(allwindows,np.int64).reshape(-1,4),np.load(folder/'windows_all.npy'))
        assert len(raw)==meta['raw_rows'] and len(I)==ref['n_cuts']==meta['gap_eligible_windows'] and sum(V)==ref['n_valid']
        gtpath=args.repo/meta['prepared_gt_relative'];assert sha(gtpath)==ltop['GT_sha256'][meta['prepared_gt_relative']];GT=load_tracks(gtpath)
        owners=np.load(root/'labels'/ref['path']/'owners.npy');derivedowners=frame_owners(GT,raw,v);assert np.array_equal(derivedowners,owners)
        bags,excluded=independent.independent_bags(R,C,I,V,owners)
        with np.load(root/'labels'/ref['path']/'bags.npz') as bank:
            assert np.array_equal(np.asarray([b['X'] for b in bags],np.int64).reshape(-1,3),bank['X']) and np.array_equal(np.asarray([b['y'] for b in bags],np.int8),bank['y']) and np.array_equal(np.asarray([b['anchors'] for b in bags],np.int64).reshape(-1,2),bank['anchors'])
        census=independent.adjacent_census(R,C,I,V,owners,ref['clip']);duplicate=read(root/'labels'/ref['path']/'duplicate_bee.json');expecteddups=[]
        for cut in census:
            if cut['kind']!='exact_switch':continue
            def presence(date,identity):
                jj=np.flatnonzero((GT[:,0]==date)&(GT[:,1]==identity))
                return 'present_known' if len(jj)==1 and (GT[jj,4:6]>0).all() else 'unknown'
            d=independent.duplicate_bee_state(cut['left_raw'],cut['right_raw'],R,C,owners,presence);expecteddups.append((cut,d))
        assert len(expecteddups)==len(duplicate['records'])
        for (cut,d),published in zip(expecteddups,duplicate['records']):
            assert published['left_raw_index']==cut['left_raw'] and published['right_raw_index']==cut['right_raw'] and published['state']==d['state'] and published['other_confirmed_raw_indices']==d['other_rows']
        input_reports.append(dict(clip=key,rows=len(raw),candidates=len(I),valid=sum(V),K206_exact=True))
        label_reports.append(dict(clip=key,all_rows_owners_exact=True,bags=len(bags),excluded=excluded,full_adjacent=dict(Counter(c['kind'] for c in census)),duplicates=dict(Counter(d['state'] for c,d in expecteddups))))
        records[key]=dict(ref=ref,meta=meta,folder=folder,rows=R,canon=C,indices=I,valid=V,owners=owners,census=census,bags=bags,GT=GT)
        print('INDEPENDENT208_INPUT_LABEL',key,flush=True);gc.collect()
    save(root/'verification/input_labels.json',dict(passed=True,seals=seals,inputs=input_reports,labels=label_reports,scope='Six exact206 K arrays/all candidate quads; all original-frame .5 Hungarian+mutual-degree owners independently recomputed; all source bags and duplicate census. Crop pixels not independently decoded.'))
    return records,pred,summary,seals,(replay,previous,native,v,graph)


def predictions(args,records,pred,seals):
    import numpy as np
    import torch
    torch.set_num_threads(1);root=args.root;reports=[];inference=[]
    outputs={r['domain']+'/'+r['clip']:r for r in pred['clips']}
    for heldout in independent.HIVES:
        fold=pred['folds'][heldout];allowed=[h for h in independent.HIVES if h!=heldout]
        assert fold['heldout']==heldout and fold['source_domains']==allowed and set(fold['models'])==set(ARMS)
        sourcekeys=sorted(k for k,r in records.items() if r['ref']['domain'] in allowed)
        for arm in ARMS:
            model=fold['models'][arm];training=read(root/'predictions'/model['training_file']);cal=model['calibration']
            assert training['seed']==208 and training['steps']==120 and training['batch_size']==32 and training['positive_per_step']==training['negative_per_step']==16 and training['learning_rate']==.003 and training['channels']==8 and training['threads']==1 and training['device']=='cpu'
            assert training['optimizer']==dict(name='Adam',lr=.003,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False,foreach=False)
            source=training['source_receipt'];assert source['source_domains']==allowed and source['label_manifest_sha256']==seals['labels'] and set(source['source_bags'])==set(sourcekeys)
            cohorts={c:{h:[] for h in independent.HIVES} for c in (0,1)}
            for key in sourcekeys:
                r=records[key];bags=r['bags'];proof=source['source_bags'][key]
                assert proof['bags']==len(bags) and proof['positive']==sum(b['y']==1 for b in bags) and proof['negative']==sum(b['y']==0 for b in bags) and proof['exact']==sum(b['length']==1 for b in bags)
                assert proof['sha256']==sha(root/'labels'/proof['file'])
                for i,b in enumerate(bags):
                    if arm!='relative_cross_exact' or b['length']==1:cohorts[b['y']][r['ref']['domain']].append((key,i))
            available={str(c):[h for h in independent.HIVES if cohorts[c][h]] for c in (0,1)};assert training['class_available_domains']==available
            for h in allowed:
                assert training['source_counts'][h]=={str(c):len(cohorts[c][h]) for c in (0,1)}
                for c in (0,1):assert training['sampler_cohort_sha256'][h][str(c)]==hashlib.sha256(json.dumps(cohorts[c][h],separators=(',',':')).encode()).hexdigest()
            noop=not all(available.values());assert (training['status']!='trained')==noop
            trace=[];rng=np.random.default_rng(208)
            if not noop:
                for step in range(120):
                    batch=[]
                    for c in (1,0):
                        for j in range(16):
                            domain=available[str(c)][int(rng.integers(len(available[str(c)])))];pool=cohorts[c][domain];batch.append(pool[int(rng.integers(len(pool)))])
                    trace.append(batch)
                assert training['first_batch']==json.loads(json.dumps(trace[0])) and training['completed_steps']==120
            else:assert training['completed_steps']==0
            assert training['sampled_batches_sha256']==hashlib.sha256(json.dumps(trace,separators=(',',':')).encode()).hexdigest()
            assert cal['quantile_probability']==.999 and cal['quantile_method']=='higher' and cal['clip_order']==sourcekeys
            expected=[]
            for clipindex,key in enumerate(sourcekeys):
                for bagindex,b in enumerate(records[key]['bags']):
                    if b['y']==0 and b['length']==1:expected.append((clipindex,bagindex,b['X'][0]))
            with np.load(root/'predictions'/cal['scores_file']) as bank:
                values=bank['probabilities'];assert values.dtype==np.float32 and np.isfinite(values).all() and len(values)==len(expected)==cal['exact_negative_bags']
                assert list(zip(bank['clip_index'].tolist(),bank['bag_index'].tolist(),bank['cut_index'].tolist()))==expected
                q=float(np.sort(values)[(999*(len(values)-1)+999)//1000]) if len(values) else None
            theta=1. if noop or q is None else max(.5,q);assert cal['quantile']==q and cal['threshold']==model['threshold']==theta
            state=torch.load(root/'predictions'/model['model_file'],map_location='cpu',weights_only=True)
            registered=sum(x.numel() for x in state.values());assert registered==training['registered_parameters']==809
            assert training['structural_gradient_parameter_limit']==(129 if arm=='geometry_interval' else 809)
            for key,r in records.items():
                if r['ref']['domain']!=heldout:continue
                out=outputs[key]['outputs'][arm];assert out['threshold']==theta
                p=np.load(root/'predictions'/out['probabilities'],mmap_mode='r');valid=np.asarray(r['valid'],bool);assert p.dtype==np.float32 and p.shape==valid.shape and np.isfinite(p[valid]).all() and np.isnan(p[~valid]).all()
                assert ((p[valid]>=0)&(p[valid]<=1)).all() and np.array_equal(np.load(root/'predictions'/out['selected']),valid&(p>theta)) and np.array_equal(np.load(root/'predictions'/out['raw05']),valid&(p>.5))
                if noop:assert (p[valid]==.5).all()
                else:
                    vi=np.flatnonzero(valid);sample=vi[np.unique([0,len(vi)//2,len(vi)-1])] if len(vi) else vi
                    if len(sample):
                        crops=np.load(r['folder']/'crops.npy',mmap_mode='r');geometry=np.load(r['folder']/'geometry.npy',mmap_mode='r');actual=sampled_forward(state,crops[sample],geometry[sample],arm)
                        np.testing.assert_allclose(actual,p[sample],rtol=2e-6,atol=2e-6);inference.append(dict(heldout=heldout,arm=arm,clip=key,cut_ordinals=sample.tolist(),max_abs=float(abs(actual-p[sample]).max())))
                normalized=[dict(c,score=float(p[c['input_ordinal']]) if c['represented'] else float('nan')) for c in r['census']]
                selected,_=independent.selected_cuts(r['rows'],r['canon'],r['indices'],p,theta,r['valid'],not noop)
                shifted,audit=independent.displaced_scores(r['rows'],r['canon'],r['indices'],p,r['valid'])
                chosen=set(selected);nmsrows=[dict(c,score=1. if c['input_ordinal'] in chosen else 0.) for c in normalized]
                nmsdiag=independent.binary_diagnostics(nmsrows,.5);nmsdiag.pop('AP_fraction');nmsdiag['AP_scope']='Use unmodified score-conditional AP; NMS is a discrete decision, not a new ranking.'
                reports.append(dict(heldout=heldout,clip=key,arm=arm,source_threshold=theta,status=training['status'],raw05=independent.binary_diagnostics(normalized,.5),calibrated=independent.binary_diagnostics(normalized,theta),
                    nms=nmsdiag,timing=independent.timing_diagnostics(nmsrows,.5),selected=selected,
                    displaced_selected=independent.selected_cuts(r['rows'],r['canon'],r['indices'],shifted,theta,r['valid'],not noop)[0]))
                r.setdefault('p',{})[arm]=p;r.setdefault('theta',{})[arm]=theta;r.setdefault('active',{})[arm]=not noop
    result=dict(passed=True,diagnostics=reports,sampled_forward=inference,scope='All four source guards/cohorts/exact stochastic draw hashes and source calibration origins/higher quantile; eighteen full scores/masks; independent functional forward sampled three valid cuts per clip/arm. No independent NN retraining or crop pixel decoding. Full known-label abstentions retained as pipeline FN; valid-score AP explicitly conditional.')
    save(root/'verification/predictions.json',result);return result


def physical(args,records,pred,summary,seals,helpers):
    import numpy as np
    replay,previous,native,v,graph=helpers;root=args.root;outputs={r['domain']+'/'+r['clip']:r for r in pred['clips']};reports=[];scores={};caches=[]
    old={ (g['unit'],str(m['member'])):m['metrics'] for g in read(args.reference206_score/'native_scores.json')['K'] for m in g['members']}
    for key,r in records.items():
        meta=r['meta'];base=np.asarray(r['rows'],float);GT=r['GT'];polygon=meta['polygon'];nf=meta['nf'];cache=FrameCache(GT,base,v);truth=native.track_structure(GT,polygon)
        selections={};probs={};thresholds={};active={}
        for arm in ARMS:
            out=outputs[key]['outputs'][arm];probs[arm]=np.load(root/'predictions'/out['probabilities'],mmap_mode='r');thresholds[arm]=out['threshold'];active[arm]=pred['folds'][r['ref']['domain']]['models'][arm]['status']=='trained'
            selections[arm]=independent.selected_cuts(r['rows'],r['canon'],r['indices'],probs[arm],thresholds[arm],r['valid'],active[arm])[0]
        def states():
            yield 'K',r['rows'],None,None
            for arm in ARMS:
                cut,segments=independent.split_rows(r['rows'],r['canon'],r['indices'],selections[arm]);yield arm+'_CUT',cut,selections[arm],None
                linked,links=independent.relink(cut,segments);yield arm+'_CUT_RELINK',linked,selections[arm],links
            arm='relative_cross_interval';shift,audit=independent.displaced_scores(r['rows'],r['canon'],r['indices'],probs[arm],r['valid']);chosen=independent.selected_cuts(r['rows'],r['canon'],r['indices'],shift,thresholds[arm],r['valid'],active[arm])[0]
            cut,segments=independent.split_rows(r['rows'],r['canon'],r['indices'],chosen);yield arm+'_DISPLACED_CUT',cut,chosen,None
            linked,links=independent.relink(cut,segments);yield arm+'_DISPLACED_CUT_RELINK',linked,chosen,links
        clip_summary=next(c for c in summary['clips'] if c['domain']==r['ref']['domain'] and c['clip']==r['ref']['clip']);assert len(clip_summary['states'])==9
        scores[key]={}
        for name,expected,chosen,links in states():
            folder=root/'score/clips'/r['ref']['domain']/r['ref']['clip']/name;result=read(folder/'score.json');actual=result['metrics'];disk=np.loadtxt(folder/'mot.txt',delimiter=',',usecols=range(7),ndmin=2)
            expected=np.asarray(expected,float);assert disk.tobytes()==expected.tobytes() and expected[:,[0,2,3,4,5,6]].tobytes()==base[:,[0,2,3,4,5,6]].tobytes()
            assert result['physical_rows']==len(base) and result['raw_row_bijection'] is True
            if chosen is not None:assert result['proof']['nms']['selected']==chosen and result['cuts']==len(chosen)
            if links is not None:
                reported=result['proof']['relink'];assert sorted((e['left'],e['right']) for e in reported['accepted_edges'])==links['links']
                assert reported['indirect_original_run_return_paths']==len(links['indirect_original_run_return_paths'])
            P=native.track_structure(expected,polygon);under=cache.owners(expected);own=native.score_from_owners(P,truth,under,polygon,nf)
            gb=replay.bins(truth['events'].values(),nf);pb=replay.bins(P['events'].values(),nf);numerator=int(abs(pb-gb).sum());denominator=int(gb.sum())
            assert np.array_equal(actual['pred_bins'],pb) and np.array_equal(actual['gt_bins'],gb) and actual['nP']==len(P['events']) and actual['nG']==len(truth['events'])
            assert result['numerator']==numerator and result['denominator']==denominator
            for t,edge in (('0.5','edges05'),('0.3','edges03')):
                for field in ('tp','pred','gt','tp_id'):assert own['iou'][t][field]==actual['iou'][t][field],(key,name,t,field)
                assert own['edges'][t]==actual['evidence'][edge]
            if name=='K':
                reference=old[meta['unit'],str(meta['member'])]
                for field in ('nG','nP','pred_bins','gt_bins'):assert actual[field]==reference[field]
                for t in ('0.5','0.3'):
                    for field in ('tp','pred','gt'):assert actual['iou'][t][field]==reference['iou'][t][field]
            compact=clip_summary['states'][name];assert compact['numerator']==numerator and compact['denominator']==denominator
            for t in ('0.5','0.3'):
                for field in ('tp','pred','gt'):assert compact['iou'][t][field]==actual['iou'][t][field]
            scores[key][name]=dict(numerator=numerator,denominator=denominator,iou={t:actual['iou'][t] for t in ('0.5','0.3')})
        caches.append(dict(clip=key,scene_frame_requests=cache.requests,unique_frame_orders=len(cache.cache)));reports.append(dict(clip=key,states=9,same_rows=True,all_both_IoU_graphs_exact=True))
        print('INDEPENDENT208_PHYSICAL',key,flush=True)
    means={}
    for hive in independent.HIVES:
        means[hive]={};keys=[k for k,r in records.items() if r['ref']['domain']==hive]
        for name in next(iter(scores.values())):
            for t in ('0.5','0.3'):
                clips=[dict(numerator=scores[k][name]['numerator'],denominator=scores[k][name]['denominator'],TP=scores[k][name]['iou'][t]['tp'],Pred=scores[k][name]['iou'][t]['pred'],GT=scores[k][name]['iou'][t]['gt']) for k in keys]
                own=independent.pooled_physical(clips);saved=summary['domains'][hive]['methods'][name];assert sum(c['numerator'] for c in clips)==saved['numerator'] and sum(c['denominator'] for c in clips)==saved['denominator']
                assert own['TP']==saved['iou'][t]['tp'] and own['Pred']==saved['iou'][t]['pred'] and own['GT']==saved['iou'][t]['gt']
                means[hive].setdefault(name,{})[t]=own
    main='relative_cross_interval_CUT_RELINK'
    W=lambda h,n:Fraction(*means[h][n]['0.5']['W_fraction'])
    F=lambda h,n:Fraction(*means[h][n]['0.5']['F1_fraction'])
    checks=dict(head35_gain_at_least1=W('35','K')-W('35',main)>=1,
        head35_strictly_beats_geometry_exact_displaced=all(W('35',main)<W('35',n) for n in ('geometry_interval_CUT_RELINK','relative_cross_exact_CUT_RELINK','relative_cross_interval_DISPLACED_CUT_RELINK')),
        head33_no_harm_halfpoint=W('33',main)-W('33','K')<=Fraction(1,2),head_F1_no_harm_onehundredth=all(F(h,main)>=F(h,'K')-Fraction(1,100) for h in ('35','33')))
    assert checks==summary['registered_gate']['checks'] and all(checks.values())==summary['registered_gate']['all_checks']
    result=dict(passed=True,seals=seals,clips=reports,frame_order_cache=caches,pooled=means,registered_gate=dict(checks=checks,all_checks=all(checks.values())),scope='All54 complete MOTs, rows/order/geometry/score unchanged except ID, NMS/displacement/relink rebuilt; original/reordered frame geometry Hungarian and both-IoU event graphs/TP/P/GT/TP_ID independently rebuilt with exact frame-order cache. K exact206. No NN retraining or pixel decoding.')
    save(root/'verification/physical.json',result);return result


def execute(args):
    records,pred,summary,seals,helpers=audit(args)
    physical_result=physical(args,records,pred,summary,seals,helpers)
    diagnostic_result=predictions(args,records,pred,seals)
    result=dict(schema='switch208-independent-v1',passed=True,prereg=PREREG,input_seals=seals,
        sources={n:sha(Path(__file__).with_name(n)) for n in ('verify208.py','audit208.py')},
        receipts={n:sha(args.root/'verification'/n) for n in ('input_labels.json','physical.json','predictions.json')},
        scopes=[physical_result['scope'],diagnostic_result['scope']])
    save(args.root/'verification/independent.json',result);return result
