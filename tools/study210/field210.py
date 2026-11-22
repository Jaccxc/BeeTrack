"""Explicit sole-CPU §210 independent field adapter. No IO on import."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import audit210 as independent

sys.path.insert(0,str(Path(__file__).parents[1]/'study209'))
import audit209 as prior

PREREG='9a90eb75b59279d81f637ea490a0e5c58677549a'
SOURCE208='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94'
LABEL208='e73ba698916d5e60fda6289b04f7a9d53d75ef1767621250370bf9bb299a473a'
DOMAINS=('35','33','W','G')
ARMS=('K','N','G','X','M','ALL')

def loss(theta,z,y,w,offset):
    eta=offset+z@theta[:-1]+theta[-1]
    probability=np.exp(-np.logaddexp(0,-eta));residual=w*(probability-y)
    return float(np.sum(w*(np.logaddexp(0,eta)-y*eta))+.05*np.sum(theta[:-1]**2)),np.r_[z.T@residual+.1*theta[:-1],residual.sum()]

def weights(records):
    result=np.empty(len(records))
    for y in (0,1):
        domains={r['domain'] for r in records if r['y']==y}
        if not domains:return None
        for domain in domains:
            ix=[j for j,r in enumerate(records) if r['y']==y and r['domain']==domain]
            result[ix]=.5/(len(domains)*len(ix))
    return result

def refit(records,feature,offset):
    from scipy.optimize import minimize
    w=weights(records);d=6 if feature=='G6' else 2
    if w is None:return dict(status='zero_missing_class',beta=np.zeros(d),intercept=0.)
    x=np.asarray([r[feature] for r in records]);y=np.asarray([r['y'] for r in records]);mean=np.sum(x*w[:,None],axis=0)
    scale=np.sqrt(np.sum(w[:,None]*(x-mean)**2,axis=0));scale[scale<1e-6]=1;z=(x-mean)/scale
    fit=minimize(lambda t:loss(t,z,y,w,offset),np.zeros(d+1),jac=True,method='L-BFGS-B',options=dict(maxiter=500,gtol=1e-8,ftol=1e-12))
    assert fit.success,fit.message
    return dict(status='fitted',mean=mean,scale=scale,beta=fit.x[:-1],intercept=fit.x[-1],weight=w,gradient=loss(fit.x,z,y,w,offset)[1],objective=fit.fun)

def logit(model,x):
    if model['status']!='fitted':return 0.
    return float(((np.asarray(x)-model['mean'])/model['scale'])@model['beta']+model['intercept'])

def self_test():
    rng=np.random.default_rng(210);z=rng.normal(size=(12,2));y=np.arange(12)%2;w=np.ones(12)/12
    offset=rng.normal(size=12);theta=rng.normal(size=3);_,gradient=loss(theta,z,y,w,offset)
    numerical=[]
    for j in range(3):
        eps=np.zeros(3);eps[j]=1e-6
        numerical.append((loss(theta+eps,z,y,w,offset)[0]-loss(theta-eps,z,y,w,offset)[0])/2e-6)
    assert np.max(abs(gradient-numerical))<1e-8
    data=[dict(domain='35' if i<4 else 'W',y=i%2,G6=[float(i),1,2,3,0,0],X2=[float(i),1]) for i in range(8)]
    ww=weights(data)
    assert sum(ww)==1 and all(abs(sum(ww[i] for i,r in enumerate(data) if r['domain']==h and r['y']==c)-.25)<1e-12 for h in ('35','W') for c in (0,1))
    head=refit(data,'G6',np.zeros(8));assert head['status']=='fitted' and np.max(abs(head['gradient']))<1e-6
    assert refit([dict(domain='35',y=1,G6=[0]*6)],'G6',np.zeros(1))['status']=='zero_missing_class'
    return {'passed':True,'scope':'synthetic offset logistic/gradient/weights/refit/fallback only'}

def slate(pairs):
    used=set();out=[]
    for p in sorted(pairs,key=lambda p:(not p['active'],p['min_offset_norm'],tuple(p['canonical_ids']))):
        if not used.intersection(p['canonical_ids']):out.append(p['pair_id']);used.update(p['canonical_ids'])
    table={p['pair_id']:p for p in pairs};active=[p for p in out if table[p]['active']]
    return out,active,(len(active)+3)//4 if active else 0

def catalog_stage(args,seals,helpers):
    """Complete GT-free interval enumeration, local geometry, feature and delta."""
    replay,previous,native,v,_=helpers
    top=prior.read(args.root/'catalog/deployment.json');assert top['prereg']==PREREG and top['source208']==SOURCE208
    assert {(r['domain'],r['clip'],r['seed']) for r in top['clips']}==prior.SPECS and len(top['clips'])==6
    assert prior.sha(args.source208/'manifest.json')==SOURCE208
    records={};reports=[]
    for ref in top['clips']:
        key=ref['domain']+'/'+ref['clip'];folder=args.root/'catalog'/ref['path'];prior.seal(folder,ref['seal'])
        cat=prior.read(folder/'catalog.json.gz');meta=cat['metadata'];src=args.source208/meta['source208_path']
        assert prior.sha(src/'rows.npy')==meta['rows_sha256'] and prior.sha(src/'canon.npy')==meta['canon_sha256']
        rows=np.load(src/'rows.npy',mmap_mode='r');canon=np.load(src/'canon.npy',mmap_mode='r');assert np.array_equal(prior.canonical(rows),canon)
        rr=independent.runs(rows,canon);expected=independent.candidate_universe(rows,canon);table={p['pair_id']:p for p in cat['pairs']}
        assert set(table)=={p['pair_id'] for p in expected}
        before={};computed=[];signs=[]
        for p in expected:
            saved=table[p['pair_id']];a,b=p['canonical_ids'];f=independent.features(rows,canon,a,b,rr)
            for ours,name in (('G','G6'),('X','X2'),('Xminus','Xminus2')):prior.close(f[ours],saved[name],2e-9)
            prior.close(p['min_offset_norm'],saved['min_offset_norm']);assert f['full_support']==saved['full_support']
            expected_brackets={tuple(br['raw_indices']):br for br in p['brackets']}
            assert set(expected_brackets)=={tuple(br['raw_indices']) for br in saved['brackets']}
            for br in saved['brackets']:
                own=expected_brackets[tuple(br['raw_indices'])]
                assert own['dates']==br['frames'];prior.close(own['b_minus_a'],br['offset_xy']);prior.close(own['normalized_offset'],br['normalized_offset'])
            sf=saved['features'];assert f['prefix_indices']==sf['prefix_indices'] and f['suffix_indices']==sf['suffix_indices'] and f['cutoff_frame']==sf['cutoff_frame']
            ix=np.sort(np.r_[rr[a],rr[b]]);local=rows[ix];lc=canon[ix]
            for c in (a,b):
                if c not in before:before[c]=list(native.track_structure(rows[rr[c]],meta['polygon'])['events'].values())
            merged=independent.merge(local,lc,[(a,b)])
            events=list(native.track_structure(merged,meta['polygon'])['events'].values())
            delta=independent.clock(events,meta['nf'])-independent.clock(before[a]+before[b],meta['nf'])
            assert np.array_equal(delta,saved['delta_windows']) and bool(np.any(delta))==saved['active']
            computed.append(dict(saved,G6=f['G'],X2=f['X'],Xminus2=f['Xminus']))
            if f['full_support']:signs.append(dict(pair_id=p['pair_id'],prefix_offset=f['prefix_offset'],X2=f['X'],Xminus2=f['Xminus'],prefix_raw_rows=sorted({j for i in f['prefix_indices'] for j in f['brackets'][i]['raw_indices']}),suffix_raw_rows=sorted({j for i in f['suffix_indices'] for j in f['brackets'][i]['raw_indices']})))
        common,active,k=slate(computed);assert common==cat['slate_ids'] and active==cat['active_slate_ids']
        records[key]=dict(ref=ref,cat=cat,meta=meta,rows=rows,canon=canon,runs=rr,pairs=computed,slate=common,active=active,k=k)
        reports.append(dict(clip=key,candidates=len(table),full_support=sum(p['full_support'] for p in computed),active=sum(p['active'] for p in computed),slate=len(common),active_slate=len(active),k=k,full_support_signs=signs))
        print('AUDIT210_CATALOG',key,flush=True)
    prior.save(args.root/'verification/catalog.json',dict(passed=True,seals=seals,clips=reports,scope='All six immutableK arrays, complete no-common-date candidate bracket universe incl singleton/far brackets, local3diag gate, unique-support median scale, all features/prefix/suffix/signs and all local physical event/30phase deltas independently rebuilt.'))
    return records

def label_stage(args,records,seals):
    from beetrack.evaluate import load_tracks
    assert prior.sha(args.labels208/'manifest.json')==LABEL208
    oldmanifest=prior.read(args.labels208/'manifest.json');top=prior.read(args.root/'labels/deployment.json')
    assert top['prereg']==PREREG and top['candidate_manifest_sha256']==seals['catalog']
    reports=[]
    for key,r in records.items():
        ref=r['ref'];path=args.root/'labels'/ref['domain']/ref['clip']/'evidence.json.gz';data=prior.read(path)
        relative=key+'/owners.npy';assert prior.sha(args.labels208/relative)==oldmanifest[relative]==data['source208_owner_sha256']
        owners=np.load(args.labels208/relative,mmap_mode='r');assert owners.shape==(len(r['rows']),2)
        oldreceipt=prior.read(args.labels208/key/'receipt.json');assert oldreceipt['input_named_files']['rows.npy']==r['meta']['rows_sha256']
        gtpath=args.repo/data['GT_relative_path'];assert prior.sha(gtpath)==data['GT_sha256']==top['GT_sha256'][data['GT_relative_path']]
        GT=load_tracks(gtpath);gtcanonical=prior.canonical(GT);lookup={}
        for pid in np.unique(GT[:,1]):
            ii=np.flatnonzero(GT[:,1]==pid);ii=ii[np.argsort(GT[ii,0],kind='stable')]
            for run,part in enumerate(np.split(ii,np.flatnonzero(np.diff(GT[ii,0])>25)+1)):
                values=set(gtcanonical[part]);assert len(values)==1;lookup[int(pid),run]=int(next(iter(values)))
        presence={}
        for j,row in enumerate(GT):presence.setdefault((int(row[0]),int(row[1])),[]).append(j)
        assert data['GT_presence']==[dict(frame=f,original_GT_ID=pid,GT_raw_indices=ii) for (f,pid),ii in sorted(presence.items())]
        table={p['pair_id']:p for p in data['records']};assert set(table)=={p['pair_id'] for p in r['pairs']}
        labels={};covisible={};bracket_counts=Counter();witness=[]
        for p in r['pairs']:
            support=independent.brackets(r['rows'],r['canon'],*p['canonical_ids'],r['runs'])
            own=independent.weak_pair(support,owners,r['rows'],presence);saved=table[p['pair_id']]
            assert own['label']==saved['class'] and own['counts']==saved['bracket_counts'] and own['co_visible']==saved['co_visible']
            savedbr={tuple(br['raw_indices']):br for br in saved['brackets']}
            for br,label in zip(support,own['brackets']):
                proof=savedbr[tuple(br['raw_indices'])];assert proof['class']==label
                ix=br['raw_indices'];assert proof['owners']==owners[ix].tolist()
                actualgt=[-1 if min(owners[j])<0 else lookup[tuple(map(int,owners[j]))] for j in ix]
                assert proof['canonical_GT_ids']==actualgt
                bracket_counts[label]+=1
            labels[p['pair_id']]=own['label'];covisible[p['pair_id']]=own['co_visible']
            if r['ref']['domain'] in ('35','33') and p['full_support']:
                witness.append(dict(pair_id=p['pair_id'],label=own['label'],co_visible=own['co_visible'],X2=p['X2'],Xminus2=p['Xminus2'],bracket_counts=own['counts']))
        r.update(GT=GT,labels=labels,covisible=covisible)
        reports.append(dict(clip=key,pairs=dict(Counter(labels.values())),brackets=dict(bracket_counts),co_visible_D=sum(covisible.values()),head_full_support=witness))
        print('AUDIT210_LABELS',key,flush=True)
    prior.save(args.root/'verification/labels.json',dict(passed=True,seals=seals,clips=reports,scope='All candidate three-row weak labels, pair mixed/multi-original aggregation and middle-date two-ID annotation presence independently rebuilt from exact previously independently verified208 full-frame owners and original GT run mapping. No new label Hungarian done; source208 owner rows pinned to exactK. Co-visible means annotations present, not physical visibility.'))

def model_stage(args,records,seals):
    pred=prior.read(args.root/'predictions/predictions.json');choices={};reports=[];compact=[]
    allrows={key+'/'+p['pair_id']:dict(p,domain=r['ref']['domain'],clip=r['ref']['clip'],y=int(r['labels'][p['pair_id']]=='S')) for key,r in records.items() for p in r['pairs'] if r['labels'][p['pair_id']]!='U'}
    for heldout in DOMAINS:
        model=prior.read(args.root/'predictions/models'/f'{heldout}.json');assert model['heldout']==heldout and set(model['source_domains'])==set(DOMAINS)-{heldout}
        expected=sorted(k for k,p in allrows.items() if p['domain']!=heldout);training=model['training_records'];assert [t['key'] for t in training]==expected
        for t in training:
            row=allrows[t['key']];assert t['y']==row['y'] and t['domain']!=heldout and t['full_support']==row['full_support']
            for feature in ('G6','X2','Xminus2'):prior.close(t[feature],row[feature])
        ownG=refit(training,'G6',np.zeros(len(training)))
        full=[t for t in training if t['full_support']]
        frozen=model['G'];offset=np.array([logit(frozen,t['G6']) for t in full]);ownC=refit(full,'X2',offset)
        for own,actual,cohort,wname in ((ownG,model['G'],training,'weight_G'),(ownC,model['correction'],full,'weight_correction')):
            assert own['status']==actual['status']
            if own['status']=='fitted':
                for f in ('mean','scale','beta','intercept'):prior.close(own[f],actual[f],2e-6)
                prior.close(own['weight'],[t[wname] for t in cohort]);assert max(abs(own['gradient']))<2e-6
            else:assert actual['intercept']==0 and all(x==0 for x in actual['beta']) and actual['mean'] is None and actual['scale'] is None
        for t in training:prior.close(t['G_offset'],logit(model['G'],t['G6']))
        for key,r in records.items():
            if r['ref']['domain']!=heldout:continue
            saved=pred['clips'][key];assert saved['slate']==r['slate'] and saved['active_slate']==r['active'] and saved['k']==r['k']
            savedrows={row['pair_id']:row for row in saved['records']};scores={};table={p['pair_id']:p for p in r['pairs']}
            for p in r['pairs']:
                g=logit(model['G'],p['G6']);x=g;m=g
                if p['full_support']:
                    x+=logit(model['correction'],p['X2']);m+=logit(model['correction'],p['Xminus2'])
                ss=dict(G=g,X=x,M=m);scores[p['pair_id']]=ss
                for head,z in ss.items():prior.close(z,savedrows[p['pair_id']]['logits'][head],1e-11)
            ids=lambda pid:tuple(map(int,pid.split('|')))
            selected=dict(K=[],ALL=list(r['active']),N=sorted(r['active'],key=lambda pid:(table[pid]['min_offset_norm'],ids(pid)))[:r['k']])
            for head in ('G','X','M'):selected[head]=sorted(r['active'],key=lambda pid:(-scores[pid][head],ids(pid)))[:r['k']]
            assert selected==saved['selected'];choices[key]=selected;r['scores']=scores
            compact.append(dict(clip=key,slate=r['slate'],active_slate=r['active'],k=r['k'],selected=selected,rows=[dict(pair_id=p['pair_id'],canonical_ids=p['canonical_ids'],min_offset_norm=p['min_offset_norm'],G6=p['G6'],X2=p['X2'],Xminus2=p['Xminus2'],full_support=p['full_support'],active=p['active'],weak=r['labels'][p['pair_id']],co_visible=r['covisible'][p['pair_id']],logits=scores[p['pair_id']]) for p in r['pairs']]))
        reports.append(dict(heldout=heldout,known_source=len(training),full_known_source=len(full),G_status=model['G']['status'],correction_status=model['correction']['status']))
        print('AUDIT210_REFIT',heldout,flush=True)
    prior.save(args.root/'verification/models.json',dict(passed=True,seals=seals,folds=reports,scope='All source-only candidate cohorts incl conflicts/inactive, independently class/domain-balanced G and full-support correction weights/moments/convex refits; frozenG offset, unsupported exact bypass, M sign/sharedscaler and all fixed-k/slate/logit masks rebuilt.'))
    prior.save(args.root/'verification/numeric_candidates.json',dict(schema='interleave210-independent-numeric-v1',seals=seals,clips=compact))
    return choices

def diagnostics(args,records,seals):
    targets={}
    for domain in DOMAINS:
        full=[(r,p) for r in records.values() if r['ref']['domain']==domain for p in r['pairs'] if p['full_support']]
        known=[(r,p) for r,p in full if r['labels'][p['pair_id']]!='U']
        y=[int(r['labels'][p['pair_id']]=='S') for r,p in known]
        scores={head:[r['scores'][p['pair_id']][head] for r,p in known] for head in ('G','X','M')}
        AP={head:prior.independent.average_precision(y,values) for head,values in scores.items()}
        enough=sum(y)>=5 and len(y)-sum(y)>=5
        targets[domain]=dict(full_population=len(full),known=len(known),S=sum(y),D=len(y)-sum(y),U=len(full)-len(known),co_visible_D=sum(r['covisible'][p['pair_id']] for r,p in known),AP=AP,sufficient=enough,passes=None if not enough else AP['X']>AP['G'] and AP['X']>AP['M'])
    sufficient=all(targets[h]['sufficient'] for h in ('35','33'))
    prior.save(args.root/'verification/diagnostics.json',dict(passed=True,seals=seals,targets=targets,attribution=dict(sufficient=sufficient,passed=None if not sufficient else all(targets[h]['passes'] for h in ('35','33'))),scope='Unpruned full-support heldout weak-label S/D/U population; grouped tied-logit AP per target separately. Both heads require5S/5D then each X>G,M; no pooling heads to hide failure. These are reliable annotation supports, not physical same-bee truth.'))

def physical_stage(args,records,choices,seals,helpers):
    replay,previous,native,v,_=helpers;summary=prior.read(args.root/'score/summary.json')
    pooled={h:{arm:dict(numerator=0,denominator=0,iou={t:dict(tp=0,pred=0,gt=0) for t in ('0.5','0.3')}) for arm in ARMS} for h in DOMAINS}
    reports=[]
    for key,r in records.items():
        rows=r['rows'];GT=r['GT'];polygon=r['meta']['polygon'];nf=r['meta']['nf'];truth=native.track_structure(GT,polygon)
        gtbin=independent.clock(list(truth['events'].values()),nf);cache=prior.NativeCache(GT,rows,v);methods={};K=None
        table={p['pair_id']:p for p in r['pairs']}
        checkpoint=args.root/'verification/physical_clips'/r['ref']['domain']/(r['ref']['clip']+'.json')
        for arm in ARMS:
            selected=choices[key][arm];pairs=[tuple(map(int,p.split('|'))) for p in selected]
            expected,output_indices=independent.merge(rows,r['canon'],pairs,return_indices=True)
            folder=args.root/'score/clips'/r['ref']['domain']/r['ref']['clip']/arm;actual=prior.read(folder/'score.json');metric=actual['metrics']
            disk=np.loadtxt(folder/'mot.txt',delimiter=',',usecols=range(7),ndmin=2)
            assert np.array_equal(expected,disk) and len(expected)==len(rows)
            assert actual['proof']['output_raw_indices']==output_indices and np.load(folder/'output_raw_indices.npy').tolist()==output_indices
            assert sorted(output_indices)==list(range(len(rows))) and actual['proof']['deleted_raw_indices']==[]
            P=native.track_structure(expected,polygon);predbin=independent.clock(list(P['events'].values()),nf)
            if arm=='K':K=predbin.copy();assert expected.tobytes()==rows.tobytes()
            additive=K.copy()
            for pid in selected:additive+=np.asarray(table[pid]['delta_windows'],np.int64)
            assert np.array_equal(additive,predbin)
            own=native.score_from_owners(P,truth,cache.owners(expected,output_indices),polygon,nf)
            assert np.array_equal(metric['pred_bins'],predbin) and np.array_equal(metric['gt_bins'],gtbin)
            numerator=int(abs(predbin-gtbin).sum());denominator=int(gtbin.sum())
            assert numerator==actual['numerator'] and denominator==actual['denominator']
            for t,e in (('0.5','edges05'),('0.3','edges03')):
                for field in ('tp','pred','gt','tp_id'):assert own['iou'][t][field]==metric['iou'][t][field],(key,arm,t,field)
                assert own['edges'][t]==metric['evidence'][e]
                assert actual['TP_FP_FN'][t]['fp']==own['iou'][t]['pred']-own['iou'][t]['tp'] and actual['TP_FP_FN'][t]['fn']==own['iou'][t]['gt']-own['iou'][t]['tp']
            if arm=='K':assert metric==prior.read(args.reference209/'clips'/r['ref']['domain']/r['ref']['clip']/'K/score.json')['metrics']
            m=dict(numerator=numerator,denominator=denominator,selected=selected,iou={t:{f:metric['iou'][t][f] for f in ('tp','pred','gt')} for t in ('0.5','0.3')})
            methods[arm]=m;target=pooled[r['ref']['domain']][arm];target['numerator']+=numerator;target['denominator']+=denominator
            for t in target['iou']:
                for f in ('tp','pred','gt'):target['iou'][t][f]+=metric['iou'][t][f]
        report=dict(clip=key,rows=len(rows),methods=methods,scene_frame_requests=cache.requests,unique_geometry_orders=len(cache.cache),all_original_rows_once=True)
        reports.append(report);prior.save(checkpoint,dict(passed=True,seals=seals,report=report))
        print('AUDIT210_ALL_NATIVE',key,flush=True)
    for h,arms in pooled.items():
        for arm,m in arms.items():
            saved=summary['domains'][h]['methods'][arm]
            assert m['numerator']==saved['numerator'] and m['denominator']==saved['denominator']
            m['W_fraction']=[100*m['numerator'],m['denominator']];m['F1_fraction']={}
            for t,c in m['iou'].items():
                for f in ('tp','pred','gt'):assert c[f]==saved['iou'][t][f]
                m['F1_fraction'][t]=[2*c['tp'],max(1,c['pred']+c['gt'])]
    W=lambda h,a:Fraction(*pooled[h][a]['W_fraction'])
    F=lambda h,a:Fraction(*pooled[h][a]['F1_fraction']['0.5'])
    checks=dict(head35_gain_at_least1=W('35','K')-W('35','X')>=1,head35_beats_N_G=all(W('35','X')<W('35',a) for a in ('N','G')),head33_loss_at_most_halfpoint=W('33','X')-W('33','K')<=Fraction(1,2),head_F1_loss_at_most_onehundredth=all(F(h,'X')>=F(h,'K')-Fraction(1,100) for h in ('35','33')))
    prior.save(args.root/'verification/physical.json',dict(passed=True,seals=seals,clips=reports,pooled=pooled,registered_gate=dict(checks=checks,passed=all(checks.values())),scope='All36 complete physicalMOTs independently reconstructed; exact original-row bijection/date/box/score retention, immutablecanon/min-ID and chronological order checked. All36 fresh fullscene Hungarian/dualIoU event graphs/maxTP/TP_ID rebuilt using prior209 exact ordered-original-row cache; all event dates/independent30clock/window numerators/additive closure and K209 anchors exact. WG raw counts pool, not E3. Shared official low-level geometry+proven public197 event scoring, no producer model/selector/scorer used.'))

def run(args):
    if not args.field_go or not args.physical_finished:raise ValueError('requires root field GO plus soleCPU release')
    sys.path.insert(0,str(args.repo.resolve()))
    # Pins populated from the formal worker handoff, not guessed from a draft.
    pins=prior.read(args.seal_pins)
    seals={name:prior.seal(args.root/name,digest) for name,digest in pins.items()}
    assert {'catalog','labels','predictions','score'}<=set(seals)
    replay=prior.module('audit210_public203',args.repo/'tools/study203/verify203.py');helpers=(replay,*replay.helpers(args.repo))
    records=catalog_stage(args,seals,helpers)
    label_stage(args,records,seals)
    choices=model_stage(args,records,seals)
    diagnostics(args,records,seals)
    physical_stage(args,records,choices,seals,helpers)
    sources={p.name:prior.sha(p) for p in (Path(__file__),Path(independent.__file__))}
    snapshot=args.root/'verification/source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in sources:(snapshot/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    reports={name:prior.sha(args.root/'verification'/name) for name in ('catalog.json','labels.json','models.json','numeric_candidates.json','diagnostics.json','physical.json')}
    prior.save(args.root/'verification/independent.json',dict(passed=True,prereg=PREREG,seals=seals,reports=reports,source_sha256=sources,prior_independent209_helpers_sha256=prior.sha(prior.__file__),scope='See staged receipts for reconstructed versus reused208 reliable owners/shared low-level helpers. No images, new detector seeds, test or GPU.'))
    print('AUDIT210_PASS',prior.sha(args.root/'verification/independent.json'),flush=True)

if __name__=='__main__':print(json.dumps(self_test(),sort_keys=True))
