"""Independent212 refit, all probabilities/guards and old86 graph replay."""
import argparse
from collections import defaultdict,Counter
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit
import field212_features as source
independent=source.independent;prior=source.prior;previous=source.previous
HEADS=('G','R','T','S');POLICIES=('K','C',*HEADS)

def feature(f):return dict(f,S_residual=(np.asarray(f['R_alt'])-f['R']).tolist())
def probability(model,f,head):
    h=model['heads']['T' if head=='S' else head]
    if h['status']!='fitted':return h['constant_probability']
    x=np.asarray(independent.model_vector(feature(f),head));return float(expit(((x-h['mean'])/h['scale'])@h['beta']+h['intercept']))
def graph_outcome(metric,nf):
    ev=metric['evidence'];left=[int(v[0]) for v in ev['predictions']];right=[int(v[0]) for v in ev['gt']]
    assert len(set(left))==len(left) and len(set(right))==len(right)
    bins=[]
    for vertices in (ev['predictions'],ev['gt']):bins.append(previous.geometry.clock([(abs(int(v[1])),1 if v[1]>0 else -1) for v in vertices],nf))
    assert np.array_equal(bins[0],metric['pred_bins']) and np.array_equal(bins[1],metric['gt_bins'])
    value=dict(numerator=int(abs(bins[0]-bins[1]).sum()),denominator=int(bins[1].sum()),nPred=len(left),nGT=len(right),iou={})
    for t,key in (('0.5','edges05'),('0.3','edges03')):
        tp=previous.independent.maximum_matching(left,right,ev[key]);assert tp==metric['iou'][t]['tp']
        value['iou'][t]=dict(TP=tp,Pred=len(left),GT=len(right),FP=len(left)-tp,FN=len(right)-tp,E=len(left)+len(right)-2*tp,F1=2*tp/(len(left)+len(right)) if left or right else 0.)
    value['W']=100*value['numerator']/value['denominator'] if value['denominator'] else None
    return value
def W(x):return Fraction(100*x['numerator'],x['denominator']) if x['denominator'] else None
def F(x):
    q=x['iou']['0.5'];return Fraction(2*q['TP'],max(1,q['Pred']+q['GT']))
def normalize_states(r):
    c=r['meta'];answer=[]
    peer_windows=[prior.read(r['folder']/next(s for s in c['scenes'] if s['scene_ref']==c['K_scene_refs'][str(i)])['events_file'])['windows'] for i in (1,2)]
    for s in c['states']:
        events=prior.read(r['folder']/next(x for x in c['scenes'] if x['scene_ref']==s['scene_ref'])['events_file'])['events']
        distance=sum(int(abs(np.asarray(s['windows'])-np.asarray(pw)).sum()) for pw in peer_windows)/2
        assert distance==s['raw_meanCountL1'] and len(events)==s['nPred']
        answer.append(dict(s,events=events,meanCountL1=distance))
    return answer

def run(args):
    if not args.field_go:raise ValueError('soleCPU GO required')
    sys.path.insert(0,str(args.repo.resolve()));assert prior.read(args.root/'verification/features.json')['passed'] and prior.read(args.root/'verification/labels.json')['passed']
    top,records=source.load_records(args);bank=prior.read(args.root/'frozen'/top['feature_bank_file'])['features']
    P=prior.read(args.root/'predictions/predictions.json');models={};model_reports=[];receiver=[]
    for key,r in records.items():
        c=r['meta'];scene=next(s for s in c['scenes'] if s['scene_ref']==c['K_scene_refs']['0'])
        for e in prior.read(r['folder']/scene['events_file'])['events']:receiver.append(dict(e,domain=c['domain'],clip=c['clip'],role='receiverK',member=scene['member'],member_index=0))
        r['states']=normalize_states(r)
    labels={e['event_ref']:e for h in ('35','33','W','G') for e in prior.read(args.root/'labels'/f'{h}.json')['records']}
    for heldout in ('35','33','W','G'):
        model=prior.read(args.root/'predictions/models'/f'{heldout}.json');assert model['heldout']==heldout and set(model['source_domains'])==set(('35','33','W','G'))-{heldout}
        expected=sorted([e for e in receiver if e['domain']!=heldout],key=lambda e:(e['domain'],e['clip'],e['event_ref']));training=model['training_records'];assert len(training)==len(expected)
        w=independent.event_weights(expected);w=w/w.sum() if len(w) else w;yy=np.asarray([labels[e['event_ref']]['y'] for e in expected])
        for i,(a,e) in enumerate(zip(training,expected)):
            for f in ('domain','clip','event_ref','role','member_index','feature_key'):assert a[f]==e[f]
            assert a['role']=='receiverK' and a['member_index']==0 and a['y']==yy[i]
            prior.close(a['weight'],w[i]);prior.close(a['features19'],independent.model_vector(feature(bank[e['feature_key']]),'T'))
        for head in ('G','R','T'):
            X=np.asarray([independent.model_vector(feature(bank[e['feature_key']]),head) for e in expected]);own=independent.fit_logistic(X,yy,w);actual=model['heads'][head]
            if own['status']=='fitted':
                assert actual['status']=='fitted'
                for f in ('mean','scale','beta','intercept'):prior.close(own[f],actual[f],3e-6)
                prior.close(own['objective'],actual['optimization']['objective'],1e-8)
                assert max(abs(own['gradient']))<2e-6
            else:
                assert actual['status'] in ('empty','single_class_smoothed') and actual['constant_probability']==own['probability'] and all(b==0 for b in actual['beta'])
        models[heldout]=model;model_reports.append(dict(heldout=heldout,source_events=len(expected),source_receivers_only=True,independent_refits=3))
        print('AUDIT212_REFIT',heldout,flush=True)
    prior.save(args.root/'verification/models.json',dict(passed=True,folds=model_reports,scope='All12 G/R/T source-only receiverK logistic fits independently reconstruct labels/weights/constant-aware moments/coefficient optima and raw-n fallbacks. S never fitted: sameT parameters/scaler with Ralt-R. Source peer labels excluded.'))
    evaluation=prior.read(args.root/'evaluation/summary.json');old=prior.read(args.joint_assessment/'assessment.json');oldmaps={c['domain']+'/'+c['clip']:c['states'] for c in old['clips']}
    outcomes={};choices={};members=[];clip_reports=[]
    for key,r in records.items():
        c=r['meta'];pc=P['clips'][key];model=models[c['domain']];ownstates=r['states'];byid={s['state_id']:s for s in ownstates};K=next(s for s in ownstates if not s['selected_action_ids']);pid=K['state_id']
        actions={a:dict(family=v['family'],endpoint_ids=v['canonical_ids']) for a,v in c['original211_actions'].items()}
        assert {s['state_id'] for s in pc['states']}==set(byid)
        probability_cache={key:{h:probability(model,f,h) for h in HEADS} for key,f in bank.items()}
        probabilities={h:{} for h in HEADS};savedstates={s['state_id']:s for s in pc['states']};state_outcomes={}
        for s in ownstates:
            sid=s['state_id'];saved=savedstates[sid];assert len(saved['event_predictions'])==s['nPred']
            for e,a in zip(s['events'],saved['event_predictions']):
                assert e['event_ref']==a['event_ref'] and e['feature_key']==a['feature_key']
                for h in HEADS:prior.close(probability_cache[e['feature_key']][h],a['probabilities'][h],1e-13)
            for h in HEADS:probabilities[h][sid]=[probability_cache[e['feature_key']][h] for e in s['events']]
            metric=oldmaps[key][sid]['metrics'];assert oldmaps[key][sid]['scope_digest']==s['original211_scope_digest'];value=graph_outcome(metric,c['nf'])
            state_outcomes[sid]=value
            stored=prior.read(args.root/'evaluation/states'/key/(sid+'.json'));assert stored['metrics']==metric
            for f in ('numerator','denominator','nPred','nGT'):assert value[f]==stored['outcome'][f]
            for t in value['iou']:
                for f in ('TP','Pred','GT','FP','FN','E'):assert value['iou'][t][f]==stored['outcome']['iou'][t][f]
        outcomes[key]=state_outcomes
        count=min(ownstates,key=lambda s:(-round(K['meanCountL1']-s['meanCountL1'],10),s['edits'],tuple(sorted((actions[a]['family'],*actions[a]['endpoint_ids']) for a in s['selected_action_ids']))))
        assert count['state_id']==c['original211_C_chosen_state_id']==pc['C_state_id']
        selected={'K':pid,'C':count['state_id']}
        for h in HEADS:
            choice=independent.choose_guarded(ownstates,probabilities[h],actions,pid);selected[h]=choice['state_id']
            assert pc['selected'][h]['state_id']==choice['state_id']
            baseH=float(K['nPred']-2*sum(probabilities[h][pid]))
            for sid,s in byid.items():
                H=float(s['nPred']-2*sum(probabilities[h][sid]));prior.close(H,savedstates[sid]['Ehat'][h]);prior.close(H-baseH,savedstates[sid]['delta_Ehat'][h]);assert (H-baseH<=1e-9)==savedstates[sid]['guard_eligible'][h]
        for h in ('K','C'):assert pc['selected'][h]['state_id']==selected[h]
        choices[key]=selected
        for original in pc['originals']:
            scene=next(s for s in c['scenes'] if s['scene_ref']==original['scene_ref']);assert scene['role']==original['role'] and scene['member_index']==original['member_index']
            gtref=next(x for x in prior.read(args.root/'labels/deployment.json')['original_K'] if (x['domain'],x['clip'],x['scene_ref'])==(c['domain'],c['clip'],scene['scene_ref']))
            graph=prior.read(args.root/'labels'/gtref['graphs']['0.5']);ys={e['canonical_event_id']:int(any(p==e['canonical_event_id'] for p,g in graph['edges'])) for e in graph['events']}
            events=prior.read(r['folder']/scene['events_file'])['events'];assert len(events)==len(original['event_predictions'])==len(ys);sse={h:0. for h in HEADS}
            for e,a in zip(events,original['event_predictions']):
                assert e['event_ref']==a['event_ref'] and e['canonical_id'] in ys
                for h in HEADS:
                    p=probability_cache[e['feature_key']][h];prior.close(p,a['probabilities'][h],1e-13);sse[h]+=(p-ys[e['canonical_id']])**2
            n=len(events);positive=sum(ys.values());members.append(dict(domain=c['domain'],clip=c['clip'],role=scene['role'],member=str(scene['member']),member_index=scene['member_index'],n=n,positive=positive,negative=n-positive,both_classes=0<positive<n,SSE=sse,Brier={h:sse[h]/n if n else None for h in HEADS}))
        clip_reports.append(dict(domain=c['domain'],clip=c['clip'],selected=selected,policies={p:state_outcomes[s] for p,s in selected.items()}))
        print('AUDIT212_PROBABILITIES_GUARDS',key,flush=True)
    assert sum(len(x) for x in outcomes.values())==86 and len(members)==18
    for own in members:
        actual=next(m for m in evaluation['Brier_members'] if (m['domain'],m['clip'],m['member_index'])==(own['domain'],own['clip'],own['member_index']))
        for f in ('n','positive','negative','both_classes'):assert own[f]==actual[f]
        for h in HEADS:
            if own['Brier'][h] is None:assert actual['Brier'][h] is None
            else:prior.close(own['Brier'][h],actual['Brier'][h],1e-12)
    domains={}
    for domain in ('35','33','W','G'):
        domains[domain]={}
        for p in POLICIES:
            cells=[c['policies'][p] for c in clip_reports if c['domain']==domain];q=dict(numerator=sum(x['numerator'] for x in cells),denominator=sum(x['denominator'] for x in cells),iou={})
            for t in ('0.5','0.3'):
                q['iou'][t]={f:sum(x['iou'][t][f] for x in cells) for f in ('TP','Pred','GT','FP','FN','E')}
            actual=evaluation['domains'][domain][p];assert q['numerator']==actual['numerator'] and q['denominator']==actual['denominator']
            for t in q['iou']:
                for f in q['iou'][t]:assert q['iou'][t][f]==actual['iou'][t][f]
            domains[domain][p]=q
    W=lambda q:Fraction(100*q['numerator'],q['denominator']) if q['denominator'] else None
    F=lambda q:Fraction(2*q['iou']['0.5']['TP'],max(1,q['iou']['0.5']['Pred']+q['iou']['0.5']['GT']))
    checks={}
    for h in ('35','33'):
        checks[h+'_W_gain_at_least_1pp']=W(domains[h]['K'])-W(domains[h]['T'])>=1
        checks[h+'_F1_loss_at_most_0.01']=F(domains[h]['T'])>=F(domains[h]['K'])-Fraction(1,100)
    for h in ('W','G'):checks[h+'_W_not_worse']=W(domains[h]['T'])<=W(domains[h]['K'])
    assert checks==evaluation['primary_gate']['checks'] and all(checks.values())==evaluation['primary_gate']['passed']
    reserve=[m for m in members if m['domain'] in ('35','33') and m['role']=='peerK'];sufficient=len(reserve)==4 and all(m['both_classes'] for m in reserve)
    comparisons=[m['Brier']['T'] is not None and m['Brier'][h] is not None and m['Brier']['T']<m['Brier'][h] for m in reserve for h in ('G','R','S')]
    differences=[choices[c['domain']+'/'+c['clip']]['T']!=choices[c['domain']+'/'+c['clip']][h] and W(c['policies']['T'])<W(c['policies'][h]) for c in clip_reports if c['domain'] in ('35','33') for h in ('R','S')]
    passed=sufficient and all(comparisons) and any(differences)
    assert evaluation['phase_attribution']['passed']==passed and evaluation['phase_attribution']['status']==('inconclusive' if not sufficient else 'passed' if passed else 'not_passed')
    # Cross-clip member pooling and member means retain three correlated members.
    pooled=defaultdict(list)
    for m in members:pooled[m['domain'],m['member'],m['member_index'],m['role']].append(m)
    for identity,cells in pooled.items():
        n=sum(x['n'] for x in cells);actual=next(x for x in evaluation['Brier_member_means']['pooled_original_members'] if (x['domain'],str(x['member']),x['member_index'],x['role'])==identity)
        assert actual['n']==n
        for h in HEADS:
            value=sum(x['SSE'][h] for x in cells)/n if n else None
            if value is None:assert actual['Brier'][h] is None
            else:prior.close(value,actual['Brier'][h],1e-12)
    prior.save(args.root/'verification/selections.json',dict(passed=True,choices=choices,scope='All86 states/all18K G/R/T/S probabilities independently computed; S fixedT weights/scaler and Ralt-R, everyH guard/relativeK/round10/C-control/familyID tie and sixchoices reproduced.'))
    prior.save(args.root/'verification/evaluation.json',dict(passed=True,domains=domains,Brier_members=members,primary_gate=dict(checks=checks,passed=all(checks.values())),phase_attribution=dict(sufficient=sufficient,passed=passed),scope='All86 old official event graphs maximum matching and signed-date30clock independently replayed; sixpolicy raw integer pool/F1/E and all18K/member Briers plus crossclip member pools/gates reproduced. This stage ran zero fresh86MOT/GT scorers. The four reserve head members share two hives/clips/GT and are not independent external trials.'))
    snap=args.root/'verification/source_snapshot_models';snap.mkdir(exist_ok=True);(snap/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('AUDIT212_MODELS_EVALUATION_PASS',prior.sha(args.root/'verification/evaluation.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--joint-assessment',type=Path,required=True);p.add_argument('--field-go',action='store_true');run(p.parse_args())
