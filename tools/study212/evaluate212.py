"""Post-common-seal 212 evaluation of existing 211 physical outcomes.

No tracking, image, GT-file or MOT-scoring entrance exists here. Every receiver
outcome is replayed from the signed old fullscene native event graph and clock.
Original-K legal-neighbor labels/forensics are separately supplied by truth212.
Member Brier observations are never deduplicated by the feature-cache key.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import fit212 as fit
import prepare212 as prepare
import truth212 as truth

ARMS=('K','C','G','R','T','S')
HEADS=('G','R','T','S')
DOMAINS=('35','33','W','G')
JOINT211_SEAL='a6446982277d9d3fee3b0efb786e6a1248536899ec04bb842d6ae08fe399f85f'
JOINT211_ASSESSMENT='5b6d317422e5a18b1e1eade522a828cff41ae6cafb15c4247cf7002de75e3861'


def read(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf8') if path.suffix=='.gz' else path.open(encoding='utf8')) as f:return json.load(f)


def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    prepare.save(path,obj)


def digest(obj):return hashlib.sha256(prepare.wire(obj)).hexdigest()


def fraction(value):return None if value is None else dict(numerator=value.numerator,denominator=value.denominator,value=float(value))


def outcome(metrics,nf):
    """Recompute graph maximum cardinalities and full 30-offset integer clock."""
    from tools.study197.candidates import window_counts
    p=metrics['evidence']['predictions'];g=metrics['evidence']['gt']
    left=[int(x[0]) for x in p];right=[int(x[0]) for x in g]
    if len(set(left))!=len(left) or len(set(right))!=len(right):raise ValueError('duplicate event vertex')
    expected=[]
    for vertices in (p,g):
        events=[dict(frame=abs(int(x[1])),direction=1 if x[1]>0 else -1) for x in vertices]
        if any(int(x[1])==0 for x in vertices):raise ValueError('zero signed event date')
        expected.append(window_counts(events,nf).astype(np.int64))
    pb,gb=map(lambda x:np.asarray(x,np.int64),(metrics['pred_bins'],metrics['gt_bins']))
    if not np.array_equal(pb,expected[0]) or not np.array_equal(gb,expected[1]):raise ValueError('official event-date/30-offset bins disagree')
    if metrics['nP']!=len(p) or metrics['nG']!=len(g):raise ValueError('event denominator disagreement')
    result=dict(numerator=int(np.abs(pb-gb).sum()),denominator=30*len(g),nPred=len(p),nGT=len(g),
                pred_bins=pb.tolist(),gt_bins=gb.tolist(),iou={})
    result['W']=float(Fraction(100*result['numerator'],result['denominator'])) if result['denominator'] else None
    for t,key in (('0.5','edges05'),('0.3','edges03')):
        matching=truth.maximum(left,right,metrics['evidence'][key]);tp=len(matching)
        if tp!=int(metrics['iou'][t]['tp']):raise ValueError('full legal graph does not reproduce native TP')
        result['iou'][t]=dict(TP=tp,Pred=len(p),GT=len(g),FP=len(p)-tp,FN=len(g)-tp,E=len(p)+len(g)-2*tp,
            F1=2*tp/(len(p)+len(g)) if len(p)+len(g) else 1.)
    return result


def pool(items):
    """Pool raw counts across the two actual W/G clips before division."""
    if not items:raise ValueError('empty policy pool')
    result={k:sum(x[k] for x in items) for k in ('numerator','denominator','nPred','nGT')}
    result['W']=100*result['numerator']/result['denominator'] if result['denominator'] else None
    result['iou']={}
    for t in ('0.5','0.3'):
        q={k:sum(x['iou'][t][k] for x in items) for k in ('TP','Pred','GT','FP','FN','E')}
        q['F1']=2*q['TP']/(q['Pred']+q['GT']) if q['Pred']+q['GT'] else 1.;result['iou'][t]=q
    return result


def W(x):return Fraction(100*x['numerator'],x['denominator']) if x['denominator'] else None


def f1(x,t='0.5'):
    q=x['iou'][t];den=q['Pred']+q['GT'];return Fraction(2*q['TP'],den) if den else Fraction(1)


def gates(domains):
    checks={}
    for h in ('35','33'):
        k,t=domains[h]['K'],domains[h]['T']
        checks[h+'_W_gain_at_least_1pp']=W(k) is not None and W(t) is not None and W(k)-W(t)>=1
        checks[h+'_F1_loss_at_most_0.01']=f1(t)>=f1(k)-Fraction(1,100)
    for h in ('W','G'):checks[h+'_W_not_worse']=W(domains[h]['T']) is not None and W(domains[h]['K']) is not None and W(domains[h]['T'])<=W(domains[h]['K'])
    return dict(passed=all(checks.values()),checks=checks)


def brier(original,graph):
    left=list(map(int,graph['pred_vertices']));right=list(map(int,graph['GT_vertices']))
    if len(set(left))!=len(left) or len(set(right))!=len(right):raise ValueError('truth duplicate vertices')
    neighbors=defaultdict(list)
    for p,g in graph['edges']:neighbors[int(p)].append(int(g))
    tp=len(truth.maximum(left,right,graph['edges']))
    if (len(left),len(right),tp,len(left)-tp,len(right)-tp)!=(graph['nPred'],graph['nGT'],graph['TP'],graph['FP'],graph['FN']):raise ValueError('original K true graph cardinality differs')
    for e in graph['events']:
        ns=sorted(set(neighbors[int(e['canonical_event_id'])]))
        if e['y']!=int(bool(ns)) or sorted(e['legal_GT_neighbors'])!=ns:raise ValueError('any-edge truth is not legal graph existence')
    events={int(e['canonical_event_id']):e for e in graph['events']}
    preds=original['event_predictions']
    if len(preds)!=len(events) or {int(e['canonical_id']) for e in preds}!=set(events):raise ValueError('K probabilities do not cover every true vertex exactly once')
    seen=set();sse={h:0. for h in HEADS};ys=[];rows=[]
    for e in sorted(preds,key=lambda e:e['event_ref']):
        if e['event_ref'] in seen:raise ValueError('duplicate event_ref')
        seen.add(e['event_ref']);y=events[int(e['canonical_id'])]['y']
        if y not in (0,1) or int(e['signed_frame'])!=int(events[int(e['canonical_id'])]['signed_frame']):raise ValueError('event truth/date join mismatch')
        ys.append(y)
        for h in HEADS:
            p=float(e['probabilities'][h])
            if not np.isfinite(p) or not 0<=p<=1:raise ValueError('invalid event probability')
            sse[h]+=(p-y)**2
        rows.append(dict(event_ref=e['event_ref'],canonical_id=e['canonical_id'],signed_frame=e['signed_frame'],y=y,
                         feature_key=e['feature_key'],probabilities=e['probabilities']))
    return dict(n=len(ys),positive=sum(ys),negative=len(ys)-sum(ys),both_classes=bool(ys) and 0<sum(ys)<len(ys),
                SSE=sse,Brier={h:sse[h]/len(ys) if ys else None for h in HEADS},events=rows,
                label_scope='any legal fullscene GT edge at IoU .5, not matched TP or bee identity')


def attribution(members,clip_results):
    reserve=[m for m in members if m['domain'] in ('35','33') and m['role']=='peerK']
    reasons=[];checks=[]
    if len(reserve)!=4 or len({(m['domain'],m['member_index']) for m in reserve})!=4:reasons.append('four distinct preregistered head reserve members not present')
    for m in reserve:
        if not m['both_classes']:reasons.append(m['domain']+'/m'+str(m['member_index'])+': insufficient positive/negative class denominator')
        for h in ('G','R','S'):
            ok=m['Brier']['T'] is not None and m['Brier'][h] is not None and m['Brier']['T']<m['Brier'][h]
            checks.append(dict(domain=m['domain'],clip=m['clip'],member=m['member'],member_index=m['member_index'],comparator=h,strictly_better=ok))
    differences=[]
    for c in clip_results:
        if c['domain'] not in ('35','33'):continue
        for h in ('R','S'):
            different=c['selected']['T']['state_id']!=c['selected'][h]['state_id']
            gain=W(c['policies'][h])-W(c['policies']['T']) if W(c['policies']['T']) is not None and W(c['policies'][h]) is not None else None
            differences.append(dict(domain=c['domain'],comparator=h,different_choice=different,W_gain=fraction(gain),different_and_strict_W_gain=different and gain is not None and gain>0))
    if not any(d['different_choice'] for d in differences):reasons.append('no head T choice differs from R or S')
    elif not any(d['different_and_strict_W_gain'] for d in differences):reasons.append('no differing head choice has strictly lower W')
    if not all(c['strictly_better'] for c in checks):reasons.append('T is not strictly better than every G/R/S reserve Brier comparator')
    insufficient=any(not m['both_classes'] for m in reserve) or len(reserve)!=4
    return dict(status='inconclusive' if insufficient else ('passed' if not reasons else 'not_passed'),passed=not reasons,
                reasons=reasons,reserve_checks=checks,choice_checks=differences,
                scope='four repeated old head reserve members; not four independent experiments')


def member_means(members):
    buckets=defaultdict(list)
    for m in members:buckets[(m['domain'],str(m['member']),m['member_index'],m['role'])].append(m)
    pooled=[]
    for (domain,member,index,role),items in sorted(buckets.items()):
        n=sum(x['n'] for x in items)
        pooled.append(dict(domain=domain,member=member,member_index=index,role=role,clips=[x['clip'] for x in items],n=n,
            positive=sum(x['positive'] for x in items),negative=sum(x['negative'] for x in items),
            Brier={h:sum(x['SSE'][h] for x in items)/n if n else None for h in HEADS}))
    means={}
    for domain in DOMAINS:
        group=[m for m in pooled if m['domain']==domain]
        if len(group)!=3:raise ValueError('three original members per domain required, no slot/global alias')
        means[domain]={h:sum(m['Brier'][h] for m in group)/3 if all(m['Brier'][h] is not None for m in group) else None for h in HEADS}
    return dict(pooled_original_members=pooled,domain_member_means=means,unit='mean of three member Briers after within-member cross-clip pooling')


def source_guard_audit(predictions,models,bank,state_outcomes):
    """Post-seal descriptive replay on source states; never fed to fitting."""
    rows=[]
    for heldout,model in models.items():
        for key,c in predictions['clips'].items():
            if c['domain']==heldout:continue
            evaluated=fit.choose_clip(c,model,bank);k=state_outcomes[key][evaluated['K_state_id']]
            for s in evaluated['states']:
                if s['state_id']==evaluated['K_state_id']:continue
                actual=state_outcomes[key][s['state_id']]
                de=actual['iou']['0.5']['E']-k['iou']['0.5']['E'];accepted=bool(s['guard_eligible']['T'])
                rows.append(dict(heldout=heldout,source_domain=c['domain'],clip=c['clip'],state_id=s['state_id'],
                    T_delta_Ehat=s['delta_Ehat']['T'],accepted=accepted,actual_delta_E05=de,
                    actual_W_gain=fraction(W(k)-W(actual)),count_utility=s['count_utility'],
                    category=('accepted_' if accepted else 'rejected_')+('actual_harm' if de>0 else ('actual_improvement' if de<0 else 'equal_actual_E'))))
    return dict(post_seal_descriptive_only=True,counts=dict(Counter(r['category'] for r in rows)),changes=rows,
                meaning='harm refers to whole-scene event error FP+FN at .5; source changes do not alter frozen choices')


def event_changes(prediction,old_states):
    """State-local action families; preserve signed-event multiplicity.

    K events are quotiented by THIS state's selected original parent pairs.
    A family with no post-edit net event is still represented by the action.
    Numeric canonical IDs may change across states and are never the join key.
    """
    states={s['state_id']:s for s in prediction['states']};K=states[prediction['K_state_id']]
    result={}
    for arm in ARMS:
        s=states[prediction['selected'][arm]['state_id']];parent_family={}
        for a in s['actions']:
            family=tuple(map(int,a['canonical_ids']))
            for p in family:
                if p in parent_family:raise ValueError('overlapping action-family parents')
                parent_family[p]=family
        def project(state):
            metric=old_states[state['state_id']]['metrics'];evidence=metric['evidence'];edges={t:defaultdict(list) for t in ('0.5','0.3')}
            for t,key in (('0.5','edges05'),('0.3','edges03')):
                for p,g in evidence[key]:edges[t][int(p)].append(int(g))
            records=[]
            for e in state['events']:
                parents=set(map(int,e['original_parent_ids']));expanded=set()
                for p in parents:expanded.update(parent_family.get(p,(p,)))
                signature=(tuple(sorted(expanded)),int(e['signed_frame']))
                records.append(dict(event_ref=e['event_ref'],original_parent_ids=sorted(parents),
                    comparison_family=list(signature[0]),signed_frame=signature[1],signature=[list(signature[0]),signature[1]],
                    legal_GT_neighbors={t:sorted(edges[t][int(e['canonical_id'])]) for t in edges}))
            return records
        before,after=project(K),project(s)
        b=Counter((tuple(e['comparison_family']),e['signed_frame']) for e in before)
        a=Counter((tuple(e['comparison_family']),e['signed_frame']) for e in after)
        def changes(records,remaining):
            records=sorted(records,key=lambda e:e['event_ref']);out=[]
            for e in records:
                key=(tuple(e['comparison_family']),e['signed_frame'])
                changed=remaining[key]>0
                if changed:remaining[key]-=1
                out.append(dict(e,changed=changed))
            return out
        before_changed=changes(before,b-a);after_changed=changes(after,a-b)
        result[arm]=dict(state_id=s['state_id'],selected_action_ids=s['selected_action_ids'],
            before=before_changed,after=after_changed,
            new_events_without_legal_edge05=sum(r['changed'] and not r['legal_GT_neighbors']['0.5'] for r in after_changed),
            interpretation='signed net-event changes within this state action-family quotient; retained signatures can still change legal neighbors')
    result['T_protection_vs_C']=dict(T_keeps_K=prediction['selected']['T']['state_id']==prediction['K_state_id'],
        T_equals_C=prediction['selected']['T']['state_id']==prediction['selected']['C']['state_id'],
        avoided_C_new_noedge_by_keeping_K=prediction['selected']['T']['state_id']==prediction['K_state_id'] and result['C']['new_events_without_legal_edge05']>0,
        physical_identity_claim=False,unknown_rule_reason='Use fixed true212 crossing-owner/direction/time forensics; no graph edge alone identifies the biological cause')
    return result


def run(args):
    # Mandatory boundary: verify the common four-fold prediction seal before any
    # label, old true-graph or target assessment file is opened.
    prediction=fit.load_predictions(args.predictions)
    receipt=prediction['receipt'];P=prediction['predictions']
    if receipt.get('prereg')!=args.prereg:raise ValueError('prediction prereg differs')
    prepare.verify(args.frozen);frozen=read(args.frozen/'deployment.json')
    prepare.verify(args.truth);T=read(args.truth/'deployment.json')
    prepare.verify(args.late_truth);late=read(args.late_truth/'deployment.json')
    prepare.verify(args.joint_assessment);J=read(args.joint_assessment/'assessment.json')
    fr=prepare.sha(args.frozen/'manifest.json');tr=prepare.sha(args.truth/'manifest.json');jr=prepare.sha(args.joint_assessment/'manifest.json')
    if jr!=JOINT211_SEAL or prepare.sha(args.joint_assessment/'assessment.json')!=JOINT211_ASSESSMENT:raise ValueError('old joint true archive differs authoritative raw-byte pins')
    if T['prereg']!=args.prereg or T['frozen_manifest_sha256']!=fr or frozen['prereg']!=args.prereg:raise ValueError('freeze/truth/prereg chain differs')
    if receipt['frozen_manifest_sha256']!=fr or receipt['label_manifest_sha256']!=tr:raise ValueError('common predictions trained against different frozen/label seals')
    if frozen['states']!=86 or frozen['originalK']!=18 or not J['all_legal_states_scored']:raise ValueError('incomplete original scope')
    joint_gate=args.joint_predictions/'ALL_FOUR_SEAL.json'
    if prepare.sha(joint_gate)!=frozen['identity']['joint_GTfree_seal'] or digest(read(joint_gate))!=J['prediction_seal']:raise ValueError('old graph not bound to same joint physical states')
    if args.out.exists():raise ValueError('complete result must not be overwritten')
    source_seals=dict(frozen=fr,truth=tr,predictions=prepare.sha(args.predictions/'manifest.json'),joint_assessment=jr,
                      late_truth=prepare.sha(args.late_truth/'manifest.json'),old_joint_gate_bytes=prepare.sha(joint_gate))
    # The late producer must bind the immutable early labels and all-fold choices.
    # Exact late deployment names are agreed with truth212; never edit early labels.
    if late['prereg']!=args.prereg or late['frozen_manifest_sha256']!=fr or late['label_manifest_sha256']!=tr or late['prediction_manifest_sha256']!=source_seals['predictions']:
        raise ValueError('late forensics not bound to exact common-sealed choices/early truth')
    old={(x['domain'],x['clip']):x for x in J['clips']};truthK={(x['domain'],x['clip'],x['scene_ref']):x for x in T['original_K']}
    clips=[];members=[];alloutcomes={};state_count=0
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.evaluate212-',dir=args.out.parent) as tmp:
        out=Path(tmp)/'payload';out.mkdir()
        for ref in frozen['clips']:
            c=prepare.load_clip(args.frozen,ref);key=c['domain']+'/'+c['clip'];pc=P['clips'][key]
            if set(pc['selected'])!=set(ARMS):raise ValueError('six policies required')
            if {s['state_id'] for s in pc['states']}!={s['state_id'] for s in c['states']}:raise ValueError('prediction and frozen all-state rosters differ')
            prior=old[c['domain'],c['clip']];state_map={};state_records=[]
            for s in c['states']:
                native=prior['states'][s['state_id']]
                if native['scope_digest']!=s['original211_scope_digest']:raise ValueError('old true graph physical scope differs')
                value=outcome(native['metrics'],c['nf'])
                if value['pred_bins']!=s['windows']:raise ValueError('frozen original state clock differs')
                state_map[s['state_id']]=value;state_count+=1
                path='states/'+key+'/'+s['state_id']+'.json'
                save(out/path,dict(domain=c['domain'],clip=c['clip'],state_id=s['state_id'],scope_digest=native['scope_digest'],
                    selected_action_ids=s['selected_action_ids'],metrics=native['metrics'],outcome=value,
                    original_joint_assessment_manifest_sha256=jr,native_replayed_not_rerun=True))
                state_records.append(dict(state_id=s['state_id'],path=path,scope_digest=native['scope_digest']))
            alloutcomes[key]=state_map
            result=dict(domain=c['domain'],clip=c['clip'],selected=pc['selected'],policies={},states=state_records,
                        event_changes=event_changes(pc,prior['states']))
            for a in ARMS:result['policies'][a]=state_map[pc['selected'][a]['state_id']]
            clips.append(result)
            for original in pc['originals']:
                meta=truthK[c['domain'],c['clip'],original['scene_ref']]
                for k in ('role','member','member_index'):
                    if (str(original[k]) if k=='member' else original[k])!=(str(meta[k]) if k=='member' else meta[k]):raise ValueError('original K member identity differs')
                graph=read(args.truth/meta['graphs']['0.5']);value=brier(original,graph)
                value.update(domain=c['domain'],clip=c['clip'],role=meta['role'],member=meta['member'],member_index=meta['member_index'],scene_ref=meta['scene_ref'])
                members.append(value)
            print('REPLAY212_COMPLETE',key,flush=True)
        if state_count!=86 or len(members)!=18 or len(clips)!=6:raise ValueError('complete 86/18/6 scope required')
        domains={h:{a:pool([c['policies'][a] for c in clips if c['domain']==h]) for a in ARMS} for h in DOMAINS}
        bank=read(args.frozen/frozen['feature_bank_file'])['features']
        bank={k:dict(G7=v['G'],R6=v['R'],P6=v['P'],R_alt6=v['R_alt']) for k,v in bank.items()}
        models={h:read(args.predictions/'models'/ (h+'.json')) for h in DOMAINS}
        # Source replay inputs need canonical actions, already preserved by fit.
        source_audit=source_guard_audit(P,models,bank,alloutcomes)
        forensic=[]
        for f in late['forensics']:
            detail=read(args.late_truth/f['path']);detail['frozen_policy_choices']=P['clips'][f['domain']+'/'+f['clip']]['selected']
            forensic.append(detail)
        summary=dict(schema='phase212-evaluation-v1',prereg=args.prereg,input_receipts=source_seals,clips=clips,domains=domains,
            primary_gate=gates(domains),Brier_members=members,Brier_member_means=member_means(members),
            phase_attribution=attribution(members,clips),source_guard_diagnostics=source_audit,
            original_fixed_forensics=forensic,scope=dict(receiver_states_replayed=86,receiver_MOT_reruns_this_stage=0,
                original_K_members=18,training_receivers=6,reserve_head_members=4,
                repeated_evidence='Same six clips and old members already used in 211/preceding experiments; no new holdout or independent replicate',
                identity_limit='No legal edge is an official-rule failure, not proof of a false bee; a legal edge is not proof of physical identity'))
        save(out/'summary.json',summary)
        save(out/'receipt.json',dict(prereg=args.prereg,inputs=source_seals,code_sha256=prepare.sha(__file__),
            prediction_common_seal_verified_before_target_access=True,physical_scoring_calls=0,source_guards_posthoc_only=True))
        save(out/'manifest.json',{p.relative_to(out).as_posix():prepare.sha(p) for p in out.rglob('*') if p.is_file()})
        (out/'SEALED.sha256').write_text(prepare.sha(out/'manifest.json')+'\n');out.rename(args.out)
    print('EVALUATE212_SEALED',prepare.sha(args.out/'manifest.json'),flush=True)


def self_test():
    from tools.study197.candidates import window_counts
    events=[dict(frame=10,direction=1)]
    bins=window_counts(events,1000).tolist()
    metric=dict(evidence=dict(predictions=[[1,10],[2,10]],gt=[[5,10]],edges05=[[1,5],[2,5]],edges03=[[1,5],[2,5]]),
        nP=2,nG=1,pred_bins=(2*np.asarray(bins)).tolist(),gt_bins=bins,iou={t:dict(tp=1) for t in ('0.5','0.3')})
    o=outcome(metric,1000)
    assert o['numerator']==30 and o['W']==100. and o['iou']['0.5']['E']==1
    original=dict(event_predictions=[dict(event_ref='a',canonical_id=1,signed_frame=10,feature_key='shared',probabilities={h:.2 for h in HEADS}),
        dict(event_ref='b',canonical_id=2,signed_frame=-11,feature_key='shared',probabilities={h:.8 for h in HEADS})])
    graph=dict(pred_vertices=[1,2],GT_vertices=[5],edges=[[2,5]],nPred=2,nGT=1,TP=1,FP=1,FN=0,
        events=[dict(canonical_event_id=1,signed_frame=10,y=0,legal_GT_neighbors=[]),dict(canonical_event_id=2,signed_frame=-11,y=1,legal_GT_neighbors=[5])])
    b=brier(original,graph);assert b['n']==2 and b['both_classes'] and abs(b['Brier']['T']-.04)<1e-12
    # Same feature cache does not collapse independent event references.
    empty=brier(dict(event_predictions=[]),dict(events=[],pred_vertices=[],GT_vertices=[],edges=[],nPred=0,nGT=0,TP=0,FP=0,FN=0));assert empty['Brier']['T'] is None
    x=dict(numerator=1,denominator=30,nPred=1,nGT=1,iou={t:dict(TP=1,Pred=1,GT=1,FP=0,FN=0,E=0,F1=1.) for t in ('0.5','0.3')})
    p=pool([x,x]);assert p['numerator']==2 and p['denominator']==60 and p['iou']['0.5']['TP']==2
    try:brier(dict(event_predictions=original['event_predictions']*2),graph)
    except ValueError:pass
    else:raise AssertionError('duplicate refs accepted')
    bank={'shared':dict(G7=[0]*7,R6=[0]*6,P6=[0]*6,R_alt6=[0]*6)}
    refs=[dict(r,original_parent_ids=[r['canonical_id']]) for r in original['event_predictions']]
    k=dict(state_id='k',actions=[],events=refs,nPred=2,edits=0,raw_meanCountL1=2.,selected_action_ids=[])
    edited=dict(k,state_id='e',actions=[dict(family='I',canonical_ids=[1,2])],events=[],nPred=0,edits=1,raw_meanCountL1=0.,selected_action_ids=['I:1|2'])
    clip=dict(domain='35',clip='synthetic',states=[k,edited],originals=[],K_state_id='k',C_state_id='e')
    model=dict(heads={h:dict(status='constant',constant_probability=.8) for h in ('G','R','T')})
    predicted=fit.choose_clip(clip,model,bank)
    assert predicted['selected']['C']['state_id']=='e' and predicted['selected']['T']['state_id']=='k'
    metrics={'k':dict(metrics=dict(evidence=dict(edges05=[[1,1]],edges03=[[1,1]]))), 'e':dict(metrics=dict(evidence=dict(edges05=[],edges03=[])))}
    changed=event_changes(predicted,metrics)
    assert len(changed['C']['before'])==2 and all(e['changed'] for e in changed['C']['before']) and changed['C']['after']==[]
    return dict(passed=True,scope='synthetic member Brier/duplicate rejection/raw pooling; actual fit policy guard integration; state-local family deletion including zero net events')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--repo',type=Path)
    for name in ('frozen','truth','late-truth','predictions','joint-assessment','joint-predictions','out'):p.add_argument('--'+name,type=Path)
    p.add_argument('--prereg');args=p.parse_args()
    if args.repo:
        project=args.repo if (args.repo/'tools').is_dir() else args.repo/'courses/dlia/final_project'
        sys.path.insert(0,str(project.resolve()))
    if args.self_test:print(json.dumps(self_test()))
    else:
        if any(getattr(args,k) is None for k in ('repo','frozen','truth','late_truth','predictions','joint_assessment','joint_predictions','out','prereg')):p.error('explicit sealed input roots, repo, prereg, new output required')
        run(args)
