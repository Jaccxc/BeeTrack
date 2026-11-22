"""Independent compact numeric replay for registered observed-phase study 212.

Recomputes phase features from anonymous native dates/inside bits, source-only
weighted logistic heads, guard choices, member Brier scores and all published
§211 state scores. It never rebuilds raw MOT ownership or GT geometry.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import features212


DOMAINS = ('35','33','W','G')
POLICIES = ('K','C','G','R','T','S')
RIDGE = .1
GUARD = 1e-9
SCHEMA = 'phase212.numeric.v1'
MARKER = '[observed-phase-v1]'


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def _features(bank):
    result={}
    for key,sequence in bank.items():
        if not isinstance(key,str) or not key or not isinstance(sequence,dict):
            raise ValueError('opaque feature key and date/inside sequence required')
        raw={'dates':sequence.get('dates'),'inside':sequence.get('inside')}
        digest=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if digest!=key:raise ValueError('date/inside feature key mismatch')
        if not isinstance(raw['dates'],list) or not isinstance(raw['inside'],list) or len(raw['dates'])!=len(raw['inside']):
            raise ValueError('aligned native dates and inside bits required')
        if any(type(v) is not int or v<=0 for v in raw['dates']) or any(type(v) is not bool for v in raw['inside']):
            raise ValueError('native dates must be positive ints and inside bits explicit booleans')
        feature=features212.run_features(raw['dates'],np.asarray(raw['inside'],dtype=bool))
        if not feature['valid']:raise ValueError('published sequence is not a valid original net event')
        result[key]=feature
    return result


def _vector(feature,head):
    g=np.asarray(feature['G'],float);r=np.asarray(feature['R'],float)
    alt=np.asarray(feature['R_alt'],float);p=np.asarray(feature['P'],float)
    if g.shape!=(7,) or any(x.shape!=(6,) for x in (r,alt,p)):
        raise ValueError('fixed G7 and six-wide phase vectors required')
    if not np.array_equal(np.asarray(feature['T'],float),p-r):
        raise ValueError('T phase residual differs from P-R')
    return {'G':g,'R':np.r_[g,r],'T':np.r_[g,r,p-r],'S':np.r_[g,r,alt-r]}[head]


def _weights(rows):
    if not rows:return np.empty(0,dtype=float)
    domains=sorted({r['domain'] for r in rows})
    clips={h:sorted({r['clip'] for r in rows if r['domain']==h}) for h in domains}
    count=Counter((r['domain'],r['clip']) for r in rows)
    w=np.asarray([1/(len(domains)*len(clips[r['domain']])*count[r['domain'],r['clip']]) for r in rows],float)
    w=w/w.sum()
    return w


def _loss(theta,x,y,w):
    beta,bias=theta[:-1],theta[-1]
    z=x@beta+bias
    residual=w*(expit(z)-y)
    return (float(w@(np.logaddexp(0.,z)-y*z)+.5*RIDGE*(beta@beta)),
            np.r_[x.T@residual+RIDGE*beta,residual.sum()])


def _fit_head(rows,features,head):
    d={'G':7,'R':13,'T':19}[head]
    if not rows:
        return {'status':'empty','constant_probability':.5,'mean':None,'scale':None,
                'beta':[0.]*d,'intercept':0.,'n':0,'npos':0,'weights':[]}
    y=np.asarray([int(r['y']) for r in rows],float)
    if not np.isin(y,[0.,1.]).all():raise ValueError('binary source K labels required')
    w=_weights(rows);x=np.stack([_vector(features[r['feature_key']],head) for r in rows])
    n,npos=len(rows),int(y.sum())
    if npos in (0,n):
        p=.5 if n==0 else (npos+1)/(n+2)
        return {'status':'empty' if n==0 else 'single_class_smoothed','constant_probability':p,
                'mean':None,'scale':None,'beta':[0.]*d,'intercept':math.log(p/(1-p)),
                'n':n,'npos':npos,'weights':w.tolist()}
    mean=w@x
    constant=np.all(x==x[0],axis=0)
    mean[constant]=x[0,constant]
    scale=np.sqrt(w@((x-mean)**2))
    zero=scale==0.
    scale[zero]=1.
    z=(x[:,:d]-mean[:d])/scale[:d]
    opt=minimize(lambda th:_loss(th,z,y,w),np.zeros(d+1),jac=True,method='L-BFGS-B',
                 options={'maxiter':500,'gtol':1e-8,'ftol':1e-12})
    if not opt.success or not np.isfinite(opt.x).all():
        raise RuntimeError('independent source logistic did not converge: '+str(opt.message))
    objective,gradient=_loss(opt.x,z,y,w)
    return {'status':'fitted','mean':mean[:d].tolist(),'scale':scale[:d].tolist(),
            'zero_scale_columns':np.flatnonzero(zero[:d]).tolist(),'beta':opt.x[:d].tolist(),
            'intercept':float(opt.x[-1]),'n':n,'npos':npos,'weights':w.tolist(),
            'objective':objective,'gradient':gradient.tolist(),'iterations':int(opt.nit)}


def _fit_fold(rows,features,heldout):
    if heldout not in DOMAINS or any(r['domain']==heldout for r in rows):
        raise ValueError('heldout receiver labels leaked into fold')
    if not {r['domain'] for r in rows}.issubset(set(DOMAINS)-{heldout}):
        raise ValueError('source receiver row outside other-three domains')
    # The projector assigns this rank before replacing source event references
    # with opaque IDs.  Preserve producer lexical order without publishing the
    # private reference strings.
    ordered=sorted(rows,key=lambda r:(r['source_order'] if 'source_order' in r
                                      else (r['domain'],r['clip'],r['event_ref'])))
    heads={h:_fit_head(ordered,features,h) for h in ('G','R','T')}
    heads['S']={'status':'shares_T_fit','refit':False}
    return {'heldout':heldout,'source_domains':[h for h in DOMAINS if h!=heldout],
            'heads':heads,'ordered_rows':ordered}


def _predict(fold,features,key,head):
    model=fold['heads']['T' if head=='S' else head]
    if model['status'] in ('empty','single_class_smoothed'):
        return float(model['constant_probability'])
    x=_vector(features[key],head)
    z=((x-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['beta'])+model['intercept']
    return float(expit(z))


def _bin_countL1(graph,nframes,bins_fn):
    # Validate both sides from signed events; the 30 fixed offsets may yield
    # more than 30 aggregate rows for long clips.
    for name in ('pred','reference_events'):
        rows=graph[name]
        if not isinstance(rows,list) or any(not isinstance(r,list) or len(r)!=2 or not isinstance(r[0],str) or type(r[1]) is not int for r in rows):
            raise ValueError('peer event vertices must be opaque string IDs and signed integer frames')
    pred=bins_fn(graph['pred'],nframes);ref=bins_fn(graph['reference_events'],nframes)
    for key,actual in (('pred_bins',pred),('reference_bins',ref)):
        packed=graph[key]
        if not isinstance(packed,list) or len(packed)!=len(actual) or any(not isinstance(r,list) or len(r)!=2 or any(type(v) is not int for v in r) for r in packed):
            raise ValueError('peer windows must match complete native-clock shape')
        if packed!=actual:raise ValueError('peer signed-event windows differ')
    return sum(abs(a-b) for x,y in zip(pred,ref) for a,b in zip(x,y))


def _public_score(state,score_arm):
    payload={'bins':state['bins'],'events':{'gt':state['gt_events'],'pred':state['pred_events'],
        'edges05':state['true_graphs']['0.5']['edges'],'edges03':state['true_graphs']['0.3']['edges']},
        'declared':state['declared']}
    result=score_arm(payload)
    for threshold,metric in (('0.5','tp05'),('0.3','tp03')):
        # Validate the independently projected edge cardinality against the
        # public scorer result, without copying/rebuilding the 86 native MOTs.
        if int(state['true_graphs'][threshold]['declared_tp'])!=int(result[metric]):
            raise ValueError('linked §211 event graph score differs')
    return result


def _canonical_action_key(state,action_map):
    rows=[]
    for aid in state['selected_action_ids']:
        action=action_map[aid]
        rows.append((action['family'],*map(int,action['endpoint_ids'])))
    return tuple(sorted(rows))


def _guarded_select(states,probs,action_map,head,k_id,count_by_id):
    byid={s['state_id']:s for s in states};k=byid[k_id]
    hvals={s['state_id']:float(s['nPred']-2*sum(probs[s['state_id']])) for s in states}
    eligible=[s for s in states if hvals[s['state_id']]-hvals[k_id]<=GUARD]
    if k not in eligible:raise AssertionError('K must satisfy own risk guard')
    choice=min(eligible,key=lambda s:(-round(float(count_by_id[k_id]-count_by_id[s['state_id']]),10),
                                       s['edits'],_canonical_action_key(s,action_map)))
    return {'state_id':choice['state_id'],'selected_action_ids':choice['selected_action_ids'],
            'edits':choice['edits'],'H':hvals[choice['state_id']],
            'H_K':hvals[k_id],'delta_H':hvals[choice['state_id']]-hvals[k_id],
            'eligible_states':len(eligible)}


def replay(bundle, public211, repo_root):
    if bundle.get('schema')!=SCHEMA or bundle.get('marker')!=MARKER:
        raise ValueError('unexpected phase212 evidence schema/marker')
    if public211.get('schema')!='joint211.numeric.v1' or public211.get('marker')!='[cross-family-peer-v1]':
        raise ValueError('pinned public joint211 package required')
    if bundle.get('receipts',{}).get('joint211_sha256')!=bundle.get('receipts',{}).get('joint211_actual_sha256'):
        raise ValueError('evidence does not pin the loaded public211 bytes')
    score_arm,bins_fn=_public_api(repo_root)
    features=_features(bundle['feature_sequences'])
    clips={(c['domain'],c['clip']):c for c in public211['clips']}
    if len(clips)!=6 or sum(len(c['states']) for c in clips.values())!=86:
        raise ValueError('public §211 six-clip/86-state scope required')
    compact_clips={(c['domain'],c['clip']):c for c in bundle['clips']}
    if set(compact_clips)!=set(clips):raise ValueError('phase/public clip roster mismatch')
    source_rows=bundle['source_rows']
    if len({r['event_ref'] for r in source_rows})!=len(source_rows):raise ValueError('duplicate receiver source event reference')
    if len(bundle.get('members',[]))!=18:raise ValueError('exact original 18 K members required')
    receiver_rows={e['event_ref']:(m,e) for m in bundle['members'] if m['role']=='receiverK' for e in m['events']}
    source_by_ref={r['event_ref']:r for r in source_rows}
    if len(receiver_rows)!=sum(len(m['events']) for m in bundle['members'] if m['role']=='receiverK') or set(source_by_ref)!=set(receiver_rows):
        raise ValueError('source rows must cover all and only receiver K events')
    for event_ref,(member,e) in receiver_rows.items():
        r=source_by_ref[event_ref]
        if r['feature_key']!=e['feature_key'] or int(r['y'])!=int(e['y']) or r['domain']!=member['domain'] or r['clip']!=member['clip']:
            raise ValueError('source row differs from receiver K event receipt')
    folds={h:_fit_fold([r for r in source_rows if r['domain']!=h],features,h) for h in DOMAINS}
    reference_models=bundle.get('models_reference')
    if reference_models is not None:
        for h in DOMAINS:
            for head in ('G','R','T'):
                expected=reference_models[h]['heads'][head]
                actual=folds[h]['heads'][head]
                if expected['status']!=actual['status']:
                    raise ValueError('source model status differs from sealed producer')
                if actual['status']=='fitted':
                    for name in ('mean','scale','beta'):
                        if not np.allclose(expected[name],actual[name],rtol=0,atol=2e-9):
                            raise ValueError('refitted source model '+name+' differs')
                    if not math.isclose(expected['intercept'],actual['intercept'],rel_tol=0,abs_tol=2e-9):
                        raise ValueError('refitted source intercept differs')
                elif not math.isclose(float(expected['constant_probability']),float(actual['constant_probability']),rel_tol=0,abs_tol=1e-14):
                    raise ValueError('source constant probability differs')
    member_event_rows=[];source_y_verified=0
    # Receiver source labels are independently regenerated from the old K true
    # graph; peer labels require the separately packed peer→GT legal graph.
    member_by_ref={}
    for member in bundle['members']:
        h,clip=member['domain'],member['clip']
        public=clips[h,clip]
        k=next(s for s in public['states'] if not s['selected_action_ids'])
        p_event_to_feature=compact_clips[h,clip]['state_event_features'][k['state_id']]
        p_to_y={str(pid):int(any(str(p)==str(pid) for p,g in k['true_graphs']['0.5']['edges']))
                for pid,_ in k['pred_events']}
        if member['role']=='receiverK':
            if set(e['public_event_id'] for e in member['events'])!=set(p_to_y):
                raise ValueError('receiver K event map does not cover old public true graph')
            for row in member['events']:
                if row['public_event_id'] not in p_to_y or row['feature_key']!=p_event_to_feature[row['public_event_id']]:
                    raise ValueError('receiver feature map differs from public211 K graph')
                row_y=p_to_y[row['public_event_id']]
                if int(row['y'])!=row_y:raise ValueError('receiver weak y differs from public211 any-edge graph')
                source_y_verified+=1
        else:
            graph=bundle['peer_truth_graphs'].get(member.get('truth_graph_key',member['member_key']))
            if graph is None:raise ValueError('peer K member requires its own true-GT legal graph')
            vertices={str(p) for p,_ in graph['pred_events']}
            if vertices!={str(e['member_event_id']) for e in member['events']}:
                raise ValueError('peer true graph event inventory differs from K feature rows')
            edge_left={str(p) for p,g in graph['edges']}
            for row in member['events']:
                truth=int(row['member_event_id'] in edge_left)
                if int(row['y'])!=truth:raise ValueError('peer y differs from its anonymous GT legal graph')
        predictions=[];labels=[]
        for row in member['events']:
            if row['feature_key'] not in features:raise ValueError('member event feature missing')
            p={head:_predict(folds[h],features,row['feature_key'],head) for head in ('G','R','T','S')}
            predictions.append(p);labels.append(int(row['y']))
            member_by_ref[row['event_ref']]=dict(domain=h,clip=clip,role=member['role'],
                member=member['member_key'],event_ref=row['event_ref'],y=int(row['y']),probabilities=p)
        member_event_rows.extend(dict(domain=h,clip=clip,member_key=member['member_key'],
            member_index=member['member_index'],role=member['role'],y=int(y),probabilities=q)
            for q,y in zip(predictions,labels))
    clip_member_rows=defaultdict(list); pooled_member_rows=defaultdict(list)
    for member in bundle['members']:
        clip_member_rows[(member['domain'],member['clip'],member['member_key'],member['member_index'],member['role'])]
        pooled_member_rows[(member['domain'],member['member_key'],member['member_index'],member['role'])]
    for row in member_event_rows:
        clip_member_rows[(row['domain'],row['clip'],row['member_key'],row['member_index'],row['role'])].append(row)
        pooled_member_rows[(row['domain'],row['member_key'],row['member_index'],row['role'])].append(row)
    def summarize_member(key,rows):
        result=dict(domain=key[0],member_key=key[-3] if len(key)==5 else key[1],
            member_index=key[-2] if len(key)==5 else key[2],role=key[-1],n=len(rows),
            positive=sum(r['y'] for r in rows),negative=sum(1-r['y'] for r in rows),
            brier={head:(float(np.mean([(r['probabilities'][head]-r['y'])**2 for r in rows])) if rows else None)
                   for head in ('G','R','T','S')})
        if len(key)==5:result['clip']=key[1]
        return result
    clip_member_brier=[summarize_member(k,v) for k,v in sorted(clip_member_rows.items())]
    pooled_member_brier=[summarize_member(k,v) for k,v in sorted(pooled_member_rows.items())]
    domain_member_means={}
    for domain in DOMAINS:
        rows=[r for r in pooled_member_brier if r['domain']==domain]
        if len(rows)!=3:raise ValueError('three original anonymous members per domain required')
        domain_member_means[domain]={head:(float(np.mean([r['brier'][head] for r in rows]))
            if all(r['brier'][head] is not None for r in rows) else None) for head in ('G','R','T','S')}
    state_results={}; selected_pooled={h:{} for h in DOMAINS}
    for key,public in clips.items():
        compact=compact_clips[key]; action_map={a['action_id']:a for a in public['actions']}
        public_by_id={s['state_id']:s for s in public['states']}
        if set(compact['state_event_features'])!=set(public_by_id):raise ValueError('86-state feature map incomplete')
        count_by_id={};state_rows=[];score_by_id={};prob_by_state={}
        for sid,state in public_by_id.items():
            if sid not in compact['state_event_features']:raise ValueError('state feature map missing')
            event_map=compact['state_event_features'][sid]
            if set(event_map)!=set(str(pid) for pid,frame in state['pred_events']):
                raise ValueError('state map does not align to complete old public event roster')
            probs={head:[] for head in ('G','R','T','S')}
            for pid,signed in state['pred_events']:
                fkey=event_map[str(pid)]
                feature=features[fkey]
                if int(feature['original_event']['direction']*feature['original_event']['date'])!=int(signed):
                    raise ValueError('event date/direction differs from anonymous raw sequence')
                for head in probs:probs[head].append(_predict(folds[key[0]],features,fkey,head))
            prob_by_state[sid]=probs
            state_rows.append({'state_id':sid,'selected_action_ids':state['selected_action_ids'],
                'edits':len(state['selected_action_ids']),'nPred':len(state['pred_events'])})
            l1=[]
            for peer in state['peer_graphs']:
                l1.append(_bin_countL1(peer,public['nframes'],bins_fn))
            count_by_id[sid]=float(sum(l1)/len(l1))
            if count_by_id[sid]!=float(state['peer_summary']['meanCountL1']):
                raise ValueError('recomputed public211 C-window utility differs')
            score=_public_score(state,score_arm)
            score_by_id[sid]=score
        k_id=next(sid for sid,s in public_by_id.items() if not s['selected_action_ids'])
        k_count=count_by_id[k_id]
        c_id=min(public['states'],key=lambda s:(-round(k_count-count_by_id[s['state_id']],10),
                  len(s['selected_action_ids']),_canonical_action_key(s,action_map)))['state_id']
        if c_id!=public['frozen_choices']['C']['state_id']:
            raise ValueError('old public211 count-C selection does not reproduce')
        choices={'K':{'state_id':k_id},'C':{'state_id':c_id}}
        for head in ('G','R','T','S'):
            choices[head]=_guarded_select(state_rows,
                {sid:prob_by_state[sid][head] for sid in prob_by_state},action_map,head,k_id,count_by_id)
        selected={a:r['state_id'] for a,r in choices.items()}
        state_results['/'.join(key)]={'selection':choices,'state_scores':score_by_id,
            'state_probabilities':prob_by_state,'count_meanL1':count_by_id,
            'state_event_features':compact['state_event_features']}
        for policy,sid in selected.items():
            selected_pooled[key[0]].setdefault(policy,[]).append(score_by_id[sid])
        # Fold-wise choice receipts copied by action state identity only.
        ref=compact.get('selection_reference',{})
        if ref:
            for head in ('G','R','T','S'):
                if ref.get(head) and ref[head]['state_id']!=selected[head]:
                    raise ValueError('replayed phase head choice differs from sealed source prediction')
    pooled={}
    for h,policies in selected_pooled.items():
        pooled[h]={}
        for policy,rows in policies.items():
            if len(rows)!=len([1 for key in clips if key[0]==h]):raise ValueError('pooled clip count differs')
            den=sum(r['window_gt_repeated'] for r in rows);num=sum(r['window_error_sum'] for r in rows)
            gt=sum(r['gt'] for r in rows);pred=sum(r['pred'] for r in rows)
            tp05=sum(r['tp05'] for r in rows);tp03=sum(r['tp03'] for r in rows)
            pooled[h][policy]={'window_numerator':num,'window_denominator':den,
                'W_percent':100*num/den,'nG':gt,'nP':pred,'tp05':tp05,'fp05':pred-tp05,'fn05':gt-tp05,
                'F1_05':2*tp05/(gt+pred) if gt+pred else 0.,
                'tp03':tp03,'fp03':pred-tp03,'fn03':gt-tp03,
                'F1_03':2*tp03/(gt+pred) if gt+pred else 0.}
    def gain(h):return 100*(pooled[h]['K']['window_numerator']-pooled[h]['T']['window_numerator'])/pooled[h]['K']['window_denominator']
    def exact_W(row):return Fraction(100*row['window_numerator'],row['window_denominator'])
    def exact_F1(row):return Fraction(2*row['tp05'],row['nG']+row['nP']) if row['nG']+row['nP'] else Fraction(1)
    gate_checks={}
    for h in ('35','33'):
        gate_checks[h+'_W_gain_at_least_1pp']=exact_W(pooled[h]['K'])-exact_W(pooled[h]['T'])>=1
        gate_checks[h+'_F1_loss_at_most_0.01']=exact_F1(pooled[h]['T'])>=exact_F1(pooled[h]['K'])-Fraction(1,100)
    for h in ('W','G'):
        gate_checks[h+'_W_not_worse']=exact_W(pooled[h]['T'])<=exact_W(pooled[h]['K'])
    gates={'passed':all(gate_checks.values()),'checks':gate_checks}
    reserve=[m for m in clip_member_brier if m['domain'] in ('35','33') and m['role']=='peerK']
    reasons=[];reserve_checks=[]
    if len(reserve)!=4 or len({(m['domain'],m['member_index']) for m in reserve})!=4:
        reasons.append('four distinct preregistered head reserve members not present')
    for m in reserve:
        both=m['positive']>0 and m['negative']>0
        if not both:reasons.append(m['domain']+'/slot'+str(m['member_index'])+': insufficient positive/negative class denominator')
        for head in ('G','R','S'):
            ok=m['brier']['T'] is not None and m['brier'][head] is not None and m['brier']['T']<m['brier'][head]
            reserve_checks.append(dict(domain=m['domain'],clip=m['clip'],member_index=m['member_index'],
                comparator=head,strictly_better=ok,both_classes=both))
    choice_checks=[]
    for clip_name,data in state_results.items():
        domain=clip_name.split('/',1)[0]
        if domain not in ('35','33'):continue
        for head in ('R','S'):
            t_id=data['selection']['T']['state_id'];other_id=data['selection'][head]['state_id']
            tw=data['state_scores'][t_id];ow=data['state_scores'][other_id]
            better=Fraction(ow['window_error_sum'],ow['window_gt_repeated'])>Fraction(tw['window_error_sum'],tw['window_gt_repeated'])
            choice_checks.append(dict(domain=domain,comparator=head,different_choice=t_id!=other_id,
                different_and_strict_W_gain=t_id!=other_id and better))
    if not any(c['different_choice'] for c in choice_checks):reasons.append('no head T choice differs from R or S')
    elif not any(c['different_and_strict_W_gain'] for c in choice_checks):reasons.append('no differing head choice has strictly lower W')
    if not all(c['strictly_better'] for c in reserve_checks):reasons.append('T is not strictly better than every G/R/S reserve Brier comparator')
    insufficient=len(reserve)!=4 or any(m['positive']==0 or m['negative']==0 for m in reserve)
    phase_attribution={'status':'inconclusive' if insufficient else ('passed' if not reasons else 'not_passed'),
        'passed':not reasons,'reasons':reasons,'reserve_checks':reserve_checks,'choice_checks':choice_checks,
        'scope':'four repeated old head reserve members; not four independent experiments'}
    return {'schema':'phase212.replay-result.v1','marker':bundle['marker'],'prereg':bundle['prereg'],
        'receipts':bundle['receipts'],'models':{h:{name:folds[h]['heads'][name] for name in ('G','R','T','S')} for h in DOMAINS},
        'members':member_by_ref,'member_brier_by_clip':clip_member_brier,
        'pooled_original_member_brier':pooled_member_brier,'domain_member_brier_means':domain_member_means,
        'source_receiver_y_rechecked':source_y_verified,'clips':state_results,'pooled':pooled,
        'gates':gates,'investment_pass':gates['passed'],'phase_attribution':phase_attribution,
        'forensics':bundle.get('forensics',[]),
        'scope':'Recomputes anonymous native-date phase features, receiver-only weighted fits, guard choices, peer-K Brier values and official old §211 graph/window scores. Does not rebuild raw owner geometry or rerun 86 MOTs.'}


def _public_api(repo_root):
    repo_root=Path(repo_root).resolve()
    if not (repo_root/'beetrack'/'p0.py').is_file():raise ValueError('--repo must point to final_project')
    sys.path.insert(0,str(repo_root/'tools'/'study209'))
    from replay209 import score_arm
    from replay_release209 import bins
    return score_arm,bins


def _load_json(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf8') as f:return json.load(f)
    return json.loads(path.read_text(encoding='utf8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path,nargs='?')
    parser.add_argument('--public211',type=Path)
    parser.add_argument('--repo',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),sort_keys=True));return
    if not args.bundle or not args.public211 or not args.repo or not args.out:
        parser.error('bundle, --public211, --repo, and --out are required')
    bundle=_load_json(args.bundle);public=_load_json(args.public211)
    bundle.setdefault('receipts',{})['joint211_actual_sha256']=_sha(args.public211)
    result=replay(bundle,public,args.repo)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')


def self_test():
    # Exercises phase extraction, fold refit/fallback, count guard, and score
    # math without opening any stage, native MOT, GT or public workspace data.
    a=features212.run_features(list(range(1,19)),[False]*8+[True]*10)
    b=features212.run_features(list(range(1,19)),[False,True]*9)
    assert a['valid'] and b['valid'] and a['P']!=b['P']
    bank={'a':a,'b':b}; rows=[]
    for domain in ('33','W','G'):
        for i,key in enumerate(('a','b','a','b')):
            rows.append({'domain':domain,'clip':domain+'c','event_ref':domain+str(i),'feature_key':key,'y':i%2})
    model=_fit_fold(rows,bank,'35')
    assert all(model['heads'][head]['status']=='fitted' for head in ('G','R','T'))
    assert 0.<=_predict(model,bank,'a','G')<=1.
    assert 0.<=_predict(model,bank,'a','S')<=1.
    assert _fit_head([],bank,'G')['constant_probability']==.5
    assert math.isclose(_fit_head([dict(rows[0],y=1)],bank,'G')['constant_probability'],2/3)
    states=[{'state_id':'K','selected_action_ids':[],'edits':0,'nPred':2},
            {'state_id':'s1','selected_action_ids':['a1'],'edits':1,'nPred':2}]
    actions={'a1':{'family':'C','endpoint_ids':[1,2]}}
    selected=_guarded_select(states,{'K':[.5,.5],'s1':[.6,.6]},actions,'G','K',{'K':10,'s1':5})
    assert selected['state_id']=='s1' and selected['delta_H']<=GUARD
    assert math.isclose(float(np.mean([(0.2-0)**2,(.8-1)**2])),.04)
    return {'status':'PASS','scope':'synthetic-only','checks':['date feature recompute','four-fold source fit core','empty/single-class fallback','risk guard/count tie','Brier arithmetic']}


if __name__=='__main__':main()
