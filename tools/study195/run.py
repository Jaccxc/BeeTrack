"""Whole-member route enumeration and exact physical-member scoring.

Pure helpers accept explicit GT arrays only at final evaluation. No field IO,
training or GT loader is enabled before the separately registered orchestration.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
from collections import Counter
import configparser
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True
import numpy as np

ARMS = ('t40', 't50')
STATES = tuple(product((0, 1), repeat=3))
DOMAINS = ('35', '33', 'W', 'G')


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [_plain(v) for v in value]
    return value


def f1(tp, pred, gt):
    return 2 * int(tp) / max(int(pred) + int(gt), 1)


def exact_f1(candidate, threshold='0.5'):
    return sum((Fraction(2*m['tp'],max(m['pred']+m['gt'],1)) for m in candidate['iou'][threshold]['members']),Fraction())/3


def score_member_native(gt, rows, polygon, nf):
    """Official full native member MOT, no filtered/reassigned event rows."""
    from beetrack.evaluate import by_track, frame_match, match_events, zone_tracks
    from tools.study193.source193 import score_native
    G, R = np.asarray(gt,float), np.asarray(rows,float)
    score = score_native(G, R, polygon, nf)
    official = match_events(G,R,polygon)
    strict = score['iou']['0.5']
    assert (official['gt'],official['pred'],official['tp']) == (score['nG'],score['nP'],strict['tp'])
    per = {}
    for threshold in (.5,.3):
        key=str(threshold); value=score['iou'][key]
        if threshold==.5:
            tp_id=official['tp_id']
        else:
            truth,pred=by_track(G),by_track(R)
            target=zone_tracks(truth,polygon); found=zone_tracks(pred,polygon)
            owners=frame_match(truth,pred,thr=threshold)
            same=set()
            for direction,start,end,cross in found.values():
                if not direction: continue
                matched=owners[start:end][owners[start:end]>=0]
                if len(matched) and matched[0]==matched[-1] and target[matched[0]][0]==direction:
                    same.add(int(matched[0]))
            tp_id=len(same)
        per[key]=dict(tp=int(value['tp']),pred=int(score['nP']),gt=int(score['nG']),tp_id=int(tp_id),
                      f1=float(value['f1']),by_direction=value['by_direction'],matching=value['matching'])
    return dict(iou=per,pred_bins=score['integer_bins']['pred'],gt_bins=score['integer_bins']['gt'],
                nG=score['nG'],nP=score['nP'],evidence=score['evidence'],
                W=score['w30_percent'],window=score['window'])


def score_video(group, gt, scorer=score_member_native):
    """Only six complete member/arm scores; the eight routes reuse them exactly."""
    if group['domain'] not in DOMAINS or len(group['members'])!=3 or len({m['member'] for m in group['members']})!=3:
        raise ValueError('three ordered distinct members in a registered domain required')
    scored=[];truth=None;ng=None
    for member in group['members']:
        if set(member['arms'])!=set(ARMS): raise ValueError('fixed t40/t50 arms required')
        arms={}
        for arm in ARMS:
            data=member['arms'][arm]
            result=scorer(gt,data['rows'],group['polygon'],group['nf'])
            bins=np.asarray(result['pred_bins'],dtype=np.int64)
            if not np.array_equal(bins,np.asarray(data['windows'])):
                raise ValueError('native full MOT events differ from frozen directional windows')
            current=np.asarray(result['gt_bins'],dtype=np.int64)
            if truth is None:truth=current;ng=result['nG']
            if not np.array_equal(truth,current) or ng!=result['nG']:
                raise ValueError('member arms do not share the exact GT clock')
            arms[arm]=result
        scored.append(dict(member=str(member['member']),tag=member['tag'],arms=arms))
    return dict(unit=group['unit'],domain=group['domain'],video=group['video'],group=group['group'],
                nf=group['nf'],members=scored,gt_bins=truth.tolist(),nG=int(ng))


def route_state(video, state):
    if tuple(state) not in STATES: raise ValueError('state must be three binary choices')
    chosen=[member['arms'][ARMS[bit]] for member,bit in zip(video['members'],state)]
    predicted=sum(np.asarray(arm['pred_bins'],dtype=np.int64) for arm in chosen)
    truth=np.asarray(video['gt_bins'],dtype=np.int64)
    numerator=int(np.abs(predicted-3*truth).sum())
    ng=int(video['nG'])
    return dict(state=list(state),arms=[ARMS[b] for b in state],numerator=numerator,denominator=90*ng,
                E3=100*numerator/(90*ng) if ng else None,
                sum_pred_bins=predicted.tolist(),gt_bins=truth.tolist(),member_metrics=[c['iou'] for c in chosen])


def combine(videos, states):
    """Pool clips first; then compute each physical member F1, then member mean."""
    if len(videos)!=len(states) or not videos: raise ValueError('one route per actual clip required')
    domain=videos[0]['domain'];group=videos[0]['group']
    if any(v['domain']!=domain or v['group']!=group for v in videos):raise ValueError('same domain/seed group required')
    order=[m['member'] for m in videos[0]['members']]
    if any([m['member'] for m in v['members']]!=order for v in videos):raise ValueError('member identities/order must align across clips')
    routes=[route_state(v,s) for v,s in zip(videos,states)]
    numerator=sum(r['numerator'] for r in routes);denominator=sum(r['denominator'] for r in routes)
    iou={}
    for threshold in ('0.5','0.3'):
        members=[]
        for i,identity in enumerate(order):
            pooled={k:sum(r['member_metrics'][i][threshold][k] for r in routes) for k in ('tp','pred','gt','tp_id')}
            pooled.update(member=identity,f1=f1(pooled['tp'],pooled['pred'],pooled['gt']))
            members.append(pooled)
        iou[threshold]=dict(members=members,mean_member_f1=float(np.mean([m['f1'] for m in members])))
    return dict(domain=domain,group=group,videos=[v['video'] for v in videos],states=[list(s) for s in states],
                numerator=numerator,denominator=denominator,E3=100*numerator/denominator if denominator else None,
                iou=iou,clip_routes=routes)


def enumerate_group(videos):
    expected=2 if videos[0]['domain'] in ('W','G') else 1
    if len(videos)!=expected or len({v['video'] for v in videos})!=expected:
        raise ValueError('head has one clip; W/G group must pool exactly two different clips')
    candidates=[combine(videos,states) for states in product(STATES,repeat=len(videos))]
    assert len(candidates)==(64 if expected==2 else 8)
    optimum=min(c['numerator'] for c in candidates)
    ties=[c for c in candidates if c['numerator']==optimum]
    frontier=[]
    for i,a in enumerate(candidates):
        ea,fa=a['numerator'],exact_f1(a)
        if not any(b['numerator']<=ea and exact_f1(b)>=fa and
                   (b['numerator']<ea or exact_f1(b)>fa) for j,b in enumerate(candidates) if j!=i):
            frontier.append(i)
    return dict(candidates=candidates,oracle_minimum_numerator=optimum,oracle_states=[t['states'] for t in ties],
                oracle_minimum_E3=ties[0]['E3'],
                oracle_tie_F1_ranges={t:[min(c['iou'][t]['mean_member_f1'] for c in ties),max(c['iou'][t]['mean_member_f1'] for c in ties)] for t in ('0.5','0.3')},
                official05_frontier_indices=frontier,
                uniform_t40=combine(videos,[(0,0,0)]*len(videos)),
                uniform_t50=combine(videos,[(1,1,1)]*len(videos)))


def guarded_oracle(groups, family):
    """Exact across-group single configurations under the FAMILY mean F1 guard."""
    target=sum((exact_f1(f) for f in family),Fraction())-Fraction(len(family),200)
    frontier=[(Fraction(),Fraction(),())]
    for group in groups:
        expanded=[]
        for cost,quality,chosen in frontier:
            for i in group['official05_frontier_indices']:
                c=group['candidates'][i]
                expanded.append((cost+Fraction(100*c['numerator'],c['denominator']),quality+exact_f1(c),chosen+(i,)))
        expanded.sort(key=lambda v:(v[0],-v[1],v[2]))
        frontier=[];best=None
        for item in expanded:
            if best is None or item[1]>best:
                frontier.append(item);best=item[1]
    valid=[r for r in frontier if r[1]>=target]
    if not valid:return dict(feasible=False,mean_f1_guard=float(target/len(groups)))
    best=min(valid,key=lambda v:(v[0],-v[1],v[2]))
    return dict(feasible=True,mean_E3=float(best[0]/len(groups)),mean_F1=float(best[1]/len(groups)),
                mean_f1_guard=float(target/len(groups)),group_candidate_indices=list(best[2]),
                configurations=[groups[k]['candidates'][i]['states'] for k,i in enumerate(best[2])],
                E3_fraction=[best[0].numerator,best[0].denominator*len(groups)],
                F1_fraction=[best[1].numerator,best[1].denominator*len(groups)])


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path,value):
    Path(path).write_text(json.dumps(_plain(value),sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _seal(root):
    files={p.relative_to(root).as_posix():_sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    _save(root/'manifest.json',files);(root/'SEALED.sha256').write_text(_sha(root/'manifest.json')+'\n')
    return _sha(root/'manifest.json')


def _verify(root):
    if (root/'SEALED.sha256').read_text().strip()!=_sha(root/'manifest.json'):raise ValueError('seal mismatch')
    for name,digest in json.loads((root/'manifest.json').read_text()).items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or _sha(p)!=digest:raise ValueError('artifact digest mismatch')
    return _sha(root/'manifest.json')


def _registration(args):
    if not args.prereg or not re.fullmatch('[0-9a-f]{40}',args.prereg):raise ValueError('full preregistration commit required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'refs/remotes/origin/main'],cwd=args.repo,capture_output=True,check=True)
    text=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,capture_output=True,check=True).stdout.decode('utf-8')
    if '[whole-member-router-v1]' not in text:raise ValueError('registered marker missing')


def _load_groups(args):
    sys.path.insert(0,str(args.repo.resolve()))
    import freeze
    groups=freeze.load_groups(args.root/'frozen')
    expected={'35':4,'33':4,'W':6,'G':6}
    if Counter(g['domain'] for g in groups)!=expected:raise ValueError('exact20 registered actual-video groups required')
    for group in groups:
        group['members'].sort(key=lambda m:m['seed'])
        seeds=[m['seed'] for m in group['members']]
        start=(group['group']-1)*3
        if seeds!=list(range(start,start+3)):raise ValueError('fixed ascending seed order required')
    groups.sort(key=lambda g:(DOMAINS.index(g['domain']),g['group'],g['video']))
    return groups


def _gt_provider(repo,groups,domains,role):
    """Source calls name exactly three domains; evaluation names four after seal."""
    if role=='source' and len(domains)!=3 or role=='evaluation' and set(domains)!=set(DOMAINS):
        raise ValueError('explicit source/evaluation GT boundary required')
    allowed={'35':{'BEE24-35-head'},'33':{'BEE24-33-head'},'W':{'BEE24-10','BEE24-15'},'G':{'BEE24-24','BEE24-28'}}
    arrays,proofs={},{}
    for group in groups:
        if group['domain'] not in domains or group['video'] in arrays:continue
        if group['video'] not in allowed[group['domain']]:raise ValueError('GT clip outside fixed allowlist')
        path=repo/'runs/gt'/group['split']/group['video']
        ini=configparser.ConfigParser();ini.read(path/'seqinfo.ini',encoding='utf-8')
        if ini.getint('Sequence','frameRate')!=25 or ini.getint('Sequence','seqLength')!=group['nf']:raise ValueError('prepared GT clock mismatch')
        gp=path/'gt/gt.txt'
        arrays[group['video']]=np.loadtxt(gp,delimiter=',',usecols=range(6),ndmin=2) if gp.read_bytes().strip() else np.empty((0,6))
        proofs[group['video']]=dict(domain=group['domain'],role=role,path=gp.relative_to(repo).as_posix(),sha256=_sha(gp),seqinfo_sha256=_sha(path/'seqinfo.ini'))
    return arrays,proofs


def _counts(group):
    return np.asarray([[m['arms'][arm]['windows'] for arm in ARMS] for m in group['members']],dtype=np.int64)


def train_predict(args):
    _registration(args);groups=_load_groups(args)
    import freeze,model
    from tools.study193.source193 import event_descriptor
    dest=args.root/'predictions'
    if dest.exists():raise FileExistsError(dest)
    receipt=freeze.verify(args.root/'frozen')
    code={Path(p).name:_sha(p) for p in (__file__,model.__file__,freeze.__file__)}
    with TemporaryDirectory(dir=args.root,prefix='.routes-') as tmp:
        work=Path(tmp)/'payload';work.mkdir()
        for heldout in DOMAINS:
            allowed=tuple(d for d in DOMAINS if d!=heldout)
            arrays,proofs=_gt_provider(args.repo,groups,allowed,'source')
            records=[]
            for group in groups:
                if group['domain'] not in allowed:continue
                truth=event_descriptor(arrays[group['video']],group['polygon'],group['nf'])
                records.append(dict(hive=group['domain'],group_id=group['group'],clip=group['video'],counts=_counts(group),gw=truth['bins']))
            fitted=model.fit(records,heldout=heldout)
            choices=[]
            for group in groups:
                if group['domain']!=heldout:continue
                prediction=model.predict(_counts(group),fitted)
                for method in ('M','D','J'):
                    if tuple(prediction[method]['bits']) not in STATES:raise ValueError('invalid deployment state')
                choices.append(dict(unit=group['unit'],video=group['video'],group=group['group'],
                                    member_order=[m['member'] for m in group['members']],prediction=prediction))
            _save(work/(heldout+'.json'),dict(heldout=heldout,source_domains=allowed,source_GT_receipts=proofs,
                                            model=fitted,choices=choices,input_receipt=receipt,code_sha256=code))
            print('FOLD_PREDICTED '+heldout,flush=True)
        _save(work/'receipt.json',dict(schema='whole-router-predictions-v1',prereg=args.prereg,input_receipt=receipt,
                                      folds=list(DOMAINS),code_sha256=code,gt_boundary='source fold sees only other3domains; final scoring is separate after all4sealed'))
        digest=_seal(work);work.rename(dest)
    print('ALL4_PREDICTIONS_SEALED '+str(dest)+' '+digest,flush=True)


def evaluate(args):
    _registration(args);groups=_load_groups(args)
    import freeze
    predroot=args.root/'predictions';prediction_sha=_verify(predroot)
    proof=json.loads((predroot/'receipt.json').read_text())
    if proof['prereg']!=args.prereg or proof['input_receipt']!=freeze.verify(args.root/'frozen'):
        raise ValueError('prediction/input registration mismatch')
    decisions={}
    for h in DOMAINS:
        fold=json.loads((predroot/(h+'.json')).read_text())
        if fold['heldout']!=h or set(fold['source_domains'])!=set(DOMAINS)-{h}:
            raise ValueError('source label boundary mismatch')
        for c in fold['choices']:decisions[c['unit']]=c
    if set(decisions)!={g['unit'] for g in groups}:raise ValueError('all20 target routes required before GT scoring')
    # No evaluation GT is opened until every fold, input and choice seal passes.
    arrays,gtproofs=_gt_provider(args.repo,groups,DOMAINS,'evaluation')
    dest=args.root/'score'
    if dest.exists():raise FileExistsError(dest)
    with TemporaryDirectory(dir=args.root,prefix='.scores-') as tmp:
        work=Path(tmp)/'payload';work.mkdir();(work/'physical').mkdir()
        scored=[];physical=[]
        for index,group in enumerate(groups):
            physical_group=dict(group,members=[])
            for m,member in enumerate(group['members']):
                copy=dict(member,arms={})
                for arm in ARMS:
                    path=work/'physical'/f'g{index:02d}_m{m}_{arm}.txt'
                    record=freeze.write_mot_exact(member['arms'][arm]['rows'],path)
                    rows=np.loadtxt(path,delimiter=',',usecols=range(7),ndmin=2) if path.stat().st_size else np.empty((0,7))
                    copy['arms'][arm]=dict(member['arms'][arm],rows=rows)
                    physical.append(dict(group=group['unit'],member=member['member'],arm=arm,file=path.relative_to(work).as_posix(),**record))
                physical_group['members'].append(copy)
            scored.append(score_video(physical_group,arrays[group['video']]))
            print('SCORED '+group['unit'],flush=True)
        summaries={};method_names=('L','H','F','M','D','J','U','O')
        for domain in DOMAINS:
            unit_groups=[];methods={k:[] for k in method_names}
            for groupid in sorted({v['group'] for v in scored if v['domain']==domain}):
                videos=[v for v in scored if v['domain']==domain and v['group']==groupid]
                enumerated=enumerate_group(videos);unit_groups.append(enumerated)
                method_states={'L':[(0,0,0)]*len(videos),'H':[(1,1,1)]*len(videos),'F':[(1,1,1) if domain in ('35','33') else (0,0,0)]*len(videos)}
                for name in ('M','D','J'):
                    method_states[name]=[decisions[v['unit']]['prediction'][name]['bits'] for v in videos]
                for name,states in method_states.items():methods[name].append(combine(videos,states))
                uniform=[combine(videos,states) for states in product(((0,0,0),(1,1,1)),repeat=len(videos))]
                for name,pool in (('U',uniform),('O',enumerated['candidates'])):
                    minimum=min(c['numerator'] for c in pool)
                    ties=[c for c in pool if c['numerator']==minimum]
                    chosen=min(ties,key=lambda c:c['states'])
                    chosen=dict(chosen,tie_count=len(ties),tie_F1_ranges={t:[float(min(exact_f1(c,t) for c in ties)),float(max(exact_f1(c,t) for c in ties))] for t in ('0.5','0.3')})
                    methods[name].append(chosen)
            mean={name:dict(E3=float(np.mean([c['E3'] for c in values])),
                            F1_05=float(sum((exact_f1(c) for c in values),Fraction())/len(values)),
                            F1_03=float(sum((exact_f1(c,'0.3') for c in values),Fraction())/len(values))) for name,values in methods.items()}
            summaries[domain]=dict(groups=unit_groups,methods=methods,means=mean,
                                   matched_differences={name:dict(E3=mean[name]['E3']-mean['F']['E3'],F1_05=mean[name]['F1_05']-mean['F']['F1_05']) for name in method_names},
                                   guarded_oracle=guarded_oracle(unit_groups,methods['F']))
        differences={h:summaries[h]['matched_differences']['J'] for h in DOMAINS}
        investment=(differences['35']['E3'] < -1.4 and differences['33']['E3'] < -1.3 and differences['W']['E3'] <= 1.4
                    and all(differences[h]['F1_05'] >= -.005-1e-15 for h in ('35','33','W')))
        joint=all(summaries[h]['means']['J']['E3']<summaries[h]['means']['D']['E3'] for h in ('35','33'))
        interpretation={}
        for h,required in (('35',1.4),('33',1.3)):
            guarded=summaries[h]['guarded_oracle'];family=summaries[h]['means']['F']['E3']
            interpretation[h]=dict(window_gain_line=required,guarded_space_can_pass=guarded['feasible'] and guarded['mean_E3']<family-required,
                                   interpretation='space_vs_learning_gap' if guarded['feasible'] and guarded['mean_E3']<family-required else 'whole_member_binary_space_below_line')
        _save(work/'native_scores.json',scored)
        _save(work/'physical_receipts.json',physical)
        _save(work/'summary.json',dict(schema='whole-router-score-v1',prereg=args.prereg,prediction_manifest_sha256=prediction_sha,
                                      GT_entry=gtproofs,hives=summaries,investment_pass=investment,joint_better_than_D_both_heads=joint,
                                      joint_gain_condition=joint,head_identification=interpretation,
                                      gt_reuse='known labels may earlier appear only as other-fold SOURCE; this final evaluation started after ALL4 prediction seal'))
        digest=_seal(work);work.rename(dest)
    print('SCORES_SEALED '+str(dest)+' '+digest,flush=True)


def joint_feasibility(enumerated, window_limit, minimum_f1, threshold='0.5', strict_window=True):
    """Known-window/F1 feasibility only; never a deployed route-selection rule."""
    accepted=[i for i,c in enumerate(enumerated['candidates']) if c['E3'] is not None and
              (c['E3']<window_limit if strict_window else c['E3']<=window_limit) and
              c['iou'][threshold]['mean_member_f1']>=minimum_f1]
    return dict(window_limit=window_limit,minimum_mean_member_f1=minimum_f1,threshold=threshold,
                strict_window=strict_window,feasible=bool(accepted),candidate_indices=accepted)


def self_test():
    # Exact small scorer fixture: keys are encoded in the mock native row.
    def scorer(gt,rows,polygon,nf):
        pred=int(rows[0,0]);tp=min(pred,int(gt[0]));ng=int(gt[0])
        metric=dict(tp=tp,pred=pred,gt=ng,tp_id=tp,f1=f1(tp,pred,ng),by_direction={},matching=[])
        return dict(pred_bins=[[pred,0]]*30,gt_bins=[[ng,0]]*30,nG=ng,nP=pred,iou={'0.5':metric,'0.3':metric})
    def group(video='BEE24-10',ng=1,counts=(0,1,2)):
        members=[]
        for i,n in enumerate(counts):
            members.append(dict(member=str(i),tag='mock'+str(i),arms={a:dict(rows=np.array([[n+b]]),windows=[[n+b,0]]*30)
                                                                          for b,a in enumerate(ARMS)}))
        return dict(unit=video,domain='W',video=video,group=1,nf=75,polygon=[],members=members)
    first=score_video(group(),np.array([1]),scorer)
    second=score_video(group('BEE24-15',counts=(2,1,0)),np.array([1]),scorer)
    e=enumerate_group([first,second])
    assert len(e['candidates'])==64 and e['uniform_t40']['numerator']==0
    c=combine([first,second],[(0,0,0),(0,0,0)])
    assert c['denominator']==180 and c['E3']==0
    assert [m['tp'] for m in c['iou']['0.5']['members']]==[1,2,1]
    assert [m['pred'] for m in c['iou']['0.5']['members']]==[2,2,2]
    assert abs(c['iou']['0.5']['mean_member_f1']-2/3)<1e-12
    assert joint_feasibility(e,1,0)['feasible']
    head=dict(first,domain='35',video='BEE24-35-head')
    assert len(enumerate_group([head])['candidates'])==8
    # The three selected MOTs remain whole; state mapping never concatenates
    # observations from opposite thresholds or introduces synthetic event IDs.
    for state in STATES:
        r=route_state(first,state)
        assert r['arms']==[ARMS[i] for i in state]
    constrained=guarded_oracle([e],[e['uniform_t40']])
    assert constrained['feasible'] and constrained['mean_E3']==0
    if (Path.cwd()/'beetrack').is_dir():
        sys.path.insert(0,str(Path.cwd()))
        gt=np.asarray([[1,7,11,4,2,2],[2,7,9,4,2,2],[3,7,7,4,2,2]],float)
        rows=np.column_stack((gt,np.full(3,.8)))
        native=score_member_native(gt,rows,[[0,0],[10,0],[10,10],[0,10]],75)
        assert all(native['iou'][t]['tp']==native['iou'][t]['tp_id']==1 for t in ('0.5','0.3'))
        assert np.asarray(native['gt_bins']).sum()==30
    print('Synthetic whole-member8/64 states, integer E3, cross-clip member F1, ties/frontier/feasibility passed')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--repo',type=Path);ap.add_argument('--root',type=Path);ap.add_argument('--prereg')
    ap.add_argument('--stage',choices=('train-predict','score'))
    args=ap.parse_args()
    if args.self_test:self_test()
    else:
        if not args.repo or not args.root or not args.prereg or not args.stage:ap.error('require --repo --root --prereg --stage')
        args.repo=args.repo.resolve();args.root=args.root.resolve()
        if args.stage=='train-predict':train_predict(args)
        else:evaluate(args)
