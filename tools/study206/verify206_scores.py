"""Independent whole-MOT and every-prefix replay for 206; no import-time IO."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
import verify206 as core
import verify206_field as field


def object_sha(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pool(videos,threshold):
    numerator=denominator=0;members={};single={}
    for video in videos:
        mm=video['members'];gt=np.asarray(mm[0]['metrics']['gt_bins'],np.int64)
        assert len(mm)==3 and all(np.array_equal(m['metrics']['gt_bins'],gt) for m in mm)
        pred=sum(np.asarray(m['metrics']['pred_bins'],np.int64) for m in mm);numerator+=int(abs(pred-3*gt).sum());denominator+=3*int(gt.sum())
        for m in mm:
            member=str(m['member']);metric=m['metrics'];counts=members.setdefault(member,[0,0,0]);flow=single.setdefault(member,[0,0])
            for j,k in enumerate(('tp','pred','gt')):counts[j]+=metric['iou'][threshold][k]
            flow[0]+=int(abs(np.asarray(metric['pred_bins'])-gt).sum());flow[1]+=int(gt.sum())
    assert len(members)==3 and denominator>0
    f1=sum((Fraction(2*t,p+g) if p+g else Fraction() for t,p,g in members.values()),Fraction())/3
    E1=sum((Fraction(100*n,d) for n,d in single.values()),Fraction())/3
    return dict(numerator=numerator,denominator=denominator,E3=Fraction(100*numerator,denominator),F1=f1,E1=E1,members=members)


def check(args):
    from beetrack.evaluate import load_tracks,frame_match,match_events
    replay,previous,native,v=field.helpers(args.repo)
    graph=field.load('independent203_graph_for206',args.repo/'tools/study203/verify203_scores.py')
    verifier_code={name:field.sha(Path(__file__).with_name(name)) for name in ('verify206.py','verify206_field.py','verify206_scores.py')}
    verifier_code.update(native=field.sha(native.__file__),geometry=field.sha(v.__file__),graph=field.sha(graph.__file__))
    root=args.root;scored=field.read(root/'score/native_scores.json');summary=field.read(root/'score/summary.json');greedy=field.read(root/'score/greedy.json');calls=field.read(root/'score/physical_receipts.json')
    identity=field.read(root/'score/input_receipt.json');assert identity==summary['input_receipt'] and identity['prereg']==field.PREREG
    assert identity['frozen_seal']==field.sha(root/'frozen/manifest.json') and identity['owners_seal']==field.sha(root/'owners/manifest.json')
    assert identity['source203_score_seal']==field.sha(args.source203_score/'manifest.json')
    expected_code=dict(official_evaluate_sha256=field.sha(args.repo/'beetrack/evaluate.py'),scorer_sha256=field.sha(args.repo/'tools/study195/run.py'),source193_sha256=field.sha(args.repo/'tools/study193/source193.py'))
    assert identity['code']==expected_code
    assert set(scored)=={'K','R05','R03'}
    states={};roles={};prefixes=[]
    for arm,videos in scored.items():
        assert len(videos)==20
        for video in videos:
            assert len(video['members'])==3
            for member in video['members']:
                key=video['unit'],str(member['member']),arm;assert key not in states
                states[key]=member['selected_query_ids'];roles[key]=member
    assert len(states)==180
    for group in greedy:
        for threshold,report in group['thresholds'].items():
            for step in report['steps']:
                prefix=step['physical_prefix'];prefixes.append((group,threshold,step,prefix))
                assert prefix['complete_group_native_scoring'] and prefix['raw_count_closure_exact']
                for request in prefix['requests']:
                    call=request['physical'];key=call['unit'],str(call['member']),call['arm'];assert key not in states
                    assert call['arm']==f'prefix:{threshold}:{step["step"]}' and request['unit']==call['unit'] and request['member']==call['member']
                    states[key]=request['query_ids'];roles[key]=request
    callmap={(c['unit'],str(c['member']),c['arm']):c for c in calls};assert len(callmap)==len(calls)==len(states) and set(callmap)==set(states)
    seen=set()
    for call in calls:
        assert call['within_job_shared']==(call['path'] in seen);seen.add(call['path'])
    scope=summary['physical_scope'];assert scope['requested']==len(calls) and scope['unique_geometry']==len(seen) and scope['within_job_shared']==sum(c['within_job_shared'] for c in calls)
    assert scope['final_arm_requests']==180 and scope['prefix_requests']==len(calls)-180
    assert scope['unique_by_origin']=={o:len({c['path'] for c in calls if c['origin']==o}) for o in ('fresh','historical')}
    references=field.read(root/'frozen/deployment.json')['groups'];domain_by_unit={r['unit']:r['domain'] for r in references}
    changed=defaultdict(set)
    for key,ids in states.items():
        if ids:changed[domain_by_unit[key[0]]].add(callmap[key]['path'])
    fresh={};cache={};GT={};raw_by_unit={};requested=0
    old={ (g['unit'],str(m['member'])):m['metrics'] for g in field.read(args.source203_score/'native_scores.json')['K'] for m in g['members']}
    for ref,group,members in field.groups(args):
        unit=group['unit'];hive=group['domain'];compact=field.read(root/'verification/owner_compact'/(ref['path']+'.json'))
        queries={q['query_id']:q for m in compact['members'] for q in m['queries']};by_member={str(m['member']):m for m in members}
        if group['video'] not in GT:GT[group['video']]=load_tracks(args.repo/'runs/gt'/group['split']/group['video']/'gt/gt.txt')
        gt=GT[group['video']]
        for key in sorted((key for key in states if key[0]==unit),key=lambda k:({'K':0,'R05':1,'R03':2}.get(k[2],3),k[2],k[1])):
            _,member_id,arm=key;member=by_member[member_id];raw=np.asarray(member['rows'],float);selected=states[key];assert selected==sorted(set(selected))
            threshold='0.3' if arm=='R03' else arm.split(':')[1] if arm.startswith('prefix:') else '0.5'
            additions=[];sources=list(member['sources']);provenance=[]
            for query_id in selected:
                q=queries[query_id];assert q['member_index']==member['member_index'] and q['labels'][threshold]['confirmed'];candidate=q['labels'][threshold]['candidate']
                additions.append(candidate['added_row']);sources.append(candidate['source_key']);provenance.append(dict(query_id=query_id,gt_row_key=candidate['GT_row_key'],source_key=candidate['source_key']))
            rows=np.vstack((raw,np.asarray(additions,float).reshape(-1,7)))
            assert rows[:len(raw)].tobytes()==raw.tobytes() and np.isfinite(rows).all() and (rows[:,4:6]>0).all()
            assert len(set(map(tuple,rows[:,:2])))==len(rows) and len(set(map(tuple,sources)))==len(sources)
            before=core.native_membership(raw);after=core.native_membership(rows);assert all(after[i]==alias for i,alias in before.items())
            for j,query_id in enumerate(selected):assert after[len(raw)+j]==queries[query_id]['canonical_id']
            proof=dict(sources=sources,provenance=provenance,original_K_exact=True,all_original_run_aliases_exact=True)
            call=callmap[key];assert call['provenance_sha256']==object_sha(proof);path=root/'score/physical'/call['path'];assert path.resolve().is_relative_to((root/'score/physical').resolve())
            if call['path'] not in cache:
                field.seal(path,call['physical_seal']);meta=field.read(path/'receipt.json');actual=field.read(path/'score.json');ident=meta['identity']
                assert ident==dict(unit=unit,member=member_id,rows_sha256=core.array_sha(rows),source_map_sha256=object_sha(sources),GT_sha256=core.array_sha(gt),polygon_sha256=core.array_sha(group['polygon']),nf=group['nf'],scorer_code=expected_code)
                assert call['path']==object_sha(ident) and meta['origin']==call['origin']
                disk=np.loadtxt(path/'mot.txt',delimiter=',',usecols=range(7),ndmin=2) if len(rows) else np.empty((0,7));assert disk.tobytes()==rows.tobytes()
                checkpoint=root/'verification/physical'/(call['path']+'.json')
                binding=dict(physical_seal=call['physical_seal'],identity=ident,verifier_code=verifier_code)
                prior_check=None
                if checkpoint.exists() and checkpoint.with_suffix('.sha256').exists():
                    assert field.sha(checkpoint)==checkpoint.with_suffix('.sha256').read_text().strip();prior_check=field.read(checkpoint);assert prior_check['binding']==binding and prior_check['passed']
                else:truth,prediction=graph._native_graph(actual,rows,gt,group['polygon'],group['nf'],native,core.bins)
                if meta['origin']=='historical':
                    assert not selected and rows.tobytes()==raw.tobytes()
                    history=meta['historical'];assert history['source_score_seal']==identity['source203_score_seal']
                    previous.sealed(args.source203_score/'physical'/history['source_leaf'],history['source_leaf_seal'])
                dofresh=bool(selected) and (len(changed[hive])<=3 or hive not in fresh)
                sample=prior_check.get('fresh_sample') if prior_check else None
                if sample:fresh.setdefault(hive,[]).append(sample)
                elif dofresh:
                    if prior_check:truth,prediction=native.track_structure(gt,group['polygon']),native.track_structure(rows,group['polygon'])
                    owner_vectors={}
                    for t in (.5,.3):
                        matching=frame_match(truth['rows'],prediction['rows'],thr=t);native_order=np.empty(len(rows),np.int64);native_order[prediction['index']]=matching;owner_vectors[str(t)]=native_order
                    own=native.score_from_owners(prediction,truth,owner_vectors,group['polygon'],group['nf'])
                    for t,edge in (('0.5','edges05'),('0.3','edges03')):
                        for f in ('tp','pred','gt','tp_id'):assert own['iou'][t][f]==actual['iou'][t][f]
                        assert own['edges'][t]==actual['evidence'][edge]
                    assert match_events(gt,rows,group['polygon'])['tp']==actual['iou']['0.5']['tp']
                    sample=dict(unit=unit,member=member_id,physical=call['path'],added_rows=len(additions),thresholds=[.5,.3]);fresh.setdefault(hive,[]).append(sample)
                field.save(checkpoint,dict(passed=True,binding=binding,graph_maximum_cardinality=True,complete_native_matrix_exact=True,fresh_sample=sample))
                light={k:actual[k] for k in ('nG','nP','pred_bins','gt_bins')};light['iou']={t:{f:actual['iou'][t][f] for f in ('tp','pred','gt','f1')} for t in ('0.5','0.3')}
                cache[call['path']]=dict(rows_sha256=core.array_sha(rows),seal=call['physical_seal'],metrics=light,edges05=actual['evidence']['edges05'] if not selected else None,edges03=actual['evidence']['edges03'] if not selected else None,predictions=actual['evidence']['predictions'] if not selected else None)
            verified=cache[call['path']];assert verified['rows_sha256']==core.array_sha(rows) and verified['seal']==call['physical_seal'];metrics=verified['metrics']
            events=[(f,s) for _,_,_,_,f,s in core.signed_events(rows,group['polygon'])];assert np.array_equal(core.bins(events,group['nf']),metrics['pred_bins'])
            expected=core.bins([(f,s) for _,_,_,_,f,s in core.signed_events(raw,group['polygon'])],group['nf'])
            for query_id in selected:q=queries[query_id];expected-=core.bins([(q['event']['frame'],q['event']['direction'])],group['nf'])
            assert np.array_equal(expected,metrics['pred_bins'])
            role=roles[key]
            if arm in ('K','R05','R03'):
                for f in ('nG','nP','pred_bins','gt_bins','iou'):assert role['metrics'][f]==metrics[f]
                denominator=int(np.asarray(metrics['gt_bins']).sum());W=100*int(abs(np.asarray(metrics['pred_bins'])-np.asarray(metrics['gt_bins'])).sum())/denominator if denominator else None
                assert role['metrics']['W'] is None if W is None else abs(role['metrics']['W']-W)<1e-11
                if arm=='K':
                    prior=old[unit,member_id]
                    for f in ('nG','nP','pred_bins','gt_bins'):assert metrics[f]==prior[f]
                    for t in ('0.5','0.3'):
                        for f in ('tp','pred','gt','f1'):assert metrics['iou'][t][f]==prior['iou'][t][f]
            for t in ('0.5','0.3'):
                metric=metrics['iou'][t];assert role['TP_FP_FN'][t]==dict(TP=metric['tp'],FP=metric['pred']-metric['tp'],FN=metric['gt']-metric['tp'])
            requested+=1
        print('INDEPENDENT206_PHYSICAL',unit,flush=True)
    assert requested==len(calls)
    means={}
    for hive in ('35','33','W','G'):
        means[hive]={}
        for arm in ('K','R05','R03'):
            method=summary['domains'][hive]['methods'][arm];videos=[g for g in scored[arm] if g['domain']==hive];groupids=sorted({g['group'] for g in videos});aggregates={t:[] for t in ('0.5','0.3')}
            for groupid in groupids:
                clips=[g for g in videos if g['group']==groupid];saved=next(g for g in method['groups'] if g['group']==groupid)
                for t in aggregates:
                    own=pool(clips,t);aggregates[t].append(own)
                    assert own['numerator']==saved['numerator'] and own['denominator']==saved['denominator']
                    assert Fraction(*saved['W_mean_member_fraction'])==own['E1']
                    reported={str(m['member']):m for m in saved['iou'][t]['members']};assert reported.keys()==own['members'].keys()
                    for member,(tp,pred,gt) in own['members'].items():
                        assert {k:reported[member][k] for k in ('tp','pred','gt')}==dict(tp=tp,pred=pred,gt=gt)
            E3=sum((x['E3'] for x in aggregates['0.5']),Fraction())/len(groupids);E1=sum((x['E1'] for x in aggregates['0.5']),Fraction())/len(groupids)
            assert Fraction(*method['E3_fraction'])==E3 and Fraction(*method['W_mean_member_fraction'])==E1
            f1={t:sum((x['F1'] for x in aa),Fraction())/len(aa) for t,aa in aggregates.items()}
            for t,value in f1.items():assert Fraction(*method['F1_fraction'][t])==value
            means[hive][arm]=dict(E3=[E3.numerator,E3.denominator],W_mean_member=[E1.numerator,E1.denominator],F1={t:[value.numerator,value.denominator] for t,value in f1.items()})
    for group,threshold,step,prefix in prefixes:
        videos={}
        for request in prefix['requests']:
            call=request['physical'];unit=call['unit'];video=videos.setdefault(unit,dict(members=[]));video['members'].append(dict(member=call['member'],metrics=cache[call['path']]['metrics']))
        for t in ('0.5','0.3'):
            own=pool(list(videos.values()),t)
            assert own['numerator']==prefix['group_numerator'] and own['denominator']==prefix['denominator']
            reported={str(m['member']):m for m in prefix['iou'][t]['members']};assert reported.keys()==own['members'].keys()
            for member,(tp,pred,gt) in own['members'].items():assert {k:reported[member][k] for k in ('tp','pred','gt')}==dict(tp=tp,pred=pred,gt=gt)
    edge_audits=field.read(root/'score/selected_original_edges.json');edge_counts={t:[0,0] for t in ('0.5','0.3')}
    compact_by_unit={r['unit']:field.read(root/'verification/owner_compact'/(r['path']+'.json')) for r in references}
    for unit,arms in edge_audits.items():
        querymap={q['query_id']:(str(m['member']),q) for m in compact_by_unit[unit]['members'] for q in m['queries']}
        for arm,audit in arms.items():
            ids=sorted(q for key,selected in states.items() if key[0]==unit and key[2]==arm for q in selected)
            assert [record['query_id'] for record in audit['records']]==ids and audit['selected_count']==audit['denominator']==len(ids)
            zeros={t:0 for t in ('0.5','0.3')}
            for record in audit['records']:
                member,q=querymap[record['query_id']];base=cache[callmap[unit,member,'K']['path']];pid=q['canonical_id']
                assert {p[0]:p[1] for p in base['predictions']}[pid]==q['event']['frame']*q['event']['direction']
                assert record['member']==member and record['canonical_id']==pid and record['event']==q['event']
                for t,fieldname in (('0.5','edges05'),('0.3','edges03')):
                    neighbors=sorted({g for p,g in base[fieldname] if p==pid});assert record['legal_GT_neighbors'][t]==neighbors and record['has_any_legal_edge'][t]==bool(neighbors);zeros[t]+=not neighbors
            for t,z in zeros.items():
                assert audit['zero_edge_count'][t]==z and audit['zero_edge_fraction'][t]==(z/len(ids) if ids else None);edge_counts[t][0]+=z;edge_counts[t][1]+=len(ids)
    K=means['35']['K'];R=means['35']['R05'];gain=Fraction(*K['E3'])-Fraction(*R['E3']);guard=Fraction(*R['F1']['0.5'])>=Fraction(*K['F1']['0.5'])-Fraction(1,200)
    result=dict(passed=True,requests=requested,unique=len(cache),fresh_changed=fresh,unchanged_domains=[h for h in ('35','33','W','G') if not changed[h]],means=means,
        main_investment=dict(gain_fraction=[gain.numerator,gain.denominator],gain_at_least_one_pp=gain>=1,F1_guard=guard,passed=gain>=1 and guard),selected_original_edges=dict(counts=edge_counts,caution='Zero edges proves unmatched; positive edges does not prove matched. Zero net same-GT-run does not prove the original prediction was FP.'),
        scope='All final and every accepted-prefix complete native matrices/source provenance/30phase events and pooled E3/E1/member F1; all supplied new or exactly sealed historical graph maximum matchings. Fresh whole-scene both-IoU rematching only on listed changed geometries; TP_ID independently recomputed only there. Remaining TP_ID not independently re-evaluated.')
    field.save(root/'verification/scores.json',result);return result
