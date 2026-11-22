"""Explicit sole-CPU joint211 independent audit; importing performs no field IO.

Own original-family reconstruction, mixed row merge, state enumeration, clock,
DFS matching, policies and arithmetic. Shared: sealed loaders, polygon kernel,
native event-edge legality and independently implemented ordered Hungarian cache.
No joint producer merge/selector/fitter is called.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,gzip,hashlib,importlib.util,json,sys
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import verify_joint211 as own

SUPPLEMENT='3323107ea3560e9ee507e65891097cf2ca605b00'
GRANT='root-unique-joint211-independent-cpu'
DOMAINS=('35','33','W','G')

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def json_scalar(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,F):return [x.numerator,x.denominator]
    raise TypeError(type(x).__name__)
def wire(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False,default=json_scalar).encode()
def digest(x):return hashlib.sha256(wire(x)).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sealed(root,expected):
    root=Path(root);assert sha(root/'manifest.json')==expected==(root/'SEALED.sha256').read_text().strip()
    files=read(root/'manifest.json')
    for name,h in files.items():
        p=(root/name).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==h,(name,'sealed file drift')
    return files
def matrix_sha(rows):return hashlib.sha256(wire(list(rows.shape))+np.asarray(rows,dtype='<f8').tobytes()).hexdigest()
def same(a,b):assert sorted(map(tuple,a))==sorted(map(tuple,b))
def record_peer(saved,result,K):
    g=result['graphs']['0.5'];same(saved['edges'],g['edges'])
    assert saved['nPred']==len(g['left']) and saved['nReference']==len(g['right'])
    assert saved['R']==own.matching(g['left'],g['right'],g['edges'])==g['maxTP']
    assert saved['J']==len({p for p,r in g['edges']})
    same(saved['matching'],[(p,r) for p,r in saved['matching']])
    assert len(saved['matching'])==g['maxTP'] and len({p for p,r in saved['matching']})==g['maxTP'] and len({r for p,r in saved['matching']})==g['maxTP']
    assert set(map(tuple,saved['matching']))<=set(map(tuple,g['edges']))
    for f in saved['flags']:
        p=f['canonical_id'];assert f['has_neighbor']==any(p==q for q,r in g['edges'])
        if K:assert f['drop_left_marginal']==g['maxTP']-own.matching(g['left'],g['right'],g['edges'],p)
    assert {f['canonical_id'] for f in saved['flags']}==set(g['left'])
    for name in ('pred_windows','reference_windows'):assert np.array_equal(saved[name],result[name])
    for name,structure,idkey in (('prediction_vertices',result['prediction_structure'],'canonical_id'),('reference_vertices',result['reference_structure'],'canonical_id')):
        assert {(v[idkey],v['frame'],v['direction']) for v in saved[name]}=={(int(p),int(f),int(d)) for p,(f,d) in structure['events'].items()}
    return dict(g,pred_windows=result['pred_windows'],reference_windows=result['reference_windows'])

def source_actions(job,c,original,main,prediction):
    """Reconstruct original common slate and every local delta, without GT."""
    roots={k:Path(v) for k,v in job['parent_roots'].items()}
    old=read(roots['predictions209']/'predictions.json')['clips'][c['domain']+'/'+c['clip']]
    features=read(roots['features209']/c['main_relative']/'features.json')
    valid={p['pair_id']:p['true']['valid'] for p in features['pairs']};table={p['pair_id']:p for p in original['pairs']}
    assert set(valid)==set(table)
    common=[];used=set()
    for p in sorted((p for p in table.values() if p['geometric_eligible'] and valid[p['pair_id']]),key=lambda p:(-p['mean_iou'],*p['canonical_ids'])):
        if used.intersection(p['canonical_ids']):continue
        used.update(p['canonical_ids']);common.append(p['pair_id'])
    assert common==old['slate']==c['original209_common_slate']
    expected={};R=original['rows'];canon=original['canon'];poly=main['metadata']['polygon'];nf=main['metadata']['nf']
    for key in common:
        p=table[key];ix=np.flatnonzero(np.isin(canon,p['canonical_ids']));local=R[ix].copy();lc=canon[ix]
        # Extracted gap-run ordinals start at zero locally: retain immutable IDs
        # explicitly instead of incorrectly re-aliasing a later raw-ID fragment.
        local[:,1]=lc
        act=dict(family='C',canonical_ids=p['canonical_ids'])
        merged,_=own.merge(local,lc,{'one':act},['one'])
        before,_=own.event_provenance(local,poly);after,_=own.event_provenance(merged,poly)
        delta=own.clock(after.values(),nf)-own.clock(before.values(),nf)
        assert np.array_equal(delta,c['original209_delta_by_pair'][key])
        if delta.any():expected[('C',*p['canonical_ids'])]=(delta,p['mean_iou'])
    active=main['active_pairs'];assert {p['pair_id'] for p in active}==set(c['original210_active_slate'])
    for p in active:
        expected[('I',*p['canonical_ids'])]=(np.asarray(p['delta_windows'],int),p['min_offset_norm'])
    actions=prediction['actions'];assert {own.action_key(a) for a in actions.values()}==set(expected)
    for aid,a in actions.items():
        d,metric=expected[own.action_key(a)];assert np.array_equal(d,a['delta_windows'])
        assert a['mean_shared_iou' if own.family(a)=='C' else 'min_local_norm']==metric
        assert aid==('209:' if own.family(a)=='C' else '210:')+'|'.join(map(str,a['canonical_ids']))
    assert prediction['source_seals']==c['source_seals']
    return actions

def run(args):
    if args.cpu_grant!=GRANT:raise PermissionError('root explicit independent sole-CPU handoff required')
    own.selftest();job=read(args.job);assert job['prereg']==SUPPLEMENT and job['marker']=='[cross-family-peer-v1]'
    repo=Path(job['repo']);local=Path(__file__).parent;sys.path.insert(0,str(repo))
    # Verify complete prediction and assessment gates BEFORE any annotations.
    predroot=Path(job['prediction_out']);assert sha(predroot/'ALL_FOUR_SEAL.json')==args.prediction_seal==(predroot/'ALL_FOUR_SEAL.sha256').read_text().strip()
    gate=read(predroot/'ALL_FOUR_SEAL.json');assert set(gate['files'])=={h+'.joint_predictions.json' for h in DOMAINS}
    for name,h in gate['files'].items():assert sha(predroot/name)==h
    sealed(job['score_out'],args.assessment_seal);assessment=read(Path(job['score_out'])/'assessment.json')
    assert assessment['prediction_seal']==digest(gate) and assessment['all_legal_states_scored']
    assert assessment['receipts']==gate['receipts'] and gate['receipts']['job_sha256']==digest(job)
    for name,h in gate['receipts']['code_sha256'].items():assert sha(local/name)==h
    builder=module('jointaudit_builder',local/'build_joint_job211.py')
    assert job['original_parent_seals']==builder.PINS
    for key,h in builder.PINS.items():
        # Input208 large files are individually verified by the frozen loaders.
        if key=='inputs208':assert sha(Path(job['parent_roots'][key])/'manifest.json')==h==(Path(job['parent_roots'][key])/'SEALED.sha256').read_text().strip()
        else:sealed(job['parent_roots'][key],h)
    sealed(job['main_models'],job['main_parameter_seal'])
    assert gate['receipts']['main_parameter_seal']==job['main_parameter_seal']
    assert {(c['domain'],c['clip']) for c in job['clips']}=={(h,c) for h,c,s in builder.SIX}
    freeze=module('jointaudit_freeze_loader',local/'freeze211.py')
    co=module('jointaudit_co_loader',local.parent/'study209/catalog209.py')
    independent=module('jointaudit_primitive',local/'audit211.py')
    field=module('jointaudit_geometry',local/'field211.py')
    public=module('jointaudit_public',repo/'tools/study203/verify203.py');previous,native,v,certificate=public.helpers(repo)
    from beetrack.evaluate import load_tracks
    out=args.out
    if out.exists():raise FileExistsError('new independent staging directory required')
    out.mkdir(parents=True);reports=[]
    actuals={(x['domain'],x['clip']):x for x in assessment['clips']}
    main_assessment=Path(job['main_frozen']).parent/'assessment'
    sealed(main_assessment,'7df94f0fbac521d715349651b5f9c1527854748535ea70399733c23b813a8d81')
    main_index=read(main_assessment/'index.json')
    scorer_receipt=digest({str(p.relative_to(repo)):sha(p) for p in (repo/'beetrack/evaluate.py',repo/'tools/study195/run.py',repo/'tools/study193/source193.py')})
    reuse_keys=set()
    for c in job['clips']:
        h,name=c['domain'],c['clip'];main=freeze.load_clip(Path(job['main_frozen'])/c['main_relative']);meta=main['metadata']
        assert meta['seed']==(3 if h in ('35','33') else 0) and 0<=meta['seed']<=14
        original=co.load_clip(Path(c['catalog209']),Path(job['inputs208']));base=main['native']['receiver'];canon=main['native']['receiver_canon'];poly=meta['polygon'];nf=meta['nf']
        assert np.array_equal(base.view(np.uint64),original['rows'].view(np.uint64)) and np.array_equal(canon,original['canon'])
        own.original_runs(base,canon)
        fold=read(predroot/(h+'.joint_predictions.json'));prediction=next(x for x in fold['clips'] if x['clip']==name)
        assert set(actuals[h,name]['states'])=={s['state_id'] for s in prediction['states']}
        model=read(Path(job['main_models'])/'models'/f'{h}.json');assert model['heldout']==h and set(model['source_domains'])==set(DOMAINS)-{h}
        assert prediction['model']=={p:{k:model[p][k] for k in ('a','b')} for p in ('Q','L','X')}
        actions=source_actions(job,c,original,main,prediction)
        assert {tuple(sorted(s['selected_action_ids'])) for s in prediction['states']}=={tuple(sorted(s)) for s in own.legal_subsets(actions)}
        # GT exact original six paths/hash bound to the already audited main22.
        gtentry=next(x for x in main_index['clips'] if (x['domain'],x['clip'])==(h,name))['GT_entry']
        gtpath=repo/gtentry['path'];assert gtpath.resolve()==Path(c['gt_path']).resolve() and sha(gtpath)==gtentry['sha256']
        refs=[load_tracks(gtpath),main['native']['peer1'],main['native']['peer2']]
        structures=[native.track_structure(r,poly) for r in refs]
        caches=[field.OrderedGeometryCache(r,base,v,('0.5','0.3') if i==0 else ('0.5',)) for i,r in enumerate(refs)]
        for r,s in zip(refs,structures):assert own.event_provenance(r,poly)[0]==s['events']
        Kbins=own.clock(own.event_provenance(base,poly)[0].values(),nf);states=[];provenance={}
        for saved in prediction['states']:
            sid=saved['state_id'];selected=saved['selected_action_ids'];rows,proof=own.merge(base,canon,actions,selected);order=proof['output_raw_indices']
            assert sid==('K' if not selected else digest(sorted(selected)))
            assert saved['retained_raw_indices']==order and saved['deleted_raw_indices']==proof['deleted_raw_indices']
            assert saved['scope_digest']==matrix_sha(rows) and saved['deletion_digest']==digest(proof['deleted_raw_indices'])
            assert saved['physical_id_vector']==rows[:,1].astype(int).tolist()
            P=native.track_structure(rows,poly);events,details=own.event_provenance(rows,poly,order,canon);assert events==P['events'];provenance[sid]=details
            bins=own.clock(events.values(),nf);expected=Kbins.copy()
            for aid in selected:expected+=np.asarray(actions[aid]['delta_windows'],int)
            assert np.array_equal(bins,expected) and np.array_equal(bins,saved['windows'])
            calculations=[]
            for i,(reference,cache) in enumerate(zip(structures,caches)):
                cache.prepare(P);calculations.append(independent.audit_native_state(rows,reference,poly,nf,order,cache,native,own.clock,P,('0.5','0.3') if i==0 else ('0.5',)))
            peers=[record_peer(g,r,not selected) for g,r in zip(prediction['peer_graphs'][sid],calculations[1:])]
            summary=own.peer_summary(peers);assert prediction['state_summaries'][sid]==dict(nPred=summary['nP'],meanJ=summary['meanJ'],meanM=summary['meanM'],meanCountL1=summary['meanCountL1'])
            current=calculations[0];record=actuals[h,name]['states'][sid];metrics=record['metrics']
            assert record['scope_digest']==saved['scope_digest'] and record['deletion_digest']==saved['deletion_digest']
            gh=matrix_sha(refs[0]) if refs[0].shape[1]==7 else hashlib.sha256(np.asarray(refs[0],dtype='<f8').tobytes()).hexdigest()
            reuse_key=digest(dict(physical=matrix_sha(rows),gt=gh,nframes=nf,polygon=np.asarray(poly).tolist(),scorer=scorer_receipt))
            assert record['strict_reuse_key']==reuse_key and record['reused']==(reuse_key in reuse_keys)
            reuse_keys.add(reuse_key)
            assert np.array_equal(metrics['pred_bins'],bins) and np.array_equal(metrics['gt_bins'],current['reference_windows'])
            assert metrics['nP']==len(events) and metrics['nG']==len(structures[0]['events'])
            for t,edge in (('0.5','edges05'),('0.3','edges03')):
                graph=current['graphs'][t];same(metrics['evidence'][edge],graph['edges']);same(record['true_graphs'][t]['edges'],graph['edges'])
                assert record['true_graphs'][t]['R']==graph['maxTP']==own.matching(graph['left'],graph['right'],graph['edges'])
                for key in ('tp','pred','gt','tp_id'):assert metrics['iou'][t][key]==current['iou'][t][key]
            N=int(np.abs(bins-current['reference_windows']).sum());D=int(current['reference_windows'].sum())
            states.append(dict(state_id=sid,selected_action_ids=selected,**summary,numerator=N,denominator=D,iou=current['iou'],graphs=current['graphs'],pred_bins=bins.tolist(),gt_bins=current['reference_windows'].tolist(),events={str(k):list(v) for k,v in events.items()},output_raw_indices=order,rows_sha256=own.array_sha(rows),proof=proof))
            # PureI must equal original main22 physical bytes/order, no producer constructor.
            if all(own.family(actions[aid])=='I' for aid in selected):
                pairs={tuple(own.endpoints(actions[aid])) for aid in selected}
                ms=next(s for s in main['states'] if {tuple(map(int,p.split('|'))) for p in s['selected_pair_ids']}==pairs)
                with np.load(Path(job['main_frozen'])/c['main_relative']/ms['materialization_file']) as z:
                    prior=base[z['output_raw_indices']].copy();prior[:,1]=z['ids']
                assert rows.tobytes()==prior.tobytes()
            print('JOINT_INDEPENDENT_STATE',h,name,sid,flush=True)
        choices=own.choose(states,actions,prediction['model']);assert choices['choices']==prediction['choices']==actuals[h,name]['selected']
        for policy in ('Q','L','X','V','C'):
            for sid,U in choices['utilities'].items():assert U[policy]==prediction['utilities'][sid][policy]
        for f in ('C','I'):
            assert choices['family_only'][f]['choices']==prediction['family_only'][f]['choices']
            assert set(choices['family_only'][f]['legal_state_ids'])==set(prediction['family_only'][f]['legal_state_ids'])
        assert [prediction['joint_Z']['dose_C'],prediction['joint_Z']['dose_I']]==list(choices['Z_dose'])
        assert prediction['joint_Z']['rank_sum']==choices['Z_rank_fraction']
        fronts={t:own.pareto(states,t) for t in ('0.5','0.3')}
        best=min(states,key=lambda s:(s['numerator'],len(s['selected_action_ids']),own.state_key(s,actions)))
        transitions=[];lookup={frozenset(s['selected_action_ids']):s for s in states}
        for after in states:
            for aid in after['selected_action_ids']:
                before=lookup[frozenset(after['selected_action_ids'])-{aid}];ends=set(own.endpoints(actions[aid]))
                old=[e for e in provenance[before['state_id']].values() if len(e['original_parent_ids'])==1 and set(e['original_parent_ids'])<=ends]
                new=[e for e in provenance[after['state_id']].values() if set(e['original_parent_ids'])==ends]
                checks={t:own.strict_duplicate(before['graphs'][t],after['graphs'][t],old,new) for t in ('0.5','0.3')}
                transitions.append(dict(before=before['state_id'],after=after['state_id'],action=aid,strict_duplicate=checks,
                    count_numerator_change=after['numerator']-before['numerator'],iou_changes={t:dict(TP=after['iou'][t]['tp']-before['iou'][t]['tp'],FP=(after['iou'][t]['pred']-after['iou'][t]['tp'])-(before['iou'][t]['pred']-before['iou'][t]['tp']),FN=(after['iou'][t]['gt']-after['iou'][t]['tp'])-(before['iou'][t]['gt']-before['iou'][t]['tp'])) for t in ('0.5','0.3')}))
        reports.append(dict(domain=h,clip=name,actions=actions,states=states,choices=choices,pareto=fronts,count_best=best['state_id'],all_count_min_states=[s['state_id'] for s in states if s['numerator']==best['numerator']],strict_duplicate_transitions=transitions,frame_requests=[x.requests for x in caches],unique_ordered_geometry=[len(x.cache) for x in caches],event_provenance=provenance))
        (out/(h+'_'+name+'.json')).write_bytes(wire(reports[-1])+b'\n')
    # Pool raw clip N/D and event totals first, never average W/G clip percentages.
    domains={};family_domains={};count_oracle={}
    def pool(selected):
        N=sum(s['numerator'] for s in selected);D=sum(s['denominator'] for s in selected)
        pooled=dict(numerator=N,denominator=D,W_fraction=[100*N,D],iou={})
        for t in ('0.5','0.3'):
            totals={k:sum(s['iou'][t][k] for s in selected) for k in ('tp','pred','gt')};value=own.f1(**totals)
            pooled['iou'][t]=dict(**totals,FP=totals['pred']-totals['tp'],FN=totals['gt']-totals['tp'],E=totals['pred']+totals['gt']-2*totals['tp'],F1_fraction=[value.numerator,value.denominator])
        return pooled
    for h in DOMAINS:
        rr=[r for r in reports if r['domain']==h];methods={}
        for policy in own.POLICIES:
            selected=[next(s for s in r['states'] if s['state_id']==r['choices']['choices'][policy]) for r in rr]
            methods[policy]=pool(selected)
        domains[h]=methods
        family_domains[h]={f:{p:pool([next(s for s in r['states'] if s['state_id']==r['choices']['family_only'][f]['choices'][p]) for r in rr]) for p in own.POLICIES} for f in ('C','I')}
        count_oracle[h]=pool([next(s for s in r['states'] if s['state_id']==r['count_best']) for r in rr])
        change=F(100*(methods['K']['numerator']-count_oracle[h]['numerator']),methods['K']['denominator'])
        count_oracle[h]['maximum_W_gain_fraction']=[change.numerator,change.denominator]
        count_oracle[h]['witnesses']={r['clip']:r['count_best'] for r in rr}
    gain=lambda h,p='X':F(100*(domains[h]['K']['numerator']-domains[h][p]['numerator']),domains[h]['K']['denominator'])
    F1=lambda h,p:F(*domains[h][p]['iou']['0.5']['F1_fraction'])
    gates=dict(head35_gain_at_least_1pp=gain('35')>=1,head33_gain_at_least_1pp=gain('33')>=1,
        head35_F1_loss_at_most_point01=F1('35','K')-F1('35','X')<=F(1,100),head33_F1_loss_at_most_point01=F1('33','K')-F1('33','X')<=F(1,100),W_no_worse=gain('W')>=0,G_no_worse=gain('G')>=0)
    attribution={};complementarity={}
    for h in ('35','33'):
        r=next(r for r in reports if r['domain']==h);byid={s['state_id']:s for s in r['states']};x=byid[r['choices']['choices']['X']];l=byid[r['choices']['choices']['L']]
        attribution[h]=dict(X_differs_L=x['state_id']!=l['state_id'],X_has_lower_E_at_05=(x['iou']['0.5']['pred']+x['iou']['0.5']['gt']-2*x['iou']['0.5']['tp'])<(l['iou']['0.5']['pred']+l['iou']['0.5']['gt']-2*l['iou']['0.5']['tp']))
        complementarity[h]=dict(X_uses_both_families={own.family(r['actions'][a]) for a in x['selected_action_ids']}=={'C','I'},X_strictly_lower_W_than_pure_C=x['numerator']<byid[r['choices']['pure_X']['C']]['numerator'],X_strictly_lower_W_than_pure_I=x['numerator']<byid[r['choices']['pure_X']['I']]['numerator'])
    report=dict(schema='joint211-independent-v1',passed=True,supplement=SUPPLEMENT,prediction_seal=args.prediction_seal,assessment_seal=args.assessment_seal,job_sha256=sha(args.job),domains=domains,family_domains=family_domains,count_oracle=count_oracle,gates=gates,all_gates_pass=all(gates.values()),capacity1_attribution=attribution,complementarity=complementarity,clips=[dict(domain=r['domain'],clip=r['clip'],states=len(r['states']),count_best=r['count_best'],pareto=r['pareto']) for r in reports],scope=dict(independent=['source slate/delta','mixed rows/winner/lost','legal matching subsets','first/last side/last crossing','30phase integer clock','DFS maximum matching','frozen utilities/ties/exact dose','F1 Pareto and pooled gates'],shared=['sealed loaders','polygon kernel','native event legal edge helper','ordered raw-index SciPy Hungarian cache'],not_repeated='sealed source-label fitting; coefficients compared exactly'),code_sha256={p.name:sha(p) for p in (Path(__file__),local/'verify_joint211.py',local/'audit211.py',local/'field211.py')})
    (out/'independent.json').write_bytes(wire(report)+b'\n');manifest={p.name:sha(p) for p in out.iterdir() if p.is_file()};(out/'manifest.json').write_bytes(wire(manifest)+b'\n');(out/'SEALED.sha256').write_text(sha(out/'manifest.json')+'\n')
    print('JOINT_INDEPENDENT_SEALED',sha(out/'manifest.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--prediction-seal',required=True);p.add_argument('--assessment-seal',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cpu-grant',required=True);run(p.parse_args())
