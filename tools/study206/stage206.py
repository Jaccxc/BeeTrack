"""Explicitly gated206 orchestration. Importing reads no field artifacts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import gc,importlib.util,inspect,json
from pathlib import Path
from fractions import Fraction
import numpy as np
import score206 as score

DOMAINS=('35','33','W','G')
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def load(path):return json.loads(Path(path).read_text())
def immutable(path,value):
    text=json.dumps(score.plain(value),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    if path.exists():
        if path.read_text(encoding='utf8')!=text:raise ValueError('immutable result differs '+str(path))
    else:path.write_text(text,encoding='utf8')

def bind_joint(unit,prepared_gt,owners):
    """Corrected owner validator, not singleton matching or a preserved old graph."""
    def validate(current,ids,t):
        if current['unit']!=unit['unit']:raise ValueError('wrong unit joint callback')
        audits=[];wanted=set(ids);known={score.query_key(q) for m in unit['members'] for q in m['queries']}
        if len(wanted)!=len(ids) or not wanted<=known:raise ValueError('foreign joint selection')
        for m in unit['members']:
            selected=[dict(query=q,candidate=score.candidate(q,t)) for q in sorted(m['queries'],key=score.query_key) if score.query_key(q) in wanted]
            if not selected:continue
            proof=owners.joint_validate(prepared_gt,m['rows'],m['sources'],selected,float(t),unit['nf'])
            audits.append(dict(member=str(m['member']),proof=proof))
        safe=all(a['proof']['valid'] for a in audits)
        return dict(safe=safe,reason='joint_valid' if safe else 'joint_return_rejected',members=audits)
    return validate

def merge_unit(frozen,owned):
    from tools.study199.owners199 import array_sha as owner_array_sha
    if frozen['unit']!=owned['unit'] or len(frozen['members'])!=3 or len(owned['members'])!=3:raise ValueError('owner/freeze roster differs')
    members=[]
    for fm,om in zip(frozen['members'],owned['members']):
        if fm['member_index']!=om['member_index'] or str(fm['member'])!=str(om['member']):raise ValueError('slot/physical member drift')
        if not 0<=fm['seed']<=14 or frozen['split']=='test':raise ValueError('outside old scope')
        if owner_array_sha(fm['rows'])!=om['original_K_sha256']:raise ValueError('owner K differs')
        original={q['query_id']:q for q in fm['roster']['queries']}
        if set(original)!={q['query_id'] for q in om['queries']}:raise ValueError('event queries filtered or introduced')
        for q in om['queries']:
            if {k:v for k,v in q.items() if k!='labels'}!=original[q['query_id']]:raise ValueError('owner changes frozen query')
        events=[(q['event']['frame'],q['event']['direction']) for q in om['queries']]
        members.append(dict(om,rows=fm['rows'],sources=fm['sources'],baseline_windows=score.bins(events,frozen['nf']).tolist()))
    pred=sum((np.asarray(m['baseline_windows']) for m in members))
    if not np.array_equal(pred,owned['pred_sum']):raise ValueError('original query events do not exhaust K bins')
    return dict(owned,members=members)

def historical_provider(root,pinned_seal,scorer_code):
    common=score._common();_,digest=common.verify_manifest(root,('native_scores.json','summary.json'))
    if digest!=pinned_seal:raise ValueError('historical203 score seal differs')
    summary=load(root/'summary.json')
    if summary['input_receipt']['source_seals']['frozen']!='439a8746c19c3857fe7192990d63f08cb17c94145c8c18c6394218cc95e01b51':raise ValueError('historical K is not fixed203')
    rows=load(root/'native_scores.json')['K'];index={(r['unit'],str(m['member'])):m for r in rows for m in r['members']};del rows
    def provider(unit,member,identity):
        old=index[(unit['unit'],str(member['member']))];path=root/'physical'/old['physical'];receipt=common.verify(path)
        if (path/'SEALED.sha256').read_text().strip()!=old['seal']:raise ValueError('historical leaf differs sealed score index')
        prior=receipt['identity'];actual=dict(rows_sha256=common.array_sha(member['rows']),
            GT_sha256=common.array_sha(unit['_actual_gt']),polygon_sha256=common.array_sha(unit['polygon']),nf=unit['nf'])
        if any(prior[k]!=v for k,v in actual.items()):raise ValueError('historical K physical input differs')
        if prior['sources_sha256']!=common.object_sha(member['sources']):raise ValueError('historical K row sources differ')
        if any(prior['code'].get(k)!=v for k,v in scorer_code.items()):return None # fresh scorer if software changed
        native=load(path/'score.json')
        common.assert_K(native,old['metrics'])
        return dict(**{k:identity[k] for k in ('rows_sha256','GT_sha256','polygon_sha256','nf')},
            scorer_code=scorer_code,score=native,source_score_seal=digest,source_leaf_seal=(path/'SEALED.sha256').read_text().strip(),source_leaf=old['physical'])
    provider.references={key:value['metrics'] for key,value in index.items()}
    return provider

def count_bees(units,t,selected=None):
    bees=set();n=0;anchors=set();anchor_queries=0;considered=0
    for u in units:
        chosen=None if selected is None else set(selected[u['unit']])
        for m,q in score.queries(u):
            if chosen is not None and score.query_key(q) not in chosen:continue
            considered+=1
            if 'GT_run_key' in q['labels'][t]:anchors.add(score.object_sha(q['labels'][t]['GT_run_key']));anchor_queries+=1
            if score.candidate(q,t) is not None:
                n+=1;bees.add(score.object_sha(q['labels'][t]['GT_run_key']))
    return dict(all_queries_or_selected=considered,anchor_same_run_queries=anchor_queries,anchor_same_run_distinct_GT_runs=len(anchors),
        confirmed_return_member_queries=n,confirmed_return_distinct_GT_runs=len(bees),not_cross_clip_physical_identity=True)

def numeric_input(unit):
    out={k:unit[k] for k in ('unit','domain','group','video','nf','pred_sum','gt_bins')};out['members']=[]
    for m in unit['members']:
        qq=[]
        for q in m['queries']:
            labels={}
            for t in score.THRESHOLDS:
                original=q['labels'][t];c=score.candidate(q,t)
                labels[t]=dict(confirmed=original['confirmed'],classification=original['classification'],reason=original['reason'],
                    GT_run_key=original.get('GT_run_key'),candidate=None if c is None else {k:c[k] for k in ('query_id','frame','GT_row_key','event_delete_only','delta_windows')})
            qq.append(dict(query_id=q['query_id'],event=q['event'],canonical_id=q['canonical_id'],eligible=q['eligible'],labels=labels))
        out['members'].append(dict(member=m['member'],member_index=m['member_index'],queries=qq))
    return out

def prefix_scorer(units,cache):
    """Complete group/member scenes at EVERY adopted greedy prefix."""
    from tools.study198.run198 import combine
    def callback(selected,t,step,residual):
        videos=[];requests=[]
        for u in units:
            vv=dict(unit=u['unit'],domain=u['domain'],group=u['group'],video=u['video'],members=[])
            for m in u['members']:
                ids=sorted(set(selected[u['unit']]) & {q['query_id'] for q in m['queries']})
                rows,proof=score.materialize_member(m,ids,t,u['nf'])
                native=cache.score(u,m,rows,proof,u['_actual_gt'],f'prefix:{t}:{step["step"]}')
                vv['members'].append(dict(member=str(m['member']),metrics=score.compact_native(native)))
                requests.append(dict(unit=u['unit'],member=str(m['member']),query_ids=ids,physical=cache.calls[-1],TP_FP_FN=score.tp_table(native)))
            predicted=sum(np.asarray(m['metrics']['pred_bins']) for m in vv['members'])
            if not np.array_equal(predicted,residual[u['unit']]+3*np.asarray(u['gt_bins'])):raise ValueError('official prefix count differs predicted window update')
            videos.append(vv)
        pooled=combine(videos);expected=sum(int(np.abs(r).sum()) for r in residual.values())
        if pooled['numerator']!=expected:raise ValueError('official prefix group numerator differs')
        return dict(group_numerator=pooled['numerator'],denominator=pooled['denominator'],iou=pooled['iou'],requests=requests,
            complete_group_native_scoring=True,raw_count_closure_exact=True)
    return callback

def field(args):
    """Only call after corrected owner seal, original owner exit and root GO."""
    import beetrack.evaluate as official
    from tools.study195.run import score_member_native,_gt_provider
    from tools.study195.freeze import write_mot_exact
    from tools.study199.owners199 import prepare_gt
    from tools.study193 import source193
    base=Path(__file__).parent;freezer=load_module('return206freeze_score',base/'freeze206.py');owners=load_module('return206owners_score',base/'owners206.py')
    root=args.root;frozen_root=root/'frozen';owner_root=root/'owners';out=root/'score';out.mkdir(parents=True,exist_ok=True)
    if not args.owners_released or not args.field_go:raise ValueError('corrected owner CPU release and explicit root GO required')
    if score.sha(frozen_root/'manifest.json')!=args.frozen_seal or score.sha(owner_root/'manifest.json')!=args.owners_seal:raise ValueError('dispatch pinned seals differ')
    freezer.verify(frozen_root);owners.verify(owner_root)
    frozen=load(frozen_root/'deployment.json');owned=load(owner_root/'deployment.json')
    if frozen['prereg']!=args.prereg or owned['prereg']!=args.prereg or owned['freeze_receipt']['manifest_sha256']!=args.frozen_seal:raise ValueError('registered206 lineage differs')
    if owned['code_sha256']!=score.sha(owners.__file__):raise ValueError('corrected owner source differs sealed producer')
    if len(frozen['groups'])!=20 or len(owned['groups'])!=20 or frozen['member_clips']!=60:raise ValueError('complete old20/60 required')
    fr={r['unit']:r for r in frozen['groups']};headers=[];roster={};clips=set()
    for ref in frozen['groups']:
        header=load(frozen_root/ref['path']/'queries.json');clips.add(header['video'])
        if header['split']=='test' or any(not 0<=m['seed']<=14 for m in header['members']):raise ValueError('outside old clips/seed14 scope')
        ids=[str(m['member']) for m in header['members']];key=(header['domain'],header['group'])
        if key in roster and roster[key]!=ids:raise ValueError('historical member identity differs across clips')
        roster[key]=ids;headers.append({k:header[k] for k in ('unit','domain','group','video','nf','polygon','split')})
    if len(clips)!=6 or {h:sum(k[0]==h for k in roster) for h in DOMAINS}!=dict(zip(DOMAINS,(4,4,3,3))):raise ValueError('fixed6clip/4-4-3-3 groups required')
    truths,GT_receipts=_gt_provider(args.repo,headers,DOMAINS,'evaluation')
    if any(owned['GT_sha256'][p['path']]!=p['sha256'] for p in GT_receipts.values()):raise ValueError('owners and native GT differ')
    code=dict(official_evaluate_sha256=score.sha(official.__file__),scorer_sha256=score.sha(inspect.getsourcefile(score_member_native)),source193_sha256=score.sha(source193.__file__))
    identity=dict(prereg=args.prereg,frozen_seal=args.frozen_seal,owners_seal=args.owners_seal,source203_score_seal=args.source203_score_seal,
        GT_receipts=GT_receipts,code=code,score_code_sha256=score.sha(score.__file__),stage_code_sha256=score.sha(__file__),corrected_owner_code_sha256=score.sha(owners.__file__))
    immutable(out/'input_receipt.json',identity)
    history=historical_provider(args.source203_score,args.source203_score_seal,code)
    cache=score.NativeCache(out/'physical',score_member_native,write_mot_exact,code,historical=history)
    all_bounds={};eligible_bounds={};confirmed={t:{} for t in score.THRESHOLDS};greedies=[];scored={a:[] for a in score.ARMS};edge_audits={};light_units=[];query_bees={};selected_bees={};numeric=[]
    refs={r['unit']:r for r in owned['groups']}
    for h,g in sorted(roster):
        units=[];joint={}
        for header in [v for v in headers if v['domain']==h and v['group']==g]:
            key=header['unit'];u=merge_unit(freezer.load_group(frozen_root/fr[key]['path']),owners.load_group(owner_root,refs[key]))
            u['_actual_gt']=truths[u['video']];units.append(u)
            prepared=prepare_gt(truths[u['video']],u['polygon']);joint[key]=bind_joint(u,prepared,owners)
            all_bounds[key]=score.bound_unit(u)
            eligible_bounds[key]=score.bound_unit(u,eligible_only=True)
            for t in score.THRESHOLDS:confirmed[t][key]=score.bound_unit(u,t)
        dispatch=lambda u,ids,t:joint[u['unit']](u,ids,t)
        gre={t:score.greedy_group(units,t,dispatch,on_accept=prefix_scorer(units,cache)) for t in score.THRESHOLDS};greedies.append(dict(domain=h,group=g,thresholds=gre))
        query_bees[(h,g)]={t:count_bees(units,t) for t in score.THRESHOLDS};selected_bees[(h,g)]={t:count_bees(units,t,gre[t]['selected']) for t in score.THRESHOLDS}
        for u in units:
            key=u['unit'];selection=dict(K=[],R05=gre['0.5']['selected'][key],R03=gre['0.3']['selected'][key])
            physical=score.score_unit(u,selection,dispatch,truths[u['video']],cache)
            for m in physical['K']['members']:
                score._common().assert_K(m['metrics'],history.references[(key,str(m['member']))])
            for arm,t in (('R05','0.5'),('R03','0.3')):
                actual=sum(np.asarray(m['metrics']['pred_bins']) for m in physical[arm]['members'])
                if not np.array_equal(actual,gre[t]['witness_pred_sum'][key]):raise ValueError('official physical group bins differ greedy witness')
            for arm in score.ARMS:scored[arm].append(physical[arm])
            edge_audits[key]=physical['selected_original_event_edges']
            light_units.append({k:u[k] for k in ('unit','domain','group','video','nf','gt_bins','pred_sum')})
            numeric.append(numeric_input(u))
            print('SCORE206_UNIT',key,len(selection['R05']),len(selection['R03']),flush=True)
        del units,joint,prepared,physical;gc.collect()
    population_counts=dict(all_native_event_roster=sum(r['query_count'] for r in all_bounds.values()),eligible_deployment_queries=sum(r['query_count'] for r in eligible_bounds.values()),
        head35_all=sum(all_bounds[u['unit']]['query_count'] for u in light_units if u['domain']=='35'),head35_eligible=sum(eligible_bounds[u['unit']]['query_count'] for u in light_units if u['domain']=='35'))
    if population_counts!=dict(all_native_event_roster=4169,eligible_deployment_queries=3641,head35_all=1568,head35_eligible=1414):raise ValueError('registered three-layer query populations differ')
    bounds=dict(all_queries=score.aggregate_bounds(light_units,all_bounds),eligible_queries=score.aggregate_bounds(light_units,eligible_bounds),
        confirmed={t:score.aggregate_bounds(light_units,confirmed[t]) for t in score.THRESHOLDS},population_counts=population_counts,
        labels=dict(all_queries='All native events including formally ineligible',eligible_queries='GT-free eligible deployment queries incl unknown/no candidate',confirmed='Confirmed same-bee fixed earliest legal candidate'))
    necessary=score.necessary_edits([u for u in light_units if u['domain']=='35'])
    necessary['confirmed_queries']={t:sum(r['query_count'] for key,r in confirmed[t].items() if next(u for u in light_units if u['unit']==key)['domain']=='35') for t in score.THRESHOLDS}
    if necessary['available']:necessary['confirmed_count_below_necessary']={t:n<necessary['necessary_member_event_edits'] for t,n in necessary['confirmed_queries'].items()}
    if sum(c['arm'] in score.ARMS for c in cache.calls)!=180:raise ValueError('registered3finalarms*60 member requests missing')
    common=score._common();summary=dict(schema='return-query206-native-score-v1',input_receipt=identity,domains=score.summarize_physical(scored),bounds=bounds,
        necessary_head35=necessary,confirmed_query_vs_selected_bees=[dict(domain=h,group=g,queries=query_bees[h,g],selected=selected_bees[h,g]) for h,g in sorted(roster)],
        physical_scope=dict(final_arm_requests=180,prefix_requests=sum(c['arm'] not in score.ARMS for c in cache.calls),requested=len(cache.calls),unique_geometry=len(cache.seen),fresh_this_attempt=cache.fresh,historical_this_attempt=cache.historical_used,resumed_this_attempt=cache.resumed,
            unique_by_origin={o:len({c['path'] for c in cache.calls if c['origin']==o}) for o in ('fresh','historical')},within_job_shared=sum(c['within_job_shared'] for c in cache.calls)),
        caveat='Oracle GT-return diagnostic,not deployable model;zero net GT-run does not imply official prediction FP;nonzero legal edges do not guarantee matching.')
    immutable(out/'native_scores.json',scored);immutable(out/'bounds.json',dict(units=all_bounds,eligible=eligible_bounds,confirmed=confirmed,summary=bounds))
    immutable(out/'numeric_inputs.json',dict(input_receipt=identity,units=numeric))
    immutable(out/'greedy.json',greedies);immutable(out/'selected_original_edges.json',edge_audits);immutable(out/'physical_receipts.json',cache.calls);immutable(out/'summary.json',summary)
    print('SCORE206_SEALED',common.seal(out),flush=True)
