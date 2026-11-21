"""Explicitly gated, per-unit adapters for the independent 206 arithmetic."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import gc
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import verify206 as core
import verify206_owners as ownership

PREREG='0528579a3678f2367a514920ef625f85f9b3fa10'
SOURCE203='439a8746c19c3857fe7192990d63f08cb17c94145c8c18c6394218cc95e01b51'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def owner_array_sha(array):
    array=np.ascontiguousarray(array,dtype='<f8')
    return hashlib.sha256(json.dumps(list(array.shape),separators=(',',':')).encode()+b'\n'+array.tobytes()).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def helpers(repo):
    replay=load('independent203_for206',repo/'tools/study203/verify203.py')
    previous,u,v,cert=replay.helpers(repo)
    return replay,previous,u,v


def seal(root,expected=None):
    root=Path(root);digest=sha(root/'manifest.json');assert digest==(root/'SEALED.sha256').read_text().strip()
    if expected:assert digest==expected
    for name,want in read(root/'manifest.json').items():
        path=(root/name).resolve();assert path.is_relative_to(root.resolve()) and sha(path)==want
    return digest


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    if path.parent.name=='verification' and isinstance(value,dict) and 'scope' in value:
        value.update(verifier_sources={name:sha(Path(__file__).with_name(name)) for name in ('verify206.py','verify206_field.py','verify206_owners.py','verify206_scores.py')},
            input_seals={s:sha(path.parent.parent/s/'manifest.json') for s in ('frozen','owners','score')})
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    digest=sha(temp);temp.replace(path);path.with_suffix('.sha256').write_text(digest+'\n')


def groups(args):
    top=read(args.root/'frozen/deployment.json');assert top['prereg']==PREREG and top['source203']==SOURCE203 and len(top['groups'])==20
    assert top['marker']=='[return-query-reach-v1]' and len({r['unit'] for r in top['groups']})==20
    for ref in top['groups']:
        folder=args.root/'frozen'/ref['path'];seal(folder);meta=read(folder/'queries.json')
        with np.load(folder/'K.npz',allow_pickle=False) as z:
            members=[dict(m,rows=z['k'+str(m['member_index'])],sources=m['source_keys']) for m in meta['members']]
        yield ref,meta,members
        del meta,members
        gc.collect()


def supply(args):
    replay,_,_,_=helpers(args.repo);assert sha(args.source203/'manifest.json')==SOURCE203;reports=[];seen=set();videos=set();seedgroups=set()
    original_refs={r['unit']:r for r in read(args.source203/'deployment.json')['groups']}
    for ref,group,members in groups(args):
        assert len(members)==3 and [m['member_index'] for m in members]==[0,1,2]
        source=members[0]['source203_receipt']['group'];folder=args.source203/source['path'];seal(folder,source['seal'])
        assert source==original_refs[group['unit']];videos.add(group['video']);seedgroups.add((group['domain'],group['group']))
        original=list(replay.stream_array(folder/'catalog.json','members'))
        with np.load(folder/'K.npz',allow_pickle=False) as z:
            for slot,m in enumerate(members):
                assert np.array_equal(m['rows'],z['k'+str(slot)]) and m['sources']==original[slot]['sources']
                provenance=original[slot]['source_record'];assert provenance==m['source203_receipt']['source202_member']
                assert m['member']==str(provenance['member']) and m['seed']==provenance['seed'] and m['tag']==provenance['tag']
                assert provenance['unit']==group['unit'] and provenance['member_index']==slot and provenance['nf']==group['nf'] and provenance['polygon']==group['polygon'] and provenance['split']==group['split'] and provenance['clip']==group['clip'] and provenance['domain']==group['domain'] and provenance['video']==group['video']
                key=group['unit'],slot,m['member'];assert key not in seen;seen.add(key)
                assert 0<=m['seed']<=14 and group['split']!='test'
        counts=[]
        for slot,m in enumerate(members):
            own=core.query_roster(m['rows'],m['sources'],group['polygon'],group['nf'],group['unit'],slot)
            saved=m['roster'];assert len(own)==saved['total_native_events']==len(saved['queries'])
            for a,b in zip(own,saved['queries']):
                assert a.keys()==b.keys()
                for key in a:
                    if key in ('L','radius'):assert abs(a[key]-b[key])<=2e-12*max(1,abs(a[key]))
                    else:assert a[key]==b[key],(group['unit'],slot,key)
            assert sum(q['eligible'] for q in own)==saved['eligible'] and sum(not q['eligible'] for q in own)==saved['ineligible']
            events=[(q['event']['frame'],q['event']['direction']) for q in own]
            assert np.array_equal(core.bins(events,group['nf']),original[slot]['windows'])
            counts.append(dict(events=len(own),eligible=saved['eligible'],ineligible=saved['ineligible']))
        report=dict(passed=True,unit=group['unit'],members=counts,source203_group_seal=source['seal'])
        save(args.root/'verification/supply'/(ref['path']+'.json'),report);reports.append(report)
        print('INDEPENDENT206_SUPPLY',group['unit'],flush=True)
        del original
    assert len(seen)==60 and len(videos)==6 and {h:sum(x==h for x,g in seedgroups) for h in ('35','33','W','G')}==dict(zip(('35','33','W','G'),(4,4,3,3)))
    result=dict(passed=True,units=len(reports),members=len(seen),events=sum(m['events'] for r in reports for m in r['members']),
        eligible=sum(m['eligible'] for r in reports for m in r['members']),scope='All60 original203 K matrices/sources and every native-event query, including ineligible/empty searches, plus exact30phase baseline.')
    save(args.root/'verification/supply.json',result);return result


def compare_geometry(ours,saved,replay,v):
    for key,value in ours.items():
        if key=='members':
            assert len(value)==len(saved[key])==3
            for a,b in zip(value,saved[key]):
                for field,content in a.items():
                    if field=='threshold_neighbor_evidence':
                        assert len(content)==len(b[field])
                        for x,y in zip(content,b[field]):
                            assert set(x)-set(y)=={'pred_row'}
                            replay.compare_evidence({k:value for k,value in x.items() if k!='pred_row'},y,v)
                    else:v.compare(content,b[field])
        else:v.compare(value,saved[key])


def compare_label(own,saved,replay,v):
    for key in ('confirmed','classification','reason'):
        assert own[key]==saved[key],(key,own[key],saved[key])
    assert len(own['anchor_rows'])==len(saved['anchor_rows'])
    for a,b in zip(own['anchor_rows'],saved['anchor_rows']):replay.compare_evidence(a,b,v)
    for key in ('GT_run_key','canonical_gt_id','gt_run_first','gt_run_last','missing_internal_frames','at_clip_start','at_clip_end'):
        if key in own:v.compare(own[key],saved[key])
    if 'GT_events' in own:assert own['GT_events']==[(e['frame'],e['direction']) for e in saved['GT_events']]
    assert len(own['enumerated'])==len(saved['enumerated'])
    for a,b in zip(own['enumerated'],saved['enumerated']):
        for key in ('GT_canonical_index','GT_raw_index','frame'):assert a[key]==b[key]
        mapped={'canonical_alias_collision':'canonical_conflict','not_exactly_one_event_retraction':'not_exactly_one_original_event_deletion'}
        assert mapped.get(a['reason'],a['reason'])==b['reason'],(a['reason'],b['reason'])
        if 'geometry' in a:compare_geometry(a['geometry'],b['geometry'],replay,v)
        if 'inserted_frames' in a:
            for key,value in a['inserted_frames'].items():v.compare(value,b['inserted'][key])
    if own['confirmed']:
        for key in ('query_id','frame','GT_raw_index','GT_canonical_index','GT_row_key','added_row','source_key','deleted_event','delta_windows','strict_no_GT_internal_gaps'):
            v.compare(own['candidate'][key],saved['candidate'][key])
        assert saved['candidate']['event_delete_only'] is True
        replay.compare_evidence(own['candidate']['required_rows'][-1],saved['candidate']['inserted']['added_row_evidence'],v)


def verify_GT_ledger(path,raw,video,nf,digest,polygon,replay,v):
    ledger=read(path);assert ledger['video']==video and ledger['nf']==nf and ledger['file_sha256']==digest
    G,origin=v.canonical_with_origin(raw);originals=defaultdict(set)
    for i,row in enumerate(G):originals[int(row[1])].add(int(raw[origin[i],1]))
    segments=v.GT_segments(dict(G=G,original_ids=originals),polygon);assert len(segments)==len(ledger['runs'])
    duplicate_runs=[];noncontiguous=[]
    for (pid,seg),saved in zip(segments.items(),ledger['runs']):
        ii=np.flatnonzero(G[:,1]==pid);rows=G[ii];sides=[ownership.side(r,polygon) for r in rows];dates=list(map(int,rows[:,0]))
        expected=dict(video=video,canonical_gt_id=pid,original_gt_id=seg['original_gt_id'],run=seg.get('run'),rows=rows.tolist(),canonical_indices=ii.tolist(),source_raw_indices=origin[ii].tolist(),sides=sides,
            probe=replay.first_return(dates,sides),at_clip_start=dates[0]==1,at_clip_end=dates[-1]==nf,missing_internal_frames=dates[-1]-dates[0]+1-len(rows),span_frames=dates[-1]-dates[0]+1,
            positive_extents=bool((rows[:,4:6]>0).all()),original_unambiguous=len(originals[pid])==1)
        for key,value in expected.items():v.compare(value,saved[key])
        assert replay.track_events(rows,polygon)==[(e['frame'],e['direction']) for e in saved['events']]
        if len(set(dates))!=len(dates):duplicate_runs.append(pid)
        if not np.all(np.diff(rows[:,0])==1):noncontiguous.append(pid)
    unique,counts=np.unique(raw[:,:2],axis=0,return_counts=True)
    return dict(file_sha256=digest,rows=len(raw),runs=len(segments),duplicate_original_date_ID=[dict(frame=int(k[0]),original_ID=int(k[1]),rows=int(n)) for k,n in zip(unique,counts) if n>1],
        duplicate_date_canonical_runs=duplicate_runs,noncontiguous_canonical_runs=noncontiguous,nonpositive_raw_indices=np.flatnonzero((raw[:,4:6]<=0).any(1)).tolist(),
        ambiguous_original_ID_canonical_runs=[pid for pid,s in segments.items() if s['original_gt_id'] is None])


def owners(args):
    from beetrack.evaluate import load_tracks
    replay,previous,_,v=helpers(args.repo);top=read(args.root/'owners/deployment.json')
    assert top['prereg']==PREREG and len(top['groups'])==20
    references={r['unit']:r for r in top['groups']};clips={};GT={};quality={};reports=[];specimens=defaultdict(set);probe_diagnostics=[]
    for ref,group,members in groups(args):
        source=references[group['unit']];path=args.root/'owners'/source['path'];assert sha(path)==source['sha256']
        video=group['video'];GTpath=args.repo/'runs/gt'/group['split']/video/'gt/gt.txt';digest=sha(GTpath)
        if video not in GT:
            GT[video]=load_tracks(GTpath);universe,_,_=replay.GT_universe(GT[video],group['polygon'],v,digest,GTpath.relative_to(args.repo).as_posix());clips[video]=universe
            quality[video]=verify_GT_ledger(args.root/'owners'/('GT_'+video+'.json'),GT[video],video,group['nf'],digest,group['polygon'],replay,v)
        destination=args.root/'verification/owners'/(ref['path']+'.json');compact_path=args.root/'verification/owner_compact'/(ref['path']+'.json');probe_path=args.root/'verification/probe_units'/(ref['path']+'.json')
        checkpoint_identity=dict(owner_unit_sha256=source['sha256'],frozen_group_sha256=sha(args.root/'frozen'/ref['path']/'manifest.json'),GT_file_sha256=digest,
            sources={name:sha(Path(__file__).with_name(name)) for name in ('verify206.py','verify206_field.py','verify206_owners.py')},public203_sha256=sha(args.repo/'tools/study203/verify203.py'))
        if all(p.exists() and p.with_suffix('.sha256').exists() for p in (destination,compact_path,probe_path)):
            for p in (destination,compact_path,probe_path):assert sha(p)==p.with_suffix('.sha256').read_text().strip()
            report=read(destination);assert report['identity']==checkpoint_identity and report['passed'];reports.append(report);probe_diagnostics.extend(read(probe_path))
            for t,values in report['confirmed_specimens_by_threshold'].items():specimens[group['domain'],t].update(tuple(x) for x in values)
            continue
        saved=read(path);assert saved['prereg']==PREREG and saved['unit']==group['unit'] and len(saved['members'])==3
        assert saved['GT_sha256']=={GTpath.relative_to(args.repo).as_posix():digest}
        assert saved['source_universe203_sha256']=='47e684d5117b7e16c8eba9176dc62ad8d237998c559fab5e45652f47acdc20ac'
        gt=GT[video];raws=[m['rows'] for m in members];sources=[m['sources'] for m in members];records=[];counts=Counter()
        assert np.array_equal(saved['gt_bins'],core.bins([(f,s) for _,_,_,_,f,s in core.signed_events(gt,group['polygon'])],group['nf']))
        assert np.array_equal(saved['pred_sum'],sum(core.bins([(f,s) for _,_,_,_,f,s in core.signed_events(raw,group['polygon'])],group['nf']) for raw in raws))
        for slot,(member,published) in enumerate(zip(members,saved['members'])):
            assert published['member']==member['member'] and published['member_index']==slot and published['seed']==member['seed']
            assert published['original_K_sha256']==owner_array_sha(raws[slot])
            assert published['source203_receipt']==member['source203_receipt']
            queries=core.query_roster(raws[slot],sources[slot],group['polygon'],group['nf'],group['unit'],slot)
            assert len(queries)==len(published['queries'])
            for query,q in zip(queries,published['queries']):
                assert query['query_id']==q['query_id'];labels={}
                for threshold in (.5,.3):
                    own=ownership.analyze(query,raws,sources,gt,group['polygon'],group['nf'],video,digest,threshold,replay,previous,v)
                    compare_label(own,q['labels'][str(threshold)],replay,v);labels[str(threshold)]=own;counts[str(threshold)+':'+own['classification']]+=1
                    if own['confirmed']:specimens[(group['domain'],str(threshold))].add((video,own['canonical_gt_id']))
                records.append(dict(query,labels=labels))
        coverage=ownership.coverage(records,raws,sources,gt,group['polygon'],group['nf'],video,replay,previous,v)
        profiles={r['canonical_gt_id']:r for r in clips[video]['runs']}
        supplied=read(args.root/'owners'/source['coverage_path']);assert supplied['unit']==group['unit'] and len(coverage)==len(supplied['runs'])
        for a,b in zip(coverage,supplied['runs']):
            for key in ('canonical_gt_id','original_gt_id','gt_run','probe'):v.compare(a[key],b[key])
            assert a['thresholds'].keys()==b['thresholds'].keys()
            for t,proof in a['thresholds'].items():
                for key,value in proof.items():
                    if key=='geometry':compare_geometry(value,b['thresholds'][t][key],replay,v)
                    else:v.compare(value,b['thresholds'][t][key])
            if a['probe'] is not None:
                for t,proof in a['thresholds'].items():
                    known=proof['geometry']['all_geometry_known'];missing=proof['geometry']['all_no_threshold_box'];pid=a['canonical_gt_id'];date=a['probe']['return_frame']
                    profile=profiles[pid];observations=profile['observations'];dates=[o['frame'] for o in observations]
                    eligible=proof['eligible_query_time_space_ids'];anchors=proof['same_run_anchor_query_ids'];exact=proof['exact_probe_first_candidate_query_ids'];same_date=[];later_unverified=[]
                    for q in records:
                        label=q['labels'][t]
                        if label.get('canonical_gt_id')!=pid or not label['confirmed']:continue
                        if label['candidate']['frame']==date:same_date.append(q['query_id'])
                        if anchors and q['query_id'] in anchors and label['candidate']['frame']<date:later_unverified.append(q['query_id'])
                    ladder='unknown_geometry' if not known else 'not_jointly_missing' if not missing else 'no_eligible_query_time_space' if not eligible else 'covered_without_same_run_anchors' if not anchors else 'same_run_anchors_without_exact_first_legal_row' if not exact else 'exact_first_legal_probe_row'
                    probe_diagnostics.append(dict(unit=group['unit'],domain=group['domain'],video=video,canonical_gt_id=pid,original_gt_id=a['original_gt_id'],GT_run=a['gt_run'],threshold=t,
                        probe_return_frame=date,geometry_known=known,all_no_threshold_box=missing,all_scene_frames_empty=proof['geometry']['all_scene_frames_empty'],
                        observed_GT_zero_net=observations[0]['inside']==observations[-1]['inside'],GT_clip_truncated=dates[0]==1 or dates[-1]==group['nf'],
                        GT_all_positive=all(o['positive_extent'] for o in observations),GT_missing_unique_dates=dates[-1]-dates[0]+1-len(set(dates)),GT_duplicate_dates=len(dates)-len(set(dates)),
                        all_event_time_space_query_ids=proof['all_query_time_space_ids'],eligible_time_space_query_ids=eligible,same_run_anchor_query_ids=anchors,exact_first_legal_probe_query_ids=exact,first_legal_same_date_query_ids=same_date,
                        some_first_legal_return_same_run_query_ids=proof['same_run_confirmed_query_ids'],later_than_first_unverified_probe_query_ids=later_unverified,ladder=ladder,
                        caveat='Later-than-first probe rows have no standalone legal insertion certificate in the frozen first-return roster; not promoted to legal.'))
        report=dict(passed=True,identity=checkpoint_identity,unit=group['unit'],queries=len(records),classes=dict(counts),all_GT_runs=len(coverage),probes=sum(c['probe'] is not None for c in coverage),
            confirmed_specimens_by_threshold={t:[list(x) for x in sorted({(video,q['labels'][t]['canonical_gt_id']) for q in records if q['labels'][t]['confirmed']})] for t in ('0.5','0.3')})
        compact=dict(unit=group['unit'],domain=group['domain'],group=group['group'],video=video,nf=group['nf'],gt_bins=saved['gt_bins'],pred_sum=saved['pred_sum'],members=[])
        for slot,member in enumerate(members):
            qs=[]
            for q in records:
                if q['member_index']!=slot:continue
                labels={}
                for t,label in q['labels'].items():
                    z={k:label[k] for k in ('confirmed','classification','reason')}
                    fields=('source_key','pred_key','row_sha256','official_owner','original_gt_id','gt_run','gt_key','gt_row_sha256','gt_raw_index','iou','pred_degree','gt_degree','isolated','qualified','pred_inside','gt_inside')
                    z['anchors']=[{k:r[k] for k in fields} for r in label['anchor_rows']]
                    for k in ('canonical_gt_id','GT_run_key','gt_run_first','gt_run_last','GT_events','missing_internal_frames','at_clip_start','at_clip_end'):
                        if k in label:z[k]=label[k]
                    z['candidate']=None if not label['confirmed'] else {k:label['candidate'][k] for k in ('query_id','frame','GT_raw_index','GT_canonical_index','GT_row_key','added_row','source_key','deleted_event','delta_windows','strict_no_GT_internal_gaps')}
                    if label['confirmed']:z['candidate']['added_annotation']={k:label['candidate']['required_rows'][-1][k] for k in fields}
                    z['enumerated']=[{k:r[k] for k in ('GT_canonical_index','GT_raw_index','frame','reason')} for r in label['enumerated']]
                    labels[t]=z
                qs.append(dict(query_id=q['query_id'],member_index=slot,native_id=q['native_id'],run_index=q['run_index'],canonical_id=q['canonical_id'],event=q['event'],eligible=q['eligible'],ineligible_reasons=q['ineligible_reasons'],labels=labels))
            compact['members'].append(dict(member=member['member'],member_index=slot,queries=qs))
        save(compact_path,compact);save(probe_path,[p for p in probe_diagnostics if p['unit']==group['unit']]);save(destination,report);reports.append(report)
        print('INDEPENDENT206_OWNERS',group['unit'],flush=True)
        del saved,records,coverage,supplied
        ownership._SCENES.clear()
        gc.collect()
    universe=read(args.root/'owners/gt_run_universe.json');assert universe['clips']==clips
    probe_summary={}
    for hive in ('35','33','W','G'):
        probe_summary[hive]={}
        for t in ('0.5','0.3'):
            rows=[r for r in probe_diagnostics if r['domain']==hive and r['threshold']==t];missing=[r for r in rows if r['all_no_threshold_box']];known=[r for r in rows if r['geometry_known']]
            probe_summary[hive][t]=dict(probe_opportunities=len(rows),known_geometry=len(known),jointly_missing=len(missing),
                jointly_missing_eligible_time_space=sum(bool(r['eligible_time_space_query_ids']) for r in missing),jointly_missing_same_run_anchors=sum(bool(r['same_run_anchor_query_ids']) for r in missing),
                jointly_missing_exact_first_legal_probe_row=sum(bool(r['exact_first_legal_probe_query_ids']) for r in missing),jointly_missing_first_legal_same_date=sum(bool(r['first_legal_same_date_query_ids']) for r in missing),
                jointly_missing_no_eligible_time_space=sum(not r['eligible_time_space_query_ids'] for r in missing),jointly_missing_non_earliest_not_certified=sum(bool(r['later_than_first_unverified_probe_query_ids']) for r in missing),
                jointly_missing_observed_GT_zero_net=sum(r['observed_GT_zero_net'] for r in missing),jointly_missing_GT_clip_truncated=sum(r['GT_clip_truncated'] for r in missing),jointly_missing_nonpositive_GT=sum(not r['GT_all_positive'] for r in missing),
                distinct_annotated_runs_all=len({(r['video'],r['canonical_gt_id']) for r in rows}),distinct_annotated_runs_with_any_jointly_missing_opportunity=len({(r['video'],r['canonical_gt_id']) for r in missing}),ladder=dict(Counter(r['ladder'] for r in missing)))
    save(args.root/'verification/probe_coverage.json',dict(passed=True,summary=probe_summary,opportunities=probe_diagnostics,source_GT_universe_sha256=sha(args.root/'owners/gt_run_universe.json'),scope='Existing complete GT-probe denominator; opportunities and deduplicated (clip,canonical GT-run) counts separated. Fixed first-row legality distinguished from later untested returns; no extra matching.'))
    result=dict(passed=True,units=len(reports),queries=sum(r['queries'] for r in reports),confirmed_specimens={h+':'+t:[list(x) for x in sorted(values)] for (h,t),values in specimens.items()},
        GT_universe_sha256=sha(args.root/'owners/gt_run_universe.json'),GT_quality=quality,probe_coverage_sha256=sha(args.root/'verification/probe_coverage.json'),probe_summary=probe_summary,scope='All queries/both thresholds: full original-scene geometry/ownership, observed same-GT-run and unknown taxonomy, earliest fixed legal real row, full six-clip GT-run/probe denominator and zero coverage; unrelated Hungarian owner changes diagnostic only.')
    save(args.root/'verification/owners.json',result);return result


def load_frozen(args,reference):
    folder=args.root/'frozen'/reference['path'];meta=read(folder/'queries.json')
    with np.load(folder/'K.npz',allow_pickle=False) as z:
        members=[dict(m,rows=z['k'+str(m['member_index'])],sources=m['source_keys']) for m in meta['members']]
    return meta,members


def counts(args):
    from beetrack.evaluate import load_tracks
    replay,previous,_,v=helpers(args.repo);refs=read(args.root/'frozen/deployment.json')['groups'];byunit={r['unit']:r for r in refs}
    units={r['unit']:read(args.root/'verification/owner_compact'/(r['path']+'.json')) for r in refs}
    saved=read(args.root/'score/bounds.json');answer={'all_queries':{},'eligible_queries':{},'0.5':{},'0.3':{}}
    for key,u in units.items():
        queries=[q for m in u['members'] for q in m['queries']]
        for t in answer:
            selected=queries if t=='all_queries' else [q for q in queries if q['eligible']] if t=='eligible_queries' else [q for q in queries if q['labels'][t]['confirmed']]
            actual=core.loose_bound(u['pred_sum'],u['gt_bins'],[(q['event']['frame'],q['event']['direction']) for q in selected],u['nf'])
            target=saved['units'][key] if t=='all_queries' else saved['eligible'][key] if t=='eligible_queries' else saved['confirmed'][t][key]
            for a,b in (('baseline_numerator','baseline_numerator'),('upper_improvement','gain_upper_numerator'),('denominator','denominator'),('capacity','capacity')):assert actual[a]==target[b]
            assert int(np.asarray(actual['capacity']).sum())==30*len(selected)
            assert target['lower_numerator']==actual['baseline_numerator']-actual['upper_improvement'] and target['query_ids']==[q['query_id'] for q in selected]
            assert target['confirmed_GT_runs'] is None if t in ('all_queries','eligible_queries') else target['confirmed_GT_runs']==len({tuple(q['labels'][t]['GT_run_key']) for q in selected})
            assert target['query_count']==len(selected);answer[t][key]=actual
    heads={}
    for t,reports in answer.items():
        heads[t]={}
        for hive in ('35','33','W','G'):
            ratios=[]
            for group in sorted({u['group'] for u in units.values() if u['domain']==hive}):
                rows=[reports[k] for k,u in units.items() if u['domain']==hive and u['group']==group]
                ratios.append(Fraction(100*sum(r['upper_improvement'] for r in rows),sum(r['denominator'] for r in rows)))
            mean=sum(ratios,Fraction())/len(ratios);published=saved['summary'][t][hive] if t in ('all_queries','eligible_queries') else saved['summary']['confirmed'][t][hive]
            assert Fraction(*published['fraction'])==mean;heads[t][hive]=[mean.numerator,mean.denominator]
    greedy=read(args.root/'score/greedy.json');GT={};checks=[]
    for report in greedy:
        groupunits={key:u for key,u in units.items() if u['domain']==report['domain'] and u['group']==report['group']};data={}
        for key in groupunits:
            meta,members=load_frozen(args,byunit[key]);data[key]=(meta,members)
            if meta['video'] not in GT:GT[meta['video']]=load_tracks(args.repo/'runs/gt'/meta['split']/meta['video']/'gt/gt.txt')
        lengths={key:np.asarray(u['gt_bins']).size for key,u in groupunits.items()};starts={};clock=0
        for key in groupunits:starts[key]=clock;clock+=lengths[key]
        residual=np.concatenate([(np.asarray(u['pred_sum'],np.int64)-3*np.asarray(u['gt_bins'],np.int64)).ravel() for u in groupunits.values()])
        for t in ('0.5','0.3'):
            options={};querymap={}
            for key,u in groupunits.items():
                for m in u['members']:
                    fullqueries={q['query_id']:q for q in data[key][1][m['member_index']]['roster']['queries']}
                    for q in m['queries']:
                        if not q['labels'][t]['confirmed']:continue
                        column=np.zeros(clock,np.int64);column[starts[key]:starts[key]+lengths[key]]=core.bins([(q['event']['frame'],q['event']['direction'])],u['nf']).ravel()
                        options[q['query_id']]=column;querymap[q['query_id']]=(key,m['member_index'],fullqueries[q['query_id']],q['labels'][t])
            def legal(ids):
                for key,(meta,members) in data.items():
                    for slot,m in enumerate(members):
                        selected=[dict(query=q,candidate=label['candidate'],canonical_gt_id=label['canonical_gt_id']) for queryid in ids for uk,si,q,label in [querymap[queryid]] if uk==key and si==slot]
                        if not selected:continue
                        audit=ownership.joint(m['rows'],m['sources'],selected,GT[meta['video']],meta['polygon'],meta['nf'],float(t),replay,previous,v)
                        if not audit['safe']:return False
                return True
            actual=core.greedy_trace(residual,options,legal);target=report['thresholds'][t]
            assert actual['selected']==[step['query_id'] for step in target['steps']]
            assert actual['numerator']==target['final_numerator'] and int(abs(residual).sum())==target['baseline_numerator']
            assert [(r['query_id'],r['proposed_gain']) for r in actual['rejected']]==[(r['query_id'],r['proposed_gain']) for r in target['rejected']]
            assert actual['remaining_nonpositive']==target['remaining_nonpositive']
            work=residual.copy()
            for a,b in zip(actual['steps'],target['steps']):
                key=querymap[a['query_id']][0];part=slice(starts[key],starts[key]+lengths[key]);before=int(abs(work[part]).sum());work-=options[a['query_id']]
                assert b['unit']==key and b['gain']==a['integer_gain'] and b['before']==before and b['after']==int(abs(work[part]).sum())
                assert b['fixed_return_frame']==querymap[a['query_id']][3]['candidate']['frame']
                assert b['gt_row_key']==querymap[a['query_id']][3]['candidate']['GT_row_key']
            for key,u in groupunits.items():
                assert sorted(q for q in actual['selected'] if querymap[q][0]==key)==target['selected'][key]
                expected=(work[starts[key]:starts[key]+lengths[key]].reshape(-1,2)+3*np.asarray(u['gt_bins'],np.int64)).tolist()
                assert expected==target['witness_pred_sum'][key]
            checks.append(dict(domain=report['domain'],group=report['group'],threshold=t,selected=len(actual['selected']),rejected=len(actual['rejected']),numerator=actual['numerator']))
        del data
        gc.collect()
    denominators=[3*int(np.asarray(u['gt_bins']).sum()) for u in units.values() if u['domain']=='35'];necessary=read(args.root/'score/summary.json')['necessary_head35']
    if len(set(denominators))==1:
        required=(sum(denominators)+99)//100;edits=(required+29)//30
        assert necessary['required_integer_gain']==required and necessary['necessary_member_event_edits']==edits and necessary['pooled_denominator']==sum(denominators)
    result=dict(passed=True,head_bound_fractions=heads,greedy=checks,necessary_head35=necessary,scope='All query-inclusive and confirmed per-unit/pooled Fraction bounds; both-threshold exact highest-positive integer greedy steps/rejections, fixed first rows, joint registered qualification and full witness count vectors. Bounds are optimistic, not attainable certificates.')
    save(args.root/'verification/counts.json',result);return result
