"""Source-only206 original-event ownership and GT return certificate core.

No field access on import. The complete GT ledger includes every run and is
checked against the sealed203 universe before any206 coverage is interpreted.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import os
import sys
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.dont_write_bytecode=True
import importlib.util
from pathlib import Path
from collections import Counter
import hashlib
import json
import argparse
import subprocess
import tempfile
import gc
import numpy as np
from beetrack.evaluate import inside,iou_matrix,frame_match
from tools.study199.owners199 import prepare_gt,event_times,array_sha,by_date


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


BASE=Path(__file__).parent
prior=module('return203owners',Path.cwd()/'tools/study203/owners203.py')
queries=module('return206queries',BASE/'queries.py')
UNIVERSE_SHA='47e684d5117b7e16c8eba9176dc62ad8d237998c559fab5e45652f47acdc20ac'


def first_return(frames,sides):
    cross=np.flatnonzero(np.asarray(sides)[1:]!=np.asarray(sides)[:-1])+1
    for j,r in zip(cross[:-1],cross[1:]):
        if sides[j]!=sides[0] and sides[r]==sides[0] and frames[r]-frames[j]<=25:
            return dict(departure_index=int(j),return_index=int(r),departure_frame=frames[j],return_frame=frames[r],initial_inside=sides[0])
    return None


def ledger(gt,video,nf,file_sha,relative):
    """Exact203 public-universe schema plus a separate full numeric206 ledger."""
    runs=[];numeric=[];G=gt['rows']
    for pid,seg in gt['segments'].items():
        ix=list(range(seg['start'],seg['end']));rows=G[ix];sides=list(map(bool,inside(rows,gt['polygon'])))
        dates=list(map(int,rows[:,0]));originals=sorted(set(int(gt['raw'][gt['source_index'][i],1]) for i in ix))
        observations=[dict(frame=f,inside=s,positive_extent=bool(G[i,4]>0 and G[i,5]>0),source_raw_index=int(gt['source_index'][i]),row_sha256=array_sha(G[i])) for i,f,s in zip(ix,dates,sides)]
        probe=first_return(dates,sides)
        runs.append(dict(canonical_gt_id=pid,original_gt_id=seg['original_gt_id'],original_ids=originals,run=seg.get('run'),observations=observations,eligible_probe_or_null=probe))
        numeric.append(dict(video=video,canonical_gt_id=pid,original_gt_id=seg['original_gt_id'],run=seg.get('run'),
            rows=rows.tolist(),canonical_indices=ix,source_raw_indices=[int(gt['source_index'][i]) for i in ix],sides=sides,
            events=event_times(rows,gt['polygon']),probe=probe,at_clip_start=dates[0]==1,at_clip_end=dates[-1]==nf,
            missing_internal_frames=int(dates[-1]-dates[0]+1-len(dates)),span_frames=dates[-1]-dates[0]+1,
            positive_extents=bool((rows[:,4:6]>0).all()),original_unambiguous=len(originals)==1))
    old=dict(GT_file_sha256=file_sha,GT_relative_path=relative,polygon=gt['polygon'],canonical_rows_sha256=array_sha(G),runs=runs)
    return old,numeric


def geometry(gt,scenes,gt_index,threshold):
    """All three complete original scenes, never just a receiver corridor."""
    cache=gt.setdefault('geometry_cache',{});cachekey=(tuple(map(id,scenes)),int(gt_index),float(threshold))
    if cachekey in cache:return cache[cachekey]
    G=gt['rows'];row=G[gt_index];date=int(row[0]);records=[]
    valid=bool(np.isfinite(row).all() and (row[4:6]>0).all())
    xy=lambda a:np.c_[a[:,2:4],a[:,2:4]+a[:,4:6]]
    for slot,scene in enumerate(scenes):
        ix=scene['frames'].get(date,np.empty(0,int));P=scene['rows'][ix]
        positive=bool(np.isfinite(P).all() and (P[:,4:6]>0).all())
        values=iou_matrix(xy(row[None]),xy(P))[0] if len(P) and valid and positive else np.zeros(len(P))
        hits=ix[np.flatnonzero(values>=threshold)]
        evidence=prior._old.isolated_owners(gt,scene,list(map(int,hits)),threshold) if len(hits) else {}
        records.append(dict(member_index=slot,scene_frame_rows=len(P),all_geometry_known=valid and positive,
            max_iou=float(values.max()) if len(values) else None,has_threshold_box=bool(len(hits)),
            confirmed_same_run=any(r['qualified'] and r['official_owner']==int(row[1]) for r in evidence.values()),
            frame_rows=[dict(canonical_row=int(i),source_key=scene['source_keys'][i],row=scene['rows'][i].tolist(),iou=float(v)) for i,v in zip(ix,values)],
            threshold_neighbor_evidence=list(evidence.values())))
    result=dict(frame=date,GT_canonical_index=int(gt_index),GT_raw_index=int(gt['source_index'][gt_index]),
        GT_row=row.tolist(),members=records,all_geometry_known=all(r['all_geometry_known'] for r in records),
        geometric_members=sum(r['has_threshold_box'] for r in records),confirmed_members=sum(r['confirmed_same_run'] for r in records),
        all_scene_frames_empty=all(r['scene_frame_rows']==0 for r in records),
        all_no_threshold_box=all(r['all_geometry_known'] and not r['has_threshold_box'] for r in records))
    cache[cachekey]=result
    return result


def single_insertion(gt,scene,query,newrow,newsource,threshold):
    """One candidate/date scene; every original box at that date participates."""
    row=np.asarray(newrow,float).copy();date=int(row[0]);oldix=scene['frames'].get(date,np.empty(0,int))
    original=scene['rows'][oldix];row[1]=query['canonical_id']
    block=np.r_[original,row[None]];order=np.argsort(block[:,1],kind='stable');P=block[order]
    keys=[scene['source_keys'][i] for i in oldix]+[tuple(newsource)];keys=[keys[i] for i in order]
    added=int(np.flatnonzero(order==len(original))[0]);compact=dict(rows=P,source_keys=keys)
    proof=prior._old.isolated_owners(gt,compact,[added],threshold)[added]
    GG=gt['rows'][gt['frames'].get(date,np.empty(0,int))]
    before=frame_match(GG,original,thr=threshold);after=frame_match(GG,P,thr=threshold)
    mapped={int(rawpos):int(after[i]) for i,rawpos in enumerate(order) if rawpos<len(original)}
    unchanged=all(mapped[i]==int(owner) for i,owner in enumerate(before))
    return dict(added_row_evidence=dict(proof,pred_row=P[added].tolist()),other_original_owners_unchanged=unchanged,
        original_frame_rows=original.tolist(),inserted_frame_rows=P.tolist(),original_owners=before.tolist(),inserted_owners=after.tolist(),
        order_from_original_plus_added=order.tolist(),source_keys=keys)


def analyze_query(gt,scenes,raws,sources,query,threshold,nf,video,gt_sha):
    """Keep every query and every rejection, then choose the first legal row."""
    slot=query['member_index'];scene=scenes[slot];anchors=prior.owners(gt,scene,query['anchor_raw_indices'],threshold)
    out=dict(query_id=query['query_id'],threshold=threshold,eligible=query['eligible'],anchor_rows=anchors,
             confirmed=False,candidate=None,enumerated=[],classification='unknown',reason=None)
    if not query['eligible']:out.update(classification='ineligible',reason=';'.join(query['ineligible_reasons']));return out
    anchorclass=prior.classify(anchors)
    if anchorclass['classification']!='S':out['reason']='anchors_'+anchorclass['reason'];return out
    pid=anchors[0]['official_owner'];seg=gt['segments'][pid];R=gt['rows'][seg['start']:seg['end']]
    out.update(GT_run_key=[video,seg['original_gt_id'],seg.get('run')],canonical_gt_id=pid,
        gt_run_first=seg['first_frame'],gt_run_last=seg['last_frame'],GT_events=event_times(R,gt['polygon']),
        missing_internal_frames=int(R[-1,0]-R[0,0]+1-len(R)),at_clip_start=seg['first_frame']==1,at_clip_end=seg['last_frame']==nf)
    if any(a['pred_inside']!=a['gt_inside'] for a in anchors):out['reason']='anchor_side_disagreement';return out
    if seg['first_frame']==1 or seg['last_frame']==nf:out['reason']='GT_run_clip_truncated';return out
    if not (R[:,4:6]>0).all():out['reason']='GT_nonpositive_geometry';return out
    if out['GT_events']:out.update(classification='interpretable_nonzero_GT_event',reason='complete_observed_GT_run_has_net_event');return out
    returns=[i for i in range(seg['start'],seg['end']) if gt['rows'][i,0]>query['endpoint_frame'] and bool(inside(gt['rows'][i:i+1],gt['polygon'])[0])==query['starting_side']]
    returns.sort(key=lambda i:(gt['rows'][i,0],int(gt['source_index'][i])))
    if not returns:out.update(classification='interpretable_no_return',reason='no_later_same_run_start_side_observation');return out
    out.update(classification='return_evidence_no_legal_candidate',reason='all_return_rows_rejected')
    if 'events' not in scene:scene['events']=event_times(raws[slot],gt['polygon'])
    oldevents=scene['events']
    expected=[e for e in oldevents if not(e['track_id']==query['canonical_id'] and e['frame']==query['event']['frame'] and e['direction']==query['event']['direction'])]
    for index in returns:
        G=gt['rows'][index];rawindex=int(gt['source_index'][index]);record=dict(GT_canonical_index=index,GT_raw_index=rawindex,frame=int(G[0]),reason=None)
        out['enumerated'].append(record)
        if not queries.in_corridor(query,G[0],G[2:6]):record['reason']='outside_fixed_time_or_disk';continue
        geo=geometry(gt,scenes,index,threshold);record['geometry']=geo
        if not geo['all_geometry_known']:record['reason']='unknown_scene_geometry';continue
        if not geo['all_no_threshold_box']:record['reason']='one_or_more_members_has_threshold_box';continue
        new=np.r_[G[:6],1.];new[1]=query['native_id'];key=['GT:'+gt_sha,rawindex]
        safety=queries.insertion_safety(raws[slot],sources[slot],query,new,key,nf);record['native_safety']=safety
        if not safety['safe']:record['reason']=safety['reason'];continue
        inserted=single_insertion(gt,scene,query,new,key,threshold);record['inserted']=inserted
        p=inserted['added_row_evidence']
        if not p['qualified'] or p['official_owner']!=pid:record['reason']='inserted_row_not_isolated_same_run';continue
        # Unrelated Hungarian owner changes are descriptive, not an eligibility
        # gate. Full selected outputs must be rematched for event F1 downstream.
        newevents=event_times(np.r_[raws[slot],new[None]],gt['polygon'])
        if newevents!=expected:record['reason']='not_exactly_one_original_event_deletion';continue
        record['reason']='first_legal_confirmed_return'
        out.update(confirmed=True,classification='confirmed_legal_return',reason=record['reason'],candidate=dict(
            query_id=query['query_id'],frame=int(G[0]),GT_raw_index=rawindex,GT_canonical_index=index,
            GT_row_key=[video,seg['original_gt_id'],seg.get('run'),int(G[0]),rawindex],added_row=new.tolist(),source_key=key,
            native_safety=safety,geometry=geo,inserted=inserted,deleted_event=query['event'],event_delete_only=True,
            strict_no_GT_internal_gaps=out['missing_internal_frames']==0))
        break
    return out


def joint_validate(gt,raw,sources,selected,threshold,nf):
    """Actual combined configuration; never substitute singleton ownership."""
    raw=np.asarray(raw,float);reasons=[];base=queries.prepare(raw,sources,nf)
    keys=[tuple(c['candidate']['source_key']) for c in selected]
    if len(keys)!=len(set(keys)):return dict(valid=False,reasons=['same_physical_GT_row_reused_within_member'])
    additions=np.asarray([c['candidate']['added_row'] for c in selected],float).reshape(-1,7)
    try:combined=queries.prepare(np.r_[raw,additions],list(sources)+keys,nf)
    except ValueError as exc:return dict(valid=False,reasons=['combined_native_conflict'],detail=str(exc))
    for i,alias in base['raw_to_canonical'].items():
        if combined['raw_to_canonical'][i]!=alias:reasons.append('combined_changes_original_run_alias')
    scene=prior.scene(combined['raw'],combined['sources']);needed=[]
    for i,item in enumerate(selected):
        q=item['query'];needed.extend(q['anchor_raw_indices']);needed.append(len(raw)+i)
        if combined['raw_to_canonical'][len(raw)+i]!=q['canonical_id']:reasons.append('combined_added_row_wrong_run')
    proofs=prior.owners(gt,scene,needed,threshold);cursor=0
    for item in selected:
        n=len(item['query']['anchor_raw_indices'])+1;rr=proofs[cursor:cursor+n];cursor+=n
        pid=item['candidate']['inserted']['added_row_evidence']['official_owner']
        if not all(r['qualified'] and r['official_owner']==pid and r['pred_inside']==r['gt_inside'] for r in rr):
            reasons.append('combined_anchor_or_added_owner_not_unique_same_run')
    oldscene=prior.scene(raw,sources);changed_dates=set(map(int,additions[:,0]))
    frameproof=[]
    for date in sorted(changed_dates):
        a=oldscene['frames'].get(date,np.empty(0,int));b=scene['frames'].get(date,np.empty(0,int));gg=gt['frames'].get(date,np.empty(0,int))
        before=frame_match(gt['rows'][gg],oldscene['rows'][a],thr=threshold)
        after=frame_match(gt['rows'][gg],scene['rows'][b],thr=threshold)
        current={tuple(scene['source_keys'][i]):int(owner) for i,owner in zip(b,after)}
        unchanged=all(current[tuple(oldscene['source_keys'][i])]==int(owner) for i,owner in zip(a,before))
        frameproof.append(dict(frame=date,original_owners=before.tolist(),combined_owners=after.tolist(),
            other_original_owners_unchanged=unchanged,
            original_source_keys=[oldscene['source_keys'][i] for i in a],combined_source_keys=[scene['source_keys'][i] for i in b]))
    removed={(x['query']['canonical_id'],x['query']['event']['frame'],x['query']['event']['direction']) for x in selected}
    expected=[e for e in event_times(raw,gt['polygon']) if (e['track_id'],e['frame'],e['direction']) not in removed]
    if event_times(combined['raw'],gt['polygon'])!=expected:reasons.append('combined_not_exact_selected_event_deletions')
    return dict(valid=not reasons,reasons=sorted(set(reasons)),selected_row_proofs=proofs,frame_owner_proofs=frameproof,
                rows_sha256=array_sha(combined['raw']),original_prefix_exact=bool(np.array_equal(combined['raw'][:len(raw)],raw)))


def analyze_group(group,gt,gt_sha):
    from tools.study197.candidates import window_counts
    members=group['members'];raws=[np.asarray(m['rows'],float) for m in members];sources=[m['sources'] for m in members]
    scenes=[prior.scene(r,s) for r,s in zip(raws,sources)];gt['geometry_cache']={}
    result={k:group[k] for k in ('unit','domain','group','video','clip','split','nf','polygon')}
    result.update(members=[],gt_bins=window_counts(event_times(gt['rows'],gt['polygon']),group['nf']).tolist(),
        pred_sum=np.sum([window_counts(event_times(r,gt['polygon']),group['nf']) for r in raws],axis=0).tolist())
    for slot,m in enumerate(members):
        roster=m['roster']['queries'];needed=sorted({i for q in roster for i in q['anchor_raw_indices']})
        for t in (.5,.3):prior.owners(gt,scenes[slot],needed,t)
        qs=[]
        for q in roster:
            labels={str(t):analyze_query(gt,scenes,raws,sources,q,t,group['nf'],group['video'],gt_sha) for t in (.5,.3)}
            for record in labels.values():
                if record['candidate']:
                    record['candidate']['delta_windows']=(-window_counts([q['event']],group['nf'])).tolist()
            qs.append(dict(q,labels=labels))
        result['members'].append(dict(member=m['member'],member_index=slot,seed=m['seed'],tag=m['tag'],queries=qs,
            original_K_sha256=array_sha(raws[slot]),source203_receipt=m['source203_receipt']))
    coverage=[]
    for pid,seg in gt['segments'].items():
        rows=gt['rows'][seg['start']:seg['end']];dates=list(map(int,rows[:,0]));sides=list(map(bool,inside(rows,gt['polygon'])))
        probe=first_return(dates,sides);item=dict(canonical_gt_id=pid,original_gt_id=seg['original_gt_id'],gt_run=seg.get('run'),probe=probe,thresholds={})
        if probe is not None:
            index=seg['start']+probe['return_index'];row=gt['rows'][index]
            for t in (.5,.3):
                geometric=geometry(gt,scenes,index,t);space=[];eligible=[];identity=[];strict=[];confirmed_run=[]
                if not np.isfinite(row).all() or not (row[4:6]>0).all():
                    item['thresholds'][str(t)]=dict(geometry=geometric,time_space_known=False,
                        reason='invalid_GT_probe_geometry',all_query_time_space_ids=None,
                        eligible_query_time_space_ids=None,same_run_anchor_query_ids=None,
                        same_run_confirmed_query_ids=None,exact_probe_first_candidate_query_ids=None)
                    continue
                for m in result['members']:
                    for q in m['queries']:
                        if not queries.in_corridor(q,row[0],row[2:6]):continue
                        space.append(q['query_id'])
                        if q['eligible']:eligible.append(q['query_id'])
                        label=q['labels'][str(t)]
                        if label.get('canonical_gt_id')==pid:identity.append(q['query_id'])
                        if label['confirmed'] and label.get('canonical_gt_id')==pid:
                            confirmed_run.append(q['query_id'])
                            if label['candidate']['GT_canonical_index']==index:strict.append(q['query_id'])
                item['thresholds'][str(t)]=dict(geometry=geometric,time_space_known=True,all_query_time_space_ids=space,
                    eligible_query_time_space_ids=eligible,same_run_anchor_query_ids=identity,
                    same_run_confirmed_query_ids=confirmed_run,exact_probe_first_candidate_query_ids=strict)
        coverage.append(item)
    gt['geometry_cache']={}
    return result,dict(unit=group['unit'],domain=group['domain'],video=group['video'],runs=coverage)


PREREG='0528579a3678f2367a514920ef625f85f9b3fa10'
MARKER='[return-query-reach-v1]'
DOMAINS=('35','33','W','G')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):prior._old.save(path,value)


def verify(root):
    root=Path(root);h=sha(root/'manifest.json')
    if h!=(root/'SEALED.sha256').read_text().strip():raise ValueError('owners206 top seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        f=(root/name).resolve()
        if not f.is_relative_to(root.resolve()) or sha(f)!=digest:raise ValueError('owners206 file drift '+name)
    return dict(manifest_sha256=h)


def load_group(root,reference):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    name=reference['path']
    if sha(root/name)!=manifest[name]:raise ValueError('owner unit differs manifest')
    return json.loads((root/name).read_text())


def run(args):
    from beetrack.evaluate import load_tracks
    freezer=module('return206freeze',BASE/'freeze206.py')
    if args.prereg!=PREREG:raise ValueError('wrong206 published registration')
    subprocess.run(['git','merge-base','--is-ancestor',PREREG,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',PREREG+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('unique206 marker required')
    if args.out.exists() or args.out.resolve().drive.upper()!='D:':raise ValueError('new D: output required')
    freezer.verify(args.frozen);frozen=json.loads((args.frozen/'deployment.json').read_text())
    if frozen['prereg']!=PREREG or frozen['source203']!=freezer.SOURCE203 or len(frozen['groups'])!=20 or frozen['member_clips']!=60:
        raise ValueError('full206 freeze required')
    if sha(args.universe203)!=UNIVERSE_SHA:raise ValueError('historical203 public universe differs pin')
    historical=json.loads(args.universe203.read_text());receipt=dict(manifest_sha256=sha(args.frozen/'manifest.json'),deployment_sha256=sha(args.frozen/'deployment.json'))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.owners206-',dir=args.out.parent) as temp:
        target=Path(temp)/'payload';target.mkdir();allrefs=[];clips={};hashes={};summary={};members=0
        for n,ref in enumerate(frozen['groups']):
            folder=args.frozen/ref['path']
            if sha(folder/'manifest.json')!=ref['seal']:raise ValueError('child freeze seal')
            group=freezer.load_group(folder);video=group['video'];domain=group['domain'];nf=group['nf']
            if len(group['members'])!=3:raise ValueError('three member scene required')
            for slot,m in enumerate(group['members']):
                if m['member_index']!=slot or not 0<=m['seed']<=14 or group['split']=='test':raise ValueError('unauthorized member')
            gtpath=args.repo/'runs/gt'/group['split']/video/'gt/gt.txt';digest=sha(gtpath)
            if digest!=prior.GT_SHA256.get(video):raise ValueError('GT differs six historical pins')
            relative=gtpath.relative_to(args.repo).as_posix();hashes[relative]=digest
            gt=prepare_gt(load_tracks(gtpath),group['polygon'])
            old,numeric=ledger(gt,video,nf,digest,relative)
            if prior._old.plain(old)!=historical['clips'][video]:raise ValueError('reconstructed GT universe differs sealed203 '+video)
            if video not in clips:
                clips[video]=old;save(target/('GT_'+video+'.json'),dict(video=video,nf=nf,file_sha256=digest,runs=numeric))
            result,coverage=analyze_group(group,gt,digest)
            result.update(schema='return-query206-owner-unit-v1',prereg=PREREG,freeze_receipt=receipt,
                          GT_sha256={relative:digest},source_universe203_sha256=UNIVERSE_SHA)
            name=f'u{n:02d}.json';cover=f'coverage{n:02d}.json';save(target/name,result);save(target/cover,coverage)
            labels=[q['labels'][t] for m in result['members'] for q in m['queries'] for t in ('0.5','0.3')]
            stats={t:dict(Counter(q['labels'][t]['classification'] for m in result['members'] for q in m['queries'])) for t in ('0.5','0.3')}
            allrefs.append(dict(unit=group['unit'],domain=domain,group=group['group'],video=video,nf=nf,path=name,coverage_path=cover,sha256=sha(target/name)))
            summary[group['unit']]=stats;members+=3
            print('OWNERS206_UNIT',group['unit'],stats,flush=True)
            del result,coverage,labels,gt,group;gc.collect()
        if members!=60 or len(clips)!=6:raise ValueError('complete60/six GT coverage missing')
        if any(sha(args.repo/p)!=h for p,h in hashes.items()):raise ValueError('GT changed')
        save(target/'gt_run_universe.json',dict(schema=historical['schema'],clips=clips))
        # Exact content was checked clip-by-clip; retain the historical raw
        # bytes too, so the published universe SHA can be verified directly.
        (target/'gt_run_universe.json').write_bytes(args.universe203.read_bytes())
        for h in DOMAINS:save(target/(h+'.json'),dict(schema='return-query206-domain-index-v1',domain=h,prereg=PREREG,groups=[g for g in allrefs if g['domain']==h]))
        save(target/'summary.json',dict(schema='return-query206-owner-summary-v1',prereg=PREREG,groups=20,members=60,
            source_universe203_sha256=UNIVERSE_SHA,freeze_receipt=receipt,GT_sha256=hashes,units=summary,code_sha256=sha(__file__)))
        save(target/'deployment.json',dict(schema='return-query206-owners-v1',prereg=PREREG,freeze_receipt=receipt,
            groups=allrefs,GT_sha256=hashes,source_universe203_sha256=UNIVERSE_SHA,code_sha256=sha(__file__),
            corrected_policy='unrelated Hungarian owner changes descriptive only; nonpositive probe geometry unknown',
            superseded_private_source_sha256='29b45e166652824abd1920114a1b9116ee962c23f5be2aa4269b1fb580c0e980',
            full_twothreshold_all20unit_rerun=True))
        save(target/'manifest.json',{p.name:sha(p) for p in target.iterdir()})
        (target/'SEALED.sha256').write_text(sha(target/'manifest.json')+'\n');target.rename(args.out)
    print('OWNERS206_SEALED',sha(args.out/'manifest.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--frozen',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--prereg',required=True)
    p.add_argument('--universe203',type=Path,required=True)
    run(p.parse_args())
