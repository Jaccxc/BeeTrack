"""Independent212 original-K actual event graphs and rule-layer forensics."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import field212_features as source
independent=source.independent;prior=source.prior;previous=source.previous

def event_check(proof,pid,P,G,under,edges,aliases=None):
    alias=lambda g:(None if g<0 else aliases[g]) if aliases is not None else int(g)
    date,direction=P['events'][pid];d,start,end,crosses=P['tracks'][pid]
    assert proof['canonical_event_id']==pid and proof['direction']==direction and proof['frame']==date and proof['signed_frame']==date*direction
    assert proof['start_canonical_row']==start and proof['end_canonical_row']==end
    for where,j in (('first',start),('last',end-1)):
        assert proof[where+'_prediction_raw_index']==int(P['index'][j]);prior.close(proof[where+'_row'],P['rows'][j],0)
    expected_cross=[j for j,sign in crosses if sign==direction];assert len(proof['all_same_direction_crossings'])==len(expected_cross)
    layers=[];distances=[];owners_seen=set()
    for saved,j in zip(proof['all_same_direction_crossings'],expected_cross):
        assert saved['after_canonical_row']==j and saved['frame']==P['rows'][j,0] and saved['direction']==direction
        assert saved['before_prediction_raw_index']==int(P['index'][j-1]) and saved['after_prediction_raw_index']==int(P['index'][j])
        prior.close(saved['before_row'],P['rows'][j-1],0);prior.close(saved['after_row'],P['rows'][j],0)
        assert saved['before_GT_owner']==alias(int(under[j-1])) and saved['after_GT_owner']==alias(int(under[j]))
        refs=sorted({int(under[j-1]),int(under[j])}-{-1});owners_seen.update(refs);assert len(refs)==len(saved['GT_owner_checks'])
        good=False;same=False
        for check,g in zip(saved['GT_owner_checks'],refs):
            gd,gs,ge,gcross=G['tracks'][g];same_cross=[(k,int(G['rows'][k,0])) for k,sign in gcross if sign==direction]
            dd=[abs(f-int(P['rows'][j,0])) for k,f in same_cross];timely=[k for k,f in same_cross if abs(f-int(P['rows'][j,0]))<=5]
            legal=gd==direction and bool(timely);same|=gd==direction;good|=legal
            assert check['GT_canonical_id']==alias(g) and check['GT_net_direction']==({1:'in',-1:'out',0:None}[gd])
            assert check['GT_same_direction_crossings']==[dict(canonical_row=k,GT_raw_index=int(G['index'][k]),frame=f) for k,f in same_cross]
            minimum=min(dd) if dd else None;assert check['minimum_same_direction_distance']==minimum
            assert check['within5_same_direction_GT_rows']==timely and check['legal']==legal
            reason='legal_GT_neighbor' if legal else 'GT_net_direction_mismatch' if gd!=direction else 'no_same_direction_GT_cross_within5'
            assert check['reason']==reason
            if minimum is not None:distances.append(minimum)
        layer='neither_crossing_row_has_official_GT_owner' if not refs else 'legal_GT_neighbor_at_this_crossing' if good else 'same_net_direction_but_time_not_supported' if same else 'owned_GT_runs_have_other_or_zero_net_direction'
        assert saved['rule_layer']==layer;layers.append(layer)
    neighbors=sorted(g for p,g in edges if p==pid);degree=Counter(g for p,g in edges)
    assert proof['y']==int(bool(neighbors)) and proof['legal_GT_neighbors']==[alias(g) for g in neighbors] and proof['legal_GT_neighbor_count']==len(neighbors)
    assert proof['neighbor_GT_event_degrees']==[degree[g] for g in neighbors] and proof['has_competing_prediction_for_any_neighbor']==any(degree[g]>1 for g in neighbors)
    return layers,distances,owners_seen

def validate_graph(saved,calculation,owners):
    P,G=calculation['prediction_structure'],calculation['reference_structure'];threshold=str(saved['threshold']);graph=calculation['graphs'][threshold]
    assert saved['pred_vertices']==sorted(graph['left']) and saved['GT_vertices']==sorted(graph['right'])
    assert sorted(map(tuple,saved['edges']))==sorted(map(tuple,graph['edges']))
    assert saved['TP']==graph['maxTP'] and saved['nPred']==len(graph['left']) and saved['nGT']==len(graph['right'])
    assert saved['FP']==len(graph['left'])-graph['maxTP'] and saved['FN']==len(graph['right'])-graph['maxTP']
    assert len(saved['matching'])==graph['maxTP'] and len({p for p,g in saved['matching']})==len(saved['matching']) and len({g for p,g in saved['matching']})==len(saved['matching'])
    assert set(map(tuple,saved['matching']))<=set(map(tuple,graph['edges']))
    events={e['canonical_event_id']:e for e in saved['events']};assert set(events)==set(P['events'])
    under=owners[P['index']]
    for pid in events:event_check(events[pid],pid,P,G,under,graph['edges'])
    assert {(e['canonical_event_id'],e['signed_frame']) for e in saved['GT_events']}=={(pid,f*d) for pid,(f,d) in G['events'].items()}
    return events

def run_records(P,raw_map,canon,parents):
    selected=[]
    for pid,(d,start,end,crosses) in P['tracks'].items():
        original=np.asarray(raw_map)[np.asarray(P['index'])[start:end]];ids=sorted(map(int,set(canon[original])))
        if set(ids).intersection(parents):selected.append((pid,d,start,end,crosses,original,ids))
    return selected

def forensic_members(proofs,P,G,owners,raw_map,canon,parents,aliases):
    expected=run_records(P,raw_map,canon,parents);assert len(proofs['runs'])==len(expected)
    layers=[];distances=[];targets=set();events=0;with_edge=0
    for saved,(pid,d,start,end,crosses,original,ids) in zip(proofs['runs'],expected):
        assert saved['canonical_run_id']==pid and saved['net_direction']==d and saved['original_parent_ids']==ids
        assert saved['original_K_raw_indices']==original.tolist() and saved['raw_row_indices']==np.asarray(P['index'])[start:end].tolist()
        assert saved['first_prediction_raw_index']==int(P['index'][start]) and saved['last_prediction_raw_index']==int(P['index'][end-1])
        prior.close(saved['first_row'],P['rows'][start],0);prior.close(saved['last_row'],P['rows'][end-1],0)
        assert saved['crossings']==[dict(before_prediction_raw_index=int(P['index'][j-1]),after_prediction_raw_index=int(P['index'][j]),frame=int(P['rows'][j,0]),direction=int(sign)) for j,sign in crosses]
        if not d:assert saved['event'] is None;continue
        events+=1;ee=saved['event'];with_edge+=ee['y']
        ll,dd,oo=event_check(ee,pid,P,G,owners[P['index']],P['_edges'],aliases);layers+=ll;distances+=dd;targets.update(oo)
    count=proofs['public_counts_only'];assert count['net_events']==events and count['no_net_event_runs']==len(expected)-events
    assert count['events_with_legal_edge']==with_edge and count['events_without_legal_edge']==events-with_edge
    assert count['same_direction_crossing_layers']==dict(Counter(layers)) and count['minimum_GT_cross_distances']==distances
    assert count['unknown_GT_owner_crossings']==layers.count('neither_crossing_row_has_official_GT_owner')
    return dict(net_events=events,events_with_legal_edge=with_edge,layers=dict(Counter(layers)),owner_runs=targets)

def run(args):
    if not args.field_go:raise ValueError('soleCPU GO required')
    sys.path.insert(0,str(args.repo.resolve()));assert prior.read(args.root/'verification/features.json')['passed']
    from beetrack.evaluate import load_tracks
    replay=prior.module('audit212_truth_public203',args.repo/'tools/study203/verify203.py');_,native,v,_=replay.helpers(args.repo)
    top,records=source.load_records(args);truth=prior.read(args.root/'labels/deployment.json');late=prior.read(args.root/'truth_late/deployment.json')
    assert truth['prereg']==source.PREREG and late['prereg']==source.PREREG and late['prediction_manifest_sha256']==source.PINS['predictions']
    original={(r['domain'],r['clip'],r['member_index']):r for r in truth['original_K']};case_refs={(r['domain'],r['clip']):[] for r in late['forensics']}
    for ref in late['forensics']:case_refs[ref['domain'],ref['clip']].append(ref)
    oldindex=prior.read(args.main_assessment/'index.json');old={(r['domain'],r['clip']):r for r in oldindex['clips']}
    reports=[];forensic_reports=[]
    for key,r in records.items():
        c=r['meta'];domain,name=c['domain'],c['clip'];gtpath=args.repo/'runs/gt'/c['split']/name/'gt/gt.txt'
        assert prior.sha(gtpath)==truth['GT_sha256'][gtpath.relative_to(args.repo).as_posix()]
        GT=load_tracks(gtpath);G=native.track_structure(GT,c['polygon']);receiver_info=None
        for slot in (0,1,2):
            scene=next(s for s in c['scenes'] if s['scene_ref']==c['K_scene_refs'][str(slot)]);rows,raw=source.scene_rows(r,scene);P=native.track_structure(rows,c['polygon'])
            cache=previous.OrderedGeometryCache(GT,rows,v);cache.prepare(P);owners=cache.owners(rows,raw)
            calculation=previous.independent.audit_native_state(rows,G,c['polygon'],c['nf'],raw,cache,native,previous.geometry.clock,P)
            ref=original[domain,name,slot];graphfiles=ref['graphs'];labels={}
            for threshold in ('0.5','0.3'):
                saved=prior.read(args.root/'labels'/graphfiles[threshold]);events=validate_graph(saved,calculation,owners[threshold]);labels[threshold]={pid:e['y'] for pid,e in events.items()}
                if slot==0:
                    Kref=next(s for s in old[domain,name]['states'] if not s['selected_pair_ids']);metric=prior.read(args.main_assessment/Kref['path']/'score.json')['metrics'];edge='edges05' if threshold=='0.5' else 'edges03'
                    assert sorted(map(tuple,saved['edges']))==sorted(map(tuple,metric['evidence'][edge]))
            eventrefs=prior.read(r['folder']/scene['events_file'])['events'];assert {e['canonical_id'] for e in eventrefs}==set(labels['0.5'])
            shard=prior.read(args.root/'labels'/f'{domain}.json')['records'] if slot==0 else prior.read(args.root/'labels/peer_labels'/domain/name/f'm{slot}.json')['records']
            by_ref={e['event_ref']:e for e in shard if e['clip']==name and e['member_index']==slot}
            assert set(by_ref)=={e['event_ref'] for e in eventrefs}
            for e in eventrefs:assert by_ref[e['event_ref']]['y']==labels['0.5'][e['canonical_id']] and by_ref[e['event_ref']]['feature_key']==e['feature_key']
            reports.append(dict(clip=key,slot=slot,role=scene['role'],member=scene['member'],events=len(eventrefs),positive=sum(labels['0.5'].values()),fresh_both_IoU_graphs=True))
            if slot==0:receiver_info=(rows,raw,P,cache,owners,calculation)
        if (domain,name) in case_refs:
            base,base_raw,beforeP,cache,beforeowners,beforecalc=receiver_info;canon=r['arrays']['receiver_canon']
            for ref in case_refs[domain,name]:
                case=prior.read(args.root/'truth_late'/ref['path']);oldstate=next(s for s in old[domain,name]['states'] if s['state_id']==ref['state_id']);folder=args.main_assessment/oldstate['path']
                rows=np.loadtxt(folder/'mot.txt',delimiter=',',usecols=range(7),ndmin=2);raw=np.load(folder/'output_raw_indices.npy');P=native.track_structure(rows,c['polygon']);cache.prepare(P);own=cache.owners(rows,raw)
                calc=previous.independent.audit_native_state(rows,G,c['polygon'],c['nf'],raw,cache,native,previous.geometry.clock,P)
                parents=set(case['original_family']);details=[]
                for threshold in ('0.5','0.3'):
                    targets=set()
                    for structure,ownership in ((beforeP,beforeowners[threshold]),(P,own[threshold])):
                        under=ownership[structure['index']]
                        for pid,d,start,end,crosses,_,_ in run_records(structure,np.arange(len(base)) if structure is beforeP else raw,canon,parents):
                            if not d:continue
                            for j,sign in crosses:
                                if sign==d:targets.update({int(under[j-1]),int(under[j])}-{-1})
                    aliases={g:'g'+str(i) for i,g in enumerate(sorted(targets))}
                    beforeP['_edges']=beforecalc['graphs'][threshold]['edges'];P['_edges']=calc['graphs'][threshold]['edges']
                    proof=case['details'][threshold]
                    b=forensic_members(proof['before'],beforeP,G,beforeowners[threshold],np.arange(len(base)),canon,parents,aliases)
                    a=forensic_members(proof['after'],P,G,own[threshold],raw,canon,parents,aliases)
                    def counts(result):
                        z=result['iou'][threshold];return dict(TP=z['tp'],FP=z['pred']-z['tp'],FN=z['gt']-z['tp'],nPred=z['pred'])
                    assert proof['whole_scene_before']==counts(beforecalc) and proof['whole_scene_after']==counts(calc)
                    original_ids=[]
                    for g in sorted(targets):
                        _,s,e,_=G['tracks'][g];ids=set(map(int,GT[np.asarray(G['index'])[s:e],1]));original_ids.append(next(iter(ids)) if len(ids)==1 else None)
                    details.append(dict(threshold=threshold,before_events=b['net_events'],after_events=a['net_events'],before_legal=b['events_with_legal_edge'],after_legal=a['events_with_legal_edge'],before_layers=b['layers'],after_layers=a['layers'],distinct_original_annotation_IDs=len(set(x for x in original_ids if x is not None)),identity_unknown=any(x is None for x in original_ids)))
                forensic_reports.append(dict(clip=key,state_id=ref['state_id'],layers=details,physical_identity_claim=False))
        print('AUDIT212_TRUTH_CLIP',key,flush=True)
    assert len(reports)==18 and len(forensic_reports)==5
    prior.save(args.root/'verification/labels.json',dict(passed=True,prereg=source.PREREG,original_K=reports,forensics=forensic_reports,scope='All18 originalK graphs independently rebuilt from fullscene Hungarian geometry at .5/.3; all any-legal-edge y labels reconstructed, receiver6 matched old211 K graphs and peer12 newly independently checked. Five fixed old211 I-singleton scenes freshly checked for crossing→owner→GT net direction→same-direction time≤5→legal-edge layers, including no-net runs and competition; GT aliases mapped within case. This is rule/annotation support, not biological identity or a declaration that no-edge means false bee. No86 fullMOT rescoring.'))
    snap=args.root/'verification/source_snapshot_truth';snap.mkdir(exist_ok=True);(snap/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('AUDIT212_TRUTH_PASS',prior.sha(args.root/'verification/labels.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--main-assessment',type=Path,required=True);p.add_argument('--field-go',action='store_true');run(p.parse_args())
