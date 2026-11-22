"""Project sealed phase212 inputs into a compact anonymous numeric bundle.

The bundle carries dates/inside bits and opaque linkage only. It does not
contain private event references, native rows, boxes, GT identities or paths.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

DOMAINS=('35','33','W','G')
SCHEMA='phase212.numeric.v1'
MARKER='[observed-phase-v1]'
PREREG='8311f4fee58caaf6712777dcf14dd37525580bf9'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf8') if path.suffix=='.gz' else path.open(encoding='utf8')) as f:
        return json.load(f)


def verify_tree(root):
    root=Path(root).resolve();manifest_path=root/'manifest.json'
    digest=sha(manifest_path)
    if (root/'SEALED.sha256').read_text(encoding='utf8').strip()!=digest:
        raise ValueError('sealed input manifest mismatch: '+root.name)
    manifest=read(manifest_path)
    for name,expected in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or sha(p)!=expected:
            raise ValueError('sealed input member mismatch: '+str(name))
    return digest,manifest


def public_state_aliases(public, meta):
    raw_actions=meta['original211_actions']
    endpoints=sorted({int(v) for action in raw_actions.values() for v in action['canonical_ids']})
    endpoint_map={v:i+1 for i,v in enumerate(endpoints)}
    by_signature={(a['family'],tuple(a['endpoint_ids'])):a['action_id'] for a in public['actions']}
    action_map={}
    for aid,a in raw_actions.items():
        signature=(a['family'],tuple(endpoint_map[int(v)] for v in a['canonical_ids']))
        if signature not in by_signature:raise ValueError('private/public action endpoint alias differs')
        action_map[aid]=by_signature[signature]
    public_by_actions={tuple(sorted(s['selected_action_ids'])):s['state_id'] for s in public['states']}
    aliases={}
    for state in meta['states']:
        selected=tuple(sorted(action_map[a] for a in state['selected_action_ids']))
        if selected not in public_by_actions:raise ValueError('private/public state action alias differs')
        aliases[state['state_id']]=public_by_actions[selected]
    if len(set(aliases.values()))!=len(public['states']) or set(aliases.values())!=set(public_by_actions.values()):
        raise ValueError('private/public full 86 state alias is not bijective')
    return aliases


def link_candidate(frozen, clip, public):
    """Return canonical event/state aliases only when every signed vertex joins."""
    scenes={s['scene_ref']:s for s in clip['_meta']['scenes']}
    canonical_ids=set()
    events_by_state={}
    for state in clip['_meta']['states']:
        events=read(frozen/clip['path']/scenes[state['scene_ref']]['events_file'])['events']
        events_by_state[state['state_id']]=events
        canonical_ids.update(int(e['canonical_id']) for e in events)
    public_ids={str(pid) for s in public['states'] for pid,_ in s['pred_events']}
    if len(canonical_ids)!=len(public_ids):raise ValueError('event inventory size mismatch')
    pids=sorted(public_ids,key=lambda x:int(x[1:]))
    canon_to_pid={str(canon):pid for pid,canon in zip(pids,sorted(canonical_ids))}
    state_alias=public_state_aliases(public,clip['_meta'])
    public_states={s['state_id']:s for s in public['states']}
    for state in clip['_meta']['states']:
        ps=public_states[state_alias[state['state_id']]]
        expected={str(pid):int(frame) for pid,frame in ps['pred_events']}
        actual={canon_to_pid[str(int(e['canonical_id']))]:int(e['signed_frame']) for e in events_by_state[state['state_id']]}
        if actual!=expected:raise ValueError('candidate clip signed event linkage differs')
    return canon_to_pid,state_alias,events_by_state


def _graph_projection(graph, member_events, prefix):
    by_canon={int(e['canonical_id']):e for e in member_events}
    if set(by_canon)!={int(v) for v in graph['pred_vertices']}:
        raise ValueError('truth graph does not cover exact K feature events')
    pmap={int(e['canonical_id']):e['member_event_id'] for e in member_events}
    gmap={int(v):prefix+('%05d'%(i+1)) for i,v in enumerate(sorted(map(int,graph['GT_vertices'])))}
    events=[]
    for event in graph['events']:
        canon=int(event['canonical_event_id'])
        if canon not in pmap:raise ValueError('truth event lacks opaque member link')
        events.append([pmap[canon],int(event['signed_frame'])])
    edges=sorted([[pmap[int(p)],gmap[int(g)]] for p,g in graph['edges']])
    return {'pred_events':sorted(events),'gt_events':sorted([gmap[int(e['canonical_event_id'])],int(e['signed_frame'])] for e in graph['GT_events']),
            'edges':edges}


def build(*, frozen, truth, predictions, evaluation, public211_path, out, prereg=PREREG):
    frozen=Path(frozen).resolve();truth=Path(truth).resolve();predictions=Path(predictions).resolve();evaluation=Path(evaluation).resolve()
    fseal,fmanifest=verify_tree(frozen);tseal,tmanifest=verify_tree(truth);pseal,pmanifest=verify_tree(predictions);eseal,emanifest=verify_tree(evaluation)
    public_path=Path(public211_path).resolve();public_sha=sha(public_path);public211=read(public_path)
    if public211.get('schema')!='joint211.numeric.v1' or public211.get('marker')!='[cross-family-peer-v1]':
        raise ValueError('pinned released joint211 numeric package required')
    frozen_top=read(frozen/'deployment.json');truth_top=read(truth/'deployment.json')
    prediction=read(predictions/'predictions.json');evaluation_summary=read(evaluation/'summary.json')
    if frozen_top.get('schema')!='phase212-prepare-v1' or truth_top.get('schema')!='phase212-truth-v1':raise ValueError('unexpected source stage schema')
    if frozen_top.get('prereg')!=prereg or truth_top.get('prereg')!=prereg:raise ValueError('phase prereg mismatch')
    if frozen_top.get('states')!=86 or frozen_top.get('originalK')!=18 or len(frozen_top.get('clips',[]))!=6:raise ValueError('incomplete frozen 86-state/18-member roster')
    if prediction.get('schema')!='phase212-source-logistic-v1' or set(prediction.get('folds',{}))!=set(DOMAINS):raise ValueError('four-fold fit source differs')
    if evaluation_summary.get('schema')!='phase212-evaluation-v1':raise ValueError('sealed phase212 evaluation required')
    if len(evaluation_summary.get('clips',[]))!=6 or sum(len(c['states']) for c in evaluation_summary['clips'])!=86:raise ValueError('evaluation lacks all 86 inherited states')
    bank_doc=read(frozen/frozen_top['feature_bank_file'])
    if bank_doc.get('schema')!='phase212-unique-feature-bank-v1':raise ValueError('feature-bank schema differs')
    feature_sequences={key:{'dates':v['dates'],'inside':v['inside']} for key,v in bank_doc['features'].items()}
    # Labels are joined in the private namespace, then references are replaced
    # with deterministic anonymous IDs before serialization.
    clip_records={(x['domain'],x['clip']):dict(x) for x in frozen_top['clips']}
    for rec in clip_records.values():
        rec['_root']=frozen;rec['_meta']=read(frozen/rec['path']/'clip.json')
    public_clip_alias={};private_to_public={};public_clips={};alias_to_private={};state_aliases={};linked_events={}
    used_aliases=set()
    for key,clip in clip_records.items():
        candidates=[]
        for public in public211['clips']:
            if public['domain']!=key[0] or (public['domain'],public['clip']) in used_aliases:continue
            try:link=link_candidate(frozen,clip,public)
            except (KeyError,ValueError,StopIteration):continue
            candidates.append((public,link))
        if len(candidates)!=1:raise ValueError('clip identity must have one exact event/action linkage to public211')
        public,link=candidates[0];alias=public['clip'];used_aliases.add((key[0],alias))
        public_clip_alias[key]=alias;private_to_public[key],state_aliases[key],linked_events[key]=link
        public_clips[key]=public;alias_to_private[key[0],alias]=key[1]
    if len(public_clip_alias)!=6:raise ValueError('public211 six-clip roster required')
    feature_event_by_ref={};event_by_clip_canon={};member_by_scene={}
    for key,clipref in clip_records.items():
        meta=clipref['_meta']
        states={}
        for state in meta['states']:
            events=linked_events[key][state['state_id']];mapping={}
            for e in events:
                canon=int(e['canonical_id']);pid=private_to_public[key].get(str(canon))
                if pid is None:raise ValueError('state event canonical ID absent from public211 map')
                mapping[pid]=e['feature_key'];feature_event_by_ref[e['event_ref']]=e
                event_by_clip_canon.setdefault(key,{})[canon]=e
            public_sid=state_aliases[key][state['state_id']]
            expected=next(s for s in public_clips[key]['states'] if s['state_id']==public_sid)
            if set(mapping)!={str(pid) for pid,frame in expected['pred_events']}:raise ValueError('old public state event coverage differs')
            for pid,frame in expected['pred_events']:
                source=next(e for e in events if private_to_public[key][str(int(e['canonical_id']))]==str(pid))
                if int(source['signed_frame'])!=int(frame):raise ValueError('old event date/direction link differs')
            states[public_sid]=mapping
        clipref['_states_features']=states
        for rec in meta['scenes']:
            if rec['role'] in ('receiverK','peerK'):member_by_scene[(key[0],key[1],rec['scene_ref'])]=rec
    # opaque member aliases are stable across clips, allowing correct pooling
    # without retaining source member labels.
    member_identity={};
    for row in truth_top['original_K']:
        key=(row['domain'],str(row['member']),int(row['member_index']))
        member_identity.setdefault(key,'m%02d'%(len(member_identity)+1))
    members=[];peer_truth_graphs={};receiver_labels={}
    truth_index={(r['domain'],r['clip'],r['scene_ref']):r for r in truth_top['original_K']}
    for source_key,row in sorted(truth_index.items()):
        domain,private_clip,scene_ref=source_key;clipref=clip_records[domain,private_clip];alias=public_clip_alias[domain,private_clip]
        member_scene=member_by_scene[source_key];scene_doc=read(frozen/clipref['path']/member_scene['events_file'])
        scene_events=scene_doc['events'];truth_ref=row['graphs']['0.5'];graph=read(truth/truth_ref)
        opaque_member=member_identity[(domain,str(row['member']),int(row['member_index']))]
        member_key=opaque_member+'_'+domain
        eventmap={int(e['canonical_id']):'q%05d'%(i+1) for i,e in enumerate(sorted(scene_events,key=lambda x:int(x['canonical_id'])))}
        labels={int(e['canonical_event_id']):int(e['y']) for e in graph['events']}
        if set(labels)!=set(eventmap):raise ValueError('member truth labels do not cover prepared event inventory')
        events=[]
        for e in scene_events:
            canon=int(e['canonical_id']);event_id=eventmap[canon]
            public_id=private_to_public[domain,private_clip].get(str(canon))
            row_event=dict(event_ref=opaque_member+'_'+domain+'_'+alias+'_'+event_id,
                member_event_id=event_id,public_event_id=public_id,feature_key=e['feature_key'],
                signed_frame=int(e['signed_frame']),y=labels[canon])
            events.append(row_event)
            if member_scene['role']=='receiverK':
                receiver_labels[row_event['event_ref']]=dict(domain=domain,clip=alias,event_ref=row_event['event_ref'],
                    feature_key=e['feature_key'],y=labels[canon],_raw_ref=e['event_ref'])
        truth_graph_key=member_key+'_'+alias
        members.append(dict(domain=domain,clip=alias,role=member_scene['role'],member_key=member_key,
            truth_graph_key=truth_graph_key,member_index=int(row['member_index']),events=events))
        if member_scene['role']=='peerK':
            graph_events=[dict(e,member_event_id=eventmap[int(e['canonical_id'])]) for e in scene_events]
            peer_truth_graphs[truth_graph_key]=_graph_projection(graph,graph_events,'u%05d_'%(len(peer_truth_graphs)+1))
    if len(members)!=18 or len(receiver_labels)==0 or len(peer_truth_graphs)!=12:raise ValueError('original K member truth roster incomplete')
    raw_source_order={r['_raw_ref']:i for i,r in enumerate(sorted(receiver_labels.values(),
        key=lambda x:(x['domain'],alias_to_private[x['domain'],x['clip']],x['_raw_ref'])))}
    source_rows=[]
    for row in receiver_labels.values():
        source_rows.append(dict(domain=row['domain'],clip=row['clip'],event_ref=row['event_ref'],feature_key=row['feature_key'],
                                y=row['y'],source_order=raw_source_order[row['_raw_ref']]))
    source_rows.sort(key=lambda r:r['source_order'])
    # Preserve the fitted reference coefficients only as a compact consistency
    # check; replay recomputes all four folds independently from source rows.
    models_reference={}
    for h in DOMAINS:
        model=read(predictions/'models'/(h+'.json'))
        models_reference[h]={'heads':{head:model['heads'][head] for head in ('G','R','T')}}
    compact_clips=[]
    for key,clipref in clip_records.items():
        alias=public_clip_alias[key];fit_clip=prediction['clips'][key[0]+'/'+key[1]]
        compact_clips.append(dict(domain=key[0],clip=alias,state_event_features=clipref['_states_features'],
            selection_reference={head:dict(state_id=state_aliases[key][fit_clip['selected'][head]['state_id']]) for head in ('G','R','T','S')}))
    # Restricted forensic summary: publish only approved categorical and signed
    # timing counts; never copy source row IDs, crossings or geometry.
    def safe_rule_summary(branch):
        counts=branch.get('public_counts_only',{})
        distances=[int(x) for x in counts.get('minimum_GT_cross_distances',[]) if type(x) is int]
        return dict(net_events=int(counts.get('net_events',0)),no_net_event_runs=int(counts.get('no_net_event_runs',0)),
            events_with_legal_edge=int(counts.get('events_with_legal_edge',0)),
            events_without_legal_edge=int(counts.get('events_without_legal_edge',0)),
            same_direction_crossing_layers=counts.get('same_direction_crossing_layers',{}),
            unknown_GT_owner_crossings=int(counts.get('unknown_GT_owner_crossings',0)),
            timing_distance_count=len(distances),timing_distance_min=min(distances) if distances else None,
            timing_distance_max=max(distances) if distances else None)
    forensic=[]
    for item in evaluation_summary.get('original_fixed_forensics',[]):
        public_item={'domain':item.get('domain'),'clip':public_clip_alias[item['domain'],item['clip']]}
        key=(item['domain'],item['clip'])
        if item.get('state_id') in state_aliases.get(key,{}):public_item['state_id']=state_aliases[key][item['state_id']]
        details=item.get('details',{})
        public_details={}
        for threshold,pair in details.items():
            if threshold not in ('0.5','0.3') or not isinstance(pair,dict):continue
            side={}
            for name in ('before','after'):
                if name in pair:side[name]=safe_rule_summary(pair[name])
            for name in ('whole_scene_before','whole_scene_after'):
                if name in pair:
                    row=pair[name]
                    side[name]={k:int(row[k]) for k in ('TP','FP','FN','nPred') if k in row}
            public_details[threshold]=side
        if public_details:public_item['details']=public_details
        public_item['physical_identity_claim']=False
        public_item['scope']='Aggregated official owner/net/time-rule counts only; source row IDs, GT IDs and coordinates omitted.'
        forensic.append(public_item)
    bundle=dict(schema=SCHEMA,marker=MARKER,prereg=prereg,
        receipts=dict(frozen_manifest_sha256=fseal,truth_manifest_sha256=tseal,prediction_manifest_sha256=pseal,
            evaluation_manifest_sha256=eseal,joint211_sha256=public_sha,
            source_receipts=dict(frozen_files=len(fmanifest),truth_files=len(tmanifest),prediction_files=len(pmanifest),evaluation_files=len(emanifest))),
        feature_sequences=feature_sequences,source_rows=source_rows,clips=compact_clips,members=members,
        peer_truth_graphs=peer_truth_graphs,models_reference=models_reference,forensics=forensic,
        scope='Recomputes anonymous date/inside features, source refits, receiver and peer any-edge labels, phase guard choices, and scores/windows from the already-published §211 event graphs. Does not rebuild raw owner geometry or rerun 86 MOTs.')
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('output must be fresh')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped:
        zipped.write((json.dumps(bundle,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
    print(json.dumps(dict(schema=SCHEMA,clips=len(compact_clips),states=sum(len(c['state_event_features']) for c in compact_clips),
        members=len(members),source_rows=len(source_rows),sha256=sha(out)),sort_keys=True))


def self_test():
    # Packer tests are intentionally limited to deterministic non-sensitive ID
    # mapping and strict output compression; full raw projection awaits audit.
    assert _graph_projection(dict(pred_vertices=[1],GT_vertices=[9],edges=[[1,9]],
        events=[dict(canonical_event_id=1,signed_frame=2)],GT_events=[dict(canonical_event_id=9,signed_frame=3)]),
        [dict(canonical_id=1,member_event_id='q00001')],'u00001_')['edges']==[['q00001','u00001_00001']]
    return {'status':'PASS','scope':'synthetic-only'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--go',action='store_true',required=True)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--frozen',type=Path);parser.add_argument('--truth',type=Path)
    parser.add_argument('--predictions',type=Path);parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--public211',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--prereg',default=PREREG)
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),sort_keys=True))
    elif any(getattr(args,k) is None for k in ('frozen','truth','predictions','evaluation','public211','out')):
        parser.error('all sealed stage roots and public211 package are required')
    else:build(frozen=args.frozen,truth=args.truth,predictions=args.predictions,evaluation=args.evaluation,
               public211_path=args.public211,out=args.out,prereg=args.prereg)
