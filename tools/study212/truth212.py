"""Source-only212 actual-GT event-neighbor truth and official-rule forensics.

y means existence of a legal neighbor in the full native event graph. It is
neither a chosen matching assignment, a deletion marginal, nor bee identity.
Only receiver K source-domain labels are exposed by the training loader.
"""
import os
import sys
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.dont_write_bytecode=True
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
import importlib.util
import argparse
import subprocess
import re
import tempfile
import gzip
import gc

DOMAINS=('35','33','W','G')


def maximum(left,right,edges):
    neighbors={p:[] for p in left};right=set(right)
    for p,g in sorted(set(map(tuple,edges))):
        if p not in neighbors or g not in right:raise ValueError('event edge outside vertex set')
        neighbors[p].append(g)
    matched={}
    def visit(p,seen):
        for g in neighbors[p]:
            if g in seen:continue
            seen.add(g)
            if g not in matched or visit(matched[g],seen):matched[g]=p;return True
        return False
    for p in sorted(left):visit(p,set())
    return [[p,g] for g,p in sorted(matched.items(),key=lambda z:(z[1],z[0]))]


def canonical(raw):
    from beetrack.evaluate import by_track
    raw=np.asarray(raw,float)
    if raw.ndim!=2 or raw.shape[1]<6 or not np.isfinite(raw).all():raise ValueError('finite native track matrix required')
    tagged=by_track(np.c_[raw,np.arange(len(raw))]);return tagged[:,:-1],tagged[:,-1].astype(np.int64)


def event_graph(raw_GT,raw_prediction,polygon,threshold=.5,verify_official=True):
    import beetrack.evaluate as E
    if threshold not in (.5,.3):raise ValueError('official .5/.3 only')
    G,gi=canonical(raw_GT);P,pi=canonical(raw_prediction)
    gz=E.zone_tracks(G,polygon);pz=E.zone_tracks(P,polygon);owners=E.frame_match(G,P,thr=threshold)
    right=sorted(g for g,t in gz.items() if t[0]);left=sorted(p for p,t in pz.items() if t[0]);edges=set();events=[]
    for pid in left:
        direction,start,end,cross=pz[pid];sign=1 if direction=='in' else -1;crossproof=[]
        for j,d in cross:
            if d!=direction:continue
            refs=sorted({int(owners[j-1]),int(owners[j])}-{-1});candidateproof=[]
            for g in refs:
                gd,gs,ge,gcross=gz[g];same=[(int(k),int(G[k,0])) for k,dg in gcross if dg==direction]
                distances=[abs(date-int(P[j,0])) for k,date in same];timely=[k for k,date in same if abs(date-int(P[j,0]))<=5]
                legal=gd==direction and bool(timely)
                if legal:edges.add((int(pid),int(g)))
                candidateproof.append(dict(GT_canonical_id=int(g),GT_net_direction=gd,
                    GT_same_direction_crossings=[dict(canonical_row=k,GT_raw_index=int(gi[k]),frame=date) for k,date in same],
                    minimum_same_direction_distance=min(distances) if distances else None,
                    within5_same_direction_GT_rows=[int(k) for k in timely],legal=legal,
                    reason='legal_GT_neighbor' if legal else ('GT_net_direction_mismatch' if gd!=direction else 'no_same_direction_GT_cross_within5')))
            if not refs:layer='neither_crossing_row_has_official_GT_owner'
            elif any(x['legal'] for x in candidateproof):layer='legal_GT_neighbor_at_this_crossing'
            elif any(x['GT_net_direction']==direction for x in candidateproof):layer='same_net_direction_but_time_not_supported'
            else:layer='owned_GT_runs_have_other_or_zero_net_direction'
            crossproof.append(dict(after_canonical_row=int(j),before_prediction_raw_index=int(pi[j-1]),after_prediction_raw_index=int(pi[j]),
                before_row=P[j-1].tolist(),after_row=P[j].tolist(),frame=int(P[j,0]),direction=sign,
                before_GT_owner=int(owners[j-1]),after_GT_owner=int(owners[j]),GT_owner_checks=candidateproof,rule_layer=layer))
        chosen=next(j for j,d in reversed(cross) if d==direction)
        events.append(dict(canonical_event_id=int(pid),direction=sign,frame=int(P[chosen,0]),signed_frame=sign*int(P[chosen,0]),
            start_canonical_row=int(start),end_canonical_row=int(end),first_prediction_raw_index=int(pi[start]),
            last_prediction_raw_index=int(pi[end-1]),first_row=P[start].tolist(),last_row=P[end-1].tolist(),
            all_same_direction_crossings=crossproof))
    edges=sorted(edges);matching=maximum(left,right,edges);GTdegree=Counter(g for p,g in edges);neighbors={p:[] for p in left}
    for p,g in edges:neighbors[p].append(g)
    for event in events:
        ns=neighbors[event['canonical_event_id']];event.update(y=int(bool(ns)),legal_GT_neighbors=ns,
            legal_GT_neighbor_count=len(ns),neighbor_GT_event_degrees=[GTdegree[g] for g in ns],
            has_competing_prediction_for_any_neighbor=any(GTdegree[g]>1 for g in ns))
    if verify_official:
        old=E.frame_match.__defaults__
        try:
            E.frame_match.__defaults__=(threshold,);official=E.match_events(raw_GT,raw_prediction,polygon,tol=5)
        finally:E.frame_match.__defaults__=old
        if (official['pred'],official['gt'],official['tp'])!=(len(left),len(right),len(matching)):
            raise AssertionError('fresh event graph differs official whole-scene score')
    run_table=[]
    for pid,(direction,start,end,crosses) in pz.items():
        run_table.append(dict(canonical_run_id=int(pid),net_direction=1 if direction=='in' else (-1 if direction=='out' else 0),
            first_prediction_raw_index=int(pi[start]),last_prediction_raw_index=int(pi[end-1]),
            first_row=P[start].tolist(),last_row=P[end-1].tolist(),raw_row_indices=pi[start:end].tolist(),
            crossings=[dict(before_prediction_raw_index=int(pi[j-1]),after_prediction_raw_index=int(pi[j]),frame=int(P[j,0]),
                direction=1 if d=='in' else -1) for j,d in crosses]))
    GT_events=[]
    for gid in right:
        direction,start,end,crosses=gz[gid];j=next(j for j,d in reversed(crosses) if d==direction);sign=1 if direction=='in' else -1
        GT_events.append(dict(canonical_event_id=int(gid),direction=sign,frame=int(G[j,0]),signed_frame=sign*int(G[j,0])))
    return dict(schema='phase212-true-native-event-graph-v1',threshold=threshold,pred_vertices=left,GT_vertices=right,GT_events=GT_events,
        edges=[list(e) for e in edges],matching=matching,nPred=len(left),nGT=len(right),TP=len(matching),
        FP=len(left)-len(matching),FN=len(right)-len(matching),events=events,run_table=run_table,
        truth_scope='legal-neighbor existence; not matched TP, essentiality, or physical identity')


def label_records(domain,clip,role,member_index,member,graph,event_references):
    """Frozen event references own the feature keys; GT cannot choose a phase."""
    references={int(r['canonical_id']):r for r in event_references}
    if references.keys()!=set(graph['pred_vertices']):raise ValueError('frozen features do not cover every original K event')
    result=[]
    for event in graph['events']:
        ref=references[event['canonical_event_id']]
        if ref['signed_frame']!=event['signed_frame']:raise ValueError('frozen event clock/direction mismatch')
        result.append(dict(domain=domain,clip=clip,role=role,member_index=int(member_index),member=str(member),
            event_ref=ref['event_ref'],canonical_event_id=event['canonical_event_id'],feature_key=ref['feature_key'],y=event['y']))
    return result


def assert_211_receiver_graph(graph,metrics,threshold):
    evidence=metrics['evidence'];edgekey='edges05' if threshold==.5 else 'edges03'
    predicted=sorted([e['canonical_event_id'],e['signed_frame']] for e in graph['events'])
    reference=sorted([e['canonical_event_id'],e['signed_frame']] for e in graph['GT_events'])
    if predicted!=sorted(evidence['predictions']) or reference!=sorted(evidence['gt']):raise ValueError('receiver211 event vertex/date mismatch')
    if sorted(graph['edges'])!=sorted(evidence[edgekey]):raise ValueError('receiver211 actual GT legal edges mismatch')
    old=metrics['iou'][str(threshold)]
    if (graph['nPred'],graph['nGT'],graph['TP'])!=(metrics['nP'],metrics['nG'],old['tp']):raise ValueError('receiver211 graph metric mismatch')
    return dict(passed=True,scope='receiver K exact signed vertices, both-IoU legal edges and whole-graph cardinality')


def family_forensics(graph,output_raw_indices,original_canon,parents):
    """Changed native family only; IDs are technical references, not bee claims."""
    rawmap=np.asarray(output_raw_indices,np.int64);canon=np.asarray(original_canon,np.int64);wanted=set(map(int,parents))
    if rawmap.shape!=(len(canon),) or not np.array_equal(np.sort(rawmap),np.arange(len(canon))):raise ValueError('complete original-row permutation required')
    selected=[];event_table={e['canonical_event_id']:e for e in graph['events']}
    for run in graph['run_table']:
        raw=np.asarray(run['raw_row_indices'],np.int64);originals=sorted(set(map(int,canon[rawmap[raw]])))
        if not wanted.intersection(originals):continue
        selected.append(dict(run,original_parent_ids=originals,original_K_raw_indices=rawmap[raw].tolist(),
            event=event_table.get(run['canonical_run_id'])))
    events=[r['event'] for r in selected if r['event'] is not None]
    layers=Counter(c['rule_layer'] for e in events for c in e['all_same_direction_crossings'])
    distances=[g['minimum_same_direction_distance'] for e in events for c in e['all_same_direction_crossings']
               for g in c['GT_owner_checks'] if g['minimum_same_direction_distance'] is not None]
    return dict(parents=sorted(wanted),runs=selected,public_counts_only=dict(
        net_events=len(events),no_net_event_runs=sum(r['event'] is None for r in selected),
        events_with_legal_edge=sum(e['y'] for e in events),events_without_legal_edge=sum(not e['y'] for e in events),
        same_direction_crossing_layers=dict(layers),minimum_GT_cross_distances=distances,
        unknown_GT_owner_crossings=layers['neither_crossing_row_has_official_GT_owner']),
        interpretation='official rule layers and timing only; not physical identity or image visibility')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source_labels(root,heldout,requested_domains=None,expected_prereg=None,expected_frozen_sha256=None):
    if heldout not in DOMAINS:raise ValueError('invalid heldout hive')
    allowed=tuple(h for h in DOMAINS if h!=heldout);requested=allowed if requested_domains is None else tuple(requested_domains)
    if len(requested)!=3 or set(requested)!=set(allowed):raise ValueError('only other3 source domains may be opened')
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('truth212 seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text());identity=None;result=[]
    for domain in requested:
        name=domain+'.json'
        if sha(root/name)!=manifest[name]:raise ValueError('source receiver shard drift')
        data=json.loads((root/name).read_text());current=(data['prereg'],data['frozen_manifest_sha256'])
        if identity is None:identity=current
        if data['domain']!=domain or current!=identity:raise ValueError('source identity mismatch')
        if expected_prereg is not None and current[0]!=expected_prereg:raise ValueError('wrong prereg')
        if expected_frozen_sha256 is not None and current[1]!=expected_frozen_sha256:raise ValueError('wrong feature parent')
        if any(r['role']!='receiverK' or r['domain']!=domain or r['member_index']!=0 for r in data['records']):raise ValueError('peer/target event in receiver-only source shard')
        result.extend(data['records'])
    return result


def selftest():
    def r(f,p,x):return [f,p,x,0,2,2,.9]
    poly=[(0,-10),(50,-10),(50,10),(0,10)]
    gt=np.array([r(1,10,-3),r(2,10,1),r(3,10,-3),r(4,10,1)],float)
    pred=gt.copy();pred[:2,1]=1;pred[2:,1]=2
    graph=event_graph(gt,pred,poly)
    assert [r['y'] for r in graph['events']]==[1,1] and graph['TP']==1
    assert all(r['has_competing_prediction_for_any_neighbor'] for r in graph['events'])
    assert all(r['legal_GT_neighbor_count']==1 for r in graph['events'])
    try:load_source_labels(Path('must-not-open'),'35',['35','W','G'])
    except ValueError:pass
    else:raise AssertionError('target request not refused before I/O')
    return 'TRUTH212_TINY_SYNTHETIC_PASS'


MARKER='[observed-phase-v1]'
SOURCE211_ASSESSMENT='7df94f0fbac521d715349651b5f9c1527854748535ea70399733c23b813a8d81'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def verify_tree(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('sealed tree mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=digest:raise ValueError('artifact drift '+name)
    return manifest


def run(args):
    """Real stage; the coordinator must grant sole CPU after full feature seal."""
    sys.path.insert(0,str(args.repo.resolve()))
    from beetrack.evaluate import load_tracks
    base=Path(__file__).parent;sys.path.insert(0,str(base))
    F=module('phase212prepare',base/'prepare212.py')
    prior=module('phase212GTpins',base.parent/'peer_suffix203'/'owners203.py')
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('published full prereg required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('unique212 marker required')
    if args.out.exists() or args.out.resolve().drive.upper()!='D:':raise ValueError('fresh D: truth root required')
    F.verify(args.frozen);top=json.loads((args.frozen/'deployment.json').read_text());parent=sha(args.frozen/'manifest.json')
    if top['schema']!='phase212-prepare-v1' or top['prereg']!=args.prereg or top['states']!=86 or top['originalK']!=18 or len(top['clips'])!=6:
        raise ValueError('all86 states/18 K features must freeze first')
    if top['identity']['main211_frozen']!=F.SOURCE211 or top['identity']['inputs208']!=F.SOURCE208:raise ValueError('wrong original K lineage')
    expected={('35','BEE24-35-head'),('33','BEE24-33-head'),('W','BEE24-10'),('W','BEE24-15'),('G','BEE24-24'),('G','BEE24-28')}
    if {(c['domain'],c['clip']) for c in top['clips']}!=expected:raise ValueError('six original clips required')
    shards={h:dict(schema='phase212-source-labels-v1',domain=h,prereg=args.prereg,frozen_manifest_sha256=parent,
        label_rule='actual GT .5 legal event-neighbor existence, receiver K only',records=[]) for h in DOMAINS}
    args.out.parent.mkdir(parents=True,exist_ok=True);indices=[];forensicrefs=[];hashes={};nK=0
    with tempfile.TemporaryDirectory(prefix='.truth212-',dir=args.out.parent) as temporary:
        target=Path(temporary)/'payload';target.mkdir()
        for ref in top['clips']:
            clip=F.load_clip(args.frozen,ref);domain=clip['domain'];name=clip['clip'];split='dev' if name.endswith('-head') else 'val'
            if clip['split']!=split:raise ValueError('unexpected source clip split')
            gtpath=args.repo/'runs/gt'/split/name/'gt/gt.txt';digest=sha(gtpath)
            if digest!=prior.GT_SHA256.get(name):raise ValueError('GT differs historical six-file pin')
            relative=gtpath.relative_to(args.repo).as_posix();hashes[relative]=digest;GT=load_tracks(gtpath)
            for slot in (0,1,2):
                loaded=F.load_scene(clip,clip['K_scene_refs'][str(slot)]);scope=loaded['metadata'];rows=loaded['rows'];role=scope['role']
                if scope['native_slot']!=slot or scope['member_index']!=slot or scope['role']!=('receiverK' if slot==0 else 'peerK'):
                    raise ValueError('original K role/slot mismatch')
                if scope['seed']!=clip['seed']+slot or not 0<=scope['seed']<=14:raise ValueError('unregistered member seed')
                graphs={str(t):event_graph(GT,rows,clip['polygon'],t) for t in (.5,.3)}
                graphpaths={};proofs=dict(pending_late_common_prediction_seal=True) if slot==0 else {}
                for t,graph in graphs.items():
                    destination=target/'assessment'/domain/name/f'm{slot}_{t}.json.gz';destination.parent.mkdir(parents=True,exist_ok=True)
                    F.save_gz(destination,graph);graphpaths[t]=destination.relative_to(target).as_posix()
                records=label_records(domain,name,role,slot,scope['member'],graphs['0.5'],loaded['events']['events'])
                if slot==0:shards[domain]['records'].extend(records)
                else:
                    path=target/'peer_labels'/domain/name/f'm{slot}.json';path.parent.mkdir(parents=True,exist_ok=True)
                    F.save(path,dict(domain=domain,clip=name,role='peerK',member_index=slot,records=records,training_prohibited=True))
                indices.append(dict(domain=domain,clip=name,role=role,member=scope['member'],member_index=slot,
                    scene_ref=scope['scene_ref'],graphs=graphpaths,receiver211_exact=proofs,rows_sha256=F.sha(clip['folder']/'native.npz')))
                nK+=1;print('TRUTH212_K_COMPLETE',domain,name,'m'+str(slot),flush=True)
                del graphs,records,rows
            # Historical receiver comparisons and five singleton forensics are
            # explicitly late-stage: no old outcome payload is opened here.
            del clip,GT;gc.collect()
        if nK!=18:raise ValueError('complete18 K required')
        if any(sha(args.repo/path)!=digest for path,digest in hashes.items()):raise ValueError('GT changed during truth stage')
        for domain,data in shards.items():F.save(target/(domain+'.json'),data)
        F.save(target/'deployment.json',dict(schema='phase212-truth-v1',prereg=args.prereg,frozen_manifest_sha256=parent,
            source211_assessment_sha256=SOURCE211_ASSESSMENT,original_K=indices,forensics=forensicrefs,GT_sha256=hashes,
            receiver_only_source_shards=True,peer_labels_and_forensics_not_training_inputs=True,
            receiver211_comparison_pending_late_stage=True,forensics_pending_late_stage=True,
            old_true_assessment_not_opened=True,code_sha256=sha(__file__)))
        (target/'truth212.py').write_bytes(Path(__file__).read_bytes())
        F.save(target/'manifest.json',{p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()})
        (target/'SEALED.sha256').write_text(sha(target/'manifest.json')+'\n');target.rename(args.out)
    print('TRUTH212_SEALED',sha(args.out/'manifest.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--self-test',action='store_true')
    p.add_argument('--frozen',type=Path);p.add_argument('--assessment211',type=Path);p.add_argument('--out',type=Path);p.add_argument('--prereg')
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    if args.self_test:print(selftest())
    else:
        if any(getattr(args,k) is None for k in ('frozen','out','prereg')):p.error('--frozen --out --prereg required')
        run(args)
