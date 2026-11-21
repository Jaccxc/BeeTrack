"""GT labels for a fixed union of K plus referenced natural orphan rows.

Continuation labels belong to ONE union scene. Existing cut labels remain the
sealed199 native-K labels; they are never recomputed in the union scene.
No field access on import; real CLI is gated by published registration.
"""
import os
import sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import cv2
import numpy as np
if '--repo' in sys.argv:sys.path.insert(0,str(Path(sys.argv[sys.argv.index('--repo')+1]).resolve()))
from beetrack.evaluate import by_track,frame_match,inside,iou_matrix,load_tracks
from tools.study199.owners199 import prepare_gt,by_date,array_sha

MARKER='[orphan-continuation-v1]'
SOURCE199='0343fa154949939f59c55d7f0a350110b9f302bff9d7b5afdfd62debf13e0c9d'
DOMAINS=('35','33','W','G')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def save(path,value):Path(path).write_text(json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')


def source_key(value):
    if len(value)!=2 or not isinstance(value[0],str) or not value[0] or isinstance(value[1],bool) or int(value[1])!=value[1] or value[1]<0:
        raise ValueError('source key must be detector hash + original nonnegative row index')
    return value[0],int(value[1])


def build_union(catalog,baseline=None):
    """Stable K native order, then referenced orphan_index/date, each source once."""
    K=np.asarray(catalog['baseline'] if baseline is None else baseline,float).reshape(-1,7)
    keys=[source_key(x) for x in catalog['baseline_sources']]
    if len(K)!=len(keys) or len(set(keys))!=len(keys):raise ValueError('K row lineage invalid')
    used=sorted(set(int(e['orphan_index']) for e in catalog['edges']))
    parts=[K];orphan_sources={};seen=set(keys);track_ids=set(K[:,1])
    for oi in used:
        o=catalog['orphans'][oi];rows=np.asarray(o['rows'],float).reshape(-1,7)
        order=np.argsort(rows[:,0],kind='stable');rows=rows[order]
        sources=[source_key(o['sources'][j]) for j in order]
        if len(rows) not in (1,2) or len(set(rows[:,1]))!=1:raise ValueError('whole one/two-row orphan required')
        if rows[0,1] in track_ids or (len(rows)>1 and (np.diff(rows[:,0])<=0).any()) or (len(rows)>1 and (np.diff(rows[:,0])>25).any()):raise ValueError('orphan tracking ID/gap is not disjoint and whole')
        track_ids.add(rows[0,1])
        if len(set(sources))!=len(sources) or set(sources)&seen:raise ValueError('orphan source reused or overlaps K')
        seen.update(sources);parts.append(rows);keys.extend(sources);orphan_sources[oi]=sources
    raw=np.concatenate(parts)
    if not np.isfinite(raw).all() or (raw[:,4:6]<=0).any() or (raw[:,:2]!=np.rint(raw[:,:2])).any() or (raw[:,0]<1).any() or (raw[:,1]<0).any() or len(np.unique(raw[:,:2],axis=0))!=len(raw):raise ValueError('invalid union native rows')
    tagged=by_track(np.c_[raw,np.arange(len(raw))]);C=tagged[:,:-1];index=tagged[:,-1].astype(int)
    if not np.array_equal(by_track(C),C):raise ValueError('union canonicalization is not idempotent')
    ordered_keys=[keys[j] for j in index];lookup={k:i for i,k in enumerate(ordered_keys)}
    if len(lookup)!=len(C):raise ValueError('union source ambiguity')
    k_map=np.fromiter((lookup[k] for k in keys[:len(K)]),dtype=np.int64,count=len(K))
    columns=[0,2,3,4,5,6]
    if not np.array_equal(C[k_map][:,columns],K[:,columns]):raise ValueError('union altered K observation')
    frame_order=np.argsort(C[:,0],kind='stable')
    order_sha=lambda values:hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
    return dict(rows=C,source_keys=ordered_keys,source_lookup=lookup,
                K_raw_to_union=k_map,
                orphan_to_union={oi:[lookup[k] for k in ss] for oi,ss in orphan_sources.items()},
                proof=dict(K_rows=len(K),union_rows=len(C),referenced_orphan_indices=used,
                           distinct_orphan_rows=len(C)-len(K),union_rows_sha256=array_sha(C),
                           canonical_source_order_sha256=order_sha(ordered_keys),
                           frame_source_order_sha256=order_sha([ordered_keys[i] for i in frame_order]),
                           ordering='K native rows then referenced orphan_index/date; official by_track IDs/frame; frame groups stable'))


def isolated_owners(gt,union,indices,threshold):
    G=gt['rows'];P=union['rows'];pf=by_date(P);wanted=defaultdict(list)
    for i in sorted(set(indices)):wanted[int(P[i,0])].append(i)
    xy=lambda a:np.c_[a[:,2:4],a[:,2:4]+a[:,4:6]]
    result={}
    for date,needed in wanted.items():
        pp=pf[date];gg=gt['frames'].get(date,np.empty(0,int));T=G[gg];R=P[pp]
        M=iou_matrix(xy(T),xy(R)) if len(T) else np.zeros((0,len(R)))
        E=M>=threshold;pd=E.sum(0);gd=E.sum(1);owners=frame_match(T,R,thr=threshold)
        ppos={int(v):j for j,v in enumerate(pp)};gpos={int(v[1]):j for j,v in enumerate(T)}
        for i in needed:
            j=ppos[i];pid=int(owners[j]);q=gpos.get(pid);seg=gt['segments'].get(pid)
            isolated=bool(q is not None and pd[j]==1 and gd[q]==1)
            record=dict(union_row=int(i),source_key=list(union['source_keys'][i]),pred_key=[date,int(P[i,1])],row_sha256=array_sha(P[i]),
                 official_owner=pid if pid>=0 else None,original_gt_id=seg['original_gt_id'] if seg else None,gt_run=seg.get('run') if seg else None,
                 gt_key=[date,pid] if q is not None else None,gt_row_sha256=array_sha(T[q]) if q is not None else None,
                 gt_raw_index=int(gt['source_index'][gg[q]]) if q is not None else None,
                 iou=float(M[q,j]) if q is not None else None,pred_degree=int(pd[j]),gt_degree=int(gd[q]) if q is not None else None,
                 isolated=isolated,qualified=bool(isolated and seg['original_gt_id'] is not None),
                 pred_inside=bool(inside(P[i:i+1],gt['polygon'])[0]),gt_inside=bool(inside(T[q:q+1],gt['polygon'])[0]) if q is not None else None,
                 graph_gt=[dict(canonical_gt_id=int(T[g,1]),original_gt_id=gt['segments'][int(T[g,1])]['original_gt_id'],iou=float(M[g,j]),gt_degree=int(gd[g])) for g in np.flatnonzero(E[:,j])],
                 owner_pred_neighbors=[])
            if q is not None:
                record['owner_pred_neighbors']=[dict(union_row=int(pp[p]),source_key=list(union['source_keys'][pp[p]]),pred_key=[date,int(R[p,1])],iou=float(M[q,p]),pred_degree=int(pd[p])) for p in np.flatnonzero(E[q])]
            result[i]=record
    return result


def edge_class(endpoint,orphans):
    rows=[endpoint,*orphans]
    originals={r['original_gt_id'] for r in rows if r['qualified']}
    orphan_originals={r['original_gt_id'] for r in orphans if r['qualified']}
    orphan_runs={r['official_owner'] for r in orphans if r['qualified']}
    info=dict(orphan_original_conflict=len(orphan_originals)>1,orphan_run_conflict=len(orphan_runs)>1)
    if not all(r['qualified'] for r in rows):return dict(**info,**{'class':'U'},reason='unmatched_competing_or_ambiguous_union_owner')
    if len({r['official_owner'] for r in rows})==1:return dict(**info,**{'class':'S'},reason='same_canonical_GT_run_all_rows')
    if len(originals)>1:return dict(**info,**{'class':'D'},reason='distinct_original_GT_ID_in_new_join')
    return dict(**info,**{'class':'U'},reason='same_original_GT_ID_different_gap_runs')


def label_member(gt,catalog,polygon,context):
    prepared=gt if isinstance(gt,dict) and 'segments' in gt else prepare_gt(gt,polygon)
    K=np.asarray(catalog['baseline'],float).reshape(-1,7)
    parents=defaultdict(list)
    for index,pid in enumerate(K[:,1]):parents[int(pid)].append(index)
    endpoints={}
    for pid,indices in parents.items():
        indices=np.asarray(indices,dtype=np.int64)
        ordered=indices[np.argsort(K[indices,0],kind='stable')]
        endpoints[pid]=(int(ordered[0]),int(ordered[-1]))
    union=build_union(catalog,baseline=K);needed=set()
    for e in catalog['edges']:
        endpoint=int(e['endpoint_K_row_index']);oi=int(e['orphan_index'])
        if not 0<=endpoint<len(K):raise ValueError('recipient endpoint index invalid')
        if int(K[endpoint,1])!=int(e['recipient']):raise ValueError('edge recipient/index mismatch')
        if endpoint!=endpoints[int(e['recipient'])][0 if e['side']=='before' else 1]:raise ValueError('nearest recipient endpoint not selected')
        needed.add(int(union['K_raw_to_union'][endpoint]));needed.update(union['orphan_to_union'][oi])
    owners={str(t):isolated_owners(prepared,union,needed,t) for t in (.5,.3)}
    output=[]
    for e in catalog['edges']:
        oi=int(e['orphan_index']);endpoint=int(union['K_raw_to_union'][e['endpoint_K_row_index']]);ix=union['orphan_to_union'][oi]
        labels={}
        for t,evidence in owners.items():
            left=evidence[endpoint];right=[evidence[i] for i in ix]
            labels[t]=dict(edge_class(left,right),recipient_row=left,orphan_rows=right)
        o=catalog['orphans'][oi];orphan_key=f"{context['unit']}|{context['member_index']}|orphan:{o['linked_id']}"
        output.append(dict(e,**{k:v for k,v in context.items() if k not in e},action_id=e['edge_id'],kind='extension',parent_key=e['recipient_cut_join_key'],orphan_key=orphan_key,
                           original_orphan_key=dict(linked_id=o['linked_id'],original_orphan_index=o.get('original_orphan_index'),sources=o['sources']),labels=labels,source_scene='K_plus_all_referenced_orphans_union'))
    return dict(edges=output,union_proof=union['proof'],counts={t:dict(Counter(e['labels'][t]['class'] for e in output)) for t in ('0.5','0.3')})


def copy_cut_labels(records):
    return [dict(r,action_id='cut:'+r['candidate_id'],kind='cut',parent_key=r['candidate_id'],orphan_key='cut-dummy|'+r['candidate_id'],source_scene='sealed199_nativeK') for r in records]


def catalog_cuts(catalog):
    reference=catalog['cut_catalog']
    if not isinstance(reference,dict) or not isinstance(reference.get('candidates'),list):
        raise ValueError('cut_catalog must be the complete frozen197 bundle metadata')
    return reference['candidates']


def diagnose_censored(catalog,labels,cuts199,polygon,context):
    result=[]
    for cut in cuts199:
        census=cut['labels']['0.5']['censor']
        if not census['supported']:continue
        run=census['gt_run'];key=[context['clip'],run['original_gt_id'],run.get('run')]
        possible=[e for e in labels['edges'] if e['parent_key']==cut['candidate_id']]
        same=[];side=[];required=run['start_inside'];expected='after' if census['type']=='later_return' else 'before'
        for e in possible:
            lab=e['labels']['0.5'];rows=lab['orphan_rows']
            if lab['class']=='S' and lab['recipient_row']['official_owner']==run['canonical_gt_id']:
                same.append(e['edge_id'])
                outer=rows[-1] if e['side']=='after' else rows[0]
                if e['side']==expected and outer['gt_inside']==required and outer['pred_inside']==required:side.append(e['edge_id'])
        result.append(dict(candidate_id=cut['candidate_id'],gt_segment_key=key,original_gt_id=run['original_gt_id'],canonical_gt_id=run['canonical_gt_id'],gt_run=run.get('run'),
                           **context,censor_type=census['type'],required_inside=required,required_side=expected,
                           natural_edge_count=len(possible),same_GT_continuation_ids=same,has_same_GT_continuation=bool(same),
                           restores_required_endpoint_side_ids=side,restores_required_endpoint_side=bool(side)))
    return result


def load_domain(root,domain):
    if domain not in DOMAINS:raise ValueError('unknown domain')
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('orphan owner seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text());name=domain+'.json'
    if sha(root/name)!=manifest[name]:raise ValueError('source-domain orphan owner shard changed')
    data=json.loads((root/name).read_text())
    if data['schema']!='orphan-owner-shard-v1' or data['domain']!=domain or data['source199_seal']!=SOURCE199:raise ValueError('wrong source label shard')
    return data


def flat_actions(root,domain):return [a for u in load_domain(root,domain)['units'] for m in u['members'] for a in [*m['edges'],*m['cuts']]]


def verify_tree(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('input tree seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=digest:raise ValueError('input tree artifact changed')
    return dict(manifest_sha256=sha(root/'manifest.json'),deployment_sha256=sha(root/'deployment.json'))


def _module(name,path):
    sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run(args):
    cv2.setNumThreads(1)
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('published full prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,capture_output=True,check=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,capture_output=True,check=True).stdout.decode('utf-8')
    if log.count(MARKER)!=1:raise ValueError('unique orphan registration marker required')
    if args.out.exists():raise ValueError('fresh owner root required')
    frozen_receipt=verify_tree(args.frozen)
    frozen=json.loads((args.frozen/'deployment.json').read_text())
    if frozen['schema']!='orphan-continuation-freeze-v1' or len(frozen['members'])!=60:raise ValueError('exact native orphan F60 freeze required')
    required=('exact_K','exact_historical_s1_IDs','exact_native_full_stage1')
    identities=[]
    for ref in frozen['members']:
        child=args.frozen/ref['path']
        if (child/'SEALED.sha256').read_text().strip()!=ref['seal'] or sha(child/'manifest.json')!=ref['seal']:
            raise ValueError('individual member seal mismatch')
        receipt=json.loads((child/'receipt.json').read_text())
        if not all(receipt.get(key) is True for key in required):raise ValueError('native exactness receipt not all true')
        identities.append((ref['unit'],int(ref['member_index'])))
    if len(set(identities))!=60:raise ValueError('member identity duplicated')
    print('PREFLIGHT_NATIVE60_ALL_EXACT',frozen_receipt['manifest_sha256'],flush=True)
    if sha(args.owners199/'manifest.json')!=SOURCE199:raise ValueError('original199 owner seal changed')
    from tools.study199.owners199 import load_domain as old_domain
    old={h:old_domain(args.owners199,h) for h in DOMAINS}
    original_units={u['unit']:u for h in DOMAINS for u in old[h]['units']}
    source=_module('orphan197_source',args.repo/'tools/study197/freeze.py')
    physical=_module('orphan197_physical',args.repo/'tools/study197/physical.py')
    source_receipt=source.verify(args.source197);source_groups=source.load_groups(args.source197)
    groups={g['unit']:g for g in source_groups}
    shards={h:dict(schema='orphan-owner-shard-v1',domain=h,prereg=args.prereg,freeze_receipt=frozen_receipt,
                   frozen_manifest_sha256=frozen_receipt['manifest_sha256'],source199_seal=SOURCE199,
                   source197_receipt=source_receipt,units=[]) for h in DOMAINS}
    by_unit={};gt_by_video={};gt_hashes={};union_proofs=[];action_ids=[];census=[]
    for member_ref in frozen['members']:
        unit=member_ref['unit'];slot=int(member_ref['member_index']);g=groups[unit];m=g['members'][slot]
        directory=args.frozen/member_ref['path']
        if sha(directory/'manifest.json')!=member_ref['seal']:raise ValueError('member orphan catalog seal differs')
        # Root manifest already covers every member file; do not treat a child
        # receipt as a substitute for the complete immutable catalog.
        catalog=json.loads((directory/'catalog.json').read_text())
        expected,_=physical.realize(m['rows'],m['bundle'],[])
        if not np.array_equal(np.asarray(catalog['baseline'],float),expected):raise ValueError('catalog K differs original197 allocation')
        if g['video'] not in gt_by_video:
            path=args.repo/'runs/gt'/g['split']/g['video']/'gt/gt.txt';digest=sha(path)
            rel=path.relative_to(args.repo).as_posix()
            if old[g['domain']]['GT_sha256'][rel]!=digest:raise ValueError('GT differs sealed199 input')
            gt_hashes[rel]=digest;gt_by_video[g['video']]=prepare_gt(load_tracks(path),g['polygon'])
        context=dict(unit=unit,domain=g['domain'],hive=g['domain'],group=g['group'],video=g['video'],clip=g['video'],member=m['member'],member_index=slot,tag=m['tag'])
        result=label_member(gt_by_video[g['video']],catalog,g['polygon'],context)
        previous=next(v for v in original_units[unit]['members'] if v['member_index']==slot)
        cuts=copy_cut_labels(previous['records'])
        if {c['candidate_id'] for c in catalog_cuts(catalog)}!={c['candidate_id'] for c in cuts}:raise ValueError('cut inventory differs199')
        opportunities=diagnose_censored(catalog,result,previous['records'],g['polygon'],context)
        census.extend(opportunities);union_proofs.append(result['union_proof'])
        record=dict(member=m['member'],member_index=slot,tag=m['tag'],source_catalog_sha256=sha(directory/'catalog.json'),
                    source_catalog_seal=member_ref['seal'],edges=result['edges'],cuts=cuts,union_proof=result['union_proof'],
                    counts=result['counts'],truncation_opportunities=opportunities)
        if unit not in by_unit:
            baseline=original_units[unit]
            item={k:g[k] for k in ('unit','domain','hive','group','video','clip','nf','nframes','split','polygon')}
            item.update(members=[],gt_bins=baseline['gt_bins'],gt_count=baseline['gt_count'],pred_sum=baseline['pred_sum'])
            by_unit[unit]=item;shards[g['domain']]['units'].append(item)
        by_unit[unit]['members'].append(record)
        action_ids.extend(e['action_id'] for e in [*result['edges'],*cuts])
        print('UNION_OWNER',unit,slot,result['counts'],flush=True)
    if len(by_unit)!=20 or len(action_ids)!=len(set(action_ids)):raise ValueError('group/action identity mismatch')
    if any(sha(args.repo/name)!=digest for name,digest in gt_hashes.items()):raise ValueError('GT changed during owner extraction')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.orphan-owners-',dir=args.out.parent) as temp:
        target=Path(temp)/'payload';target.mkdir()
        for h,data in shards.items():
            for u in data['units']:u['members'].sort(key=lambda x:x['member_index'])
            data['GT_sha256']=gt_hashes;data['code_sha256']=sha(__file__);save(target/(h+'.json'),data)
        unique_gt={tuple(x['gt_segment_key']) for x in census}
        save(target/'summary.json',dict(schema='orphan-owner-summary-v1',prereg=args.prereg,freeze_receipt=frozen_receipt,source199_seal=SOURCE199,
              GT_sha256=gt_hashes,code_sha256=sha(__file__),members=60,groups=20,actions=len(action_ids),
              truncation_opportunities=len(census),unique_GT_segments=len(unique_gt),
              same_GT_continuation_opportunities=sum(x['has_same_GT_continuation'] for x in census),
              restored_required_side_opportunities=sum(x['restores_required_endpoint_side'] for x in census),
              unique_GT_with_same_continuation=len({tuple(x['gt_segment_key']) for x in census if x['has_same_GT_continuation']}),
              unique_GT_with_required_side=len({tuple(x['gt_segment_key']) for x in census if x['restores_required_endpoint_side']})))
        save(target/'manifest.json',{p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file()})
        (target/'SEALED.sha256').write_text(sha(target/'manifest.json')+'\n');target.rename(args.out)
    print('ORPHAN_OWNERS_SEALED',sha(args.out/'manifest.json'),flush=True)


def self_test():
    poly=[[0,0],[10,0],[10,10],[0,10]];row=lambda f,p,x:[f,p,x-1,4,2,2,.8]
    truth=np.array([row(f,1,x) for f,x in enumerate((12,8,7,6,12),1)],float)
    K=truth[:4].copy();K[:,1]=1;orphan=truth[4:].copy();orphan[:,1]=9
    edge=dict(edge_id='toy|0|7|90|after',orphan_index=0,recipient=1,parent_canonical_id=7,endpoint_K_row_index=3,side='after',recipient_cut_join_key='toy|0|7')
    catalog=dict(baseline=K,baseline_sources=[('toy',i) for i in range(4)],orphans=[dict(rows=orphan,sources=[('toy',4)],linked_id=90)],edges=[edge])
    context=dict(unit='toy',member_index=0,member='0',clip='synthetic',domain='W',hive='W',group=1)
    label=label_member(truth,catalog,poly,context)
    assert label['edges'][0]['labels']['0.5']['class']=='S' and label['union_proof']['union_rows']==5
    twice=dict(catalog,edges=[edge,dict(edge,edge_id='second-edge')])
    assert label_member(truth,twice,poly,context)['union_proof']['union_rows']==5
    Dtruth=truth.copy();Dtruth[-1,1]=2
    assert label_member(Dtruth,catalog,poly,context)['edges'][0]['labels']['0.5']['class']=='D'
    extra=orphan.copy();extra[:,1]=10
    rival=dict(rows=extra,sources=[('toy',5)],linked_id=91)
    # Unreferenced pool rows are not part of the fixed matching scene.
    pool=dict(catalog,orphans=[catalog['orphans'][0],rival])
    assert label_member(truth,pool,poly,context)['edges'][0]['labels']['0.5']['class']=='S'
    pool['edges']=[edge,dict(edge,edge_id='rival-edge',orphan_index=1)]
    result=label_member(truth,pool,poly,context)
    assert all(e['labels']['0.5']['class']=='U' for e in result['edges'])
    reverse=result['edges'][0]['labels']['0.5']['orphan_rows'][0]['owner_pred_neighbors']
    assert len(reverse)==2
    L=dict(qualified=True,official_owner=1,original_gt_id=1)
    R=dict(qualified=True,official_owner=100001,original_gt_id=1)
    assert edge_class(L,[R])['class']=='U'
    R['original_gt_id']=2
    assert edge_class(L,[L,R])['class']=='D' and edge_class(L,[L,R])['orphan_original_conflict']
    assert catalog_cuts(dict(cut_catalog=dict(candidates=[dict(candidate_id='cut-test')],feature_names_G=[])))[0]['candidate_id']=='cut-test'
    cut=copy_cut_labels([dict(candidate_id='cut-test',labels={})])[0]
    assert cut['orphan_key']=='cut-dummy|cut-test' and cut['parent_key']=='cut-test'
    print('Union orphan owners synthetic source dedup, S/D/U, fullscene competition/reverse graph and gap-run passed')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',required=True,type=Path);ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--frozen',type=Path);ap.add_argument('--source197',type=Path);ap.add_argument('--owners199',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--prereg')
    ap.add_argument('--wait-for-seal',action='store_true',help='wait read-only for the producer top seal; then run once or fail')
    args=ap.parse_args()
    if args.self_test:self_test()
    else:
        if any(getattr(args,k) is None for k in ('frozen','source197','owners199','out','prereg')):ap.error('--frozen --source197 --owners199 --out --prereg required')
        args.repo=args.repo.resolve();args.frozen=args.frozen.resolve();args.source197=args.source197.resolve();args.owners199=args.owners199.resolve();args.out=args.out.resolve()
        if args.wait_for_seal:
            print('WAITING_FOR_FROZEN_TOP_SEAL',args.frozen/'SEALED.sha256',flush=True)
            while not (args.frozen/'SEALED.sha256').is_file():time.sleep(30)
        run(args)
