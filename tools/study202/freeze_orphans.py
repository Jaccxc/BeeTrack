"""GT-free native orphan catalog, source/synthetic only until published GO.

Historical195 s1 is a per-K-row ID vector, NOT full stage1. Full native
stage1 is checked against independent official reconstruction, not history.
No GT loader exists here. All60 individual publications are immutable/resumable.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from source_rows import stage1_sources,linked_partition
from prototype import candidates,event_change

MARKER='[orphan-continuation-v1]'
SOURCE195='4f78f24d7fd13dc2eca9b206501a5326b4e42d476e1a9c4dd1ba98fe4bf61377'
SOURCE197='e7fd48078f8d0a32bb9018f2bac93dc4b5270d374d5bcd0a631fa97dd4d790e4'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def save(path,obj):Path(path).write_text(json.dumps(plain(obj),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')


def verify(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError('input changed: '+name)
    return manifest


EXTENSION_FEATURES=('gap_over12','short_rows_over2','before','mean_prediction_error',
                    'max_prediction_error','endpoint_distance','abs_log_area_ratio',
                    'abs_log_aspect_ratio','core_fit_rms','endpoint_opposite_side')


def extension_features(core,short,side,polygon):
    from beetrack.evaluate import inside
    core=np.asarray(core,float);short=np.asarray(short,float)
    t=core[:,0];centers=core[:,2:4]+core[:,4:6]/2
    design=np.column_stack((np.ones(3),t-t.mean()))
    beta=np.linalg.lstsq(design,centers,rcond=None)[0]
    diag=float(np.median(np.hypot(core[:,4],core[:,5])))
    predicted=np.column_stack((np.ones(len(short)),short[:,0]-t.mean()))@beta
    err=np.linalg.norm(predicted-(short[:,2:4]+short[:,4:6]/2),axis=1)/diag
    a=core[0] if side=='before' else core[-1]
    b=short[-1] if side=='before' else short[0]
    gap=abs(a[0]-b[0])
    distance=np.linalg.norm((a[2:4]+a[4:6]/2)-(b[2:4]+b[4:6]/2))/diag
    features=[gap/12,len(short)/2,float(side=='before'),float(err.mean()),float(err.max()),
              float(distance),float(abs(np.log((b[4]*b[5])/(a[4]*a[5])))),
              float(abs(np.log((b[4]/b[5])/(a[4]/a[5])))),
              float(np.sqrt(np.mean(np.sum((design@beta-centers)**2,axis=1)))/diag),
              float(inside(a[None],polygon)[0]!=inside(b[None],polygon)[0])]
    if not np.isfinite(features).all():raise ValueError('invalid extension features')
    return features


def build_catalog(raw,partition,polygon,nf,unit,slot):
    """Stable197 allocated K; candidate delta from official isolated parent events."""
    from tools.study197.physical import canonicalize,realize
    from tools.study197.candidates import window_counts
    bundle=canonicalize(raw);bundle['candidates']=[]
    baseline,allocation=realize(raw,bundle,[])
    sources=partition['retained_sources']
    if sources is None:raise ValueError('original D source lineage required')
    pool=[];excluded=[]
    maxid=int(baseline[:,1].max()) if len(baseline) else 0
    for oi,old in enumerate(partition['orphans']):
        r=old['rows'].copy();r=r[np.argsort(r[:,0],kind='stable')]
        if len(r)>1 and (np.diff(r[:,0])>25).any():
            excluded.append(dict(orphan_index=oi,reason='internal official gap >25'));continue
        # Orphan temporary IDs never collide with retained stable allocator IDs.
        r[:,1]=maxid+1+oi
        pool.append(dict(rows=r,sources=old['sources'],linked_id=int(old['rows'][0,1]),original_orphan_index=oi,
                         source_s1_index=old['source_s1_index']))
    # A row-capacity uniqueness proof includes unchosen orphan pool too.
    allsources=list(map(tuple,sources))+[tuple(s) for o in pool for s in o['sources']]
    if len(set(allsources))!=len(allsources):raise ValueError('source pool not disjoint')
    edges=[]
    # Date-indexed endpoint filtering avoids crossing every recipient with all orphans.
    starts=np.array([o['rows'][0,0] for o in pool]);ends=np.array([o['rows'][-1,0] for o in pool])
    for block in bundle['parent_blocks']:
        pid=2*block['rank']+1
        parent_indices=np.asarray(bundle['source_raw_index'][block['start']:block['end']],int)
        parent=baseline[parent_indices]
        if len(parent)<3:continue
        lo,hi=parent[:,0].min(),parent[:,0].max()
        eligible=np.flatnonzero(((starts>hi)&(starts<=hi+12))|((ends<lo)&(ends>=lo-12)))
        if not len(eligible):continue
        parent_sources=[sources[j] for j in parent_indices]
        local=[pool[j] for j in eligible]
        for edge in candidates(parent,parent_sources,local):
            oi=int(eligible[edge['orphan_index']]);o=pool[oi]
            extended=np.vstack((parent,o['rows'].copy()));extended[len(parent):,1]=pid
            change=event_change(parent,extended,polygon)
            delta=window_counts(change['after'],nf)-window_counts(change['before'],nf)
            key=f'{unit}|{slot}|{block["parent_id"]}|{o["linked_id"]}|{edge["side"]}'
            ordered=parent_indices[np.argsort(baseline[parent_indices,0],kind='stable')]
            core_indices=ordered[:3] if edge['side']=='before' else ordered[-3:]
            edges.append(dict(edge,edge_id=key,orphan_index=oi,recipient=pid,
                              member_index=slot,unit=unit,
                              endpoint_K_row_index=int(core_indices[0] if edge['side']=='before' else core_indices[-1]),
                              core_K_row_indices=core_indices,
                              features10=extension_features(baseline[core_indices],o['rows'],edge['side'],polygon),
                              parent_canonical_id=block['parent_id'],parent_rank=block['rank'],
                              recipient_cut_join_key=f'{unit}|{slot}|{block["parent_id"]}',
                              orphan_linked_id=o['linked_id'],sources=o['sources'],
                              event_change=change,delta_windows=delta,
                              event_effect='unchanged' if not change['changes'] else change['changes'][0]['kind']))
    return dict(baseline=baseline,baseline_sources=sources,orphans=pool,edges=edges,
                excluded_orphans=excluded,allocation=allocation,
                constraints=dict(recipient_capacity=1,orphan_capacity=1,source_capacity=1,
                                 no_existing_K_row_or_ID_changes=True),
                parameters=dict(max_gap=12,max_extrapolation_error_core_median_diagonal=2,
                                nearest_core_rows=3,formal_status='prototype pending prereg'))


def freeze_member(repo,source195,unit,member,clip,arm,out,cut_reference):
    from beetrack import p0
    from tools.study193.freeze193 import MODEL_SHA
    repo=Path(repo);dest=Path(out)
    dp=repo/'runs/dets'/member['tag']/clip['split']/(clip['clip']+'.npy')
    source=Path(source195)/arm['rows_file'];detsha=sha(dp)
    if detsha!=arm['source_receipt']['detector_sha256']:raise ValueError('detector drift')
    linker=arm['source_receipt']['linker_name'];lp=repo/'beetrack/models'/(linker+'.json')
    if sha(lp)!=MODEL_SHA[linker] or sha(lp)!=arm['source_receipt']['linker_sha256']:raise ValueError('linker drift')
    code={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('source_rows.py'),Path(__file__).with_name('prototype.py'))}
    identity=dict(source195_rows_sha256=sha(source),detector_sha256=detsha,linker_sha256=sha(lp),code=code,
                  source197_cut_sha256=hashlib.sha256(json.dumps(cut_reference,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                  unit=unit['unit'],member_index=unit['slot'],threshold=.5 if member['hive'] in ('35','33') else .4)
    if dest.exists():
        verify(dest)
        if json.loads((dest/'receipt.json').read_text())['identity']!=identity:raise ValueError('resume inputs differ')
        return
    with np.load(source,allow_pickle=False) as z:raw=z['rows'];historical_ids=z['s1']
    traced=stage1_sources(np.load(dp,allow_pickle=False),clip['nframes'],detsha,identity['threshold'])
    partition=linked_partition(traced['rows'],p0.link_model(linker),traced['sources'],raw)
    # Tagged exact correspondence supplies original s1 IDs without bbox lookup.
    rowids=traced['rows'][partition['retained_s1_index'],1]
    if not np.array_equal(rowids,historical_ids):raise ValueError('historical per-K s1 IDs differ')
    catalog=build_catalog(raw,partition,clip['polygon'],clip['nframes'],unit['unit'],unit['slot'])
    from tools.study197.candidates import build_candidates,G_NAMES,T_NAMES
    rebuilt=build_candidates(raw,clip['polygon'],clip['nframes'])
    if cut_reference['feature_names_G']!=list(G_NAMES) or cut_reference['feature_names_T']!=list(T_NAMES):
        raise ValueError('cut feature schema drift')
    old={c['parent_id']:c for c in cut_reference['candidates']}
    if set(old)!={c['parent_id'] for c in rebuilt['candidates']}:raise ValueError('cut catalog drift')
    for c in rebuilt['candidates']:
        for key in ('delta_windows','features_G','features_T','cut_after_canonical_row','end','k','span'):
            if not np.array_equal(c[key],old[c['parent_id']][key]):raise ValueError('cut exact mismatch '+key)
    catalog.update(feature_names_extension=list(EXTENSION_FEATURES),cut_catalog=cut_reference,
                   cut_delta_and_features_exact197=True)
    if sha(dp)!=detsha or sha(source)!=identity['source195_rows_sha256']:raise ValueError('source changed during freeze')
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        np.savez(temp/'rows.npz',original_K=raw,full_stage1=traced['rows'],stage1_D_indices=traced['raw_row_indices'],
                 linked=partition['linked_rows'],linked_s1_indices=partition['source_s1_index'],baseline=catalog['baseline'])
        save(temp/'catalog.json',catalog)
        save(temp/'receipt.json',dict(identity=identity,source_det_path=dp.relative_to(repo).as_posix(),
             exact_K=True,exact_historical_s1_IDs=True,exact_native_full_stage1=True,
             historical_s1_format='per-K-row original stage1 ID, not full stage1',
             canonical_allocator='study197 rank r ->2r+1; original K retained separately',
             candidate_count=len(catalog['edges']),orphan_count=len(catalog['orphans'])))
        save(temp/'manifest.json',{p.name:sha(p) for p in temp.iterdir()})
        (temp/'SEALED.sha256').write_text(sha(temp/'manifest.json')+'\n');temp.rename(dest)
    finally:
        if temp.exists():shutil.rmtree(temp)


def materialize(group,member,cut_ids,edge_ids):
    """Explicit fullscene realization; scorer MUST rematch full GT freshly.

    member rows/raw,bundle are197; catalog is loaded catalog.json. Every action
    consumes its parent, and continuations also consume exactly one orphan.
    Native baseline row order is preserved, appended orphan order is edge-ID.
    """
    from tools.study197.physical import realize,array_sha
    catalog=member['catalog'];raw=np.asarray(member.get('raw',member.get('rows')),float)
    rows,proof=realize(raw,member['bundle'],cut_ids)
    if len(set(edge_ids))!=len(edge_ids):raise ValueError('duplicate action')
    byid={e['edge_id']:e for e in catalog['edges']}
    cuts={c['candidate_id']:c for c in member['bundle']['candidates']}
    used_parent={int(cuts[c]['parent_id']) for c in cut_ids};used_orphan=set()
    sources=list(map(tuple,catalog['baseline_sources']));parts=[rows]
    for eid in sorted(edge_ids):
        e=byid[eid];pid=int(e['parent_canonical_id']);oi=int(e['orphan_index'])
        if pid in used_parent or oi in used_orphan:raise ValueError('parent/orphan action conflict')
        used_parent.add(pid);used_orphan.add(oi)
        o=catalog['orphans'][oi];add=np.asarray(o['rows'],float).copy();add[:,1]=e['recipient']
        parts.append(add);sources.extend(map(tuple,o['sources']))
    result=np.concatenate(parts)
    if len(set(sources))!=len(sources) or len(np.unique(result[:,:2],axis=0))!=len(result):
        raise ValueError('source or track/date reused')
    return result,dict(cut_ids=sorted(cut_ids),edge_ids=sorted(edge_ids),sources=sources,
                       native_rows_sha256=array_sha(result),baseline_proof=proof,
                       fullscene_fresh_frame_matching_required=True)


def run(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('published prereg full SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('unique registered marker required: '+MARKER)
    from tools.study193.freeze193 import members,CLIPS
    if sha(args.source195/'manifest.json')!=SOURCE195:raise ValueError('wrong195 seal')
    manifest=verify(args.source195);data=json.loads((args.source195/'deployment.json').read_text())
    if sha(args.source197/'manifest.json')!=SOURCE197:raise ValueError('wrong197 seal')
    verify(args.source197);cutdata=json.loads((args.source197/'deployment.json').read_text())
    if cutdata['source195_seal']!=SOURCE195:raise ValueError('197 source mismatch')
    cuts={(g['unit'],m['member_index']):m['bundle'] for g in cutdata['groups'] for m in g['members']}
    if data['members']!=members() or len(data['units'])!=60:raise ValueError('195 allowlist changed')
    if [(c['hive'],c['split'],c['clip'],c['nframes']) for c in data['clips']]!=list(CLIPS):raise ValueError('clip whitelist changed')
    records=[]
    for group in data['groups']:
        clip=data['clips'][group['clip_index']];name=f'{group["hive"]}_g{group["group"]}_{clip["clip"]}'
        for slot,uid in enumerate(group['units']):
            u=data['units'][uid];m=data['members'][u['member']];arm=u['arms']['t50' if u['hive'] in ('35','33') else 't40']
            if m['seed']>=15 or clip['split']=='test':raise ValueError('forbidden source')
            if sha(args.source195/arm['rows_file'])!=manifest[arm['rows_file']]:raise ValueError('195 arm drift')
            token=f'u{uid:02d}';freeze_member(args.repo,args.source195,dict(unit=name,slot=slot),m,clip,arm,args.out/token,cuts[(name,slot)])
            records.append(dict(unit=name,member_index=slot,source195_unit=uid,path=token,
                                domain=group['hive'],hive=group['hive'],group=group['group'],
                                clip=clip['clip'],video=clip['clip'],split=clip['split'],
                                nf=clip['nframes'],polygon=clip['polygon'],member=str(u['member']),
                                seed=m['seed'],tag=m['tag'],
                                seal=(args.out/token/'SEALED.sha256').read_text().strip()))
            print('SEALED',token,name,slot,flush=True)
    if len(records)!=60:raise ValueError('not60')
    top=dict(schema='orphan-continuation-freeze-v1',prereg=args.prereg,marker=MARKER,members=records,
             source195_seal=SOURCE195,source197_seal=SOURCE197,feature_names_extension=list(EXTENSION_FEATURES))
    target=args.out/'deployment.json'
    if target.exists():
        if json.loads(target.read_text())!=top:raise ValueError('top resume mismatch')
    else:save(target,top)
    topmanifest={'deployment.json':sha(target)}
    for record in records:
        folder=args.out/record['path'];verify(folder)
        for file in folder.iterdir():
            if file.is_file():topmanifest[file.relative_to(args.out).as_posix()]=sha(file)
    if (args.out/'manifest.json').exists():
        if json.loads((args.out/'manifest.json').read_text())!=topmanifest:raise ValueError('top seal resume mismatch')
    else:
        save(args.out/'manifest.json',topmanifest)
    seal=sha(args.out/'manifest.json');sealpath=args.out/'SEALED.sha256'
    if sealpath.exists() and sealpath.read_text().strip()!=seal:raise ValueError('top seal mismatch')
    if not sealpath.exists():sealpath.write_text(seal+'\n')
    print('FROZEN60',target,'TOP_SEAL',seal)


def self_test():
    from source_rows import self_test as source_test
    from beetrack.p0 import stage1
    D=np.array([[f,x,0,x+4,4,.9] for f,x in [(1,0),(2,0),(3,0),(4,0),(5,0),(6,5)]],float)
    traced=stage1_sources(D,6,'synthetic')
    p=linked_partition(traced['rows'],lambda X:np.zeros(len(X)),traced['sources'])
    c=build_catalog(p['retained'],p,[(0,0),(100,0),(100,100),(0,100)],6,'toy',0)
    assert c['edges'] and len(c['orphans'])==1
    f=c['edges'][0]['features10']
    assert len(f)==10 and f[:3]==[1/12,.5,0.] and f[6:9]==[0.,0.,0.]
    assert np.isclose(f[3],5/np.sqrt(32)) and f[3]==f[4] and f[4]==f[5]
    assert np.array_equal(c['baseline'][:,[0,2,3,4,5,6]],p['retained'][:,[0,2,3,4,5,6]])
    assert len(c['baseline_sources'])==len(c['baseline'])
    from tools.study197.physical import canonicalize
    b=canonicalize(p['retained']);b['candidates']=[]
    m=dict(rows=p['retained'],bundle=b,catalog=c)
    realized,proof=materialize({},m,[],[c['edges'][0]['edge_id']])
    assert np.array_equal(realized[:len(c['baseline'])],c['baseline'])
    assert len(realized)==len(c['baseline'])+len(c['orphans'][0]['rows'])
    assert len(set(proof['sources']))==len(realized)
    return source_test()+'; catalog source capacity, stable197 allocation and official event deltas PASS'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--self-test',action='store_true')
    p.add_argument('--source195',type=Path);p.add_argument('--source197',type=Path);p.add_argument('--out',type=Path);p.add_argument('--prereg')
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    if args.self_test:print(self_test())
    else:
        if args.source195 is None or args.source197 is None or args.out is None:p.error('--source195 --source197 --out --prereg required')
        run(args)
