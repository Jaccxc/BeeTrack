"""Verify §117 full-frame ownership and causal donors on the §116 event complement."""
import argparse
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


ARMS=('U','A','Z','Q','Zm','D')
TAUS=(.5,.3)


def require(ok,message):
    if not ok:
        raise ValueError(message)


def gt_roster(gt,polygon,n,excluded):
    from beetrack.evaluate import by_track, zone_tracks
    a=by_track(np.asarray(gt,dtype=float).reshape(-1,6))
    require(len(a)==len(set(map(tuple,a[:,:2]))) and np.all((a[:,0]>=1)&(a[:,0]<=n)),'Invalid GT frame or ID')
    by_frame=defaultdict(list)
    for f,k,x,y,w,h in a:
        by_frame[int(f)].append((int(k),[x,y,x+w,y+h]))
    events=[]
    for k,(direction,_,_,changes) in zone_tracks(a,polygon).items():
        if direction:
            i=next(i for i,d in reversed(changes) if d==direction)
            events.append(dict(id=k,dir=direction,before=int(a[i-1,0]),arrival=int(a[i,0]),excluded=k in excluded))
    require(set(excluded)=={e['id'] for e in events if e['excluded']},'Previously opened IDs outside source event universe')
    return by_frame,events


def xy(rows):
    return np.asarray(rows,dtype=float).reshape(-1,4)


def overlap_components(a,b):
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(axis=2)
    area_a=np.prod(a[:,2:]-a[:,:2],axis=1)
    area_b=np.prod(b[:,2:]-b[:,:2],axis=1)
    return inter,area_a[:,None]+area_b[None,:]-inter


def side(poly,box):
    return cv2.pointPolygonTest(poly,((box[0]+box[2])/2,(box[1]+box[3])/2),False)>=0


def frame_boxes(truth):
    return [k for k,_ in truth],xy([box for _,box in truth])


def unique_owner(box,truth):
    """Top GT overlaps of one pre-NMS native box in the coordinates supplied by the caller."""
    ids,gt=frame_boxes(truth)
    inter,union=overlap_components(xy([box]),gt)
    ratios=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)[0]
    order=sorted(range(len(truth)),key=lambda i:(-ratios[i],ids[i]))[:2]
    best=[[ids[i],float(ratios[i])] for i in order]
    return best


def owner_status(best):
    if not best or best[0][1]<.3:
        return 'missing',None
    if len(best)>1 and best[1][1]>=.3:
        return 'ambiguous',None
    if best[0][1]<.5:
        return 'weak',None
    return 'unique',best[0][0]


def joint_owner(a,b):
    sa,ia=owner_status(a); sb,ib=owner_status(b)
    if sa==sb=='unique':
        return ('same',ia) if ia==ib else ('other',None)
    return next(k for k in ('ambiguous','missing','weak') if k in (sa,sb)),None


def assignment(ids,boxes,truth,polygon):
    keys,gt=frame_boxes(truth)
    predictions=xy([boxes[str(k)] for k in ids])
    # Official frame_match converts output to MOT xywh and back before its IoU computation.
    canonical=predictions.copy()
    canonical[:,2:]=predictions[:,:2]+(predictions[:,2:]-predictions[:,:2])
    inter,union=overlap_components(gt,canonical)
    q=inter/(union+1e-9)
    g,p=linear_sum_assignment(-q)
    return [[keys[i],ids[j],float(q[i,j]),side(polygon,predictions[j])==side(polygon,gt[i])]
            for i,j in zip(g,p) if q[i,j]>=.3]


def clipped(box,hw):
    h,w=hw
    a=np.asarray(box,float)
    require(a.shape==(4,) and np.isfinite(a).all() and np.all(a[2:]>=a[:2]),'Inverted or nonfinite raw receiver/donor box')
    return np.clip(a,[0,0,0,0],[w,h,w,h]).tolist()


def validate_raster(record,ids):
    h,w=record['source_hw']; oh,ow=record['input_bchw'][-2:]; rows,cols=record['p3_hw']
    require(min(w,h,rows,cols)>0 and (oh,ow)==(8*rows,8*cols),'Invalid P3 or source geometry')
    r=min(ow/w,oh/h)
    nh,nw=max(1,round(h*r)),max(1,round(w*r))
    left,top=(ow-nw)//2,(oh-nh)//2
    def inside(k):
        y,x=divmod(k,cols)
        return left<=8*x+4<left+nw and top<=8*y+4<top+nh
    require(all(map(inside,ids)),'P3 receiver or donor is in letterbox padding')
    if 'raster' in record:
        valid_cols=sum(left<=8*x+4<left+nw for x in range(cols))
        valid_rows=sum(top<=8*y+4<top+nh for y in range(rows))
        require(record['raster']==dict(pad_xy=[left,top],resized_hw=[nh,nw],valid_cells=valid_cols*valid_rows),
                'Sealed P3 padding or valid-cell count differs')


def build_evidence(records,gt,polygon,excluded,meta):
    n=meta['nframes']
    by_frame,events=gt_roster(gt,polygon,n,set(excluded))
    prior=deque(maxlen=2)
    frames=[]
    poly=np.asarray(polygon,dtype=np.float32)
    for f,r in enumerate(records,1):
        require(f==r['frame'] and f<=n and r['donor_frame']==(None if f<3 else f-2),'Invalid temporal history')
        truth=by_frame.get(f,[])
        boxes=r['boxes']; wide=r['p3_hw'][1]; cells=wide*r['p3_hw'][0]
        require(r['input_bchw'][-2:]==[r['p3_hw'][0]*8,wide*8],'P3 grid or stride differs')
        require(set(r['arms'])==set(ARMS),'Six post-NMS arms required')
        rawcoords=xy(list(boxes.values()))
        h,w=r['source_hw']
        require(np.isfinite(rawcoords).all() and np.all(rawcoords>=0) and np.all(rawcoords<=[w,h,w,h])
                and np.all(rawcoords[:,2:]>=rawcoords[:,:2]),'Invalid clipped NMS input box')
        cache={}; pools=[]; lookup={}; survivors={}
        for arm in ARMS:
            pool=r['arms'][arm]
            require(all(type(k) is int and k>=0 and 0.05-1e-7<=score<=1 for k,score in pool),
                    'Nonfinite or invalid survivor score/anchor')
            ordered=sorted(((-score,*boxes[str(k)],k) for k,score in pool if score>=.25))
            keys=tuple(row[-1] for row in ordered)
            require(len(keys)==len(set(keys)) and len(pool)==len(set(k for k,_ in pool)),'Repeated native NMS anchor')
            if keys not in cache:
                cache[keys]=len(pools)
                pools.append(dict(ids=list(keys),matches=assignment(keys,boxes,truth,poly)))
            lookup[arm]=cache[keys]
            survivors[arm]=dict(pool)
        eligible=r['eligible']
        cur=[q['current_id'] for q in eligible]; donor=[q['donor_id'] for q in eligible]; z=r['z_ids']
        require(cur==sorted(set(cur)) and len(donor)==len(set(donor)) and z==sorted(set(z)),
                'Receiver order, zero-motion order or reciprocal uniqueness changed')
        require(all(type(k) is int and 0<=k<cells for k in (*cur,*donor,*z)), 'Invalid P3 anchor')
        validate_raster(r,(*cur,*donor,*z))
        if f<3:
            require(not cur and not z and all(r['arms'][name]==r['arms']['U'] for name in ARMS),'Cold frame changed')
        if not cur:
            require(all(r['arms'][name]==r['arms']['U'] for name in ('A','Q','Zm','D')),'No A receiver but treatment changed')
        if len(cur)<2:
            require(r['arms']['D']==r['arms']['A'],'A singleton cyclic permutation changed detector output')
        if not z:
            require(r['arms']['Z']==r['arms']['U'],'Zero-motion output changed without a Z receiver')
        if cur==donor:
            require(r['arms']['Zm']==r['arms']['A'],'All donors stationary but same-cell control differs')
        checked=[]
        for i,row in enumerate(eligible):
            q=row['current_id']; p=row['donor_id']; fs=row['scores']
            require(set(fs)==set(ARMS) and all(0<=score<=1 for score in fs.values()) and
                    .05<=fs['U']<.25 and .25<=row['donor_score']<=1,
                    'Native weak receiver or strong original donor outside score gate')
            dy,dx=q//wide-p//wide,q%wide-p%wide
            require(max(abs(dx),abs(dy))<=2 and row['moved']==(p!=q),'Donor is outside the 5x5 P3 search window')
            a=clipped(row['current_raw_box'],r['source_hw']); b=clipped(row['donor_raw_box'],r['source_hw'])
            require(a==boxes[str(q)] and b==row['donor_box'],'Pre-NMS donor or receiver clipping changed')
            # Where the old native survivor is present, verify both source geometry and unmodified confidence.
            oldboxes,oldscores=prior[0]
            if str(p) in oldboxes:
                require(b==oldboxes[str(p)],'Donor box is not the native t-2 anchor box')
            if p in oldscores:
                require(abs(oldscores[p]-row['donor_score'])<=1e-6,'Donor confidence differs from native t-2')
            oldtruth=by_frame.get(f-2,[])
            aowner=unique_owner(row['current_raw_box'],truth)
            bowner=unique_owner(row['donor_raw_box'],oldtruth)
            ac=aowner if a==row['current_raw_box'] else unique_owner(a,truth)
            bc=bowner if b==row['donor_raw_box'] else unique_owner(b,oldtruth)
            for arm in ARMS:
                require(fs[arm]+1e-7>=fs['U'],'Temporal confidence fell below native confidence')
                if q in survivors[arm]:
                    require(abs(survivors[arm][q]-fs[arm])<=1e-6,'Surviving receiver confidence mismatch')
            checked.append(dict(current=q,donor=p,D_donor=donor[i-1],scores=fs,donor_score=row['donor_score'],
                current_raw=row['current_raw_box'],donor_raw=row['donor_raw_box'],current_box=a,donor_box=b,
                current_top2=aowner,donor_top2=bowner,current_top2_clipped=ac,donor_top2_clipped=bc,
                retained=[arm for arm in ARMS if q in survivors[arm]]))
        runtime={k:r[k] for k in ('input_bchw','source_hw','p3_hw','p3_channels','raster','timing_ms',
                 'backbone_forwards','process_rss_bytes','process_peak_rss_bytes','history_native_bytes',
                 'history_diagnostic_boxes_bytes','stock_u_exact') if k in r}
        frames.append(dict(f=f,donor_frame=r['donor_frame'],gt=[k for k,_ in truth],
                           pools=pools,arms=lookup,eligible=checked,z_ids=z,runtime=runtime))
        prior.append((boxes,survivors['U']))
    require(len(frames)==n,'Incomplete detector stream')
    return dict(format='temporal117-donor-audit-v1',owner_geometry='unclipped_primary_clipped_sensitivity',
                meta=meta,events=events,frames=frames)


def _replay(evidence,use_clipped):
    require(evidence['format']=='temporal117-donor-audit-v1','Unknown donor evidence format')
    events=[e for e in evidence['events'] if not e['excluded']]
    lookup=defaultdict(list)
    for e in events:
        lookup[e['before']].append((e['id'],'before'))
        lookup[e['arrival']].append((e['id'],'arrival'))
    valid={a:{str(t):{} for t in TAUS} for a in ARMS}
    totals={a:{str(t):Counter(tp=0,fp=0,fn=0) for t in TAUS} for a in ARMS}
    moved_cert={}; provenance=Counter(); moved_owner=Counter()
    stage_owners={stage:{motion:Counter(same=0,other=0,unknown=0) for motion in ('moved','stationary')}
                  for stage in ('eligible','raised','crossed_025','survived_025','claimed_05','correct_side_05')}
    changes={a:Counter(gained=0,lost=0,eligible_boosted=0,eligible_crossed=0) for a in ARMS}
    for fr in evidence['frames']:
        f=fr['f']; n_gt=len(fr['gt']); pools=fr['pools']; idx=fr['arms']
        baseline=set(pools[idx['U']]['ids'])
        eligible={}
        for x in fr['eligible']:
            if use_clipped:
                kind,ident=joint_owner(x['current_top2_clipped'],x['donor_top2_clipped'])
            else:
                kind,ident=joint_owner(x['current_top2'],x['donor_top2'])
            provenance[kind]+=1
            k=x['current']
            motion='moved' if k!=x['donor'] else 'stationary'
            category=kind if kind in ('same','other') else 'unknown'
            a_pool=pools[idx['A']]
            assigned=next((m for m in a_pool['matches'] if m[1]==k and m[2]>=.5),None)
            flags=(True,x['scores']['A']>x['scores']['U'],x['scores']['A']>=.25,
                   k in a_pool['ids'],assigned is not None,assigned is not None and assigned[3])
            for stage,flag in zip(stage_owners,flags):
                if flag:
                    stage_owners[stage][motion][category]+=1
            if x['current']!=x['donor']:
                moved_owner[kind]+=1
                if ident is not None:
                    eligible[x['current']]=ident
            for a in ARMS:
                changes[a]['eligible_boosted']+=int(x['scores'][a]>x['scores']['U'])
                changes[a]['eligible_crossed']+=int(x['scores'][a]>=.25)
        provenance['z']+=len(fr['z_ids'])
        provenance['permutable_frames']+=int(len(fr['eligible'])>=2)
        for a in ARMS:
            data=pools[idx[a]]; now=set(data['ids'])
            changes[a]['gained']+=len(now-baseline)
            changes[a]['lost']+=len(baseline-now)
            for tau in TAUS:
                label=str(tau)
                links={gt:(k,ok) for gt,k,iou,ok in data['matches'] if iou>=tau}
                n_tp=len(links); row=totals[a][label]
                row['tp']+=n_tp; row['fp']+=len(now)-n_tp; row['fn']+=n_gt-n_tp
                for gt,side in lookup.get(f,()):
                    k,ok=links.get(gt,(None,False))
                    valid[a][label][gt,side]=bool(ok)
                    if a=='A' and tau==.5 and ok and eligible.get(k)==gt:
                        moved_cert[gt,side]=dict(frame=f,anchor=k)
    result={}
    for tau in TAUS:
        label=str(tau)
        visibility={}; gained={}; lost={}; ids_for_arm={}
        for a in ARMS:
            ends=valid[a][label]
            flags={e['id']:[ends[e['id'],'before'],ends[e['id'],'arrival']] for e in events}
            ids={k for k,values in flags.items() if all(values)}
            ids_for_arm[a]=ids
            visibility[a]=dict(before=sum(v[0] for v in flags.values()),arrival=sum(v[1] for v in flags.values()),
                               both=len(ids),ids=sorted(ids),by_direction={d:sum(e['id'] in ids for e in events if e['dir']==d)
                                                                         for d in ('in','out')},
                               endpoint_ids={side:sorted(k for k,v in flags.items() if v[j])
                                             for j,side in enumerate(('before','arrival'))},
                               direction_endpoints={d:{side:sum(flags[e['id']][j] for e in events if e['dir']==d)
                                             for j,side in enumerate(('before','arrival'))} for d in ('in','out')})
            gained[a]=sorted(ids-ids_for_arm['U']); lost[a]=sorted(ids_for_arm['U']-ids)
        cert=[]
        if tau==.5:
            for gt in gained['A']:
                cert.extend(dict(id=gt,end=side,**moved_cert[gt,side]) for side in ('before','arrival')
                            if not valid['U'][label][gt,side] and (gt,side) in moved_cert)
        u,a=totals['U'][label],totals['A'][label]
        dtp,dfp=a['tp']-u['tp'],a['fp']-u['fp']
        result[label]=dict(visible=visibility,gained=gained,lost=lost,qualifying=cert,
            frame_counts={name:dict(totals[name][label]) for name in ARMS},delta_tp=dtp,delta_fp=dfp)
    primary=result['0.5']
    passed=(all(primary['visible']['A']['both']>primary['visible'][a]['both'] for a in ('U','Z','Q'))
            and bool(primary['qualifying']) and primary['delta_tp']>max(0,primary['delta_fp']))
    return dict(meta=evidence['meta'],roster_size=len(events),roster_ids=sorted(e['id'] for e in events),
                excluded_ids=sorted(e['id'] for e in evidence['events'] if e['excluded']),thresholds=result,
                eligible=sum(len(f['eligible']) for f in evidence['frames']),ownership=dict(provenance),
                moved_ownership=dict(moved_owner),owner_by_stage=stage_owners,
                changes={k:dict(v) for k,v in changes.items()},seed_pass=passed)


def replay_mechanism(evidence):
    require(evidence.get('owner_geometry')=='unclipped_primary_clipped_sensitivity','Unspecified donor identity coordinates')
    main=_replay(evidence,False)
    secondary=_replay(evidence,True)
    main['identity_coordinates']='inverse_letterbox_unclipped'
    main['clipped_sensitivity']=dict(identity_coordinates='camera_clipped',ownership=secondary['ownership'],
        moved_ownership=secondary['moved_ownership'],owner_by_stage=secondary['owner_by_stage'],
        qualifying=secondary['thresholds']['0.5']['qualifying'],seed_pass=secondary['seed_pass'])
    return main


def aggregate(summaries):
    """Each input is one seed on the same physical clip. Preserve inter-seed overlap."""
    require(len(summaries)==3 and len({r['meta']['clip'] for r in summaries})==1 and
            sorted(r['meta']['seed'] for r in summaries)==[0,1,2], 'Need three seeds from one source')
    require(all(r['roster_ids']==summaries[0]['roster_ids'] for r in summaries),'Physical population differs by seed')
    overlap={}
    for arm in ARMS:
        sources=[set(r['thresholds']['0.5']['visible'][arm]['ids']) for r in summaries]
        overlap[arm]=dict(union=sorted(set.union(*sources)),intersection=sorted(set.intersection(*sources)))
    recover=[{v['id'] for v in r['thresholds']['0.5']['qualifying']} for r in summaries]
    secondary=[{v['id'] for v in r['clipped_sensitivity']['qualifying']} for r in summaries]
    sp_seeds=sorted(r['meta']['seed'] for r in summaries if r['clipped_sensitivity']['seed_pass'])
    return dict(clip=summaries[0]['meta']['clip'],n=summaries[0]['roster_size'],visible=overlap,
                qualifying_union=sorted(set.union(*recover)),qualifying_intersection=sorted(set.intersection(*recover)),
                passed_seeds=sorted(r['meta']['seed'] for r in summaries if r['seed_pass']),
                source_pass=sum(r['seed_pass'] for r in summaries)>=2,
                clipped_sensitivity=dict(qualifying_union=sorted(set.union(*secondary)),
                    qualifying_intersection=sorted(set.intersection(*secondary)),passed_seeds=sp_seeds,
                    source_pass=len(sp_seeds)>=2))


def wire(obj):
    return (json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=False)+'\n').encode()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=wire(obj)
    if path.suffix=='.gz':
        raw=gzip.compress(raw,compresslevel=5,mtime=0)
    checksum=hashlib.sha256(raw).hexdigest()
    with NamedTemporaryFile(dir=path.parent,delete=False) as candidate:
        candidate.write(raw)
    temp=Path(candidate.name)
    try:
        try:
            os.link(temp,path)
        except FileExistsError:
            require(file_hash(path)==checksum,f'Conflicting immutable evidence at {path}')
    finally:
        temp.unlink()
    return checksum


def roster116(repo):
    historical=repo/'docs/study116_replay.json.gz'
    bundle=json.loads(gzip.decompress(historical.read_bytes()))
    sys.path.insert(0,str((repo/'tools/study116').resolve()))
    from release116 import scalar_codec
    require(bundle['prereg']=='c8fabf6c55c9737086b02050fabb3360bb5f1db2','Unexpected §116 registration')
    original={}
    for clip,c in bundle['clips'].items():
        labels=[]
        for e in scalar_codec(c['events'],inverse=True):
            tag=f"beetrack-warp-pilot-v1|{clip}|{e['dir']}|{e['gt_tid']}|{e['arrival']}"
            require(hashlib.sha256(tag.encode()).hexdigest()==e['rank'],'Invalid §116 rank')
            labels.append([e['rank'],e['gt_tid'],e['arrival'],e['dir'],e['before']])
        original[clip]=dict(gt_sha=scalar_codec(c['sources_sha256']['gt_mot'],inverse=True),total=c['population']['available'],
                            selected=sorted(labels),frames=sorted({f for e in labels for f in (e[2],e[4])}))
    board='3f144a920a99ef438bc2552284f77bc8cab579c891e3db956280402ded62566e'
    require(hashlib.sha256(wire(original)).hexdigest()==board,'§116 frozen exclusion roster mismatch')
    return original,board


def real_run(root,repo,detector_sha,tracking_sha,prereg):
    from store117 import iter_records
    detector=root/'detector_done.json'
    require(file_hash(detector)==detector_sha,'Global six-case detector seal mismatch')
    index=json.loads(detector.read_bytes())
    require(index['study']==117 and index['prereg']==prereg and index['backbone_forwards']==6000,
            'Incomplete §117 detector seal')
    tracker=root/'tracking/index.json'
    require(file_hash(tracker)==tracking_sha,'Global 36-MOT tracking fingerprint mismatch')
    tracked=json.loads(tracker.read_bytes())
    expected={(fold,seed,arm) for fold in (33,35) for seed in range(3) for arm in ARMS}
    require(tracked['stage']=='predictions_only' and tracked['prereg']==prereg and
            tracked['detector_index_sha256']==detector_sha and len(tracked['cases'])==36 and
            {(c['fold'],c['seed'],c['arm']) for c in tracked['cases']}==expected,'Tracking dependency or coverage differs')
    roster,prior_sha=roster116(repo)
    zones=repo/'beetrack/zones.json'
    zone_sha=file_hash(zones)
    require(zone_sha=='fa89f3f7fe2d5f9b9825407641e5d3f10e692c0e8f08d83630aa4cebc6c13212','Historical polygons changed')
    polygon=json.loads(zones.read_bytes())
    dest=root/'mechanism'
    case_index={}; folds={}
    for fold in (33,35):
        clip=f'BEE24-{68-fold}-tail'
        old=roster[clip]; gt_path=repo/'runs/gt/val'/clip/'gt/gt.txt'
        require(file_hash(gt_path)==old['gt_sha'] and len(old['selected'])==16,'Historical tail GT or exclusion size mismatch')
        gt=np.loadtxt(gt_path,delimiter=',',ndmin=2,usecols=range(6))
        cohort={e[1] for e in old['selected']}
        summaries=[]
        for seed in range(3):
            name=f'{fold}_{seed}'; db=f'cases/{name}.sqlite'; case=f'cases/{name}.json'
            require(file_hash(root/db)==index['files'][db] and file_hash(root/case)==index['files'][case],
                    f'Unsealed source case {name}')
            m=json.loads((root/case).read_bytes())['config']
            require((m['fold'],m['seed'],m['clip'],m['nframes'],m['prereg'])==(fold,seed,clip,1000,prereg),
                    'Detector/camera/seed provenance mismatch')
            meta=dict(clip=clip,fold=fold,seed=seed,nframes=1000,prereg=prereg,source_gt_sha256=old['gt_sha'],
                      detector_index_sha256=detector_sha,tracking_index_sha256=tracking_sha,
                      sqlite_sha256=index['files'][db],roster116_sha256=prior_sha,
                      zones_sha256=zone_sha,cpu=m['cpu'])
            file=dest/(name+'.json.gz'); score=dest/(name+'.summary.json'); receipt=dest/(name+'.done.json')
            if receipt.is_file():
                done=json.loads(receipt.read_bytes())
                require(done['meta']==meta and file_hash(file)==done['sha256'] and file_hash(score)==done['summary_sha256'],
                        'Incomplete or changed mechanism case')
                summary=json.loads(score.read_bytes())
            else:
                ev=build_evidence(iter_records(root/db),gt,polygon[clip],cohort,meta)
                require(len(ev['events'])==old['total'] and
                        sorted((e['id'],e['arrival'],e['dir'],e['before']) for e in ev['events'] if e['excluded'])==
                        sorted(tuple(x[1:]) for x in old['selected']),'Final roster differs from pinned §116')
                summary=replay_mechanism(ev)
                done=dict(meta=meta,sha256=save(file,ev),summary_sha256=save(score,summary))
                save(receipt,done)
                del ev
            summaries.append(summary)
            case_index[name]=dict(file=file.name,sha256=done['sha256'],summary_file=score.name,
                                   summary_sha256=done['summary_sha256'],receipt_sha256=file_hash(receipt))
            print('Verified',name,summary['roster_size'],flush=True)
        folds[str(fold)]=aggregate(summaries)
    print(save(dest/'index.json',dict(study=117,prereg=prereg,detector_index_sha256=detector_sha,tracking_index_sha256=tracking_sha,
               cases=case_index,folds=folds)),flush=True)


def self_check():
    from copy import deepcopy
    from beetrack.evaluate import by_track,frame_match
    poly=[[10,0],[62,0],[62,62],[10,62]]
    left=[2,10,8,16]; right=[12,10,18,16]; neighbor=[26,20,30,24]
    rows=lambda f,d,origin:dict(frame=f,donor_frame=None if f<3 else f-2,source_hw=[64,64],
        p3_hw=[8,8],input_bchw=[1,3,64,64],boxes=d,arms={k:list(origin) for k in ARMS},eligible=[],z_ids=[])
    frames=[rows(1,{'8':left,'19':neighbor},[[8,.9],[19,.8]]),
            rows(2,{'5':left,'19':neighbor},[[5,.8],[19,.8]]),
            rows(3,{'9':right,'19':neighbor},[[19,.8],[9,.15]])]
    scores=dict.fromkeys(ARMS,.15); scores['A']=scores['D']=.65
    frames[2]['arms']['A']=[[19,.8],[9,.65]]
    frames[2]['arms']['D']=deepcopy(frames[2]['arms']['A'])
    frames[2]['eligible']=[dict(current_id=9,donor_id=8,moved=True,current_raw_box=right,
                                donor_raw_box=left,donor_box=left,donor_score=.9,scores=scores)]
    gt=np.array([[f,k,b[0],b[1],b[2]-b[0],b[3]-b[1]] for f in range(1,4)
                 for k,b in ((1,right if f==3 else left),(2,neighbor))])
    meta=dict(clip='synthetic',seed=0,nframes=3)
    def score(r=frames,g=gt,excluded=()):
        return replay_mechanism(build_evidence(r,g,poly,excluded,meta))
    good=score(); primary=good['thresholds']['0.5']
    assert good['seed_pass'] and primary['gained']['A']==[1] and primary['qualifying']==[dict(id=1,end='arrival',frame=3,anchor=9)]
    assert primary['delta_tp']==1 and primary['delta_fp']==0 and not score(excluded=[1])['seed_pass']
    other=deepcopy(frames)
    other[0]['boxes']['8']=neighbor
    other[2]['eligible'][0].update(donor_raw_box=neighbor,donor_box=neighbor)
    assert not score(other)['seed_pass'] and score(other)['ownership']['other']==1
    no=deepcopy(frames)
    no[0]['boxes']['8']=[0,0,0,0]
    no[2]['eligible'][0].update(donor_raw_box=[-2,-2,0,0],donor_box=[0,0,0,0])
    assert score(no)['ownership']['missing']==1
    stationary=deepcopy(frames)
    stationary[0]['boxes']['9']=stationary[0]['boxes'].pop('8')
    for a in ARMS:
        stationary[0]['arms'][a][0][0]=9
    stationary[2]['eligible'][0].update(donor_id=9,moved=False)
    stationary[2]['eligible'][0]['scores']['Zm']=.65
    stationary[2]['arms']['Zm']=deepcopy(stationary[2]['arms']['A'])
    assert not score(stationary)['seed_pass']
    crowded=np.vstack((gt,[1,3,2,10,6,6]))
    assert not score(g=crowded)['seed_pass'] and score(g=crowded)['ownership']['ambiguous']==1
    # Same actual NMS rows and full-frame visibility on both branches. The raw donor
    # extends left of the sensor and fails IoU .5; its clipped counterpart fits the bee.
    border=deepcopy(frames)
    border[0]['boxes']['8']=[0,10,8,16]
    border[2]['eligible'][0].update(donor_raw_box=[-15,10,8,16],donor_box=[0,10,8,16])
    edge=score(border)
    assert edge['thresholds']['0.5']['gained']['A']==[1] and not edge['seed_pass']
    assert edge['thresholds']['0.5']['qualifying']==[] and edge['ownership']['missing']==1
    assert edge['clipped_sensitivity']['seed_pass'] and edge['clipped_sensitivity']['qualifying'][0]['id']==1
    assert owner_status([[1,.5],[2,.3]])==('ambiguous',None) and owner_status([[1,.5],[2,.299]])==('unique',1)
    wrong_anchor=deepcopy(frames)
    wrong_anchor[2]['boxes']['20']=right
    wrong_anchor[2]['arms']['A']=[[19,.8],[20,.67],[9,.65]]
    wrong_anchor[2]['arms']['D']=deepcopy(wrong_anchor[2]['arms']['A'])
    temp=score(wrong_anchor)
    assert temp['thresholds']['0.5']['visible']['A']['both']==1 and not temp['seed_pass']
    for tau in TAUS:
        f=np.asarray([[3,19,*neighbor[:2],4,4],[3,9,*right[:2],6,6]],float)
        ours=assignment([19,9],frames[2]['boxes'],[(1,right),(2,neighbor)],np.asarray(poly,np.float32))
        got={k:g for g,k,quality,_ in ours if quality>=tau}
        expected=frame_match(by_track(gt),f,thr=tau)
        assert got=={int(row[1]):int(y) for row,y in zip(f,expected) if y>=0}
    tampered=deepcopy(frames);tampered[2]['donor_frame']=2
    try:
        score(tampered)
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong historical frame accepted')
    s1,s2=deepcopy(good),deepcopy(good);s1['meta']['seed']=1;s2['meta']['seed']=2
    assert aggregate([s2,good,s1])['qualifying_union']==[1]
    border_seeds=[{**edge,'meta':{**edge['meta'],'seed':seed}} for seed in range(3)]
    source=aggregate(border_seeds)
    assert not source['source_pass'] and source['passed_seeds']==[]
    assert source['clipped_sensitivity']['source_pass'] and source['clipped_sensitivity']['passed_seeds']==[0,1,2]
    print('Verified source complement, identity competition, wrong/ambiguous/missing/stationary donors, '
          'wrong surviving anchor, two-frame history, both IoUs, whole-frame net gate, cross-seed identity, '
          'and an out-of-bounds donor with different raw/clipped seed and source decisions.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--self-check',action='store_true')
    action.add_argument('--run',type=Path)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--detector-sha')
    parser.add_argument('--tracking-sha')
    parser.add_argument('--prereg')
    parser.add_argument('--store-dir',type=Path,default=Path(__file__).resolve().parent/'acquisition')
    args=parser.parse_args()
    sys.path[:0]=[str(args.repo.resolve()),str(args.store_dir.resolve())]
    if args.self_check:
        self_check()
    else:
        require(args.detector_sha and args.tracking_sha and args.prereg,
                'Supply globally sealed detector and tracker fingerprints and preregistration revision')
        real_run(args.run,args.repo,args.detector_sha,args.tracking_sha,args.prereg)
