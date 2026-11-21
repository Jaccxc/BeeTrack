"""Independent endpoint-peer verification; phase receipts remain separate.

Default execution uses synthetic rows only. Field entry awaits coordinator GO.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib
import json
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(1)


def canonical_rows(raw):
    """Independent sorted native-ID gap runs, including invalid-size dates."""
    raw=np.asarray(raw,float);segments=[];aliases=set()
    for pid in sorted(set(raw[:,1])):
        rows=raw[raw[:,1]==pid];rows=rows[np.argsort(rows[:,0],kind='stable')]
        cuts=[0,*list(np.flatnonzero(np.diff(rows[:,0])>25)+1),len(rows)]
        for run,(lo,hi) in enumerate(zip(cuts,cuts[1:])):
            identity=int(pid+100000*run);assert identity not in aliases;aliases.add(identity)
            piece=rows[lo:hi].copy();piece[:,1]=identity;segments.append(piece)
    if not segments:return np.empty_like(raw)
    result=np.concatenate(segments);return result[np.lexsort((result[:,0],result[:,1]))]


def scalar_iou(left,right):
    a=list(map(float,left));b=list(map(float,right))
    if not np.isfinite(a+b).all() or min(a[2],a[3],b[2],b[3])<=0:return 0.
    overlap=max(0.,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0.,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    union=a[2]*a[3]+b[2]*b[3]-overlap
    return overlap/union if union>0 else 0.


def peer_dates(raw,polygon):
    rows=canonical_rows(raw);dates={};poly=np.asarray(polygon,np.float32)
    for row in rows:
        frame,pid=map(int,row[:2]);entries=dates.setdefault(frame,{})
        assert pid not in entries
        valid=bool(min(row[4:6])>0)
        side=cv2.pointPolygonTest(poly,(float(row[2]+row[4]/2),float(row[3]+row[5]/2)),False)>=0 if valid else None
        entries[pid]=dict(box=row[2:6].copy(),side=side,valid=valid)
    return dates


def best_at_date(dates,frame,own):
    entries=dates.get(int(frame),{})
    ranked=sorted(((pid,scalar_iou(own,entry['box'])) for pid,entry in entries.items() if entry['valid']),key=lambda p:(-p[1],p[0]))
    return entries,ranked


def rebuild_one_peer(short,core,dates):
    assert len(core)==3 and len(short) in (1,2)
    frames=[int(r[0]) for r in core]
    identities=sorted({pid for frame in frames for pid in dates.get(frame,{})})
    qualified=[]
    for pid in identities:
        overlaps=[]
        for row in core:
            entry=dates.get(int(row[0]),{}).get(pid)
            overlaps.append(scalar_iou(row[2:6],entry['box']) if entry and entry['valid'] else 0.)
        if sum(x>=.5 for x in overlaps)>=2:qualified.append(dict(track_id=pid,iou=overlaps,mean=sum(overlaps)/3))
    qualified.sort(key=lambda r:(-r['mean'],r['track_id']))
    best=qualified[0] if qualified else None;second=qualified[1] if len(qualified)>1 else None
    margin=best['mean']-(second['mean'] if second else 0.) if best else 0.
    identified=best is not None and margin>=.1;fixed=best['track_id'] if identified else None
    small=[]
    for row in short:
        entries,ranked=best_at_date(dates,row[0],row[2:6]);winner=ranked[0] if ranked else None;nextbest=ranked[1] if len(ranked)>1 else None
        entry=entries.get(fixed) if fixed is not None else None
        small.append(dict(frame=int(row[0]),any_best_id=winner[0] if winner else None,any_iou=winner[1] if winner else 0.,
            any_second_id=nextbest[0] if nextbest else None,any_second_iou=nextbest[1] if nextbest else 0.,
            any_side=entries[winner[0]]['side'] if winner else None,observable=bool(winner and winner[1]>=.5),missing=not bool(ranked),
            fixed_core_id=fixed,core_row_present=entry is not None,core_valid=bool(entry and entry['valid']),
            core_iou=scalar_iou(row[2:6],entry['box']) if entry and entry['valid'] else None,core_side=entry['side'] if entry else None))
    free=[]
    for row in core:
        entries,ranked=best_at_date(dates,row[0],row[2:6]);bestfree=ranked[0] if ranked else None
        free.append(dict(frame=int(row[0]),best_id=bestfree[0] if bestfree else None,iou=bestfree[1] if bestfree else 0.,side=entries[bestfree[0]]['side'] if bestfree else None,missing=not bool(ranked)))
    return dict(core_dates=frames,qualified=qualified,best_id=best['track_id'] if best else None,best_mean=best['mean'] if best else 0.,
                second_id=second['track_id'] if second else None,second_mean=second['mean'] if second else 0.,margin=margin,identified=identified,
                identified_id=fixed,short=small,free_core=free,short_observable=any(s['observable'] for s in small))


def rebuild_peer_evidence(candidate,canonical,peers,polygon):
    window=canonical[candidate['window_indices']];k=candidate['k'];short=window[:k];core=window[k:]
    records=[rebuild_one_peer(short,core,raw if isinstance(raw,dict) else peer_dates(raw,polygon)) for raw in peers]
    same=any(r['identified'] and r['short_observable'] for r in records)
    either=any(r['identified'] for r in records) and any(r['short_observable'] for r in records)
    frame_only=[any(p['short'][j]['observable'] for p in records) for j in range(k)]
    free_peer=[sum(c['iou']>=.5 for c in p['free_core'])>=2 and p['short_observable'] for p in records]
    result=dict(candidate_id=candidate['candidate_id'],parent_id=candidate['parent_id'],k=k,
                short_dates=[int(r[0]) for r in short],core_dates=[int(r[0]) for r in core],peers=records,
                gate=same,gate_same_peer=same,gate_any_peer=either,ablation_eligible=same,eligible=same,qualifies=same,
                total_short_frames=k,frame_only_observable=frame_only,frame_only_observable_count=sum(frame_only),
                frame_only_peers=free_peer,frame_only_gate=any(free_peer),usable_peer_count=sum(p['identified'] and p['short_observable'] for p in records))
    if 'features_G' in candidate:
        vectors=identity_vectors(candidate,result,candidate['short_inside'])
        result.update(C11=vectors['N11'],U4=vectors['U4'],I7=vectors['I7'],features_N11=vectors['N11'],features_B15=vectors['B15'],features_I22=vectors['I22'])
    return result


def identity_vectors(candidate,evidence,short_inside):
    usable=[p for p in evidence['peers'] if p['identified'] and p['short_observable']]
    U=[];I=[];core_side=not short_inside
    for peer in usable:
        observed=[s for s in peer['short'] if s['observable']]
        assert observed
        U.append([sum(r['iou'] for r in peer['free_core'])/3,
                  len(observed)/len(peer['short']),sum(r['any_iou'] for r in observed)/len(observed),
                  sum(r['any_side']==core_side for r in observed)/len(observed)])
        I.append([peer['best_mean'],peer['margin'],
                  sum(r['core_row_present'] for r in observed)/len(observed),
                  sum(0 if r['core_iou'] is None else r['core_iou'] for r in observed)/len(observed),
                  sum(r['core_iou'] is not None and r['core_iou']>=.5 and r['core_side']==short_inside for r in observed)/len(observed),
                  sum(r['core_iou'] is not None and r['core_iou']>=.5 and r['core_side']==core_side for r in observed)/len(observed),
                  sum(r['any_best_id']==peer['identified_id'] for r in observed)/len(observed)])
    u=np.mean(U,axis=0).tolist() if U else [0.]*4;i=np.mean(I,axis=0).tolist() if I else [0.]*7
    common=[*candidate['features_G'],float(candidate['direction']==1),len(usable)/2]
    return dict(available_peer_count=len(usable),U4=u,I7=i,N11=common,B15=common+u,I22=common+u+i)


def match_count(nodes,edges):
    owner={}
    def visit(pid,seen):
        for gid in sorted(edges.get(pid,())):
            if gid in seen:continue
            seen.add(gid)
            if gid not in owner or visit(owner[gid],seen):owner[gid]=pid;return True
        return False
    for pid in sorted(nodes):visit(pid,set())
    return len(owner)


def bin_events(signed_dates,nf):
    bins=[]
    for offset in range(30):
        local=[[0,0] for _ in range((nf+25*offset+749)//750)]
        for date in signed_dates:local[(abs(int(date))-1+25*offset)//750][int(date<0)]+=1
        bins.extend(local)
    return np.asarray(bins,np.int64)


def build_graph_family(K,A,candidates,parent_blocks,nf):
    """K/A legal graphs select one independent event ON/OFF per parent.

    This never replays TP_ID: cutting changes a segment's first/last owners.
    """
    assert K['evidence']['gt']==A['evidence']['gt']
    nodes=[{int(r[0]):r for r in m['evidence']['predictions']} for m in (K,A)]
    assert len(nodes[0])==K['nP'] and len(nodes[1])==A['nP']
    ranks={b['parent_id']:b['rank'] for b in parent_blocks};options={};onids=set()
    for c in candidates:
        rank=ranks[c['parent_id']];possible={2*rank+1,2*rank+2}
        before=possible&set(nodes[0]);after=possible&set(nodes[1])
        assert (len(before),len(after))==((0,1) if c['delta']==1 else (1,0))
        state=1 if c['delta']==1 else 0;pid=next(iter(after if state else before));onids.add(pid)
        edges={}
        for t,key in (('0.5','edges05'),('0.3','edges03')):
            edges[t]=sorted({int(gid) for p,gid in (K,A)[state]['evidence'][key] if int(p)==pid})
        assert int(nodes[state][pid][1])==int(c['frame'])*int(c['direction'])
        options[c['candidate_id']]=dict(delta=c['delta'],pid=pid,record=nodes[state][pid],edges=edges)
    fixed={pid:r for pid,r in nodes[0].items() if pid not in onids}
    assert fixed=={pid:r for pid,r in nodes[1].items() if pid not in onids}
    fixed_edges={}
    for t,key in (('0.5','edges05'),('0.3','edges03')):
        left=sorted([int(p),int(g)] for p,g in K['evidence'][key] if int(p) in fixed)
        right=sorted([int(p),int(g)] for p,g in A['evidence'][key] if int(p) in fixed)
        assert left==right;fixed_edges[t]={}
        for pid,gid in left:fixed_edges[t].setdefault(pid,[]).append(gid)
    return dict(fixed=fixed,fixed_edges=fixed_edges,options=options,gt=K['evidence']['gt'],nf=nf)


def replay_mask(family,selected):
    selected=set(selected);assert selected<=set(family['options'])
    nodes=dict(family['fixed']);edges={t:{p:list(gs) for p,gs in e.items()} for t,e in family['fixed_edges'].items()}
    for cid,c in family['options'].items():
        if (cid in selected)==(c['delta']==1):
            nodes[c['pid']]=c['record']
            for t in edges:edges[t][c['pid']]=c['edges'][t]
    ng=len(family['gt']);npred=len(nodes)
    return dict(pred_bins=bin_events([r[1] for r in nodes.values()],family['nf']),nG=ng,nP=npred,
                iou={t:dict(tp=match_count(nodes,edge),pred=npred,gt=ng) for t,edge in edges.items()})


def event_marginal_z(family,candidate_id):
    """Full-member marginal maxTP; all other candidates remain K."""
    option=family['options'][candidate_id]
    baseline=replay_mask(family,[])['iou']['0.5']['tp']
    changed=replay_mask(family,[candidate_id])['iou']['0.5']['tp']
    result=(changed-baseline)*option['delta'];assert result in (0,1)
    return result


def fit_source_scores(records,heldout,feature_key):
    dimensions={'features_N11':11,'features_B15':15,'features_I22':22}
    source=[r for r in records if r['hive']!=heldout and r['eligible']]
    d=dimensions[feature_key]
    if not source:return dict(mean=np.zeros(d),scale=np.ones(d),coef=np.zeros((d,2)),intercept=np.array([0.,.5]),weights=[],empty=True)
    hives={r['hive'] for r in source};weights=[]
    for row in source:
        groups={r['group'] for r in source if r['hive']==row['hive']}
        clips={r['clip'] for r in source if (r['hive'],r['group'])==(row['hive'],row['group'])}
        classes={r['delta'] for r in source if (r['hive'],r['group'],r['clip'])==(row['hive'],row['group'],row['clip'])}
        n=sum((r['hive'],r['group'],r['clip'],r['delta'])==(row['hive'],row['group'],row['clip'],row['delta']) for r in source)
        weights.append(1/len(hives)/len(groups)/len(clips)/len(classes)/n)
    w=np.asarray(weights);assert abs(w.sum()-1)<1e-12
    X=np.asarray([r[feature_key] for r in source],float);Y=np.asarray([[r['y'],r['z']] for r in source],float)
    mean=(w[:,None]*X).sum(axis=0);constant=np.all(X==X[0],axis=0);mean[constant]=X[0,constant]
    scale=np.sqrt((w[:,None]*(X-mean)**2).sum(axis=0));scale[(scale==0)|constant]=1
    Z=(X-mean)/scale;intercept=w@Y
    coef=np.linalg.solve(Z.T@(w[:,None]*Z)+.2*np.eye(d),Z.T@(w[:,None]*(Y-intercept)))
    for col in range(2):
        if np.all(Y[:,col]==Y[0,col]):coef[:,col]=0;intercept[col]=Y[0,col]
    gradient=2*(Z.T@(w[:,None]*(Z@coef+intercept-Y))+.2*coef)
    assert np.max(abs(gradient),initial=0)<1e-10
    return dict(mean=mean,scale=scale,coef=coef,intercept=intercept,weights=w,empty=False)


def score_candidate(record,fitted,feature_key):
    yz=fitted['intercept']+((np.asarray(record[feature_key])-fitted['mean'])/fitted['scale'])@fitted['coef']
    y=float(np.clip(yz[0],-1,1));z=float(np.clip(yz[1],0,1));identity=record['delta']*(2*z-1)
    return dict(y=y,z=z,utility=(y+identity)/2,cut=(y+identity)/2>.1,Y=y>.1,F=identity>.1)


def matched_peer_null(records,selected):
    selected=set(selected);groups={}
    for r in records:
        if r['eligible']:groups.setdefault((r['unit'],r['member_index'],r['delta'],r['direction']),[]).append(r)
    result=[]
    for pool in groups.values():
        number=sum(r['candidate_id'] in selected for r in pool)
        key=lambda r:(hashlib.sha256(('endpoint-peer-null-v1|'+r['candidate_id']).encode('utf-8')).digest(),r['candidate_id'])
        result.extend(r['candidate_id'] for r in sorted(pool,key=key)[:number])
    return sorted(result)


def own197_module(repo):
    import importlib.util
    path=repo/'tools/study197/verify.py'
    spec=importlib.util.spec_from_file_location('independent_verify197_support',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def assert_record(own,saved,path='record'):
    if isinstance(own,dict):
        for key,value in own.items():assert key in saved,path+'/'+key;assert_record(value,saved[key],path+'/'+key)
    elif isinstance(own,list):
        assert len(own)==len(saved),path
        for i,(a,b) in enumerate(zip(own,saved)):assert_record(a,b,path+'/'+str(i))
    elif isinstance(own,(float,np.floating)):
        assert saved is not None and abs(own-saved)<2e-12,(path,own,saved)
    else:assert own==saved,(path,own,saved)


def coverage_rows(rows):
    result=dict(all_candidates=len(rows),frame_only_gate_candidates=sum(bool(r['frame_only_gate']) for r in rows),
                frame_only_good_frames=sum(r['frame_only_observable_count'] for r in rows),short_frames=sum(r['k'] for r in rows))
    for sign,name in ((1,'add'),(-1,'drop')):
        result['all_'+name]=sum(r['delta']==sign for r in rows)
        result['qualified_'+name]=sum(r['delta']==sign and r['eligible'] for r in rows)
        for direction,label in ((1,'in'),(-1,'out')):
            result['all_'+name+'_'+label]=sum(r['delta']==sign and r['direction']==direction for r in rows)
            result['qualified_'+name+'_'+label]=sum(r['delta']==sign and r['direction']==direction and r['eligible'] for r in rows)
    return result


def audit_peers_and_graphs(repo,root,source197):
    """Phase1 only: fresh GT-free peer audit and historical420 graph replay."""
    import sys
    sys.path.insert(0,str(repo));V=own197_module(repo)
    peer_seal=V.verify_sealed(root/'frozen');source_seal=V.verify_sealed(source197/'frozen')
    assert source_seal=='e7fd48078f8d0a32bb9018f2bac93dc4b5270d374d5bcd0a631fa97dd4d790e4'
    peers=read_json(root/'frozen/deployment.json');original=read_json(source197/'frozen/deployment.json')
    assert peer_seal=='d53e70b757984aa42a7d2e8b6c6ea0f287bedfe564d3a6a1fde685dc2d85765f'
    assert peers['source197_seal']==source_seal and len(peers['groups'])==len(original['groups'])==20
    for key,filename in (('peers','peers.py'),('freezer','freeze.py')):assert peers['source_hashes'][key]==digest(Path(__file__).with_name(filename))
    source_manifest=read_json(source197/'score/manifest.json');assert digest(source197/'score/manifest.json')=='3b21533355dd1ba4ff441c2fd4d6bdc518c22b6b806b456e7c699fdb9a56e7d8'
    for file in ('native_scores.json','summary.json'):assert digest(source197/'score'/file)==source_manifest[file]
    scored=read_json(source197/'score/native_scores.json');oldmetrics={(v['unit'],m['member'],arm):m['metrics'] for arm,videos in scored.items() for v in videos for m in v['members']}
    oldmasks={};source_summary=read_json(source197/'score/summary.json')
    assert V.verify_sealed(source197/'predictions')=='9a726ff7c097e0f2c595c63e2a50bd4b05390006439bd12c0c539eac4a6f752f'
    for h in ('35','33','W','G'):
        for item in read_json(source197/'predictions'/f'{h}.masks.json')['member_clips']:
            for arm,mask in item['selected'].items():oldmasks[item['unit'],str(item['member']),arm]=mask
    for proof in source_summary['oracle_certificates']:
        unit=proof['unit'];g=next(g for g in original['groups'] if g['unit']==unit)
        for m in g['members']:oldmasks[unit,str(m['member']),'O']=[proof['global_to_local'][cid][1] for cid in proof['oracle']['selected_ids'] if int(proof['global_to_local'][cid][0])==m['member_index']]
    sourcegroups={g['unit']:g for g in original['groups']};assert {g['unit'] for g in peers['groups']}==set(sourcegroups)
    families={};bundles={};records={};coverage={};allids=[];replay_count=0
    for g in peers['groups']:
        source=sourcegroups[g['unit']]
        for key in ('domain','hive','group','video','clip','nf','nframes','split','polygon'):assert g[key]==source[key]
        h=g['domain'];cov=coverage.setdefault(h,dict(all=0,eligible=0,frame_only=0,add=0,drop=0,eligible_add=0,eligible_drop=0,short_frames=0,short_observable_frames=0))
        rawrows=[];canonical=[];date_maps=[]
        for m in source['members']:
            assert digest(source197/'frozen'/m['rows_file'])==m['rows_sha256']
            with np.load(source197/'frozen'/m['rows_file'],allow_pickle=False) as z:raw=z['rows']
            rawrows.append(raw);canonical.append(canonical_rows(raw));date_maps.append(peer_dates(raw,g['polygon']))
        for slot,(member,sm) in enumerate(zip(g['members'],source['members'])):
            assert member['member_index']==slot and member['member']==sm['member']
            assert len(member['records'])==len(sm['bundle']['candidates'])
            key=(g['unit'],str(sm['member']));candidates=sm['bundle']['candidates'];peer_indices=[i for i in range(3) if i!=slot]
            C,positions,blocks=V.canonicalize(rawrows[slot])
            assert np.array_equal(C,canonical[slot]);bundles[key]=(rawrows[slot],C,positions,blocks,source,sm)
            for candidate,saved in zip(candidates,member['records']):
                rebuilt=rebuild_peer_evidence(candidate,C,[date_maps[i] for i in peer_indices],g['polygon'])
                assert_record(rebuilt,saved)
                for record,peer_index in zip(saved['peers'],peer_indices):assert record['member_index']==peer_index and record['member']==source['members'][peer_index]['member']
                for name in ('frame','direction','delta','delta_windows'):assert_record(candidate[name],saved[name])
                assert saved['candidate_id']==candidate['candidate_id'];allids.append(saved['candidate_id']);records[saved['candidate_id']]=dict(saved)
                cov['all']+=1;cov['eligible']+=int(rebuilt['gate']);cov['frame_only']+=int(rebuilt['frame_only_gate'])
                label='add' if candidate['delta']==1 else 'drop';cov[label]+=1;cov['eligible_'+label]+=int(rebuilt['gate'])
                cov['short_frames']+=candidate['k'];cov['short_observable_frames']+=rebuilt['frame_only_observable_count']
            family=build_graph_family(oldmetrics[*key,'K'],oldmetrics[*key,'A'],candidates,blocks,g['nf']);families[key]=family
            for arm in ('K','A','H','G','T','M','O'):
                result=replay_mask(family,oldmasks[*key,arm]);metric=oldmetrics[*key,arm]
                assert np.array_equal(result['pred_bins'],metric['pred_bins']) and result['nG']==metric['nG'] and result['nP']==metric['nP']
                for threshold in ('0.5','0.3'):assert result['iou'][threshold]=={k:metric['iou'][threshold][k] for k in ('tp','pred','gt')}
                replay_count+=1
        print('INDEPENDENT_PEERS_AND_LEGACY_GRAPHS '+g['unit'],flush=True)
    assert len(allids)==len(set(allids))==1196 and replay_count==420
    for h,c in coverage.items():assert {k:c[k] for k in ('all','eligible','frame_only')}==peers['coverage'][h]
    coverage_groups={f"{h}/{gid}":coverage_rows([r for r in records.values() if r['hive']==h and r['group']==gid]) for h in ('35','33','W','G') for gid in sorted({r['group'] for r in records.values() if r['hive']==h})}
    return dict(coverage_by_group=coverage_groups,peer_manifest=peer_seal,source197_manifest=source_seal,peer_records=2392,candidates=1196,coverage=coverage,legacy_full_MOT_graph_replays=420,
                graph_scope='K/A official legal graphs from independently verified197; rebuild full-mask maximumTP at.5/.3 for all7legacyarms. TP_ID is deliberately not graph-replayed.'),records,families,bundles,peers


def phase1(repo,root,source197,capacity_path,physical_batch,physical_root,prereg):
    """Write phase1 receipt independently of any optional phase2 files."""
    import sys
    from fractions import Fraction
    sys.path.insert(0,str(repo))
    report,records,families,bundles,peerdata=audit_peers_and_graphs(repo,root,source197);V=own197_module(repo)
    assert peerdata['prereg']==prereg
    capacity=read_json(capacity_path);assert capacity['prereg']==prereg
    stage1_seal=V.verify_sealed(capacity_path.parent)
    units={g['unit']:g for g in peerdata['groups']};certified=[];masks={};graphs={};group_counts={};oracle_metrics={}
    for unit,g in units.items():
        row=capacity['units'][unit];candidates=[];pred=[]
        for member in g['members']:
            key=(unit,str(member['member']));pred.append(replay_mask(families[key],[])['pred_bins'])
            candidates.extend(r for r in member['records'] if records[r['candidate_id']]['eligible'])
        base=next(iter(g['members']));family=families[unit,str(base['member'])];gt_bins=bin_events([r[1] for r in family['gt']],g['nf']);pred_sum=sum(pred)
        answer=row['oracle'];proof=V.verify_oracle_exact(pred_sum,gt_bins,candidates,g['nf'],answer);certified.append(dict(unit=unit,**proof))
        per_member={}
        for m in g['members']:
            key=(unit,str(m['member']));selected=[cid for cid in answer['selected_ids'] if records[cid]['member_index']==m['member_index']]
            assert sorted(row['member_masks'][str(m['member_index'])])==sorted(selected)
            masks[key]=selected;result=replay_mask(families[key],selected);graphs[key]=result;per_member[m['member_index']]=result
        assert np.array_equal(sum(v['pred_bins'] for v in per_member.values()),answer['witness_pred_sum'])
        grouped=group_counts.setdefault((g['domain'],g['group']),dict(K=0,lower=0,upper=0,den=0))
        grouped['K']+=int(abs(pred_sum-3*gt_bins).sum());grouped['lower']+=proof['lower'];grouped['upper']+=proof['upper'];grouped['den']+=3*int(gt_bins.sum())
    batch=read_json(physical_batch);assert batch['complete_members']==60 and len(batch['members'])==60
    expected_masks={unit:row['member_masks'] for unit,row in capacity['units'].items()}
    assert batch['requested_masks_sha256']==hashlib.sha256(json.dumps(expected_masks,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    source_gt=read_json(source197/'score/summary.json')['GT_entry']
    seen=set();native_ious=0;official_calls=0;gt_cache={}
    for receipt in batch['members']:
        g=units[receipt['unit']];member=next(m for m in g['members'] if m['member_index']==receipt['member_index']);key=(g['unit'],str(member['member']))
        assert key not in seen;seen.add(key);directory=physical_root/receipt['path'];seal=V.verify_sealed(directory);assert seal==receipt['seal']
        metric=read_json(directory/'score.json');raw,C,positions,blocks,source,sm=bundles[key]
        physical=np.loadtxt(directory/'mot.txt',delimiter=',',usecols=range(7),ndmin=2)
        expected=V.realize_independent(raw,C,positions,blocks,sm['bundle']['candidates'],masks[key]);assert np.array_equal(expected,physical)
        actual=V.track_structure(physical,g['polygon']);baseline=V.track_structure(raw,g['polygon']);assert np.array_equal(actual['frame_order'],baseline['frame_order'])
        if g['video'] not in gt_cache:
            path=repo/'runs/gt'/g['split']/g['video']/'gt/gt.txt';assert digest(path)==source_gt[g['video']]['sha256']
            gt=np.loadtxt(path,delimiter=',',usecols=range(6),ndmin=2)
            truth=V.track_structure(gt,g['polygon']);gt_cache[g['video']]=(gt,truth)
        gt,truth=gt_cache[g['video']]
        def array_hash(value):
            a=np.ascontiguousarray(value,dtype='<f8');return hashlib.sha256(json.dumps(list(a.shape),separators=(',',':')).encode()+b'\n'+a.tobytes()).hexdigest()
        binding=read_json(directory/'receipt.json')['identity']
        assert binding['code_sha256']==digest(Path(__file__).with_name('physical198.py'))
        assert binding['raw_sha256']==array_hash(raw) and binding['rows_sha256']==array_hash(physical)
        assert binding['gt_sha256']==array_hash(gt) and binding['polygon_sha256']==array_hash(g['polygon']) and binding['nframes']==g['nf']
        assert binding['selected_ids']==sorted(masks[key])
        graph=graphs[key];actual_pred=bin_events([frame*sign for frame,sign in actual['events'].values()],g['nf'])
        actual_gt=bin_events([frame*sign for frame,sign in truth['events'].values()],g['nf'])
        assert np.array_equal(actual_pred,graph['pred_bins']) and np.array_equal(actual_pred,metric['pred_bins']) and np.array_equal(actual_gt,metric['gt_bins'])
        assert sorted([pid,frame*sign] for pid,(frame,sign) in actual['events'].items())==sorted([int(r[0]),int(r[1])] for r in metric['evidence']['predictions'])
        assert sorted([pid,frame*sign] for pid,(frame,sign) in truth['events'].items())==sorted(families[key]['gt'])
        assert metric['evidence']['gt']==families[key]['gt']
        assert (graph['nG'],graph['nP'])==(metric['nG'],metric['nP'])
        for t in ('0.5','0.3'):assert all(graph['iou'][t][k]==metric['iou'][t][k] for k in ('tp','pred','gt'))
        oracle_metrics[key]=graph
    assert len(seen)==60
    hives={};gate={}
    for h in ('35','33','W','G'):
        gs=sorted({g['group'] for g in units.values() if g['domain']==h});K=[];lower=[];upper=[];F={t:[] for t in ('0.5','0.3')}
        for gid in gs:
            raw=group_counts[h,gid];saved_group=capacity['groups'][f'{h}/{gid}']
            assert (raw['K'],raw['lower'],raw['upper'],raw['den'])==(saved_group['baseline'],saved_group['lower'],saved_group['upper'],saved_group['denominator'])
            K.append(Fraction(100*raw['K'],raw['den']));lower.append(Fraction(100*raw['lower'],raw['den']));upper.append(Fraction(100*raw['upper'],raw['den']))
            clips=[g for g in units.values() if (g['domain'],g['group'])==(h,gid)]
            for t in F:
                mf=[]
                for index in range(3):
                    rows=[oracle_metrics[g['unit'],str(g['members'][index]['member'])]['iou'][t] for g in clips];tp=sum(r['tp'] for r in rows);den=sum(r['pred']+r['gt'] for r in rows);mf.append(Fraction(2*tp,max(1,den)))
                    if t=='0.5':
                        native_member=saved_group['native_witness_F1']['members'][str(index)]
                        assert (native_member['tp'],native_member['pred'],native_member['gt'])==(tp,sum(r['pred'] for r in rows),sum(r['gt'] for r in rows))
                F[t].append(sum(mf)/3)
                if t=='0.5':assert abs(saved_group['native_witness_F1']['mean_member_F1']-float(sum(mf)/3))<1e-14
        baseline=sum(K)/len(K);low=sum(lower)/len(lower);up=sum(upper)/len(upper)
        for field,want in [('mean_gain_ceiling',float(baseline-low)),('mean_witness_gain',float(baseline-up)),('mean_native_witness_F1',float(sum(F['0.5'])/len(F['0.5'])))]:assert abs(capacity['hives'][h][field]-want)<1e-12
        hives[h]=dict(K_E3=float(baseline),O_lower_E3=float(low),O_witness_E3=float(up),gain_ceiling=float(baseline-low),witness_gain=float(baseline-up),F1={t:float(sum(v)/len(v)) for t,v in F.items()})
        if h in ('35','33'):
            line=Fraction(14 if h=='35' else 13,10);gate[h]=dict(witness_pass=baseline-up>line,excluded=baseline-low<=line,unknown=baseline-up<=line<baseline-low)
    allowed=all(v['witness_pass'] for v in gate.values())
    assert capacity['stage2_authorized_by_count_gate']==allowed and capacity['all60_native_graph_verified'] and capacity['physical_verified_members']==60
    expected_gate='pass' if allowed else ('fail_count_space' if any(v['excluded'] for v in gate.values()) else 'unknown_gap_stop')
    assert capacity['resource_gate']==expected_gate
    for key,c in report['coverage_by_group'].items():
        for name,value in capacity['groups'][key]['coverage'].items():assert c[name]==value
    report.update(phase=1,passed=True,prereg=prereg,stage1_manifest=stage1_seal,capacity_sha256=digest(capacity_path),physical_batch_sha256=digest(physical_batch),restricted_certificates=certified,native_O_members=60,
                  native_frame_match_calls=native_ious,direct_official_match_events_calls=official_calls,physical_scope='All60 sealed native official O scores bound to exact GT/raw/polygon/MOT hashes; independently rebuilt actual event windows and K/A full-scene graphs at.5/.3. No expensive native IoU rematching in this phase, per coordinator.',hives=hives,gate=gate,phase2_count_allowed=allowed,
                  limits=['Graph replay does not infer TP_ID.','O F1 is one physically realized count witness, not an optimum over count ties.','Repeated design scenes are not external confirmation.'],verifier_sha256=digest(__file__))
    out=root/'independent_phase1.json';out.write_text(json.dumps(report,indent=2)+'\n');(root/'independent_phase1.sha256').write_text(digest(out)+'\n');print('INDEPENDENT_PHASE1_PASS '+str(out)+' '+digest(out),flush=True)
    return report


def synthetic():
    row=lambda frame,pid:[frame,pid,0,0,4,4,.8]
    own=np.array([row(f,1) for f in range(1,5)],float);peer=np.array([row(f,9) for f in range(1,5)],float)
    polygon=[[-1,-1],[8,-1],[8,8],[-1,8]];candidate=dict(candidate_id='fixture',parent_id=1,k=1,window_indices=[0,1,2,3])
    fixed=rebuild_peer_evidence(candidate,own,[peer,peer],polygon)
    changed=peer.copy();changed[0,1]=11
    switched=rebuild_peer_evidence(candidate,own,[changed,changed],polygon)
    assert fixed['gate'] and switched['gate'] and fixed['peers'][0]['short'][0]['core_iou']==1
    assert switched['peers'][0]['short'][0]['core_iou'] is None
    missing=rebuild_peer_evidence(candidate,own,[peer[1:],peer[1:]],polygon);assert not missing['gate']
    transient=peer.copy();transient[1:,1]=[10,11,12]
    cross_peer=rebuild_peer_evidence(candidate,own,[peer[1:],transient],polygon)
    assert not cross_peer['gate'] and cross_peer['gate_any_peer'] and cross_peer['frame_only_gate']
    duplicate=peer.copy();duplicate[:,1]=10
    ambiguous=rebuild_peer_evidence(candidate,own,[np.r_[peer,duplicate],np.r_[peer,duplicate]],polygon)
    assert not ambiguous['gate'] and ambiguous['peers'][0]['margin']==0
    assert fixed['peers'][0]['free_core']==switched['peers'][0]['free_core']
    feature_candidate=dict(candidate,features_G=[0.]*9,direction=1)
    f=identity_vectors(feature_candidate,fixed,True);g=identity_vectors(feature_candidate,switched,True)
    assert f['U4']==g['U4'] and f['I7']!=g['I7'] and len(f['N11'])==11 and len(f['B15'])==15 and len(f['I22'])==22
    def metric(nodes,edges):return dict(nP=len(nodes),evidence=dict(gt=[[1,10],[2,-30],[3,-20]],predictions=[[pid,date,0,0,2,2] for pid,date in nodes],edges05=edges,edges03=edges))
    K=metric([(5,10),(7,-20)],[[5,1],[7,3]])
    A=metric([(2,15),(4,-30),(5,10)],[[2,1],[4,2],[5,1]])
    candidates=[dict(candidate_id='redundant',parent_id=10,delta=1,frame=15,direction=1),dict(candidate_id='useful',parent_id=20,delta=1,frame=30,direction=-1),dict(candidate_id='drop',parent_id=40,delta=-1,frame=20,direction=-1)]
    family=build_graph_family(K,A,candidates,[dict(parent_id=pid,rank=i) for i,pid in enumerate((10,20,30,40))],50)
    import itertools
    for bits in itertools.product((0,1),repeat=3):
        selected=[c['candidate_id'] for c,bit in zip(candidates,bits) if bit];m=replay_mask(family,selected)
        assert m['nP']==2+bits[0]+bits[1]-bits[2] and m['iou']['0.5']['tp']==2+bits[1]-bits[2]
    assert [event_marginal_z(family,c['candidate_id']) for c in candidates]==[0,1,1]
    rng=np.random.default_rng(199);training=[]
    for h in ('35','33','W','G'):
        for group in (1,2):
            for clip in ('a','b'):
                for delta in (-1,1):
                    for i in range(2 if delta<0 else 5):
                        x=rng.normal(size=22)
                        training.append(dict(hive=h,group=group,clip=clip,delta=delta,eligible=True,
                            features_N11=x[:11],features_B15=x[:15],features_I22=x,y=float(rng.choice([-1,0,1])),z=int(rng.integers(2))))
    for key in ('features_N11','features_B15','features_I22'):
        fitted=fit_source_scores(training,'35',key)
        for r in training:
            if r['hive']=='35':r.update(y=999,z=999)
        changed=fit_source_scores(training,'35',key);assert np.array_equal(fitted['coef'],changed['coef'])
        empty=fit_source_scores([],'35',key)
        assert not score_candidate(dict(training[0],delta=1),empty,key)['cut'] and not score_candidate(dict(training[0],delta=-1),empty,key)['cut']
    return dict(passed=True,core_identity_on_three_dates=True,short_identity_vs_ID_free_control=True,missing_abstains=True,margin_tie_ambiguous=True,
                U4_identity_free_I7_identity_sensitive=True,full_scene_graph_replay=True,marginal_maxTP_not_local_edge=True,
                same_peer_not_cross_peer_gate=True,balanced_add_drop_source_only_ridge=True,empty_source_neutral=True,
                scope='synthetic only; phase1/phase2 field entries await protocol and GO')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',type=int,choices=(1,2));p.add_argument('--repo',type=Path);p.add_argument('--root',type=Path);p.add_argument('--source197',type=Path);p.add_argument('--capacity',type=Path);p.add_argument('--physical-batch',type=Path);p.add_argument('--physical-root',type=Path);p.add_argument('--prereg');p.add_argument('--field-go',action='store_true');a=p.parse_args()
    if a.field_go:
        if a.phase!=1:p.error('phase2 entry is separate and awaits its own coordinator release/schema')
        if not all((a.repo,a.root,a.source197,a.capacity,a.physical_batch,a.physical_root,a.prereg)):p.error('phase1 requires explicit paths and coordinator GO')
        phase1(a.repo.resolve(),a.root.resolve(),a.source197.resolve(),a.capacity.resolve(),a.physical_batch.resolve(),a.physical_root.resolve(),a.prereg)
    elif a.phase or a.root:p.error('field reads disabled without explicit --field-go coordinator release')
    else:print(json.dumps(synthetic(),indent=2))
