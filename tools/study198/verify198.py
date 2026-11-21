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
CODE_DIR=Path(__file__).resolve().parent
AUDIT_DIR=None
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
    return dict(pred_bins=bin_events([r[1] for r in nodes.values()],family['nf']),nG=ng,nP=npred,nodes=nodes,edges=edges,
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
    for key,filename in (('peers','peers.py'),('freezer','freeze.py')):assert peers['source_hashes'][key]==digest(CODE_DIR/filename)
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
        assert binding['code_sha256']==digest(CODE_DIR/'physical198.py')
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
    output=Path(AUDIT_DIR or root);output.mkdir(parents=True,exist_ok=True);out=output/'independent_phase1.json';assert not out.exists() and not (output/'independent_phase1.sha256').exists(),'fresh audit destination required';out.write_text(json.dumps(report,indent=2)+'\n');(output/'independent_phase1.sha256').write_text(digest(out)+'\n');print('INDEPENDENT_PHASE1_PASS '+str(out)+' '+digest(out),flush=True)
    return report


def stage2_context(repo,root,source197):
    """Reuse only exact phase1-sealed peer features; never repeat peer IoUs."""
    import sys
    sys.path.insert(0,str(repo));V=own197_module(repo)
    prior=read_json(root/'independent_phase1.json');assert digest(root/'independent_phase1.json')==(root/'independent_phase1.sha256').read_text().strip()
    assert prior['passed'] and prior['phase2_count_allowed'] and prior['native_O_members']==60 and prior['legacy_full_MOT_graph_replays']==420
    assert V.verify_sealed(root/'frozen')==prior['peer_manifest'] and V.verify_sealed(root/'stage1')==prior['stage1_manifest']
    assert digest(root/'stage1/capacity.json')==prior['capacity_sha256']
    original=read_json(source197/'frozen/deployment.json');assert V.verify_sealed(source197/'frozen')==prior['source197_manifest']
    peers=read_json(root/'frozen/deployment.json');source_groups={g['unit']:g for g in original['groups']}
    manifest=read_json(source197/'score/manifest.json');assert digest(source197/'score/native_scores.json')==manifest['native_scores.json']
    native=read_json(source197/'score/native_scores.json');oldmetrics={(v['unit'],m['member'],a):m['metrics'] for a,videos in native.items() for v in videos for m in v['members']}
    families={};bundles={};records={};T197={}
    for h in ('35','33','W','G'):
        for item in read_json(source197/'predictions'/f'{h}.masks.json')['member_clips']:T197[item['unit'],str(item['member'])]=set(item['selected']['T'])
    for g in peers['groups']:
        source=source_groups[g['unit']]
        for m,sm in zip(g['members'],source['members']):
            key=(g['unit'],str(sm['member']))
            with np.load(source197/'frozen'/sm['rows_file'],allow_pickle=False) as z:raw=z['rows']
            C,index,blocks=V.canonicalize(raw);families[key]=build_graph_family(oldmetrics[*key,'K'],oldmetrics[*key,'A'],sm['bundle']['candidates'],blocks,g['nf'])
            bundles[key]=(raw,C,index,blocks,g,sm)
            for r in m['records']:records[r['candidate_id']]=dict(r,T197_cut=r['candidate_id'] in T197[key])
    return prior,peers,records,families,bundles,oldmetrics


def posthoc_arithmetic(root,source197,report):
    """Exact arithmetic after sealing; never modifies models or deployment."""
    import itertools
    from fractions import Fraction
    original=read_json(source197/'frozen/deployment.json');native=read_json(source197/'score/native_scores.json')
    K={v['unit']:v for v in native['K']};checked=0;groups=[];baseline_total=0;add_only_total=0;relaxed_add_total=0;total_configurations=0
    for g in original['groups']:
        for m in g['members']:
            for c in m['bundle']['candidates']:
                column=int(c['delta'])*bin_events([int(c['frame'])*int(c['direction'])],g['nf'])
                assert np.array_equal(column,c['delta_windows'])
                start=0
                for offset in range(30):
                    length=(g['nf']+25*offset+749)//750;block=column[start:start+length];start+=length
                    assert np.count_nonzero(block)==1 and int(abs(block).sum())==1 and int(block.sum())==c['delta']
                assert start==len(column);checked+=1
        if g['domain']!='35':continue
        metrics=[m['metrics'] for m in K[g['unit']]['members']];gt=np.asarray(metrics[0]['gt_bins'],np.int64)
        assert all(m['nG']==129 and np.array_equal(m['gt_bins'],gt) for m in metrics) and int(gt.sum())==30*129
        residual=(sum(np.asarray(m['pred_bins'],np.int64) for m in metrics)-3*gt).ravel()
        adds=[c for m in g['members'] for c in m['bundle']['candidates'] if c['delta']==1]
        matrix=np.column_stack([bin_events([c['frame']*c['direction']],g['nf']).ravel() for c in adds]) if adds else np.zeros((len(residual),0),np.int64)
        histogram={};best=None;witness=None
        for bits in itertools.product((0,1),repeat=len(adds)):
            value=int(abs(residual+matrix@np.asarray(bits,np.int64)).sum());histogram[value]=histogram.get(value,0)+1
            if best is None or value<best:best=value;witness=list(bits)
        assert sum(histogram.values())==2**len(adds) and best==min(histogram)
        capacity=matrix.sum(axis=1);relaxed=int((np.maximum(residual,0)+np.maximum(-residual-capacity,0)).sum());assert relaxed<=best
        relaxed_add_total+=relaxed
        N=int(abs(residual).sum());baseline_total+=N;add_only_total+=best;total_configurations+=sum(histogram.values())
        groups.append(dict(unit=g['unit'],GT=129,baseline_numerator=N,add_only_optimum_numerator=best,windowwise_add_relaxation_numerator=relaxed,
            residual=residual.tolist(),add_columns=[dict(candidate_id=c['candidate_id'],frame=c['frame'],direction=c['direction']) for c in adds],
            witness=witness,configurations=2**len(adds),objective_histogram=histogram))
    assert checked==1196 and len(groups)==4
    denominator=4*90*129;baseline=Fraction(100*baseline_total,denominator);add_floor=Fraction(100*add_only_total,denominator)
    relaxed_floor=Fraction(100*relaxed_add_total,denominator);relaxed_gain=baseline-relaxed_floor
    add_gain=baseline-add_floor;per_drop=Fraction(100*30,denominator)
    labels={};records={};masks={};operation_summary={}
    for h in ('35','33','W','G'):
        model=read_json(root/'predictions'/f'{h}.model.json')
        for r in model['training_records']:
            value=(r['y'],r['z'],r['improvement_numerator'])
            if r['key'] in labels:assert labels[r['key']]==value
            labels[r['key']]=value
    for g in read_json(root/'frozen/deployment.json')['groups']:
        for m in g['members']:
            for r in m['records']:records[r['candidate_id']]=r
    for h in ('35','33','W','G'):
        fold=read_json(root/'predictions'/f'{h}.masks.json');operation_summary[h]={}
        for arm in ('I','B'):
            ids=[cid for m in fold['member_clips'] for cid in m['selected'][arm]];masks[h,arm]=ids
            summary={}
            for delta,name in ((1,'add'),(-1,'drop')):
                chosen=[cid for cid in ids if records[cid]['delta']==delta]
                summary[name]=dict(count=len(chosen),y_sum=sum(labels[cid][0] for cid in chosen),y_positive=sum(labels[cid][0]>0 for cid in chosen),z1=sum(labels[cid][1] for cid in chosen),
                    candidates=[dict(candidate_id=cid,y=labels[cid][0],z=labels[cid][1]) for cid in chosen])
            operation_summary[h][arm]=summary
    fixed_window_operations={}
    for h in ('35','33'):
        sums={name:Fraction() for name in ('K','add_only','drop_only','all_I')}
        for g in original['groups']:
            if g['domain']!=h:continue
            metric=[m['metrics'] for m in K[g['unit']]['members']];gt=np.asarray(metric[0]['gt_bins'],np.int64);r=sum(np.asarray(m['pred_bins'],np.int64) for m in metric)-3*gt
            changes={}
            for sign,name in ((1,'add_only'),(-1,'drop_only')):
                ids=[cid for cid in masks[h,'I'] if records[cid]['unit']==g['unit'] and records[cid]['delta']==sign]
                changes[name]=sum((sign*bin_events([records[cid]['frame']*records[cid]['direction']],g['nf']) for cid in ids),np.zeros_like(r))
            den=3*int(gt.sum())
            for name,value in [('K',r),('add_only',r+changes['add_only']),('drop_only',r+changes['drop_only']),('all_I',r+changes['add_only']+changes['drop_only'])]:sums[name]+=Fraction(100*int(abs(value).sum()),den)
        sums={name:value/4 for name,value in sums.items()}
        assert abs(float(sums['all_I'])-report['means'][h]['I']['E3'])<1e-12
        fixed_window_operations[h]=dict(E3={name:float(value) for name,value in sums.items()},gain_vs_K={name:float(sums['K']-value) for name,value in sums.items() if name!='K'},interaction=float(sums['all_I']-sums['add_only']-sums['drop_only']+sums['K']),scope='Retrospective window arithmetic for the fixed sealed I decisions, not new native/F1 scoring; the two effects do not add without the displayed absolute-error interaction.')
    d=operation_summary['35']['I']['drop']['count'];line=Fraction(14,10)
    gain_upper=lambda budget:Fraction(100*(baseline_total-max(0,add_only_total-30*budget)),denominator)
    relaxed_upper=lambda budget:Fraction(100*(baseline_total-max(0,relaxed_add_total-30*budget)),denominator)
    necessary=next(budget for budget in range(1000) if gain_upper(budget)>line)
    relaxed_necessary=next(budget for budget in range(1000) if relaxed_upper(budget)>line)
    bound=dict(all_phase_single_event_columns_checked=checked,GT_per_group=[129]*4,groups=groups,total_enumerated_configurations=total_configurations,
        baseline_numerator_sum=baseline_total,add_only_optimum_numerator_sum=add_only_total,windowwise_add_relaxation_numerator_sum=relaxed_add_total,mean_denominator=denominator,
        K_fraction=[baseline.numerator,baseline.denominator],add_only_floor_fraction=[add_floor.numerator,add_floor.denominator],
        add_only_gain_fraction=[add_gain.numerator,add_gain.denominator],per_drop_gain_ceiling_fraction=[per_drop.numerator,per_drop.denominator],
        I35_drop_count=d,I35_gain_upper=float(gain_upper(d)),I35_gain_upper_fraction=[gain_upper(d).numerator,gain_upper(d).denominator],
        at14_upper=float(gain_upper(14)),at14_upper_fraction=[gain_upper(14).numerator,gain_upper(14).denominator],
        necessary_drop_count_for_strict_1_4=necessary,
        windowwise_relaxation=dict(add_floor_fraction=[relaxed_floor.numerator,relaxed_floor.denominator],add_gain_ceiling_fraction=[relaxed_gain.numerator,relaxed_gain.denominator],at_I_drop_upper_fraction=[relaxed_upper(d).numerator,relaxed_upper(d).denominator],at14_upper_fraction=[relaxed_upper(14).numerator,relaxed_upper(14).denominator],necessary_drops=relaxed_necessary,scope='Allows each window to use different additions; optimistic lower bound, not a common30phase physical optimum.'),
        proof='For any chosen additions alone L>=exact add-only minimum. At most d drops move at most30d integer L1 units, by the triangle inequality. All4groups have identical90*129 denominator; use all197 additions for an optimistic superset.',
        scope='Retrospective necessary condition only. The necessary drop count is not sufficient, and the bound omits F1. No new mask/model/threshold was deployed.')
    report.update(posthoc_I_B_add_drop=operation_summary,posthoc_fixed_I_window_operations=fixed_window_operations,posthoc_35_deletion_budget_bound=bound)
    return report


def phase2(repo,root,source197,score_root,prereg):
    """Second receipt only after a separately completed score/prediction seal."""
    import sys
    from fractions import Fraction
    sys.path.insert(0,str(repo));V=own197_module(repo)
    prior,peerdata,records,families,bundles,oldmetrics=stage2_context(repo,root,source197)
    assert prior['prereg']==peerdata['prereg']==prereg
    seals={k:V.verify_sealed(path) for k,path in (('predictions',root/'predictions'),('score',score_root))}
    receipt=read_json(root/'predictions/receipt.json');assert not receipt['stopped'] and receipt['prereg']==prereg
    assert receipt['model_code_sha256']==digest(CODE_DIR/'model198.py')
    assert receipt['guard_receipt']['guard_sha256']==prior['capacity_sha256']
    # y and z always use the complete member scene, all OTHER parents at K.
    groups={g['unit']:g for g in peerdata['groups']};baselines={key:replay_mask(f,[]) for key,f in families.items()}
    for unit,g in groups.items():
        pred=sum(baselines[unit,str(m['member'])]['pred_bins'] for m in g['members'])
        gt=bin_events([r[1] for r in families[unit,str(g['members'][0]['member'])]['gt']],g['nf']);residual=pred-3*gt;loss=int(abs(residual).sum())
        for m in g['members']:
            key=(unit,str(m['member']));base=baselines[key]['iou']['0.5']['tp']
            for r in m['records']:
                if not r['eligible']:continue
                cid=r['candidate_id'];record=records[cid];single=replay_mask(families[key],[cid])['iou']['0.5']['tp']
                on,off=(single,base) if r['delta']==1 else (base,single);z=on-off;assert z in (0,1)
                delta=int(r['delta'])*bin_events([int(r['frame'])*int(r['direction'])],g['nf']);gain=loss-int(abs(residual+delta).sum())
                record.update(y=gain/30,z=z,event_on_TP=on,event_off_TP=off,improvement_numerator=gain)
    decisions={};fit_audit={};allrecords=list(records.values())
    for target in ('35','33','W','G'):
        model=read_json(root/'predictions'/f'{target}.model.json');fold=read_json(root/'predictions'/f'{target}.masks.json')
        assert model['heldout']==fold['heldout']==target and set(model['source_domains'])=={'35','33','W','G'}-{target}
        source=[r for r in allrecords if r['hive']!=target and r['eligible']];assert {a['key'] for a in model['training_records']}=={r['candidate_id'] for r in source}
        assert model['eligible_source_rows']==len(source)
        fits={name:fit_source_scores(allrecords,target,key) for name,key in (('N','features_N11'),('B','features_B15'),('I','features_I22'))}
        for name,fit in fits.items():
            head=model['heads'][name]
            for a,b in (('mean','mean'),('scale','scale'),('coef','beta'),('intercept','intercept')):np.testing.assert_allclose(fit[a],head[b],atol=3e-9,rtol=1e-8)
            assert head['ridge']==.2
        weight={r['candidate_id']:float(w) for r,w in zip(source,fits['I']['weights'])}
        for a in model['training_records']:
            r=records[a['key']];assert r['hive']!=target and r['eligible']
            for field in ('y','z','event_on_TP','event_off_TP','improvement_numerator'):assert a[field]==r[field]
            assert abs(a['weight']-weight[r['candidate_id']])<1e-15
            for key in ('features_N11','features_B15','features_I22'):np.testing.assert_allclose(a[key],r[key],atol=2e-12,rtol=0)
        source_receipts=model['source_receipts']
        if 'GT' in source_receipts:
            expected_clips={r['clip'] for r in allrecords if r['hive']!=target}
            assert set(source_receipts['GT'])==expected_clips
            for clip,p in source_receipts['GT'].items():assert p['domain']!=target and p['role']=='source' and digest(repo/p['path'])==p['sha256']
        else:
            assert set(source_receipts['source_shards'])=={'35','33','W','G'}-{target}
            shard_root=root/'source_shards';manifest=read_json(shard_root/'manifest.json');shash=digest(shard_root/'manifest.json')
            assert shash==(shard_root/'SEALED.sha256').read_text().strip()==source_receipts['shard_manifest_sha256']
            assert manifest['prereg']==prereg and source_receipts['stage1_receipt']==manifest['stage1_receipt']
            assert manifest['stage1_receipt']['manifest_sha256']==prior['stage1_manifest'] and manifest['stage1_receipt']['capacity_sha256']==prior['capacity_sha256']
            for h,proof in source_receipts['source_shards'].items():
                assert h!=target and proof==manifest['shards'][h] and proof['file']==h+'.json' and digest(shard_root/proof['file'])==proof['sha256']
                payload=read_json(shard_root/proof['file']);assert payload['hive']==h and payload['prereg']==prereg
                assert set(payload['models'])=={g['unit'] for g in groups.values() if g['domain']==h}
                assert all(m['context']['domain']==h for members in payload['models'].values() for m in members.values())
        assert {p['unit'] for p in source_receipts['graphs']}=={g['unit'] for g in groups.values() if g['domain']!=target}
        for proof in source_receipts['graphs']:
            assert proof['domain']!=target
            g=groups[proof['unit']]
            for i,m in enumerate(g['members']):
                key=(g['unit'],str(m['member']))
                for arm in ('K','A'):
                    evidence=oldmetrics[*key,arm]['evidence']
                    assert proof['graph_inputs'][i][arm]==evidence['inputs_sha256']
                    wire=(json.dumps(evidence,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
                    assert proof['graph_sha256'][i][arm]==hashlib.sha256(wire).hexdigest()
        for item in fold['member_clips']:
            key=(item['unit'],str(item['member']));rlist=sorted([r for r in allrecords if (r['unit'],str(r['member']))==key],key=lambda r:r['candidate_id'])
            assert item['candidate_ids']==[r['candidate_id'] for r in rlist] and item['eligible']==[r['eligible'] for r in rlist]
            masks={a:[] for a in ('K','T197','Tg','N','B','I','F','Y','M')}
            for i,r in enumerate(rlist):
                cid=r['candidate_id']
                if r['T197_cut']:masks['T197'].append(cid)
                if r['T197_cut'] and r['eligible']:masks['Tg'].append(cid)
                for name,keyfeature in (('N','features_N11'),('B','features_B15'),('I','features_I22')):
                    if not r['eligible']:assert item['scores'][name][i] is None;continue
                    predicted=score_candidate(r,fits[name],keyfeature);saved=item['scores'][name][i]
                    for a,b in (('y','yhat'),('z','zhat'),('utility','utility')):assert abs(predicted[a]-saved[b])<3e-9
                    if predicted['cut']:masks[name].append(cid)
                    if name=='I':
                        if predicted['F']:masks['F'].append(cid)
                        if predicted['Y']:masks['Y'].append(cid)
            masks['M']=matched_peer_null(rlist,masks['I'])
            for arm,ids in masks.items():assert sorted(ids)==sorted(item['selected'][arm]) and item['masks'][arm]==[r['candidate_id'] in ids for r in rlist]
            decisions[key]=masks
        fit_audit[target]=dict(eligible_sources=len(source),source_domains=sorted({r['hive'] for r in source}),source_graph_units=len(source_receipts['graphs']))
    assert len(decisions)==60
    capacity=read_json(root/'stage1/capacity.json')
    for unit,g in groups.items():
        for m in g['members']:decisions[unit,str(m['member'])]['O']=capacity['units'][unit]['member_masks'][str(m['member_index'])]
    native=read_json(score_root/'native_scores.json');summary=read_json(score_root/'summary.json');physical=read_json(score_root/'physical_receipts.json')
    maskscores={};checked_graphs=0
    for arm,videos in native.items():
        for video in videos:
            for m in video['members']:
                key=(video['unit'],str(m['member']));actual=replay_mask(families[key],decisions[key][arm]);saved=m['metrics']
                assert (actual['nG'],actual['nP'])==(saved['nG'],saved['nP']) and np.array_equal(actual['pred_bins'],saved['pred_bins'])
                assert np.array_equal(bin_events([r[1] for r in families[key]['gt']],families[key]['nf']),saved['gt_bins'])
                for t in ('0.5','0.3'):assert all(actual['iou'][t][k]==saved['iou'][t][k] for k in ('tp','pred','gt'))
                maskscores[*key,arm]=actual;checked_graphs+=1
    assert checked_graphs==600
    native_checks={'I':0,'controls':0};source_gt=read_json(source197/'score/summary.json')['GT_entry'];GT={}
    for record in physical:
        key=(record['unit'],str(record['member']));arm=record['arm'];directory=score_root/record['path'];V.verify_sealed(directory)
        raw,C,index,blocks,g,sm=bundles[key];ids=decisions[key][arm]
        rows=np.loadtxt(directory/'mot.txt',delimiter=',',usecols=range(7),ndmin=2);expected=V.realize_independent(raw,C,index,blocks,sm['bundle']['candidates'],ids);assert np.array_equal(rows,expected)
        structure=V.track_structure(rows,g['polygon']);original=V.track_structure(raw,g['polygon']);assert np.array_equal(structure['frame_order'],original['frame_order'])
        saved=read_json(directory/'score.json');binding=read_json(directory/'receipt.json')['identity'];graph=maskscores[*key,arm]
        if g['video'] not in GT:
            path=repo/'runs/gt'/g['split']/g['video']/'gt/gt.txt';assert digest(path)==source_gt[g['video']]['sha256'];GT[g['video']]=np.loadtxt(path,delimiter=',',usecols=range(6),ndmin=2)
        def array_hash(value):
            a=np.ascontiguousarray(value,dtype='<f8');return hashlib.sha256(json.dumps(list(a.shape),separators=(',',':')).encode()+b'\n'+a.tobytes()).hexdigest()
        assert binding['raw_sha256']==array_hash(raw) and binding['rows_sha256']==array_hash(rows) and binding['gt_sha256']==array_hash(GT[g['video']]) and binding['selected_ids']==sorted(ids)
        assert np.array_equal(bin_events([f*s for f,s in structure['events'].values()],g['nf']),graph['pred_bins'])
        assert (graph['nG'],graph['nP'])==(saved['nG'],saved['nP'])
        for t,keyedge in (('0.5','edges05'),('0.3','edges03')):
            expected_edges=sorted([pid,gid] for pid,gids in graph['edges'][t].items() for gid in gids)
            assert expected_edges==saved['evidence'][keyedge]
            assert all(graph['iou'][t][k]==saved['iou'][t][k] for k in ('tp','pred','gt'))
        native_checks['I' if arm=='I' else 'controls']+=1
    assert native_checks=={'I':60,'controls':24}
    means={};exact_means={};aggregate_groups=0;cut_counts={}
    for h in ('35','33','W','G'):
        means[h]={};exact_means[h]={};cut_counts[h]={}
        for arm in native:
            E=[];F={t:[] for t in ('0.5','0.3')}
            for gid in sorted({g['group'] for g in groups.values() if g['domain']==h}):
                clips=[g for g in groups.values() if (g['domain'],g['group'])==(h,gid)];N=D=0
                for g in clips:
                    members=[maskscores[g['unit'],str(m['member']),arm] for m in g['members']];gb=bin_events([r[1] for r in families[g['unit'],str(g['members'][0]['member'])]['gt']],g['nf'])
                    N+=int(abs(sum(m['pred_bins'] for m in members)-3*gb).sum());D+=3*int(gb.sum())
                E.append(Fraction(100*N,D));published=next(row for row in summary['domains'][h]['methods'][arm]['groups'] if row['group']==gid);assert (N,D)==(published['numerator'],published['denominator'])
                for t in F:
                    mf=[]
                    for i in range(3):
                        rows=[maskscores[g['unit'],str(g['members'][i]['member']),arm]['iou'][t] for g in clips];tot={k:sum(r[k] for r in rows) for k in ('tp','pred','gt')};mf.append(Fraction(2*tot['tp'],max(1,tot['pred']+tot['gt'])))
                        assert all(tot[k]==published['iou'][t]['members'][i][k] for k in tot)
                    F[t].append(sum(mf)/3)
                aggregate_groups+=1
            exact_means[h][arm]=dict(E3=sum(E)/len(E),F1={t:sum(v)/len(v) for t,v in F.items()});means[h][arm]=dict(E3=float(sum(E)/len(E)),F1={t:float(sum(v)/len(v)) for t,v in F.items()})
            published=summary['domains'][h]['methods'][arm];assert abs(published['E3']-means[h][arm]['E3'])<1e-12
            for t in F:assert abs(published['F1'][t]-means[h][arm]['F1'][t])<1e-14
            cuts=sum(len(decisions[key][arm]) for key,(_,_,_,_,g,_) in bundles.items() if g['domain']==h);cut_counts[h][arm]=cuts
            if arm!='O':assert cuts==published['cut_count']
    gain={h:exact_means[h]['K']['E3']-exact_means[h]['I']['E3'] for h in means}
    investment=gain['35']>Fraction(14,10) and gain['33']>Fraction(13,10) and gain['W']>=-Fraction(14,10) and all(exact_means[h]['I']['F1']['0.5']>=exact_means[h]['K']['F1']['0.5']-Fraction(5,1000) for h in ('35','33','W'))
    attribution=all(exact_means[h]['I']['E3']<exact_means[h][other]['E3'] for h in ('35','33') for other in ('B','M'))
    assert investment==summary['investment_pass'] and attribution==summary['identity_attribution_pass']
    # Retrospective heldout-domain labels below never enter a fitted model.
    diagnostics={}
    for h in ('35','33','W','G'):
        different={'I_cut_B_keep':[],'B_cut_I_keep':[]}
        for key,masks in decisions.items():
            if bundles[key][4]['domain']!=h:continue
            for label,ids in (('I_cut_B_keep',set(masks['I'])-set(masks['B'])),('B_cut_I_keep',set(masks['B'])-set(masks['I']))):
                different[label].extend(records[cid] for cid in sorted(ids))
        diagnostics[h]={}
        for label,rows in different.items():
            diagnostics[h][label]=dict(count=len(rows),add=sum(r['delta']==1 for r in rows),drop=sum(r['delta']==-1 for r in rows),
                y_sum=sum(r['y'] for r in rows),y_mean=float(np.mean([r['y'] for r in rows])) if rows else None,
                y_positive=sum(r['y']>0 for r in rows),eventON_z1=sum(r['z']==1 for r in rows),singleton_delta_TP_sum=sum(r['delta']*r['z'] for r in rows),
                candidates=[dict(candidate_id=r['candidate_id'],delta=r['delta'],direction=r['direction'],y=r['y'],z=r['z']) for r in rows])
        diagnostics[h]['I_minus_controls']={other:dict(E3=means[h]['I']['E3']-means[h][other]['E3'],F1_05=means[h]['I']['F1']['0.5']-means[h][other]['F1']['0.5']) for other in ('B','M')}
        diagnostics[h]['scope']='Retrospective design-domain y/z at complete K, never passed into heldout fitting; singleton marginal totals need not equal batch TP/error changes.'
    report=dict(phase=2,passed=True,prereg=prereg,seals=seals,phase1_receipt_sha256=digest(root/'independent_phase1.json'),source_training=fit_audit,posthoc_I_vs_B=diagnostics,all_source_y_z_labels=True,all_four_model_heads_and_masks=True,graph_members=checked_graphs,native_sealed_matrix_checks=native_checks,reused_phase1_O_native=60,fresh_native_rematching_calls=0,aggregate_groups=aggregate_groups,cut_counts=cut_counts,means=means,investment_pass=investment,identity_attribution_pass=attribution,
                limits=['Native I60/control24 official scores are sealed producer evaluations checked by independent exact MOT/hash/rank/events and full K/A graphs; no duplicate IoU rematching.','TP_ID is not inferred by graph replay.','Stage1 screened these known design labels; stage2 is not external confirmation.'],verifier_sha256=digest(__file__))
    report=posthoc_arithmetic(root,source197,report)
    output=Path(AUDIT_DIR or root);output.mkdir(parents=True,exist_ok=True);out=output/'independent_phase2.json';assert not out.exists() and not (output/'independent_phase2.sha256').exists(),'fresh audit destination required';out.write_text(json.dumps(report,indent=2)+'\n');(output/'independent_phase2.sha256').write_text(digest(out)+'\n');print('INDEPENDENT_PHASE2_PASS '+str(out)+' '+digest(out),flush=True)
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
    p=argparse.ArgumentParser();p.add_argument('--phase',type=int,choices=(1,2));p.add_argument('--repo',type=Path);p.add_argument('--root',type=Path);p.add_argument('--source197',type=Path);p.add_argument('--capacity',type=Path);p.add_argument('--physical-batch',type=Path);p.add_argument('--physical-root',type=Path);p.add_argument('--score-root',type=Path);p.add_argument('--prereg');p.add_argument('--field-go',action='store_true');p.add_argument('--source-code-dir',type=Path);p.add_argument('--audit-out',type=Path);a=p.parse_args()
    if a.source_code_dir:CODE_DIR=a.source_code_dir.resolve()
    if a.audit_out:AUDIT_DIR=a.audit_out.resolve()
    if a.field_go:
        if a.phase==1:
            if not all((a.repo,a.root,a.source197,a.capacity,a.physical_batch,a.physical_root,a.prereg)):p.error('phase1 requires explicit paths and coordinator GO')
            phase1(a.repo.resolve(),a.root.resolve(),a.source197.resolve(),a.capacity.resolve(),a.physical_batch.resolve(),a.physical_root.resolve(),a.prereg)
        elif a.phase==2:
            if not all((a.repo,a.root,a.source197,a.prereg)):p.error('phase2 requires explicit paths and separate coordinator GO after score seal')
            phase2(a.repo.resolve(),a.root.resolve(),a.source197.resolve(),(a.score_root or a.root/'score').resolve(),a.prereg)
        else:p.error('select released phase explicitly')
    elif a.phase or a.root:p.error('field reads disabled without explicit --field-go coordinator release')
    else:print(json.dumps(synthetic(),indent=2))
