"""Native full-scene and single-suffix isolated owners, with source-only shards.

The real CLI requires the published203 registration and complete input seal.
Return probes remain in a separate file excluded from the source-label loader.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import importlib.util
import os
import sys
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.dont_write_bytecode=True
import json
import hashlib
import subprocess
import re
import tempfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from beetrack.evaluate import by_track, inside, iou_matrix
from tools.study199.owners199 import prepare_gt, array_sha, event_times,by_date

_spec=importlib.util.spec_from_file_location('orphan_owner_core',Path.cwd()/'tools/study202/owners_orphans.py')
_old=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)


def scene(raw,source_keys):
    """Canonicalize the complete native scene, retaining a bijective row map."""
    raw=np.asarray(raw,float).reshape(-1,7)
    keys=[_old.source_key(k) for k in source_keys]
    if len(raw)!=len(keys) or len(keys)!=len(set(keys)):
        raise ValueError('scene must have unique detector-source rows')
    if not np.isfinite(raw).all() or (raw[:,4:6]<=0).any():
        raise ValueError('invalid native boxes')
    tagged=by_track(np.c_[raw,np.arange(len(raw))])
    rows=tagged[:,:-1];order=tagged[:,-1].astype(int)
    inverse=np.empty(len(order),dtype=int);inverse[order]=np.arange(len(order))
    return dict(rows=rows,source_keys=[keys[i] for i in order],raw_to_canonical=inverse,frames=by_date(rows),owner_cache={},
                proof=dict(rows_sha256=array_sha(rows),native_rows_sha256=array_sha(raw),
                           row_order='official by_track of complete native scene',rows=len(rows)))


def owners(prepared_gt,prepared_scene,native_indices,threshold):
    mapping=prepared_scene['raw_to_canonical']
    indices=[int(i) for i in native_indices]
    if any(i<0 or i>=len(mapping) for i in indices):raise ValueError('native row index outside scene')
    cache=prepared_scene.setdefault('owner_cache',{}).setdefault(str(threshold),{})
    wanted=[int(mapping[i]) for i in indices];missing=[i for i in wanted if i not in cache]
    if missing:cache.update(_old.isolated_owners(prepared_gt,prepared_scene,missing,threshold))
    return [dict(cache[int(mapping[i])],native_row=i,pred_row=prepared_scene['rows'][mapping[i]].tolist()) for i in indices]


def inserted_suffix_owners(gt,host_scene,candidate,threshold):
    """Rematch only inserted dates with every original K box at those dates."""
    suffix=np.asarray(candidate['suffix_rows'],float).reshape(-1,7).copy()
    last_anchor=int(host_scene['raw_to_canonical'][candidate['host_anchor_K_indices'][-1]])
    suffix[:,1]=host_scene['rows'][last_anchor,1]
    if not (suffix[:,0]>host_scene['rows'][last_anchor,0]).all():raise ValueError('suffix must follow both anchors')
    parts=[];keys=[];suffix_positions=[]
    for row,key in zip(suffix,candidate['suffix_source_keys']):
        ix=host_scene['frames'].get(int(row[0]),np.empty(0,int))
        native=host_scene['rows'][ix]
        if np.any(native[:,1]==row[1]):raise ValueError('suffix overwrites existing host row')
        start=len(keys);block=np.r_[native,row[None]]
        order=np.argsort(block[:,1],kind='stable');parts.append(block[order])
        bk=[host_scene['source_keys'][i] for i in ix]+[_old.source_key(key)]
        keys.extend(bk[i] for i in order);suffix_positions.append(start+int(np.flatnonzero(order==len(native))[0]))
    compact=dict(rows=np.concatenate(parts),source_keys=keys)
    evidence=_old.isolated_owners(gt,compact,suffix_positions,threshold)
    return [dict(evidence[i],pred_row=compact['rows'][i].tolist()) for i in suffix_positions]


def label_natural(gt,scenes,candidate):
    hscene=scenes[candidate['host_member_index']];dscene=scenes[candidate['donor_member_index']]
    labels={}
    for t in (.5,.3):
        h=owners(gt,hscene,candidate['host_anchor_K_indices'],t)
        d=owners(gt,dscene,candidate['donor_anchor_K_indices']+candidate['donor_suffix_K_indices'],t)
        inserted=inserted_suffix_owners(gt,hscene,candidate,t)
        formal=classify(h+d+h+inserted)
        labels[str(t)]=dict(**{'class':formal.pop('classification')},**formal,
            native_class=classify(h+d)['classification'],insert_class=classify(h+inserted)['classification'],
            original_host_rows=h,original_donor_rows=d,inserted_host_rows=h,inserted_suffix_rows=inserted,
            inserted_anchor_proof='unchanged dates before forward suffix; native full-scene match reused')
    return dict(candidate,labels=labels)


def classify(rows):
    """Conservative complete-evidence taxonomy; ambiguity always remains U."""
    if not rows or not all(r['qualified'] for r in rows):
        return dict(classification='U',reason='incomplete_or_competing_owner')
    if len({r['official_owner'] for r in rows})==1:
        return dict(classification='S',reason='all_same_canonical_GT_run')
    if len({r['original_gt_id'] for r in rows})>1:
        return dict(classification='D',reason='complete_distinct_original_GT_IDs')
    return dict(classification='U',reason='same_original_ID_distinct_gap_runs')


def label_candidate(gt,host_scene,donor_scene,inserted_scene,candidate):
    """Only this candidate's suffix may appear in inserted_scene.

    The materializer supplies native indices after appending the suffix; all
    complete host K rows still participate in each inserted-frame assignment.
    """
    output={}
    host_indices=candidate['host_anchor_K_indices']
    donor_indices=candidate['donor_anchor_K_indices']+candidate['donor_suffix_K_indices']
    insert_indices=candidate['insert_host_anchor_indices']+candidate['insert_suffix_indices']
    if len(host_indices)!=2 or len(candidate['donor_anchor_K_indices'])!=2:
        raise ValueError('exactly two host/donor anchors required')
    for threshold in (.5,.3):
        h=owners(gt,host_scene,host_indices,threshold)
        d=owners(gt,donor_scene,donor_indices,threshold)
        inserted=owners(gt,inserted_scene,insert_indices,threshold)
        native=classify(h+d);after=classify(inserted);formal=classify(h+d+inserted)
        output[str(threshold)]=dict(**{'class':formal.pop('classification')},**formal,
            native_class=native['classification'],insert_class=after['classification'],
            original_host_rows=h,original_donor_rows=d,inserted_rows=inserted)
    return dict(candidate_id=candidate['candidate_id'],action_id=candidate['candidate_id'],labels=output)


def host_event_supervision(gt,host_scene,start_and_two_end_indices,threshold):
    """An empty event list is valid when all three anchors certify one run."""
    evidence=owners(gt,host_scene,start_and_two_end_indices,threshold)
    label=classify(evidence)
    if label['classification']!='S':
        return dict(valid=False,reason=label['reason'],anchors=evidence,events=None,
                    missing_owner_rows=sum(r['official_owner'] is None for r in evidence),
                    competing_rows=sum(r['official_owner'] is not None and not r['isolated'] for r in evidence))
    pid=evidence[0]['official_owner'];seg=gt['segments'][pid]
    rows=gt['rows'][seg['start']:seg['end']]
    return dict(valid=True,anchors=evidence,canonical_gt_id=pid,
                original_gt_id=seg['original_gt_id'],gt_run=seg['run'],
                first_frame=seg['first_frame'],last_frame=seg['last_frame'],
                events=event_times(rows,gt['polygon']),gt_run_rows=rows.tolist(),gt_run_rows_sha256=array_sha(rows))


def return_probes(gt,scenes):
    """One first qualifying observed return per full GT run, independent of K.

    These are descriptive probes, never source-training labels/features.
    """
    if len(scenes)!=3:raise ValueError('exactly three original K scenes required')
    output=[]
    xy=lambda rows:np.c_[rows[:,2:4],rows[:,2:4]+rows[:,4:6]]
    for pid,seg in gt['segments'].items():
        rows=gt['rows'][seg['start']:seg['end']]
        sides=inside(rows,gt['polygon']);initial=bool(sides[0]);chosen=None
        crossings=np.flatnonzero(sides[1:]!=sides[:-1])+1
        for pos,j in enumerate(crossings):
            if bool(sides[j])==initial or pos+1==len(crossings):continue
            r=int(crossings[pos+1])
            if bool(sides[r])==initial and rows[r,0]-rows[j,0]<=25:
                chosen=(int(j),r);break
        if chosen is None:continue
        j,r=chosen;date=int(rows[r,0]);gt_index=seg['start']+r
        proof=dict(canonical_gt_id=int(pid),original_gt_id=seg['original_gt_id'],gt_run=seg.get('run'),
            departure_frame=int(rows[j,0]),return_frame=date,initial_inside=initial,
            full_GT_run_rows=rows.tolist(),full_GT_run_sha256=array_sha(rows),
            departure_previous_row=rows[j-1].tolist(),departure_row=rows[j].tolist(),
            return_previous_row=rows[r-1].tolist(),return_row=rows[r].tolist(),thresholds={})
        G=gt['rows'];gg=gt['frames'][date];gpos=int(np.flatnonzero(gg==gt_index)[0])
        for threshold in (.5,.3):
            members=[]
            for slot,s in enumerate(scenes):
                P=s['rows'];pp=np.flatnonzero(P[:,0]==date)
                M=iou_matrix(xy(G[gg]),xy(P[pp])) if len(pp) else np.zeros((len(gg),0))
                neighbors=pp[np.flatnonzero(M[gpos]>=threshold)]
                evidence=_old.isolated_owners(gt,s,[int(i) for i in neighbors],threshold)
                confirmed=any(x['qualified'] and x['official_owner']==pid for x in evidence.values())
                members.append(dict(member_index=slot,has_geometric_box=bool(len(neighbors)),
                    confirmed_same_GT=confirmed,GT_degree=int(len(neighbors)),
                    neighbor_rows=list(evidence.values())))
            proof['thresholds'][str(threshold)]=dict(members=members,
                geometric_members=sum(m['has_geometric_box'] for m in members),
                confirmed_members=sum(m['confirmed_same_GT'] for m in members))
        output.append(proof)
    return output


def selftest():
    polygon=np.array([[0,0],[100,0],[100,100],[0,100]],np.float32)
    raw=np.array([[f,1,10+f,10,4,4,.9] for f in range(1,6)],float)
    gt=prepare_gt(raw[:,:6],polygon);s=scene(raw,[('synthetic',i) for i in range(5)])
    for threshold in (.5,.3):
        r=owners(gt,s,[0,3,4],threshold)
        assert classify(r)['classification']=='S'
        assert host_event_supervision(gt,s,[0,3,4],threshold)['events']==[]
        competing=np.r_[raw,[[5,2,15,10,4,4,.8]]]
        t=scene(competing,[('synthetic',i) for i in range(6)])
        assert classify(owners(gt,t,[0,3,4],threshold))['classification']=='U'
    host=scene(raw[:3],[('host',i) for i in range(3)])
    donor=scene(raw,[('donor',i) for i in range(5)])
    inserted=scene(raw,[('host',i) if i<3 else ('donor',i) for i in range(5)])
    c=dict(candidate_id='synthetic',host_anchor_K_indices=[1,2],donor_anchor_K_indices=[1,2],
           donor_suffix_K_indices=[3,4],insert_host_anchor_indices=[1,2],insert_suffix_indices=[3,4])
    assert all(v['class']=='S' for v in label_candidate(gt,host,donor,inserted,c)['labels'].values())
    c.update(host_member_index=0,donor_member_index=1,suffix_rows=raw[3:].tolist(),
             suffix_source_keys=[('donor',3),('donor',4)])
    compact=label_natural(gt,[host,donor,donor],c)
    assert all(v['class']=='S' for v in compact['labels'].values())
    for t in (.5,.3):
        full=owners(gt,inserted,[3,4],t);short=inserted_suffix_owners(gt,host,c,t)
        for a,b in zip(full,short):
            for key in ('official_owner','original_gt_id','gt_run','pred_degree','gt_degree','iou','qualified'):
                assert a[key]==b[key],key
    d=[dict(qualified=True,official_owner=1,original_gt_id=1),dict(qualified=True,official_owner=2,original_gt_id=2)]
    assert classify(d)['classification']=='D'
    d[1]['original_gt_id']=1
    assert classify(d)['classification']=='U'
    turn=np.array([[1,1,10,10,4,4,.9],[2,1,110,10,4,4,.9],[3,1,10,10,4,4,.9]],float)
    tg=prepare_gt(turn[:,:6],polygon);ts=scene(turn,[('turn',i) for i in range(3)])
    probes=return_probes(tg,[ts,ts,scene(np.empty((0,7)),[])])
    assert len(probes)==1 and probes[0]['return_frame']==3
    assert all(p['geometric_members']==p['confirmed_members']==2 for p in probes[0]['thresholds'].values())
    print('OWNERS203_SYNTHETIC_PASS')


MARKER='[peer-suffix-observed-v1]'
PREREG='69d52924d596d74c56edd25f1bc9bb7c8e66edb4'
DOMAINS=('35','33','W','G')
GT_SHA256={
    'BEE24-33-head':'96bde4a21985ae03e592eb535cb4980106de5f6260b2ea9dc5a75dabe3da6b68',
    'BEE24-35-head':'4a6d01a55c52046bfcd1f617bb5dcab5ded79c2cc28f1ebba8ea8b66ba2adef6',
    'BEE24-10':'9e8972b068cf1e755e5cf7dba44e04d7ef98c671f36e56fb40103a8d60e188af',
    'BEE24-15':'b8eb32ee611db6541b13b5405987208e9e9d93a968700873e686fb82015f3b1d',
    'BEE24-24':'2e7a033c68d4b7e0e4db07fbe8ab213f03a239ad4c7a712c3bcb5be1a48cf71b',
    'BEE24-28':'cbd1624258acedfd8d7a706edd25fee24a772b2ba556dd94d3f4bcc8032fae88'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_domain(root,domain):
    if domain not in DOMAINS:raise ValueError('invalid source domain')
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('owner top seal')
    manifest=json.loads((root/'manifest.json').read_text());name=domain+'.json'
    if sha(root/name)!=manifest[name]:raise ValueError('owner shard hash')
    data=json.loads((root/name).read_text())
    if data['domain']!=domain or data['prereg']!=PREREG:raise ValueError('owner shard identity')
    return data


def run(args):
    from beetrack.evaluate import load_tracks
    from tools.study197.candidates import window_counts
    spec=importlib.util.spec_from_file_location('suffix203freeze',Path(__file__).with_name('freeze203.py'))
    freezer=importlib.util.module_from_spec(spec);spec.loader.exec_module(freezer)
    if args.prereg!=PREREG:raise ValueError('wrong published registration')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('unique203 marker required')
    if args.out.exists() or args.out.resolve().drive.upper()!='D:':raise ValueError('fresh D: output required')
    freezer.verify(args.frozen);top=json.loads((args.frozen/'deployment.json').read_text())
    if top['prereg']!=PREREG or top['source202']!=freezer.SOURCE202 or len(top['groups'])!=20:raise ValueError('incomplete203 freeze')
    receipt=dict(manifest_sha256=sha(args.frozen/'manifest.json'),deployment_sha256=sha(args.frozen/'deployment.json'))
    shards={h:dict(schema='peer-suffix203-owner-shard-v1',domain=h,prereg=PREREG,freeze_receipt=receipt,
                   frozen_manifest_sha256=receipt['manifest_sha256'],units=[]) for h in DOMAINS}
    gt_cache={};gt_hashes={};probes=[];actions=[];member_count=0
    for ref in top['groups']:
        directory=args.frozen/ref['path']
        if sha(directory/'manifest.json')!=ref['seal']:raise ValueError('group seal differs')
        c=json.loads((directory/'catalog.json').read_text());meta=c['members']
        if [m['member_index'] for m in meta]!=[0,1,2]:raise ValueError('three ordered members required')
        with np.load(directory/'K.npz',allow_pickle=False) as z:raws=[z['k'+str(i)] for i in range(3)]
        scenes=[scene(raw,m['sources']) for raw,m in zip(raws,meta)];member_count+=3
        domain=c['domain'];video=c['video'];split='dev' if video.endswith('-head') else 'val'
        path=args.repo/'runs/gt'/split/video/'gt/gt.txt'
        if video not in gt_cache:
            rel=path.relative_to(args.repo).as_posix();gt_hashes[rel]=sha(path)
            if gt_hashes[rel]!=GT_SHA256.get(video):raise ValueError('GT not identical to authorized old six-clip input')
            gt_cache[video]=prepare_gt(load_tracks(path),c['polygon'])
        gt=gt_cache[video];nf=c['nf']
        needed=[set() for _ in range(3)]
        for candidate in c['candidates']:
            needed[candidate['host_member_index']].update(candidate['host_anchor_K_indices'])
            needed[candidate['donor_member_index']].update(candidate['donor_anchor_K_indices']+candidate['donor_suffix_K_indices'])
        for host in c['host_contexts']:
            if not host['option_ids']:continue
            slot=host['host_member_index'];ix=np.flatnonzero(raws[slot][:,1]==host['host_parent_id'])
            ix=ix[np.argsort(raws[slot][ix,0],kind='stable')];needed[slot].update(map(int,[ix[0],ix[-2],ix[-1]]))
        for slot in range(3):
            for t in (.5,.3):owners(gt,scenes[slot],sorted(needed[slot]),t)
        unit={k:c[k] for k in ('unit','domain','group','video','nf','polygon')}
        unit.update(hive=domain,clip=video,split=split,members=[],hosts=[],resources=c['resources'],
                    gt_bins=window_counts(event_times(gt['rows'],c['polygon']),nf).tolist(),
                    pred_sum=np.sum([np.asarray(m['windows'],dtype=np.int64) for m in meta],axis=0).tolist())
        byhost=defaultdict(list)
        for candidate in c['candidates']:
            result=label_natural(gt,scenes,candidate);byhost[candidate['host_member_index']].append(result)
            actions.append(candidate['candidate_id'])
        for slot,m in enumerate(meta):
            unit['members'].append(dict(member=m['member'],member_index=slot,actions=byhost[slot],native_scene_proof=scenes[slot]['proof']))
        for host in c['host_contexts']:
            if not host['option_ids']:continue
            slot=host['host_member_index'];raw=raws[slot];ix=np.flatnonzero(raw[:,1]==host['host_parent_id'])
            ix=ix[np.argsort(raw[ix,0],kind='stable')]
            if len(ix)<3:raise ValueError('candidate host too short')
            targets={str(t):host_event_supervision(gt,scenes[slot],[int(ix[0]),int(ix[-2]),int(ix[-1])],t) for t in (.5,.3)}
            primary=targets['0.5'];unit['hosts'].append(dict(host,event_target_valid=primary['valid'],
                events=primary['events'],target_windows=window_counts(primary['events'],nf).tolist() if primary['valid'] else None,
                supervision=targets))
        shards[domain]['units'].append(unit)
        probes.append(dict(unit=c['unit'],domain=domain,group=c['group'],video=video,probes=return_probes(gt,scenes)))
        print('OWNERS203_UNIT',c['unit'],len(c['candidates']),dict(Counter(a['labels']['0.5']['class'] for m in unit['members'] for a in m['actions'])),flush=True)
    if member_count!=60 or len(set(actions))!=len(actions):raise ValueError('member/action inventory')
    if any(sha(args.repo/p)!=h for p,h in gt_hashes.items()):raise ValueError('GT changed')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.owners203-',dir=args.out.parent) as temp:
        target=Path(temp)/'payload';target.mkdir()
        for h,data in shards.items():
            data.update(GT_sha256={p:v for p,v in gt_hashes.items() if any(u['video'] in p for u in data['units'])},code_sha256=sha(__file__))
            freezer.save(target/(h+'.json'),data)
        freezer.save(target/'probes.json',dict(schema='peer-suffix203-return-probes-v1',prereg=PREREG,
            freeze_receipt=receipt,GT_sha256=gt_hashes,groups=probes,training_forbidden=True))
        summary=dict(schema='peer-suffix203-owner-summary-v1',prereg=PREREG,freeze_receipt=receipt,GT_sha256=gt_hashes,
                     groups=20,members=member_count,actions=len(actions),code_sha256=sha(__file__),domains={})
        for h,data in shards.items():
            aa=[a for u in data['units'] for m in u['members'] for a in m['actions']];hh=[x for u in data['units'] for x in u['hosts']]
            summary['domains'][h]=dict(actions=len(aa),labels={t:dict(Counter(a['labels'][t]['class'] for a in aa)) for t in ('0.5','0.3')},
                label_reasons={t:dict(Counter(a['labels'][t]['reason'] for a in aa)) for t in ('0.5','0.3')},
                host_targets=len(hh),valid_targets=sum(x['event_target_valid'] for x in hh),
                invalid_reasons=dict(Counter(x['supervision']['0.5']['reason'] for x in hh if not x['event_target_valid'])))
        freezer.save(target/'summary.json',summary)
        freezer.save(target/'classes.json',dict(schema='peer-suffix203-owner-classes-v1',prereg=PREREG,
            freeze_receipt=receipt,GT_sha256=gt_hashes,units=[dict(unit=u['unit'],domain=h,
                actions=[dict(action_id=a['action_id'],labels={t:dict(**{'class':a['labels'][t]['class']}) for t in ('0.5','0.3')})
                         for m in u['members'] for a in m['actions']]) for h,data in shards.items() for u in data['units']]))
        freezer.save(target/'manifest.json',{p.name:sha(p) for p in target.iterdir()})
        (target/'SEALED.sha256').write_text(sha(target/'manifest.json')+'\n');target.rename(args.out)
    print('OWNERS203_SEALED',sha(args.out/'manifest.json'),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--repo',type=Path)
    p.add_argument('--frozen',type=Path);p.add_argument('--out',type=Path);p.add_argument('--prereg')
    args=p.parse_args()
    if args.selftest:selftest()
    elif any(getattr(args,k) is None for k in ('repo','frozen','out','prereg')):p.error('--repo --frozen --out --prereg required')
    else:run(args)
