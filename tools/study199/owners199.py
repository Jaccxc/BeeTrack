"""Native full-scene isolated-owner taxonomy and observed truncation census.

Pure analyze_member has no field access. Real CLI requires the published199
registration; output is split by domain so source fitting need not read its
held-out label payload. No detector, GPU, feature extraction or model fitting.
"""
import os
import sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
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
import cv2
import numpy as np
if '--repo' in sys.argv:sys.path.insert(0,str(Path(sys.argv[sys.argv.index('--repo')+1]).resolve()))
from beetrack.evaluate import by_track,frame_match,inside,iou_matrix,load_tracks,zone_tracks

MARKER='[endpoint-identity-v1]'
SOURCE197_SEAL='e7fd48078f8d0a32bb9018f2bac93dc4b5270d374d5bcd0a631fa97dd4d790e4'
DOMAINS=('35','33','W','G')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    a=np.ascontiguousarray(a,dtype='<f8')
    return hashlib.sha256(json.dumps(list(a.shape),separators=(',',':')).encode()+b'\n'+a.tobytes()).hexdigest()


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x


def save(path,data):Path(path).write_text(json.dumps(plain(data),separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')


def by_date(rows):
    order=np.argsort(rows[:,0],kind='stable');dates,starts=np.unique(rows[order,0],return_index=True)
    return {int(f):ix for f,ix in zip(dates,np.split(order,starts[1:]))}


def prepare_gt(raw,polygon):
    raw=np.asarray(raw,dtype=float)
    if raw.ndim!=2 or raw.shape[1]<6 or not np.isfinite(raw).all():raise ValueError('finite GT rows required')
    if np.any(raw[:,:2]!=np.rint(raw[:,:2])) or len(np.unique(raw[:,:2],axis=0))!=len(raw):raise ValueError('GT native keys invalid')
    tagged=by_track(np.c_[raw,np.arange(len(raw))]);G=tagged[:,:-1];index=tagged[:,-1].astype(int)
    segments={};original_runs=defaultdict(list)
    for pid,(event,start,end,cross) in zone_tracks(G,polygon).items():
        originals=sorted(set(int(x) for x in raw[index[start:end],1]));orig=originals[0] if len(originals)==1 else None
        record=dict(canonical_gt_id=int(pid),original_gt_id=orig,original_ids=originals,start=start,end=end,
                    first_frame=int(G[start,0]),last_frame=int(G[end-1,0]),
                    start_inside=bool(inside(G[start:start+1],polygon)[0]),end_inside=bool(inside(G[end-1:end],polygon)[0]),event=event)
        segments[int(pid)]=record
        if orig is not None:original_runs[orig].append(record)
    for records in original_runs.values():
        for run,record in enumerate(sorted(records,key=lambda x:x['first_frame'])):record['run']=run
    return dict(raw=raw,rows=G,source_index=index,frames=by_date(G),segments=segments,polygon=polygon)


def event_times(rows,polygon):
    rows=by_track(rows);events=[]
    for pid,(direction,start,end,cross) in zone_tracks(rows,polygon).items():
        if direction:
            j=next(j for j,d in reversed(cross) if d==direction)
            events.append(dict(track_id=int(pid),frame=int(rows[j,0]),direction=1 if direction=='in' else -1))
    return events


def bins(events,nf):
    result=[]
    for offset in range(30):
        a=np.zeros(((nf+25*offset+749)//750,2),dtype=np.int64)
        for e in events:a[(e['frame']+25*offset-1)//750,int(e['direction']<0)]+=1
        result.extend(a.tolist())
    return result


def row_owners(gt,K,indices,threshold):
    """Official matching plus full-scene threshold degrees at requested dates.

    Every GT/prediction at each requested frame participates. Nothing is
    cropped to one track or the candidate's endpoint window for assignment.
    """
    G=gt['rows'];frames=by_date(K);requested=defaultdict(list)
    for index in sorted(set(indices)):requested[int(K[index,0])].append(index)
    xyxy=lambda a:np.c_[a[:,2:4],a[:,2:4]+a[:,4:6]]
    result={}
    for date,needed in requested.items():
        pi=frames[date];gi=gt['frames'].get(date,np.empty(0,int));P=K[pi];T=G[gi]
        matrix=iou_matrix(xyxy(T),xyxy(P)) if len(T) else np.zeros((0,len(P)))
        graph=matrix>=threshold;pd=graph.sum(0);gd=graph.sum(1)
        owners=frame_match(T,P,thr=threshold)
        predpos={int(index):j for j,index in enumerate(pi)};gtpos={int(r[1]):j for j,r in enumerate(T)}
        for index in needed:
            j=predpos[index];owner=int(owners[j]);q=gtpos.get(owner);segment=gt['segments'].get(owner)
            qual=bool(q is not None and pd[j]==1 and gd[q]==1 and segment['original_gt_id'] is not None)
            ownerrow=T[q] if q is not None else None
            result[index]=dict(canonical_row=int(index),pred_key=[date,int(K[index,1])],row_sha256=array_sha(K[index]),
                official_owner=owner if owner>=0 else None,owner_gt_row=int(gi[q]) if q is not None else None,
                owner_gt_original_id=segment['original_gt_id'] if segment else None,owner_gt_run=segment.get('run') if segment else None,
                owner_gt_raw_index=int(gt['source_index'][gi[q]]) if q is not None else None,
                owner_gt_key=[date,owner] if owner>=0 else None,owner_gt_row_sha256=array_sha(ownerrow) if ownerrow is not None else None,
                iou=float(matrix[q,j]) if q is not None else None,pred_degree=int(pd[j]),gt_degree=int(gd[q]) if q is not None else None,
                matched=owner>=0,isolated=qual,qualified=qual,
                pred_inside=bool(inside(K[index:index+1],gt['polygon'])[0]),
                gt_inside=bool(inside(ownerrow[None],gt['polygon'])[0]) if ownerrow is not None else None,
                graph_gt=[dict(canonical_gt_id=int(T[g,1]),original_gt_id=gt['segments'][int(T[g,1])]['original_gt_id'],iou=float(matrix[g,j]),gt_degree=int(gd[g])) for g in np.flatnonzero(graph[:,j])])
    return result


def classify(left,right):
    if not left['qualified'] or not right['qualified']:
        return 'U','unmatched_competing_or_ambiguous_owner'
    if left['official_owner']==right['official_owner']:return 'S','same_canonical_GT_segment'
    if left['owner_gt_original_id']!=right['owner_gt_original_id']:return 'D','distinct_original_GT_IDs'
    return 'U','same_original_GT_ID_different_gap_runs'


def truncation(candidate,block,K,gt,evidence,label):
    empty=dict(supported=False,type='unconfirmed',reason='not_drop_S')
    if candidate['delta']!=-1 or label!='S':return empty
    i0,i1=block['start'],block['end']-1;cut=candidate['cut_after_canonical_row']
    ix=[i0,i1,cut,cut+1];rows=[evidence[i] for i in ix]
    if not all(r['qualified'] for r in rows) or len({r['official_owner'] for r in rows})!=1:
        return dict(empty,reason='parent_endpoints_and_cut_not_one_isolated_GT_run',row_indices=ix)
    owner=rows[0]['official_owner'];segment=gt['segments'][owner]
    pstart,pend=rows[0]['pred_inside'],rows[1]['pred_inside']
    info=dict(gt_run=segment,rows=rows,row_indices=ix,p_first_frame=int(K[i0,0]),p_last_frame=int(K[i1,0]),
              p_start_inside=pstart,p_end_inside=pend,K_event_direction='in' if pend else 'out')
    if rows[0]['gt_inside']!=pstart or rows[1]['gt_inside']!=pend:
        return dict(empty,**info,reason='GT_at_P_endpoint_dates_side_disagrees')
    if segment['start_inside']!=segment['end_inside']:
        return dict(empty,**info,reason='GT_run_has_different_endpoint_sides')
    common=segment['start_inside'];G=gt['rows'][segment['start']:segment['end']]
    sides=inside(G,gt['polygon'])
    witness=lambda j:dict(frame=int(G[j,0]),canonical_gt_id=owner,original_gt_id=segment['original_gt_id'],run=segment.get('run'),inside=bool(sides[j]),row_sha256=array_sha(G[j]))
    excursion=np.flatnonzero(sides!=common)
    info['excursion_observation']=witness(int(excursion[0])) if len(excursion) else None
    if common==pstart and segment['last_frame']>K[i1,0]:
        returned=np.flatnonzero((G[:,0]>K[i1,0])&(sides==common))
        if len(returned):return dict(supported=True,type='later_return',reason='same_GT_run_returns_after_P_end',return_observation=witness(int(returned[0])),**info)
    if common==pend and segment['first_frame']<K[i0,0]:
        earlier=np.flatnonzero((G[:,0]<K[i0,0])&(sides==common))
        if len(earlier):return dict(supported=True,type='earlier_part_missed',reason='same_GT_run_starts_before_P_start',earlier_observation=witness(int(earlier[-1])),**info)
    return dict(empty,**info,reason='no_observed_supporting_GT_extension')


def analyze_member(gt,k_rows,bundle,polygon,nframes):
    prepared=gt if isinstance(gt,dict) and 'segments' in gt else prepare_gt(gt,polygon)
    K=by_track(k_rows);C=bundle['canonical_rows']
    cols=[0,*range(2,C.shape[1])]
    if K.shape!=C.shape or not np.array_equal(K[:,cols],C[:,cols]):raise ValueError('K canonical row/date/box order changed')
    blocks={b['parent_id']:b for b in bundle['parent_blocks']};needed=set()
    for c in bundle['candidates']:
        b=blocks[c['parent_id']];j=c['cut_after_canonical_row']
        if not b['start']<=j<b['end']-1:raise ValueError('cut index outside canonical parent')
        if (j-b['start']+1 if c['end']=='start' else b['end']-j-1)!=c['k']:raise ValueError('short segment cut offset changed')
        if int(K[j,0])>=int(K[j+1,0]):raise ValueError('cut rows are not consecutive parent observations')
        side=inside(K[b['start']:b['end']],polygon)
        short=side[:c['k']] if c['end']=='start' else side[-c['k']:]
        if len(set(short))!=1 or side[j-b['start']]==side[j+1-b['start']]:raise ValueError('cut is not native side boundary')
        needed.update((b['start'],b['end']-1,j,j+1))
    owners={str(t):row_owners(prepared,K,needed,t) for t in (.5,.3)}
    for values in owners.values():
        for index,evidence in values.items():
            evidence.update(source_raw_index=int(bundle['source_raw_index'][index]),source_parent_id=int(C[index,1]),source_row_sha256=array_sha(C[index]))
    records=[];counts={str(t):Counter() for t in (.5,.3)}
    for c in bundle['candidates']:
        j=c['cut_after_canonical_row'];labels={}
        for t,rows in owners.items():
            left,right=rows[j],rows[j+1];label,reason=classify(left,right)
            censor=truncation(c,blocks[c['parent_id']],K,prepared,rows,label)
            labels[t]=dict(**{'class':label},reason=reason,left=left,right=right,censor=censor)
            counts[t][label]+=1
            if censor['supported']:counts[t][censor['type']]+=1
        keys=('candidate_id','key','unit','hive','domain','group','clip','video','member','member_index','tag','parent_id','parent_canonical_id','k','end','span','frame','direction','delta','delta_windows')
        record={k:c[k] for k in keys if k in c};record['labels']=labels
        records.append(record)
    ge=event_times(prepared['rows'],polygon);pe=event_times(K,polygon)
    return dict(records=records,counts={t:dict(x) for t,x in counts.items()},K_rows_sha256=array_sha(K),GT_rows_sha256=array_sha(prepared['rows']),
                nG=len(ge),gt_bins=bins(ge,nframes),pred_bins=bins(pe,nframes),requested_rows=len(needed))


def load_domain(root,domain):
    """Validate global receipt, then read only requested source label shard."""
    if domain not in DOMAINS:raise ValueError('unknown domain')
    root=Path(root)
    if (root/'SEALED.sha256').read_text().strip()!=sha(root/'manifest.json'):raise ValueError('owners seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text());name=domain+'.json'
    if sha(root/name)!=manifest[name]:raise ValueError('source owner shard changed')
    data=json.loads((root/name).read_text())
    if data['domain']!=domain or data['source197_seal']!=SOURCE197_SEAL:raise ValueError('wrong owner source')
    return data


def flat_records(root,domain):return [r for u in load_domain(root,domain)['units'] for m in u['members'] for r in m['records']]


def _module(name,path):
    sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run(args):
    cv2.setNumThreads(1)
    if not re.fullmatch('[0-9a-f]{40}',args.prereg):raise ValueError('full prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,capture_output=True,check=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,capture_output=True,check=True).stdout.decode('utf-8')
    if log.count(MARKER)!=1:raise ValueError('unique199 prereg marker required')
    if args.out.exists():raise FileExistsError(args.out)
    if sha(args.source197/'manifest.json')!=SOURCE197_SEAL:raise ValueError('original197 input seal changed')
    source=_module('source197_freeze',args.repo/'tools/study197/freeze.py')
    physical=_module('source197_physical',args.repo/'tools/study197/physical.py')
    receipt=source.verify(args.source197);groups=source.load_groups(args.source197)
    if len(groups)!=20 or sum(len(g['members']) for g in groups)!=60:raise ValueError('source197 F60 changed')
    truths={};gt_hashes={}
    for g in groups:
        if g['video'] not in truths:
            path=args.repo/'runs/gt'/g['split']/g['video']/'gt/gt.txt'
            gt_hashes[path.relative_to(args.repo).as_posix()]=sha(path)
            truths[g['video']]=prepare_gt(load_tracks(path),g['polygon'])
    if len(truths)!=6:raise ValueError('exactly six authorized GT clips required')
    shards={h:dict(schema='endpoint-owner-shard-v1',domain=h,prereg=args.prereg,source197_seal=SOURCE197_SEAL,source197_receipt=receipt,units=[]) for h in DOMAINS}
    summary={h:{str(t):Counter() for t in (.5,.3)} for h in DOMAINS};allids=[]
    for g in groups:
        members=[];truth=None
        for m in g['members']:
            K,_=physical.realize(m['rows'],m['bundle'],[])
            evidence=analyze_member(truths[g['video']],K,m['bundle'],g['polygon'],g['nf'])
            if truth is None:truth=evidence['gt_bins']
            if truth!=evidence['gt_bins']:raise ValueError('members GT clock differs')
            if not np.array_equal(evidence['pred_bins'],m['baseline_windows']):raise ValueError('native K windows differ197')
            allids.extend(r['candidate_id'] for r in evidence['records'])
            for t,values in evidence['counts'].items():summary[g['domain']][t].update(values)
            members.append(dict(member=m['member'],member_index=m['member_index'],tag=m['tag'],raw_receipt=m['raw_receipt'],**evidence))
        keys=('unit','domain','hive','group','video','clip','nf','nframes','split','polygon')
        unit={k:g[k] for k in keys};unit.update(members=members,gt_bins=truth,gt_count=members[0]['nG'],pred_sum=np.sum([m['pred_bins'] for m in members],axis=0).tolist())
        shards[g['domain']]['units'].append(unit)
        print('OWNER_UNIT',g['unit'],{t:dict(summary[g['domain']][t]) for t in ('0.5','0.3')},flush=True)
    if len(allids)!=1196 or len(set(allids))!=1196:raise ValueError('all1196 source candidate IDs required')
    if any(sha(args.repo/p)!=v for p,v in gt_hashes.items()):raise ValueError('GT changed during owner census')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.owners199-',dir=args.out.parent) as tmp:
        target=Path(tmp)/'payload';target.mkdir()
        code=dict(owners199=sha(__file__),official_evaluate=sha(args.repo/'beetrack/evaluate.py'),source197_freezer=sha(args.repo/'tools/study197/freeze.py'),source197_physical=sha(args.repo/'tools/study197/physical.py'))
        for h,data in shards.items():
            data.update(counts={t:dict(v) for t,v in summary[h].items()},GT_sha256=gt_hashes,code_sha256=code)
            save(target/(h+'.json'),data)
        save(target/'summary.json',dict(schema='endpoint-owner-summary-v1',prereg=args.prereg,source197_seal=SOURCE197_SEAL,source197_receipt=receipt,GT_sha256=gt_hashes,code_sha256=code,members=60,candidates=1196,counts={h:{t:dict(v) for t,v in q.items()} for h,q in summary.items()},definition='qualified=official frame_match owner AND full-frame preddegree=1 AND owner GTdegree=1; S=same canonicalGTsegment,D=distinctoriginalGTIDs,otherwiseU'))
        save(target/'manifest.json',{p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file()})
        (target/'SEALED.sha256').write_text(sha(target/'manifest.json')+'\n');target.rename(args.out)
    print('OWNERS_SEALED',sha(args.out/'manifest.json'),flush=True)


def self_test(repo):
    candidates=_module('synthetic197_candidates',repo/'tools/study197/candidates.py')
    physical=_module('synthetic197_physical',repo/'tools/study197/physical.py')
    poly=[[0,0],[10,0],[10,10],[0,10]]
    row=lambda f,p,x:[f,p,x-1,4,2,2,.8]
    truth=np.array([row(f,1,x) for f,x in enumerate((12,8,7,6,12),1)],float)
    pred=truth[:4].copy();pred[:,1]=9
    b=candidates.build_candidates(pred,poly,5);K,_=physical.realize(pred,b,[])
    result=analyze_member(truth,K,b,poly,5);r=result['records'][0]
    for t in ('0.5','0.3'):
        assert r['labels'][t]['class']=='S' and r['labels'][t]['censor']['type']=='later_return'
        assert r['labels'][t]['censor']['return_observation']['frame']==5
    # Full-scene identical-box rival makes the cut unknown despite assignment.
    competitor=np.r_[pred,pred[:2].copy()];competitor[4:,1]=10
    cb=candidates.build_candidates(competitor,poly,5);CK,_=physical.realize(competitor,cb,[])
    rr=analyze_member(truth,CK,cb,poly,5)
    assert all(r['labels']['0.5']['class']=='U' for r in rr['records'])
    truth_rival=np.r_[truth,truth[:2].copy()];truth_rival[5:,1]=77
    tied=analyze_member(truth_rival,K,b,poly,5)['records'][0]
    assert tied['labels']['0.5']['class']=='U' and tied['labels']['0.5']['left']['pred_degree']==2
    # Earlier part missed: true starts inside, leaves, then returns; prediction
    # omits the first true row and ends with a short inside suffix.
    earlier=np.array([row(f,2,x) for f,x in enumerate((8,12,13,14,8),1)],float)
    P=earlier[1:].copy();P[:,1]=5
    eb=candidates.build_candidates(P,poly,5);EK,_=physical.realize(P,eb,[])
    er=analyze_member(earlier,EK,eb,poly,5)['records'][0]
    assert er['labels']['0.5']['censor']['type']=='earlier_part_missed'
    # Direct gap-run and distinct-original taxonomy, no modulo-ID shortcuts.
    L=dict(qualified=True,official_owner=1,owner_gt_original_id=1)
    R=dict(qualified=True,official_owner=100001,owner_gt_original_id=1)
    assert classify(L,R)[0]=='U'
    R['owner_gt_original_id']=7;assert classify(L,R)[0]=='D'
    # True box centred exactly on polygon boundary remains inside.
    assert inside(np.array([row(1,1,10)],float),poly)[0]
    print('owners199 synthetic five-frame return, earlier mirror, competition/ties, gap-run/original IDs and boundary passed')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',required=True,type=Path);ap.add_argument('--source197',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--prereg');ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args();args.repo=args.repo.resolve()
    if args.self_test:self_test(args.repo)
    else:
        if not args.source197 or not args.out or not args.prereg:ap.error('--source197 --out --prereg required')
        args.source197=args.source197.resolve();args.out=args.out.resolve();run(args)
