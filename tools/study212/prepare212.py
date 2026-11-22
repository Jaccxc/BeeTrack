"""GT-free212 unique date/inside event features on old joint86 +18 original K.

No joint_score, true assessment, GT, image, fitting or model execution entrance.
Reconstruct every state with original joint materializer, not a new search.
All numeric phase evidence is deduplicated by exact raw dates+inside bits.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import features212 as F

MARKER='[observed-phase-v1]'
SOURCE211='c21b74bbf3accd85a8a48863d16c1d241ce3045032b67ab8e6d92f390b88d6ae'
SOURCE208='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94'
FEATURE_SHA='a6d5a17af04f6e4e6e5211c882166978782c2efe52ca740f9762d8bd8b9e9f70'
SIX={('35','BEE24-35-head',3),('33','BEE24-33-head',3),('W','BEE24-10',0),('W','BEE24-15',0),('G','BEE24-24',0),('G','BEE24-28',0)}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def wire(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def save(path,obj):
    with Path(path).open('w',encoding='utf8',newline='\n') as f:json.dump(obj,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')


def save_gz(path,obj):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
            with io.TextIOWrapper(gz,encoding='utf8',newline='\n') as f:json.dump(obj,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')


def verify(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('phase seal mismatch')
    files=json.loads((root/'manifest.json').read_text())
    for name,digest in files.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError('frozen data drift '+name)
    return files


class FeatureBank:
    def __init__(self,evidence):self.rows={};self.evidence=evidence

    def get(self,dates,bits):
        sequence=dict(dates=np.asarray(dates,np.int64).tolist(),inside=np.asarray(bits,bool).tolist())
        key=hashlib.sha256(wire(sequence)).hexdigest()
        if key not in self.rows:
            full=F.run_features(np.asarray(dates,np.int64),np.asarray(bits,bool))
            if not full['valid']:raise ValueError('official net event rejected by phase feature API')
            if full['dates']!=sequence['dates'] or full['inside']!=sequence['inside']:raise AssertionError('feature sequence changed')
            self.rows[key]=dict(feature_key=key,**sequence,**{k:full[k] for k in ('G','R','R_alt','P','T')})
            json.dump(dict(feature_key=key,evidence=full),self.evidence,sort_keys=True,separators=(',',':'),allow_nan=False);self.evidence.write('\n')
        return key,self.rows[key]


def scene_events(rows,polygon,nf,bank,scope,raw_indices,original_canon):
    from beetrack.evaluate import by_track,inside,zone_tracks
    from tools.study197.candidates import window_counts
    R=np.asarray(rows,float);raw_indices=np.asarray(raw_indices,np.int64);original_canon=np.asarray(original_canon,np.int64)
    if len(raw_indices)!=len(R) or len(set(raw_indices))!=len(raw_indices):raise ValueError('scene original-row map not bijective onto kept rows')
    tagged=by_track(np.column_stack((R,np.arange(len(R)))))
    C=tagged[:,:7];scene_index=tagged[:,-1].astype(np.int64);bits=inside(C,polygon)
    zones=zone_tracks(C,polygon);events=[];run_count=len(zones)
    for pid,(direction,start,end,cross) in zones.items():
        if not direction:continue
        dates=C[start:end,0].astype(np.int64);runbits=bits[start:end];key,feature=bank.get(dates,runbits)
        indices=[j for j,d in cross if d==direction];last=indices[-1] if indices else end-1
        date=int(C[last,0]);sign=1 if direction=='in' else -1
        if F.event_summary(dates,runbits)!=dict(direction=sign,date=date):raise AssertionError('official event date/direction drift')
        raw=raw_indices[scene_index[start:end]];parents=sorted(map(int,np.unique(original_canon[raw])))
        event_ref=f'{scope["domain"]}|{scope["clip"]}|{scope["scene_ref"]}|{int(pid)}|{sign*date}'
        events.append(dict(event_ref=event_ref,canonical_id=int(pid),direction=sign,event_frame=date,signed_frame=sign*date,
            feature_key=key,original_parent_ids=parents,observations=int(end-start),
            first_original_raw_index=int(raw[0]),last_original_raw_index=int(raw[-1])))
    windows=window_counts([dict(frame=e['event_frame'],direction=e['direction']) for e in events],nf).tolist()
    return dict(scope,events=events,nPred=len(events),windows=windows,canonical_runs=run_count,
                non_event_runs=run_count-len(events),all_original_run_net_events_included=True)


def canonical_map(rows):
    from beetrack.evaluate import by_track
    R=np.asarray(rows,float);tag=by_track(np.column_stack((R,np.arange(len(R)))))
    result=np.empty(len(R),np.int64);result[tag[:,-1].astype(int)]=tag[:,1].astype(np.int64)
    return result


def load_clip(root,record):
    root=Path(root);folder=root/record['path'];data=json.loads((folder/'clip.json').read_text())
    return dict(data,folder=folder,prepare_root=root)


def load_scene(clip,scene_ref):
    """Explicit original native reconstruction for labels/forensics, no GT I/O."""
    scene=next(s for s in clip['scenes'] if s['scene_ref']==scene_ref)
    with np.load(clip['folder']/'native.npz',allow_pickle=False) as z:
        base=z['receiver' if scene['native_slot']==0 else 'peer'+str(scene['native_slot'])]
    if scene['map_file'] is None:indices=np.arange(len(base));rows=base.copy()
    else:
        with np.load(clip['folder']/scene['map_file'],allow_pickle=False) as z:indices=z['output_raw_indices'];rows=base[indices].copy();rows[:,1]=z['ids']
    events_path=clip['folder']/scene['events_file']
    with gzip.open(events_path,'rt',encoding='utf8') as f:events=json.load(f)
    return dict(rows=rows,output_raw_indices=indices,native_slot=scene['native_slot'],metadata=scene,events=events)


def run(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('full published prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    text=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if text.count(MARKER)!=1:raise ValueError('unique212marker required')
    if args.out.resolve().drive.upper()!='D:':raise ValueError('D: output required')
    if args.out.exists():verify(args.out);raise FileExistsError('complete frozen root already exists; verified, never overwrite')
    if sha(F.__file__)!=FEATURE_SHA:raise ValueError('registered phase feature source drift')
    if sha(args.main_frozen/'manifest.json')!=SOURCE211 or sha(args.inputs208/'manifest.json')!=SOURCE208:raise ValueError('fixed K source mismatch')
    # Deliberately no score/true/assessment argument or file open.
    sealpath=args.joint_predictions/'ALL_FOUR_SEAL.json';joint_seal=sha(sealpath)
    if (args.joint_predictions/'ALL_FOUR_SEAL.sha256').read_text().strip()!=joint_seal or (args.joint_seal and joint_seal!=args.joint_seal):
        raise ValueError('joint GT-free prediction seal mismatch')
    seal=json.loads(sealpath.read_text())
    if seal['schema']!='joint211-all-four-gtfree-seal-v1' or seal['receipts']['supplement_prereg']!='3323107ea3560e9ee507e65891097cf2ca605b00' or seal['receipts']['main_frozen_seal']!=SOURCE211:
        raise ValueError('wrong original joint family source')
    job=json.loads(args.job.read_text())
    if hashlib.sha256(wire(job)).hexdigest()!=seal['receipts']['job_sha256']:raise ValueError('original job does not match frozen predictions')
    if len(job['clips'])!=6:raise ValueError('fixed six job required')
    for name,digest in seal['files'].items():
        if Path(name).name!=name or sha(args.joint_predictions/name)!=digest:raise ValueError('GT-free joint fold file changed')
    local=Path(__file__).parent.parent/'study211';sys.path.insert(0,str(local))
    import freeze211 as main
    import joint_actions211 as J
    import joint_deploy211 as D
    from graphs211 import event_vertices
    from tools.study197.candidates import event_times,window_counts
    main.verify(args.main_frozen)
    inputs_manifest=json.loads((args.inputs208/'manifest.json').read_text())
    jobs={(c['domain'],c['clip']):c for c in job['clips']}
    if {(d,c,3 if d in ('35','33') else 0) for d,c in jobs}!=SIX:raise ValueError('original six seeds/clips changed')
    args.out.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.'+args.out.name+'-',dir=args.out.parent))
    references=[];total_states=0;total_K=0
    try:
        with (temp/'feature_evidence.jsonl.gz').open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
                with io.TextIOWrapper(gz,encoding='utf8',newline='\n') as evidence:
                    bank=FeatureBank(evidence)
                    for domain in ('35','33','W','G'):
                        name=domain+'.joint_predictions.json'
                        if name not in seal['files']:raise ValueError('missing frozen domain')
                        fold=json.loads((args.joint_predictions/name).read_text())
                        for prediction in fold['clips']:
                            clip=prediction['clip'];j=jobs[domain,clip];old=main.load_clip(args.main_frozen/j['main_relative']);meta=old['metadata']
                            base=old['native']['receiver'];canon=old['native']['receiver_canon']
                            source208=args.inputs208/domain/clip
                            for filename in ('rows.npy','canon.npy','metadata.json'):
                                if sha(source208/filename)!=inputs_manifest[f'{domain}/{clip}/{filename}']:raise ValueError('208numeric source drift')
                            source_meta=json.loads((source208/'metadata.json').read_text())
                            if not np.array_equal(base.view(np.uint64),np.load(source208/'rows.npy',mmap_mode='r').view(np.uint64)) or not np.array_equal(canon,np.load(source208/'canon.npy',mmap_mode='r')):
                                raise ValueError('original208/211 K/canon mismatch')
                            # Original catalog is GT-free. No prior GT-score file is opened.
                            with gzip.open(Path(j['catalog209'])/'catalog.json.gz','rt',encoding='utf8') as f:co=json.load(f)
                            parent=Path(job['input_job_proof']['original209_catalog']['root'])
                            if sha(parent/'manifest.json')!=j['source_seals']['209'] or (parent/'SEALED.sha256').read_text().strip()!=j['source_seals']['209']:
                                raise ValueError('209top catalog source seal mismatch')
                            parent_files=json.loads((parent/'manifest.json').read_text())
                            rel=(Path(j['catalog209'])/'manifest.json').resolve().relative_to(parent.resolve()).as_posix()
                            if sha(Path(j['catalog209'])/'manifest.json')!=parent_files[rel]:raise ValueError('209child catalog source seal mismatch')
                            child=json.loads((Path(j['catalog209'])/'manifest.json').read_text())
                            if sha(Path(j['catalog209'])/'catalog.json.gz')!=child['catalog.json.gz']:raise ValueError('209catalog numeric drift')
                            roster=J.make_roster(co['pairs'],j['original209_common_slate'],j['original209_delta_by_pair'],old['active_pairs'],j['original210_active_slate'],source_seals=j['source_seals'])
                            legal={tuple(sorted(s)) for s in J.feasible_subsets(roster)}
                            actual={tuple(sorted(s['selected_action_ids'])) for s in prediction['states']}
                            if legal!=actual or len(actual)!=len(prediction['states']):raise ValueError('joint full state roster changed')
                            folder=temp/domain/clip;folder.mkdir(parents=True)
                            np.savez(folder/'native.npz',**old['native'])
                            with gzip.open(args.main_frozen/j['main_relative']/'sources.json.gz','rt',encoding='utf8') as f:sources=json.load(f)
                            save_gz(folder/'sources.json.gz',sources)
                            scenes=[];state_records=[];Krefs={}
                            kstate=next(s for s in prediction['states'] if not s['selected_action_ids'])
                            for slot in (0,1,2):
                                R=base if slot==0 else old['native']['peer'+str(slot)]
                                C=canon if slot==0 else canonical_map(R)
                                scene_ref=f'K/m{slot}';sub=folder/scene_ref;sub.mkdir(parents=True)
                                original_K=next(s for s in old['states'] if s['is_K'])
                                member_meta=dict(member=meta['member'],member_index=0,seed=meta['seed'],tag=source_meta['tag']) if slot==0 else original_K['peer_graphs'][slot-1]['reference_member']
                                scope=dict(domain=domain,clip=clip,scene_ref=scene_ref,native_slot=slot,
                                           role='receiverK' if slot==0 else 'peerK',state_id=kstate['state_id'] if slot==0 else None,
                                           member=member_meta['member'],member_index=slot,seed=member_meta['seed'],tag=member_meta['tag'])
                                events=scene_events(R,meta['polygon'],meta['nf'],bank,scope,np.arange(len(R)),C)
                                save_gz(sub/'events.json.gz',events)
                                eventfile=scene_ref+'/events.json.gz'
                                scenes.append(dict(scope,map_file=None,events_file=eventfile,nPred=events['nPred']))
                                Krefs[slot]=scene_ref;total_K+=1
                            for state in prediction['states']:
                                R,proof=J.materialize(base,canon,roster,state['selected_action_ids'])
                                rawidx=np.asarray(proof['output_raw_indices'],np.int64)
                                if proof['output_raw_indices']!=state['retained_raw_indices'] or proof['deleted_raw_indices']!=state['deleted_raw_indices'] or R[:,1].astype(int).tolist()!=state['physical_id_vector'] or D.matrix_sha(R)!=state['scope_digest']:
                                    raise ValueError('joint original materialization/IDs/order/scope differs')
                                if not state['selected_action_ids'] and R.tobytes()!=base.tobytes():raise AssertionError('empty K raw bytes changed')
                                windows=window_counts(event_times(R,meta['polygon']),meta['nf']).tolist()
                                if windows!=state['windows']:raise ValueError('original timing/window closure differs')
                                _,_,vertices=event_vertices(R,meta['polygon'])
                                expected=prediction['peer_graphs'][state['state_id']][0]['prediction_vertices']
                                for v in vertices:v.pop('box',None)
                                if vertices!=expected:raise ValueError('canonical event IDs/crossings differ211 native schema')
                                if state['selected_action_ids']:
                                    scene_ref='states/'+state['state_id'];sub=folder/scene_ref;sub.mkdir(parents=True)
                                    np.savez_compressed(sub/'map.npz',ids=R[:,1].astype(np.int64),output_raw_indices=rawidx)
                                    scope=dict(domain=domain,clip=clip,scene_ref=scene_ref,native_slot=0,role='receiverState',state_id=state['state_id'],
                                               member=meta['member'],member_index=0,seed=meta['seed'],tag=source_meta['tag'])
                                    events=scene_events(R,meta['polygon'],meta['nf'],bank,scope,rawidx,canon)
                                    save_gz(sub/'events.json.gz',events)
                                    scenes.append(dict(scope,map_file=scene_ref+'/map.npz',events_file=scene_ref+'/events.json.gz',nPred=events['nPred']))
                                else:scene_ref=Krefs[0]
                                summary=prediction['state_summaries'][state['state_id']]
                                state_records.append(dict(state_id=state['state_id'],scene_ref=scene_ref,selected_action_ids=state['selected_action_ids'],
                                    tie_tuple=sorted([[prediction['actions'][aid]['family'],*prediction['actions'][aid]['canonical_ids']] for aid in state['selected_action_ids']]),
                                    edits=len(state['selected_action_ids']),nPred=summary['nPred'],raw_meanCountL1=summary['meanCountL1'],
                                    windows=windows,original211_scope_digest=state['scope_digest']))
                                total_states+=1
                            clipmeta={k:meta[k] for k in ('domain','clip','unit','nf','polygon','split','region','seed','member')}
                            clipmeta.update(schema='phase212-clip-v1',scenes=scenes,states=state_records,K_scene_refs={str(k):v for k,v in Krefs.items()},
                                original211_C_chosen_state_id=prediction['choices']['C'],original211_actions=prediction['actions'],
                                original211_GTfree_fold=dict(file=name,sha256=seal['files'][name]),
                                original211_joint_prediction_seal=joint_seal,original211_main_path=j['main_relative'],
                                original211_true_graph_locator=dict(domain=domain,clip=clip,state_ids=[s['state_id'] for s in state_records],
                                    note='locator only; no true/score/assessment opened during prepare'))
                            save(folder/'clip.json',clipmeta)
                            references.append(dict(domain=domain,clip=clip,path=domain+'/'+clip,states=len(state_records),originalK=3))
                            print('PHASE_PREPARED',domain,clip,'states',len(state_records),flush=True)
                        del fold
                    save_gz(temp/'feature_bank.json.gz',dict(schema='phase212-unique-feature-bank-v1',feature_schema=F.FEATURE_SCHEMA,
                        G_names=list(F.G_NAMES),phase_names=list(F.PHASE_NAMES),features=bank.rows,
                        raw_positive_dates_and_inside_bits_preserved=True))
                    unique_count=len(bank.rows)
        if total_states!=86 or total_K!=18 or len(references)!=6:raise ValueError('complete original86/18K/sixclip family required')
        identity=dict(prereg=args.prereg,marker=MARKER,joint_GTfree_seal=joint_seal,main211_frozen=SOURCE211,inputs208=SOURCE208,
            original_job_sha256=sha(args.job),code_sha256={p.name:sha(p) for p in (Path(__file__),Path(F.__file__),local/'joint_actions211.py',local/'joint_deploy211.py')})
        save(temp/'deployment.json',dict(schema='phase212-prepare-v1',prereg=args.prereg,clips=references,states=total_states,originalK=total_K,
            feature_bank_file='feature_bank.json.gz',unique_sequences=unique_count,identity=identity))
        save(temp/'receipt.json',dict(identity=identity,GT_access=False,true_assessment_access=False,image_access=False,
            all_original86_scope_and_event_windows_exact=True,empty_K_bytes_exact=True,feature_sequences_have_raw_dates_and_inside=True,
            feature_computation_once_per_sequence=True))
        files={p.relative_to(temp).as_posix():sha(p) for p in temp.rglob('*') if p.is_file()}
        save(temp/'manifest.json',files);(temp/'SEALED.sha256').write_text(sha(temp/'manifest.json')+'\n');temp.rename(args.out)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print('PHASE212_FROZEN86_K18',sha(args.out/'manifest.json'))


def self_test():
    import io
    bank=FeatureBank(io.StringIO());R=np.array([[f,1,x,0,2,2,.9] for f,x in [(1,-3),(2,-2),(3,0),(4,1)]],float)
    poly=[(0,-10),(20,-10),(20,10),(0,10)];scope=dict(domain='synthetic',clip='synthetic',scene_ref='K/m0',native_slot=0,role='receiverK',state_id='K')
    a=scene_events(R,poly,30,bank,scope,np.arange(4),np.ones(4,int))
    b=scene_events(R,poly,30,bank,dict(scope,scene_ref='states/copy'),np.arange(4),np.ones(4,int))
    assert len(bank.rows)==1 and a['events'][0]['feature_key']==b['events'][0]['feature_key']
    saved=next(iter(bank.rows.values()));assert saved['dates']==[1,2,3,4] and saved['inside']==[False,False,True,True]
    assert all(len(saved[k])==6 for k in ('R','R_alt','P','T')) and len(saved['G'])==7
    assert a['events'][0]['canonical_id']==1 and a['events'][0]['event_frame']==3
    wire(a);wire(bank.rows)
    return dict(passed=True,scope='synthetic event timing/canonicalID, exact dates+inside dedup, phasebank P/R/Ralt noGT')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--self-test',action='store_true')
    p.add_argument('--joint-predictions',type=Path);p.add_argument('--joint-seal');p.add_argument('--job',type=Path)
    p.add_argument('--main-frozen',type=Path);p.add_argument('--inputs208',type=Path);p.add_argument('--out',type=Path);p.add_argument('--prereg')
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    if args.self_test:print(json.dumps(self_test()))
    else:
        if any(getattr(args,k) is None for k in ('joint_predictions','job','main_frozen','inputs208','out')):p.error('explicit GT-free old sources/output required')
        run(args)
