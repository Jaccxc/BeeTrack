"""Durable first independent212 field stage: original physical carriers/features."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,gzip,hashlib,json,re,sys,gc
from pathlib import Path
import numpy as np
import audit212 as independent
sys.path.insert(0,str(Path(__file__).parents[1]/'study211'))
import field211 as previous
prior=previous.prior
PREREG='8311f4fee58caaf6712777dcf14dd37525580bf9'
PINS={'frozen':'61774c5f725073b1d887fdc06a8dd3a390a7566406eef12fe31c3fc859fd3b2b','labels':'46b3ab84c85053161925d09f2008af455c6a5c76e3916403dcc579b5dea12091','truth_late':'8149898901c37c54c980c4277d2ff9f6631b02ca0878244d920e151563d278b1','predictions':'66c2ecfd756a878e4e50fa37cf9b8883675510a02ef57f766f95c9faffbe08d8','evaluation':'dd2f15fb9d57afc26688edfa9c2326c5fd5c8077191c61141bdfaacf3230caae'}
JOINT='a6446982277d9d3fee3b0efb786e6a1248536899ec04bb842d6ae08fe399f85f'
ASSESSMENT='5b6d317422e5a18b1e1eade522a828cff41ae6cafb15c4247cf7002de75e3861'
GATE='3be147c2efbb2eeabf85d322880cb32e2d9760d86e1443f534e93ce09db4c578'

def wire(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def matrix_sha(x):return hashlib.sha256(wire(list(x.shape))+np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()
def groups(rows):
    order=np.lexsort((np.arange(len(rows)),rows[:,0],rows[:,1]));ids,starts=np.unique(rows[order,1],return_index=True);answer=[];seen=set()
    for pid,a,b in zip(ids,starts,[*starts[1:],len(rows)]):
        ii=order[a:b];assert np.all(np.diff(rows[ii,0])>0)
        for n,part in enumerate(np.split(ii,np.flatnonzero(np.diff(rows[ii,0])>25)+1)):
            canonical=int(pid)+100000*n;assert canonical not in seen;seen.add(canonical);answer.append((canonical,part))
    return answer

def stream_physical_states(path):
    """Old canonical JSON state objects start with deleted_raw_indices.

    Decode one complete physical vector object at a time, not a350MB fold.
    Source files are raw-byte pinned before this format-specific reader runs.
    """
    decoder=json.JSONDecoder();pattern=re.compile(r'\{\s*"deleted_raw_indices"\s*:');buffer=''
    with Path(path).open(encoding='utf8') as stream:
        while True:
            found=pattern.search(buffer)
            if found:
                buffer=buffer[found.start():]
                while True:
                    try:value,end=decoder.raw_decode(buffer);break
                    except json.JSONDecodeError:
                        chunk=stream.read(1<<20)
                        if not chunk:raise
                        buffer+=chunk
                buffer=buffer[end:]
                if all(k in value for k in ('state_id','scope_digest','retained_raw_indices','physical_id_vector')):yield value
            else:
                chunk=stream.read(1<<20)
                if not chunk:break
                buffer=buffer[-100:]+chunk

def load_records(args):
    top=prior.read(args.root/'frozen/deployment.json');records={}
    for ref in top['clips']:
        folder=args.root/'frozen'/ref['path'];c=prior.read(folder/'clip.json')
        with np.load(folder/'native.npz') as z:arrays={k:z[k] for k in z.files}
        records[c['domain']+'/'+c['clip']]=dict(meta=c,folder=folder,arrays=arrays,ref=ref)
    return top,records

def scene_rows(r,scene):
    base=r['arrays']['receiver' if scene['native_slot']==0 else 'peer'+str(scene['native_slot'])]
    if scene['map_file'] is None:return base.copy(),np.arange(len(base))
    with np.load(r['folder']/scene['map_file']) as z:
        raw=z['output_raw_indices'];rows=base[raw].copy();rows[:,1]=z['ids']
    return rows,raw

def verify_scene(r,scene,rows,raw_indices,sequences):
    c=r['meta'];document=prior.read(r['folder']/scene['events_file']);bits=independent.inside_bits(rows,c['polygon']);events=[]
    slot=scene['native_slot'];base=r['arrays']['receiver' if slot==0 else 'peer'+str(slot)]
    original=r['arrays']['receiver_canon'] if slot==0 else prior.canonical(base)
    for pid,ii in groups(rows):
        dates=rows[ii,0].astype(np.int64);side=bits[ii];event=independent.net_event(dates,side)
        if event is None:continue
        sequence=dict(dates=dates.tolist(),inside=side.tolist());key=hashlib.sha256(wire(sequence)).hexdigest()
        if key in sequences:assert sequences[key]==sequence
        else:sequences[key]=sequence
        raw=raw_indices[ii];sign=event['direction'];date=event['date'];event_ref=f'{c["domain"]}|{c["clip"]}|{scene["scene_ref"]}|{pid}|{sign*date}'
        events.append(dict(event_ref=event_ref,canonical_id=pid,direction=sign,event_frame=date,signed_frame=sign*date,feature_key=key,original_parent_ids=sorted(map(int,set(original[raw]))),observations=len(ii),first_original_raw_index=int(raw[0]),last_original_raw_index=int(raw[-1])))
    assert events==document['events'] and len(events)==document['nPred']==scene['nPred']
    assert len(groups(rows))==document['canonical_runs'] and document['non_event_runs']==document['canonical_runs']-len(events)
    bins=previous.geometry.clock([(e['event_frame'],e['direction']) for e in events],c['nf']);assert np.array_equal(bins,document['windows'])
    return events

def phase_proof(saved,own,dates):
    assert saved['segments']==own['segments'] and saved['events']==own['events']
    assert saved['dates']==[dates[i] for i in own['indices']]
    for f in ('phase','O','D','T'):assert saved[f]==own[f]

def run(args):
    if not args.field_go:raise ValueError('soleCPU GO required')
    sys.path.insert(0,str(args.repo.resolve()));seals={s:prior.seal(args.root/s,h) for s,h in PINS.items()}
    prior.seal(args.joint_assessment,JOINT);assert prior.sha(args.joint_assessment/'assessment.json')==ASSESSMENT
    top,records=load_records(args);assert top['prereg']==PREREG and top['states']==86 and top['originalK']==18
    gatefile=args.joint_predictions/'ALL_FOUR_SEAL.json';assert prior.sha(gatefile)==GATE==top['identity']['joint_GTfree_seal']
    gate=prior.read(gatefile);old=prior.read(args.joint_assessment/'assessment.json')
    assert hashlib.sha256(wire(gate)).hexdigest()==old['prediction_seal']
    prior.seal(args.main211,previous.FROZEN)
    allseq={};seen_scenes=set();event_counts={};scene_count=0
    for key,r in records.items():
        with np.load(args.main211/key/'native.npz') as z:
            for name,array in r['arrays'].items():assert array.tobytes()==z[name].tobytes()
        for scene in r['meta']['scenes']:
            if scene['role'] in ('receiverK','peerK'):
                rows,raw=scene_rows(r,scene);ev=verify_scene(r,scene,rows,raw,allseq);seen_scenes.add((key,scene['scene_ref']));event_counts[key,scene['scene_ref']]=len(ev);scene_count+=1
    old_by_scope={}
    for key,r in records.items():
        for state in r['meta']['states']:old_by_scope[state['original211_scope_digest']]=(key,state)
    actual_count=0;old_graphs={(c['domain']+'/'+c['clip']):c['states'] for c in old['clips']}
    for domain in ('35','33','W','G'):
        path=args.joint_predictions/(domain+'.joint_predictions.json');assert prior.sha(path)==gate['files'][path.name]
        for state in stream_physical_states(path):
            if state['scope_digest'] not in old_by_scope:continue
            key,current=old_by_scope[state['scope_digest']];r=records[key]
            assert current['state_id']==state['state_id'] and current['selected_action_ids']==state['selected_action_ids']
            raw=np.asarray(state['retained_raw_indices'],np.int64);rows=r['arrays']['receiver'][raw].copy();rows[:,1]=state['physical_id_vector'];assert matrix_sha(rows)==state['scope_digest']
            scene=next(s for s in r['meta']['scenes'] if s['scene_ref']==current['scene_ref']);prepared,indices=scene_rows(r,scene)
            assert rows.tobytes()==prepared.tobytes() and np.array_equal(raw,indices)
            assert old_graphs[key][state['state_id']]['scope_digest']==state['scope_digest']
            ev=verify_scene(r,scene,rows,raw,allseq);assert len(ev)==current['nPred']
            assert np.array_equal(current['windows'],old_graphs[key][state['state_id']]['metrics']['pred_bins'])
            event_counts[key,scene['scene_ref']]=len(ev);seen_scenes.add((key,scene['scene_ref']));actual_count+=1
        print('AUDIT212_ORIGINAL_CARRIERS',domain,flush=True);gc.collect()
    assert actual_count==86 and scene_count==18
    bank=prior.read(args.root/'frozen'/top['feature_bank_file'])['features'];assert set(bank)==set(allseq) and len(bank)==top['unique_sequences']
    verified=set()
    with gzip.open(args.root/'frozen/feature_evidence.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            item=json.loads(line);key=item['feature_key'];assert key not in verified;verified.add(key)
            sequence=allseq[key];own=independent.features(np.asarray(sequence['dates']),np.asarray(sequence['inside'],bool));saved=item['evidence'];assert own['valid']
            for f in ('G','P','R','R_alt','T'):prior.close(own[f],bank[key][f],1e-13);prior.close(own[f],saved[f],1e-13)
            assert saved['dates']==sequence['dates'] and saved['inside']==sequence['inside'] and saved['original_event']==own['original_event']
            for k in ('2','3'):
                for a,b in zip(saved['phases'][k],own['phases'][k]):phase_proof(a,b,sequence['dates'])
                for a,b in zip(saved['controls'][k],own['controls'][k]):
                    for f in ('salt','seed','permutation','labels','ODT'):assert a[f]==b[f]
                    for aa,bb in zip(a['phases'],b['phases']):phase_proof(aa,bb,sequence['dates'])
            if len(verified)%100==0:print('AUDIT212_UNIQUE_FEATURES',len(verified),flush=True)
    assert verified==set(bank)
    prior.save(args.root/'verification/features.json',dict(passed=True,prereg=PREREG,seals=seals,old_joint_assessment=JOINT,old_joint_gate_bytes=GATE,old_joint_gate_logical=old['prediction_seal'],states=actual_count,original_K=scene_count,unique_sequences=len(bank),scene_event_counts=[dict(clip=k,scene_ref=s,n=n) for (k,s),n in sorted(event_counts.items())],scope='All original86 physical matrices reconstructed from raw main211 native carriers and streamed OLD sealed ID/raw-index state vectors; exact212 encoded-map/scope/oldgraph window equality. All18 originalK full canonical gaps/sides/net events rebuilt. All unique raw dates+inside sequences, G7/P6/R6/Ralt6/T6 and every16-salt seed/permutation/phase segmentation independently rebuilt. Polygon containment shares registered boundary-inclusive cv2 primitive; no interpolation/min3. No86 fresh GT/MOT rescoring.'))
    snapshot=args.root/'verification/source_snapshot_features';snapshot.mkdir(exist_ok=True)
    for p in (Path(__file__),Path(independent.__file__)):(snapshot/p.name).write_bytes(p.read_bytes())
    print('AUDIT212_FEATURES_PASS',prior.sha(args.root/'verification/features.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--joint-predictions',type=Path,required=True);p.add_argument('--joint-assessment',type=Path,required=True);p.add_argument('--main211',type=Path,required=True);p.add_argument('--field-go',action='store_true');run(p.parse_args())
