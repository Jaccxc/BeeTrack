"""Late212 receiver checks and five original211 singleton rule forensics.

No old true outcome is opened until the complete four-fold predictions seal is
validated. The early receiver-source label archive is never modified.
"""
import os
import sys
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import gzip
import tempfile
import numpy as np


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('complete top seal required')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=h:raise ValueError('sealed file drift '+name)
    return sha(root/'manifest.json'),manifest


def anonymize_GT(details):
    """Stable within-case aliases preserve before/after owner relations."""
    scalar={'GT_canonical_id','before_GT_owner','after_GT_owner'};found=set()
    def collect(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in scalar and isinstance(v,(int,np.integer)) and v>=0:found.add(int(v))
                elif k=='legal_GT_neighbors':found.update(map(int,v))
                else:collect(v)
        elif isinstance(x,list):
            for v in x:collect(v)
    collect(details);mapping={v:'g'+str(i) for i,v in enumerate(sorted(found))}
    def replace(x):
        if isinstance(x,dict):
            result={}
            for k,v in x.items():
                if k in scalar:result[k]=None if v is None or v<0 else mapping[int(v)]
                elif k=='legal_GT_neighbors':result[k]=[mapping[int(owner)] for owner in v]
                else:result[k]=replace(v)
            return result
        if isinstance(x,list):return [replace(v) for v in x]
        return x
    return replace(details)


def assert_211_graph(graph,metrics,threshold):
    """Historical prediction vertices have six columns, GT vertices two."""
    evidence=metrics['evidence'];old={int(v[0]):v for v in evidence['predictions']}
    predicted=sorted([e['canonical_event_id'],e['signed_frame']] for e in graph['events'])
    reference=sorted([e['canonical_event_id'],e['signed_frame']] for e in graph['GT_events'])
    if predicted!=sorted(v[:2] for v in evidence['predictions']) or reference!=sorted(evidence['gt']):
        raise ValueError('receiver211 signed vertex/date mismatch')
    for event in graph['events']:
        row=event['all_same_direction_crossings'][-1]['after_row'];x,y,w,h=row[2:6]
        if not np.array_equal([x,y,x+w,y+h],old[event['canonical_event_id']][2:6]):raise ValueError('211 event representative box mismatch')
    key='edges05' if threshold==.5 else 'edges03'
    if sorted(graph['edges'])!=sorted(evidence[key]):raise ValueError('211 legal GT graph edges mismatch')
    if (graph['nPred'],graph['nGT'],graph['TP'])!=(metrics['nP'],metrics['nG'],metrics['iou'][str(threshold)]['tp']):
        raise ValueError('211 whole-graph cardinality mismatch')
    return dict(passed=True,scope='signed vertices, representative boxes, legal edges and cardinality exact')


def run(args):
    sys.path.insert(0,str(args.repo.resolve()));base=Path(__file__).parent;sys.path.insert(0,str(base))
    # The only payload opened before this gate is the common prediction itself.
    fit=module('phase212_late_fit_gate',base/'fit212.py');fit.verify_seal(args.predictions)
    prediction_sha,_=verify(args.predictions)
    if args.out.exists() or args.out.resolve().drive.upper()!='D:':raise ValueError('fresh D: late output required')
    frozen_sha,_=verify(args.frozen);truth_sha,truth_manifest=verify(args.truth)
    truth_deployment=json.loads((args.truth/'deployment.json').read_text())
    if truth_deployment['prereg']!=args.prereg or truth_deployment['frozen_manifest_sha256']!=frozen_sha:
        raise ValueError('early truth/frozen/prereg mismatch')
    # Import the exact source bytes sealed with the early fresh-GT labels.
    T=module('phase212_early_sealed_core',args.truth/'truth212.py')
    if sha(args.truth/'truth212.py')!=truth_deployment['code_sha256']:raise ValueError('early executed core source differs')
    old_sha,oldmanifest=verify(args.assessment211)
    if old_sha!=T.SOURCE211_ASSESSMENT:raise ValueError('wrong original211 main assessment')
    oldindex=json.loads((args.assessment211/'index.json').read_text());oldclips={(x['domain'],x['clip']):x for x in oldindex['clips']}
    F=module('phase212_late_prepare',base/'prepare212.py');top=json.loads((args.frozen/'deployment.json').read_text())
    if top['prereg']!=args.prereg or top['states']!=86 or top['originalK']!=18:raise ValueError('wrong prepared feature scope')
    from beetrack.evaluate import load_tracks
    prior=module('phase212_late_GTpins',base.parent/'peer_suffix203'/'owners203.py')
    args.out.parent.mkdir(parents=True,exist_ok=True);comparisons=[];refs=[];GThashes={}
    with tempfile.TemporaryDirectory(prefix='.late212-',dir=args.out.parent) as temporary:
        target=Path(temporary)/'payload';target.mkdir()
        for ref in top['clips']:
            clip=F.load_clip(args.frozen,ref);domain=clip['domain'];name=clip['clip'];oldclip=oldclips[domain,name]
            original=next(s for s in oldclip['states'] if not s['selected_pair_ids'])
            Kscore=json.loads((args.assessment211/original['path']/'score.json').read_text());graphs={}
            index=next(x for x in truth_deployment['original_K'] if x['domain']==domain and x['clip']==name and x['member_index']==0)
            checks={}
            for t in (.5,.3):
                with gzip.open(args.truth/index['graphs'][str(t)],'rt',encoding='utf8') as stream:graphs[str(t)]=json.load(stream)
                checks[str(t)]=assert_211_graph(graphs[str(t)],Kscore['metrics'],t)
            comparisons.append(dict(domain=domain,clip=name,checks=checks,old_K_state_id=original['state_id'],
                old_score_sha256=oldmanifest[original['path']+'/score.json']))
            if domain!='35' and name not in ('BEE24-15','BEE24-24'):continue
            loaded=F.load_scene(clip,clip['K_scene_refs']['0']);receiver=loaded['rows'];canon=F.canonical_map(receiver)
            gtpath=args.repo/'runs/gt'/clip['split']/name/'gt/gt.txt';digest=sha(gtpath)
            if digest!=prior.GT_SHA256.get(name):raise ValueError('GT differs historical six pin')
            GThashes[gtpath.relative_to(args.repo).as_posix()]=digest;GT=load_tracks(gtpath)
            for state in oldclip['states']:
                if len(state['selected_pair_ids'])!=1:continue
                folder=args.assessment211/state['path'];score=json.loads((folder/'score.json').read_text())
                rows=np.loadtxt(folder/'mot.txt',delimiter=',',usecols=range(7),ndmin=2)
                order=np.load(folder/'output_raw_indices.npy',allow_pickle=False)
                if not np.array_equal(rows[:,[0,2,3,4,5,6]],receiver[order][:,[0,2,3,4,5,6]]):raise ValueError('old singleton changed original rows')
                parents=list(map(int,state['selected_pair_ids'][0].split('|')));details={}
                for t in (.5,.3):
                    fresh=T.event_graph(GT,rows,clip['polygon'],t);assert_211_graph(fresh,score['metrics'],t)
                    details[str(t)]=dict(before=T.family_forensics(graphs[str(t)],np.arange(len(receiver)),canon,parents),
                        after=T.family_forensics(fresh,order,canon,parents),
                        whole_scene_before={k:graphs[str(t)][k] for k in ('TP','FP','FN','nPred')},
                        whole_scene_after={k:fresh[k] for k in ('TP','FP','FN','nPred')})
                path=target/domain/name/(state['state_id']+'.json.gz');path.parent.mkdir(parents=True,exist_ok=True)
                F.save_gz(path,dict(domain=domain,clip=name,state_id=state['state_id'],original_family=parents,
                    details=anonymize_GT(details),GT_aliases='within-case opaque sorted canonical-owner aliases; original GT IDs omitted',
                    old211_score_sha256=oldmanifest[state['path']+'/score.json'],physical_identity_claim=False))
                refs.append(dict(domain=domain,clip=name,state_id=state['state_id'],path=path.relative_to(target).as_posix()))
            print('LATE212_CLIP_COMPLETE',domain,name,flush=True)
        if len(comparisons)!=6 or len(refs)!=5:raise ValueError('six receiver comparisons/five fixed forensic cases required')
        if any(sha(args.repo/p)!=h for p,h in GThashes.items()):raise ValueError('GT changed during forensics')
        F.save(target/'deployment.json',dict(schema='phase212-late-truth-v1',prereg=args.prereg,frozen_manifest_sha256=frozen_sha,
            label_manifest_sha256=truth_sha,prediction_manifest_sha256=prediction_sha,source211_assessment_sha256=old_sha,
            receiver211_exact=comparisons,forensics=refs,GT_sha256=GThashes,
            old_outcome_payloads_first_opened_after_common_prediction_gate=True,early_labels_unchanged=True,
            shared_graph_core_sha256=truth_deployment['code_sha256'],late_code_sha256=sha(__file__)))
        (target/'truth212_late.py').write_bytes(Path(__file__).read_bytes())
        F.save(target/'manifest.json',{p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()})
        (target/'SEALED.sha256').write_text(sha(target/'manifest.json')+'\n');target.rename(args.out)
    print('LATE212_SEALED',sha(args.out/'manifest.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--frozen',required=True,type=Path)
    p.add_argument('--truth',required=True,type=Path);p.add_argument('--predictions',required=True,type=Path)
    p.add_argument('--assessment211',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--prereg',required=True)
    run(p.parse_args())
