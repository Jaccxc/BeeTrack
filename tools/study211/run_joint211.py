"""Explicit joint211 field entry, gated; importing performs no field work.

prepare: only original K/catalogs, complete peers, already sealed main models.
score: only AFTER all four joint choices seal, then complete native GT scoring.
Requires a root-reviewed job spec and explicit sole-CPU handoff for EACH stage.
No default input paths, new seed discovery, training, images, GPU or Git calls.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

SUPPLEMENT='3323107ea3560e9ee507e65891097cf2ca605b00'
CPU_GRANT='root-unique-joint211-field-cpu'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_job(path,grant):
    if grant!=CPU_GRANT:raise PermissionError('root sole joint CPU handoff not supplied')
    job=json.loads(Path(path).read_text())
    if job['prereg']!=SUPPLEMENT or job['marker']!='[cross-family-peer-v1]':
        raise ValueError('root-verified supplemental registration required')
    if len(job['clips'])!=6 or {c['domain'] for c in job['clips']}!={'35','33','W','G'}:
        raise ValueError('exact original six-clip/four-domain roster required')
    if len({(c['domain'],c['clip']) for c in job['clips']})!=6:
        raise ValueError('duplicate native clip')
    for key in ('prediction_out','score_out'):
        if Path(job[key]).resolve().drive.upper()!='D:':raise ValueError('new outputs belong on D:')
    return job


def dependencies(job):
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    repo=Path(job['repo']);local=Path(__file__).parent
    sys.path.insert(0,str(local));sys.path.insert(0,str(repo))
    import numpy as np
    import joint_deploy211 as deploy
    import joint_actions211 as joint
    import graphs211 as graphs
    import freeze211 as freeze
    import fit211 as fit
    co=module('joint_original_catalog209',local.parent/'study209'/'catalog209.py')
    return np,deploy,joint,graphs,freeze,fit,co


def verify_priors(job,freeze,fit):
    import build_joint_job211
    build_joint_job211.validate_job(job,cpu_grant=CPU_GRANT)
    freeze.verify(Path(job['main_frozen']))
    if (Path(job['main_frozen'])/'SEALED.sha256').read_text().strip()!=job['main_frozen_seal']:
        raise ValueError('main peer freeze receipt differs')
    fit.verify_seal(Path(job['main_models']))
    if (Path(job['main_models'])/'SEALED.sha256').read_text().strip()!=job['main_parameter_seal']:
        raise ValueError('main all-four coefficient/choice seal differs')
    main=json.loads((Path(job['main_frozen'])/'deployment.json').read_text())
    if {(c['domain'],c['clip']) for c in main['clips']}!={(c['domain'],c['clip']) for c in job['clips']}:
        raise ValueError('joint clip whitelist differs from original main22')
    if main['states']!=22:raise ValueError('original main22 closure required')


def receipts(job):
    local=Path(__file__).parent
    code={name:file_sha(local/name) for name in ('run_joint211.py','joint_deploy211.py',
                                              'joint_actions211.py','graphs211.py','fit211.py','freeze211.py',
                                              'build_joint_job211.py','true211.py')}
    return dict(supplement_prereg=job['prereg'],main_parameter_seal=job['main_parameter_seal'],
                main_frozen_seal=job['main_frozen_seal'],code_sha256=code,
                job_sha256=hashlib.sha256(json.dumps(job,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def load_unit(job,c,np,freeze,co):
    main=freeze.load_clip(Path(job['main_frozen'])/c['main_relative'])
    meta=main['metadata']
    if (meta['domain'],meta['clip'])!=(c['domain'],c['clip']):raise ValueError('frozen clip linkage differs')
    expected_seed=3 if c['domain'] in ('35','33') else 0
    if meta['seed']!=expected_seed or not 0<=meta['seed']<=14:raise ValueError('original seed whitelist differs')
    original=co.load_clip(Path(c['catalog209']),Path(job['inputs208']))
    if not np.array_equal(original['rows'],main['native']['receiver']) or not np.array_equal(original['canon'],main['native']['receiver_canon']):
        raise ValueError('209/main211 original K/canon differs')
    if original['metadata']['nf']!=meta['nf'] or original['metadata']['polygon']!=meta['polygon']:
        raise ValueError('original209 clock/polygon mismatch')
    unit=dict(unit=meta['unit'],rows=main['native']['receiver'],canon=main['native']['receiver_canon'],
              polygon=meta['polygon'],nframes=meta['nf'],peer_rows=[main['native']['peer1'],main['native']['peer2']])
    return unit,main,original


def prepare(job):
    np,deploy,joint,graphs,freeze,fit,co=dependencies(job)
    verify_priors(job,freeze,fit)
    from tools.study197.candidates import event_times,window_counts
    docs={h:dict(domain=h,clips=[]) for h in ('35','33','W','G')}
    for c in job['clips']:
        unit,main,original=load_unit(job,c,np,freeze,co)
        loaded=json.loads((Path(job['main_models'])/'models'/f'{c["domain"]}.json').read_text())
        if loaded['heldout']!=c['domain'] or set(loaded['source_domains'])!={'35','33','W','G'}-{c['domain']}:
            raise ValueError('sealed main model source scope mismatch')
        # Do not propagate source labels or original GT IDs into deployment.
        model={p:{k:loaded[p][k] for k in ('a','b')} for p in ('Q','L','X')}
        ipairs=main['active_pairs'];isl=[p['pair_id'] for p in ipairs]
        if set(isl)!=set(c['original210_active_slate']):raise ValueError('210 original fixed slate/all-active mismatch')
        roster=joint.make_roster(original['pairs'],c['original209_common_slate'],c['original209_delta_by_pair'],
                                 ipairs,c['original210_active_slate'],source_seals=c['source_seals'])
        count=lambda rows:window_counts(event_times(rows,unit['polygon']),unit['nframes'])
        prediction=deploy.build_native_clip(unit,roster,model,joint=joint,graphs=graphs,count_windows=count)
        prediction['main210_closure']=deploy.assert_main210_closure(prediction,main,main_rows=lambda s:freeze.state_rows(main,s))
        prediction.update(domain=c['domain'],clip=c['clip'])
        docs[c['domain']]['clips'].append(prediction)
        print('JOINT_GT_FREE_CLIP',c['domain'],c['clip'],len(prediction['states']),flush=True)
    seal=deploy.write_four_fold_seal(job['prediction_out'],docs,receipts(job))
    print('JOINT_ALL_FOUR_CHOICES_SEALED',deploy.digest(seal),flush=True)


def score(job):
    np,deploy,joint,graphs,freeze,fit,co=dependencies(job)
    verify_priors(job,freeze,fit)
    pinned=receipts(job)
    gate=deploy.verify_target_gate(job['prediction_out'],pinned,cpu_grant=CPU_GRANT)
    out=Path(job['score_out'])
    if out.exists():raise FileExistsError('immutable new assessment directory required')
    out.mkdir(parents=True)
    # First import/read true annotations ONLY below the complete choices gate.
    from beetrack.evaluate import load_tracks
    from tools.study195.run import score_member_native
    import true211
    main_assessment=Path(job['main_frozen']).parent/'assessment'
    main_assessment_sha=true211.verify(main_assessment)
    if main_assessment_sha!='7df94f0fbac521d715349651b5f9c1527854748535ea70399733c23b813a8d81':
        raise ValueError('original main22 complete true assessment seal differs')
    main_index=json.loads((main_assessment/'index.json').read_text())
    scorer_receipt=deploy.digest({str(p.relative_to(job['repo'])):file_sha(p) for p in
        (Path(job['repo'])/'beetrack'/'evaluate.py',Path(job['repo'])/'tools'/'study195'/'run.py',
         Path(job['repo'])/'tools'/'study193'/'source193.py')})
    cache={};records=[]
    for c in job['clips']:
        unit,main,_=load_unit(job,c,np,freeze,co)
        fold=json.loads((Path(job['prediction_out'])/f'{c["domain"]}.joint_predictions.json').read_text())
        prediction=next(x for x in fold['clips'] if x['clip']==c['clip'])
        # Explicit original six GT paths in reviewed job; never discover splits.
        allowed=Path(job['repo'])/'runs'/'gt'/main['metadata']['split']/main['metadata']['clip']/'gt'/'gt.txt'
        if Path(c['gt_path']).resolve()!=allowed.resolve():
            raise ValueError('GT path outside original main22 clip/split whitelist')
        GT=load_tracks(Path(c['gt_path']))
        scores=deploy.score_native_states(prediction,unit,GT,score_native=score_member_native,
            gate=gate,expected_gate=gate,reuse=cache,scorer_receipt=scorer_receipt)
        reference_clip=next(x for x in main_index['clips'] if (x['domain'],x['clip'])==(c['domain'],c['clip']))
        references={s['state_id']:s for s in reference_clip['states']}
        closure={x['state_id']:x['main_state_id'] for x in prediction['main210_closure']['states']}
        for sid,record in scores.items():
            state=next(s for s in prediction['states'] if s['state_id']==sid)
            metrics=record['metrics']
            if metrics['pred_bins']!=state['windows']:raise AssertionError('true/native full30phase windows differ')
            record['true_graphs']={t:true211.graph_from_metrics(metrics,t) for t in ('0.5','0.3')}
            if sid in closure:
                reference=references[closure[sid]];folder=main_assessment/reference['path']
                if true211.verify(folder)!=reference['seal']:raise ValueError('main22 state seal differs')
                original_score=json.loads((folder/'score.json').read_text())
                deploy.assert_reference(metrics,original_score['metrics'])
                deploy.assert_reference(record['true_graphs'],original_score['true_graphs'])
                record['main210_true_exact']=dict(main_state_id=closure[sid],metrics_exact=True,
                                                 legal_graphs_both_thresholds_exact=True)
        records.append(dict(domain=c['domain'],clip=c['clip'],states=scores,
                            selected=prediction['choices'],family_only=prediction['family_only']))
        print('JOINT_TRUE_CLIP_COMPLETE',c['domain'],c['clip'],len(scores),flush=True)
    payload=dict(schema='joint211-native-assessment-v1',prediction_seal=deploy.digest(gate),
                 receipts=pinned,clips=records,all_legal_states_scored=True,
                 main22_true_reference_seal=main_assessment_sha,
                 numerical_replay_requires_anonymized_projection=True)
    data=deploy.canonical_bytes(payload)+b'\n';(out/'assessment.json').write_bytes(data)
    manifest={'assessment.json':hashlib.sha256(data).hexdigest()}
    (out/'manifest.json').write_bytes(deploy.canonical_bytes(manifest)+b'\n')
    (out/'SEALED.sha256').write_text(file_sha(out/'manifest.json')+'\n')
    print('JOINT_TRUE_ALL_STATES_SEALED',file_sha(out/'manifest.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','score'))
    parser.add_argument('--job',required=True,type=Path)
    parser.add_argument('--cpu-grant',required=True)
    args=parser.parse_args();job=load_job(args.job,args.cpu_grant)
    (prepare if args.stage=='prepare' else score)(job)
