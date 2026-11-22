"""Automatic joint211 job construction; NO work on import.

Runtime is a field-stage operation requiring labels185's sole CPU handoff.
Only original GT-free catalogs/K/peers/choices and sealed main coefficients
are used. No joint results, assessment, score file or GT file is opened.
The only required changing CLI input is --main-model-seal.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

SUPPLEMENT='3323107ea3560e9ee507e65891097cf2ca605b00'
CPU_GRANT='root-unique-joint211-field-cpu'
PINS=dict(catalog209='daba4221ee9d297c4d47c786f550d9f62d6aef3318feec3bc0fc7128ffae696f',
          features209='c39fdb1b28b79450e0706c6b52f7f629f40d907d175da199067307657d286bda',
          predictions209='238bda44bcbf29fa5cc42db2139d07aff9f386baa36ed49b0a50ffb113bc511e',
          catalog210='3810c1afabe45160642b6187c8fed89dac4e5e05dabb9f5be2cb696d666a7005',
          main_frozen='c21b74bbf3accd85a8a48863d16c1d241ce3045032b67ab8e6d92f390b88d6ae',
          inputs208='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94')
SIX={('35','BEE24-35-head',3),('33','BEE24-33-head',3),('W','BEE24-10',0),('W','BEE24-15',0),('G','BEE24-24',0),('G','BEE24-28',0)}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def wire(x):return (json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def sealed(root,expected,verify_all=True):
    root=Path(root)
    if sha(root/'manifest.json')!=expected or (root/'SEALED.sha256').read_text().strip()!=expected:
        raise ValueError('fixed parent seal mismatch: '+str(root))
    files=json.loads((root/'manifest.json').read_text())
    for name,digest in files.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()):raise ValueError('manifest escape')
        if verify_all and sha(p)!=digest:raise ValueError('parent file drift '+name)
    return files


def checked(root,files,name):
    path=Path(root)/name
    if name not in files or sha(path)!=files[name]:raise ValueError('named signed input drift '+name)
    return path


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj;spec.loader.exec_module(obj);return obj


def common_slate(pairs,valid):
    table={p['pair_id']:p for p in pairs}
    if set(valid)!=set(table):raise ValueError('complete209 true.valid mask required')
    used=set();slate=[]
    for p in sorted((p for p in pairs if p['geometric_eligible'] and valid[p['pair_id']]),
                    key=lambda p:(-p['mean_iou'],*p['canonical_ids'])):
        if used.intersection(p['canonical_ids']):continue
        used.update(p['canonical_ids']);slate.append(p['pair_id'])
    return slate


def local_deltas(rows,canon,pairs,slate,polygon,nf):
    """Official count-only single209 unions, original global row-index tie rule.

    No GT scalar is accepted. Nonshared dates retain every original observation;
    shared winners use score then original raw index. Local counts are additive
    only because this is the ORIGINAL disjoint209 common slate.
    """
    import numpy as np
    from tools.study197.candidates import event_times,window_counts
    R=np.asarray(rows,float);C=np.asarray(canon,np.int64);table={p['pair_id']:p for p in pairs}
    order=np.argsort(C,kind='stable');unique,starts=np.unique(C[order],return_index=True)
    runs={int(pid):order[a:b][np.argsort(R[order[a:b],0],kind='stable')] for pid,a,b in zip(unique,starts,[*starts[1:],len(R)])}
    result={};used=set()
    for key in slate:
        p=table[key];a,b=p['canonical_ids']
        if used.intersection((a,b)):raise ValueError('original common slate not disjoint')
        used.update((a,b));ia,ib=runs[a],runs[b]
        left={int(R[j,0]):int(j) for j in ia};right={int(R[j,0]):int(j) for j in ib}
        dates=sorted(left.keys() & right.keys());shared=[[left[f],right[f]] for f in dates]
        if not dates or dates!=p['shared_dates'] or shared!=p['row_indices']:raise ValueError('original209 pair raw/date mapping differs')
        drop=set()
        for i,j in shared:
            winner,loser=(i,j) if R[i,6]>R[j,6] or R[i,6]==R[j,6] and i<j else (j,i)
            drop.add(loser)
        union=np.r_[ia,ib];kept=np.array([j for j in union if j not in drop],np.int64)
        merged=R[kept].copy();merged[:,1]=a
        chronological=np.sort(merged[:,0])
        if (np.diff(chronological)>25).any() or len(set(chronological))!=len(chronological):raise ValueError('209 merged run invalid')
        before=window_counts(event_times(R[union],polygon),nf);after=window_counts(event_times(merged,polygon),nf)
        result[key]=(after-before).tolist()
    return result


def build_job(args):
    if args.cpu_grant!=CPU_GRANT:raise PermissionError('sole joint field CPU handoff required even for job generation')
    if not re.fullmatch('[0-9a-f]{64}',args.main_model_seal):raise ValueError('explicit final main model SHA256 required')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    repo=args.repo.resolve();local=Path(__file__).parent;sys.path.insert(0,str(repo));sys.path.insert(0,str(local))
    import numpy as np
    import freeze211
    import fit211
    roots={key:Path(getattr(args,key)).resolve() for key in PINS}
    manifests={key:sealed(root,PINS[key],verify_all=key not in ('inputs208',)) for key,root in roots.items()}
    sealed(args.main_models,args.main_model_seal);fit211.verify_seal(args.main_models)
    # No true labels/source shards are opened; main models are only verified.
    co=module('joint211_job_catalog209',local.parent/'common209/catalog209.py')
    with (checked(roots['predictions209'],manifests['predictions209'],'predictions.json')).open(encoding='utf8') as f:pred209=json.load(f)
    top209=json.loads(checked(roots['catalog209'],manifests['catalog209'],'deployment.json').read_text())
    top210=json.loads(checked(roots['catalog210'],manifests['catalog210'],'deployment.json').read_text())
    main=json.loads(checked(roots['main_frozen'],manifests['main_frozen'],'deployment.json').read_text())
    if main['states']!=22 or len(main['clips'])!=6:raise ValueError('original22 closure missing')
    refs209={(r['domain'],r['clip']):r for r in top209['clips']};refs210={(r['domain'],r['clip']):r for r in top210['clips']}
    if {(r['domain'],r['clip'],r['seed']) for r in top209['clips']}!=SIX:raise ValueError('original209 fixed-six whitelist differs')
    if {(r['domain'],r['clip'],r['seed']) for r in top210['clips']}!=SIX:raise ValueError('original210 fixed-six whitelist differs')
    clips=[]
    for ref in main['clips']:
        domain,clip=ref['domain'],ref['clip'];key=(domain,clip);path=ref['path']
        native=freeze211.load_clip(roots['main_frozen']/path);meta=native['metadata']
        original=co.load_clip(roots['catalog209']/refs209[key]['path'],roots['inputs208'])
        if not np.array_equal(original['rows'].view(np.uint64),native['native']['receiver'].view(np.uint64)) or not np.array_equal(original['canon'],native['native']['receiver_canon']):
            raise ValueError('209/211 immutable originalK differs')
        with gzip.open(roots['main_frozen']/path/'sources.json.gz','rt',encoding='utf8') as f:main_sources=json.load(f)
        if original['source_keys']!=main_sources['receiver']:raise ValueError('209/211 original detector row lineage differs')
        if original['metadata']['nf']!=meta['nf'] or original['metadata']['polygon']!=meta['polygon']:
            raise ValueError('209/211 original clock/polygon differs')
        feature_path=checked(roots['features209'],manifests['features209'],path+'/features.json')
        pixels=json.loads(feature_path.read_text());valid={p['pair_id']:p['true']['valid'] for p in pixels['pairs']}
        common=common_slate(original['pairs'],valid)
        old=pred209['clips'][domain+'/'+clip]
        if old['slate']!=common:raise ValueError('reconstructed common slate differs original sealed209 slate')
        with gzip.open(checked(roots['catalog210'],manifests['catalog210'],refs210[key]['path']+'/catalog.json.gz'),'rt',encoding='utf8') as f:inter=json.load(f)
        active=inter['active_slate_ids'];allactive={p['pair_id'] for p in inter['pairs'] if p['active']}
        if set(active)!=allactive or allactive!={p['pair_id'] for p in native['active_pairs']}:raise ValueError('original210 all-active/slate/main22 lineage differs')
        endpoints=[i for p in native['active_pairs'] for i in p['canonical_ids']]
        if len(endpoints)!=len(set(endpoints)):raise ValueError('original210 active family is not disjoint')
        oldtable={p['pair_id']:p for p in inter['pairs']}
        for p in native['active_pairs']:
            if (p['delta_windows']!=oldtable[p['pair_id']]['delta_windows'] or
                    p['min_offset_norm']!=oldtable[p['pair_id']]['min_offset_norm']):raise ValueError('main/210 delta/local-distance lineage differs')
        delta=local_deltas(original['rows'],original['canon'],original['pairs'],common,meta['polygon'],meta['nf'])
        clips.append(dict(domain=domain,clip=clip,main_relative=path,catalog209=str(roots['catalog209']/refs209[key]['path']),
            original209_common_slate=common,original209_delta_by_pair=delta,original210_active_slate=active,
            source_seals={'209':PINS['catalog209'],'210':PINS['catalog210']},
            gt_path=str(repo/'runs/gt'/meta['split']/meta['clip']/'gt/gt.txt')))
    if {(c['domain'],c['clip']) for c in clips}!={(d,c) for d,c,seed in SIX}:raise ValueError('no missing/replaced joint clip')
    job=dict(prereg=SUPPLEMENT,marker='[cross-family-peer-v1]',repo=str(repo),main_frozen=str(roots['main_frozen']),
        main_frozen_seal=PINS['main_frozen'],main_models=str(Path(args.main_models).resolve()),main_parameter_seal=args.main_model_seal,
        inputs208=str(roots['inputs208']),prediction_out=str(args.prediction_out.resolve()),score_out=str(args.score_out.resolve()),clips=clips,
        original_parent_seals=dict(PINS),builder_source_sha256=sha(__file__),
        parent_roots={key:str(root) for key,root in roots.items()},
        input_job_proof=dict(original209_catalog=dict(root=str(roots['catalog209']),seal=PINS['catalog209']),
            original209_predictions=dict(root=str(roots['predictions209']),seal=PINS['predictions209']),
            original209_features=dict(root=str(roots['features209']),seal=PINS['features209']),
            original210_catalog=dict(root=str(roots['catalog210']),seal=PINS['catalog210']),
            original211_frozen=dict(root=str(roots['main_frozen']),seal=PINS['main_frozen']),
            main_parameters=dict(root=str(Path(args.main_models).resolve()),seal=args.main_model_seal),
            singleton209_delta=dict(origin='prediction-only original-source-rule replay, not a true-GT outcome table',
                builder_code_sha256=sha(__file__),official_event_helper_sha256=sha(repo/'tools/study197/candidates.py'),
                official_evaluate_sha256=sha(repo/'beetrack/evaluate.py'),original_catalog_seal=PINS['catalog209'])),
        count_delta_source='runtime prediction-only local209 physical union; no GT/scorer outcomes',GT_read_during_job_generation=False)
    for key in ('prediction_out','score_out'):
        if Path(job[key]).drive.upper()!='D:':raise ValueError('D: outputs required')
    return job


def validate_job(job,*,cpu_grant):
    """Full GT-free reconstruction, rejecting manually altered slates/deltas/paths.

    Field runtime only: run under the same sole CPU grant as job generation.
    No joint assessment/GT file is read. Code hashes and signed original choices
    are checked again, rather than trusting the job's candidate list.
    """
    from types import SimpleNamespace
    if cpu_grant!=CPU_GRANT:raise PermissionError('sole joint field CPU handoff required')
    fields={key:Path(value) for key,value in job['parent_roots'].items()}
    fields.update(repo=Path(job['repo']),main_models=Path(job['main_models']),main_model_seal=job['main_parameter_seal'],
                  prediction_out=Path(job['prediction_out']),score_out=Path(job['score_out']),cpu_grant=cpu_grant)
    expected=build_job(SimpleNamespace(**fields))
    if wire(expected)!=wire(job):raise ValueError('automatic input job differs from signed sources/code; no manual slate/delta edits')
    return dict(job_sha256=hashlib.sha256(wire(job)).hexdigest(),all_original_parent_seals_checked=True,
                common_slate_and_singleton_deltas_rebuilt=True,GT_access=False)


def publish_job(job,path):
    path=Path(path);data=wire(job)
    if path.exists():
        if path.read_bytes()!=data:raise FileExistsError('immutable joint job differs')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data)
        if path.exists():raise FileExistsError(path)
        Path(temp).rename(path)
    finally:
        if Path(temp).exists():Path(temp).unlink()


def self_test(repo):
    sys.path.insert(0,str(repo));import numpy as np
    from tools.study197.candidates import event_times,window_counts
    from tools.study209.physical209 import materialize
    R=np.array([[f,p,x,0,2,2,.9] for p in (1,2) for f,x in [(1,-3),(2,0),(3,1)]],float)
    C=R[:,1].astype(np.int64);poly=[(0,-10),(20,-10),(20,10),(0,10)]
    pair=dict(pair_id='1|2',canonical_ids=[1,2],geometric_eligible=True,mean_iou=1.,shared_dates=[1,2,3],row_indices=[[0,3],[1,4],[2,5]])
    delta=local_deltas(R,C,[pair],['1|2'],poly,10)['1|2']
    merged,_=materialize(R,C,[pair],['1|2'])
    assert np.array_equal(delta,window_counts(event_times(merged,poly),10)-window_counts(event_times(R,poly),10))
    assert int(np.abs(delta).sum())==30
    assert common_slate([pair],{'1|2':True})==['1|2'] and common_slate([pair],{'1|2':False})==[]
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'job.json';publish_job({'synthetic':True},p);publish_job({'synthetic':True},p)
        try:publish_job({'synthetic':False},p)
        except FileExistsError:pass
        else:raise AssertionError('job overwritten')
    return dict(passed=True,scope='synthetic local209δ exact original materializer; slate; immutable job; NO FIELD')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--main-model-seal');parser.add_argument('--cpu-grant',default=CPU_GRANT)
    defaults=dict(catalog209='common209/catalog',features209='common209/features',predictions209='common209/predictions',
                  catalog210='interleave210/catalog',main_frozen='match211/frozen',inputs208='switch208/inputs',main_models='match211/predictions',
                  prediction_out='match211/joint_predictions',score_out='match211/joint_score',out='match211/joint_job.json')
    for name,relative in defaults.items():parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(args.repo.resolve())))
    else:
        if args.main_model_seal is None:parser.error('--main-model-seal required after main all-four seal/CPU release')
        job=build_job(args);publish_job(job,args.out);print('JOINT_JOB',args.out,'SHA256',sha(args.out))
