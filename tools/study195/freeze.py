"""GT-free two-threshold whole-member inputs; coding/self-test until prereg GO.

Only named .40/.50 old head caches and clean W/G caches are allowed. No tracker
execution, score tables, GT, image read, threshold choice, or new seed scan.
"""
import os
import sys
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
if '--repo' in sys.argv:
    sys.path.insert(0,str(Path(sys.argv[sys.argv.index('--repo')+1]).resolve()))
from tools.study193 import freeze193 as F

ARMS={'t40':0.40,'t50':0.50}


def write_mot_exact(rows,path):
    """Materialize one arm for official scoring; preserve all native float bits."""
    rows=np.asarray(rows,dtype=np.float64).reshape(-1,7)
    path=Path(path)
    with path.open('x',encoding='ascii',newline='\n') as stream:
        for row in rows:
            stream.write(','.join(format(float(x),'.17g') for x in row)+',-1,-1,-1\n')
    reread=(np.loadtxt(path,delimiter=',',usecols=range(7),ndmin=2) if len(rows) else np.empty((0,7),float))
    if reread.dtype!=rows.dtype or reread.shape!=rows.shape or reread.tobytes()!=rows.tobytes():
        raise ValueError('physical MOT roundtrip changed native values')
    return dict(sha256=F.sha(path),rows=len(rows),precision='17 significant digits; exact float64 roundtrip')


def verify(root):
    root=Path(root)
    if (root/'SEALED.sha256').read_text().strip()!=F.sha(root/'manifest.json'):
        raise ValueError('global manifest seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or F.sha(path)!=digest:
            raise ValueError('unsafe or changed frozen artifact')
    return dict(manifest_sha256=F.sha(root/'manifest.json'),deployment_sha256=F.sha(root/'deployment.json'))


def load_groups(root):
    root=Path(root);receipt=verify(root)
    data=json.loads((root/'deployment.json').read_text())
    if data['schema']!='whole-member-router-freeze-v1': raise ValueError('wrong schema')
    if data['members']!=F.members() or len(data['units'])!=60 or data['arms']!=ARMS:
        raise ValueError('whitelist or arm count changed')
    for unit in data['units']:
        member=data['members'][unit['member']]
        clip=data['clips'][unit['clip_index']]
        if set(unit['arms'])!=set(ARMS) or unit['hive']!=member['hive'] or unit['hive']!=clip['hive']:
            raise ValueError('unit mapping changed')
    result=[]
    for group in data['groups']:
        clip=data['clips'][group['clip_index']]
        mm=[]
        for uid in group['units']:
            unit=data['units'][uid];m=data['members'][unit['member']];arms={}
            for key in ARMS:
                arm=unit['arms'][key]
                with np.load(root/arm['rows_file'],allow_pickle=False) as z: raw=z['rows']
                arms[key]=dict(rows=raw,windows=arm['windows'],seconds=arm['seconds'],event_records=arm['event_records'],
                               raw_receipt=dict(file=arm['rows_file'],sha256=arm['rows_sha256'],source=arm['source_receipt']))
            mm.append(dict(member=str(m['member']),tag=m['tag'],seed=m['seed'],arms=arms))
        result.append(dict(unit=f"{group['hive']}_g{group['group']}_{clip['clip']}",domain=group['hive'],video=clip['clip'],group=group['group'],nf=clip['nframes'],polygon=clip['polygon'],split=clip['split'],members=mm,freeze_receipt=receipt,linker_model=group['linker_name'],linker_sha256=F.MODEL_SHA[group['linker_name']],linker_excluded_domains=[group['hive']] if group['hive'] in ('W','G') else ['33','35']))
    return result


def freeze(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg) or args.section!=195:
        raise ValueError('explicit full prereg commit and section195 required')
    ancestry=subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,capture_output=True)
    if ancestry.returncode!=0: raise ValueError('prereg commit is not published on origin/main')
    raw=subprocess.run(['git','show',f'{args.prereg}:docs/experiment_log.md'],cwd=args.repo,capture_output=True,check=True).stdout.decode('utf-8')
    marker='[whole-member-router-v1]'
    if raw.count(marker)!=1: raise ValueError('unique literal router registration marker required')
    temporary=re.search(r'^### §195 預登記 \[whole-member-router-v1\][^\n]*$',raw,re.M)
    formal=re.search(r'^## 195\.[^\n]*\n(?:(?!^## ).)*(?=^## |\Z)',raw,re.M|re.S)
    if not temporary and not (formal and marker in formal.group()):
        raise ValueError('correct formal or temporary section195 registration absent')
    if args.out.exists(): raise FileExistsError(args.out)
    zonepath=args.repo/'beetrack/zones.json';zones=json.loads(zonepath.read_text(encoding='utf-8'))
    mm=F.members();specs=[]
    model_receipts={name:F.sha(args.repo/'beetrack/models'/f'{name}.json') for name in F.MODEL_SHA}
    if model_receipts!=F.MODEL_SHA: raise ValueError('linker hashes changed')
    # Verify every named artifact for BOTH thresholds before opening any arrays.
    for m in mm:
        for ci,(h,sp,clip,nf) in enumerate(F.CLIPS):
            if h!=m['hive']: continue
            directory=args.head_cache if h in ('35','33') else args.clean_cache
            paths={a:directory/f"{m['tag']}_{sp}_{clip}_{threshold:.2f}.npz" for a,threshold in ARMS.items()}
            if any(not p.is_file() for p in paths.values()): raise FileNotFoundError('both explicit cache arms required')
            dp=args.repo/'runs/dets'/m['tag']/sp/f'{clip}.npy'
            specs.append((m,ci,sp,clip,nf,paths,{a:F.sha(p) for a,p in paths.items()},F.sha(dp)))
    if len(specs)!=60: raise ValueError('expected60 member/clip pairs')
    reference=None;refmap={};reference_receipt=None
    if args.reference193:
        reference_receipt=verify(args.reference193)
        reference=json.loads((args.reference193/'deployment.json').read_text())
        if reference['schema']!='budget193-deployment-v1': raise ValueError('wrong193 reference schema')
        refmap={(u['member'],u['clip_index']):u for u in reference['units']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.router-freeze-',dir=args.out.parent) as temp:
        target=Path(temp)/'payload';target.mkdir();(target/'rows').mkdir()
        units=[]
        for m,ci,sp,clip,nf,paths,hashes,detsha in specs:
            uid=len(units);arms={};linker='m_no'+m['hive'] if m['hive'] in ('W','G') else 'm_dev_strict'
            for key,threshold in ARMS.items():
                src=paths[key]
                if F.sha(src)!=hashes[key]: raise ValueError('source changed after preflight')
                with np.load(src,allow_pickle=False) as z:
                    rows=z['rows'];s1=z['s1']
                if rows.dtype!=np.float64 or s1.dtype!=np.float64: raise ValueError('full float64 source required')
                events=F.extract(rows,s1,zones[clip],nf)
                windows,seconds=F.counts(events,nf)
                if key=='t40' and reference is not None:
                    ref=refmap[m['member'],ci]
                    with np.load(args.reference193/ref['rows_file'],allow_pickle=False) as z:
                        for name,actual in [('rows',rows),('s1',s1)]:
                            if actual.dtype!=z[name].dtype or actual.shape!=z[name].shape or actual.tobytes()!=z[name].tobytes():
                                raise ValueError('t40 differs from193 fullprecision reference')
                    if windows!=ref['windows'] or seconds!=ref['seconds']: raise ValueError('t40 clocks differ from193')
                    original=[{k:e[k] for k in ('track_id','frame','sign','features','canonical_row','box')} for e in reference['events'] if e['unit']==ref['unit']]
                    if events!=original: raise ValueError('t40 events/features differ from193')
                file=f'rows/u{uid:02d}_{key}.npz';np.savez(target/file,rows=rows,s1=s1)
                ev=[dict(pid=e['track_id'],frame=e['frame'],direction=e['sign'],box=e['box'],features10=e['features']) for e in events]
                arms[key]=dict(threshold=threshold,rows_file=file,rows_sha256=F.sha(target/file),row_count=len(rows),event_records=ev,windows=windows,seconds=seconds,
                               source_receipt=dict(kind='section189_head_cache' if m['hive'] in ('35','33') else 'section192_clean_cache',cache_name=src.name,cache_sha256=hashes[key],detector_sha256=detsha,linker_name=linker,linker_sha256=F.MODEL_SHA[linker],precision='native unrounded float64'))
            units.append(dict(unit=uid,member=m['member'],hive=m['hive'],group=m['group'],seed=m['seed'],tag=m['tag'],clip_index=ci,clip=clip,split=sp,nframes=nf,arms=arms))
        groups=[]
        for h,sp,clip,nf in F.CLIPS:
            ci=next(i for i,s in enumerate(F.CLIPS) if s[2]==clip)
            for group in sorted(set(m['group'] for m in mm if m['hive']==h)):
                us=[u['unit'] for u in units if u['hive']==h and u['group']==group and u['clip_index']==ci]
                if len(us)!=3: raise ValueError('three orderedmembers required')
                groups.append(dict(hive=h,group=group,clip_index=ci,units=us,linker_name='m_no'+h if h in ('W','G') else 'm_dev_strict'))
        algorithm_hashes={name:F.sha(args.repo/'beetrack'/name) for name in ('p0.py','track.py','link.py','evaluate.py','common.py')}
        for name,folder in [('score189.py',args.head_cache.parent),('score192.py',args.clean_cache.parent)]:
            algorithm_hashes[name]=F.sha(folder/name) if (folder/name).is_file() else None
        data=dict(schema='whole-member-router-freeze-v1',prereg=args.prereg,section=args.section,arms=ARMS,members=mm,units=units,groups=groups,
                  clips=[dict(clip_index=i,hive=h,split=sp,clip=c,nframes=nf,polygon=zones[c]) for i,(h,sp,c,nf) in enumerate(F.CLIPS)],
                  clock=dict(fps=25,frame_origin=1,offset_seconds=list(range(30)),window_frames=750,layout='stackoffset0..29; eachnbins=ceil((nf+25*offset)/750); in,out columns; bin=(frame+25*offset-1)//750; include partialwindows',seconds='(frame-1)//25'),
                  source_hashes=dict(freezer=F.sha(__file__),study193_freezer=F.sha(F.__file__),zones=F.sha(zonepath),linkers=model_receipts,algorithms=algorithm_hashes),reference193_receipt=reference_receipt,
                  physical_scoring='write_mot_exact materializes one Nx7 arm to17g fullprecision MOT; rereadbit equality checked; no rowmix')
        (target/'deployment.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
        manifest={p.relative_to(target).as_posix():F.sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (target/'SEALED.sha256').write_text(F.sha(target/'manifest.json')+'\n')
        target.rename(args.out)
    print('Sealed120 two-threshold arms withoutGT:',args.out)


def selfcheck():
    F.selfcheck()
    rows=np.array([[1,1,1.23456789012345,2,3,4,.4],[3,1,2,2,3,4,.7]],float)
    with tempfile.TemporaryDirectory(prefix='whole-router-synthetic-') as td:
        write_mot_exact(rows,Path(td)/'native.txt')
        write_mot_exact(np.empty((0,7)),Path(td)/'empty.txt')
    assert tuple(ARMS)==('t40','t50')
    print('whole-member freezer synthetic fullprecisionphysical roundtrip passed')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path)
    ap.add_argument('--head-cache',type=Path);ap.add_argument('--clean-cache',type=Path)
    ap.add_argument('--reference193',type=Path);ap.add_argument('--prereg');ap.add_argument('--section',type=int)
    ap.add_argument('--self-check',action='store_true')
    args=ap.parse_args()
    if args.self_check: selfcheck()
    else:
        if any(getattr(args,n) is None for n in ('out','head_cache','clean_cache','prereg','section')):
            ap.error('--out,--head-cache,--clean-cache,--prereg,--section required')
        args.repo=args.repo.resolve();args.out=args.out.resolve()
        freeze(args)
