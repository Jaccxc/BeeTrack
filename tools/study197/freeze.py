"""Seal only §195 fixed-F60 arms for endpoint cuts, before any GT access."""
import os
import sys
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
if '--repo' in sys.argv:sys.path.insert(0,str(Path(sys.argv[sys.argv.index('--repo')+1]).resolve()))
from tools.study193 import freeze193 as F
from candidates import build_candidates,_canonical

SOURCE_SEAL='4f78f24d7fd13dc2eca9b206501a5326b4e42d476e1a9c4dd1ba98fe4bf61377'
MARKER='[endpoint-cut-v1]'


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x


def save(path,x):Path(path).write_text(json.dumps(plain(x),separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')


def verify(root):
    root=Path(root)
    if (root/'SEALED.sha256').read_text().strip()!=F.sha(root/'manifest.json'):raise ValueError('freeze seal mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or F.sha(path)!=digest:raise ValueError('frozen artifact changed')
    return dict(manifest_sha256=F.sha(root/'manifest.json'),deployment_sha256=F.sha(root/'deployment.json'))


def load_groups(root):
    root=Path(root);receipt=verify(root);data=json.loads((root/'deployment.json').read_text())
    if data['schema']!='endpoint-cut-freeze-v1' or data['source195_seal']!=SOURCE_SEAL:raise ValueError('wrong source freeze')
    result=[]
    for group in data['groups']:
        members=[]
        for record in group['members']:
            with np.load(root/record['rows_file'],allow_pickle=False) as z:raw=z['rows']
            _,canonical,index=_canonical(raw,group['nf'])
            bundle=dict(raw_rows=raw,canonical_rows=canonical,source_raw_index=index,polygon=group['polygon'],nframes=group['nf'],**record['bundle'])
            bundle['baseline_windows']=np.asarray(bundle['baseline_windows'],dtype=np.int64)
            for c in bundle['candidates']:c['delta_windows']=np.asarray(c['delta_windows'],dtype=np.int64)
            members.append(dict(record,rows=raw,raw=raw,bundle=bundle,candidates=bundle['candidates'],baseline_windows=bundle['baseline_windows'],
                                unit_key=f"{group['unit']}|{record['member_index']}",raw_receipt=dict(file=record['rows_file'],sha256=record['rows_sha256'],source=record['source_receipt'])))
        result.append(dict(group,members=members,freeze_receipt=receipt))
    return result


def freeze(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg):raise ValueError('full prereg commit required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,capture_output=True,check=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,capture_output=True,check=True).stdout.decode('utf-8')
    if log.count(MARKER)!=1:raise ValueError('unique endpoint prereg marker required')
    if args.out.exists():raise FileExistsError(args.out)
    source=args.source195
    if F.sha(source/'manifest.json')!=SOURCE_SEAL or (source/'SEALED.sha256').read_text().strip()!=SOURCE_SEAL:
        raise ValueError('fixed195 source seal mismatch')
    source_manifest=json.loads((source/'manifest.json').read_text())
    if F.sha(source/'deployment.json')!=source_manifest['deployment.json']:raise ValueError('source195 metadata changed')
    data=json.loads((source/'deployment.json').read_text())
    if data['members']!=F.members() or len(data['units'])!=60:raise ValueError('source195 whitelist changed')
    # Exactly the60 authorized F arrays are opened/hashed, not the other60 arm.
    for unit in data['units']:
        arm=unit['arms']['t50' if unit['hive'] in ('35','33') else 't40']
        if F.sha(source/arm['rows_file'])!=source_manifest[arm['rows_file']] or arm['rows_sha256']!=source_manifest[arm['rows_file']]:
            raise ValueError('selected complete F arm hash mismatch')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.endpoint-freeze-',dir=args.out.parent) as tmp:
        dest=Path(tmp)/'payload';dest.mkdir();(dest/'rows').mkdir()
        groups=[];total=Counter();all_candidates=[]
        for group in data['groups']:
            clip=data['clips'][group['clip_index']]
            unit_name=f"{group['hive']}_g{group['group']}_{clip['clip']}"
            members=[];fixed='t50' if group['hive'] in ('35','33') else 't40'
            for slot,uid in enumerate(group['units']):
                source_unit=data['units'][uid];member=data['members'][source_unit['member']];arm=source_unit['arms'][fixed]
                with np.load(source/arm['rows_file'],allow_pickle=False) as z:raw=z['rows']
                bundle=build_candidates(raw,clip['polygon'],clip['nframes'])
                if not np.array_equal(bundle['baseline_windows'],arm['windows']):raise ValueError('baseline F clock differs195')
                old=[(e['track_id'],e['frame'],e['direction']) for e in bundle['baseline_events']]
                expected=[(e['pid'],e['frame'],e['direction']) for e in arm['event_records']]
                if old!=expected:raise ValueError('baseline F official events differ195')
                for c in bundle['candidates']:
                    pid=c['parent_id'];key=f'{unit_name}|{slot}|{pid}'
                    c.update(local_candidate_id=c['candidate_id'],candidate_id=key,key=key,unit=unit_name,domain=group['hive'],hive=group['hive'],group=group['group'],clip=clip['clip'],video=clip['clip'],member_index=slot,member=str(member['member']),tag=member['tag'],parent_canonical_id=pid)
                    all_candidates.append(c)
                    total[group['hive']]+=1
                file=f"rows/u{len(groups):02d}_m{slot}.npz";np.savez(dest/file,rows=raw)
                # Verify native shape/bytes, no serialization rounding.
                with np.load(dest/file,allow_pickle=False) as z:assert z['rows'].tobytes()==raw.tobytes()
                meta={k:v for k,v in bundle.items() if k not in ('raw_rows','canonical_rows','source_raw_index','polygon','nframes')}
                members.append(dict(member=str(member['member']),member_index=slot,tag=member['tag'],seed=member['seed'],fixed_arm=fixed,rows_file=file,rows_sha256=F.sha(dest/file),source_receipt=dict(source195_unit=uid,source195_rows_file=arm['rows_file'],source195_rows_sha256=arm['rows_sha256'],upstream=arm['source_receipt']),bundle=meta))
            groups.append(dict(unit=unit_name,domain=group['hive'],hive=group['hive'],video=clip['clip'],clip=clip['clip'],group=group['group'],nf=clip['nframes'],nframes=clip['nframes'],polygon=clip['polygon'],split=clip['split'],members=members))
        source_hashes=dict(freezer=F.sha(__file__),candidates=F.sha(Path(__file__).with_name('candidates.py')),study193=F.sha(F.__file__),evaluate=F.sha(args.repo/'beetrack/evaluate.py'))
        save(dest/'deployment.json',dict(schema='endpoint-cut-freeze-v1',prereg=args.prereg,marker=MARKER,source195_seal=SOURCE_SEAL,source195_deployment_sha256=source_manifest['deployment.json'],source_hashes=source_hashes,groups=groups,candidate_counts=dict(total),authorized_arms=60,
              clock=dict(fps=25,frame_origin=1,window_frames=750,offset_seconds=list(range(30))),candidate_key='unit|member_index(local0..2)|parent_canonical_id'))
        manifest={p.relative_to(dest).as_posix():F.sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
        save(dest/'manifest.json',manifest);(dest/'SEALED.sha256').write_text(F.sha(dest/'manifest.json')+'\n')
        dest.rename(args.out)
    print('FROZEN',args.out,'SEAL',F.sha(args.out/'manifest.json'),'CANDIDATES',dict(total))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',required=True,type=Path);ap.add_argument('--source195',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--prereg')
    args=ap.parse_args()
    if not args.source195 or not args.out or not args.prereg:ap.error('--source195 --out --prereg required')
    args.repo=args.repo.resolve();args.source195=args.source195.resolve();args.out=args.out.resolve()
    freeze(args)
