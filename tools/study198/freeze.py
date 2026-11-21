"""GT-free peer evidence seal for the exact frozen197 candidate/action family."""
import os
import sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import Counter
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
from peers import prepare_peer,peer_evidence

MARKER='[endpoint-peer-v1]'
SOURCE197_SEAL='e7fd48078f8d0a32bb9018f2bac93dc4b5270d374d5bcd0a631fa97dd4d790e4'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x


def save(path,data):Path(path).write_text(json.dumps(plain(data),separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')


def verify(root):
    root=Path(root)
    if (root/'SEALED.sha256').read_text().strip()!=sha(root/'manifest.json'):raise ValueError('peer seal mismatch')
    for name,digest in json.loads((root/'manifest.json').read_text()).items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=digest:raise ValueError('peer artifact changed')
    return dict(manifest_sha256=sha(root/'manifest.json'),deployment_sha256=sha(root/'deployment.json'))


def load_groups(root):
    root=Path(root);receipt=verify(root)
    data=json.loads((root/'deployment.json').read_text())
    if data['schema']!='endpoint-peer-freeze-v1' or data['source197_seal']!=SOURCE197_SEAL:raise ValueError('wrong peer source schema')
    return [dict(g,freeze_receipt=receipt) for g in data['groups']]


def load_records(root):return [r for g in load_groups(root) for m in g['members'] for r in m['records']]


def run(args):
    cv2.setNumThreads(1)
    if not re.fullmatch('[0-9a-f]{40}',args.prereg):raise ValueError('full prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,capture_output=True,check=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,capture_output=True,check=True).stdout.decode('utf-8')
    if log.count(MARKER)!=1:raise ValueError('unique peer prereg marker required')
    if args.out.exists():raise FileExistsError(args.out)
    if sha(args.source197/'manifest.json')!=SOURCE197_SEAL:raise ValueError('original197 source seal changed')
    source_code=args.repo/'tools/study197/freeze.py'
    sys.path.insert(0,str(source_code.parent))
    spec=importlib.util.spec_from_file_location('endpoint197_source_freezer',source_code)
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    source_receipt=source.verify(args.source197)
    source_groups=source.load_groups(args.source197)
    if len(source_groups)!=20 or sum(len(g['members']) for g in source_groups)!=60:raise ValueError('fixed source group count changed')
    expected=[c['candidate_id'] for g in source_groups for m in g['members'] for c in m['candidates']]
    if len(expected)!=1196 or len(set(expected))!=1196:raise ValueError('fixed1196 candidate identity changed')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.peer-freeze-',dir=args.out.parent) as tmp:
        target=Path(tmp)/'payload';target.mkdir();groups=[];coverage=Counter()
        for group in source_groups:
            dates=set()
            for member in group['members']:
                for c in member['candidates']:
                    dates.update(int(x) for x in member['bundle']['canonical_rows'][c['window_indices'],0])
            peers=[prepare_peer(m['rows'],group['polygon'],dates=dates) for m in group['members']]
            members=[]
            for slot,member in enumerate(group['members']):
                other=[i for i in range(3) if i!=slot];records=[]
                for c in member['candidates']:
                    evidence=peer_evidence(c,member['bundle'],[peers[i] for i in other],group['polygon'])
                    for p,index in zip(evidence['peers'],other):p['member_index']=index;p['member']=group['members'][index]['member']
                    fields=('candidate_id','key','unit','hive','domain','group','clip','video','member','member_index','tag','parent_id','parent_canonical_id','k','end','span','frame','direction','delta','delta_windows','features_G','short_inside','core_inside')
                    record={k:c[k] for k in fields};record.update(evidence)
                    records.append(record)
                    coverage[(group['domain'],'all')]+=1
                    coverage[(group['domain'],'eligible')]+=int(record['eligible'])
                    coverage[(group['domain'],'frame_only')]+=int(record['frame_only_gate'])
                members.append(dict(member=member['member'],member_index=slot,tag=member['tag'],raw_receipt=member['raw_receipt'],records=records,baseline_windows=member['baseline_windows']))
            keep=('unit','domain','hive','group','video','clip','nf','nframes','split','polygon')
            groups.append(dict({k:group[k] for k in keep},members=members))
        actual=[r['candidate_id'] for g in groups for m in g['members'] for r in m['records']]
        if actual!=expected:raise ValueError('candidate order changed')
        save(target/'deployment.json',dict(schema='endpoint-peer-freeze-v1',prereg=args.prereg,marker=MARKER,source197_seal=SOURCE197_SEAL,source197_receipt=source_receipt,groups=groups,
             source_hashes=dict(freezer=sha(__file__),peers=sha(Path(__file__).with_name('peers.py')),source197_freezer=sha(source_code),official_evaluate=sha(args.repo/'beetrack/evaluate.py')),
             feature_dimensions=dict(N11=11,B15=15,I22=22),gate='exists SAMEpeer stablecore identifiedANDownshortobservable',
             frame_only_diagnostic='exists SAMEpeer >=2/3core dates independentbestIoU>=.50 AND >=1shortbestIoU>=.50; never formaleligible',
             coverage={h:{name:coverage[(h,name)] for name in ('all','eligible','frame_only')} for h in ('35','33','W','G')}))
        save(target/'manifest.json',{p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file()})
        (target/'SEALED.sha256').write_text(sha(target/'manifest.json')+'\n');target.rename(args.out)
    print('PEERS_SEALED',sha(args.out/'manifest.json'))
    print(json.dumps({h:{name:coverage[(h,name)] for name in ('all','eligible','frame_only')} for h in ('35','33','W','G')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',required=True,type=Path);ap.add_argument('--source197',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--prereg',required=True)
    args=ap.parse_args();args.repo=args.repo.resolve();args.source197=args.source197.resolve();args.out=args.out.resolve();run(args)
