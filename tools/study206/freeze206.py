"""GT-free206 adapter: immutable203 K/sources -> all native-event queries.

SOURCE ONLY until explicit root GO and published preregistration. No GT metric,
natural image, detector, model, score, reserve seed or GT loader is accessed.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from queries import generate

SOURCE203='439a8746c19c3857fe7192990d63f08cb17c94145c8c18c6394218cc95e01b51'
SOURCE202='7261c2573773a23342286a39ce712950cb986aa6f34248cd12f5705ab420f469'
MARKER='[return-query-reach-v1]'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,x):Path(path).write_text(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')


def manifest(root):
    root=Path(root);digest=sha(root/'manifest.json')
    if (root/'SEALED.sha256').read_text().strip()!=digest:raise ValueError('seal mismatch')
    return json.loads((root/'manifest.json').read_text())


def verify(root):
    root=Path(root);data=manifest(root)
    for name,digest in data.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError('changed frozen file '+name)
    return data


def load_group(folder):
    folder=Path(folder);verify(folder);group=json.loads((folder/'queries.json').read_text())
    with np.load(folder/'K.npz',allow_pickle=False) as z:
        members=[dict(m,rows=z[f'k{m["member_index"]}'],sources=m['source_keys']) for m in group['members']]
    return dict(group,members=members)


def load_groups(root):
    """Explicit consumer I/O; caller must obey its own GT/CPU gate."""
    root=Path(root);manifest(root);top=json.loads((root/'deployment.json').read_text())
    if sha(root/'deployment.json')!=manifest(root)['deployment.json']:raise ValueError('deployment hash mismatch')
    if top['schema']!='return-query206-freeze-v1' or len(top['groups'])!=20:raise ValueError('wrong206 roster')
    return [load_group(root/g['path']) for g in top['groups']]


def freeze_group(group,members,dest,identity):
    """Whole group fail-closed. Input metadata/arrays must already be verified."""
    from tools.study197.candidates import window_counts
    dest=Path(dest)
    if dest.exists():
        verify(dest)
        if json.loads((dest/'receipt.json').read_text())['identity']!=identity:raise ValueError('resume source/code drift')
        return
    frozen=[];counts=Counter()
    for slot,m in enumerate(members):
        if m['member_index']!=slot:raise ValueError('historical slot order changed')
        roster=generate(m['rows'],m['sources'],group['polygon'],group['nf'],group['unit'],slot)
        original=[dict(track_id=q['canonical_id'],frame=q['event']['frame'],direction=q['event']['direction']) for q in roster['queries']]
        if original!=m['historical_events']:raise ValueError('native query events differ203 GT-free event table')
        if not np.array_equal(window_counts(original,group['nf']),np.asarray(m['historical_windows'],int)):
            raise ValueError('query window counts differ203 GT-free windows')
        counts.update(native_events=roster['total_native_events'],eligible=roster['eligible'],ineligible=roster['ineligible'])
        frozen.append(dict(member_index=slot,member=m['member'],seed=m['seed'],tag=m['tag'],
                           source_keys=[list(s) for s in m['sources']],source203_receipt=m['source203_receipt'],roster=roster))
    dest.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        np.savez(temp/'K.npz',**{f'k{i}':m['rows'] for i,m in enumerate(members)})
        save(temp/'queries.json',dict(schema='return-query206-group-v1',**group,members=frozen,counts=dict(counts)))
        save(temp/'receipt.json',dict(identity=identity,members=3,counts=dict(counts),
            exact203_K=True,exact203_sources=True,exact203_events_and_windows=True,
            native_ID_policy='Keep202baseline IDs used by203; never renumber or bridge independent gap runs',GT_access=False))
        save(temp/'manifest.json',{p.name:sha(p) for p in temp.iterdir()});(temp/'SEALED.sha256').write_text(sha(temp/'manifest.json')+'\n')
        temp.rename(dest)
    finally:
        if temp.exists():shutil.rmtree(temp)


def run(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('full published prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('unique206marker required '+MARKER)
    if args.out.resolve().drive.upper()!='D:':raise ValueError('all field artifacts on D:')
    if sha(args.source203/'manifest.json')!=SOURCE203:raise ValueError('wrong203source')
    source_manifest=manifest(args.source203)
    if sha(args.source203/'deployment.json')!=source_manifest['deployment.json']:raise ValueError('203deployment drift')
    source=json.loads((args.source203/'deployment.json').read_text())
    if source['source202']!=SOURCE202 or len(source['groups'])!=20:raise ValueError('source202/20unit mismatch')
    from tools.study193.freeze193 import members as historical_members,CLIPS
    allowmembers=historical_members();allowed=set()
    for m in allowmembers:
        for domain,split,clip,nf in CLIPS:
            if domain==m['hive']:
                allowed.add((f'{domain}_g{m["group"]}_{clip}',m['seed']%3,str(m['member']),m['seed'],m['tag'],split,clip,nf))
    actual=set();groups=[];seenunits=set()
    code={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('queries.py'))}
    for g in source['groups']:
        if g['unit'] in seenunits:raise ValueError('duplicate203unit')
        seenunits.add(g['unit']);folder=args.source203/g['path']
        # Open only signed GT-free K/catalog/receipt. Never open metric files.
        for name in ('manifest.json','SEALED.sha256','K.npz','catalog.json','receipt.json'):
            rel=f'{g["path"]}/{name}'
            if sha(args.source203/rel)!=source_manifest[rel]:raise ValueError('203named input drift '+rel)
        child=manifest(folder)
        if sha(folder/'manifest.json')!=g['seal']:raise ValueError('203membergroup seal drift')
        catalog=json.loads((folder/'catalog.json').read_text());receipt=json.loads((folder/'receipt.json').read_text())
        if receipt['identity']['source202']!=SOURCE202 or not receipt['exact202_K'] or not receipt['exact202_sourcekeys']:
            raise ValueError('203Ksource checks missing')
        meta=catalog['members']
        if len(meta)!=3 or [m['member_index'] for m in meta]!=[0,1,2]:raise ValueError('three historical slots required')
        group={k:catalog[k] for k in ('unit','domain','group','video','nf','polygon')}
        group.update(clip=meta[0]['source_record']['clip'],split=meta[0]['source_record']['split'])
        mm=[]
        with np.load(folder/'K.npz',allow_pickle=False) as z:
            for slot,m in enumerate(meta):
                r=m['source_record'];key=(group['unit'],slot,str(r['member']),r['seed'],r['tag'],r['split'],r['clip'],r['nf'])
                if key not in allowed or key in actual or not 0<=r['seed']<=14 or r['split']=='test':raise ValueError('historical roster/seed mismatch')
                actual.add(key)
                if r['unit']!=group['unit'] or r['member_index']!=slot or r['polygon']!=group['polygon']:raise ValueError('group metadata changed')
                mm.append(dict(member_index=slot,member=str(r['member']),seed=r['seed'],tag=r['tag'],rows=z[f'k{slot}'],
                    sources=m['sources'],historical_events=m['events'],historical_windows=m['windows'],
                    source203_receipt=dict(group=g,source202_member=r,files={name:child[name] for name in ('K.npz','catalog.json','receipt.json')})))
        identity=dict(prereg=args.prereg,source203=SOURCE203,source203_group_seal=g['seal'],code=code)
        token=g['path'];freeze_group(group,mm,args.out/token,identity)
        groups.append(dict(group,path=token,seal=(args.out/token/'SEALED.sha256').read_text().strip()))
        print('SEALED',token,group['unit'],flush=True)
    if actual!=allowed or len(actual)!=60:raise ValueError('full60 historical roster coverage missing')
    top=dict(schema='return-query206-freeze-v1',prereg=args.prereg,marker=MARKER,source203=SOURCE203,
             source202=SOURCE202,groups=groups,member_clips=60,GT_access=False)
    if (args.out/'deployment.json').exists():
        if json.loads((args.out/'deployment.json').read_text())!=top:raise ValueError('top resume drift')
    else:save(args.out/'deployment.json',top)
    files={p.relative_to(args.out).as_posix():sha(p) for p in args.out.rglob('*') if p.is_file() and p not in (args.out/'manifest.json',args.out/'SEALED.sha256')}
    if (args.out/'manifest.json').exists() and json.loads((args.out/'manifest.json').read_text())!=files:raise ValueError('top manifest drift')
    save(args.out/'manifest.json',files);(args.out/'SEALED.sha256').write_text(sha(args.out/'manifest.json')+'\n')
    print('FROZEN20/60',sha(args.out/'manifest.json'))


def self_test():
    from tools.study197.candidates import event_times,window_counts
    R=np.array([[f,1,x,0,2,2,.9] for f,x in [(1,-4),(2,-3),(3,1)]],float)
    poly=[(0,-10),(20,-10),(20,10),(0,10)]
    group=dict(unit='synthetic',domain='W',group=1,video='synthetic',clip='synthetic',split='synthetic',nf=30,polygon=poly)
    ms=[]
    for slot in range(3):
        ev=event_times(R,poly);ms.append(dict(member_index=slot,member=str(slot),seed=slot,tag='synthetic',rows=R,
            sources=[('synthetic'+str(slot),i) for i in range(3)],historical_events=ev,
            historical_windows=window_counts(ev,30).tolist(),source203_receipt=dict(synthetic=True)))
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'g00';freeze_group(group,ms,p,dict(synthetic=True));verify(p)
        loaded=load_group(p);assert len(loaded['members'])==3 and loaded['counts']['native_events']==3
        freeze_group(group,ms,p,dict(synthetic=True))
        assert all(np.array_equal(m['rows'],R) for m in loaded['members'])
        try:freeze_group(group,ms,p,dict(synthetic=False))
        except ValueError:pass
        else:raise AssertionError('input drift accepted')
        bad=[dict(m) for m in ms];bad[0]['historical_events']=[]
        try:freeze_group(group,bad,Path(tmp)/'bad',dict(synthetic=True))
        except ValueError:assert not (Path(tmp)/'bad').exists()
        else:raise AssertionError('event mismatch silently skipped')
    return 'PASS synthetic: three slots/K/source roundtrip; complete event+window exact; all-query roster; atomic resume/drift refusal. NO FIELD.'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--self-test',action='store_true')
    p.add_argument('--source203',type=Path);p.add_argument('--out',type=Path);p.add_argument('--prereg')
    a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
    if a.self_test:print(self_test())
    else:
        if a.source203 is None or a.out is None:p.error('--source203 --out --prereg required')
        run(a)
