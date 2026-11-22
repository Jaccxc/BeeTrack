"""GT-free211 original210-family22-state peer agreement freeze.

No actual GT, metric, image or model accesses. Native receiver and two peer K
arrays come only from signed208/206 inputs; all states use study210 materialize.
Joint209/210 branch is outside this CLI and requires separate registration.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from graphs211 import peer_graph,k_event_features,self_test as graph_self_test

MARKER='[peer-event-capacity-v1]'
SOURCE210='3810c1afabe45160642b6187c8fed89dac4e5e05dabb9f5be2cb696d666a7005'
SOURCE208='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94'
SOURCE206='066aaa245d78ee5d926cafeb709d200746e9ea9f736f820606270c35a2188d5d'
EXPECTED_STATES={('35','BEE24-35-head'):8,('33','BEE24-33-head'):8,('W','BEE24-10'):1,
                 ('W','BEE24-15'):2,('G','BEE24-24'):2,('G','BEE24-28'):1}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def array_sha(a):
    a=np.ascontiguousarray(a,dtype='<f8');h=hashlib.sha256(json.dumps(list(a.shape)).encode()+b'\n');h.update(a.tobytes());return h.hexdigest()


def save(path,x):
    with Path(path).open('w',encoding='utf8',newline='\n') as f:json.dump(x,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')


def save_gz(path,x):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
            with io.TextIOWrapper(gz,encoding='utf8',newline='\n') as f:json.dump(x,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')


def verify(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('seal mismatch')
    m=json.loads((root/'manifest.json').read_text())
    for name,digest in m.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError('frozen file drift '+name)
    return m


def states(pairs):
    """All actual active-family subsets, not the210 disjoint slate shortcut."""
    active=sorted([p for p in pairs if p['active']],key=lambda p:p['canonical_ids'])
    result=[]
    for bits in itertools.product((0,1),repeat=len(active)):
        selected=[p for p,b in zip(active,bits) if b];parents=[i for p in selected for i in p['canonical_ids']]
        if len(set(parents))!=len(parents):continue
        result.append(dict(state_id=''.join(map(str,bits)) or '0',selected_pair_ids=[p['pair_id'] for p in selected],
                           edits=len(selected),is_K=not selected))
    return result


def freeze_clip(catalog,peers,metadata,identity,dest):
    from tools.study210.physical210 import materialize
    dest=Path(dest)
    if dest.exists():
        verify(dest)
        if json.loads((dest/'receipt.json').read_text())['identity']!=identity:raise ValueError('resume inputs/code drift')
        return
    R=np.asarray(catalog['rows'],float);C=np.asarray(catalog['canon'],np.int64);polygon=metadata['polygon'];nf=metadata['nf']
    allstates=states(catalog['pairs'])
    if len(allstates)!=metadata['expected_states']:raise ValueError('actual210 active-family state count changed; no substitutes')
    dest.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        np.savez(temp/'native.npz',receiver=R,receiver_canon=C,peer1=peers[0]['rows'],peer2=peers[1]['rows'])
        save_gz(temp/'sources.json.gz',dict(receiver=catalog['source_keys'],peers=[p['sources'] for p in peers]))
        records=[];kfeatures=None
        for index,s in enumerate(allstates):
            output,proof=materialize(R,C,catalog['pairs'],s['selected_pair_ids'])
            rawindex=np.asarray(proof['output_raw_indices'],np.int64)
            if not np.array_equal(np.sort(rawindex),np.arange(len(R))) or not np.array_equal(output[:,[0,2,3,4,5,6]],R[rawindex][:,[0,2,3,4,5,6]]):
                raise ValueError('materialization is not an exact original-row bijection')
            file=f'state_s{index:03d}.npz';np.savez(temp/file,ids=output[:,1].astype(np.int64),output_raw_indices=rawindex)
            graphs=[peer_graph(p['rows'],output,polygon,nf,deletion=s['is_K']) for p in peers]
            if graphs[0]['nPred']!=graphs[1]['nPred'] or graphs[0]['pred_windows']!=graphs[1]['pred_windows']:
                raise AssertionError('state prediction events depend on peer reference')
            family={int(pid):[int(pid)] for pid in np.unique(C)}
            table={p['pair_id']:p for p in catalog['pairs']}
            for key in s['selected_pair_ids']:
                a,b=table[key]['canonical_ids'];family[min(a,b)]=[a,b]
            for graph,p in zip(graphs,peers):
                graph['reference_member']=p['metadata']
                for v in graph['prediction_vertices']:v['constituent_original_canonical_ids']=family[v['canonical_id']]
            if s['is_K']:
                kfeatures=k_event_features(graphs)
                vertices={v['canonical_id']:v for v in graphs[0]['prediction_vertices']}
                for f in kfeatures:
                    v=vertices[f['canonical_id']]
                    f.update(signed_frame=v['signed_frame'],event_key=f'{metadata["clip"]}|{f["canonical_id"]}|{v["signed_frame"]}')
            count_l1=[int(np.abs(np.asarray(g['pred_windows'],np.int64)-np.asarray(g['reference_windows'],np.int64)).sum()) for g in graphs]
            record=dict(s,state_index=index,materialization_file=file,rows_sha256=array_sha(output),
                nP=graphs[0]['nPred'],meanJ=sum(g['J'] for g in graphs)/2,meanM=sum(g['R'] for g in graphs)/2,
                meanCountL1=sum(count_l1)/2,peerCountL1=count_l1,pred_windows=graphs[0]['pred_windows'],
                peer_graphs=graphs,materialization=dict(output_raw_indices_file=file,ids_field='ids',raw_index_field='output_raw_indices',
                    all_original_rows_once=True,geometry_score_dates_exact=True,original_K_exact=s['is_K'],
                    implementation='tools.study210.physical210.materialize'))
            records.append(record)
            print('PEER_STATE',metadata['domain'],metadata['clip'],index+1,'/',len(allstates),flush=True)
        if kfeatures is None:raise AssertionError('missing original K state')
        save_gz(temp/'states.json.gz',dict(schema='match211-clip-v1',metadata=metadata,active_pairs=[p for p in catalog['pairs'] if p['active']],
            states=records,K_event_features=kfeatures,reference_type='other two original prediction members, not actual GT'))
        save(temp/'receipt.json',dict(identity=identity,states=len(records),peer_graphs=2*len(records),GT_access=False,
            original_receiver_K_exact208_and206=True,all_states_original_row_bijection=True,official_peer_match_checked=True))
        save(temp/'manifest.json',{p.name:sha(p) for p in temp.iterdir()});(temp/'SEALED.sha256').write_text(sha(temp/'manifest.json')+'\n')
        temp.rename(dest)
    finally:
        if temp.exists():shutil.rmtree(temp)


def load_clip(folder):
    folder=Path(folder);verify(folder)
    with gzip.open(folder/'states.json.gz','rt',encoding='utf8') as f:data=json.load(f)
    with np.load(folder/'native.npz',allow_pickle=False) as z:native={k:z[k] for k in z.files}
    return dict(data,native=native,folder=folder)


def state_rows(clip,state):
    with np.load(clip['folder']/state['materialization_file'],allow_pickle=False) as z:
        rows=clip['native']['receiver'][z['output_raw_indices']].copy();rows[:,1]=z['ids']
    if array_sha(rows)!=state['rows_sha256']:raise ValueError('state physical reconstruction drift')
    return rows


def run(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('full published prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    text=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if text.count(MARKER)!=1:raise ValueError('unique211marker required '+MARKER)
    if args.out.resolve().drive.upper()!='D:':raise ValueError('D: output required')
    for root,digest in ((args.catalog210,SOURCE210),(args.inputs208,SOURCE208),(args.source206,SOURCE206)):
        if sha(root/'manifest.json')!=digest or (root/'SEALED.sha256').read_text().strip()!=digest:raise ValueError('historical source seal mismatch')
    # Verify only explicit numeric named old inputs, not images/metric files.
    m208=json.loads((args.inputs208/'manifest.json').read_text());m206=json.loads((args.source206/'manifest.json').read_text())
    for root,m in ((args.inputs208,m208),(args.source206,m206)):
        if sha(root/'deployment.json')!=m['deployment.json']:raise ValueError('historical deployment changed')
    d206=json.loads((args.source206/'deployment.json').read_text());byunit={g['unit']:g for g in d206['groups']}
    verify(args.catalog210);d210=json.loads((args.catalog210/'deployment.json').read_text())
    from tools.study210.catalog210 import load_clip as load210
    records=[];total=0
    for c in d210['clips']:
        catalog=load210(args.catalog210/c['path'],args.inputs208);meta=catalog['metadata'];unit=meta['unit']
        source=byunit[unit];folder=args.source206/source['path']
        for name in ('queries.json','K.npz','receipt.json'):
            if sha(folder/name)!=m206[source['path']+'/'+name]:raise ValueError('named206source drift')
        q=json.loads((folder/'queries.json').read_text())
        if q['nf']!=meta['nf'] or q['polygon']!=meta['polygon']:raise ValueError('same-unit clock/polygon mismatch')
        peers=[]
        with np.load(folder/'K.npz',allow_pickle=False) as z:
            receiver=z['k0'];R=np.asarray(catalog['rows'])
            if receiver.dtype!=R.dtype or receiver.shape!=R.shape or not np.array_equal(receiver.view(np.uint64),R.view(np.uint64)):
                raise ValueError('original K differs bit-for-bit between208 and206')
            if q['members'][0]['source_keys']!=catalog['source_keys']:raise ValueError('receiver original detector row lineage differs')
            expected_seed=3 if meta['domain'] in ('35','33') else 0
            if q['members'][0]['seed']!=expected_seed or meta['seed']!=expected_seed:raise ValueError('receiver seed changed')
            for slot in (1,2):
                m=q['members'][slot]
                if m['member_index']!=slot or m['seed']!=expected_seed+slot or not 0<=m['seed']<=14:raise ValueError('same-unit oldpeer whitelist changed')
                peers.append(dict(rows=z[f'k{slot}'],sources=m['source_keys'],metadata={k:m[k] for k in ('member','member_index','seed','tag')}))
        metadata={k:meta[k] for k in ('domain','clip','unit','nf','polygon','split','region','seed','member')}
        metadata.update(expected_states=EXPECTED_STATES[(meta['domain'],meta['clip'])],peer206_group=source)
        identity=dict(prereg=args.prereg,source210=SOURCE210,source208=SOURCE208,source206=SOURCE206,
            source210_clip=c['seal'],peer206_K_sha256=m206[source['path']+'/K.npz'],
            code_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('graphs211.py'),args.repo/'tools/study210/physical210.py')})
        dest=args.out/c['path'];freeze_clip(catalog,peers,metadata,identity,dest)
        count=json.loads((dest/'receipt.json').read_text())['states'];total+=count
        records.append(dict(domain=meta['domain'],clip=meta['clip'],unit=unit,path=c['path'],states=count,
                            seal=(dest/'SEALED.sha256').read_text().strip()))
        print('SEALED211',meta['domain'],meta['clip'],flush=True)
    if len(records)!=6 or total!=22:raise ValueError('full original-family sixclip22state roster required')
    top=dict(schema='match211-freeze-v1',prereg=args.prereg,marker=MARKER,source210=SOURCE210,source208=SOURCE208,source206=SOURCE206,
             clips=records,states=total,GT_access=False)
    if (args.out/'deployment.json').exists() and json.loads((args.out/'deployment.json').read_text())!=top:raise ValueError('top resume drift')
    save(args.out/'deployment.json',top)
    files={p.relative_to(args.out).as_posix():sha(p) for p in args.out.rglob('*') if p.is_file() and p not in (args.out/'manifest.json',args.out/'SEALED.sha256')}
    if (args.out/'manifest.json').exists() and json.loads((args.out/'manifest.json').read_text())!=files:raise ValueError('top manifest drift')
    save(args.out/'manifest.json',files);(args.out/'SEALED.sha256').write_text(sha(args.out/'manifest.json')+'\n')
    print('FROZEN211_STATES22',sha(args.out/'manifest.json'))


def self_test():
    graph_self_test()
    from tools.study210.catalog210 import build
    R=np.array([[24,1,-3,0,2,2,.9],[26,1,1,0,2,2,.9],[25,2,-1,0,2,2,.9]],float)
    poly=[(0,-20),(50,-20),(50,20),(0,20)];catalog=build(R,R[:,1],40,poly)
    catalog.update(rows=R,canon=R[:,1].astype(int),source_keys=[['synthetic',i] for i in range(3)])
    peers=[dict(rows=R.copy(),sources=[['peer'+str(j),i] for i in range(3)],metadata=dict(seed=j)) for j in (1,2)]
    meta=dict(domain='synthetic',clip='synthetic',nf=40,polygon=poly,expected_states=2)
    with tempfile.TemporaryDirectory() as tmp:
        dest=Path(tmp)/'clip';freeze_clip(catalog,peers,meta,dict(synthetic=True),dest);loaded=load_clip(dest)
        assert len(loaded['states'])==2
        for state in loaded['states']:assert len(state_rows(loaded,state))==len(R)
        assert loaded['states'][0]['is_K'] and loaded['states'][0]['selected_pair_ids']==[]
    return dict(passed=True,scope='synthetic original210-family two states, native row reconstruction, peer official graphs and vertex marginals')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--self-test',action='store_true')
    p.add_argument('--catalog210',type=Path);p.add_argument('--inputs208',type=Path);p.add_argument('--source206',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--prereg')
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()));sys.path.insert(0,str(args.repo.resolve()/'tools/study210'))
    if args.self_test:print(json.dumps(self_test()))
    else:
        if any(getattr(args,k) is None for k in ('catalog210','inputs208','source206','out')):p.error('explicit old sources and output required')
        run(args)
