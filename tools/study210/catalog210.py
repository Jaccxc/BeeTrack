"""GT-free210 actual interleaving catalog; source/synthetic until root GO.

No images, GT, scores or models. Two passes retain every temporal bracket of
qualified no-common-date pairs, including distant brackets and singleton B.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from collections import defaultdict
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
import feature210 as F

MARKER='[alternating-offset-v1]'
SOURCE208='1472ff42c53f7630b53e7feb9b91872a14fe8cc19cf98117772955d66b6d7a94'
SPECS={('35','BEE24-35-head',3),('33','BEE24-33-head',3),('W','BEE24-10',0),('W','BEE24-15',0),('G','BEE24-24',0),('G','BEE24-28',0)}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x


def save(path,x):
    with Path(path).open('w',encoding='utf8',newline='\n') as f:json.dump(plain(x),f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')


def save_gz(path,x):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
            with io.TextIOWrapper(gz,encoding='utf8',newline='\n') as text:
                json.dump(plain(x),text,sort_keys=True,separators=(',',':'),allow_nan=False);text.write('\n')


def verify(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'SEALED.sha256').read_text().strip():raise ValueError('seal mismatch')
    m=json.loads((root/'manifest.json').read_text())
    for name,digest in m.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError('frozen artifact drift '+name)
    return m


def prepare(rows,canon,nf):
    from tools.study197.physical import canonicalize
    R=np.asarray(rows,float);C=np.asarray(canon)
    if R.ndim!=2 or R.shape[1]!=7 or C.shape!=(len(R),):raise ValueError('rows[M,7],canon[M] required')
    if not np.isfinite(C).all() or (C!=np.floor(C)).any() or (R[:,0]>nf).any():raise ValueError('invalid canonical clock')
    b=canonicalize(R);mapped=np.empty(len(R),np.int64);mapped[b['source_raw_index']]=b['canonical_rows'][:,1].astype(np.int64)
    if not np.array_equal(mapped,C):raise ValueError('canon differs from original K official mapping')
    tracks={}
    for block in b['parent_blocks']:tracks[block['parent_id']]=np.asarray(b['source_raw_index'][block['start']:block['end']],np.int64)
    order=np.argsort(R[:,0],kind='stable');dates,starts=np.unique(R[order,0],return_index=True)
    byframe={int(f):order[a:z] for f,a,z in zip(dates,starts,[*starts[1:],len(R)])}
    return R,C.astype(np.int64),tracks,byframe


def bracket_batches(R,C,tracks,byframe):
    centers=R[:,2:4]+R[:,4:6]/2;diagonal=np.hypot(R[:,4],R[:,5])
    for pid,ix in tracks.items():
        for left,right in zip(ix[:-1],ix[1:]):
            tl,tr=int(R[left,0]),int(R[right,0])
            if not 2<=tr-tl<=6:continue
            for date in range(max(tl+1,tr-3),min(tl+3,tr-1)+1):
                inner=byframe.get(date,np.empty(0,np.int64));inner=inner[C[inner]!=pid]
                if not len(inner):continue
                alpha=(date-tl)/(tr-tl);prediction=(1-alpha)*centers[left]+alpha*centers[right]
                offset=centers[inner]-prediction
                scale=(diagonal[left]+diagonal[inner]+diagonal[right])/3
                norm=np.linalg.norm(offset,axis=1)/scale
                yield int(pid),int(left),int(right),date,inner,offset,scale,norm


def merge_rows(rows,canon,pairs,selected_ids):
    """No deletion/interpolation/reordering: only selected parent IDs coalesce."""
    from beetrack.evaluate import by_track
    R=np.asarray(rows,float);C=np.asarray(canon,np.int64);table={p['pair_id']:p for p in pairs}
    if len(set(selected_ids))!=len(selected_ids):raise ValueError('duplicate pair selection')
    out=R.copy();used=set()
    if selected_ids:out[:,1]=C
    for key in sorted(selected_ids,key=lambda k:table[k]['canonical_ids']):
        a,b=table[key]['canonical_ids']
        if a in used or b in used:raise ValueError('selected pairs share a canonical run')
        used.update((a,b));ia=np.flatnonzero(C==a);ib=np.flatnonzero(C==b)
        if np.intersect1d(R[ia,0],R[ib,0]).size:raise ValueError('pair has common actual dates')
        union=np.r_[ia,ib];out[union,1]=min(a,b)
        dates=np.sort(out[union,0])
        if (np.diff(dates)>25).any():raise ValueError('merged pair would split again under official gap')
    output_index=np.lexsort((np.arange(len(R)),out[:,0])) if selected_ids else np.arange(len(R))
    out=out[output_index]
    if not np.array_equal(out[:,[0,2,3,4,5,6]],R[output_index][:,[0,2,3,4,5,6]]):raise AssertionError('native observations changed')
    if len(out)!=len(R) or len(set(map(tuple,out[:,:2])))!=len(out):raise ValueError('merged date/ID collision')
    return out,dict(selected_ids=sorted(selected_ids),rows_preserved=len(R),output_raw_indices=output_index.tolist(),
                    order='K exact original order; intervention stable(frame,original raw index)',
                    geometry_score_dates_exact=True,allocator='min selected canonical ID',fresh_official_matching_required=True)


def build(rows,canon,nf,polygon):
    from tools.study197.candidates import event_times,window_counts
    R,C,tracks,byframe=prepare(rows,canon,nf);qualified=set();date_sets={p:set(map(int,R[ix,0])) for p,ix in tracks.items()}
    for outer,left,right,date,inner,offset,scale,norm in bracket_batches(R,C,tracks,byframe):
        for j in inner[norm<=1]:qualified.add(tuple(sorted((outer,int(C[j])))))
    qualified={p for p in qualified if date_sets[p[0]].isdisjoint(date_sets[p[1]])}
    brackets={p:[] for p in qualified}
    for outer,left,right,date,inner,offset,scale,norm in bracket_batches(R,C,tracks,byframe):
        for z,middle in enumerate(inner):
            key=tuple(sorted((outer,int(C[middle]))))
            if key not in qualified:continue
            brackets[key].append(dict(outer_parent=outer,inner_parent=int(C[middle]),raw_indices=[left,int(middle),right],
                frames=[int(R[left,0]),date,int(R[right,0])],left_dt=date-int(R[left,0]),right_dt=int(R[right,0])-date,
                offset_xy=(offset[z] if outer==key[0] else -offset[z]).tolist(),offset_orientation='B-minus-A',
                raw_middle_minus_outer_interpolation=offset[z].tolist(),
                local_mean_diagonal=float(scale[z]),normalized_offset=float(norm[z]),
                qualifying_local_offset=bool(norm[z]<=1)))
    pairs=[];resource=defaultdict(list)
    for a,b in sorted(qualified):
        key=f'{a}|{b}';br=sorted(brackets[(a,b)],key=lambda x:(x['frames'][1],x['outer_parent'],x['raw_indices']))
        ia,ib=tracks[a],tracks[b];indices=np.sort(np.r_[ia,ib]);original=R[indices]
        before=event_times(original,polygon);merged=original.copy();merged[:,1]=a
        after=event_times(merged,polygon)
        delta=window_counts(after,nf)-window_counts(before,nf)
        numeric=F.pair_features(R[ia],R[ib])
        if not numeric['valid']:raise ValueError('qualified bracket pair rejected by pure feature API: '+key)
        pair=dict(pair_id=key,canonical_ids=[a,b],eligible=True,no_common_dates=True,
            track_raw_indices=[ia.tolist(),ib.tolist()],brackets=br,min_offset_norm=min(q['normalized_offset'] for q in br),
            events_before=before,events_after=after,delta_windows=delta.tolist(),active=bool(np.any(delta)),
            merge_retains_all_rows=True,features=numeric,G6=numeric['G'],X2=numeric['X'],Xminus2=numeric['Xminus'],
            full_support=numeric['full_support'])
        pairs.append(pair);resource[a].append(key);resource[b].append(key)
    used=set();slate=[]
    for p in sorted(pairs,key=lambda p:(not p['active'],p['min_offset_norm'],p['canonical_ids'])):
        if used.intersection(p['canonical_ids']):p['slate_order']=None;continue
        p['slate_order']=len(slate);slate.append(p['pair_id']);used.update(p['canonical_ids'])
    table={p['pair_id']:p for p in pairs}
    active_slate=[p for p in slate if table[p]['active']]
    return dict(schema='interleave210-catalog-v1',nframes=int(nf),raw_rows=len(R),canonical_runs=len(tracks),pairs=pairs,
        slate_ids=slate,active_slate_ids=active_slate,
        conflicts=[dict(canonical_id=int(pid),pair_ids=ids,capacity=1) for pid,ids in sorted(resource.items()) if len(ids)>1],
        denominator=dict(eligible_pairs=len(pairs),active_pairs=sum(p['active'] for p in pairs),
                         slate_pairs=len(slate),active_slate_pairs=len(active_slate)),
        rules=dict(no_common_actual_date=True,outer_adjacent_observations=True,left_right_raw_dt=[1,3],
                   candidate_gate='any offset / arithmetic mean of THREE diagonals <=1',
                   pair_features_scale='separate common median of unique support diagonals; no candidate offset filter',
                   all_both_direction_temporal_brackets_retained=True,slate='active first, min local offset, numeric pair IDs; greedy disjoint',
                   feature_validity_never_changes_common_slate=True))


def load_clip(folder,inputs208):
    folder=Path(folder);verify(folder)
    with gzip.open(folder/'catalog.json.gz','rt',encoding='utf8') as f:catalog=json.load(f)
    m=catalog['metadata'];source=Path(inputs208)/m['source208_path']
    for name,key in (('rows.npy','rows_sha256'),('canon.npy','canon_sha256'),('source_keys.json','source_keys_sha256')):
        if sha(source/name)!=m[key]:raise ValueError('208native source drift '+name)
    return dict(catalog,rows=np.load(source/'rows.npy',mmap_mode='r'),canon=np.load(source/'canon.npy',mmap_mode='r'),
                source_keys=json.loads((source/'source_keys.json').read_text()))


def freeze_clip(rows,canon,nf,polygon,metadata,identity,dest):
    dest=Path(dest)
    if dest.exists():
        verify(dest)
        if json.loads((dest/'receipt.json').read_text())['identity']!=identity:raise ValueError('resume code/source drift')
        return
    catalog=build(rows,canon,nf,polygon)
    dest.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        save_gz(temp/'catalog.json.gz',dict(catalog,metadata=metadata))
        save(temp/'receipt.json',dict(identity=identity,denominator=catalog['denominator'],GT_access=False,image_access=False))
        save(temp/'manifest.json',{p.name:sha(p) for p in temp.iterdir()});(temp/'SEALED.sha256').write_text(sha(temp/'manifest.json')+'\n')
        temp.rename(dest)
    finally:
        if temp.exists():shutil.rmtree(temp)


def run(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('published full prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('unique210marker required '+MARKER)
    if args.out.resolve().drive.upper()!='D:':raise ValueError('D: output required')
    if sha(args.inputs/'manifest.json')!=SOURCE208 or (args.inputs/'SEALED.sha256').read_text().strip()!=SOURCE208:raise ValueError('208input source mismatch')
    manifest=json.loads((args.inputs/'manifest.json').read_text())
    if sha(args.inputs/'deployment.json')!=manifest['deployment.json']:raise ValueError('208deployment drift')
    top=json.loads((args.inputs/'deployment.json').read_text())
    if {(c['domain'],c['clip'],c['seed']) for c in top['clips']}!=SPECS or len(top['clips'])!=6:raise ValueError('fixedsix roster mismatch')
    clips=[]
    for c in top['clips']:
        source=args.inputs/c['path']
        for name in ('rows.npy','canon.npy','metadata.json','source_keys.json'):
            if sha(source/name)!=manifest[c['path']+'/'+name]:raise ValueError('named208source changed')
        m=json.loads((source/'metadata.json').read_text());m.update(source208_path=c['path'],
            rows_sha256=manifest[c['path']+'/rows.npy'],canon_sha256=manifest[c['path']+'/canon.npy'],
            source_keys_sha256=manifest[c['path']+'/source_keys.json'])
        identity=dict(prereg=args.prereg,source208=SOURCE208,source208_clip_seal=c['seal'],
                      code_sha256={p.name:sha(p) for p in (Path(__file__),Path(F.__file__))})
        dest=args.out/c['path'];freeze_clip(np.load(source/'rows.npy',mmap_mode='r'),np.load(source/'canon.npy',mmap_mode='r'),m['nf'],m['polygon'],m,identity,dest)
        clips.append(dict(domain=c['domain'],clip=c['clip'],seed=c['seed'],path=c['path'],seal=(dest/'SEALED.sha256').read_text().strip()))
        print('SEALED210',c['domain'],c['clip'],flush=True)
    deployment=dict(schema='interleave210-freeze-v1',prereg=args.prereg,marker=MARKER,source208=SOURCE208,clips=clips)
    if (args.out/'deployment.json').exists() and json.loads((args.out/'deployment.json').read_text())!=deployment:raise ValueError('top resume mismatch')
    save(args.out/'deployment.json',deployment)
    files={p.relative_to(args.out).as_posix():sha(p) for p in args.out.rglob('*') if p.is_file() and p not in (args.out/'manifest.json',args.out/'SEALED.sha256')}
    if (args.out/'manifest.json').exists() and json.loads((args.out/'manifest.json').read_text())!=files:raise ValueError('top files drift')
    save(args.out/'manifest.json',files);(args.out/'SEALED.sha256').write_text(sha(args.out/'manifest.json')+'\n')
    print('FROZEN210',sha(args.out/'manifest.json'))


def self_test():
    def r(f,p,x,y=0):return [f,p,x,y,2,2,.9]
    poly=[(0,-100),(100,-100),(100,100),(0,100)]
    # Nested singleton ABA is legal; two different observers need not each have2.
    R=np.array([r(1,1,-3),r(3,1,1),r(2,2,-1)],float)
    c=build(R,R[:,1],10,poly);assert len(c['pairs'])==1
    p=c['pairs'][0];assert len(p['brackets'])==1 and not p['active'] and not p['full_support']
    merged,proof=merge_rows(R,R[:,1],c['pairs'],['1|2'])
    assert len(merged)==len(R) and proof['geometry_score_dates_exact']
    from tools.study197.candidates import event_times,window_counts
    assert np.array_equal(window_counts(event_times(merged,poly),10)-window_counts(event_times(R,poly),10),p['delta_windows'])
    shifted=R.copy();shifted[:,0]+=(23)
    shifted_catalog=build(shifted,shifted[:,1],40,poly)
    assert shifted_catalog['pairs'][0]['active']  # entrydate26->25 changes an edge window
    # Qualified near bracket does not discard a later geometrically far one.
    R=np.array([r(1,1,0),r(3,1,0),r(5,1,0),r(7,1,0),r(2,2,0),r(4,2,30),r(6,2,0)],float)
    c=build(R,R[:,1],10,poly);p=c['pairs'][0]
    assert any(b['normalized_offset']>1 for b in p['brackets']) and {b['outer_parent'] for b in p['brackets']}=={1,2}
    # Any actual shared date excludes the pair, even away from the near bracket.
    R=np.vstack((R,r(7,2,0)));assert not build(R,R[:,1],10,poly)['pairs']
    # Gap4 on either side is forbidden, regardless of perfect geometry.
    R=np.array([r(1,1,0),r(6,1,0),r(2,2,0)],float);assert not build(R,R[:,1],10,poly)['pairs']
    return dict(passed=True,scope='synthetic only',tests=['nested singleton','both orientations incl far brackets',
        'no common actual date','rawdt1..3','full no-delete merge','official countdelta','fixed geometry slate'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--self-test',action='store_true')
    p.add_argument('--inputs',type=Path);p.add_argument('--out',type=Path);p.add_argument('--prereg')
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    if args.self_test:print(json.dumps(self_test()))
    else:
        if args.inputs is None or args.out is None:p.error('--inputs --out --prereg required')
        run(args)
