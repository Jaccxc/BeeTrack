"""Sealed202 -> compact eight-head capacity. No raw tracks/images are opened.

Run only after the authorized fit owner reports completion. --wait only watches
the three top-level seal names; all inventories/hashes are verified before data.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse, hashlib, json, time
from pathlib import Path
from types import SimpleNamespace
import cap_orphans as cap

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def sealed(root):
    root=Path(root).resolve();digest=sha(root/'manifest.json')
    if (root/'SEALED.sha256').read_text().strip()!=digest:raise ValueError('seal mismatch '+str(root))
    files=load(root/'manifest.json')
    for name,value in files.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha(path)!=value:raise ValueError('manifest mismatch '+name)
    return digest

def prepare(root,prereg,old199,old197):
    """All three top seals must exist before even hash-verifying any payload."""
    roots={k:Path(root)/k for k in ('frozen','owners','predictions')}
    if not all((p/'SEALED.sha256').is_file() for p in roots.values()):raise FileNotFoundError('three complete top seals required')
    times=[(roots[k]/'SEALED.sha256').stat().st_mtime_ns for k in ('frozen','owners','predictions')]
    if times!=sorted(times):raise ValueError('top seals out of authorized order')
    seals={k:sealed(p) for k,p in roots.items()}
    frozen=load(roots['frozen']/'deployment.json');receipt=load(roots['predictions']/'receipt.json')
    if frozen['prereg']!=prereg or receipt['prereg']!=prereg:raise ValueError('prereg differs')
    if receipt['freeze_receipt']['manifest_sha256']!=seals['frozen'] or receipt['owners_manifest_sha256']!=seals['owners']:
        raise ValueError('prediction inputs differ')
    if set(receipt['arms'])!={'K','G','L','M','J'} or set(receipt['folds'])!={'35','33','W','G'}:raise ValueError('fourfold arms incomplete')
    refs=frozen['members']
    if len(refs)!=60 or len({m['clip'] for m in refs})!=6:raise ValueError('old six clip sixty streams only')
    if any(not 0<=m['seed']<=14 or m['split']=='test' for m in refs):raise ValueError('old seed <=14 / no test required')
    identity={(m['unit'],m['member_index']):str(m['member']) for m in refs}
    if len(identity)!=60:raise ValueError('duplicate member slot')
    predicted=set()
    for h in ('35','33','W','G'):
        model=load(roots['predictions']/(h+'.model.json'));fold=load(roots['predictions']/(h+'.masks.json'))
        if model['heldout']!=h or fold['heldout']!=h or set(model['source_domains'])!={'35','33','W','G'}-{h}:raise ValueError('fold differs')
        for m in fold['member_clips']:
            if set(m['selected'])!={'K','G','L','M','J'}:raise ValueError('mask arms incomplete')
            key=(m['unit'],str(m['member']))
            if key in predicted:raise ValueError('duplicate prediction')
            predicted.add(key)
    if predicted!={(u,m) for (u,s),m in identity.items()}:raise ValueError('prediction roster differs')
    units=[]
    for h in ('35','33'):
        shard=load(roots['owners']/(h+'.json'))
        if shard['prereg']!=prereg or shard['frozen_manifest_sha256']!=seals['frozen']:raise ValueError('owners frozen differs')
        for u in shard['units']:
            for m in u['members']:
                if str(m['member'])!=identity[(u['unit'],m['member_index'])]:raise ValueError('slot is not detector seed')
            units.append(u)
    normalized=cap.normalize_catalogue(dict(units=units))
    # Old outputs are numeric only. Map physical IDs, never seed-based positions.
    oldseals={'199':sealed(old199)};prior=load(Path(old199)/'capacity199.json')
    old197=Path(old197);old_manifest=old197.parent/'manifest.json'
    if sha(old_manifest)!=(old197.parent/'SEALED.sha256').read_text().strip():raise ValueError('old197 score seal mismatch')
    if load(old_manifest)[old197.name]!=sha(old197):raise ValueError('old197 summary changed')
    summary=load(old197);oldseals['197_summary']=sha(old197)
    warm={};comparisons={}
    for u in normalized:
        key=u['unit'];cuts={a['candidate_id']:a['action_id'] for a in u['actions'] if a['kind']=='cut'}
        warm[key]=[];comparisons[key]={}
        for family in ('O_D','O_DU'):
            answer=prior['units'][key]['families'][family]['oracle'];ids=answer['selected_ids']
            if not set(ids)<=cuts.keys():raise ValueError('old199 candidate outside frozen cuts')
            warm[key].append([cuts[c] for c in ids])
            comparisons[key][family]=dict(lower=answer['lower_bound'],upper=answer['upper_bound'],denominator=3*sum(map(sum,u['gt_bins'])))
        proof=next(x for x in summary['oracle_certificates'] if x['unit']==key);ids=[]
        for global_id in proof['oracle']['selected_ids']:
            slot,local=proof['global_to_local'][global_id]
            if local not in cuts:raise ValueError('old197 candidate outside cuts')
            a=next(a for a in u['actions'] if a['action_id']==cuts[local])
            if a['member_index']!=slot:raise ValueError('old197 slot mismatch')
            ids.append(cuts[local])
        warm[key].append(ids);a=proof['oracle']
        comparisons[key]['allcut197']=dict(lower=a['lower_bound'],upper=a['upper_bound'],denominator=3*sum(map(sum,u['gt_bins'])))
    return dict(prereg=prereg,units=units,warm_starts=warm,references=dict(units=comparisons),
                source_seals=seals,old_source_seals=oldseals,scope=dict(clips=6,max_seed=14,head_units=8,raw_reads=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prereg',required=True)
    p.add_argument('--old199',type=Path,required=True);p.add_argument('--old197-summary',type=Path,required=True)
    p.add_argument('--fit-finished',action='store_true',required=True);p.add_argument('--wait',action='store_true')
    p.add_argument('--lp-limit',type=float,default=30);p.add_argument('--mip-limit',type=float,default=300);p.add_argument('--all-limit',type=float,default=120)
    args=p.parse_args()
    while not all((args.root/k/'SEALED.sha256').is_file() for k in ('frozen','owners','predictions')):
        if not args.wait:raise FileNotFoundError('waiting complete top seals')
        time.sleep(30)
    data=prepare(args.root,args.prereg,args.old199,args.old197_summary);out=args.root/'capacity';out.mkdir(exist_ok=True)
    path=out/'compact_input.json';text=json.dumps(data,sort_keys=True,separators=(',',':'))+'\n'
    if path.exists():
        if path.read_text()!=text:raise ValueError('compact immutable mismatch')
    else:path.write_text(text)
    cap.run_cli(SimpleNamespace(input=path,sha256=sha(path),prereg=args.prereg,out=out,
        lp_limit=args.lp_limit,mip_limit=args.mip_limit,all_limit=args.all_limit,linear_limit=200))

if __name__=='__main__':main()
