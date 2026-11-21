"""Authorized FIFO adapter: complete three-stage seals -> eight-head capacity.

Never decodes K.npz, images or GT tracks. Hashes all frozen bytes to verify the
complete source seal; only owner35/33 numeric payloads enter capacity.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import argparse,json
from pathlib import Path
from types import SimpleNamespace
import capacity203 as cap

DOMAINS=('35','33','W','G');ARMS=('K','G','I','L','T','M')
def sealed(root):
    root=Path(root).resolve();digest=cap.sha(root/'manifest.json')
    if (root/'SEALED.sha256').read_text().strip()!=digest:raise ValueError('top seal differs '+str(root))
    manifest=cap.load(root/'manifest.json')
    for name,value in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or cap.sha(path)!=value:raise ValueError('sealed file differs '+name)
    return dict(manifest_sha256=digest,files=manifest)

def prepare(root,prereg,expected_seals=None):
    roots={k:Path(root)/k for k in ('frozen','owners','predictions')}
    if not all((p/'SEALED.sha256').is_file() for p in roots.values()):raise FileNotFoundError('three complete top seals required')
    times=[(roots[k]/'SEALED.sha256').stat().st_mtime_ns for k in roots]
    if times!=sorted(times):raise ValueError('FIFO seal order differs')
    receipts={k:sealed(p) for k,p in roots.items()}
    if expected_seals is not None and any(receipts[k]['manifest_sha256']!=expected_seals[k] for k in roots):raise ValueError('CPU dispatch pinned seals differ')
    fr=cap.load(roots['frozen']/'deployment.json');pr=cap.load(roots['predictions']/'receipt.json')
    if fr['prereg']!=prereg or pr['prereg']!=prereg:raise ValueError('prereg differs')
    if pr['freeze_receipt']['manifest_sha256']!=receipts['frozen']['manifest_sha256'] or pr['owners_manifest_sha256']!=receipts['owners']['manifest_sha256']:
        raise ValueError('prediction source lineage differs')
    if set(pr['arms'])!=set(ARMS) or set(pr['folds'])!=set(DOMAINS):raise ValueError('incomplete fourfold six arms')
    if set(receipts['predictions']['files'])!={'receipt.json'}|{h+'.'+k+'.json' for h in DOMAINS for k in ('model','masks')}:
        raise ValueError('prediction manifest inventory differs')
    if len(fr['groups'])!=20 or len({g['unit'] for g in fr['groups']})!=20:raise ValueError('old twenty-unit source required')
    roster={};clips=set();catalog_paths={}
    for ref in fr['groups']:
        folder=roots['frozen']/ref['path']
        if cap.sha(folder/'manifest.json')!=ref['seal']:raise ValueError('catalog seal differs')
        c=cap.load(folder/'catalog.json');catalog_paths[c['unit']]=folder/'catalog.json'
        if c['unit']!=ref['unit'] or c['domain']!=ref['domain']:raise ValueError('catalog identity differs')
        if {m['member_index'] for m in c['members']}!={0,1,2} or len(c['members'])!=3:raise ValueError('three source members required')
        for m in c['members']:
            old=m['source_record'];clips.add(old['clip'])
            if not 0<=old['seed']<=14 or old['split']=='test':raise ValueError('old seed0..14 no test only')
            roster[(c['unit'],str(m['member']))]=m['member_index']
        del c
    if len(clips)!=6 or len(roster)!=60:raise ValueError('old six clips / sixty streams only')
    predictions={};counts={};seen=set();last_unit=None;prediction_table={}
    for h in DOMAINS:
        model=cap.load(roots['predictions']/(h+'.model.json'));fold=cap.load(roots['predictions']/(h+'.masks.json'))
        if model['heldout']!=h or fold['heldout']!=h or set(model['source_domains'])!=set(DOMAINS)-{h}:raise ValueError('LOHO fold differs')
        for m in fold['member_clips']:
            key=(m['unit'],str(m['host_member']))
            if key not in roster or key in seen or set(m['selected'])!=set(ARMS):raise ValueError('prediction roster/arms differs')
            seen.add(key);unit=m['unit']
            if last_unit!=unit:
                c=cap.load(catalog_paths[unit]);prediction_table={a['action_id']:str(a['host_member']) for a in c['candidates']};del c;last_unit=unit
            for arm,mask in m['selected'].items():
                if set(mask)!={'candidate_ids'}:raise ValueError('suffix-only mask required')
                ids=mask['candidate_ids']
                if len(ids)!=len(set(ids)) or any(cid not in prediction_table or prediction_table[cid]!=key[1] for cid in ids):raise ValueError('foreign prediction ID')
                predictions.setdefault(unit,{}).setdefault(arm,[]).extend(ids)
    if seen!=roster.keys():raise ValueError('prediction sixty roster incomplete')
    units=[]
    for h in ('35','33'):
        owner=cap.load(roots['owners']/(h+'.json'))
        if owner['prereg']!=prereg or owner['frozen_manifest_sha256']!=receipts['frozen']['manifest_sha256']:raise ValueError('owners source differs')
        for u in owner['units']:
            raw=cap.load(catalog_paths[u['unit']]);table={a['action_id']:a for a in raw['candidates']};covered=[];compact_members=[]
            for m in u['members']:
                if roster[(u['unit'],str(m['member']))]!=m['member_index']:raise ValueError('slot/member drift; never infer seed')
                compact_actions=[]
                for a in m['actions']:
                    if a['action_id'] not in table:raise ValueError('owner introduces action')
                    original=table[a['action_id']]
                    for key in ('host_member','host_member_index','host_track_id','suffix','delta_windows'):
                        if a[key]!=original[key]:raise ValueError('owner changes frozen action '+key)
                    covered.append(a['action_id'])
                    compact_actions.append(dict(action_id=a['action_id'],candidate_id=a['candidate_id'],
                        host_member=a['host_member'],host_member_index=a['host_member_index'],host_track_id=a['host_track_id'],
                        suffix=[{k:s[k] for k in ('source','frame','box')} for s in a['suffix']],delta_windows=a['delta_windows'],
                        labels={t:{'class':a['labels'][t]['class']} for t in ('0.5','0.3')}))
                compact_members.append(dict(member=m['member'],member_index=m['member_index'],actions=compact_actions))
            if len(covered)!=len(set(covered)) or set(covered)!=table.keys():raise ValueError('owner complete candidate inventory differs')
            units.append(dict(unit=u['unit'],domain=h,group=u['group'],video=u['video'],nf=u['nf'],
                pred_sum=u['pred_sum'],gt_bins=u['gt_bins'],members=compact_members))
            del raw,table
        del owner
    normalized=cap.normalize(dict(units=units));warm={}
    for u in normalized:
        key=u['unit'];warm[key]=[];counts[key]={}
        problem=cap.compile_problem(u,'A');known={a['action_id'] for a in problem['actions']}
        for arm in ARMS:
            ids=predictions[key][arm];cap.witness(problem,[cid for cid in ids if cid in known])
            warm[key].append(ids);counts[key][arm]=len(ids)
    return dict(prereg=prereg,prediction_seal=receipts['predictions']['manifest_sha256'],units=units,warm_starts=warm,
        adapter_code_sha256=cap.sha(__file__),
        source_receipts={k:dict(manifest_sha256=r['manifest_sha256']) for k,r in receipts.items()},
        prediction_selected_counts=counts,scope=dict(clip_count=6,max_seed=14,head_units=8,raw_decode=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prereg',required=True)
    for name in ('frozen','owners','prediction'):p.add_argument('--'+name+'-seal',required=True)
    p.add_argument('--fit-finished',action='store_true',required=True);p.add_argument('--lp-limit',type=float,default=30)
    p.add_argument('--mip-limit',type=float,default=300);args=p.parse_args()
    data=prepare(args.root,args.prereg,dict(frozen=args.frozen_seal,owners=args.owners_seal,predictions=args.prediction_seal));out=args.root/'capacity';out.mkdir(exist_ok=True)
    path=out/'compact_input.json';cap.save(path,data)
    prediction_seal=data['prediction_seal'];del data
    cap.run(SimpleNamespace(input=path,sha256=cap.sha(path),prereg=args.prereg,out=out,predictions=args.root/'predictions',
        prediction_seal=prediction_seal,lp_limit=args.lp_limit,mip_limit=args.mip_limit))

if __name__=='__main__':main()
