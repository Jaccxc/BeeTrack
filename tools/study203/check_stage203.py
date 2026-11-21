"""Tiny synthetic full sealed-adapter/CLI/resume check; never reads field roots."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import tempfile
from pathlib import Path
from types import SimpleNamespace
import capacity203 as cap
import stage203 as stage

def seal(root):
    cap.save(root/'manifest.json',{p.relative_to(root).as_posix():cap.sha(p) for p in root.rglob('*') if p.is_file()})
    (root/'SEALED.sha256').write_text(cap.sha(root/'manifest.json'))

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);units=[];refs=[];skeleton=[]
        for h,ng in [('35',4),('33',4),('W',6),('G',6)]:
            for g in range(ng):
                key=f'{h}_{g}';video=h+str(g%2) if h in ('W','G') else h
                members=[]
                for slot in range(3):
                    member=key+'|'+str(slot);old=dict(seed=slot+10,clip=video,split='val')
                    members.append(dict(member=member,member_index=slot,source_record=old))
                    skeleton.append(dict(unit=key,host_member=member,hive=h))
                action=dict(action_id=key+'edge',candidate_id=key+'edge',host_member=members[0]['member'],host_member_index=0,host_track_id=1,
                    suffix=[dict(source=['SHA',0],frame=3,box=[0,0,2,2])],delta_windows=[[-1,0]]*30)
                catalog=dict(unit=key,domain=h,members=members,candidates=[action]);folder=root/'frozen'/key;folder.mkdir(parents=True)
                cap.save(folder/'catalog.json',catalog);seal(folder)
                refs.append(dict(unit=key,domain=h,group=g,video=video,path=key,seal=cap.sha(folder/'manifest.json')))
                if h in ('35','33'):
                    mm=[dict(member=m['member'],member_index=m['member_index'],actions=[dict(action,labels={t:{'class':'S'} for t in ('0.5','0.3')})] if m['member_index']==0 else []) for m in members]
                    units.append(dict(unit=key,domain=h,group=g,video=video,nf=3,pred_sum=[[4,0]]*30,gt_bins=[[1,0]]*30,members=mm))
        cap.save(root/'frozen/deployment.json',dict(prereg='toy',groups=refs));seal(root/'frozen');fr=cap.sha(root/'frozen/manifest.json')
        (root/'owners').mkdir()
        for h in ('35','33'):cap.save(root/f'owners/{h}.json',dict(prereg='toy',frozen_manifest_sha256=fr,units=[u for u in units if u['domain']==h]))
        seal(root/'owners');owner=cap.sha(root/'owners/manifest.json');(root/'predictions').mkdir()
        cap.save(root/'predictions/receipt.json',dict(prereg='toy',freeze_receipt={'manifest_sha256':fr},owners_manifest_sha256=owner,arms=list(stage.ARMS),folds=list(stage.DOMAINS)))
        for h in stage.DOMAINS:
            cap.save(root/f'predictions/{h}.model.json',dict(heldout=h,source_domains=[d for d in stage.DOMAINS if d!=h]))
            cap.save(root/f'predictions/{h}.masks.json',dict(heldout=h,member_clips=[dict(m,selected={a:{'candidate_ids':[]} for a in stage.ARMS}) for m in skeleton if m['hive']==h]))
        seal(root/'predictions');data=stage.prepare(root,'toy');assert len(data['units'])==8
        out=root/'capacity';out.mkdir();path=out/'compact_input.json';cap.save(path,data)
        args=SimpleNamespace(input=path,sha256=cap.sha(path),prereg='toy',out=out,predictions=root/'predictions',prediction_seal=data['prediction_seal'],lp_limit=1,mip_limit=1)
        cap.run(args);before=cap.sha(out/'capacity.json');cap.run(args);assert cap.sha(out/'capacity.json')==before
        (root/'predictions/SEALED.sha256').unlink()
        try:stage.prepare(root,'toy')
        except FileNotFoundError:pass
        else:raise AssertionError('incomplete predictions accepted')
        print('PASS synthetic three-seal adapter; old6clip/seed14; member namespaces; 24 certificates; immutable resume; missing seal rejects')

if __name__=='__main__':main()
