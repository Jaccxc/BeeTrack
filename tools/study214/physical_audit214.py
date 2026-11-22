"""Independent214 full90 materialization, fresh geometry, events and bounds."""
import argparse,sys
from collections import Counter
from pathlib import Path
from fractions import Fraction
import numpy as np
import cv2
import field214 as f
p=f.p;own=f.own

def independent_events(R,polygon):
    table={};sides=cv2_polygon_sides(R,polygon)
    for pid,ix in f.scene(R):table[pid]=own.event(R[ix,0],sides[ix])
    return table,sides

def cv2_polygon_sides(R,polygon):
    polygon=np.asarray(polygon,np.float32)
    return np.asarray([cv2.pointPolygonTest(polygon,tuple(map(float,r[2:4]+r[4:6]/2)),False)>=0 for r in R],bool)

def diagnostics(R,A,endpoints,GT,owner,polygon,saved):
    records=saved['records'];assert len(records)==sum(e['eligible'] for e in endpoints);matched=[];flips=repairs=damages=moved=0
    for e,s in zip([e for e in endpoints if e['eligible']],records):
        j=e['raw_index'];before=R[j,2:4]+R[j,4:6]/2;after=A[j,2:4]+A[j,4:6]/2;poly=np.asarray(polygon,np.float32)
        ds=[cv2.pointPolygonTest(poly,tuple(map(float,z)),True) for z in (before,after)];g=owner[j];flip=(ds[0]>=0)!=(ds[1]>=0);move=not np.array_equal(R[j,2:4],A[j,2:4])
        assert s['raw_index']==j and s['original_matched']==bool(g>=0) and s['side_flipped']==flip and s['moved']==move;flips+=flip;moved+=move
        if g>=0:
            center=GT[g,2:4]+GT[g,4:6]/2;gd=cv2.pointPolygonTest(poly,tuple(map(float,center)),True);diag=np.hypot(R[j,4],R[j,5]);b=abs(ds[0]-gd)/diag;a=abs(ds[1]-gd)/diag
            p.close([b,a],[s['before_boundary_MAE'],s['after_boundary_MAE']],1e-13)
            err0=(ds[0]>=0)!=(gd>=0);err1=(ds[1]>=0)!=(gd>=0);repairs+=flip and err0 and not err1;damages+=flip and not err0 and err1
            matched.append((b,a,gd>=0))
    assert saved['matched']==len(matched) and saved['side_flips']==flips and saved['moved']==moved and saved['flip_repaired']==repairs and saved['flip_damaged']==damages
    if matched:
        p.close(saved['after_boundary_MAE'],sum(z[1] for z in matched)/len(matched),1e-13)
        assert saved['GT_inside']==sum(z[2] for z in matched) and saved['GT_outside']==sum(not z[2] for z in matched)
    return {k:saved[k] for k in ('matched','GT_inside','GT_outside','after_boundary_MAE')}

def check_bound(R,O,E,polygon,nf,saved_member):
    sid=cv2_polygon_sides(R,polygon);oid=cv2_polygon_sides(O,polygon);allowed={e['raw_index'] for e in E if e['eligible']};menus={'all_side':[],'owner_side':[]};fixed=own.clock([],nf);run_count=0
    for pid,ix in f.scene(R):
        pos=[j for j,i in enumerate(ix) if i in allowed]
        if not pos:
            event=own.event(R[ix,0],sid[ix]);fixed+=own.clock([] if event is None else [event],nf);continue
        run_count+=1
        evidence=saved_member['runs'][run_count-1]
        assert evidence['frames']==R[ix,0].astype(int).tolist() and evidence['sides']==sid[ix].tolist() and evidence['oracle_sides']==oid[ix].tolist() and evidence['eligible']==pos
        for mode in menus:
            options={j:(False,True) if mode=='all_side' else tuple(sorted({bool(sid[ix[j]]),bool(oid[ix[j]])})) for j in pos}
            menus[mode].append(own.side_menu(R[ix,0],sid[ix],options))
    assert np.array_equal(fixed,saved_member['fixed_windows']) and len(saved_member['runs'])==run_count
    answer={}
    for mode,mm in menus.items():
        lo,hi=own.envelope(mm,nf);answer[mode]=(lo+fixed,hi+fixed)
    return answer

def run(a):
    assert a.field_go and a.score_sha;sys.path.insert(0,str(a.repo.resolve()));cv2.setNumThreads(1)
    for stage,digest in f.PINS.items():p.seal(a.root/stage,digest)
    p.seal(a.root/'score',a.score_sha);assert p.read(a.out/'models.json')['passed']
    native,v=f.helpers(a);from beetrack.evaluate import load_tracks
    labels=p.read(a.root/'labels/deployment.json');refs={(r['domain'],r['clip'],r['member_index']):r for r in labels['original_K']};P=p.read(a.root/'predictions/predictions.json');summary=p.read(a.root/'score/summary.json');allclips=[];diag=[]
    for ref,folder,c,arrays in f.records(a):
        destination=a.out/'physical'/ref['domain']/(ref['clip']+'.json');destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            cached=p.read(destination);assert cached['score_seal']==a.score_sha;allclips.append(cached);diag+=cached['diagnostics'];continue
        truth=p.read(a.root/'labels'/refs[c['domain'],c['clip'],0]['path']);GTpath=a.repo/truth['GT_relative_path'];assert p.sha(GTpath)==truth['GT_sha256'];GT=load_tracks(GTpath);geo=f.Geometry(GT,native,v,c['polygon']);Gevents,Gsides=independent_events(GT,c['polygon']);Gbins=own.clock([e for e in Gevents.values() if e is not None],c['nf'])
        arms={a:[] for a in f.ARMS};members_bounds=[];clipdiag=[]
        bound=p.read(a.root/'score/capacity'/c['domain']/(c['clip']+'.json'))
        for slot in range(3):
            R=f.slotrows(arrays,slot);E=f.ends(c,slot);K=native.track_structure(R,c['polygon']);_,owners=geo.match(K);baseline,_=independent_events(R,c['polygon'])
            pred=next(r for r in P['members'] if (r['domain'],r['clip'],r['member_index'])==(c['domain'],c['clip'],slot));oracle=np.zeros((len(E),2))
            for j,e in enumerate(E):
                i=e['raw_index'];g=owners[i]
                if e['eligible'] and g>=0:oracle[j]=np.clip(((geo.G['rows'][g,2:4]+geo.G['rows'][g,4:6]/2)-(R[i,2:4]+R[i,4:6]/2))/R[i,4:6],-.25,.25)
            O=None;memberdiag=dict(domain=c['domain'],clip=c['clip'],member=str(pred['member']),member_index=slot,arms={})
            for arm in f.ARMS:
                offsets=oracle if arm=='O' else np.load(a.root/'predictions'/pred['shifts'][arm]);A=own.materialize(R,E,offsets);where=a.root/'score/clips'/c['domain']/c['clip']/('m'+str(slot))/arm;M=p.read(where/'score.json');reloaded=np.loadtxt(where/'mot.txt',delimiter=',',ndmin=2)
                assert np.array_equal(A,reloaded);own.row_integrity(R,reloaded,[e['raw_index'] for e in E if e['eligible']])
                Pt=native.track_structure(A,c['polygon']);ownersA,_=geo.match(Pt);calc=native.score_from_owners(Pt,geo.G,ownersA,c['polygon'],c['nf']);events,sides=independent_events(A,c['polygon']);assert Pt['events']=={pid:e for pid,e in events.items() if e is not None}
                bins=own.clock([e for e in events.values() if e is not None],c['nf']);metric=M['metrics'];assert np.array_equal(bins,metric['pred_bins']) and np.array_equal(Gbins,metric['gt_bins']);assert calc['nP']==metric['nP'] and calc['nG']==metric['nG']
                for t,edgekey in (('0.5','edges05'),('0.3','edges03')):
                    assert sorted(map(tuple,calc['edges'][t]))==sorted(map(tuple,metric['evidence'][edgekey]));tp=f.old.independent.maximum_matching(Pt['events'],geo.G['events'],calc['edges'][t]);assert tp==metric['iou'][t]['tp'] and calc['iou'][t]['tp_id']==metric['iou'][t]['tp_id']
                if arm=='K':
                    oldref=next(r for r in p.read(a.labels212/'deployment.json')['original_K'] if (r['domain'],r['clip'],r['member_index'])==(c['domain'],c['clip'],slot))
                    for t,edgekey in (('0.5','edges05'),('0.3','edges03')):
                        original=p.read(a.labels212/oldref['graphs'][t]);assert sorted(map(tuple,original['edges']))==sorted(map(tuple,calc['edges'][t])) and original['TP']==calc['iou'][t]['tp']
                changes=Counter('unchanged' if baseline[pid]==e else 'added' if baseline[pid] is None else 'removed' if e is None else 'direction_changed' if baseline[pid][1]!=e[1] else 'time_shifted' for pid,e in events.items());assert dict(changes)==M['event_changes']['counts']
                memberdiag['arms'][arm]=diagnostics(R,A,E,geo.G['rows'],owners,c['polygon'],M['endpoint_diagnostics'])
                arms[arm].append(dict(member=str(pred['member']),pred_bins=bins.tolist(),gt_bins=Gbins.tolist(),nP=calc['nP'],nG=calc['nG'],iou=calc['iou']))
                if arm=='O':O=A
                print('AUDIT214_NATIVE',c['domain'],c['clip'],slot,arm,flush=True)
            members_bounds.append(check_bound(R,O,E,c['polygon'],c['nf'],bound['replay_evidence']['members'][slot]));clipdiag.append(memberdiag)
        for mode in ('all_side','owner_side'):
            lo=sum((x[mode][0] for x in members_bounds),np.zeros_like(Gbins));hi=sum((x[mode][1] for x in members_bounds),np.zeros_like(Gbins));b=bound['layers'][mode]
            assert np.array_equal(lo,b['lower_counts']) and np.array_equal(hi,b['upper_counts']) and own.interval_distance(lo,hi,3*Gbins)==b['lower_bound_numerator']
        member_bounds=[dict(member_index=i,member=str(arms['K'][i]['member']),denominator=int(Gbins.sum()),lower_bound_numerators={mode:own.interval_distance(env[mode][0],env[mode][1],Gbins) for mode in ('all_side','owner_side')}) for i,env in enumerate(members_bounds)]
        report=dict(passed=True,score_seal=a.score_sha,domain=c['domain'],clip=c['clip'],arms=arms,diagnostics=clipdiag,bounds={mode:bound['layers'][mode]['lower_bound_numerator'] for mode in ('all_side','owner_side')},member_bounds=member_bounds,denominator=3*int(Gbins.sum()),fresh_Hungarian_frame_geometries=geo.calls)
        p.save(destination,report);allclips.append(report);diag+=clipdiag
    domains={}
    for h in ('35','33','W','G'):
        domains[h]={}
        for arm in f.ARMS:
            clips=[c for c in allclips if c['domain']==h];N=sum(int(abs(sum((np.asarray(m['pred_bins']) for m in c['arms'][arm]),np.zeros_like(c['arms'][arm][0]['gt_bins']))-3*np.asarray(c['arms'][arm][0]['gt_bins'])).sum()) for c in clips);D=sum(c['denominator'] for c in clips);members=[]
            for member in {m['member'] for c in clips for m in c['arms'][arm]}:
                cells=[m for c in clips for m in c['arms'][arm] if m['member']==member];iou={}
                for t in ('0.5','0.3'):
                    tp=sum(m['iou'][t]['tp'] for m in cells);pr=sum(m['nP'] for m in cells);gt=sum(m['nG'] for m in cells);iou[t]=dict(TP=tp,Pred=pr,GT=gt,F1=2*tp/(pr+gt) if pr+gt else 1.)
                members.append(dict(member=member,iou=iou))
            actual=summary['domains'][h][arm];assert (N,D)==(actual['numerator'],actual['denominator'])
            for m in members:
                saved=next(x for x in actual['members'] if x['member']==m['member'])
                for t in m['iou']:
                    for k,val in m['iou'][t].items():p.close(val,saved['iou'][t][k])
            domains[h][arm]=dict(numerator=N,denominator=D,members=members)
    W=lambda h,arm:Fraction(100*domains[h][arm]['numerator'],domains[h][arm]['denominator'])
    F=lambda h,arm:sum((Fraction(2*m['iou']['0.5']['TP'],m['iou']['0.5']['Pred']+m['iou']['0.5']['GT']) for m in domains[h][arm]['members']),Fraction())/3
    checks={}
    for h in ('35','33'):checks[h+'_E3_gain_at_least_1pp']=W(h,'K')-W(h,'I')>=1;checks[h+'_mean_member_F1_loss_at_most_0.01']=F(h,'I')>=F(h,'K')-Fraction(1,100)
    for h in ('W','G'):checks[h+'_E3_not_worse']=W(h,'I')<=W(h,'K')
    assert checks==summary['gates']['primary']['checks']
    reserve=[d for d in diag if d['domain'] in ('35','33') and d['member_index'] in (1,2)];sufficient=len(reserve)==4 and all(d['arms']['I']['GT_inside']>0 and d['arms']['I']['GT_outside']>0 for d in reserve);better=all(d['arms']['I']['matched']>0 and all(d['arms']['I']['after_boundary_MAE']<d['arms'][a]['after_boundary_MAE'] for a in ('G','Z')) for d in reserve);choice=any(all(W(h,'I')<W(h,a) for a in ('G','Z')) for h in ('35','33'))
    assert summary['gates']['crop_spatial_information']['passed']==(sufficient and better and choice)
    bounds={}
    for h in domains:
        clips=[c for c in allclips if c['domain']==h];D=sum(c['denominator'] for c in clips);layer={}
        for mode in ('all_side','owner_side'):
            N=sum(c['bounds'][mode] for c in clips);published=summary['registered_count_capacity']['pooled'][h][mode];assert (N,D)==(published['lower_bound_numerator'],published['denominator'])
            assert N<=min(domains[h][a]['numerator'] for a in ('K','O'))
            layer[mode]=dict(numerator=N,denominator=D,lower_percent=100*N/D,maximum_gain_percent=100*(domains[h]['K']['numerator']-N)/D)
        assert layer['all_side']['numerator']<=layer['owner_side']['numerator']
        mb=[]
        for member in {m['member'] for c in clips for m in c['member_bounds']}:
            cells=[m for c in clips for m in c['member_bounds'] if m['member']==member];den=sum(m['denominator'] for m in cells)
            mb.append(dict(member=member,denominator=den,lower_bound_numerators={mode:sum(m['lower_bound_numerators'][mode] for m in cells) for mode in ('all_side','owner_side')}))
        bounds[h]=dict(E3=layer,members=mb)
    p.save(a.out/'physical.json',dict(passed=True,domains=domains,count_bounds=bounds,primary_checks=checks,mechanism=dict(sufficient=sufficient,passed=sufficient and better and choice),full90_fresh_geometry=True,scope='All90 exact materializations and fullscene native geometry/edges at both IoUs recomputed, independent DFS maximum matching and30clock. Public197 event-edge kernel shared; no production214 scorer/model imported. All original-owner MAE records and full joint A/B endpoint menus independently checked. Member W bounds supplied independently in this receipt; original scored capacity summary reports E3 bounds.'))
    print('AUDIT214_PHYSICAL_PASS',p.sha(a.out/'physical.json'),flush=True)

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--repo',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--labels212',type=Path,required=True);q.add_argument('--score-sha',required=True);q.add_argument('--field-go',action='store_true');run(q.parse_args())
