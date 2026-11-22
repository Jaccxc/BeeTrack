"""214 fullscene endpoint-centre intervention scorer (post common-fold seal).

Pure core accepts explicit original rows, registered eligible endpoints and
normalized dx/width,dy/height offsets. GT/provider access belongs after the
all-four-fold prediction gate. Every arm is rematched from its actual boxes;
no old graph, local TP delta or count-only surrogate supplies changed-arm F1.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import gzip
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

ARMS=('K','G','I','Z','O')
DOMAINS=('35','33','W','G')


def eligible(endpoint):return bool(endpoint.get('gate',{}).get('eligible',endpoint.get('eligible',True)))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('w',encoding='utf8',newline='\n') as f:json.dump(value,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')


def read(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf8') if path.suffix=='.gz' else path.open(encoding='utf8')) as f:return json.load(f)


def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value,dtype='<f8').tobytes()).hexdigest()


class NativeCache:
    """Resume only an exact same member/arm/GT/physical/native-code request."""
    def __init__(self,root,inputs):
        self.root=Path(root);self.inputs=inputs;self.fresh=0;self.reused=0
        from tools.study195 import run
        from tools.study193 import source193
        import beetrack.evaluate as official
        self.code={Path(m.__file__).name:sha(m.__file__) for m in (run,source193,official)}

    def scorer(self,domain,clip,slot):
        order=iter(ARMS)
        def call(gt,physical,polygon,nf):
            arm=next(order);folder=self.root/domain/clip/('m'+str(slot))/arm
            identity=dict(inputs=self.inputs,domain=domain,clip=clip,slot=slot,arm=arm,GT_sha256=array_sha(gt),
                physical_sha256=array_sha(physical),polygon=np.asarray(polygon).tolist(),nf=nf,scorer_code=self.code)
            if folder.exists():
                manifest=json.loads((folder/'manifest.json').read_text())
                if sha(folder/'manifest.json')!=(folder/'SEALED.sha256').read_text().strip() or any(sha(folder/k)!=v for k,v in manifest.items()):raise ValueError('native resume cache hash differs')
                if json.loads((folder/'identity.json').read_text())!=identity:raise ValueError('native resume identity differs')
                cached=np.loadtxt(folder/'mot.txt',delimiter=',',ndmin=2)
                if not np.array_equal(cached,physical):raise ValueError('native cache raw boxes/order differ')
                self.reused+=1;return json.loads((folder/'metrics.json').read_text())
            from tools.study195.run import score_member_native
            folder.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.native214-',dir=folder.parent) as temp:
                partial=Path(temp)/'payload';partial.mkdir();np.savetxt(partial/'mot.txt',physical,delimiter=',',fmt='%.17g')
                reread=np.loadtxt(partial/'mot.txt',delimiter=',',ndmin=2)
                if not np.array_equal(reread,physical):raise ValueError('native cache physical serialization differs')
                metric=score_member_native(gt,reread,polygon,nf)
                save(partial/'metrics.json',metric);save(partial/'identity.json',identity)
                save(partial/'manifest.json',{p.name:sha(p) for p in partial.iterdir()})
                (partial/'SEALED.sha256').write_text(sha(partial/'manifest.json')+'\n');partial.rename(folder)
            self.fresh+=1;return metric
        return call


def materialize(rows,endpoints,offsets):
    """Only eligible row x/y changes; fixed original row order/size/date/ID/score."""
    R=np.asarray(rows,np.float64);D=np.asarray(offsets,np.float64)
    if R.ndim!=2 or R.shape[1]!=7 or not np.isfinite(R).all() or np.any(R[:,4:6]<=0):raise ValueError('finite original M7, positive box size required')
    if D.shape!=(len(endpoints),2) or not np.isfinite(D).all() or np.any(np.abs(D)>.25+1e-12):raise ValueError('registered Nx2 normalized offsets clipped within .25 required')
    indices=[int(e['raw_index']) for e in endpoints]
    if len(set(indices))!=len(indices) or any(i<0 or i>=len(R) for i in indices):raise ValueError('unique original endpoint row indices required')
    if any(not eligible(e) and np.any(d!=0) for e,d in zip(endpoints,D)):raise ValueError('ineligible/unknown endpoint must remain K')
    from beetrack.evaluate import by_track
    tagged=by_track(np.c_[R,np.arange(len(R))]);native=tagged[:,:7];raw=tagged[:,-1].astype(np.int64)
    _,starts=np.unique(native[:,1],return_index=True);last=np.r_[starts[1:],len(native)]-1
    endpoint_rows=set(raw[np.r_[starts,last]].tolist())
    if not set(indices)<=endpoint_rows:raise ValueError('registered endpoint is not an original canonical run endpoint')
    result=R.copy()
    if indices:result[indices,2:4]+=D*R[indices,4:6]
    if not np.array_equal(result[:,[0,1,4,5,6]],R[:,[0,1,4,5,6]]):raise AssertionError('non-position geometry/date/ID/score changed')
    unchanged=np.ones(len(R),bool);unchanged[indices]=False
    if not np.array_equal(result[unchanged],R[unchanged]):raise AssertionError('nonendpoint row changed')
    return result,dict(raw_indices=indices,normalized_offsets=D.tolist(),changed_raw_indices=[i for i,d in zip(indices,D) if np.any(d!=0)],
        row_count=len(R),row_order_preserved=True,frame_ID_size_score_exact=True,eligible_endpoints_only=True)


def original_owners(gt,rows,polygon):
    """Whole-scene .5 frame ownership, indexed back to original prediction rows."""
    from beetrack.evaluate import by_track,frame_match,iou_matrix
    G=by_track(gt);tag=by_track(np.c_[rows,np.arange(len(rows))]);P=tag[:,:7];raw=tag[:,-1].astype(int)
    owners=frame_match(G,P,thr=.5);gt_rows=np.full(len(rows),-1,np.int64)
    # owners are canonical GT track IDs; exact GT observation is same-date row.
    lookup=defaultdict(list)
    for j,g in enumerate(G):lookup[int(g[1]),int(g[0])].append(j)
    recovered={}
    for j,gid in enumerate(owners):
        if gid<0:continue
        choices=lookup[int(gid),int(P[j,0])]
        if len(choices)==1:gt_rows[raw[j]]=choices[0];continue
        # Recover the exact official full-frame Hungarian row, never choose a
        # nearest box or add an isolation gate when GT canonical IDs collide.
        date=int(P[j,0])
        if date not in recovered:
            from scipy.optimize import linear_sum_assignment
            gg=np.flatnonzero(G[:,0]==date);pp=np.flatnonzero(P[:,0]==date)
            xy=lambda a:np.c_[a[:,2:4],a[:,2:4]+a[:,4:6]]
            matrix=iou_matrix(xy(G[gg]),xy(P[pp]));a,b=linear_sum_assignment(-matrix)
            mapping={int(pp[y]):int(gg[x]) for x,y in zip(a,b) if matrix[x,y]>=.5}
            if any((int(G[mapping[int(p)],1]) if int(p) in mapping else -1)!=owners[p] for p in pp):raise AssertionError('recovered GT observation differs official owner')
            recovered[date]=mapping
        gt_rows[raw[j]]=recovered[date][j]
    return G,gt_rows


def oracle_offsets(rows,endpoints,canonical_GT,owner_rows):
    result=np.zeros((len(endpoints),2),np.float64)
    for k,e in enumerate(endpoints):
        if not eligible(e):continue
        i=int(e['raw_index']);g=int(owner_rows[i])
        if g<0:continue
        p=np.asarray(rows[i]);q=canonical_GT[g]
        result[k]=np.clip(((q[2:4]+q[4:6]/2)-(p[2:4]+p[4:6]/2))/p[4:6],-.25,.25)
    return result


def endpoint_diagnostics(original,corrected,endpoints,GT,owners,polygon):
    import cv2
    poly=np.asarray(polygon,np.float32);records=[]
    def distance(p):return float(cv2.pointPolygonTest(poly,tuple(map(float,p)),True))
    for e in endpoints:
        if not eligible(e):continue
        i=int(e['raw_index']);a=original[i];b=corrected[i];before=a[2:4]+a[4:6]/2;after=b[2:4]+b[4:6]/2;diag=float(np.hypot(a[4],a[5]));g=int(owners[i])
        bd,ad=distance(before),distance(after);moved=not np.array_equal(a[2:4],b[2:4]);r=dict(endpoint_id=e['endpoint_id'],raw_index=i,
            moved=moved,original_matched=g>=0,original_side=bd>=0,corrected_side=ad>=0,side_flipped=(bd>=0)!=(ad>=0),
            original_signed_polygon_distance=bd,corrected_signed_polygon_distance=ad,original_box_diagonal=diag)
        if g>=0:
            q=GT[g];center=q[2:4]+q[4:6]/2;gd=distance(center);old_error=(bd>=0)!=(gd>=0);new_error=(ad>=0)!=(gd>=0)
            r.update(GT_canonical_id=int(q[1]),GT_frame=int(q[0]),GT_side=gd>=0,GT_signed_polygon_distance=gd,
                before_side_error=old_error,after_side_error=new_error,
                before_boundary_MAE=abs(bd-gd)/diag,after_boundary_MAE=abs(ad-gd)/diag,
                before_center_error=float(np.linalg.norm(before-center)/diag),after_center_error=float(np.linalg.norm(after-center)/diag),
                flip_repaired=bool(r['side_flipped'] and old_error and not new_error),flip_damaged=bool(r['side_flipped'] and not old_error and new_error))
        records.append(r)
    matched=[r for r in records if r['original_matched']];n=len(matched)
    result=dict(registered_endpoints=len(endpoints),ineligible_or_unknown=sum(not eligible(e) for e in endpoints),eligible=len(records),matched=n,unmatched=len(records)-n,moved=sum(r['moved'] for r in records),
        moved_original_unmatched=sum(r['moved'] and not r['original_matched'] for r in records),
        GT_inside=sum(r['GT_side'] for r in matched),GT_outside=sum(not r['GT_side'] for r in matched),
        side_flips=sum(r['side_flipped'] for r in records),flip_repaired=sum(r.get('flip_repaired',False) for r in records),
        flip_damaged=sum(r.get('flip_damaged',False) for r in records),
        normalization='original prediction box hypot(width,height); fixed original fullscene .5 GT owner',records=records)
    for label,key in (('boundary_MAE','boundary_MAE'),('center_error','center_error'),('side_error_rate','side_error')):
        for phase in ('before','after'):result[phase+'_'+label]=sum(r[phase+'_'+key] for r in matched)/n if n else None
    return result


def event_changes(before,after,polygon):
    from beetrack.evaluate import by_track,zone_tracks
    def table(rows):
        C=by_track(rows);result={}
        for pid,(direction,start,end,crosses) in zone_tracks(C,polygon).items():
            if not direction:result[int(pid)]=None;continue
            j=next(j for j,d in reversed(crosses) if d==direction)
            result[int(pid)]=(1 if direction=='in' else -1,int(C[j,0]))
        return result
    a,b=table(before),table(after)
    if a.keys()!=b.keys():raise ValueError('position-only intervention altered canonical run partition')
    rows=[]
    for pid in sorted(a):
        x,y=a[pid],b[pid]
        kind='unchanged' if x==y else ('added' if x is None else ('removed' if y is None else ('direction_changed' if x[0]!=y[0] else 'time_shifted')))
        rows.append(dict(canonical_run_id=pid,before=x,after=y,change=kind))
    return dict(counts=dict(Counter(r['change'] for r in rows)),runs=rows)


def metric_counts(metric):
    p=np.asarray(metric['pred_bins'],np.int64);g=np.asarray(metric['gt_bins'],np.int64)
    if p.shape!=g.shape:raise ValueError('native GT/pred clock shape differs')
    value=dict(numerator=int(np.abs(p-g).sum()),denominator=30*int(metric['nG']),nGT=int(metric['nG']),nPred=int(metric['nP']),iou={})
    value['W']=100*value['numerator']/value['denominator'] if value['denominator'] else None
    for t in ('0.5','0.3'):
        q=metric['iou'][t];tp=int(q['tp']);pr=int(q['pred']);gt=int(q['gt'])
        value['iou'][t]=dict(TP=tp,Pred=pr,GT=gt,FP=pr-tp,FN=gt-tp,E=pr+gt-2*tp,F1=2*tp/(pr+gt) if pr+gt else 1.,TP_ID=int(q['tp_id']))
    return value


def ensemble(members):
    if len(members)!=3:raise ValueError('exactly three original members required')
    G=np.asarray(members[0]['gt_bins'],np.int64);ng=int(members[0]['nG'])
    if any(int(m['nG'])!=ng or not np.array_equal(G,m['gt_bins']) for m in members):raise ValueError('member GT clocks differ')
    P=sum((np.asarray(m['pred_bins'],np.int64) for m in members),np.zeros_like(G));raw=int(np.abs(P-3*G).sum());den=90*ng
    return dict(numerator=raw,denominator=den,E3=100*raw/den if den else None,GT=ng,pred_sum_bins=P.tolist(),gt_bins=G.tolist(),
        member_metrics=[metric_counts(m) for m in members],mean_member_F1={t:sum(metric_counts(m)['iou'][t]['F1'] for m in members)/3 for t in ('0.5','0.3')},
        count_rule='sum three predicted bins versus three GT bins, then divide by 90*GT; never mean three W percentages')


def pool_clips(clips):
    if not clips:raise ValueError('empty domain clips')
    n=sum(x['numerator'] for x in clips);d=sum(x['denominator'] for x in clips)
    roster=[str(m['member']) for m in clips[0]['members']]
    if len(set(roster))!=3 or any({str(m['member']) for m in c['members']}!=set(roster) for c in clips):raise ValueError('cross-clip original member roster differs')
    pooled=[]
    for member in roster:
        values=[next(m['counts'] for m in c['members'] if str(m['member'])==member) for c in clips];q={}
        for t in ('0.5','0.3'):
            v={k:sum(x['iou'][t][k] for x in values) for k in ('TP','Pred','GT','FP','FN','E','TP_ID')};v['F1']=2*v['TP']/(v['Pred']+v['GT']) if v['Pred']+v['GT'] else 1.;q[t]=v
        pooled.append(dict(member=member,iou=q,numerator=sum(x['numerator'] for x in values),denominator=sum(x['denominator'] for x in values)))
    return dict(numerator=n,denominator=d,E3=100*n/d if d else None,members=pooled,
        mean_member_F1={t:sum(m['iou'][t]['F1'] for m in pooled)/3 for t in ('0.5','0.3')})


def assert_K212(metric,graphs,nf):
    from tools.study197.candidates import window_counts
    for t,g in graphs.items():
        e=metric['evidence'];pred=sorted((int(v[0]),int(v[1])) for v in e['predictions']);gt=sorted((int(v[0]),int(v[1])) for v in e['gt'])
        if pred!=sorted((v['canonical_event_id'],v['signed_frame']) for v in g['events']) or gt!=sorted((v['canonical_event_id'],v['signed_frame']) for v in g['GT_events']):raise ValueError('K differs212 signed event roster')
        if sorted(map(tuple,e['edges05' if t=='0.5' else 'edges03']))!=sorted(map(tuple,g['edges'])) or metric['iou'][t]['tp']!=g['TP']:raise ValueError('K differs212 full legal graph/TP')
        for key,rows in (('pred_bins',pred),('gt_bins',gt)):
            bins=window_counts([dict(frame=abs(date),direction=1 if date>0 else -1) for pid,date in rows],nf)
            if not np.array_equal(bins,np.asarray(metric[key])):raise ValueError('K differs212 full30 clock')
    return dict(all18_original_K=True,signed_events_exact=True,both_IoU_edges_TP_exact=True,all30_integer_windows_exact=True,
                TP_ID_reference='212 graph archive does not expose TP_ID; current native value retained without a historical equality claim')


def score_member(gt,rows,endpoints,offsets,polygon,nf,old212_graphs,*,scorer=None,original_endpoint_truth=None):
    """Late explicit native API; caller must verify all-four/18-member seal first."""
    if scorer is None:
        from tools.study195.run import score_member_native
        scorer=score_member_native
    if set(offsets)!=set(('K','G','I','Z')):raise ValueError('sealed four blind arm predictions required; O must be generated late')
    if np.any(np.asarray(offsets['K'])!=0):raise ValueError('K must be exactly zero displacement')
    GT,owners=original_owners(gt,rows,polygon)
    if original_endpoint_truth is not None:
        records={r['endpoint_id']:r for r in original_endpoint_truth}
        if len(records)!=len(endpoints) or set(records)!={e['endpoint_id'] for e in endpoints}:raise ValueError('214 original truth endpoint roster differs')
        for e in endpoints:
            r=records[e['endpoint_id']];i=int(e['raw_index']);g=int(owners[i])
            if r['raw_index']!=i or bool(r['has_owner'])!=(g>=0):raise ValueError('214 original wholeframe ownership differs labels')
            if g>=0 and (r['canonical_GT_owner']!=int(GT[g,1]) or not np.array_equal(np.asarray(r['GT_row']),GT[g])):raise ValueError('214 exact original GT observation differs labels')
    all_offsets=dict(offsets,O=oracle_offsets(rows,endpoints,GT,owners));result={}
    for arm in ARMS:
        physical,proof=materialize(rows,endpoints,all_offsets[arm]);metric=scorer(gt,physical,polygon,nf)
        if arm=='K':closure=assert_K212(metric,old212_graphs,nf)
        result[arm]=dict(metrics=metric,counts=metric_counts(metric),proof=proof,
            endpoint_diagnostics=endpoint_diagnostics(rows,physical,endpoints,GT,owners,polygon),
            event_changes=event_changes(rows,physical,polygon),physical_rows=physical)
    return dict(arms=result,K212_closure=closure,oracle_scope='posthoc clipped true-owner centre displacement, not an upper bound')


def gates(domains,member_diagnostics):
    def e3(x):return Fraction(100*x['numerator'],x['denominator']) if x['denominator'] else None
    checks={}
    for h in ('35','33'):
        k,i=domains[h]['K'],domains[h]['I'];checks[h+'_E3_gain_at_least_1pp']=e3(k) is not None and e3(i) is not None and e3(k)-e3(i)>=1
        def meanF(x):
            return sum((Fraction(2*m['iou']['0.5']['TP'],m['iou']['0.5']['Pred']+m['iou']['0.5']['GT']) if m['iou']['0.5']['Pred']+m['iou']['0.5']['GT'] else Fraction(1) for m in x['members']),Fraction())/3
        checks[h+'_mean_member_F1_loss_at_most_0.01']=meanF(i)>=meanF(k)-Fraction(1,100)
    for h in ('W','G'):checks[h+'_E3_not_worse']=e3(domains[h]['I']) is not None and e3(domains[h]['K']) is not None and e3(domains[h]['I'])<=e3(domains[h]['K'])
    reserve=[m for m in member_diagnostics if m['domain'] in ('35','33') and m['member_index'] in (1,2)];reasons=[];cells=[]
    if len(reserve)!=4:reasons.append('four registered head reserve member cells required')
    for m in reserve:
        a=m['arms'];n=a['I']['matched'];both=a['I']['GT_inside']>0 and a['I']['GT_outside']>0
        better=n>0 and all(a['I']['after_boundary_MAE']<a[c]['after_boundary_MAE'] for c in ('G','Z'))
        cells.append(dict(domain=m['domain'],clip=m['clip'],member=m['member'],member_index=m['member_index'],matched=n,both_GT_sides=both,I_strictly_better_G_Z=better))
        if not both:reasons.append(m['domain']+'/m'+str(m['member_index'])+': lacks both original matched GT sides')
        if not better:reasons.append(m['domain']+'/m'+str(m['member_index'])+': I not strictly lower boundary-distance MAE than G and Z')
    changed=any(e3(domains[h]['I']) is not None and all(e3(domains[h]['I'])<e3(domains[h][c]) for c in ('G','Z')) for h in ('35','33'))
    if not changed:reasons.append('no head I E3 strictly below both G and Z')
    insufficient=len(reserve)!=4 or any(not c['both_GT_sides'] or c['matched']==0 for c in cells)
    status='inconclusive_support' if insufficient else ('passed' if not reasons else 'not_passed')
    return dict(primary=dict(passed=all(checks.values()),checks=checks),crop_spatial_information=dict(passed=not reasons,status=status,cells=cells,reasons=reasons,
        interpretation='crop spatial information only, not biological bee identity'))


def evaluate_explicit(groups,decision_provider,gt_provider,reference212_provider,out,input_receipts,final_validation=None):
    """Explicit late runner, after caller verifies all4/18 prediction seal.

    groups is a lazy six-clip iterable: domain/clip/polygon/nf, members[3] with
    member(global),member_index(slot),rows(M7),endpoints[{endpoint_id,raw_index}].
    Provider calls occur only here, after the caller's common-seal gate. Each
    decision_provider(group,member) returns K/G/I/Z normalized arrays aligned to
    frozen endpoint order. reference212_provider returns both original graphs.
    """
    out=Path(out)
    if out.exists():raise FileExistsError('immutable score output exists')
    if not input_receipts.get('all_four_folds_18_members_common_seal_verified'):raise ValueError('full common prediction gate required before provider access')
    out.parent.mkdir(parents=True,exist_ok=True);clips=[];diagnostics=[];seen=set();native=0;bound_refs=[]
    cache=NativeCache(out.with_name(out.name+'_native_cache'),input_receipts)
    with tempfile.TemporaryDirectory(prefix='.eval214-',dir=out.parent) as temp:
        root=Path(temp)/'payload';root.mkdir()
        for group in groups:
            domain=group['domain'];clip=group['clip'];key=(domain,clip)
            if domain not in DOMAINS or key in seen or len(group['members'])!=3:raise ValueError('six unique original three-member clips required')
            seen.add(key);slots=[m['member_index'] for m in group['members']]
            if sorted(slots)!=[0,1,2] or len({str(m['member']) for m in group['members']})!=3:raise ValueError('three distinct registered members/slots required')
            gt=gt_provider(group);per_arm={a:[] for a in ARMS};member_refs=[];bound_members=[]
            for member in sorted(group['members'],key=lambda m:m['member_index']):
                offsets=decision_provider(group,member);reference=reference212_provider(group,member)
                scored=score_member(gt,member['rows'],member['endpoints'],offsets,group['polygon'],group['nf'],reference,
                    scorer=cache.scorer(domain,clip,member['member_index']),original_endpoint_truth=member.get('original_endpoint_truth'))
                bound_members.append(dict(rows=member['rows'],endpoints=[e for e in member['endpoints'] if eligible(e)],oracle_rows=scored['arms']['O']['physical_rows']))
                diag=dict(domain=domain,clip=clip,member=str(member['member']),member_index=member['member_index'],arms={})
                refs={}
                for arm,value in scored['arms'].items():
                    relative=Path('clips')/domain/clip/('m'+str(member['member_index']))/arm;folder=root/relative;folder.mkdir(parents=True)
                    physical=value.pop('physical_rows');np.savetxt(folder/'mot.txt',physical,delimiter=',',fmt='%.17g')
                    reloaded=np.loadtxt(folder/'mot.txt',delimiter=',',ndmin=2)
                    if not np.array_equal(physical,reloaded):raise ValueError('physical MOT serialization changed boxes')
                    # Re-reading .17g is exact, so the fullscene match remains the
                    # match on these identical numeric rows; no second scorer run.
                    value['physical_receipt']=dict(path='mot.txt',sha256=sha(folder/'mot.txt'),rows=len(physical),roundtrip_numeric_exact=True)
                    save(folder/'score.json',dict(domain=domain,clip=clip,member=str(member['member']),member_index=member['member_index'],arm=arm,**value))
                    refs[arm]=(relative/'score.json').as_posix();per_arm[arm].append(dict(member=str(member['member']),metric=value['metrics'],counts=value['counts']))
                    diag['arms'][arm]={k:v for k,v in value['endpoint_diagnostics'].items() if k!='records'};native+=1
                member_refs.append(dict(member=str(member['member']),member_index=member['member_index'],arms=refs,K212_closure=scored['K212_closure']))
                diagnostics.append(diag)
            result=dict(domain=domain,clip=clip,nf=group['nf'],members=member_refs,arms={})
            for arm,values in per_arm.items():
                value=ensemble([v['metric'] for v in values]);value['members']=[dict(member=v['member'],counts=v['counts']) for v in values]
                result['arms'][arm]=value
            from bound214 import bound_scene
            bound=bound_scene(bound_members,group['polygon'],group['nf'],np.asarray(per_arm['K'][0]['metric']['gt_bins'],np.int64))
            bound_path='capacity/'+domain+'/'+clip+'.json';save(root/bound_path,bound)
            bound_refs.append(dict(domain=domain,clip=clip,path=bound_path,denominator=bound['denominator'],
                lower_bound_numerators={layer:value['lower_bound_numerator'] for layer,value in bound['layers'].items()}))
            del bound_members,bound
            clips.append(result);print('CENTER214_CLIP_COMPLETE',domain,clip,flush=True)
        if len(seen)!=6 or native!=90 or len(diagnostics)!=18:raise ValueError('complete 6clips/18members/90native states required')
        domains={h:{a:pool_clips([c['arms'][a] for c in clips if c['domain']==h]) for a in ARMS} for h in DOMAINS}
        summary=dict(schema='center214-native-evaluation-v1',input_receipts=input_receipts,clips=clips,domains=domains,
            member_endpoint_diagnostics=diagnostics,gates=gates(domains,diagnostics),
            registered_count_capacity=dict(clips=bound_refs,pooled={h:{layer:dict(
                lower_bound_numerator=sum(b['lower_bound_numerators'][layer] for b in bound_refs if b['domain']==h),
                denominator=sum(b['denominator'] for b in bound_refs if b['domain']==h)) for layer in ('all_side','owner_side')} for h in DOMAINS},
                role='two registered per-window count lower bounds; not a changed primary gate or universal pixel-output bound'),
            scope=dict(original_K_members=18,native_fullscene_requests=90,fresh_native_scores_this_attempt=cache.fresh,resumed_sealed_native_scores_this_attempt=cache.reused,
                native_cache_root=str(cache.root),changed_arm_F1_from_old_graph=False,
                oracle='posthoc original-owner centre displacement clip±.25, not an upper bound',
                inference='Repeated six original clips/three old members; crop spatial information is not bee identity'))
        if final_validation is not None:final_validation()
        save(root/'summary.json',summary);save(root/'receipt.json',dict(inputs=input_receipts,code_sha256=sha(__file__),native_fullscene_requests=native,
            fresh_native_scores_this_attempt=cache.fresh,resumed_sealed_native_scores_this_attempt=cache.reused))
        save(root/'manifest.json',{p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()})
        (root/'SEALED.sha256').write_text(sha(root/'manifest.json')+'\n');root.rename(out)
    return summary


def run(args):
    """Single CPU late CLI; fourfold/18-member gate precedes all truth I/O."""
    import fit214
    import prepare214
    prediction=fit214.load_predictions(args.predictions)
    # No truth file is opened above this line.
    P=prediction['predictions'];receipt=prediction['receipt'];predsha=sha(args.predictions/'manifest.json')
    prepare214.verify(args.frozen);prepare214.verify(args.labels);prepare214.verify(args.labels212)
    frozen=read(args.frozen/'deployment.json');labels=read(args.labels/'deployment.json');old=read(args.labels212/'deployment.json')
    fr=sha(args.frozen/'manifest.json');lr=sha(args.labels/'manifest.json');oldsha=sha(args.labels212/'manifest.json')
    if oldsha!='46b3ab84c85053161925d09f2008af455c6a5c76e3916403dcc579b5dea12091':raise ValueError('wrong original18 K212 reference archive')
    if frozen['identity']['prereg']!=args.prereg or labels['prereg']!=args.prereg or labels['frozen_manifest_sha256']!=fr:raise ValueError('214 frozen/labels registration chain differs')
    if receipt['prereg']!=args.prereg or receipt['frozen_manifest_sha256']!=fr or receipt['label_manifest_sha256']!=lr:raise ValueError('blind common predictions do not bind exact214 inputs')
    if frozen['identity']['supplement']!=labels['supplement'] or receipt['supplement']!=labels['supplement']:raise ValueError('registered capacity supplement differs')
    roster={(p['domain'],p['clip'],p['member_index']):p for p in P['members']}
    if len(roster)!=18 or len(P['members'])!=18 or frozen['originalK']!=18:raise ValueError('all18 target members must be sealed before scoring')
    truthrefs={(r['domain'],r['clip'],r['member_index']):r for r in labels['original_K']}
    oldrefs={(r['domain'],r['clip'],r['member_index']):r for r in old['original_K']}
    if set(roster)!=set(truthrefs) or set(roster)!=set(oldrefs):raise ValueError('214 predictions/truth and212 original member scene rosters differ')
    GTpaths={};GTrecords={}
    def groups():
        for ref in frozen['clips']:
            clip=prepare214.load_clip(args.frozen,ref);members=[];domain=clip['domain'];name=clip['clip']
            for slot in (0,1,2):
                scene=prepare214.load_scene(clip,slot);m=scene['metadata'];p=roster[domain,name,slot]
                truthdata=read(args.labels/truthrefs[domain,name,slot]['path'])
                if str(p['member'])!=str(m['member']) or str(truthdata['member'])!=str(m['member']):raise ValueError('original global member identity differs')
                endpoints=scene['endpoints']
                if p['endpoint_ids']!=[e['endpoint_id'] for e in endpoints] or p['raw_indices']!=[e['raw_index'] for e in endpoints] or p['eligible']!=[eligible(e) for e in endpoints] or p['original_boxes']!=[e['box'] for e in endpoints]:raise ValueError('blind prediction row alignment/eligibility/original geometry differs')
                relative=truthdata['GT_relative_path']
                if relative not in labels['GT_sha256'] or truthdata['GT_sha256']!=labels['GT_sha256'][relative]:raise ValueError('GT source not in six prepared official whitelist')
                path=(args.repo/relative).resolve()
                if not path.is_relative_to(args.repo.resolve()) or sha(path)!=labels['GT_sha256'][relative]:raise ValueError('official GT bytes differ')
                if (domain,name) in GTpaths and GTpaths[domain,name]!=path:raise ValueError('member GT source differs')
                GTpaths[domain,name]=path;GTrecords[str(path)]=labels['GT_sha256'][relative]
                members.append(dict(member=str(m['member']),member_index=slot,seed=m['seed'],rows=scene['rows'],endpoints=endpoints,original_endpoint_truth=truthdata['records']))
            yield dict(domain=domain,clip=name,polygon=clip['polygon'],nf=clip['nf'],members=members)
    def decisions(group,member):
        p=roster[group['domain'],group['clip'],member['member_index']]
        if set(p['shifts'])!=set(('K','G','I','Z')):raise ValueError('four sealed blind arm arrays required')
        return {a:np.load(args.predictions/file,allow_pickle=False) for a,file in p['shifts'].items()}
    def truth_provider(group):
        from beetrack.evaluate import load_tracks
        return load_tracks(GTpaths[group['domain'],group['clip']])
    def references(group,member):
        ref=oldrefs[group['domain'],group['clip'],member['member_index']]
        if str(ref['member'])!=member['member']:raise ValueError('212 original K member differs')
        return {t:read(args.labels212/file) for t,file in ref['graphs'].items()}
    inputs=dict(prereg=args.prereg,supplement=labels['supplement'],frozen_manifest_sha256=fr,label_manifest_sha256=lr,
        predictions_manifest_sha256=predsha,original212_labels_manifest_sha256=oldsha,all_four_folds_18_members_common_seal_verified=True,
        GT_sha256=labels['GT_sha256'])
    def final_validation():
        if any(sha(path)!=h for path,h in GTrecords.items()):raise ValueError('GT changed during scoring')
    summary=evaluate_explicit(groups(),decisions,truth_provider,references,args.out,inputs,final_validation)
    print('CENTER214_SCORE_SEALED',sha(args.out/'manifest.json'),flush=True)
    return summary


def self_test():
    rows=np.array([[1,1,-2,0,2,2,.8],[2,1,-1,0,2,2,.7],[3,1,0,0,2,2,.9]],float)
    endpoints=[dict(endpoint_id='first',raw_index=0),dict(endpoint_id='last',raw_index=2)]
    fixed,proof=materialize(rows,endpoints,[[.25,0],[-.25,0]])
    assert fixed[0,2]==-1.5 and fixed[2,2]==-.5 and np.array_equal(rows[1],fixed[1])
    assert np.array_equal(rows[:,[0,1,4,5,6]],fixed[:,[0,1,4,5,6]])
    try:materialize(rows,[dict(endpoint_id='wrong',raw_index=1)],[[0,0]])
    except ValueError:pass
    else:raise AssertionError('nonendpoint accepted')
    G=rows.copy();G[:,2]+=.5;owner=np.arange(3);o=oracle_offsets(rows,endpoints,G,owner);assert np.allclose(o[:,0],.25)
    from tools.study195.run import score_member_native
    # Fresh fullscene scorer on a tiny artificial entrance; never field files.
    polygon=[(0,-5),(10,-5),(10,5),(0,5)]
    metric=score_member_native(G,rows,polygon,30);z=np.zeros((2,2));corrected=materialize(rows,endpoints,o)[0]
    gt,owners=original_owners(G,rows,polygon)
    d=endpoint_diagnostics(rows,corrected,endpoints,gt,owners,polygon)
    assert d['matched']==2 and d['after_boundary_MAE']==0 and d['before_boundary_MAE']>0
    e=ensemble([metric,metric,metric]);assert e['numerator']==3*metric_counts(metric)['numerator']
    assert e['E3']==metric_counts(metric)['W'] and e['mean_member_F1']['0.5']==metric['iou']['0.5']['f1']
    assert event_changes(rows,rows,polygon)['counts']=={'unchanged':1}
    with tempfile.TemporaryDirectory(prefix='synthetic214-cache-') as temp:
        cache=NativeCache(Path(temp)/'cache',dict(synthetic=True));cache.scorer('35','toy',0)(G,rows,polygon,30)
        cache.scorer('35','toy',0)(G,rows,polygon,30)
        assert cache.fresh==1 and cache.reused==1
        try:cache.scorer('35','toy',0)(G,corrected,polygon,30)
        except ValueError:pass
        else:raise AssertionError('resume accepted changed actual boxes')
    return dict(passed=True,scope='synthetic endpoint invariants/oracle, actual fullscene official metric, signed-boundary MAE, three-member integer aggregation, unchanged event')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path);p.add_argument('--self-test',action='store_true')
    for arg in ('frozen','labels','labels212','predictions','out'):p.add_argument('--'+arg,type=Path)
    p.add_argument('--prereg');args=p.parse_args()
    if args.repo:sys.path.insert(0,str(args.repo.resolve()))
    if args.self_test:print(json.dumps(self_test()))
    else:
        if any(getattr(args,k) is None for k in ('repo','frozen','labels','labels212','predictions','out','prereg')):p.error('explicit sealed roots/repo/prereg/new out required')
        run(args)
