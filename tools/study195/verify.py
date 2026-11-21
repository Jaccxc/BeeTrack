"""Independent whole-member threshold router verifier. Synthetic-only by default.

No production candidate/router selection or scoring helper is imported.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import itertools
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(1)


def events_from_native(rows,polygon):
    """Original ID gap runs; boundary included; last matching-direction crossing."""
    rows=np.asarray(rows,float);poly=np.asarray(polygon,np.float32);events={}
    for native_id in sorted(set(rows[:,1])):
        track=rows[rows[:,1]==native_id];track=track[np.argsort(track[:,0],kind='stable')]
        cuts=[0,*list(np.flatnonzero(np.diff(track[:,0])>25)+1),len(track)]
        for run,(a,b) in enumerate(zip(cuts,cuts[1:])):
            r=track[a:b];sides=[]
            for x in r:sides.append(cv2.pointPolygonTest(poly,(float(x[2]+x[4]/2),float(x[3]+x[5]/2)),False)>=0)
            if sides[0]==sides[-1]:continue
            direction=1 if sides[-1] else -1
            dates=[int(r[j,0]) for j in range(1,len(r)) if sides[j]!=sides[j-1] and sides[j]==sides[-1]]
            identity=int(native_id+100000*run)
            assert identity not in events,'official gap alias collision'
            events[identity]=(dates[-1],direction)
    return events


def bins(events,nf):
    out=[]
    for offset in range(30):
        counts=[[0,0] for _ in range((nf+25*offset+749)//750)]
        for frame,direction in events:counts[(int(frame)-1+25*offset)//750][int(direction<0)]+=1
        out.extend(counts)
    return np.asarray(out,np.int64)


def maximum_matching(kept,edges):
    owner={}
    def augment(p,seen):
        for g in sorted(edges.get(p,())):
            if g in seen:continue
            seen.add(g)
            if g not in owner or augment(owner[g],seen):owner[g]=p;return True
        return False
    for p in sorted(kept):augment(p,set())
    return len(owner)


def enumerate_joint(clips):
    """Each clip has gt_bins and 3x2 member candidates {bins,tp,pred,gt}.

    Every state is one physical whole-MOT choice per member per clip. Videos
    pool their loss numerator; member F1 pools video counts before averaging.
    """
    assert len(clips) in (1,2)
    denominator=3*sum(int(np.asarray(c['gt_bins']).sum()) for c in clips)
    assert denominator>0
    states=[]
    for bits in itertools.product((0,1),repeat=3*len(clips)):
        numerator=over=under=0;tp=[0]*3;pred=[0]*3;truth=[0]*3
        for vi,clip in enumerate(clips):
            selected=[clip['members'][m][bits[3*vi+m]] for m in range(3)]
            residual=sum(np.asarray(a['bins'],np.int64) for a in selected)-3*np.asarray(clip['gt_bins'],np.int64)
            numerator+=int(np.abs(residual).sum());over+=int(np.maximum(residual,0).sum());under+=int(np.maximum(-residual,0).sum())
            for m,a in enumerate(selected):tp[m]+=a['tp'];pred[m]+=a['pred'];truth[m]+=a['gt']
        member_f1=[Fraction(2*t,max(1,p+g)) for t,p,g in zip(tp,pred,truth)]
        mean=sum(member_f1,Fraction())/3
        states.append(dict(bits=''.join(map(str,bits)),numerator=numerator,denominator=denominator,E3=100*numerator/denominator,
            over=100*over/denominator,under=100*under/denominator,tp=tp,pred=pred,gt=truth,
            member_f1=[dict(numerator=f.numerator,denominator=f.denominator) for f in member_f1],
            mean_f1=dict(numerator=mean.numerator,denominator=mean.denominator,value=float(mean))))
    # Keep count and F1 tied to the SAME state, including count-optimum ties.
    best=min(s['numerator'] for s in states)
    count_optimal=[s['bits'] for s in states if s['numerator']==best]
    pareto=[]
    for s in states:
        fs=Fraction(s['mean_f1']['numerator'],s['mean_f1']['denominator'])
        if not any(t['numerator']<=s['numerator'] and Fraction(t['mean_f1']['numerator'],t['mean_f1']['denominator'])>=fs and
                   (t['numerator']<s['numerator'] or Fraction(t['mean_f1']['numerator'],t['mean_f1']['denominator'])>fs) for t in states):pareto.append(s['bits'])
    return dict(states=states,count_optimal=count_optimal,pareto=pareto)


def router_features(six):
    """Protocol seven features; six has shape (member, arm, window, direction)."""
    six=np.asarray(six,float)
    assert six.shape[:2]==(3,2) and six.shape[-1]==2
    flat=six.reshape(6,*six.shape[2:])
    centre=np.median(flat,axis=0)
    scale=max(90.,3*float(centre.sum()))
    wide=np.ptp(flat,axis=0)>=2
    features=[]
    for bits in itertools.product((0,1),repeat=3):
        deviations=np.stack([six[i,bits[i]]-centre for i in range(3)])
        total=deviations.sum(axis=0)
        features.append([abs(total).sum()/scale,abs(deviations).sum()/scale,
                         total[:,0].sum()/scale,total[:,1].sum()/scale,
                         abs(total)[wide].sum()/scale,abs(deviations[:,wide]).sum()/scale,
                         sum(bits)/3])
    return np.asarray(features),scale


def independent_fit(records,heldout,kind):
    """Records carry domain/group/video, features[8,7], B, integer losses[8]."""
    source=[r for r in records if r['domain']!=heldout]
    domains=sorted({r['domain'] for r in source})
    assert len(domains)==3
    columns=[0,1,2,3,4,5,6] if kind=='J' else [1,2,3,5,6]
    prior=0 if kind=='J' else 1
    xs=[];ys=[];weights=[]
    for domain in domains:
        groups=sorted({r['group'] for r in source if r['domain']==domain})
        for group in groups:
            videos=[r for r in source if r['domain']==domain and r['group']==group]
            denom=sum(r['B'] for r in videos)
            for r in videos:
                x=np.asarray(r['features'],float)
                centered=x-x.mean(axis=0)
                losses=np.asarray(r['losses'],float)
                target=(losses-losses.mean())/r['B']-centered[:,prior]
                xs.extend(centered[:,columns]);ys.extend(target)
                weights.extend([r['B']/denom/len(groups)/len(domains)/8]*8)
    x=np.asarray(xs);y=np.asarray(ys);w=np.asarray(weights)
    assert abs(w.sum()-1)<1e-12
    rms=np.sqrt((w[:,None]*x*x).sum(axis=0));rms[rms==0]=1
    z=x/rms
    coef=np.linalg.solve(z.T@(w[:,None]*z)+.01*np.eye(len(columns)),z.T@(w*y))
    return dict(columns=columns,prior=prior,rms=rms,coef=coef,weights=w,
                gradient=2*(z.T@(w*(z@coef-y))+.01*coef))


def guarded_configuration(group_results,reference_f1,guard=Fraction(1,200)):
    """Exact cross-group Pareto dynamic program; returns actual physical states."""
    threshold=reference_f1-guard
    frontier=[(Fraction(0),Fraction(0),())]
    for group in group_results:
        # Equal count errors share only their maximum feasible F1 representative.
        options={}
        for state in group['states']:
            e=Fraction(100*state['numerator'],state['denominator'])
            f=Fraction(state['mean_f1']['numerator'],state['mean_f1']['denominator'])
            if e not in options or f>options[e][0]:options[e]=(f,state['bits'])
        expanded={}
        for e,f,path in frontier:
            for de,(df,bits) in options.items():
                ne=e+de;nf=f+df
                if ne not in expanded or nf>expanded[ne][0]:expanded[ne]=(nf,path+(bits,))
        frontier=[];best=None
        for e,(f,path) in sorted(expanded.items()):
            if best is None or f>best:frontier.append((e,f,path));best=f
    n=len(group_results)
    feasible=[(e/n,f/n,path) for e,f,path in frontier if f/n>=threshold]
    if not feasible:return None
    e,f,path=min(feasible,key=lambda v:(v[0],-v[1],v[2]))
    return dict(E3=float(e),mean_F1=float(f),states=list(path),
                exact_E3=[e.numerator,e.denominator],exact_F1=[f.numerator,f.denominator])


def synthetic():
    truth=np.tile([1,0],(30,1))
    low=dict(bins=np.zeros_like(truth),tp=0,pred=0,gt=1)
    high=dict(bins=2*truth,tp=1,pred=2,gt=1)
    one=dict(gt_bins=truth,members=[[low,high] for _ in range(3)])
    a=enumerate_joint([one]);b=enumerate_joint([one,one])
    assert len(a['states'])==8 and len(b['states'])==64
    assert set(a['count_optimal'])=={'001','010','100','011','101','110'}
    # Count and F1 optima cannot be combined into a nonexistent state.
    assert max(s['mean_f1']['value'] for s in a['states'] if s['bits'] in a['count_optimal'])==float(Fraction(4,9))
    assert next(s for s in a['states'] if s['bits']=='111')['mean_f1']['value']==float(Fraction(2,3))
    poly=[[0,0],[10,0],[10,10],[0,10]]
    rows=np.array([[1,7,12,2,2,2],[2,7,1,2,2,2],[40,7,1,2,2,2],[41,7,12,2,2,2]],float)
    events=events_from_native(rows,poly)
    assert events=={7:(2,1),100007:(41,-1)} and bins(events.values(),60).sum()==60
    assert maximum_matching([1,2],{1:[8],2:[8,9]})==2
    guarded=guarded_configuration([a,a],Fraction(2,3),Fraction(0))
    assert guarded['states']==['111','111'] and guarded['mean_F1']==float(Fraction(2,3))
    x,B=router_features(np.zeros((3,2,5,2)))
    assert B==90 and np.all(x[:,:6]==0) and x[-1,6]==1
    rng=np.random.default_rng(195)
    records=[]
    for domain in ('33','35','W','G'):
        for group in range(2):
            for video in range(1 if domain in ('33','35') else 2):
                x,B=router_features(rng.integers(0,5,(3,2,6,2)))
                records.append(dict(domain=domain,group=group,video=video,features=x,B=B,losses=rng.integers(0,100,8)))
    for kind in ('J','D'):
        fitted=independent_fit(records,'33',kind)
        assert np.max(abs(fitted['gradient']))<1e-12
        # A held-out label cannot affect normalization or coefficients.
        for r in records:
            if r['domain']=='33':r['losses'][:]=99999
        changed=independent_fit(records,'33',kind)
        assert np.array_equal(fitted['coef'],changed['coef'])
    return dict(native_gap_crossing=True,eight_states=True,sixtyfour_states=True,joint_count_F1=True,
                matching=True,cross_group_guard=True,seven_features=True,source_only_ridge=True,passed=True)


def supplementary_checks(root,report):
    """Exact replay from the member counts/448 states already raw-verified above."""
    import hashlib
    summary=json.loads((root/'score/summary.json').read_text())
    tie_checks=0
    for h,head in summary['hives'].items():
        for method,groups in head['methods'].items():
            exact=[]
            for group in groups:
                members=group['iou']['0.3']['members']
                exact.append(sum((Fraction(2*m['tp'],max(1,m['pred']+m['gt'])) for m in members),Fraction())/3)
            assert abs(float(sum(exact)/len(exact))-head['means'][method]['F1_03'])<1e-14
        for index,g in enumerate(head['groups']):
            for method in ('U','O'):
                pool=[c for c in g['candidates'] if method=='O' or all(len(set(bits))==1 for bits in c['states'])]
                minimum=min(c['numerator'] for c in pool);ties=[c for c in pool if c['numerator']==minimum]
                chosen=head['methods'][method][index];assert chosen['tie_count']==len(ties)
                for t in ('0.5','0.3'):
                    fs=[sum((Fraction(2*m['tp'],max(1,m['pred']+m['gt'])) for m in c['iou'][t]['members']),Fraction())/3 for c in ties]
                    assert chosen['tie_F1_ranges'][t]==[float(min(fs)),float(max(fs))]
                tie_checks+=1
    prior=root.parent/'budget193/score/summary193.json'
    old=json.loads(prior.read_text());references={}
    for h,head in summary['hives'].items():
        low=head['means']['L'];base=old['heads'][h]['B']
        assert abs(low['E3']-base['E3'])<1e-12 and abs(low['F1_05']-base['mean_single_f1'])<1e-14
        references[h]=dict(L_E3=low['E3'],H_E3=head['means']['H']['E3'],section193_B_exact=True)
    published_rounded={'35':(6.83,4.34),'33':(19.36,19.07),'W':(35.77,41.08),'G':(37.08,42.18)}
    for h,(l,r) in published_rounded.items():assert abs(references[h]['L_E3']-l)<.005 and abs(references[h]['H_E3']-r)<.005
    report['supplementary']=dict(all_dense_F1_means=True,oracle_tie_range_groups=tie_checks,baseline_references=references,
        section193_summary_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
        public_log_rounded_L_H_match=True,
        physical_guard_witness='Each saved guard bit selects one of the 120 native-identical materialized whole-MOT files; no concatenation or event splice.',
        chronology_scope='Seals and all source/target receipt roles are checked; historical process order is attested by producer sealed receipts, not independently observable retroactively.')
    return report


def field(repo,root):
    import hashlib,sys
    sys.path.insert(0,str(repo))
    from beetrack.evaluate import by_track,frame_match,match_events
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def read(p):return json.loads(Path(p).read_text())
    seals={}
    for directory in ('frozen','predictions','score'):
        base=root/directory;manifest=read(base/'manifest.json')
        digest=sha(base/'manifest.json');assert (base/'SEALED.sha256').read_text().strip()==digest
        for name,want in manifest.items():
            path=(base/name).resolve();assert path.is_relative_to(base.resolve()) and sha(path)==want
        seals[directory]=digest
    assert seals['score']=='bb0b7412a73e616583362c95d036f440389ed98403617c53153bf68363ca6710'
    d=read(root/'frozen/deployment.json');published=read(root/'score/summary.json');native=read(root/'score/native_scores.json')
    assert d['prereg']==published['prereg']=='b7e85bb9679d771394a578c6962b4b870f15b5b9'
    assert len(d['units'])==60 and len(d['groups'])==20 and len(d['members'])==42
    for name,h in d['source_hashes']['linkers'].items():assert sha(repo/'beetrack/models'/f'{name}.json')==h
    zones=read(repo/'beetrack/zones.json');truth={};records=[];videos={};arm_metrics={};native_map={v['unit']:v for v in native};raws={}
    def tracks(rows,polygon):
        a=by_track(rows);poly=np.asarray(polygon,np.float32);answer={}
        for pid in sorted(set(a[:,1])):
            idx=np.flatnonzero(a[:,1]==pid);r=a[idx]
            side=np.array([cv2.pointPolygonTest(poly,(float(v[2]+v[4]/2),float(v[3]+v[5]/2)),False)>=0 for v in r])
            direction=0 if side[0]==side[-1] else (1 if side[-1] else -1)
            crosses=[(int(idx[j]),1 if side[j] else -1) for j in range(1,len(r)) if side[j]!=side[j-1]]
            answer[int(pid)]=(direction,idx,crosses)
        return a,answer
    for clip in d['clips']:
        name=clip['clip'];assert name in {'BEE24-35-head','BEE24-33-head','BEE24-10','BEE24-15','BEE24-24','BEE24-28'}
        assert clip['polygon']==zones[name]
        path=repo/'runs/gt'/clip['split']/name/'gt/gt.txt'
        gt=np.loadtxt(path,delimiter=',',usecols=range(6),ndmin=2)
        ev=events_from_native(gt,clip['polygon']);G,gtracks=tracks(gt,clip['polygon'])
        truth[name]=(gt,ev,bins(ev.values(),clip['nframes']),G,gtracks)
        proof=published['GT_entry'][name];assert proof['sha256']==sha(path)
    def score(gt,raw,poly,nf,G,gtracks):
        ev=events_from_native(raw,poly);P,ptracks=tracks(raw,poly);metrics={};edgeproof={}
        for threshold in (.5,.3):
            under=frame_match(G,P,thr=threshold);edges={};same=set()
            for pid,(direction,idx,crosses) in ptracks.items():
                if not direction:continue
                u=under[idx];u=u[u>=0]
                if len(u) and u[0]==u[-1] and gtracks[int(u[0])][0]==direction:same.add(int(u[0]))
                for j,sign in crosses:
                    if sign!=direction:continue
                    for owner in {int(under[j-1]),int(under[j])}-{-1}:
                        gd,_,gc=gtracks[owner]
                        if gd==direction and any(s==direction and abs(G[k,0]-P[j,0])<=5 for k,s in gc):edges.setdefault(pid,set()).add(owner)
            tp=maximum_matching(ev,edges)
            metrics[str(threshold)]=dict(tp=tp,pred=len(ev),gt=sum(bool(v[0]) for v in gtracks.values()),tp_id=len(same))
            edgeproof[str(threshold)]=sorted([p,g] for p,gs in edges.items() for g in gs)
        official=match_events(gt,raw,poly)
        assert all(official[k]==metrics['0.5'][k] for k in ('tp','gt','pred','tp_id'))
        return ev,bins(ev.values(),nf),metrics,edgeproof
    for group in sorted(d['groups'],key=lambda g:({'35':0,'33':1,'W':2,'G':3}[g['hive']],g['group'],d['clips'][g['clip_index']]['clip'])):
        clip=d['clips'][group['clip_index']];name=clip['clip'];h=group['hive'];gid=group['group'];unit=f'{h}_g{gid}_{name}'
        gt,gev,gb,G,gs=truth[name];six=[];members=[];threshold_members={'.5':[],'.3':[]}
        for uid in group['units']:
            u=d['units'][uid];assert u['seed'] in (range(3,15) if h in ('35','33') else range(9))
            assert u['seed']==(gid-1)*3+len(members)
            these=[];per_threshold={'.5':[],'.3':[]};bc=[]
            for arm in ('t40','t50'):
                entry=u['arms'][arm];path=root/'frozen'/entry['rows_file'];assert sha(path)==entry['rows_sha256']==entry['source_receipt']['cache_sha256']
                linker='m_no'+h if h in ('W','G') else 'm_dev_strict';assert entry['source_receipt']['linker_name']==linker
                assert sha(repo/'runs/dets'/u['tag']/u['split']/f'{name}.npy')==entry['source_receipt']['detector_sha256']
                with np.load(path,allow_pickle=False) as z:raw=z['rows']
                raws[(unit,str(u['member']),arm)]=raw
                ev,pb,met,edges=score(gt,raw,clip['polygon'],clip['nframes'],G,gs)
                archived=native_map[unit]['members'][len(members)]['arms'][arm]
                assert np.array_equal(pb,entry['windows']) and np.array_equal(pb,archived['pred_bins']) and np.array_equal(gb,archived['gt_bins'])
                assert {int(e['pid']):(int(e['frame']),int(e['direction'])) for e in entry['event_records']}==ev
                assert sorted([p,f*s] for p,(f,s) in ev.items())==sorted([int(e[0]),int(e[1])] for e in archived['evidence']['predictions'])
                for threshold,key in (('0.5','edges05'),('0.3','edges03')):
                    assert edges[threshold]==archived['evidence'][key]
                    assert all(met[threshold][k]==archived['iou'][threshold][k] for k in ('tp','pred','gt','tp_id'))
                    per_threshold['.'+threshold[-1]].append(dict(bins=pb,**met[threshold]))
                bc.append(pb)
            six.append(bc);members.append(per_threshold['.5'])
            for t in threshold_members:threshold_members[t].append(per_threshold[t])
        x,B=router_features(six);losses=[int(abs(np.asarray(six)[np.arange(3),bits].sum(axis=0)-3*gb).sum()) for bits in itertools.product((0,1),repeat=3)]
        records.append(dict(domain=h,group=gid,video=name,features=x,B=B,losses=losses,unit=unit))
        videos[unit]=dict(domain=h,group=gid,video=name,gt_bins=gb,members=members,other_members=threshold_members['.3'])
        print('VERIFIED_NATIVE '+unit,flush=True)
    for r in read(root/'score/physical_receipts.json'):
        actual=np.loadtxt(root/'score'/r['file'],delimiter=',',usecols=range(7),ndmin=2)
        assert np.array_equal(actual,raws[(r['group'],str(r['member']),r['arm'])])
    choices={};training={}
    for h in ('35','33','W','G'):
        fold=read(root/'predictions'/f'{h}.json');model=fold['model'];assert set(fold['source_domains'])==set(('35','33','W','G'))-{h}
        assert model['heldout']==h and set(model['training_sources'])==set(fold['source_domains'])
        allowed={r['video'] for r in records if r['domain']!=h};assert set(fold['source_GT_receipts'])==allowed
        for video,p in fold['source_GT_receipts'].items():assert p['domain']!=h and sha(repo/p['path'])==p['sha256']
        for fn,digest in fold['code_sha256'].items():assert sha(Path(__file__).parent/fn)==digest
        fit={kind:independent_fit(records,h,kind) for kind in ('D','J')}
        for kind,f in fit.items():
            np.testing.assert_allclose(f['coef'],model[kind]['beta'],atol=1e-11,rtol=1e-11);np.testing.assert_allclose(f['rms'],model[kind]['scale'],atol=1e-14)
        for a in model['training_records']:
            r=next(r for r in records if (r['domain'],r['group'],r['video'])==(a['hive'],a['group_id'],a['clip']))
            np.testing.assert_allclose(r['features'],a['raw_z'],atol=1e-14);assert r['losses']==a['true_cost'] and r['B']==a['B']
        for c in fold['choices']:
            r=next(r for r in records if r['unit']==c['unit']);x=r['features'];cx=x-x.mean(axis=0)
            scores={'M':x[:,0]}
            for kind,f in fit.items():scores[kind]=cx[:,f['prior']]+cx[:,f['columns']]/f['rms']@f['coef']
            picks={}
            for method,values in scores.items():
                np.testing.assert_allclose(values,c['prediction'][method]['scores'],atol=1e-11,rtol=1e-11)
                ties=np.flatnonzero(values<=values.min()+1e-12);pick=min(ties,key=lambda k:(bin(k).count('1'),k));bits=format(pick,'03b')
                assert list(map(int,bits))==c['prediction'][method]['bits'];picks[method]=bits
            choices[c['unit']]=picks
        training[h]=dict(source_videos=sorted(allowed),records=len(model['training_records']),coefficient_max_error={k:float(np.max(abs(fit[k]['coef']-model[k]['beta']))) for k in fit})
    report={};total=0
    for h in ('35','33','W','G'):
        pub=published['hives'][h];results=[];means={name:[] for name in pub['methods']};exact_means={name:[] for name in pub['methods']}
        for gi,gid in enumerate(sorted({v['group'] for v in videos.values() if v['domain']==h})):
            clips=sorted([v for v in videos.values() if v['domain']==h and v['group']==gid],key=lambda c:c['video'])
            enumerated=enumerate_joint(clips);dense=enumerate_joint([dict(c,members=c['other_members']) for c in clips]);results.append(enumerated);total+=len(enumerated['states'])
            for state,p,ds in zip(enumerated['states'],pub['groups'][gi]['candidates'],dense['states']):
                assert state['bits']==''.join(str(b) for route in p['states'] for b in route)
                assert (state['numerator'],state['denominator'])==(p['numerator'],p['denominator'])
                for st,t in ((state,'0.5'),(ds,'0.3')):
                    assert st['tp']==[m['tp'] for m in p['iou'][t]['members']]
                    assert abs(st['mean_f1']['value']-p['iou'][t]['mean_member_f1'])<1e-14
            for method,arr in pub['methods'].items():
                p=arr[gi];bits=''.join(str(b) for route in p['states'] for b in route);st=next(s for s in enumerated['states'] if s['bits']==bits)
                if method in ('M','D','J'):assert bits==''.join(choices[next(r['unit'] for r in records if r['domain']==h and r['group']==gid and r['video']==c['video'])][method] for c in clips)
                elif method in ('L','H','F'):assert bits==('1' if method=='H' or method=='F' and h in ('33','35') else '0')*len(bits)
                elif method=='O':assert st['numerator']==min(s['numerator'] for s in enumerated['states'])
                elif method=='U':assert st['numerator']==min(s['numerator'] for s in enumerated['states'] if all(len(set(s['bits'][j:j+3]))==1 for j in range(0,len(bits),3)))
                assert st['numerator']==p['numerator'];means[method].append(Fraction(100*st['numerator'],st['denominator']));exact_means[method].append(Fraction(st['mean_f1']['numerator'],st['mean_f1']['denominator']))
        mean={k:dict(E3=float(sum(v)/len(v)),F1=float(sum(exact_means[k])/len(v))) for k,v in means.items()}
        for k,v in mean.items():assert abs(v['E3']-pub['means'][k]['E3'])<1e-12 and abs(v['F1']-pub['means'][k]['F1_05'])<1e-14
        ref=sum(exact_means['F'])/len(results);guarded=guarded_configuration(results,ref)
        assert abs(guarded['E3']-pub['guarded_oracle']['mean_E3'])<1e-12 and abs(guarded['mean_F1']-pub['guarded_oracle']['mean_F1'])<1e-14
        report[h]=dict(means=mean,guarded=guarded,exact_J_less_D=sum(means['J'])<sum(means['D']),groups=len(results))
    assert total==448
    gain={h:report[h]['means']['F']['E3']-report[h]['means']['J']['E3'] for h in report}
    investment=gain['35']>1.4 and gain['33']>1.3 and gain['W']>=-1.4 and all(report[h]['means']['J']['F1']>=report[h]['means']['F']['F1']-.005-1e-15 for h in ('35','33','W'))
    joint=all(report[h]['exact_J_less_D'] for h in ('35','33'))
    assert investment==published['investment_pass'] and joint==published['joint_better_than_D_both_heads']
    out=dict(passed=True,seals=seals,member_arms=120,physical_MOT_roundtrips=120,joint_states=total,fold_training=training,hives=report,investment_pass=investment,joint_strict_both=joint,
             independently_rebuilt=['native gap/event/endpoints','30-phase integer bins','crossing graph at .5/.3 and own augmenting TP','official match_events all120','all448jointstates','source-only features/weights/RMS/ridge/predictions','exact cross-group F1 guard physical choices'],
             shared_primitive=['official by_track/frame_match/match_events; production router/model/scorer not imported'],limitations=['Original detector/tracker inference not rerun; frozen full native NPZ and original detector hashes verified.'])
    out=supplementary_checks(root,out)
    out['verifier_sha256']=sha(Path(__file__))
    dest=root/'independent.json';dest.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print('INDEPENDENT_PASS '+str(dest)+' '+sha(dest),flush=True)
    return out


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path);p.add_argument('--root',type=Path);a=p.parse_args()
    if a.root:field(a.repo.resolve(),a.root.resolve())
    else:print(json.dumps(synthetic(),indent=2))
