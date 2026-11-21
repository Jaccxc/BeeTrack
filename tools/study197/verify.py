"""Independent endpoint-cut audit. Default synthetic, no field IO at import."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(1)


def canonicalize(raw):
    """Independent official gap segments, with an exact native-row bijection."""
    raw=np.asarray(raw,float);positions=[];identities=[];aliases=set()
    for native in sorted(set(raw[:,1])):
        idx=np.flatnonzero(raw[:,1]==native)
        idx=idx[np.argsort(raw[idx,0],kind='stable')]
        cuts=[0,*list(np.flatnonzero(np.diff(raw[idx,0])>25)+1),len(idx)]
        for run,(lo,hi) in enumerate(zip(cuts,cuts[1:])):
            pid=int(native+100000*run)
            assert pid not in aliases;aliases.add(pid)
            positions.extend(idx[lo:hi]);identities.extend([pid]*(hi-lo))
    positions=np.asarray(positions,int);ids=np.asarray(identities,int)
    order=np.lexsort((positions,raw[positions,0],ids));positions=positions[order];ids=ids[order]
    canonical=raw[positions].copy();canonical[:,1]=ids
    blocks=[]
    for rank,pid in enumerate(sorted(aliases)):
        rows=np.flatnonzero(ids==pid)
        blocks.append(dict(rank=rank,parent_id=pid,start=int(rows[0]),end=int(rows[-1]+1)))
    assert np.array_equal(np.sort(positions),np.arange(len(raw)))
    return canonical,positions,blocks


def realize_independent(raw,canonical,positions,blocks,candidates,selected):
    selected=set(selected);byid={c['candidate_id']:c for c in candidates}
    assert selected<=set(byid)
    parent_cuts={}
    for cid in selected:
        c=byid[cid];assert c['parent_id'] not in parent_cuts
        parent_cuts[c['parent_id']]=c
    result=np.asarray(raw,float).copy()
    for block in blocks:
        lo,hi=block['start'],block['end'];base=2*block['rank']+1
        result[positions[lo:hi],1]=base
        cut=parent_cuts.get(block['parent_id'])
        if cut is not None:
            after=int(cut['cut_after_canonical_row']);assert lo<=after<hi-1
            result[positions[after+1:hi],1]=base+1
    cols=[i for i in range(result.shape[1]) if i!=1]
    assert np.array_equal(result[:,cols],np.asarray(raw)[:,cols])
    return result


def event_list(rows,polygon):
    canonical,_,blocks=canonicalize(rows);poly=np.asarray(polygon,np.float32);events=[]
    for block in blocks:
        segment=canonical[block['start']:block['end']]
        sides=[cv2.pointPolygonTest(poly,(float(r[2]+r[4]/2),float(r[3]+r[5]/2)),False)>=0 for r in segment]
        if sides[0]==sides[-1]:continue
        sign=1 if sides[-1] else -1
        crossing=[int(segment[i,0]) for i in range(1,len(segment)) if sides[i]!=sides[i-1] and sides[i]==sides[-1]]
        events.append((crossing[-1],sign))
    return events


def windows(events,nf):
    out=[]
    for offset in range(30):
        bins=[[0,0] for _ in range((nf+25*offset+749)//750)]
        for frame,sign in events:bins[(frame-1+25*offset)//750][int(sign<0)]+=1
        out.extend(bins)
    return np.asarray(out,np.int64)


def check_candidate_effect(raw,polygon,nf,candidate):
    canonical,positions,blocks=canonicalize(raw)
    baseline=realize_independent(raw,canonical,positions,blocks,[candidate],[])
    changed=realize_independent(raw,canonical,positions,blocks,[candidate],[candidate['candidate_id']])
    before=windows(event_list(baseline,polygon),nf);after=windows(event_list(changed,polygon),nf)
    expected=np.zeros_like(before)
    expected+=int(candidate['delta'])*windows([(candidate['frame'],candidate['direction'])],nf)
    assert np.array_equal(after-before,expected)
    assert int(expected.sum())==30*int(candidate['delta'])
    return baseline,changed,expected


def rebuild_candidates(raw,polygon,nf):
    """Rebuild the entire allowed option family and fixed features from rows."""
    from collections import Counter
    canonical,positions,blocks=canonicalize(raw);poly=np.asarray(polygon,np.float32)
    result=[]
    for block in blocks:
        start,end=block['start'],block['end'];parent=canonical[start:end]
        options=[]
        for endpoint in ('start','end'):
            order=list(range(start,end)) if endpoint=='start' else list(range(end-1,start-1,-1))
            for k in (1,2):
                if len(order)<k+3:continue
                idx=order[:k+3];w=canonical[idx]
                if np.any(w[:,4:6]<=0):continue
                span=int(abs(w[-1,0]-w[0,0]))
                if span>12:continue
                signed=np.array([cv2.pointPolygonTest(poly,(float(r[2]+r[4]/2),float(r[3]+r[5]/2)),True) for r in w])
                inside=signed>=0
                if not all(inside[:k]==inside[0]) or not all(inside[k:]!=inside[0]):continue
                cut=start+k-1 if endpoint=='start' else end-k-1
                options.append((k,span,endpoint!='start',endpoint,idx,cut,signed,inside))
        if not options:continue
        k,span,is_end,endpoint,idx,cut,signed,inside=min(options,key=lambda o:o[:3])
        w=canonical[idx];scale=float(sum(np.sqrt(r[4]*r[5]) for r in w)/len(w));d=signed/scale
        tau=abs(w[:,0]-w[0,0]);x=tau[k:];y=d[k:]
        slope=float(sum((x-x.mean())*(y-y.mean()))/sum((x-x.mean())**2));intercept=float(y.mean()-slope*x.mean())
        prediction=intercept+slope*tau[:k];fit=intercept+slope*x
        original=event_list(parent,polygon)
        G=[float(k-1),span/12,float(is_end),float(inside[0]),float(bool(original)),
           float(w[:k,6].mean()-w[k:,6].mean()),float(np.log1p(len(parent))/np.log(51)),
           float(np.clip(abs(d[:k]).mean(),-4,4)),float(np.clip(abs(d[k:]).mean(),-4,4))]
        extra=np.clip([abs(d[:k]-prediction).mean(),(-1 if inside[0] else 1)*prediction.mean(),abs(y-fit).mean()],-4,4).tolist()
        cid=f"cut:{block['parent_id']}:{endpoint}:{k}"
        c=dict(candidate_id=cid,parent_id=block['parent_id'],end=endpoint,k=k,span=span,
               cut_after_canonical_row=cut,window_indices=idx,features_G=G,features_T=G+extra,
               H=bool(np.all((prediction>=0)==bool(not inside[0]))),d_normalized=d.tolist(),tau=tau.tolist(),
               core_fit_ab=[intercept,slope],pred_short=prediction.tolist())
        physical=parent.copy();physical[:cut-start+1,1]=1;physical[cut-start+1:,1]=2
        new=event_list(physical,polygon)
        old=Counter(original);newcounter=Counter(new);added=list((newcounter-old).elements());dropped=list((old-newcounter).elements())
        assert len(added)+len(dropped)==1
        frame,direction=(added or dropped)[0];c.update(frame=frame,direction=direction,delta=1 if added else -1)
        delta=windows(new,nf)-windows(original,nf)
        assert np.array_equal(delta,c['delta']*windows([(frame,direction)],nf))
        c['delta_windows']=delta
        result.append(c)
    return dict(canonical_rows=canonical,source_raw_index=positions,parent_blocks=blocks,candidates=result,
                baseline_windows=windows(event_list(raw,polygon),nf))


def oracle_columns(candidates,nf):
    shape=windows([],nf).shape
    return np.column_stack([(int(c['delta'])*windows([(int(c['frame']),int(c['direction']))],nf)).ravel() for c in candidates]) if candidates else np.zeros((np.prod(shape),0),np.int64)


def verify_oracle_exact(pred_sum,gt_bins,candidates,nf,answer):
    """Integer weak-duality proof and an actual whole-column binary witness."""
    residual=(np.asarray(pred_sum,np.int64)-3*np.asarray(gt_bins,np.int64)).ravel()
    matrix=oracle_columns(candidates,nf);proof=answer['certificate'];p=proof['p'];q=proof['q']
    assert type(q) is int and q>0 and len(p)==len(residual)
    assert all(type(v) is int and abs(v)<=q for v in p)
    linear=sum(int(r)*v for r,v in zip(residual,p))
    dots=[sum(int(v)*w for v,w in zip(matrix[:,j],p)) for j in range(matrix.shape[1])]
    negative=sum(min(v,0) for v in dots);scaled=linear+negative;lower=-((-scaled)//q)
    assert proof['linear_term']==linear and proof['column_dot_products']==dots
    assert proof['negative_part_sum']==negative and proof['scaled_numerator']==scaled and proof['lower_bound']==lower
    x=answer['x'];assert len(x)==len(candidates) and all(type(v) is int and v in (0,1) for v in x)
    assert answer['selected_ids']==[c['candidate_id'] for c,bit in zip(candidates,x) if bit]
    witness=np.asarray(pred_sum,np.int64).ravel()+matrix@np.asarray(x,np.int64)
    assert np.all(witness>=0) and np.array_equal(witness.reshape(np.asarray(pred_sum).shape),answer['witness_pred_sum'])
    upper=sum(abs(int(v)) for v in witness-3*np.asarray(gt_bins,np.int64).ravel())
    certified=max(0,lower)
    if answer.get('enumeration') is not None:
        assert len(candidates)<=18
        histogram={}
        for bits in itertools.product((0,1),repeat=len(candidates)):
            cost=sum(abs(int(v)) for v in residual+matrix@np.asarray(bits,np.int64))
            histogram[cost]=histogram.get(cost,0)+1
        saved={int(k):v for k,v in answer['enumeration']['objective_histogram'].items()}
        assert histogram==saved and answer['enumeration']['configurations']==2**len(candidates)
        certified=min(histogram)
    assert (answer['lower_bound'],answer['upper_bound'],answer['gap'],answer['optimal'])==(certified,upper,upper-certified,upper==certified)
    assert answer['denominator']==3*int(np.asarray(gt_bins).sum())
    return dict(lower=certified,upper=upper,optimal=upper==certified)


def independent_ridge(records,heldout,feature_key):
    source=[r for r in records if r['domain']!=heldout]
    dimension=9 if feature_key=='features_G' else 12
    if not source:return dict(mean=np.zeros(dimension),scale=np.ones(dimension),coef=np.zeros(dimension),intercept=0.,empty=True)
    domains=sorted({r['domain'] for r in source});weight=[]
    for r in source:
        groups={a['group'] for a in source if a['domain']==r['domain']}
        clips={a['clip'] for a in source if (a['domain'],a['group'])==(r['domain'],r['group'])}
        size=sum((a['domain'],a['group'],a['clip'])==(r['domain'],r['group'],r['clip']) for a in source)
        weight.append(1/len(domains)/len(groups)/len(clips)/size)
    weight=np.asarray(weight);assert abs(weight.sum()-1)<1e-12
    x=np.asarray([r[feature_key] for r in source],float);y=np.array([r['y'] for r in source],float)
    mean=(weight[:,None]*x).sum(axis=0);constant=np.all(x==x[0],axis=0);mean[constant]=x[0,constant]
    scale=np.sqrt((weight[:,None]*(x-mean)**2).sum(axis=0));scale[(scale==0)|constant]=1
    z=(x-mean)/scale;intercept=float(weight@y)
    coef=np.linalg.solve(z.T@(weight[:,None]*z)+.2*np.eye(dimension),z.T@(weight*(y-intercept)))
    if np.all(y==y[0]):coef=np.zeros(dimension);intercept=float(y[0])
    gradient=2*(z.T@(weight*(z@coef+intercept-y))+.2*coef)
    assert np.max(abs(gradient),initial=0)<1e-10
    return dict(mean=mean,scale=scale,coef=coef,intercept=intercept,empty=False,weights=weight)


def hash_matched_mask(candidates,T_ids,unit,member_index):
    T_ids=set(T_ids);pools={};counts={}
    for c in candidates:
        key=(c['delta'],c['direction']);pools.setdefault(key,[]).append(c)
        counts[key]=counts.get(key,0)+int(c['candidate_id'] in T_ids)
    selected=[]
    for key,pool in pools.items():
        def rank(c):
            text=f"endpoint-null-v1|{unit}|{member_index}|{c['parent_id']}"
            return hashlib.sha256(text.encode('utf-8')).digest(),c['parent_id']
        selected.extend(c['candidate_id'] for c in sorted(pool,key=rank)[:counts[key]])
    return sorted(selected)


def track_structure(raw,polygon):
    from beetrack.evaluate import by_track
    tagged=by_track(np.c_[raw,np.arange(len(raw))]);rows=tagged[:,:-1];indices=tagged[:,-1].astype(int)
    poly=np.asarray(polygon,np.float32)
    sides=np.array([cv2.pointPolygonTest(poly,(float(r[2]+r[4]/2),float(r[3]+r[5]/2)),False)>=0 for r in rows])
    ids,starts=np.unique(rows[:,1],return_index=True);tracks={};events={}
    for pid,start,end in zip(ids,starts,[*starts[1:],len(rows)]):
        d=0 if sides[start]==sides[end-1] else (1 if sides[end-1] else -1)
        crosses=[(int(start+j+1),1 if sides[start+j+1] else -1) for j in np.flatnonzero(sides[start+1:end]!=sides[start:end-1])]
        tracks[int(pid)]=(d,int(start),int(end),crosses)
        if d:events[int(pid)]=(int(rows[next(j for j,sign in reversed(crosses) if sign==d),0]),d)
    return dict(rows=rows,index=indices,tracks=tracks,events=events,frame_order=indices[np.argsort(rows[:,0],kind='stable')])


def augmenting_tp(predictions,edges):
    occupied={}
    def visit(pid,seen):
        for target in sorted(edges.get(pid,())):
            if target in seen:continue
            seen.add(target)
            if target not in occupied or visit(occupied[target],seen):occupied[target]=pid;return True
        return False
    for pid in sorted(predictions):visit(pid,set())
    return len(occupied)


def score_from_owners(structure,truth,owners_raw,polygon,nf):
    P=structure['rows'];G=truth['rows'];pt=structure['tracks'];gt=truth['tracks'];pe=structure['events'];ge=truth['events']
    result=dict(nG=len(ge),nP=len(pe),gt_bins=windows(ge.values(),nf),pred_bins=windows(pe.values(),nf),iou={},edges={})
    for threshold,raw_owners in owners_raw.items():
        under=raw_owners[structure['index']];edges={};same=set()
        for pid,(direction,start,end,crosses) in pt.items():
            if not direction:continue
            matched=under[start:end];matched=matched[matched>=0]
            if len(matched) and matched[0]==matched[-1] and gt[int(matched[0])][0]==direction:same.add(int(matched[0]))
            for j,sign in crosses:
                if sign!=direction:continue
                for target in {int(under[j-1]),int(under[j])}-{-1}:
                    if target in ge and ge[target][1]==direction and any(s==direction and abs(G[k,0]-P[j,0])<=5 for k,s in gt[target][3]):edges.setdefault(pid,set()).add(target)
        tp=augmenting_tp(pe,edges);result['iou'][threshold]=dict(tp=tp,pred=len(pe),gt=len(ge),tp_id=len(same))
        result['edges'][threshold]=sorted([pid,target] for pid,targets in edges.items() for target in targets)
    return result


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sealed(root):
    root=Path(root);digest=file_sha(root/'manifest.json');assert (root/'SEALED.sha256').read_text().strip()==digest
    for name,expected in json.loads((root/'manifest.json').read_text()).items():
        path=(root/name).resolve();assert path.is_relative_to(root.resolve()) and file_sha(path)==expected
    return digest


def stream_json_array(path):
    """One physical receipt at a time; avoids another466MB parsed object."""
    decoder=json.JSONDecoder();buffer=''
    with Path(path).open(encoding='utf-8') as stream:
        while not buffer.strip():buffer=stream.read(1<<20)
        buffer=buffer.lstrip();assert buffer[0]=='[';buffer=buffer[1:]
        while True:
            buffer=buffer.lstrip()
            if buffer.startswith(','):buffer=buffer[1:].lstrip()
            if buffer.startswith(']'):return
            try:value,end=decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk=stream.read(1<<20)
                if not chunk:raise
                buffer+=chunk;continue
            buffer=buffer[end:];yield value


def final_summary_checks(root,report):
    """Replay compact group bounds after all primitive certificates were checked."""
    from fractions import Fraction
    data=json.loads((root/'frozen/deployment.json').read_text());summary=json.loads((root/'score/summary.json').read_text())
    native=json.loads((root/'score/native_scores.json').read_text())
    native_map={(v['unit'],m['member'],arm):m['metrics'] for arm,videos in native.items() for v in videos for m in v['members']}
    domain={g['unit']:g['domain'] for g in data['groups']};cuts={h:{a:0 for a in ('K','A','H','G','T','M','O')} for h in ('35','33','W','G')};copies=0
    for receipt in stream_json_array(root/'score/physical_receipts.json'):
        assert native_map[receipt['unit'],receipt['member'],receipt['arm']]==receipt['proof']['metrics']
        cuts[domain[receipt['unit']]][receipt['arm']]+=len(receipt['mask']);copies+=1
    assert copies==420
    for h in cuts:
        for a,n in cuts[h].items():assert summary['domains'][h]['methods'][a]['cut_count']==n
    report.update(native_score_artifact_copies=copies,cut_counts=cuts)
    verified={p['unit']:p for p in report['oracle_certificates']};original={p['unit']:p['oracle'] for p in summary['oracle_certificates']}
    bounds={};excluded={}
    for hive in ('35','33','W','G'):
        rows=[]
        for gid in sorted({g['group'] for g in data['groups'] if g['domain']==hive}):
            units=[g['unit'] for g in data['groups'] if (g['domain'],g['group'])==(hive,gid)]
            lower=sum(verified[u]['lower'] for u in units);upper=sum(verified[u]['upper'] for u in units);den=sum(original[u]['denominator'] for u in units)
            saved=next(g for g in summary['domains'][hive]['oracle_bounds']['groups'] if g['group']==gid)
            assert (lower,upper,den)==(saved['lower'],saved['upper'],saved['denominator'])
            rows.append((Fraction(100*lower,den),Fraction(100*upper,den)))
        lo=sum(r[0] for r in rows)/len(rows);hi=sum(r[1] for r in rows)/len(rows)
        published=summary['domains'][hive]['oracle_bounds'];assert abs(float(lo)-published['mean_lower_E3'])<1e-12 and abs(float(hi)-published['mean_upper_E3'])<1e-12
        assert published['all_optimal']==all(l==u for l,u in rows)
        bounds[hive]=dict(mean_lower_E3=float(lo),mean_witness_E3=float(hi),all_optimal=all(l==u for l,u in rows))
        for arm in ('K','A','H','G','T','M','O'):assert summary['domains'][hive]['methods'][arm]['candidate_count']==report['candidates'][hive]
    for hive,line in (('35',1.4),('33',1.3)):
        excluded[hive]=bounds[hive]['mean_lower_E3']>=report['means'][hive]['K']['E3']-line
        assert excluded[hive]==summary['oracle_family_excluded'][hive]
    report.update(oracle_bounds=bounds,oracle_family_excluded=excluded)
    return report


def field(repo,root,source195,prereg,source_code_dir=None,require_original_seals=False,audit_out=None):
    audit_out=Path(audit_out or root).resolve()
    if (audit_out/"independent.json").exists() or (audit_out/"independent.sha256").exists(): raise FileExistsError("fresh independent receipt destination required")
    source_code_dir=Path(source_code_dir or Path(__file__).parent).resolve()
    import sys
    from fractions import Fraction
    sys.path.insert(0,str(repo))
    from beetrack.evaluate import by_track,frame_match,match_events
    read=lambda p:json.loads(Path(p).read_text())
    seals={k:verify_sealed(root/k) for k in ('frozen','predictions','score')}
    if require_original_seals: assert seals['score']=='3b21533355dd1ba4ff441c2fd4d6bdc518c22b6b806b456e7c699fdb9a56e7d8'
    if require_original_seals: assert seals['frozen']=='e7fd48078f8d0a32bb9018f2bac93dc4b5270d374d5bcd0a631fa97dd4d790e4'
    if require_original_seals: assert seals['predictions']=='9a726ff7c097e0f2c595c63e2a50bd4b05390006439bd12c0c539eac4a6f752f'
    data=read(root/'frozen/deployment.json');summary=read(root/'score/summary.json');physical=read(root/'score/physical_receipts.json')
    assert data['prereg']==prereg and data['source195_seal']=='4f78f24d7fd13dc2eca9b206501a5326b4e42d476e1a9c4dd1ba98fe4bf61377'
    # Read/hash only the authorized F60 arrays from the prior120-arm freeze.
    assert file_sha(source195/'frozen/manifest.json')==data['source195_seal']
    olddata=read(source195/'frozen/deployment.json');oldnative=read(source195/'score/native_scores.json');oldmap={g['unit']:g for g in oldnative}
    assert file_sha(source195/'frozen/deployment.json')==data['source195_deployment_sha256']
    receipt=read(root/'predictions/receipt.json');assert file_sha(source_code_dir/'model.py')==receipt['model_code_sha256']
    assert receipt['prereg']==prereg
    assert receipt['freeze_receipt']['manifest_sha256']==seals['frozen']
    for fn,digest in receipt['input_receipt']['code_receipts'].items():assert file_sha(source_code_dir/fn)==digest
    groups=data['groups'];assert len(groups)==20 and sum(len(g['members']) for g in groups)==60
    expected={'35':{'BEE24-35-head'},'33':{'BEE24-33-head'},'W':{'BEE24-10','BEE24-15'},'G':{'BEE24-24','BEE24-28'}}
    truth={};bundles={};records=[];frozen_counts={};candidate_by_key={};baseline_bins={}
    for g in groups:
        assert g['video'] in expected[g['domain']]
        if g['video'] not in truth:
            path=repo/'runs/gt'/g['split']/g['video']/'gt/gt.txt';gt=np.loadtxt(path,delimiter=',',usecols=range(6),ndmin=2)
            truth[g['video']]=(gt,track_structure(gt,g['polygon']),file_sha(path))
        for m in g['members']:
            assert m['seed'] in (range(3,15) if g['domain'] in ('35','33') else range(9))
            with np.load(root/'frozen'/m['rows_file'],allow_pickle=False) as z:raw=z['rows']
            upstream=m['source_receipt'];sourcefile=source195/'frozen'/upstream['source195_rows_file'];assert file_sha(sourcefile)==upstream['source195_rows_sha256']
            with np.load(sourcefile,allow_pickle=False) as z:assert np.array_equal(raw,z['rows'])
            assert m['fixed_arm']==('t50' if g['domain'] in ('35','33') else 't40')
            own=rebuild_candidates(raw,g['polygon'],g['nf']);saved=m['bundle'];assert len(own['candidates'])==len(saved['candidates'])
            for key in ('rank','parent_id','start','end'):assert [b[key] for b in own['parent_blocks']]==[b[key] for b in saved['parent_blocks']]
            for c,a in zip(own['candidates'],saved['candidates']):
                assert c['candidate_id']==a['local_candidate_id']
                for key in ('parent_id','end','k','span','cut_after_canonical_row','window_indices','frame','direction','delta','H'):assert c[key]==a[key]
                for key in ('features_G','features_T','d_normalized','tau','core_fit_ab','pred_short','delta_windows'):np.testing.assert_allclose(c[key],a[key],atol=2e-10,rtol=1e-11)
                cid=f"{g['unit']}|{m['member_index']}|{c['parent_id']}";assert cid==a['candidate_id'];c['candidate_id']=cid
                c.update(domain=g['domain'],group=g['group'],clip=g['video'],unit=g['unit'],member_index=m['member_index'])
                records.append(c);candidate_by_key[cid]=c
            assert np.array_equal(own['baseline_windows'],saved['baseline_windows'])
            fixed=m['fixed_arm'];old=oldmap[g['unit']]['members'][m['member_index']]['arms'][fixed]
            assert np.array_equal(own['baseline_windows'],old['pred_bins'])
            key=(g['unit'],m['member']);bundles[key]=(raw,own,g,m);baseline_bins[key]=own['baseline_windows']
            frozen_counts[g['domain']]=frozen_counts.get(g['domain'],0)+len(own['candidates'])
        print('INDEPENDENT_CANDIDATES '+g['unit'],flush=True)
    # Labels are rebuilt from the other-three SOURCE pools for each fit below.
    for g in groups:
        gb=windows(truth[g['video']][1]['events'].values(),g['nf']);r=sum(baseline_bins[g['unit'],m['member']] for m in g['members'])-3*gb;baseline=int(abs(r).sum())
        for m in g['members']:
            for c in bundles[g['unit'],m['member']][1]['candidates']:
                gain=baseline-int(abs(r+c['delta_windows']).sum());assert -30<=gain<=30;c.update(y=gain/30,improvement_numerator=gain)
    decisions={};training={}
    for h in ('35','33','W','G'):
        model=read(root/'predictions'/f'{h}.model.json');fold=read(root/'predictions'/f'{h}.masks.json')
        assert model['heldout']==fold['heldout']==h and set(model['source_domains'])==set(expected)-{h}
        source=[r for r in records if r['domain']!=h];assert {r['key'] for r in model['training_records']}=={r['candidate_id'] for r in source}
        assert set(model['source_receipts'])=={r['clip'] for r in source}
        for clip,p in model['source_receipts'].items():assert p['role']=='source' and p['domain']!=h and file_sha(repo/p['path'])==p['sha256']==truth[clip][2]
        fits={kind:independent_ridge(source,h,'features_'+kind) for kind in ('G','T')}
        for kind,fit in fits.items():
            saved=model['heads'][kind]
            for ownkey,savedkey in (('mean','mean'),('scale','scale'),('coef','beta')):np.testing.assert_allclose(fit[ownkey],saved[savedkey],atol=2e-9,rtol=1e-8)
            assert abs(fit['intercept']-saved['intercept'])<1e-10
        for a in model['training_records']:
            c=candidate_by_key[a['key']];assert a['hive']!=h and a['improvement_numerator']==c['improvement_numerator'] and a['y']==c['y']
            np.testing.assert_allclose(a['features_G9'],c['features_G'],atol=2e-10);np.testing.assert_allclose(a['features_T12'],c['features_T'],atol=2e-10)
            domain=c['domain'];gs={r['group'] for r in source if r['domain']==domain};cs={r['clip'] for r in source if (r['domain'],r['group'])==(domain,c['group'])};n=sum((r['domain'],r['group'],r['clip'])==(domain,c['group'],c['clip']) for r in source)
            w=1/len({r['domain'] for r in source})/len(gs)/len(cs)/n;assert abs(w-a['weight'])<1e-15
        for item in fold['member_clips']:
            key=(item['unit'],str(item['member']));raw,b,g,m=bundles[key];cands=sorted(b['candidates'],key=lambda c:c['candidate_id']);masks={'K':[],'A':[c['candidate_id'] for c in cands],'H':[c['candidate_id'] for c in cands if c['H']]}
            for kind,fit in fits.items():
                values=[float(fit['intercept']+((np.asarray(c['features_'+kind])-fit['mean'])/fit['scale'])@fit['coef']) for c in cands]
                np.testing.assert_allclose(values,item['scores'][kind],atol=2e-9,rtol=1e-8);masks[kind]=[c['candidate_id'] for c,value in zip(cands,values) if value>.1]
            masks['M']=hash_matched_mask(cands,masks['T'],g['unit'],m['member_index'])
            for arm,ids in masks.items():
                assert sorted(ids)==sorted(item['selected'][arm])
                assert item['masks'][arm]==[c['candidate_id'] in ids for c in cands]
            assert all('y' not in c and 'improvement_numerator' not in c for c in item['candidates'])
            decisions[key]=masks
        training[h]=dict(source_candidates=len(source),source_domains=sorted({r['domain'] for r in source}))
    assert len(decisions)==60
    oracles={p['unit']:p for p in summary['oracle_certificates']};certificates=[]
    for g in groups:
        candidates=[]
        for m in g['members']:
            for c in bundles[g['unit'],m['member']][1]['candidates']:candidates.append(dict(c,candidate_id=f"{m['member_index']}|{c['candidate_id']}"))
        pred=sum(baseline_bins[g['unit'],m['member']] for m in g['members']);gb=windows(truth[g['video']][1]['events'].values(),g['nf']);answer=oracles[g['unit']]['oracle']
        certificates.append(dict(unit=g['unit'],**verify_oracle_exact(pred,gb,candidates,g['nf'],answer)))
        for m in g['members']:decisions[g['unit'],m['member']]['O']=[cid.split('|',1)[1] for cid in answer['selected_ids'] if int(cid.split('|',1)[0])==m['member_index']]
    saved_records={(r['unit'],r['member'],r['arm']):r for r in physical}
    assert len(saved_records)==420
    scores={};direct=[];owner_count=0;unique_masks=0
    first_group={h:min(g['group'] for g in groups if g['domain']==h) for h in expected}
    for g in groups:
        gt,GT,_=truth[g['video']]
        for m in g['members']:
            key=(g['unit'],m['member']);raw,b,_,_=bundles[key];baseline=realize_independent(raw,b['canonical_rows'],b['source_raw_index'],b['parent_blocks'],b['candidates'],[])
            base=track_structure(baseline,g['polygon']);owners={}
            native_order=by_track(np.c_[raw,np.arange(len(raw))])
            native_frame_order=native_order[np.argsort(native_order[:,0],kind='stable'),-1].astype(int)
            assert np.array_equal(native_frame_order,base['frame_order'])
            for threshold in (.5,.3):
                values=frame_match(GT['rows'],base['rows'],thr=threshold);scatter=np.empty(len(raw),int);scatter[base['index']]=values;owners[str(threshold)]=scatter
            owner_count+=1;cache={}
            for arm in ('K','A','H','G','T','M','O'):
                ids=decisions[key][arm];rows=realize_independent(raw,b['canonical_rows'],b['source_raw_index'],b['parent_blocks'],b['candidates'],ids)
                saved=saved_records[g['unit'],m['member'],arm];assert set(saved['mask'])==set(ids)
                actual=np.loadtxt(root/'score'/saved['path'],delimiter=',',usecols=range(7),ndmin=2);assert np.array_equal(actual,rows)
                structure=track_structure(rows,g['polygon']);assert np.array_equal(structure['frame_order'],base['frame_order'])
                maskkey=tuple(sorted(ids))
                if maskkey not in cache:cache[maskkey]=score_from_owners(structure,GT,owners,g['polygon'],g['nf']);unique_masks+=1
                metrics=cache[maskkey];published=saved['proof']['metrics']
                assert metrics['nG']==published['nG'] and metrics['nP']==published['nP']
                assert np.array_equal(metrics['pred_bins'],published['pred_bins']) and np.array_equal(metrics['gt_bins'],published['gt_bins'])
                expectedbins=b['baseline_windows']+sum((candidate_by_key[cid]['delta_windows'] for cid in ids),np.zeros_like(b['baseline_windows']))
                assert np.array_equal(metrics['pred_bins'],expectedbins)
                for t,ekey in (('0.5','edges05'),('0.3','edges03')):
                    assert all(metrics['iou'][t][k]==published['iou'][t][k] for k in ('tp','pred','gt','tp_id'))
                    assert metrics['edges'][t]==published['evidence'][ekey]
                if arm=='K':
                    reference=oldmap[g['unit']]['members'][m['member_index']]['arms'][m['fixed_arm']]
                    assert metrics['nP']==reference['nP']
                    for t in ('0.5','0.3'):assert metrics['iou'][t]=={k:reference['iou'][t][k] for k in ('tp','pred','gt','tp_id')}
                # Direct original official .5 re-evaluation on all native K60,
                # plus seven complete physical masks in one member/clip per fold.
                native_K=arm=='K'
                sample=g['group']==first_group[g['domain']] and m['member_index']==0 and g['video']==min(v['video'] for v in groups if v['domain']==g['domain'])
                if native_K or sample:
                    value=match_events(gt,raw if native_K else rows,g['polygon']);assert all(value[k]==metrics['iou']['0.5'][k] for k in ('tp','pred','gt','tp_id'))
                    direct.append(dict(unit=g['unit'],member=m['member'],arm=arm,native=native_K))
                scores[g['unit'],m['member'],arm]=metrics
            assert np.array_equal(scores[g['unit'],m['member'],'T']['pred_bins'].sum(axis=0),scores[g['unit'],m['member'],'M']['pred_bins'].sum(axis=0))
        print('INDEPENDENT_FULL_MOT '+g['unit'],flush=True)
    means={};aggregate_checks=0
    for h in expected:
        means[h]={}
        for arm in ('K','A','H','G','T','M','O'):
            E=[];F={t:[] for t in ('0.5','0.3')}
            for gid in sorted({g['group'] for g in groups if g['domain']==h}):
                clips=[g for g in groups if (g['domain'],g['group'])==(h,gid)];numerator=denominator=0
                for g in clips:
                    ss=[scores[g['unit'],m['member'],arm] for m in g['members']];gb=ss[0]['gt_bins'];numerator+=int(abs(sum(s['pred_bins'] for s in ss)-3*gb).sum());denominator+=3*int(gb.sum())
                E.append(Fraction(100*numerator,denominator))
                published=next(p for p in summary['domains'][h]['methods'][arm]['groups'] if p['group']==gid);assert (published['numerator'],published['denominator'])==(numerator,denominator)
                for t in F:
                    fs=[]
                    for i in range(3):
                        ss=[scores[g['unit'],g['members'][i]['member'],arm]['iou'][t] for g in clips];tot={k:sum(v[k] for v in ss) for k in ('tp','pred','gt','tp_id')};fs.append(Fraction(2*tot['tp'],max(1,tot['pred']+tot['gt'])))
                        assert all(tot[k]==published['iou'][t]['members'][i][k] for k in tot)
                    F[t].append(sum(fs)/3)
                aggregate_checks+=1
            means[h][arm]=dict(E3=float(sum(E)/len(E)),F1={t:float(sum(v)/len(v)) for t,v in F.items()})
            p=summary['domains'][h]['methods'][arm];assert abs(p['E3']-means[h][arm]['E3'])<1e-12
            assert all(abs(p['F1'][t]-means[h][arm]['F1'][t])<1e-14 for t in F)
    gain={h:means[h]['K']['E3']-means[h]['T']['E3'] for h in expected}
    investment=gain['35']>1.4 and gain['33']>1.3 and gain['W']>=-1.4 and all(means[h]['T']['F1']['0.5']>=means[h]['K']['F1']['0.5']-.005 for h in ('35','33','W'))
    motion=all(means[h]['T']['E3']<means[h][other]['E3'] for h in ('35','33') for other in ('G','M'))
    assert investment==summary['investment_pass'] and motion==summary['motion_attribution_pass']
    report=dict(passed=True,seals=seals,candidates=frozen_counts,source_training=training,full_MOTs=420,unique_MOT_masks=unique_masks,baseline_owner_pairs=owner_count,
                direct_official_calls=len(direct),direct_official_scope=direct,oracle_certificates=certificates,aggregate_groups=aggregate_checks,means=means,investment_pass=investment,motion_attribution_pass=motion,
                rebuilt=['candidate completeness/geometry/G9T12/H','all source singleton labels/weighted normalization/ridge and masks/M','all420raw physical MOT matrices and exact per-frame native rank','all420full-scene crossing graphs/.5/.3TP plus TP_ID using proved-invariant official owners','all20oracle integer proofs and physical binary witnesses','all98group aggregations'],
                limits=['Frame owners computed once per member/clip at both IoUs and reused only after exact raw-index per-frame rank invariance; direct match_events calls separately listed.','Oracle F1 is measured only for the saved count witness, not best F1 among count ties.','Historical seal order is attested by receipts, not independently observable retroactively.'],verifier_sha256=file_sha(__file__))
    report=final_summary_checks(root,report)
    audit_out.mkdir(parents=True,exist_ok=True);destination=audit_out/'independent.json';destination.write_text(json.dumps(report,indent=2)+'\n');digest=file_sha(destination);(audit_out/'independent.sha256').write_text(digest+'\n');print('INDEPENDENT_PASS '+str(destination)+' '+digest,flush=True)
    return report


def synthetic():
    polygon=[[0,0],[10,0],[10,10],[0,10]]
    results=[]
    for inside in (False,True):
        for end in ('start','end'):
            for added in (False,True):
                short=2 if inside else 12;other=12 if inside else 2
                xs=[short,other,other,other]+([short,short] if added else [])
                if end=='end':xs=xs[::-1]
                raw=np.array([[i+1,7,x,2,2,2,.9] for i,x in enumerate(xs)],float)
                # Shuffle native order while preserving the exact physical rows.
                raw=raw[[*range(1,len(raw),2),*range(0,len(raw),2)]]
                canonical,positions,blocks=canonicalize(raw)
                cut=0 if end=='start' else len(raw)-2
                b=realize_independent(raw,canonical,positions,blocks,[],[])
                c=dict(candidate_id='fixture',parent_id=7,cut_after_canonical_row=cut)
                alt=realize_independent(raw,canonical,positions,blocks,[c],['fixture'])
                delta=windows(event_list(alt,polygon),50)-windows(event_list(b,polygon),50)
                sign=1 if delta.sum()>0 else -1
                nonzero=np.argwhere(delta)
                assert len(nonzero)==30 and np.all(delta[delta!=0]==sign)
                event=(event_list(alt,polygon) if sign>0 else event_list(b,polygon))[0]
                c.update(frame=event[0],direction=event[1],delta=sign)
                check_candidate_effect(raw,polygon,50,c);results.append((end,inside,added,sign))
    rng=np.random.default_rng(197);records=[]
    for domain in ('33','35','W','G'):
        for group in (1,2):
            for clip in ('a','b'):
                for _ in range(3):records.append(dict(domain=domain,group=group,clip=clip,features_G=rng.normal(size=9),features_T=rng.normal(size=12),y=float(rng.choice([-1,0,1]))))
    for key in ('features_G','features_T'):
        a=independent_ridge(records,'33',key)
        for r in records:
            if r['domain']=='33':r['y']=999
        b=independent_ridge(records,'33',key);assert np.array_equal(a['coef'],b['coef'])
        constant=independent_ridge([dict(r,y=.1) for r in records],'33',key)
        assert constant['intercept']==.1 and np.all(constant['coef']==0)
    cands=[dict(candidate_id=str(i),parent_id=i,delta=(-1,1)[i%2],direction=(-1,1)[(i//2)%2]) for i in range(16)]
    M=hash_matched_mask(cands,['0','3','4'],'unit',0)
    assert len(M)==3 and sorted((c['delta'],c['direction']) for c in cands if c['candidate_id'] in M)==sorted((c['delta'],c['direction']) for c in cands if c['candidate_id'] in {'0','3','4'})
    gt=windows([(5,1)],50);pred=4*gt;c=[dict(candidate_id='delete',frame=5,direction=1,delta=-1)]
    residual=(pred-3*gt).ravel();p=[int(v>0)-int(v<0) for v in residual]
    proof=dict(p=p,q=1,linear_term=30,column_dot_products=[-30],negative_part_sum=-30,scaled_numerator=0,lower_bound=0)
    answer=dict(certificate=proof,x=[1],selected_ids=['delete'],witness_pred_sum=(3*gt).tolist(),lower_bound=0,upper_bound=0,gap=0,optimal=True,denominator=90)
    assert verify_oracle_exact(pred,gt,c,50,answer)['optimal']
    owner_checks=0
    if (Path.cwd()/'beetrack').is_dir():
        import sys
        sys.path.insert(0,str(Path.cwd()))
        from beetrack.evaluate import frame_match,match_events
        raw=np.array([[f,pid,x,2,2,2,.9] for f,x in enumerate([12,2,2,2,2],1) for pid in (7,9)],float)
        b=rebuild_candidates(raw,polygon,50);GT=track_structure(raw,polygon)
        base=realize_independent(raw,b['canonical_rows'],b['source_raw_index'],b['parent_blocks'],b['candidates'],[])
        baseline=track_structure(base,polygon);owners={}
        for threshold in (.5,.3):
            owner=np.empty(len(raw),int);owner[baseline['index']]=frame_match(GT['rows'],baseline['rows'],thr=threshold);owners[str(threshold)]=owner
        for bits in itertools.product((0,1),repeat=len(b['candidates'])):
            selected=[c['candidate_id'] for c,bit in zip(b['candidates'],bits) if bit]
            physical=realize_independent(raw,b['canonical_rows'],b['source_raw_index'],b['parent_blocks'],b['candidates'],selected)
            structure=track_structure(physical,polygon);assert np.array_equal(structure['frame_order'],baseline['frame_order'])
            metrics=score_from_owners(structure,GT,owners,polygon,50);official=match_events(raw,physical,polygon)
            for key in ('tp','gt','pred','tp_id'):assert metrics['iou']['0.5'][key]==official[key]
            for threshold in (.5,.3):assert np.array_equal(frame_match(GT['rows'],structure['rows'],thr=threshold),owners[str(threshold)][structure['index']])
            owner_checks+=1
    return dict(passed=True,endpoint_cases=len(results),native_row_geometry_unchanged=True,
                thirty_phase_single_add_drop=True,source_only_ridge=True,constant_label_strict_threshold=True,matched_count_control=True,
                exact_integer_oracle_certificate=True,
                identical_box_owner_invariance_and_official_graph=owner_checks,
                scope='synthetic only; field disabled until coordinator GO')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path);p.add_argument('--root',type=Path);p.add_argument('--source195',type=Path);p.add_argument('--prereg');p.add_argument('--field-go',action='store_true');p.add_argument('--source-code-dir',type=Path);p.add_argument('--require-original-seals',action='store_true');p.add_argument('--audit-out',type=Path);a=p.parse_args()
    if a.field_go:
        if not all((a.repo,a.root,a.source195,a.prereg)):p.error('field requires repo/root/source195/prereg and coordinator GO')
        field(a.repo.resolve(),a.root.resolve(),a.source195.resolve(),a.prereg,a.source_code_dir,a.require_original_seals,a.audit_out)
    elif any((a.repo,a.root,a.source195,a.prereg)):p.error('no field reads before --field-go coordinator release')
    else:print(json.dumps(synthetic(),indent=2))
