"""Post-registration exact WINDOW support on the original fixed209 slate/k.

Marker [cross-part-window-support-v1]. No field IO on import. No ranking,
training, GT association or F1 scoring: saved integer GT windows are accepted
only in the explicitly gated field adapter. The main36 states/gate stay fixed.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,gzip,hashlib,json,math,subprocess,sys
from fractions import Fraction
from itertools import combinations
from numbers import Integral
from pathlib import Path
import numpy as np

MARKER='[cross-part-window-support-v1]'
PREREG='9efbea732621dec804aea9dcc9df87fc1ca6293e'
ARMS=('U','G','F','X')
def integer(x):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral):raise ValueError('integer required')
    return int(x)
def pair_key(s):return tuple(map(int,s.split('|')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    if str(p).endswith('.gz'):
        with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
    return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    if p.exists():
        if p.read_text()!=text:raise ValueError('immutable output differs '+str(p))
    else:p.write_text(text)
def vector_hash(a):
    a=np.ascontiguousarray(a,dtype='<i8')
    return hashlib.sha256(json.dumps(list(a.shape),separators=(',',':')).encode()+b'\n'+a.tobytes()).hexdigest()

def enumerate_clip(pred_bins,gt_bins,pair_deltas,slate,k,actual=None,max_active=20):
    """Exact all-window support for exactly k of n disjoint actions.

    Active subset S can be completed iff 0<=k-|S|<=n-a. A completion's zero
    actions are selected by ascending numeric pair ID, independent of scores.
    Every physical k-subset maps to exactly one active subset, with multiplicity
    C(n-a,k-|S|). Deduplicated vectors retain BOTH multiplicities explicitly.
    """
    P=np.asarray(pred_bins);G=np.asarray(gt_bins)
    if P.ndim!=2 or P.shape[1]!=2 or G.shape!=P.shape:raise ValueError('paired integer directional windows required')
    if any(not np.issubdtype(a.dtype,np.integer) or (a<0).any() for a in (P,G)):raise ValueError('nonnegative integer counts required')
    slate=list(slate);n=len(slate);k=integer(k)
    if len(set(slate))!=n or not 0<=k<=n:raise ValueError('fixed slate/k differs')
    table={r['pair_id']:r for r in pair_deltas}
    if len(table)!=len(pair_deltas) or set(table)!=set(slate):raise ValueError('complete fixed-slate delta table required')
    used=set();columns={}
    for key in slate:
        ids=pair_key(key)
        if len(ids)!=2 or ids[0]>=ids[1] or used&set(ids):raise ValueError('slate must contain disjoint canonical runs')
        used.update(ids);d=np.asarray(table[key]['delta_windows'])
        if d.shape!=P.shape or not np.issubdtype(d.dtype,np.integer):raise ValueError('integer delta shape differs')
        columns[key]=d.astype(np.int64,copy=False)
    active=sorted([s for s in slate if np.any(columns[s])],key=pair_key);zero=sorted(set(slate)-set(active),key=pair_key)
    a,z=len(active),len(zero)
    if a>max_active:raise ValueError('exact small-support cap exceeded; do not claim solved')
    baseline=sum(abs(int(p)-int(g)) for p,g in zip(P.ravel(),G.ravel()));den=sum(map(int,G.ravel()))
    if den%30:raise ValueError('GT event population not repeated exactly30 times')
    vectors={};feasible=0;physical_count=0;minimum=None;maximum=None
    for size in range(max(0,k-z),min(k,a)+1):
        for selected_active in combinations(active,size):
            selected=list(sorted((*selected_active,*zero[:k-size]),key=pair_key));B=P.astype(np.int64,copy=True)
            for key in selected_active:B+=columns[key]
            if (B<0).any():raise ValueError('nonphysical negative count in additive support')
            numerator=sum(abs(int(p)-int(g)) for p,g in zip(B.ravel(),G.ravel()));digest=vector_hash(B);multiplicity=math.comb(z,k-size)
            state=dict(active_subset=list(selected_active),zero_completion=zero[:k-size],selected=selected,vector_sha256=digest,
                numerator=numerator,pred_bins=B.tolist(),gain_numerator=baseline-numerator)
            tie=tuple(pair_key(s) for s in selected)
            if minimum is None or (numerator,tie)<(minimum['numerator'],tuple(pair_key(s) for s in minimum['selected'])):minimum=state
            if maximum is None or (-numerator,tie)<(-maximum['numerator'],tuple(pair_key(s) for s in maximum['selected'])):maximum=state
            if digest not in vectors:vectors[digest]=dict(pred_bins=B.tolist(),numerator=numerator,active_subset_count=0,physical_subset_count=0,representative=state)
            v=vectors[digest]
            if v['pred_bins']!=B.tolist():raise AssertionError('vector hash collision')
            v['active_subset_count']+=1;v['physical_subset_count']+=multiplicity
            if tie<tuple(pair_key(s) for s in v['representative']['selected']):v['representative']=state
            feasible+=1;physical_count+=multiplicity
    if physical_count!=math.comb(n,k) or minimum is None:raise AssertionError('complete k-subset coverage identity failed')
    realized={}
    for arm,record in (actual or {}).items():
        if arm not in ARMS:raise ValueError('J is outside this fixed common-slate support')
        ids=record['selected']
        if len(ids)!=len(set(ids)) or not set(ids)<=set(slate):raise ValueError('actual model selection outside common slate')
        B=P.astype(np.int64,copy=True)
        for key in ids:B+=columns[key]
        if not np.array_equal(B,record['pred_bins']):raise ValueError('actual saved arm vector disagrees additive realization')
        value=sum(abs(int(p)-int(g)) for p,g in zip(B.ravel(),G.ravel()))
        if value!=integer(record['numerator']):raise ValueError('actual raw numerator disagrees integer window error')
        digest=vector_hash(B);realized[arm]=dict(selected=ids,numerator=value,gain_numerator=baseline-value,vector_sha256=digest,
            exact_k_mask=len(ids)==k,within_exact_k_vector_support=digest in vectors,additive_vector_exact=True)
    result=dict(n=n,k=k,active_count=a,zero_count=z,active_pair_ids=active,zero_pair_ids=zero,
        feasibility='0<=k-|active_subset|<=n-active_count',feasible_active_subsets=feasible,
        feasible_physical_subsets=physical_count,physical_subset_binomial=math.comb(n,k),distinct_window_vectors=len(vectors),
        baseline_numerator=baseline,denominator=den,minimum=minimum,maximum=maximum,
        vectors=[dict(vector_sha256=h,**v) for h,v in sorted(vectors.items())],actual=realized,
        scope='Exact count/window support only; zero-delta actions can change boxes and F1; J excluded; original fixed k retained.')
    if den:
        result.update(min_W_percent=float(Fraction(100*minimum['numerator'],den)),max_W_percent=float(Fraction(100*maximum['numerator'],den)),
            best_gain_fraction=[100*(baseline-minimum['numerator']),den],best_gain_percent=float(Fraction(100*(baseline-minimum['numerator']),den)),
            can_gain_at_least_1pp=100*(baseline-minimum['numerator'])>=den,can_gain_more_than_1pp=100*(baseline-minimum['numerator'])>den)
    else:result.update(min_W_percent=None,max_W_percent=None,best_gain_percent=None,can_gain_at_least_1pp=None,can_gain_more_than_1pp=None)
    return result

def counts_independent(events,nf):
    """Second integer histogram implementation; not attainable209.window_counts."""
    E=np.asarray(events,int).reshape(-1,2);out=[]
    if len(E) and ((E[:,0]<1).any() or (E[:,0]>nf).any() or not np.isin(E[:,1],[-1,1]).all()):raise ValueError('invalid event clock')
    for shift in range(0,750,25):
        bins=(E[:,0]-1+shift)//750;direction=(E[:,1]<0).astype(int);n=(nf+shift+749)//750
        out.append(np.bincount(2*bins+direction,minlength=2*n).reshape(n,2))
    return np.vstack(out)

def verify_extreme(rows,canon,pairs,polygon,nframes,state,gt_bins,gt_events=None,out=None):
    """Actual full MOT -> official native events -> independently counted bins.

    No frame_match, GT rectangle association, event matching or F1 is invoked.
    """
    import physical209
    from beetrack.p0 import events,window_err
    R,proof=physical209.materialize(rows,canon,pairs,state['selected']);E=events(R,polygon).astype(int)
    B=counts_independent(E,nframes);G=np.asarray(gt_bins,np.int64);raw=sum(abs(int(b)-int(g)) for b,g in zip(B.ravel(),G.ravel()))
    if B.tolist()!=state['pred_bins'] or raw!=state['numerator'] or vector_hash(B)!=state['vector_sha256']:raise ValueError('physical extreme is not exact enumerated vector')
    official=None
    if gt_events is not None:
        GE=np.asarray(gt_events,int).reshape(-1,2)
        if not np.array_equal(counts_independent(GE,nframes),G):raise ValueError('saved GT event clock disagrees saved GT windows')
        official=float(window_err(E,GE,nframes))
        if abs(30*official-raw)>1e-7:raise ValueError('official30-phase window_err disagrees integer numerator')
    result=dict(selected=state['selected'],events=E.tolist(),integer_bins_exact=True,numerator=raw,vector_sha256=vector_hash(B),
        official_window_err=official,retained_raw_indices=proof['retained_raw_indices'],deleted_raw_indices=proof['deleted_raw_indices'],
        actions=proof['actions'],no_GT_matching_or_F1=True)
    if out is not None:
        from tools.study195.freeze import write_mot_exact
        out=Path(out);out.mkdir(parents=True,exist_ok=True);result['mot']=write_mot_exact(R,out/'mot.txt');result['mot_sha256']=sha(out/'mot.txt')
        save(out/'counts.json',result)
    return result

def pooled(clips):
    """W/G clips pool integer numerators+denominators BEFORE division."""
    result={}
    for h in sorted({c['domain'] for c in clips}):
        rows=[c for c in clips if c['domain']==h];D=sum(c['exact']['denominator'] for c in rows)
        K=sum(c['exact']['baseline_numerator'] for c in rows);lo=sum(c['exact']['minimum']['numerator'] for c in rows);hi=sum(c['exact']['maximum']['numerator'] for c in rows)
        result[h]=dict(baseline_numerator=K,denominator=D,min_numerator=lo,max_numerator=hi,best_gain_numerator=K-lo,
            baseline_W=float(Fraction(100*K,D)) if D else None,min_W=float(Fraction(100*lo,D)) if D else None,max_W=float(Fraction(100*hi,D)) if D else None,
            best_gain_percent=float(Fraction(100*(K-lo),D)) if D else None,can_gain_at_least_1pp=100*(K-lo)>=D if D else None,
            can_gain_more_than_1pp=100*(K-lo)>D if D else None,actual={})
        for arm in ARMS:
            N=sum(c['exact']['actual'][arm]['numerator'] for c in rows)
            result[h]['actual'][arm]=dict(numerator=N,W=float(Fraction(100*N,D)) if D else None,gain_numerator=K-N)
    return result

def checked_seal(root,expected):
    root=Path(root).resolve();digest=sha(root/'manifest.json')
    if digest!=expected or (root/'SEALED.sha256').read_text().strip()!=digest:raise ValueError('pinned seal differs')
    manifest=read(root/'manifest.json')
    for name,value in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha(path)!=value:raise ValueError('sealed input differs '+name)
    return digest

def field(args):
    # Explicit post-publication and unique-CPU gate. No changes to main36 files.
    if not args.field_go or not args.cpu_released:raise ValueError('published supplement and unique CPU handoff required')
    subprocess.run(['git','merge-base','--is-ancestor',args.supplement_sha,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.supplement_sha+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('published support supplement marker differs')
    score_seal=checked_seal(args.root/'score',args.score_seal);checked_seal(args.root/'catalog',args.catalog_seal)
    if sha(args.root/'attainable_diagnostic.json')!=args.triangle_sha:raise ValueError('fixed triangle evidence differs')
    triangle=read(args.root/'attainable_diagnostic.json')
    if triangle['prereg']!=PREREG or triangle['score_seal']!=score_seal:raise ValueError('original prereg/score linkage differs')
    import catalog209
    deployment=read(args.root/'catalog/deployment.json');refs={(r['domain'],r['clip']):r for r in deployment['clips']}
    out=args.out;out.mkdir(parents=True,exist_ok=True);results=[]
    receipt=dict(original_prereg=PREREG,supplement_sha=args.supplement_sha,marker=MARKER,score_seal=score_seal,
        catalog_seal=args.catalog_seal,triangle_sha256=args.triangle_sha,code_sha256=sha(__file__),no_new_matching_or_F1=True)
    save(out/'input_receipt.json',receipt)
    for diagnostic in triangle['clips']:
        domain,clip=diagnostic['domain'],diagnostic['clip'];folder=args.root/'score/clips'/domain/clip
        records={a:read(folder/a/'score.json') for a in ('K',*ARMS)};K=records['K']['metrics']
        P,G=K['pred_bins'],K['gt_bins'];actual={a:dict(selected=records[a]['selected_pair_ids'],pred_bins=records[a]['metrics']['pred_bins'],numerator=records[a]['numerator']) for a in ARMS}
        exact=enumerate_clip(P,G,diagnostic['pairs'],diagnostic['common_slate'],diagnostic['declared_k'],actual)
        source=catalog209.load_clip(args.root/'catalog'/refs[domain,clip]['path'],args.source208);meta=source['metadata']
        GTevents=[[abs(int(signed)),1 if signed>0 else -1] for _,signed in K['evidence']['gt']]
        checks={}
        for name,state in (('min',exact['minimum']),('max',exact['maximum'])):
            checks[name]=verify_extreme(source['rows'],source['canon'],source['pairs'],meta['polygon'],meta['nf'],state,G,GTevents,out/domain/clip/name)
        value=dict(domain=domain,clip=clip,exact=exact,physical_count_checks=checks)
        save(out/domain/clip/'exact.json',value);results.append(value);print('EXACT209',domain,clip,exact['active_count'],exact['distinct_window_vectors'],exact['minimum']['numerator'],exact['maximum']['numerator'],flush=True)
    summary=dict(schema='common209-exact-window-support-v1',receipt=receipt,clips=results,pooled=pooled(results),
        scope='Post-registration exact window support on original disjoint slate and k; main36 metrics/gate unchanged; not J; no new F1.')
    save(out/'exact.json',summary);save(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('manifest.json','SEALED.sha256')})
    seal=sha(out/'manifest.json');path=out/'SEALED.sha256'
    if path.exists():
        if path.read_text().strip()!=seal:raise ValueError('sidecar output seal differs')
    else:path.write_text(seal+'\n')
    print('EXACT209_SEALED',seal,flush=True)

def selftest():
    P=np.asarray([[2,0]]*30);G=np.asarray([[1,0]]*30)
    D=[dict(pair_id='1|2',delta_windows=[[-1,0]]*30),dict(pair_id='3|4',delta_windows=[[1,0]]*30),
       dict(pair_id='5|6',delta_windows=[[0,0]]*30),dict(pair_id='7|8',delta_windows=[[0,0]]*30)]
    e=enumerate_clip(P,G,D,[r['pair_id'] for r in D],2)
    assert e['feasible_active_subsets']==4 and e['feasible_physical_subsets']==6 and e['distinct_window_vectors']==3
    assert (e['minimum']['numerator'],e['maximum']['numerator'])==(0,60)
    assert sorted(v['active_subset_count'] for v in e['vectors'])==[1,1,2]
    # No zero filler: exactly k active, so k1 excludes empty/both.
    a=enumerate_clip(P,G,D[:2],[r['pair_id'] for r in D[:2]],1);assert a['feasible_active_subsets']==2
    z=enumerate_clip(P,G,D[2:],[r['pair_id'] for r in D[2:]],1);assert z['distinct_window_vectors']==1 and z['feasible_physical_subsets']==2
    empty=enumerate_clip(P,G,[],[],0);assert empty['feasible_active_subsets']==1
    # Actual known arm closure, independent of whether its registered model abstained.
    actual={arm:dict(selected=e['minimum']['selected'],pred_bins=e['minimum']['pred_bins'],numerator=0) for arm in ARMS}
    assert all(v['within_exact_k_vector_support'] for v in enumerate_clip(P,G,D,[r['pair_id'] for r in D],2,actual)['actual'].values())
    import catalog209,attainable209
    from beetrack.p0 import events
    def row(f,p,x,y=0,s=.9):return [f,p,x,y,4,4,s]
    rows=[row(f,1,-3) for f in range(1,6)]+[row(f,2,-1,s=.8) for f in range(3,8)]
    rows += [row(f,p,-3 if f<6 else -1,20,.7) for p in (3,4) for f in range(1,8)]
    rows += [row(f,p,-3,y,.7) for p,y in ((5,40),(6,40),(7,60),(8,60)) for f in range(1,8)]
    R=np.asarray(rows,float);C=R[:,1].astype(np.int64);poly=[[0,-20],[100,-20],[100,100],[0,100]];nf=7
    pairs=catalog209.build(R,C,nf)['pairs'];slate=['1|2','3|4','5|6','7|8']
    prediction=counts_independent(events(R,poly).astype(int),nf);truth=counts_independent([[6,1]],nf)
    diagnostic=attainable209.analyze(R,C,pairs,slate,poly,nf,1)
    exact=enumerate_clip(prediction,truth,diagnostic['pairs'],slate,1)
    assert exact['minimum']['numerator']==0 and exact['maximum']['numerator']==60
    for state in (exact['minimum'],exact['maximum']):verify_extreme(R,C,pairs,poly,nf,state,truth,[[6,1]])
    print('PASS synthetic exact support;zero-fill iff;vector collisions;binomial coverage;empty;actual-arm closure;physical min/max official event+window verification')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--synthetic',action='store_true');p.add_argument('--field-go',action='store_true');p.add_argument('--cpu-released',action='store_true')
    for name in ('root','out','source208'):p.add_argument('--'+name,type=Path)
    for name in ('supplement-sha','score-seal','catalog-seal','triangle-sha'):p.add_argument('--'+name)
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    if args.synthetic:selftest()
    else:
        if not all((args.root,args.out,args.source208,args.supplement_sha,args.score_seal,args.catalog_seal,args.triangle_sha,args.field_go,args.cpu_released)):p.error('gated pinned inputs and CPU handoff required')
        field(args)
