"""Fixed disjoint ACTIVE-slate210 count ceiling; source-only until CPU GO.

Exactly ceil(a/4) actions (0 if empty). Enumerate <=65536 combinations;
otherwise give integer coordinate L1 bounds and actual attainable witnesses.
No F1, full-candidate-graph claim, optimizer, ranking or threshold selection.
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

EXACT_CAP=65536
ARMS=('N','G','X','M')
def integer(x):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral):raise ValueError('integer required')
    return int(x)
def pair_key(key):return tuple(map(int,key.split('|')))
def raw_loss(P,G):return sum(abs(int(p)-int(g)) for p,g in zip(P.ravel(),G.ravel()))
def digest(B):
    B=np.ascontiguousarray(B,dtype='<i8')
    return hashlib.sha256(json.dumps(list(B.shape),separators=(',',':')).encode()+b'\n'+B.tobytes()).hexdigest()

def coordinate_certificate(residual,columns,k):
    """Each coord's k-smallest/largest delta sum encloses EVERY k-subset.

    The coordinate optimizers may select different subsets. Thus the resulting
    L1 lower/upper losses are rigorous relaxations, not attainable extrema.
    """
    R=np.asarray(residual,np.int64);D=np.asarray(columns,np.int64).reshape(-1,R.size);a=len(D)
    if not 0<=k<=a:raise ValueError('fixed cardinality outside slate')
    if k:
        low=np.partition(D,k-1,axis=0)[:k].sum(axis=0)
        high=np.partition(D,a-k,axis=0)[a-k:].sum(axis=0)
    else:low=high=np.zeros(R.size,np.int64)
    left=R.ravel()+low;right=R.ravel()+high
    minimum=np.where(left>0,left,np.where(right<0,-right,0));maximum=np.maximum(abs(left),abs(right))
    return dict(k=k,delta_sum_lower=low.reshape(R.shape).tolist(),delta_sum_upper=high.reshape(R.shape).tolist(),
        residual=R.tolist(),coordinate_loss_lower=minimum.reshape(R.shape).tolist(),coordinate_loss_upper=maximum.reshape(R.shape).tolist(),
        loss_lower=sum(map(int,minimum)),loss_upper=sum(map(int,maximum)),
        proof='For each coordinate sum k smallest delta <= any k-subset sum <= sum k largest;sum dist(0,interval) <= L1 <= sum max(abs endpoints).')

def diagnose_clip(pred_bins,gt_bins,slate_actions,actual,population,*,exact_cap=EXACT_CAP):
    P=np.asarray(pred_bins);G=np.asarray(gt_bins)
    if P.ndim!=2 or P.shape[1]!=2 or G.shape!=P.shape or any(not np.issubdtype(v.dtype,np.integer) or (v<0).any() for v in (P,G)):
        raise ValueError('nonnegative integer directional count arrays required')
    P=P.astype(np.int64,copy=False);G=G.astype(np.int64,copy=False);a=len(slate_actions);k=(a+3)//4 if a else 0
    if exact_cap!=EXACT_CAP:raise ValueError('registered exact cap is fixed65536')
    table={x['pair_id']:x for x in slate_actions};used=set();columns={}
    if len(table)!=a:raise ValueError('unique fixed-slate pair IDs required')
    for key,action in table.items():
        ids=tuple(map(integer,action['canonical_ids']))
        if len(ids)!=2 or ids[0]>=ids[1] or key!='|'.join(map(str,ids)) or set(ids)&used:raise ValueError('fixed slate must be disjoint original runs')
        used.update(ids);delta=np.asarray(action['delta_windows'])
        if delta.shape!=P.shape or not np.issubdtype(delta.dtype,np.integer) or not np.any(delta):raise ValueError('ACTIVE slate needs nonzero integer window delta')
        columns[key]=delta.astype(np.int64,copy=False)
    pop={key:integer(population[key]) for key in ('trigger_candidates','active_candidates','slate_pairs')}
    if not 0<=a==pop['slate_pairs']<=pop['active_candidates']<=pop['trigger_candidates']:raise ValueError('frozen trigger/active/slate population differs')
    if 'all_slate_pairs' in population:
        pop['all_slate_pairs']=integer(population['all_slate_pairs'])
        if not a<=pop['all_slate_pairs']<=pop['trigger_candidates']:raise ValueError('full versus active slate population differs')
    order=sorted(table,key=pair_key);den=sum(map(int,G.ravel()))
    if den%30:raise ValueError('full30phase GT census differs')
    base=raw_loss(P,G)
    def state(ids):
        if len(ids)!=len(set(ids)) or not set(ids)<=table.keys():raise ValueError('foreign/repeated fixed-slate selection')
        B=P.copy()
        for key in ids:B+=columns[key]
        if (B<0).any():raise ValueError('nonphysical negative additive counts')
        return dict(selected=sorted(ids,key=pair_key),numerator=raw_loss(B,G),pred_bins=B.tolist(),vector_sha256=digest(B))
    actual_checked={};witnesses=[state(order[:k])]
    for arm in ARMS:
        record=actual[arm];value=state(record['selected'])
        if value['pred_bins']!=record['pred_bins'] or value['numerator']!=integer(record['numerator']):raise ValueError('saved actual arm disagrees frozen additive window vector')
        exact_budget=len(record['selected'])==k
        if exact_budget:witnesses.append(value)
        actual_checked[arm]=dict(**value,exact_budget=exact_budget,additive_vector_exact=True)
    best=min(witnesses,key=lambda s:(s['numerator'],tuple(pair_key(t) for t in s['selected'])))
    worst=min(witnesses,key=lambda s:(-s['numerator'],tuple(pair_key(t) for t in s['selected'])))
    cert=coordinate_certificate(P-G,[columns[key].ravel() for key in order],k);count=math.comb(a,k);enumerated=0
    exact=count<=exact_cap
    if exact:
        for ids in combinations(order,k):
            value=state(ids);enumerated+=1;tie=tuple(pair_key(t) for t in value['selected'])
            if (value['numerator'],tie)<(best['numerator'],tuple(pair_key(t) for t in best['selected'])):best=value
            if (-value['numerator'],tie)<(-worst['numerator'],tuple(pair_key(t) for t in worst['selected'])):worst=value
        if enumerated!=count:raise AssertionError('exact budget subset coverage differs')
    if not cert['loss_lower']<=best['numerator']<=worst['numerator']<=cert['loss_upper']:raise AssertionError('coordinate certificate or attainable witness inconsistent')
    lo=best['numerator'] if exact else cert['loss_lower'];hi=worst['numerator'] if exact else cert['loss_upper']
    all_state=state(order)
    reference_checks={}
    for arm,expected in (('K',state([])),('ALL',all_state)):
        if arm in actual:
            record=actual[arm]
            if set(record['selected'])!=set(expected['selected']) or record['pred_bins']!=expected['pred_bins'] or integer(record['numerator'])!=expected['numerator']:
                raise ValueError('K/ALL reference is not this same fixed-slate count realization')
            reference_checks[arm]=True
    result=dict(population=pop,a=a,k=k,combinations=count,exact_cap=exact_cap,mode='exact' if exact else 'coordinate_bounds',
        enumerated=enumerated,denominator=den,baseline_numerator=base,K=state([]),ALL=all_state,
        min_loss_interval=[lo,best['numerator']],max_loss_interval=[worst['numerator'],hi],
        best_known=best,worst_known=worst,actual=actual_checked,coordinate_certificate=cert,
        fixed_slate_pair_ids=order,reference_checks=reference_checks,
        count_input=dict(pred_bins=P.tolist(),gt_bins=G.tolist(),slate_actions=[dict(pair_id=key,canonical_ids=table[key]['canonical_ids'],delta_windows=columns[key].tolist()) for key in order]),
        scope='Only this frozen disjoint ACTIVE slate at original quarter budget;ALL means all this slate,not full trigger graph;not F1.')
    if den:
        result.update(gain_lower_fraction=[100*(base-best['numerator']),den],gain_upper_fraction=[100*(base-lo),den],
            gain_lower_percent=float(Fraction(100*(base-best['numerator']),den)),gain_upper_percent=float(Fraction(100*(base-lo),den)))
    else:result.update(gain_lower_percent=None,gain_upper_percent=None)
    return result

def pool(clips):
    """Pool W/G actual-clip integer losses and GT counts, never mean clip W."""
    result={}
    for hive in sorted({c['domain'] for c in clips}):
        rows=[c['capacity'] for c in clips if c['domain']==hive];D=sum(r['denominator'] for r in rows);K=sum(r['baseline_numerator'] for r in rows)
        minlo=sum(r['min_loss_interval'][0] for r in rows);minhi=sum(r['min_loss_interval'][1] for r in rows)
        maxlo=sum(r['max_loss_interval'][0] for r in rows);maxhi=sum(r['max_loss_interval'][1] for r in rows)
        exact=all(r['mode']=='exact' for r in rows)
        result[hive]=dict(mode='exact' if exact else 'pooled_certified_bounds',denominator=D,baseline_numerator=K,
            min_loss_interval=[minlo,minhi],max_loss_interval=[maxlo,maxhi],
            best_gain_percent_interval=[float(Fraction(100*(K-minhi),D)),float(Fraction(100*(K-minlo),D))] if D else None,
            actual={arm:sum(r['actual'][arm]['numerator'] for r in rows) for arm in ARMS},
            K=sum(r['K']['numerator'] for r in rows),ALL=sum(r['ALL']['numerator'] for r in rows))
    return result

def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt',encoding='utf8') as f:return json.load(f)
    return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=text:raise ValueError('immutable output differs '+str(path))
    else:path.write_text(text)
def sealed(root,expected):
    root=Path(root).resolve()
    if sha(root/'manifest.json')!=expected or (root/'SEALED.sha256').read_text().strip()!=expected:raise ValueError('pinned artifact seal differs')
    for name,value in read(root/'manifest.json').items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha(path)!=value:raise ValueError('sealed source changed '+name)

def count_events(events,nf):
    E=np.asarray(events,int).reshape(-1,2);parts=[]
    if len(E) and (not np.isin(E[:,1],[-1,1]).all() or (E[:,0]<1).any() or (E[:,0]>nf).any()):raise ValueError('invalid event clock')
    for shift in range(0,750,25):
        n=(nf+shift+749)//750;bins=(E[:,0]-1+shift)//750;direction=(E[:,1]<0).astype(int)
        parts.append(np.bincount(2*bins+direction,minlength=2*n).reshape(n,2))
    return np.vstack(parts)

def verify_count_witness(rows,canon,pairs,polygon,nframes,witness,gt_bins):
    """Original-row bijection -> official events -> count-only witness check.

    No GT rectangle loader, frame Hungarian, event pairing or F1 is called.
    Stable date ordering is allowed; every non-ID observation must survive.
    """
    import physical210
    from beetrack.p0 import events
    out,proof=physical210.materialize(rows,canon,pairs,witness['selected']);order=np.asarray(proof['output_raw_indices'],int);raw=np.asarray(rows,float)
    if len(out)!=len(raw) or not np.array_equal(np.sort(order),np.arange(len(raw))) or not np.array_equal(out[:,[0,2,3,4,5,6]],raw[order][:,[0,2,3,4,5,6]]):raise ValueError('210 action loses/changes a raw observation')
    E=events(out,polygon).astype(int);B=count_events(E,nframes)
    if B.tolist()!=witness['pred_bins'] or digest(B)!=witness['vector_sha256'] or raw_loss(B,np.asarray(gt_bins))!=witness['numerator']:raise ValueError('official count realization differs budget witness')
    return dict(selected=witness['selected'],numerator=witness['numerator'],vector_sha256=digest(B),events=E.tolist(),
        output_raw_indices=order.tolist(),all_original_rows_once=True,official_count_exact=True,no_new_F1=True)

def field(args):
    if not args.field_go or not args.independent_cpu_released:raise ValueError('explicit root/FIFO CPU GO required')
    import catalog210,physical210
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count('[alternating-offset-v1]')!=1:raise ValueError('published210 preregistration marker differs')
    sealed(args.root/'catalog',args.catalog_seal);sealed(args.root/'score',args.score_seal)
    top=read(args.root/'catalog/deployment.json')
    if top['prereg']!=args.prereg or {(r['domain'],r['clip'],r['seed']) for r in top['clips']}!=catalog210.SPECS or len(top['clips'])!=6:raise ValueError('fixed six-case scope differs')
    args.out.mkdir(parents=True,exist_ok=True)
    receipt=dict(prereg=args.prereg,catalog_seal=args.catalog_seal,score_seal=args.score_seal,code_sha256=sha(__file__),
        physical_code_sha256=sha(physical210.__file__),catalog_code_sha256=sha(catalog210.__file__),exact_cap=EXACT_CAP,
        scope='After main36 and independent CPU release;count-only explanation;formal model selections/gate untouched.')
    save(args.out/'input_receipt.json',receipt);reports=[]
    for ref in top['clips']:
        source=catalog210.load_clip(args.root/'catalog'/ref['path'],args.source208);meta=source['metadata'];table={r['pair_id']:r for r in source['pairs']}
        frozen=physical210.fixed_slate(source['pairs'])
        if frozen['slate']!=source['slate_ids'] or frozen['active_slate']!=source['active_slate_ids']:raise ValueError('frozen active slate differs')
        active=[table[k] for k in frozen['active_slate']];domain,clip=ref['domain'],ref['clip'];folder=args.root/'score/clips'/domain/clip
        if read(folder/'summary.json')['signature']['prereg']!=args.prereg:raise ValueError('main clip registration differs')
        records={arm:read(folder/arm/'score.json') for arm in ('K',*ARMS,'ALL')}
        actual={arm:dict(selected=r['selected_pair_ids'],pred_bins=r['metrics']['pred_bins'],numerator=r['numerator']) for arm,r in records.items()}
        pop=source['denominator'];population=dict(trigger_candidates=pop['eligible_pairs'],active_candidates=pop['active_pairs'],
            slate_pairs=pop['active_slate_pairs'],all_slate_pairs=pop['slate_pairs'])
        K=records['K']['metrics'];answer=diagnose_clip(K['pred_bins'],K['gt_bins'],active,actual,population)
        if any(integer(r['denominator'])!=answer['denominator'] for r in records.values()):raise ValueError('main raw GT denominator differs')
        if not all(r['exact_budget'] for r in answer['actual'].values()) or answer['k']!=frozen['k']:raise ValueError('actual controls do not spend the frozen budget')
        checks={name:verify_count_witness(source['rows'],source['canon'],source['pairs'],meta['polygon'],meta['nf'],answer[name],K['gt_bins']) for name in ('best_known','worst_known')}
        report=dict(domain=domain,clip=clip,capacity=answer,physical_count_witnesses=checks);reports.append(report)
        save(args.out/domain/clip/'capacity.json',report)
        print('CAPACITY210',domain,clip,answer['a'],answer['k'],answer['combinations'],answer['mode'],answer['min_loss_interval'],answer['max_loss_interval'],flush=True)
    summary=dict(schema='interleave210-fixed-budget-count-v1',receipt=receipt,clips=reports,pooled=pool(reports))
    save(args.out/'capacity.json',summary);save(args.out/'manifest.json',{p.relative_to(args.out).as_posix():sha(p) for p in args.out.rglob('*') if p.is_file() and p.name not in ('manifest.json','SEALED.sha256')})
    seal=sha(args.out/'manifest.json');path=args.out/'SEALED.sha256'
    if path.exists():
        if path.read_text().strip()!=seal:raise ValueError('output seal differs')
    else:path.write_text(seal+'\n')
    print('CAPACITY210_SEALED',seal,flush=True)

def selftest():
    P=np.asarray([[4,1]]*30);G=np.asarray([[1,1]]*30)
    actions=[dict(pair_id=f'{2*i+1}|{2*i+2}',canonical_ids=[2*i+1,2*i+2],delta_windows=[[d,0]]*30) for i,d in enumerate((-1,-1,1,1))]
    selections=dict(N=['1|2'],G=['3|4'],X=['5|6'],M=['7|8'])
    actual={arm:dict(selected=ids,pred_bins=(P+np.asarray(actions[[a['pair_id'] for a in actions].index(ids[0])]['delta_windows'])).tolist(),numerator=60 if arm in ('N','G') else 120) for arm,ids in selections.items()}
    e=diagnose_clip(P,G,actions,actual,dict(trigger_candidates=9,active_candidates=6,slate_pairs=4))
    assert e['mode']=='exact' and e['combinations']==4 and e['min_loss_interval']==[60,60] and e['max_loss_interval']==[120,120]
    # Non-enumerated larger family: actual feasible witnesses need not reach the
    # coordinate bound; never label those values true extrema.
    many=[]
    for i in range(24):
        d=np.asarray([[-1,1] if i%2 else [1,-1]]*30)
        many.append(dict(pair_id=f'{2*i+1}|{2*i+2}',canonical_ids=[2*i+1,2*i+2],delta_windows=d.tolist()))
    PP=np.asarray([[50,50]]*30);GG=np.asarray([[45,45]]*30);ids=[a['pair_id'] for a in many[:6]];B=PP.copy()
    for a in many[:6]:B+=np.asarray(a['delta_windows'])
    aa={arm:dict(selected=ids,pred_bins=B.tolist(),numerator=raw_loss(B,GG)) for arm in ARMS}
    b=diagnose_clip(PP,GG,many,aa,dict(trigger_candidates=30,active_candidates=24,slate_pairs=24))
    assert b['mode']=='coordinate_bounds' and b['combinations']==134596 and b['enumerated']==0
    assert b['min_loss_interval'][0]<=300<=b['min_loss_interval'][1]
    empty_actual={arm:dict(selected=[],pred_bins=P.tolist(),numerator=90) for arm in ARMS}
    z=diagnose_clip(P,G,[],empty_actual,dict(trigger_candidates=0,active_candidates=0,slate_pairs=0));assert z['k']==0 and z['combinations']==1
    print('PASS exact fixed-quarter subsets; coordinate certificate fallback; actual-arm closure; empty; no field')

def physical_selftest():
    import catalog210
    from beetrack.p0 import events
    def row(f,p,x):return [f,p,x,0,4,4,.9]
    R=np.asarray([row(1,1,-3),row(3,1,-1),row(5,1,-1),row(2,2,-3),row(4,2,-1)],float);C=R[:,1].astype(np.int64)
    poly=[[0,-10],[100,-10],[100,100],[0,100]];nf=5;catalog=catalog210.build(R,C,nf,poly)
    table={p['pair_id']:p for p in catalog['pairs']};active=[table[s] for s in catalog['active_slate_ids']]
    assert len(active)==1
    P=count_events(events(R,poly).astype(int),nf);G=count_events([[3,1]],nf);B=P+np.asarray(active[0]['delta_windows'])
    actual={arm:dict(selected=[active[0]['pair_id']],pred_bins=B.tolist(),numerator=raw_loss(B,G)) for arm in ARMS}
    d=catalog['denominator'];value=diagnose_clip(P,G,active,actual,dict(trigger_candidates=d['eligible_pairs'],active_candidates=d['active_pairs'],slate_pairs=d['active_slate_pairs'],all_slate_pairs=d['slate_pairs']))
    assert value['min_loss_interval']==[0,0]
    check=verify_count_witness(R,C,catalog['pairs'],poly,nf,value['best_known'],G)
    assert check['all_original_rows_once'] and check['official_count_exact']
    print('PASS synthetic all-rows temporal interleave and official count-only witness')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--synthetic',action='store_true');p.add_argument('--repo',type=Path)
    for name in ('root','out','source208'):p.add_argument('--'+name,type=Path)
    for name in ('prereg','catalog-seal','score-seal'):p.add_argument('--'+name)
    p.add_argument('--field-go',action='store_true');p.add_argument('--independent-cpu-released',action='store_true');a=p.parse_args()
    if a.synthetic:
        selftest()
        if a.repo:
            sys.path.insert(0,str(a.repo.resolve()));physical_selftest()
    else:
        if not all((a.repo,a.root,a.out,a.source208,a.prereg,a.catalog_seal,a.score_seal,a.field_go,a.independent_cpu_released)):p.error('published pinned inputs and FIFO CPU release required')
        sys.path.insert(0,str(a.repo.resolve()));field(a)
