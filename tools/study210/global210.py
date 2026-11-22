"""Supplement [all-disjoint-interleave-v1]: unbudgeted ONE-SHOT count support.

If ALL frozen nonzero-delta pair IDs equal the disjoint active slate, inert
edges can be removed from any matching without changing its windows. Conversely
each active subset is legal. Thus their 2^a vectors cover the WHOLE frozen edge
graph's one-shot matching count support, not iterative/shared-run operations.
No field IO on import. Original fixed-k model evaluation remains separate.
"""
from itertools import combinations
from fractions import Fraction
from pathlib import Path
import argparse,math,subprocess,sys
import numpy as np
import capacity210 as C

MARKER='[all-disjoint-interleave-v1]'
def endpoint_disjoint(ids,table):
    seen=set()
    for key in ids:
        endpoints=set(map(C.integer,table[key]['canonical_ids']))
        if len(endpoints)!=2 or seen&endpoints:return False
        seen.update(endpoints)
    return True

def analyze(P,G,all_pairs,active_slate_ids,actual,max_active=16):
    P=np.asarray(P);G=np.asarray(G)
    if P.shape!=G.shape or P.ndim!=2 or P.shape[1]!=2 or any(not np.issubdtype(a.dtype,np.integer) or (a<0).any() for a in (P,G)):raise ValueError('integer nonnegative directional windows required')
    P=P.astype(np.int64,copy=False);G=G.astype(np.int64,copy=False);table={p['pair_id']:p for p in all_pairs}
    if len(table)!=len(all_pairs):raise ValueError('unique original frozen pairs required')
    deltas={};global_active=[]
    for key,p in table.items():
        endpoints=tuple(map(C.integer,p['canonical_ids']))
        if len(endpoints)!=2 or not 0<endpoints[0]<endpoints[1] or endpoints!=C.pair_key(key):raise ValueError('original canonical pair identity differs')
        d=np.asarray(p['delta_windows'])
        if d.shape!=P.shape or not np.issubdtype(d.dtype,np.integer):raise ValueError('all-graph delta clock differs')
        active=bool(d.any())
        if not isinstance(p['active'],(bool,np.bool_)) or bool(p['active'])!=active:raise ValueError('frozen active flag differs actual delta vector')
        deltas[key]=d.astype(np.int64,copy=False)
        if active:global_active.append(key)
    global_active=sorted(global_active,key=C.pair_key);slate=sorted(active_slate_ids,key=C.pair_key)
    if len(set(slate))!=len(slate) or not set(slate)<=table.keys() or any(not deltas[k].any() for k in slate) or not endpoint_disjoint(slate,table):raise ValueError('disjoint original active slate required')
    equality=set(global_active)==set(slate);disjoint=endpoint_disjoint(global_active,table);global_scope=equality and disjoint
    support_ids=global_active if global_scope else slate;a=len(support_ids)
    if a>max_active:raise ValueError('small-support source assumption exceeded; no exact result')
    baseline=C.raw_loss(P,G);den=sum(map(int,G.ravel()))
    if den%30:raise ValueError('GT census is not30 repeated offsets')
    vectors={};budgets=[];best=None;worst=None;enumerated=0
    for size in range(a+1):
        local=[];hashes=set()
        for selected in combinations(support_ids,size):
            B=P.copy()
            for key in selected:B+=deltas[key]
            if (B<0).any():raise ValueError('nonphysical additive count')
            raw=C.raw_loss(B,G);h=C.digest(B);state=dict(selected=list(selected),active_budget=size,numerator=raw,pred_bins=B.tolist(),vector_sha256=h)
            tie=tuple(C.pair_key(k) for k in selected)
            if best is None or (raw,tie)<(best['numerator'],tuple(C.pair_key(k) for k in best['selected'])):best=state
            if worst is None or (-raw,tie)<(-worst['numerator'],tuple(C.pair_key(k) for k in worst['selected'])):worst=state
            if h not in vectors:vectors[h]=dict(pred_bins=B.tolist(),numerator=raw,active_subset_count=0,by_active_budget={},representative=state)
            v=vectors[h]
            if v['pred_bins']!=B.tolist():raise AssertionError('hash collision')
            v['active_subset_count']+=1;v['by_active_budget'][str(size)]=v['by_active_budget'].get(str(size),0)+1
            if tie<tuple(C.pair_key(k) for k in v['representative']['selected']):v['representative']=state
            local.append(state);hashes.add(h);enumerated+=1
        budgets.append(dict(active_budget=size,active_subset_count=len(local),binomial=math.comb(a,size),unique_window_vectors=len(hashes),
            min_numerator=min(s['numerator'] for s in local),max_numerator=max(s['numerator'] for s in local)))
    if enumerated!=2**a or sum(v['active_subset_count'] for v in vectors.values())!=enumerated:raise AssertionError('powerset enumeration incomplete')
    actual_checked={}
    for arm,record in actual.items():
        ids=record['selected']
        if len(ids)!=len(set(ids)) or not set(ids)<=table.keys() or not endpoint_disjoint(ids,table):raise ValueError('actual state is not one-shot disjoint')
        B=P.copy()
        for key in ids:B+=deltas[key]
        if B.tolist()!=record['pred_bins'] or C.raw_loss(B,G)!=record['numerator']:raise ValueError('actual arm count closure differs')
        actual_checked[arm]=dict(selected=ids,numerator=record['numerator'],vector_sha256=C.digest(B),
            within_reported_count_support=C.digest(B) in vectors,nonzero_selected=[key for key in ids if deltas[key].any()],exact_vector_closure=True)
    return dict(scope='all_frozen_eligible_pairs_one_shot_disjoint_no_budget' if global_scope else 'fixed_active_slate_only_no_budget',
        global_scope_proved=global_scope,premise=dict(global_nonzero_pair_ids=global_active,active_slate_pair_ids=slate,
            ID_set_equality=equality,global_nonzero_endpoint_disjoint=disjoint,all_pair_count=len(table)),
        proof='Remove inert edges from any ONE-SHOT disjoint matching: additive windows unchanged. Under set equality/disjointness the remainder is an active subset;every active subset itself is legal.',
        a=a,active_support_pair_ids=support_ids,enumerated_active_subsets=enumerated,by_active_budget=budgets,
        distinct_window_vectors=len(vectors),vectors=[dict(vector_sha256=h,**v) for h,v in sorted(vectors.items())],
        baseline_numerator=baseline,denominator=den,best=best,worst=worst,actual=actual_checked,
        best_gain_percent=float(Fraction(100*(baseline-best['numerator']),den)) if den else None,
        count_input=dict(pred_bins=P.tolist(),gt_bins=G.tolist(),all_pairs=[dict(pair_id=k,canonical_ids=table[k]['canonical_ids'],active=bool(deltas[k].any()),delta_windows=deltas[k].tolist()) for k in sorted(table,key=C.pair_key)]),
        budget_note='Budget counts NONZERO active edits after removing inert edges,not all physical graph matchings of that cardinality.',
        limits='Count only; no F1 claim; not shared-run/multiway/iterative edits, recutting, new observations, or pairs outside the frozen trigger graph.')

def pooled(clips):
    result={}
    for h in sorted({c['domain'] for c in clips}):
        rows=[c['support'] for c in clips if c['domain']==h];D=sum(r['denominator'] for r in rows);K=sum(r['baseline_numerator'] for r in rows)
        lo=sum(r['best']['numerator'] for r in rows);hi=sum(r['worst']['numerator'] for r in rows)
        result[h]=dict(scope='all_frozen_graphs_one_shot' if all(r['global_scope_proved'] for r in rows) else 'fixed_slate_only',
            denominator=D,baseline_numerator=K,min_numerator=lo,max_numerator=hi,best_gain_numerator=K-lo,
            best_gain_percent=float(Fraction(100*(K-lo),D)) if D else None,can_gain_at_least_1pp=100*(K-lo)>=D if D else None,
            can_gain_more_than_1pp=100*(K-lo)>D if D else None,
            actual={arm:sum(r['actual'][arm]['numerator'] for r in rows) for arm in rows[0]['actual']})
    return result

def field(args):
    if not args.field_go or not args.independent_cpu_released:raise ValueError('public supplement plus FIFO GO required')
    subprocess.run(['git','merge-base','--is-ancestor',args.supplement_sha,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.supplement_sha+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(MARKER)!=1:raise ValueError('published all-disjoint supplement differs')
    C.sealed(args.root/'catalog',args.catalog_seal);C.sealed(args.root/'score',args.score_seal)
    import catalog210,physical210
    top=C.read(args.root/'catalog/deployment.json')
    if top['prereg']!=args.prereg or len(top['clips'])!=6 or {(r['domain'],r['clip'],r['seed']) for r in top['clips']}!=catalog210.SPECS:raise ValueError('original six-case scope differs')
    args.out.mkdir(parents=True,exist_ok=True)
    receipt=dict(original_prereg=args.prereg,supplement_sha=args.supplement_sha,marker=MARKER,catalog_seal=args.catalog_seal,score_seal=args.score_seal,
        source_sha256=C.sha(__file__),count_helper_sha256=C.sha(C.__file__),physical_sha256=C.sha(physical210.__file__),no_GT_pairing_or_F1=True)
    C.save(args.out/'input_receipt.json',receipt);reports=[]
    for ref in top['clips']:
        source=catalog210.load_clip(args.root/'catalog'/ref['path'],args.source208);meta=source['metadata'];domain,clip=ref['domain'],ref['clip']
        audit=physical210.fixed_slate(source['pairs'])
        if audit['active_slate']!=source['active_slate_ids']:raise ValueError('frozen slate identity differs')
        folder=args.root/'score/clips'/domain/clip;records={arm:C.read(folder/arm/'score.json') for arm in ('K',*C.ARMS,'ALL')}
        actual={arm:dict(selected=r['selected_pair_ids'],pred_bins=r['metrics']['pred_bins'],numerator=r['numerator']) for arm,r in records.items()}
        K=records['K']['metrics'];support=analyze(K['pred_bins'],K['gt_bins'],source['pairs'],audit['active_slate'],actual)
        checks={}
        if domain in ('35','33'):
            for key in ('best','worst'):
                checks[key]=C.verify_count_witness(source['rows'],source['canon'],source['pairs'],meta['polygon'],meta['nf'],support[key],K['gt_bins'])
        value=dict(domain=domain,clip=clip,support=support,physical_count_checks=checks);reports.append(value)
        C.save(args.out/domain/clip/'support.json',value)
        print('GLOBAL210',domain,clip,support['global_scope_proved'],support['a'],support['enumerated_active_subsets'],support['best']['numerator'],support['worst']['numerator'],flush=True)
    result=dict(schema='interleave210-unbudgeted-count-support-v1',receipt=receipt,clips=reports,pooled=pooled(reports))
    C.save(args.out/'support.json',result);C.save(args.out/'manifest.json',{p.relative_to(args.out).as_posix():C.sha(p) for p in args.out.rglob('*') if p.is_file() and p.name not in ('manifest.json','SEALED.sha256')})
    seal=C.sha(args.out/'manifest.json');path=args.out/'SEALED.sha256'
    if path.exists():
        if path.read_text().strip()!=seal:raise ValueError('output seal differs')
    else:path.write_text(seal+'\n')
    print('GLOBAL210_SEALED',seal,flush=True)

def selftest():
    P=np.asarray([[3,1]]*30);G=np.asarray([[1,1]]*30)
    pairs=[dict(pair_id='1|2',canonical_ids=[1,2],delta_windows=[[-1,0]]*30,active=True),
        dict(pair_id='3|4',canonical_ids=[3,4],delta_windows=[[-1,0]]*30,active=True),
        dict(pair_id='1|5',canonical_ids=[1,5],delta_windows=[[0,0]]*30,active=False)]
    actual={'K':dict(selected=[],pred_bins=P.tolist(),numerator=60)}
    value=analyze(P,G,pairs,['1|2','3|4'],actual)
    assert value['global_scope_proved'] and value['best']['numerator']==0 and value['distinct_window_vectors']==3
    assert [b['active_subset_count'] for b in value['by_active_budget']]==[1,2,1]
    # An extra nonzero original ID invalidates whole-graph scope.
    extra=dict(pair_id='5|6',canonical_ids=[5,6],delta_windows=[[1,0]]*30,active=True)
    fallback=analyze(P,G,pairs+[extra],['1|2','3|4'],actual)
    assert not fallback['global_scope_proved'] and fallback['scope']=='fixed_active_slate_only_no_budget'
    shared=dict(pair_id='1|6',canonical_ids=[1,6],delta_windows=[[1,0]]*30,active=True)
    fail=analyze(P,G,pairs+[shared],['1|2','3|4'],actual);assert not fail['premise']['global_nonzero_endpoint_disjoint']
    print('PASS all-active-ID/disjoint premise;inert removal;complete active powerset;budget/vector multiplicities;scope fallback;no field')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--synthetic',action='store_true');p.add_argument('--repo',type=Path)
    for key in ('root','out','source208'):p.add_argument('--'+key,type=Path)
    for key in ('prereg','supplement-sha','catalog-seal','score-seal'):p.add_argument('--'+key)
    p.add_argument('--field-go',action='store_true');p.add_argument('--independent-cpu-released',action='store_true');a=p.parse_args()
    if a.synthetic:selftest()
    else:
        if not all((a.repo,a.root,a.out,a.source208,a.prereg,a.supplement_sha,a.catalog_seal,a.score_seal,a.field_go,a.independent_cpu_released)):p.error('published supplement/pinned artifacts/CPU release required')
        sys.path.insert(0,str(a.repo.resolve()));field(a)
